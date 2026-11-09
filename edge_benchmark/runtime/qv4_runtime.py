#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from edge_common import default_base_root, inverse_quantiles, load_timer, resolve_device, safe_torch_load, validate_common_npz
from qv4_model import QV4Forecaster


def load_qv4(root: Path, base_root: Path, device: torch.device):
    ckpt_path = root / "checkpoints" / "qv4_seed2026_best.pt"
    ck = safe_torch_load(ckpt_path)
    state = ck.get("trainable_state")
    if not isinstance(state, dict):
        raise RuntimeError(f"QV4 checkpoint lacks trainable_state: {ckpt_path}")
    a = ck["args"]
    cities = list(ck["cities"])
    variables = list(ck["variables"])
    if "O3" not in variables:
        raise RuntimeError("O3 missing from QV4 checkpoint variables")
    timer = load_timer(base_root, device)
    m = QV4Forecaster(
        timer=timer,
        n_vars=len(variables),
        n_cities=len(cities),
        input_hidden=int(a["input_hidden"]),
        output_hidden=int(a["output_hidden"]),
        context_hidden=int(a["context_hidden"]),
        emb_dim=int(a["emb_dim"]),
        city_emb_dim=int(a["city_emb_dim"]),
        dropout=float(a["dropout"]),
        o3_idx=variables.index("O3"),
        corr_temperature=float(a["corr_temperature"]),
        quantile_inner_gap_init=float(a.get("quantile_inner_gap_init", 0.8)),
        quantile_outer_gap_init=float(a.get("quantile_outer_gap_init", 0.4)),
    )
    missing, unexpected = m.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"QV4 unexpected checkpoint keys: {unexpected}")
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    absent = sorted(trainable - set(state))
    if absent:
        raise RuntimeError(f"QV4 checkpoint missing trainable keys: {absent[:30]}")
    m.to(device).eval()
    return m


@torch.inference_mode()
def infer(m, z, device: torch.device, batch_size: int):
    validate_common_npz(z)
    out = []
    n = len(z["x"])
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        q = m(
            torch.from_numpy(z["x"][s:e]).to(device=device, dtype=torch.float32),
            torch.from_numpy(z["x_multi"][s:e]).to(device=device, dtype=torch.float32),
            torch.from_numpy(z["time_feat"][s:e]).to(device=device, dtype=torch.float32),
            torch.from_numpy(z["var_idx"][s:e]).to(device=device, dtype=torch.long),
            torch.from_numpy(z["city_id"][s:e]).to(device=device, dtype=torch.long),
        )
        out.append(q.float().cpu().numpy())
    q_std = np.concatenate(out, axis=0).astype(np.float32, copy=False)
    q_phys = inverse_quantiles(q_std, z["mu"].astype(np.float32), z["std"].astype(np.float32))
    return q_phys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--base-root", type=Path, default=None)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = a.base_root.resolve() if a.base_root else default_base_root(root)
    device = resolve_device(a.device)
    z = np.load(a.input, allow_pickle=False)
    pred = infer(load_qv4(root, base, device), z, device, a.batch_size)
    crossing = int(np.count_nonzero(pred[..., :-1] > pred[..., 1:]))
    if not np.isfinite(pred).all():
        raise RuntimeError("QV4 produced non-finite output")
    if crossing != 0:
        raise RuntimeError(f"QV4 quantile crossing count={crossing}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.output, pred, allow_pickle=False)
    print(f"saved {a.output} {pred.shape} {pred.dtype} crossing_count={crossing}")


if __name__ == "__main__":
    main()
