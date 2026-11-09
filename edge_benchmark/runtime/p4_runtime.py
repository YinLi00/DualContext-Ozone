#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from edge_common import default_base_root, inverse_point, load_timer, resolve_device, safe_torch_load, validate_common_npz
from p4_model import P4Forecaster


def load_p4(root: Path, base_root: Path, device: torch.device):
    ckpt_path = root / "checkpoints" / "p4_seed2026_best.pt"
    ck = safe_torch_load(ckpt_path)
    state = ck.get("adapter_state")
    if not isinstance(state, dict):
        raise RuntimeError(f"P4 checkpoint lacks adapter_state: {ckpt_path}")
    a = ck["args"]
    cities = list(ck["cities"])
    variables = list(ck["variables"])
    timer = load_timer(base_root, device)
    m = P4Forecaster(
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
    )
    missing, unexpected = m.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if not k.startswith("timer.")]
    if unexpected or bad_missing:
        raise RuntimeError(f"P4 checkpoint mismatch unexpected={unexpected} non_timer_missing={bad_missing}")
    m.to(device).eval()
    return m


@torch.inference_mode()
def infer(m, z, device: torch.device, batch_size: int):
    validate_common_npz(z)
    out = []
    n = len(z["x"])
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        p = m(
            torch.from_numpy(z["x"][s:e]).to(device=device, dtype=torch.float32),
            torch.from_numpy(z["x_multi"][s:e]).to(device=device, dtype=torch.float32),
            torch.from_numpy(z["time_feat"][s:e]).to(device=device, dtype=torch.float32),
            torch.from_numpy(z["var_idx"][s:e]).to(device=device, dtype=torch.long),
            torch.from_numpy(z["city_id"][s:e]).to(device=device, dtype=torch.long),
        )
        out.append(p.float().cpu().numpy())
    p_std = np.concatenate(out, axis=0).astype(np.float32, copy=False)
    return inverse_point(p_std, z["mu"].astype(np.float32), z["std"].astype(np.float32))


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
    pred = infer(load_p4(root, base, device), z, device, a.batch_size)
    if not np.isfinite(pred).all():
        raise RuntimeError("P4 produced non-finite output")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.output, pred, allow_pickle=False)
    print(f"saved {a.output} {pred.shape} {pred.dtype}")


if __name__ == "__main__":
    main()
