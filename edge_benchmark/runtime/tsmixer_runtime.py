#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from edge_common import inverse_point, resolve_device, safe_torch_load, validate_common_npz
from tsmixer_backbone import TSMixerBackbone


def _semantic_value(name: str, batch: dict):
    n = name.lower()
    if "city" in n:
        return batch["city_id"]
    if "time" in n or "mark" in n:
        t = batch["time_feat"]
        if t.ndim == 2:
            if t.shape[1] % 48 != 0:
                raise RuntimeError(f"Cannot reshape time_feat {tuple(t.shape)} to [B,48,F]")
            t = t.reshape(t.shape[0], 48, t.shape[1] // 48)
        return t
    if n in {"x", "x_multi", "input", "inputs", "x_enc", "history"} or n.startswith("x_"):
        return batch["x_multi"]
    return None


def call_tsmixer(model, batch: dict, o3_idx: int) -> torch.Tensor:
    """Call the exact extracted wrapper without assuming argument order."""
    params = list(inspect.signature(model.forward).parameters.values())
    kwargs = {}
    unresolved = []
    for p in params:
        if p.name == "self":
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        v = _semantic_value(p.name, batch)
        if v is None:
            if p.default is not inspect._empty:
                continue
            unresolved.append(p.name)
        else:
            kwargs[p.name] = v
    if unresolved:
        raise RuntimeError(
            "Cannot map exact GlobalTSMixer.forward parameters "
            f"{unresolved}; signature={inspect.signature(model.forward)}"
        )
    out = model(**kwargs)
    if isinstance(out, dict):
        for key in ("point", "pred", "prediction", "output"):
            if key in out:
                out = out[key]
                break
        else:
            raise RuntimeError(f"Unsupported TSMixer dict output keys={list(out)}")
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.ndim == 3:
        if out.shape[-1] <= o3_idx:
            raise RuntimeError(f"Cannot select O3 channel {o3_idx} from {tuple(out.shape)}")
        out = out[:, -48:, o3_idx]
    elif out.ndim == 2:
        out = out[:, -48:]
    else:
        raise RuntimeError(f"Unexpected TSMixer output shape {tuple(out.shape)}")
    return out.float()


def load_tsmixer(root: Path, device: torch.device):
    wrapper_file = root / "runtime" / "tsmixer_wrapper.py"
    if not wrapper_file.is_file():
        raise RuntimeError(
            "tsmixer_wrapper.py is missing. Do NOT guess the wrapper. "
            "Run finalize_tsmixer_exact.py on Windows with the exact "
            "train_global_tsmixer.py source before Jetson deployment."
        )
    from tsmixer_wrapper import GlobalTSMixer

    ckpt_path = root / "checkpoints" / "tsmixer_seed2026_best.pt"
    ck = safe_torch_load(ckpt_path)
    state = ck.get("model")
    if not isinstance(state, dict):
        raise RuntimeError(f"TSMixer checkpoint lacks model state: {ckpt_path}")
    a = dict(ck.get("args", {}))
    variables = list(a.get("variables", ["PM2.5","PM10","SO2","NO2","O3","CO","temperature","relative_humidity"]))
    cities = list(a.get("cities", []))
    if not cities:
        cities = [str(i) for i in range(int(state["city_emb.weight"].shape[0]))]
    cfg = SimpleNamespace(
        task_name="long_term_forecast", seq_len=96, pred_len=48,
        d_model=int(a.get("d_model", 128)), dropout=float(a.get("dropout", 0.1)),
        enc_in=len(variables), e_layers=int(a.get("e_layers", 2)),
    )
    backbone = TSMixerBackbone(cfg)
    m = GlobalTSMixer(
        backbone, len(cities), len(variables), variables.index("O3"),
        str(a.get("variant", "city_time")),
        int(a.get("city_emb_dim", 16)), int(a.get("condition_hidden", 64)),
    )
    m.load_state_dict(state, strict=True)
    if sum(p.numel() for p in m.parameters()) != 60586:
        raise RuntimeError("TSMixer parameter count mismatch")
    m.to(device).eval()
    return m, variables.index("O3")


@torch.inference_mode()
def infer(m, o3_idx: int, z, device: torch.device, batch_size: int):
    validate_common_npz(z)
    out = []
    n = len(z["x_multi"])
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        batch = {
            "x_multi": torch.from_numpy(z["x_multi"][s:e]).to(device=device, dtype=torch.float32),
            "time_feat": torch.from_numpy(z["time_feat"][s:e]).to(device=device, dtype=torch.float32),
            "city_id": torch.from_numpy(z["city_id"][s:e]).to(device=device, dtype=torch.long),
        }
        out.append(call_tsmixer(m, batch, o3_idx).cpu().numpy())
    p_std = np.concatenate(out).astype(np.float32, copy=False)
    return inverse_point(p_std, z["mu"].astype(np.float32), z["std"].astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    device = resolve_device(a.device)
    z = np.load(a.input, allow_pickle=False)
    m, o3_idx = load_tsmixer(root, device)
    pred = infer(m, o3_idx, z, device, a.batch_size)
    if not np.isfinite(pred).all():
        raise RuntimeError("TSMixer produced non-finite output")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.output, pred, allow_pickle=False)
    print(f"saved {a.output} {pred.shape} {pred.dtype}")


if __name__ == "__main__":
    main()
