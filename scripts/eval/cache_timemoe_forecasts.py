#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cache TimeMoE-50M zero-shot O3 forecasts using existing Sundial with-inputs cache.

Why this design:
  We already have clean train/val/test with-input caches from Sundial:
    x.npy          historical O3 input, shape [N,96]
    true.npy       O3 target, shape [N,48]
    city_id.npy
    optional x_multi.npy / time_feat.npy / var_idx.npy

  TimeMoE is univariate. We run it on x.npy only and save:
    pred_mean.npy
    pred_median.npy  # same as pred_mean for deterministic TimeMoE generate
    true.npy
    city_id.npy
    x.npy, x_multi.npy, time_feat.npy, var_idx.npy if available

  This makes the output cache compatible with existing Sundial S1/S2 adapter scripts.

TimeMoE official generate pattern:
  normalize context -> model.generate(normed, max_new_tokens=pred_len) -> inverse normalize.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[2]


def metric_np(pred, true):
    d = pred.astype(np.float64) - true.astype(np.float64)
    mae = np.abs(d).mean()
    mse = (d ** 2).mean()
    return float(mae), float(mse), float(math.sqrt(mse)), int(d.size)


def pick_model_path(user_path):
    if user_path:
        return user_path
    candidates = [
        ROOT / "models_hf_local/timemoe_200m",
        ROOT / "models_hf_local/TimeMoE-200M",
        ROOT / "models_hf_local/TimeMoE-50M",
        ROOT / "models_hf_local/timemoe_50m",
        ROOT / "models_hf_local/TimeMoE-50M-main",
        ROOT / "models_hf_local/Maple728_TimeMoE-50M",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(ROOT / "models_hf_local/timemoe_200m")


def load_array(cache_dir, name, required=True):
    p = Path(cache_dir) / name
    if p.exists():
        return np.load(p, mmap_mode="r")
    if required:
        raise FileNotFoundError(f"Missing {p}")
    return None


def copy_optional_arrays(src_dir, out_dir, names):
    copied = {}
    for name in names:
        src = Path(src_dir) / name
        if src.exists():
            arr = np.load(src, mmap_mode="r")
            np.save(Path(out_dir) / name, np.asarray(arr))
            copied[name] = list(arr.shape)
    return copied


@torch.no_grad()
def forecast_batch(model, x_np, pred_len, device, input_mode="phys", revin=True):
    """
    x_np: [B,96] physical scale if input_mode=phys.
    """
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)

    if revin:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        std = torch.clamp(std, min=1e-5)
        xin = (x - mean) / std
    else:
        mean = torch.zeros_like(x[:, :1])
        std = torch.ones_like(x[:, :1])
        xin = x

    # TimeMoE weights may be bf16; cast only the model input, then inverse-transform in fp32.
    model_dtype = next(model.parameters()).dtype
    xin = xin.to(dtype=model_dtype)

    # Official TimeMoE example: output shape [B, context_len + pred_len].
    out = model.generate(xin, max_new_tokens=pred_len)
    if isinstance(out, dict):
        # Just in case a future version returns dict-like generation output.
        out = out["sequences"]
    pred_norm = out[:, -pred_len:].float()
    pred = pred_norm * std.float() + mean.float()
    return pred.detach().cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_cache", type=str, required=True,
                    help="Existing with-inputs cache, e.g. results/stage1_sundial_cache/sundial_o3_test_phys_s20_with_inputs")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--model_path", type=str, default="")
    ap.add_argument("--pred_len", type=int, default=48)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--input_mode", type=str, default="phys")
    ap.add_argument("--revin", action="store_true")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    input_cache = Path(args.input_cache)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "pred_mean.npy").exists() and not args.overwrite:
        print(f"Found existing cache: {out_dir}. Use --overwrite to rerun.")
        return

    x = load_array(input_cache, "x.npy", required=True)
    true = load_array(input_cache, "true.npy", required=True)
    city_id = load_array(input_cache, "city_id.npy", required=False)

    # The with-inputs cache stores x.npy in standardized O3 space, while true.npy is physical O3.
    # Recover physical O3 input before feeding TimeMoE.
    mu_arr = load_array(input_cache, "mu.npy", required=False)
    std_arr = load_array(input_cache, "std.npy", required=False)

    n = x.shape[0]
    if args.max_samples and args.max_samples > 0:
        n = min(n, args.max_samples)

    print("==== TimeMoE cache ====")
    print("input_cache:", input_cache)
    print("out_dir:", out_dir)
    print("n_samples:", n)
    print("x shape:", x.shape, "true shape:", true.shape)
    print("device:", args.device)
    if mu_arr is not None and std_arr is not None:
        print("Recover x from standardized space using mu/std before TimeMoE inference.")
    else:
        print("WARNING: mu/std not found; x will be used as-is.")

    model_path = pick_model_path(args.model_path)
    print("Loading TimeMoE from:", model_path)

    torch_dtype = "auto"
    if args.dtype == "float32":
        torch_dtype = torch.float32
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=args.device,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.eval()

    # Some remote-code models dislike use_cache mismatches; keep default, but disable grads.
    preds = np.empty((n, args.pred_len), dtype=np.float32)

    abs_sum = 0.0
    sq_sum = 0.0
    count = 0

    num_batches = math.ceil(n / args.batch_size)
    for bi, st in enumerate(range(0, n, args.batch_size), start=1):
        ed = min(n, st + args.batch_size)
        xb = np.asarray(x[st:ed], dtype=np.float32).copy()

        # Recover physical O3 input if global scaler is available.
        # x.npy is standardized, true.npy is physical.
        if mu_arr is not None and std_arr is not None:
            mu_b = np.asarray(mu_arr[st:ed], dtype=np.float32).reshape(-1, 1)
            std_b = np.asarray(std_arr[st:ed], dtype=np.float32).reshape(-1, 1)
            xb = xb * std_b + mu_b

        pb = forecast_batch(model, xb, args.pred_len, args.device, args.input_mode, args.revin)
        preds[st:ed] = pb

        tb = np.asarray(true[st:ed, :args.pred_len], dtype=np.float32)
        d = pb - tb
        abs_sum += np.abs(d).sum()
        sq_sum += np.square(d).sum()
        count += d.size

        if bi == 1 or bi % 50 == 0 or bi == num_batches:
            mae = abs_sum / max(count, 1)
            rmse = math.sqrt(sq_sum / max(count, 1))
            print(f"[{bi:5d}/{num_batches:5d}] MAE={mae:.4f}, RMSE={rmse:.4f}, count={count:,}", flush=True)

    true_arr = np.asarray(true[:n, :args.pred_len], dtype=np.float32)
    np.save(out_dir / "pred_mean.npy", preds)
    np.save(out_dir / "pred_median.npy", preds)
    np.save(out_dir / "true.npy", true_arr)

    if city_id is not None:
        np.save(out_dir / "city_id.npy", np.asarray(city_id[:n]))
    else:
        np.save(out_dir / "city_id.npy", np.zeros((n,), dtype=np.int16))

    # Copy inputs so existing S2 adapter scripts can reuse the cache.
    copied = copy_optional_arrays(input_cache, out_dir, ["x.npy", "x_multi.npy", "time_feat.npy", "var_idx.npy", "mu.npy", "std.npy"])

    mae, mse, rmse, total_count = metric_np(preds, true_arr)
    metrics = pd.DataFrame([{
        "split": input_cache.name,
        "method": "timemoe200m_zeroshot",
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "count": total_count,
        "n_samples": n,
        "input_mode": args.input_mode,
        "revin": bool(args.revin),
        "model_path": str(model_path),
    }])
    metrics.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    meta = {
        "input_cache": str(input_cache),
        "out_dir": str(out_dir),
        "model_path": str(model_path),
        "n_samples": n,
        "pred_len": args.pred_len,
        "batch_size": args.batch_size,
        "input_mode": args.input_mode,
        "revin": bool(args.revin),
        "copied_optional_arrays": copied,
        "arrays": {
            "pred_mean": list(preds.shape),
            "pred_median": list(preds.shape),
            "true": list(true_arr.shape),
        },
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n==== Done ====")
    print(metrics.to_string(index=False))
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
