#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cache SEMPO zero-shot O3 forecasts using existing with-inputs cache.

Existing cache:
  x.npy      standardized O3 input [N,96]
  true.npy   physical O3 target [N,48]
  mu.npy/std.npy global scaler for recovering physical x
  city_id.npy, optional x_multi/time_feat/var_idx

SEMPO checkpoint:
  seq_len=512, pred_len=96.
Minimal adaptation:
  standardized x_96 -> physical x_96 -> left pad to 512 -> SEMPO output 96 -> take first 48.
"""

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]


def make_config(args):
    return SimpleNamespace(
        task_name="long_term_forecast",
        is_pretraining=0,
        is_training=0,
        is_zeroshot=1,
        train_test=0,
        model_id="O3",
        model="SEMPO",
        data="O3",
        features="M",
        target="O3",
        freq="h",
        percent=100,
        setting="sempo_o3_cache",
        inverse=False,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.sempo_pred_len,
        horizon_lengths=[1, 96, 192, 336, 720],
        patch_len=args.patch_len,
        stride=args.stride,
        c_in=args.c_in,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        d_ff=args.d_ff,
        moving_avg=25,
        factor=1,
        dropout=0.1,
        distil=True,
        embed="timeF",
        activation="gelu",
        output_attention=False,
        head_type="prediction",
        domain_len=args.domain_len,
        use_gpu=True,
        use_multi_gpu=False,
        device=args.device,
    )


def smart_load_state_dict(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    if not isinstance(sd, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(sd)}")

    model_keys = list(model.state_dict().keys())
    sd_keys = list(sd.keys())
    if sd_keys and sd_keys[0].startswith("module.") and not model_keys[0].startswith("module."):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("load_state_dict missing/unexpected:", len(missing), len(unexpected))
    if len(missing) > 10 or len(unexpected) > 10:
        print("first missing:", missing[:10])
        print("first unexpected:", unexpected[:10])
    return model


def metric_np(pred, true):
    d = pred.astype(np.float64) - true.astype(np.float64)
    mae = np.abs(d).mean()
    mse = (d ** 2).mean()
    return float(mae), float(mse), float(math.sqrt(mse)), int(d.size)


def load_array(cache_dir, name, required=True):
    p = Path(cache_dir) / name
    if p.exists():
        return np.load(p, mmap_mode="r")
    if required:
        raise FileNotFoundError(f"Missing {p}")
    return None


def pad_left(x96, seq_len=512, mode="repeat_context"):
    """
    x96: [B,96] physical O3.
    Return [B,512].
    """
    b, l = x96.shape
    if l >= seq_len:
        return x96[:, -seq_len:].astype(np.float32)

    need = seq_len - l

    if mode == "repeat_first":
        left = np.repeat(x96[:, :1], need, axis=1)
    elif mode == "context_mean":
        left = np.repeat(x96.mean(axis=1, keepdims=True), need, axis=1)
    elif mode == "zero":
        left = np.zeros((b, need), dtype=np.float32)
    elif mode == "repeat_context":
        reps = int(np.ceil(need / l))
        tiled = np.tile(x96, (1, reps))
        left = tiled[:, -need:]
    elif mode == "reverse_context":
        rev = x96[:, ::-1]
        reps = int(np.ceil(need / l))
        tiled = np.tile(rev, (1, reps))
        left = tiled[:, -need:]
    else:
        raise ValueError(f"Unknown pad mode: {mode}")

    return np.concatenate([left, x96], axis=1).astype(np.float32)


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
def forecast_batch(model, cfg, x96_phys, args):
    x512 = pad_left(x96_phys, seq_len=args.seq_len, mode=args.pad_mode)
    x = torch.as_tensor(x512[:, :, None], dtype=torch.float32, device=args.device)
    b = x.shape[0]

    x_mark = torch.zeros(b, args.seq_len, 4, dtype=torch.float32, device=args.device)
    x_dec = torch.zeros(b, args.label_len + args.sempo_pred_len, args.c_in, dtype=torch.float32, device=args.device)
    x_mark_dec = torch.zeros(b, args.label_len + args.sempo_pred_len, 4, dtype=torch.float32, device=args.device)

    ys, recon = model(x, x_mark, x_dec, x_mark_dec)
    idx96 = cfg.horizon_lengths.index(96)
    pred96 = ys[idx96][:, -args.sempo_pred_len:, :]
    pred48 = pred96[:, :args.task_pred_len, 0]
    return pred48.detach().cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_cache", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--sempo_root", type=str, default=str(ROOT / "SEMPO/SEMPO-main"))
    ap.add_argument("--ckpt", type=str, default=str(ROOT / "SEMPO/SEMPO-main/checkpoints/long_term_forecast_SEMPO_UTSD_ftM_sl512_ll48_pl96_pl64_dm256_nh2_el3_dl3_df128_fc1_ebtimeF_dtTrue_0/checkpoint.pth"))

    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--label_len", type=int, default=48)
    ap.add_argument("--sempo_pred_len", type=int, default=96)
    ap.add_argument("--task_pred_len", type=int, default=48)
    ap.add_argument("--patch_len", type=int, default=64)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--c_in", type=int, default=1)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=2)
    ap.add_argument("--e_layers", type=int, default=3)
    ap.add_argument("--d_layers", type=int, default=3)
    ap.add_argument("--d_ff", type=int, default=128)
    ap.add_argument("--domain_len", type=int, default=128)

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pad_mode", type=str, default="repeat_context",
                    choices=["repeat_context", "repeat_first", "context_mean", "zero", "reverse_context"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # Resolve paths before SEMPO script changes working directory.
    input_cache = Path(args.input_cache).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "pred_mean.npy").exists() and not args.overwrite:
        print(f"Found existing cache: {out_dir}. Use --overwrite to rerun.")
        return

    x_std = load_array(input_cache, "x.npy", required=True)
    true = load_array(input_cache, "true.npy", required=True)
    city_id = load_array(input_cache, "city_id.npy", required=False)
    mu_arr = load_array(input_cache, "mu.npy", required=True)
    std_arr = load_array(input_cache, "std.npy", required=True)

    n = x_std.shape[0]
    if args.max_samples and args.max_samples > 0:
        n = min(n, args.max_samples)

    # Resolve paths before chdir, otherwise relative checkpoint paths break after os.chdir().
    sempo_root = Path(args.sempo_root).expanduser().resolve()
    args.ckpt = str(Path(args.ckpt).expanduser().resolve())

    sys.path.insert(0, str(sempo_root))
    os.chdir(str(sempo_root))

    cfg = make_config(args)
    module = importlib.import_module("models.SEMPO")
    model = module.Model(cfg).to(args.device)
    model = smart_load_state_dict(model, args.ckpt)
    model.eval()

    print("==== SEMPO cache ====")
    print("input_cache:", input_cache)
    print("out_dir:", out_dir)
    print("n_samples:", n)
    print("pad_mode:", args.pad_mode)
    print("ckpt:", args.ckpt)
    print("param count:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    preds = np.empty((n, args.task_pred_len), dtype=np.float32)
    abs_sum = 0.0
    sq_sum = 0.0
    count = 0
    num_batches = math.ceil(n / args.batch_size)

    for bi, st in enumerate(range(0, n, args.batch_size), start=1):
        ed = min(n, st + args.batch_size)
        xb = np.asarray(x_std[st:ed], dtype=np.float32).copy()
        mu_b = np.asarray(mu_arr[st:ed], dtype=np.float32).reshape(-1, 1)
        std_b = np.asarray(std_arr[st:ed], dtype=np.float32).reshape(-1, 1)
        xb_phys = xb * std_b + mu_b

        pb = forecast_batch(model, cfg, xb_phys, args)
        preds[st:ed] = pb

        tb = np.asarray(true[st:ed, :args.task_pred_len], dtype=np.float32)
        d = pb - tb
        abs_sum += np.abs(d).sum()
        sq_sum += np.square(d).sum()
        count += d.size

        if bi == 1 or bi % 50 == 0 or bi == num_batches:
            mae = abs_sum / max(count, 1)
            rmse = math.sqrt(sq_sum / max(count, 1))
            print(f"[{bi:5d}/{num_batches:5d}] MAE={mae:.4f}, RMSE={rmse:.4f}, count={count:,}", flush=True)

    true_arr = np.asarray(true[:n, :args.task_pred_len], dtype=np.float32)
    np.save(out_dir / "pred_mean.npy", preds)
    np.save(out_dir / "pred_median.npy", preds)
    np.save(out_dir / "true.npy", true_arr)

    if city_id is not None:
        np.save(out_dir / "city_id.npy", np.asarray(city_id[:n]))
    else:
        np.save(out_dir / "city_id.npy", np.zeros((n,), dtype=np.int16))

    copied = copy_optional_arrays(input_cache, out_dir, ["x.npy", "x_multi.npy", "time_feat.npy", "var_idx.npy", "mu.npy", "std.npy"])

    mae, mse, rmse, total_count = metric_np(preds, true_arr)
    metrics = pd.DataFrame([{
        "split": input_cache.name,
        "method": f"sempo_zeroshot_pad_{args.pad_mode}",
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "count": total_count,
        "n_samples": n,
        "pad_mode": args.pad_mode,
        "seq_len": args.seq_len,
        "sempo_pred_len": args.sempo_pred_len,
        "task_pred_len": args.task_pred_len,
        "ckpt": str(args.ckpt),
    }])
    metrics.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    meta = {
        "input_cache": str(input_cache),
        "out_dir": str(out_dir),
        "ckpt": str(args.ckpt),
        "n_samples": n,
        "pad_mode": args.pad_mode,
        "arrays": {
            "pred_mean": list(preds.shape),
            "pred_median": list(preds.shape),
            "true": list(true_arr.shape),
        },
        "copied_optional_arrays": copied,
        "args": vars(args),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n==== Done ====")
    print(metrics.to_string(index=False))
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
