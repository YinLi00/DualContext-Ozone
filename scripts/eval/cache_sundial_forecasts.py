#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cache Sundial zero-shot predictions for Stage-1 O3.

S1-0 purpose:
  Generate Sundial predictions once and save them to disk, so later adapter
  training can read cached base forecasts instead of repeatedly calling
  Sundial.generate(), which is slow.

Default selected zero-shot setting:
  input_mode = phys
  sample aggregation = mean/median both saved
  num_samples = 20
  revin = True

Outputs in out_dir:
  pred_samples.npy    [N, S, H]  physical-space generated samples
  pred_mean.npy       [N, H]
  pred_median.npy     [N, H]
  true.npy            [N, H]     physical-space target
  city_id.npy         [N]
  mu.npy              [N]
  std.npy             [N]
  metrics.csv
  meta.json

Optional with --save_inputs:
  x.npy               [N, 96]       standardized target history
  x_multi.npy         [N, 96, V]    standardized multivariate history
  time_feat.npy       shape depends on dataset
  var_idx.npy         [N] or dataset-provided shape

Run example:
  python scripts/eval/cache_sundial_forecasts.py \
    --split test \
    --batch_size 16 \
    --num_samples 20 \
    --input_mode phys \
    --revin \
    --device cuda \
    --out_dir results/stage1_sundial_cache/sundial_o3_test_phys_s20
"""

import argparse
import json
import sys
import time
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

# ---- Sundial / transformers DynamicCache compatibility patch ----
# Safe even when running transformers==4.40.1; useful if the environment changes.
try:
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

    if not hasattr(DynamicCache, "get_max_length"):
        def _sundial_get_max_length(self):
            if hasattr(self, "get_max_cache_shape"):
                try:
                    return self.get_max_cache_shape()
                except Exception:
                    pass
            return None
        DynamicCache.get_max_length = _sundial_get_max_length

    if not hasattr(DynamicCache, "get_usable_length"):
        def _sundial_get_usable_length(self, new_seq_length=None, layer_idx=0):
            try:
                previous_seq_length = self.get_seq_length(layer_idx)
            except TypeError:
                previous_seq_length = self.get_seq_length()

            max_length = self.get_max_length()
            if max_length is not None and new_seq_length is not None:
                try:
                    if previous_seq_length + new_seq_length > max_length:
                        return max_length - new_seq_length
                except Exception:
                    pass
            return previous_seq_length

        DynamicCache.get_usable_length = _sundial_get_usable_length
except Exception:
    pass
# ----------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "scripts" / "train"
sys.path.insert(0, str(TRAIN_DIR))

from train_timer_context_ablation import Stage1O3CoRAContextDataset


class MetricAgg:
    def __init__(self):
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, pred, true):
        pred = np.asarray(pred, dtype=np.float64)
        true = np.asarray(true, dtype=np.float64)
        d = pred - true
        self.abs_sum += np.abs(d).sum()
        self.sq_sum += (d ** 2).sum()
        self.count += d.size

    def result(self):
        mae = self.abs_sum / max(self.count, 1)
        mse = self.sq_sum / max(self.count, 1)
        return {"MAE": mae, "MSE": mse, "RMSE": mse ** 0.5, "count": self.count}


def to_tensor_output(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for k in ["sequences", "predictions", "prediction", "samples", "forecast", "outputs"]:
            if k in output and isinstance(output[k], torch.Tensor):
                return output[k]
        for v in output.values():
            if isinstance(v, torch.Tensor):
                return v
    if hasattr(output, "sequences") and isinstance(output.sequences, torch.Tensor):
        return output.sequences
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise RuntimeError(f"Cannot find tensor output from generate output type={type(output)}")


def extract_forecast(output, pred_len):
    y = to_tensor_output(output)

    if y.ndim == 2:
        return y[:, -pred_len:].unsqueeze(1)

    if y.ndim == 3:
        # Expected Sundial local output: [B, S, H].
        if y.shape[-1] >= pred_len:
            return y[:, :, -pred_len:]
        # Possible [B, H, S].
        if y.shape[1] >= pred_len:
            return y[:, -pred_len:, :].transpose(1, 2)

    if y.ndim == 4:
        if y.shape[-1] == 1:
            return extract_forecast(y.squeeze(-1), pred_len)
        if y.shape[2] >= pred_len:
            return y[:, :, -pred_len:, 0]

    raise RuntimeError(f"Unsupported Sundial output shape: {tuple(y.shape)}")


def prepare_out_dir(out_dir: Path, overwrite: bool):
    if out_dir.exists():
        if overwrite:
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(
                f"out_dir already exists: {out_dir}\n"
                f"Use --overwrite to remove and regenerate it."
            )
    out_dir.mkdir(parents=True, exist_ok=True)


def make_memmap(path: Path, shape, dtype=np.float32):
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=str(ROOT / "models_hf_local" / "sundial_base_128m"))
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--pred_len", type=int, default=48)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--input_mode", type=str, default="phys", choices=["std", "phys"])
    ap.add_argument("--revin", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_inputs", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    prepare_out_dir(out_dir, args.overwrite)

    print("Loading dataset...")
    ds = Stage1O3CoRAContextDataset(
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
        time_feature_mode="both",
    )
    n = len(ds)
    if n <= 0:
        raise RuntimeError(f"No samples found for split={args.split}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Split       : {args.split}")
    print(f"Samples     : {n:,}")
    print(f"Out dir     : {out_dir}")
    print(f"Input mode  : {args.input_mode}")
    print(f"num_samples : {args.num_samples}")
    print(f"revin       : {args.revin}")

    print("Loading Sundial:", args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    model.eval().to(args.device)

    print("Model class:", type(model))
    print("Device     :", next(model.parameters()).device)

    # Inspect one batch for shapes.
    first_batch = next(iter(loader))
    y0 = first_batch["y_phys"]
    x0 = first_batch["x"]
    x_multi0 = first_batch.get("x_multi", None)
    time0 = first_batch.get("time_feat", None)
    city0 = first_batch["city_id"]

    h = args.pred_len
    pred_samples_mm = make_memmap(out_dir / "pred_samples.npy", (n, args.num_samples, h), np.float32)
    pred_mean_mm = make_memmap(out_dir / "pred_mean.npy", (n, h), np.float32)
    pred_median_mm = make_memmap(out_dir / "pred_median.npy", (n, h), np.float32)
    true_mm = make_memmap(out_dir / "true.npy", (n, h), np.float32)
    city_mm = make_memmap(out_dir / "city_id.npy", (n,), np.int16)
    mu_mm = make_memmap(out_dir / "mu.npy", (n,), np.float32)
    std_mm = make_memmap(out_dir / "std.npy", (n,), np.float32)

    input_maps = {}
    if args.save_inputs:
        x_shape = tuple(x0.shape[1:]) if x0.ndim > 1 else (1,)
        input_maps["x"] = make_memmap(out_dir / "x.npy", (n,) + x_shape, np.float32)

        if x_multi0 is not None:
            input_maps["x_multi"] = make_memmap(out_dir / "x_multi.npy", (n,) + tuple(x_multi0.shape[1:]), np.float32)

        if time0 is not None:
            input_maps["time_feat"] = make_memmap(out_dir / "time_feat.npy", (n,) + tuple(time0.shape[1:]), np.float32)

        if "var_idx" in first_batch:
            v0 = first_batch["var_idx"]
            var_shape = tuple(v0.shape[1:]) if v0.ndim > 1 else ()
            input_maps["var_idx"] = make_memmap(out_dir / "var_idx.npy", (n,) + var_shape, np.int16)

    metric_mean = MetricAgg()
    metric_median = MetricAgg()

    start = 0
    t0 = time.time()

    for step, batch in enumerate(loader, start=1):
        bsz = batch["y_phys"].shape[0]
        end = start + bsz

        x = batch["x"].to(args.device, dtype=torch.float32, non_blocking=True)
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        y_phys = batch["y_phys"].cpu().numpy().astype(np.float32)
        mu = batch["mu"].to(args.device, dtype=torch.float32, non_blocking=True).view(-1, 1)
        std = batch["std"].to(args.device, dtype=torch.float32, non_blocking=True).view(-1, 1)

        if args.input_mode == "phys":
            x_in = x * std + mu
        else:
            x_in = x

        output = model.generate(
            x_in,
            max_new_tokens=args.pred_len,
            num_samples=args.num_samples,
            revin=args.revin,
        )
        pred_samples = extract_forecast(output, args.pred_len)  # [B,S,H]

        if args.input_mode == "std":
            pred_samples_phys = pred_samples * std.view(-1, 1, 1) + mu.view(-1, 1, 1)
        else:
            pred_samples_phys = pred_samples

        pred_samples_np = pred_samples_phys.detach().cpu().numpy().astype(np.float32)
        pred_mean_np = pred_samples_np.mean(axis=1).astype(np.float32)
        pred_median_np = np.median(pred_samples_np, axis=1).astype(np.float32)

        pred_samples_mm[start:end] = pred_samples_np
        pred_mean_mm[start:end] = pred_mean_np
        pred_median_mm[start:end] = pred_median_np
        true_mm[start:end] = y_phys
        city_mm[start:end] = batch["city_id"].cpu().numpy().astype(np.int16)
        mu_mm[start:end] = batch["mu"].cpu().numpy().astype(np.float32)
        std_mm[start:end] = batch["std"].cpu().numpy().astype(np.float32)

        if args.save_inputs:
            input_maps["x"][start:end] = batch["x"].cpu().numpy().astype(np.float32)
            if "x_multi" in input_maps:
                input_maps["x_multi"][start:end] = batch["x_multi"].cpu().numpy().astype(np.float32)
            if "time_feat" in input_maps:
                input_maps["time_feat"][start:end] = batch["time_feat"].cpu().numpy().astype(np.float32)
            if "var_idx" in input_maps:
                input_maps["var_idx"][start:end] = batch["var_idx"].cpu().numpy().astype(np.int16)

        metric_mean.update(pred_mean_np, y_phys)
        metric_median.update(pred_median_np, y_phys)

        start = end

        if step == 1 or step % args.log_every == 0 or step == len(loader):
            rm = metric_mean.result()
            rmed = metric_median.result()
            elapsed = time.time() - t0
            print(
                f"[{step:5d}/{len(loader):5d}] "
                f"mean: MAE={rm['MAE']:.4f}, RMSE={rm['RMSE']:.4f} | "
                f"median: MAE={rmed['MAE']:.4f}, RMSE={rmed['RMSE']:.4f} | "
                f"count={rm['count']:,}, elapsed={elapsed:.1f}s",
                flush=True,
            )

    # Flush memmaps.
    for obj in [pred_samples_mm, pred_mean_mm, pred_median_mm, true_mm, city_mm, mu_mm, std_mm]:
        obj.flush()
    for obj in input_maps.values():
        obj.flush()

    mean_result = metric_mean.result()
    median_result = metric_median.result()

    rows = [
        {
            "split": args.split,
            "method": f"sundial_cache_{args.input_mode}_mean_s{args.num_samples}",
            **mean_result,
            "num_samples": args.num_samples,
            "sample_agg": "mean",
            "input_mode": args.input_mode,
            "revin": args.revin,
            "n_samples": n,
        },
        {
            "split": args.split,
            "method": f"sundial_cache_{args.input_mode}_median_s{args.num_samples}",
            **median_result,
            "num_samples": args.num_samples,
            "sample_agg": "median",
            "input_mode": args.input_mode,
            "revin": args.revin,
            "n_samples": n,
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    meta = {
        "model_path": args.model_path,
        "split": args.split,
        "n_samples": n,
        "pred_len": args.pred_len,
        "num_samples": args.num_samples,
        "input_mode": args.input_mode,
        "revin": args.revin,
        "save_inputs": args.save_inputs,
        "arrays": {
            "pred_samples": list(pred_samples_mm.shape),
            "pred_mean": list(pred_mean_mm.shape),
            "pred_median": list(pred_median_mm.shape),
            "true": list(true_mm.shape),
            "city_id": list(city_mm.shape),
            "mu": list(mu_mm.shape),
            "std": list(std_mm.shape),
        },
        "input_arrays": {k: list(v.shape) for k, v in input_maps.items()},
        "args": vars(args),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n==== Done ====")
    print(df.to_string(index=False))
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
