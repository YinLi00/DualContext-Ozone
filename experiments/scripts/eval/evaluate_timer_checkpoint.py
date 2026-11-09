#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inference-only interventions on trained Timer checkpoints.

A normally trained Timer checkpoint is kept fixed. Only its base forecast is replaced
at test time. This answers whether the trained contextual residual model actually
depends on the sample-specific Timer forecast.

Interventions:
  correct_timer              : original Timer base forecast
  zero_intervention          : all-zero base
  persistence_intervention   : repeat the last standardized O3 value for 48 h
  raw_seasonal24_intervention: repeat the last raw standardized 24 h twice
  global_cross_city_shuffle  : use a Timer forecast from a randomly selected sample
                               belonging to a different city
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

L, H = 96, 48
INTERVENTIONS = [
    "correct_timer",
    "zero_intervention",
    "persistence_intervention",
    "raw_seasonal24_intervention",
    "global_cross_city_shuffle",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path,
                   default=Path(__file__).resolve().parents[3])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--shuffle_seed", type=int, default=7312026)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--keep_cache", action="store_true")
    p.add_argument("--max_test_samples", type=int, default=0)
    return p.parse_args()


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MetricBundle:
    def __init__(self, n_cities: int):
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.count = 0
        self.city_abs = np.zeros(n_cities, dtype=np.float64)
        self.city_sq = np.zeros(n_cities, dtype=np.float64)
        self.city_n = np.zeros(n_cities, dtype=np.int64)
        self.h_abs = np.zeros(H, dtype=np.float64)
        self.h_sq = np.zeros(H, dtype=np.float64)
        self.h_n = np.zeros(H, dtype=np.int64)

    def update(self, pred: np.ndarray, truth: np.ndarray, city_ids: np.ndarray) -> None:
        d = np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
        self.abs_sum += float(np.abs(d).sum())
        self.sq_sum += float(np.square(d).sum())
        self.count += int(d.size)

        for cid in np.unique(city_ids):
            mask = city_ids == cid
            dc = d[mask]
            self.city_abs[int(cid)] += float(np.abs(dc).sum())
            self.city_sq[int(cid)] += float(np.square(dc).sum())
            self.city_n[int(cid)] += int(dc.size)

        self.h_abs += np.abs(d).sum(axis=0)
        self.h_sq += np.square(d).sum(axis=0)
        self.h_n += d.shape[0]

    def overall(self) -> dict:
        mse = self.sq_sum / self.count
        return {
            "MAE": self.abs_sum / self.count,
            "MSE": mse,
            "RMSE": math.sqrt(mse),
            "count": self.count,
        }


def load_strict_module(root: Path):
    train_dir = root / "experiments" / "scripts" / "train"
    if str(train_dir) not in sys.path:
        sys.path.insert(0, str(train_dir))
    return importlib.import_module("train_attribution_reference")


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_cross_city_permutation(city_ids: np.ndarray, seed: int) -> np.ndarray:
    """Assign every recipient a donor from a different city.

    Sampling is without replacement within each recipient-city group. Donors may be
    reused across different recipient cities, which is acceptable because the goal is
    information destruction rather than a one-to-one global derangement.
    """
    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(city_ids), dtype=np.int64)
    perm = np.empty(len(city_ids), dtype=np.int64)
    for cid in np.unique(city_ids):
        recipients = all_idx[city_ids == cid]
        donors = all_idx[city_ids != cid]
        if len(donors) < len(recipients):
            raise RuntimeError(f"Not enough cross-city donors for city_id={cid}")
        perm[recipients] = rng.choice(donors, size=len(recipients), replace=False)
    if np.any(city_ids[perm] == city_ids):
        raise RuntimeError("Cross-city permutation validation failed")
    return perm


def forward_with_base(model, x, xm, tf, vi, ci, base):
    """Exact original residual head, with an externally supplied base forecast."""
    x_adapt = model.input_adapter(x, ci)
    pred = model.gamma(vi) * base + model.beta(vi)
    ctx = model.context_encoder(xm)
    feat = torch.cat([
        pred, x, x_adapt, tf, ctx,
        model.var_emb(vi), model.city_emb(ci),
    ], dim=-1)
    return pred + model.output_gate * model.output_net(feat)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = choose_device(args.device)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = (root / "experiments" / "results" / "01_attribution" /
                     f"timer_seed{args.seed}" / "best.pt")
    elif not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    ckpt_path = ckpt_path.resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    strict = load_strict_module(root)
    index_root = root / "experiments" / "data_v2" / "indices"
    stats_path = root / "experiments" / "data_v2" / "city_var_stats_train_only.json"
    cipda, cora = strict.load_project_modules(root, index_root, stats_path)
    Dataset = cora.Stage1O3CoRAContextDataset

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    cfg = checkpoint.get("args", {})
    time_feature_mode = cfg.get("time_feature_mode", "both")
    test_ds = Dataset("test", args.max_test_samples, args.seed + 2, time_feature_mode)
    loader = strict.make_loader(
        test_ds, args.batch_size, False, args.num_workers, device.type == "cuda")

    print(f"Loading Timer from {cipda.MODEL_PATH}", flush=True)
    timer = strict.AutoModelForCausalLM.from_pretrained(
        cipda.MODEL_PATH, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.float32,
    )
    model = strict.StrictAttributionForecaster(
        timer=timer, base_mode="timer",
        n_vars=len(test_ds.variables), n_cities=len(test_ds.cities),
        InputAdapter=cora.InputAdapter, ContextEncoder=cora.CoRAContextEncoder,
        time_feature_dim=cora.time_feature_dim,
        input_hidden=int(cfg.get("input_hidden", 64)),
        output_hidden=int(cfg.get("output_hidden", 128)),
        context_hidden=int(cfg.get("context_hidden", 128)),
        emb_dim=int(cfg.get("emb_dim", 16)),
        city_emb_dim=int(cfg.get("city_emb_dim", 16)),
        dropout=float(cfg.get("dropout", 0.05)),
        time_feature_mode=time_feature_mode,
        o3_idx=test_ds.o3_idx,
        corr_temperature=float(cfg.get("corr_temperature", 2.0)),
    ).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["adapter_state"], strict=False)
    bad_missing = [x for x in missing if not x.startswith("timer.")]
    if unexpected or bad_missing:
        raise RuntimeError(f"Checkpoint mismatch: missing={bad_missing}, unexpected={unexpected}")
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_samples = len(test_ds)
    cache_path = out_dir / "timer_base_cache.npy"
    city_path = out_dir / "test_city_ids.npy"

    print(f"seed={args.seed} test={n_samples:,} batch={args.batch_size} device={device}")
    print(f"checkpoint={ckpt_path}")
    print(f"params total={total_params:,}, trainable={trainable_params:,}")

    # Pass 1: cache sample-specific Timer forecasts produced by the fixed checkpoint.
    cache = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.float32, shape=(n_samples, H))
    city_ids = np.empty(n_samples, dtype=np.int16)
    offset = 0
    cache_t0 = time.perf_counter()
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            x = batch["x"].to(device, dtype=torch.float32, non_blocking=True)
            ci = batch["city_id"].to(device, non_blocking=True)
            x_adapt = model.input_adapter(x, ci)
            base = model.timer(x_adapt).logits[:, :H].float()
            b = x.shape[0]
            cache[offset:offset+b] = base.cpu().numpy()
            city_ids[offset:offset+b] = batch["city_id"].numpy().astype(np.int16)
            offset += b
            if step == 1 or step % 500 == 0:
                print(f"cache step={step}/{len(loader)} samples={offset:,}", flush=True)
    cache.flush()
    np.save(city_path, city_ids)
    cache_seconds = time.perf_counter() - cache_t0
    if offset != n_samples:
        raise RuntimeError(f"Cache size mismatch: {offset} != {n_samples}")

    permutation_seed = int(args.shuffle_seed + args.seed)
    permutation = make_cross_city_permutation(city_ids, permutation_seed)
    perm_path = out_dir / "cross_city_permutation.npy"
    np.save(perm_path, permutation)

    # Reopen read-only to avoid accidental modification.
    del cache
    cache = np.load(cache_path, mmap_mode="r")

    metrics: Dict[str, MetricBundle] = {
        name: MetricBundle(len(test_ds.cities)) for name in INTERVENTIONS
    }
    offset = 0
    eval_t0 = time.perf_counter()
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            x = batch["x"].to(device, dtype=torch.float32, non_blocking=True)
            xm = batch["x_multi"].to(device, dtype=torch.float32, non_blocking=True)
            tf = batch["time_feat"].to(device, dtype=torch.float32, non_blocking=True)
            vi = batch["var_idx"].to(device, non_blocking=True)
            ci = batch["city_id"].to(device, non_blocking=True)
            y_phys = batch["y_phys"].numpy()
            mu = batch["mu"].numpy()[:, None]
            std = batch["std"].numpy()[:, None]
            city_batch = batch["city_id"].numpy()
            b = x.shape[0]

            bases = {
                "correct_timer": torch.from_numpy(
                    np.array(cache[offset:offset+b], copy=True)).to(device),
                "zero_intervention": torch.zeros((b, H), device=device, dtype=x.dtype),
                "persistence_intervention": x[:, -1:].repeat(1, H),
                "raw_seasonal24_intervention": x[:, -24:].repeat(1, H // 24),
                "global_cross_city_shuffle": torch.from_numpy(
                    np.array(cache[permutation[offset:offset+b]], copy=True)).to(device),
            }

            for name, base in bases.items():
                pred_s = forward_with_base(model, x, xm, tf, vi, ci, base.float())
                pred = pred_s.cpu().numpy() * std + mu
                metrics[name].update(pred, y_phys, city_batch)

            offset += b
            if step == 1 or step % 500 == 0:
                print(f"eval step={step}/{len(loader)} samples={offset:,}", flush=True)
    eval_seconds = time.perf_counter() - eval_t0

    overall_rows = []
    city_rows = []
    horizon_rows = []
    for name in INTERVENTIONS:
        bundle = metrics[name]
        overall_rows.append({
            "intervention": name,
            "seed": args.seed,
            **bundle.overall(),
            "checkpoint": str(ckpt_path),
            "cache_seconds": cache_seconds,
            "evaluation_seconds": eval_seconds,
            "total_params": total_params,
            "trainable_params": trainable_params,
        })
        for cid, city in enumerate(test_ds.cities):
            n = int(bundle.city_n[cid])
            mse = bundle.city_sq[cid] / n if n else math.nan
            city_rows.append({
                "intervention": name, "seed": args.seed,
                "city_id": cid, "city": city,
                "MAE": bundle.city_abs[cid] / n if n else math.nan,
                "MSE": mse, "RMSE": math.sqrt(mse) if n else math.nan,
                "count": n,
            })
        for h in range(H):
            n = int(bundle.h_n[h])
            mse = bundle.h_sq[h] / n if n else math.nan
            horizon_rows.append({
                "intervention": name, "seed": args.seed, "horizon": h + 1,
                "MAE": bundle.h_abs[h] / n if n else math.nan,
                "MSE": mse, "RMSE": math.sqrt(mse) if n else math.nan,
                "count": n,
            })

    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(out_dir / "metrics_overall.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(city_rows).to_csv(
        out_dir / "metrics_by_city.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(horizon_rows).to_csv(
        out_dir / "metrics_by_horizon.csv", index=False, encoding="utf-8-sig")

    original_metrics_path = ckpt_path.parent / "metrics_overall.csv"
    sanity = {}
    if original_metrics_path.exists():
        original = pd.read_csv(original_metrics_path).iloc[0]
        correct = overall_df[overall_df["intervention"] == "correct_timer"].iloc[0]
        sanity = {
            "original_MAE": float(original["MAE"]),
            "recomputed_MAE": float(correct["MAE"]),
            "MAE_abs_diff": abs(float(original["MAE"]) - float(correct["MAE"])),
            "original_RMSE": float(original["RMSE"]),
            "recomputed_RMSE": float(correct["RMSE"]),
            "RMSE_abs_diff": abs(float(original["RMSE"]) - float(correct["RMSE"])),
        }
        pd.DataFrame([sanity]).to_csv(
            out_dir / "sanity_check.csv", index=False, encoding="utf-8-sig")
        if sanity["MAE_abs_diff"] > 1e-4 or sanity["RMSE_abs_diff"] > 1e-4:
            print("WARNING: correct_timer does not reproduce original metrics within 1e-4")

    meta = {
        "seed": args.seed,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "n_test_samples": n_samples,
        "prediction_points": n_samples * H,
        "batch_size": args.batch_size,
        "device": str(device),
        "shuffle_definition": "random donor from a different city for every recipient",
        "permutation_seed": permutation_seed,
        "cross_city_violations": int(np.sum(city_ids[permutation] == city_ids)),
        "cache_seconds": cache_seconds,
        "evaluation_seconds": eval_seconds,
        "sanity_check": sanity,
    }
    (out_dir / "intervention_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nFinal intervention metrics")
    print(overall_df[["intervention", "seed", "MAE", "RMSE", "count"]].to_string(index=False))
    print(f"cache_seconds={cache_seconds:.2f}, evaluation_seconds={eval_seconds:.2f}")

    if not args.keep_cache:
        cache_path.unlink(missing_ok=True)
        city_path.unlink(missing_ok=True)
        print("Temporary Timer cache removed; permutation and metrics were retained.")


if __name__ == "__main__":
    main()
