#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controller and merger for city-pooled/station-specific revision-v2 runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DEFAULT = Path(__file__).resolve().parents[3]
MODELS = [
    "TSMixer", "DLinear", "PatchTST", "LSTM", "iTransformer",
    "TimeMixer", "TimesNet", "CNNLSTM", "Transformer", "TimeXer", "Koopa",
]
H = 48


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["run", "status", "summarize"])
    p.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    p.add_argument("--scope", choices=["city", "station", "all"], default="all")
    p.add_argument("--models", nargs="*", default=MODELS)
    p.add_argument("--seeds", nargs="*", type=int, default=[2026])
    p.add_argument("--result_subdir", default="04_local_supervised_v2")
    p.add_argument("--max_units", type=int, default=0)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=-1)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--max_test_samples", type=int, default=0)
    p.add_argument("--require_cuda", action="store_true")
    p.add_argument("--keep_unit_arrays", type=int, choices=[0, 1], default=1)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_meta(root: Path):
    p = root / "experiments/data_v2/indices/meta.json"
    return json.loads(p.read_text(encoding="utf-8"))


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def units_for_scope(meta: dict, scope: str):
    if scope == "city":
        return [(c, f"city_{safe_name(c)}") for c in meta["cities"]]
    out = []
    for sm in meta["stations"]:
        aid = int(sm["array_id"])
        label = f"station_{aid:03d}_{safe_name(sm['city'])}_{safe_name(sm['station_id'])}"
        out.append((str(aid), label))
    return out


def model_seed_root(root: Path, subdir: str, scope: str, model: str, seed: int):
    return root / "experiments/results" / subdir / scope / f"{model}_seed{seed}"


def unit_complete(d: Path) -> bool:
    try:
        s = json.loads((d / "run_status.json").read_text(encoding="utf-8"))
        return (s.get("state") == "complete" and
                (d / "pred.npy").exists() and (d / "true.npy").exists() and
                (d / "sample_index.npy").exists())
    except Exception:
        return False


def aggregate_complete(d: Path) -> bool:
    try:
        s = json.loads((d / "aggregate_status.json").read_text(encoding="utf-8"))
        return s.get("state") == "complete" and (d / "pred.npy").exists()
    except Exception:
        return False


def metrics(pred: np.ndarray, true: np.ndarray):
    diff = pred.astype(np.float64) - true.astype(np.float64)
    mse = float(np.square(diff).mean())
    return {"MAE": float(np.abs(diff).mean()), "MSE": mse,
            "RMSE": math.sqrt(mse), "count": int(diff.size)}


def write_group_metrics(out_dir: Path, pred: np.ndarray, true: np.ndarray,
                        group: np.ndarray, meta: dict, group_kind: str):
    rows = []
    for gid in np.unique(group):
        mask = group == gid
        row = metrics(pred[mask], true[mask])
        if group_kind == "city":
            row = {"city_id": int(gid), "city": meta["cities"][int(gid)], **row}
        else:
            sm = meta["stations"][int(gid)]
            row = {"array_id": int(gid), "city": sm["city"],
                   "station_id": sm["station_id"], **row}
        rows.append(row)
    return pd.DataFrame(rows)


def merge_model_seed(root: Path, subdir: str, scope: str, model: str,
                     seed: int, units, keep_unit_arrays: bool):
    base = model_seed_root(root, subdir, scope, model, seed)
    if aggregate_complete(base):
        print(f"[MERGE SKIP] {scope} {model} seed={seed}")
        return
    test_idx = np.load(root / "experiments/data_v2/indices/test.npz",
                       allow_pickle=False)
    n = len(test_idx["array_id"])
    pred = np.full((n, H), np.nan, dtype=np.float32)
    true = np.full((n, H), np.nan, dtype=np.float32)
    forecast_ns = np.zeros(n, dtype=np.int64)
    coverage = np.zeros(n, dtype=np.int16)

    for _, label in units:
        d = base / "units" / label
        if not unit_complete(d):
            raise RuntimeError(f"Unit incomplete: {d}")
        idx = np.load(d / "sample_index.npy", allow_pickle=False).astype(np.int64)
        p = np.load(d / "pred.npy", allow_pickle=False).astype(np.float32)
        y = np.load(d / "true.npy", allow_pickle=False).astype(np.float32)
        f = np.load(d / "forecast_start_time_ns.npy", allow_pickle=False).astype(np.int64)
        if p.shape != y.shape or p.shape != (len(idx), H):
            raise RuntimeError(f"Shape mismatch in {d}: pred={p.shape} true={y.shape}")
        pred[idx] = p
        true[idx] = y
        forecast_ns[idx] = f
        coverage[idx] += 1

    if not np.all(coverage == 1):
        vals, cnt = np.unique(coverage, return_counts=True)
        raise RuntimeError(f"Coverage is not exactly one: {dict(zip(vals.tolist(), cnt.tolist()))}")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise RuntimeError("Merged arrays contain non-finite values")

    canonical_true = root / "experiments/results/03_zeroshot_baselines_v2_reindexed/Timer/true.npy"
    if not canonical_true.is_file():
        canonical_true = root / "experiments/data_v2/indices/test_targets.npy"
    if canonical_true.is_file():
        ref = np.load(canonical_true, mmap_mode="r")
        if ref.shape != true.shape or not np.array_equal(np.asarray(ref), true):
            raise RuntimeError("Merged true.npy differs from canonical revision-v2 true.npy")

    np.save(base / "pred.npy", pred, allow_pickle=False)
    np.save(base / "true.npy", true, allow_pickle=False)
    np.save(base / "sample_index.npy", np.arange(n, dtype=np.int64), allow_pickle=False)
    np.save(base / "city_id.npy", test_idx["city_id"].astype(np.int64), allow_pickle=False)
    np.save(base / "array_id.npy", test_idx["array_id"].astype(np.int32), allow_pickle=False)
    np.save(base / "start_idx.npy", test_idx["start_idx"].astype(np.int32), allow_pickle=False)
    np.save(base / "forecast_start_time_ns.npy", forecast_ns, allow_pickle=False)

    meta = load_meta(root)
    overall = {"scope": scope, "model_name": model, "seed": seed, **metrics(pred, true)}
    city_df = write_group_metrics(base, pred, true, test_idx["city_id"], meta, "city")
    station_df = write_group_metrics(base, pred, true, test_idx["array_id"], meta, "station")
    hrows = []
    for h in range(H):
        hrows.append({"horizon": h + 1, **metrics(pred[:, h:h+1], true[:, h:h+1])})
    city_df.to_csv(base / "metrics_by_city.csv", index=False, encoding="utf-8-sig")
    station_df.to_csv(base / "metrics_by_station.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(hrows).to_csv(base / "metrics_by_horizon.csv", index=False,
                                encoding="utf-8-sig")
    overall.update({
        "city_MAE_macro": float(city_df.MAE.mean()),
        "city_RMSE_macro": float(city_df.RMSE.mean()),
        "station_MAE_macro": float(station_df.MAE.mean()),
        "station_RMSE_macro": float(station_df.RMSE.mean()),
        "n_units": len(units), "n_test_samples": n,
    })
    pd.DataFrame([overall]).to_csv(base / "metrics_overall.csv", index=False,
                                    encoding="utf-8-sig")
    (base / "aggregate_manifest.json").write_text(json.dumps({
        "protocol": "revision-v2", "scope": scope, "model_name": model,
        "seed": seed, "n_units": len(units), "n_test_samples": n,
        "coverage": "exactly one prediction per canonical test sample",
        "true_verified_against_timer": canonical_true.exists(),
        "hyperparameter_policy": "reuse global-selected hyperparameters; no local retuning",
        "normalization": "same city-level train-only scaler as global revision-v2",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (base / "aggregate_status.json").write_text(json.dumps({
        "state": "complete", "scope": scope, "model_name": model,
        "seed": seed, "n_units": len(units), "n_test_samples": n,
        "MAE": overall["MAE"], "RMSE": overall["RMSE"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if not keep_unit_arrays:
        for _, label in units:
            d = base / "units" / label
            for name in ["pred.npy", "true.npy", "city_id.npy", "array_id.npy",
                         "sample_index.npy", "start_idx.npy",
                         "forecast_start_time_ns.npy"]:
                (d / name).unlink(missing_ok=True)
    print(f"[MERGED] {scope} {model} seed={seed} MAE={overall['MAE']:.6f} RMSE={overall['RMSE']:.6f}")


def run_scope(args, scope: str):
    root = args.root.expanduser().resolve()
    meta = load_meta(root)
    units = units_for_scope(meta, scope)
    if args.max_units > 0:
        units = units[:args.max_units]
    trainer = root / "experiments/scripts/train/train_supervised_scopes.py"

    for model in args.models:
        if model not in MODELS:
            raise ValueError(f"Unsupported model: {model}")
        for seed in args.seeds:
            base = model_seed_root(root, args.result_subdir, scope, model, seed)
            (base / "units").mkdir(parents=True, exist_ok=True)
            for pos, (unit, label) in enumerate(units, 1):
                d = base / "units" / label
                if unit_complete(d) and not args.force:
                    print(f"[SKIP {pos}/{len(units)}] {scope} {model} seed={seed} {unit}")
                    continue
                cmd = [
                    sys.executable, str(trainer), "--root", str(root),
                    "--scope", scope, "--unit", unit,
                    "--model_name", model, "--seed", str(seed),
                    "--out_dir", str(d), "--keep_checkpoint", "0",
                ]
                if args.require_cuda:
                    cmd.append("--require_cuda")
                for key in ["epochs", "patience", "batch_size", "max_train_samples",
                            "max_val_samples", "max_test_samples"]:
                    value = getattr(args, key)
                    if value > 0:
                        cmd += [f"--{key}", str(value)]
                if args.num_workers >= 0:
                    cmd += ["--num_workers", str(args.num_workers)]
                if args.force:
                    cmd.append("--force")
                print("\n" + "=" * 90)
                print(f"[{pos}/{len(units)}] {scope} {model} seed={seed} unit={unit}")
                print(" ".join(cmd))
                print("=" * 90, flush=True)
                subprocess.run(cmd, check=True)
            if args.max_units > 0:
                print(
                    f"[PARTIAL/SMOKE] completed {len(units)} unit(s) for "
                    f"{scope} {model} seed={seed}; full-test merge intentionally skipped."
                )
            else:
                merge_model_seed(root, args.result_subdir, scope, model, seed,
                                 units, bool(args.keep_unit_arrays))


def summarize(args):
    root = args.root.expanduser().resolve()
    base = root / "experiments/results" / args.result_subdir
    rows = []
    for p in base.glob("*/*_seed*/metrics_overall.csv"):
        try:
            df = pd.read_csv(p)
            if not df.empty:
                row = df.iloc[0].to_dict()
                row["result_dir"] = str(p.parent)
                rows.append(row)
        except Exception as e:
            print(f"[WARN] {p}: {e}")
    if not rows:
        print("No aggregate metrics found.")
        return
    out = pd.DataFrame(rows).sort_values(["scope", "model_name", "seed"])
    out.to_csv(base / "summary_all_runs.csv", index=False, encoding="utf-8-sig")
    summary = (out.groupby(["scope", "model_name"], as_index=False)
               .agg(n_seeds=("seed", "nunique"),
                    MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
                    RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
                    count=("count", "first")))
    summary.to_csv(base / "summary_by_scope_model.csv", index=False,
                   encoding="utf-8-sig")
    print(summary.to_string(index=False))


def status(args):
    root = args.root.expanduser().resolve()
    meta = load_meta(root)
    scopes = ["city", "station"] if args.scope == "all" else [args.scope]
    for scope in scopes:
        units = units_for_scope(meta, scope)
        if args.max_units > 0:
            units = units[:args.max_units]
        print("\n" + "=" * 88)
        print(f"{scope.upper()} expected_units={len(units)}")
        for model in args.models:
            for seed in args.seeds:
                base = model_seed_root(root, args.result_subdir, scope, model, seed)
                done = sum(unit_complete(base / "units" / label) for _, label in units)
                agg = "COMPLETE" if aggregate_complete(base) else "pending"
                print(f"{model:<13} seed={seed} units={done:>3}/{len(units):<3} aggregate={agg}")


def main():
    args = parse_args()
    if args.command == "run":
        scopes = ["city", "station"] if args.scope == "all" else [args.scope]
        for scope in scopes:
            run_scope(args, scope)
        summarize(args)
    elif args.command == "status":
        status(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
