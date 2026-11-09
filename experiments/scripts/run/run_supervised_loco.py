#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controller/status/summarizer for 11 supervised Global baselines under strict LOCO."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DEFAULT = Path(__file__).resolve().parents[3]
SEEDS = [2026, 2027, 2028, 2029, 2030]
MODELS = [
    "TSMixer", "DLinear", "PatchTST", "LSTM", "iTransformer",
    "TimeMixer", "TimesNet", "CNNLSTM", "Transformer", "TimeXer", "Koopa",
]
H = 48


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["setup", "smoke", "formal", "status", "summarize"])
    p.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    p.add_argument("--models", nargs="*", choices=MODELS, default=None)
    p.add_argument("--cities", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=0,
                   help="0 preserves each model's Global11 selected value")
    p.add_argument("--patience", type=int, default=0,
                   help="0 preserves each model's Global11 selected value")
    p.add_argument("--batch_size", type=int, default=0,
                   help="0 preserves each model's Global11 selected value")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def result_root(root: Path) -> Path:
    return root / "experiments/results/16_loco_global11_v2"


def fold_root(root: Path, city: str) -> Path:
    return (
        root / "experiments/results/14_loco_22cities_v2/00_fold_indices"
        / safe_name(city)
    )


def script(root: Path, rel: str) -> Path:
    return root / "experiments/scripts" / rel


def all_cities(root: Path) -> list[str]:
    meta = json.loads(
        (root / "experiments/data_v2/indices/meta.json").read_text(encoding="utf-8")
    )
    cities = [str(x) for x in meta["cities"]]
    if len(cities) != 22:
        raise RuntimeError(f"Expected 22 canonical cities, found {len(cities)}")
    return cities


def selections(args: argparse.Namespace) -> tuple[list[str], list[str], list[int]]:
    root = args.root.expanduser().resolve()
    models = args.models or MODELS
    cities = args.cities or all_cities(root)
    seeds = args.seeds or SEEDS
    unknown = sorted(set(cities) - set(all_cities(root)))
    if unknown:
        raise ValueError(f"Unknown cities: {unknown}")
    return list(models), list(cities), list(seeds)


def setup(root: Path) -> dict[str, Any]:
    out = result_root(root)
    for rel in ["00_smoke_gpu", "01_runs", "02_summary", "logs"]:
        (out / rel).mkdir(parents=True, exist_ok=True)
    design = {
        "protocol": "revision-v2 strict LOCO for 11 Global supervised baselines",
        "models": MODELS,
        "model_count": 11,
        "city_count": 22,
        "seeds": SEEDS,
        "expected_formal_runs": 11 * 22 * 5,
        "train_val": "21 source cities",
        "test": "one held-out city",
        "normalization": "source-21 pooled train-only",
        "city_identity_mode": "shared_single_id",
        "target_city_stats_used": False,
        "hyperparameter_policy": "reuse completed Global11 final-run args; no LOCO retuning",
        "fold_source": str(
            root / "experiments/results/14_loco_22cities_v2/00_fold_indices"
        ),
    }
    (out / "experiment_design.json").write_text(
        json.dumps(design, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return design


def launch(cmd: list[str]) -> None:
    print("\n" + "=" * 110, flush=True)
    print(" ".join(cmd), flush=True)
    print("=" * 110, flush=True)
    subprocess.run(cmd, check=True)


def prepare_folds(root: Path) -> None:
    launch([
        sys.executable,
        str(script(root, "data/build_loco_folds.py")),
        "--root", str(root),
    ])


def run_dir(root: Path, model: str, city: str, seed: int, smoke: bool = False) -> Path:
    if smoke:
        return result_root(root) / "00_smoke_gpu" / safe_name(model) / safe_name(city) / f"seed{seed}"
    return result_root(root) / "01_runs" / safe_name(model) / safe_name(city) / f"seed{seed}"


def run_complete(directory: Path, formal: bool) -> bool:
    status = directory / "run_status.json"
    if not status.exists():
        return False
    try:
        s = json.loads(status.read_text(encoding="utf-8"))
    except Exception:
        return False
    if s.get("state") != "complete":
        return False
    required = ["metrics_overall.csv", "pred.npy", "true.npy"]
    if formal:
        required += ["canonical_test_positions.npy", "run_manifest.json", "metrics_by_horizon.csv"]
        if s.get("formal_full_fold") is not True:
            return False
    return all((directory / name).exists() for name in required)


def trainer_command(
    root: Path,
    model: str,
    city: str,
    seed: int,
    out: Path,
    args: argparse.Namespace,
    smoke: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(script(root, "train/train_supervised_loco.py")),
        "--root", str(root),
        "--heldout_city", city,
        "--model_name", model,
        "--seed", str(seed),
        "--out_dir", str(out),
        "--num_workers", str(0 if smoke else args.num_workers),
        "--keep_checkpoint", "0",
        "--require_cuda",
    ]
    if smoke:
        cmd += [
            "--epochs", "1", "--patience", "1", "--batch_size", "16",
            "--max_train_samples", "512",
            "--max_val_samples", "256",
            "--max_test_samples", "256",
        ]
    else:
        if args.epochs > 0:
            cmd += ["--epochs", str(args.epochs)]
        if args.patience > 0:
            cmd += ["--patience", str(args.patience)]
        if args.batch_size > 0:
            cmd += ["--batch_size", str(args.batch_size)]
    if args.force:
        cmd.append("--force")
    return cmd


def smoke(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup(root)
    prepare_folds(root)
    models, cities, _ = selections(args)
    city = cities[0]
    seed = 2026
    for pos, model in enumerate(models, 1):
        out = run_dir(root, model, city, seed, smoke=True)
        if run_complete(out, formal=False) and not args.force:
            print(f"[SMOKE SKIP] {model} {city}")
            continue
        print(f"\n[SMOKE {pos}/{len(models)}] model={model} city={city}")
        launch(trainer_command(root, model, city, seed, out, args, smoke=True))
    print("\nAll requested Global11 LOCO GPU smokes completed.")


def formal(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup(root)
    prepare_folds(root)
    models, cities, seeds = selections(args)
    total = len(models) * len(cities) * len(seeds)
    pos = 0

    # Model-major order: one comparison model becomes complete as early as possible.
    for model in models:
        for city in cities:
            for seed in seeds:
                pos += 1
                out = run_dir(root, model, city, seed, smoke=False)
                if run_complete(out, formal=True) and not args.force:
                    print(f"[SKIP {pos}/{total}] {model} {city} seed={seed}", flush=True)
                    continue
                print(
                    f"\n######## [{pos}/{total}] model={model} "
                    f"heldout={city} seed={seed} ########",
                    flush=True,
                )
                launch(trainer_command(root, model, city, seed, out, args, smoke=False))


def read_one(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if len(df) != 1:
        raise RuntimeError(f"Expected exactly one row: {path}")
    return df.iloc[0].to_dict()


def audit_one(root: Path, model: str, city: str, seed: int) -> dict[str, Any]:
    d = run_dir(root, model, city, seed, smoke=False)
    row: dict[str, Any] = {
        "model_name": model,
        "heldout_city": city,
        "seed": seed,
        "run_dir": str(d),
        "complete": run_complete(d, formal=True),
    }
    if not row["complete"]:
        row.update({
            "audit_ok": False,
            "shape_ok": False,
            "true_canonical_equal": False,
            "positions_canonical_equal": False,
            "metric_match": False,
        })
        return row

    metric = read_one(d / "metrics_overall.csv")
    pred = np.load(d / "pred.npy", mmap_mode="r")
    true = np.load(d / "true.npy", mmap_mode="r")
    pos = np.load(d / "canonical_test_positions.npy", mmap_mode="r")
    fold = fold_root(root, city)
    canonical_true = np.load(fold / "canonical_test_true.npy", mmap_mode="r")
    canonical_pos = np.load(fold / "canonical_test_positions.npy", mmap_mode="r")

    diff = np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt(np.square(diff).mean()))
    shape_ok = pred.shape == true.shape == canonical_true.shape and pred.ndim == 2 and pred.shape[1] == H
    true_equal = bool(np.array_equal(np.asarray(true), np.asarray(canonical_true), equal_nan=True))
    pos_equal = bool(np.array_equal(np.asarray(pos), np.asarray(canonical_pos)))
    metric_match = abs(mae - float(metric["MAE"])) <= 5e-5 and abs(rmse - float(metric["RMSE"])) <= 5e-5
    nonfinite_pred = int((~np.isfinite(pred)).sum())
    nonfinite_true = int((~np.isfinite(true)).sum())

    fold_manifest = json.loads((fold / "fold_manifest.json").read_text(encoding="utf-8"))
    no_leakage = all([
        fold_manifest.get("target_city_in_train") is False,
        fold_manifest.get("target_city_in_val") is False,
        fold_manifest.get("target_city_stats_used") is False,
        int(fold_manifest.get("source_city_count", -1)) == 21,
        city not in fold_manifest.get("source_cities", []),
        metric.get("city_identity_mode") == "shared_single_id",
        metric.get("normalization") == "source-21 pooled train-only",
        metric.get("target_city_stats_used") in [False, 0, "False", "false"],
    ])

    row.update(metric)
    row.update({
        "shape_ok": shape_ok,
        "true_canonical_equal": true_equal,
        "positions_canonical_equal": pos_equal,
        "metric_match": metric_match,
        "nonfinite_pred": nonfinite_pred,
        "nonfinite_true": nonfinite_true,
        "array_MAE_check": mae,
        "array_RMSE_check": rmse,
        "abs_error_sum_check": float(np.abs(diff).sum(dtype=np.float64)),
        "sq_error_sum_check": float(np.square(diff).sum(dtype=np.float64)),
        "scalar_count_check": int(diff.size),
        "no_leakage": bool(no_leakage),
    })
    row["audit_ok"] = all([
        shape_ok, true_equal, pos_equal, metric_match,
        nonfinite_pred == 0, nonfinite_true == 0, no_leakage,
    ])
    return row


def summarize(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup(root)
    models, cities, seeds = selections(args)
    out = result_root(root) / "02_summary"

    rows = []
    for model in models:
        for city in cities:
            for seed in seeds:
                rows.append(audit_one(root, model, city, seed))
    run_df = pd.DataFrame(rows)
    run_df.to_csv(out / "run_metrics_audit.csv", index=False, encoding="utf-8-sig")

    completed = run_df[run_df["complete"] == True].copy()  # noqa: E712
    if completed.empty:
        raise RuntimeError("No completed formal runs to summarize")

    city_summary = (
        completed.groupby(["model_name", "heldout_city"], as_index=False)
        .agg(
            n_seeds=("MAE", "count"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            n_test_samples=("n_test", "first"),
            audit_ok_runs=("audit_ok", "sum"),
        )
    )
    city_summary.to_csv(out / "city_summary.csv", index=False, encoding="utf-8-sig")

    # Macro: every held-out city receives equal weight.
    macro_by_seed = (
        completed.groupby(["model_name", "seed"], as_index=False)
        .agg(
            city_count=("heldout_city", "nunique"),
            Macro_MAE=("MAE", "mean"),
            Macro_RMSE=("RMSE", "mean"),
        )
    )
    macro_by_seed.to_csv(out / "macro_by_seed.csv", index=False, encoding="utf-8-sig")

    macro_summary = (
        macro_by_seed.groupby("model_name", as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            Macro_MAE_mean=("Macro_MAE", "mean"),
            Macro_MAE_std=("Macro_MAE", "std"),
            Macro_RMSE_mean=("Macro_RMSE", "mean"),
            Macro_RMSE_std=("Macro_RMSE", "std"),
        )
    )
    macro_summary.to_csv(out / "macro_summary.csv", index=False, encoding="utf-8-sig")

    # Micro: pool every scalar prediction from all 22 held-out cities per seed.
    micro_rows = []
    for model in models:
        for seed in seeds:
            part = completed[
                (completed["model_name"] == model) & (completed["seed"] == seed)
            ]
            if part.empty:
                continue
            count = float(part["scalar_count_check"].sum())
            abs_sum = float(part["abs_error_sum_check"].sum())
            sq_sum = float(part["sq_error_sum_check"].sum())
            micro_rows.append({
                "model_name": model,
                "seed": seed,
                "city_count": int(part["heldout_city"].nunique()),
                "Micro_MAE": abs_sum / count,
                "Micro_MSE": sq_sum / count,
                "Micro_RMSE": math.sqrt(sq_sum / count),
                "scalar_count": int(count),
            })
    micro_df = pd.DataFrame(micro_rows)
    micro_df.to_csv(out / "micro_by_seed.csv", index=False, encoding="utf-8-sig")

    micro_summary = (
        micro_df.groupby("model_name", as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            Micro_MAE_mean=("Micro_MAE", "mean"),
            Micro_MAE_std=("Micro_MAE", "std"),
            Micro_RMSE_mean=("Micro_RMSE", "mean"),
            Micro_RMSE_std=("Micro_RMSE", "std"),
        )
        .sort_values("Micro_MAE_mean")
    )
    micro_summary.to_csv(out / "micro_summary.csv", index=False, encoding="utf-8-sig")

    comparison = micro_summary.merge(macro_summary, on=["model_name", "seed_count"], how="outer")
    comparison = comparison.sort_values("Micro_MAE_mean")
    comparison.to_csv(out / "paper_table_global11_loco.csv", index=False, encoding="utf-8-sig")

    expected = len(models) * len(cities) * len(seeds)
    manifest = {
        "state": "complete" if (
            len(run_df) == expected
            and int(run_df["complete"].sum()) == expected
            and int(run_df["audit_ok"].sum()) == expected
        ) else "incomplete",
        "protocol": "revision-v2 strict LOCO Global11",
        "models": models,
        "cities": cities,
        "seeds": seeds,
        "expected_runs": expected,
        "completed_runs": int(run_df["complete"].sum()),
        "audit_ok_runs": int(run_df["audit_ok"].sum()),
        "city_identity_mode": "shared_single_id",
        "normalization": "source-21 pooled train-only",
        "target_city_stats_used": False,
    }
    (out / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nMICRO SUMMARY (all held-out test scalars pooled per seed)")
    print(micro_summary.to_string(index=False))
    print("\nMACRO SUMMARY (22 held-out cities equally weighted)")
    print(macro_summary.sort_values("Macro_MAE_mean").to_string(index=False))
    print("\n" + json.dumps(manifest, indent=2, ensure_ascii=False))

    if manifest["state"] != "complete":
        raise SystemExit("ERROR: selected LOCO Global11 set is incomplete or failed audit")


def status(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup(root)
    models, cities, seeds = selections(args)
    total = len(models) * len(cities) * len(seeds)
    done_total = 0
    print("\nGlobal11 strict-LOCO status")
    print("-" * 90)
    for model in models:
        expected = len(cities) * len(seeds)
        done = sum(
            run_complete(run_dir(root, model, city, seed), formal=True)
            for city in cities for seed in seeds
        )
        done_total += done
        print(f"{model:<14} {done:>4}/{expected:<4} ({100.0*done/max(expected,1):6.2f}%)")
    print("-" * 90)
    print(f"TOTAL          {done_total}/{total} ({100.0*done_total/max(total,1):.2f}%)")


def main() -> None:
    args = parse_args()
    args.root = args.root.expanduser().resolve()
    if args.command == "setup":
        setup(args.root)
        prepare_folds(args.root)
    elif args.command == "smoke":
        smoke(args)
    elif args.command == "formal":
        formal(args)
    elif args.command == "status":
        status(args)
    elif args.command == "summarize":
        summarize(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
