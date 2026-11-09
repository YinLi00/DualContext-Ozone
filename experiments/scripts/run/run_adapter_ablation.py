#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controller for revision-v2 progressive-adapter and time-feature ablations."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODELS = [
    "p4_io_time_context",
    "p1_o3_output",
    "p2_context",
    "p3_time_context",
    "t0_none",
    "t1_hour",
    "t2_doy",
]
SEEDS = [2026, 2027, 2028, 2029, 2030]
EXPECTED_SHAPE = (106_200, 48)

RELATIVE_DIRS = {
    "p1_o3_output": "01_progressive_adapter/p1_o3_output",
    "p2_context": "01_progressive_adapter/p2_context",
    "p3_time_context": "01_progressive_adapter/p3_time_context",
    "p4_io_time_context": "01_progressive_adapter/p4_io_time_context",
    "t0_none": "02_time_features/t0_none",
    "t1_hour": "02_time_features/t1_hour",
    "t2_doy": "02_time_features/t2_doy",
}

PAPER_NAMES = {
    "p0_timer_zeroshot": "Timer zero-shot",
    "p1_o3_output": "O3-output adapter",
    "p2_context": "Context adapter",
    "p3_time_context": "Time-context adapter",
    "p4_io_time_context": "IO-Time-Context adapter",
    "t0_none": "None",
    "t1_hour": "Hour only",
    "t2_doy": "DOY only",
    "t3_hour_doy": "Hour + DOY",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "command",
        choices=["setup", "smoke", "formal", "summarize", "status"],
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    p.add_argument("--models", nargs="*", default=MODELS)
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--cpu_threads", type=int, default=4)
    return p.parse_args()


def result_root(root: Path) -> Path:
    return root / "experiments/results/10_adapter_time_ablation_v2"


def trainer(root: Path) -> Path:
    return (
        root / "experiments/scripts/train/"
        "train_adapter_ablation.py"
    )


def safe_symlink(target: Path, link: Path) -> str:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return "existing"
        link.unlink()
    elif link.exists():
        return "real_directory_preserved"
    relative = os.path.relpath(target.resolve(), link.parent.resolve())
    os.symlink(relative, link)
    return "created"


def setup_tree(root: Path) -> dict[str, Any]:
    out = result_root(root)
    for rel in [
        "00_smoke_cpu",
        "01_progressive_adapter/p1_o3_output",
        "01_progressive_adapter/p2_context",
        "01_progressive_adapter/p3_time_context",
        "01_progressive_adapter/p4_io_time_context",
        "02_time_features/t0_none",
        "02_time_features/t1_hour",
        "02_time_features/t2_doy",
        "03_summary",
        "logs",
    ]:
        (out / rel).mkdir(parents=True, exist_ok=True)

    timer_source = (
        root / "experiments/results/"
        "03_zeroshot_baselines_v2_archive/Timer"
    )
    if not timer_source.exists():
        timer_source = (
            root / "experiments/results/"
            "03_zeroshot_baselines_v2_reindexed/Timer"
        )
    p0_link = out / "01_progressive_adapter/p0_timer_zeroshot"
    p0_status = safe_symlink(timer_source, p0_link)

    p4 = out / "01_progressive_adapter/p4_io_time_context"
    t3 = out / "02_time_features/t3_hour_doy"
    t3_status = safe_symlink(p4, t3)

    design = {
        "protocol": "revision-v2",
        "input_length": 96,
        "prediction_length": 48,
        "seeds": SEEDS,
        "unique_trainable_variants": MODELS,
        "n_formal_trainings": len(MODELS) * len(SEEDS),
        "p0_timer_zeroshot": str(timer_source),
        "t3_reuses_p4": True,
        "p0_link_status": p0_status,
        "t3_link_status": t3_status,
        "formal_order": "seed-major; P4 first, then P1/P2/P3/T0/T1/T2",
        "fixed_hyperparameters": {
            "epochs": 20,
            "patience": 5,
            "batch_size": 64,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "mse_weight": 0.5,
            "grad_clip": 1.0,
            "input_hidden": 64,
            "output_hidden": 128,
            "context_hidden": 128,
            "emb_dim": 16,
            "city_emb_dim": 16,
            "dropout": 0.05,
            "corr_temperature": 2.0,
            "timer_frozen": True,
        },
    }
    (out / "experiment_design.json").write_text(
        json.dumps(design, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return design


def run_dir(root: Path, variant: str, seed: int, smoke: bool) -> Path:
    out = result_root(root)
    if smoke:
        return out / "00_smoke_cpu" / f"{variant}_seed{seed}"
    return out / RELATIVE_DIRS[variant] / f"seed{seed}"


def is_complete(path: Path, formal: bool) -> bool:
    status = path / "run_status.json"
    if not status.exists():
        return False
    try:
        state = json.loads(status.read_text(encoding="utf-8")).get("state")
    except Exception:
        return False
    if state != "complete":
        return False
    required = ["metrics_overall.csv"]
    if formal:
        required += ["pred.npy", "true.npy", "array_manifest.json"]
    return all((path / name).exists() for name in required)


def launch(cmd: list[str]) -> None:
    print("\n" + "=" * 100, flush=True)
    print(" ".join(cmd), flush=True)
    print("=" * 100, flush=True)
    subprocess.run(cmd, check=True)


def run_smoke(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup_tree(root)
    for variant in MODELS:
        out = run_dir(root, variant, 2026, True)
        if is_complete(out, formal=False):
            print(f"[SMOKE SKIP] {variant}", flush=True)
            continue
        cmd = [
            sys.executable,
            str(trainer(root)),
            "--root", str(root),
            "--variant", variant,
            "--seed", "2026",
            "--epochs", "1",
            "--patience", "1",
            "--batch_size", "16",
            "--num_workers", "0",
            "--max_train_samples", "128",
            "--max_val_samples", "64",
            "--max_test_samples", "64",
            "--device", "cpu",
            "--cpu_threads", str(args.cpu_threads),
            "--save_arrays", "1",
            "--keep_checkpoint", "0",
            "--log_every", "1",
            "--out_dir", str(out),
        ]
        launch(cmd)
    print("\nCPU smoke complete: 7/7 variants.", flush=True)


def run_formal(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup_tree(root)
    unknown = sorted(set(args.models) - set(MODELS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    for seed in args.seeds:
        for variant in args.models:
            out = run_dir(root, variant, seed, False)
            if is_complete(out, formal=True):
                print(f"[FORMAL SKIP] {variant} seed={seed}", flush=True)
                continue
            cmd = [
                sys.executable,
                str(trainer(root)),
                "--root", str(root),
                "--variant", variant,
                "--seed", str(seed),
                "--epochs", "20",
                "--patience", "5",
                "--batch_size", "64",
                "--num_workers", str(args.num_workers),
                "--lr", "0.001",
                "--weight_decay", "0.0001",
                "--mse_weight", "0.5",
                "--grad_clip", "1.0",
                "--input_hidden", "64",
                "--output_hidden", "128",
                "--context_hidden", "128",
                "--emb_dim", "16",
                "--city_emb_dim", "16",
                "--dropout", "0.05",
                "--corr_temperature", "2.0",
                "--device", "cuda",
                "--require_cuda",
                "--save_arrays", "1",
                "--keep_checkpoint", "1",
                "--log_every", "1000",
                "--out_dir", str(out),
            ]
            launch(cmd)


def read_metric(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if len(df) != 1:
        raise RuntimeError(f"Expected one row in {path}")
    return df.iloc[0].to_dict()


def canonical_true_path(root: Path) -> Path:
    path = (
        root / "experiments/results/"
        "03_zeroshot_baselines_v2_archive/Timer/true.npy"
    )
    if not path.exists():
        path = (
            root / "experiments/results/"
            "03_zeroshot_baselines_v2_reindexed/Timer/true.npy"
        )
    return path if path.is_file() else root / "experiments/data_v2/indices/test_targets.npy"


def summarize(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    out = result_root(root)
    setup_tree(root)
    canonical_path = canonical_true_path(root)
    canonical = np.load(canonical_path, mmap_mode="r")

    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for variant in MODELS:
        for seed in SEEDS:
            directory = run_dir(root, variant, seed, False)
            metric_path = directory / "metrics_overall.csv"
            pred_path = directory / "pred.npy"
            true_path = directory / "true.npy"
            row = {
                "variant": variant,
                "paper_name": PAPER_NAMES[variant],
                "seed": seed,
                "run_dir": str(directory),
                "complete": is_complete(directory, formal=True),
            }
            if metric_path.exists():
                row.update(read_metric(metric_path))
            if pred_path.exists() and true_path.exists():
                pred = np.load(pred_path, mmap_mode="r")
                true = np.load(true_path, mmap_mode="r")
                row["pred_shape"] = str(tuple(pred.shape))
                row["true_shape"] = str(tuple(true.shape))
                row["shape_ok"] = (
                    tuple(pred.shape) == EXPECTED_SHAPE
                    and tuple(true.shape) == EXPECTED_SHAPE
                )
                row["true_canonical_equal"] = (
                    tuple(true.shape) == tuple(canonical.shape)
                    and np.array_equal(
                        np.asarray(true), np.asarray(canonical), equal_nan=True
                    )
                )
                row["nonfinite_pred"] = int((~np.isfinite(pred)).sum())
                row["nonfinite_true"] = int((~np.isfinite(true)).sum())
            else:
                row["shape_ok"] = False
                row["true_canonical_equal"] = False

            checks = [
                bool(row["complete"]),
                bool(row.get("shape_ok", False)),
                bool(row.get("true_canonical_equal", False)),
                int(row.get("nonfinite_pred", -1)) == 0,
                int(row.get("nonfinite_true", -1)) == 0,
            ]
            row["audit_ok"] = all(checks)
            if not row["audit_ok"]:
                issues.append({
                    "variant": variant,
                    "seed": seed,
                    "run_dir": str(directory),
                    "row": json.dumps(row, ensure_ascii=False, default=str),
                })
            rows.append(row)

    seed_df = pd.DataFrame(rows)
    seed_df.to_csv(
        out / "03_summary/all_trainable_seed_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(issues).to_csv(
        out / "03_summary/audit_issues.csv",
        index=False,
        encoding="utf-8-sig",
    )

    def aggregate(variant_order: list[str]) -> pd.DataFrame:
        agg_rows = []
        for variant in variant_order:
            part = seed_df[seed_df["variant"] == variant]
            mae = pd.to_numeric(part.get("MAE"), errors="coerce").dropna()
            rmse = pd.to_numeric(part.get("RMSE"), errors="coerce").dropna()
            agg_rows.append({
                "variant": variant,
                "paper_name": PAPER_NAMES[variant],
                "n": len(mae),
                "MAE_mean": float(mae.mean()) if len(mae) else math.nan,
                "MAE_std": float(mae.std(ddof=1)) if len(mae) > 1 else math.nan,
                "RMSE_mean": float(rmse.mean()) if len(rmse) else math.nan,
                "RMSE_std": (
                    float(rmse.std(ddof=1)) if len(rmse) > 1 else math.nan
                ),
                "audit_ok_runs": int(part["audit_ok"].sum()),
            })
        return pd.DataFrame(agg_rows)

    progressive = aggregate([
        "p1_o3_output",
        "p2_context",
        "p3_time_context",
        "p4_io_time_context",
    ])

    timer_metric = read_metric(
        root / "experiments/results/"
        "03_zeroshot_baselines_v2_archive/Timer/"
        "metrics_recomputed_from_arrays.csv"
    )
    p0 = pd.DataFrame([{
        "variant": "p0_timer_zeroshot",
        "paper_name": PAPER_NAMES["p0_timer_zeroshot"],
        "n": 1,
        "MAE_mean": float(timer_metric["MAE"]),
        "MAE_std": math.nan,
        "RMSE_mean": float(timer_metric["RMSE"]),
        "RMSE_std": math.nan,
        "audit_ok_runs": 1,
    }])
    progressive = pd.concat([p0, progressive], ignore_index=True)
    progressive["MAE_reduction_vs_P0_pct"] = (
        (progressive.loc[0, "MAE_mean"] - progressive["MAE_mean"])
        / progressive.loc[0, "MAE_mean"] * 100.0
    )
    progressive["RMSE_reduction_vs_P0_pct"] = (
        (progressive.loc[0, "RMSE_mean"] - progressive["RMSE_mean"])
        / progressive.loc[0, "RMSE_mean"] * 100.0
    )
    progressive.to_csv(
        out / "03_summary/progressive_summary_5seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    time_summary = aggregate(["t0_none", "t1_hour", "t2_doy"])
    p4_row = aggregate(["p4_io_time_context"]).iloc[0].to_dict()
    p4_row["variant"] = "t3_hour_doy"
    p4_row["paper_name"] = PAPER_NAMES["t3_hour_doy"]
    time_summary = pd.concat(
        [time_summary, pd.DataFrame([p4_row])],
        ignore_index=True,
    )
    time_summary.to_csv(
        out / "03_summary/time_feature_summary_5seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "protocol": "revision-v2",
        "expected_unique_trainings": 35,
        "completed_runs": int(seed_df["complete"].sum()),
        "audit_ok_runs": int(seed_df["audit_ok"].sum()),
        "issues": len(issues),
        "p0_reused_from_zero_shot": True,
        "t3_reused_from_p4": True,
        "canonical_true": str(canonical_path),
        "expected_shape": list(EXPECTED_SHAPE),
    }
    (out / "03_summary/archive_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nProgressive adapter summary")
    print(progressive.to_string(index=False))
    print("\nTime feature summary")
    print(time_summary.to_string(index=False))
    print("\n" + json.dumps(manifest, indent=2, ensure_ascii=False))

    if manifest["completed_runs"] != 35 or manifest["audit_ok_runs"] != 35:
        raise SystemExit("ERROR: formal archive is not complete or audit-clean")


def status(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup_tree(root)
    print("=" * 92)
    print("CPU SMOKE")
    for variant in MODELS:
        directory = run_dir(root, variant, 2026, True)
        print(
            f"{variant:<20} "
            f"{'COMPLETE' if is_complete(directory, False) else 'pending'}"
        )
    print("\n" + "=" * 92)
    print("FORMAL GPU")
    complete = 0
    for variant in MODELS:
        parts = []
        for seed in SEEDS:
            ok = is_complete(run_dir(root, variant, seed, False), True)
            complete += int(ok)
            parts.append(f"{seed}:{'OK' if ok else '--'}")
        print(f"{variant:<20} {' '.join(parts)}")
    print(f"\nFormal complete: {complete}/35")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if args.command == "setup":
        print(json.dumps(setup_tree(root), indent=2, ensure_ascii=False))
    elif args.command == "smoke":
        run_smoke(args)
    elif args.command == "formal":
        run_formal(args)
    elif args.command == "summarize":
        summarize(args)
    elif args.command == "status":
        status(args)


if __name__ == "__main__":
    main()
