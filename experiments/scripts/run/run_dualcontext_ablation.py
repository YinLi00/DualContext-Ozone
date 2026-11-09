#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controller, directory setup, status and final audit for DualContext v2."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VARIANTS = [
    "c4_full_dualcontext",
    "c0_no_context",
    "c1_o3_only",
    "c2_naive_multivariate",
    "c3_dual_no_corr",
    "b0_positive_only",
    "b1_negative_only",
    "b2_equal_fusion",
    "b3_shared_gate",
]
SEEDS = [2026, 2027, 2028, 2029, 2030]
EXPECTED_SHAPE = (106_200, 48)

RELATIVE_DIRS = {
    "c0_no_context": "01_overall/c0_no_context",
    "c1_o3_only": "01_overall/c1_o3_only",
    "c2_naive_multivariate": "01_overall/c2_naive_multivariate",
    "c3_dual_no_corr": "01_overall/c3_dual_no_corr",
    "c4_full_dualcontext": "01_overall/c4_full_dualcontext",
    "b0_positive_only": "02_branch_gate/b0_positive_only",
    "b1_negative_only": "02_branch_gate/b1_negative_only",
    "b2_equal_fusion": "02_branch_gate/b2_equal_fusion",
    "b3_shared_gate": "02_branch_gate/b3_shared_gate",
}

PAPER_NAMES = {
    "c0_no_context": "No Context",
    "c1_o3_only": "O3-only Context",
    "c2_naive_multivariate": "Naive Multivariate",
    "c3_dual_no_corr": "DualContext w/o Pearson",
    "c4_full_dualcontext": "Full DualContext",
    "b0_positive_only": "Positive-only",
    "b1_negative_only": "Negative-only",
    "b2_equal_fusion": "Equal Fusion",
    "b3_shared_gate": "Shared Gate",
    "b4_independent_gates": "Independent Gates",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "command",
        choices=["setup", "smoke", "formal", "status", "summarize"],
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    p.add_argument("--variants", nargs="*", default=VARIANTS)
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--cpu_threads", type=int, default=4)
    return p.parse_args()


def result_root(root: Path) -> Path:
    return root / "experiments/results/11_dualcontext_ablation_v2"


def trainer(root: Path) -> Path:
    return (
        root / "experiments/scripts/train/"
        "train_dualcontext_ablation.py"
    )


def safe_symlink(target: Path, link: Path) -> str:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return "existing"
        link.unlink()
    elif link.exists():
        return "real_path_preserved"
    relative = os.path.relpath(target.resolve(), link.parent.resolve())
    os.symlink(relative, link)
    return "created"


def setup_tree(root: Path) -> dict[str, Any]:
    out = result_root(root)
    for rel in [
        "00_smoke_cpu",
        "01_overall/c0_no_context",
        "01_overall/c1_o3_only",
        "01_overall/c2_naive_multivariate",
        "01_overall/c3_dual_no_corr",
        "01_overall/c4_full_dualcontext",
        "02_branch_gate/b0_positive_only",
        "02_branch_gate/b1_negative_only",
        "02_branch_gate/b2_equal_fusion",
        "02_branch_gate/b3_shared_gate",
        "03_diagnostics",
        "04_summary",
        "logs",
    ]:
        (out / rel).mkdir(parents=True, exist_ok=True)

    c4 = out / "01_overall/c4_full_dualcontext"
    b4 = out / "02_branch_gate/b4_independent_gates"
    b4_status = safe_symlink(c4, b4)

    design = {
        "protocol": "revision-v2",
        "input_length": 96,
        "prediction_length": 48,
        "seeds": SEEDS,
        "unique_trainable_variants": VARIANTS,
        "n_unique_variants": len(VARIANTS),
        "n_formal_trainings": len(VARIANTS) * len(SEEDS),
        "b4_reuses_c4": True,
        "b4_link_status": b4_status,
        "formal_order": (
            "seed-major; C4 first, followed by C0/C1/C2/C3/"
            "B0/B1/B2/B3"
        ),
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
            "time_feature_mode": "both",
            "timer_frozen": True,
        },
    }
    (out / "experiment_design.json").write_text(
        json.dumps(design, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return design


def run_dir(root: Path, variant: str,
            seed: int, smoke: bool) -> Path:
    out = result_root(root)
    if smoke:
        return out / "00_smoke_cpu" / f"{variant}_seed{seed}"
    return out / RELATIVE_DIRS[variant] / f"seed{seed}"


def is_complete(path: Path, formal: bool) -> bool:
    status = path / "run_status.json"
    if not status.exists():
        return False
    try:
        state = json.loads(
            status.read_text(encoding="utf-8")
        ).get("state")
    except Exception:
        return False
    if state != "complete":
        return False
    required = [
        "metrics_overall.csv",
        "gate_summary.csv",
        "variable_gate_and_correlation_summary.csv",
        "branch_diagnostics_by_city.csv",
    ]
    if formal:
        required += [
            "pred.npy", "true.npy", "array_manifest.json"
        ]
    return all((path / name).exists() for name in required)


def launch(command: list[str]) -> None:
    print("\n" + "=" * 104, flush=True)
    print(" ".join(command), flush=True)
    print("=" * 104, flush=True)
    subprocess.run(command, check=True)


def smoke(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup_tree(root)
    for variant in VARIANTS:
        out = run_dir(root, variant, 2026, True)
        if is_complete(out, formal=False):
            print(f"[SMOKE SKIP] {variant}", flush=True)
            continue
        command = [
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
        launch(command)
    print("\nCPU smoke complete: 9/9 unique variants.", flush=True)


def formal(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup_tree(root)
    unknown = sorted(set(args.variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    for seed in args.seeds:
        for variant in args.variants:
            out = run_dir(root, variant, seed, False)
            if is_complete(out, formal=True):
                print(
                    f"[FORMAL SKIP] {variant} seed={seed}",
                    flush=True,
                )
                continue
            command = [
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
            launch(command)


def canonical_true(root: Path) -> Path:
    for path in [
        root / "experiments/results/"
        "03_zeroshot_baselines_v2_archive/Timer/true.npy",
        root / "experiments/results/"
        "03_zeroshot_baselines_v2_reindexed/Timer/true.npy",
        root / "experiments/data_v2/indices/test_targets.npy",
    ]:
        if path.exists():
            return path
    raise FileNotFoundError("Canonical true.npy not found")


def read_metric(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if len(df) != 1:
        raise RuntimeError(f"Expected one row: {path}")
    return df.iloc[0].to_dict()


def summarize(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    out = result_root(root)
    setup_tree(root)
    ref_path = canonical_true(root)
    ref = np.load(ref_path, mmap_mode="r")

    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in SEEDS:
            directory = run_dir(root, variant, seed, False)
            row: dict[str, Any] = {
                "variant": variant,
                "paper_name": PAPER_NAMES[variant],
                "seed": seed,
                "run_dir": str(directory),
                "complete": is_complete(directory, formal=True),
            }
            metric_path = directory / "metrics_overall.csv"
            pred_path = directory / "pred.npy"
            true_path = directory / "true.npy"
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
                    tuple(true.shape) == tuple(ref.shape)
                    and np.array_equal(
                        np.asarray(true),
                        np.asarray(ref),
                        equal_nan=True,
                    )
                )
                row["nonfinite_pred"] = int(
                    (~np.isfinite(pred)).sum()
                )
                row["nonfinite_true"] = int(
                    (~np.isfinite(true)).sum()
                )
                if "MAE" in row and "RMSE" in row:
                    diff = (
                        np.asarray(pred, dtype=np.float64)
                        - np.asarray(true, dtype=np.float64)
                    )
                    array_mae = float(np.mean(np.abs(diff)))
                    array_rmse = float(np.sqrt(np.mean(diff ** 2)))
                    row["array_MAE"] = array_mae
                    row["array_RMSE"] = array_rmse
                    row["metric_match"] = (
                        abs(array_mae - float(row["MAE"])) <= 5e-5
                        and abs(array_rmse - float(row["RMSE"])) <= 5e-5
                    )
                else:
                    row["metric_match"] = False
            else:
                row["shape_ok"] = False
                row["true_canonical_equal"] = False
                row["metric_match"] = False

            row["audit_ok"] = all([
                bool(row["complete"]),
                bool(row.get("shape_ok", False)),
                bool(row.get("true_canonical_equal", False)),
                bool(row.get("metric_match", False)),
                int(row.get("nonfinite_pred", -1)) == 0,
                int(row.get("nonfinite_true", -1)) == 0,
            ])
            if not row["audit_ok"]:
                issues.append({
                    "variant": variant,
                    "seed": seed,
                    "run_dir": str(directory),
                    "detail": json.dumps(
                        row, ensure_ascii=False, default=str
                    ),
                })
            rows.append(row)

    seed_df = pd.DataFrame(rows)
    seed_df.to_csv(
        out / "04_summary/all_unique_seed_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(issues).to_csv(
        out / "04_summary/audit_issues.csv",
        index=False,
        encoding="utf-8-sig",
    )

    def aggregate(order: list[tuple[str, str]]) -> pd.DataFrame:
        aggregate_rows = []
        for display_variant, source_variant in order:
            part = seed_df[
                seed_df["variant"] == source_variant
            ]
            mae = pd.to_numeric(
                part.get("MAE"), errors="coerce"
            ).dropna()
            rmse = pd.to_numeric(
                part.get("RMSE"), errors="coerce"
            ).dropna()
            trainable = pd.to_numeric(
                part.get("trainable_params"),
                errors="coerce",
            ).dropna()
            aggregate_rows.append({
                "variant": display_variant,
                "source_variant": source_variant,
                "paper_name": PAPER_NAMES[display_variant],
                "n_seeds": len(mae),
                "MAE_mean": float(mae.mean()) if len(mae) else math.nan,
                "MAE_std": (
                    float(mae.std(ddof=1))
                    if len(mae) > 1 else math.nan
                ),
                "RMSE_mean": (
                    float(rmse.mean()) if len(rmse) else math.nan
                ),
                "RMSE_std": (
                    float(rmse.std(ddof=1))
                    if len(rmse) > 1 else math.nan
                ),
                "trainable_params_mean": (
                    float(trainable.mean())
                    if len(trainable) else math.nan
                ),
                "audit_ok_runs": int(part["audit_ok"].sum()),
            })
        return pd.DataFrame(aggregate_rows)

    overall = aggregate([
        ("c0_no_context", "c0_no_context"),
        ("c1_o3_only", "c1_o3_only"),
        ("c2_naive_multivariate", "c2_naive_multivariate"),
        ("c3_dual_no_corr", "c3_dual_no_corr"),
        ("c4_full_dualcontext", "c4_full_dualcontext"),
    ])
    branch = aggregate([
        ("b0_positive_only", "b0_positive_only"),
        ("b1_negative_only", "b1_negative_only"),
        ("b2_equal_fusion", "b2_equal_fusion"),
        ("b3_shared_gate", "b3_shared_gate"),
        ("b4_independent_gates", "c4_full_dualcontext"),
    ])
    overall.to_csv(
        out / "04_summary/overall_summary_5seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    branch.to_csv(
        out / "04_summary/branch_gate_summary_5seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Collect per-run diagnostic tables into root-level long tables.
    diagnostic_specs = {
        "gate_summary_all_runs.csv": "gate_summary.csv",
        "variable_gate_and_correlation_all_runs.csv":
            "variable_gate_and_correlation_summary.csv",
        "branch_diagnostics_by_city_all_runs.csv":
            "branch_diagnostics_by_city.csv",
    }
    diagnostic_counts = {}
    diag_root = out / "03_diagnostics"
    for output_name, source_name in diagnostic_specs.items():
        tables = []
        for variant in VARIANTS:
            for seed in SEEDS:
                path = run_dir(
                    root, variant, seed, False
                ) / source_name
                if path.exists():
                    table = pd.read_csv(
                        path, encoding="utf-8-sig"
                    )
                    table.insert(0, "seed", seed)
                    table.insert(0, "source_variant", variant)
                    tables.append(table)
        merged = (
            pd.concat(tables, ignore_index=True)
            if tables else pd.DataFrame()
        )
        merged.to_csv(
            diag_root / output_name,
            index=False,
            encoding="utf-8-sig",
        )
        diagnostic_counts[output_name] = len(merged)

    manifest = {
        "protocol": "revision-v2",
        "expected_unique_variants": 9,
        "expected_unique_trainings": 45,
        "completed_runs": int(seed_df["complete"].sum()),
        "audit_ok_runs": int(seed_df["audit_ok"].sum()),
        "issues": len(issues),
        "b4_reuses_c4": True,
        "canonical_true": str(ref_path),
        "expected_shape": list(EXPECTED_SHAPE),
        "diagnostic_rows": diagnostic_counts,
    }
    (out / "04_summary/archive_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nOverall DualContext ablation")
    print(overall.to_string(index=False))
    print("\nBranch/gate ablation")
    print(branch.to_string(index=False))
    print("\n" + json.dumps(
        manifest, indent=2, ensure_ascii=False
    ))

    if (
        manifest["completed_runs"] != 45
        or manifest["audit_ok_runs"] != 45
        or manifest["issues"] != 0
    ):
        raise SystemExit(
            "ERROR: DualContext archive incomplete or audit failed"
        )


def status(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup_tree(root)
    print("=" * 96)
    print("CPU SMOKE")
    for variant in VARIANTS:
        directory = run_dir(root, variant, 2026, True)
        print(
            f"{variant:<25} "
            f"{'COMPLETE' if is_complete(directory, False) else 'pending'}"
        )
    print("\n" + "=" * 96)
    print("FORMAL GPU")
    complete = 0
    for variant in VARIANTS:
        marks = []
        for seed in SEEDS:
            ok = is_complete(
                run_dir(root, variant, seed, False),
                formal=True,
            )
            complete += int(ok)
            marks.append(f"{seed}:{'OK' if ok else '--'}")
        print(f"{variant:<25} {' '.join(marks)}")
    print(f"\nUnique formal complete: {complete}/45")
    print(
        "B4 independent-gates reuses C4 full-DualContext; "
        "it is not an additional training."
    )


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if args.command == "setup":
        print(json.dumps(
            setup_tree(root), indent=2, ensure_ascii=False
        ))
    elif args.command == "smoke":
        smoke(args)
    elif args.command == "formal":
        formal(args)
    elif args.command == "status":
        status(args)
    elif args.command == "summarize":
        summarize(args)


if __name__ == "__main__":
    main()
