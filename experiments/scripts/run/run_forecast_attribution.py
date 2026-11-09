#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Setup, CPU smoke, formal queue, status and audit for TSFM necessity."""
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
    "n0_no_backbone",
    "n1_persistence",
    "n3_random_timer",
]
SEEDS = [2026, 2027, 2028, 2029, 2030]
INTERVENTIONS = [
    "i0_correct_timer",
    "i1_zero",
    "i2_sample_shuffle",
    "i3_persistence",
    "i4_horizon_shuffle",
]
EXPECTED_SHAPE = (106_200, 48)
DISPLAY = {
    "n0_no_backbone": "No Backbone",
    "n1_persistence": "Persistence Prior",
    "n3_random_timer": "Random Timer Prior",
    "n4_pretrained_timer": "Pretrained Timer Prior",
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
    return root / "experiments/results/13_tsfm_necessity_v2"


def trainer(root: Path) -> Path:
    return (
        root / "experiments/scripts/train/"
        "train_forecast_attribution.py"
    )


def evaluator(root: Path) -> Path:
    return (
        root / "experiments/scripts/eval/"
        "evaluate_forecast_interventions.py"
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


def setup(root: Path) -> dict[str, Any]:
    out = result_root(root)
    for rel in [
        "00_smoke_cpu",
        "00_backbone_cache/random_timer",
        "00_backbone_cache/pretrained_timer",
        "01_training_replacement/n0_no_backbone",
        "01_training_replacement/n1_persistence",
        "01_training_replacement/n3_random_timer",
        "02_inference_intervention/i0_correct_timer",
        "02_inference_intervention/i1_zero",
        "02_inference_intervention/i2_sample_shuffle",
        "02_inference_intervention/i3_persistence",
        "02_inference_intervention/i4_horizon_shuffle",
        "03_diagnostics",
        "04_summary",
        "05_supervised_reference",
        "logs",
    ]:
        (out / rel).mkdir(parents=True, exist_ok=True)

    p4 = (
        root / "experiments/results/"
        "10_adapter_time_ablation_v2/"
        "01_progressive_adapter/p4_io_time_context"
    )
    n4_state = safe_symlink(
        p4,
        out / "01_training_replacement/n4_pretrained_timer",
    )
    tsmixer = (
        root / "experiments/results/"
        "02_global_baselines/00_global11_v2_array_archive/TSMixer"
    )
    tsmixer_state = safe_symlink(
        tsmixer,
        out / "05_supervised_reference/TSMixer",
    )
    timer_zero = (
        root / "experiments/results/"
        "03_zeroshot_baselines_v2_archive/Timer"
    )
    zero_state = safe_symlink(
        timer_zero,
        out / "05_supervised_reference/Timer_ZeroShot",
    )

    design = {
        "protocol": "revision-v2",
        "new_training_variants": VARIANTS,
        "new_formal_trainings": 15,
        "pretrained_timer_reused_from_group1_p4": str(p4),
        "pretrained_timer_link_state": n4_state,
        "supervised_tsmixer_reference": str(tsmixer),
        "tsmixer_link_state": tsmixer_state,
        "timer_zeroshot_reference": str(timer_zero),
        "timer_zero_link_state": zero_state,
        "interventions": INTERVENTIONS,
        "intervention_evaluations": 25,
        "fixed_components": [
            "InputAdapter",
            "full CoRAContextEncoder",
            "Hour+DOY",
            "variable embedding",
            "city embedding",
            "horizon gamma/beta",
            "same residual output MLP",
        ],
        "note": (
            "TSMixer is reported as a supervised external reference, not as "
            "an adapter-trained replacement in this core TSFM-removal package."
        ),
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
    return (
        out / "01_training_replacement"
        / variant / f"seed{seed}"
    )


def is_complete(directory: Path, formal: bool) -> bool:
    status = directory / "run_status.json"
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
        required += [
            "pred.npy", "base_pred.npy", "true.npy",
            "array_manifest.json",
        ]
    return all((directory / name).exists() for name in required)


def launch(command: list[str]) -> None:
    print("\n" + "=" * 104, flush=True)
    print(" ".join(command), flush=True)
    print("=" * 104, flush=True)
    subprocess.run(command, check=True)


def smoke(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup(root)
    for variant in VARIANTS:
        out = run_dir(root, variant, 2026, True)
        if is_complete(out, False):
            print(f"[SMOKE SKIP] {variant}", flush=True)
            continue
        launch([
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
        ])
    print("\nCPU smoke complete: 3/3 variants.", flush=True)


def formal(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup(root)
    unknown = sorted(set(args.variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    # Seed-major order exposes any variant-specific problem early.
    for seed in args.seeds:
        for variant in args.variants:
            out = run_dir(root, variant, seed, False)
            if is_complete(out, True):
                print(
                    f"[FORMAL SKIP] {variant} seed={seed}",
                    flush=True,
                )
                continue
            launch([
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
            ])

    # P4 is available now; run all five inference interventions per seed.
    for seed in args.seeds:
        launch([
            sys.executable,
            str(evaluator(root)),
            "--root", str(root),
            "--seed", str(seed),
            "--batch_size", "64",
            "--num_workers", str(args.num_workers),
            "--device", "cuda",
            "--require_cuda",
        ])


def read_one(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if len(df) != 1:
        raise RuntimeError(f"Expected one row: {path}")
    return df.iloc[0].to_dict()


def audit_array_run(directory: Path) -> dict[str, Any]:
    metric = read_one(directory / "metrics_overall.csv")
    pred = np.load(directory / "pred.npy", mmap_mode="r")
    true = np.load(directory / "true.npy", mmap_mode="r")
    base = np.load(directory / "base_pred.npy", mmap_mode="r")
    diff = (
        np.asarray(pred, dtype=np.float64)
        - np.asarray(true, dtype=np.float64)
    )
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return {
        **metric,
        "shape_ok": (
            tuple(pred.shape) == EXPECTED_SHAPE
            and tuple(true.shape) == EXPECTED_SHAPE
            and tuple(base.shape) == EXPECTED_SHAPE
        ),
        "nonfinite_pred": int((~np.isfinite(pred)).sum()),
        "nonfinite_true": int((~np.isfinite(true)).sum()),
        "nonfinite_base": int((~np.isfinite(base)).sum()),
        "array_MAE": mae,
        "array_RMSE": rmse,
        "metric_match": (
            abs(mae - float(metric["MAE"])) <= 5e-5
            and abs(rmse - float(metric["RMSE"])) <= 5e-5
        ),
    }


def summarize(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    out = result_root(root)
    setup(root)
    reference_path = (
        root / "experiments/results/"
        "03_zeroshot_baselines_v2_archive/Timer/true.npy"
    )
    if not reference_path.is_file():
        reference_path = root / "experiments/data_v2/indices/test_targets.npy"
    canonical = np.load(reference_path, mmap_mode="r")
    rows = []
    issues = []

    for variant in VARIANTS:
        for seed in SEEDS:
            directory = run_dir(root, variant, seed, False)
            row: dict[str, Any] = {
                "variant": variant,
                "paper_name": DISPLAY[variant],
                "seed": seed,
                "run_dir": str(directory),
                "complete": is_complete(directory, True),
            }
            if row["complete"]:
                row.update(audit_array_run(directory))
                true = np.load(directory / "true.npy", mmap_mode="r")
                row["true_canonical_equal"] = np.array_equal(
                    np.asarray(true), np.asarray(canonical), equal_nan=True
                )
            else:
                row.update({
                    "shape_ok": False,
                    "metric_match": False,
                    "true_canonical_equal": False,
                    "nonfinite_pred": -1,
                    "nonfinite_true": -1,
                    "nonfinite_base": -1,
                })
            row["audit_ok"] = all([
                bool(row["complete"]),
                bool(row["shape_ok"]),
                bool(row["metric_match"]),
                bool(row["true_canonical_equal"]),
                int(row["nonfinite_pred"]) == 0,
                int(row["nonfinite_true"]) == 0,
                int(row["nonfinite_base"]) == 0,
            ])
            if not row["audit_ok"]:
                issues.append({
                    "kind": "training",
                    "variant": variant,
                    "seed": seed,
                    "detail": json.dumps(
                        row, ensure_ascii=False, default=str
                    ),
                })
            rows.append(row)

    seed_df = pd.DataFrame(rows)
    seed_df.to_csv(
        out / "04_summary/necessity_seed_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    aggregate_rows = []
    for variant in VARIANTS:
        part = seed_df[seed_df["variant"] == variant]
        aggregate_rows.append({
            "variant": variant,
            "paper_name": DISPLAY[variant],
            "n_seeds": int(part["MAE"].notna().sum()),
            "MAE_mean": float(part["MAE"].mean()),
            "MAE_std": float(part["MAE"].std(ddof=1)),
            "RMSE_mean": float(part["RMSE"].mean()),
            "RMSE_std": float(part["RMSE"].std(ddof=1)),
            "base_MAE_mean": float(part["base_MAE"].mean()),
            "base_RMSE_mean": float(part["base_RMSE"].mean()),
            "mean_abs_correction_mean":
                float(part["mean_abs_correction"].mean()),
            "correction_to_base_l2_ratio_mean":
                float(part["correction_to_base_l2_ratio"].mean()),
            "trainable_params_mean":
                float(part["trainable_params"].mean()),
            "audit_ok_runs": int(part["audit_ok"].sum()),
        })

    # Add N4 P4 result without retraining.
    p4_rows = []
    for seed in SEEDS:
        path = (
            root / "experiments/results/"
            "10_adapter_time_ablation_v2/"
            "01_progressive_adapter/p4_io_time_context/"
            f"seed{seed}/metrics_overall.csv"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        item = read_one(path)
        item["seed"] = seed
        p4_rows.append(item)
    p4 = pd.DataFrame(p4_rows)
    aggregate_rows.append({
        "variant": "n4_pretrained_timer",
        "paper_name": DISPLAY["n4_pretrained_timer"],
        "n_seeds": 5,
        "MAE_mean": float(p4["MAE"].mean()),
        "MAE_std": float(p4["MAE"].std(ddof=1)),
        "RMSE_mean": float(p4["RMSE"].mean()),
        "RMSE_std": float(p4["RMSE"].std(ddof=1)),
        "base_MAE_mean": math.nan,
        "base_RMSE_mean": math.nan,
        "mean_abs_correction_mean": math.nan,
        "correction_to_base_l2_ratio_mean": math.nan,
        "trainable_params_mean": float(
            p4["trainable_params"].mean()
        ),
        "audit_ok_runs": 5,
    })
    summary = pd.DataFrame(aggregate_rows)
    summary.to_csv(
        out / "04_summary/necessity_summary_5seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # External references, kept separate to avoid implying same adapter.
    refs = []
    timer_zero = read_one(
        root / "experiments/results/"
        "03_zeroshot_baselines_v2_archive/Timer/"
        "metrics_recomputed_from_arrays.csv"
    )
    refs.append({
        "reference": "Timer Zero-shot",
        "n": 1,
        "MAE_mean": float(timer_zero["MAE"]),
        "MAE_std": math.nan,
        "RMSE_mean": float(timer_zero["RMSE"]),
        "RMSE_std": math.nan,
        "role": "zero-shot TSFM reference",
    })
    tsmixer_rows = []
    for seed in SEEDS:
        tsmixer_rows.append(read_one(
            root / "experiments/results/"
            "02_global_baselines/00_global11_v2_array_archive/"
            f"TSMixer/seed{seed}/metrics_recomputed_from_arrays.csv"
        ))
    tsmixer = pd.DataFrame(tsmixer_rows)
    refs.append({
        "reference": "Global TSMixer",
        "n": 5,
        "MAE_mean": float(tsmixer["MAE"].mean()),
        "MAE_std": float(tsmixer["MAE"].std(ddof=1)),
        "RMSE_mean": float(tsmixer["RMSE"].mean()),
        "RMSE_std": float(tsmixer["RMSE"].std(ddof=1)),
        "role": "supervised task-specific external reference",
    })
    pd.DataFrame(refs).to_csv(
        out / "04_summary/external_references.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Intervention audit and summary.
    intervention_rows = []
    for intervention in INTERVENTIONS:
        for seed in SEEDS:
            directory = (
                out / "02_inference_intervention"
                / intervention / f"seed{seed}"
            )
            status = directory / "run_status.json"
            complete = False
            if status.exists():
                try:
                    complete = (
                        json.loads(
                            status.read_text(encoding="utf-8")
                        ).get("state") == "complete"
                    )
                except Exception:
                    pass
            row = {
                "intervention": intervention,
                "seed": seed,
                "complete": complete,
            }
            if complete:
                row.update(read_one(
                    directory / "metrics_overall.csv"
                ))
                pred = np.load(directory / "pred.npy", mmap_mode="r")
                true = np.load(directory / "true.npy", mmap_mode="r")
                row["shape_ok"] = (
                    tuple(pred.shape) == EXPECTED_SHAPE
                    and tuple(true.shape) == EXPECTED_SHAPE
                )
                row["true_canonical_equal"] = np.array_equal(
                    np.asarray(true), np.asarray(canonical), equal_nan=True
                )
                row["nonfinite_pred"] = int(
                    (~np.isfinite(pred)).sum()
                )
            else:
                row.update({
                    "shape_ok": False,
                    "true_canonical_equal": False,
                    "nonfinite_pred": -1,
                })
            row["audit_ok"] = all([
                row["complete"],
                row["shape_ok"],
                row["true_canonical_equal"],
                int(row["nonfinite_pred"]) == 0,
            ])
            if not row["audit_ok"]:
                issues.append({
                    "kind": "intervention",
                    "intervention": intervention,
                    "seed": seed,
                    "detail": json.dumps(
                        row, ensure_ascii=False, default=str
                    ),
                })
            intervention_rows.append(row)

    intervention_df = pd.DataFrame(intervention_rows)
    intervention_df.to_csv(
        out / "04_summary/intervention_seed_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    intervention_summary = (
        intervention_df.groupby("intervention", as_index=False)
        .agg(
            n=("MAE", "count"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            base_MAE_mean=("base_MAE", "mean"),
            base_RMSE_mean=("base_RMSE", "mean"),
            audit_ok_runs=("audit_ok", "sum"),
        )
    )
    intervention_summary.to_csv(
        out / "04_summary/intervention_summary_5seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    correct = intervention_df[
        intervention_df["intervention"] == "i0_correct_timer"
    ][["seed", "MAE", "RMSE"]].rename(
        columns={"MAE": "correct_MAE", "RMSE": "correct_RMSE"}
    )
    paired = intervention_df.merge(correct, on="seed", how="left")
    paired = paired[
        paired["intervention"] != "i0_correct_timer"
    ].copy()
    paired["delta_MAE_vs_correct"] = (
        paired["MAE"] - paired["correct_MAE"]
    )
    paired["delta_RMSE_vs_correct"] = (
        paired["RMSE"] - paired["correct_RMSE"]
    )
    paired.to_csv(
        out / "04_summary/intervention_paired_vs_correct.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(issues).to_csv(
        out / "04_summary/audit_issues.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "protocol": "revision-v2",
        "expected_new_training_runs": 15,
        "completed_new_training_runs":
            int(seed_df["complete"].sum()),
        "audit_ok_new_training_runs":
            int(seed_df["audit_ok"].sum()),
        "expected_intervention_runs": 25,
        "completed_intervention_runs":
            int(intervention_df["complete"].sum()),
        "audit_ok_intervention_runs":
            int(intervention_df["audit_ok"].sum()),
        "p4_reused": True,
        "timer_zeroshot_reference": True,
        "global_tsmixer_external_reference": True,
        "issues": len(issues),
    }
    (out / "04_summary/archive_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nTSFM necessity summary")
    print(summary.to_string(index=False))
    print("\nIntervention summary")
    print(intervention_summary.to_string(index=False))
    print("\n" + json.dumps(
        manifest, indent=2, ensure_ascii=False
    ))
    if (
        manifest["completed_new_training_runs"] != 15
        or manifest["audit_ok_new_training_runs"] != 15
        or manifest["completed_intervention_runs"] != 25
        or manifest["audit_ok_intervention_runs"] != 25
        or manifest["issues"] != 0
    ):
        raise SystemExit(
            "ERROR: TSFM necessity archive is incomplete or failed audit"
        )


def status(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup(root)
    print("=" * 100)
    print("CPU SMOKE")
    for variant in VARIANTS:
        print(
            f"{variant:<22} "
            f"{'COMPLETE' if is_complete(run_dir(root, variant, 2026, True), False) else 'pending'}"
        )

    print("\nFORMAL TRAINING")
    complete = 0
    for variant in VARIANTS:
        marks = []
        for seed in SEEDS:
            ok = is_complete(
                run_dir(root, variant, seed, False), True
            )
            complete += int(ok)
            marks.append(f"{seed}:{'OK' if ok else '--'}")
        print(f"{variant:<22} {' '.join(marks)}")
    print(f"New formal training complete: {complete}/15")

    print("\nP4 PRETRAINED TIMER")
    p4_complete = 0
    for seed in SEEDS:
        path = (
            root / "experiments/results/"
            "10_adapter_time_ablation_v2/"
            "01_progressive_adapter/p4_io_time_context/"
            f"seed{seed}/metrics_overall.csv"
        )
        p4_complete += int(path.exists())
    print(f"P4 available: {p4_complete}/5")

    print("\nINFERENCE INTERVENTIONS")
    int_complete = 0
    for intervention in INTERVENTIONS:
        marks = []
        for seed in SEEDS:
            status_path = (
                result_root(root)
                / "02_inference_intervention"
                / intervention / f"seed{seed}/run_status.json"
            )
            ok = False
            if status_path.exists():
                try:
                    ok = (
                        json.loads(
                            status_path.read_text(encoding="utf-8")
                        ).get("state") == "complete"
                    )
                except Exception:
                    pass
            int_complete += int(ok)
            marks.append(f"{seed}:{'OK' if ok else '--'}")
        print(f"{intervention:<22} {' '.join(marks)}")
    print(f"Interventions complete: {int_complete}/25")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if args.command == "setup":
        print(json.dumps(
            setup(root), indent=2, ensure_ascii=False
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
