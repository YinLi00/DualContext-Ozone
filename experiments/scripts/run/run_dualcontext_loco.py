#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controller, status and audit for 22-city LOCO revision-v2."""
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

SEEDS = [2026, 2027, 2028, 2029, 2030]
REFERENCES = ["persistence", "timer_source_scaled"]
H = 48


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "command",
        choices=[
            "setup", "smoke", "formal", "status", "summarize",
        ],
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    p.add_argument("--cities", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--cpu_threads", type=int, default=4)
    return p.parse_args()


def safe_name(value: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in "._-" else "_"
        for ch in value
    )


def result_root(root: Path) -> Path:
    return root / "experiments/results/14_loco_22cities_v2"


def cities(root: Path) -> list[str]:
    meta = json.loads(
        (
            root / "experiments/data_v2/"
            "indices/meta.json"
        ).read_text(encoding="utf-8")
    )
    values = list(meta["cities"])
    if len(values) != 22:
        raise RuntimeError(
            f"Expected 22 cities, found {len(values)}"
        )
    return values


def script(root: Path, relative: str) -> Path:
    return root / "experiments/scripts" / relative


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
        "00_fold_indices",
        "00_smoke_cpu",
        "01_loco_full",
        "03_reference/persistence",
        "03_reference/timer_source_scaled",
        "03_reference/external",
        "04_summary",
        "logs",
    ]:
        (out / rel).mkdir(parents=True, exist_ok=True)

    local_city = (
        root / "experiments/results/"
        "04_local_supervised_v2/city"
    )
    local_state = safe_symlink(
        local_city,
        out / "03_reference/external/local_city_oracle",
    )
    audited_timer = (
        root / "experiments/results/"
        "03_zeroshot_baselines_v2_archive/Timer"
    )
    timer_state = safe_symlink(
        audited_timer,
        out / "03_reference/external/audited_timer_allcity",
    )

    meta = json.loads(
        (
            root / "experiments/data_v2/"
            "indices/meta.json"
        ).read_text(encoding="utf-8")
    )
    design = {
        "protocol": "revision-v2-LOCO",
        "city_count": 22,
        "seeds": SEEDS,
        "new_formal_trainings": 110,
        "core_model": "LOCO Full",
        "train_val": "21 source cities",
        "test": "one held-out city",
        "fold_order": "city-major, five seeds per city",
        "city_identity_mode": "shared_single_id",
        "normalization": "source-21 pooled train-only",
        "references": REFERENCES,
        "external_local_city_oracle": str(local_city),
        "local_city_link_state": local_state,
        "external_audited_timer": str(audited_timer),
        "audited_timer_link_state": timer_state,
        "canonical_splits_inherited": meta.get("splits", {}),
        "excluded_from_core_queue": [
            "LOCO TSMixer 110-run extension",
            "All-city agnostic 5-run upper bound",
        ],
    }
    (out / "experiment_design.json").write_text(
        json.dumps(design, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return design


def launch(command: list[str]) -> None:
    print("\n" + "=" * 112, flush=True)
    print(" ".join(command), flush=True)
    print("=" * 112, flush=True)
    subprocess.run(command, check=True)


def fold_manifest(root: Path, city: str) -> Path:
    return (
        result_root(root)
        / "00_fold_indices" / safe_name(city)
        / "fold_manifest.json"
    )


def run_dir(root: Path, city: str, seed: int, smoke: bool) -> Path:
    if smoke:
        return (
            result_root(root)
            / "00_smoke_cpu"
            / f"{safe_name(city)}_seed{seed}"
        )
    return (
        result_root(root)
        / "01_loco_full" / safe_name(city)
        / f"seed{seed}"
    )


def train_complete(directory: Path, formal: bool) -> bool:
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
            "canonical_test_positions.npy",
            "array_manifest.json",
        ]
    return all((directory / name).exists() for name in required)


def reference_dir(root: Path, reference: str, city: str) -> Path:
    return (
        result_root(root)
        / "03_reference" / reference / safe_name(city)
    )


def reference_complete(root: Path, reference: str, city: str) -> bool:
    directory = reference_dir(root, reference, city)
    status = directory / "run_status.json"
    if not status.exists():
        return False
    try:
        data = json.loads(status.read_text(encoding="utf-8"))
        expected = int(
            json.loads(
                fold_manifest(root, city).read_text(encoding="utf-8")
            )["split_counts"]["test"]
        )
        pred = np.load(directory / "pred.npy", mmap_mode="r")
        true = np.load(directory / "true.npy", mmap_mode="r")
        return (
            data.get("state") == "complete"
            and data.get("formal_full_fold") is True
            and pred.shape == true.shape == (expected, H)
            and (directory / "metrics_overall.csv").exists()
        )
    except Exception:
        return False


def prepare_folds(root: Path) -> None:
    launch([
        sys.executable,
        str(script(
            root, "data/build_loco_folds.py"
        )),
        "--root", str(root),
    ])


def smoke(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup(root)
    prepare_folds(root)
    city = (args.cities or cities(root))[0]
    out = run_dir(root, city, 2026, True)
    if not train_complete(out, False):
        launch([
            sys.executable,
            str(script(root, "train/train_dualcontext_loco.py")),
            "--root", str(root),
            "--heldout_city", city,
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
    else:
        print(f"[SMOKE SKIP] {city}", flush=True)

    launch([
        sys.executable,
        str(script(
            root, "eval/evaluate_loco_references.py"
        )),
        "--root", str(root),
        "--heldout_city", city,
        "--batch_size", "16",
        "--num_workers", "0",
        "--max_test_samples", "64",
        "--device", "cpu",
        "--cpu_threads", str(args.cpu_threads),
        "--overwrite",
    ])
    print(f"\nLOCO CPU smoke complete: city={city}", flush=True)


def formal(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup(root)
    prepare_folds(root)
    selected = args.cities or cities(root)
    unknown = sorted(set(selected) - set(cities(root)))
    if unknown:
        raise ValueError(f"Unknown cities: {unknown}")

    for position, city in enumerate(selected, 1):
        print(
            f"\n######## LOCO CITY {position}/{len(selected)}: "
            f"{city} ########",
            flush=True,
        )
        if not all(
            reference_complete(root, reference, city)
            for reference in REFERENCES
        ):
            launch([
                sys.executable,
                str(script(
                    root, "eval/evaluate_loco_references.py"
                )),
                "--root", str(root),
                "--heldout_city", city,
                "--batch_size", "256",
                "--num_workers", str(args.num_workers),
                "--device", "cuda",
                "--require_cuda",
            ])
        for seed in args.seeds:
            out = run_dir(root, city, seed, False)
            if train_complete(out, True):
                print(
                    f"[FORMAL SKIP] {city} seed={seed}",
                    flush=True,
                )
                continue
            launch([
                sys.executable,
                str(script(
                    root, "train/train_dualcontext_loco.py"
                )),
                "--root", str(root),
                "--heldout_city", city,
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


def read_one(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if len(frame) != 1:
        raise RuntimeError(f"Expected one row: {path}")
    return frame.iloc[0].to_dict()


def array_audit(
    root: Path,
    city: str,
    directory: Path,
) -> dict[str, Any]:
    pred = np.load(directory / "pred.npy", mmap_mode="r")
    base = np.load(directory / "base_pred.npy", mmap_mode="r")
    true = np.load(directory / "true.npy", mmap_mode="r")
    positions = np.load(
        directory / "canonical_test_positions.npy",
        mmap_mode="r",
    )
    canonical = np.load(
        result_root(root) / "00_fold_indices" / safe_name(city)
        / "canonical_test_true.npy",
        mmap_mode="r",
    )
    canonical_positions = np.load(
        result_root(root) / "00_fold_indices" / safe_name(city)
        / "canonical_test_positions.npy",
        mmap_mode="r",
    )
    metric = read_one(directory / "metrics_overall.csv")
    diff = (
        np.asarray(pred, dtype=np.float64)
        - np.asarray(true, dtype=np.float64)
    )
    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt(np.square(diff).mean()))
    return {
        **metric,
        "shape_ok": (
            pred.shape == base.shape == true.shape == canonical.shape
            and pred.ndim == 2 and pred.shape[1] == H
        ),
        "true_canonical_equal": np.array_equal(
            np.asarray(true), np.asarray(canonical), equal_nan=True
        ),
        "positions_canonical_equal": np.array_equal(
            np.asarray(positions), np.asarray(canonical_positions)
        ),
        "nonfinite_pred": int((~np.isfinite(pred)).sum()),
        "nonfinite_base": int((~np.isfinite(base)).sum()),
        "nonfinite_true": int((~np.isfinite(true)).sum()),
        "array_MAE": mae,
        "array_RMSE": rmse,
        "metric_match": (
            abs(mae - float(metric["MAE"])) <= 5e-5
            and abs(rmse - float(metric["RMSE"])) <= 5e-5
        ),
        "sq_error_sum": float(np.square(diff).sum(dtype=np.float64)),
        "abs_error_sum": float(np.abs(diff).sum(dtype=np.float64)),
        "scalar_count": int(diff.size),
    }


def reference_audit(
    root: Path,
    city: str,
    reference: str,
) -> dict[str, Any]:
    directory = reference_dir(root, reference, city)
    pred = np.load(directory / "pred.npy", mmap_mode="r")
    true = np.load(directory / "true.npy", mmap_mode="r")
    positions = np.load(
        directory / "canonical_test_positions.npy",
        mmap_mode="r",
    )
    canonical = np.load(
        result_root(root) / "00_fold_indices" / safe_name(city)
        / "canonical_test_true.npy",
        mmap_mode="r",
    )
    canonical_positions = np.load(
        result_root(root) / "00_fold_indices" / safe_name(city)
        / "canonical_test_positions.npy",
        mmap_mode="r",
    )
    metric = read_one(directory / "metrics_overall.csv")
    diff = (
        np.asarray(pred, dtype=np.float64)
        - np.asarray(true, dtype=np.float64)
    )
    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt(np.square(diff).mean()))
    return {
        **metric,
        "shape_ok": (
            pred.shape == true.shape == canonical.shape
            and pred.ndim == 2 and pred.shape[1] == H
        ),
        "true_canonical_equal": np.array_equal(
            np.asarray(true), np.asarray(canonical), equal_nan=True
        ),
        "positions_canonical_equal": np.array_equal(
            np.asarray(positions), np.asarray(canonical_positions)
        ),
        "nonfinite_pred": int((~np.isfinite(pred)).sum()),
        "nonfinite_true": int((~np.isfinite(true)).sum()),
        "array_MAE": mae,
        "array_RMSE": rmse,
        "metric_match": (
            abs(mae - float(metric["MAE"])) <= 5e-5
            and abs(rmse - float(metric["RMSE"])) <= 5e-5
        ),
    }


def bootstrap_ci(values: np.ndarray, seed: int = 2026) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    samples = rng.choice(
        values,
        size=(10000, len(values)),
        replace=True,
    )
    means = samples.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    out = result_root(root)
    setup(root)
    all_cities = cities(root)
    issues: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for city in all_cities:
        manifest = json.loads(
            fold_manifest(root, city).read_text(encoding="utf-8")
        )
        fold_ok = all([
            manifest.get("state") == "complete",
            manifest.get("source_city_count") == 21,
            manifest.get("target_city_in_train") is False,
            manifest.get("target_city_in_val") is False,
            manifest.get("target_city_stats_used") is False,
            city not in manifest.get("source_cities", []),
        ])
        if not fold_ok:
            issues.append({
                "kind": "fold",
                "city": city,
                "detail": json.dumps(
                    manifest, ensure_ascii=False
                ),
            })

        for seed in SEEDS:
            directory = run_dir(root, city, seed, False)
            row: dict[str, Any] = {
                "heldout_city": city,
                "seed": seed,
                "complete": train_complete(directory, True),
                "run_dir": str(directory),
            }
            if row["complete"]:
                row.update(array_audit(root, city, directory))
            else:
                row.update({
                    "shape_ok": False,
                    "true_canonical_equal": False,
                    "positions_canonical_equal": False,
                    "metric_match": False,
                    "nonfinite_pred": -1,
                    "nonfinite_base": -1,
                    "nonfinite_true": -1,
                })
            row["audit_ok"] = all([
                fold_ok,
                bool(row["complete"]),
                bool(row["shape_ok"]),
                bool(row["true_canonical_equal"]),
                bool(row["positions_canonical_equal"]),
                bool(row["metric_match"]),
                int(row["nonfinite_pred"]) == 0,
                int(row["nonfinite_base"]) == 0,
                int(row["nonfinite_true"]) == 0,
                row.get("city_identity_mode") == "shared_single_id",
                row.get("normalization")
                    == "source-21 pooled train-only",
            ])
            if not row["audit_ok"]:
                issues.append({
                    "kind": "training",
                    "city": city,
                    "seed": seed,
                    "detail": json.dumps(
                        row, ensure_ascii=False, default=str
                    ),
                })
            rows.append(row)

    seed_df = pd.DataFrame(rows)
    seed_df.to_csv(
        out / "04_summary/loco_seed_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    city_rows = []
    for city in all_cities:
        part = seed_df[seed_df["heldout_city"] == city]
        city_rows.append({
            "heldout_city": city,
            "n_seeds": int(part["MAE"].notna().sum()),
            "MAE_mean": float(part["MAE"].mean()),
            "MAE_std": float(part["MAE"].std(ddof=1)),
            "RMSE_mean": float(part["RMSE"].mean()),
            "RMSE_std": float(part["RMSE"].std(ddof=1)),
            "base_MAE_mean": float(part["base_MAE"].mean()),
            "base_MAE_std": float(part["base_MAE"].std(ddof=1)),
            "base_RMSE_mean": float(part["base_RMSE"].mean()),
            "base_RMSE_std": float(part["base_RMSE"].std(ddof=1)),
            "mean_abs_correction_mean":
                float(part["mean_abs_correction"].mean()),
            "correction_to_base_l2_ratio_mean":
                float(part["correction_to_base_l2_ratio"].mean()),
            "n_test_samples": int(part["n_test_samples"].iloc[0])
                if part["n_test_samples"].notna().any() else 0,
            "audit_ok_runs": int(part["audit_ok"].sum()),
        })
    city_df = pd.DataFrame(city_rows)
    city_df.to_csv(
        out / "04_summary/loco_city_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    macro_by_seed = (
        seed_df.groupby("seed", as_index=False)
        .agg(
            city_count=("heldout_city", "nunique"),
            Macro_MAE=("MAE", "mean"),
            Macro_RMSE=("RMSE", "mean"),
            Macro_base_MAE=("base_MAE", "mean"),
            Macro_base_RMSE=("base_RMSE", "mean"),
        )
    )
    macro_by_seed.to_csv(
        out / "04_summary/loco_macro_by_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    macro_summary = pd.DataFrame([{
        "city_count": 22,
        "seed_count": 5,
        "Macro_MAE_mean": float(
            macro_by_seed["Macro_MAE"].mean()
        ),
        "Macro_MAE_std": float(
            macro_by_seed["Macro_MAE"].std(ddof=1)
        ),
        "Macro_RMSE_mean": float(
            macro_by_seed["Macro_RMSE"].mean()
        ),
        "Macro_RMSE_std": float(
            macro_by_seed["Macro_RMSE"].std(ddof=1)
        ),
        "city_MAE_median": float(city_df["MAE_mean"].median()),
        "city_MAE_p90": float(
            city_df["MAE_mean"].quantile(0.90)
        ),
        "best_city": str(
            city_df.loc[city_df["MAE_mean"].idxmin(), "heldout_city"]
        ),
        "best_city_MAE": float(city_df["MAE_mean"].min()),
        "worst_city": str(
            city_df.loc[city_df["MAE_mean"].idxmax(), "heldout_city"]
        ),
        "worst_city_MAE": float(city_df["MAE_mean"].max()),
    }])
    macro_summary.to_csv(
        out / "04_summary/loco_macro_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    micro_rows = []
    for seed in SEEDS:
        part = seed_df[seed_df["seed"] == seed]
        count = float(part["scalar_count"].sum())
        abs_sum = float(part["abs_error_sum"].sum())
        sq_sum = float(part["sq_error_sum"].sum())
        micro_rows.append({
            "seed": seed,
            "city_count": int(part["heldout_city"].nunique()),
            "Micro_MAE": abs_sum / count,
            "Micro_MSE": sq_sum / count,
            "Micro_RMSE": math.sqrt(sq_sum / count),
            "scalar_count": int(count),
        })
    micro_df = pd.DataFrame(micro_rows)
    micro_df.to_csv(
        out / "04_summary/loco_micro_by_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([{
        "Micro_MAE_mean": float(micro_df["Micro_MAE"].mean()),
        "Micro_MAE_std": float(
            micro_df["Micro_MAE"].std(ddof=1)
        ),
        "Micro_RMSE_mean": float(
            micro_df["Micro_RMSE"].mean()
        ),
        "Micro_RMSE_std": float(
            micro_df["Micro_RMSE"].std(ddof=1)
        ),
    }]).to_csv(
        out / "04_summary/loco_micro_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    reference_rows = []
    for city in all_cities:
        for reference in REFERENCES:
            ok = reference_complete(root, reference, city)
            row = {
                "heldout_city": city,
                "reference": reference,
                "complete": ok,
            }
            if ok:
                row.update(reference_audit(root, city, reference))
            else:
                row.update({
                    "shape_ok": False,
                    "true_canonical_equal": False,
                    "positions_canonical_equal": False,
                    "metric_match": False,
                    "nonfinite_pred": -1,
                    "nonfinite_true": -1,
                })
            row["audit_ok"] = all([
                row["complete"],
                row["shape_ok"],
                row["true_canonical_equal"],
                row["positions_canonical_equal"],
                row["metric_match"],
                int(row["nonfinite_pred"]) == 0,
                int(row["nonfinite_true"]) == 0,
            ])
            if not row["audit_ok"]:
                issues.append({
                    "kind": "reference",
                    "city": city,
                    "reference": reference,
                    "detail": json.dumps(
                        row, ensure_ascii=False, default=str
                    ),
                })
            reference_rows.append(row)
    reference_df = pd.DataFrame(reference_rows)
    reference_df.to_csv(
        out / "04_summary/loco_reference_city_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    timer_city = reference_df[
        reference_df["reference"] == "timer_source_scaled"
    ][["heldout_city", "MAE", "RMSE"]].rename(columns={
        "MAE": "timer_MAE",
        "RMSE": "timer_RMSE",
    })
    paired_seed = seed_df.merge(
        timer_city, on="heldout_city", how="left"
    )
    paired_seed["delta_MAE_vs_timer"] = (
        paired_seed["MAE"] - paired_seed["timer_MAE"]
    )
    paired_seed["delta_RMSE_vs_timer"] = (
        paired_seed["RMSE"] - paired_seed["timer_RMSE"]
    )
    paired_seed["wins_vs_timer_MAE"] = (
        paired_seed["delta_MAE_vs_timer"] < 0
    )
    paired_seed.to_csv(
        out / "04_summary/paired_vs_timer_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired_city = (
        paired_seed.groupby("heldout_city", as_index=False)
        .agg(
            loco_MAE=("MAE", "mean"),
            loco_MAE_std=("MAE", "std"),
            loco_RMSE=("RMSE", "mean"),
            timer_MAE=("timer_MAE", "first"),
            timer_RMSE=("timer_RMSE", "first"),
            delta_MAE_vs_timer=("delta_MAE_vs_timer", "mean"),
            delta_RMSE_vs_timer=("delta_RMSE_vs_timer", "mean"),
        )
    )
    paired_city["win_vs_timer"] = (
        paired_city["delta_MAE_vs_timer"] < 0
    )
    paired_city.to_csv(
        out / "04_summary/paired_vs_timer_city.csv",
        index=False,
        encoding="utf-8-sig",
    )
    deltas = paired_city["delta_MAE_vs_timer"].to_numpy(
        dtype=np.float64
    )
    ci_low, ci_high = bootstrap_ci(deltas)
    try:
        from scipy.stats import wilcoxon
        statistic, p_value = wilcoxon(
            deltas, alternative="two-sided", zero_method="wilcox"
        )
        statistic = float(statistic)
        p_value = float(p_value)
    except Exception:
        statistic = math.nan
        p_value = math.nan
    pd.DataFrame([{
        "city_count": 22,
        "mean_delta_MAE_LOCO_minus_Timer": float(deltas.mean()),
        "median_delta_MAE_LOCO_minus_Timer":
            float(np.median(deltas)),
        "bootstrap95_low": ci_low,
        "bootstrap95_high": ci_high,
        "win_cities": int((deltas < 0).sum()),
        "tie_cities": int((deltas == 0).sum()),
        "loss_cities": int((deltas > 0).sum()),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_value": p_value,
    }]).to_csv(
        out / "04_summary/paired_vs_timer_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_seed_rows = []
    for city in all_cities:
        for seed in SEEDS:
            frame = pd.read_csv(
                run_dir(root, city, seed, False)
                / "metrics_by_horizon.csv",
                encoding="utf-8-sig",
            )
            frame.insert(0, "seed", seed)
            frame.insert(0, "heldout_city", city)
            horizon_seed_rows.append(frame)
    horizon_seed = pd.concat(
        horizon_seed_rows, ignore_index=True
    )
    horizon_city = (
        horizon_seed.groupby(
            ["heldout_city", "horizon"], as_index=False
        )
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            base_MAE_mean=("base_MAE", "mean"),
            base_RMSE_mean=("base_RMSE", "mean"),
        )
    )
    horizon_city.to_csv(
        out / "04_summary/loco_horizon_city_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    horizon_macro = (
        horizon_city.groupby("horizon", as_index=False)
        .agg(
            Macro_MAE=("MAE_mean", "mean"),
            Macro_RMSE=("RMSE_mean", "mean"),
            Macro_base_MAE=("base_MAE_mean", "mean"),
            Macro_base_RMSE=("base_RMSE_mean", "mean"),
        )
    )
    horizon_macro.to_csv(
        out / "04_summary/loco_horizon_macro.csv",
        index=False,
        encoding="utf-8-sig",
    )

    period_rows = []
    periods = [
        ("short_1_12", 1, 12),
        ("medium_13_24", 13, 24),
        ("long_25_48", 25, 48),
    ]
    for city in all_cities:
        for seed in SEEDS:
            frame = horizon_seed[
                (horizon_seed["heldout_city"] == city)
                & (horizon_seed["seed"] == seed)
            ]
            for name, start, end in periods:
                part = frame[
                    (frame["horizon"] >= start)
                    & (frame["horizon"] <= end)
                ]
                # Each horizon has the same held-out sample count.
                period_rows.append({
                    "heldout_city": city,
                    "seed": seed,
                    "period": name,
                    "MAE": float(part["MAE"].mean()),
                    "RMSE": float(
                        np.sqrt(np.mean(np.square(part["RMSE"])))
                    ),
                })
    period_seed = pd.DataFrame(period_rows)
    period_seed.to_csv(
        out / "04_summary/loco_period_seed_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    period_macro = (
        period_seed.groupby(["seed", "period"], as_index=False)
        .agg(
            Macro_MAE=("MAE", "mean"),
            Macro_RMSE=("RMSE", "mean"),
        )
        .groupby("period", as_index=False)
        .agg(
            Macro_MAE_mean=("Macro_MAE", "mean"),
            Macro_MAE_std=("Macro_MAE", "std"),
            Macro_RMSE_mean=("Macro_RMSE", "mean"),
            Macro_RMSE_std=("Macro_RMSE", "std"),
        )
    )
    period_macro.to_csv(
        out / "04_summary/loco_period_macro_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(issues).to_csv(
        out / "04_summary/audit_issues.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "state": "complete" if not issues else "incomplete",
        "protocol": "revision-v2-LOCO",
        "expected_folds": 22,
        "complete_folds": sum(
            fold_manifest(root, city).exists()
            for city in all_cities
        ),
        "expected_training_runs": 110,
        "completed_training_runs":
            int(seed_df["complete"].sum()),
        "audit_ok_training_runs":
            int(seed_df["audit_ok"].sum()),
        "expected_reference_runs": 44,
        "completed_reference_runs":
            int(reference_df["complete"].sum()),
        "audit_ok_reference_runs":
            int(reference_df["audit_ok"].sum()),
        "city_identity_mode": "shared_single_id",
        "normalization": "source-21 pooled train-only",
        "target_city_stats_used": False,
        "issues": len(issues),
    }
    (out / "04_summary/archive_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nLOCO city summary")
    print(city_df.to_string(index=False))
    print("\nMacro summary")
    print(macro_summary.to_string(index=False))
    print("\n" + json.dumps(
        manifest, indent=2, ensure_ascii=False
    ))
    if (
        manifest["complete_folds"] != 22
        or manifest["completed_training_runs"] != 110
        or manifest["audit_ok_training_runs"] != 110
        or manifest["completed_reference_runs"] != 44
        or manifest["audit_ok_reference_runs"] != 44
        or manifest["issues"] != 0
    ):
        raise SystemExit("ERROR: LOCO archive incomplete or failed audit")


def status(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup(root)
    all_cities = cities(root)
    folds = 0
    for city in all_cities:
        path = fold_manifest(root, city)
        if path.exists():
            try:
                folds += int(
                    json.loads(
                        path.read_text(encoding="utf-8")
                    ).get("state") == "complete"
                )
            except Exception:
                pass
    print("=" * 108)
    print(f"FOLDS: {folds}/22")

    smoke_city = (args.cities or all_cities)[0]
    print(
        "CPU SMOKE:",
        f"{smoke_city} "
        f"{'COMPLETE' if train_complete(run_dir(root, smoke_city, 2026, True), False) else 'pending'}",
    )

    print("\nREFERENCES")
    reference_count = 0
    for city in all_cities:
        marks = []
        for reference in REFERENCES:
            ok = reference_complete(root, reference, city)
            reference_count += int(ok)
            marks.append(
                f"{reference}:{'OK' if ok else '--'}"
            )
        print(f"{city:<14} {' '.join(marks)}")
    print(f"Reference complete: {reference_count}/44")

    print("\nLOCO FULL FORMAL")
    formal_count = 0
    for city in all_cities:
        marks = []
        for seed in SEEDS:
            ok = train_complete(
                run_dir(root, city, seed, False), True
            )
            formal_count += int(ok)
            marks.append(f"{seed}:{'OK' if ok else '--'}")
        print(f"{city:<14} {' '.join(marks)}")
    print(f"\nFormal complete: {formal_count}/110")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if args.command == "setup":
        print(json.dumps(
            setup(root), indent=2, ensure_ascii=False
        ))
        prepare_folds(root)
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
