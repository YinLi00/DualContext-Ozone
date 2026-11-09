#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit and summarize all 11 global supervised revision-v2 baselines.

Expected experiment:
    11 models × 5 seeds (2026-2030)
    test samples = 106,200
    horizon = 48
    scalar predictions = 5,097,600

This script never modifies or relocates the original run directories.
It writes a compact summary/index under:
    experiments/results/02_global_baselines/00_all11_global_v2_summary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODELS = {
    "TSMixer": "1b1_global_tsmixer",
    "DLinear": "1b2_global_dlinear",
    "PatchTST": "1b2_global_patchtst",
    "LSTM": "1d_global_lstm",
    "iTransformer": "1d_global_itransformer",
    "TimeMixer": "1d_global_timemixer",
    "TimesNet": "1d_global_timesnet",
    "CNNLSTM": "1e_global_cnnlstm",
    "Transformer": "1e_global_transformer",
    "TimeXer": "1e_global_timexer",
    "Koopa": "1e_global_koopa",
}
SEEDS = [2026, 2027, 2028, 2029, 2030]
EXPECTED_SHAPE = (106_200, 48)
EXPECTED_COUNT = EXPECTED_SHAPE[0] * EXPECTED_SHAPE[1]

METRICS = [
    "MAE", "RMSE",
    "city_MAE_macro", "city_RMSE_macro",
    "station_MAE_macro", "station_RMSE_macro",
    "train_seconds", "test_seconds", "peak_gpu_memory_mb",
    "total_params", "trainable_params",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path(
            "experiments/results/02_global_baselines/"
            "00_all11_global_v2_summary"
        ),
    )
    p.add_argument(
        "--verify_arrays",
        action="store_true",
        help=(
            "Read every pred/true array, recompute MAE/RMSE, check finite values, "
            "and compare true.npy against the canonical Timer v2 true.npy."
        ),
    )
    p.add_argument("--chunk_rows", type=int, default=4096)
    p.add_argument(
        "--metric_tolerance",
        type=float,
        default=1e-6,
        help="Absolute tolerance for CSV-vs-array MAE/RMSE checks.",
    )
    return p.parse_args()


def resolve(root: Path, p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def read_one_row(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if len(df) != 1:
        raise RuntimeError(f"Expected one row in {path}, found {len(df)}")
    return df.iloc[0].to_dict()


def normalize_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "invalid_json"
    value = obj.get("state", obj.get("status", "unknown"))
    return str(value)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def compare_true_and_metrics(
    pred_path: Path,
    true_path: Path,
    canonical_true_path: Path,
    chunk_rows: int,
) -> dict[str, Any]:
    pred = np.load(pred_path, mmap_mode="r")
    true = np.load(true_path, mmap_mode="r")
    canonical = np.load(canonical_true_path, mmap_mode="r")

    result: dict[str, Any] = {
        "pred_shape": str(tuple(pred.shape)),
        "true_shape": str(tuple(true.shape)),
        "pred_dtype": str(pred.dtype),
        "true_dtype": str(true.dtype),
        "shape_ok": tuple(pred.shape) == EXPECTED_SHAPE
                    and tuple(true.shape) == EXPECTED_SHAPE,
        "canonical_true_equal": False,
        "nonfinite_pred": 0,
        "nonfinite_true": 0,
        "array_MAE": math.nan,
        "array_MSE": math.nan,
        "array_RMSE": math.nan,
    }
    if tuple(pred.shape) != EXPECTED_SHAPE or tuple(true.shape) != EXPECTED_SHAPE:
        return result
    if tuple(canonical.shape) != EXPECTED_SHAPE:
        raise RuntimeError(
            f"Canonical true shape is {canonical.shape}, expected {EXPECTED_SHAPE}"
        )

    abs_sum = 0.0
    sq_sum = 0.0
    n = 0
    true_equal = True
    nf_pred = 0
    nf_true = 0

    for start in range(0, EXPECTED_SHAPE[0], chunk_rows):
        end = min(start + chunk_rows, EXPECTED_SHAPE[0])
        p = np.asarray(pred[start:end], dtype=np.float64)
        y = np.asarray(true[start:end], dtype=np.float64)
        c = np.asarray(canonical[start:end])

        finite_p = np.isfinite(p)
        finite_y = np.isfinite(y)
        nf_pred += int((~finite_p).sum())
        nf_true += int((~finite_y).sum())
        if not np.array_equal(np.asarray(true[start:end]), c, equal_nan=True):
            true_equal = False

        valid = finite_p & finite_y
        if valid.any():
            diff = p[valid] - y[valid]
            abs_sum += float(np.abs(diff).sum(dtype=np.float64))
            sq_sum += float(np.square(diff).sum(dtype=np.float64))
            n += int(valid.sum())

    result["canonical_true_equal"] = true_equal
    result["nonfinite_pred"] = nf_pred
    result["nonfinite_true"] = nf_true
    if n:
        result["array_MAE"] = abs_sum / n
        result["array_MSE"] = sq_sum / n
        result["array_RMSE"] = math.sqrt(result["array_MSE"])
    return result


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    result_root = root / "experiments/results/02_global_baselines"
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical_true = (
        root
        / "experiments/results/03_zeroshot_baselines_v2_reindexed"
        / "Timer/true.npy"
    )
    if args.verify_arrays and not canonical_true.exists():
        raise FileNotFoundError(f"Canonical v2 true.npy not found: {canonical_true}")

    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for model, group in MODELS.items():
        for seed in SEEDS:
            run_dir = result_root / group / f"{model}_city_time_seed{seed}"
            metric_path = run_dir / "metrics_overall.csv"
            pred_path = run_dir / "pred.npy"
            true_path = run_dir / "true.npy"
            status_path = run_dir / "run_status.json"
            manifest_path = run_dir / "prediction_manifest.json"
            args_path = run_dir / "args.json"

            row: dict[str, Any] = {
                "model_name": model,
                "seed": seed,
                "group_dir": group,
                "run_dir": str(run_dir),
                "status": normalize_status(status_path),
                "metrics_exists": metric_path.exists(),
                "pred_exists": pred_path.exists(),
                "true_exists": true_path.exists(),
                "manifest_exists": manifest_path.exists(),
                "args_exists": args_path.exists(),
            }

            run_issues: list[str] = []
            if not metric_path.exists():
                run_issues.append("missing metrics_overall.csv")
            else:
                try:
                    metric = read_one_row(metric_path)
                    for key, value in metric.items():
                        if key not in row:
                            row[key] = value
                    count = int(float(metric.get("count", -1)))
                    row["count_ok"] = count == EXPECTED_COUNT
                    if count != EXPECTED_COUNT:
                        run_issues.append(f"metric count={count}")
                except Exception as exc:
                    row["metric_read_error"] = repr(exc)
                    run_issues.append(f"cannot read metrics: {exc}")

            if not pred_path.exists():
                run_issues.append("missing pred.npy")
            if not true_path.exists():
                run_issues.append("missing true.npy")

            if args.verify_arrays and pred_path.exists() and true_path.exists():
                try:
                    checked = compare_true_and_metrics(
                        pred_path, true_path, canonical_true, args.chunk_rows
                    )
                    row.update(checked)
                    if not checked["shape_ok"]:
                        run_issues.append(
                            f"array shape pred={checked['pred_shape']} "
                            f"true={checked['true_shape']}"
                        )
                    if not checked["canonical_true_equal"]:
                        run_issues.append("true.npy differs from canonical v2 true.npy")
                    if checked["nonfinite_pred"] != 0:
                        run_issues.append(
                            f"pred nonfinite={checked['nonfinite_pred']}"
                        )
                    if checked["nonfinite_true"] != 0:
                        run_issues.append(
                            f"true nonfinite={checked['nonfinite_true']}"
                        )

                    if metric_path.exists() and "MAE" in row and "RMSE" in row:
                        mae_delta = abs(float(row["MAE"]) - checked["array_MAE"])
                        rmse_delta = abs(float(row["RMSE"]) - checked["array_RMSE"])
                        row["MAE_csv_array_absdiff"] = mae_delta
                        row["RMSE_csv_array_absdiff"] = rmse_delta
                        row["array_metric_match"] = (
                            mae_delta <= args.metric_tolerance
                            and rmse_delta <= args.metric_tolerance
                        )
                        if not row["array_metric_match"]:
                            run_issues.append(
                                f"metric mismatch MAE_diff={mae_delta:.3g}, "
                                f"RMSE_diff={rmse_delta:.3g}"
                            )
                except Exception as exc:
                    row["array_check_error"] = repr(exc)
                    run_issues.append(f"array verification error: {exc}")

            row["audit_ok"] = len(run_issues) == 0
            row["issues"] = " | ".join(run_issues)
            rows.append(row)
            for issue in run_issues:
                issues.append({
                    "model_name": model,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "issue": issue,
                })

    df = pd.DataFrame(rows)
    df.to_csv(
        out_dir / "seed_level_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    issue_df = pd.DataFrame(
        issues,
        columns=["model_name", "seed", "run_dir", "issue"],
    )
    issue_df.to_csv(
        out_dir / "audit_issues.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows: list[dict[str, Any]] = []
    for model in MODELS:
        part = df[df["model_name"] == model].copy()
        # Model-level ranking only requires valid metric files.
        # Prediction arrays are audited separately.
        complete = part[
            part["metrics_exists"].fillna(False)
        ]
        out: dict[str, Any] = {
            "model_name": model,
            "n_expected_seeds": len(SEEDS),
            "n_metric_seeds": int(part["metrics_exists"].fillna(False).sum()),
            "n_array_seeds": int(
                (
                    part["pred_exists"].fillna(False)
                    & part["true_exists"].fillna(False)
                ).sum()
            ),
            "n_audit_ok": int(part["audit_ok"].fillna(False).sum()),
        }

        for metric in METRICS:
            if metric not in complete.columns:
                continue
            vals = pd.to_numeric(complete[metric], errors="coerce").dropna()
            if len(vals):
                out[f"{metric}_mean"] = float(vals.mean())
                out[f"{metric}_std"] = (
                    float(vals.std(ddof=1)) if len(vals) > 1 else math.nan
                )
                out[f"{metric}_min"] = float(vals.min())
                out[f"{metric}_max"] = float(vals.max())

        if "MAE_mean" in out and "MAE_std" in out:
            out["MAE_mean_std"] = (
                f"{out['MAE_mean']:.6f} ± {out['MAE_std']:.6f}"
            )
        if "RMSE_mean" in out and "RMSE_std" in out:
            out["RMSE_mean_std"] = (
                f"{out['RMSE_mean']:.6f} ± {out['RMSE_std']:.6f}"
            )
        summary_rows.append(out)

    summary = pd.DataFrame(summary_rows)
    if "MAE_mean" in summary.columns:
        summary = summary.sort_values(
            ["MAE_mean", "RMSE_mean"], na_position="last"
        ).reset_index(drop=True)
        summary.insert(0, "MAE_rank", np.arange(1, len(summary) + 1))
    if "RMSE_mean" in summary.columns:
        summary["RMSE_rank"] = (
            summary["RMSE_mean"].rank(method="min").astype("Int64")
        )

    summary.to_csv(
        out_dir / "summary_5seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paper_cols = [
        c for c in [
            "MAE_rank", "RMSE_rank", "model_name",
            "MAE_mean_std", "RMSE_mean_std",
            "city_MAE_macro_mean", "city_MAE_macro_std",
            "station_MAE_macro_mean", "station_MAE_macro_std",
            "total_params_mean", "trainable_params_mean",
            "train_seconds_mean", "peak_gpu_memory_mb_mean",
            "n_metric_seeds", "n_array_seeds", "n_audit_ok",
        ]
        if c in summary.columns
    ]
    summary[paper_cols].to_csv(
        out_dir / "paper_table_global11.csv",
        index=False,
        encoding="utf-8-sig",
    )

    index_cols = [
        "model_name", "seed", "run_dir", "status",
        "metrics_exists", "pred_exists", "true_exists",
        "manifest_exists", "args_exists", "audit_ok", "issues",
    ]
    df[[c for c in index_cols if c in df.columns]].to_csv(
        out_dir / "result_file_index.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "protocol": "revision-v2",
        "training_scope": "global pooled: all cities and all stations",
        "models": list(MODELS),
        "seeds": SEEDS,
        "expected_runs": len(MODELS) * len(SEEDS),
        "expected_test_shape": list(EXPECTED_SHAPE),
        "expected_scalar_count": EXPECTED_COUNT,
        "verify_arrays": bool(args.verify_arrays),
        "canonical_true": str(canonical_true) if args.verify_arrays else None,
        "n_runs_with_metrics": int(df["metrics_exists"].fillna(False).sum()),
        "n_runs_with_arrays": int(
            (
                df["pred_exists"].fillna(False)
                & df["true_exists"].fillna(False)
            ).sum()
        ),
        "n_runs_audit_ok": int(df["audit_ok"].fillna(False).sum()),
        "n_issues": len(issues),
        "output_dir": str(out_dir),
    }
    (out_dir / "summary_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Global 11 supervised baselines — revision-v2",
        "",
        f"- Expected runs: {len(MODELS) * len(SEEDS)}",
        f"- Runs with metrics: {manifest['n_runs_with_metrics']}",
        f"- Runs with prediction arrays: {manifest['n_runs_with_arrays']}",
        f"- Audit-passed runs: {manifest['n_runs_audit_ok']}",
        f"- Audit issues: {manifest['n_issues']}",
        f"- Test shape per run: {EXPECTED_SHAPE}",
        f"- Scalar prediction count per run: {EXPECTED_COUNT:,}",
        "",
        "## Ranking by 5-seed mean MAE",
        "",
    ]
    if len(summary):
        for _, r in summary.iterrows():
            mae = r.get("MAE_mean_std", "NA")
            rmse = r.get("RMSE_mean_std", "NA")
            lines.append(
                f"{int(r['MAE_rank'])}. **{r['model_name']}** — "
                f"MAE {mae}; RMSE {rmse}"
            )
    lines.extend([
        "",
        "## Generated files",
        "",
        "- `seed_level_metrics.csv`: all 55 seed-level rows",
        "- `summary_5seeds.csv`: model-level mean/SD/min/max",
        "- `paper_table_global11.csv`: compact paper-facing table",
        "- `result_file_index.csv`: paths and completion flags",
        "- `audit_issues.csv`: missing or inconsistent artifacts",
        "- `summary_manifest.json`: protocol and audit manifest",
    ])
    (out_dir / "README_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("GLOBAL 11 × 5-SEED SUMMARY")
    print("=" * 100)
    display_cols = [
        c for c in [
            "MAE_rank", "RMSE_rank", "model_name",
            "MAE_mean_std", "RMSE_mean_std",
            "n_metric_seeds", "n_array_seeds", "n_audit_ok",
        ]
        if c in summary.columns
    ]
    print(summary[display_cols].to_string(index=False))
    print("\nAudit manifest:")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {out_dir}")

    if manifest["n_runs_with_metrics"] != 55:
        raise SystemExit("ERROR: not all 55 metrics files were found")
    if manifest["n_runs_with_arrays"] != 55:
        print(
            f"WARNING: only {manifest['n_runs_with_arrays']}/55 runs "
            "contain pred.npy and true.npy"
        )
    if args.verify_arrays and manifest["n_issues"] != 0:
        print(
            f"WARNING: array audit reported "
            f"{manifest['n_issues']} issue records"
        )


if __name__ == "__main__":
    main()
