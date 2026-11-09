#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a dedicated revision-v2 archive for all six zero-shot baselines.

Source:
experiments/results/03_zeroshot_baselines_v2_reindexed/

Models:
- Timer
- Sundial
- TimeMoE-200M
- SEMPO
- Chronos-Bolt-Base
- Chronos-2

Destination:
experiments/results/03_zeroshot_baselines_v2_archive/

The source directories are never modified. Existing arrays and metadata are
hard-linked when possible, otherwise symlinked or copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODELS = [
    "Timer",
    "Sundial",
    "TimeMoE-200M",
    "SEMPO",
    "Chronos-Bolt-Base",
    "Chronos-2",
]
EXPECTED_SHAPE = (106_200, 48)
EXPECTED_COUNT = EXPECTED_SHAPE[0] * EXPECTED_SHAPE[1]

OPTIONAL_FILES = [
    "metrics_overall.csv",
    "metrics_by_horizon.csv",
    "metrics_by_city.csv",
    "metrics_by_station.csv",
    "metrics_by_city_variable.csv",
    "prediction_manifest.json",
    "run_status.json",
    "progress.json",
    "args.json",
    "metadata.json",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    p.add_argument(
        "--source_dir",
        type=Path,
        default=Path(
            "experiments/results/"
            "03_zeroshot_baselines_v2_reindexed"
        ),
    )
    p.add_argument(
        "--archive_dir",
        type=Path,
        default=Path(
            "experiments/results/"
            "03_zeroshot_baselines_v2_archive"
        ),
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="Use normal copies instead of hard links/symlinks.",
    )
    p.add_argument(
        "--hash_arrays",
        action="store_true",
        help="Compute SHA-256 for every pred.npy and true.npy.",
    )
    p.add_argument(
        "--metric_tolerance",
        type=float,
        default=1e-8,
    )
    return p.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def link_or_copy(src: Path, dst: Path, force_copy: bool = False) -> str:
    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        try:
            if dst.resolve() == src:
                return "existing-link"
        except Exception:
            pass
        dst.unlink()

    if force_copy:
        shutil.copy2(src, dst)
        return "copy"

    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        pass

    try:
        relative = os.path.relpath(src, dst.parent)
        os.symlink(relative, dst)
        return "symlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy-fallback"


def compute_metrics(
    pred_path: Path,
    true_path: Path,
    chunk_rows: int = 4096,
) -> dict[str, Any]:
    pred = np.load(pred_path, mmap_mode="r")
    true = np.load(true_path, mmap_mode="r")

    if tuple(pred.shape) != EXPECTED_SHAPE:
        raise RuntimeError(
            f"Unexpected pred shape {pred.shape}: {pred_path}"
        )
    if tuple(true.shape) != EXPECTED_SHAPE:
        raise RuntimeError(
            f"Unexpected true shape {true.shape}: {true_path}"
        )

    abs_sum = 0.0
    sq_sum = 0.0
    count = 0
    nonfinite_pred = 0
    nonfinite_true = 0

    for start in range(0, EXPECTED_SHAPE[0], chunk_rows):
        end = min(start + chunk_rows, EXPECTED_SHAPE[0])
        p = np.asarray(pred[start:end], dtype=np.float64)
        y = np.asarray(true[start:end], dtype=np.float64)

        fp = np.isfinite(p)
        fy = np.isfinite(y)
        nonfinite_pred += int((~fp).sum())
        nonfinite_true += int((~fy).sum())

        valid = fp & fy
        diff = p[valid] - y[valid]
        abs_sum += float(np.abs(diff).sum(dtype=np.float64))
        sq_sum += float(np.square(diff).sum(dtype=np.float64))
        count += int(valid.sum())

    mse = sq_sum / count
    return {
        "MAE": abs_sum / count,
        "MSE": mse,
        "RMSE": math.sqrt(mse),
        "count": count,
        "nonfinite_pred": nonfinite_pred,
        "nonfinite_true": nonfinite_true,
        "pred_shape": str(tuple(pred.shape)),
        "true_shape": str(tuple(true.shape)),
        "pred_dtype": str(pred.dtype),
        "true_dtype": str(true.dtype),
    }


def write_metrics_by_horizon(
    pred_path: Path,
    true_path: Path,
    out_path: Path,
) -> None:
    pred = np.load(pred_path, mmap_mode="r")
    true = np.load(true_path, mmap_mode="r")
    rows = []
    for h in range(EXPECTED_SHAPE[1]):
        p = np.asarray(pred[:, h], dtype=np.float64)
        y = np.asarray(true[:, h], dtype=np.float64)
        diff = p - y
        mse = float(np.mean(diff ** 2))
        rows.append(
            {
                "horizon": h + 1,
                "MAE": float(np.mean(np.abs(diff))),
                "MSE": mse,
                "RMSE": math.sqrt(mse),
                "count": int(diff.size),
            }
        )
    pd.DataFrame(rows).to_csv(
        out_path,
        index=False,
        encoding="utf-8-sig",
    )


def read_one_row(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if len(df) != 1:
        raise RuntimeError(f"Expected one row in {path}")
    return df.iloc[0].to_dict()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    source_root = resolve(root, args.source_dir)
    archive_root = resolve(root, args.archive_dir)
    archive_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    canonical_true: Path | None = None

    for model_name in MODELS:
        source_dir = source_root / model_name
        archive_dir = archive_root / model_name
        archive_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 88)
        print(model_name)
        print("=" * 88)

        pred_src = source_dir / "pred.npy"
        true_src = source_dir / "true.npy"
        if not pred_src.exists():
            raise FileNotFoundError(f"Missing pred.npy: {pred_src}")
        if not true_src.exists():
            raise FileNotFoundError(f"Missing true.npy: {true_src}")

        if canonical_true is None:
            canonical_true = true_src
        else:
            a = np.load(canonical_true, mmap_mode="r")
            b = np.load(true_src, mmap_mode="r")
            if tuple(a.shape) != tuple(b.shape):
                raise RuntimeError(
                    f"{model_name}: true shape differs from Timer"
                )
            same = True
            for start in range(0, EXPECTED_SHAPE[0], 4096):
                end = min(start + 4096, EXPECTED_SHAPE[0])
                if not np.array_equal(
                    np.asarray(a[start:end]),
                    np.asarray(b[start:end]),
                    equal_nan=True,
                ):
                    same = False
                    break
            if not same:
                raise RuntimeError(
                    f"{model_name}: true.npy differs from canonical Timer true"
                )

        pred_method = link_or_copy(
            pred_src,
            archive_dir / "pred.npy",
            force_copy=args.copy,
        )
        # All archive true.npy paths point to Timer's canonical true when possible.
        true_method = link_or_copy(
            canonical_true,
            archive_dir / "true.npy",
            force_copy=args.copy,
        )

        linked_files = {
            "pred.npy": pred_method,
            "true.npy": true_method,
        }
        for name in OPTIONAL_FILES:
            src = source_dir / name
            if src.exists():
                linked_files[name] = link_or_copy(
                    src,
                    archive_dir / name,
                    force_copy=args.copy,
                )

        recomputed = compute_metrics(
            archive_dir / "pred.npy",
            archive_dir / "true.npy",
        )
        pd.DataFrame([{"model_name": model_name, **recomputed}]).to_csv(
            archive_dir / "metrics_recomputed_from_arrays.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if not (archive_dir / "metrics_by_horizon.csv").exists():
            write_metrics_by_horizon(
                archive_dir / "pred.npy",
                archive_dir / "true.npy",
                archive_dir / "metrics_by_horizon.csv",
            )
            linked_files["metrics_by_horizon.csv"] = "generated"

        reported_path = source_dir / "metrics_overall.csv"
        reported = read_one_row(reported_path) if reported_path.exists() else {}
        mae_diff = (
            abs(float(reported["MAE"]) - recomputed["MAE"])
            if "MAE" in reported else math.nan
        )
        rmse_diff = (
            abs(float(reported["RMSE"]) - recomputed["RMSE"])
            if "RMSE" in reported else math.nan
        )
        metric_match = (
            "MAE" in reported
            and "RMSE" in reported
            and mae_diff <= args.metric_tolerance
            and rmse_diff <= args.metric_tolerance
        )

        manifest = {
            "protocol": "revision-v2",
            "model_name": model_name,
            "role": "zero-shot baseline",
            "source_dir": str(source_dir),
            "archive_dir": str(archive_dir),
            "array_shape": list(EXPECTED_SHAPE),
            "scalar_count": EXPECTED_COUNT,
            "linked_files": linked_files,
            "reported_metrics": reported,
            "recomputed_metrics": recomputed,
            "MAE_absdiff_reported_vs_array": mae_diff,
            "RMSE_absdiff_reported_vs_array": rmse_diff,
            "metric_match": metric_match,
        }

        if args.hash_arrays:
            manifest["pred_sha256"] = sha256_file(
                archive_dir / "pred.npy"
            )
            manifest["true_sha256"] = sha256_file(
                archive_dir / "true.npy"
            )

        (archive_dir / "array_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        row = {
            "model_name": model_name,
            "source_dir": str(source_dir),
            "archive_dir": str(archive_dir),
            **recomputed,
            "reported_MAE": reported.get("MAE"),
            "reported_MSE": reported.get("MSE"),
            "reported_RMSE": reported.get("RMSE"),
            "MAE_absdiff_reported_vs_array": mae_diff,
            "RMSE_absdiff_reported_vs_array": rmse_diff,
            "metric_match": metric_match,
            "pred_link_method": pred_method,
            "true_link_method": true_method,
        }
        rows.append(row)
        print(
            f"MAE={recomputed['MAE']:.6f} "
            f"RMSE={recomputed['RMSE']:.6f} "
            f"metric_match={metric_match}"
        )

    summary = pd.DataFrame(rows)
    summary["MAE_rank"] = summary["MAE"].rank(
        method="min"
    ).astype("Int64")
    summary["RMSE_rank"] = summary["RMSE"].rank(
        method="min"
    ).astype("Int64")
    summary = summary.sort_values(
        ["MAE_rank", "RMSE_rank"]
    ).reset_index(drop=True)
    summary.to_csv(
        archive_root / "summary_all_6_zeroshot_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        archive_root / "archive_index.csv",
        index=False,
        encoding="utf-8-sig",
    )

    n_complete = len(summary)
    n_match = int(summary["metric_match"].sum())
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "revision-v2",
        "role": "zero-shot baselines",
        "models": MODELS,
        "expected_models": 6,
        "expected_shape_per_model": list(EXPECTED_SHAPE),
        "expected_scalar_count_per_model": EXPECTED_COUNT,
        "n_models_with_arrays": n_complete,
        "n_metric_matches": n_match,
        "source_root": str(source_root),
        "archive_root": str(archive_root),
    }
    (archive_root / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    readme = f"""# Six zero-shot baselines — revision-v2

- Models: 6
- Protocol: revision-v2
- Test shape per model: `(106200, 48)`
- Scalar count per model: `5,097,600`
- Models with arrays: {n_complete}/6
- Metrics matching arrays: {n_match}/6

Each model directory contains at least:

- `pred.npy`
- `true.npy`
- `metrics_overall.csv`
- `metrics_recomputed_from_arrays.csv`
- `metrics_by_horizon.csv`
- `array_manifest.json`

The original source directory was not modified.
"""
    (archive_root / "README.md").write_text(
        readme,
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("ZERO-SHOT 6-MODEL ARCHIVE")
    print("=" * 88)
    print(summary[
        [
            "MAE_rank",
            "RMSE_rank",
            "model_name",
            "MAE",
            "RMSE",
            "metric_match",
        ]
    ].to_string(index=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Saved to: {archive_root}")

    if n_complete != 6:
        raise SystemExit("ERROR: fewer than six model arrays archived")
    if n_match != 6:
        raise SystemExit("ERROR: one or more metrics do not match arrays")


if __name__ == "__main__":
    main()
