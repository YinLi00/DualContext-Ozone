#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finalize native frozen-backbone predictions for Output Adapter training."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap

MODELS = ("sundial", "timemoe_200m", "sempo")
SPLITS = ("train", "val", "test")
H = 48
DISPLAY_NAMES = {
    "sundial": "Sundial",
    "timemoe_200m": "TimeMoE-200M",
    "sempo": "SEMPO",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    p.add_argument("--mode", choices=["smoke", "formal"], required=True)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def result_root(root: Path) -> Path:
    return root / "experiments/results/12_cross_backbone_output_adapter_v2"


def input_dir(root: Path, split: str) -> Path:
    return result_root(root) / f"00_native_cache/input/{split}"


def raw_dir(root: Path, model: str, split: str, smoke: bool) -> Path:
    prefix = "smoke_raw" if smoke else "raw"
    return result_root(root) / f"00_native_cache/{prefix}/{model}/{split}"


def aligned_dir(root: Path, model: str, split: str, smoke: bool) -> Path:
    prefix = "smoke_aligned" if smoke else "aligned"
    return result_root(root) / f"00_native_cache/{prefix}/{model}/{split}"


def archive_pred(root: Path, model: str) -> Path:
    return (
        root / "experiments/results/"
        f"03_zeroshot_baselines_v2_archive/"
        f"{DISPLAY_NAMES[model]}/pred.npy"
    )


def json_dump(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def replace_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    relative = os.path.relpath(target.resolve(), link.parent.resolve())
    os.symlink(relative, link)


def prediction_file(directory: Path) -> Path:
    for name in ("pred_mean.npy", "pred.npy"):
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No prediction file in {directory}")


def chunk_equal(a: np.ndarray, b: np.ndarray, atol: float = 2e-3) -> float:
    if a.shape != b.shape:
        raise RuntimeError(f"Shape mismatch: {a.shape} vs {b.shape}")
    maximum = 0.0
    for start in range(0, a.shape[0], 8192):
        end = min(start + 8192, a.shape[0])
        diff = float(
            np.max(
                np.abs(
                    np.asarray(a[start:end], dtype=np.float64)
                    - np.asarray(b[start:end], dtype=np.float64)
                )
            )
        )
        maximum = max(maximum, diff)
        if diff > atol:
            raise RuntimeError(
                f"Array mismatch exceeds atol={atol}: {maximum}"
            )
    return maximum


def finalize_one(
    root: Path,
    model: str,
    split: str,
    smoke: bool,
    overwrite: bool,
) -> None:
    out = aligned_dir(root, model, split, smoke)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not overwrite:
        try:
            if json.loads(
                manifest_path.read_text(encoding="utf-8")
            ).get("state") == "complete":
                print(f"[FINALIZE SKIP] {model}/{split}", flush=True)
                return
        except Exception:
            pass
    out.mkdir(parents=True, exist_ok=True)

    if smoke:
        source = raw_dir(root, model, "test", True)
        source_input = source
        pred_path = prediction_file(source)
    elif split == "test":
        source = archive_pred(root, model).parent
        source_input = input_dir(root, "test")
        pred_path = archive_pred(root, model)
    else:
        source = raw_dir(root, model, split, False)
        source_input = source
        pred_path = prediction_file(source)

    pred = np.load(pred_path, mmap_mode="r")
    true = np.load(source_input / "true.npy", mmap_mode="r")
    x = np.load(source_input / "x.npy", mmap_mode="r")
    mu = np.load(source_input / "mu.npy", mmap_mode="r")
    std = np.load(source_input / "std.npy", mmap_mode="r")
    city = np.load(source_input / "city_id.npy", mmap_mode="r")
    var_idx = np.load(source_input / "var_idx.npy", mmap_mode="r")

    n = int(pred.shape[0])
    if pred.shape != (n, H):
        raise RuntimeError(f"{model}/{split}: pred shape={pred.shape}")
    for name, array in {
        "true": true,
        "x": x,
        "mu": mu,
        "std": std,
        "city": city,
        "var_idx": var_idx,
    }.items():
        if array.shape[0] < n:
            raise RuntimeError(
                f"{model}/{split}: {name} length={array.shape[0]}, pred={n}"
            )

    if not smoke and split in ("train", "val"):
        native_true = np.load(
            input_dir(root, split) / "true.npy", mmap_mode="r"
        )
        native_city = np.load(
            input_dir(root, split) / "city_id.npy", mmap_mode="r"
        )
        if true.shape != native_true.shape:
            raise RuntimeError(
                f"{model}/{split}: raw true shape differs from native input"
            )
        true_max = chunk_equal(true, native_true, atol=2e-3)
        if not np.array_equal(np.asarray(city), np.asarray(native_city)):
            raise RuntimeError(
                f"{model}/{split}: city IDs differ from native input"
            )
    else:
        true_max = 0.0

    base_phys = open_memmap(
        out / "base_pred_phys.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n, H),
    )
    base_std = open_memmap(
        out / "base_pred_std.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n, H),
    )
    nonfinite = 0
    for start in range(0, n, 8192):
        end = min(start + 8192, n)
        p = np.asarray(pred[start:end], dtype=np.float32)
        m = np.asarray(mu[start:end], dtype=np.float32)[:, None]
        s = np.asarray(std[start:end], dtype=np.float32)[:, None]
        z = (p - m) / s
        base_phys[start:end] = p
        base_std[start:end] = z
        nonfinite += int((~np.isfinite(p)).sum())
        nonfinite += int((~np.isfinite(z)).sum())
    base_phys.flush()
    base_std.flush()
    if nonfinite:
        raise RuntimeError(
            f"{model}/{split}: nonfinite predictions={nonfinite}"
        )

    mapping = {
        "x.npy": "x.npy",
        "true.npy": "true.npy",
        "y_phys.npy": "true.npy",
        "mu.npy": "mu.npy",
        "std.npy": "std.npy",
        "city_id.npy": "city_id.npy",
        "var_idx.npy": "var_idx.npy",
    }
    y_std_source = source_input / "y_std.npy"
    if y_std_source.exists():
        mapping["y_std.npy"] = "y_std.npy"
    for dest, src in mapping.items():
        replace_symlink(source_input / src, out / dest)

    json_dump(
        manifest_path,
        {
            "state": "complete",
            "protocol": "revision-v2",
            "mode": "smoke" if smoke else "formal",
            "model": model,
            "backbone": DISPLAY_NAMES[model],
            "split": split,
            "prediction_source": str(pred_path),
            "input_source": str(source_input),
            "n_samples": n,
            "base_pred_shape": [n, H],
            "raw_true_max_abs_diff_vs_native": true_max,
            "test_anchored_to_audited_zero_shot": (
                (not smoke) and split == "test"
            ),
            "nonfinite_count": nonfinite,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    smoke = args.mode == "smoke"
    for split in SPLITS:
        finalize_one(
            root,
            args.model,
            split,
            smoke,
            args.overwrite,
        )


if __name__ == "__main__":
    main()
