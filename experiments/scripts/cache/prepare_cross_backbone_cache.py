#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Align existing frozen-TSFM caches to the canonical revision-v2 samples.

This script does not run or fine-tune the TSFM backbones. The project already
contains deterministic frozen predictions for Sundial, TimeMoE-200M and SEMPO
on train/validation/test caches. We:

1. export the canonical revision-v2 dataset order;
2. align each revision-v2 sample to its original cached sample using physical
   96-hour O3 history, 48-hour target and city ID;
3. verify every aligned target against the revision-v2 target;
4. verify the aligned test predictions against the audited revision-v2
   zero-shot archive;
5. save standardized and physical base forecasts for output-adapter training.

The script stops rather than silently using a cache when any alignment or
zero-shot verification fails.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader

L, H = 96, 48
SPLITS = ("train", "val", "test")
MODEL_KEYS = ("sundial", "timemoe_200m", "sempo")
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
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--match_atol", type=float, default=2e-3)
    return p.parse_args()


def import_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def result_root(root: Path) -> Path:
    return root / "experiments/results/12_cross_backbone_output_adapter_v2"


def canonical_root(root: Path) -> Path:
    return result_root(root) / "00_cache_alignment/canonical_v2"


def aligned_root(root: Path) -> Path:
    return result_root(root) / "00_cache_alignment/aligned"


def mapping_root(root: Path) -> Path:
    return result_root(root) / "00_cache_alignment/mappings"


def json_dump(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def find_station_key(batch: dict[str, Any]) -> Optional[str]:
    for key in (
        "station_id", "station_idx", "site_id",
        "site_idx", "station_code", "station",
    ):
        if key in batch:
            return key
    return None


def export_revision_v2(
    root: Path,
    split: str,
    *,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
) -> Path:
    out = canonical_root(root) / split
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not overwrite:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if data.get("state") == "complete":
            print(f"[CANONICAL SKIP] {split}: {out}", flush=True)
            return out

    out.mkdir(parents=True, exist_ok=True)
    source_path = (
        root / "experiments/scripts/train/"
        "train_attribution_reference.py"
    )
    source = import_source(
        source_path, f"_cross_backbone_source_{split}"
    )
    index_root = root / "experiments/data_v2/indices"
    stats_path = (
        root / "experiments/data_v2/"
        "city_var_stats_train_only.json"
    )
    _, cora = source.load_project_modules(
        root, index_root, stats_path
    )
    Dataset = cora.Stage1O3CoRAContextDataset
    dataset = Dataset(split, 0, 2026, "both")
    n = len(dataset)

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "drop_last": False,
        "pin_memory": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(**loader_kwargs)

    arrays = {
        "x": open_memmap(
            out / "x.npy", mode="w+", dtype=np.float32, shape=(n, L)
        ),
        "y_std": open_memmap(
            out / "y_std.npy", mode="w+", dtype=np.float32, shape=(n, H)
        ),
        "y_phys": open_memmap(
            out / "y_phys.npy", mode="w+", dtype=np.float32, shape=(n, H)
        ),
        "mu": open_memmap(
            out / "mu.npy", mode="w+", dtype=np.float32, shape=(n,)
        ),
        "std": open_memmap(
            out / "std.npy", mode="w+", dtype=np.float32, shape=(n,)
        ),
        "city_id": open_memmap(
            out / "city_id.npy", mode="w+", dtype=np.int16, shape=(n,)
        ),
        "var_idx": open_memmap(
            out / "var_idx.npy", mode="w+", dtype=np.int16, shape=(n,)
        ),
    }
    station_values: list[str] = []
    station_seen = False
    offset = 0
    for step, batch in enumerate(loader, 1):
        b = int(batch["x"].shape[0])
        end = offset + b
        arrays["x"][offset:end] = to_numpy(batch["x"]).astype(
            np.float32, copy=False
        )
        arrays["y_std"][offset:end] = to_numpy(batch["y_std"]).astype(
            np.float32, copy=False
        )
        arrays["y_phys"][offset:end] = to_numpy(batch["y_phys"]).astype(
            np.float32, copy=False
        )
        arrays["mu"][offset:end] = to_numpy(batch["mu"]).astype(
            np.float32, copy=False
        ).reshape(-1)
        arrays["std"][offset:end] = to_numpy(batch["std"]).astype(
            np.float32, copy=False
        ).reshape(-1)
        arrays["city_id"][offset:end] = to_numpy(batch["city_id"]).astype(
            np.int16, copy=False
        ).reshape(-1)
        arrays["var_idx"][offset:end] = to_numpy(batch["var_idx"]).astype(
            np.int16, copy=False
        ).reshape(-1)

        station_key = find_station_key(batch)
        if station_key is not None:
            station_seen = True
            station_values.extend(
                to_numpy(batch[station_key]).reshape(-1).astype(str).tolist()
            )

        offset = end
        if step == 1 or step % 500 == 0 or offset == n:
            print(
                f"[CANONICAL {split}] {offset:,}/{n:,}",
                flush=True,
            )

    if offset != n:
        raise RuntimeError(f"{split}: wrote {offset}, expected {n}")
    for array in arrays.values():
        array.flush()
    if station_seen:
        if len(station_values) != n:
            raise RuntimeError(
                f"{split}: station length={len(station_values)}, expected={n}"
            )
        np.save(out / "station_id.npy", np.asarray(station_values))

    x = np.load(out / "x.npy", mmap_mode="r")
    y_std = np.load(out / "y_std.npy", mmap_mode="r")
    y_phys = np.load(out / "y_phys.npy", mmap_mode="r")
    mu = np.load(out / "mu.npy", mmap_mode="r")
    std = np.load(out / "std.npy", mmap_mode="r")
    reconstructed = y_std[: min(n, 10000)] * std[: min(n, 10000), None] + mu[
        : min(n, 10000), None
    ]
    max_err = float(
        np.max(np.abs(reconstructed - y_phys[: min(n, 10000)]))
    )
    if max_err > 2e-3:
        raise RuntimeError(
            f"{split}: standardized/physical target mismatch={max_err}"
        )

    json_dump(
        manifest_path,
        {
            "state": "complete",
            "protocol": "revision-v2",
            "split": split,
            "n_samples": n,
            "x_shape": list(x.shape),
            "y_shape": list(y_phys.shape),
            "variables": list(dataset.variables),
            "cities": list(dataset.cities),
            "o3_idx": int(dataset.o3_idx),
            "station_id_saved": station_seen,
            "sample_reconstruction_max_abs_error": max_err,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    return out


def first_existing(candidates: list[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    return None


def locate_sundial_cache(root: Path, split: str) -> Path:
    base = root / "results/stage1_sundial_cache"
    candidates = [
        base / f"sundial_o3_{split}_phys_s20_with_inputs",
        base / f"sundial_o3_{split}_phys_s20",
    ]
    found = first_existing(candidates)
    if found:
        return found

    globs = sorted(
        [
            p for p in base.glob(f"*{split}*")
            if p.is_dir()
            and (p / "pred_mean.npy").exists()
            and (p / "true.npy").exists()
            and (p / "x.npy").exists()
        ],
        key=lambda p: (
            "with_inputs" not in p.name,
            "s20" not in p.name,
            p.name,
        ),
    )
    if not globs:
        raise FileNotFoundError(
            f"No Sundial {split} cache with pred/true/x under {base}"
        )
    return globs[0]


def locate_model_cache(root: Path, model: str, split: str) -> Path:
    if model == "sundial":
        return locate_sundial_cache(root, split)

    if model == "timemoe_200m":
        base = root / "results/stage1_timemoe_cache"
        exact = base / f"timemoe200m_o3_{split}_phys"
        if exact.exists():
            return exact
        patterns = [f"*200m*{split}*phys*", f"*{split}*"]
    elif model == "sempo":
        base = root / "results/stage1_sempo_cache"
        exact = base / f"sempo_o3_{split}_repeat_context"
        if exact.exists():
            return exact
        patterns = [f"*{split}*repeat_context*", f"*{split}*"]
    else:
        raise ValueError(model)

    choices: list[Path] = []
    for pattern in patterns:
        choices.extend(
            p for p in base.glob(pattern)
            if p.is_dir()
            and ((p / "pred_mean.npy").exists() or (p / "pred.npy").exists())
            and (p / "true.npy").exists()
        )
        if choices:
            break
    choices = sorted(set(choices))
    if not choices:
        raise FileNotFoundError(
            f"No cache found for model={model}, split={split}, base={base}"
        )
    return choices[0]


def prediction_file(cache_dir: Path) -> Path:
    for name in ("pred_mean.npy", "pred.npy", "prediction.npy"):
        path = cache_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No point prediction in {cache_dir}")


def old_physical_x(cache_dir: Path) -> np.ndarray:
    x = np.load(cache_dir / "x.npy", mmap_mode="r")
    mu = np.load(cache_dir / "mu.npy", mmap_mode="r")
    std = np.load(cache_dir / "std.npy", mmap_mode="r")
    if x.ndim != 2 or x.shape[1] != L:
        raise RuntimeError(f"Unexpected x shape in {cache_dir}: {x.shape}")
    return np.asarray(x, dtype=np.float32) * np.asarray(
        std, dtype=np.float32
    )[:, None] + np.asarray(mu, dtype=np.float32)[:, None]


def new_physical_x(canonical_dir: Path) -> np.ndarray:
    x = np.load(canonical_dir / "x.npy", mmap_mode="r")
    mu = np.load(canonical_dir / "mu.npy", mmap_mode="r")
    std = np.load(canonical_dir / "std.npy", mmap_mode="r")
    return np.asarray(x, dtype=np.float32) * np.asarray(
        std, dtype=np.float32
    )[:, None] + np.asarray(mu, dtype=np.float32)[:, None]


def chunk_allclose(
    a: np.ndarray,
    b: np.ndarray,
    atol: float,
    chunk: int = 4096,
) -> tuple[bool, float]:
    if a.shape != b.shape:
        return False, math.inf
    max_diff = 0.0
    for start in range(0, a.shape[0], chunk):
        end = min(start + chunk, a.shape[0])
        aa = np.asarray(a[start:end], dtype=np.float64)
        bb = np.asarray(b[start:end], dtype=np.float64)
        diff = float(np.max(np.abs(aa - bb)))
        max_diff = max(max_diff, diff)
        if diff > atol:
            return False, max_diff
    return True, max_diff


def locate_verified_test_mapping(root: Path) -> Optional[Path]:
    candidates = sorted(
        root.glob("v2_vs_original_test_index_audit_*/v2_to_old_index.npy"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        summary = path.parent / "AUDIT_SUMMARY.json"
        if not summary.exists():
            continue
        data = json.loads(summary.read_text(encoding="utf-8"))
        if (
            data.get("verdict") == "FULL_SUBSET_UNIQUE_TRUE_MATCH"
            and data.get("safe_to_reindex_old_predictions")
        ):
            return path
    return None


def row_keys(
    x_phys: np.ndarray,
    y_phys: np.ndarray,
    city: np.ndarray,
    decimals: int = 3,
) -> np.ndarray:
    x_cols = [0, 11, 23, 35, 47, 59, 71, 83, 95]
    y_cols = [0, 5, 11, 17, 23, 29, 35, 41, 47]
    scale = 10 ** decimals
    selected = np.concatenate(
        [
            np.asarray(city).reshape(-1, 1).astype(np.int64),
            np.rint(np.asarray(x_phys[:, x_cols]) * scale).astype(np.int64),
            np.rint(np.asarray(y_phys[:, y_cols]) * scale).astype(np.int64),
        ],
        axis=1,
    )
    return selected


def build_mapping_by_content(
    old_x: np.ndarray,
    old_y: np.ndarray,
    old_city: np.ndarray,
    new_x: np.ndarray,
    new_y: np.ndarray,
    new_city: np.ndarray,
    atol: float,
) -> np.ndarray:
    print("[MAPPING] building physical-content index", flush=True)
    old_keys = row_keys(old_x, old_y, old_city, decimals=3)
    new_keys = row_keys(new_x, new_y, new_city, decimals=3)

    buckets: dict[bytes, list[int]] = {}
    for idx, row in enumerate(old_keys):
        buckets.setdefault(row.tobytes(), []).append(idx)

    used = np.zeros(old_x.shape[0], dtype=bool)
    mapping = np.full(new_x.shape[0], -1, dtype=np.int64)
    misses: list[int] = []
    for new_idx, row in enumerate(new_keys):
        candidates = buckets.get(row.tobytes(), [])
        selected = -1
        for old_idx in candidates:
            if used[old_idx]:
                continue
            if int(old_city[old_idx]) != int(new_city[new_idx]):
                continue
            if not np.allclose(
                old_x[old_idx], new_x[new_idx], atol=atol, rtol=0.0
            ):
                continue
            if not np.allclose(
                old_y[old_idx], new_y[new_idx], atol=atol, rtol=0.0
            ):
                continue
            selected = old_idx
            break
        if selected < 0:
            misses.append(new_idx)
            if len(misses) >= 20:
                break
        else:
            mapping[new_idx] = selected
            used[selected] = True

        if (new_idx + 1) % 50000 == 0:
            print(
                f"[MAPPING] {new_idx + 1:,}/{new_x.shape[0]:,}",
                flush=True,
            )

    if misses:
        raise RuntimeError(
            "Could not map revision-v2 samples into the old cache. "
            f"First unmatched revision-v2 rows: {misses}. "
            "The cache may use a different split or data preprocessing."
        )
    if np.any(mapping < 0):
        raise RuntimeError("Mapping contains unresolved rows")
    if len(np.unique(mapping)) != len(mapping):
        raise RuntimeError("Mapping is not one-to-one")
    return mapping


def make_or_verify_mapping(
    root: Path,
    split: str,
    sundial_dir: Path,
    canonical_dir: Path,
    atol: float,
    overwrite: bool,
) -> Path:
    out = mapping_root(root) / f"{split}_v2_to_old.npy"
    manifest_path = mapping_root(root) / f"{split}_mapping_manifest.json"
    if out.exists() and manifest_path.exists() and not overwrite:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if data.get("state") == "complete":
            print(f"[MAPPING SKIP] {split}", flush=True)
            return out

    old_y = np.load(sundial_dir / "true.npy", mmap_mode="r")
    old_city = np.load(sundial_dir / "city_id.npy", mmap_mode="r")
    old_x = old_physical_x(sundial_dir)
    new_y = np.load(canonical_dir / "y_phys.npy", mmap_mode="r")
    new_city = np.load(canonical_dir / "city_id.npy", mmap_mode="r")
    new_x = new_physical_x(canonical_dir)

    mapping_source = "content_match"
    mapping: Optional[np.ndarray] = None
    if split == "test":
        verified = locate_verified_test_mapping(root)
        if verified is not None:
            candidate = np.load(verified)
            if candidate.shape == (new_y.shape[0],):
                mapping = candidate.astype(np.int64, copy=False)
                mapping_source = str(verified)

    if mapping is None and old_y.shape[0] == new_y.shape[0]:
        city_equal = np.array_equal(
            np.asarray(old_city), np.asarray(new_city)
        )
        x_equal, x_max = chunk_allclose(old_x, new_x, atol)
        y_equal, y_max = chunk_allclose(old_y, new_y, atol)
        if city_equal and x_equal and y_equal:
            mapping = np.arange(new_y.shape[0], dtype=np.int64)
            mapping_source = "identity"
            print(
                f"[MAPPING {split}] identity, x_max={x_max:.6g}, "
                f"y_max={y_max:.6g}",
                flush=True,
            )

    if mapping is None:
        mapping = build_mapping_by_content(
            old_x,
            np.asarray(old_y),
            np.asarray(old_city),
            new_x,
            np.asarray(new_y),
            np.asarray(new_city),
            atol,
        )

    if mapping.min() < 0 or mapping.max() >= old_y.shape[0]:
        raise RuntimeError(f"{split}: mapping outside old-cache bounds")
    if len(np.unique(mapping)) != len(mapping):
        raise RuntimeError(f"{split}: mapping is not unique")

    mapped_city = np.asarray(old_city[mapping])
    if not np.array_equal(mapped_city, np.asarray(new_city)):
        raise RuntimeError(f"{split}: mapped city IDs differ")
    x_ok, x_max = chunk_allclose(old_x[mapping], new_x, atol)
    y_ok, y_max = chunk_allclose(old_y[mapping], new_y, atol)
    if not x_ok or not y_ok:
        raise RuntimeError(
            f"{split}: final mapping verification failed: "
            f"x_max={x_max}, y_max={y_max}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, mapping)
    json_dump(
        manifest_path,
        {
            "state": "complete",
            "protocol": "revision-v2",
            "split": split,
            "source_cache": str(sundial_dir),
            "mapping_source": mapping_source,
            "old_samples": int(old_y.shape[0]),
            "v2_samples": int(new_y.shape[0]),
            "mapping_unique": True,
            "mapped_x_max_abs_diff": x_max,
            "mapped_y_max_abs_diff": y_max,
            "match_atol": atol,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    return out


def symlink_or_replace(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    relative = os.path.relpath(target.resolve(), link.parent.resolve())
    os.symlink(relative, link)


def archive_zero_shot_pred(root: Path, model: str) -> Path:
    display = DISPLAY_NAMES[model]
    return (
        root / "experiments/results/"
        f"03_zeroshot_baselines_v2_archive/{display}/pred.npy"
    )


def align_model_split(
    root: Path,
    model: str,
    split: str,
    source_dir: Path,
    canonical_dir: Path,
    mapping_path: Path,
    *,
    overwrite: bool,
    atol: float,
) -> None:
    out = aligned_root(root) / model / split
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not overwrite:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if data.get("state") == "complete":
            print(f"[ALIGN SKIP] {model}/{split}", flush=True)
            return
    out.mkdir(parents=True, exist_ok=True)

    mapping = np.load(mapping_path)
    pred_src = prediction_file(source_dir)
    old_pred = np.load(pred_src, mmap_mode="r")
    old_true = np.load(source_dir / "true.npy", mmap_mode="r")
    old_city = np.load(source_dir / "city_id.npy", mmap_mode="r")
    new_true = np.load(canonical_dir / "y_phys.npy", mmap_mode="r")
    new_city = np.load(canonical_dir / "city_id.npy", mmap_mode="r")
    mu = np.load(canonical_dir / "mu.npy", mmap_mode="r")
    std = np.load(canonical_dir / "std.npy", mmap_mode="r")

    if old_pred.shape[0] != old_true.shape[0]:
        raise RuntimeError(
            f"{model}/{split}: pred/true source lengths differ"
        )
    if mapping.max() >= old_pred.shape[0]:
        raise RuntimeError(
            f"{model}/{split}: canonical mapping does not fit model cache"
        )
    if not np.array_equal(np.asarray(old_city[mapping]), np.asarray(new_city)):
        raise RuntimeError(
            f"{model}/{split}: model-cache city order differs from Sundial cache"
        )
    true_ok, true_max = chunk_allclose(
        old_true[mapping], new_true, atol
    )
    if not true_ok:
        raise RuntimeError(
            f"{model}/{split}: mapped true mismatch={true_max}; "
            "model cache does not share the canonical sample order"
        )

    n = new_true.shape[0]
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
        p = np.asarray(old_pred[mapping[start:end]], dtype=np.float32)
        base_phys[start:end] = p
        standardized = (
            p - np.asarray(mu[start:end], dtype=np.float32)[:, None]
        ) / np.asarray(std[start:end], dtype=np.float32)[:, None]
        base_std[start:end] = standardized
        nonfinite += int((~np.isfinite(p)).sum())
        nonfinite += int((~np.isfinite(standardized)).sum())
    base_phys.flush()
    base_std.flush()
    if nonfinite:
        raise RuntimeError(
            f"{model}/{split}: aligned base predictions contain nonfinite values"
        )

    for name in (
        "x.npy", "y_std.npy", "y_phys.npy", "mu.npy", "std.npy",
        "city_id.npy", "var_idx.npy", "station_id.npy",
    ):
        source = canonical_dir / name
        if source.exists():
            symlink_or_replace(source, out / name)

    archived_match = None
    archived_max_diff = None
    if split == "test":
        archived = archive_zero_shot_pred(root, model)
        if not archived.exists():
            raise FileNotFoundError(
                f"Audited zero-shot archive missing: {archived}"
            )
        archived_pred = np.load(archived, mmap_mode="r")
        aligned_pred = np.load(out / "base_pred_phys.npy", mmap_mode="r")
        archived_match, archived_max_diff = chunk_allclose(
            aligned_pred, archived_pred, atol=5e-5
        )
        if not archived_match:
            raise RuntimeError(
                f"{model}: aligned test forecast does not reproduce the "
                f"audited zero-shot archive; max_diff={archived_max_diff}"
            )

    json_dump(
        manifest_path,
        {
            "state": "complete",
            "protocol": "revision-v2",
            "model": model,
            "display_name": DISPLAY_NAMES[model],
            "split": split,
            "source_cache": str(source_dir),
            "source_prediction": str(pred_src),
            "mapping": str(mapping_path),
            "n_samples": int(n),
            "base_pred_shape": [int(n), H],
            "mapped_true_max_abs_diff": true_max,
            "test_matches_audited_zero_shot": archived_match,
            "test_zero_shot_max_abs_diff": archived_max_diff,
            "nonfinite_count": nonfinite,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out = result_root(root)
    out.mkdir(parents=True, exist_ok=True)

    inventory: dict[str, Any] = {
        "protocol": "revision-v2",
        "models": {},
        "splits": {},
    }

    canonical_dirs: dict[str, Path] = {}
    sundial_dirs: dict[str, Path] = {}
    mapping_paths: dict[str, Path] = {}

    for split in SPLITS:
        canonical_dirs[split] = export_revision_v2(
            root,
            split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
        )
        sundial_dirs[split] = locate_sundial_cache(root, split)
        mapping_paths[split] = make_or_verify_mapping(
            root,
            split,
            sundial_dirs[split],
            canonical_dirs[split],
            args.match_atol,
            args.overwrite,
        )
        inventory["splits"][split] = {
            "canonical_dir": str(canonical_dirs[split]),
            "sundial_source": str(sundial_dirs[split]),
            "mapping": str(mapping_paths[split]),
        }

    for model in MODEL_KEYS:
        inventory["models"][model] = {}
        for split in SPLITS:
            source = locate_model_cache(root, model, split)
            align_model_split(
                root,
                model,
                split,
                source,
                canonical_dirs[split],
                mapping_paths[split],
                overwrite=args.overwrite,
                atol=args.match_atol,
            )
            inventory["models"][model][split] = str(source)

    inventory.update(
        {
            "state": "complete",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "aligned_root": str(aligned_root(root)),
        }
    )
    json_dump(
        result_root(root) / "00_cache_alignment/cache_inventory.json",
        inventory,
    )
    print(
        json.dumps(inventory, indent=2, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
