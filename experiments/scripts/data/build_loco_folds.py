#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build 22 leave-one-city-out folds from canonical revision-v2 indices.

This script never rebuilds chronological windows. It only subsets the already
audited revision-v2 train/val/cal/test indices:

  train/val/cal: all source cities except the held-out city
  test         : held-out city only

For each fold it computes an exact source-only pooled scaler by combining the
canonical city-level train-only population moments of the 21 source cities:

  E[X]  = sum_i n_i * mu_i / sum_i n_i
  E[X²] = sum_i n_i * (sigma_i² + mu_i²) / sum_i n_i

The held-out city's statistics are never used by the fold dataset.

The fold also stores canonical test positions and exact physical test targets,
which are later used for array-order and no-leakage audits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap

L, H = 96, 48
SPLITS = ("train", "val", "cal", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    p.add_argument("--cities", nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_name(value: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in "._-" else "_"
        for ch in value
    )


def result_root(root: Path) -> Path:
    return root / "experiments/results/14_loco_22cities_v2"


def fold_root(root: Path, city: str) -> Path:
    return result_root(root) / "00_fold_indices" / safe_name(city)


def json_dump(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def pool_source_stats(
    canonical_stats: dict[str, Any],
    source_cities: list[str],
    variables: list[str],
) -> dict[str, Any]:
    pooled: dict[str, Any] = {}
    for variable in variables:
        total_n = 0
        total_sum = 0.0
        total_sumsq = 0.0
        source_rows = []
        for city in source_cities:
            stat = canonical_stats["city_variable_stats"][city][variable]
            n = int(stat["count"])
            mu = float(stat["mean"])
            std = float(stat["std"])
            total_n += n
            total_sum += n * mu
            total_sumsq += n * (std * std + mu * mu)
            source_rows.append({
                "city": city,
                "count": n,
                "mean": mu,
                "std": std,
            })
        if total_n <= 0:
            raise RuntimeError(
                f"No source training observations for {variable}"
            )
        mean = total_sum / total_n
        variance = max(total_sumsq / total_n - mean * mean, 0.0)
        std = float(np.sqrt(variance))
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        pooled[variable] = {
            "mean": float(mean),
            "std": float(std),
            "count": int(total_n),
            "source_city_count": len(source_cities),
            "moment_definition": "population",
            "source_city_moments": source_rows,
        }
    return pooled


def subset_index(
    canonical_index: Path,
    output: Path,
    *,
    heldout_id: int,
    split: str,
    o3_idx: int,
) -> tuple[int, np.ndarray]:
    idx = np.load(canonical_index, allow_pickle=False)
    array_id = idx["array_id"].astype(np.int32)
    city_id = idx["city_id"].astype(np.int16)
    var_idx = idx["var_idx"].astype(np.int8)
    start_idx = idx["start_idx"].astype(np.int32)

    o3_mask = var_idx == int(o3_idx)
    if split == "test":
        keep = o3_mask & (city_id == heldout_id)
    else:
        keep = o3_mask & (city_id != heldout_id)

    positions = np.flatnonzero(keep).astype(np.int64)
    np.savez_compressed(
        output,
        array_id=array_id[keep],
        city_id=city_id[keep],
        var_idx=var_idx[keep],
        start_idx=start_idx[keep],
        canonical_position=positions,
    )
    return int(keep.sum()), positions


def write_canonical_test_arrays(
    root: Path,
    meta: dict[str, Any],
    fold: Path,
    test_index: Path,
    o3_idx: int,
) -> dict[str, Any]:
    idx = np.load(test_index, allow_pickle=False)
    array_id = idx["array_id"].astype(np.int32)
    start_idx = idx["start_idx"].astype(np.int32)
    positions = idx["canonical_position"].astype(np.int64)
    n = len(array_id)

    true_path = fold / "canonical_test_true.npy"
    last_path = fold / "canonical_test_last_o3.npy"
    out_true = open_memmap(
        true_path, mode="w+", dtype=np.float32, shape=(n, H)
    )
    out_last = open_memmap(
        last_path, mode="w+", dtype=np.float32, shape=(n,)
    )

    cache: dict[int, np.ndarray] = {}
    for i, (aid, start) in enumerate(zip(array_id, start_idx)):
        aid_i = int(aid)
        if aid_i not in cache:
            path = root / meta["array_paths"][aid_i]
            with np.load(path, allow_pickle=True) as data:
                cache[aid_i] = data["values"].astype(np.float32)
        values = cache[aid_i]
        st = int(start)
        out_last[i] = values[st + L - 1, o3_idx]
        out_true[i] = values[st + L:st + L + H, o3_idx]
    out_true.flush()
    out_last.flush()

    return {
        "n_samples": n,
        "canonical_test_positions_shape": list(positions.shape),
        "true_shape": [n, H],
        "last_o3_shape": [n],
        "true_sha256": sha256(true_path),
        "last_o3_sha256": sha256(last_path),
    }


def build_one(
    root: Path,
    city: str,
    meta: dict[str, Any],
    stats: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    cities = list(meta["cities"])
    if city not in cities:
        raise ValueError(f"Unknown city: {city}")
    heldout_id = cities.index(city)
    source_cities = [value for value in cities if value != city]
    if len(source_cities) != len(cities) - 1:
        raise RuntimeError("Source-city count mismatch")

    fold = fold_root(root, city)
    manifest_path = fold / "fold_manifest.json"
    if manifest_path.exists() and not overwrite:
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            if (
                manifest.get("state") == "complete"
                and manifest.get("heldout_city") == city
            ):
                print(f"[FOLD SKIP] {city}", flush=True)
                return manifest
        except Exception:
            pass

    fold.mkdir(parents=True, exist_ok=True)
    index_root = root / "experiments/data_v2/indices"
    counts: dict[str, int] = {}
    test_positions = None
    index_hashes: dict[str, str] = {}
    for split in SPLITS:
        output = fold / f"{split}.npz"
        count, positions = subset_index(
            index_root / f"{split}.npz",
            output,
            heldout_id=heldout_id,
            split=split,
            o3_idx=int(meta["o3_idx"]),
        )
        counts[split] = count
        index_hashes[split] = sha256(output)
        if split == "test":
            test_positions = positions
            np.save(
                fold / "canonical_test_positions.npy",
                positions,
                allow_pickle=False,
            )

    if counts["train"] <= 0 or counts["val"] <= 0 or counts["test"] <= 0:
        raise RuntimeError(f"Empty LOCO split for {city}: {counts}")

    pooled = pool_source_stats(
        stats, source_cities, list(meta["variables"])
    )
    scaler = {
        "state": "complete",
        "protocol": "revision-v2",
        "normalization": "source-21 global train-only pooled scaler",
        "heldout_city": city,
        "heldout_city_id": heldout_id,
        "heldout_city_used_in_scaler": False,
        "source_cities": source_cities,
        "source_city_count": len(source_cities),
        "stats_source": stats.get("stats_source"),
        "stats_range": stats.get("stats_range"),
        "variables": list(meta["variables"]),
        "pooled_variable_stats": pooled,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_dump(fold / "source_train_stats.json", scaler)
    json_dump(fold / "source_cities.json", {
        "heldout_city": city,
        "source_cities": source_cities,
    })

    canonical = write_canonical_test_arrays(
        root,
        meta,
        fold,
        fold / "test.npz",
        int(meta["o3_idx"]),
    )

    if test_positions is None:
        raise RuntimeError("Missing canonical test positions")

    manifest = {
        "state": "complete",
        "protocol": "revision-v2",
        "heldout_city": city,
        "heldout_city_id": heldout_id,
        "source_cities": source_cities,
        "source_city_count": len(source_cities),
        "city_identity_mode": "shared_single_id",
        "target_city_in_train": False,
        "target_city_in_val": False,
        "target_city_stats_used": False,
        "split_counts": counts,
        "canonical_split_definition": meta.get("splits", {}),
        "index_sha256": index_hashes,
        "source_stats_sha256": sha256(
            fold / "source_train_stats.json"
        ),
        "canonical_test": canonical,
        "canonical_test_position_min": int(test_positions.min()),
        "canonical_test_position_max": int(test_positions.max()),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_dump(manifest_path, manifest)
    print(
        f"[FOLD BUILT] {city:<14} "
        f"train={counts['train']:,} val={counts['val']:,} "
        f"test={counts['test']:,}",
        flush=True,
    )
    return manifest


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    index_root = root / "experiments/data_v2/indices"
    meta_path = index_root / "meta.json"
    stats_path = (
        root / "experiments/data_v2/"
        "city_var_stats_train_only.json"
    )
    if not meta_path.exists() or not stats_path.exists():
        raise FileNotFoundError(
            "Canonical revision-v2 index/statistics are missing"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    cities = list(meta["cities"])
    if len(cities) != 22:
        raise RuntimeError(
            f"Expected 22 canonical cities, found {len(cities)}"
        )
    selected = args.cities or cities
    unknown = sorted(set(selected) - set(cities))
    if unknown:
        raise ValueError(f"Unknown cities: {unknown}")

    inventory = {
        "state": "running",
        "protocol": "revision-v2",
        "canonical_city_count": len(cities),
        "cities": {},
    }
    for city in selected:
        manifest = build_one(
            root, city, meta, stats, args.overwrite
        )
        inventory["cities"][city] = {
            "fold": str(fold_root(root, city)),
            "split_counts": manifest["split_counts"],
            "state": manifest["state"],
        }

    inventory["state"] = "complete"
    inventory["built_city_count"] = len(inventory["cities"])
    inventory["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json_dump(
        result_root(root) / "00_fold_indices/fold_inventory.json",
        inventory,
    )
    print(json.dumps(inventory, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
