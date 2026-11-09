#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build revision-v2 chronological indices and train-only city-variable statistics.

This script reads the existing compact station arrays under processed/stage1_arrays.
It never modifies legacy arrays, indices, statistics, checkpoints, or results.

Splits:
  train: 2023-01-01 -- 2024-10-31
  val  : 2024-11-01 -- 2024-11-30
  cal  : 2024-12-01 -- 2024-12-31
  test : 2025-01-01 -- 2025-12-31

Every 96->48 window must lie completely inside one split. City-variable means and
standard deviations are estimated from the training period only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

L, H = 96, 48
TOTAL = L + H
DEFAULT_VARIABLES = [
    "PM2.5", "PM10", "SO2", "NO2", "O3", "CO",
    "temperature", "relative_humidity",
]


@dataclass(frozen=True)
class SplitSpec:
    start: pd.Timestamp
    end: pd.Timestamp
    stride: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument("--array_root", type=Path, default=None)
    p.add_argument("--out_root", type=Path, default=None)
    p.add_argument("--exclude_file", type=Path, default=None,
                   help="Optional text file: one city/station_id per line")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_exclusions(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    out: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.add(line.replace("\\", "/"))
    return out


def in_split(start: pd.Timestamp, end: pd.Timestamp, spec: SplitSpec) -> bool:
    return start >= spec.start and end <= spec.end


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    array_root = (args.array_root or root / "processed" / "stage1_arrays").resolve()
    out_root = (args.out_root or root / "experiments" / "data_v2").resolve()
    index_root = out_root / "indices"
    stats_path = out_root / "city_var_stats_train_only.json"
    qa_path = out_root / "index_qa.csv"
    manifest_path = out_root / "data_manifest.json"

    if not array_root.exists():
        raise FileNotFoundError(f"Array root not found: {array_root}")
    if out_root.exists() and any(out_root.iterdir()) and not args.overwrite:
        if stats_path.exists() and (index_root / "meta.json").exists():
            print(f"V2 data already exist: {out_root}")
            print("Use --overwrite to rebuild.")
            return
        raise RuntimeError(f"Output directory is non-empty: {out_root}")

    out_root.mkdir(parents=True, exist_ok=True)
    index_root.mkdir(parents=True, exist_ok=True)

    splits: Dict[str, SplitSpec] = {
        "train": SplitSpec(pd.Timestamp("2023-01-01 00:00:00"), pd.Timestamp("2024-10-31 23:00:00"), 12),
        "val": SplitSpec(pd.Timestamp("2024-11-01 00:00:00"), pd.Timestamp("2024-11-30 23:00:00"), 24),
        "cal": SplitSpec(pd.Timestamp("2024-12-01 00:00:00"), pd.Timestamp("2024-12-31 23:00:00"), 24),
        "test": SplitSpec(pd.Timestamp("2025-01-01 00:00:00"), pd.Timestamp("2025-12-31 23:00:00"), 24),
    }

    exclusions = load_exclusions(args.exclude_file)
    files = sorted(array_root.glob("*/*.npz"))
    if not files:
        raise RuntimeError(f"No station arrays found under {array_root}")

    records: List[Tuple[Path, str, str]] = []
    variables: List[str] | None = None
    for fp in files:
        city = fp.parent.name
        station_id = fp.stem
        key = f"{city}/{station_id}"
        if key in exclusions:
            continue
        with np.load(fp, allow_pickle=True) as z:
            vars_here = [str(x) for x in z["variables"].tolist()]
        if variables is None:
            variables = vars_here
        elif vars_here != variables:
            raise ValueError(f"Variable order mismatch in {fp}: {vars_here} vs {variables}")
        records.append((fp, city, station_id))

    assert variables is not None
    if "O3" not in variables:
        raise ValueError(f"O3 not found in variables: {variables}")
    o3_idx = variables.index("O3")
    cities = sorted({city for _, city, _ in records})
    city_to_id = {city: i for i, city in enumerate(cities)}

    stats_acc = {
        city: {var: {"sum": 0.0, "sumsq": 0.0, "count": 0} for var in variables}
        for city in cities
    }
    arrays_rel: List[str] = []
    station_meta: List[dict] = []
    idx_store = {
        split: {"array_id": [], "city_id": [], "var_idx": [], "start_idx": []}
        for split in splits
    }
    qa_rows: List[dict] = []
    test_targets: list[np.ndarray] = []

    for array_id, (fp, city, station_id) in enumerate(records):
        with np.load(fp, allow_pickle=True) as z:
            values = z["values"].astype(np.float32)
            time_ns = z["time_ns"].astype(np.int64)
        times = pd.to_datetime(time_ns)
        if values.ndim != 2 or values.shape[1] != len(variables):
            raise ValueError(f"Unexpected values shape in {fp}: {values.shape}")
        if len(times) != len(values):
            raise ValueError(f"Time/value length mismatch in {fp}")

        rel = str(fp.relative_to(root)) if fp.is_relative_to(root) else str(fp)
        arrays_rel.append(rel)
        station_meta.append({"array_id": array_id, "city": city, "station_id": station_id, "array_path": rel})

        train_mask = (times >= splits["train"].start) & (times <= splits["train"].end)
        train_values = values[np.asarray(train_mask)]
        for vi, var in enumerate(variables):
            vv = train_values[:, vi]
            vv = vv[np.isfinite(vv)]
            if vv.size:
                stats_acc[city][var]["sum"] += float(vv.sum(dtype=np.float64))
                stats_acc[city][var]["sumsq"] += float(np.square(vv, dtype=np.float64).sum())
                stats_acc[city][var]["count"] += int(vv.size)

        # Require all historical covariates and the O3 target to be finite.
        all_finite = np.all(np.isfinite(values), axis=1).astype(np.int32)
        rolling = np.convolve(all_finite, np.ones(TOTAL, dtype=np.int32), mode="valid")
        valid_starts = np.flatnonzero(rolling == TOTAL)

        per_split = {k: 0 for k in splits}
        for st in valid_starts:
            win_start = pd.Timestamp(times[st])
            win_end = pd.Timestamp(times[st + TOTAL - 1])
            for split, spec in splits.items():
                if st % spec.stride == 0 and in_split(win_start, win_end, spec):
                    idx_store[split]["array_id"].append(array_id)
                    idx_store[split]["city_id"].append(city_to_id[city])
                    idx_store[split]["var_idx"].append(o3_idx)
                    idx_store[split]["start_idx"].append(int(st))
                    if split == "test":
                        test_targets.append(values[st + L:st + L + H, o3_idx].copy())
                    per_split[split] += 1

        qa_rows.append({
            "city": city,
            "station_id": station_id,
            "n_rows": len(values),
            "finite_ratio_all_vars": float(np.all(np.isfinite(values), axis=1).mean()),
            **{f"n_{k}": v for k, v in per_split.items()},
        })
        if (array_id + 1) % 25 == 0 or array_id + 1 == len(records):
            print(f"Processed {array_id + 1}/{len(records)} station arrays", flush=True)

    stats_out = {
        "stats_source": "training split only",
        "stats_range": [str(splits["train"].start), str(splits["train"].end)],
        "variables": variables,
        "cities": cities,
        "city_variable_stats": {},
    }
    for city in cities:
        stats_out["city_variable_stats"][city] = {}
        for var in variables:
            a = stats_acc[city][var]
            n = a["count"]
            if n <= 0:
                raise RuntimeError(f"No training values for {city}/{var}")
            mean = a["sum"] / n
            var_pop = max(a["sumsq"] / n - mean * mean, 0.0)
            std = float(np.sqrt(var_pop))
            if not np.isfinite(std) or std <= 0:
                std = 1.0
            stats_out["city_variable_stats"][city][var] = {
                "mean": float(mean), "std": std, "count": int(n)
            }
    stats_path.write_text(json.dumps(stats_out, indent=2, ensure_ascii=False), encoding="utf-8")

    for split, payload in idx_store.items():
        np.savez_compressed(
            index_root / f"{split}.npz",
            array_id=np.asarray(payload["array_id"], dtype=np.int32),
            city_id=np.asarray(payload["city_id"], dtype=np.int16),
            var_idx=np.asarray(payload["var_idx"], dtype=np.int8),
            start_idx=np.asarray(payload["start_idx"], dtype=np.int32),
        )

    # Reference targets follow test.npz row order and use physical O3 units.
    targets = np.asarray(test_targets, dtype=np.float32).reshape(-1, H)
    np.save(index_root / "test_targets.npy", targets, allow_pickle=False)

    meta = {
        "root": str(root),
        "array_root": str(array_root),
        "variables": variables,
        "cities": cities,
        "array_paths": arrays_rel,
        "stations": station_meta,
        "o3_idx": o3_idx,
        "seq_len": L,
        "pred_len": H,
        "splits": {
            k: {"start": str(v.start), "end": str(v.end), "stride": v.stride,
                "n_samples": len(idx_store[k]["array_id"])}
            for k, v in splits.items()
        },
        "stats_path": str(stats_path),
        "exclusions": sorted(exclusions),
    }
    (index_root / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(qa_rows).to_csv(qa_path, index=False, encoding="utf-8-sig")

    manifest = {
        "meta_sha256": sha256(index_root / "meta.json"),
        "stats_sha256": sha256(stats_path),
        "test_targets_sha256": sha256(index_root / "test_targets.npy"),
        "index_sha256": {k: sha256(index_root / f"{k}.npz") for k in splits},
        "counts": {k: len(v["array_id"]) for k, v in idx_store.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nRevision-v2 data built successfully")
    print(f"Output: {out_root}")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
