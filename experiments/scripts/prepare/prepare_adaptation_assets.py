#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare deterministic train-only assets for reviewer45 experiments.

Assets:
  * city x 12 seasonal bins x 24 hour climatology in standardized O3 units;
  * frozen global multi-output ridge AR mapping standardized 96 -> 48;
  * city-specific variable-to-O3 mutual-information relevance table.

No validation, calibration or test observation is used.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

L, H = 96, 48


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    p.add_argument("--mode", choices=["smoke", "formal"], default="formal")
    p.add_argument("--linear_samples", type=int, default=None)
    p.add_argument("--linear_ridge", type=float, default=1.0)
    p.add_argument("--mi_windows_per_city", type=int, default=None)
    p.add_argument("--mi_neighbors", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def result_root(root: Path) -> Path:
    return root / "experiments/results/15_reviewer45_v2"


def asset_root(root: Path, mode: str) -> Path:
    return result_root(root) / "00_assets" / mode


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_dump(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def resolve_array(root: Path, raw: str) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else root / path


def as_numpy(value: Any, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def prepare_climatology(
    root: Path,
    out: Path,
    meta: dict[str, Any],
    stats: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    directory = out / "climatology"
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and not overwrite:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if data.get("state") == "complete":
            print("[ASSET SKIP] climatology", flush=True)
            return data
    directory.mkdir(parents=True, exist_ok=True)

    cities = list(meta["cities"])
    variables = list(meta["variables"])
    o3_idx = int(meta.get("o3_idx", variables.index("O3")))
    index_root = root / "experiments/data_v2/indices"
    train_idx = np.load(index_root / "train.npz", allow_pickle=False)
    aid_all = train_idx["array_id"].astype(np.int32)
    cid_all = train_idx["city_id"].astype(np.int16)
    vid_all = train_idx["var_idx"].astype(np.int8)
    mask = vid_all == o3_idx
    aid_all = aid_all[mask]
    cid_all = cid_all[mask]

    aid_to_city: dict[int, int] = {}
    for aid, cid in zip(aid_all, cid_all):
        aid_i, cid_i = int(aid), int(cid)
        old = aid_to_city.get(aid_i)
        if old is not None and old != cid_i:
            raise RuntimeError(f"array_id={aid_i} maps to two cities")
        aid_to_city[aid_i] = cid_i

    split = meta["splits"]["train"]
    start = np.datetime64(str(split["start"]), "ns")
    end = np.datetime64(str(split["end"]), "ns")
    sums = np.zeros((len(cities), 12, 24), dtype=np.float64)
    counts = np.zeros((len(cities), 12, 24), dtype=np.int64)
    finite_total = 0

    for number, (aid, cid) in enumerate(sorted(aid_to_city.items()), 1):
        path = resolve_array(root, meta["array_paths"][aid])
        with np.load(path, allow_pickle=True) as data:
            values = np.asarray(data["values"], dtype=np.float32)
            time_ns = np.asarray(data["time_ns"]).astype("datetime64[ns]")
        select = (
            (time_ns >= start)
            & (time_ns <= end)
            & np.isfinite(values[:, o3_idx])
        )
        if not np.any(select):
            continue
        times = time_ns[select]
        ozone = values[select, o3_idx].astype(np.float64)
        hour = (
            times.astype("datetime64[h]").astype(np.int64) % 24
        ).astype(np.int64)
        day = times.astype("datetime64[D]")
        year_start = (
            times.astype("datetime64[Y]").astype("datetime64[D]")
        )
        doy = (day - year_start).astype(np.int64)
        season = np.clip(
            np.floor(doy * 12.0 / 366.0).astype(np.int64),
            0, 11,
        )
        np.add.at(sums, (cid, season, hour), ozone)
        np.add.at(counts, (cid, season, hour), 1)
        finite_total += len(ozone)
        if number == 1 or number % 50 == 0 or number == len(aid_to_city):
            print(
                f"[CLIMATOLOGY] arrays {number}/{len(aid_to_city)}",
                flush=True,
            )

    physical = np.zeros_like(sums, dtype=np.float32)
    standardized = np.zeros_like(sums, dtype=np.float32)
    missing_bins = 0
    for cid, city in enumerate(cities):
        stat = stats["city_variable_stats"][city][variables[o3_idx]]
        mu = float(stat["mean"])
        std = float(stat["std"])
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        valid = counts[cid] > 0
        physical[cid, valid] = (
            sums[cid, valid] / counts[cid, valid]
        ).astype(np.float32)
        physical[cid, ~valid] = np.float32(mu)
        standardized[cid] = (physical[cid] - mu) / std
        missing_bins += int((~valid).sum())

    np.save(
        directory / "climatology_phys.npy",
        physical, allow_pickle=False,
    )
    np.save(
        directory / "climatology_std.npy",
        standardized, allow_pickle=False,
    )
    np.save(directory / "counts.npy", counts, allow_pickle=False)
    manifest = {
        "state": "complete",
        "protocol": "revision-v2",
        "source": "raw station arrays restricted to canonical train range",
        "train_range": split,
        "shape": list(standardized.shape),
        "season_definition": "12 equal day-of-year bins",
        "hour_definition": "local timestamp hour 0-23",
        "finite_o3_observations": int(finite_total),
        "station_arrays": len(aid_to_city),
        "missing_city_season_hour_bins_filled_with_city_train_mean":
            missing_bins,
        "target_stats_source": stats.get("stats_source"),
        "climatology_std_sha256":
            sha256(directory / "climatology_std.npy"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_dump(manifest_path, manifest)
    print("[ASSET COMPLETE] climatology", flush=True)
    return manifest


def prepare_linear_ar(
    out: Path,
    Dataset,
    sample_count: int,
    ridge: float,
    overwrite: bool,
) -> dict[str, Any]:
    directory = out / "linear_ar"
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and not overwrite:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            data.get("state") == "complete"
            and int(data.get("requested_samples", -1)) == sample_count
            and abs(float(data.get("ridge", -1)) - ridge) < 1e-12
        ):
            print("[ASSET SKIP] linear_ar", flush=True)
            return data
    directory.mkdir(parents=True, exist_ok=True)

    dataset = Dataset("train", sample_count, 2026, "both")
    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    gram = np.zeros((L + 1, L + 1), dtype=np.float64)
    rhs = np.zeros((L + 1, H), dtype=np.float64)
    rows = 0
    for step, batch in enumerate(loader, 1):
        x = as_numpy(batch["x"], np.float64)
        y = as_numpy(batch["y_std"], np.float64)
        x_aug = np.concatenate(
            [x, np.ones((len(x), 1), dtype=np.float64)],
            axis=1,
        )
        gram += x_aug.T @ x_aug
        rhs += x_aug.T @ y
        rows += len(x)
        if step == 1 or step % 20 == 0 or step == len(loader):
            print(
                f"[LINEAR AR] batches {step}/{len(loader)} rows={rows:,}",
                flush=True,
            )
    penalty = np.eye(L + 1, dtype=np.float64) * float(ridge)
    penalty[-1, -1] = 0.0
    coef = np.linalg.solve(gram + penalty, rhs)
    w = coef[:-1].astype(np.float32)
    b = coef[-1].astype(np.float32)
    np.save(directory / "W.npy", w, allow_pickle=False)
    np.save(directory / "b.npy", b, allow_pickle=False)

    manifest = {
        "state": "complete",
        "protocol": "revision-v2",
        "model": "global multi-output ridge AR",
        "input": "standardized O3 history [96]",
        "output": "standardized O3 future [48]",
        "requested_samples": int(sample_count),
        "actual_samples": int(rows),
        "ridge": float(ridge),
        "W_shape": list(w.shape),
        "b_shape": list(b.shape),
        "W_sha256": sha256(directory / "W.npy"),
        "b_sha256": sha256(directory / "b.npy"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_dump(manifest_path, manifest)
    print("[ASSET COMPLETE] linear_ar", flush=True)
    return manifest


def histogram_mi(x: np.ndarray, y: np.ndarray,
                 bins: int = 16) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 100:
        return 0.0
    x_edges = np.unique(np.quantile(
        x, np.linspace(0, 1, bins + 1)
    ))
    y_edges = np.unique(np.quantile(
        y, np.linspace(0, 1, bins + 1)
    ))
    if len(x_edges) < 3 or len(y_edges) < 3:
        return 0.0
    table, _, _ = np.histogram2d(
        x, y, bins=(x_edges, y_edges)
    )
    pxy = table / max(table.sum(), 1.0)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px @ py
    valid = (pxy > 0) & (denom > 0)
    return float(np.sum(
        pxy[valid] * np.log(pxy[valid] / denom[valid])
    ))


def prepare_mi(
    out: Path,
    Dataset,
    windows_per_city: int,
    neighbors: int,
    overwrite: bool,
) -> dict[str, Any]:
    directory = out / "mutual_information"
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and not overwrite:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            data.get("state") == "complete"
            and int(data.get("windows_per_city", -1))
            == windows_per_city
        ):
            print("[ASSET SKIP] mutual_information", flush=True)
            return data
    directory.mkdir(parents=True, exist_ok=True)

    dataset = Dataset("train", 0, 2026, "both")
    n_cities = len(dataset.cities)
    n_vars = len(dataset.variables)
    raw = np.zeros((n_cities, n_vars), dtype=np.float64)
    counts = np.zeros(n_cities, dtype=np.int64)
    rng = np.random.default_rng(2026)
    method = "sklearn_mutual_info_regression"
    try:
        from sklearn.feature_selection import mutual_info_regression
    except Exception:
        mutual_info_regression = None
        method = "quantile_histogram_fallback"

    city_vector = np.asarray(dataset.city_id, dtype=np.int64)
    for cid, city in enumerate(dataset.cities):
        candidates = np.flatnonzero(city_vector == cid)
        if len(candidates) == 0:
            continue
        take = min(int(windows_per_city), len(candidates))
        selected = rng.choice(candidates, size=take, replace=False)
        blocks = []
        for number, index in enumerate(selected, 1):
            sample = dataset[int(index)]
            blocks.append(
                as_numpy(sample["x_multi"], np.float32)
            )
        matrix = np.concatenate(blocks, axis=0).astype(np.float64)
        target = matrix[:, dataset.o3_idx]
        counts[cid] = len(matrix)
        if mutual_info_regression is not None:
            values = mutual_info_regression(
                matrix,
                target,
                discrete_features=False,
                n_neighbors=int(neighbors),
                random_state=2026,
            )
        else:
            values = np.asarray([
                histogram_mi(matrix[:, vid], target)
                for vid in range(n_vars)
            ], dtype=np.float64)
        raw[cid] = np.nan_to_num(
            values, nan=0.0, posinf=0.0, neginf=0.0
        )
        # O3-to-O3 MI is a trivial self-dependence and must not participate
        # in covariate relevance. O3 history remains available via o3_ctx.
        raw[cid, dataset.o3_idx] = 0.0
        print(
            f"[MI] city {cid+1}/{n_cities} {city} "
            f"windows={take} points={len(matrix):,}",
            flush=True,
        )

    normalized = np.zeros_like(raw, dtype=np.float32)
    for cid in range(n_cities):
        row = np.maximum(raw[cid], 0.0)
        maximum = float(row.max())
        normalized[cid] = (
            (row / maximum).astype(np.float32)
            if maximum > 0 else np.zeros(n_vars, dtype=np.float32)
        )
    np.save(directory / "mi_city_var_raw.npy",
            raw.astype(np.float32), allow_pickle=False)
    np.save(directory / "mi_city_var.npy",
            normalized, allow_pickle=False)
    manifest = {
        "state": "complete",
        "protocol": "revision-v2",
        "source": "canonical train historical windows only",
        "method": method,
        "windows_per_city": int(windows_per_city),
        "neighbors": int(neighbors),
        "points_per_city": counts.tolist(),
        "shape": list(normalized.shape),
        "target_channel_excluded": True,
        "target_variable": dataset.variables[dataset.o3_idx],
        "target_index": int(dataset.o3_idx),
        "variables": list(dataset.variables),
        "cities": list(dataset.cities),
        "mi_sha256": sha256(directory / "mi_city_var.npy"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_dump(manifest_path, manifest)
    print("[ASSET COMPLETE] mutual_information", flush=True)
    return manifest


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out = asset_root(root, args.mode)
    out.mkdir(parents=True, exist_ok=True)
    linear_samples = (
        args.linear_samples
        if args.linear_samples is not None
        else (512 if args.mode == "smoke" else 0)
    )
    mi_windows = (
        args.mi_windows_per_city
        if args.mi_windows_per_city is not None
        else (8 if args.mode == "smoke" else 128)
    )

    source = import_file(
        root / "experiments/scripts/train/"
        "train_attribution_reference.py",
        "_reviewer45_asset_source",
    )
    cipda, cora = source.load_project_modules(
        root,
        root / "experiments/data_v2/indices",
        root / "experiments/data_v2/"
        "city_var_stats_train_only.json",
    )
    Dataset = cora.Stage1O3CoRAContextDataset
    meta = json.loads(
        (
            root / "experiments/data_v2/indices/meta.json"
        ).read_text(encoding="utf-8")
    )
    stats = json.loads(
        (
            root / "experiments/data_v2/"
            "city_var_stats_train_only.json"
        ).read_text(encoding="utf-8")
    )

    inventory = {
        "state": "running",
        "mode": args.mode,
        "protocol": "revision-v2",
        "linear_samples": linear_samples,
        "mi_windows_per_city": mi_windows,
    }
    json_dump(out / "asset_inventory.json", inventory)
    inventory["climatology"] = prepare_climatology(
        root, out, meta, stats, args.overwrite
    )
    inventory["linear_ar"] = prepare_linear_ar(
        out, Dataset, linear_samples,
        args.linear_ridge, args.overwrite,
    )
    inventory["mutual_information"] = prepare_mi(
        out, Dataset, mi_windows,
        args.mi_neighbors, args.overwrite,
    )
    inventory["state"] = "complete"
    inventory["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json_dump(out / "asset_inventory.json", inventory)
    print(json.dumps(inventory, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
