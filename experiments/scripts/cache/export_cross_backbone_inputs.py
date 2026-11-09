#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export canonical revision-v2 O3 input caches for frozen TSFM inference.

The existing failed alignment package already exported part of the canonical
revision-v2 dataset. This script safely reuses any complete split and exports
missing splits. It then creates the exact file interface required by the
existing TimeMoE and SEMPO cache scripts:

    x.npy       standardized 96-hour O3 history
    true.npy    physical 48-hour O3 target
    y_std.npy   standardized target
    mu.npy
    std.npy
    city_id.npy
    var_idx.npy

No GPU is used.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader

L, H = 96, 48
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def import_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def result_root(root: Path) -> Path:
    return root / "experiments/results/12_cross_backbone_output_adapter_v2"


def canonical_dir(root: Path, split: str) -> Path:
    # Reuse the already-created canonical tree from the failed mapping attempt.
    return (
        result_root(root)
        / f"00_cache_alignment/canonical_v2/{split}"
    )


def native_input_dir(root: Path, split: str) -> Path:
    return result_root(root) / f"00_native_cache/input/{split}"


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def json_dump(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def canonical_complete(path: Path) -> bool:
    manifest = path / "manifest.json"
    required = [
        "x.npy", "y_std.npy", "y_phys.npy",
        "mu.npy", "std.npy", "city_id.npy", "var_idx.npy",
    ]
    if not manifest.exists() or not all((path / x).exists() for x in required):
        return False
    try:
        return json.loads(
            manifest.read_text(encoding="utf-8")
        ).get("state") == "complete"
    except Exception:
        return False


def export_split(
    root: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
) -> Path:
    out = canonical_dir(root, split)
    if canonical_complete(out) and not overwrite:
        print(f"[CANONICAL SKIP] {split}: {out}", flush=True)
        return out

    out.mkdir(parents=True, exist_ok=True)
    source_path = (
        root / "experiments/scripts/train/"
        "train_attribution_reference.py"
    )
    source = import_source(
        source_path, f"_native_input_source_{split}"
    )
    index_root = root / "experiments/data_v2/indices"
    stats_path = (
        root / "experiments/data_v2/"
        "city_var_stats_train_only.json"
    )
    _, cora = source.load_project_modules(root, index_root, stats_path)
    Dataset = cora.Stage1O3CoRAContextDataset
    dataset = Dataset(split, 0, 2026, "both")
    n = len(dataset)

    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "drop_last": False,
        "pin_memory": False,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    loader = DataLoader(**kwargs)

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
        offset = end
        if step == 1 or step % 500 == 0 or offset == n:
            print(
                f"[CANONICAL {split}] {offset:,}/{n:,}",
                flush=True,
            )

    if offset != n:
        raise RuntimeError(f"{split}: wrote {offset}, expected {n}")
    for arr in arrays.values():
        arr.flush()

    probe_n = min(n, 10000)
    y_std = np.load(out / "y_std.npy", mmap_mode="r")
    y_phys = np.load(out / "y_phys.npy", mmap_mode="r")
    mu = np.load(out / "mu.npy", mmap_mode="r")
    std = np.load(out / "std.npy", mmap_mode="r")
    reconstructed = (
        y_std[:probe_n] * std[:probe_n, None] + mu[:probe_n, None]
    )
    max_err = float(
        np.max(np.abs(reconstructed - y_phys[:probe_n]))
    )
    if max_err > 2e-3:
        raise RuntimeError(
            f"{split}: target reconstruction mismatch={max_err}"
        )

    json_dump(
        out / "manifest.json",
        {
            "state": "complete",
            "protocol": "revision-v2",
            "split": split,
            "n_samples": n,
            "x_shape": [n, L],
            "y_shape": [n, H],
            "variables": list(dataset.variables),
            "cities": list(dataset.cities),
            "o3_idx": int(dataset.o3_idx),
            "sample_reconstruction_max_abs_error": max_err,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    return out


def replace_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    relative = os.path.relpath(target.resolve(), link.parent.resolve())
    os.symlink(relative, link)


def make_native_view(root: Path, split: str, canonical: Path) -> Path:
    out = native_input_dir(root, split)
    out.mkdir(parents=True, exist_ok=True)
    mapping = {
        "x.npy": "x.npy",
        "true.npy": "y_phys.npy",
        "y_phys.npy": "y_phys.npy",
        "y_std.npy": "y_std.npy",
        "mu.npy": "mu.npy",
        "std.npy": "std.npy",
        "city_id.npy": "city_id.npy",
        "var_idx.npy": "var_idx.npy",
    }
    for dest, source in mapping.items():
        replace_symlink(canonical / source, out / dest)

    manifest = json.loads(
        (canonical / "manifest.json").read_text(encoding="utf-8")
    )
    manifest.update(
        {
            "state": "complete",
            "interface": "native frozen-TSFM input cache",
            "canonical_source": str(canonical),
            "true_alias": "y_phys.npy",
        }
    )
    json_dump(out / "manifest.json", manifest)
    return out


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    inventory = {
        "state": "running",
        "protocol": "revision-v2",
        "splits": {},
    }
    for split in SPLITS:
        canonical = export_split(
            root,
            split,
            args.batch_size,
            args.num_workers,
            args.overwrite,
        )
        view = make_native_view(root, split, canonical)
        n = int(np.load(view / "true.npy", mmap_mode="r").shape[0])
        inventory["splits"][split] = {
            "canonical": str(canonical),
            "native_input": str(view),
            "n_samples": n,
        }

    inventory.update(
        {
            "state": "complete",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    json_dump(
        result_root(root)
        / "00_native_cache/native_input_inventory.json",
        inventory,
    )
    print(json.dumps(inventory, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
