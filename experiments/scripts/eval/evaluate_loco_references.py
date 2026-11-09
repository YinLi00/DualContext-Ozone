#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute no-training references for one held-out city.

References use the exact same fold, source-21 pooled scaler and canonical test
order as LOCO Full:

  persistence          : last physical O3 repeated for 48 h
  timer_source_scaled  : frozen Timer on source-normalized O3 history

The second reference isolates what the pretrained Timer alone can do under the
strict LOCO normalization protocol; it does not use InputAdapter, target-city
identity, target-city training data or target-city statistics.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    p.add_argument("--heldout_city", required=True)
    p.add_argument(
        "--references",
        nargs="*",
        default=["persistence", "timer_source_scaled"],
        choices=["persistence", "timer_source_scaled"],
    )
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_test_samples", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--require_cuda", action="store_true")
    p.add_argument("--cpu_threads", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def safe_name(value: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in "._-" else "_"
        for ch in value
    )


def result_root(root: Path) -> Path:
    return root / "experiments/results/14_loco_22cities_v2"


def fold_root(root: Path, city: str) -> Path:
    return result_root(root) / "00_fold_indices" / safe_name(city)


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_device(args: argparse.Namespace) -> torch.device:
    has_cuda = torch.cuda.is_available()
    if args.require_cuda and not has_cuda:
        raise RuntimeError("--require_cuda set but CUDA unavailable")
    if args.device == "cpu":
        torch.set_num_threads(max(1, int(args.cpu_threads)))
        return torch.device("cpu")
    if args.device == "cuda":
        if not has_cuda:
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if has_cuda else "cpu")


def metric(pred: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    diff = (
        pred.astype(np.float64)
        - true.astype(np.float64)
    )
    mse = float(np.square(diff).mean())
    return {
        "MAE": float(np.abs(diff).mean()),
        "MSE": mse,
        "RMSE": math.sqrt(mse),
        "count": int(diff.size),
    }


def complete(directory: Path, n: int) -> bool:
    status = directory / "run_status.json"
    if not status.exists():
        return False
    try:
        data = json.loads(status.read_text(encoding="utf-8"))
        if data.get("state") != "complete":
            return False
        pred = np.load(directory / "pred.npy", mmap_mode="r")
        true = np.load(directory / "true.npy", mmap_mode="r")
        return pred.shape == true.shape == (n, 48)
    except Exception:
        return False


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    trainer_module = import_file(
        root / "experiments/scripts/train/"
        "train_dualcontext_loco.py",
        f"_loco_reference_dataset_{safe_name(args.heldout_city)}",
    )
    source = import_file(
        root / "experiments/scripts/train/"
        "train_attribution_reference.py",
        f"_loco_reference_source_{safe_name(args.heldout_city)}",
    )
    index_root = root / "experiments/data_v2/indices"
    stats_path = (
        root / "experiments/data_v2/"
        "city_var_stats_train_only.json"
    )
    cipda, cora = source.load_project_modules(
        root, index_root, stats_path
    )
    dataset = trainer_module.LOCOStage1Dataset(
        root=root,
        heldout_city=args.heldout_city,
        split="test",
        build_time_feature_matrix=cora.build_time_feature_matrix,
        max_samples=args.max_test_samples,
        seed=2028,
    )
    loader = trainer_module.make_loader(
        dataset,
        args.batch_size,
        False,
        args.num_workers,
        resolve_device(args).type == "cuda",
    )
    device = resolve_device(args)
    timer = None
    if "timer_source_scaled" in args.references:
        print(f"Loading frozen Timer from {cipda.MODEL_PATH}", flush=True)
        timer = AutoModelForCausalLM.from_pretrained(
            cipda.MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32,
        ).to(device)
        timer.eval()
        if hasattr(timer.config, "use_cache"):
            timer.config.use_cache = False

    pred_lists: dict[str, list[np.ndarray]] = {
        name: [] for name in args.references
    }
    true_list: list[np.ndarray] = []
    position_list: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        mu = (
            trainer_module.to_numpy(batch["mu"])
            .astype(np.float32)[:, None]
        )
        std = (
            trainer_module.to_numpy(batch["std"])
            .astype(np.float32)[:, None]
        )
        true = (
            trainer_module.to_numpy(batch["y_phys"])
            .astype(np.float32)
        )
        if "persistence" in args.references:
            persistence_std = (
                x[:, -1:].expand(-1, 48)
                .detach().cpu().numpy().astype(np.float32)
            )
            pred_lists["persistence"].append(
                persistence_std * std + mu
            )
        if "timer_source_scaled" in args.references:
            timer_std = (
                timer(x).logits[:, :48].float()
                .detach().cpu().numpy().astype(np.float32)
            )
            pred_lists["timer_source_scaled"].append(
                timer_std * std + mu
            )
        true_list.append(true)
        position_list.append(
            trainer_module.to_numpy(
                batch["canonical_position"]
            ).astype(np.int64)
        )

    true = np.concatenate(true_list, axis=0)
    positions = np.concatenate(position_list, axis=0)
    canonical = np.load(
        fold_root(root, args.heldout_city)
        / "canonical_test_true.npy",
        mmap_mode="r",
    )
    canonical_positions = np.load(
        fold_root(root, args.heldout_city)
        / "canonical_test_positions.npy",
        mmap_mode="r",
    )
    full = args.max_test_samples == 0
    if full and (
        not np.array_equal(np.asarray(canonical), true)
        or not np.array_equal(
            np.asarray(canonical_positions), positions
        )
    ):
        raise RuntimeError("Reference test order differs from fold canonical")

    for name in args.references:
        out = (
            result_root(root)
            / "03_reference" / name
            / safe_name(args.heldout_city)
        )
        if complete(out, len(dataset)) and not args.overwrite:
            print(
                f"[REFERENCE SKIP] {args.heldout_city} {name}",
                flush=True,
            )
            continue
        out.mkdir(parents=True, exist_ok=True)
        pred = np.concatenate(pred_lists[name], axis=0)
        np.save(out / "pred.npy", pred, allow_pickle=False)
        np.save(out / "true.npy", true, allow_pickle=False)
        np.save(
            out / "canonical_test_positions.npy",
            positions,
            allow_pickle=False,
        )

        overall = {
            "reference": name,
            "heldout_city": args.heldout_city,
            "protocol": "revision-v2-LOCO",
            "normalization": "source-21 pooled train-only",
            "target_city_stats_used": False,
            "n_test_samples": len(dataset),
            **metric(pred, true),
        }
        pd.DataFrame([overall]).to_csv(
            out / "metrics_overall.csv",
            index=False,
            encoding="utf-8-sig",
        )
        horizon_rows = []
        for horizon in range(48):
            horizon_rows.append({
                "horizon": horizon + 1,
                **metric(
                    pred[:, horizon:horizon + 1],
                    true[:, horizon:horizon + 1],
                ),
            })
        pd.DataFrame(horizon_rows).to_csv(
            out / "metrics_by_horizon.csv",
            index=False,
            encoding="utf-8-sig",
        )

        manifest = {
            "state": "complete",
            "reference": name,
            "heldout_city": args.heldout_city,
            "shape": list(pred.shape),
            "nonfinite_pred": int((~np.isfinite(pred)).sum()),
            "nonfinite_true": int((~np.isfinite(true)).sum()),
            "formal_full_fold": full,
            "true_exactly_matches_fold_canonical": (
                bool(full) if full else False
            ),
            "positions_exactly_match_fold_canonical": (
                bool(full) if full else False
            ),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (out / "array_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out / "run_status.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(pd.DataFrame([overall]).to_string(index=False))


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
