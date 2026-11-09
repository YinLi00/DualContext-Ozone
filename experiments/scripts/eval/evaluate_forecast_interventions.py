#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inference-time interventions on the trained P4 pretrained-Timer model.

For each P4 seed, this script first computes and caches the correctly aligned
standardized Timer base forecast over the revision-v2 test set. It then evaluates:

I0 correct_timer
I1 zero
I2 sample_shuffle
I3 persistence
I4 horizon_shuffle

The adapter weights are never retrained. I2 uses a single global sample
permutation; I4 uses one fixed 48-horizon permutation per seed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

L, H = 96, 48
INTERVENTIONS = (
    "i0_correct_timer",
    "i1_zero",
    "i2_sample_shuffle",
    "i3_persistence",
    "i4_horizon_shuffle",
)
EXPECTED_SHAPE = (106_200, H)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--interventions",
        nargs="*",
        default=list(INTERVENTIONS),
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    p.add_argument("--require_cuda", action="store_true")
    p.add_argument("--overwrite_base_cache", action="store_true")
    return p.parse_args()


def result_root(root: Path) -> Path:
    return root / "experiments/results/13_tsfm_necessity_v2"


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
        return torch.device("cpu")
    if args.device == "cuda":
        if not has_cuda:
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if has_cuda else "cpu")


def make_loader(dataset, batch_size: int, workers: int,
                cuda: bool) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": cuda,
        "drop_last": False,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def canonical_true(root: Path) -> Path:
    archive = (
        root / "experiments/results/"
        "03_zeroshot_baselines_v2_archive/Timer/true.npy"
    )
    return archive if archive.is_file() else root / "experiments/data_v2/indices/test_targets.npy"


def load_p4(root: Path, seed: int, device: torch.device):
    p4_script = (
        root / "experiments/scripts/train/"
        "train_adapter_ablation.py"
    )
    p4_module = import_file(
        p4_script, f"_p4_intervention_module_{seed}"
    )
    source_path = (
        root / "experiments/scripts/train/"
        "train_attribution_reference.py"
    )
    source = import_file(
        source_path, f"_p4_source_module_{seed}"
    )
    index_root = root / "experiments/data_v2/indices"
    stats_path = (
        root / "experiments/data_v2/"
        "city_var_stats_train_only.json"
    )
    cipda, cora = source.load_project_modules(
        root, index_root, stats_path
    )
    Dataset = cora.Stage1O3CoRAContextDataset
    test_ds = Dataset("test", 0, seed + 2, "both")

    checkpoint_path = (
        root / "experiments/results/"
        "10_adapter_time_ablation_v2/"
        "01_progressive_adapter/p4_io_time_context/"
        f"seed{seed}/best.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    try:
        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = dict(ckpt.get("args", {}))

    timer = AutoModelForCausalLM.from_pretrained(
        cipda.MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    model = p4_module.AdapterTimeAblationForecaster(
        source_module=source,
        timer=timer,
        variant="p4_io_time_context",
        n_vars=len(test_ds.variables),
        n_cities=len(test_ds.cities),
        InputAdapter=cora.InputAdapter,
        ContextEncoder=cora.CoRAContextEncoder,
        time_feature_dim=cora.time_feature_dim,
        input_hidden=int(cfg.get("input_hidden", 64)),
        output_hidden=int(cfg.get("output_hidden", 128)),
        context_hidden=int(cfg.get("context_hidden", 128)),
        emb_dim=int(cfg.get("emb_dim", 16)),
        city_emb_dim=int(cfg.get("city_emb_dim", 16)),
        dropout=float(cfg.get("dropout", 0.05)),
        o3_idx=test_ds.o3_idx,
        corr_temperature=float(
            cfg.get("corr_temperature", 2.0)
        ),
    ).to(device)
    missing, unexpected = model.load_state_dict(
        ckpt["adapter_state"], strict=False
    )
    missing_non_timer = [
        name for name in missing if not name.startswith("timer.")
    ]
    if unexpected or missing_non_timer:
        raise RuntimeError(
            f"P4 checkpoint mismatch: missing={missing_non_timer}, "
            f"unexpected={unexpected}"
        )
    model.eval()
    return model, test_ds, list(test_ds.cities), ckpt


def base_cache_paths(root: Path, seed: int) -> tuple[Path, Path]:
    directory = (
        result_root(root)
        / f"00_backbone_cache/pretrained_timer/seed{seed}"
    )
    return directory / "base_std.npy", directory / "manifest.json"


@torch.no_grad()
def build_base_cache(
    root: Path,
    seed: int,
    model,
    loader: DataLoader,
    device: torch.device,
    overwrite: bool,
) -> Path:
    base_path, manifest_path = base_cache_paths(root, seed)
    if base_path.exists() and manifest_path.exists() and not overwrite:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            data.get("state") == "complete"
            and data.get("shape") == list(EXPECTED_SHAPE)
        ):
            return base_path

    base_path.parent.mkdir(parents=True, exist_ok=True)
    out = open_memmap(
        base_path,
        mode="w+",
        dtype=np.float32,
        shape=EXPECTED_SHAPE,
    )
    offset = 0
    for step, batch in enumerate(loader, 1):
        x = batch["x"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        city = batch["city_id"].to(device, non_blocking=True)
        x_adapt = model.input_adapter(x, city)
        base = model.timer(x_adapt).logits[:, :H].float()
        b = base.shape[0]
        out[offset:offset + b] = (
            base.detach().cpu().numpy().astype(np.float32)
        )
        offset += b
        if step == 1 or step % 500 == 0 or offset == EXPECTED_SHAPE[0]:
            print(
                f"[Timer base seed={seed}] "
                f"{offset:,}/{EXPECTED_SHAPE[0]:,}",
                flush=True,
            )
    out.flush()
    if offset != EXPECTED_SHAPE[0]:
        raise RuntimeError(
            f"Base cache wrote {offset}, expected {EXPECTED_SHAPE[0]}"
        )
    manifest_path.write_text(
        json.dumps({
            "state": "complete",
            "seed": seed,
            "shape": list(EXPECTED_SHAPE),
            "dtype": "float32",
            "nonfinite": int(
                (~np.isfinite(np.load(base_path, mmap_mode="r"))).sum()
            ),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return base_path


class Metric:
    def __init__(self):
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, pred, true):
        diff = (
            np.asarray(pred, dtype=np.float64)
            - np.asarray(true, dtype=np.float64)
        )
        self.abs_sum += float(np.abs(diff).sum())
        self.sq_sum += float(np.square(diff).sum())
        self.count += int(diff.size)

    def result(self):
        mse = self.sq_sum / self.count
        return {
            "MAE": self.abs_sum / self.count,
            "MSE": mse,
            "RMSE": math.sqrt(mse),
            "count": self.count,
        }


@torch.no_grad()
def evaluate_intervention(
    root: Path,
    seed: int,
    intervention: str,
    model,
    loader: DataLoader,
    base_path: Path,
    cities: list[str],
    device: torch.device,
) -> None:
    out_dir = (
        result_root(root)
        / f"02_inference_intervention/{intervention}/seed{seed}"
    )
    status_path = out_dir / "run_status.json"
    if status_path.exists():
        try:
            if json.loads(
                status_path.read_text(encoding="utf-8")
            ).get("state") == "complete":
                print(
                    f"[INTERVENTION SKIP] {intervention} seed={seed}",
                    flush=True,
                )
                return
        except Exception:
            pass
    out_dir.mkdir(parents=True, exist_ok=True)

    base_all = np.load(base_path, mmap_mode="r")
    rng = np.random.default_rng(seed + 91_337)
    sample_perm = rng.permutation(EXPECTED_SHAPE[0])
    horizon_perm = rng.permutation(H)

    pred_out = open_memmap(
        out_dir / "pred.npy",
        mode="w+",
        dtype=np.float32,
        shape=EXPECTED_SHAPE,
    )
    base_out = open_memmap(
        out_dir / "intervened_base_pred.npy",
        mode="w+",
        dtype=np.float32,
        shape=EXPECTED_SHAPE,
    )
    true_out = open_memmap(
        out_dir / "true.tmp.npy",
        mode="w+",
        dtype=np.float32,
        shape=EXPECTED_SHAPE,
    )
    metric = Metric()
    base_metric = Metric()
    by_city: dict[int, Metric] = defaultdict(Metric)
    h_abs = np.zeros(H, dtype=np.float64)
    h_sq = np.zeros(H, dtype=np.float64)
    h_n = np.zeros(H, dtype=np.int64)
    correction_abs = 0.0
    correction_sq = 0.0
    base_sq = 0.0

    offset = 0
    for batch in loader:
        x = batch["x"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        x_multi = batch["x_multi"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        time_feat = batch["time_feat"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        var_idx = batch["var_idx"].to(device, non_blocking=True)
        city_id = batch["city_id"].to(device, non_blocking=True)
        y_phys = to_numpy(batch["y_phys"]).astype(
            np.float32, copy=False
        )
        mu = to_numpy(batch["mu"]).astype(
            np.float32, copy=False
        )[:, None]
        std = to_numpy(batch["std"]).astype(
            np.float32, copy=False
        )[:, None]
        b = x.shape[0]
        indices = np.arange(offset, offset + b)

        x_adapt = model.input_adapter(x, city_id)
        if intervention == "i0_correct_timer":
            base_np = np.asarray(base_all[indices], dtype=np.float32)
        elif intervention == "i1_zero":
            base_np = np.zeros((b, H), dtype=np.float32)
        elif intervention == "i2_sample_shuffle":
            base_np = np.asarray(
                base_all[sample_perm[indices]], dtype=np.float32
            )
        elif intervention == "i3_persistence":
            # Intervention prior must be classical persistence from the raw
            # observed history, not a trainable InputAdapter output.
            base_np = (
                x[:, -1:].expand(-1, H)
                .detach().cpu().numpy().astype(np.float32)
            )
        elif intervention == "i4_horizon_shuffle":
            base_np = np.asarray(
                base_all[indices][:, horizon_perm], dtype=np.float32
            )
        else:
            raise ValueError(intervention)

        base_std = torch.from_numpy(base_np).to(
            device, dtype=torch.float32
        )
        calibrated = (
            model.gamma(var_idx) * base_std
            + model.beta(var_idx)
        )
        context = model.context_encoder(x_multi)
        features = torch.cat([
            calibrated,
            x,
            x_adapt,
            time_feat,
            context,
            model.var_emb(var_idx),
            model.city_emb(city_id),
        ], dim=-1)
        pred_std = (
            calibrated
            + model.output_gate * model.output_net(features)
        )
        pred = (
            pred_std.detach().cpu().numpy().astype(np.float32)
            * std + mu
        )
        base_phys = base_np * std + mu

        pred_out[offset:offset + b] = pred
        base_out[offset:offset + b] = base_phys
        true_out[offset:offset + b] = y_phys
        metric.update(pred, y_phys)
        base_metric.update(base_phys, y_phys)

        city_np = to_numpy(batch["city_id"]).astype(np.int64)
        for cid in np.unique(city_np):
            mask = city_np == cid
            by_city[int(cid)].update(
                pred[mask], y_phys[mask]
            )
        diff = (
            pred.astype(np.float64)
            - y_phys.astype(np.float64)
        )
        h_abs += np.abs(diff).sum(axis=0)
        h_sq += np.square(diff).sum(axis=0)
        h_n += diff.shape[0]
        correction = (
            pred.astype(np.float64)
            - base_phys.astype(np.float64)
        )
        correction_abs += float(np.abs(correction).sum())
        correction_sq += float(np.square(correction).sum())
        base_sq += float(
            np.square(base_phys.astype(np.float64)).sum()
        )
        offset += b

    pred_out.flush()
    base_out.flush()
    true_out.flush()
    if offset != EXPECTED_SHAPE[0]:
        raise RuntimeError(
            f"Intervention wrote {offset}, expected {EXPECTED_SHAPE[0]}"
        )

    temp_true = out_dir / "true.tmp.npy"
    true = np.load(temp_true, mmap_mode="r")
    reference = np.load(canonical_true(root), mmap_mode="r")
    if not np.array_equal(
        np.asarray(true), np.asarray(reference), equal_nan=True
    ):
        raise RuntimeError("Intervention true array differs from canonical")
    temp_true.unlink()
    try:
        os.link(canonical_true(root).resolve(), out_dir / "true.npy")
        true_storage = "hardlink_to_canonical"
    except OSError:
        relative = os.path.relpath(
            canonical_true(root).resolve(), out_dir.resolve()
        )
        os.symlink(relative, out_dir / "true.npy")
        true_storage = "symlink_to_canonical"

    final = metric.result()
    base_result = base_metric.result()
    count = max(final["count"], 1)
    row = {
        "intervention": intervention,
        "seed": seed,
        **final,
        "base_MAE": base_result["MAE"],
        "base_MSE": base_result["MSE"],
        "base_RMSE": base_result["RMSE"],
        "mean_abs_correction": correction_abs / count,
        "correction_RMSE": math.sqrt(correction_sq / count),
        "correction_to_base_l2_ratio": (
            math.sqrt(correction_sq)
            / (math.sqrt(base_sq) + 1e-12)
        ),
    }
    pd.DataFrame([row]).to_csv(
        out_dir / "metrics_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    city_rows = []
    for cid, item in sorted(by_city.items()):
        city_rows.append({
            "city_id": cid,
            "city": (
                cities[cid]
                if 0 <= cid < len(cities) else str(cid)
            ),
            **item.result(),
        })
    pd.DataFrame(city_rows).to_csv(
        out_dir / "metrics_by_city.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_rows = []
    for h in range(H):
        n = int(h_n[h])
        mse = h_sq[h] / n
        horizon_rows.append({
            "horizon": h + 1,
            "MAE": h_abs[h] / n,
            "MSE": mse,
            "RMSE": math.sqrt(mse),
            "count": n,
        })
    pd.DataFrame(horizon_rows).to_csv(
        out_dir / "metrics_by_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "state": "complete",
        "protocol": "revision-v2",
        "intervention": intervention,
        "seed": seed,
        "shape": list(EXPECTED_SHAPE),
        "true_storage": true_storage,
        "sample_permutation_seed": (
            seed + 91_337
            if intervention == "i2_sample_shuffle" else None
        ),
        "horizon_permutation": (
            horizon_perm.tolist()
            if intervention == "i4_horizon_shuffle" else None
        ),
        "nonfinite_pred": int(
            (~np.isfinite(
                np.load(out_dir / "pred.npy", mmap_mode="r")
            )).sum()
        ),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "array_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    status_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(pd.DataFrame([row]).to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    unknown = sorted(set(args.interventions) - set(INTERVENTIONS))
    if unknown:
        raise ValueError(f"Unknown interventions: {unknown}")
    device = resolve_device(args)
    model, test_ds, cities, _ = load_p4(
        root, args.seed, device
    )
    loader = make_loader(
        test_ds,
        args.batch_size,
        args.num_workers,
        device.type == "cuda",
    )
    base_path = build_base_cache(
        root,
        args.seed,
        model,
        loader,
        device,
        args.overwrite_base_cache,
    )
    for intervention in args.interventions:
        evaluate_intervention(
            root,
            args.seed,
            intervention,
            model,
            loader,
            base_path,
            cities,
            device,
        )


if __name__ == "__main__":
    main()
