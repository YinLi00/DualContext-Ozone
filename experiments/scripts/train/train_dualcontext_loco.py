#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train one city-agnostic LOCO Full model under revision-v2.

For a held-out city c:
  train/val: canonical revision-v2 samples from the other 21 cities
  test     : canonical revision-v2 samples from c only

Leakage controls:
  * target city is absent from train and validation indices;
  * all samples receive shared city_id=0;
  * all variables use a source-21 pooled train-only scaler;
  * held-out-city statistics are never read by this Dataset.

Architecture:
  frozen pretrained Timer
  + InputAdapter
  + full CoRAContextEncoder
  + Hour/DOY target features
  + variable embedding
  + one shared city embedding
  + horizon-wise gamma/beta
  + residual Output Adapter
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM

L, H = 96, 48


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    p.add_argument("--heldout_city", required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--mse_weight", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--input_hidden", type=int, default=64)
    p.add_argument("--output_hidden", type=int, default=128)
    p.add_argument("--context_hidden", type=int, default=128)
    p.add_argument("--emb_dim", type=int, default=16)
    p.add_argument("--city_emb_dim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--corr_temperature", type=float, default=2.0)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--max_test_samples", type=int, default=0)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--require_cuda", action="store_true")
    p.add_argument("--save_arrays", type=int, choices=[0, 1], default=1)
    p.add_argument("--keep_checkpoint", type=int, choices=[0, 1], default=1)
    p.add_argument("--cpu_threads", type=int, default=4)
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


def import_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def seed_everything(seed: int, cuda: bool) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = cuda


def resolve_device(args: argparse.Namespace) -> torch.device:
    has_cuda = torch.cuda.is_available()
    if args.require_cuda and not has_cuda:
        raise RuntimeError("--require_cuda set but CUDA is unavailable")
    if args.device == "cpu":
        torch.set_num_threads(max(1, int(args.cpu_threads)))
        return torch.device("cpu")
    if args.device == "cuda":
        if not has_cuda:
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if has_cuda else "cpu")


class LOCOStage1Dataset(Dataset):
    """Raw-array Dataset normalized only with source-21 train moments."""

    def __init__(
        self,
        *,
        root: Path,
        heldout_city: str,
        split: str,
        build_time_feature_matrix,
        max_samples: int = 0,
        seed: int = 2026,
    ) -> None:
        self.root = root
        self.heldout_city = heldout_city
        self.split = split
        self.build_time_feature_matrix = build_time_feature_matrix

        canonical_meta = json.loads(
            (
                root / "experiments/data_v2/"
                "indices/meta.json"
            ).read_text(encoding="utf-8")
        )
        self.meta = canonical_meta
        self.variables = list(canonical_meta["variables"])
        self.original_cities = list(canonical_meta["cities"])
        self.cities = ["shared_source_city"]
        self.array_paths = list(canonical_meta["array_paths"])
        self.stations = list(canonical_meta["stations"])
        self.o3_idx = int(canonical_meta["o3_idx"])
        self.heldout_city_id = self.original_cities.index(heldout_city)

        fold = fold_root(root, heldout_city)
        self.fold_manifest = json.loads(
            (fold / "fold_manifest.json").read_text(encoding="utf-8")
        )
        scaler = json.loads(
            (fold / "source_train_stats.json").read_text(encoding="utf-8")
        )
        if scaler.get("heldout_city_used_in_scaler") is not False:
            raise RuntimeError("Held-out city appears in the fold scaler")
        if heldout_city in scaler["source_cities"]:
            raise RuntimeError("Held-out city appears in source city list")

        mu = []
        std = []
        for variable in self.variables:
            stat = scaler["pooled_variable_stats"][variable]
            mu.append(np.float32(stat["mean"]))
            value = np.float32(stat["std"])
            if not np.isfinite(value) or value <= 0:
                value = np.float32(1.0)
            std.append(value)
        self.source_mu = np.asarray(mu, dtype=np.float32)
        self.source_std = np.asarray(std, dtype=np.float32)
        self.o3_mu = np.float32(self.source_mu[self.o3_idx])
        self.o3_std = np.float32(self.source_std[self.o3_idx])

        idx = np.load(fold / f"{split}.npz", allow_pickle=False)
        array_id = idx["array_id"].astype(np.int32)
        original_city_id = idx["city_id"].astype(np.int16)
        var_idx = idx["var_idx"].astype(np.int8)
        start_idx = idx["start_idx"].astype(np.int32)
        canonical_position = idx["canonical_position"].astype(np.int64)

        if split == "test":
            if not np.all(original_city_id == self.heldout_city_id):
                raise RuntimeError("Test fold contains a source city")
        else:
            if np.any(original_city_id == self.heldout_city_id):
                raise RuntimeError(
                    f"{split} fold contains the held-out city"
                )

        if max_samples and 0 < max_samples < len(array_id):
            rng = np.random.default_rng(seed)
            selection = rng.choice(
                len(array_id), size=max_samples, replace=False
            )
            # Preserve canonical order for validation/test diagnostics.
            if split != "train":
                selection.sort()
            array_id = array_id[selection]
            original_city_id = original_city_id[selection]
            var_idx = var_idx[selection]
            start_idx = start_idx[selection]
            canonical_position = canonical_position[selection]

        self.array_id = array_id
        self.original_city_id = original_city_id
        self.var_idx = var_idx
        self.start_idx = start_idx
        self.canonical_position = canonical_position
        self.cache: dict[int, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.array_id)

    def _load(self, array_id: int) -> dict[str, np.ndarray]:
        aid = int(array_id)
        if aid not in self.cache:
            path = self.root / self.array_paths[aid]
            with np.load(path, allow_pickle=True) as data:
                values = data["values"].astype(np.float32)
                time_ns = data["time_ns"].astype(np.int64)
            self.cache[aid] = {
                "values": values,
                "time_feat": self.build_time_feature_matrix(
                    time_ns, "both"
                ).astype(np.float32),
            }
        return self.cache[aid]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        aid = int(self.array_id[index])
        start = int(self.start_idx[index])
        var_idx = int(self.var_idx[index])
        original_city_id = int(self.original_city_id[index])
        sample = self._load(aid)
        values = sample["values"]

        x_multi_phys = values[start:start + L, :]
        x_o3_phys = x_multi_phys[:, var_idx]
        y_o3_phys = values[
            start + L:start + L + H, var_idx
        ]
        x_multi = (
            x_multi_phys - self.source_mu[None, :]
        ) / self.source_std[None, :]
        x = (x_o3_phys - self.o3_mu) / self.o3_std
        y_std = (y_o3_phys - self.o3_mu) / self.o3_std
        time_feat = sample["time_feat"][
            start + L:start + L + H
        ].reshape(-1)

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "x_multi": torch.tensor(
                x_multi, dtype=torch.float32
            ),
            "time_feat": torch.tensor(
                time_feat, dtype=torch.float32
            ),
            "y_std": torch.tensor(y_std, dtype=torch.float32),
            "y_phys": torch.tensor(
                y_o3_phys, dtype=torch.float32
            ),
            "mu": torch.tensor(self.o3_mu, dtype=torch.float32),
            "std": torch.tensor(self.o3_std, dtype=torch.float32),
            # Explicitly anonymous/shared city identity.
            "city_id": torch.tensor(0, dtype=torch.long),
            "orig_city_id": torch.tensor(
                original_city_id, dtype=torch.long
            ),
            "var_idx": torch.tensor(var_idx, dtype=torch.long),
            "array_id": torch.tensor(aid, dtype=torch.long),
            "start_idx": torch.tensor(start, dtype=torch.long),
            "canonical_position": torch.tensor(
                int(self.canonical_position[index]),
                dtype=torch.long,
            ),
        }


class LOCOFullForecaster(nn.Module):
    def __init__(
        self,
        *,
        timer: nn.Module,
        n_vars: int,
        InputAdapter,
        ContextEncoder,
        time_feature_dim,
        input_hidden: int,
        output_hidden: int,
        context_hidden: int,
        emb_dim: int,
        city_emb_dim: int,
        dropout: float,
        o3_idx: int,
        corr_temperature: float,
        init_gate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.timer = timer
        for parameter in self.timer.parameters():
            parameter.requires_grad_(False)
        self.timer.eval()
        if hasattr(self.timer.config, "use_cache"):
            self.timer.config.use_cache = False

        self.input_adapter = InputAdapter(
            seq_len=L,
            n_cities=1,
            city_emb_dim=city_emb_dim,
            hidden=input_hidden,
            dropout=dropout,
            init_gate=init_gate,
        )
        self.context_encoder = ContextEncoder(
            seq_len=L,
            n_vars=n_vars,
            hidden=context_hidden,
            dropout=dropout,
            o3_idx=o3_idx,
            corr_temperature=corr_temperature,
        )
        self.gamma = nn.Embedding(n_vars, H)
        self.beta = nn.Embedding(n_vars, H)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        self.var_emb = nn.Embedding(n_vars, emb_dim)
        self.city_emb = nn.Embedding(1, city_emb_dim)

        in_dim = (
            H + L + L + H * time_feature_dim("both")
            + context_hidden + emb_dim + city_emb_dim
        )
        self.output_net = nn.Sequential(
            nn.Linear(in_dim, output_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_hidden, H),
        )
        self.output_gate = nn.Parameter(
            torch.tensor(float(init_gate))
        )
        nn.init.zeros_(self.output_net[-1].weight)
        nn.init.zeros_(self.output_net[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.timer.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        x_multi: torch.Tensor,
        time_feat: torch.Tensor,
        var_idx: torch.Tensor,
        city_id: torch.Tensor,
        *,
        return_base: bool = False,
    ):
        if city_id.numel() and torch.any(city_id != 0):
            raise RuntimeError("LOCO city_id must be shared ID zero")
        x_adapt = self.input_adapter(x, city_id)
        base = self.timer(x_adapt).logits[:, :H].float()
        calibrated = (
            self.gamma(var_idx) * base
            + self.beta(var_idx)
        )
        context = self.context_encoder(x_multi)
        features = torch.cat(
            [
                calibrated,
                x,
                x_adapt,
                time_feat,
                context,
                self.var_emb(var_idx),
                self.city_emb(city_id),
            ],
            dim=-1,
        )
        final = (
            calibrated
            + self.output_gate * self.output_net(features)
        )
        return (final, base) if return_base else final


class SumMetric:
    def __init__(self) -> None:
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        diff = (
            np.asarray(pred, dtype=np.float64)
            - np.asarray(true, dtype=np.float64)
        )
        self.abs_sum += float(np.abs(diff).sum(dtype=np.float64))
        self.sq_sum += float(np.square(diff).sum(dtype=np.float64))
        self.count += int(diff.size)

    def result(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "MAE": math.nan,
                "MSE": math.nan,
                "RMSE": math.nan,
                "count": 0,
            }
        mse = self.sq_sum / self.count
        return {
            "MAE": self.abs_sum / self.count,
            "MSE": mse,
            "RMSE": math.sqrt(mse),
            "count": self.count,
        }


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    cuda: bool,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
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


def run_epoch(
    model: LOCOFullForecaster,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer,
    mse_weight: float,
    grad_clip: float,
    log_every: int,
    collect: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    final_metric = SumMetric()
    base_metric = SumMetric()
    by_station: dict[int, SumMetric] = defaultdict(SumMetric)
    h_abs = np.zeros(H, dtype=np.float64)
    h_sq = np.zeros(H, dtype=np.float64)
    base_h_abs = np.zeros(H, dtype=np.float64)
    base_h_sq = np.zeros(H, dtype=np.float64)
    h_n = np.zeros(H, dtype=np.int64)
    pred_batches: list[np.ndarray] = []
    base_batches: list[np.ndarray] = []
    true_batches: list[np.ndarray] = []
    array_batches: list[np.ndarray] = []
    position_batches: list[np.ndarray] = []
    loss_sum = 0.0
    batch_count = 0
    correction_abs_sum = 0.0
    correction_sq_sum = 0.0
    base_sq_sum = 0.0

    for step, batch in enumerate(loader, 1):
        x = batch["x"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        x_multi = batch["x_multi"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        time_feat = batch["time_feat"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        y_std = batch["y_std"].to(
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

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            pred_std, base_std = model(
                x,
                x_multi,
                time_feat,
                var_idx,
                city_id,
                return_base=True,
            )
            loss = (
                F.l1_loss(pred_std, y_std)
                + mse_weight * F.mse_loss(pred_std, y_std)
            )
            if training:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [
                            p for p in model.parameters()
                            if p.requires_grad
                        ],
                        grad_clip,
                    )
                optimizer.step()

        loss_sum += float(loss.detach().cpu())
        batch_count += 1
        pred = (
            pred_std.detach().cpu().numpy().astype(
                np.float32, copy=False
            ) * std + mu
        )
        base = (
            base_std.detach().cpu().numpy().astype(
                np.float32, copy=False
            ) * std + mu
        )
        final_metric.update(pred, y_phys)
        base_metric.update(base, y_phys)

        correction = (
            pred.astype(np.float64)
            - base.astype(np.float64)
        )
        correction_abs_sum += float(np.abs(correction).sum())
        correction_sq_sum += float(np.square(correction).sum())
        base_sq_sum += float(
            np.square(base.astype(np.float64)).sum()
        )

        if collect:
            arrays = to_numpy(batch["array_id"]).astype(np.int64)
            for aid in np.unique(arrays):
                mask = arrays == aid
                by_station[int(aid)].update(
                    pred[mask], y_phys[mask]
                )
            diff = (
                pred.astype(np.float64)
                - y_phys.astype(np.float64)
            )
            base_diff = (
                base.astype(np.float64)
                - y_phys.astype(np.float64)
            )
            h_abs += np.abs(diff).sum(axis=0)
            h_sq += np.square(diff).sum(axis=0)
            base_h_abs += np.abs(base_diff).sum(axis=0)
            base_h_sq += np.square(base_diff).sum(axis=0)
            h_n += diff.shape[0]
            pred_batches.append(pred)
            base_batches.append(base)
            true_batches.append(y_phys)
            array_batches.append(arrays)
            position_batches.append(
                to_numpy(batch["canonical_position"]).astype(
                    np.int64
                )
            )

        if training and (step == 1 or step % log_every == 0):
            metric = final_metric.result()
            print(
                f"step={step}/{len(loader)} "
                f"loss={loss_sum/max(batch_count,1):.6f} "
                f"MAE={metric['MAE']:.4f} "
                f"RMSE={metric['RMSE']:.4f}",
                flush=True,
            )

    final_result = final_metric.result()
    base_result = base_metric.result()
    count = max(final_result["count"], 1)
    detail: dict[str, Any] = {
        "base_metrics": base_result,
        "by_station": by_station,
        "h_abs": h_abs,
        "h_sq": h_sq,
        "base_h_abs": base_h_abs,
        "base_h_sq": base_h_sq,
        "h_n": h_n,
        "mean_abs_correction": correction_abs_sum / count,
        "correction_RMSE": math.sqrt(correction_sq_sum / count),
        "correction_to_base_l2_ratio": (
            math.sqrt(correction_sq_sum)
            / (math.sqrt(base_sq_sum) + 1e-12)
        ),
    }
    if collect:
        detail.update({
            "pred": np.concatenate(pred_batches, axis=0),
            "base": np.concatenate(base_batches, axis=0),
            "true": np.concatenate(true_batches, axis=0),
            "array_id": np.concatenate(array_batches, axis=0),
            "canonical_position": np.concatenate(
                position_batches, axis=0
            ),
        })
    return {
        "loss": loss_sum / max(batch_count, 1),
        **final_result,
    }, detail


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("timer.")
    }


def save_outputs(
    out: Path,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    detail: dict[str, Any],
    test_ds: LOCOStage1Dataset,
    *,
    best_epoch: int,
    best_val_rmse: float,
    train_seconds: float,
    test_seconds: float,
    total_params: int,
    trainable_params: int,
    peak_gpu_memory_mb: float,
    device_name: str,
) -> None:
    base = detail["base_metrics"]
    row = {
        **metrics,
        "heldout_city": args.heldout_city,
        "heldout_city_id": test_ds.heldout_city_id,
        "seed": args.seed,
        "protocol": "revision-v2-LOCO",
        "source_city_count": 21,
        "city_identity_mode": "shared_single_id",
        "normalization": "source-21 pooled train-only",
        "base_MAE": base["MAE"],
        "base_MSE": base["MSE"],
        "base_RMSE": base["RMSE"],
        "mean_abs_correction": detail["mean_abs_correction"],
        "correction_RMSE": detail["correction_RMSE"],
        "correction_to_base_l2_ratio":
            detail["correction_to_base_l2_ratio"],
        "best_epoch": best_epoch,
        "best_val_RMSE": best_val_rmse,
        "train_seconds": train_seconds,
        "test_seconds": test_seconds,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "device_name": device_name,
        "n_test_samples": len(test_ds),
    }
    pd.DataFrame([row]).to_csv(
        out / "metrics_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_rows = []
    for h in range(H):
        n = int(detail["h_n"][h])
        mse = detail["h_sq"][h] / n if n else math.nan
        base_mse = (
            detail["base_h_sq"][h] / n if n else math.nan
        )
        horizon_rows.append({
            "horizon": h + 1,
            "MAE": detail["h_abs"][h] / n if n else math.nan,
            "MSE": mse,
            "RMSE": math.sqrt(mse) if n else math.nan,
            "base_MAE": (
                detail["base_h_abs"][h] / n if n else math.nan
            ),
            "base_MSE": base_mse,
            "base_RMSE": (
                math.sqrt(base_mse) if n else math.nan
            ),
            "count": n,
        })
    pd.DataFrame(horizon_rows).to_csv(
        out / "metrics_by_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    station_rows = []
    for aid, metric in sorted(detail["by_station"].items()):
        station = test_ds.stations[aid]
        station_rows.append({
            "array_id": aid,
            "city": station["city"],
            "station_id": station["station_id"],
            **metric.result(),
        })
    pd.DataFrame(station_rows).to_csv(
        out / "metrics_by_station.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if args.save_arrays:
        pred = np.asarray(detail["pred"], dtype=np.float32)
        base_pred = np.asarray(detail["base"], dtype=np.float32)
        true = np.asarray(detail["true"], dtype=np.float32)
        array_id = np.asarray(detail["array_id"], dtype=np.int32)
        positions = np.asarray(
            detail["canonical_position"], dtype=np.int64
        )
        np.save(out / "pred.npy", pred, allow_pickle=False)
        np.save(
            out / "base_pred.npy", base_pred, allow_pickle=False
        )
        np.save(out / "true.npy", true, allow_pickle=False)
        np.save(out / "array_id.npy", array_id, allow_pickle=False)
        np.save(
            out / "canonical_test_positions.npy",
            positions,
            allow_pickle=False,
        )
        np.save(
            out / "shared_city_id.npy",
            np.zeros(len(pred), dtype=np.int16),
            allow_pickle=False,
        )
        np.save(
            out / "original_city_id.npy",
            np.full(
                len(pred),
                test_ds.heldout_city_id,
                dtype=np.int16,
            ),
            allow_pickle=False,
        )

        full_formal = (
            args.max_test_samples == 0
            and pred.shape == (len(test_ds), H)
        )
        canonical_path = (
            fold_root(args.root, args.heldout_city)
            / "canonical_test_true.npy"
        )
        canonical_positions = (
            fold_root(args.root, args.heldout_city)
            / "canonical_test_positions.npy"
        )
        exact_true = False
        exact_positions = False
        if full_formal:
            canonical = np.load(canonical_path, mmap_mode="r")
            expected_pos = np.load(
                canonical_positions, mmap_mode="r"
            )
            exact_true = (
                canonical.shape == true.shape
                and np.array_equal(
                    np.asarray(canonical),
                    true,
                    equal_nan=True,
                )
            )
            exact_positions = np.array_equal(
                np.asarray(expected_pos), positions
            )
            if not exact_true or not exact_positions:
                raise RuntimeError(
                    "Formal LOCO test arrays do not match fold canonical order"
                )

        manifest = {
            "state": "complete",
            "protocol": "revision-v2-LOCO",
            "heldout_city": args.heldout_city,
            "seed": args.seed,
            "pred_shape": list(pred.shape),
            "base_pred_shape": list(base_pred.shape),
            "true_shape": list(true.shape),
            "canonical_positions_shape": list(positions.shape),
            "nonfinite_pred": int((~np.isfinite(pred)).sum()),
            "nonfinite_base_pred":
                int((~np.isfinite(base_pred)).sum()),
            "nonfinite_true": int((~np.isfinite(true)).sum()),
            "formal_full_fold": full_formal,
            "true_exactly_matches_fold_canonical": exact_true,
            "positions_exactly_match_fold_canonical": exact_positions,
            "city_identity_mode": "shared_single_id",
            "target_city_stats_used": False,
        }
        (out / "array_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def run_main(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    args.root = root
    out = (
        args.out_dir.expanduser().resolve()
        if args.out_dir.is_absolute()
        else (root / args.out_dir).resolve()
    )
    out.mkdir(parents=True, exist_ok=True)

    fold = fold_root(root, args.heldout_city)
    if not (fold / "fold_manifest.json").exists():
        raise FileNotFoundError(
            f"LOCO fold is missing: {fold}"
        )

    source_path = (
        root / "experiments/scripts/train/"
        "train_attribution_reference.py"
    )
    source = import_source(
        source_path, f"_loco_source_{safe_name(args.heldout_city)}"
    )
    index_root = root / "experiments/data_v2/indices"
    stats_path = (
        root / "experiments/data_v2/"
        "city_var_stats_train_only.json"
    )
    cipda, cora = source.load_project_modules(
        root, index_root, stats_path
    )

    device = resolve_device(args)
    cuda = device.type == "cuda"
    seed_everything(args.seed, cuda)

    dataset_kwargs = {
        "root": root,
        "heldout_city": args.heldout_city,
        "build_time_feature_matrix":
            cora.build_time_feature_matrix,
    }
    train_ds = LOCOStage1Dataset(
        **dataset_kwargs,
        split="train",
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    val_ds = LOCOStage1Dataset(
        **dataset_kwargs,
        split="val",
        max_samples=args.max_val_samples,
        seed=args.seed + 1,
    )
    test_ds = LOCOStage1Dataset(
        **dataset_kwargs,
        split="test",
        max_samples=args.max_test_samples,
        seed=args.seed + 2,
    )
    train_loader = make_loader(
        train_ds, args.batch_size, True, args.num_workers, cuda
    )
    val_loader = make_loader(
        val_ds, args.batch_size, False, args.num_workers, cuda
    )
    test_loader = make_loader(
        test_ds, args.batch_size, False, args.num_workers, cuda
    )

    print(f"Loading frozen Timer from {cipda.MODEL_PATH}", flush=True)
    timer = AutoModelForCausalLM.from_pretrained(
        cipda.MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    model = LOCOFullForecaster(
        timer=timer,
        n_vars=len(train_ds.variables),
        InputAdapter=cora.InputAdapter,
        ContextEncoder=cora.CoRAContextEncoder,
        time_feature_dim=cora.time_feature_dim,
        input_hidden=args.input_hidden,
        output_hidden=args.output_hidden,
        context_hidden=args.context_hidden,
        emb_dim=args.emb_dim,
        city_emb_dim=args.city_emb_dim,
        dropout=args.dropout,
        o3_idx=train_ds.o3_idx,
        corr_temperature=args.corr_temperature,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    device_name = (
        torch.cuda.get_device_name(device)
        if cuda else f"CPU ({torch.get_num_threads()} threads)"
    )
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)

    fold_manifest = json.loads(
        (fold / "fold_manifest.json").read_text(encoding="utf-8")
    )
    print("=" * 100)
    print("[22-CITY LOCO FULL / REVISION-V2]")
    print(
        f"heldout_city={args.heldout_city} seed={args.seed} "
        f"device={device_name}"
    )
    print(
        f"train={len(train_ds):,} val={len(val_ds):,} "
        f"test={len(test_ds):,}"
    )
    print(
        "city_identity=shared_single_id "
        "normalization=source-21 pooled train-only"
    )
    print(
        f"epochs={args.epochs} patience={args.patience} "
        f"batch={args.batch_size} lr={args.lr}"
    )
    print(
        f"params total={total_params:,} "
        f"trainable={trainable_params:,}"
    )
    print("=" * 100)

    args_dict = vars(args).copy()
    for key, value in list(args_dict.items()):
        if isinstance(value, Path):
            args_dict[key] = str(value)
    args_dict.update({
        "protocol": "revision-v2-LOCO",
        "fold_root": str(fold),
        "source_cities": fold_manifest["source_cities"],
        "source_city_count": 21,
        "target_city_in_train": False,
        "target_city_in_val": False,
        "target_city_stats_used": False,
        "city_identity_mode": "shared_single_id",
        "normalization": "source-21 pooled train-only",
        "total_params": total_params,
        "trainable_params": trainable_params,
        "device_name": device_name,
    })
    (out / "args.json").write_text(
        json.dumps(args_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out / "source_cities.json").write_text(
        json.dumps({
            "heldout_city": args.heldout_city,
            "source_cities": fold_manifest["source_cities"],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    shutil_source = fold / "source_train_stats.json"
    (out / "source_train_stats.json").write_bytes(
        shutil_source.read_bytes()
    )

    status_path = out / "run_status.json"
    status_path.write_text(
        json.dumps({
            "state": "running",
            "heldout_city": args.heldout_city,
            "seed": args.seed,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_rmse = float("inf")
    bad_epochs = 0
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} train", flush=True)
        train_metrics, _ = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            mse_weight=args.mse_weight,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
            collect=False,
        )
        print(f"Epoch {epoch}/{args.epochs} val", flush=True)
        val_metrics, _ = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            mse_weight=args.mse_weight,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
            collect=False,
        )
        history.append({
            "epoch": epoch,
            **{
                f"train_{key}": value
                for key, value in train_metrics.items()
            },
            **{
                f"val_{key}": value
                for key, value in val_metrics.items()
            },
        })
        pd.DataFrame(history).to_csv(
            out / "history.csv",
            index=False,
            encoding="utf-8-sig",
        )

        checkpoint = {
            "epoch": epoch,
            "adapter_state": adapter_state(model),
            "args": args_dict,
            "heldout_city": args.heldout_city,
            "source_cities": fold_manifest["source_cities"],
            "variables": train_ds.variables,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, out / "last.pt")
        if val_metrics["RMSE"] < best_rmse - 1e-12:
            best_rmse = float(val_metrics["RMSE"])
            bad_epochs = 0
            torch.save(checkpoint, out / "best.pt")
            print(
                f"Saved best.pt val_RMSE={best_rmse:.6f}",
                flush=True,
            )
        else:
            bad_epochs += 1
            print(
                f"No improvement: {bad_epochs}/{args.patience}",
                flush=True,
            )
            if (
                args.patience > 0
                and bad_epochs >= args.patience
            ):
                print(
                    f"Early stopping at epoch={epoch}",
                    flush=True,
                )
                break

    if cuda:
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - train_start

    try:
        best = torch.load(
            out / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        best = torch.load(out / "best.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(
        best["adapter_state"], strict=False
    )
    missing_non_timer = [
        name for name in missing if not name.startswith("timer.")
    ]
    if unexpected or missing_non_timer:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing_non_timer}, "
            f"unexpected={unexpected}"
        )

    print("\nFinal held-out-city test", flush=True)
    test_start = time.perf_counter()
    test_metrics, detail = run_epoch(
        model,
        test_loader,
        device,
        optimizer=None,
        mse_weight=args.mse_weight,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        collect=True,
    )
    if cuda:
        torch.cuda.synchronize(device)
    test_seconds = time.perf_counter() - test_start
    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if cuda else 0.0
    )

    save_outputs(
        out,
        args,
        test_metrics,
        detail,
        test_ds,
        best_epoch=int(best["epoch"]),
        best_val_rmse=best_rmse,
        train_seconds=train_seconds,
        test_seconds=test_seconds,
        total_params=total_params,
        trainable_params=trainable_params,
        peak_gpu_memory_mb=peak_gpu_memory_mb,
        device_name=device_name,
    )

    summary = {
        "state": "complete",
        "protocol": "revision-v2-LOCO",
        "heldout_city": args.heldout_city,
        "heldout_city_id": test_ds.heldout_city_id,
        "seed": args.seed,
        "best_epoch": int(best["epoch"]),
        "best_val_RMSE": best_rmse,
        "epochs_ran": len(history),
        "train_seconds": train_seconds,
        "test_seconds": test_seconds,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "city_identity_mode": "shared_single_id",
        "target_city_stats_used": False,
        **test_metrics,
        "base_MAE": detail["base_metrics"]["MAE"],
        "base_RMSE": detail["base_metrics"]["RMSE"],
        "mean_abs_correction": detail["mean_abs_correction"],
        "correction_RMSE": detail["correction_RMSE"],
        "correction_to_base_l2_ratio":
            detail["correction_to_base_l2_ratio"],
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    status_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not args.keep_checkpoint:
        for name in ("best.pt", "last.pt"):
            path = out / name
            if path.exists():
                path.unlink()

    print(pd.DataFrame([summary]).to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out = (
        args.out_dir.expanduser().resolve()
        if args.out_dir.is_absolute()
        else (root / args.out_dir).resolve()
    )
    try:
        run_main(args)
    except Exception as exc:
        out.mkdir(parents=True, exist_ok=True)
        (out / "run_status.json").write_text(
            json.dumps({
                "state": "failed",
                "heldout_city": args.heldout_city,
                "seed": args.seed,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
