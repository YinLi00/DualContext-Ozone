#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Progressive adapter and temporal-feature ablations under revision-v2.

The script reuses the project's canonical revision-v2 dataset and the exact
frozen-Timer/InputAdapter/CoRAContext modules used by
train_attribution_reference.py. It adds seven unique trainable variants:

Progressive adapter:
  p1_o3_output       output-side residual correction only
  p2_context         P1 + multivariate CoRAContext
  p3_time_context    P2 + future hour/DOY features
  p4_io_time_context P3 + input-side adaptation (full model)

Temporal features (all use the full P4 architecture and identical parameters):
  t0_none            four time channels retained but set to zero
  t1_hour            hour sin/cos retained; DOY sin/cos set to zero
  t2_doy             DOY sin/cos retained; hour sin/cos set to zero
  t3_hour_doy        alias of P4; not trained separately by the controller

Timer remains frozen in every variant. All trained runs use the same
revision-v2 split, train-only normalization, 96->48 task, loss, optimizer and
hyperparameters. Formal runs save pred.npy/true.npy and reported metrics.
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
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

L, H = 96, 48

VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "p1_o3_output": {
        "group": "progressive",
        "use_input_adapter": False,
        "use_context": False,
        "include_time_vector": False,
        "time_mask": "none",
        "paper_name": "O3-output adapter",
    },
    "p2_context": {
        "group": "progressive",
        "use_input_adapter": False,
        "use_context": True,
        "include_time_vector": False,
        "time_mask": "none",
        "paper_name": "Context adapter",
    },
    "p3_time_context": {
        "group": "progressive",
        "use_input_adapter": False,
        "use_context": True,
        "include_time_vector": True,
        "time_mask": "hour_doy",
        "paper_name": "Time-context adapter",
    },
    "p4_io_time_context": {
        "group": "progressive",
        "use_input_adapter": True,
        "use_context": True,
        "include_time_vector": True,
        "time_mask": "hour_doy",
        "paper_name": "IO-Time-Context adapter",
    },
    "t0_none": {
        "group": "time",
        "use_input_adapter": True,
        "use_context": True,
        "include_time_vector": True,
        "time_mask": "none",
        "paper_name": "No temporal features",
    },
    "t1_hour": {
        "group": "time",
        "use_input_adapter": True,
        "use_context": True,
        "include_time_vector": True,
        "time_mask": "hour",
        "paper_name": "Hour only",
    },
    "t2_doy": {
        "group": "time",
        "use_input_adapter": True,
        "use_context": True,
        "include_time_vector": True,
        "time_mask": "doy",
        "paper_name": "DOY only",
    },
    "t3_hour_doy": {
        "group": "time",
        "use_input_adapter": True,
        "use_context": True,
        "include_time_vector": True,
        "time_mask": "hour_doy",
        "paper_name": "Hour + DOY",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path,
                   default=Path(__file__).resolve().parents[3])
    p.add_argument("--variant", required=True, choices=sorted(VARIANT_SPECS))
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


def seed_everything(seed: int, cuda: bool) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    # Match the existing revision-v2 Timer trainers.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = cuda


def import_source(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_device(args: argparse.Namespace) -> torch.device:
    has_cuda = torch.cuda.is_available()
    if args.require_cuda and not has_cuda:
        raise RuntimeError("--require_cuda was set, but CUDA is unavailable")
    if args.device == "cpu":
        torch.set_num_threads(max(1, int(args.cpu_threads)))
        return torch.device("cpu")
    if args.device == "cuda":
        if not has_cuda:
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if has_cuda else "cpu")


class SumMetric:
    def __init__(self) -> None:
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, pred: np.ndarray, truth: np.ndarray) -> None:
        d = np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
        self.abs_sum += float(np.abs(d).sum(dtype=np.float64))
        self.sq_sum += float(np.square(d).sum(dtype=np.float64))
        self.count += int(d.size)

    def result(self) -> dict[str, Any]:
        if self.count == 0:
            return {"MAE": math.nan, "MSE": math.nan,
                    "RMSE": math.nan, "count": 0}
        mse = self.sq_sum / self.count
        return {
            "MAE": self.abs_sum / self.count,
            "MSE": mse,
            "RMSE": math.sqrt(mse),
            "count": self.count,
        }


class AdapterTimeAblationForecaster(nn.Module):
    """Variant-specific wrapper assembled from the canonical full model modules."""

    def __init__(
        self,
        *,
        source_module,
        timer: nn.Module,
        variant: str,
        n_vars: int,
        n_cities: int,
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
    ):
        super().__init__()
        self.variant = variant
        self.spec = dict(VARIANT_SPECS[variant])
        self.time_mask = str(self.spec["time_mask"])
        self.use_input_adapter = bool(self.spec["use_input_adapter"])
        self.use_context = bool(self.spec["use_context"])
        self.include_time_vector = bool(self.spec["include_time_vector"])

        # Instantiate the canonical full architecture once, then retain only the
        # modules required by this variant. P4/T3 use its exact module set.
        core = source_module.StrictAttributionForecaster(
            timer=timer,
            base_mode="timer",
            n_vars=n_vars,
            n_cities=n_cities,
            InputAdapter=InputAdapter,
            ContextEncoder=ContextEncoder,
            time_feature_dim=time_feature_dim,
            input_hidden=input_hidden,
            output_hidden=output_hidden,
            context_hidden=context_hidden,
            emb_dim=emb_dim,
            city_emb_dim=city_emb_dim,
            dropout=dropout,
            time_feature_mode="both",
            o3_idx=o3_idx,
            corr_temperature=corr_temperature,
            init_gate=init_gate,
        )

        self.timer = core.timer
        self.gamma = core.gamma
        self.beta = core.beta
        self.var_emb = core.var_emb
        self.city_emb = core.city_emb
        self.output_gate = core.output_gate

        if self.use_input_adapter:
            self.input_adapter = core.input_adapter
        if self.use_context:
            self.context_encoder = core.context_encoder

        time_dim = H * 4 if self.include_time_vector else 0
        in_dim = H + L + emb_dim + city_emb_dim
        if self.use_input_adapter:
            in_dim += L
        if self.use_context:
            in_dim += context_hidden
        in_dim += time_dim

        full_exact = (
            self.use_input_adapter
            and self.use_context
            and self.include_time_vector
        )
        if full_exact:
            self.output_net = core.output_net
        else:
            self.output_net = nn.Sequential(
                nn.Linear(in_dim, output_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_hidden, H),
            )
            nn.init.zeros_(self.output_net[-1].weight)
            nn.init.zeros_(self.output_net[-1].bias)

        # Timer is frozen but gradients through its input remain enabled, so the
        # input adapter in P4/T0-T3 can learn.
        for parameter in self.timer.parameters():
            parameter.requires_grad_(False)
        self.timer.eval()
        if hasattr(self.timer.config, "use_cache"):
            self.timer.config.use_cache = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.timer.eval()
        return self

    @staticmethod
    def mask_time(time_feat: torch.Tensor, mode: str) -> torch.Tensor:
        if time_feat.ndim != 2 or time_feat.shape[1] != H * 4:
            raise ValueError(
                f"Expected flattened both-mode time features [B,{H*4}], "
                f"got {tuple(time_feat.shape)}"
            )
        view = time_feat.reshape(time_feat.shape[0], H, 4)
        if mode == "hour_doy":
            out = view
        elif mode == "none":
            out = torch.zeros_like(view)
        elif mode == "hour":
            out = view.clone()
            out[:, :, 2:4] = 0.0
        elif mode == "doy":
            out = view.clone()
            out[:, :, 0:2] = 0.0
        else:
            raise ValueError(mode)
        return out.reshape(time_feat.shape[0], H * 4)

    def forward(
        self,
        x: torch.Tensor,
        x_multi: torch.Tensor,
        time_feat: torch.Tensor,
        var_idx: torch.Tensor,
        city_id: torch.Tensor,
    ) -> torch.Tensor:
        x_adapt = self.input_adapter(x, city_id) if self.use_input_adapter else x
        base = self.timer(x_adapt).logits[:, :H].float()
        pred = self.gamma(var_idx) * base + self.beta(var_idx)

        features = [pred, x]
        if self.use_input_adapter:
            features.append(x_adapt)
        if self.include_time_vector:
            features.append(self.mask_time(time_feat, self.time_mask))
        if self.use_context:
            features.append(self.context_encoder(x_multi))
        features.extend([self.var_emb(var_idx), self.city_emb(city_id)])

        return pred + self.output_gate * self.output_net(
            torch.cat(features, dim=-1)
        )


def make_loader(
    dataset,
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


def tensor_or_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def extract_optional_ids(batch: dict[str, Any]) -> Optional[np.ndarray]:
    for key in ("station_id", "station_idx", "site_id", "site_idx",
                "station_code", "station"):
        if key in batch:
            value = batch[key]
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)
    return None


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer=None,
    mse_weight: float,
    grad_clip: float,
    log_every: int,
    collect_detail: bool,
    collect_arrays: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    training = optimizer is not None
    model.train(training)

    overall = SumMetric()
    by_city: Dict[int, SumMetric] = defaultdict(SumMetric)
    by_station: dict[str, SumMetric] = defaultdict(SumMetric)
    h_abs = np.zeros(H, dtype=np.float64)
    h_sq = np.zeros(H, dtype=np.float64)
    h_n = np.zeros(H, dtype=np.int64)
    pred_batches: list[np.ndarray] = []
    true_batches: list[np.ndarray] = []
    city_batches: list[np.ndarray] = []
    station_batches: list[np.ndarray] = []

    loss_sum = 0.0
    batches = 0

    for step, batch in enumerate(loader, 1):
        x = batch["x"].to(device, dtype=torch.float32, non_blocking=True)
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

        y_phys = tensor_or_array(batch["y_phys"]).astype(np.float32, copy=False)
        mu = tensor_or_array(batch["mu"]).astype(np.float32, copy=False)[:, None]
        std = tensor_or_array(batch["std"]).astype(np.float32, copy=False)[:, None]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            pred_std = model(x, x_multi, time_feat, var_idx, city_id)
            loss = (
                F.l1_loss(pred_std, y_std)
                + mse_weight * F.mse_loss(pred_std, y_std)
            )
            if training:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        grad_clip,
                    )
                optimizer.step()

        loss_sum += float(loss.detach().cpu())
        batches += 1
        pred = (
            pred_std.detach().cpu().numpy().astype(np.float32, copy=False)
            * std
            + mu
        )
        overall.update(pred, y_phys)

        if collect_detail:
            city_ids = tensor_or_array(batch["city_id"]).astype(np.int64)
            for cid in np.unique(city_ids):
                mask = city_ids == cid
                by_city[int(cid)].update(pred[mask], y_phys[mask])

            station_ids = extract_optional_ids(batch)
            if station_ids is not None:
                station_ids = np.asarray(station_ids).reshape(-1)
                for sid in np.unique(station_ids):
                    mask = station_ids == sid
                    by_station[str(sid)].update(pred[mask], y_phys[mask])

            diff = pred.astype(np.float64) - y_phys.astype(np.float64)
            h_abs += np.abs(diff).sum(axis=0, dtype=np.float64)
            h_sq += np.square(diff).sum(axis=0, dtype=np.float64)
            h_n += diff.shape[0]

        if collect_arrays:
            pred_batches.append(np.asarray(pred, dtype=np.float32))
            true_batches.append(np.asarray(y_phys, dtype=np.float32))
            city_batches.append(
                tensor_or_array(batch["city_id"]).astype(np.int16)
            )
            station_ids = extract_optional_ids(batch)
            if station_ids is not None:
                station_batches.append(np.asarray(station_ids))

        if training and (step == 1 or step % log_every == 0):
            result = overall.result()
            print(
                f"step={step}/{len(loader)} "
                f"loss={loss_sum/max(batches,1):.6f} "
                f"MAE={result['MAE']:.4f} "
                f"RMSE={result['RMSE']:.4f}",
                flush=True,
            )

    detail: dict[str, Any] = {
        "by_city": by_city,
        "by_station": by_station,
        "h_abs": h_abs,
        "h_sq": h_sq,
        "h_n": h_n,
    }
    if collect_arrays:
        detail["pred"] = np.concatenate(pred_batches, axis=0)
        detail["true"] = np.concatenate(true_batches, axis=0)
        detail["city_id"] = np.concatenate(city_batches, axis=0)
        if station_batches and len(station_batches) == len(pred_batches):
            detail["station_id"] = np.concatenate(station_batches, axis=0)

    return (
        {"loss": loss_sum / max(batches, 1), **overall.result()},
        detail,
    )


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("timer.")
    }


def safe_hardlink_or_save_true(
    true: np.ndarray,
    out_path: Path,
    canonical_path: Path,
    formal_full_test: bool,
) -> str:
    if out_path.exists() or out_path.is_symlink():
        out_path.unlink()

    if formal_full_test and canonical_path.exists():
        canonical = np.load(canonical_path, mmap_mode="r")
        if canonical.shape != true.shape:
            raise RuntimeError(
                f"Canonical true shape={canonical.shape}, run true shape={true.shape}"
            )
        if not np.array_equal(np.asarray(canonical), true, equal_nan=True):
            max_diff = float(
                np.nanmax(
                    np.abs(
                        np.asarray(canonical, dtype=np.float64)
                        - true.astype(np.float64)
                    )
                )
            )
            raise RuntimeError(
                "Test true array does not exactly match canonical revision-v2 "
                f"true.npy; max_abs_diff={max_diff}"
            )
        try:
            os.link(canonical_path.resolve(), out_path)
            return "hardlink_to_canonical"
        except OSError:
            try:
                rel = os.path.relpath(canonical_path.resolve(), out_path.parent)
                os.symlink(rel, out_path)
                return "symlink_to_canonical"
            except OSError:
                np.save(out_path, true)
                return "copied_array"

    np.save(out_path, true)
    return "saved_subset_array"


def save_test_outputs(
    out_dir: Path,
    metrics: dict[str, Any],
    detail: dict[str, Any],
    cities: list[str],
    *,
    args: argparse.Namespace,
    total_params: int,
    trainable_params: int,
    best_epoch: int,
    best_val_rmse: float,
    train_seconds: float,
    test_seconds: float,
    peak_gpu_memory_mb: float,
    device_name: str,
    canonical_true: Path,
) -> None:
    row = {
        **metrics,
        "variant": args.variant,
        "paper_name": VARIANT_SPECS[args.variant]["paper_name"],
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_RMSE": best_val_rmse,
        "train_seconds": train_seconds,
        "test_seconds": test_seconds,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "device_name": device_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "protocol": "revision-v2",
    }
    pd.DataFrame([row]).to_csv(
        out_dir / "metrics_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    city_rows = []
    for cid, metric in sorted(detail["by_city"].items()):
        name = cities[cid] if 0 <= cid < len(cities) else str(cid)
        city_rows.append({
            "city_id": cid,
            "city": name,
            **metric.result(),
        })
    pd.DataFrame(city_rows).to_csv(
        out_dir / "metrics_by_city.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_rows = []
    for horizon in range(H):
        n = int(detail["h_n"][horizon])
        mse = detail["h_sq"][horizon] / n if n else math.nan
        horizon_rows.append({
            "horizon": horizon + 1,
            "MAE": detail["h_abs"][horizon] / n if n else math.nan,
            "MSE": mse,
            "RMSE": math.sqrt(mse) if n else math.nan,
            "count": n,
        })
    pd.DataFrame(horizon_rows).to_csv(
        out_dir / "metrics_by_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    station_rows = []
    for station, metric in sorted(detail["by_station"].items()):
        station_rows.append({"station_id": station, **metric.result()})
    pd.DataFrame(
        station_rows,
        columns=["station_id", "MAE", "MSE", "RMSE", "count"],
    ).to_csv(
        out_dir / "metrics_by_station.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if args.save_arrays:
        pred = np.asarray(detail["pred"], dtype=np.float32)
        true = np.asarray(detail["true"], dtype=np.float32)
        np.save(out_dir / "pred.npy", pred)
        formal_full_test = (
            args.max_test_samples == 0
            and pred.shape == (106_200, H)
            and true.shape == (106_200, H)
        )
        true_method = safe_hardlink_or_save_true(
            true,
            out_dir / "true.npy",
            canonical_true,
            formal_full_test,
        )
        np.save(out_dir / "city_id.npy", detail["city_id"])
        if "station_id" in detail:
            np.save(out_dir / "station_id.npy", detail["station_id"])

        array_manifest = {
            "protocol": "revision-v2",
            "variant": args.variant,
            "seed": args.seed,
            "pred_shape": list(pred.shape),
            "true_shape": list(true.shape),
            "pred_dtype": str(pred.dtype),
            "true_dtype": str(true.dtype),
            "count": int(pred.size),
            "nonfinite_pred": int((~np.isfinite(pred)).sum()),
            "nonfinite_true": int((~np.isfinite(true)).sum()),
            "true_storage": true_method,
            "canonical_true": str(canonical_true),
        }
        (out_dir / "array_manifest.json").write_text(
            json.dumps(array_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def run_main(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir.is_absolute()
        else (root / args.out_dir).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    source_path = (
        root / "experiments/scripts/train/"
        "train_attribution_reference.py"
    )
    if not source_path.exists():
        raise FileNotFoundError(
            f"Required canonical trainer not found: {source_path}"
        )
    source = import_source(source_path, "_adapter_time_source")

    device = resolve_device(args)
    cuda = device.type == "cuda"
    seed_everything(args.seed, cuda)

    index_root = root / "experiments/data_v2/indices"
    stats_path = (
        root / "experiments/data_v2/"
        "city_var_stats_train_only.json"
    )
    if not (index_root / "meta.json").exists() or not stats_path.exists():
        raise FileNotFoundError(
            "Revision-v2 index/statistics are missing. "
            "Run build_data_splits.py first."
        )

    cipda, cora = source.load_project_modules(root, index_root, stats_path)
    Dataset = cora.Stage1O3CoRAContextDataset

    # Always request both time channels. T0/T1/T2 use information masks while
    # retaining the exact same four-dimensional time input and parameter count.
    train_ds = Dataset("train", args.max_train_samples, args.seed, "both")
    val_ds = Dataset("val", args.max_val_samples, args.seed + 1, "both")
    test_ds = Dataset("test", args.max_test_samples, args.seed + 2, "both")

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

    model = AdapterTimeAblationForecaster(
        source_module=source,
        timer=timer,
        variant=args.variant,
        n_vars=len(train_ds.variables),
        n_cities=len(train_ds.cities),
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

    print("=" * 88)
    print("[ADAPTER + TIME ABLATION / REVISION-V2]")
    print(f"variant={args.variant} seed={args.seed}")
    print(f"paper_name={VARIANT_SPECS[args.variant]['paper_name']}")
    print(f"device={device} ({device_name})")
    print(
        f"train={len(train_ds):,} val={len(val_ds):,} "
        f"test={len(test_ds):,}"
    )
    print(
        f"epochs={args.epochs} patience={args.patience} "
        f"batch={args.batch_size} lr={args.lr}"
    )
    print(
        f"params total={total_params:,}, "
        f"trainable={trainable_params:,}"
    )
    print("=" * 88)

    args_dict = vars(args).copy()
    args_dict.update({
        "root": str(root),
        "out_dir": str(out_dir),
        "index_root": str(index_root),
        "stats_path": str(stats_path),
        "source_trainer": str(source_path),
        "time_dataset_mode": "both",
        "variant_spec": VARIANT_SPECS[args.variant],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "device_name": device_name,
        "protocol": "revision-v2",
    })
    for key, value in list(args_dict.items()):
        if isinstance(value, Path):
            args_dict[key] = str(value)
    (out_dir / "args.json").write_text(
        json.dumps(args_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    status_path = out_dir / "run_status.json"
    status_path.write_text(
        json.dumps({
            "state": "running",
            "variant": args.variant,
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
            collect_detail=False,
            collect_arrays=False,
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
            collect_detail=False,
            collect_arrays=False,
        )
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            out_dir / "history.csv",
            index=False,
            encoding="utf-8-sig",
        )

        checkpoint = {
            "epoch": epoch,
            "adapter_state": adapter_state(model),
            "args": args_dict,
            "cities": train_ds.cities,
            "variables": train_ds.variables,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, out_dir / "last.pt")

        if val_metrics["RMSE"] < best_rmse - 1e-12:
            best_rmse = float(val_metrics["RMSE"])
            bad_epochs = 0
            torch.save(checkpoint, out_dir / "best.pt")
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
            if args.patience > 0 and bad_epochs >= args.patience:
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
            out_dir / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        best = torch.load(out_dir / "best.pt", map_location="cpu")

    missing, unexpected = model.load_state_dict(
        best["adapter_state"],
        strict=False,
    )
    missing_non_timer = [
        name for name in missing if not name.startswith("timer.")
    ]
    if unexpected or missing_non_timer:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )

    print("\nFinal test evaluation", flush=True)
    test_start = time.perf_counter()
    test_metrics, detail = run_epoch(
        model,
        test_loader,
        device,
        optimizer=None,
        mse_weight=args.mse_weight,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        collect_detail=True,
        collect_arrays=bool(args.save_arrays),
    )
    if cuda:
        torch.cuda.synchronize(device)
    test_seconds = time.perf_counter() - test_start
    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if cuda else 0.0
    )

    canonical_true = (
        root / "experiments/results/"
        "03_zeroshot_baselines_v2_archive/Timer/true.npy"
    )
    if not canonical_true.exists():
        canonical_true = (
            root / "experiments/results/"
            "03_zeroshot_baselines_v2_reindexed/Timer/true.npy"
        )
    if not canonical_true.is_file():
        canonical_true = root / "experiments/data_v2/indices/test_targets.npy"

    save_test_outputs(
        out_dir,
        test_metrics,
        detail,
        train_ds.cities,
        args=args,
        total_params=total_params,
        trainable_params=trainable_params,
        best_epoch=int(best["epoch"]),
        best_val_rmse=best_rmse,
        train_seconds=train_seconds,
        test_seconds=test_seconds,
        peak_gpu_memory_mb=peak_gpu_memory_mb,
        device_name=device_name,
        canonical_true=canonical_true,
    )

    summary = {
        "state": "complete",
        "variant": args.variant,
        "paper_name": VARIANT_SPECS[args.variant]["paper_name"],
        "seed": args.seed,
        "protocol": "revision-v2",
        "best_epoch": int(best["epoch"]),
        "best_val_RMSE": best_rmse,
        "epochs_ran": len(history),
        "train_seconds": train_seconds,
        "test_seconds": test_seconds,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "total_params": total_params,
        "trainable_params": trainable_params,
        **test_metrics,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    status_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not args.keep_checkpoint:
        for name in ("last.pt", "best.pt"):
            path = out_dir / name
            if path.exists():
                path.unlink()

    print(pd.DataFrame([summary]).to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir.is_absolute()
        else (args.root.expanduser().resolve() / args.out_dir).resolve()
    )
    try:
        run_main(args)
    except Exception as exc:
        out_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "state": "failed",
            "variant": args.variant,
            "seed": args.seed,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (out_dir / "run_status.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
