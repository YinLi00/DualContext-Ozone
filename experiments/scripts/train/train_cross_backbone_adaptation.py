#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the same lightweight output adapter on a cached frozen-TSFM forecast.

The adapter is intentionally identical in information content to P1 from the
progressive ablation:
- 48-step standardized frozen-backbone forecast;
- horizon-wise gamma/beta;
- original 96-hour standardized O3 history;
- variable embedding;
- city embedding;
- one residual MLP and a scalar output gate.

It excludes InputAdapter, multivariate context and future calendar features.
Only the cached frozen base forecast differs across Sundial, TimeMoE-200M and
SEMPO.
"""
from __future__ import annotations

import argparse
import json
import math
import os
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

L, H = 96, 48
MODELS = ("sundial", "timemoe_200m", "sempo")
DISPLAY_NAMES = {
    "sundial": "Sundial",
    "timemoe_200m": "TimeMoE-200M",
    "sempo": "SEMPO",
}
EXPECTED_TEST_SHAPE = (106_200, H)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--mse_weight", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--emb_dim", type=int, default=16)
    p.add_argument("--city_emb_dim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--max_test_samples", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--require_cuda", action="store_true")
    p.add_argument("--cpu_threads", type=int, default=4)
    p.add_argument("--save_arrays", type=int, choices=[0, 1], default=1)
    p.add_argument("--keep_checkpoint", type=int, choices=[0, 1], default=1)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument(
        "--cache_root",
        type=Path,
        default=Path(
            "experiments/results/"
            "12_cross_backbone_output_adapter_v2/"
            "00_native_cache/aligned"
        ),
    )
    return p.parse_args()


def result_root(root: Path) -> Path:
    return root / "experiments/results/12_cross_backbone_output_adapter_v2"


def cache_dir(
    root: Path,
    cache_root: Path,
    model: str,
    split: str,
) -> Path:
    base = (
        cache_root.expanduser().resolve()
        if cache_root.is_absolute()
        else (root / cache_root).resolve()
    )
    return base / model / split


class CachedBaseDataset(Dataset):
    def __init__(
        self,
        directory: Path,
        max_samples: int,
        seed: int,
    ) -> None:
        manifest = directory / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"Aligned cache manifest missing: {manifest}"
            )
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("state") != "complete":
            raise RuntimeError(f"Incomplete aligned cache: {directory}")

        self.directory = directory
        self.base = np.load(
            directory / "base_pred_std.npy", mmap_mode="r"
        )
        self.x = np.load(directory / "x.npy", mmap_mode="r")
        self.y_std = np.load(
            directory / "y_std.npy", mmap_mode="r"
        )
        self.y_phys = np.load(
            directory / "y_phys.npy", mmap_mode="r"
        )
        self.mu = np.load(directory / "mu.npy", mmap_mode="r")
        self.std = np.load(directory / "std.npy", mmap_mode="r")
        self.city = np.load(
            directory / "city_id.npy", mmap_mode="r"
        )
        self.var_idx = np.load(
            directory / "var_idx.npy", mmap_mode="r"
        )
        station_path = directory / "station_id.npy"
        self.station = (
            np.load(station_path, allow_pickle=False)
            if station_path.exists() else None
        )

        n = self.base.shape[0]
        for array in (
            self.x, self.y_std, self.y_phys,
            self.mu, self.std, self.city, self.var_idx,
        ):
            if array.shape[0] != n:
                raise RuntimeError(
                    f"Cache length mismatch in {directory}"
                )
        if max_samples and max_samples < n:
            rng = np.random.default_rng(seed)
            self.indices = np.sort(
                rng.choice(n, size=max_samples, replace=False)
            )
        else:
            self.indices = np.arange(n, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> dict[str, Any]:
        i = int(self.indices[index])
        item: dict[str, Any] = {
            "base": torch.from_numpy(
                np.array(self.base[i], dtype=np.float32, copy=True)
            ),
            "x": torch.from_numpy(
                np.array(self.x[i], dtype=np.float32, copy=True)
            ),
            "y_std": torch.from_numpy(
                np.array(self.y_std[i], dtype=np.float32, copy=True)
            ),
            "y_phys": torch.from_numpy(
                np.array(self.y_phys[i], dtype=np.float32, copy=True)
            ),
            "mu": torch.tensor(float(self.mu[i]), dtype=torch.float32),
            "std": torch.tensor(float(self.std[i]), dtype=torch.float32),
            "city_id": torch.tensor(int(self.city[i]), dtype=torch.long),
            "var_idx": torch.tensor(int(self.var_idx[i]), dtype=torch.long),
        }
        if self.station is not None:
            item["station_id"] = str(self.station[i])
        return item


class OutputAdapter(nn.Module):
    def __init__(
        self,
        n_vars: int,
        n_cities: int,
        hidden: int,
        emb_dim: int,
        city_emb_dim: int,
        dropout: float,
        init_gate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.gamma = nn.Embedding(n_vars, H)
        self.beta = nn.Embedding(n_vars, H)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        self.var_emb = nn.Embedding(n_vars, emb_dim)
        self.city_emb = nn.Embedding(n_cities, city_emb_dim)
        in_dim = H + L + emb_dim + city_emb_dim
        self.output_net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, H),
        )
        self.output_gate = nn.Parameter(
            torch.tensor(float(init_gate))
        )
        nn.init.zeros_(self.output_net[-1].weight)
        nn.init.zeros_(self.output_net[-1].bias)

    def forward(
        self,
        base: torch.Tensor,
        x: torch.Tensor,
        var_idx: torch.Tensor,
        city_id: torch.Tensor,
    ) -> torch.Tensor:
        calibrated = (
            self.gamma(var_idx) * base + self.beta(var_idx)
        )
        features = torch.cat(
            [
                calibrated,
                x,
                self.var_emb(var_idx),
                self.city_emb(city_id),
            ],
            dim=-1,
        )
        return calibrated + self.output_gate * self.output_net(
            features
        )


class SumMetric:
    def __init__(self) -> None:
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, pred: np.ndarray, truth: np.ndarray) -> None:
        diff = (
            np.asarray(pred, dtype=np.float64)
            - np.asarray(truth, dtype=np.float64)
        )
        self.abs_sum += float(np.abs(diff).sum())
        self.sq_sum += float(np.square(diff).sum())
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


def resolve_device(args: argparse.Namespace) -> torch.device:
    has_cuda = torch.cuda.is_available()
    if args.require_cuda and not has_cuda:
        raise RuntimeError("--require_cuda set but CUDA unavailable")
    if args.device == "cpu":
        torch.set_num_threads(max(1, args.cpu_threads))
        return torch.device("cpu")
    if args.device == "cuda":
        if not has_cuda:
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if has_cuda else "cpu")


def seed_all(seed: int, cuda: bool) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = cuda


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


def run_epoch(
    model: OutputAdapter,
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
    overall = SumMetric()
    by_city: dict[int, SumMetric] = defaultdict(SumMetric)
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
        base = batch["base"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        x = batch["x"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        y_std = batch["y_std"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        var_idx = batch["var_idx"].to(
            device, non_blocking=True
        )
        city_id = batch["city_id"].to(
            device, non_blocking=True
        )
        y_phys = batch["y_phys"].numpy().astype(
            np.float32, copy=False
        )
        mu = batch["mu"].numpy().astype(
            np.float32, copy=False
        )[:, None]
        std = batch["std"].numpy().astype(
            np.float32, copy=False
        )[:, None]

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            pred_std = model(base, x, var_idx, city_id)
            loss = (
                F.l1_loss(pred_std, y_std)
                + mse_weight * F.mse_loss(pred_std, y_std)
            )
            if training:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip
                    )
                optimizer.step()

        loss_sum += float(loss.detach().cpu())
        batches += 1
        pred = (
            pred_std.detach().cpu().numpy().astype(
                np.float32, copy=False
            ) * std + mu
        )
        overall.update(pred, y_phys)

        if collect:
            city = batch["city_id"].numpy().astype(np.int64)
            for cid in np.unique(city):
                mask = city == cid
                by_city[int(cid)].update(
                    pred[mask], y_phys[mask]
                )
            if "station_id" in batch:
                stations = np.asarray(batch["station_id"])
                for station in np.unique(stations):
                    mask = stations == station
                    by_station[str(station)].update(
                        pred[mask], y_phys[mask]
                    )
                station_batches.append(stations)

            diff = pred.astype(np.float64) - y_phys.astype(
                np.float64
            )
            h_abs += np.abs(diff).sum(axis=0)
            h_sq += np.square(diff).sum(axis=0)
            h_n += diff.shape[0]
            pred_batches.append(pred)
            true_batches.append(y_phys)
            city_batches.append(city)

        if training and (step == 1 or step % log_every == 0):
            metric = overall.result()
            print(
                f"step={step}/{len(loader)} "
                f"loss={loss_sum/max(batches,1):.6f} "
                f"MAE={metric['MAE']:.4f} "
                f"RMSE={metric['RMSE']:.4f}",
                flush=True,
            )

    detail: dict[str, Any] = {
        "by_city": by_city,
        "by_station": by_station,
        "h_abs": h_abs,
        "h_sq": h_sq,
        "h_n": h_n,
    }
    if collect:
        detail["pred"] = np.concatenate(pred_batches)
        detail["true"] = np.concatenate(true_batches)
        detail["city_id"] = np.concatenate(city_batches)
        if station_batches:
            detail["station_id"] = np.concatenate(
                station_batches
            )
    return (
        {
            "loss": loss_sum / max(batches, 1),
            **overall.result(),
        },
        detail,
    )


def canonical_true(root: Path) -> Path:
    archive = (
        root / "experiments/results/"
        "03_zeroshot_baselines_v2_archive/Timer/true.npy"
    )
    return archive if archive.is_file() else root / "experiments/data_v2/indices/test_targets.npy"


def save_true(
    true: np.ndarray,
    output: Path,
    reference: Path,
    formal: bool,
) -> str:
    if output.exists() or output.is_symlink():
        output.unlink()
    if formal:
        ref = np.load(reference, mmap_mode="r")
        if ref.shape != true.shape or not np.array_equal(
            np.asarray(ref), true, equal_nan=True
        ):
            raise RuntimeError(
                "Adapter true array differs from canonical revision-v2 true"
            )
        try:
            os.link(reference.resolve(), output)
            return "hardlink_to_canonical"
        except OSError:
            relative = os.path.relpath(
                reference.resolve(), output.parent.resolve()
            )
            os.symlink(relative, output)
            return "symlink_to_canonical"
    np.save(output, true)
    return "saved_smoke_subset"


def save_outputs(
    out: Path,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    detail: dict[str, Any],
    *,
    best_epoch: int,
    best_val_rmse: float,
    train_seconds: float,
    test_seconds: float,
    trainable_params: int,
    device_name: str,
    peak_gpu_memory_mb: float,
) -> None:
    row = {
        **metrics,
        "model": args.model,
        "backbone": DISPLAY_NAMES[args.model],
        "method": "Output Adapter",
        "seed": args.seed,
        "protocol": "revision-v2",
        "best_epoch": best_epoch,
        "best_val_RMSE": best_val_rmse,
        "train_seconds": train_seconds,
        "test_seconds": test_seconds,
        "trainable_params": trainable_params,
        "device_name": device_name,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
    }
    pd.DataFrame([row]).to_csv(
        out / "metrics_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    city_rows = []
    for cid, metric in sorted(detail["by_city"].items()):
        city_rows.append({
            "city_id": cid,
            **metric.result(),
        })
    pd.DataFrame(city_rows).to_csv(
        out / "metrics_by_city.csv",
        index=False,
        encoding="utf-8-sig",
    )

    station_rows = []
    for station, metric in sorted(detail["by_station"].items()):
        station_rows.append({
            "station_id": station,
            **metric.result(),
        })
    pd.DataFrame(
        station_rows,
        columns=["station_id", "MAE", "MSE", "RMSE", "count"],
    ).to_csv(
        out / "metrics_by_station.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_rows = []
    for h in range(H):
        n = int(detail["h_n"][h])
        mse = detail["h_sq"][h] / n if n else math.nan
        horizon_rows.append({
            "horizon": h + 1,
            "MAE": detail["h_abs"][h] / n if n else math.nan,
            "MSE": mse,
            "RMSE": math.sqrt(mse) if n else math.nan,
            "count": n,
        })
    pd.DataFrame(horizon_rows).to_csv(
        out / "metrics_by_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if args.save_arrays:
        pred = np.asarray(detail["pred"], dtype=np.float32)
        true = np.asarray(detail["true"], dtype=np.float32)
        np.save(out / "pred.npy", pred)
        formal = (
            args.max_test_samples == 0
            and pred.shape == EXPECTED_TEST_SHAPE
            and true.shape == EXPECTED_TEST_SHAPE
        )
        true_storage = save_true(
            true, out / "true.npy", canonical_true(args.root), formal
        )
        np.save(out / "city_id.npy", detail["city_id"])
        if "station_id" in detail:
            np.save(out / "station_id.npy", detail["station_id"])
        manifest = {
            "protocol": "revision-v2",
            "model": args.model,
            "seed": args.seed,
            "pred_shape": list(pred.shape),
            "true_shape": list(true.shape),
            "count": int(pred.size),
            "nonfinite_pred": int((~np.isfinite(pred)).sum()),
            "nonfinite_true": int((~np.isfinite(true)).sum()),
            "true_storage": true_storage,
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

    device = resolve_device(args)
    cuda = device.type == "cuda"
    seed_all(args.seed, cuda)

    train_ds = CachedBaseDataset(
        cache_dir(root, args.cache_root, args.model, "train"),
        args.max_train_samples,
        args.seed,
    )
    val_ds = CachedBaseDataset(
        cache_dir(root, args.cache_root, args.model, "val"),
        args.max_val_samples,
        args.seed + 1,
    )
    test_ds = CachedBaseDataset(
        cache_dir(root, args.cache_root, args.model, "test"),
        args.max_test_samples,
        args.seed + 2,
    )
    train_loader = make_loader(
        train_ds, args.batch_size, True,
        args.num_workers, cuda
    )
    val_loader = make_loader(
        val_ds, args.batch_size, False,
        args.num_workers, cuda
    )
    test_loader = make_loader(
        test_ds, args.batch_size, False,
        args.num_workers, cuda
    )

    all_city = np.load(
        cache_dir(root, args.cache_root, args.model, "train") / "city_id.npy",
        mmap_mode="r",
    )
    all_var = np.load(
        cache_dir(root, args.cache_root, args.model, "train") / "var_idx.npy",
        mmap_mode="r",
    )
    n_cities = int(np.max(all_city)) + 1
    n_vars = int(np.max(all_var)) + 1

    model = OutputAdapter(
        n_vars=n_vars,
        n_cities=n_cities,
        hidden=args.hidden,
        emb_dim=args.emb_dim,
        city_emb_dim=args.city_emb_dim,
        dropout=args.dropout,
    ).to(device)
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    device_name = (
        torch.cuda.get_device_name(device)
        if cuda else f"CPU ({torch.get_num_threads()} threads)"
    )
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)

    print("=" * 88)
    print("[CROSS-BACKBONE OUTPUT ADAPTER / REVISION-V2]")
    print(
        f"backbone={DISPLAY_NAMES[args.model]} "
        f"seed={args.seed} device={device_name}"
    )
    print(
        f"train={len(train_ds):,} val={len(val_ds):,} "
        f"test={len(test_ds):,}"
    )
    print(
        f"epochs={args.epochs} patience={args.patience} "
        f"batch={args.batch_size} lr={args.lr}"
    )
    print(f"trainable_params={trainable_params:,}")
    print("=" * 88)

    args_dict = vars(args).copy()
    for key, value in list(args_dict.items()):
        if isinstance(value, Path):
            args_dict[key] = str(value)
    args_dict.update({
        "protocol": "revision-v2",
        "backbone": DISPLAY_NAMES[args.model],
        "adapter_information": [
            "frozen_base_forecast_48",
            "raw_o3_history_96",
            "variable_embedding",
            "city_embedding",
        ],
        "excluded_information": [
            "input_adapter",
            "multivariate_context",
            "future_calendar",
            "pearson_cue",
        ],
        "trainable_params": trainable_params,
    })
    (out / "args.json").write_text(
        json.dumps(args_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    status_path = out / "run_status.json"
    status_path.write_text(
        json.dumps({
            "state": "running",
            "model": args.model,
            "seed": args.seed,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2),
        encoding="utf-8",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
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
            model, train_loader, device,
            optimizer=optimizer,
            mse_weight=args.mse_weight,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
            collect=False,
        )
        print(f"Epoch {epoch}/{args.epochs} val", flush=True)
        val_metrics, _ = run_epoch(
            model, val_loader, device,
            optimizer=None,
            mse_weight=args.mse_weight,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
            collect=False,
        )
        history.append({
            "epoch": epoch,
            **{
                f"train_{k}": v
                for k, v in train_metrics.items()
            },
            **{
                f"val_{k}": v
                for k, v in val_metrics.items()
            },
        })
        pd.DataFrame(history).to_csv(
            out / "history.csv",
            index=False,
            encoding="utf-8-sig",
        )
        checkpoint = {
            "epoch": epoch,
            "model_state": {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            },
            "args": args_dict,
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
    model.load_state_dict(best["model_state"], strict=True)

    test_start = time.perf_counter()
    test_metrics, detail = run_epoch(
        model, test_loader, device,
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
        best_epoch=int(best["epoch"]),
        best_val_rmse=best_rmse,
        train_seconds=train_seconds,
        test_seconds=test_seconds,
        trainable_params=trainable_params,
        device_name=device_name,
        peak_gpu_memory_mb=peak_gpu_memory_mb,
    )
    summary = {
        "state": "complete",
        "model": args.model,
        "backbone": DISPLAY_NAMES[args.model],
        "method": "Output Adapter",
        "seed": args.seed,
        "protocol": "revision-v2",
        "best_epoch": int(best["epoch"]),
        "best_val_RMSE": best_rmse,
        "epochs_ran": len(history),
        "train_seconds": train_seconds,
        "test_seconds": test_seconds,
        "trainable_params": trainable_params,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        **test_metrics,
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
                "model": args.model,
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
