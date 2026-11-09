#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global 22-city TSMixer baseline under the revision-v2 protocol.

Key properties
--------------
* One model is trained jointly on all 22 cities / 295 retained stations.
* Uses revision-v2 chronological indices and train-only city-variable statistics.
* Input: 96 h x 8 historical variables; target: 48 h O3.
* Optional information-matched conditioning supplies city identity and future
  calendar features without exposing future pollutant or meteorological values.
* Model selection uses validation physical RMSE. Calibration and test are not
  used for hyperparameter selection.
* Final test metrics include micro, city-macro, station-macro, city, station,
  and horizon summaries in physical O3 units.

The TSMixer implementation is imported from the project's existing
Time-Series-Library-main/models/TSMixer.py.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

L, H = 96, 48


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument("--tslib_root", type=Path, default=None)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--variant", choices=["plain", "city_time"], default="city_time")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--mse_weight", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--e_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--city_emb_dim", type=int, default=16)
    p.add_argument("--condition_hidden", type=int, default=64)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--max_test_samples", type=int, default=0)
    p.add_argument("--skip_test", action="store_true")
    p.add_argument("--require_cuda", action="store_true")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--log_every", type=int, default=500)
    return p.parse_args()


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Use a reproducible FP32 CUDA policy for all final neural baselines.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


class ArrayCache:
    """Small per-worker LRU cache for station arrays."""
    def __init__(self, root: Path, array_paths: list[str], max_items: int = 32):
        self.root = root
        self.array_paths = array_paths
        self.max_items = max_items
        self.cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, array_id: int) -> tuple[np.ndarray, np.ndarray]:
        if array_id in self.cache:
            value = self.cache.pop(array_id)
            self.cache[array_id] = value
            return value
        path = Path(self.array_paths[array_id])
        if not path.is_absolute():
            path = self.root / path
        with np.load(path, allow_pickle=True) as z:
            values = z["values"].astype(np.float32)
            time_ns = z["time_ns"].astype(np.int64)
        self.cache[array_id] = (values, time_ns)
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return values, time_ns


def future_calendar_features(time_ns: np.ndarray) -> np.ndarray:
    """hour and day-of-year sine/cosine features, shape [H, 4]."""
    ts = pd.to_datetime(time_ns)
    hour = ts.hour.to_numpy(dtype=np.float32)
    doy = ts.dayofyear.to_numpy(dtype=np.float32)
    return np.stack([
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * (doy - 1.0) / 365.25),
        np.cos(2 * np.pi * (doy - 1.0) / 365.25),
    ], axis=-1).astype(np.float32)


class GlobalV2Dataset(Dataset):
    def __init__(self, root: Path, split: str, max_samples: int, sample_seed: int):
        self.root = root
        self.index_root = root / "experiments" / "data_v2" / "indices"
        self.stats_path = root / "experiments" / "data_v2" / "city_var_stats_train_only.json"
        self.meta = json.loads((self.index_root / "meta.json").read_text(encoding="utf-8"))
        self.stats = json.loads(self.stats_path.read_text(encoding="utf-8"))
        z = np.load(self.index_root / f"{split}.npz", allow_pickle=False)
        self.array_id = z["array_id"].astype(np.int32)
        self.city_id = z["city_id"].astype(np.int64)
        self.start_idx = z["start_idx"].astype(np.int32)
        if max_samples and max_samples > 0 and len(self.array_id) > max_samples:
            rng = np.random.default_rng(sample_seed)
            sel = np.sort(rng.choice(len(self.array_id), size=max_samples, replace=False))
            self.array_id = self.array_id[sel]
            self.city_id = self.city_id[sel]
            self.start_idx = self.start_idx[sel]
        self.variables = [str(x) for x in self.meta["variables"]]
        self.cities = [str(x) for x in self.meta["cities"]]
        self.o3_idx = int(self.meta["o3_idx"])
        self.array_paths = [str(x) for x in self.meta["array_paths"]]
        self.stations = self.meta["stations"]
        self.cache = ArrayCache(root, self.array_paths)
        means, stds = [], []
        for city in self.cities:
            cstats = self.stats["city_variable_stats"][city]
            means.append([float(cstats[v]["mean"]) for v in self.variables])
            stds.append([max(float(cstats[v]["std"]), 1e-6) for v in self.variables])
        self.city_mean = np.asarray(means, dtype=np.float32)
        self.city_std = np.asarray(stds, dtype=np.float32)
        self.split = split

    def __len__(self) -> int:
        return len(self.array_id)

    def __getitem__(self, i: int) -> dict:
        aid = int(self.array_id[i])
        cid = int(self.city_id[i])
        st = int(self.start_idx[i])
        values, time_ns = self.cache.get(aid)
        mu = self.city_mean[cid]
        sd = self.city_std[cid]
        x = (values[st:st + L] - mu) / sd
        y_phys = values[st + L:st + L + H, self.o3_idx]
        y_std = (y_phys - mu[self.o3_idx]) / sd[self.o3_idx]
        ft = future_calendar_features(time_ns[st + L:st + L + H])
        return {
            "x": torch.from_numpy(np.asarray(x, dtype=np.float32)),
            "future_time": torch.from_numpy(ft),
            "y_std": torch.from_numpy(np.asarray(y_std, dtype=np.float32)),
            "y_phys": torch.from_numpy(np.asarray(y_phys, dtype=np.float32)),
            "city_id": torch.tensor(cid, dtype=torch.long),
            "array_id": torch.tensor(aid, dtype=torch.long),
            "target_mu": torch.tensor(mu[self.o3_idx], dtype=torch.float32),
            "target_std": torch.tensor(sd[self.o3_idx], dtype=torch.float32),
        }


class GlobalTSMixer(nn.Module):
    def __init__(self, backbone: nn.Module, n_cities: int, n_vars: int, o3_idx: int,
                 variant: str, city_emb_dim: int, condition_hidden: int):
        super().__init__()
        self.backbone = backbone
        self.o3_idx = o3_idx
        self.variant = variant
        if variant == "city_time":
            self.city_emb = nn.Embedding(n_cities, city_emb_dim)
            self.city_to_input = nn.Linear(city_emb_dim, n_vars)
            self.condition_head = nn.Sequential(
                nn.Linear(1 + 4 + city_emb_dim, condition_hidden),
                nn.GELU(),
                nn.Linear(condition_hidden, 1),
            )
            self.condition_gate = nn.Parameter(torch.tensor(1e-3))
            nn.init.zeros_(self.condition_head[-1].weight)
            nn.init.zeros_(self.condition_head[-1].bias)

    def forward(self, x: torch.Tensor, future_time: torch.Tensor,
                city_id: torch.Tensor) -> torch.Tensor:
        x_in = x
        city_vec = None
        if self.variant == "city_time":
            city_vec = self.city_emb(city_id)
            x_in = x + self.city_to_input(city_vec).unsqueeze(1)
        # Official TSMixer ignores time marks and decoder inputs.
        out = self.backbone(x_in, None, None, None)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim != 3:
            raise RuntimeError(f"Unexpected TSMixer output shape: {tuple(out.shape)}")
        base = out[:, -H:, self.o3_idx]
        if self.variant == "plain":
            return base
        cv = city_vec.unsqueeze(1).expand(-1, H, -1)
        cond = torch.cat([base.unsqueeze(-1), future_time, cv], dim=-1)
        correction = self.condition_head(cond).squeeze(-1)
        return base + self.condition_gate * correction


class SumMetric:
    def __init__(self) -> None:
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, pred: np.ndarray, truth: np.ndarray) -> None:
        d = np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
        self.abs_sum += float(np.abs(d).sum())
        self.sq_sum += float(np.square(d).sum())
        self.count += int(d.size)

    def result(self) -> dict:
        if not self.count:
            return {"MAE": math.nan, "MSE": math.nan, "RMSE": math.nan, "count": 0}
        mse = self.sq_sum / self.count
        return {"MAE": self.abs_sum / self.count, "MSE": mse,
                "RMSE": math.sqrt(mse), "count": self.count}


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, workers: int, cuda: bool) -> DataLoader:
    kwargs = dict(dataset=ds, batch_size=batch_size, shuffle=shuffle,
                  num_workers=workers, drop_last=False, pin_memory=cuda)
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def import_tsmixer(tslib_root: Path, cfg: SimpleNamespace) -> nn.Module:
    if str(tslib_root) not in sys.path:
        sys.path.insert(0, str(tslib_root))
    mod = importlib.import_module("models.TSMixer")
    return mod.Model(cfg)


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
              optimizer: torch.optim.Optimizer | None, mse_weight: float,
              grad_clip: float, log_every: int, collect_detail: bool,
              cities: list[str], stations: list[dict]) -> tuple[dict, dict]:
    training = optimizer is not None
    model.train(training)
    overall = SumMetric()
    by_city: Dict[int, SumMetric] = defaultdict(SumMetric)
    by_station: Dict[int, SumMetric] = defaultdict(SumMetric)
    h_abs = np.zeros(H, dtype=np.float64)
    h_sq = np.zeros(H, dtype=np.float64)
    h_n = np.zeros(H, dtype=np.int64)
    loss_sum, batches = 0.0, 0

    for step, batch in enumerate(loader, 1):
        x = batch["x"].to(device, dtype=torch.float32, non_blocking=True)
        ft = batch["future_time"].to(device, dtype=torch.float32, non_blocking=True)
        ys = batch["y_std"].to(device, dtype=torch.float32, non_blocking=True)
        ci = batch["city_id"].to(device, non_blocking=True)
        mu = batch["target_mu"].numpy()[:, None]
        sd = batch["target_std"].numpy()[:, None]
        y_phys = batch["y_phys"].numpy()

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            pred_s = model(x, ft, ci)
            loss = F.l1_loss(pred_s, ys) + mse_weight * F.mse_loss(pred_s, ys)
            if training:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        loss_sum += float(loss.detach().cpu())
        batches += 1
        pred = pred_s.detach().cpu().numpy() * sd + mu
        overall.update(pred, y_phys)

        if collect_detail:
            city_ids = batch["city_id"].numpy()
            array_ids = batch["array_id"].numpy()
            for cid in np.unique(city_ids):
                mask = city_ids == cid
                by_city[int(cid)].update(pred[mask], y_phys[mask])
            for aid in np.unique(array_ids):
                mask = array_ids == aid
                by_station[int(aid)].update(pred[mask], y_phys[mask])
            d = pred.astype(np.float64) - y_phys.astype(np.float64)
            h_abs += np.abs(d).sum(axis=0)
            h_sq += np.square(d).sum(axis=0)
            h_n += d.shape[0]

        if training and (step == 1 or step % log_every == 0):
            r = overall.result()
            print(f"step={step}/{len(loader)} loss={loss_sum/batches:.6f} "
                  f"MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f}", flush=True)

    detail = {"by_city": by_city, "by_station": by_station,
              "h_abs": h_abs, "h_sq": h_sq, "h_n": h_n}
    return {"loss": loss_sum / max(batches, 1), **overall.result()}, detail


def save_details(out_dir: Path, detail: dict, cities: list[str], stations: list[dict]) -> dict:
    city_rows = []
    for cid, meter in sorted(detail["by_city"].items()):
        city_rows.append({"city_id": cid, "city": cities[cid], **meter.result()})
    city_df = pd.DataFrame(city_rows)
    city_df.to_csv(out_dir / "metrics_by_city.csv", index=False, encoding="utf-8-sig")

    station_rows = []
    for aid, meter in sorted(detail["by_station"].items()):
        sm = stations[aid]
        station_rows.append({"array_id": aid, "city": sm["city"],
                             "station_id": sm["station_id"], **meter.result()})
    station_df = pd.DataFrame(station_rows)
    station_df.to_csv(out_dir / "metrics_by_station.csv", index=False, encoding="utf-8-sig")

    h_rows = []
    for h in range(H):
        n = int(detail["h_n"][h])
        mse = detail["h_sq"][h] / n if n else math.nan
        h_rows.append({"horizon": h + 1,
                       "MAE": detail["h_abs"][h] / n if n else math.nan,
                       "MSE": mse, "RMSE": math.sqrt(mse) if n else math.nan,
                       "count": n})
    pd.DataFrame(h_rows).to_csv(out_dir / "metrics_by_horizon.csv", index=False, encoding="utf-8-sig")

    return {
        "city_MAE_macro": float(city_df["MAE"].mean()),
        "city_RMSE_macro": float(city_df["RMSE"].mean()),
        "city_MAE_median": float(city_df["MAE"].median()),
        "city_RMSE_median": float(city_df["RMSE"].median()),
        "station_MAE_macro": float(station_df["MAE"].mean()),
        "station_RMSE_macro": float(station_df["RMSE"].mean()),
        "station_MAE_median": float(station_df["MAE"].median()),
        "station_RMSE_median": float(station_df["RMSE"].median()),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    tslib_root = (args.tslib_root or root / "Time-Series-Library-main").resolve()
    out_dir = (root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    cuda = torch.cuda.is_available()
    if args.require_cuda and not cuda:
        raise RuntimeError("CUDA is required for this run but torch.cuda.is_available() is False")
    device = torch.device("cuda:0" if cuda else "cpu")

    train_ds = GlobalV2Dataset(root, "train", args.max_train_samples, args.seed)
    val_ds = GlobalV2Dataset(root, "val", args.max_val_samples, args.seed + 1)
    test_ds = GlobalV2Dataset(root, "test", args.max_test_samples, args.seed + 2)
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, cuda)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers, cuda)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers, cuda)

    cfg = SimpleNamespace(task_name="long_term_forecast", seq_len=L, pred_len=H,
                          d_model=args.d_model, dropout=args.dropout,
                          enc_in=len(train_ds.variables), e_layers=args.e_layers)
    backbone = import_tsmixer(tslib_root, cfg)
    model = GlobalTSMixer(backbone, len(train_ds.cities), len(train_ds.variables),
                          train_ds.o3_idx, args.variant, args.city_emb_dim,
                          args.condition_hidden).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    gpu_name = torch.cuda.get_device_name(0) if cuda else "CPU"
    print("============================================================")
    print("[GLOBAL TSMIXER REVISION-V2]")
    print(f"variant={args.variant} seed={args.seed} device={device} ({gpu_name})")
    print(f"train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}")
    print(f"batch={args.batch_size} lr={args.lr} d_model={args.d_model} e_layers={args.e_layers}")
    print(f"params total={total_params:,}, trainable={trainable_params:,}")
    print("============================================================", flush=True)

    run_meta = vars(args).copy()
    run_meta.update({"root": str(root), "tslib_root": str(tslib_root),
                     "out_dir": str(out_dir), "device": str(device),
                     "gpu_name": gpu_name, "total_params": total_params,
                     "trainable_params": trainable_params,
                     "variables": train_ds.variables, "cities": train_ds.cities,
                     "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
                     "metric_scale": "physical O3 after city-level train-only inverse standardization",
                     "loss": f"L1 + {args.mse_weight} * MSE on standardized O3"})
    (out_dir / "args.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best_rmse = float("inf")
    best_epoch = -1
    bad_epochs = 0
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    train_t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} train")
        tr, _ = run_epoch(model, train_loader, device, optimizer, args.mse_weight,
                          args.grad_clip, args.log_every, False,
                          train_ds.cities, train_ds.stations)
        print(f"Epoch {epoch}/{args.epochs} val")
        va, _ = run_epoch(model, val_loader, device, None, args.mse_weight,
                          args.grad_clip, args.log_every, False,
                          train_ds.cities, train_ds.stations)
        history.append({"epoch": epoch,
                        **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in va.items()}})
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False, encoding="utf-8-sig")
        print(f"epoch={epoch} val_MAE={va['MAE']:.5f} val_RMSE={va['RMSE']:.5f}", flush=True)

        if va["RMSE"] < best_rmse - 1e-9:
            best_rmse = va["RMSE"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "best_val_RMSE": best_rmse, "args": run_meta}, out_dir / "best.pt")
            print(f"Saved best.pt val_RMSE={best_rmse:.5f}", flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}", flush=True)
                break

    train_seconds = time.perf_counter() - train_t0
    peak_memory_mb = (torch.cuda.max_memory_allocated() / 1024**2) if cuda else math.nan
    pd.DataFrame([{
        "variant": args.variant, "seed": args.seed, "lr": args.lr,
        "d_model": args.d_model, "e_layers": args.e_layers,
        "best_epoch": best_epoch, "best_val_RMSE": best_rmse,
        "train_seconds": train_seconds, "peak_gpu_memory_mb": peak_memory_mb,
        "total_params": total_params, "trainable_params": trainable_params,
    }]).to_csv(out_dir / "run_summary.csv", index=False, encoding="utf-8-sig")

    if args.skip_test:
        print("skip_test=True: validation-only tuning run completed.")
        return

    best = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"])
    print("\nFinal test evaluation")
    test_t0 = time.perf_counter()
    te, detail = run_epoch(model, test_loader, device, None, args.mse_weight,
                           args.grad_clip, args.log_every, True,
                           train_ds.cities, train_ds.stations)
    test_seconds = time.perf_counter() - test_t0
    macro = save_details(out_dir, detail, train_ds.cities, train_ds.stations)
    row = {**te, **macro, "variant": args.variant, "seed": args.seed,
           "lr": args.lr, "d_model": args.d_model, "e_layers": args.e_layers,
           "best_epoch": best_epoch, "best_val_RMSE": best_rmse,
           "train_seconds": train_seconds, "test_seconds": test_seconds,
           "peak_gpu_memory_mb": peak_memory_mb, "gpu_name": gpu_name,
           "total_params": total_params, "trainable_params": trainable_params}
    pd.DataFrame([row]).to_csv(out_dir / "metrics_overall.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
