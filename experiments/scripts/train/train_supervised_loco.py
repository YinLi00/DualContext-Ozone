#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train one supervised Global11 baseline under strict 22-city LOCO.

Protocol for held-out city c:
  * train/val: canonical revision-v2 samples from the other 21 cities;
  * test: canonical revision-v2 samples from c only;
  * normalization: source-21 pooled train-only scaler from the existing
    build_loco_folds.py folds;
  * city identity: a single shared city ID (0) for ALL source and target samples;
  * future hour/DOY conditioning is retained;
  * model architecture/hyperparameters are reused from the completed Global11
    final runs via train_supervised_scopes.read_reference_args().

Supported models (same Global11 set):
  TSMixer, DLinear, PatchTST, LSTM, iTransformer, TimeMixer, TimesNet,
  CNNLSTM, Transformer, TimeXer, Koopa.
"""
from __future__ import annotations

import argparse
import gc
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
import torch.nn.functional as F
from torch.utils.data import Dataset

ROOT_DEFAULT = Path(__file__).resolve().parents[3]
L, H = 96, 48
MODELS = [
    "TSMixer", "DLinear", "PatchTST", "LSTM", "iTransformer",
    "TimeMixer", "TimesNet", "CNNLSTM", "Transformer", "TimeXer", "Koopa",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    p.add_argument("--heldout_city", required=True)
    p.add_argument("--model_name", choices=MODELS, required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--reference_args", type=Path, default=None)

    # 0 / -1 mean: preserve the selected Global11 final-run value.
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=-1)

    # Only for smoke/diagnostics. Formal run uses all samples (0).
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--max_test_samples", type=int, default=0)

    p.add_argument("--keep_checkpoint", type=int, choices=[0, 1], default=0)
    p.add_argument("--require_cuda", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def fold_root(root: Path, city: str) -> Path:
    return (
        root / "experiments/results/14_loco_22cities_v2/00_fold_indices"
        / safe_name(city)
    )


def safe_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def add_train_modules(root: Path):
    train_root = root / "experiments/scripts/train"
    if str(train_root) not in sys.path:
        sys.path.insert(0, str(train_root))
    import train_supervised_scopes as scoped
    modules = scoped.add_train_path(root)
    return scoped, modules


def subset_indices(n: int, max_samples: int, seed: int, keep_order: bool) -> np.ndarray:
    idx = np.arange(n, dtype=np.int64)
    if max_samples <= 0 or n <= max_samples:
        return idx
    rng = np.random.default_rng(seed)
    out = rng.choice(n, size=max_samples, replace=False)
    if keep_order:
        out.sort()
    return out.astype(np.int64)


class LOCOGlobalDataset(Dataset):
    """Global11-compatible Dataset using strict source-only LOCO normalization."""

    def __init__(
        self,
        root: Path,
        heldout_city: str,
        split: str,
        max_samples: int,
        sample_seed: int,
        tsm,
    ) -> None:
        self.root = root
        self.heldout_city = heldout_city
        self.split = split
        self.tsm = tsm

        meta_path = root / "experiments/data_v2/indices/meta.json"
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.variables = [str(v) for v in self.meta["variables"]]
        self.original_cities = [str(c) for c in self.meta["cities"]]
        if heldout_city not in self.original_cities:
            raise ValueError(f"Unknown held-out city: {heldout_city}")
        self.heldout_city_id = self.original_cities.index(heldout_city)

        # Critical: one shared identity so no unseen city embedding is needed.
        self.cities = ["shared_single_id"]
        self.stations = self.meta["stations"]
        self.array_paths = [str(p) for p in self.meta["array_paths"]]
        self.o3_idx = int(self.meta["o3_idx"])

        fold = fold_root(root, heldout_city)
        manifest_path = fold / "fold_manifest.json"
        scaler_path = fold / "source_train_stats.json"
        index_path = fold / f"{split}.npz"
        for path in [manifest_path, scaler_path, index_path]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing LOCO fold artifact: {path}\n"
                    "Run build_loco_folds.py first."
                )

        self.fold_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scaler = json.loads(scaler_path.read_text(encoding="utf-8"))

        leakage_checks = {
            "target_city_in_train": self.fold_manifest.get("target_city_in_train") is False,
            "target_city_in_val": self.fold_manifest.get("target_city_in_val") is False,
            "target_city_stats_used": self.fold_manifest.get("target_city_stats_used") is False,
            "source_city_count": int(self.fold_manifest.get("source_city_count", -1)) == 21,
            "shared_identity": self.fold_manifest.get("city_identity_mode") == "shared_single_id",
            "scaler_excludes_target": scaler.get("heldout_city_used_in_scaler") is False,
            "heldout_not_source": heldout_city not in scaler.get("source_cities", []),
        }
        if not all(leakage_checks.values()):
            raise RuntimeError(
                "LOCO leakage check failed: " + json.dumps(leakage_checks, ensure_ascii=False)
            )

        mu, sd = [], []
        for variable in self.variables:
            stat = scaler["pooled_variable_stats"][variable]
            mu.append(float(stat["mean"]))
            value = float(stat["std"])
            sd.append(value if np.isfinite(value) and value > 0 else 1.0)
        self.source_mu = np.asarray(mu, dtype=np.float32)
        self.source_std = np.asarray(sd, dtype=np.float32)
        self.target_mu_value = np.float32(self.source_mu[self.o3_idx])
        self.target_std_value = np.float32(self.source_std[self.o3_idx])

        z = np.load(index_path, allow_pickle=False)
        array_id = z["array_id"].astype(np.int32)
        original_city_id = z["city_id"].astype(np.int16)
        var_idx = z["var_idx"].astype(np.int8)
        start_idx = z["start_idx"].astype(np.int32)
        canonical_position = z["canonical_position"].astype(np.int64)

        if not np.all(var_idx == self.o3_idx):
            raise RuntimeError(f"Fold {heldout_city}/{split} contains non-O3 target indices")
        if split == "test":
            if not np.all(original_city_id == self.heldout_city_id):
                raise RuntimeError("LOCO test fold contains source-city samples")
        else:
            if np.any(original_city_id == self.heldout_city_id):
                raise RuntimeError(f"LOCO {split} fold contains held-out city samples")

        sel = subset_indices(
            len(array_id), max_samples, sample_seed, keep_order=(split != "train")
        )
        self.array_id = array_id[sel]
        self.original_city_id = original_city_id[sel]
        self.start_idx = start_idx[sel]
        self.canonical_position = canonical_position[sel]
        self.cache = tsm.ArrayCache(root, self.array_paths, max_items=len(self.array_paths))

    def __len__(self) -> int:
        return len(self.array_id)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        aid = int(self.array_id[i])
        st = int(self.start_idx[i])
        orig_cid = int(self.original_city_id[i])
        values, time_ns = self.cache.get(aid)

        x_phys = values[st:st + L, :]
        y_phys = values[st + L:st + L + H, self.o3_idx]
        x = (x_phys - self.source_mu[None, :]) / self.source_std[None, :]
        y_std = (y_phys - self.target_mu_value) / self.target_std_value
        future_time = self.tsm.future_calendar_features(
            time_ns[st + L:st + L + H]
        )

        return {
            "x": torch.from_numpy(np.asarray(x, dtype=np.float32)),
            "future_time": torch.from_numpy(np.asarray(future_time, dtype=np.float32)),
            "y_std": torch.from_numpy(np.asarray(y_std, dtype=np.float32)),
            "y_phys": torch.from_numpy(np.asarray(y_phys, dtype=np.float32)),
            "city_id": torch.tensor(0, dtype=torch.long),
            "orig_city_id": torch.tensor(orig_cid, dtype=torch.long),
            "array_id": torch.tensor(aid, dtype=torch.long),
            "target_mu": torch.tensor(self.target_mu_value, dtype=torch.float32),
            "target_std": torch.tensor(self.target_std_value, dtype=torch.float32),
            "sample_index": torch.tensor(int(self.canonical_position[i]), dtype=torch.long),
            "start_idx": torch.tensor(st, dtype=torch.long),
            "forecast_start_time_ns": torch.tensor(
                int(time_ns[st + L]), dtype=torch.long
            ),
        }


class SumMetric:
    def __init__(self) -> None:
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        diff = np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)
        self.abs_sum += float(np.abs(diff).sum(dtype=np.float64))
        self.sq_sum += float(np.square(diff).sum(dtype=np.float64))
        self.count += int(diff.size)

    def result(self) -> dict[str, Any]:
        if self.count <= 0:
            return {"MAE": math.nan, "MSE": math.nan, "RMSE": math.nan, "count": 0}
        mse = self.sq_sum / self.count
        return {
            "MAE": self.abs_sum / self.count,
            "MSE": mse,
            "RMSE": math.sqrt(mse),
            "count": self.count,
        }


def run_epoch(
    model,
    loader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    mse_weight: float,
    grad_clip: float,
    log_every: int,
    collect_detail: bool,
    collect_arrays: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[np.ndarray]]]:
    training = optimizer is not None
    model.train(training)

    overall = SumMetric()
    by_station = defaultdict(SumMetric)
    h_abs = np.zeros(H, dtype=np.float64)
    h_sq = np.zeros(H, dtype=np.float64)
    h_n = np.zeros(H, dtype=np.int64)
    loss_sum = 0.0
    batches = 0

    arrays: dict[str, list[np.ndarray]] = {
        "true": [], "pred": [], "shared_city_id": [], "original_city_id": [],
        "array_id": [], "canonical_position": [], "start_idx": [],
        "forecast_start_time_ns": [],
    }

    for step, batch in enumerate(loader, 1):
        x = batch["x"].to(device, dtype=torch.float32, non_blocking=True)
        ft = batch["future_time"].to(device, dtype=torch.float32, non_blocking=True)
        ys = batch["y_std"].to(device, dtype=torch.float32, non_blocking=True)
        shared_ci = batch["city_id"].to(device, non_blocking=True)

        if shared_ci.numel() and torch.any(shared_ci != 0):
            raise RuntimeError("LOCO Global11 model received nonzero shared city_id")

        mu = batch["target_mu"].numpy()[:, None]
        sd = batch["target_std"].numpy()[:, None]
        y_phys = batch["y_phys"].numpy().astype(np.float32, copy=False)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            pred_s = model(x, ft, shared_ci)
            if pred_s.shape != ys.shape:
                raise RuntimeError(
                    f"Prediction/target mismatch: {tuple(pred_s.shape)} vs {tuple(ys.shape)}"
                )
            loss = F.l1_loss(pred_s, ys) + mse_weight * F.mse_loss(pred_s, ys)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            if training:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        loss_sum += float(loss.detach().cpu())
        batches += 1
        pred = (
            pred_s.detach().cpu().numpy().astype(np.float32, copy=False) * sd + mu
        ).astype(np.float32, copy=False)
        overall.update(pred, y_phys)

        if collect_detail:
            array_ids = batch["array_id"].numpy()
            for aid in np.unique(array_ids):
                mask = array_ids == aid
                by_station[int(aid)].update(pred[mask], y_phys[mask])
            diff = pred.astype(np.float64) - y_phys.astype(np.float64)
            h_abs += np.abs(diff).sum(axis=0)
            h_sq += np.square(diff).sum(axis=0)
            h_n += diff.shape[0]

        if collect_arrays:
            arrays["true"].append(y_phys)
            arrays["pred"].append(pred)
            arrays["shared_city_id"].append(batch["city_id"].numpy())
            arrays["original_city_id"].append(batch["orig_city_id"].numpy())
            arrays["array_id"].append(batch["array_id"].numpy())
            arrays["canonical_position"].append(batch["sample_index"].numpy())
            arrays["start_idx"].append(batch["start_idx"].numpy())
            arrays["forecast_start_time_ns"].append(batch["forecast_start_time_ns"].numpy())

        if training and (step == 1 or step % log_every == 0):
            r = overall.result()
            print(
                f"step={step}/{len(loader)} loss={loss_sum/max(batches,1):.6f} "
                f"MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f}",
                flush=True,
            )

    return (
        {"loss": loss_sum / max(batches, 1), **overall.result()},
        {"by_station": by_station, "h_abs": h_abs, "h_sq": h_sq, "h_n": h_n},
        arrays,
    )


def merge_and_save_arrays(
    out_dir: Path,
    arrays: dict[str, list[np.ndarray]],
    root: Path,
    heldout_city: str,
    full_test: bool,
) -> dict[str, Any]:
    merged: dict[str, np.ndarray] = {}
    for key, chunks in arrays.items():
        if not chunks:
            raise RuntimeError(f"No test array chunks for {key}")
        merged[key] = np.concatenate(chunks, axis=0)

    order = np.argsort(merged["canonical_position"], kind="stable")
    for key in merged:
        merged[key] = merged[key][order]

    if len(np.unique(merged["canonical_position"])) != len(merged["canonical_position"]):
        raise RuntimeError("Duplicate canonical test positions")
    if not np.all(merged["shared_city_id"] == 0):
        raise RuntimeError("Saved LOCO shared_city_id is not all zero")

    file_map = {
        "true": "true.npy",
        "pred": "pred.npy",
        "shared_city_id": "shared_city_id.npy",
        "original_city_id": "original_city_id.npy",
        "array_id": "array_id.npy",
        "canonical_position": "canonical_test_positions.npy",
        "start_idx": "start_idx.npy",
        "forecast_start_time_ns": "forecast_start_time_ns.npy",
    }
    info = {}
    for key, filename in file_map.items():
        arr = np.asarray(merged[key])
        np.save(out_dir / filename, arr, allow_pickle=False)
        info[filename] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}

    audit: dict[str, Any] = {
        "full_test": bool(full_test),
        "shape_ok": merged["pred"].shape == merged["true"].shape
        and merged["pred"].ndim == 2 and merged["pred"].shape[1] == H,
        "nonfinite_pred": int((~np.isfinite(merged["pred"])).sum()),
        "nonfinite_true": int((~np.isfinite(merged["true"])).sum()),
        "shared_city_id_all_zero": bool(np.all(merged["shared_city_id"] == 0)),
    }

    if full_test:
        fold = fold_root(root, heldout_city)
        canonical_true = np.load(fold / "canonical_test_true.npy", allow_pickle=False)
        canonical_pos = np.load(fold / "canonical_test_positions.npy", allow_pickle=False)
        audit["true_canonical_equal"] = bool(
            np.array_equal(merged["true"], canonical_true, equal_nan=True)
        )
        audit["positions_canonical_equal"] = bool(
            np.array_equal(merged["canonical_position"], canonical_pos)
        )
        if not audit["true_canonical_equal"]:
            raise RuntimeError("Saved true.npy differs from canonical LOCO held-out targets")
        if not audit["positions_canonical_equal"]:
            raise RuntimeError("Saved canonical test positions differ from LOCO fold")
    else:
        audit["true_canonical_equal"] = None
        audit["positions_canonical_equal"] = None

    if not audit["shape_ok"] or audit["nonfinite_pred"] or audit["nonfinite_true"]:
        raise RuntimeError(f"Prediction array audit failed: {audit}")

    return {"arrays": info, "audit": audit, "merged": merged}


def save_detail_metrics(
    out_dir: Path,
    detail: dict[str, Any],
    meta: dict[str, Any],
    heldout_city: str,
) -> dict[str, Any]:
    station_rows = []
    for aid, meter in sorted(detail["by_station"].items()):
        sm = meta["stations"][int(aid)]
        station_rows.append({
            "array_id": int(aid),
            "city": sm["city"],
            "station_id": sm["station_id"],
            **meter.result(),
        })
    station_df = pd.DataFrame(station_rows)
    station_df.to_csv(out_dir / "metrics_by_station.csv", index=False, encoding="utf-8-sig")

    h_rows = []
    for h in range(H):
        n = int(detail["h_n"][h])
        mse = detail["h_sq"][h] / n if n else math.nan
        h_rows.append({
            "horizon": h + 1,
            "MAE": detail["h_abs"][h] / n if n else math.nan,
            "MSE": mse,
            "RMSE": math.sqrt(mse) if n else math.nan,
            "count": n,
        })
    pd.DataFrame(h_rows).to_csv(
        out_dir / "metrics_by_horizon.csv", index=False, encoding="utf-8-sig"
    )

    return {
        "heldout_city": heldout_city,
        "station_count": int(len(station_df)),
        "station_MAE_macro": float(station_df["MAE"].mean()) if len(station_df) else math.nan,
        "station_RMSE_macro": float(station_df["RMSE"].mean()) if len(station_df) else math.nan,
    }


def main_impl(cli: argparse.Namespace) -> None:
    root = cli.root.expanduser().resolve()
    out_dir = cli.out_dir if cli.out_dir.is_absolute() else root / cli.out_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "run_status.json"

    if status_path.exists() and not cli.force:
        try:
            s = json.loads(status_path.read_text(encoding="utf-8"))
            if (
                s.get("state") == "complete"
                and (out_dir / "pred.npy").exists()
                and (out_dir / "true.npy").exists()
            ):
                print(f"[SKIP] complete: {out_dir}")
                return
        except Exception:
            pass

    scoped, modules = add_train_modules(root)
    tsm, _, _, _ = modules
    tsm.seed_everything(cli.seed)

    cuda = torch.cuda.is_available()
    if cli.require_cuda and not cuda:
        raise RuntimeError("CUDA required but unavailable")
    device = torch.device("cuda:0" if cuda else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if cuda else "CPU"

    ref = scoped.read_reference_args(root, cli.model_name, cli.reference_args)
    args = scoped.namespace_from_reference(ref, cli, root, cli.model_name)
    if str(args.variant) != "city_time":
        raise RuntimeError(
            f"Expected Global11 selected variant=city_time, got {args.variant!r}"
        )

    train_ds = LOCOGlobalDataset(
        root, cli.heldout_city, "train", cli.max_train_samples, cli.seed, tsm
    )
    val_ds = LOCOGlobalDataset(
        root, cli.heldout_city, "val", cli.max_val_samples, cli.seed + 1, tsm
    )
    test_ds = LOCOGlobalDataset(
        root, cli.heldout_city, "test", cli.max_test_samples, cli.seed + 2, tsm
    )

    train_loader = tsm.make_loader(train_ds, args.batch_size, True, args.num_workers, cuda)
    val_loader = tsm.make_loader(val_ds, args.batch_size, False, args.num_workers, cuda)
    test_loader = tsm.make_loader(test_ds, args.batch_size, False, args.num_workers, cuda)

    # Koopa's train-spectrum mask depends only on the held-out fold, not on seed.
    # Reuse one fold-specific cache across the five seeds to avoid recomputing the
    # same FFT spectrum five times. train_supervised_scopes.build_model()
    # intentionally uses out_dir/koopa_mask_spectrum_smoke.npy for non-global data.
    koopa_shared_cache = (
        root / "experiments/results/16_loco_global11_v2/00_koopa_masks"
        / f"{safe_name(cli.heldout_city)}.npy"
    )
    koopa_local_cache = out_dir / "koopa_mask_spectrum_smoke.npy"
    if cli.model_name == "Koopa" and koopa_shared_cache.exists() and not koopa_local_cache.exists():
        koopa_local_cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(koopa_shared_cache, koopa_local_cache)
        except OSError:
            import shutil
            shutil.copy2(koopa_shared_cache, koopa_local_cache)
        print(f"Reused fold-specific Koopa spectrum: {koopa_shared_cache}", flush=True)

    model = scoped.build_model(cli.model_name, args, train_ds, device, out_dir, modules)

    if cli.model_name == "Koopa" and koopa_local_cache.exists() and not koopa_shared_cache.exists():
        koopa_shared_cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = koopa_shared_cache.with_suffix(".tmp.npy")
        np.save(tmp, np.load(koopa_local_cache, allow_pickle=False), allow_pickle=False)
        os.replace(tmp, koopa_shared_cache)
        print(f"Saved fold-specific Koopa spectrum: {koopa_shared_cache}", flush=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    formal_full_fold = (
        cli.max_train_samples <= 0
        and cli.max_val_samples <= 0
        and cli.max_test_samples <= 0
    )

    run_meta = {
        "protocol": "revision-v2 strict LOCO Global11 supervised baseline",
        "heldout_city": cli.heldout_city,
        "model_name": cli.model_name,
        "seed": cli.seed,
        "reference_args_path": ref["reference_args_path"],
        "hyperparameter_policy": "exactly reuse completed Global11 final-run hyperparameters; no LOCO retuning",
        "variant": args.variant,
        "city_identity_mode": "shared_single_id",
        "normalization": "source-21 pooled train-only",
        "target_city_in_train": False,
        "target_city_in_val": False,
        "target_city_stats_used": False,
        "source_city_count": 21,
        "formal_full_fold": formal_full_fold,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "mse_weight": float(args.mse_weight),
        "grad_clip": float(args.grad_clip),
        "device": str(device),
        "gpu_name": gpu_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "variables": train_ds.variables,
        "o3_idx": train_ds.o3_idx,
        "selected_args": {
            k: v for k, v in vars(args).items()
            if isinstance(v, (str, int, float, bool, type(None)))
        },
    }
    (out_dir / "args.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    safe_status(status_path, {
        "state": "running",
        "started_at": pd.Timestamp.now().isoformat(),
        "heldout_city": cli.heldout_city,
        "model_name": cli.model_name,
        "seed": cli.seed,
        "formal_full_fold": formal_full_fold,
    })

    print("=" * 88)
    print("[GLOBAL11 SUPERVISED — STRICT LOCO REVISION-V2]")
    print(f"heldout_city={cli.heldout_city} model={cli.model_name} seed={cli.seed}")
    print(f"device={device} ({gpu_name})")
    print(f"train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}")
    print(
        f"epochs={args.epochs} patience={args.patience} batch={args.batch_size} "
        f"workers={args.num_workers} lr={args.lr}"
    )
    print("city_identity=shared_single_id normalization=source-21 pooled train-only")
    print(f"params total={total_params:,} trainable={trainable_params:,}")
    print("=" * 88, flush=True)

    # Keep optimizer semantics matched to the original Global11 baselines.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_path = out_dir / "best.pt"
    history = []
    best_rmse = float("inf")
    best_epoch = -1
    bad_epochs = 0
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} train", flush=True)
        tr, _, _ = run_epoch(
            model, train_loader, device, optimizer, args.mse_weight,
            args.grad_clip, args.log_every, False, False,
        )
        print(f"Epoch {epoch}/{args.epochs} val", flush=True)
        va, _, _ = run_epoch(
            model, val_loader, device, None, args.mse_weight,
            args.grad_clip, args.log_every, False, False,
        )
        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in va.items()},
        })
        pd.DataFrame(history).to_csv(
            out_dir / "history.csv", index=False, encoding="utf-8-sig"
        )
        print(f"epoch={epoch} val_MAE={va['MAE']:.6f} val_RMSE={va['RMSE']:.6f}")
        if va["RMSE"] < best_rmse - 1e-9:
            best_rmse = va["RMSE"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "best_val_RMSE": best_rmse},
                best_path,
            )
        else:
            bad_epochs += 1
            print(f"No improvement: {bad_epochs}/{args.patience}", flush=True)
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}")
                break

    train_seconds = time.perf_counter() - t0
    if not best_path.exists():
        raise RuntimeError("best.pt was not created")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    print("\nFinal held-out-city test evaluation", flush=True)
    test_t0 = time.perf_counter()
    te, detail, arrays = run_epoch(
        model, test_loader, device, None, args.mse_weight,
        args.grad_clip, args.log_every, True, True,
    )
    test_seconds = time.perf_counter() - test_t0

    saved = merge_and_save_arrays(
        out_dir, arrays, root, cli.heldout_city, full_test=formal_full_fold
    )
    detail_summary = save_detail_metrics(
        out_dir, detail, train_ds.meta, cli.heldout_city
    )

    diff = (
        saved["merged"]["pred"].astype(np.float64)
        - saved["merged"]["true"].astype(np.float64)
    )
    array_mae = float(np.abs(diff).mean())
    array_rmse = float(np.sqrt(np.square(diff).mean()))
    metric_match = abs(array_mae - te["MAE"]) <= 5e-5 and abs(array_rmse - te["RMSE"]) <= 5e-5
    if not metric_match:
        raise RuntimeError(
            f"Saved-array metrics mismatch: eval=({te['MAE']},{te['RMSE']}) "
            f"array=({array_mae},{array_rmse})"
        )

    row = {
        **te,
        **detail_summary,
        "heldout_city": cli.heldout_city,
        "model_name": cli.model_name,
        "seed": cli.seed,
        "best_epoch": best_epoch,
        "best_val_RMSE": best_rmse,
        "train_seconds": train_seconds,
        "test_seconds": test_seconds,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / 1024**2 if cuda else math.nan
        ),
        "gpu_name": gpu_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "formal_full_fold": formal_full_fold,
        "city_identity_mode": "shared_single_id",
        "normalization": "source-21 pooled train-only",
        "target_city_stats_used": False,
        "array_MAE": array_mae,
        "array_RMSE": array_rmse,
        "metric_match": metric_match,
        "abs_error_sum": float(np.abs(diff).sum(dtype=np.float64)),
        "sq_error_sum": float(np.square(diff).sum(dtype=np.float64)),
        "scalar_count": int(diff.size),
    }
    pd.DataFrame([row]).to_csv(
        out_dir / "metrics_overall.csv", index=False, encoding="utf-8-sig"
    )

    manifest = {
        **run_meta,
        "metrics": row,
        "arrays": saved["arrays"],
        "array_audit": saved["audit"],
        "completed_at": pd.Timestamp.now().isoformat(),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    if not cli.keep_checkpoint:
        best_path.unlink(missing_ok=True)

    safe_status(status_path, {
        "state": "complete",
        "completed_at": pd.Timestamp.now().isoformat(),
        "heldout_city": cli.heldout_city,
        "model_name": cli.model_name,
        "seed": cli.seed,
        "formal_full_fold": formal_full_fold,
        "MAE": te["MAE"],
        "RMSE": te["RMSE"],
        "n_test": len(test_ds),
        "city_identity_mode": "shared_single_id",
        "normalization": "source-21 pooled train-only",
        "target_city_stats_used": False,
    })

    print(pd.DataFrame([row]).to_string(index=False))

    del model, optimizer, train_loader, val_loader, test_loader
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()


def main() -> None:
    cli = parse_args()
    root = cli.root.expanduser().resolve()
    out_dir = cli.out_dir if cli.out_dir.is_absolute() else root / cli.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "run_status.json"
    try:
        main_impl(cli)
    except Exception as exc:
        safe_status(status_path, {
            "state": "failed",
            "failed_at": pd.Timestamp.now().isoformat(),
            "heldout_city": cli.heldout_city,
            "model_name": cli.model_name,
            "seed": cli.seed,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
