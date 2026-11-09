#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global supervised baselines missing from the original 11-model set.

Models
------
CNNLSTM, Transformer, TimeXer, Koopa

Protocol
--------
- Reuse revision-v2 GlobalV2Dataset exactly.
- Pool all training windows from all 22 cities/stations into one global model.
- Input length 96, horizon 48, eight variables.
- City identity and future calendar information use the same wrapper as the
  existing revision-v2 global baselines.
- Hyperparameters are selected on validation only.
- Final test runs save per-sample arrays and metadata.

This script intentionally imports the existing data/metric utilities from:
experiments/scripts/train/train_global_tsmixer.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_global_tsmixer import (
    H,
    L,
    GlobalV2Dataset,
    SumMetric,
    make_loader,
    save_details,
    seed_everything,
)

MODEL_CHOICES = ["CNNLSTM", "Transformer", "TimeXer", "Koopa"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    p.add_argument("--tslib_root", type=Path, default=None)
    p.add_argument("--model_name", choices=MODEL_CHOICES, required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--variant", choices=["plain", "city_time"], default="city_time")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--mse_weight", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Shared model controls
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--d_ff", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--e_layers", type=int, default=2)
    p.add_argument("--d_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--factor", type=int, default=1)
    p.add_argument("--activation", type=str, default="gelu")
    p.add_argument("--embed", type=str, default="timeF")
    p.add_argument("--freq", type=str, default="h")
    p.add_argument("--use_norm", type=int, choices=[0, 1], default=1)
    p.add_argument("--label_len", type=int, default=48)

    # CNN-LSTM
    p.add_argument("--conv_channel", type=int, default=64)

    # TimeXer
    p.add_argument("--patch_len", type=int, default=16)

    # Koopa
    p.add_argument("--koopa_dynamic_dim", type=int, default=32)
    p.add_argument("--koopa_hidden_dim", type=int, default=64)
    p.add_argument("--koopa_hidden_layers", type=int, default=2)
    p.add_argument("--koopa_num_blocks", type=int, default=2)
    p.add_argument("--koopa_multistep", type=int, choices=[0, 1], default=0)
    p.add_argument("--koopa_alpha", type=float, default=0.2)
    p.add_argument(
        "--koopa_spectrum_batch_size",
        type=int,
        default=512,
        help="Batch size used once to derive Koopa train-only frequency mask.",
    )

    # Information-matched wrapper
    p.add_argument("--city_emb_dim", type=int, default=16)
    p.add_argument("--condition_hidden", type=int, default=64)

    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--max_test_samples", type=int, default=0)
    p.add_argument(
        "--skip_test",
        action="store_true",
        help="Validation-only tuning. Test split is not constructed or read.",
    )
    p.add_argument("--require_cuda", action="store_true")
    p.add_argument("--save_arrays", type=int, choices=[0, 1], default=1)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--log_every", type=int, default=1000)
    return p.parse_args()


class GlobalV2DatasetIndexed(GlobalV2Dataset):
    """GlobalV2Dataset plus stable sample metadata for prediction archiving."""

    def __getitem__(self, i: int) -> dict:
        item = super().__getitem__(i)
        aid = int(self.array_id[i])
        st = int(self.start_idx[i])
        _, time_ns = self.cache.get(aid)
        forecast_start_ns = int(time_ns[st + L])
        item.update(
            {
                "sample_index": torch.tensor(i, dtype=torch.long),
                "start_idx": torch.tensor(st, dtype=torch.long),
                "forecast_start_time_ns": torch.tensor(
                    forecast_start_ns, dtype=torch.long
                ),
            }
        )
        return item


def import_tslib_module(tslib_root: Path, model_name: str):
    model_file = tslib_root / "models" / f"{model_name}.py"
    if not model_file.exists():
        raise FileNotFoundError(f"TSLib model file not found: {model_file}")

    root_str = str(tslib_root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)

    # Avoid silently reusing a module imported from another repository path.
    module_name = f"models.{model_name}"
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def patch_timexer_forecast_multi_out_of_place(backbone: nn.Module) -> None:
    """Avoid in-place normalization so gradients reach the city input branch."""

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            centered = x_enc - means
            stdev = torch.sqrt(
                torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_norm = centered / stdev
        else:
            x_norm = x_enc
            means = None
            stdev = None

        en_embed, n_vars = self.en_embedding(x_norm.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_norm, x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out,
            (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]),
        )
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            assert means is not None and stdev is not None
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1)

        return dec_out

    backbone.forecast_multi = MethodType(forecast_multi, backbone)


def build_cfg(args: argparse.Namespace, n_vars: int) -> SimpleNamespace:
    """Superset of attributes consumed by the four server-side model files."""
    return SimpleNamespace(
        task_name="long_term_forecast",
        is_training=1,
        model_id=f"global_{args.model_name}_revision_v2",
        model=args.model_name,
        data="revision_v2_global",
        features="M",
        target="O3",
        seq_len=L,
        label_len=args.label_len,
        pred_len=H,
        enc_in=n_vars,
        dec_in=n_vars,
        c_out=n_vars,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        dropout=args.dropout,
        factor=args.factor,
        activation=args.activation,
        embed=args.embed,
        freq=args.freq,
        output_attention=False,
        use_norm=args.use_norm,
        conv_channel=args.conv_channel,
        patch_len=args.patch_len,
        stride=max(1, args.patch_len // 2),
        num_class=2,
        individual=False,
        distil=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


def koopa_mask_cache_path(root: Path, alpha: float) -> Path:
    alpha_tag = str(alpha).replace(".", "p")
    return (
        root
        / "experiments"
        / "data_v2"
        / f"koopa_mask_spectrum_train_all_alpha{alpha_tag}.npy"
    )


@torch.no_grad()
def compute_koopa_mask_spectrum(
    train_ds: GlobalV2DatasetIndexed,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> np.ndarray:
    """
    Compute Koopa's shared frequency mask from training windows only.

    Full-data runs share a cached mask. Subsampled smoke runs use a local mask
    so they can never overwrite the full training mask.
    """
    full_training = args.max_train_samples <= 0
    cache_path = (
        koopa_mask_cache_path(args.root.resolve(), args.koopa_alpha)
        if full_training
        else out_dir / "koopa_mask_spectrum_smoke.npy"
    )

    if cache_path.exists():
        mask = np.load(cache_path, allow_pickle=False).astype(np.int64)
        print(f"Loaded Koopa train-only mask: {cache_path} -> {mask.tolist()}")
        return mask

    loader = make_loader(
        train_ds,
        batch_size=max(1, args.koopa_spectrum_batch_size),
        shuffle=False,
        workers=args.num_workers,
        cuda=device.type == "cuda",
    )
    amplitude_sum: Optional[torch.Tensor] = None
    denominator = 0

    print(
        f"Computing Koopa spectrum from training split only "
        f"(n={len(train_ds):,})...",
        flush=True,
    )
    for step, batch in enumerate(loader, 1):
        x = batch["x"].to(device, dtype=torch.float32, non_blocking=True)
        amp = torch.abs(torch.fft.rfft(x, dim=1))
        # Sum across samples and variables, leaving frequency dimension.
        cur = amp.sum(dim=(0, 2)).detach().cpu().double()
        amplitude_sum = cur if amplitude_sum is None else amplitude_sum + cur
        denominator += int(x.shape[0] * x.shape[2])
        if step == 1 or step % 500 == 0:
            print(f"Koopa spectrum step={step}/{len(loader)}", flush=True)

    if amplitude_sum is None or denominator <= 0:
        raise RuntimeError("Failed to compute Koopa training spectrum.")

    mean_amp = amplitude_sum / denominator
    k = max(1, int(mean_amp.numel() * args.koopa_alpha))
    mask = torch.topk(mean_amp, k=k, largest=True).indices.numpy().astype(np.int64)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, mask)
    os.replace(tmp, cache_path)
    print(f"Saved Koopa train-only mask: {cache_path} -> {mask.tolist()}")
    return mask


def build_backbone(
    args: argparse.Namespace,
    tslib_root: Path,
    n_vars: int,
    train_ds: GlobalV2DatasetIndexed,
    device: torch.device,
    out_dir: Path,
) -> nn.Module:
    cfg = build_cfg(args, n_vars)
    mod = import_tslib_module(tslib_root, args.model_name)

    if args.model_name == "Koopa":
        mask_np = compute_koopa_mask_spectrum(train_ds, args, device, out_dir)
        mask_cpu = torch.as_tensor(mask_np, dtype=torch.long)

        original = mod.Model._get_mask_spectrum

        def _use_precomputed_mask(self, configs):
            return mask_cpu.clone()

        mod.Model._get_mask_spectrum = _use_precomputed_mask
        try:
            backbone = mod.Model(
                cfg,
                dynamic_dim=args.koopa_dynamic_dim,
                hidden_dim=args.koopa_hidden_dim,
                hidden_layers=args.koopa_hidden_layers,
                num_blocks=args.koopa_num_blocks,
                multistep=bool(args.koopa_multistep),
            )
        finally:
            mod.Model._get_mask_spectrum = original

        # The upstream FourierFilter stores the mask as a plain tensor.
        # Register it as a buffer so model.to(device) moves it reliably.
        mask = backbone.disentanglement.mask_spectrum.detach().long()
        delattr(backbone.disentanglement, "mask_spectrum")
        backbone.disentanglement.register_buffer("mask_spectrum", mask)
        return backbone

    backbone = mod.Model(cfg)
    if args.model_name == "TimeXer":
        patch_timexer_forecast_multi_out_of_place(backbone)
        print(
            "Applied out-of-place TimeXer normalization patch for "
            "differentiable city input.",
            flush=True,
        )
    return backbone


class InformationMatchedBaseline(nn.Module):
    """Same city identity and future-calendar conditioning as existing baselines."""

    def __init__(
        self,
        backbone: nn.Module,
        model_name: str,
        n_cities: int,
        n_vars: int,
        o3_idx: int,
        variant: str,
        city_emb_dim: int,
        condition_hidden: int,
        label_len: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.model_name = model_name
        self.n_vars = n_vars
        self.o3_idx = o3_idx
        self.variant = variant
        self.label_len = label_len

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

    def _backbone_forward(self, x_in: torch.Tensor) -> torch.Tensor:
        if self.model_name == "Transformer":
            label_len = min(self.label_len, x_in.shape[1])
            dec_hist = x_in[:, -label_len:, :] if label_len > 0 else x_in[:, :0, :]
            dec_zeros = torch.zeros(
                x_in.shape[0],
                H,
                self.n_vars,
                device=x_in.device,
                dtype=x_in.dtype,
            )
            x_dec = torch.cat([dec_hist, dec_zeros], dim=1)
            return self.backbone(x_in, None, x_dec, None)

        return self.backbone(x_in, None, None, None)

    def forward(
        self,
        x: torch.Tensor,
        future_time: torch.Tensor,
        city_id: torch.Tensor,
    ) -> torch.Tensor:
        x_in = x
        city_vec: Optional[torch.Tensor] = None
        if self.variant == "city_time":
            city_vec = self.city_emb(city_id)
            x_in = x + self.city_to_input(city_vec).unsqueeze(1)

        out = self._backbone_forward(x_in)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim != 3:
            raise RuntimeError(
                f"Unexpected {self.model_name} output shape: {tuple(out.shape)}"
            )
        if out.shape[1] < H:
            raise RuntimeError(
                f"{self.model_name} returned {out.shape[1]} steps; expected >= {H}"
            )
        if out.shape[2] <= self.o3_idx:
            raise RuntimeError(
                f"{self.model_name} returned {out.shape[2]} variables; "
                f"O3 index is {self.o3_idx}"
            )

        base = out[:, -H:, self.o3_idx]
        if self.variant == "plain":
            return base

        assert city_vec is not None
        cv = city_vec.unsqueeze(1).expand(-1, H, -1)
        cond = torch.cat([base.unsqueeze(-1), future_time, cv], dim=-1)
        correction = self.condition_head(cond).squeeze(-1)
        return base + self.condition_gate * correction


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    mse_weight: float,
    grad_clip: float,
    log_every: int,
    collect_detail: bool,
    collect_arrays: bool,
) -> tuple[dict, dict, dict]:
    training = optimizer is not None
    model.train(training)

    overall = SumMetric()
    by_city = defaultdict(SumMetric)
    by_station = defaultdict(SumMetric)
    h_abs = np.zeros(H, dtype=np.float64)
    h_sq = np.zeros(H, dtype=np.float64)
    h_n = np.zeros(H, dtype=np.int64)
    loss_sum = 0.0
    batches = 0

    arrays: dict[str, list[np.ndarray]] = {
        "true": [],
        "pred": [],
        "city_id": [],
        "array_id": [],
        "sample_index": [],
        "start_idx": [],
        "forecast_start_time_ns": [],
    }

    for step, batch in enumerate(loader, 1):
        x = batch["x"].to(device, dtype=torch.float32, non_blocking=True)
        ft = batch["future_time"].to(device, dtype=torch.float32, non_blocking=True)
        ys = batch["y_std"].to(device, dtype=torch.float32, non_blocking=True)
        ci = batch["city_id"].to(device, non_blocking=True)

        mu = batch["target_mu"].numpy()[:, None]
        sd = batch["target_std"].numpy()[:, None]
        y_phys = batch["y_phys"].numpy().astype(np.float32, copy=False)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            pred_s = model(x, ft, ci)
            if pred_s.shape != ys.shape:
                raise RuntimeError(
                    f"Prediction/target mismatch: {tuple(pred_s.shape)} vs "
                    f"{tuple(ys.shape)}"
                )
            loss = F.l1_loss(pred_s, ys) + mse_weight * F.mse_loss(pred_s, ys)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at step {step}: {float(loss.detach().cpu())}"
                )
            if training:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        loss_sum += float(loss.detach().cpu())
        batches += 1

        pred = (
            pred_s.detach().cpu().numpy().astype(np.float32, copy=False) * sd
            + mu
        ).astype(np.float32, copy=False)
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

            diff = pred.astype(np.float64) - y_phys.astype(np.float64)
            h_abs += np.abs(diff).sum(axis=0)
            h_sq += np.square(diff).sum(axis=0)
            h_n += diff.shape[0]

        if collect_arrays:
            arrays["true"].append(np.asarray(y_phys, dtype=np.float32))
            arrays["pred"].append(np.asarray(pred, dtype=np.float32))
            for key in [
                "city_id",
                "array_id",
                "sample_index",
                "start_idx",
                "forecast_start_time_ns",
            ]:
                arrays[key].append(batch[key].numpy())

        if training and (step == 1 or step % log_every == 0):
            r = overall.result()
            print(
                f"step={step}/{len(loader)} loss={loss_sum / batches:.6f} "
                f"MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f}",
                flush=True,
            )

    detail = {
        "by_city": by_city,
        "by_station": by_station,
        "h_abs": h_abs,
        "h_sq": h_sq,
        "h_n": h_n,
    }
    metrics = {"loss": loss_sum / max(batches, 1), **overall.result()}
    return metrics, detail, arrays


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_prediction_arrays(
    out_dir: Path,
    arrays: dict[str, list[np.ndarray]],
    dataset: GlobalV2DatasetIndexed,
    reported_metrics: dict,
) -> dict:
    merged: dict[str, np.ndarray] = {}
    for key, chunks in arrays.items():
        if not chunks:
            raise RuntimeError(f"No chunks collected for {key}")
        merged[key] = np.concatenate(chunks, axis=0)

    n = len(dataset)
    if merged["true"].shape != (n, H):
        raise RuntimeError(
            f"Unexpected true shape: {merged['true'].shape}; expected {(n, H)}"
        )
    if merged["pred"].shape != (n, H):
        raise RuntimeError(
            f"Unexpected pred shape: {merged['pred'].shape}; expected {(n, H)}"
        )
    if not np.array_equal(merged["sample_index"], np.arange(n, dtype=np.int64)):
        raise RuntimeError("Test sample order is not the canonical 0..N-1 order.")

    file_map = {
        "true": "true.npy",
        "pred": "pred.npy",
        "city_id": "city_id.npy",
        "array_id": "array_id.npy",
        "sample_index": "sample_index.npy",
        "start_idx": "start_idx.npy",
        "forecast_start_time_ns": "forecast_start_time_ns.npy",
    }
    for key, filename in file_map.items():
        np.save(out_dir / filename, merged[key], allow_pickle=False)

    city_ids = merged["city_id"].astype(np.int64)
    array_ids = merged["array_id"].astype(np.int64)
    station_ids = [dataset.stations[int(a)]["station_id"] for a in array_ids]
    city_names = [dataset.cities[int(c)] for c in city_ids]
    times = pd.to_datetime(
        merged["forecast_start_time_ns"].astype(np.int64), utc=True
    )
    metadata = pd.DataFrame(
        {
            "sample_index": merged["sample_index"].astype(np.int64),
            "array_id": array_ids,
            "city_id": city_ids,
            "city": city_names,
            "station_id": station_ids,
            "start_idx": merged["start_idx"].astype(np.int64),
            "forecast_start_time_ns": merged[
                "forecast_start_time_ns"
            ].astype(np.int64),
            "forecast_start_time_utc": times.astype(str),
        }
    )
    metadata.to_csv(
        out_dir / "sample_metadata.csv", index=False, encoding="utf-8-sig"
    )

    diff = merged["pred"].astype(np.float64) - merged["true"].astype(np.float64)
    mae = float(np.abs(diff).mean())
    mse = float(np.square(diff).mean())
    rmse = float(math.sqrt(mse))

    tol = 1e-5
    validation = {
        "n_samples": n,
        "horizon": H,
        "true_shape": list(merged["true"].shape),
        "pred_shape": list(merged["pred"].shape),
        "dtype": str(merged["pred"].dtype),
        "recomputed_MAE": mae,
        "recomputed_MSE": mse,
        "recomputed_RMSE": rmse,
        "reported_MAE": float(reported_metrics["MAE"]),
        "reported_MSE": float(reported_metrics["MSE"]),
        "reported_RMSE": float(reported_metrics["RMSE"]),
        "mae_abs_difference": abs(mae - float(reported_metrics["MAE"])),
        "rmse_abs_difference": abs(rmse - float(reported_metrics["RMSE"])),
        "metric_match_tolerance": tol,
        "metric_match": (
            abs(mae - float(reported_metrics["MAE"])) <= tol
            and abs(rmse - float(reported_metrics["RMSE"])) <= tol
        ),
        "files": {},
    }

    for filename in [*file_map.values(), "sample_metadata.csv"]:
        path = out_dir / filename
        validation["files"][filename] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    (out_dir / "prediction_manifest.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not validation["metric_match"]:
        raise RuntimeError(
            "Saved-array metrics do not match evaluation metrics: "
            f"MAE diff={validation['mae_abs_difference']}, "
            f"RMSE diff={validation['rmse_abs_difference']}"
        )
    return validation


def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_out_dir(root: Path, out_dir: Path) -> Path:
    return out_dir.resolve() if out_dir.is_absolute() else (root / out_dir).resolve()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    args.root = root
    tslib_root = (
        args.tslib_root or root / "Time-Series-Library-main"
    ).expanduser().resolve()
    out_dir = resolve_out_dir(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    cuda = torch.cuda.is_available()
    if args.require_cuda and not cuda:
        raise RuntimeError("CUDA is required but is not available.")
    device = torch.device("cuda:0" if cuda else "cpu")

    train_ds = GlobalV2DatasetIndexed(
        root, "train", args.max_train_samples, args.seed
    )
    val_ds = GlobalV2DatasetIndexed(
        root, "val", args.max_val_samples, args.seed + 1
    )
    train_loader = make_loader(
        train_ds, args.batch_size, True, args.num_workers, cuda
    )
    val_loader = make_loader(
        val_ds, args.batch_size, False, args.num_workers, cuda
    )

    # Strict tuning: never construct/read test data when --skip_test is set.
    test_ds = None
    test_loader = None
    if not args.skip_test:
        test_ds = GlobalV2DatasetIndexed(
            root, "test", args.max_test_samples, args.seed + 2
        )
        test_loader = make_loader(
            test_ds, args.batch_size, False, args.num_workers, cuda
        )

    backbone = build_backbone(
        args,
        tslib_root,
        len(train_ds.variables),
        train_ds,
        device,
        out_dir,
    )
    model = InformationMatchedBaseline(
        backbone=backbone,
        model_name=args.model_name,
        n_cities=len(train_ds.cities),
        n_vars=len(train_ds.variables),
        o3_idx=train_ds.o3_idx,
        variant=args.variant,
        city_emb_dim=args.city_emb_dim,
        condition_hidden=args.condition_hidden,
        label_len=args.label_len,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gpu_name = torch.cuda.get_device_name(0) if cuda else "CPU"

    print("=" * 76)
    print("[GLOBAL MISSING-4 BASELINE — REVISION-V2]")
    print(
        f"model={args.model_name} variant={args.variant} seed={args.seed}"
    )
    print(f"device={device} ({gpu_name})")
    print(
        f"train={len(train_ds):,} val={len(val_ds):,} "
        f"test={'not loaded' if test_ds is None else f'{len(test_ds):,}'}"
    )
    print(
        f"batch={args.batch_size} workers={args.num_workers} lr={args.lr}"
    )
    print(f"params total={total_params:,}, trainable={trainable_params:,}")
    print("=" * 76, flush=True)

    run_meta = vars(args).copy()
    run_meta.update(
        {
            "root": str(root),
            "tslib_root": str(tslib_root),
            "out_dir": str(out_dir),
            "device": str(device),
            "gpu_name": gpu_name,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "variables": train_ds.variables,
            "cities": train_ds.cities,
            "o3_idx": train_ds.o3_idx,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": None if test_ds is None else len(test_ds),
            "protocol": (
                "revision-v2 global pooled training over all cities/stations"
            ),
            "metric_scale": (
                "physical O3 after city-level train-only inverse standardization"
            ),
            "loss": f"L1 + {args.mse_weight} * MSE on standardized O3",
            "test_split_constructed": test_ds is not None,
        }
    )
    (out_dir / "args.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    history = []
    best_rmse = float("inf")
    best_epoch = -1
    bad_epochs = 0

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    train_t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} train", flush=True)
        tr, _, _ = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            args.mse_weight,
            args.grad_clip,
            args.log_every,
            collect_detail=False,
            collect_arrays=False,
        )
        print(f"Epoch {epoch}/{args.epochs} val", flush=True)
        va, _, _ = run_epoch(
            model,
            val_loader,
            device,
            None,
            args.mse_weight,
            args.grad_clip,
            args.log_every,
            collect_detail=False,
            collect_arrays=False,
        )
        history.append(
            {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in tr.items()},
                **{f"val_{k}": v for k, v in va.items()},
            }
        )
        pd.DataFrame(history).to_csv(
            out_dir / "history.csv", index=False, encoding="utf-8-sig"
        )
        print(
            f"epoch={epoch} val_MAE={va['MAE']:.5f} "
            f"val_RMSE={va['RMSE']:.5f}",
            flush=True,
        )

        if va["RMSE"] < best_rmse - 1e-9:
            best_rmse = va["RMSE"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_val_RMSE": best_rmse,
                    "args": run_meta,
                },
                out_dir / "best.pt",
            )
            print(
                f"Saved best.pt val_RMSE={best_rmse:.5f}", flush=True
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(
                    f"Early stopping at epoch={epoch}; "
                    f"best_epoch={best_epoch}",
                    flush=True,
                )
                break

    train_seconds = time.perf_counter() - train_t0
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / 1024**2 if cuda else math.nan
    )

    summary_row = {
        "model_name": args.model_name,
        "variant": args.variant,
        "seed": args.seed,
        "lr": args.lr,
        "best_epoch": best_epoch,
        "best_val_RMSE": best_rmse,
        "train_seconds": train_seconds,
        "peak_gpu_memory_mb": peak_memory_mb,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    pd.DataFrame([summary_row]).to_csv(
        out_dir / "run_summary.csv", index=False, encoding="utf-8-sig"
    )

    if args.skip_test:
        (out_dir / "run_status.json").write_text(
            json.dumps(
                {
                    "status": "completed_validation_only",
                    "model_name": args.model_name,
                    "seed": args.seed,
                    "best_epoch": best_epoch,
                    "best_val_RMSE": best_rmse,
                    "test_split_constructed": False,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print("Validation-only tuning completed; test split was not loaded.")
        return

    assert test_ds is not None and test_loader is not None
    best = safe_torch_load(out_dir / "best.pt", device)
    model.load_state_dict(best["model"])

    print("\nFinal test evaluation", flush=True)
    test_t0 = time.perf_counter()
    te, detail, arrays = run_epoch(
        model,
        test_loader,
        device,
        None,
        args.mse_weight,
        args.grad_clip,
        args.log_every,
        collect_detail=True,
        collect_arrays=bool(args.save_arrays),
    )
    test_seconds = time.perf_counter() - test_t0
    macro = save_details(
        out_dir, detail, train_ds.cities, train_ds.stations
    )

    row = {
        **te,
        **macro,
        "model_name": args.model_name,
        "variant": args.variant,
        "seed": args.seed,
        "lr": args.lr,
        "best_epoch": best_epoch,
        "best_val_RMSE": best_rmse,
        "train_seconds": train_seconds,
        "test_seconds": test_seconds,
        "peak_gpu_memory_mb": peak_memory_mb,
        "gpu_name": gpu_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    pd.DataFrame([row]).to_csv(
        out_dir / "metrics_overall.csv", index=False, encoding="utf-8-sig"
    )

    prediction_manifest = None
    if args.save_arrays:
        prediction_manifest = save_prediction_arrays(
            out_dir, arrays, test_ds, row
        )

    (out_dir / "run_status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "model_name": args.model_name,
                "seed": args.seed,
                "best_epoch": best_epoch,
                "best_val_RMSE": best_rmse,
                "test_MAE": te["MAE"],
                "test_RMSE": te["RMSE"],
                "arrays_saved": bool(args.save_arrays),
                "array_metric_match": (
                    None
                    if prediction_manifest is None
                    else prediction_manifest["metric_match"]
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
