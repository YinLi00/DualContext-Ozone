#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extended global forecasting baselines under revision-v2.

Models:
  - LSTM: local direct multi-horizon recurrent baseline
  - iTransformer, TimeMixer, TimesNet: imported from Time-Series-Library

All models reuse the exact revision-v2 data protocol and metric code from
train_global_tsmixer.py. The city_time wrapper supplies the same city
identity and future-calendar information used by the existing global baselines.

Tuning mode (--skip_test) does not construct or read the test split.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn

from train_global_tsmixer import (
    H,
    L,
    GlobalV2Dataset,
    make_loader,
    run_epoch,
    save_details,
    seed_everything,
)


MODEL_CHOICES = ["LSTM", "iTransformer", "TimeMixer", "TimesNet"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path,
                   default=Path(__file__).resolve().parents[3])
    p.add_argument("--tslib_root", type=Path, default=None)
    p.add_argument("--model_name", choices=MODEL_CHOICES, required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--variant", choices=["plain", "city_time"],
                   default="city_time")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--mse_weight", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Shared architecture controls
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--d_ff", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--e_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--factor", type=int, default=1)
    p.add_argument("--activation", type=str, default="gelu")
    p.add_argument("--embed", type=str, default="timeF")
    p.add_argument("--freq", type=str, default="h")
    p.add_argument("--use_norm", type=int, default=1)

    # LSTM
    p.add_argument("--lstm_hidden", type=int, default=64)
    p.add_argument("--lstm_layers", type=int, default=2)

    # TimesNet
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--num_kernels", type=int, default=6)

    # TimeMixer
    p.add_argument("--moving_avg", type=int, default=25)
    p.add_argument("--decomp_method",
                   choices=["moving_avg", "dft_decomp"],
                   default="moving_avg")
    p.add_argument("--down_sampling_layers", type=int, default=2)
    p.add_argument("--down_sampling_window", type=int, default=2)
    p.add_argument("--down_sampling_method",
                   choices=["avg", "max", "conv"], default="avg")
    p.add_argument("--channel_independence", type=int,
                   choices=[0, 1], default=0)

    # Information-matched wrapper
    p.add_argument("--city_emb_dim", type=int, default=16)
    p.add_argument("--condition_hidden", type=int, default=64)

    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--max_test_samples", type=int, default=0)
    p.add_argument("--skip_test", action="store_true")
    p.add_argument("--require_cuda", action="store_true")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--log_every", type=int, default=1000)
    return p.parse_args()


class DirectLSTMBackbone(nn.Module):
    """Classic LSTM with direct 48-step multivariate projection."""

    def __init__(self, n_vars: int, hidden: int, layers: int,
                 dropout: float):
        super().__init__()
        self.n_vars = n_vars
        self.pred_len = H
        self.lstm = nn.LSTM(
            input_size=n_vars,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, H * n_vars),
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None,
                x_mark_dec=None, mask=None):
        out, _ = self.lstm(x_enc)
        last = self.norm(out[:, -1, :])
        return self.head(last).reshape(x_enc.shape[0], H, self.n_vars)


def import_tslib_model(tslib_root: Path, model_name: str,
                       cfg: SimpleNamespace) -> nn.Module:
    model_file = tslib_root / "models" / f"{model_name}.py"
    if not model_file.exists():
        raise FileNotFoundError(
            f"TSLib model file not found: {model_file}\n"
            f"Expected the project Time-Series-Library under: {tslib_root}"
        )

    # Avoid accidentally importing another installed top-level package named
    # "models" before the project repository.
    root_str = str(tslib_root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)

    mod = importlib.import_module(f"models.{model_name}")
    return mod.Model(cfg)


def patch_itransformer_forecast_out_of_place(backbone: nn.Module) -> None:
    """Use explicit out-of-place normalization for differentiable city input."""

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_norm = centered / stdev

        _, _, n_vars = x_norm.shape
        enc_out = self.enc_embedding(x_norm, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)
        dec_out = dec_out[:, :, :n_vars]
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1)
        return dec_out

    backbone.forecast = MethodType(forecast, backbone)


class InformationMatchedBaseline(nn.Module):
    """Match city identity and future calendar information across baselines."""

    def __init__(
        self,
        backbone: nn.Module,
        n_cities: int,
        n_vars: int,
        o3_idx: int,
        variant: str,
        city_emb_dim: int,
        condition_hidden: int,
    ):
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

        out = self.backbone(x_in, None, None, None)

        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim != 3:
            raise RuntimeError(
                f"Unexpected backbone output shape: {tuple(out.shape)}"
            )
        if out.shape[1] < H:
            raise RuntimeError(
                f"Backbone returned {out.shape[1]} steps; expected at least {H}"
            )
        if out.shape[2] <= self.o3_idx:
            raise RuntimeError(
                f"Backbone returned {out.shape[2]} variables; "
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


def build_cfg(args: argparse.Namespace, n_vars: int) -> SimpleNamespace:
    """Superset of attributes consumed by supported TSLib model files."""

    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=L,
        label_len=0,
        pred_len=H,
        enc_in=n_vars,
        dec_in=n_vars,
        c_out=n_vars,

        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=1,
        dropout=args.dropout,
        factor=args.factor,
        activation=args.activation,
        embed=args.embed,
        freq=args.freq,
        output_attention=False,
        use_norm=args.use_norm,

        # TimesNet
        top_k=args.top_k,
        num_kernels=args.num_kernels,

        # TimeMixer
        moving_avg=args.moving_avg,
        decomp_method=args.decomp_method,
        down_sampling_layers=args.down_sampling_layers,
        down_sampling_window=args.down_sampling_window,
        down_sampling_method=args.down_sampling_method,
        channel_independence=bool(args.channel_independence),

        # Harmless compatibility attributes for differing TSLib snapshots
        num_class=2,
        individual=False,
        distil=True,
    )


def build_backbone(
    args: argparse.Namespace,
    tslib_root: Path,
    n_vars: int,
) -> nn.Module:
    if args.model_name == "LSTM":
        return DirectLSTMBackbone(
            n_vars=n_vars,
            hidden=args.lstm_hidden,
            layers=args.lstm_layers,
            dropout=args.dropout,
        )

    cfg = build_cfg(args, n_vars)
    backbone = import_tslib_model(tslib_root, args.model_name, cfg)

    if args.model_name == "iTransformer":
        patch_itransformer_forecast_out_of_place(backbone)
        print(
            "Applied out-of-place iTransformer normalization patch "
            "for differentiable city input.",
            flush=True,
        )

    return backbone


def resolve_out_dir(root: Path, out_dir: Path) -> Path:
    return out_dir.resolve() if out_dir.is_absolute() else (root / out_dir).resolve()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    tslib_root = (
        args.tslib_root or root / "Time-Series-Library-main"
    ).expanduser().resolve()
    out_dir = resolve_out_dir(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)

    cuda = torch.cuda.is_available()
    if args.require_cuda and not cuda:
        raise RuntimeError(
            "CUDA is required but torch.cuda.is_available() is False"
        )
    device = torch.device("cuda:0" if cuda else "cpu")

    train_ds = GlobalV2Dataset(
        root, "train", args.max_train_samples, args.seed
    )
    val_ds = GlobalV2Dataset(
        root, "val", args.max_val_samples, args.seed + 1
    )
    train_loader = make_loader(
        train_ds, args.batch_size, True, args.num_workers, cuda
    )
    val_loader = make_loader(
        val_ds, args.batch_size, False, args.num_workers, cuda
    )

    # Strict validation-only tuning: do not construct/read the test split.
    test_ds = None
    test_loader = None
    if not args.skip_test:
        test_ds = GlobalV2Dataset(
            root, "test", args.max_test_samples, args.seed + 2
        )
        test_loader = make_loader(
            test_ds, args.batch_size, False, args.num_workers, cuda
        )

    backbone = build_backbone(
        args, tslib_root, len(train_ds.variables)
    )
    model = InformationMatchedBaseline(
        backbone=backbone,
        n_cities=len(train_ds.cities),
        n_vars=len(train_ds.variables),
        o3_idx=train_ds.o3_idx,
        variant=args.variant,
        city_emb_dim=args.city_emb_dim,
        condition_hidden=args.condition_hidden,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    gpu_name = torch.cuda.get_device_name(0) if cuda else "CPU"

    print("=" * 68)
    print("[GLOBAL EXTENDED BASELINE — REVISION-V2]")
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
    if args.model_name == "LSTM":
        print(
            f"hidden={args.lstm_hidden} layers={args.lstm_layers} "
            f"dropout={args.dropout}"
        )
    else:
        print(
            f"d_model={args.d_model} d_ff={args.d_ff} "
            f"e_layers={args.e_layers}"
        )
    print(
        f"params total={total_params:,}, trainable={trainable_params:,}"
    )
    print("=" * 68, flush=True)

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
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": 0 if test_ds is None else len(test_ds),
            "test_loaded_during_tuning": False,
            "metric_scale": (
                "physical O3 after city-level train-only inverse standardization"
            ),
            "loss": f"L1 + {args.mse_weight} * MSE on standardized O3",
            "selection": (
                "minimum validation physical RMSE; "
                "calibration/test not used"
            ),
        }
    )
    (out_dir / "args.json").write_text(
        json.dumps(
            run_meta,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
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
        print(f"\nEpoch {epoch}/{args.epochs} train")
        tr, _ = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            args.mse_weight,
            args.grad_clip,
            args.log_every,
            False,
            train_ds.cities,
            train_ds.stations,
        )

        print(f"Epoch {epoch}/{args.epochs} val")
        va, _ = run_epoch(
            model,
            val_loader,
            device,
            None,
            args.mse_weight,
            args.grad_clip,
            args.log_every,
            False,
            train_ds.cities,
            train_ds.stations,
        )

        history.append(
            {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in tr.items()},
                **{f"val_{k}": v for k, v in va.items()},
            }
        )
        pd.DataFrame(history).to_csv(
            out_dir / "history.csv",
            index=False,
            encoding="utf-8-sig",
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
                f"Saved best.pt val_RMSE={best_rmse:.5f}",
                flush=True,
            )
        else:
            bad_epochs += 1
            print(
                f"No improvement: {bad_epochs}/{args.patience}",
                flush=True,
            )
            if bad_epochs >= args.patience:
                print(
                    f"Early stopping at epoch={epoch}; "
                    f"best_epoch={best_epoch}",
                    flush=True,
                )
                break

    train_seconds = time.perf_counter() - train_t0
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / 1024**2
        if cuda
        else math.nan
    )

    summary_row = {
        "model_name": args.model_name,
        "variant": args.variant,
        "seed": args.seed,
        "lr": args.lr,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "n_heads": args.n_heads,
        "e_layers": args.e_layers,
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": args.lstm_layers,
        "top_k": args.top_k,
        "num_kernels": args.num_kernels,
        "moving_avg": args.moving_avg,
        "decomp_method": args.decomp_method,
        "down_sampling_layers": args.down_sampling_layers,
        "down_sampling_window": args.down_sampling_window,
        "down_sampling_method": args.down_sampling_method,
        "channel_independence": args.channel_independence,
        "best_epoch": best_epoch,
        "best_val_RMSE": best_rmse,
        "epochs_ran": len(history),
        "train_seconds": train_seconds,
        "peak_gpu_memory_mb": peak_memory_mb,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    pd.DataFrame([summary_row]).to_csv(
        out_dir / "run_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if args.skip_test:
        print(
            "skip_test=True: validation-only tuning completed; "
            "test split was not constructed.",
            flush=True,
        )
        return

    if test_ds is None or test_loader is None:
        raise RuntimeError("Internal error: final run has no test loader")

    best = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"])

    print("\nFinal test evaluation")
    test_t0 = time.perf_counter()
    te, detail = run_epoch(
        model,
        test_loader,
        device,
        None,
        args.mse_weight,
        args.grad_clip,
        args.log_every,
        True,
        train_ds.cities,
        train_ds.stations,
    )
    test_seconds = time.perf_counter() - test_t0

    macro = save_details(
        out_dir,
        detail,
        train_ds.cities,
        train_ds.stations,
    )
    row = {
        **te,
        **macro,
        **summary_row,
        "test_seconds": test_seconds,
        "gpu_name": gpu_name,
    }
    pd.DataFrame([row]).to_csv(
        out_dir / "metrics_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
