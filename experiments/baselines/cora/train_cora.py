#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Formal revision-v2 CoRA baseline.

Primary protocol:
- official Timer forecasting checkpoint (67.398M trainable parameters);
- official CoRA correlation-aware architecture, with the audited H=48 bridge;
- 96 h x 8 historical variables -> 48 h x 8 supervised targets;
- city x variable train-only revision-v2 normalization;
- model selection on validation O3 RMSE;
- final paper metrics on physical-unit O3 only;
- no city ID is provided to the model;
- no future pollutant/met variables are inference inputs.

The official repository wrapper has an optimizer ambiguity: its branch named
"plugin" optimizes self.model.fm.parameters(), while fm is the Timer backbone.
The primary run here uses optimizer_scope=all, explicitly stepping BOTH Timer
and CoRA parameters, which matches the intended end-to-end CoRA adaptation.
The literal repository behavior remains available as --optimizer_scope repo_literal
for sensitivity/audit, but is not the primary formal setting.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cora_common import (
    TIMER_PARAMETER_COUNT,
    build_cora_model,
    make_cora_config,
    parameter_groups,
    seed_everything,
)
from cora_dataset import CoRARevisionV2Dataset, H


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--backbone_lr", type=float, default=1e-4)
    p.add_argument("--plugin_lr", type=float, default=1e-4)
    p.add_argument("--alpha_mae", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument(
        "--optimizer_scope",
        choices=["all", "repo_literal", "plugin_only"],
        default="all",
    )
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--max_test_samples", type=int, default=0)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("experiments/results/18_cora_baseline"),
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("Timer_forecast_1.0.ckpt"),
    )
    p.add_argument(
        "--plugin_path",
        type=Path,
        default=Path(
            "experiments/baselines/cora/vendor/plugin_ar_v2.py"
        ),
    )
    return p.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (root / path).resolve()


def make_loader(ds, batch_size, shuffle, workers, cuda):
    kw = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=cuda,
        drop_last=False,
    )
    if workers > 0:
        kw["persistent_workers"] = True
    return DataLoader(**kw)


def model_inputs(x: torch.Tensor):
    b = x.shape[0]
    device = x.device
    mark_enc = torch.zeros(b, 96, 1, dtype=x.dtype, device=device)
    mark_dec = torch.zeros(b, 48, 1, dtype=x.dtype, device=device)
    dec_inp = torch.zeros(b, 48, 8, dtype=x.dtype, device=device)
    return dec_inp, mark_enc, mark_dec


class Metric:
    def __init__(self):
        self.abs = 0.0
        self.sq = 0.0
        self.err = 0.0
        self.n = 0

    def update(self, pred, true):
        d = pred.astype(np.float64) - true.astype(np.float64)
        self.abs += float(np.abs(d).sum())
        self.sq += float(np.square(d).sum())
        self.err += float(d.sum())
        self.n += d.size

    def result(self):
        if self.n == 0:
            return {"MAE": math.nan, "RMSE": math.nan, "Bias": math.nan}
        return {
            "MAE": self.abs / self.n,
            "RMSE": math.sqrt(self.sq / self.n),
            "Bias": self.err / self.n,
        }


def build_optimizer(model, args):
    backbone, plugin = parameter_groups(model)
    if not backbone or not plugin:
        raise RuntimeError("Could not separate Timer and CoRA parameter groups")

    if args.optimizer_scope == "all":
        groups = [
            {"params": backbone, "lr": args.backbone_lr, "name": "Timer-67M"},
            {"params": plugin, "lr": args.plugin_lr, "name": "CoRA"},
        ]
    elif args.optimizer_scope == "repo_literal":
        # Literal active branch in official adapters_for_plugin.py:
        # optimizer(self.model.fm.parameters()).
        groups = [
            {"params": backbone, "lr": args.plugin_lr, "name": "repo_literal_fm"}
        ]
    else:
        groups = [{"params": plugin, "lr": args.plugin_lr, "name": "CoRA_only"}]

    optimizer = torch.optim.Adam(groups)
    return optimizer, len(backbone), len(plugin)


def train_epoch(model, loader, device, optimizer, args):
    model.train()
    model.config.is_train = 1

    loss_sum = mse_sum = mae_sum = aux_sum = 0.0
    batches = 0
    t0 = time.perf_counter()

    for step, batch in enumerate(loader, 1):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        dec_inp, mark_enc, mark_dec = model_inputs(x)

        optimizer.zero_grad(set_to_none=True)
        pred, aux = model(x, dec_inp, mark_enc, mark_dec)

        if pred.shape != y.shape:
            raise RuntimeError(f"Train shape mismatch pred={pred.shape}, y={y.shape}")

        mse = F.mse_loss(pred, y)
        mae = F.l1_loss(pred, y)
        # Effective official wrapper objective:
        # MSE + alpha*MAE + CoRA auxiliary loss (already gamma-scaled).
        loss = mse + args.alpha_mae * mae + aux
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite train loss at step {step}")

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_sum += float(loss.detach().cpu())
        mse_sum += float(mse.detach().cpu())
        mae_sum += float(mae.detach().cpu())
        aux_sum += float(aux.detach().cpu())
        batches += 1

        if step == 1 or step % args.log_every == 0:
            print(
                f"  step={step}/{len(loader)} "
                f"loss={loss_sum/batches:.6f} "
                f"mse={mse_sum/batches:.6f} "
                f"mae={mae_sum/batches:.6f} "
                f"aux={aux_sum/batches:.6f}",
                flush=True,
            )

    return {
        "loss": loss_sum / max(1, batches),
        "mse_all_std": mse_sum / max(1, batches),
        "mae_all_std": mae_sum / max(1, batches),
        "aux": aux_sum / max(1, batches),
        "seconds": time.perf_counter() - t0,
    }


@torch.no_grad()
def evaluate(model, loader, device, o3_idx, collect=False):
    model.eval()
    model.config.is_train = 0

    metric = Metric()
    all_mse_sum = 0.0
    all_mae_sum = 0.0
    batches = 0

    pred_list = []
    true_list = []
    city_list = []
    aid_list = []
    start_list = []
    pos_list = []

    t0 = time.perf_counter()
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        dec_inp, mark_enc, mark_dec = model_inputs(x)

        pred = model(x, dec_inp, mark_enc, mark_dec)
        if pred.shape != y.shape:
            raise RuntimeError(f"Eval shape mismatch pred={pred.shape}, y={y.shape}")
        if not torch.isfinite(pred).all():
            raise RuntimeError("Non-finite evaluation prediction")

        all_mse_sum += float(F.mse_loss(pred, y).cpu())
        all_mae_sum += float(F.l1_loss(pred, y).cpu())
        batches += 1

        o3_mu = batch["o3_mu"].numpy()[:, None]
        o3_std = batch["o3_std"].numpy()[:, None]
        pred_o3_std = pred[:, :, o3_idx].cpu().numpy()
        pred_o3 = pred_o3_std * o3_std + o3_mu
        true_o3 = batch["y_o3_phys"].numpy()
        metric.update(pred_o3, true_o3)

        if collect:
            pred_list.append(pred_o3.astype(np.float32))
            true_list.append(true_o3.astype(np.float32))
            city_list.append(batch["city_id"].numpy().astype(np.int16))
            aid_list.append(batch["array_id"].numpy().astype(np.int32))
            start_list.append(batch["start_idx"].numpy().astype(np.int32))
            pos_list.append(
                batch["canonical_position"].numpy().astype(np.int64)
            )

    result = {
        **metric.result(),
        "mse_all_std": all_mse_sum / max(1, batches),
        "mae_all_std": all_mae_sum / max(1, batches),
        "seconds": time.perf_counter() - t0,
    }

    arrays = None
    if collect:
        arrays = {
            "pred_o3": np.concatenate(pred_list, axis=0),
            "true_o3": np.concatenate(true_list, axis=0),
            "city_ids": np.concatenate(city_list, axis=0),
            "array_id": np.concatenate(aid_list, axis=0),
            "start_idx": np.concatenate(start_list, axis=0),
            "canonical_test_positions": np.concatenate(pos_list, axis=0),
        }
    return result, arrays


def save_test_metrics(seed_dir: Path, arrays: dict, cities: list[str]):
    pred = arrays["pred_o3"]
    true = arrays["true_o3"]
    city_ids = arrays["city_ids"]

    d = pred.astype(np.float64) - true.astype(np.float64)
    overall = {
        "N_windows": int(pred.shape[0]),
        "N_scalar": int(pred.size),
        "MAE": float(np.abs(d).mean()),
        "RMSE": float(np.sqrt(np.square(d).mean())),
        "Bias": float(d.mean()),
    }
    pd.DataFrame([overall]).to_csv(
        seed_dir / "metrics_overall.csv", index=False, encoding="utf-8-sig"
    )

    rows = []
    for cid, city in enumerate(cities):
        m = city_ids == cid
        if not np.any(m):
            continue
        dc = d[m]
        rows.append(
            {
                "city_id": cid,
                "city": city,
                "N_windows": int(m.sum()),
                "MAE": float(np.abs(dc).mean()),
                "RMSE": float(np.sqrt(np.square(dc).mean())),
                "Bias": float(dc.mean()),
            }
        )
    pd.DataFrame(rows).to_csv(
        seed_dir / "metrics_by_city.csv", index=False, encoding="utf-8-sig"
    )

    hrows = []
    for h in range(H):
        dh = d[:, h]
        hrows.append(
            {
                "horizon": h + 1,
                "N": int(len(dh)),
                "MAE": float(np.abs(dh).mean()),
                "RMSE": float(np.sqrt(np.square(dh).mean())),
                "Bias": float(dh.mean()),
            }
        )
    pd.DataFrame(hrows).to_csv(
        seed_dir / "metrics_by_horizon.csv", index=False, encoding="utf-8-sig"
    )
    return overall


def cpu_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def main():
    args = parse_args()
    root = args.root.resolve()
    checkpoint = resolve(root, args.checkpoint)
    plugin_path = resolve(root, args.plugin_path)
    out_root = resolve(root, args.out_dir)
    seed_dir = out_root / f"seed{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    cuda = device.type == "cuda"

    seed_everything(args.seed)

    print("=" * 100)
    print("CoRA / official Timer-67M / revision-v2")
    print("=" * 100)
    print("seed       =", args.seed)
    print("device     =", device)
    print("checkpoint =", checkpoint)
    print("plugin     =", plugin_path)
    print("out        =", seed_dir)
    print("optimizer_scope =", args.optimizer_scope)

    ds_kwargs = dict(root=root)
    train_ds = CoRARevisionV2Dataset(
        **ds_kwargs, split="train",
        max_samples=args.max_train_samples, seed=args.seed,
    )
    val_ds = CoRARevisionV2Dataset(
        **ds_kwargs, split="val",
        max_samples=args.max_val_samples, seed=args.seed + 1,
    )
    test_ds = CoRARevisionV2Dataset(
        **ds_kwargs, split="test",
        max_samples=args.max_test_samples, seed=args.seed + 2,
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

    cfg = make_cora_config(
        n_vars=len(train_ds.variables),
        seq_len=96,
        horizon=48,
        plugin_lr=args.plugin_lr,
        backbone_lr=args.backbone_lr,
    )
    model = build_cora_model(
        root=root,
        checkpoint=checkpoint,
        plugin_path=plugin_path,
        config=cfg,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(
        p.numel() for n, p in model.named_parameters() if n.startswith("fm.")
    )
    plugin_params = total_params - backbone_params
    if backbone_params != TIMER_PARAMETER_COUNT:
        raise RuntimeError("Timer parameter count changed unexpectedly")

    optimizer, n_backbone_tensors, n_plugin_tensors = build_optimizer(model, args)

    optimized_ids = {
        id(p) for group in optimizer.param_groups for p in group["params"]
    }
    optimized_params = sum(
        p.numel() for p in model.parameters() if id(p) in optimized_ids
    )

    print(f"train/val/test = {len(train_ds):,}/{len(val_ds):,}/{len(test_ds):,}")
    print(f"params total   = {total_params:,}")
    print(f"Timer params   = {backbone_params:,}")
    print(f"CoRA params    = {plugin_params:,}")
    print(f"optimized      = {optimized_params:,}")
    print(
        f"optimizer tensors Timer/CoRA = "
        f"{n_backbone_tensors}/{n_plugin_tensors}"
    )

    if cuda:
        torch.cuda.reset_peak_memory_stats(device)

    history = []
    best_rmse = float("inf")
    best_epoch = None
    bad_epochs = 0
    train_t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} - TRAIN", flush=True)
        tr = train_epoch(model, train_loader, device, optimizer, args)

        print(f"Epoch {epoch}/{args.epochs} - VAL", flush=True)
        va, _ = evaluate(model, val_loader, device, train_ds.o3_idx, collect=False)
        print(
            f"  val O3 MAE={va['MAE']:.6f} "
            f"RMSE={va['RMSE']:.6f} Bias={va['Bias']:.6f}",
            flush=True,
        )

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in va.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            seed_dir / "training_log.csv", index=False, encoding="utf-8-sig"
        )

        if va["RMSE"] < best_rmse:
            best_rmse = va["RMSE"]
            best_epoch = epoch
            bad_epochs = 0
            ckpt = {
                "epoch": epoch,
                "model_state": cpu_state_dict(model),
                "val_metrics": va,
                "seed": args.seed,
                "optimizer_scope": args.optimizer_scope,
            }
            torch.save(ckpt, seed_dir / "best.pt")
            print(f"  saved best.pt val_RMSE={best_rmse:.6f}", flush=True)
        else:
            bad_epochs += 1
            print(f"  no improvement ({bad_epochs}/{args.patience})", flush=True)
            if bad_epochs >= args.patience:
                print("  early stopping", flush=True)
                break

    train_seconds = time.perf_counter() - train_t0
    if best_epoch is None:
        raise RuntimeError("No best checkpoint was saved")

    best = torch.load(seed_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state"], strict=True)
    model.to(device)

    print("\nFINAL CANONICAL TEST", flush=True)
    te, arrays = evaluate(
        model, test_loader, device, train_ds.o3_idx, collect=True
    )
    assert arrays is not None

    # Save canonical arrays.
    for key, value in arrays.items():
        np.save(seed_dir / f"{key}.npy", value)

    overall = save_test_metrics(seed_dir, arrays, test_ds.cities)

    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if cuda else None
    )
    checkpoint_bytes = int((seed_dir / "best.pt").stat().st_size)
    throughput = (
        len(test_ds) / te["seconds"] if te["seconds"] > 0 else math.nan
    )

    manifest = {
        "method": "CoRA",
        "backbone": "official Timer forecasting checkpoint / Timer-67M architecture",
        "seed": args.seed,
        "protocol": {
            "history_hours": 96,
            "horizon_hours": 48,
            "n_variables": 8,
            "training_supervision": "all 8 future variables",
            "paper_evaluation": "O3 only",
            "o3_idx": train_ds.o3_idx,
            "normalization": "revision-v2 city-variable train-only mean/std",
            "city_id_input": False,
            "future_pollutant_or_met_input": False,
            "model_selection": "validation physical-unit O3 RMSE",
        },
        "cora_compatibility_edits": [
            "removed confirmed-unused TinyTimeMixer import from local vendored copy",
            "CoRA output head 96 -> configured horizon 48",
            "output_patch_len 96 -> configured horizon 48",
            "for H<96, fuse FIRST H Timer forecast steps as in official Timer inference",
            "PyTorch>=2.6 checkpoint is explicitly loaded with weights_only=False",
        ],
        "optimizer": {
            "scope": args.optimizer_scope,
            "type": "Adam",
            "backbone_lr": args.backbone_lr,
            "plugin_lr": args.plugin_lr,
            "alpha_mae": args.alpha_mae,
            "objective": "MSE(all 8 std) + alpha*MAE(all 8 std) + gamma*CoRA_aux",
            "gamma": cfg.plugin.gama,
            "grad_clip": args.grad_clip,
        },
        "samples": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "parameters": {
            "total": total_params,
            "Timer": backbone_params,
            "CoRA": plugin_params,
            "optimized": optimized_params,
        },
        "selected_epoch": best_epoch,
        "best_val_rmse_o3": best_rmse,
        "test_o3": overall,
        "timing": {
            "train_seconds": train_seconds,
            "test_seconds": te["seconds"],
            "test_windows_per_second": throughput,
        },
        "memory": {
            "peak_cuda_allocated_bytes": peak_bytes,
        },
        "files": {
            "official_timer_checkpoint": str(checkpoint),
            "vendored_cora_plugin": str(plugin_path),
            "best_checkpoint_bytes": checkpoint_bytes,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "device": (
                torch.cuda.get_device_name(device) if cuda else "CPU"
            ),
        },
        "args": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
        },
    }
    (seed_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print(
        f"DONE seed={args.seed} epoch={best_epoch} "
        f"MAE={overall['MAE']:.6f} RMSE={overall['RMSE']:.6f} "
        f"Bias={overall['Bias']:.6f}"
    )
    print(f"train_seconds={train_seconds:.1f}")
    print(f"test_seconds={te['seconds']:.1f}")
    print(f"test_windows/s={throughput:.2f}")
    if peak_bytes is not None:
        print(f"peak_cuda_allocated_GiB={peak_bytes / 2**30:.3f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
