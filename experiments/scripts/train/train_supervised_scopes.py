#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train one revision-v2 supervised baseline for one city or one station.

This script intentionally reuses the already validated global baseline model
builders. Only the training sample scope changes:

* city: one model is trained from all stations in one city;
* station: one model is trained from one station only.

Everything else remains matched to the global revision-v2 protocol:
chronological splits, 96 -> 48 task, eight variables, city-level train-only
normalization, global-selected hyperparameters, city/time conditioning, loss,
and physical-scale evaluation.
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
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT_DEFAULT = Path(__file__).resolve().parents[3]
MODELS = [
    "TSMixer", "DLinear", "PatchTST", "LSTM", "iTransformer",
    "TimeMixer", "TimesNet", "CNNLSTM", "Transformer", "TimeXer", "Koopa",
]
GLOBAL_RUNS = {
    "TSMixer": "1b1_global_tsmixer/TSMixer_city_time_seed2026",
    "DLinear": "1b2_global_dlinear/DLinear_city_time_seed2026",
    "PatchTST": "1b2_global_patchtst/PatchTST_city_time_seed2026",
    "LSTM": "1d_global_lstm/LSTM_city_time_seed2026",
    "iTransformer": "1d_global_itransformer/iTransformer_city_time_seed2026",
    "TimeMixer": "1d_global_timemixer/TimeMixer_city_time_seed2026",
    "TimesNet": "1d_global_timesnet/TimesNet_city_time_seed2026",
    "CNNLSTM": "1e_global_cnnlstm/CNNLSTM_city_time_seed2026",
    "Transformer": "1e_global_transformer/Transformer_city_time_seed2026",
    "TimeXer": "1e_global_timexer/TimeXer_city_time_seed2026",
    "Koopa": "1e_global_koopa/Koopa_city_time_seed2026",
}

DEFAULTS: dict[str, Any] = {
    "variant": "city_time", "epochs": 20, "patience": 5,
    "batch_size": 64, "num_workers": 8, "lr": 3e-4,
    "weight_decay": 1e-4, "mse_weight": 0.5, "grad_clip": 1.0,
    "d_model": 64, "d_ff": 128, "n_heads": 8, "e_layers": 2,
    "d_layers": 1, "dropout": 0.1, "factor": 1,
    "activation": "gelu", "embed": "timeF", "freq": "h",
    "use_norm": 1, "label_len": 48, "moving_avg": 25,
    "patch_len": 16, "stride": 8, "city_emb_dim": 16,
    "condition_hidden": 64, "lstm_hidden": 64, "lstm_layers": 2,
    "top_k": 3, "num_kernels": 6, "decomp_method": "moving_avg",
    "down_sampling_layers": 2, "down_sampling_window": 2,
    "down_sampling_method": "avg", "channel_independence": 0,
    "conv_channel": 64, "koopa_dynamic_dim": 32,
    "koopa_hidden_dim": 64, "koopa_hidden_layers": 2,
    "koopa_num_blocks": 2, "koopa_multistep": 0,
    "koopa_alpha": 0.2, "koopa_spectrum_batch_size": 512,
    "log_every": 1000,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    p.add_argument("--scope", choices=["city", "station"], required=True)
    p.add_argument("--unit", required=True,
                   help="City name for city scope, array_id for station scope.")
    p.add_argument("--model_name", choices=MODELS, required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--reference_args", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=-1)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--max_test_samples", type=int, default=0)
    p.add_argument("--keep_checkpoint", type=int, choices=[0, 1], default=0)
    p.add_argument("--require_cuda", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def add_train_path(root: Path):
    train_root = root / "experiments" / "scripts" / "train"
    if str(train_root) not in sys.path:
        sys.path.insert(0, str(train_root))
    import train_global_tsmixer as tsm
    import train_global_tslib_baseline as classic
    import train_global_extended_baselines as extended
    import train_global_additional_baselines as missing4
    return tsm, classic, extended, missing4


def read_reference_args(root: Path, model_name: str, explicit: Path | None) -> dict:
    path = explicit
    if path is None:
        path = (root / "experiments" / "results" /
                "02_global_baselines" / GLOBAL_RUNS[model_name] / "args.json")
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(
            f"Global reference args not found: {path}\n"
            "The local-scope experiment must start after all 11 global baselines finish."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = dict(DEFAULTS)
    cfg.update(data)
    cfg["reference_args_path"] = str(path.resolve())
    return cfg


def safe_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    os.replace(tmp, path)


def subset_after_filter(index: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or len(index) <= max_samples:
        return index
    rng = np.random.default_rng(seed)
    pos = np.sort(rng.choice(len(index), size=max_samples, replace=False))
    return index[pos]


def make_scoped_dataset_class(tsm):
    class ScopedV2Dataset(tsm.GlobalV2Dataset):
        def __init__(self, root: Path, split: str, scope: str, unit: str,
                     max_samples: int, sample_seed: int):
            super().__init__(root, split, 0, sample_seed)
            canonical = np.arange(len(self.array_id), dtype=np.int64)
            if scope == "city":
                if unit not in self.cities:
                    raise ValueError(f"Unknown city: {unit}")
                unit_city_id = self.cities.index(unit)
                keep = np.flatnonzero(self.city_id == unit_city_id)
                self.unit_name = unit
                self.unit_city_id = unit_city_id
                self.unit_array_id = None
            else:
                aid = int(unit)
                if aid < 0 or aid >= len(self.stations):
                    raise ValueError(f"array_id out of range: {aid}")
                keep = np.flatnonzero(self.array_id == aid)
                sm = self.stations[aid]
                self.unit_name = f"{sm['city']}/{sm['station_id']}"
                self.unit_city_id = self.cities.index(sm["city"])
                self.unit_array_id = aid
            keep = subset_after_filter(keep, max_samples, sample_seed)
            if len(keep) == 0:
                raise RuntimeError(
                    f"No {split} samples for scope={scope}, unit={unit}"
                )
            self.sample_index = canonical[keep]
            self.array_id = self.array_id[keep]
            self.city_id = self.city_id[keep]
            self.start_idx = self.start_idx[keep]
            self.scope = scope
            self.unit = unit

        def __getitem__(self, i: int) -> dict:
            item = super().__getitem__(i)
            aid = int(self.array_id[i])
            st = int(self.start_idx[i])
            _, time_ns = self.cache.get(aid)
            item.update({
                "sample_index": torch.tensor(int(self.sample_index[i]), dtype=torch.long),
                "start_idx": torch.tensor(st, dtype=torch.long),
                "forecast_start_time_ns": torch.tensor(
                    int(time_ns[st + tsm.L]), dtype=torch.long
                ),
            })
            return item

    return ScopedV2Dataset


def namespace_from_reference(ref: dict, cli: argparse.Namespace,
                             root: Path, model_name: str) -> argparse.Namespace:
    cfg = dict(DEFAULTS)
    cfg.update(ref)
    cfg.update({
        "root": root,
        "tslib_root": root / "Time-Series-Library-main",
        "model_name": model_name,
        "seed": cli.seed,
        "out_dir": cli.out_dir,
        "max_train_samples": cli.max_train_samples,
        "max_val_samples": cli.max_val_samples,
        "max_test_samples": cli.max_test_samples,
        "skip_test": False,
        "require_cuda": cli.require_cuda,
        "save_arrays": 1,
    })
    if cli.epochs > 0:
        cfg["epochs"] = cli.epochs
    if cli.patience > 0:
        cfg["patience"] = cli.patience
    if cli.batch_size > 0:
        cfg["batch_size"] = cli.batch_size
    if cli.num_workers >= 0:
        cfg["num_workers"] = cli.num_workers
    # Normalize aliases emitted by different global trainers.
    cfg["epochs"] = int(cfg.get("epochs", cfg.get("train_epochs", 20)))
    cfg["patience"] = int(cfg.get("patience", 5))
    cfg["batch_size"] = int(cfg.get("batch_size", 64))
    cfg["num_workers"] = int(cfg.get("num_workers", 8))
    cfg["lr"] = float(cfg.get("lr", cfg.get("learning_rate", 3e-4)))
    cfg["weight_decay"] = float(cfg.get("weight_decay", 1e-4))
    cfg["mse_weight"] = float(cfg.get("mse_weight", 0.5))
    cfg["grad_clip"] = float(cfg.get("grad_clip", 1.0))
    return argparse.Namespace(**cfg)


def build_model(model_name: str, args: argparse.Namespace, train_ds,
                device: torch.device, out_dir: Path, modules):
    tsm, classic, extended, missing4 = modules
    tslib_root = Path(args.tslib_root).resolve()
    n_vars = len(train_ds.variables)
    n_cities = len(train_ds.cities)

    if model_name == "TSMixer":
        cfg = SimpleNamespace(
            task_name="long_term_forecast", seq_len=tsm.L, pred_len=tsm.H,
            d_model=args.d_model, dropout=args.dropout, enc_in=n_vars,
            e_layers=args.e_layers,
        )
        backbone = tsm.import_tsmixer(tslib_root, cfg)
        return tsm.GlobalTSMixer(
            backbone, n_cities, n_vars, train_ds.o3_idx, args.variant,
            args.city_emb_dim, args.condition_hidden,
        ).to(device)

    if model_name in {"DLinear", "PatchTST"}:
        cfg = classic.build_cfg(args, n_vars)
        backbone = classic.import_tslib_model(tslib_root, model_name, cfg)
        if model_name == "PatchTST":
            classic.patch_patchtst_forecast_out_of_place(backbone)
            print(
                "Applied out-of-place PatchTST normalization patch.",
                flush=True,
            )
        return classic.InformationMatchedBaseline(
            backbone, n_cities, n_vars, train_ds.o3_idx, args.variant,
            args.city_emb_dim, args.condition_hidden,
        ).to(device)

    if model_name in {"LSTM", "iTransformer", "TimeMixer", "TimesNet"}:
        backbone = extended.build_backbone(args, tslib_root, n_vars)
        return extended.InformationMatchedBaseline(
            backbone, n_cities, n_vars, train_ds.o3_idx, args.variant,
            args.city_emb_dim, args.condition_hidden,
        ).to(device)

    # Koopa's spectrum must be computed from this unit's training split, not
    # loaded from the global all-city cache. The existing builder uses a local
    # cache whenever max_train_samples > 0; this flag only controls cache path
    # here because train_ds has already been constructed explicitly.
    original_max = args.max_train_samples
    if model_name == "Koopa" and args.max_train_samples <= 0:
        args.max_train_samples = 1
    try:
        backbone = missing4.build_backbone(
            args, tslib_root, n_vars, train_ds, device, out_dir
        )
    finally:
        args.max_train_samples = original_max
    return missing4.InformationMatchedBaseline(
        backbone=backbone, model_name=model_name, n_cities=n_cities,
        n_vars=n_vars, o3_idx=train_ds.o3_idx, variant=args.variant,
        city_emb_dim=args.city_emb_dim,
        condition_hidden=args.condition_hidden, label_len=args.label_len,
    ).to(device)


def merge_chunks(arrays: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    merged = {}
    for key, chunks in arrays.items():
        if not chunks:
            raise RuntimeError(f"No test chunks collected for {key}")
        merged[key] = np.concatenate(chunks, axis=0)
    return merged


def save_unit_arrays(out_dir: Path, arrays: dict[str, list[np.ndarray]]) -> dict:
    merged = merge_chunks(arrays)
    order = np.argsort(merged["sample_index"], kind="stable")
    for key in merged:
        merged[key] = merged[key][order]
    if len(np.unique(merged["sample_index"])) != len(merged["sample_index"]):
        raise RuntimeError("Duplicate canonical sample_index in unit test arrays")
    file_names = {
        "true": "true.npy", "pred": "pred.npy", "city_id": "city_id.npy",
        "array_id": "array_id.npy", "sample_index": "sample_index.npy",
        "start_idx": "start_idx.npy",
        "forecast_start_time_ns": "forecast_start_time_ns.npy",
    }
    info = {}
    for key, filename in file_names.items():
        a = np.asarray(merged[key])
        np.save(out_dir / filename, a, allow_pickle=False)
        info[filename] = {"shape": list(a.shape), "dtype": str(a.dtype)}
    return info


def main_impl(cli: argparse.Namespace) -> None:
    root = cli.root.expanduser().resolve()
    out_dir = cli.out_dir if cli.out_dir.is_absolute() else root / cli.out_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "run_status.json"

    if status_path.exists() and not cli.force:
        try:
            s = json.loads(status_path.read_text(encoding="utf-8"))
            if s.get("state") == "complete" and (out_dir / "pred.npy").exists():
                print(f"[SKIP] complete: {out_dir}")
                return
        except Exception:
            pass

    modules = add_train_path(root)
    tsm, _, _, missing4 = modules
    tsm.seed_everything(cli.seed)
    cuda = torch.cuda.is_available()
    if cli.require_cuda and not cuda:
        raise RuntimeError("CUDA required but unavailable")
    device = torch.device("cuda:0" if cuda else "cpu")

    ref = read_reference_args(root, cli.model_name, cli.reference_args)
    args = namespace_from_reference(ref, cli, root, cli.model_name)
    ScopedV2Dataset = make_scoped_dataset_class(tsm)
    train_ds = ScopedV2Dataset(root, "train", cli.scope, cli.unit,
                               cli.max_train_samples, cli.seed)
    val_ds = ScopedV2Dataset(root, "val", cli.scope, cli.unit,
                             cli.max_val_samples, cli.seed + 1)
    test_ds = ScopedV2Dataset(root, "test", cli.scope, cli.unit,
                              cli.max_test_samples, cli.seed + 2)
    train_loader = tsm.make_loader(train_ds, args.batch_size, True,
                                   args.num_workers, cuda)
    val_loader = tsm.make_loader(val_ds, args.batch_size, False,
                                 args.num_workers, cuda)
    test_loader = tsm.make_loader(test_ds, args.batch_size, False,
                                  args.num_workers, cuda)

    model = build_model(cli.model_name, args, train_ds, device, out_dir, modules)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gpu_name = torch.cuda.get_device_name(0) if cuda else "CPU"

    run_meta = {
        "protocol": "revision-v2 scoped supervised training",
        "scope": cli.scope, "unit": cli.unit,
        "unit_name": train_ds.unit_name,
        "model_name": cli.model_name, "seed": cli.seed,
        "reference_args_path": ref["reference_args_path"],
        "hyperparameter_policy": "reuse global-selected hyperparameters; no local retuning",
        "normalization": "same city-level train-only scaler as global revision-v2",
        "variant": args.variant,
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "epochs": args.epochs, "patience": args.patience,
        "batch_size": args.batch_size, "num_workers": args.num_workers,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "mse_weight": args.mse_weight, "grad_clip": args.grad_clip,
        "device": str(device), "gpu_name": gpu_name,
        "total_params": total_params, "trainable_params": trainable_params,
        "variables": train_ds.variables, "o3_idx": train_ds.o3_idx,
        "selected_args": {k: v for k, v in vars(args).items()
                          if isinstance(v, (str, int, float, bool, type(None)))},
    }
    (out_dir / "args.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    safe_status(status_path, {
        "state": "running", "started_at": pd.Timestamp.now().isoformat(),
        **{k: run_meta[k] for k in ["scope", "unit", "unit_name", "model_name", "seed"]},
    })

    print("=" * 80)
    print("[SCOPED SUPERVISED REVISION-V2]")
    print(f"scope={cli.scope} unit={cli.unit} ({train_ds.unit_name})")
    print(f"model={cli.model_name} seed={cli.seed} device={device} ({gpu_name})")
    print(f"train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}")
    print(f"epochs={args.epochs} patience={args.patience} batch={args.batch_size} lr={args.lr}")
    print(f"params={total_params:,}")
    print("=" * 80, flush=True)

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
        tr, _, _ = missing4.run_epoch(
            model, train_loader, device, optimizer, args.mse_weight,
            args.grad_clip, args.log_every, False, False,
        )
        print(f"Epoch {epoch}/{args.epochs} val", flush=True)
        va, _, _ = missing4.run_epoch(
            model, val_loader, device, None, args.mse_weight,
            args.grad_clip, args.log_every, False, False,
        )
        history.append({"epoch": epoch,
                        **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in va.items()}})
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False,
                                     encoding="utf-8-sig")
        print(f"epoch={epoch} val_MAE={va['MAE']:.6f} val_RMSE={va['RMSE']:.6f}")
        if va["RMSE"] < best_rmse - 1e-9:
            best_rmse = va["RMSE"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "best_val_RMSE": best_rmse}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}")
                break

    train_seconds = time.perf_counter() - t0
    if not best_path.exists():
        raise RuntimeError("best.pt was not created")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    print("\nFinal unit test evaluation", flush=True)
    test_t0 = time.perf_counter()
    te, detail, arrays = missing4.run_epoch(
        model, test_loader, device, None, args.mse_weight,
        args.grad_clip, args.log_every, True, True,
    )
    test_seconds = time.perf_counter() - test_t0
    macro = tsm.save_details(out_dir, detail, train_ds.cities, train_ds.stations)
    row = {
        **te, **macro, "scope": cli.scope, "unit": cli.unit,
        "unit_name": train_ds.unit_name, "model_name": cli.model_name,
        "seed": cli.seed, "best_epoch": best_epoch,
        "best_val_RMSE": best_rmse, "train_seconds": train_seconds,
        "test_seconds": test_seconds,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / 1024**2 if cuda else math.nan
        ),
        "gpu_name": gpu_name, "total_params": total_params,
        "trainable_params": trainable_params,
    }
    pd.DataFrame([row]).to_csv(out_dir / "metrics_overall.csv", index=False,
                                encoding="utf-8-sig")
    array_info = save_unit_arrays(out_dir, arrays)
    manifest = {**run_meta, "metrics": row, "arrays": array_info,
                "completed_at": pd.Timestamp.now().isoformat()}
    (out_dir / "unit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    if not cli.keep_checkpoint:
        best_path.unlink(missing_ok=True)
    safe_status(status_path, {
        "state": "complete", "completed_at": pd.Timestamp.now().isoformat(),
        "scope": cli.scope, "unit": cli.unit, "unit_name": train_ds.unit_name,
        "model_name": cli.model_name, "seed": cli.seed,
        "n_test": len(test_ds), "MAE": te["MAE"], "RMSE": te["RMSE"],
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
            "state": "failed", "failed_at": pd.Timestamp.now().isoformat(),
            "scope": cli.scope, "unit": cli.unit,
            "model_name": cli.model_name, "seed": cli.seed,
            "error": repr(exc), "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
