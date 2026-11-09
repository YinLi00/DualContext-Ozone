#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core TSFM-necessity training under the canonical revision-v2 protocol.

This script trains three controlled replacements of the full P4 model:

N0 no_backbone:
    The 48-step base forecast is zero in standardized space.

N1 persistence:
    The 48-step base forecast repeats the last value of the adapted O3 input.

N3 random_timer:
    The same Timer architecture is randomly initialized, frozen, and used as
    the base forecaster. The random Timer state is stored once per seed.

All three variants keep the rest of the full P4 architecture fixed:
InputAdapter, full CoRAContextEncoder, future Hour+DOY features, variable/city
embeddings, horizon gamma/beta and the same residual output MLP.

N4 pretrained Timer is not retrained here. It is reused from Group 1 P4.
"""
from __future__ import annotations

import argparse
import hashlib
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
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM

L, H = 96, 48
VARIANTS = ("n0_no_backbone", "n1_persistence", "n3_random_timer")
DISPLAY_NAMES = {
    "n0_no_backbone": "No Backbone",
    "n1_persistence": "Persistence Prior",
    "n3_random_timer": "Random Timer Prior",
}
EXPECTED_TEST_SHAPE = (106_200, H)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    p.add_argument("--variant", required=True, choices=VARIANTS)
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
        raise RuntimeError("--require_cuda was set, but CUDA is unavailable")
    if args.device == "cpu":
        torch.set_num_threads(max(1, int(args.cpu_threads)))
        return torch.device("cpu")
    if args.device == "cuda":
        if not has_cuda:
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if has_cuda else "cpu")


def seed_everything(seed: int, cuda: bool) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = cuda


def make_loader(dataset, batch_size: int, shuffle: bool,
                workers: int, cuda: bool) -> DataLoader:
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


def sha256_state_dict(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(tensor.numpy().tobytes(order="C"))
    return h.hexdigest()


def reset_module_parameters(module: nn.Module) -> None:
    if hasattr(module, "reset_parameters") and callable(module.reset_parameters):
        try:
            module.reset_parameters()
        except Exception:
            pass


def create_random_timer(
    model_path: Path,
    seed: int,
    cache_dir: Path,
) -> tuple[nn.Module, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    state_path = cache_dir / "random_timer_state.pt"
    manifest_path = cache_dir / "manifest.json"

    torch.manual_seed(seed)
    method = "from_config"
    fallback_reason = None
    if state_path.exists() and manifest_path.exists():
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        try:
            timer = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=True,
            )
        except Exception:
            timer = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=torch.float32,
            )
            timer.apply(reset_module_parameters)
            method = "pretrained_structure_then_reset"
        state = torch.load(state_path, map_location="cpu")
        timer.load_state_dict(state, strict=True)
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        return timer, manifest

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    try:
        timer = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
        )
    except Exception as exc:
        method = "pretrained_structure_then_reset"
        fallback_reason = repr(exc)
        timer = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        torch.manual_seed(seed)
        timer.apply(reset_module_parameters)
        # Explicitly randomize any remaining floating trainable tensors that
        # a custom module did not reset. This guarantees no pretrained tensor
        # is intentionally preserved.
        for name, parameter in timer.named_parameters():
            if not parameter.is_floating_point():
                continue
            if parameter.ndim >= 2:
                nn.init.xavier_uniform_(parameter)
            elif "norm" in name.lower() and "weight" in name.lower():
                nn.init.ones_(parameter)
            else:
                nn.init.zeros_(parameter)

    state = {
        key: value.detach().cpu()
        for key, value in timer.state_dict().items()
    }
    torch.save(state, state_path)
    manifest = {
        "state": "complete",
        "seed": seed,
        "model_path": str(model_path),
        "initialization_method": method,
        "parameter_count": sum(p.numel() for p in timer.parameters()),
        "state_sha256": sha256_state_dict(state),
        "fallback_reason": fallback_reason,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return timer, manifest


class NecessityForecaster(nn.Module):
    def __init__(
        self,
        *,
        variant: str,
        timer: Optional[nn.Module],
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
    ) -> None:
        super().__init__()
        self.variant = variant
        self.timer = timer
        if self.timer is not None:
            for parameter in self.timer.parameters():
                parameter.requires_grad_(False)
            self.timer.eval()
            if hasattr(self.timer.config, "use_cache"):
                self.timer.config.use_cache = False

        self.input_adapter = InputAdapter(
            seq_len=L,
            n_cities=n_cities,
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
        self.city_emb = nn.Embedding(n_cities, city_emb_dim)

        tdim = time_feature_dim("both")
        in_dim = (
            H + L + L + H * tdim
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
        if self.timer is not None:
            self.timer.eval()
        return self

    def base_forecast(
        self,
        x: torch.Tensor,
        x_adapt: torch.Tensor,
    ) -> torch.Tensor:
        if self.variant == "n0_no_backbone":
            return torch.zeros(
                x.shape[0],
                H,
                device=x.device,
                dtype=x.dtype,
            )
        if self.variant == "n1_persistence":
            # Persistence is a fixed classical prior.  The trainable
            # InputAdapter remains available to the residual head but cannot
            # manipulate the persistence trajectory itself.
            return x[:, -1:].expand(-1, H)
        if self.variant == "n3_random_timer":
            if self.timer is None:
                raise RuntimeError("Random Timer is missing")
            return self.timer(x_adapt).logits[:, :H].float()
        raise ValueError(self.variant)

    def forward_with_base(
        self,
        base: torch.Tensor,
        x: torch.Tensor,
        x_adapt: torch.Tensor,
        x_multi: torch.Tensor,
        time_feat: torch.Tensor,
        var_idx: torch.Tensor,
        city_id: torch.Tensor,
    ) -> torch.Tensor:
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
        return (
            calibrated
            + self.output_gate * self.output_net(features)
        )

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
        x_adapt = self.input_adapter(x, city_id)
        base = self.base_forecast(x, x_adapt)
        final = self.forward_with_base(
            base,
            x,
            x_adapt,
            x_multi,
            time_feat,
            var_idx,
            city_id,
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


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def station_ids(batch: dict[str, Any]) -> Optional[np.ndarray]:
    for key in (
        "station_id", "station_idx", "site_id",
        "site_idx", "station_code", "station",
    ):
        if key in batch:
            return to_numpy(batch[key]).reshape(-1)
    return None


def run_epoch(
    model: NecessityForecaster,
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
    by_city: dict[int, SumMetric] = defaultdict(SumMetric)
    base_by_city: dict[int, SumMetric] = defaultdict(SumMetric)
    by_station: dict[str, SumMetric] = defaultdict(SumMetric)
    h_abs = np.zeros(H, dtype=np.float64)
    h_sq = np.zeros(H, dtype=np.float64)
    base_h_abs = np.zeros(H, dtype=np.float64)
    base_h_sq = np.zeros(H, dtype=np.float64)
    h_n = np.zeros(H, dtype=np.int64)

    pred_batches: list[np.ndarray] = []
    base_batches: list[np.ndarray] = []
    true_batches: list[np.ndarray] = []
    city_batches: list[np.ndarray] = []
    station_batches: list[np.ndarray] = []
    correction_abs_sum = 0.0
    correction_sq_sum = 0.0
    base_sq_sum = 0.0
    loss_sum = 0.0
    batches = 0

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
        batches += 1
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
            city = to_numpy(batch["city_id"]).astype(np.int64)
            for cid in np.unique(city):
                mask = city == cid
                by_city[int(cid)].update(
                    pred[mask], y_phys[mask]
                )
                base_by_city[int(cid)].update(
                    base[mask], y_phys[mask]
                )
            sid = station_ids(batch)
            if sid is not None:
                for value in np.unique(sid):
                    mask = sid == value
                    by_station[str(value)].update(
                        pred[mask], y_phys[mask]
                    )
                station_batches.append(sid)

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
            city_batches.append(city)

        if training and (step == 1 or step % log_every == 0):
            metric = final_metric.result()
            print(
                f"step={step}/{len(loader)} "
                f"loss={loss_sum/max(batches,1):.6f} "
                f"MAE={metric['MAE']:.4f} "
                f"RMSE={metric['RMSE']:.4f}",
                flush=True,
            )

    final_result = final_metric.result()
    base_result = base_metric.result()
    count = max(final_result["count"], 1)
    correction_mse = correction_sq_sum / count
    detail: dict[str, Any] = {
        "by_city": by_city,
        "base_by_city": base_by_city,
        "by_station": by_station,
        "h_abs": h_abs,
        "h_sq": h_sq,
        "base_h_abs": base_h_abs,
        "base_h_sq": base_h_sq,
        "h_n": h_n,
        "base_metrics": base_result,
        "mean_abs_correction": correction_abs_sum / count,
        "correction_RMSE": math.sqrt(correction_mse),
        "correction_to_base_l2_ratio": (
            math.sqrt(correction_sq_sum)
            / (math.sqrt(base_sq_sum) + 1e-12)
        ),
    }
    if collect:
        detail["pred"] = np.concatenate(pred_batches, axis=0)
        detail["base"] = np.concatenate(base_batches, axis=0)
        detail["true"] = np.concatenate(true_batches, axis=0)
        detail["city_id"] = np.concatenate(city_batches, axis=0)
        if station_batches:
            detail["station_id"] = np.concatenate(
                station_batches, axis=0
            )

    return (
        {
            "loss": loss_sum / max(batches, 1),
            **final_result,
        },
        detail,
    )


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("timer.")
    }


def canonical_true(root: Path) -> Path:
    candidates = [
        root / "experiments/results/"
        "03_zeroshot_baselines_v2_archive/Timer/true.npy",
        root / "experiments/results/"
        "03_zeroshot_baselines_v2_reindexed/Timer/true.npy",
        root / "experiments/data_v2/indices/test_targets.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Canonical revision-v2 true.npy not found")


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
                "Run true array differs from canonical revision-v2 true"
            )
        try:
            os.link(reference.resolve(), output)
            return "hardlink_to_canonical"
        except OSError:
            try:
                relative = os.path.relpath(
                    reference.resolve(), output.parent.resolve()
                )
                os.symlink(relative, output)
                return "symlink_to_canonical"
            except OSError:
                np.save(output, true)
                return "copied_canonical_array"
    np.save(output, true)
    return "saved_smoke_subset"


def save_test_outputs(
    out: Path,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    detail: dict[str, Any],
    cities: list[str],
    *,
    best_epoch: int,
    best_val_rmse: float,
    train_seconds: float,
    test_seconds: float,
    total_params: int,
    trainable_params: int,
    device_name: str,
    peak_gpu_memory_mb: float,
    random_manifest: Optional[dict[str, Any]],
) -> None:
    base = detail["base_metrics"]
    improvement = (
        (base["MAE"] - metrics["MAE"]) / base["MAE"] * 100.0
        if base["MAE"] != 0 else math.nan
    )
    row = {
        **metrics,
        "variant": args.variant,
        "paper_name": DISPLAY_NAMES[args.variant],
        "seed": args.seed,
        "protocol": "revision-v2",
        "base_MAE": base["MAE"],
        "base_MSE": base["MSE"],
        "base_RMSE": base["RMSE"],
        "MAE_improvement_vs_base_pct": improvement,
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
        "device_name": device_name,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "random_timer_state_sha256": (
            random_manifest.get("state_sha256")
            if random_manifest else None
        ),
    }
    pd.DataFrame([row]).to_csv(
        out / "metrics_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    city_rows = []
    for cid, metric in sorted(detail["by_city"].items()):
        final = metric.result()
        base_city = detail["base_by_city"][cid].result()
        city_rows.append({
            "city_id": cid,
            "city": (
                cities[cid]
                if 0 <= cid < len(cities)
                else str(cid)
            ),
            **final,
            "base_MAE": base_city["MAE"],
            "base_MSE": base_city["MSE"],
            "base_RMSE": base_city["RMSE"],
        })
    pd.DataFrame(city_rows).to_csv(
        out / "base_vs_final_by_city.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([
        {
            "city_id": row["city_id"],
            "city": row["city"],
            "MAE": row["MAE"],
            "MSE": row["MSE"],
            "RMSE": row["RMSE"],
            "count": row["count"],
        }
        for row in city_rows
    ]).to_csv(
        out / "metrics_by_city.csv",
        index=False,
        encoding="utf-8-sig",
    )

    station_rows = []
    for sid, metric in sorted(detail["by_station"].items()):
        station_rows.append({
            "station_id": sid,
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
        final_mse = (
            detail["h_sq"][h] / n if n else math.nan
        )
        base_mse = (
            detail["base_h_sq"][h] / n if n else math.nan
        )
        horizon_rows.append({
            "horizon": h + 1,
            "MAE": (
                detail["h_abs"][h] / n if n else math.nan
            ),
            "MSE": final_mse,
            "RMSE": math.sqrt(final_mse) if n else math.nan,
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
        out / "base_vs_final_by_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([
        {
            "horizon": row["horizon"],
            "MAE": row["MAE"],
            "MSE": row["MSE"],
            "RMSE": row["RMSE"],
            "count": row["count"],
        }
        for row in horizon_rows
    ]).to_csv(
        out / "metrics_by_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if args.save_arrays:
        pred = np.asarray(detail["pred"], dtype=np.float32)
        base_pred = np.asarray(detail["base"], dtype=np.float32)
        true = np.asarray(detail["true"], dtype=np.float32)
        np.save(out / "pred.npy", pred)
        np.save(out / "base_pred.npy", base_pred)
        formal = (
            args.max_test_samples == 0
            and pred.shape == EXPECTED_TEST_SHAPE
            and true.shape == EXPECTED_TEST_SHAPE
        )
        storage = save_true(
            true,
            out / "true.npy",
            canonical_true(args.root),
            formal,
        )
        np.save(out / "city_id.npy", detail["city_id"])
        if "station_id" in detail:
            np.save(out / "station_id.npy", detail["station_id"])
        manifest = {
            "protocol": "revision-v2",
            "variant": args.variant,
            "seed": args.seed,
            "pred_shape": list(pred.shape),
            "base_pred_shape": list(base_pred.shape),
            "true_shape": list(true.shape),
            "count": int(pred.size),
            "nonfinite_pred": int((~np.isfinite(pred)).sum()),
            "nonfinite_base_pred":
                int((~np.isfinite(base_pred)).sum()),
            "nonfinite_true": int((~np.isfinite(true)).sum()),
            "true_storage": storage,
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

    source_path = (
        root / "experiments/scripts/train/"
        "train_attribution_reference.py"
    )
    source = import_file(source_path, "_necessity_canonical_source")
    index_root = root / "experiments/data_v2/indices"
    stats_path = (
        root / "experiments/data_v2/"
        "city_var_stats_train_only.json"
    )
    cipda, cora = source.load_project_modules(
        root, index_root, stats_path
    )
    Dataset = cora.Stage1O3CoRAContextDataset

    device = resolve_device(args)
    cuda = device.type == "cuda"
    seed_everything(args.seed, cuda)

    train_ds = Dataset(
        "train", args.max_train_samples, args.seed, "both"
    )
    val_ds = Dataset(
        "val", args.max_val_samples, args.seed + 1, "both"
    )
    test_ds = Dataset(
        "test", args.max_test_samples, args.seed + 2, "both"
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

    timer = None
    random_manifest = None
    if args.variant == "n3_random_timer":
        cache = (
            result_root(root)
            / f"00_backbone_cache/random_timer/seed{args.seed}"
        )
        timer, random_manifest = create_random_timer(
            Path(cipda.MODEL_PATH),
            args.seed,
            cache,
        )

    model = NecessityForecaster(
        variant=args.variant,
        timer=timer,
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

    print("=" * 96)
    print("[TSFM NECESSITY / REVISION-V2]")
    print(
        f"variant={args.variant} "
        f"paper_name={DISPLAY_NAMES[args.variant]}"
    )
    print(f"seed={args.seed} device={device_name}")
    print(
        f"train={len(train_ds):,} val={len(val_ds):,} "
        f"test={len(test_ds):,}"
    )
    print(
        f"epochs={args.epochs} patience={args.patience} "
        f"batch={args.batch_size} lr={args.lr}"
    )
    print(
        f"params total={total_params:,} "
        f"trainable={trainable_params:,}"
    )
    print("=" * 96)

    args_dict = vars(args).copy()
    for key, value in list(args_dict.items()):
        if isinstance(value, Path):
            args_dict[key] = str(value)
    args_dict.update({
        "protocol": "revision-v2",
        "full_context": True,
        "time_feature_mode": "both",
        "timer_frozen": args.variant == "n3_random_timer",
        "random_timer_manifest": random_manifest,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "device_name": device_name,
    })
    (out / "args.json").write_text(
        json.dumps(args_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    status_path = out / "run_status.json"
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
            "adapter_state": adapter_state(model),
            "args": args_dict,
            "cities": list(train_ds.cities),
            "variables": list(train_ds.variables),
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
        name for name in missing
        if not name.startswith("timer.")
    ]
    if unexpected or missing_non_timer:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing_non_timer}, "
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
        collect=True,
    )
    if cuda:
        torch.cuda.synchronize(device)
    test_seconds = time.perf_counter() - test_start
    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if cuda else 0.0
    )

    save_test_outputs(
        out,
        args,
        test_metrics,
        detail,
        list(train_ds.cities),
        best_epoch=int(best["epoch"]),
        best_val_rmse=best_rmse,
        train_seconds=train_seconds,
        test_seconds=test_seconds,
        total_params=total_params,
        trainable_params=trainable_params,
        device_name=device_name,
        peak_gpu_memory_mb=peak_gpu_memory_mb,
        random_manifest=random_manifest,
    )

    summary = {
        "state": "complete",
        "variant": args.variant,
        "paper_name": DISPLAY_NAMES[args.variant],
        "seed": args.seed,
        "protocol": "revision-v2",
        "best_epoch": int(best["epoch"]),
        "best_val_RMSE": best_rmse,
        "epochs_ran": len(history),
        "train_seconds": train_seconds,
        "test_seconds": test_seconds,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
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
                "variant": args.variant,
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
