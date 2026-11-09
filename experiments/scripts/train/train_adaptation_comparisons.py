#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nine reviewer-requested variants under the canonical revision-v2 protocol.

New formal runs:
  q_quantile_full       Quantile-DualContext, q05/q10/q50/q90/q95
  n2_climatology        frozen city-season-hour climatology prior + full adapter
  n3_linear_ar          frozen global ridge AR(96->48) prior + full adapter
  a0_affine_only        frozen Timer + horizon-wise gamma/beta only
  a1_lora_r8            Timer attention LoRA-r8 + full adapter
  a2_full_finetune      full Timer fine-tuning + full adapter
  r1_lagged_corr        lagged-correlation DualContext
  r2_mutual_information train-only city-specific MI-guided DualContext
  r3_learned_attention  target-query learned-attention DualContext

All variants reuse the project's canonical revision-v2 Dataset, train-only
normalization, 96->48 task, five seeds and exact 2025 test order.  Quantile
training follows the previous project formulation:
  pinball + 1.0*MAE(q50) + 0.5*MSE(q50) + 0.05*crossing penalty.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
import traceback
from contextlib import nullcontext
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

L, H = 96, 48
QUANTILES = [0.05, 0.10, 0.50, 0.90, 0.95]

VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "q_quantile_full": {
        "group": "quantile",
        "paper_name": "Quantile-DualContext",
        "needs_timer": True,
        "timer_mode": "frozen",
        "relation_mode": "pearson",
        "full_adapter": True,
        "quantile": True,
    },
    "n2_climatology": {
        "group": "necessity",
        "paper_name": "Climatology Prior",
        "needs_timer": False,
        "timer_mode": "none",
        "relation_mode": "pearson",
        "full_adapter": True,
        "quantile": False,
    },
    "n3_linear_ar": {
        "group": "necessity",
        "paper_name": "Linear AR Prior",
        "needs_timer": False,
        "timer_mode": "none",
        "relation_mode": "pearson",
        "full_adapter": True,
        "quantile": False,
    },
    "a0_affine_only": {
        "group": "adaptation",
        "paper_name": "Horizon-wise Affine",
        "needs_timer": True,
        "timer_mode": "frozen",
        "relation_mode": "none",
        "full_adapter": False,
        "quantile": False,
    },
    "a1_lora_r8": {
        "group": "adaptation",
        "paper_name": "LoRA-r8 Full Adapter",
        "needs_timer": True,
        "timer_mode": "lora",
        "relation_mode": "pearson",
        "full_adapter": True,
        "quantile": False,
    },
    "a2_full_finetune": {
        "group": "adaptation",
        "paper_name": "Full Fine-tuning",
        "needs_timer": True,
        "timer_mode": "full",
        "relation_mode": "pearson",
        "full_adapter": True,
        "quantile": False,
    },
    "r1_lagged_corr": {
        "group": "relation",
        "paper_name": "Lagged-correlation DualContext",
        "needs_timer": True,
        "timer_mode": "frozen",
        "relation_mode": "lagged_corr",
        "full_adapter": True,
        "quantile": False,
    },
    "r2_mutual_information": {
        "group": "relation",
        "paper_name": "Mutual-information DualContext",
        "needs_timer": True,
        "timer_mode": "frozen",
        "relation_mode": "mutual_information",
        "full_adapter": True,
        "quantile": False,
    },
    "r3_learned_attention": {
        "group": "relation",
        "paper_name": "Learned-attention DualContext",
        "needs_timer": True,
        "timer_mode": "frozen",
        "relation_mode": "learned_attention",
        "full_adapter": True,
        "quantile": False,
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
    p.add_argument("--batch_size", type=int, default=64,
                   help="Physical micro-batch size.")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--adapter_lr", type=float, default=1e-3)
    p.add_argument("--lora_lr", type=float, default=1e-4)
    p.add_argument("--backbone_lr", type=float, default=1e-5)
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
    p.add_argument("--lag_max", type=int, default=24)
    p.add_argument("--lag_step", type=int, default=3)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lambda_mid_mae", type=float, default=1.0)
    p.add_argument("--lambda_mid_mse", type=float, default=0.5)
    p.add_argument("--lambda_cross", type=float, default=0.05)
    p.add_argument("--asset_mode", choices=["smoke", "formal"],
                   default="formal")
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--max_test_samples", type=int, default=0)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--require_cuda", action="store_true")
    p.add_argument("--amp_bf16", type=int, choices=[0, 1], default=0,
                   help="Training autocast only; validation/test stay FP32.")
    p.add_argument("--gradient_checkpointing", type=int, choices=[0, 1],
                   default=0)
    p.add_argument("--save_arrays", type=int, choices=[0, 1], default=1)
    p.add_argument("--keep_checkpoint", type=int, choices=[0, 1], default=1)
    p.add_argument("--cpu_threads", type=int, default=4)
    return p.parse_args()


def result_root(root: Path) -> Path:
    return root / "experiments/results/15_reviewer45_v2"


def asset_root(root: Path, mode: str) -> Path:
    return result_root(root) / "00_assets" / mode


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_device(args: argparse.Namespace) -> torch.device:
    available = torch.cuda.is_available()
    if args.require_cuda and not available:
        raise RuntimeError("--require_cuda set but CUDA is unavailable")
    if args.device == "cpu":
        torch.set_num_threads(max(1, int(args.cpu_threads)))
        return torch.device("cpu")
    if args.device == "cuda":
        if not available:
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if available else "cpu")


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


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class SumMetric:
    def __init__(self) -> None:
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        diff = (np.asarray(pred, dtype=np.float64)
                - np.asarray(true, dtype=np.float64))
        self.abs_sum += float(np.abs(diff).sum(dtype=np.float64))
        self.sq_sum += float(np.square(diff).sum(dtype=np.float64))
        self.count += int(diff.size)

    def result(self) -> dict[str, Any]:
        if self.count <= 0:
            return {"MAE": math.nan, "MSE": math.nan,
                    "RMSE": math.nan, "count": 0}
        mse = self.sq_sum / self.count
        return {
            "MAE": self.abs_sum / self.count,
            "MSE": mse,
            "RMSE": math.sqrt(mse),
            "count": self.count,
        }


class LoRALinear(nn.Module):
    """Dependency-free LoRA wrapper for an existing frozen nn.Linear."""

    def __init__(self, base: nn.Linear, rank: int,
                 alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.lora_dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(
            torch.empty(rank, base.in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, rank)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = F.linear(
            F.linear(self.lora_dropout(x), self.lora_A),
            self.lora_B,
        )
        return self.base(x) + self.scale * update


def get_parent_module(root: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def set_child_module(parent: nn.Module, name: str, module: nn.Module) -> None:
    if name.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(name)] = module
    else:
        setattr(parent, name, module)


def inject_lora(timer: nn.Module, rank: int, alpha: float,
                dropout: float) -> list[str]:
    linear_names = [
        name for name, module in timer.named_modules()
        if name and isinstance(module, nn.Linear)
    ]
    preferred_patterns = (
        "q_proj", "v_proj", ".query", ".value", "query_proj",
        "value_proj", "wq", "wv",
    )
    combined_patterns = ("qkv", "c_attn", "in_proj")
    selected = [
        name for name in linear_names
        if any(token in name.lower() for token in preferred_patterns)
    ]
    selected += [
        name for name in linear_names
        if any(token in name.lower() for token in combined_patterns)
    ]
    selected = sorted(set(selected))
    if not selected:
        selected = sorted({
            name for name in linear_names
            if ("attn" in name.lower() or "attention" in name.lower())
            and ("proj" in name.lower() or "linear" in name.lower())
        })
    if not selected:
        raise RuntimeError(
            "No Timer attention Linear modules were found for LoRA. "
            f"Available linear modules (first 80): {linear_names[:80]}"
        )

    for name in selected:
        parent, child = get_parent_module(timer, name)
        current = (parent[int(child)] if child.isdigit()
                   and isinstance(parent, (nn.ModuleList, nn.Sequential))
                   else getattr(parent, child))
        if not isinstance(current, nn.Linear):
            raise TypeError(f"LoRA target is not nn.Linear: {name}")
        set_child_module(
            parent, child,
            LoRALinear(current, rank, alpha, dropout),
        )
    return selected


def enable_gradient_checkpointing(timer: nn.Module) -> dict[str, Any]:
    report = {"requested": True, "enabled": False, "method": None}
    if hasattr(timer, "gradient_checkpointing_enable"):
        try:
            timer.gradient_checkpointing_enable()
            report.update(enabled=True,
                          method="gradient_checkpointing_enable")
            return report
        except Exception as exc:
            report["first_error"] = repr(exc)
    for module in timer.modules():
        if hasattr(module, "gradient_checkpointing"):
            try:
                setattr(module, "gradient_checkpointing", True)
                report.update(enabled=True,
                              method="module_attribute")
            except Exception:
                pass
    return report


class RelationContextEncoder(nn.Module):
    """Controlled context encoder with replaceable relation cue."""

    def __init__(
        self,
        *,
        seq_len: int,
        n_vars: int,
        n_cities: int,
        hidden: int,
        dropout: float,
        o3_idx: int,
        corr_temperature: float,
        relation_mode: str,
        lag_max: int,
        lag_step: int,
        mi_table: Optional[np.ndarray],
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.n_vars = int(n_vars)
        self.n_cities = int(n_cities)
        self.hidden = int(hidden)
        self.o3_idx = int(o3_idx)
        self.corr_temperature = float(corr_temperature)
        self.relation_mode = str(relation_mode)
        self.lags = list(range(-int(lag_max), int(lag_max) + 1,
                               max(1, int(lag_step))))
        if 0 not in self.lags:
            self.lags.append(0)
            self.lags.sort()

        self.channel_encoder = nn.Sequential(
            nn.LayerNorm(seq_len),
            nn.Linear(seq_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.pos_score = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.neg_score = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.relation_proj = nn.Sequential(
            nn.LayerNorm(n_vars),
            nn.Linear(n_vars, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        if self.relation_mode == "learned_attention":
            self.query_proj = nn.Linear(hidden, hidden, bias=False)
            self.key_proj = nn.Linear(hidden, hidden, bias=False)
        else:
            self.query_proj = None
            self.key_proj = None

        if mi_table is None:
            table = np.zeros((n_cities, n_vars), dtype=np.float32)
        else:
            table = np.asarray(mi_table, dtype=np.float32).copy()
            if table.shape != (n_cities, n_vars):
                raise ValueError(
                    f"MI table shape {table.shape}, expected "
                    f"{(n_cities, n_vars)}"
                )
        # The target O3 channel is represented separately by o3_ctx and must
        # never compete with exogenous covariates in a relation mechanism.
        table[:, self.o3_idx] = 0.0
        self.register_buffer(
            "mi_table",
            torch.tensor(table, dtype=torch.float32),
            persistent=True,
        )

    @staticmethod
    def pearson(x: torch.Tensor, target_idx: int,
                eps: float = 1e-6) -> torch.Tensor:
        target = x[:, :, target_idx:target_idx + 1]
        xc = x - x.mean(dim=1, keepdim=True)
        yc = target - target.mean(dim=1, keepdim=True)
        numerator = (xc * yc).mean(dim=1)
        xs = torch.sqrt((xc ** 2).mean(dim=1) + eps)
        ys = torch.sqrt((yc ** 2).mean(dim=1) + eps)
        corr = numerator / (xs * ys + eps)
        corr = torch.nan_to_num(
            corr, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(-1.0, 1.0)
        corr = corr.clone()
        corr[:, target_idx] = 0.0
        return corr

    def mask_target_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] <= 1:
            raise ValueError("Relation attention requires non-target covariates")
        masked = logits.clone()
        masked[:, self.o3_idx] = -torch.inf
        return masked

    def lagged_correlation(self, x: torch.Tensor) -> torch.Tensor:
        best = torch.zeros(
            x.shape[0], x.shape[2], device=x.device, dtype=torch.float32
        )
        best_abs = torch.full_like(best, -1.0)
        for lag in self.lags:
            if lag > 0:
                xv = x[:, lag:, :]
                yt = x[:, :-lag, self.o3_idx:self.o3_idx + 1]
            elif lag < 0:
                shift = -lag
                xv = x[:, :-shift, :]
                yt = x[:, shift:, self.o3_idx:self.o3_idx + 1]
            else:
                xv = x
                yt = x[:, :, self.o3_idx:self.o3_idx + 1]
            xc = xv - xv.mean(dim=1, keepdim=True)
            yc = yt - yt.mean(dim=1, keepdim=True)
            num = (xc * yc).mean(dim=1)
            xs = torch.sqrt((xc ** 2).mean(dim=1) + 1e-6)
            ys = torch.sqrt((yc ** 2).mean(dim=1) + 1e-6)
            corr = torch.nan_to_num(
                num / (xs * ys + 1e-6),
                nan=0.0, posinf=0.0, neginf=0.0,
            ).clamp(-1.0, 1.0)
            replace = corr.abs() > best_abs
            best = torch.where(replace, corr, best)
            best_abs = torch.where(replace, corr.abs(), best_abs)
        best = best.clone()
        best[:, self.o3_idx] = 0.0
        return best

    def forward(self, x_multi: torch.Tensor,
                city_id: torch.Tensor) -> torch.Tensor:
        B, length, n_vars = x_multi.shape
        if length != self.seq_len or n_vars != self.n_vars:
            raise ValueError(
                f"Expected [B,{self.seq_len},{self.n_vars}], "
                f"got {tuple(x_multi.shape)}"
            )
        x = x_multi.float()
        h = self.channel_encoder(
            x.transpose(1, 2).contiguous().reshape(B * n_vars, length)
        ).reshape(B, n_vars, self.hidden)
        o3_ctx = h[:, self.o3_idx, :]
        pos_score = self.pos_score(h).squeeze(-1)
        neg_score = self.neg_score(h).squeeze(-1)

        if self.relation_mode == "pearson":
            relation = self.pearson(x, self.o3_idx)
            pos_logits = (
                pos_score
                + self.corr_temperature * relation.clamp(min=0.0)
            )
            neg_logits = (
                neg_score
                + self.corr_temperature * (-relation).clamp(min=0.0)
            )
        elif self.relation_mode == "lagged_corr":
            relation = self.lagged_correlation(x)
            pos_logits = (
                pos_score
                + self.corr_temperature * relation.clamp(min=0.0)
            )
            neg_logits = (
                neg_score
                + self.corr_temperature * (-relation).clamp(min=0.0)
            )
        elif self.relation_mode == "mutual_information":
            relation = self.mi_table[city_id].float()
            pos_logits = (
                pos_score + self.corr_temperature * relation
            )
            neg_logits = (
                neg_score + self.corr_temperature * relation
            )
        elif self.relation_mode == "learned_attention":
            query = self.query_proj(o3_ctx).unsqueeze(1)
            keys = self.key_proj(h)
            relation = (
                (query * keys).sum(dim=-1)
                / math.sqrt(float(self.hidden))
            ).tanh()
            pos_logits = (
                pos_score + self.corr_temperature * relation
            )
            neg_logits = (
                neg_score - self.corr_temperature * relation
            )
        else:
            raise ValueError(self.relation_mode)

        relation = relation.clone()
        relation[:, self.o3_idx] = 0.0
        pos_logits = self.mask_target_logits(pos_logits)
        neg_logits = self.mask_target_logits(neg_logits)
        pos_w = torch.softmax(pos_logits, dim=-1)
        neg_w = torch.softmax(neg_logits, dim=-1)
        pos_ctx = torch.sum(pos_w.unsqueeze(-1) * h, dim=1)
        neg_ctx = torch.sum(neg_w.unsqueeze(-1) * h, dim=1)
        relation_ctx = self.relation_proj(relation)
        return self.fusion(torch.cat(
            [pos_ctx, neg_ctx, o3_ctx, relation_ctx], dim=-1
        ))


class Reviewer45Forecaster(nn.Module):
    def __init__(
        self,
        *,
        variant: str,
        timer: Optional[nn.Module],
        n_vars: int,
        n_cities: int,
        InputAdapter,
        time_feature_dim,
        input_hidden: int,
        output_hidden: int,
        context_hidden: int,
        emb_dim: int,
        city_emb_dim: int,
        dropout: float,
        o3_idx: int,
        corr_temperature: float,
        lag_max: int,
        lag_step: int,
        mi_table: Optional[np.ndarray],
        climatology: Optional[np.ndarray],
        linear_w: Optional[np.ndarray],
        linear_b: Optional[np.ndarray],
        init_gate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.spec = VARIANT_SPECS[variant]
        self.timer = timer
        self.o3_idx = int(o3_idx)
        self.quantile = bool(self.spec["quantile"])
        self.full_adapter = bool(self.spec["full_adapter"])
        self.timer_mode = str(self.spec["timer_mode"])

        if climatology is None:
            climatology = np.zeros((n_cities, 12, 24), dtype=np.float32)
        if linear_w is None:
            linear_w = np.zeros((L, H), dtype=np.float32)
        if linear_b is None:
            linear_b = np.zeros((H,), dtype=np.float32)
        self.register_buffer(
            "climatology",
            torch.tensor(climatology, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "linear_w",
            torch.tensor(linear_w, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "linear_b",
            torch.tensor(linear_b, dtype=torch.float32),
            persistent=True,
        )

        if self.timer is not None:
            if self.timer_mode == "full":
                for p in self.timer.parameters():
                    p.requires_grad_(True)
            elif self.timer_mode == "lora":
                for name, p in self.timer.named_parameters():
                    p.requires_grad_(
                        name.endswith("lora_A")
                        or name.endswith("lora_B")
                    )
            else:
                for p in self.timer.parameters():
                    p.requires_grad_(False)
            if hasattr(self.timer.config, "use_cache"):
                self.timer.config.use_cache = False

        if not self.full_adapter:
            self.gamma = nn.Parameter(torch.ones(H))
            self.beta = nn.Parameter(torch.zeros(H))
            self.input_adapter = None
            self.context_encoder = None
            self.var_emb = None
            self.city_emb = None
            self.output_net = None
            self.output_gate = None
            return

        self.gamma = nn.Embedding(n_vars, H)
        self.beta = nn.Embedding(n_vars, H)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

        self.input_adapter = InputAdapter(
            seq_len=L,
            n_cities=n_cities,
            city_emb_dim=city_emb_dim,
            hidden=input_hidden,
            dropout=dropout,
            init_gate=init_gate,
        )
        self.context_encoder = RelationContextEncoder(
            seq_len=L,
            n_vars=n_vars,
            n_cities=n_cities,
            hidden=context_hidden,
            dropout=dropout,
            o3_idx=o3_idx,
            corr_temperature=corr_temperature,
            relation_mode=str(self.spec["relation_mode"]),
            lag_max=lag_max,
            lag_step=lag_step,
            mi_table=mi_table,
        )
        self.var_emb = nn.Embedding(n_vars, emb_dim)
        self.city_emb = nn.Embedding(n_cities, city_emb_dim)
        in_dim = (
            H + L + L + H * time_feature_dim("both")
            + context_hidden + emb_dim + city_emb_dim
        )
        output_dim = H * len(QUANTILES) if self.quantile else H
        self.output_net = nn.Sequential(
            nn.Linear(in_dim, output_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_hidden, output_dim),
        )
        self.output_gate = nn.Parameter(
            torch.tensor(float(init_gate))
        )
        nn.init.zeros_(self.output_net[-1].weight)
        nn.init.zeros_(self.output_net[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.timer is None:
            return self
        if self.timer_mode == "full":
            self.timer.train(mode)
        elif self.timer_mode == "lora":
            self.timer.eval()
            for module in self.timer.modules():
                if isinstance(module, LoRALinear):
                    module.train(mode)
        else:
            self.timer.eval()
        return self

    @staticmethod
    def decode_time_bins(time_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tf = time_feat.view(time_feat.shape[0], H, 4)
        hour_angle = torch.atan2(tf[..., 0], tf[..., 1])
        hour_angle = torch.remainder(hour_angle, 2.0 * math.pi)
        hour = torch.remainder(
            torch.round(hour_angle * 24.0 / (2.0 * math.pi)).long(),
            24,
        )
        doy_angle = torch.atan2(tf[..., 2], tf[..., 3])
        doy_angle = torch.remainder(doy_angle, 2.0 * math.pi)
        doy = doy_angle * 366.0 / (2.0 * math.pi)
        season_bin = torch.clamp(
            torch.floor(doy * 12.0 / 366.0).long(),
            min=0, max=11,
        )
        return season_bin, hour

    def base_forecast(
        self,
        *,
        x: torch.Tensor,
        x_adapt: torch.Tensor,
        time_feat: torch.Tensor,
        city_id: torch.Tensor,
    ) -> torch.Tensor:
        if self.variant == "n2_climatology":
            season_bin, hour = self.decode_time_bins(time_feat)
            city = city_id[:, None].expand(-1, H)
            return self.climatology[city, season_bin, hour]
        if self.variant == "n3_linear_ar":
            # Frozen classical AR prior: InputAdapter features may be used by
            # the residual head, but cannot manipulate the AR base forecast.
            return x @ self.linear_w + self.linear_b
        if self.timer is None:
            raise RuntimeError(f"Timer missing for {self.variant}")
        timer_input = x if not self.full_adapter else x_adapt
        return self.timer(timer_input).logits[:, :H].float()

    def forward(
        self,
        x: torch.Tensor,
        x_multi: torch.Tensor,
        time_feat: torch.Tensor,
        var_idx: torch.Tensor,
        city_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not self.full_adapter:
            base = self.base_forecast(
                x=x, x_adapt=x, time_feat=time_feat, city_id=city_id
            )
            point = (
                self.gamma.unsqueeze(0) * base
                + self.beta.unsqueeze(0)
            )
            return {"point": point, "base": base}

        x_adapt = self.input_adapter(x, city_id)
        base = self.base_forecast(
            x=x, x_adapt=x_adapt,
            time_feat=time_feat, city_id=city_id,
        )
        calibrated = self.gamma(var_idx) * base + self.beta(var_idx)
        context = self.context_encoder(x_multi, city_id)
        features = torch.cat([
            calibrated,
            x,
            x_adapt,
            time_feat,
            context,
            self.var_emb(var_idx),
            self.city_emb(city_id),
        ], dim=-1)

        if self.quantile:
            delta = self.output_net(features).view(
                x.shape[0], H, len(QUANTILES)
            )
            raw_quantiles = (
                calibrated.unsqueeze(-1)
                + self.output_gate * delta
            )
            rearranged_quantiles, _ = torch.sort(
                raw_quantiles, dim=-1
            )
            return {
                # Point forecasting always uses the network's directly learned
                # q0.50 channel, never the median after rearrangement.
                "point": raw_quantiles[..., 2],
                "base": base,
                "quantiles": raw_quantiles,
                "raw_quantiles": raw_quantiles,
                "rearranged_quantiles": rearranged_quantiles,
            }

        point = calibrated + self.output_gate * self.output_net(features)
        return {"point": point, "base": base}


def pinball_loss(pred_q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    target = y.unsqueeze(-1)
    losses = []
    for index, tau in enumerate(QUANTILES):
        error = target[..., 0] - pred_q[..., index]
        losses.append(
            torch.maximum(tau * error, (tau - 1.0) * error).mean()
        )
    return torch.stack(losses).mean()


def crossing_penalty(pred_q: torch.Tensor) -> torch.Tensor:
    return F.relu(pred_q[..., :-1] - pred_q[..., 1:]).mean()


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    names = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    state = model.state_dict()
    return {
        name: state[name].detach().cpu()
        for name in sorted(names)
        if name in state
    }


def load_trainable_state(model: nn.Module,
                         state: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    trainable = {
        name for name, p in model.named_parameters()
        if p.requires_grad
    }
    absent = sorted(trainable - set(state))
    if absent:
        raise RuntimeError(
            f"Checkpoint lacks trainable keys: {absent[:30]}"
        )


def parameter_manifest(
    model: nn.Module,
    *,
    variant: str,
    lora_targets: list[str],
    optimizer_groups: list[dict[str, Any]],
    checkpointing: dict[str, Any],
) -> dict[str, Any]:
    module_counts: dict[str, int] = defaultdict(int)
    trainable_names = []
    total = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        n = int(parameter.numel())
        total += n
        if parameter.requires_grad:
            trainable += n
            trainable_names.append(name)
            top = name.split(".", 1)[0]
            module_counts[top] += n
    return {
        "variant": variant,
        "paper_name": VARIANT_SPECS[variant]["paper_name"],
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "trainable_ratio": trainable / max(total, 1),
        "trainable_by_top_module": dict(sorted(module_counts.items())),
        "trainable_parameter_names": trainable_names,
        "lora_target_modules": lora_targets,
        "optimizer_groups": optimizer_groups,
        "gradient_checkpointing": checkpointing,
    }


def create_optimizer(model: nn.Module, args: argparse.Namespace):
    external, lora, backbone = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("timer.") and (
            ".lora_A" in name or ".lora_B" in name
        ):
            lora.append(parameter)
        elif name.startswith("timer."):
            backbone.append(parameter)
        else:
            external.append(parameter)
    groups = []
    report = []
    for label, params, lr in [
        ("external_adapter", external, args.adapter_lr),
        ("lora", lora, args.lora_lr),
        ("timer_backbone", backbone, args.backbone_lr),
    ]:
        if params:
            groups.append({
                "params": params,
                "lr": lr,
                "weight_decay": args.weight_decay,
            })
            report.append({
                "name": label,
                "lr": lr,
                "weight_decay": args.weight_decay,
                "params": int(sum(p.numel() for p in params)),
            })
    if not groups:
        raise RuntimeError("No trainable parameters")
    use_fused = (
        args.variant == "a2_full_finetune"
        and torch.cuda.is_available()
    )
    return torch.optim.AdamW(
        groups,
        fused=use_fused,
    ), report


def extract_station_ids(batch: dict[str, Any]) -> Optional[np.ndarray]:
    for key in (
        "station_id", "station_idx", "site_id", "site_idx",
        "station_code", "station",
    ):
        if key in batch:
            return to_numpy(batch[key]).reshape(-1)
    return None


def run_epoch(
    model: Reviewer45Forecaster,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer,
    args: argparse.Namespace,
    collect: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    overall = SumMetric()
    base_metric = SumMetric()
    by_city: dict[int, SumMetric] = defaultdict(SumMetric)
    by_station: dict[str, SumMetric] = defaultdict(SumMetric)
    h_abs = np.zeros(H, dtype=np.float64)
    h_sq = np.zeros(H, dtype=np.float64)
    h_n = np.zeros(H, dtype=np.int64)
    preds, bases, truths, city_arrays = [], [], [], []
    station_arrays: list[np.ndarray] = []
    raw_quantile_arrays: list[np.ndarray] = []
    rearranged_quantile_arrays: list[np.ndarray] = []
    loss_sum = 0.0
    batches = 0
    accumulation = max(1, int(args.grad_accum))
    if training:
        optimizer.zero_grad(set_to_none=True)

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

        amp_enabled = bool(
            training and args.amp_bf16 and device.type == "cuda"
        )
        amp_context = (
            torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )
            if amp_enabled else nullcontext()
        )
        with torch.set_grad_enabled(training):
            with amp_context:
                output = model(
                    x, x_multi, time_feat, var_idx, city_id,
                )
                point = output["point"]
                if model.quantile:
                    pred_q = output["quantiles"]
                    loss = (
                        pinball_loss(pred_q, y_std)
                        + args.lambda_mid_mae
                        * F.l1_loss(point, y_std)
                        + args.lambda_mid_mse
                        * F.mse_loss(point, y_std)
                        + args.lambda_cross
                        * crossing_penalty(pred_q)
                    )
                else:
                    loss = (
                        F.l1_loss(point, y_std)
                        + args.mse_weight * F.mse_loss(point, y_std)
                    )
                backward_loss = loss / accumulation
            if training:
                backward_loss.backward()

        if training and (
            step % accumulation == 0 or step == len(loader)
        ):
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_sum += float(loss.detach().float().cpu())
        batches += 1
        point_np = point.detach().float().cpu().numpy()
        base_np = output["base"].detach().float().cpu().numpy()
        pred_phys = point_np * std + mu
        base_phys = base_np * std + mu
        overall.update(pred_phys, y_phys)
        base_metric.update(base_phys, y_phys)

        if collect:
            city_np = to_numpy(batch["city_id"]).astype(np.int64)
            for cid in np.unique(city_np):
                mask = city_np == cid
                by_city[int(cid)].update(
                    pred_phys[mask], y_phys[mask]
                )
            station_np = extract_station_ids(batch)
            if station_np is not None:
                for station in np.unique(station_np):
                    mask = station_np == station
                    by_station[str(station)].update(
                        pred_phys[mask], y_phys[mask]
                    )
                station_arrays.append(station_np)
            diff = (
                pred_phys.astype(np.float64)
                - y_phys.astype(np.float64)
            )
            h_abs += np.abs(diff).sum(axis=0)
            h_sq += np.square(diff).sum(axis=0)
            h_n += diff.shape[0]
            preds.append(pred_phys.astype(np.float32, copy=False))
            bases.append(base_phys.astype(np.float32, copy=False))
            truths.append(y_phys)
            city_arrays.append(city_np)
            if model.quantile:
                raw_std = (
                    output["raw_quantiles"]
                    .detach().float().cpu().numpy()
                )
                rearranged_std = (
                    output["rearranged_quantiles"]
                    .detach().float().cpu().numpy()
                )
                raw_phys = (
                    raw_std * std[:, :, None] + mu[:, :, None]
                )
                rearranged_phys = (
                    rearranged_std * std[:, :, None] + mu[:, :, None]
                )
                raw_quantile_arrays.append(
                    raw_phys.astype(np.float32, copy=False)
                )
                rearranged_quantile_arrays.append(
                    rearranged_phys.astype(np.float32, copy=False)
                )

        if training and (step == 1 or step % args.log_every == 0):
            result = overall.result()
            print(
                f"step={step}/{len(loader)} "
                f"loss={loss_sum/max(batches,1):.6f} "
                f"MAE={result['MAE']:.4f} "
                f"RMSE={result['RMSE']:.4f}",
                flush=True,
            )

    metrics = {
        "loss": loss_sum / max(batches, 1),
        **overall.result(),
    }
    detail: dict[str, Any] = {
        "base_metrics": base_metric,
        "by_city": by_city,
        "by_station": by_station,
        "h_abs": h_abs,
        "h_sq": h_sq,
        "h_n": h_n,
    }
    if collect:
        detail.update({
            "pred": np.concatenate(preds, axis=0),
            "base": np.concatenate(bases, axis=0),
            "true": np.concatenate(truths, axis=0),
            "city_id": np.concatenate(city_arrays, axis=0),
        })
        if station_arrays:
            detail["station_id"] = np.concatenate(
                station_arrays, axis=0
            )
        if raw_quantile_arrays:
            detail["raw_quantiles"] = np.concatenate(
                raw_quantile_arrays, axis=0
            )
            detail["rearranged_quantiles"] = np.concatenate(
                rearranged_quantile_arrays, axis=0
            )
    return metrics, detail


def canonical_true_path(root: Path) -> Path:
    candidates = [
        root / "experiments/results/02_global_baselines/"
        "00_global11_v2_array_archive/_canonical/true.npy",
        root / "experiments/results/02_global_baselines/"
        "00_global11_v2_array_archive/TSMixer/seed2026/true.npy",
        root / "experiments/results/03_zeroshot_baselines_v2_archive/"
        "Timer/true.npy",
        root / "experiments/data_v2/indices/test_targets.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def save_true_array(
    true: np.ndarray,
    out: Path,
    canonical_path: Path,
    formal: bool,
) -> str:
    if out.exists() or out.is_symlink():
        out.unlink()
    if formal and canonical_path.exists():
        canonical = np.load(canonical_path, mmap_mode="r")
        if (
            canonical.shape != true.shape
            or not np.array_equal(
                np.asarray(canonical), true, equal_nan=True
            )
        ):
            raise RuntimeError(
                "Run true differs from canonical revision-v2 true.npy"
            )
        try:
            os.link(canonical_path.resolve(), out)
            return "hardlink_to_canonical"
        except OSError:
            try:
                rel = os.path.relpath(
                    canonical_path.resolve(), out.parent
                )
                os.symlink(rel, out)
                return "symlink_to_canonical"
            except OSError:
                np.save(out, true, allow_pickle=False)
                return "copied_canonical"
    np.save(out, true, allow_pickle=False)
    return "saved_subset"


def save_outputs(
    out: Path,
    *,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    detail: dict[str, Any],
    cities: list[str],
    manifest: dict[str, Any],
    best_epoch: int,
    best_val_rmse: float,
    train_seconds: float,
    test_seconds: float,
    peak_allocated_mb: float,
    peak_reserved_mb: float,
    device_name: str,
) -> None:
    base = detail["base_metrics"].result()
    row = {
        **metrics,
        "variant": args.variant,
        "paper_name": VARIANT_SPECS[args.variant]["paper_name"],
        "group": VARIANT_SPECS[args.variant]["group"],
        "seed": args.seed,
        "protocol": "revision-v2",
        "best_epoch": best_epoch,
        "best_val_RMSE": best_val_rmse,
        "base_MAE": base["MAE"],
        "base_MSE": base["MSE"],
        "base_RMSE": base["RMSE"],
        "train_seconds": train_seconds,
        "test_seconds": test_seconds,
        "peak_gpu_memory_allocated_mb": peak_allocated_mb,
        "peak_gpu_memory_reserved_mb": peak_reserved_mb,
        "device_name": device_name,
        "total_params": manifest["total_params"],
        "trainable_params": manifest["trainable_params"],
        "trainable_ratio": manifest["trainable_ratio"],
        "physical_batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "amp_bf16_training": bool(args.amp_bf16),
    }
    pd.DataFrame([row]).to_csv(
        out / "metrics_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    city_rows = []
    for cid, accumulator in sorted(detail["by_city"].items()):
        city_rows.append({
            "city_id": cid,
            "city": cities[cid] if 0 <= cid < len(cities) else str(cid),
            **accumulator.result(),
        })
    pd.DataFrame(city_rows).to_csv(
        out / "metrics_by_city.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_rows = []
    for horizon in range(H):
        count = int(detail["h_n"][horizon])
        mse = detail["h_sq"][horizon] / count if count else math.nan
        horizon_rows.append({
            "horizon": horizon + 1,
            "MAE": (
                detail["h_abs"][horizon] / count
                if count else math.nan
            ),
            "MSE": mse,
            "RMSE": math.sqrt(mse) if count else math.nan,
            "count": count,
        })
    pd.DataFrame(horizon_rows).to_csv(
        out / "metrics_by_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    station_rows = [
        {"station_id": station, **acc.result()}
        for station, acc in sorted(detail["by_station"].items())
    ]
    pd.DataFrame(station_rows).to_csv(
        out / "metrics_by_station.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not args.save_arrays:
        return
    pred = np.asarray(detail["pred"], dtype=np.float32)
    base_pred = np.asarray(detail["base"], dtype=np.float32)
    true = np.asarray(detail["true"], dtype=np.float32)
    city_id = np.asarray(detail["city_id"], dtype=np.int16)
    np.save(out / "pred.npy", pred, allow_pickle=False)
    np.save(out / "base_pred.npy", base_pred, allow_pickle=False)
    np.save(out / "city_id.npy", city_id, allow_pickle=False)
    if "station_id" in detail:
        np.save(
            out / "station_id.npy",
            detail["station_id"],
            allow_pickle=False,
        )

    formal = (
        args.max_test_samples == 0
        and pred.shape == true.shape == (106_200, H)
    )
    true_storage = save_true_array(
        true, out / "true.npy",
        canonical_true_path(args.root), formal,
    )
    array_manifest: dict[str, Any] = {
        "state": "complete",
        "protocol": "revision-v2",
        "variant": args.variant,
        "seed": args.seed,
        "pred_shape": list(pred.shape),
        "base_pred_shape": list(base_pred.shape),
        "true_shape": list(true.shape),
        "formal_full_test": formal,
        "nonfinite_pred": int((~np.isfinite(pred)).sum()),
        "nonfinite_base_pred": int((~np.isfinite(base_pred)).sum()),
        "nonfinite_true": int((~np.isfinite(true)).sum()),
        "true_storage": true_storage,
        "canonical_true": str(canonical_true_path(args.root)),
    }
    if "raw_quantiles" in detail:
        raw_quantiles = np.asarray(
            detail["raw_quantiles"], dtype=np.float32
        )
        rearranged_quantiles = np.asarray(
            detail["rearranged_quantiles"], dtype=np.float32
        )
        np.save(
            out / "quantile_pred_raw.npy",
            raw_quantiles,
            allow_pickle=False,
        )
        np.save(
            out / "quantile_pred_rearranged.npy",
            rearranged_quantiles,
            allow_pickle=False,
        )
        labels = ["q05", "q10", "q50", "q90", "q95"]
        for index, label in enumerate(labels):
            np.save(
                out / f"raw_pred_{label}.npy",
                raw_quantiles[..., index],
                allow_pickle=False,
            )
            np.save(
                out / f"rearranged_pred_{label}.npy",
                rearranged_quantiles[..., index],
                allow_pickle=False,
            )

        raw_violation = (
            raw_quantiles[..., :-1] - raw_quantiles[..., 1:]
        )
        raw_mask = raw_violation > 0
        rearranged_violation = (
            rearranged_quantiles[..., :-1]
            - rearranged_quantiles[..., 1:]
        )
        rearranged_mask = rearranged_violation > 0
        raw_pairs = int(raw_mask.sum())
        rearranged_pairs = int(rearranged_mask.sum())
        total_pairs = int(raw_mask.size)
        raw_any = int(raw_mask.any(axis=-1).sum())
        total_positions = int(raw_mask.shape[0] * raw_mask.shape[1])
        raw_positive = raw_violation[raw_mask]
        rearranged_positive = rearranged_violation[rearranged_mask]
        array_manifest.update({
            "quantile_shape": list(raw_quantiles.shape),
            "quantiles": QUANTILES,
            "point_semantics": "raw learned q0.50 channel",
            "interval_semantics": (
                "use rearranged quantiles for monotone intervals; "
                "raw quantiles are retained for crossing diagnostics"
            ),
            "nonfinite_raw_quantiles":
                int((~np.isfinite(raw_quantiles)).sum()),
            "nonfinite_rearranged_quantiles":
                int((~np.isfinite(rearranged_quantiles)).sum()),
            "raw_q50_equals_pred": bool(np.array_equal(
                raw_quantiles[..., 2], pred, equal_nan=True
            )),
            "raw_crossing_pairs": raw_pairs,
            "raw_crossing_pair_frequency":
                raw_pairs / max(total_pairs, 1),
            "raw_positions_with_any_crossing": raw_any,
            "raw_any_crossing_frequency":
                raw_any / max(total_positions, 1),
            "raw_mean_crossing_magnitude": (
                float(raw_positive.mean())
                if raw_positive.size else 0.0
            ),
            "raw_max_crossing_magnitude": (
                float(raw_positive.max())
                if raw_positive.size else 0.0
            ),
            "rearranged_crossing_pairs": rearranged_pairs,
            "rearranged_crossing_pair_frequency":
                rearranged_pairs / max(total_pairs, 1),
            "rearranged_mean_crossing_magnitude": (
                float(rearranged_positive.mean())
                if rearranged_positive.size else 0.0
            ),
            "rearranged_max_crossing_magnitude": (
                float(rearranged_positive.max())
                if rearranged_positive.size else 0.0
            ),
        })
    (out / "array_manifest.json").write_text(
        json.dumps(array_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_assets(root: Path, mode: str, variant: str,
                n_cities: int, n_vars: int):
    base = asset_root(root, mode)
    climatology = linear_w = linear_b = mi_table = None
    required = []
    if variant == "n2_climatology":
        required.append(base / "climatology/climatology_std.npy")
    if variant == "n3_linear_ar":
        required += [
            base / "linear_ar/W.npy",
            base / "linear_ar/b.npy",
        ]
    if variant == "r2_mutual_information":
        required.append(base / "mutual_information/mi_city_var.npy")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required reviewer45 assets are missing: " + ", ".join(missing)
        )
    if variant == "n2_climatology":
        climatology = np.load(required[0], allow_pickle=False)
        if climatology.shape != (n_cities, 12, 24):
            raise ValueError(f"Bad climatology shape {climatology.shape}")
    if variant == "n3_linear_ar":
        linear_w = np.load(
            base / "linear_ar/W.npy", allow_pickle=False
        )
        linear_b = np.load(
            base / "linear_ar/b.npy", allow_pickle=False
        )
        if linear_w.shape != (L, H) or linear_b.shape != (H,):
            raise ValueError(
                f"Bad Linear AR shapes: {linear_w.shape}, {linear_b.shape}"
            )
    if variant == "r2_mutual_information":
        mi_table = np.load(required[0], allow_pickle=False)
        if mi_table.shape != (n_cities, n_vars):
            raise ValueError(f"Bad MI shape {mi_table.shape}")
    return climatology, linear_w, linear_b, mi_table


def run_main(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    args.root = root
    out = (
        args.out_dir.expanduser().resolve()
        if args.out_dir.is_absolute()
        else (root / args.out_dir).resolve()
    )
    out.mkdir(parents=True, exist_ok=True)
    source = import_file(
        root / "experiments/scripts/train/"
        "train_attribution_reference.py",
        "_reviewer45_canonical_source",
    )
    cipda, cora = source.load_project_modules(
        root,
        root / "experiments/data_v2/indices",
        root / "experiments/data_v2/"
        "city_var_stats_train_only.json",
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

    climatology, linear_w, linear_b, mi_table = load_assets(
        root, args.asset_mode, args.variant,
        len(train_ds.cities), len(train_ds.variables),
    )

    timer = None
    lora_targets: list[str] = []
    checkpointing = {
        "requested": bool(args.gradient_checkpointing),
        "enabled": False,
        "method": None,
    }
    if VARIANT_SPECS[args.variant]["needs_timer"]:
        print(f"Loading Timer from {cipda.MODEL_PATH}", flush=True)
        timer = AutoModelForCausalLM.from_pretrained(
            cipda.MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        if args.variant == "a1_lora_r8":
            lora_targets = inject_lora(
                timer,
                args.lora_rank,
                args.lora_alpha,
                args.lora_dropout,
            )
            print(
                f"Injected LoRA into {len(lora_targets)} modules:",
                flush=True,
            )
            for name in lora_targets:
                print(f"  {name}", flush=True)
        if (
            args.variant == "a2_full_finetune"
            and args.gradient_checkpointing
        ):
            checkpointing = enable_gradient_checkpointing(timer)

    model = Reviewer45Forecaster(
        variant=args.variant,
        timer=timer,
        n_vars=len(train_ds.variables),
        n_cities=len(train_ds.cities),
        InputAdapter=cora.InputAdapter,
        time_feature_dim=cora.time_feature_dim,
        input_hidden=args.input_hidden,
        output_hidden=args.output_hidden,
        context_hidden=args.context_hidden,
        emb_dim=args.emb_dim,
        city_emb_dim=args.city_emb_dim,
        dropout=args.dropout,
        o3_idx=train_ds.o3_idx,
        corr_temperature=args.corr_temperature,
        lag_max=args.lag_max,
        lag_step=args.lag_step,
        mi_table=mi_table,
        climatology=climatology,
        linear_w=linear_w,
        linear_b=linear_b,
    ).to(device)

    optimizer, optimizer_report = create_optimizer(model, args)
    manifest = parameter_manifest(
        model,
        variant=args.variant,
        lora_targets=lora_targets,
        optimizer_groups=optimizer_report,
        checkpointing=checkpointing,
    )
    (out / "trainable_parameter_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    device_name = (
        torch.cuda.get_device_name(device)
        if cuda else f"CPU ({torch.get_num_threads()} threads)"
    )
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)

    print("=" * 104)
    print("[REVIEWER45 / REVISION-V2]")
    print(
        f"variant={args.variant} "
        f"paper_name={VARIANT_SPECS[args.variant]['paper_name']}"
    )
    print(
        f"seed={args.seed} device={device_name} "
        f"asset_mode={args.asset_mode}"
    )
    print(
        f"train={len(train_ds):,} val={len(val_ds):,} "
        f"test={len(test_ds):,}"
    )
    print(
        f"epochs={args.epochs} patience={args.patience} "
        f"micro_batch={args.batch_size} accum={args.grad_accum} "
        f"effective_batch={args.batch_size*args.grad_accum}"
    )
    print(
        f"params total={manifest['total_params']:,} "
        f"trainable={manifest['trainable_params']:,} "
        f"ratio={manifest['trainable_ratio']:.6%}"
    )
    print("=" * 104)

    args_dict = vars(args).copy()
    for key, value in list(args_dict.items()):
        if isinstance(value, Path):
            args_dict[key] = str(value)
    args_dict.update({
        "protocol": "revision-v2",
        "quantiles": QUANTILES if model.quantile else None,
        "parameter_manifest":
            str(out / "trainable_parameter_manifest.json"),
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

    history = []
    best_rmse = float("inf")
    bad_epochs = 0
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} train", flush=True)
        tr, _ = run_epoch(
            model, train_loader, device,
            optimizer=optimizer, args=args, collect=False,
        )
        print(f"Epoch {epoch}/{args.epochs} val", flush=True)
        va, _ = run_epoch(
            model, val_loader, device,
            optimizer=None, args=args, collect=False,
        )
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in va.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            out / "history.csv", index=False, encoding="utf-8-sig"
        )
        checkpoint = {
            "epoch": epoch,
            "trainable_state": trainable_state(model),
            "args": args_dict,
            "variables": train_ds.variables,
            "cities": train_ds.cities,
            "val_metrics": va,
            "parameter_manifest": manifest,
        }
        torch.save(checkpoint, out / "last.pt")
        if va["RMSE"] < best_rmse - 1e-12:
            best_rmse = float(va["RMSE"])
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
            if args.patience > 0 and bad_epochs >= args.patience:
                print(f"Early stopping at epoch={epoch}", flush=True)
                break

    if cuda:
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - train_start
    try:
        best = torch.load(
            out / "best.pt", map_location="cpu", weights_only=False
        )
    except TypeError:
        best = torch.load(out / "best.pt", map_location="cpu")
    load_trainable_state(model, best["trainable_state"])

    print("\nFinal revision-v2 test", flush=True)
    test_start = time.perf_counter()
    te, detail = run_epoch(
        model, test_loader, device,
        optimizer=None, args=args, collect=True,
    )
    if cuda:
        torch.cuda.synchronize(device)
    test_seconds = time.perf_counter() - test_start
    peak_allocated = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if cuda else 0.0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) / 1024**2
        if cuda else 0.0
    )

    save_outputs(
        out,
        args=args,
        metrics=te,
        detail=detail,
        cities=train_ds.cities,
        manifest=manifest,
        best_epoch=int(best["epoch"]),
        best_val_rmse=best_rmse,
        train_seconds=train_seconds,
        test_seconds=test_seconds,
        peak_allocated_mb=peak_allocated,
        peak_reserved_mb=peak_reserved,
        device_name=device_name,
    )
    summary = {
        "state": "complete",
        "variant": args.variant,
        "paper_name": VARIANT_SPECS[args.variant]["paper_name"],
        "group": VARIANT_SPECS[args.variant]["group"],
        "seed": args.seed,
        "protocol": "revision-v2",
        "best_epoch": int(best["epoch"]),
        "best_val_RMSE": best_rmse,
        "epochs_ran": len(history),
        "train_seconds": train_seconds,
        "test_seconds": test_seconds,
        "peak_gpu_memory_allocated_mb": peak_allocated,
        "peak_gpu_memory_reserved_mb": peak_reserved,
        "total_params": manifest["total_params"],
        "trainable_params": manifest["trainable_params"],
        "trainable_ratio": manifest["trainable_ratio"],
        **te,
        "base_MAE": detail["base_metrics"].result()["MAE"],
        "base_RMSE": detail["base_metrics"].result()["RMSE"],
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
