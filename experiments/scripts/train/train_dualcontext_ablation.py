#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive DualContext ablations under the canonical revision-v2 protocol.

Nine unique trainable variants are implemented.

Overall structure:
  c0_no_context
  c1_o3_only
  c2_naive_multivariate
  c3_dual_no_corr
  c4_full_dualcontext

Branch / gate:
  b0_positive_only
  b1_negative_only
  b2_equal_fusion
  b3_shared_gate

B4 independent-gates is exactly C4 and is not trained a second time.

The frozen Timer, InputAdapter, future Hour+DOY features, city/variable
embeddings, horizon-wise affine calibration, output residual head, optimizer,
loss, split and train-only scaler are held fixed. Only the context encoder
variant changes.

The Full encoder follows the project's established implementation:
- shared variable-wise channel encoder;
- O3-centered Pearson coefficients from the 96-hour history only, with the
  target O3 channel excluded from covariate competition;
- separate positive and negative covariate relevance distributions;
- a dedicated O3 context and a projected covariate-relation vector;
- learned fusion to a fixed 128-dimensional context.

Diagnostics are collected only during final test evaluation.
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
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

L, H = 96, 48

VARIANT_SPECS: dict[str, dict[str, str]] = {
    "c0_no_context": {
        "section": "overall",
        "paper_name": "No Context",
        "description": "Full adapter backbone with a fixed zero context vector.",
    },
    "c1_o3_only": {
        "section": "overall",
        "paper_name": "O3-only Context",
        "description": "Only the historical O3 channel is encoded in the context branch.",
    },
    "c2_naive_multivariate": {
        "section": "overall",
        "paper_name": "Naive Multivariate",
        "description": "Single learned multivariate relevance distribution without Pearson polarity.",
    },
    "c3_dual_no_corr": {
        "section": "overall",
        "paper_name": "DualContext w/o Pearson",
        "description": "Separate positive/negative learned branches with all Pearson cues removed.",
    },
    "c4_full_dualcontext": {
        "section": "overall",
        "paper_name": "Full DualContext",
        "description": "Pearson-guided positive/negative branches with independent relevance gates.",
    },
    "b0_positive_only": {
        "section": "branch_gate",
        "paper_name": "Positive-only",
        "description": "Full Pearson design with the negative branch set to zero.",
    },
    "b1_negative_only": {
        "section": "branch_gate",
        "paper_name": "Negative-only",
        "description": "Full Pearson design with the positive branch set to zero.",
    },
    "b2_equal_fusion": {
        "section": "branch_gate",
        "paper_name": "Equal Fusion",
        "description": "Positive/negative branch contexts are averaged before fusion.",
    },
    "b3_shared_gate": {
        "section": "branch_gate",
        "paper_name": "Shared Gate",
        "description": "One shared variable relevance distribution replaces independent branches.",
    },
}

ALL_VARIANTS = list(VARIANT_SPECS)
EXPECTED_SHAPE = (106_200, H)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path,
                   default=Path(__file__).resolve().parents[3])
    p.add_argument("--variant", required=True, choices=ALL_VARIANTS)
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


def import_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import source module: {path}")
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
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
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
        self.abs_sum += float(np.abs(diff).sum(dtype=np.float64))
        self.sq_sum += float(np.square(diff).sum(dtype=np.float64))
        self.count += int(diff.size)

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


class DiagnosticAccumulator:
    """Streaming aggregate for input-window branch diagnostics."""

    def __init__(self, n_vars: int, city_names: list[str]) -> None:
        self.n_vars = n_vars
        self.city_names = city_names
        self.reset()

    def reset(self) -> None:
        V = self.n_vars
        self.n = 0
        self.corr_sum = np.zeros(V, dtype=np.float64)
        self.corr_sq = np.zeros(V, dtype=np.float64)
        self.corr_abs = np.zeros(V, dtype=np.float64)
        self.corr_pos = np.zeros(V, dtype=np.int64)
        self.corr_neg = np.zeros(V, dtype=np.int64)
        self.pos_w_sum = np.zeros(V, dtype=np.float64)
        self.neg_w_sum = np.zeros(V, dtype=np.float64)
        self.shared_w_sum = np.zeros(V, dtype=np.float64)
        self.pos_entropy_sum = 0.0
        self.neg_entropy_sum = 0.0
        self.shared_entropy_sum = 0.0
        self.pos_norm_sum = 0.0
        self.neg_norm_sum = 0.0
        self.final_norm_sum = 0.0
        self.city: dict[int, dict[str, float]] = defaultdict(
            lambda: {
                "n": 0.0,
                "pos_norm_sum": 0.0,
                "neg_norm_sum": 0.0,
                "final_norm_sum": 0.0,
                "pos_entropy_sum": 0.0,
                "neg_entropy_sum": 0.0,
                "shared_entropy_sum": 0.0,
                "corr_abs_mean_sum": 0.0,
                "positive_variable_ratio_sum": 0.0,
                "negative_variable_ratio_sum": 0.0,
            }
        )

    @staticmethod
    def entropy(weights: torch.Tensor) -> torch.Tensor:
        w = weights.clamp_min(1e-12)
        return -(w * w.log()).sum(dim=-1)

    def update(self, diag: dict[str, torch.Tensor],
               city_id: torch.Tensor) -> None:
        corr = diag["corr"].detach().float().cpu().numpy()
        pos_w = diag["pos_w"].detach().float().cpu().numpy()
        neg_w = diag["neg_w"].detach().float().cpu().numpy()
        shared_w = diag["shared_w"].detach().float().cpu().numpy()
        pos_ctx = diag["pos_ctx"].detach().float().cpu()
        neg_ctx = diag["neg_ctx"].detach().float().cpu()
        final_ctx = diag["final_ctx"].detach().float().cpu()
        city = city_id.detach().cpu().numpy().astype(np.int64)

        batch_n = corr.shape[0]
        self.n += batch_n
        self.corr_sum += corr.sum(axis=0, dtype=np.float64)
        self.corr_sq += np.square(corr).sum(axis=0, dtype=np.float64)
        self.corr_abs += np.abs(corr).sum(axis=0, dtype=np.float64)
        self.corr_pos += (corr > 0).sum(axis=0)
        self.corr_neg += (corr < 0).sum(axis=0)
        self.pos_w_sum += pos_w.sum(axis=0, dtype=np.float64)
        self.neg_w_sum += neg_w.sum(axis=0, dtype=np.float64)
        self.shared_w_sum += shared_w.sum(axis=0, dtype=np.float64)

        pos_entropy = self.entropy(
            torch.from_numpy(pos_w)
        ).numpy()
        neg_entropy = self.entropy(
            torch.from_numpy(neg_w)
        ).numpy()
        shared_entropy = self.entropy(
            torch.from_numpy(shared_w)
        ).numpy()
        pos_norm = torch.linalg.vector_norm(
            pos_ctx, dim=-1
        ).numpy()
        neg_norm = torch.linalg.vector_norm(
            neg_ctx, dim=-1
        ).numpy()
        final_norm = torch.linalg.vector_norm(
            final_ctx, dim=-1
        ).numpy()

        self.pos_entropy_sum += float(pos_entropy.sum())
        self.neg_entropy_sum += float(neg_entropy.sum())
        self.shared_entropy_sum += float(shared_entropy.sum())
        self.pos_norm_sum += float(pos_norm.sum())
        self.neg_norm_sum += float(neg_norm.sum())
        self.final_norm_sum += float(final_norm.sum())

        corr_abs_mean = np.abs(corr).mean(axis=1)
        pos_var_ratio = (corr > 0).mean(axis=1)
        neg_var_ratio = (corr < 0).mean(axis=1)
        for cid in np.unique(city):
            mask = city == cid
            d = self.city[int(cid)]
            n = int(mask.sum())
            d["n"] += n
            d["pos_norm_sum"] += float(pos_norm[mask].sum())
            d["neg_norm_sum"] += float(neg_norm[mask].sum())
            d["final_norm_sum"] += float(final_norm[mask].sum())
            d["pos_entropy_sum"] += float(pos_entropy[mask].sum())
            d["neg_entropy_sum"] += float(neg_entropy[mask].sum())
            d["shared_entropy_sum"] += float(
                shared_entropy[mask].sum()
            )
            d["corr_abs_mean_sum"] += float(
                corr_abs_mean[mask].sum()
            )
            d["positive_variable_ratio_sum"] += float(
                pos_var_ratio[mask].sum()
            )
            d["negative_variable_ratio_sum"] += float(
                neg_var_ratio[mask].sum()
            )

    def save(self, out_dir: Path, variable_names: list[str],
             variant: str) -> None:
        n = max(self.n, 1)
        mean = self.corr_sum / n
        var = np.maximum(self.corr_sq / n - np.square(mean), 0.0)

        variable_rows = []
        for idx, name in enumerate(variable_names):
            variable_rows.append({
                "variant": variant,
                "variable_id": idx,
                "variable": name,
                "corr_mean": mean[idx],
                "corr_std": math.sqrt(var[idx]),
                "corr_abs_mean": self.corr_abs[idx] / n,
                "corr_positive_ratio": self.corr_pos[idx] / n,
                "corr_negative_ratio": self.corr_neg[idx] / n,
                "positive_gate_mean": self.pos_w_sum[idx] / n,
                "negative_gate_mean": self.neg_w_sum[idx] / n,
                "shared_gate_mean": self.shared_w_sum[idx] / n,
                "samples": self.n,
            })
        pd.DataFrame(variable_rows).to_csv(
            out_dir / "variable_gate_and_correlation_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        gate_summary = [{
            "variant": variant,
            "samples": self.n,
            "positive_gate_entropy_mean": self.pos_entropy_sum / n,
            "negative_gate_entropy_mean": self.neg_entropy_sum / n,
            "shared_gate_entropy_mean": self.shared_entropy_sum / n,
            "positive_branch_norm_mean": self.pos_norm_sum / n,
            "negative_branch_norm_mean": self.neg_norm_sum / n,
            "final_context_norm_mean": self.final_norm_sum / n,
        }]
        pd.DataFrame(gate_summary).to_csv(
            out_dir / "gate_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        city_rows = []
        for cid, d in sorted(self.city.items()):
            cn = max(int(d["n"]), 1)
            city_rows.append({
                "variant": variant,
                "city_id": cid,
                "city": (
                    self.city_names[cid]
                    if 0 <= cid < len(self.city_names)
                    else str(cid)
                ),
                "samples": int(d["n"]),
                "positive_branch_norm_mean":
                    d["pos_norm_sum"] / cn,
                "negative_branch_norm_mean":
                    d["neg_norm_sum"] / cn,
                "final_context_norm_mean":
                    d["final_norm_sum"] / cn,
                "positive_gate_entropy_mean":
                    d["pos_entropy_sum"] / cn,
                "negative_gate_entropy_mean":
                    d["neg_entropy_sum"] / cn,
                "shared_gate_entropy_mean":
                    d["shared_entropy_sum"] / cn,
                "corr_abs_mean":
                    d["corr_abs_mean_sum"] / cn,
                "positive_variable_ratio":
                    d["positive_variable_ratio_sum"] / cn,
                "negative_variable_ratio":
                    d["negative_variable_ratio_sum"] / cn,
            })
        pd.DataFrame(city_rows).to_csv(
            out_dir / "branch_diagnostics_by_city.csv",
            index=False,
            encoding="utf-8-sig",
        )


class ComprehensiveDualContextEncoder(nn.Module):
    """A single controlled implementation for all context variants.

    O3 is represented by the dedicated ``o3_ctx`` and by the model's raw O3
    history path.  Every multivariate relevance distribution therefore
    competes only across the seven non-target covariates; the trivial
    corr(O3, O3) entry is zeroed before relation projection and the O3 logit is
    masked before every covariate softmax.
    """

    _SHARED_VARIANTS = {"c2_naive_multivariate", "b3_shared_gate"}
    _DUAL_VARIANTS = {
        "c3_dual_no_corr",
        "c4_full_dualcontext",
        "b0_positive_only",
        "b1_negative_only",
        "b2_equal_fusion",
    }

    def __init__(
        self,
        seq_len: int,
        n_vars: int,
        hidden: int,
        dropout: float,
        o3_idx: int,
        corr_temperature: float,
        variant: str,
        city_names: list[str],
    ) -> None:
        super().__init__()
        if variant not in VARIANT_SPECS:
            raise ValueError(variant)
        if int(n_vars) <= 1:
            raise ValueError("DualContext requires at least one non-target covariate")
        self.seq_len = int(seq_len)
        self.n_vars = int(n_vars)
        self.hidden = int(hidden)
        self.o3_idx = int(o3_idx)
        self.corr_temperature = float(corr_temperature)
        self.variant = variant

        self.channel_encoder = nn.Sequential(
            nn.LayerNorm(seq_len),
            nn.Linear(seq_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        def make_score() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1),
            )

        # C2 and B3 now have one actual scoring network, not the average of
        # two separately parameterized scoring networks.
        if variant in self._SHARED_VARIANTS:
            self.shared_score: Optional[nn.Module] = make_score()
            self.pos_score: Optional[nn.Module] = None
            self.neg_score: Optional[nn.Module] = None
        elif variant in self._DUAL_VARIANTS:
            self.shared_score = None
            self.pos_score = make_score()
            self.neg_score = make_score()
        else:
            self.shared_score = None
            self.pos_score = None
            self.neg_score = None

        self.corr_proj = nn.Sequential(
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
        self.diagnostics = DiagnosticAccumulator(n_vars, city_names)
        self.collect_diagnostics = False
        self._batch_city_id: Optional[torch.Tensor] = None

        self._freeze_unused_modules()

    def _freeze(self, module: nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    def _freeze_unused_modules(self) -> None:
        if self.variant == "c0_no_context":
            self._freeze(self.channel_encoder)
            self._freeze(self.corr_proj)
            self._freeze(self.fusion)
        elif self.variant in {
            "c1_o3_only",
            "c2_naive_multivariate",
            "c3_dual_no_corr",
        }:
            self._freeze(self.corr_proj)

    @staticmethod
    def pearson_to_target(
        x_multi: torch.Tensor,
        target_idx: int,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        x = x_multi.float()
        target = x[:, :, target_idx:target_idx + 1]
        x_center = x - x.mean(dim=1, keepdim=True)
        y_center = target - target.mean(dim=1, keepdim=True)
        numerator = (x_center * y_center).mean(dim=1)
        x_std = torch.sqrt((x_center ** 2).mean(dim=1) + eps)
        y_std = torch.sqrt((y_center ** 2).mean(dim=1) + eps)
        corr = numerator / (x_std * y_std + eps)
        corr = torch.nan_to_num(
            corr, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(-1.0, 1.0)
        corr = corr.clone()
        corr[:, target_idx] = 0.0
        return corr

    def mask_target_logits(self, logits: torch.Tensor) -> torch.Tensor:
        masked = logits.clone()
        masked[:, self.o3_idx] = -torch.inf
        return masked

    def set_batch_city_id(self, city_id: torch.Tensor) -> None:
        self._batch_city_id = city_id

    def reset_diagnostics(self, enabled: bool = True) -> None:
        self.diagnostics.reset()
        self.collect_diagnostics = enabled

    def forward(self, x_multi: torch.Tensor) -> torch.Tensor:
        B, L_, V = x_multi.shape
        if L_ != self.seq_len or V != self.n_vars:
            raise ValueError(
                f"Expected [B,{self.seq_len},{self.n_vars}], "
                f"got {tuple(x_multi.shape)}"
            )

        corr = self.pearson_to_target(x_multi, self.o3_idx)
        zeros_w = torch.zeros_like(corr)

        if self.variant == "c0_no_context":
            final_ctx = torch.zeros(
                B, self.hidden, device=x_multi.device, dtype=x_multi.dtype
            )
            pos_ctx = torch.zeros_like(final_ctx)
            neg_ctx = torch.zeros_like(final_ctx)
            pos_w = zeros_w
            neg_w = zeros_w
            shared_w = zeros_w
        else:
            x_ch = x_multi.transpose(1, 2).contiguous()
            h = self.channel_encoder(
                x_ch.reshape(B * V, L_)
            ).reshape(B, V, self.hidden)
            o3_ctx = h[:, self.o3_idx, :]

            if self.variant == "c1_o3_only":
                pos_w = zeros_w.clone()
                pos_w[:, self.o3_idx] = 1.0
                neg_w = zeros_w
                shared_w = pos_w
                pos_ctx = o3_ctx
                neg_ctx = torch.zeros_like(o3_ctx)
                corr_ctx = torch.zeros_like(o3_ctx)
                final_ctx = self.fusion(torch.cat([
                    torch.zeros_like(o3_ctx),
                    torch.zeros_like(o3_ctx),
                    o3_ctx,
                    corr_ctx,
                ], dim=-1))

            elif self.variant == "c2_naive_multivariate":
                if self.shared_score is None:
                    raise RuntimeError("C2 shared_score is missing")
                shared_logits = self.mask_target_logits(
                    self.shared_score(h).squeeze(-1)
                )
                shared_w = torch.softmax(shared_logits, dim=-1)
                shared_ctx = torch.sum(
                    shared_w.unsqueeze(-1) * h, dim=1
                )
                pos_w = shared_w
                neg_w = shared_w
                pos_ctx = shared_ctx
                neg_ctx = shared_ctx
                corr_ctx = torch.zeros_like(shared_ctx)
                final_ctx = self.fusion(torch.cat([
                    shared_ctx, shared_ctx, o3_ctx, corr_ctx
                ], dim=-1))

            elif self.variant == "c3_dual_no_corr":
                if self.pos_score is None or self.neg_score is None:
                    raise RuntimeError("C3 dual score networks are missing")
                pos_logits = self.mask_target_logits(
                    self.pos_score(h).squeeze(-1)
                )
                neg_logits = self.mask_target_logits(
                    self.neg_score(h).squeeze(-1)
                )
                pos_w = torch.softmax(pos_logits, dim=-1)
                neg_w = torch.softmax(neg_logits, dim=-1)
                shared_w = zeros_w
                pos_ctx = torch.sum(
                    pos_w.unsqueeze(-1) * h, dim=1
                )
                neg_ctx = torch.sum(
                    neg_w.unsqueeze(-1) * h, dim=1
                )
                corr_ctx = torch.zeros_like(pos_ctx)
                final_ctx = self.fusion(torch.cat([
                    pos_ctx, neg_ctx, o3_ctx, corr_ctx
                ], dim=-1))

            elif self.variant == "b3_shared_gate":
                if self.shared_score is None:
                    raise RuntimeError("B3 shared_score is missing")
                shared_logits = (
                    self.shared_score(h).squeeze(-1)
                    + self.corr_temperature * corr.abs()
                )
                shared_logits = self.mask_target_logits(shared_logits)
                shared_w = torch.softmax(shared_logits, dim=-1)
                shared_ctx = torch.sum(
                    shared_w.unsqueeze(-1) * h, dim=1
                )
                pos_w = shared_w
                neg_w = shared_w
                pos_ctx = shared_ctx
                neg_ctx = shared_ctx
                corr_ctx = self.corr_proj(corr)
                final_ctx = self.fusion(torch.cat([
                    shared_ctx, shared_ctx, o3_ctx, corr_ctx
                ], dim=-1))

            else:
                if self.pos_score is None or self.neg_score is None:
                    raise RuntimeError("Dual score networks are missing")
                pos_logits = (
                    self.pos_score(h).squeeze(-1)
                    + self.corr_temperature * corr.clamp(min=0.0)
                )
                neg_logits = (
                    self.neg_score(h).squeeze(-1)
                    + self.corr_temperature * (-corr).clamp(min=0.0)
                )
                pos_logits = self.mask_target_logits(pos_logits)
                neg_logits = self.mask_target_logits(neg_logits)
                pos_w = torch.softmax(pos_logits, dim=-1)
                neg_w = torch.softmax(neg_logits, dim=-1)
                shared_w = zeros_w
                pos_ctx = torch.sum(
                    pos_w.unsqueeze(-1) * h, dim=1
                )
                neg_ctx = torch.sum(
                    neg_w.unsqueeze(-1) * h, dim=1
                )
                corr_ctx = self.corr_proj(corr)

                if self.variant == "b0_positive_only":
                    neg_ctx = torch.zeros_like(neg_ctx)
                elif self.variant == "b1_negative_only":
                    pos_ctx = torch.zeros_like(pos_ctx)
                elif self.variant == "b2_equal_fusion":
                    average_ctx = 0.5 * (pos_ctx + neg_ctx)
                    pos_ctx = average_ctx
                    neg_ctx = average_ctx
                elif self.variant != "c4_full_dualcontext":
                    raise ValueError(self.variant)

                final_ctx = self.fusion(torch.cat([
                    pos_ctx, neg_ctx, o3_ctx, corr_ctx
                ], dim=-1))

        if (
            self.collect_diagnostics
            and self._batch_city_id is not None
        ):
            self.diagnostics.update(
                {
                    "corr": corr,
                    "pos_w": pos_w,
                    "neg_w": neg_w,
                    "shared_w": shared_w,
                    "pos_ctx": pos_ctx,
                    "neg_ctx": neg_ctx,
                    "final_ctx": final_ctx,
                },
                self._batch_city_id,
            )
        return final_ctx

class DualContextAblationForecaster(nn.Module):
    def __init__(
        self,
        *,
        timer: nn.Module,
        n_vars: int,
        n_cities: int,
        InputAdapter,
        time_feature_dim,
        variant: str,
        city_names: list[str],
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
        self.context_encoder = ComprehensiveDualContextEncoder(
            seq_len=L,
            n_vars=n_vars,
            hidden=context_hidden,
            dropout=dropout,
            o3_idx=o3_idx,
            corr_temperature=corr_temperature,
            variant=variant,
            city_names=city_names,
        )
        self.gamma = nn.Embedding(n_vars, H)
        self.beta = nn.Embedding(n_vars, H)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        self.var_emb = nn.Embedding(n_vars, emb_dim)
        self.city_emb = nn.Embedding(n_cities, city_emb_dim)

        time_dim = time_feature_dim("both")
        in_dim = (
            H + L + L + H * time_dim
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
        self.timer.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        x_multi: torch.Tensor,
        time_feat: torch.Tensor,
        var_idx: torch.Tensor,
        city_id: torch.Tensor,
    ) -> torch.Tensor:
        x_adapt = self.input_adapter(x, city_id)
        base = self.timer(x_adapt).logits[:, :H].float()
        pred = self.gamma(var_idx) * base + self.beta(var_idx)

        self.context_encoder.set_batch_city_id(city_id)
        context = self.context_encoder(x_multi)

        features = torch.cat([
            pred,
            x,
            x_adapt,
            time_feat,
            context,
            self.var_emb(var_idx),
            self.city_emb(city_id),
        ], dim=-1)
        return pred + self.output_gate * self.output_net(features)


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def extract_station_ids(batch: dict[str, Any]) -> Optional[np.ndarray]:
    for key in (
        "station_id", "station_idx", "site_id",
        "site_idx", "station_code", "station",
    ):
        if key in batch:
            return to_numpy(batch[key]).reshape(-1)
    return None


def run_epoch(
    model: DualContextAblationForecaster,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer,
    mse_weight: float,
    grad_clip: float,
    log_every: int,
    collect_detail: bool,
    collect_arrays: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    overall = SumMetric()
    by_city: dict[int, SumMetric] = defaultdict(SumMetric)
    by_station: dict[str, SumMetric] = defaultdict(SumMetric)
    h_abs = np.zeros(H, dtype=np.float64)
    h_sq = np.zeros(H, dtype=np.float64)
    h_n = np.zeros(H, dtype=np.int64)

    preds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    cities: list[np.ndarray] = []
    stations: list[np.ndarray] = []
    loss_sum = 0.0
    n_batches = 0

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
            pred_std = model(
                x, x_multi, time_feat, var_idx, city_id
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
        n_batches += 1
        pred = (
            pred_std.detach().cpu().numpy().astype(
                np.float32, copy=False
            ) * std + mu
        )
        overall.update(pred, y_phys)

        if collect_detail:
            city_np = to_numpy(batch["city_id"]).astype(np.int64)
            for cid in np.unique(city_np):
                mask = city_np == cid
                by_city[int(cid)].update(
                    pred[mask], y_phys[mask]
                )
            station_np = extract_station_ids(batch)
            if station_np is not None:
                for sid in np.unique(station_np):
                    mask = station_np == sid
                    by_station[str(sid)].update(
                        pred[mask], y_phys[mask]
                    )

            diff = pred.astype(np.float64) - y_phys.astype(
                np.float64
            )
            h_abs += np.abs(diff).sum(axis=0, dtype=np.float64)
            h_sq += np.square(diff).sum(axis=0, dtype=np.float64)
            h_n += diff.shape[0]

        if collect_arrays:
            preds.append(np.asarray(pred, dtype=np.float32))
            truths.append(np.asarray(y_phys, dtype=np.float32))
            cities.append(
                to_numpy(batch["city_id"]).astype(np.int16)
            )
            station_np = extract_station_ids(batch)
            if station_np is not None:
                stations.append(station_np)

        if training and (step == 1 or step % log_every == 0):
            current = overall.result()
            print(
                f"step={step}/{len(loader)} "
                f"loss={loss_sum/max(n_batches,1):.6f} "
                f"MAE={current['MAE']:.4f} "
                f"RMSE={current['RMSE']:.4f}",
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
        detail["pred"] = np.concatenate(preds, axis=0)
        detail["true"] = np.concatenate(truths, axis=0)
        detail["city_id"] = np.concatenate(cities, axis=0)
        if stations and len(stations) == len(preds):
            detail["station_id"] = np.concatenate(stations, axis=0)

    return (
        {
            "loss": loss_sum / max(n_batches, 1),
            **overall.result(),
        },
        detail,
    )


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("timer.")
    }


def canonical_true_path(root: Path) -> Path | None:
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
    return None


def save_true(
    true: np.ndarray,
    output: Path,
    canonical: Path | None,
    formal: bool,
) -> str:
    if output.exists() or output.is_symlink():
        output.unlink()
    if formal and canonical is not None:
        ref = np.load(canonical, mmap_mode="r")
        if ref.shape != true.shape:
            raise RuntimeError(
                f"Canonical shape={ref.shape}, run shape={true.shape}"
            )
        if not np.array_equal(
            np.asarray(ref), true, equal_nan=True
        ):
            diff = np.abs(
                np.asarray(ref, dtype=np.float64)
                - true.astype(np.float64)
            )
            raise RuntimeError(
                "Run true.npy differs from canonical revision-v2 "
                f"true.npy; max_abs_diff={float(np.nanmax(diff))}"
            )
        try:
            os.link(canonical.resolve(), output)
            return "hardlink_to_canonical"
        except OSError:
            try:
                rel = os.path.relpath(
                    canonical.resolve(), output.parent
                )
                os.symlink(rel, output)
                return "symlink_to_canonical"
            except OSError:
                np.save(output, true)
                return "copied_canonical_array"
    np.save(output, true)
    return "saved_test_targets" if formal else "saved_smoke_subset"


def save_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    detail: dict[str, Any],
    city_names: list[str],
    *,
    total_params: int,
    trainable_params: int,
    best_epoch: int,
    best_val_rmse: float,
    train_seconds: float,
    test_seconds: float,
    peak_gpu_memory_mb: float,
    device_name: str,
    canonical_true: Path | None,
) -> None:
    overall = {
        **metrics,
        "variant": args.variant,
        "paper_name": VARIANT_SPECS[args.variant]["paper_name"],
        "seed": args.seed,
        "protocol": "revision-v2",
        "best_epoch": best_epoch,
        "best_val_RMSE": best_val_rmse,
        "train_seconds": train_seconds,
        "test_seconds": test_seconds,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "device_name": device_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    pd.DataFrame([overall]).to_csv(
        out_dir / "metrics_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    city_rows = []
    for cid, metric in sorted(detail["by_city"].items()):
        city_rows.append({
            "city_id": cid,
            "city": (
                city_names[cid]
                if 0 <= cid < len(city_names)
                else str(cid)
            ),
            **metric.result(),
        })
    pd.DataFrame(city_rows).to_csv(
        out_dir / "metrics_by_city.csv",
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
        columns=[
            "station_id", "MAE", "MSE", "RMSE", "count"
        ],
    ).to_csv(
        out_dir / "metrics_by_station.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_rows = []
    for horizon in range(H):
        n = int(detail["h_n"][horizon])
        mse = (
            detail["h_sq"][horizon] / n
            if n else math.nan
        )
        horizon_rows.append({
            "horizon": horizon + 1,
            "MAE": (
                detail["h_abs"][horizon] / n
                if n else math.nan
            ),
            "MSE": mse,
            "RMSE": math.sqrt(mse) if n else math.nan,
            "count": n,
        })
    pd.DataFrame(horizon_rows).to_csv(
        out_dir / "metrics_by_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if args.save_arrays:
        pred = np.asarray(detail["pred"], dtype=np.float32)
        true = np.asarray(detail["true"], dtype=np.float32)
        np.save(out_dir / "pred.npy", pred)
        formal = (
            args.max_test_samples == 0
            and pred.shape == EXPECTED_SHAPE
            and true.shape == EXPECTED_SHAPE
        )
        true_storage = save_true(
            true, out_dir / "true.npy",
            canonical_true, formal
        )
        np.save(out_dir / "city_id.npy", detail["city_id"])
        if "station_id" in detail:
            np.save(
                out_dir / "station_id.npy",
                detail["station_id"],
            )
        manifest = {
            "protocol": "revision-v2",
            "variant": args.variant,
            "seed": args.seed,
            "pred_shape": list(pred.shape),
            "true_shape": list(true.shape),
            "count": int(pred.size),
            "pred_dtype": str(pred.dtype),
            "true_dtype": str(true.dtype),
            "nonfinite_pred": int(
                (~np.isfinite(pred)).sum()
            ),
            "nonfinite_true": int(
                (~np.isfinite(true)).sum()
            ),
            "true_storage": true_storage,
            "canonical_true": str(canonical_true) if canonical_true is not None else None,
        }
        (out_dir / "array_manifest.json").write_text(
            json.dumps(
                manifest, indent=2, ensure_ascii=False
            ),
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
            f"Canonical revision-v2 trainer missing: {source_path}"
        )
    source = import_source(
        source_path, "_dualcontext_revision_v2_source"
    )

    device = resolve_device(args)
    cuda = device.type == "cuda"
    seed_everything(args.seed, cuda)

    index_root = root / "experiments/data_v2/indices"
    stats_path = (
        root / "experiments/data_v2/"
        "city_var_stats_train_only.json"
    )
    cipda, cora = source.load_project_modules(
        root, index_root, stats_path
    )
    Dataset = cora.Stage1O3CoRAContextDataset

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

    print(f"Loading frozen Timer from {cipda.MODEL_PATH}", flush=True)
    timer = AutoModelForCausalLM.from_pretrained(
        cipda.MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
    )

    model = DualContextAblationForecaster(
        timer=timer,
        n_vars=len(train_ds.variables),
        n_cities=len(train_ds.cities),
        InputAdapter=cora.InputAdapter,
        time_feature_dim=cora.time_feature_dim,
        variant=args.variant,
        city_names=train_ds.cities,
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
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    device_name = (
        torch.cuda.get_device_name(device)
        if cuda else f"CPU ({torch.get_num_threads()} threads)"
    )
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)

    print("=" * 92)
    print("[DUALCONTEXT ABLATION / REVISION-V2]")
    print(
        f"variant={args.variant} "
        f"paper_name={VARIANT_SPECS[args.variant]['paper_name']}"
    )
    print(f"seed={args.seed} device={device} ({device_name})")
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
    print("=" * 92)

    args_dict = vars(args).copy()
    args_dict.update({
        "root": str(root),
        "out_dir": str(out_dir),
        "source_trainer": str(source_path),
        "index_root": str(index_root),
        "stats_path": str(stats_path),
        "protocol": "revision-v2",
        "variant_spec": VARIANT_SPECS[args.variant],
        "time_feature_mode": "both",
        "timer_frozen": True,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "device_name": device_name,
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
        [
            p for p in model.parameters()
            if p.requires_grad
        ],
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
            collect_detail=False,
            collect_arrays=False,
        )
        print(f"Epoch {epoch}/{args.epochs} val", flush=True)
        val_metrics, _ = run_epoch(
            model, val_loader, device,
            optimizer=None,
            mse_weight=args.mse_weight,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
            collect_detail=False,
            collect_arrays=False,
        )
        row = {
            "epoch": epoch,
            **{
                f"train_{k}": v
                for k, v in train_metrics.items()
            },
            **{
                f"val_{k}": v
                for k, v in val_metrics.items()
            },
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
            out_dir / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        best = torch.load(
            out_dir / "best.pt", map_location="cpu"
        )
    missing, unexpected = model.load_state_dict(
        best["adapter_state"], strict=False
    )
    missing_non_timer = [
        name for name in missing
        if not name.startswith("timer.")
    ]
    if unexpected or missing_non_timer:
        raise RuntimeError(
            "Checkpoint mismatch: "
            f"missing={missing_non_timer}, unexpected={unexpected}"
        )

    model.context_encoder.reset_diagnostics(enabled=True)
    print("\nFinal test evaluation", flush=True)
    test_start = time.perf_counter()
    test_metrics, detail = run_epoch(
        model, test_loader, device,
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

    canonical_true = canonical_true_path(root) if args.save_arrays else None
    save_outputs(
        out_dir,
        args,
        test_metrics,
        detail,
        train_ds.cities,
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
    model.context_encoder.diagnostics.save(
        out_dir,
        train_ds.variables,
        args.variant,
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
        for name in ("best.pt", "last.pt"):
            path = out_dir / name
            if path.exists():
                path.unlink()

    print(pd.DataFrame([summary]).to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir.is_absolute()
        else (root / args.out_dir).resolve()
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
