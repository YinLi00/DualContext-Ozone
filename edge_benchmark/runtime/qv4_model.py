#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

L, H = 96, 48
QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)


def time_feature_dim(mode: str) -> int:
    if mode == "hour":
        return 2
    if mode == "doy":
        return 2
    if mode == "both":
        return 4
    raise ValueError(mode)


class InputAdapter(nn.Module):
    def __init__(self, seq_len=96, n_cities=22, city_emb_dim=16,
                 hidden=64, dropout=0.05, init_gate=1e-3):
        super().__init__()
        self.city_emb = nn.Embedding(n_cities, city_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(seq_len + city_emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, seq_len),
        )
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, city_id):
        h = torch.cat([x, self.city_emb(city_id)], dim=-1)
        return x + self.gate * self.net(h)


class RelationContextEncoder(nn.Module):
    """Exact q_quantile_full Pearson relation encoder."""
    def __init__(self, *, seq_len: int, n_vars: int, n_cities: int,
                 hidden: int, dropout: float, o3_idx: int,
                 corr_temperature: float, lag_max: int = 24,
                 lag_step: int = 3, mi_table: Optional[np.ndarray] = None):
        super().__init__()
        self.seq_len = int(seq_len)
        self.n_vars = int(n_vars)
        self.n_cities = int(n_cities)
        self.hidden = int(hidden)
        self.o3_idx = int(o3_idx)
        self.corr_temperature = float(corr_temperature)
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
        table = np.zeros((n_cities, n_vars), dtype=np.float32)
        table[:, self.o3_idx] = 0.0
        self.register_buffer("mi_table", torch.tensor(table), persistent=True)

    @staticmethod
    def pearson(x: torch.Tensor, target_idx: int, eps: float = 1e-6):
        target = x[:, :, target_idx:target_idx + 1]
        xc = x - x.mean(dim=1, keepdim=True)
        yc = target - target.mean(dim=1, keepdim=True)
        numerator = (xc * yc).mean(dim=1)
        xs = torch.sqrt((xc ** 2).mean(dim=1) + eps)
        ys = torch.sqrt((yc ** 2).mean(dim=1) + eps)
        corr = numerator / (xs * ys + eps)
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        corr = corr.clone()
        corr[:, target_idx] = 0.0
        return corr

    def _mask_target_logits(self, logits):
        out = logits.clone()
        out[:, self.o3_idx] = -torch.inf
        return out

    def forward(self, x_multi: torch.Tensor, city_id: torch.Tensor):
        B, length, n_vars = x_multi.shape
        if length != self.seq_len or n_vars != self.n_vars:
            raise ValueError(f"Expected [B,{self.seq_len},{self.n_vars}], got {tuple(x_multi.shape)}")
        x = x_multi.float()
        h = self.channel_encoder(
            x.transpose(1, 2).contiguous().reshape(B * n_vars, length)
        ).reshape(B, n_vars, self.hidden)
        o3_ctx = h[:, self.o3_idx, :]
        pos_score = self.pos_score(h).squeeze(-1)
        neg_score = self.neg_score(h).squeeze(-1)
        relation = self.pearson(x, self.o3_idx)
        pos_logits = pos_score + self.corr_temperature * relation.clamp(min=0.0)
        neg_logits = neg_score + self.corr_temperature * (-relation).clamp(min=0.0)
        relation = relation.clone()
        relation[:, self.o3_idx] = 0.0
        pos_logits = self._mask_target_logits(pos_logits)
        neg_logits = self._mask_target_logits(neg_logits)
        pos_w = torch.softmax(pos_logits, dim=-1)
        neg_w = torch.softmax(neg_logits, dim=-1)
        pos_ctx = torch.sum(pos_w.unsqueeze(-1) * h, dim=1)
        neg_ctx = torch.sum(neg_w.unsqueeze(-1) * h, dim=1)
        relation_ctx = self.relation_proj(relation)
        return self.fusion(torch.cat([pos_ctx, neg_ctx, o3_ctx, relation_ctx], dim=-1))


class QV4Forecaster(nn.Module):
    """Deployment-only exact q_quantile_full architecture."""
    def __init__(self, *, timer: nn.Module, n_vars: int, n_cities: int,
                 input_hidden: int, output_hidden: int, context_hidden: int,
                 emb_dim: int, city_emb_dim: int, dropout: float,
                 o3_idx: int, corr_temperature: float,
                 quantile_inner_gap_init: float = 0.8,
                 quantile_outer_gap_init: float = 0.4,
                 init_gate: float = 1e-3):
        super().__init__()
        self.timer = timer
        for p in self.timer.parameters():
            p.requires_grad_(False)
        self.timer.eval()
        if hasattr(self.timer.config, "use_cache"):
            self.timer.config.use_cache = False

        self.gamma = nn.Embedding(n_vars, H)
        self.beta = nn.Embedding(n_vars, H)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        self.input_adapter = InputAdapter(
            seq_len=L, n_cities=n_cities, city_emb_dim=city_emb_dim,
            hidden=input_hidden, dropout=dropout, init_gate=init_gate,
        )
        self.context_encoder = RelationContextEncoder(
            seq_len=L, n_vars=n_vars, n_cities=n_cities,
            hidden=context_hidden, dropout=dropout, o3_idx=o3_idx,
            corr_temperature=corr_temperature,
        )
        self.var_emb = nn.Embedding(n_vars, emb_dim)
        self.city_emb = nn.Embedding(n_cities, city_emb_dim)
        in_dim = H + L + L + H * 4 + context_hidden + emb_dim + city_emb_dim
        self.output_net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, output_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_hidden, H * 5),
        )
        self.output_gate = nn.Parameter(torch.tensor(float(init_gate)))
        nn.init.zeros_(self.output_net[-1].weight)
        nn.init.zeros_(self.output_net[-1].bias)

        def spi(v: float) -> float:
            return math.log(math.expm1(float(v)))

        with torch.no_grad():
            bias = self.output_net[-1].bias.view(H, 5)
            bias[:, 0].zero_()
            bias[:, 1].fill_(spi(quantile_inner_gap_init))
            bias[:, 2].fill_(spi(quantile_outer_gap_init))
            bias[:, 3].fill_(spi(quantile_inner_gap_init))
            bias[:, 4].fill_(spi(quantile_outer_gap_init))

    def train(self, mode: bool = True):
        super().train(mode)
        self.timer.eval()
        return self

    def forward(self, x, x_multi, time_feat, var_idx, city_id):
        x_adapt = self.input_adapter(x, city_id)
        base = self.timer(x_adapt).logits[:, :H].float()
        calibrated = self.gamma(var_idx) * base + self.beta(var_idx)
        context = self.context_encoder(x_multi, city_id)
        features = torch.cat([
            calibrated, x, x_adapt, time_feat, context,
            self.var_emb(var_idx), self.city_emb(city_id),
        ], dim=-1)
        head = self.output_net(features).view(x.shape[0], H, 5)
        q50 = calibrated + self.output_gate * head[..., 0]
        lower_inner = F.softplus(head[..., 1])
        lower_outer = F.softplus(head[..., 2])
        upper_inner = F.softplus(head[..., 3])
        upper_outer = F.softplus(head[..., 4])
        q10 = q50 - lower_inner
        q05 = q10 - lower_outer
        q90 = q50 + upper_inner
        q95 = q90 + upper_outer
        return torch.stack([q05, q10, q50, q90, q95], dim=-1)
