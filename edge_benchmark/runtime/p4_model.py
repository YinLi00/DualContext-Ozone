#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn as nn

L, H = 96, 48


class InputAdapter(nn.Module):
    def __init__(self, seq_len=96, n_cities=22, city_emb_dim=16,
                 hidden=64, dropout=0.05, init_gate=1e-3):
        super().__init__()
        self.city_emb = nn.Embedding(n_cities, city_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(seq_len + city_emb_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, seq_len),
        )
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, city_id):
        return x + self.gate * self.net(torch.cat([x, self.city_emb(city_id)], dim=-1))


class CoRAContextEncoder(nn.Module):
    def __init__(self, seq_len=96, n_vars=8, hidden=128, dropout=0.05,
                 o3_idx=4, corr_temperature=2.0):
        super().__init__()
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.hidden = hidden
        self.o3_idx = int(o3_idx)
        self.corr_temperature = float(corr_temperature)
        self.channel_encoder = nn.Sequential(
            nn.LayerNorm(seq_len), nn.Linear(seq_len, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.pos_score = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.neg_score = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.corr_proj = nn.Sequential(
            nn.LayerNorm(n_vars), nn.Linear(n_vars, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden * 4), nn.Linear(hidden * 4, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
        )

    @staticmethod
    def _pearson_to_target(x_multi, target_idx, eps=1e-6):
        x = x_multi.float()
        target = x[:, :, target_idx:target_idx + 1]
        xc = x - x.mean(dim=1, keepdim=True)
        yc = target - target.mean(dim=1, keepdim=True)
        num = (xc * yc).mean(dim=1)
        xs = torch.sqrt((xc ** 2).mean(dim=1) + eps)
        ys = torch.sqrt((yc ** 2).mean(dim=1) + eps)
        corr = torch.nan_to_num(num / (xs * ys + eps), nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        corr = corr.clone()
        corr[:, target_idx] = 0.0
        return corr

    @staticmethod
    def _mask_target_logits(logits, target_idx):
        out = logits.clone()
        out[:, target_idx] = -torch.inf
        return out

    def forward(self, x_multi):
        B, L_, V = x_multi.shape
        if L_ != self.seq_len or V != self.n_vars:
            raise ValueError(f"Expected [B,{self.seq_len},{self.n_vars}], got {tuple(x_multi.shape)}")
        h = self.channel_encoder(
            x_multi.transpose(1, 2).contiguous().reshape(B * V, L_)
        ).reshape(B, V, self.hidden)
        corr = self._pearson_to_target(x_multi, self.o3_idx)
        pos_logits = self.pos_score(h).squeeze(-1) + self.corr_temperature * corr.clamp(min=0.0)
        neg_logits = self.neg_score(h).squeeze(-1) + self.corr_temperature * (-corr).clamp(min=0.0)
        pos_logits = self._mask_target_logits(pos_logits, self.o3_idx)
        neg_logits = self._mask_target_logits(neg_logits, self.o3_idx)
        pos_w = torch.softmax(pos_logits, dim=-1)
        neg_w = torch.softmax(neg_logits, dim=-1)
        pos_ctx = torch.sum(pos_w.unsqueeze(-1) * h, dim=1)
        neg_ctx = torch.sum(neg_w.unsqueeze(-1) * h, dim=1)
        o3_ctx = h[:, self.o3_idx, :]
        corr_ctx = self.corr_proj(corr)
        return self.fusion(torch.cat([pos_ctx, neg_ctx, o3_ctx, corr_ctx], dim=-1))


class P4Forecaster(nn.Module):
    def __init__(self, *, timer, n_vars: int, n_cities: int,
                 input_hidden: int, output_hidden: int, context_hidden: int,
                 emb_dim: int, city_emb_dim: int, dropout: float,
                 o3_idx: int, corr_temperature: float, init_gate: float = 1e-3):
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
        self.var_emb = nn.Embedding(n_vars, emb_dim)
        self.city_emb = nn.Embedding(n_cities, city_emb_dim)
        self.output_gate = nn.Parameter(torch.tensor(float(init_gate)))
        self.input_adapter = InputAdapter(
            seq_len=L, n_cities=n_cities, city_emb_dim=city_emb_dim,
            hidden=input_hidden, dropout=dropout, init_gate=init_gate,
        )
        self.context_encoder = CoRAContextEncoder(
            seq_len=L, n_vars=n_vars, hidden=context_hidden, dropout=dropout,
            o3_idx=o3_idx, corr_temperature=corr_temperature,
        )
        in_dim = H + L + L + H * 4 + context_hidden + emb_dim + city_emb_dim
        self.output_net = nn.Sequential(
            nn.Linear(in_dim, output_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(output_hidden, H),
        )
        nn.init.zeros_(self.output_net[-1].weight)
        nn.init.zeros_(self.output_net[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.timer.eval()
        return self

    def forward(self, x, x_multi, time_feat, var_idx, city_id):
        x_adapt = self.input_adapter(x, city_id)
        base = self.timer(x_adapt).logits[:, :H].float()
        pred = self.gamma(var_idx) * base + self.beta(var_idx)
        features = [
            pred, x, x_adapt, time_feat, self.context_encoder(x_multi),
            self.var_emb(var_idx), self.city_emb(city_id),
        ]
        return pred + self.output_gate * self.output_net(torch.cat(features, dim=-1))
