#!/usr/bin/env python3
from __future__ import annotations
import torch
import torch.nn as nn
L, H = 96, 48

class GlobalTSMixer(nn.Module):
    def __init__(self, backbone: nn.Module, n_cities: int, n_vars: int, o3_idx: int,
                 variant: str, city_emb_dim: int, condition_hidden: int):
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

    def forward(self, x: torch.Tensor, future_time: torch.Tensor,
                city_id: torch.Tensor) -> torch.Tensor:
        x_in = x
        city_vec = None
        if self.variant == "city_time":
            city_vec = self.city_emb(city_id)
            x_in = x + self.city_to_input(city_vec).unsqueeze(1)
        # Official TSMixer ignores time marks and decoder inputs.
        out = self.backbone(x_in, None, None, None)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim != 3:
            raise RuntimeError(f"Unexpected TSMixer output shape: {tuple(out.shape)}")
        base = out[:, -H:, self.o3_idx]
        if self.variant == "plain":
            return base
        cv = city_vec.unsqueeze(1).expand(-1, H, -1)
        cond = torch.cat([base.unsqueeze(-1), future_time, cv], dim=-1)
        correction = self.condition_head(cond).squeeze(-1)
        return base + self.condition_gate * correction
