# Auto-generated inference-only C4 definitions.
from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from torch import nn

L = 96
H = 48

VARIANT_SPECS = {
    "c4_full_dualcontext": {
        "section": "overall",
        "paper_name": "Full DualContext",
        "description": (
            "Pearson-guided positive/negative branches "
            "with independent relevance gates."
        ),
    }
}


def time_feature_dim(mode):
    if mode == "hour":
        return 2
    if mode == "doy":
        return 2
    if mode == "both":
        return 4
    raise ValueError(f"Unknown time_feature_mode: {mode}")


class InputAdapter(nn.Module):
    def __init__(self, seq_len=96, n_cities=22, city_emb_dim=16, hidden=64, dropout=0.05, init_gate=1e-3):
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
        delta = self.net(h)
        return x + self.gate * delta


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
