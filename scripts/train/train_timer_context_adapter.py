#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train Timer IO-Time-CoRAContext Adapter for O3 forecasting.

Extension of the existing IO-Time-Context O3 adapter:
- Timer backbone still receives only standardized O3 history: x [B, 96].
- A correlation-aware multivariate branch receives standardized [96, 8] history.
- The context branch uses variable-wise encoders and O3-centered positive/negative
  correlation relevance, inspired by CoRA's correlation-aware adapter idea.
- Output adapter fuses base forecast, original/adapted O3 input, target-time features,
  CoRA-style covariate context, variable embedding and city embedding.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "scripts" / "train"
sys.path.insert(0, str(TRAIN_DIR))

from train_timer_base_adapter import MetricAgg, MODEL_PATH, L, H
from train_timer_calendar_adapter import build_time_feature_matrix, time_feature_dim

INDEX_ROOT = ROOT / "processed" / "stage1_indices_compact_qc"
STATS_PATH = ROOT / "processed" / "stage1_city_var_stats.json"


class Stage1O3CoRAContextDataset(Dataset):
    """
    O3 target dataset with multivariate historical context.

    Returns:
      x         : standardized O3 history, [96]
      x_multi   : standardized multivariate history, [96, n_vars]
      time_feat : flattened target-horizon time features, [48 * time_dim]
      y_std     : standardized future O3, [48]
      y_phys    : physical future O3, [48]
      mu/std    : city-level O3 mean/std
      city_id, var_idx
    """
    def __init__(self, split="train", max_samples=0, seed=2026, time_feature_mode="both"):
        self.split = split
        self.time_feature_mode = time_feature_mode

        self.meta = json.loads((INDEX_ROOT / "meta.json").read_text(encoding="utf-8"))
        self.stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))

        self.variables = self.meta["variables"]
        self.cities = self.meta["cities"]
        self.array_paths = self.meta["array_paths"]
        self.o3_idx = self.variables.index("O3")

        idx = np.load(INDEX_ROOT / f"{split}.npz")
        array_id = idx["array_id"].astype(np.int32)
        city_id = idx["city_id"].astype(np.int16)
        var_idx = idx["var_idx"].astype(np.int8)
        start_idx = idx["start_idx"].astype(np.int32)

        mask = var_idx == self.o3_idx
        array_id = array_id[mask]
        city_id = city_id[mask]
        var_idx = var_idx[mask]
        start_idx = start_idx[mask]

        if max_samples and max_samples > 0 and max_samples < len(array_id):
            rng = np.random.default_rng(seed)
            sel = rng.choice(len(array_id), size=max_samples, replace=False)
            array_id = array_id[sel]
            city_id = city_id[sel]
            var_idx = var_idx[sel]
            start_idx = start_idx[sel]

        self.array_id = array_id
        self.city_id = city_id
        self.var_idx = var_idx
        self.start_idx = start_idx

        self.cache = {}

        # City-level stats for multivariate standardization.
        self.city_var_mu = {}
        self.city_var_std = {}
        for city in self.cities:
            mu = []
            std = []
            for var in self.variables:
                stat = self.stats["city_variable_stats"][city][var]
                mu.append(np.float32(stat["mean"]))
                std_val = np.float32(stat["std"])
                if not np.isfinite(std_val) or std_val <= 0:
                    std_val = np.float32(1.0)
                std.append(std_val)
            self.city_var_mu[city] = np.asarray(mu, dtype=np.float32)
            self.city_var_std[city] = np.asarray(std, dtype=np.float32)

    def __len__(self):
        return len(self.array_id)

    def _load_npz(self, aid):
        aid = int(aid)
        if aid not in self.cache:
            path = ROOT / self.array_paths[aid]
            z = np.load(path, allow_pickle=True)
            values = z["values"].astype(np.float32)
            time_ns = z["time_ns"]
            time_feat = build_time_feature_matrix(time_ns, self.time_feature_mode)
            self.cache[aid] = {
                "values": values,
                "time_feat": time_feat,
            }
        return self.cache[aid]

    def __getitem__(self, i):
        aid = int(self.array_id[i])
        cid = int(self.city_id[i])
        vid = int(self.var_idx[i])
        st = int(self.start_idx[i])

        city = self.cities[cid]
        var = self.variables[vid]

        z = self._load_npz(aid)
        values = z["values"]
        time_feat_mat = z["time_feat"]

        # Timer input and target: O3 only.
        x_o3 = values[st:st + L, vid]
        y_o3 = values[st + L:st + L + H, vid]

        # Multivariate context: all variables over the same input window.
        x_multi = values[st:st + L, :]
        mu_vec = self.city_var_mu[city]
        std_vec = self.city_var_std[city]
        x_multi_std = (x_multi - mu_vec[None, :]) / std_vec[None, :]

        # City-level O3 standardization for Timer and target.
        stat = self.stats["city_variable_stats"][city][var]
        mu = np.float32(stat["mean"])
        std = np.float32(stat["std"])
        if not np.isfinite(std) or std <= 0:
            std = np.float32(1.0)

        x_std = (x_o3 - mu) / std
        y_std = (y_o3 - mu) / std

        target_time_feat = time_feat_mat[st + L:st + L + H]
        time_feat = target_time_feat.reshape(-1).astype(np.float32)

        return {
            "x": torch.tensor(x_std, dtype=torch.float32),
            "x_multi": torch.tensor(x_multi_std, dtype=torch.float32),
            "time_feat": torch.tensor(time_feat, dtype=torch.float32),
            "y_std": torch.tensor(y_std, dtype=torch.float32),
            "y_phys": torch.tensor(y_o3, dtype=torch.float32),
            "mu": torch.tensor(mu, dtype=torch.float32),
            "std": torch.tensor(std, dtype=torch.float32),
            "city_id": torch.tensor(cid, dtype=torch.long),
            "var_idx": torch.tensor(vid, dtype=torch.long),
        }


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


class CoRAContextEncoder(nn.Module):
    """
    Correlation-aware multivariate context encoder for [B, 96, n_vars].

    This is a lightweight O3-targeted adaptation of the CoRA idea:
      1) encode each variable separately with a shared variable-wise encoder;
      2) compute per-window Pearson correlation between each covariate and O3,
         explicitly excluding the target O3 channel from relation competition;
      3) build positive-correlation and negative-correlation covariate contexts;
      4) fuse them into a compact context embedding.

    It intentionally avoids flattening all variables together, so it can test whether
    explicit correlation/relevance modeling is better than naive multivariate context.
    """
    def __init__(
        self,
        seq_len=96,
        n_vars=8,
        hidden=128,
        dropout=0.05,
        o3_idx=4,
        corr_temperature=2.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.hidden = hidden
        self.o3_idx = int(o3_idx)
        self.corr_temperature = float(corr_temperature)

        # Shared variable-wise encoder: each channel [96] -> hidden.
        self.channel_encoder = nn.Sequential(
            nn.LayerNorm(seq_len),
            nn.Linear(seq_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        # Separate scoring heads for positive and negative relation contexts.
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

        # Project the raw O3-centered correlation vector as an explicit context clue.
        self.corr_proj = nn.Sequential(
            nn.LayerNorm(n_vars),
            nn.Linear(n_vars, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Fuse positive context, negative context, O3 self context and correlation vector.
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden * 4),
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

    @staticmethod
    def _pearson_to_target(x_multi, target_idx, eps=1e-6):
        """
        x_multi: [B, L, V], already standardized at city-variable level.
        Return: corr [B, V], window-level Pearson correlation to O3.
        """
        x = x_multi.float()
        target = x[:, :, target_idx:target_idx + 1]

        x_center = x - x.mean(dim=1, keepdim=True)
        y_center = target - target.mean(dim=1, keepdim=True)

        numerator = (x_center * y_center).mean(dim=1)
        x_std = torch.sqrt((x_center ** 2).mean(dim=1) + eps)
        y_std = torch.sqrt((y_center ** 2).mean(dim=1) + eps)
        corr = numerator / (x_std * y_std + eps)
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = corr.clamp(-1.0, 1.0)
        # O3 is the target and already has a dedicated o3_ctx plus raw-history path.
        # Exclude the trivial corr(O3,O3)≈1 cue from covariate relevance.
        corr = corr.clone()
        corr[:, target_idx] = 0.0
        return corr

    @staticmethod
    def _mask_target_logits(logits, target_idx):
        if logits.shape[-1] <= 1:
            raise ValueError("Covariate attention requires at least one non-target variable")
        masked = logits.clone()
        masked[:, target_idx] = -torch.inf
        return masked

    def forward(self, x_multi):
        """
        x_multi: [B, 96, n_vars]
        return : [B, hidden]
        """
        B, L_, V = x_multi.shape
        if V != self.n_vars:
            raise ValueError(f"Expected n_vars={self.n_vars}, got {V}")
        if L_ != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {L_}")

        # [B, V, L] -> [B*V, L] -> [B, V, hidden]
        x_ch = x_multi.transpose(1, 2).contiguous()
        h = self.channel_encoder(x_ch.reshape(B * V, L_)).reshape(B, V, self.hidden)

        corr = self._pearson_to_target(x_multi, self.o3_idx)  # [B, V]

        # CoRA-style polarity separation. Positive context is encouraged by positive
        # O3-covariate correlation; negative context is encouraged by negative relation.
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

        ctx = self.fusion(torch.cat([pos_ctx, neg_ctx, o3_ctx, corr_ctx], dim=-1))
        return ctx



class IOTimeCoRAContextAdapter(nn.Module):
    def __init__(
        self,
        timer,
        n_vars,
        n_cities,
        input_hidden=64,
        output_hidden=128,
        context_hidden=128,
        emb_dim=16,
        city_emb_dim=16,
        dropout=0.05,
        time_feature_mode="both",
        init_gate=1e-3,
        o3_idx=4,
        corr_temperature=2.0,
    ):
        super().__init__()

        self.timer = timer
        for p in self.timer.parameters():
            p.requires_grad_(False)
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

        self.context_encoder = CoRAContextEncoder(
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

        self.time_dim = time_feature_dim(time_feature_mode)

        in_dim = (
            H                 # base/pred
            + L               # original O3 input
            + L               # adapted O3 input
            + H * self.time_dim
            + context_hidden
            + emb_dim
            + city_emb_dim
        )

        self.output_net = nn.Sequential(
            nn.Linear(in_dim, output_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_hidden, H),
        )
        self.output_gate = nn.Parameter(torch.tensor(float(init_gate)))
        nn.init.zeros_(self.output_net[-1].weight)
        nn.init.zeros_(self.output_net[-1].bias)

    def forward(self, x, x_multi, time_feat, var_idx, city_id):
        # Need gradients through Timer output w.r.t. x_adapt, while Timer params stay frozen.
        x_adapt = self.input_adapter(x, city_id)

        out = self.timer(x_adapt)
        base = out.logits[:, :H].float()

        pred = self.gamma(var_idx) * base + self.beta(var_idx)
        ctx = self.context_encoder(x_multi)

        feat = torch.cat(
            [
                pred,
                x,
                x_adapt,
                time_feat,
                ctx,
                self.var_emb(var_idx),
                self.city_emb(city_id),
            ],
            dim=-1,
        )

        delta = self.output_net(feat)
        pred = pred + self.output_gate * delta
        return pred


def make_loader(dataset, batch_size, shuffle, num_workers):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def run_one_epoch(model, loader, optimizer, device, train, mse_weight=0.5, grad_clip=1.0, log_every=100):
    model.train(train)
    model.timer.eval()

    loss_sum = 0.0
    n_loss = 0
    metric = MetricAgg()

    for step, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, dtype=torch.float32, non_blocking=True)
        x_multi = batch["x_multi"].to(device, dtype=torch.float32, non_blocking=True)
        time_feat = batch["time_feat"].to(device, dtype=torch.float32, non_blocking=True)
        y_std = batch["y_std"].to(device, dtype=torch.float32, non_blocking=True)
        var_idx = batch["var_idx"].to(device, non_blocking=True)
        city_id = batch["city_id"].to(device, non_blocking=True)

        y_phys = batch["y_phys"].numpy()
        mu = batch["mu"].numpy()[:, None]
        std = batch["std"].numpy()[:, None]

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            pred_std = model(x, x_multi, time_feat, var_idx, city_id)
            mae_loss = F.l1_loss(pred_std, y_std)
            mse_loss = F.mse_loss(pred_std, y_std)
            loss = mae_loss + mse_weight * mse_loss

            if train:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        grad_clip,
                    )
                optimizer.step()

        loss_sum += float(loss.detach().cpu())
        n_loss += 1

        pred_phys = pred_std.detach().cpu().numpy() * std + mu
        metric.update(pred_phys, y_phys)

        if train and (step % log_every == 0 or step == 1):
            r = metric.result()
            print(
                f"  step {step:6d}/{len(loader):6d} "
                f"loss={loss_sum / max(n_loss, 1):.5f} "
                f"MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f}",
                flush=True,
            )

    return {
        "loss": loss_sum / max(n_loss, 1),
        **metric.result(),
    }


def adapter_state_dict(model):
    return {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("timer.")}


def count_trainable(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--input_hidden", type=int, default=64)
    parser.add_argument("--output_hidden", type=int, default=128)
    parser.add_argument("--context_hidden", type=int, default=128)
    parser.add_argument("--corr_temperature", type=float, default=2.0)
    parser.add_argument("--emb_dim", type=int, default=16)
    parser.add_argument("--city_emb_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--time_feature_mode", type=str, default="both", choices=["hour", "doy", "both"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mse_weight", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out_dir", type=str, default="checkpoints/stage1_cipda/io_time_coracontext_o3")
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==== Args ====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    print("\n==== Loading O3 correlation-aware multi-context datasets ====")
    train_set = Stage1O3CoRAContextDataset(
        split="train",
        max_samples=args.max_train_samples,
        seed=args.seed,
        time_feature_mode=args.time_feature_mode,
    )
    val_set = Stage1O3CoRAContextDataset(
        split="val",
        max_samples=args.max_val_samples,
        seed=args.seed + 1,
        time_feature_mode=args.time_feature_mode,
    )

    train_loader = make_loader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Train samples: {len(train_set):,}")
    print(f"Val samples  : {len(val_set):,}")
    print(f"Cities       : {len(train_set.cities)}")
    print(f"Variables    : {train_set.variables}")

    print("\n==== Loading frozen Timer ====")
    timer = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    timer.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timer.to(device)

    model = IOTimeCoRAContextAdapter(
        timer=timer,
        n_vars=len(train_set.variables),
        n_cities=len(train_set.cities),
        input_hidden=args.input_hidden,
        output_hidden=args.output_hidden,
        context_hidden=args.context_hidden,
        corr_temperature=args.corr_temperature,
        o3_idx=train_set.o3_idx,
        emb_dim=args.emb_dim,
        city_emb_dim=args.city_emb_dim,
        dropout=args.dropout,
        time_feature_mode=args.time_feature_mode,
    ).to(device)

    total, trainable = count_trainable(model)
    print(f"Total params    : {total / 1e6:.3f}M")
    print(f"Trainable params: {trainable / 1e6:.6f}M")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    history = []
    best_val_rmse = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\n==== Epoch {epoch}/{args.epochs} train ====")
        train_metrics = run_one_epoch(
            model, train_loader, optimizer, device,
            train=True,
            mse_weight=args.mse_weight,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
        )

        print(f"\n==== Epoch {epoch}/{args.epochs} val ====")
        val_metrics = run_one_epoch(
            model, val_loader, optimizer, device,
            train=False,
            mse_weight=args.mse_weight,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_MAE": train_metrics["MAE"],
            "train_MSE": train_metrics["MSE"],
            "train_RMSE": train_metrics["RMSE"],
            "train_count": train_metrics["count"],
            "val_loss": val_metrics["loss"],
            "val_MAE": val_metrics["MAE"],
            "val_MSE": val_metrics["MSE"],
            "val_RMSE": val_metrics["RMSE"],
            "val_count": val_metrics["count"],
        }

        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "metrics_epoch.csv", index=False, encoding="utf-8-sig")

        print("\nEpoch summary:")
        print(pd.DataFrame([row]))

        ckpt = {
            "epoch": epoch,
            "adapter_state": adapter_state_dict(model),
            "args": vars(args),
            "variables": train_set.variables,
            "cities": train_set.cities,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

        torch.save(ckpt, out_dir / "last.pt")

        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            torch.save(ckpt, out_dir / "best.pt")
            print(f"Saved best.pt with val_RMSE={best_val_rmse:.4f}")

    print("\n==== Finished training ====")
    print("Best val RMSE:", best_val_rmse)
    print("Output dir:", out_dir)


if __name__ == "__main__":
    main()
