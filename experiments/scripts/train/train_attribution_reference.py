#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict base-forecast attribution with an identical contextual residual head.

All modes share the same trainable architecture and contextual information. Only the
base forecast is changed:
  timer            : frozen Timer forecast
  zero             : all-zero base (same residual architecture)
  shuffled_timer   : Timer forecast shifted across samples within each batch
  raw_seasonal24   : raw standardized O3 last 24 h repeated twice
  persistence      : raw standardized O3 last value repeated 48 times

The script trains on revision-v2 train, selects the checkpoint on revision-v2 val,
and evaluates once on revision-v2 test. The independent cal split is not used here.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

L, H = 96, 48


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument("--base_mode", required=True,
                   choices=["timer", "zero", "shuffled_timer", "raw_seasonal24", "persistence"])
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
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
    p.add_argument("--time_feature_mode", default="both", choices=["hour", "doy", "both"])
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--max_test_samples", type=int, default=0)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--log_every", type=int, default=200)
    return p.parse_args()


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class SumMetric:
    def __init__(self) -> None:
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, pred: np.ndarray, truth: np.ndarray) -> None:
        d = np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
        self.abs_sum += float(np.abs(d).sum())
        self.sq_sum += float(np.square(d).sum())
        self.count += int(d.size)

    def result(self) -> dict:
        if self.count == 0:
            return {"MAE": math.nan, "MSE": math.nan, "RMSE": math.nan, "count": 0}
        mse = self.sq_sum / self.count
        return {"MAE": self.abs_sum / self.count, "MSE": mse,
                "RMSE": math.sqrt(mse), "count": self.count}


def load_project_modules(root: Path, index_root: Path, stats_path: Path):
    train_dir = root / "scripts" / "train"
    if str(train_dir) not in sys.path:
        sys.path.insert(0, str(train_dir))
    import train_timer_base_adapter as cipda
    import train_timer_context_adapter as cora
    # Redirect only this process to revision-v2 data.
    cora.ROOT = root
    cora.INDEX_ROOT = index_root
    cora.STATS_PATH = stats_path
    return cipda, cora


class StrictAttributionForecaster(nn.Module):
    def __init__(self, *, timer: Optional[nn.Module], base_mode: str, n_vars: int,
                 n_cities: int, InputAdapter, ContextEncoder, time_feature_dim,
                 input_hidden: int, output_hidden: int, context_hidden: int,
                 emb_dim: int, city_emb_dim: int, dropout: float,
                 time_feature_mode: str, o3_idx: int, corr_temperature: float,
                 init_gate: float = 1e-3):
        super().__init__()
        self.timer = timer
        self.base_mode = base_mode
        if self.timer is not None:
            for p in self.timer.parameters():
                p.requires_grad_(False)
            self.timer.eval()
            if hasattr(self.timer.config, "use_cache"):
                self.timer.config.use_cache = False

        self.input_adapter = InputAdapter(
            seq_len=L, n_cities=n_cities, city_emb_dim=city_emb_dim,
            hidden=input_hidden, dropout=dropout, init_gate=init_gate,
        )
        self.context_encoder = ContextEncoder(
            seq_len=L, n_vars=n_vars, hidden=context_hidden, dropout=dropout,
            o3_idx=o3_idx, corr_temperature=corr_temperature,
        )
        self.gamma = nn.Embedding(n_vars, H)
        self.beta = nn.Embedding(n_vars, H)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        self.var_emb = nn.Embedding(n_vars, emb_dim)
        self.city_emb = nn.Embedding(n_cities, city_emb_dim)
        tdim = time_feature_dim(time_feature_mode)
        in_dim = H + L + L + H * tdim + context_hidden + emb_dim + city_emb_dim
        self.output_net = nn.Sequential(
            nn.Linear(in_dim, output_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(output_hidden, H),
        )
        self.output_gate = nn.Parameter(torch.tensor(float(init_gate)))
        nn.init.zeros_(self.output_net[-1].weight)
        nn.init.zeros_(self.output_net[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.timer is not None:
            self.timer.eval()
        return self

    def _base(self, x: torch.Tensor, x_adapt: torch.Tensor) -> torch.Tensor:
        if self.base_mode in {"timer", "shuffled_timer"}:
            if self.timer is None:
                raise RuntimeError("Timer model is required for this base mode")
            base = self.timer(x_adapt).logits[:, :H].float()
            if self.base_mode == "shuffled_timer":
                if base.size(0) > 1:
                    base = torch.roll(base, shifts=1, dims=0)
                else:
                    base = torch.flip(base, dims=[1])
            return base
        if self.base_mode == "zero":
            return torch.zeros((x.size(0), H), device=x.device, dtype=x.dtype)
        if self.base_mode == "raw_seasonal24":
            return x[:, -24:].repeat(1, H // 24)
        if self.base_mode == "persistence":
            return x[:, -1:].repeat(1, H)
        raise ValueError(self.base_mode)

    def forward(self, x, x_multi, time_feat, var_idx, city_id):
        x_adapt = self.input_adapter(x, city_id)
        base = self._base(x, x_adapt)
        pred = self.gamma(var_idx) * base + self.beta(var_idx)
        ctx = self.context_encoder(x_multi)
        feat = torch.cat([
            pred, x, x_adapt, time_feat, ctx,
            self.var_emb(var_idx), self.city_emb(city_id),
        ], dim=-1)
        return pred + self.output_gate * self.output_net(feat)


def make_loader(ds, batch_size: int, shuffle: bool, workers: int, cuda: bool) -> DataLoader:
    kw = dict(dataset=ds, batch_size=batch_size, shuffle=shuffle,
              num_workers=workers, pin_memory=cuda, drop_last=False)
    if workers > 0:
        kw["persistent_workers"] = True
    return DataLoader(**kw)


def run_epoch(model, loader, device, optimizer=None, mse_weight=0.5,
              grad_clip=1.0, log_every=200, collect_detail=False):
    training = optimizer is not None
    model.train(training)
    overall = SumMetric()
    by_city: Dict[int, SumMetric] = defaultdict(SumMetric)
    h_abs = np.zeros(H, dtype=np.float64)
    h_sq = np.zeros(H, dtype=np.float64)
    h_n = np.zeros(H, dtype=np.int64)
    loss_sum, batches = 0.0, 0

    for step, batch in enumerate(loader, 1):
        x = batch["x"].to(device, dtype=torch.float32, non_blocking=True)
        xm = batch["x_multi"].to(device, dtype=torch.float32, non_blocking=True)
        tf = batch["time_feat"].to(device, dtype=torch.float32, non_blocking=True)
        ys = batch["y_std"].to(device, dtype=torch.float32, non_blocking=True)
        vi = batch["var_idx"].to(device, non_blocking=True)
        ci = batch["city_id"].to(device, non_blocking=True)
        y_phys = batch["y_phys"].numpy()
        mu = batch["mu"].numpy()[:, None]
        std = batch["std"].numpy()[:, None]

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            pred_s = model(x, xm, tf, vi, ci)
            loss = F.l1_loss(pred_s, ys) + mse_weight * F.mse_loss(pred_s, ys)
            if training:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], grad_clip)
                optimizer.step()
        loss_sum += float(loss.detach().cpu())
        batches += 1
        pred = pred_s.detach().cpu().numpy() * std + mu
        overall.update(pred, y_phys)

        if collect_detail:
            city_ids = batch["city_id"].numpy()
            for cid in np.unique(city_ids):
                m = city_ids == cid
                by_city[int(cid)].update(pred[m], y_phys[m])
            d = pred.astype(np.float64) - y_phys.astype(np.float64)
            h_abs += np.abs(d).sum(axis=0)
            h_sq += np.square(d).sum(axis=0)
            h_n += d.shape[0]

        if training and (step == 1 or step % log_every == 0):
            r = overall.result()
            print(f"step={step}/{len(loader)} loss={loss_sum/batches:.5f} "
                  f"MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f}", flush=True)

    detail = {"by_city": by_city, "h_abs": h_abs, "h_sq": h_sq, "h_n": h_n}
    return {"loss": loss_sum / max(batches, 1), **overall.result()}, detail


def adapter_state(model: nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if not k.startswith("timer.")}


def save_test_details(out_dir: Path, detail: dict, cities: list[str]) -> None:
    rows = []
    for cid, metric in sorted(detail["by_city"].items()):
        rows.append({"city_id": cid, "city": cities[cid], **metric.result()})
    pd.DataFrame(rows).to_csv(out_dir / "metrics_by_city.csv", index=False, encoding="utf-8-sig")
    h_rows = []
    for h in range(H):
        n = int(detail["h_n"][h])
        mse = detail["h_sq"][h] / n if n else math.nan
        h_rows.append({"horizon": h + 1,
                       "MAE": detail["h_abs"][h] / n if n else math.nan,
                       "MSE": mse, "RMSE": math.sqrt(mse) if n else math.nan,
                       "count": n})
    pd.DataFrame(h_rows).to_csv(out_dir / "metrics_by_horizon.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = (root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    index_root = root / "experiments" / "data_v2" / "indices"
    stats_path = root / "experiments" / "data_v2" / "city_var_stats_train_only.json"
    if not (index_root / "meta.json").exists() or not stats_path.exists():
        raise FileNotFoundError("Revision-v2 data not found. Run build_data_splits.py first.")

    cipda, cora = load_project_modules(root, index_root, stats_path)
    Dataset = cora.Stage1O3CoRAContextDataset
    train_ds = Dataset("train", args.max_train_samples, args.seed, args.time_feature_mode)
    val_ds = Dataset("val", args.max_val_samples, args.seed + 1, args.time_feature_mode)
    test_ds = Dataset("test", args.max_test_samples, args.seed + 2, args.time_feature_mode)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, cuda)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers, cuda)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers, cuda)

    timer = None
    if args.base_mode in {"timer", "shuffled_timer"}:
        print(f"Loading Timer from {cipda.MODEL_PATH}")
        timer = AutoModelForCausalLM.from_pretrained(
            cipda.MODEL_PATH, trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.float32,
        )

    model = StrictAttributionForecaster(
        timer=timer, base_mode=args.base_mode,
        n_vars=len(train_ds.variables), n_cities=len(train_ds.cities),
        InputAdapter=cora.InputAdapter, ContextEncoder=cora.CoRAContextEncoder,
        time_feature_dim=cora.time_feature_dim,
        input_hidden=args.input_hidden, output_hidden=args.output_hidden,
        context_hidden=args.context_hidden, emb_dim=args.emb_dim,
        city_emb_dim=args.city_emb_dim, dropout=args.dropout,
        time_feature_mode=args.time_feature_mode, o3_idx=train_ds.o3_idx,
        corr_temperature=args.corr_temperature,
    ).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"mode={args.base_mode} train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}")
    print(f"params total={total:,}, trainable={trainable:,}, device={device}")

    args_dict = vars(args).copy()
    args_dict.update({"root": str(root), "out_dir": str(out_dir),
                      "index_root": str(index_root), "stats_path": str(stats_path),
                      "total_params": total, "trainable_params": trainable})
    (out_dir / "args.json").write_text(json.dumps(args_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    history = []
    best_rmse = float("inf")
    train_t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} train")
        tr, _ = run_epoch(model, train_loader, device, optimizer, args.mse_weight,
                          args.grad_clip, args.log_every, False)
        print(f"Epoch {epoch}/{args.epochs} val")
        va, _ = run_epoch(model, val_loader, device, None, args.mse_weight,
                          args.grad_clip, args.log_every, False)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
               **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "metrics_epoch.csv", index=False, encoding="utf-8-sig")
        ckpt = {"epoch": epoch, "adapter_state": adapter_state(model),
                "args": args_dict, "cities": train_ds.cities,
                "variables": train_ds.variables, "val_metrics": va}
        torch.save(ckpt, out_dir / "last.pt")
        if va["RMSE"] < best_rmse:
            best_rmse = va["RMSE"]
            torch.save(ckpt, out_dir / "best.pt")
            print(f"Saved best.pt val_RMSE={best_rmse:.4f}")

    train_seconds = time.perf_counter() - train_t0
    best = torch.load(out_dir / "best.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(best["adapter_state"], strict=False)
    if unexpected or [x for x in missing if not x.startswith("timer.")]:
        raise RuntimeError(f"Checkpoint mismatch, missing={missing}, unexpected={unexpected}")
    print("\nFinal test evaluation")
    test_t0 = time.perf_counter()
    te, detail = run_epoch(model, test_loader, device, None, args.mse_weight,
                           args.grad_clip, args.log_every, True)
    test_seconds = time.perf_counter() - test_t0
    pd.DataFrame([{**te, "base_mode": args.base_mode, "seed": args.seed,
                   "best_epoch": best["epoch"], "best_val_RMSE": best_rmse,
                   "train_seconds": train_seconds, "test_seconds": test_seconds,
                   "total_params": total, "trainable_params": trainable}]).to_csv(
        out_dir / "metrics_overall.csv", index=False, encoding="utf-8-sig")
    save_test_details(out_dir, detail, train_ds.cities)
    print(pd.read_csv(out_dir / "metrics_overall.csv").to_string(index=False))


if __name__ == "__main__":
    main()
