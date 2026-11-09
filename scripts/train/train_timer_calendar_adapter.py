import argparse
import json
import math
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

INDEX_ROOT = ROOT / "processed" / "stage1_indices_compact_qc"
STATS_PATH = ROOT / "processed" / "stage1_city_var_stats.json"


def time_feature_dim(mode):
    if mode == "hour":
        return 2
    if mode == "doy":
        return 2
    if mode == "both":
        return 4
    raise ValueError(f"Unknown time_feature_mode: {mode}")


def build_time_feature_matrix(time_ns, mode="both"):
    """
    time_ns: np.ndarray, shape [T], int64 ns or datetime64
    mode:
      - hour: hour_sin, hour_cos
      - doy : doy_sin, doy_cos
      - both: hour_sin, hour_cos, doy_sin, doy_cos
    return: np.ndarray, shape [T, C]
    """
    dt = time_ns.astype("datetime64[ns]")

    hour = dt.astype("datetime64[h]").astype(np.int64) % 24

    day = dt.astype("datetime64[D]")
    year_start = dt.astype("datetime64[Y]").astype("datetime64[D]")
    doy = (day - year_start).astype(np.int64)

    hour_angle = 2.0 * np.pi * hour / 24.0
    doy_angle = 2.0 * np.pi * doy / 366.0

    hour_feat = np.stack(
        [np.sin(hour_angle), np.cos(hour_angle)],
        axis=-1,
    ).astype(np.float32)

    doy_feat = np.stack(
        [np.sin(doy_angle), np.cos(doy_angle)],
        axis=-1,
    ).astype(np.float32)

    if mode == "hour":
        return hour_feat
    if mode == "doy":
        return doy_feat
    if mode == "both":
        return np.concatenate([hour_feat, doy_feat], axis=-1)

    raise ValueError(f"Unknown time_feature_mode: {mode}")


class Stage1O3TimeDataset(Dataset):
    def __init__(self, split="train", max_samples=0, seed=2026, time_feature_mode="both"):
        self.split = split
        self.time_feature_mode = time_feature_mode

        self.meta = json.loads((INDEX_ROOT / "meta.json").read_text(encoding="utf-8"))
        self.stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))

        self.variables = self.meta["variables"]
        self.cities = self.meta["cities"]
        self.array_paths = self.meta["array_paths"]

        idx = np.load(INDEX_ROOT / f"{split}.npz")

        array_id = idx["array_id"].astype(np.int32)
        city_id = idx["city_id"].astype(np.int16)
        var_idx = idx["var_idx"].astype(np.int8)
        start_idx = idx["start_idx"].astype(np.int32)

        o3_idx = self.variables.index("O3")
        mask = var_idx == o3_idx

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

        x = values[st:st + L, vid]
        y = values[st + L:st + L + H, vid]
        target_time_feat = time_feat_mat[st + L:st + L + H]

        stat = self.stats["city_variable_stats"][city][var]
        mu = np.float32(stat["mean"])
        std = np.float32(stat["std"])

        x_std = (x - mu) / std
        y_std = (y - mu) / std
        time_feat = target_time_feat.reshape(-1).astype(np.float32)

        return {
            "x": torch.tensor(x_std, dtype=torch.float32),
            "time_feat": torch.tensor(time_feat, dtype=torch.float32),
            "y_std": torch.tensor(y_std, dtype=torch.float32),
            "y_phys": torch.tensor(y, dtype=torch.float32),
            "mu": torch.tensor(mu, dtype=torch.float32),
            "std": torch.tensor(std, dtype=torch.float32),
            "city_id": torch.tensor(cid, dtype=torch.long),
            "var_idx": torch.tensor(vid, dtype=torch.long),
        }


class TimeContextOutputAdapter(nn.Module):
    def __init__(
        self,
        timer,
        n_vars,
        n_cities,
        hidden=128,
        emb_dim=16,
        city_emb_dim=16,
        dropout=0.05,
        time_feature_mode="both",
        init_gate=1e-3,
    ):
        super().__init__()

        self.timer = timer
        for p in self.timer.parameters():
            p.requires_grad_(False)
        self.timer.eval()

        self.gamma = nn.Embedding(n_vars, H)
        self.beta = nn.Embedding(n_vars, H)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

        self.var_emb = nn.Embedding(n_vars, emb_dim)
        self.city_emb = nn.Embedding(n_cities, city_emb_dim)

        # pred[48] + input[96] + time[48*C] + var emb + city emb
        self.time_feature_mode = time_feature_mode
        self.time_dim = time_feature_dim(time_feature_mode)
        in_dim = H + L + H * self.time_dim + emb_dim + city_emb_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, H),
        )

        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, time_feat, var_idx, city_id):
        with torch.no_grad():
            out = self.timer(x)
            base = out.logits[:, :H].float()

        pred = self.gamma(var_idx) * base + self.beta(var_idx)

        feat = torch.cat(
            [
                pred,
                x,
                time_feat,
                self.var_emb(var_idx),
                self.city_emb(city_id),
            ],
            dim=-1,
        )

        delta = self.net(feat)
        pred = pred + self.gate * delta
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
            pred_std = model(x, time_feat, var_idx, city_id)

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
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=128)
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
    parser.add_argument("--out_dir", type=str, default="checkpoints/stage1_cipda/time_context_o3")
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==== Args ====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    print("\n==== Loading O3 time datasets ====")
    train_set = Stage1O3TimeDataset(
        split="train",
        max_samples=args.max_train_samples,
        seed=args.seed,
        time_feature_mode=args.time_feature_mode,
    )
    val_set = Stage1O3TimeDataset(
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

    model = TimeContextOutputAdapter(
        timer=timer,
        n_vars=len(train_set.variables),
        n_cities=len(train_set.cities),
        hidden=args.hidden,
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
