import argparse
import json
import math
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[2]
INDEX_ROOT = ROOT / "processed" / "stage1_indices_compact_qc"
STATS_PATH = ROOT / "processed" / "stage1_city_var_stats.json"
MODEL_PATH = Path(
    os.environ.get("IOTJ_TIMER_MODEL_PATH", str(ROOT / "models_hf_local" / "timer_base_84m"))
).expanduser()

L = 96
H = 48


class Stage1Dataset(Dataset):
    def __init__(self, split="train", only_var="ALL", only_city="ALL", max_samples=0, seed=2026):
        self.split = split

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

        if only_var.upper() != "ALL":
            if only_var not in self.variables:
                raise ValueError(f"Unknown variable {only_var}. Available: {self.variables}")
            target_vid = self.variables.index(only_var)
            mask = var_idx == target_vid
            array_id = array_id[mask]
            city_id = city_id[mask]
            var_idx = var_idx[mask]
            start_idx = start_idx[mask]

        if only_city.upper() != "ALL":
            if only_city not in self.cities:
                raise ValueError(f"Unknown city {only_city}. Available: {self.cities}")
            target_cid = self.cities.index(only_city)
            mask = city_id == target_cid
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

    def _load_values(self, aid):
        aid = int(aid)
        if aid not in self.cache:
            path = ROOT / self.array_paths[aid]
            self.cache[aid] = np.load(path, allow_pickle=True)["values"].astype(np.float32)
        return self.cache[aid]

    def __getitem__(self, i):
        aid = int(self.array_id[i])
        cid = int(self.city_id[i])
        vid = int(self.var_idx[i])
        st = int(self.start_idx[i])

        city = self.cities[cid]
        var = self.variables[vid]

        values = self._load_values(aid)

        x = values[st:st + L, vid]
        y = values[st + L:st + L + H, vid]

        stat = self.stats["city_variable_stats"][city][var]
        mu = np.float32(stat["mean"])
        std = np.float32(stat["std"])

        x_std = (x - mu) / std
        y_std = (y - mu) / std

        return {
            "x": torch.tensor(x_std, dtype=torch.float32),
            "y_std": torch.tensor(y_std, dtype=torch.float32),
            "y_phys": torch.tensor(y, dtype=torch.float32),
            "mu": torch.tensor(mu, dtype=torch.float32),
            "std": torch.tensor(std, dtype=torch.float32),
            "city_id": torch.tensor(cid, dtype=torch.long),
            "var_idx": torch.tensor(vid, dtype=torch.long),
        }


class TemporalAdapter(nn.Module):
    def __init__(
        self,
        seq_len,
        n_vars,
        n_cities=0,
        hidden=64,
        emb_dim=16,
        city_emb_dim=0,
        dropout=0.1,
        init_gate=1e-3,
    ):
        super().__init__()
        self.var_emb = nn.Embedding(n_vars, emb_dim)
        self.use_city_emb = (n_cities is not None) and (n_cities > 0) and (city_emb_dim > 0)

        if self.use_city_emb:
            self.city_emb = nn.Embedding(n_cities, city_emb_dim)
            in_dim = seq_len + emb_dim + city_emb_dim
        else:
            self.city_emb = None
            in_dim = seq_len + emb_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, seq_len),
        )
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

        # Make the residual branch small at initialization.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, var_idx, city_id=None):
        parts = [x, self.var_emb(var_idx)]

        if self.use_city_emb:
            if city_id is None:
                raise ValueError("city_id is required when city embedding is enabled.")
            parts.append(self.city_emb(city_id))

        h = torch.cat(parts, dim=-1)
        delta = self.net(h)
        return x + self.gate * delta


class TimerCIPDA(nn.Module):
    def __init__(
        self,
        timer,
        n_vars,
        n_cities=0,
        adapter_mode="output_affine",
        hidden=64,
        emb_dim=16,
        city_emb_dim=16,
        dropout=0.1,
    ):
        super().__init__()
        assert adapter_mode in ["output_affine", "output_mlp", "output_mlp_city", "input_output"]
        self.timer = timer
        self.adapter_mode = adapter_mode
        self.n_vars = n_vars

        for p in self.timer.parameters():
            p.requires_grad_(False)
        self.timer.eval()

        self.gamma = nn.Embedding(n_vars, H)
        self.beta = nn.Embedding(n_vars, H)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

        if adapter_mode == "input_output":
            self.input_adapter = TemporalAdapter(
                seq_len=L,
                n_vars=n_vars,
                n_cities=0,
                hidden=hidden,
                emb_dim=emb_dim,
                city_emb_dim=0,
                dropout=dropout,
                init_gate=1e-3,
            )
        else:
            self.input_adapter = None

        if adapter_mode in ["output_mlp", "output_mlp_city", "input_output"]:
            use_city = adapter_mode == "output_mlp_city"
            self.output_adapter = TemporalAdapter(
                seq_len=H,
                n_vars=n_vars,
                n_cities=n_cities if use_city else 0,
                hidden=hidden,
                emb_dim=emb_dim,
                city_emb_dim=city_emb_dim if use_city else 0,
                dropout=dropout,
                init_gate=1e-3,
            )
        else:
            self.output_adapter = None

    def forward(self, x, var_idx, city_id=None):
        # x: [B, 96], standardized
        if self.input_adapter is not None:
            x_in = self.input_adapter(x, var_idx, city_id)
            out = self.timer(x_in)
        else:
            # Output-only modes do not need gradients through Timer.
            with torch.no_grad():
                out = self.timer(x)

        base = out.logits[:, :H].float()

        gamma = self.gamma(var_idx)
        beta = self.beta(var_idx)
        pred = gamma * base + beta

        if self.output_adapter is not None:
            pred = self.output_adapter(pred, var_idx, city_id)

        return pred


class MetricAgg:
    def __init__(self):
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, pred, true):
        mask = np.isfinite(pred) & np.isfinite(true)
        if not mask.any():
            return
        err = pred[mask] - true[mask]
        self.abs_sum += float(np.abs(err).sum())
        self.sq_sum += float((err ** 2).sum())
        self.count += int(err.size)

    def result(self):
        if self.count == 0:
            return {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "count": 0}
        mae = self.abs_sum / self.count
        mse = self.sq_sum / self.count
        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": math.sqrt(mse),
            "count": self.count,
        }


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


def run_one_epoch(model, loader, optimizer, device, train, mse_weight=0.5, o3_loss_weight=1.0, grad_clip=1.0, log_every=100):
    model.train(train)
    model.timer.eval()

    loss_sum = 0.0
    n_loss = 0

    metric = MetricAgg()
    by_var = defaultdict(MetricAgg)

    variables = loader.dataset.variables

    for step, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, dtype=torch.float32, non_blocking=True)
        y_std = batch["y_std"].to(device, dtype=torch.float32, non_blocking=True)
        var_idx = batch["var_idx"].to(device, non_blocking=True)
        city_id = batch["city_id"].to(device, non_blocking=True)

        y_phys = batch["y_phys"].numpy()
        mu = batch["mu"].numpy()[:, None]
        std = batch["std"].numpy()[:, None]
        var_np = batch["var_idx"].numpy()

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            pred_std = model(x, var_idx, city_id)

            # sample-wise weighted loss; useful for O3-targeted multi-variable training
            per_sample_mae = torch.abs(pred_std - y_std).mean(dim=1)
            per_sample_mse = ((pred_std - y_std) ** 2).mean(dim=1)

            if o3_loss_weight != 1.0:
                # variable order comes from processed/stage1_city_var_stats.json
                # ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO', 'temperature', 'relative_humidity']
                o3_idx = 4
                weights = torch.ones_like(per_sample_mae)
                weights = torch.where(var_idx == o3_idx, weights * o3_loss_weight, weights)
                mae_loss = (per_sample_mae * weights).sum() / weights.sum()
                mse_loss = (per_sample_mse * weights).sum() / weights.sum()
            else:
                mae_loss = per_sample_mae.mean()
                mse_loss = per_sample_mse.mean()

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

        for i in range(pred_phys.shape[0]):
            var = variables[int(var_np[i])]
            by_var[var].update(pred_phys[i], y_phys[i])

        if train and (step % log_every == 0 or step == 1):
            r = metric.result()
            print(
                f"  step {step:6d}/{len(loader):6d} "
                f"loss={loss_sum / max(n_loss, 1):.5f} "
                f"MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f}",
                flush=True,
            )

    out = {
        "loss": loss_sum / max(n_loss, 1),
        **metric.result(),
    }

    by_var_rows = []
    for var, agg in sorted(by_var.items()):
        by_var_rows.append({"var_name": var, **agg.result()})

    return out, pd.DataFrame(by_var_rows)


def count_trainable(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_mode", type=str, default="output_affine",
                        choices=["output_affine", "output_mlp", "output_mlp_city", "input_output"])
    parser.add_argument("--only_var", type=str, default="ALL")
    parser.add_argument("--only_city", type=str, default="ALL")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mse_weight", type=float, default=0.5)
    parser.add_argument("--o3_loss_weight", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--emb_dim", type=int, default=16)
    parser.add_argument("--city_emb_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out_dir", type=str, default="checkpoints/stage1_cipda/run")
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==== Args ====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    print("\n==== Loading datasets ====")
    train_set = Stage1Dataset(
        split="train",
        only_var=args.only_var,
        only_city=args.only_city,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    val_set = Stage1Dataset(
        split="val",
        only_var=args.only_var,
        only_city=args.only_city,
        max_samples=args.max_val_samples,
        seed=args.seed + 1,
    )

    train_loader = make_loader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Train samples: {len(train_set):,}")
    print(f"Val samples  : {len(val_set):,}")
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

    model = TimerCIPDA(
        timer=timer,
        n_vars=len(train_set.variables),
        n_cities=len(train_set.cities),
        adapter_mode=args.adapter_mode,
        hidden=args.hidden,
        emb_dim=args.emb_dim,
        city_emb_dim=args.city_emb_dim,
        dropout=args.dropout,
    ).to(device)

    total, trainable = count_trainable(model)
    print(f"Total params    : {total / 1e6:.3f}M")
    print(f"Trainable params: {trainable / 1e6:.6f}M")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_val_rmse = float("inf")

    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    for epoch in range(1, args.epochs + 1):
        print(f"\n==== Epoch {epoch}/{args.epochs} train ====")
        train_metrics, train_by_var = run_one_epoch(
            model, train_loader, optimizer, device,
            train=True,
            mse_weight=args.mse_weight,
            o3_loss_weight=args.o3_loss_weight,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
        )

        print(f"\n==== Epoch {epoch}/{args.epochs} val ====")
        val_metrics, val_by_var = run_one_epoch(
            model, val_loader, optimizer, device,
            train=False,
            mse_weight=args.mse_weight,
            o3_loss_weight=args.o3_loss_weight,
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
        train_by_var.to_csv(out_dir / f"train_by_variable_epoch{epoch}.csv", index=False, encoding="utf-8-sig")
        val_by_var.to_csv(out_dir / f"val_by_variable_epoch{epoch}.csv", index=False, encoding="utf-8-sig")

        print("\nEpoch summary:")
        print(pd.DataFrame([row]))

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
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
