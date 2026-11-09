import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[2]
COMPACT_ROOT = ROOT / "processed" / "stage1_indices_compact_qc"
STATS_PATH = ROOT / "processed" / "stage1_city_var_stats.json"
MODEL_PATH = ROOT / "models_hf_local" / "timer_base_84m"

L = 96
H = 48


class Stage1CompactDataset(Dataset):
    def __init__(self, split="test", only_var=None, max_samples=None):
        self.split = split
        self.meta = json.loads((COMPACT_ROOT / "meta.json").read_text(encoding="utf-8"))
        self.stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))

        idx = np.load(COMPACT_ROOT / f"{split}.npz")
        array_id = idx["array_id"]
        city_id = idx["city_id"]
        var_idx = idx["var_idx"]
        start_idx = idx["start_idx"]

        self.variables = self.meta["variables"]
        self.cities = self.meta["cities"]
        self.array_paths = self.meta["array_paths"]

        if only_var is not None:
            if only_var not in self.variables:
                raise ValueError(f"Unknown variable {only_var}. Available: {self.variables}")
            target_vid = self.variables.index(only_var)
            mask = var_idx == target_vid
            array_id = array_id[mask]
            city_id = city_id[mask]
            var_idx = var_idx[mask]
            start_idx = start_idx[mask]

        if max_samples is not None and max_samples > 0:
            array_id = array_id[:max_samples]
            city_id = city_id[:max_samples]
            var_idx = var_idx[:max_samples]
            start_idx = start_idx[:max_samples]

        self.array_id = array_id.astype(np.int32)
        self.city_id = city_id.astype(np.int16)
        self.var_idx = var_idx.astype(np.int8)
        self.start_idx = start_idx.astype(np.int32)

        self.cache = {}

    def __len__(self):
        return len(self.array_id)

    def _load_array(self, aid):
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

        values = self._load_array(aid)
        x = values[st:st+L, vid]
        y = values[st+L:st+L+H, vid]

        stat = self.stats["city_variable_stats"][city][var]
        mu = np.float32(stat["mean"])
        std = np.float32(stat["std"])

        x_std = (x - mu) / std
        y_std = (y - mu) / std

        return {
            "x": torch.tensor(x_std, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),          # physical y
            "y_std": torch.tensor(y_std, dtype=torch.float32),  # standardized y
            "mu": torch.tensor(mu, dtype=torch.float32),
            "std": torch.tensor(std, dtype=torch.float32),
            "city_id": torch.tensor(cid, dtype=torch.long),
            "var_idx": torch.tensor(vid, dtype=torch.long),
        }


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
            "RMSE": float(np.sqrt(mse)),
            "count": self.count,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--only_var", type=str, default="O3", help="Use O3 first; set ALL for all variables.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means full split.")
    parser.add_argument("--out_dir", type=str, default="results/stage1_timer_zeroshot")
    parser.add_argument("--model_dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--infer_mode", type=str, default="generate", choices=["generate", "forward"])
    args = parser.parse_args()

    only_var = None if args.only_var.upper() == "ALL" else args.only_var
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = Stage1CompactDataset(
        split=args.split,
        only_var=only_var,
        max_samples=args.max_samples if args.max_samples > 0 else None,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Split       : {args.split}")
    print(f"Only var    : {only_var if only_var else 'ALL'}")
    print(f"Samples     : {len(dataset):,}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Num workers : {args.num_workers}")
    print(f"Out dir     : {out_dir}")

    print("\nLoading Timer-base-84m...")
    if args.model_dtype == "fp16":
        load_dtype = torch.float16
    elif args.model_dtype == "bf16":
        load_dtype = torch.bfloat16
    else:
        load_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=load_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    first_param = next(model.parameters())
    device = first_param.device
    model_dtype = first_param.dtype
    print("Model device:", device)
    print("Model dtype :", model_dtype)

    cities = dataset.cities
    variables = dataset.variables

    overall = MetricAgg()
    by_city = defaultdict(MetricAgg)
    by_var = defaultdict(MetricAgg)
    by_city_var = defaultdict(MetricAgg)

    n_batches = len(loader)

    with torch.no_grad():
        for bi, batch in enumerate(loader, start=1):
            x = batch["x"].to(device=device, dtype=model_dtype, non_blocking=True)
            y = batch["y"].cpu().numpy()
            mu = batch["mu"].cpu().numpy()[:, None]
            std = batch["std"].cpu().numpy()[:, None]
            city_id = batch["city_id"].cpu().numpy()
            var_idx = batch["var_idx"].cpu().numpy()

            # Timer expects [B, 96].
            # generate: official generation path; forward: differentiable path for adapter training.
            if args.infer_mode == "generate":
                pred_std = model.generate(x, max_new_tokens=H)
                if pred_std.shape[-1] != H:
                    pred_std = pred_std[:, -H:]
            else:
                out = model(x)
                pred_std = out.logits
                if pred_std.shape[-1] < H:
                    raise RuntimeError(f"Forward logits too short: {pred_std.shape}")
                pred_std = pred_std[:, :H]

            pred_std = pred_std.float().cpu().numpy()

            if not np.isfinite(pred_std).all():
                n_bad = np.size(pred_std) - np.isfinite(pred_std).sum()
                print(f"[WARN] non-finite pred_std at batch {bi}: bad={n_bad}", flush=True)

            pred = pred_std * std + mu

            overall.update(pred, y)

            for i in range(pred.shape[0]):
                city = cities[int(city_id[i])]
                var = variables[int(var_idx[i])]
                by_city[city].update(pred[i], y[i])
                by_var[var].update(pred[i], y[i])
                by_city_var[(city, var)].update(pred[i], y[i])

            if bi % 50 == 0 or bi == 1 or bi == n_batches:
                res = overall.result()
                print(
                    f"[{bi:5d}/{n_batches:5d}] "
                    f"MAE={res['MAE']:.4f}, RMSE={res['RMSE']:.4f}, count={res['count']:,}",
                    flush=True,
                )

    # Save metrics.
    overall_df = pd.DataFrame([{
        "split": args.split,
        "only_var": only_var if only_var else "ALL",
        "infer_mode": args.infer_mode,
        **overall.result(),
    }])
    overall_df.to_csv(out_dir / "metrics_overall.csv", index=False, encoding="utf-8-sig")

    city_rows = []
    for city, agg in sorted(by_city.items()):
        city_rows.append({"city": city, **agg.result()})
    pd.DataFrame(city_rows).to_csv(out_dir / "metrics_by_city.csv", index=False, encoding="utf-8-sig")

    var_rows = []
    for var, agg in sorted(by_var.items()):
        var_rows.append({"var_name": var, **agg.result()})
    pd.DataFrame(var_rows).to_csv(out_dir / "metrics_by_variable.csv", index=False, encoding="utf-8-sig")

    cv_rows = []
    for (city, var), agg in sorted(by_city_var.items()):
        cv_rows.append({"city": city, "var_name": var, **agg.result()})
    pd.DataFrame(cv_rows).to_csv(out_dir / "metrics_by_city_variable.csv", index=False, encoding="utf-8-sig")

    print("\n==== Done ====")
    print(overall_df)
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
