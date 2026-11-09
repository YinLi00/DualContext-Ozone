#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = [
    "Timer",
    "Sundial",
    "TimeMoE-200M",
    "SEMPO",
    "Chronos-Bolt-Base",
    "Chronos-2",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = p.parse_args()
    base = args.root / "experiments/results/03_zeroshot_baselines_v2_reindexed"
    rows = []
    for model in MODELS:
        d = base / model
        metric = d / "metrics_overall.csv"
        pred_path = d / "pred.npy"
        true_path = d / "true.npy"
        if not metric.exists() or not pred_path.exists() or not true_path.exists():
            rows.append({"model_name": model, "status": "MISSING"})
            continue
        df = pd.read_csv(metric)
        row = df.iloc[0].to_dict()
        pred = np.load(pred_path, mmap_mode="r", allow_pickle=False)
        true = np.load(true_path, mmap_mode="r", allow_pickle=False)
        row.update(
            {
                "model_name": model,
                "status": "COMPLETE",
                "pred_shape": str(tuple(pred.shape)),
                "true_shape": str(tuple(true.shape)),
            }
        )
        if pred.shape != (106200, 48) or true.shape != (106200, 48):
            row["status"] = "BAD_SHAPE"
        rows.append(row)
    out = pd.DataFrame(rows)
    preferred = ["model_name", "status", "MAE", "MSE", "RMSE", "count", "pred_shape", "true_shape"]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    out = out[cols]
    out.to_csv(base / "summary_all_6_zeroshot_v2.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print("Saved:", base / "summary_all_6_zeroshot_v2.csv")


if __name__ == "__main__":
    main()
