#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate completed CoRA seeds into mean +/- std tables."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    ap.add_argument(
        "--result_root",
        type=Path,
        default=Path("experiments/results/18_cora_baseline"),
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2026, 2027, 2028, 2029, 2030],
    )
    args = ap.parse_args()
    rr = args.result_root if args.result_root.is_absolute() else args.root / args.result_root

    rows = []
    for seed in args.seeds:
        p = rr / f"seed{seed}/metrics_overall.csv"
        if not p.exists():
            print(f"[skip] seed{seed}: missing {p}")
            continue
        r = pd.read_csv(p).iloc[0].to_dict()
        r["seed"] = seed
        rows.append(r)
    if not rows:
        raise RuntimeError("No completed seed metrics found")

    df = pd.DataFrame(rows)
    df.to_csv(rr / "metrics_seeds.csv", index=False, encoding="utf-8-sig")

    out = []
    for metric in ["MAE", "RMSE", "Bias"]:
        x = df[metric].to_numpy(dtype=float)
        out.append({
            "metric": metric,
            "n_seeds": len(x),
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
        })
    sdf = pd.DataFrame(out)
    sdf.to_csv(rr / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    print(df[["seed", "MAE", "RMSE", "Bias"]].to_string(index=False))
    print()
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
