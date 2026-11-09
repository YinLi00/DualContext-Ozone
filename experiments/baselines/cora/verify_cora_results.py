#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit one completed formal CoRA seed against canonical revision-v2 test indices."""

from __future__ import annotations

import argparse
import json
import math
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
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--result_root",
        type=Path,
        default=Path("experiments/results/18_cora_baseline"),
    )
    args = ap.parse_args()
    root = args.root.resolve()
    rr = args.result_root if args.result_root.is_absolute() else root / args.result_root
    d = rr / f"seed{args.seed}"

    required = [
        "pred_o3.npy",
        "true_o3.npy",
        "city_ids.npy",
        "array_id.npy",
        "start_idx.npy",
        "canonical_test_positions.npy",
        "metrics_overall.csv",
        "metrics_by_city.csv",
        "metrics_by_horizon.csv",
        "training_log.csv",
        "run_manifest.json",
        "best.pt",
    ]
    missing = [x for x in required if not (d / x).exists()]
    if missing:
        raise RuntimeError(f"Missing result files: {missing}")

    pred = np.load(d / "pred_o3.npy")
    true = np.load(d / "true_o3.npy")
    cid = np.load(d / "city_ids.npy")
    aid = np.load(d / "array_id.npy")
    start = np.load(d / "start_idx.npy")
    pos = np.load(d / "canonical_test_positions.npy")

    test_path = root / "experiments/data_v2/indices/test.npz"
    z = np.load(test_path, allow_pickle=False)
    n = len(z["array_id"])

    print("=" * 96)
    print(f"CoRA result audit seed={args.seed}")
    print("=" * 96)
    print("pred shape =", pred.shape)
    print("true shape =", true.shape)
    print("test n     =", n)

    if pred.shape != (n, 48) or true.shape != (n, 48):
        raise RuntimeError("Prediction/truth shape is not canonical [106200,48]")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise RuntimeError("Non-finite pred/true")
    if not np.array_equal(aid.astype(np.int32), z["array_id"].astype(np.int32)):
        raise RuntimeError("array_id differs from canonical test index")
    if not np.array_equal(cid.astype(np.int16), z["city_id"].astype(np.int16)):
        raise RuntimeError("city_id differs from canonical test index")
    if not np.array_equal(start.astype(np.int32), z["start_idx"].astype(np.int32)):
        raise RuntimeError("start_idx differs from canonical test index")
    if not np.array_equal(pos.astype(np.int64), np.arange(n, dtype=np.int64)):
        raise RuntimeError("canonical_test_positions is not 0..N-1")

    meta = json.loads(
        (root / "experiments/data_v2/indices/meta.json").read_text(
            encoding="utf-8"
        )
    )
    o3_idx = int(meta["o3_idx"])
    array_paths = list(meta["array_paths"])

    # Strong truth audit: reconstruct physical O3 targets directly from stage1 arrays.
    truth_ok = True
    maxdiff = 0.0
    for this_aid in np.unique(aid):
        rows = np.flatnonzero(aid == this_aid)
        path = root / array_paths[int(this_aid)]
        with np.load(path, allow_pickle=True) as data:
            values = data["values"].astype(np.float32)
        for r in rows:
            st = int(start[r])
            expected = values[st + 96:st + 144, o3_idx]
            diff = float(np.max(np.abs(expected - true[r])))
            maxdiff = max(maxdiff, diff)
            if diff != 0.0:
                truth_ok = False
                break
        if not truth_ok:
            break
    print("truth max abs diff =", maxdiff)
    if not truth_ok:
        raise RuntimeError("Saved true_o3 does not exactly match stage1 arrays")

    delta = pred.astype(np.float64) - true.astype(np.float64)
    mae = float(np.abs(delta).mean())
    rmse = float(np.sqrt(np.square(delta).mean()))
    bias = float(delta.mean())
    csv = pd.read_csv(d / "metrics_overall.csv").iloc[0]
    for name, val in [("MAE", mae), ("RMSE", rmse), ("Bias", bias)]:
        if not math.isclose(float(csv[name]), val, rel_tol=0, abs_tol=1e-9):
            raise RuntimeError(f"{name} CSV mismatch: {csv[name]} vs {val}")

    man = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    if int(man["samples"]["test"]) != n:
        raise RuntimeError("Manifest test count mismatch")
    if int(man["protocol"]["o3_idx"]) != o3_idx:
        raise RuntimeError("Manifest O3 index mismatch")

    print(f"MAE  = {mae:.6f}")
    print(f"RMSE = {rmse:.6f}")
    print(f"Bias = {bias:.6f}")
    print("cities present =", len(np.unique(cid)))
    print("[PASS] canonical identity, truth, finiteness, and metrics all verified")


if __name__ == "__main__":
    main()
