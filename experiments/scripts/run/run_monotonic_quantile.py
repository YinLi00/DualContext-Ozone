#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runner for Monotonic Quantile-DualContext v4 with an independent calibration split.

Install:
  experiments/scripts/train/train_monotonic_quantile.py
  experiments/scripts/run/run_monotonic_quantile.py

Run:
  python experiments/scripts/run/run_monotonic_quantile.py formal
  python experiments/scripts/run/run_monotonic_quantile.py status
  python experiments/scripts/run/run_monotonic_quantile.py audit

Result root:
  experiments/results/17_quantile_monotonic_v4_cal/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = [2026, 2027, 2028, 2029, 2030]


def args():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["formal", "status", "audit"])
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--cal_fraction", type=float, default=0.25)
    return p.parse_args()


def trainer(root: Path) -> Path:
    return (
        root
        / "experiments/scripts/train/"
        "train_monotonic_quantile.py"
    )


def rr(root: Path) -> Path:
    return root / "experiments/results/17_quantile_monotonic_v4_cal"


def sd(root: Path, seed: int) -> Path:
    return rr(root) / "q_quantile_monotonic_cal" / f"seed{seed}"


def complete(out: Path) -> bool:
    required = [
        out / "run_summary.json",
        out / "pred.npy",
        out / "true.npy",
        out / "quantile_pred_raw.npy",
        out / "val_cal_split_manifest.json",
        out / "calibration/manifest.json",
        out / "calibration/true.npy",
        out / "calibration/quantile_pred_raw.npy",
        out / "calibration/city_id.npy",
    ]
    return all(p.exists() for p in required)


def formal(a):
    root = a.root.expanduser().resolve()
    t = trainer(root)
    if not t.exists():
        raise FileNotFoundError(t)

    for seed in a.seeds:
        out = sd(root, seed)
        if complete(out):
            print(f"[SKIP COMPLETE] seed {seed}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(t),
            "--root", str(root),
            "--variant", "q_quantile_full",
            "--seed", str(seed),
            "--epochs", "20",
            "--patience", "5",
            "--batch_size", "64",
            "--grad_accum", "1",
            "--num_workers", str(a.num_workers),
            "--adapter_lr", "0.001",
            "--weight_decay", "0.0001",
            "--grad_clip", "1.0",
            "--input_hidden", "64",
            "--output_hidden", "128",
            "--context_hidden", "128",
            "--emb_dim", "16",
            "--city_emb_dim", "16",
            "--dropout", "0.05",
            "--corr_temperature", "2.0",
            "--lambda_mid_mae", "1.0",
            "--lambda_mid_mse", "0.5",
            "--lambda_cross", "0.0",
            "--quantile_inner_gap_init", "0.80",
            "--quantile_outer_gap_init", "0.40",
            "--cal_fraction", str(a.cal_fraction),
            "--asset_mode", "formal",
            "--max_train_samples", "0",
            "--max_val_samples", "0",
            "--max_test_samples", "0",
            "--device", "cuda",
            "--require_cuda",
            "--amp_bf16", "0",
            "--gradient_checkpointing", "0",
            "--save_arrays", "1",
            "--keep_checkpoint", "1",
            "--log_every", "1000",
            "--out_dir", str(out),
        ]
        print("\n" + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


def status(a):
    root = a.root.expanduser().resolve()
    print("QV4 independent-calibration status")
    print("-" * 84)
    done = 0
    for seed in a.seeds:
        out = sd(root, seed)
        if complete(out):
            done += 1
            state = "COMPLETE"
        else:
            state = "PENDING"
        split = "-"
        if (out / "val_cal_split_manifest.json").exists():
            try:
                m = json.loads(
                    (out / "val_cal_split_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                split = (
                    f"val={m.get('validation_selection_samples')} "
                    f"cal={m.get('calibration_samples')} "
                    f"drop={m.get('dropped_boundary_samples')}"
                )
            except Exception:
                pass
        print(f"seed {seed}: {state:8s} {split}")
    print("-" * 84)
    print(f"TOTAL {done}/{len(a.seeds)}")


def audit(a):
    root = a.root.expanduser().resolve()
    rows = []
    for seed in a.seeds:
        out = sd(root, seed)
        if not complete(out):
            print(f"[SKIP] incomplete seed {seed}")
            continue

        split = json.loads(
            (out / "val_cal_split_manifest.json").read_text(encoding="utf-8")
        )
        calm = json.loads(
            (out / "calibration/manifest.json").read_text(encoding="utf-8")
        )

        q = np.load(out / "quantile_pred_raw.npy", mmap_mode="r")
        pred = np.load(out / "pred.npy", mmap_mode="r")
        true = np.load(out / "true.npy", mmap_mode="r")

        cq = np.load(
            out / "calibration/quantile_pred_raw.npy", mmap_mode="r"
        )
        cy = np.load(out / "calibration/true.npy", mmap_mode="r")
        cc = np.load(out / "calibration/city_id.npy", mmap_mode="r")

        test_cross = int((q[..., :-1] > q[..., 1:]).sum())
        cal_cross = int((cq[..., :-1] > cq[..., 1:]).sum())

        rows.append({
            "seed": seed,
            "strict_no_shared_timestamp": bool(
                split["strict_no_shared_timestamp"]
            ),
            "validation_selection_samples": int(
                split["validation_selection_samples"]
            ),
            "calibration_samples": int(split["calibration_samples"]),
            "dropped_boundary_samples": int(
                split["dropped_boundary_samples"]
            ),
            "cutoff_iso": split["cutoff_iso"],
            "latest_val_end_iso": split["latest_validation_used_end_iso"],
            "earliest_cal_start_iso": split[
                "earliest_calibration_used_start_iso"
            ],
            "test_crossing_pairs": test_cross,
            "calibration_crossing_pairs": cal_cross,
            "test_all_finite": bool(
                np.isfinite(q).all()
                and np.isfinite(pred).all()
                and np.isfinite(true).all()
            ),
            "cal_all_finite": bool(
                np.isfinite(cq).all()
                and np.isfinite(cy).all()
            ),
            "calibration_cities": int(np.unique(cc).size),
            "test_q50_pred_max_abs": float(
                np.max(
                    np.abs(
                        np.asarray(q[..., 2], dtype=np.float64)
                        - np.asarray(pred, dtype=np.float64)
                    )
                )
            ),
            "test_q05_max_abs": float(np.max(np.abs(q[..., 0]))),
            "test_q95_max_abs": float(np.max(np.abs(q[..., 4]))),
            "cal_q05_max_abs": float(np.max(np.abs(cq[..., 0]))),
            "cal_q95_max_abs": float(np.max(np.abs(cq[..., 4]))),
            "calibration_not_used_for_checkpoint_selection": bool(
                calm["calibration_not_used_for_checkpoint_selection"]
            ),
        })

    df = pd.DataFrame(rows)
    outcsv = rr(root) / "audit_v4_independent_calibration.csv"
    outcsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outcsv, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"\nSaved: {outcsv}")

    if not df.empty:
        hard = (
            df["strict_no_shared_timestamp"].all()
            and df["calibration_not_used_for_checkpoint_selection"].all()
            and (df["test_crossing_pairs"] == 0).all()
            and (df["calibration_crossing_pairs"] == 0).all()
            and df["test_all_finite"].all()
            and df["cal_all_finite"].all()
            and (df["calibration_cities"] == 22).all()
            and (df["test_q50_pred_max_abs"] <= 5e-5).all()
        )
        if not hard:
            raise RuntimeError("QV4 protocol audit FAILED.")
        print("\n[PASS] Independent calibration protocol and monotonic arrays passed.")


def main():
    a = args()
    if a.command == "formal":
        formal(a)
    elif a.command == "status":
        status(a)
    else:
        audit(a)


if __name__ == "__main__":
    main()
