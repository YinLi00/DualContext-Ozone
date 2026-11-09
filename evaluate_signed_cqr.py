#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Independent signed-CQR analysis for Monotonic Quantile-DualContext v4.

Expected root:
ROOT/
  seed2026/
    pred.npy
    true.npy
    city_id.npy
    quantile_pred_raw.npy
    calibration/
      true.npy
      city_id.npy
      quantile_pred_raw.npy
      manifest.json
  ...
  seed2030/

The script computes, separately for nominal 80% and 90% intervals:
  1) Raw quantile interval;
  2) Global signed CQR;
  3) Horizon-wise signed CQR;
  4) City-horizon signed CQR.

Signed CQR score:
    s = max(q_lower - y, y - q_upper)
(no max(..., 0)), so a negative conformal correction can SHRINK an
over-conservative interval.

It also evaluates a physically truncated sensitivity:
    lower = max(lower, 0), upper = max(upper, 0)
which is appropriate for non-negative O3 concentration.  The untruncated
results remain the primary statistical results; floor=0 is reported separately.

This script does NOT use test labels to determine any correction.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


SEEDS = [2026, 2027, 2028, 2029, 2030]
H = 48
INTERVALS = {
    "80": {"alpha": 0.20, "li": 1, "ui": 3, "nominal": 0.80},
    "90": {"alpha": 0.10, "li": 0, "ui": 4, "nominal": 0.90},
}


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--physical-floor", type=float, default=0.0)
    return p.parse_args()


def seed_root(root: Path, seed: int) -> Path:
    direct = root / f"seed{seed}"
    if direct.exists():
        return direct
    nested = root / "q_quantile_monotonic_cal" / f"seed{seed}"
    if nested.exists():
        return nested
    raise FileNotFoundError(
        f"Cannot locate seed{seed} under {root}; expected either "
        f"{direct} or {nested}"
    )


def finite_conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    """
    Split-conformal finite-sample order statistic:
        k = ceil((n+1) * coverage)
    clipped to [1, n].
    Works for signed scores and therefore can return a negative correction.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("No finite calibration scores.")
    n = s.size
    k = int(math.ceil((n + 1) * float(coverage)))
    k = max(1, min(k, n))
    return float(np.partition(s, k - 1)[k - 1])


def signed_scores(y, lo, hi):
    return np.maximum(lo - y, y - hi)


def interval_metrics(y, lo, hi, alpha, median):
    y = np.asarray(y, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    median = np.asarray(median, dtype=np.float64)

    finite = (
        np.isfinite(y)
        & np.isfinite(lo)
        & np.isfinite(hi)
        & np.isfinite(median)
    )
    y = y[finite]
    lo = lo[finite]
    hi = hi[finite]
    median = median[finite]

    invalid = lo > hi
    invalid_rate = float(np.mean(invalid)) if len(y) else np.nan

    cover = (y >= lo) & (y <= hi)
    width = hi - lo
    score = (
        width
        + (2.0 / alpha) * (lo - y) * (y < lo)
        + (2.0 / alpha) * (y - hi) * (y > hi)
    )

    return {
        "n": int(y.size),
        "PICP": float(np.mean(cover)),
        "MPIW": float(np.mean(width)),
        "IS": float(np.mean(score)),
        "invalid_rate": invalid_rate,
        "median_MAE": float(np.mean(np.abs(median - y))),
    }


def wis_from_intervals(y, median, lo80, hi80, lo90, hi90):
    y = np.asarray(y, dtype=np.float64)
    median = np.asarray(median, dtype=np.float64)
    is80 = (
        (hi80 - lo80)
        + 10.0 * (lo80 - y) * (y < lo80)
        + 10.0 * (y - hi80) * (y > hi80)
    )
    is90 = (
        (hi90 - lo90)
        + 20.0 * (lo90 - y) * (y < lo90)
        + 20.0 * (y - hi90) * (y > hi90)
    )
    return float(
        np.mean(
            (0.5 * np.abs(y - median) + 0.10 * is80 + 0.05 * is90)
            / 2.5
        )
    )


def raw_negative_audit(seed, cal_q, test_q, floor):
    rows = []
    for split, q in [("calibration", cal_q), ("test", test_q)]:
        for idx, name in [(0, "q05"), (1, "q10"), (2, "q50"), (3, "q90"), (4, "q95")]:
            vals = np.asarray(q[..., idx], dtype=np.float64)
            rows.append({
                "seed": seed,
                "split": split,
                "quantile": name,
                "n": int(vals.size),
                "negative_count": int(np.sum(vals < floor)),
                "negative_rate": float(np.mean(vals < floor)),
                "min": float(np.min(vals)),
                "p001": float(np.quantile(vals, 0.001)),
                "p01": float(np.quantile(vals, 0.01)),
                "median": float(np.median(vals)),
                "max": float(np.max(vals)),
            })
    return rows


def build_corrections(cal_y, cal_q, cal_city, interval_key):
    spec = INTERVALS[interval_key]
    alpha = spec["alpha"]
    li, ui = spec["li"], spec["ui"]
    coverage = 1.0 - alpha

    scores = signed_scores(cal_y, cal_q[..., li], cal_q[..., ui])

    global_corr = finite_conformal_quantile(scores, coverage)

    horizon_corr = {}
    for h in range(H):
        horizon_corr[h] = finite_conformal_quantile(
            scores[:, h], coverage
        )

    city_horizon_corr = {}
    for cid in np.unique(cal_city):
        mask = cal_city == cid
        for h in range(H):
            city_horizon_corr[(int(cid), h)] = finite_conformal_quantile(
                scores[mask, h], coverage
            )

    return global_corr, horizon_corr, city_horizon_corr


def apply_mode(q, city, interval_key, mode, corrections):
    spec = INTERVALS[interval_key]
    lo = np.asarray(q[..., spec["li"]], dtype=np.float64).copy()
    hi = np.asarray(q[..., spec["ui"]], dtype=np.float64).copy()

    if mode == "raw":
        return lo, hi

    global_corr, horizon_corr, city_horizon_corr = corrections

    if mode == "global":
        lo -= global_corr
        hi += global_corr
    elif mode == "horizon":
        for h in range(H):
            c = horizon_corr[h]
            lo[:, h] -= c
            hi[:, h] += c
    elif mode == "city_horizon":
        for cid in np.unique(city):
            mask = city == cid
            for h in range(H):
                c = city_horizon_corr[(int(cid), h)]
                lo[mask, h] -= c
                hi[mask, h] += c
    else:
        raise ValueError(mode)

    return lo, hi


def aggregate_mean_std(df, group_cols, metrics):
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for m in metrics:
            x = g[m].to_numpy(float)
            row[f"{m}_mean"] = float(np.mean(x))
            row[f"{m}_std"] = (
                float(np.std(x, ddof=1)) if len(x) > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    a = args()
    root = a.root.resolve()
    out = (
        a.out_dir.resolve()
        if a.out_dir is not None
        else root / "signed_cqr_analysis"
    )
    out.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    horizon_rows = []
    city_rows = []
    correction_rows = []
    negative_rows = []
    protocol_rows = []

    for seed in a.seeds:
        sd = seed_root(root, seed)
        cald = sd / "calibration"

        required = [
            sd / "true.npy",
            sd / "city_id.npy",
            sd / "quantile_pred_raw.npy",
            cald / "true.npy",
            cald / "city_id.npy",
            cald / "quantile_pred_raw.npy",
            cald / "manifest.json",
        ]
        missing = [p for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing qv4/CQR input:\n  "
                + "\n  ".join(str(p) for p in missing)
            )

        y = np.load(sd / "true.npy", allow_pickle=False).astype(np.float64)
        city = np.load(sd / "city_id.npy", allow_pickle=False).reshape(-1)
        q = np.load(
            sd / "quantile_pred_raw.npy", allow_pickle=False
        ).astype(np.float64)

        cy = np.load(
            cald / "true.npy", allow_pickle=False
        ).astype(np.float64)
        ccity = np.load(
            cald / "city_id.npy", allow_pickle=False
        ).reshape(-1)
        cq = np.load(
            cald / "quantile_pred_raw.npy", allow_pickle=False
        ).astype(np.float64)

        manifest = json.loads(
            (cald / "manifest.json").read_text(encoding="utf-8")
        )
        if not manifest.get(
            "calibration_not_used_for_checkpoint_selection", False
        ):
            raise RuntimeError(
                f"seed{seed}: calibration independence manifest failed"
            )

        if q.shape[-2:] != (48, 5) or cq.shape[-2:] != (48, 5):
            raise RuntimeError(
                f"seed{seed}: unexpected q shapes test={q.shape}, cal={cq.shape}"
            )
        if np.any(q[..., :-1] > q[..., 1:]) or np.any(
            cq[..., :-1] > cq[..., 1:]
        ):
            raise RuntimeError(
                f"seed{seed}: quantile crossing found before CQR"
            )
        if len(np.unique(city)) != 22 or len(np.unique(ccity)) != 22:
            raise RuntimeError(
                f"seed{seed}: expected all 22 cities in test/calibration"
            )

        negative_rows.extend(
            raw_negative_audit(
                seed, cq, q, floor=a.physical_floor
            )
        )

        corr_by_interval = {}
        for ikey in ["80", "90"]:
            corr = build_corrections(cy, cq, ccity, ikey)
            corr_by_interval[ikey] = corr
            global_corr, horizon_corr, city_horizon_corr = corr
            correction_rows.append({
                "seed": seed,
                "interval": ikey,
                "mode": "global",
                "city_id": np.nan,
                "horizon": np.nan,
                "correction": global_corr,
            })
            for h, c in horizon_corr.items():
                correction_rows.append({
                    "seed": seed,
                    "interval": ikey,
                    "mode": "horizon",
                    "city_id": np.nan,
                    "horizon": h + 1,
                    "correction": c,
                })
            for (cid, h), c in city_horizon_corr.items():
                correction_rows.append({
                    "seed": seed,
                    "interval": ikey,
                    "mode": "city_horizon",
                    "city_id": cid,
                    "horizon": h + 1,
                    "correction": c,
                })

        protocol_rows.append({
            "seed": seed,
            "calibration_n_windows": int(cy.shape[0]),
            "test_n_windows": int(y.shape[0]),
            "calibration_cities": int(np.unique(ccity).size),
            "test_cities": int(np.unique(city).size),
            "strict_independent_calibration": True,
            "calibration_cutoff_iso": manifest.get("cutoff_iso"),
            "calibration_crossing_pairs": manifest.get("crossing_pairs"),
        })

        for mode in ["raw", "global", "horizon", "city_horizon"]:
            interval_data = {}
            for ikey in ["80", "90"]:
                lo, hi = apply_mode(
                    q, city, ikey, mode, corr_by_interval[ikey]
                )
                interval_data[ikey] = (lo, hi)

            for floor_mode in ["untruncated", "floor0"]:
                use = {}
                for ikey, (lo, hi) in interval_data.items():
                    if floor_mode == "floor0":
                        lo2 = np.maximum(lo, a.physical_floor)
                        hi2 = np.maximum(hi, a.physical_floor)
                    else:
                        lo2, hi2 = lo, hi
                    use[ikey] = (lo2, hi2)

                invalid80 = float(np.mean(use["80"][0] > use["80"][1]))
                invalid90 = float(np.mean(use["90"][0] > use["90"][1]))
                if invalid80 > 0 or invalid90 > 0:
                    raise RuntimeError(
                        f"seed{seed} {mode} {floor_mode}: signed CQR produced "
                        f"invalid intervals: 80={invalid80}, 90={invalid90}"
                    )

                m80 = interval_metrics(
                    y, use["80"][0], use["80"][1], 0.20, q[..., 2]
                )
                m90 = interval_metrics(
                    y, use["90"][0], use["90"][1], 0.10, q[..., 2]
                )
                wis = wis_from_intervals(
                    y, q[..., 2],
                    use["80"][0], use["80"][1],
                    use["90"][0], use["90"][1],
                )

                overall_rows.append({
                    "seed": seed,
                    "mode": mode,
                    "physical": floor_mode,
                    "PICP80": m80["PICP"],
                    "coverage_error80": m80["PICP"] - 0.80,
                    "abs_coverage_error80": abs(m80["PICP"] - 0.80),
                    "MPIW80": m80["MPIW"],
                    "IS80": m80["IS"],
                    "PICP90": m90["PICP"],
                    "coverage_error90": m90["PICP"] - 0.90,
                    "abs_coverage_error90": abs(m90["PICP"] - 0.90),
                    "MPIW90": m90["MPIW"],
                    "IS90": m90["IS"],
                    "WIS": wis,
                    "median_MAE": m80["median_MAE"],
                })

                # Horizon metrics.
                for h in range(H):
                    hm80 = interval_metrics(
                        y[:, h],
                        use["80"][0][:, h],
                        use["80"][1][:, h],
                        0.20,
                        q[:, h, 2],
                    )
                    hm90 = interval_metrics(
                        y[:, h],
                        use["90"][0][:, h],
                        use["90"][1][:, h],
                        0.10,
                        q[:, h, 2],
                    )
                    hwis = wis_from_intervals(
                        y[:, h], q[:, h, 2],
                        use["80"][0][:, h],
                        use["80"][1][:, h],
                        use["90"][0][:, h],
                        use["90"][1][:, h],
                    )
                    horizon_rows.append({
                        "seed": seed,
                        "mode": mode,
                        "physical": floor_mode,
                        "horizon": h + 1,
                        "PICP80": hm80["PICP"],
                        "coverage_error80": hm80["PICP"] - 0.80,
                        "MPIW80": hm80["MPIW"],
                        "IS80": hm80["IS"],
                        "PICP90": hm90["PICP"],
                        "coverage_error90": hm90["PICP"] - 0.90,
                        "MPIW90": hm90["MPIW"],
                        "IS90": hm90["IS"],
                        "WIS": hwis,
                    })

                # City metrics.
                for cid in np.unique(city):
                    mask = city == cid
                    cm80 = interval_metrics(
                        y[mask],
                        use["80"][0][mask],
                        use["80"][1][mask],
                        0.20,
                        q[mask, :, 2],
                    )
                    cm90 = interval_metrics(
                        y[mask],
                        use["90"][0][mask],
                        use["90"][1][mask],
                        0.10,
                        q[mask, :, 2],
                    )
                    cwis = wis_from_intervals(
                        y[mask], q[mask, :, 2],
                        use["80"][0][mask],
                        use["80"][1][mask],
                        use["90"][0][mask],
                        use["90"][1][mask],
                    )
                    city_rows.append({
                        "seed": seed,
                        "mode": mode,
                        "physical": floor_mode,
                        "city_id": int(cid),
                        "PICP80": cm80["PICP"],
                        "coverage_error80": cm80["PICP"] - 0.80,
                        "MPIW80": cm80["MPIW"],
                        "IS80": cm80["IS"],
                        "PICP90": cm90["PICP"],
                        "coverage_error90": cm90["PICP"] - 0.90,
                        "MPIW90": cm90["MPIW"],
                        "IS90": cm90["IS"],
                        "WIS": cwis,
                    })

        print(f"[DONE] seed {seed}", flush=True)

    overall = pd.DataFrame(overall_rows)
    horizon = pd.DataFrame(horizon_rows)
    citydf = pd.DataFrame(city_rows)
    corrdf = pd.DataFrame(correction_rows)
    negdf = pd.DataFrame(negative_rows)
    protodf = pd.DataFrame(protocol_rows)

    metrics = [
        "PICP80", "coverage_error80", "abs_coverage_error80",
        "MPIW80", "IS80",
        "PICP90", "coverage_error90", "abs_coverage_error90",
        "MPIW90", "IS90", "WIS", "median_MAE",
    ]
    overall_summary = aggregate_mean_std(
        overall, ["mode", "physical"], metrics
    )
    horizon_summary = aggregate_mean_std(
        horizon, ["mode", "physical", "horizon"],
        [
            "PICP80", "coverage_error80", "MPIW80", "IS80",
            "PICP90", "coverage_error90", "MPIW90", "IS90", "WIS",
        ],
    )
    city_summary = aggregate_mean_std(
        citydf, ["mode", "physical", "city_id"],
        [
            "PICP80", "coverage_error80", "MPIW80", "IS80",
            "PICP90", "coverage_error90", "MPIW90", "IS90", "WIS",
        ],
    )

    overall.to_csv(
        out / "cqr_overall_by_seed.csv",
        index=False, encoding="utf-8-sig"
    )
    overall_summary.to_csv(
        out / "cqr_overall_mean_std.csv",
        index=False, encoding="utf-8-sig"
    )
    horizon.to_csv(
        out / "cqr_horizon_by_seed.csv",
        index=False, encoding="utf-8-sig"
    )
    horizon_summary.to_csv(
        out / "cqr_horizon_mean_std.csv",
        index=False, encoding="utf-8-sig"
    )
    citydf.to_csv(
        out / "cqr_city_by_seed.csv",
        index=False, encoding="utf-8-sig"
    )
    city_summary.to_csv(
        out / "cqr_city_mean_std.csv",
        index=False, encoding="utf-8-sig"
    )
    corrdf.to_csv(
        out / "signed_cqr_corrections.csv",
        index=False, encoding="utf-8-sig"
    )
    negdf.to_csv(
        out / "negative_quantile_audit.csv",
        index=False, encoding="utf-8-sig"
    )
    protodf.to_csv(
        out / "protocol_audit.csv",
        index=False, encoding="utf-8-sig"
    )

    print()
    print("=" * 96)
    print("SIGNED-CQR SUMMARY (mean across seeds)")
    print("=" * 96)
    show = overall_summary[
        overall_summary["physical"] == "untruncated"
    ].copy()
    cols = [
        "mode",
        "PICP80_mean", "MPIW80_mean", "IS80_mean",
        "PICP90_mean", "MPIW90_mean", "IS90_mean", "WIS_mean",
    ]
    print(show[cols].to_string(index=False))
    print()
    print("Physical floor=0 sensitivity:")
    show2 = overall_summary[
        overall_summary["physical"] == "floor0"
    ].copy()
    print(show2[cols].to_string(index=False))
    print()
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
