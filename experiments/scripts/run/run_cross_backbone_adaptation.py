#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native revision-v2 cross-backbone Output Adapter controller."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODELS = ["sundial", "timemoe_200m", "sempo"]
DISPLAY_NAMES = {
    "sundial": "Sundial",
    "timemoe_200m": "TimeMoE-200M",
    "sempo": "SEMPO",
}
SEEDS = [2026, 2027, 2028, 2029, 2030]
SPLITS = ["train", "val"]
EXPECTED_SHAPE = (106_200, 48)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "command",
        choices=[
            "setup", "prepare-input", "smoke",
            "infer", "formal", "status", "summarize",
        ],
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    p.add_argument("--models", nargs="*", default=MODELS)
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--cpu_threads", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def result_root(root: Path) -> Path:
    return root / "experiments/results/12_cross_backbone_output_adapter_v2"


def script(root: Path, relative: str) -> Path:
    return root / "experiments/scripts" / relative


def safe_symlink(target: Path, link: Path) -> str:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return "existing"
        link.unlink()
    elif link.exists():
        return "real_path_preserved"
    relative = os.path.relpath(target.resolve(), link.parent.resolve())
    os.symlink(relative, link)
    return "created"


def setup(root: Path) -> dict[str, Any]:
    out = result_root(root)
    for rel in [
        "00_native_cache/input",
        "00_native_cache/smoke_raw/sundial/test",
        "00_native_cache/smoke_raw/timemoe_200m/test",
        "00_native_cache/smoke_raw/sempo/test",
        "00_native_cache/smoke_aligned",
        "00_native_cache/raw/sundial",
        "00_native_cache/raw/timemoe_200m",
        "00_native_cache/raw/sempo",
        "00_native_cache/aligned",
        "00_smoke_cpu",
        "01_output_adapter/sundial",
        "01_output_adapter/timemoe_200m",
        "01_output_adapter/sempo",
        "03_summary",
        "logs",
    ]:
        (out / rel).mkdir(parents=True, exist_ok=True)

    timer_p1 = (
        root / "experiments/results/"
        "10_adapter_time_ablation_v2/"
        "01_progressive_adapter/p1_o3_output"
    )
    link_state = safe_symlink(timer_p1, out / "02_timer_reference")
    design = {
        "protocol": "revision-v2",
        "native_backbones": MODELS,
        "seeds": SEEDS,
        "new_formal_trainings": 15,
        "timer_output_adapter_reused": str(timer_p1),
        "timer_link_state": link_state,
        "native_inference": {
            "Sundial": "20-sample mean, physical input + RevIN",
            "TimeMoE-200M": "physical input + RevIN",
            "SEMPO": "repeat_context padding to 512",
        },
        "test_forecasts": (
            "anchored to audited revision-v2 zero-shot archive"
        ),
        "formal_order": (
            "native train/val inference -> cache finalization -> "
            "15 output-adapter runs"
        ),
    }
    (out / "experiment_design.json").write_text(
        json.dumps(design, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return design


def launch(command: list[str]) -> None:
    print("\n" + "=" * 108, flush=True)
    print(" ".join(command), flush=True)
    print("=" * 108, flush=True)
    subprocess.run(command, check=True)


def input_ready(root: Path) -> bool:
    path = result_root(root) / "00_native_cache/native_input_inventory.json"
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return (
            data.get("state") == "complete"
            and all(
                split in data.get("splits", {})
                for split in ("train", "val", "test")
            )
        )
    except Exception:
        return False


def prepare_input(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup(root)
    command = [
        sys.executable,
        str(script(
            root,
            "cache/export_cross_backbone_inputs.py",
        )),
        "--root", str(root),
        "--batch_size", "1024",
        "--num_workers", str(min(args.num_workers, 4)),
    ]
    if args.overwrite:
        command.append("--overwrite")
    launch(command)


def raw_dir(root: Path, model: str, split: str, smoke: bool) -> Path:
    prefix = "smoke_raw" if smoke else "raw"
    return result_root(root) / f"00_native_cache/{prefix}/{model}/{split}"


def aligned_root(root: Path, smoke: bool) -> Path:
    prefix = "smoke_aligned" if smoke else "aligned"
    return result_root(root) / f"00_native_cache/{prefix}"


def raw_complete(root: Path, model: str, split: str, smoke: bool) -> bool:
    directory = raw_dir(root, model, split, smoke)
    pred = directory / "pred_mean.npy"
    true = directory / "true.npy"
    if not pred.exists() or not true.exists():
        return False
    try:
        p = np.load(pred, mmap_mode="r")
        y = np.load(true, mmap_mode="r")
        return (
            p.ndim == 2
            and p.shape[1] == 48
            and y.shape == p.shape
        )
    except Exception:
        return False


def run_sundial(
    root: Path,
    split: str,
    out: Path,
    *,
    smoke: bool,
) -> None:
    if raw_complete(root, "sundial", split, smoke):
        print(f"[RAW SKIP] sundial/{split}", flush=True)
        return
    command = [
        sys.executable,
        str(script(
            root, "cache/cache_sundial_native.py"
        )),
        "--root", str(root),
        "--split", split,
        "--pred_len", "48",
        "--batch_size", "2" if smoke else "16",
        "--num_workers", "0" if smoke else "4",
        "--num_samples", "2" if smoke else "20",
        "--input_mode", "phys",
        "--revin",
        "--device", "cpu" if smoke else "cuda",
        "--save_inputs",
        "--overwrite",
        "--out_dir", str(out),
    ]
    if smoke:
        command += ["--max_samples", "8"]
    launch(command)


def run_timemoe(
    root: Path,
    split: str,
    out: Path,
    *,
    smoke: bool,
) -> None:
    if raw_complete(root, "timemoe_200m", split, smoke):
        print(f"[RAW SKIP] timemoe_200m/{split}", flush=True)
        return
    input_cache = (
        result_root(root) / f"00_native_cache/input/{split}"
    )
    command = [
        sys.executable,
        str(root / "scripts/eval/cache_timemoe_forecasts.py"),
        "--input_cache", str(input_cache),
        "--out_dir", str(out),
        "--model_path", str(root / "models_hf_local/timemoe_200m"),
        "--pred_len", "48",
        "--batch_size", "2" if smoke else "256",
        "--device", "cpu" if smoke else "cuda",
        "--input_mode", "phys",
        "--revin",
        "--dtype", "float32" if smoke else "auto",
        "--overwrite",
    ]
    if smoke:
        command += ["--max_samples", "8"]
    launch(command)


def run_sempo(
    root: Path,
    split: str,
    out: Path,
    *,
    smoke: bool,
) -> None:
    if raw_complete(root, "sempo", split, smoke):
        print(f"[RAW SKIP] sempo/{split}", flush=True)
        return
    input_cache = (
        result_root(root) / f"00_native_cache/input/{split}"
    )
    command = [
        sys.executable,
        str(root / "scripts/eval/cache_sempo_forecasts.py"),
        "--input_cache", str(input_cache),
        "--out_dir", str(out),
        "--batch_size", "2" if smoke else "256",
        "--device", "cpu" if smoke else "cuda",
        "--pad_mode", "repeat_context",
        "--overwrite",
    ]
    if smoke:
        command += ["--max_samples", "8"]
    launch(command)


def finalize(root: Path, model: str, mode: str, overwrite: bool) -> None:
    command = [
        sys.executable,
        str(script(
            root,
            "cache/finalize_cross_backbone_cache.py",
        )),
        "--root", str(root),
        "--mode", mode,
        "--model", model,
    ]
    if overwrite:
        command.append("--overwrite")
    launch(command)


def smoke(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup(root)
    if not input_ready(root):
        prepare_input(args)

    for model in MODELS:
        out = raw_dir(root, model, "test", True)
        if model == "sundial":
            run_sundial(root, "test", out, smoke=True)
        elif model == "timemoe_200m":
            run_timemoe(root, "test", out, smoke=True)
        else:
            run_sempo(root, "test", out, smoke=True)
        finalize(root, model, "smoke", args.overwrite)

        adapter_out = (
            result_root(root) / "00_smoke_cpu"
            / f"{model}_seed2026"
        )
        if is_complete(adapter_out, formal=False):
            print(f"[ADAPTER SMOKE SKIP] {model}", flush=True)
            continue
        launch([
            sys.executable,
            str(script(
                root,
                "train/train_cross_backbone_adaptation.py",
            )),
            "--root", str(root),
            "--cache_root", str(aligned_root(root, True)),
            "--model", model,
            "--seed", "2026",
            "--epochs", "1",
            "--patience", "1",
            "--batch_size", "4",
            "--num_workers", "0",
            "--device", "cpu",
            "--cpu_threads", str(args.cpu_threads),
            "--save_arrays", "1",
            "--keep_checkpoint", "0",
            "--log_every", "1",
            "--out_dir", str(adapter_out),
        ])
    print("\nNative CPU smoke complete: 3/3.", flush=True)


def infer(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup(root)
    if not input_ready(root):
        prepare_input(args)

    for model in args.models:
        for split in SPLITS:
            out = raw_dir(root, model, split, False)
            if model == "sundial":
                run_sundial(root, split, out, smoke=False)
            elif model == "timemoe_200m":
                run_timemoe(root, split, out, smoke=False)
            elif model == "sempo":
                run_sempo(root, split, out, smoke=False)
            else:
                raise ValueError(model)
        finalize(root, model, "formal", args.overwrite)


def run_dir(root: Path, model: str, seed: int) -> Path:
    return (
        result_root(root) / "01_output_adapter"
        / model / f"seed{seed}"
    )


def is_complete(directory: Path, formal: bool) -> bool:
    status = directory / "run_status.json"
    if not status.exists():
        return False
    try:
        state = json.loads(status.read_text(encoding="utf-8")).get("state")
    except Exception:
        return False
    if state != "complete":
        return False
    required = ["metrics_overall.csv"]
    if formal:
        required += ["pred.npy", "true.npy", "array_manifest.json"]
    return all((directory / name).exists() for name in required)


def formal(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup(root)
    cache_base = aligned_root(root, False)
    for model in args.models:
        for split in ("train", "val", "test"):
            manifest = cache_base / model / split / "manifest.json"
            if not manifest.exists():
                raise RuntimeError(
                    f"Native aligned cache missing: {manifest}"
                )
        for seed in args.seeds:
            out = run_dir(root, model, seed)
            if is_complete(out, formal=True):
                print(f"[FORMAL SKIP] {model} seed={seed}", flush=True)
                continue
            launch([
                sys.executable,
                str(script(
                    root,
                    "train/train_cross_backbone_adaptation.py",
                )),
                "--root", str(root),
                "--cache_root", str(cache_base),
                "--model", model,
                "--seed", str(seed),
                "--epochs", "20",
                "--patience", "5",
                "--batch_size", "64",
                "--num_workers", str(args.num_workers),
                "--lr", "0.001",
                "--weight_decay", "0.0001",
                "--mse_weight", "0.5",
                "--grad_clip", "1.0",
                "--hidden", "128",
                "--emb_dim", "16",
                "--city_emb_dim", "16",
                "--dropout", "0.05",
                "--device", "cuda",
                "--require_cuda",
                "--save_arrays", "1",
                "--keep_checkpoint", "1",
                "--log_every", "1000",
                "--out_dir", str(out),
            ])


def read_one(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if len(df) != 1:
        raise RuntimeError(f"Expected one row: {path}")
    return df.iloc[0].to_dict()


def summarize(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    out = result_root(root)
    reference_path = (
        root / "experiments/results/"
        "03_zeroshot_baselines_v2_archive/Timer/true.npy"
    )
    if not reference_path.is_file():
        reference_path = root / "experiments/data_v2/indices/test_targets.npy"
    reference = np.load(reference_path, mmap_mode="r")
    rows = []
    issues = []
    for model in MODELS:
        for seed in SEEDS:
            directory = run_dir(root, model, seed)
            row: dict[str, Any] = {
                "model": model,
                "backbone": DISPLAY_NAMES[model],
                "seed": seed,
                "run_dir": str(directory),
                "complete": is_complete(directory, True),
            }
            if row["complete"]:
                row.update(read_one(directory / "metrics_overall.csv"))
                pred = np.load(directory / "pred.npy", mmap_mode="r")
                true = np.load(directory / "true.npy", mmap_mode="r")
                row["shape_ok"] = (
                    tuple(pred.shape) == EXPECTED_SHAPE
                    and tuple(true.shape) == EXPECTED_SHAPE
                )
                row["true_canonical_equal"] = np.array_equal(
                    np.asarray(true), np.asarray(reference), equal_nan=True
                )
                row["nonfinite_pred"] = int((~np.isfinite(pred)).sum())
                row["nonfinite_true"] = int((~np.isfinite(true)).sum())
                diff = (
                    np.asarray(pred, dtype=np.float64)
                    - np.asarray(true, dtype=np.float64)
                )
                array_mae = float(np.mean(np.abs(diff)))
                array_rmse = float(np.sqrt(np.mean(diff ** 2)))
                row["array_MAE"] = array_mae
                row["array_RMSE"] = array_rmse
                row["metric_match"] = (
                    abs(array_mae - float(row["MAE"])) <= 5e-5
                    and abs(array_rmse - float(row["RMSE"])) <= 5e-5
                )
            else:
                row.update({
                    "shape_ok": False,
                    "true_canonical_equal": False,
                    "metric_match": False,
                    "nonfinite_pred": -1,
                    "nonfinite_true": -1,
                })
            row["audit_ok"] = all([
                bool(row["complete"]),
                bool(row["shape_ok"]),
                bool(row["true_canonical_equal"]),
                bool(row["metric_match"]),
                int(row["nonfinite_pred"]) == 0,
                int(row["nonfinite_true"]) == 0,
            ])
            if not row["audit_ok"]:
                issues.append({
                    "model": model,
                    "seed": seed,
                    "detail": json.dumps(
                        row, ensure_ascii=False, default=str
                    ),
                })
            rows.append(row)

    seed_df = pd.DataFrame(rows)
    seed_df.to_csv(
        out / "03_summary/new_backbone_seed_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(issues).to_csv(
        out / "03_summary/audit_issues.csv",
        index=False,
        encoding="utf-8-sig",
    )

    adapted_rows = []
    for model in MODELS:
        part = seed_df[seed_df["model"] == model]
        mae = pd.to_numeric(part["MAE"], errors="coerce").dropna()
        rmse = pd.to_numeric(part["RMSE"], errors="coerce").dropna()
        params = pd.to_numeric(
            part["trainable_params"], errors="coerce"
        ).dropna()
        adapted_rows.append({
            "backbone": DISPLAY_NAMES[model],
            "model_key": model,
            "method": "Output Adapter",
            "n_seeds": len(mae),
            "MAE_mean": float(mae.mean()),
            "MAE_std": float(mae.std(ddof=1)),
            "RMSE_mean": float(rmse.mean()),
            "RMSE_std": float(rmse.std(ddof=1)),
            "trainable_params": float(params.mean()),
            "audit_ok_runs": int(part["audit_ok"].sum()),
        })

    timer_metrics = []
    for seed in SEEDS:
        timer_metrics.append(read_one(
            root / "experiments/results/"
            "10_adapter_time_ablation_v2/"
            f"01_progressive_adapter/p1_o3_output/seed{seed}/"
            "metrics_overall.csv"
        ))
    timer_df = pd.DataFrame(timer_metrics)
    adapted_rows.insert(0, {
        "backbone": "Timer",
        "model_key": "timer",
        "method": "Output Adapter",
        "n_seeds": 5,
        "MAE_mean": float(timer_df["MAE"].mean()),
        "MAE_std": float(timer_df["MAE"].std(ddof=1)),
        "RMSE_mean": float(timer_df["RMSE"].mean()),
        "RMSE_std": float(timer_df["RMSE"].std(ddof=1)),
        "trainable_params": float(
            timer_df["trainable_params"].mean()
        ),
        "audit_ok_runs": 5,
    })
    adapted = pd.DataFrame(adapted_rows)

    zero_rows = []
    for display in ["Timer", "Sundial", "TimeMoE-200M", "SEMPO"]:
        metric = read_one(
            root / "experiments/results/"
            f"03_zeroshot_baselines_v2_archive/{display}/"
            "metrics_recomputed_from_arrays.csv"
        )
        zero_rows.append({
            "backbone": display,
            "zero_shot_MAE": float(metric["MAE"]),
            "zero_shot_RMSE": float(metric["RMSE"]),
        })
    comparison = adapted.merge(
        pd.DataFrame(zero_rows), on="backbone", how="left"
    )
    comparison["delta_MAE"] = (
        comparison["MAE_mean"] - comparison["zero_shot_MAE"]
    )
    comparison["delta_RMSE"] = (
        comparison["RMSE_mean"] - comparison["zero_shot_RMSE"]
    )
    comparison["relative_MAE_change_pct"] = (
        comparison["delta_MAE"]
        / comparison["zero_shot_MAE"] * 100.0
    )
    comparison["relative_RMSE_change_pct"] = (
        comparison["delta_RMSE"]
        / comparison["zero_shot_RMSE"] * 100.0
    )

    adapted.to_csv(
        out / "03_summary/output_adapter_summary_5seeds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison.to_csv(
        out / "03_summary/cross_backbone_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "protocol": "revision-v2",
        "native_cache": True,
        "test_forecasts_anchored_to_zero_shot_archive": True,
        "expected_new_trainings": 15,
        "completed_new_runs": int(seed_df["complete"].sum()),
        "audit_ok_new_runs": int(seed_df["audit_ok"].sum()),
        "issues": len(issues),
        "timer_p1_reused": True,
    }
    (out / "03_summary/archive_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(adapted.to_string(index=False))
    print(comparison.to_string(index=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if (
        manifest["completed_new_runs"] != 15
        or manifest["audit_ok_new_runs"] != 15
        or manifest["issues"] != 0
    ):
        raise SystemExit("ERROR: final audit incomplete")


def status(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup(root)
    print("=" * 96)
    print(
        "NATIVE INPUT:",
        "COMPLETE" if input_ready(root) else "pending",
    )
    print("\nCPU SMOKE")
    for model in MODELS:
        smoke_dir = (
            result_root(root) / "00_smoke_cpu"
            / f"{model}_seed2026"
        )
        print(
            f"{DISPLAY_NAMES[model]:<16} "
            f"{'COMPLETE' if is_complete(smoke_dir, False) else 'pending'}"
        )

    print("\nNATIVE FORMAL CACHE")
    cache_base = aligned_root(root, False)
    for model in MODELS:
        marks = []
        for split in ("train", "val", "test"):
            path = cache_base / model / split / "manifest.json"
            marks.append(f"{split}:{'OK' if path.exists() else '--'}")
        print(f"{DISPLAY_NAMES[model]:<16} {' '.join(marks)}")

    print("\nOUTPUT ADAPTER FORMAL")
    complete = 0
    for model in MODELS:
        marks = []
        for seed in SEEDS:
            ok = is_complete(run_dir(root, model, seed), True)
            complete += int(ok)
            marks.append(f"{seed}:{'OK' if ok else '--'}")
        print(f"{DISPLAY_NAMES[model]:<16} {' '.join(marks)}")
    print(f"\nNew formal complete: {complete}/15")
    print("Timer Output Adapter is reused from Group 1 P1.")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if args.command == "setup":
        print(json.dumps(setup(root), indent=2, ensure_ascii=False))
    elif args.command == "prepare-input":
        prepare_input(args)
    elif args.command == "smoke":
        smoke(args)
    elif args.command == "infer":
        infer(args)
    elif args.command == "formal":
        formal(args)
    elif args.command == "status":
        status(args)
    elif args.command == "summarize":
        summarize(args)


if __name__ == "__main__":
    main()
