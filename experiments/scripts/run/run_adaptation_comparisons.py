#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Setup, smoke, adaptive GPU probe, formal queue, status and audit."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEEDS = [2026, 2027, 2028, 2029, 2030]
CPU_SAFE = [
    "q_quantile_full",
    "n2_climatology",
    "n3_linear_ar",
    "a0_affine_only",
    "r1_lagged_corr",
    "r2_mutual_information",
    "r3_learned_attention",
]
GPU_PROBE = ["a1_lora_r8", "a2_full_finetune"]
VARIANTS = CPU_SAFE + GPU_PROBE
RELATIVE_DIRS = {
    "q_quantile_full": "01_quantile/q_quantile_full",
    "n2_climatology": "02_necessity/n2_climatology",
    "n3_linear_ar": "02_necessity/n3_linear_ar",
    "a0_affine_only": "03_adaptation/a0_affine_only",
    "a1_lora_r8": "03_adaptation/a1_lora_r8",
    "a2_full_finetune": "03_adaptation/a2_full_finetune",
    "r1_lagged_corr": "04_relation/r1_lagged_corr",
    "r2_mutual_information": "04_relation/r2_mutual_information",
    "r3_learned_attention": "04_relation/r3_learned_attention",
}
PAPER_NAMES = {
    "q_quantile_full": "Quantile-DualContext",
    "n2_climatology": "Climatology Prior",
    "n3_linear_ar": "Linear AR Prior",
    "a0_affine_only": "Horizon-wise Affine",
    "a1_lora_r8": "LoRA-r8 Full Adapter",
    "a2_full_finetune": "Full Fine-tuning",
    "r1_lagged_corr": "Lagged-correlation DualContext",
    "r2_mutual_information": "Mutual-information DualContext",
    "r3_learned_attention": "Learned-attention DualContext",
}
GROUPS = {
    "q_quantile_full": "quantile",
    "n2_climatology": "necessity",
    "n3_linear_ar": "necessity",
    "a0_affine_only": "adaptation",
    "a1_lora_r8": "adaptation",
    "a2_full_finetune": "adaptation",
    "r1_lagged_corr": "relation",
    "r2_mutual_information": "relation",
    "r3_learned_attention": "relation",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "command",
        choices=[
            "setup", "prepare", "cpu_smoke", "gpu_smoke",
            "formal", "status", "summarize",
        ],
    )
    p.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--cpu_threads", type=int, default=4)
    return p.parse_args()


def result_root(root: Path) -> Path:
    return root / "experiments/results/15_reviewer45_v2"


def trainer(root: Path) -> Path:
    return (
        root / "experiments/scripts/train/"
        "train_adaptation_comparisons.py"
    )


def preparer(root: Path) -> Path:
    return (
        root / "experiments/scripts/prepare/"
        "prepare_adaptation_assets.py"
    )


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
    directories = [
        "00_assets/smoke",
        "00_assets/formal",
        "00_smoke_cpu",
        "00_smoke_gpu",
        *RELATIVE_DIRS.values(),
        "05_reference",
        "06_summary",
        "logs",
    ]
    for relative in directories:
        (out / relative).mkdir(parents=True, exist_ok=True)

    references = {
        "C4_Full_DualContext": (
            root / "experiments/results/"
            "11_dualcontext_ablation_v2/01_overall/"
            "c4_full_dualcontext"
        ),
        "P4_Frozen_Full_Adapter": (
            root / "experiments/results/"
            "10_adapter_time_ablation_v2/01_progressive_adapter/"
            "p4_io_time_context"
        ),
        "Timer_ZeroShot": (
            root / "experiments/results/"
            "03_zeroshot_baselines_v2_archive/Timer"
        ),
        "N0_No_Backbone": (
            root / "experiments/results/"
            "13_tsfm_necessity_v2/01_training_replacement/"
            "n0_no_backbone"
        ),
        "N1_Persistence": (
            root / "experiments/results/"
            "13_tsfm_necessity_v2/01_training_replacement/"
            "n1_persistence"
        ),
        "N3_Random_Timer": (
            root / "experiments/results/"
            "13_tsfm_necessity_v2/01_training_replacement/"
            "n3_random_timer"
        ),
    }
    link_states = {}
    for name, target in references.items():
        link_states[name] = safe_symlink(
            target, out / "05_reference" / name
        )

    design = {
        "protocol": "revision-v2",
        "seeds": SEEDS,
        "variants": VARIANTS,
        "cpu_safe_variants": CPU_SAFE,
        "deferred_gpu_probe_variants": GPU_PROBE,
        "maximum_formal_runs": 45,
        "quantile_outputs": [0.05, 0.10, 0.50, 0.90, 0.95],
        "quantile_point_semantics": "raw learned q0.50",
        "quantile_interval_semantics": (
            "raw outputs retained for crossing diagnostics; "
            "rearranged outputs retained for monotone intervals"
        ),
        "relation_semantics": (
            "target O3 excluded from covariate relation cues and softmax; "
            "O3 enters through dedicated o3_ctx and raw-history paths"
        ),
        "fixed_task": {
            "input_length": 96,
            "prediction_length": 48,
            "variables": 8,
            "test_samples": 106200,
        },
        "formal_hyperparameters": {
            "epochs": 20,
            "patience": 5,
            "effective_batch_size": 64,
            "adapter_lr": 0.001,
            "lora_lr": 0.0001,
            "backbone_lr": 0.00001,
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
        },
        "gpu_policy": (
            "LoRA and full fine-tuning are adaptively smoke-tested only "
            "after the existing LOCO queue releases the GPU. Unsupported "
            "variants are recorded and skipped without blocking the other "
            "formal experiments."
        ),
        "reference_link_states": link_states,
    }
    (out / "experiment_design.json").write_text(
        json.dumps(design, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return design


def run_dir(root: Path, variant: str, seed: int,
            smoke_kind: str | None = None,
            attempt: int | None = None) -> Path:
    out = result_root(root)
    if smoke_kind == "cpu":
        return out / "00_smoke_cpu" / f"{variant}_seed{seed}"
    if smoke_kind == "gpu":
        base = out / "00_smoke_gpu" / variant
        return base / f"attempt{attempt}" if attempt else base
    return out / RELATIVE_DIRS[variant] / f"seed{seed}"


def read_status(path: Path) -> dict[str, Any]:
    status = path / "run_status.json"
    if not status.exists():
        return {}
    try:
        return json.loads(status.read_text(encoding="utf-8"))
    except Exception:
        return {}


def is_complete(path: Path, formal: bool,
                quantile: bool = False) -> bool:
    data = read_status(path)
    if data.get("state") != "complete":
        return False
    if not (path / "metrics_overall.csv").exists():
        return False
    if formal:
        required = [
            "pred.npy", "base_pred.npy", "true.npy",
            "city_id.npy", "array_manifest.json",
            "trainable_parameter_manifest.json",
        ]
        if quantile:
            required += [
                "quantile_pred_raw.npy",
                "quantile_pred_rearranged.npy",
                "raw_pred_q05.npy", "raw_pred_q10.npy",
                "raw_pred_q50.npy", "raw_pred_q90.npy",
                "raw_pred_q95.npy",
                "rearranged_pred_q05.npy",
                "rearranged_pred_q10.npy",
                "rearranged_pred_q50.npy",
                "rearranged_pred_q90.npy",
                "rearranged_pred_q95.npy",
            ]
        return all((path / name).exists() for name in required)
    return True


def launch(command: list[str], *, check: bool = True,
           stdout=None, stderr=None) -> subprocess.CompletedProcess:
    print("\n" + "=" * 116, flush=True)
    print(" ".join(command), flush=True)
    print("=" * 116, flush=True)
    return subprocess.run(
        command, check=check, stdout=stdout, stderr=stderr
    )


def prepare_assets(root: Path, mode: str) -> None:
    launch([
        sys.executable,
        str(preparer(root)),
        "--root", str(root),
        "--mode", mode,
    ])


def common_train_command(
    root: Path,
    variant: str,
    seed: int,
    out: Path,
    *,
    epochs: int,
    patience: int,
    batch: int,
    accum: int,
    workers: int,
    max_train: int,
    max_val: int,
    max_test: int,
    device: str,
    require_cuda: bool,
    amp_bf16: int,
    checkpointing: int,
    asset_mode: str,
    keep_checkpoint: int,
    log_every: int,
) -> list[str]:
    command = [
        sys.executable,
        str(trainer(root)),
        "--root", str(root),
        "--variant", variant,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--batch_size", str(batch),
        "--grad_accum", str(accum),
        "--num_workers", str(workers),
        "--adapter_lr", "0.001",
        "--lora_lr", "0.0001",
        "--backbone_lr", "0.00001",
        "--weight_decay", "0.0001",
        "--mse_weight", "0.5",
        "--grad_clip", "1.0",
        "--input_hidden", "64",
        "--output_hidden", "128",
        "--context_hidden", "128",
        "--emb_dim", "16",
        "--city_emb_dim", "16",
        "--dropout", "0.05",
        "--corr_temperature", "2.0",
        "--lag_max", "24",
        "--lag_step", "3",
        "--lora_rank", "8",
        "--lora_alpha", "16",
        "--lora_dropout", "0.05",
        "--lambda_mid_mae", "1.0",
        "--lambda_mid_mse", "0.5",
        "--lambda_cross", "0.05",
        "--asset_mode", asset_mode,
        "--max_train_samples", str(max_train),
        "--max_val_samples", str(max_val),
        "--max_test_samples", str(max_test),
        "--device", device,
        "--amp_bf16", str(amp_bf16),
        "--gradient_checkpointing", str(checkpointing),
        "--save_arrays", "1",
        "--keep_checkpoint", str(keep_checkpoint),
        "--log_every", str(log_every),
        "--out_dir", str(out),
    ]
    if require_cuda:
        command.append("--require_cuda")
    return command


def cpu_smoke(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup(root)
    prepare_assets(root, "smoke")
    for variant in CPU_SAFE:
        out = run_dir(root, variant, 2026, "cpu")
        if is_complete(out, False):
            print(f"[CPU SMOKE SKIP] {variant}", flush=True)
            continue
        command = common_train_command(
            root, variant, 2026, out,
            epochs=1, patience=1,
            batch=16, accum=1, workers=0,
            max_train=128, max_val=64, max_test=64,
            device="cpu", require_cuda=False,
            amp_bf16=0, checkpointing=0,
            asset_mode="smoke", keep_checkpoint=0,
            log_every=1,
        )
        command += ["--cpu_threads", str(args.cpu_threads)]
        launch(command)
    print("\nCPU-safe smoke complete: 7/7 variants.", flush=True)


def gpu_profiles(variant: str) -> list[dict[str, int]]:
    if variant == "a1_lora_r8":
        return [
            {"batch": 32, "accum": 2, "checkpointing": 0},
            {"batch": 16, "accum": 4, "checkpointing": 0},
            {"batch": 8, "accum": 8, "checkpointing": 0},
        ]
    return [
        {"batch": 8, "accum": 8, "checkpointing": 0},
        {"batch": 4, "accum": 16, "checkpointing": 1},
        {"batch": 2, "accum": 32, "checkpointing": 1},
        {"batch": 1, "accum": 64, "checkpointing": 1},
    ]


def gpu_smoke(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    out_root = result_root(root) / "00_smoke_gpu"
    out_root.mkdir(parents=True, exist_ok=True)
    support: dict[str, Any] = {
        "state": "running",
        "tested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_available": bool(
            subprocess.run(
                [sys.executable, "-c",
                 "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"],
                check=False,
            ).returncode == 0
        ),
        "variants": {},
    }
    if not support["gpu_available"]:
        raise RuntimeError("GPU smoke requested but CUDA is unavailable")

    for variant in GPU_PROBE:
        variant_report = {
            "supported": False,
            "attempts": [],
            "selected_profile": None,
        }
        for attempt, profile in enumerate(gpu_profiles(variant), 1):
            out = run_dir(
                root, variant, 2026, "gpu", attempt=attempt
            )
            out.mkdir(parents=True, exist_ok=True)
            log_path = out / "attempt.log"
            command = common_train_command(
                root, variant, 2026, out,
                epochs=1, patience=1,
                batch=profile["batch"],
                accum=profile["accum"],
                workers=0,
                max_train=128, max_val=64, max_test=64,
                device="cuda", require_cuda=True,
                amp_bf16=1,
                checkpointing=profile["checkpointing"],
                asset_mode="smoke",
                keep_checkpoint=0,
                log_every=1,
            )
            print(
                f"\n[GPU PROBE] {variant} attempt={attempt} "
                f"profile={profile}",
                flush=True,
            )
            with log_path.open("w", encoding="utf-8") as handle:
                result = launch(
                    command,
                    check=False,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            attempt_report = {
                **profile,
                "attempt": attempt,
                "returncode": result.returncode,
                "log": str(log_path),
                "complete": is_complete(out, False),
            }
            variant_report["attempts"].append(attempt_report)
            if result.returncode == 0 and attempt_report["complete"]:
                variant_report["supported"] = True
                variant_report["selected_profile"] = profile
                break
        if not variant_report["supported"]:
            variant_report["reason"] = (
                "All adaptive GPU smoke profiles failed. "
                "See per-attempt logs."
            )
        support["variants"][variant] = variant_report

    support["state"] = "complete"
    support["supported_variants"] = [
        variant for variant in GPU_PROBE
        if support["variants"][variant]["supported"]
    ]
    support["unsupported_variants"] = [
        variant for variant in GPU_PROBE
        if not support["variants"][variant]["supported"]
    ]
    support["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_root / "gpu_support_manifest.json").write_text(
        json.dumps(support, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(support, indent=2, ensure_ascii=False))


def load_gpu_support(root: Path) -> dict[str, Any]:
    path = (
        result_root(root) / "00_smoke_gpu/"
        "gpu_support_manifest.json"
    )
    if not path.exists():
        raise FileNotFoundError(
            "GPU support manifest is missing. Run gpu_smoke first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("state") != "complete":
        raise RuntimeError("GPU support manifest is incomplete")
    return data


def supported_variants(root: Path) -> list[str]:
    support = load_gpu_support(root)
    return CPU_SAFE + list(support.get("supported_variants", []))


def formal(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup(root)
    prepare_assets(root, "formal")
    support = load_gpu_support(root)
    selected = args.variants or (
        CPU_SAFE + list(support.get("supported_variants", []))
    )
    unknown = sorted(set(selected) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    for variant in selected:
        if variant in GPU_PROBE:
            report = support["variants"][variant]
            if not report.get("supported"):
                print(
                    f"[FORMAL SKIP UNSUPPORTED] {variant}",
                    flush=True,
                )
                continue
            profile = report["selected_profile"]
            batch = int(profile["batch"])
            accum = int(profile["accum"])
            checkpointing = int(profile["checkpointing"])
            amp = 1
        else:
            batch, accum, checkpointing, amp = 64, 1, 0, 0

        print(
            f"\n######## FORMAL {variant}: "
            f"{PAPER_NAMES[variant]} ########",
            flush=True,
        )
        for seed in args.seeds:
            out = run_dir(root, variant, seed)
            if is_complete(
                out, True, quantile=(variant == "q_quantile_full")
            ):
                print(
                    f"[FORMAL SKIP] {variant} seed={seed}",
                    flush=True,
                )
                continue
            launch(common_train_command(
                root, variant, seed, out,
                epochs=20, patience=5,
                batch=batch, accum=accum,
                workers=args.num_workers,
                max_train=0, max_val=0, max_test=0,
                device="cuda", require_cuda=True,
                amp_bf16=amp,
                checkpointing=checkpointing,
                asset_mode="formal",
                keep_checkpoint=1,
                log_every=1000,
            ))


def read_one_csv(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if len(frame) != 1:
        raise RuntimeError(f"Expected one row: {path}")
    return frame.iloc[0].to_dict()


def canonical_true_path(root: Path) -> Path:
    candidates = [
        root / "experiments/results/02_global_baselines/"
        "00_global11_v2_array_archive/_canonical/true.npy",
        root / "experiments/results/02_global_baselines/"
        "00_global11_v2_array_archive/TSMixer/seed2026/true.npy",
        root / "experiments/results/"
        "03_zeroshot_baselines_v2_archive/Timer/true.npy",
        root / "experiments/data_v2/indices/test_targets.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No canonical revision-v2 true.npy found; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def audit_run(root: Path, variant: str, seed: int) -> dict[str, Any]:
    directory = run_dir(root, variant, seed)
    pred = np.load(directory / "pred.npy", mmap_mode="r")
    base = np.load(directory / "base_pred.npy", mmap_mode="r")
    true = np.load(directory / "true.npy", mmap_mode="r")
    metric = read_one_csv(directory / "metrics_overall.csv")
    diff = (
        np.asarray(pred, dtype=np.float64)
        - np.asarray(true, dtype=np.float64)
    )
    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt(np.square(diff).mean()))
    row = {
        **metric,
        "run_dir": str(directory),
        "complete": True,
        "shape_ok": (
            pred.shape == base.shape == true.shape == (106_200, 48)
        ),
        "nonfinite_pred": int((~np.isfinite(pred)).sum()),
        "nonfinite_base": int((~np.isfinite(base)).sum()),
        "nonfinite_true": int((~np.isfinite(true)).sum()),
        "array_MAE": mae,
        "array_RMSE": rmse,
        "metric_match": (
            abs(mae - float(metric["MAE"])) <= 5e-5
            and abs(rmse - float(metric["RMSE"])) <= 5e-5
        ),
    }
    canonical = np.load(
        canonical_true_path(root),
        mmap_mode="r",
    )
    row["true_canonical_equal"] = np.array_equal(
        np.asarray(true), np.asarray(canonical), equal_nan=True
    )
    if variant == "q_quantile_full":
        raw_quantiles = np.load(
            directory / "quantile_pred_raw.npy", mmap_mode="r"
        )
        rearranged_quantiles = np.load(
            directory / "quantile_pred_rearranged.npy", mmap_mode="r"
        )
        raw_q50 = np.load(
            directory / "raw_pred_q50.npy", mmap_mode="r"
        )
        raw_crossings = int(
            (
                np.asarray(raw_quantiles[..., :-1])
                > np.asarray(raw_quantiles[..., 1:])
            ).sum()
        )
        rearranged_crossings = int(
            (
                np.asarray(rearranged_quantiles[..., :-1])
                > np.asarray(rearranged_quantiles[..., 1:])
            ).sum()
        )
        row.update({
            "quantile_shape_ok": (
                raw_quantiles.shape == (106_200, 48, 5)
                and rearranged_quantiles.shape == (106_200, 48, 5)
            ),
            "nonfinite_quantiles": int(
                (~np.isfinite(raw_quantiles)).sum()
                + (~np.isfinite(rearranged_quantiles)).sum()
            ),
            "q50_equals_pred": np.array_equal(
                np.asarray(raw_q50), np.asarray(pred), equal_nan=True
            ),
            "raw_quantile_crossings": raw_crossings,
            "rearranged_quantile_crossings": rearranged_crossings,
        })
    else:
        row.update({
            "quantile_shape_ok": True,
            "nonfinite_quantiles": 0,
            "q50_equals_pred": True,
            "raw_quantile_crossings": 0,
            "rearranged_quantile_crossings": 0,
        })
    row["audit_ok"] = all([
        row["shape_ok"],
        row["true_canonical_equal"],
        row["metric_match"],
        row["nonfinite_pred"] == 0,
        row["nonfinite_base"] == 0,
        row["nonfinite_true"] == 0,
        row["quantile_shape_ok"],
        row["nonfinite_quantiles"] == 0,
        row["q50_equals_pred"],
        # Raw crossings are a measured model property and are not forced to
        # zero. Only the explicit monotone rearrangement must be crossing-free.
        row["rearranged_quantile_crossings"] == 0,
    ])
    return row


def reference_seed_metrics(root: Path, reference: str) -> pd.DataFrame:
    reference_root = result_root(root) / "05_reference" / reference
    rows = []
    for seed in SEEDS:
        path = reference_root / f"seed{seed}" / "metrics_overall.csv"
        if not path.exists():
            continue
        row = read_one_csv(path)
        rows.append({
            "seed": seed,
            "reference": reference,
            "reference_MAE": float(row["MAE"]),
            "reference_RMSE": float(row["RMSE"]),
            "reference_trainable_params":
                row.get("trainable_params", np.nan),
            "reference_train_seconds":
                row.get("train_seconds", np.nan),
            "reference_peak_gpu_memory_mb":
                row.get(
                    "peak_gpu_memory_mb",
                    row.get("peak_gpu_memory_allocated_mb", np.nan),
                ),
        })
    return pd.DataFrame(rows)


def paired_table(seed_df: pd.DataFrame, variants: list[str],
                 reference_df: pd.DataFrame,
                 output: Path) -> pd.DataFrame:
    part = seed_df[seed_df["variant"].isin(variants)].copy()
    merged = part.merge(reference_df, on="seed", how="left")
    merged["delta_MAE"] = (
        merged["MAE"] - merged["reference_MAE"]
    )
    merged["delta_RMSE"] = (
        merged["RMSE"] - merged["reference_RMSE"]
    )
    merged["win_MAE"] = merged["delta_MAE"] < 0
    merged.to_csv(output, index=False, encoding="utf-8-sig")
    return merged


def summarize(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    out = result_root(root)
    support = load_gpu_support(root)
    supported = CPU_SAFE + list(support.get("supported_variants", []))
    unsupported = list(support.get("unsupported_variants", []))
    issues = []
    rows = []

    for variant in supported:
        for seed in SEEDS:
            directory = run_dir(root, variant, seed)
            if not is_complete(
                directory, True,
                quantile=(variant == "q_quantile_full"),
            ):
                issues.append({
                    "variant": variant,
                    "seed": seed,
                    "kind": "missing_or_incomplete",
                    "path": str(directory),
                })
                rows.append({
                    "variant": variant,
                    "seed": seed,
                    "complete": False,
                    "audit_ok": False,
                })
                continue
            try:
                row = audit_run(root, variant, seed)
            except Exception as exc:
                row = {
                    "variant": variant,
                    "seed": seed,
                    "complete": True,
                    "audit_ok": False,
                    "audit_error": repr(exc),
                }
            if not row.get("audit_ok", False):
                issues.append({
                    "variant": variant,
                    "seed": seed,
                    "kind": "audit_failure",
                    "detail": json.dumps(
                        row, ensure_ascii=False, default=str
                    ),
                })
            rows.append(row)

    seed_df = pd.DataFrame(rows)
    seed_df.to_csv(
        out / "06_summary/reviewer45_seed_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = []
    for variant in supported:
        part = seed_df[
            (seed_df["variant"] == variant)
            & (seed_df["complete"] == True)
        ]
        summary_rows.append({
            "variant": variant,
            "paper_name": PAPER_NAMES[variant],
            "group": GROUPS[variant],
            "n": int(len(part)),
            "MAE_mean": float(part["MAE"].mean()),
            "MAE_std": float(part["MAE"].std(ddof=1)),
            "RMSE_mean": float(part["RMSE"].mean()),
            "RMSE_std": float(part["RMSE"].std(ddof=1)),
            "base_MAE_mean": float(part["base_MAE"].mean()),
            "base_RMSE_mean": float(part["base_RMSE"].mean()),
            "trainable_params":
                float(part["trainable_params"].mean()),
            "trainable_ratio":
                float(part["trainable_ratio"].mean()),
            "train_seconds_mean":
                float(part["train_seconds"].mean()),
            "test_seconds_mean":
                float(part["test_seconds"].mean()),
            "peak_gpu_memory_allocated_mb_mean":
                float(part["peak_gpu_memory_allocated_mb"].mean()),
            "peak_gpu_memory_reserved_mb_mean":
                float(part["peak_gpu_memory_reserved_mb"].mean()),
            "audit_ok_runs": int(part["audit_ok"].sum()),
        })
    variant_df = pd.DataFrame(summary_rows)
    variant_df.to_csv(
        out / "06_summary/reviewer45_variant_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    c4 = reference_seed_metrics(root, "C4_Full_DualContext")
    p4 = reference_seed_metrics(root, "P4_Frozen_Full_Adapter")
    paired_table(
        seed_df,
        ["q_quantile_full"],
        c4,
        out / "06_summary/paired_quantile_q50_vs_c4.csv",
    )
    paired_table(
        seed_df,
        ["r1_lagged_corr", "r2_mutual_information",
         "r3_learned_attention"],
        c4,
        out / "06_summary/paired_relation_vs_pearson_c4.csv",
    )
    paired_table(
        seed_df,
        ["n2_climatology", "n3_linear_ar"],
        p4,
        out / "06_summary/paired_priors_vs_pretrained_p4.csv",
    )
    adaptation_variants = [
        variant for variant in
        ["a0_affine_only", "a1_lora_r8", "a2_full_finetune"]
        if variant in supported
    ]
    paired_adaptation = paired_table(
        seed_df,
        adaptation_variants,
        p4,
        out / "06_summary/paired_adaptation_vs_frozen_p4.csv",
    )

    efficiency = variant_df[[
        "variant", "paper_name", "MAE_mean", "MAE_std",
        "RMSE_mean", "RMSE_std", "trainable_params",
        "trainable_ratio", "train_seconds_mean",
        "test_seconds_mean",
        "peak_gpu_memory_allocated_mb_mean",
        "peak_gpu_memory_reserved_mb_mean",
    ]].copy()
    if len(p4):
        efficiency = pd.concat([
            pd.DataFrame([{
                "variant": "reference_p4",
                "paper_name": "Frozen Full Adapter (P4)",
                "MAE_mean": p4["reference_MAE"].mean(),
                "MAE_std": p4["reference_MAE"].std(ddof=1),
                "RMSE_mean": p4["reference_RMSE"].mean(),
                "RMSE_std": p4["reference_RMSE"].std(ddof=1),
                "trainable_params":
                    p4["reference_trainable_params"].mean(),
                "trainable_ratio": np.nan,
                "train_seconds_mean":
                    p4["reference_train_seconds"].mean(),
                "test_seconds_mean": np.nan,
                "peak_gpu_memory_allocated_mb_mean":
                    p4["reference_peak_gpu_memory_mb"].mean(),
                "peak_gpu_memory_reserved_mb_mean": np.nan,
            }]),
            efficiency,
        ], ignore_index=True)
    efficiency.to_csv(
        out / "06_summary/adaptation_efficiency_table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(issues).to_csv(
        out / "06_summary/audit_issues.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "state": (
            "complete" if not issues and not unsupported
            else "complete_with_unsupported"
            if not issues else "incomplete"
        ),
        "protocol": "revision-v2",
        "maximum_variants": 9,
        "supported_variants": supported,
        "unsupported_variants": unsupported,
        "expected_runs_for_supported_variants":
            len(supported) * len(SEEDS),
        "maximum_possible_runs": 45,
        "completed_runs": int(seed_df["complete"].sum()),
        "audit_ok_runs": int(seed_df["audit_ok"].sum()),
        "issues": len(issues),
        "gpu_support_manifest": str(
            out / "00_smoke_gpu/gpu_support_manifest.json"
        ),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "06_summary/archive_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nReviewer45 variant summary")
    print(variant_df.to_string(index=False))
    print("\n" + json.dumps(
        manifest, indent=2, ensure_ascii=False
    ))
    if issues:
        raise SystemExit("ERROR: reviewer45 archive failed audit")


def status(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    setup(root)
    out = result_root(root)
    print("=" * 108)
    for mode in ("smoke", "formal"):
        inventory = out / f"00_assets/{mode}/asset_inventory.json"
        state = "--"
        if inventory.exists():
            try:
                state = json.loads(
                    inventory.read_text(encoding="utf-8")
                ).get("state", "--")
            except Exception:
                state = "BROKEN"
        print(f"ASSETS {mode:<6}: {state}")

    print("\nCPU-SAFE SMOKE")
    cpu_count = 0
    for variant in CPU_SAFE:
        ok = is_complete(run_dir(root, variant, 2026, "cpu"), False)
        cpu_count += int(ok)
        print(f"{variant:<26} {'COMPLETE' if ok else '--'}")
    print(f"CPU-safe smoke: {cpu_count}/7")

    print("\nDEFERRED GPU SMOKE")
    support_path = out / "00_smoke_gpu/gpu_support_manifest.json"
    if support_path.exists():
        support = json.loads(support_path.read_text(encoding="utf-8"))
        for variant in GPU_PROBE:
            report = support.get("variants", {}).get(variant, {})
            print(
                f"{variant:<26} "
                f"{'SUPPORTED' if report.get('supported') else 'UNSUPPORTED'} "
                f"{report.get('selected_profile')}"
            )
    else:
        print("Not run yet; it will run automatically after LOCO.")

    print("\nFORMAL")
    formal_count = 0
    for variant in VARIANTS:
        marks = []
        for seed in SEEDS:
            ok = is_complete(
                run_dir(root, variant, seed),
                True,
                quantile=(variant == "q_quantile_full"),
            )
            formal_count += int(ok)
            marks.append(f"{seed}:{'OK' if ok else '--'}")
        print(f"{variant:<26} {' '.join(marks)}")
    print(f"\nFormal complete on disk: {formal_count}/45 maximum")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if args.command == "setup":
        print(json.dumps(
            setup(root), indent=2, ensure_ascii=False
        ))
    elif args.command == "prepare":
        setup(root)
        prepare_assets(root, "formal")
    elif args.command == "cpu_smoke":
        cpu_smoke(args)
    elif args.command == "gpu_smoke":
        gpu_smoke(args)
    elif args.command == "formal":
        formal(args)
    elif args.command == "status":
        status(args)
    elif args.command == "summarize":
        summarize(args)


if __name__ == "__main__":
    main()
