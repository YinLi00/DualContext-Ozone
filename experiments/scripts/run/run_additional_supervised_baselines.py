#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controller for the four missing revision-v2 global supervised baselines.

Commands
--------
smoke      Small one-epoch tests for all four models.
tune       Validation-only hyperparameter runs; test split is never loaded.
select     Select minimum validation physical RMSE for each model.
final      Run five final seeds (2026-2030) from selected configs.
summarize  Aggregate final metrics.
status     Print completion status.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

MODELS = ["CNNLSTM", "Transformer", "TimeXer", "Koopa"]
SEEDS = [2026, 2027, 2028, 2029, 2030]

# Four validation candidates per model. The test split is never constructed.
CANDIDATES = {
    "CNNLSTM": [
        {
            "name": "h64_c32_lr1em3",
            "lr": 1e-3,
            "d_model": 64,
            "e_layers": 2,
            "conv_channel": 32,
            "batch_size": 64,
        },
        {
            "name": "h128_c64_lr1em3",
            "lr": 1e-3,
            "d_model": 128,
            "e_layers": 2,
            "conv_channel": 64,
            "batch_size": 64,
        },
        {
            "name": "h64_c64_lr3em4",
            "lr": 3e-4,
            "d_model": 64,
            "e_layers": 2,
            "conv_channel": 64,
            "batch_size": 64,
        },
        {
            "name": "h128_c64_lr3em4",
            "lr": 3e-4,
            "d_model": 128,
            "e_layers": 2,
            "conv_channel": 64,
            "batch_size": 64,
        },
    ],
    "Transformer": [
        {
            "name": "d64_lr3em4",
            "lr": 3e-4,
            "d_model": 64,
            "d_ff": 128,
            "n_heads": 8,
            "e_layers": 2,
            "d_layers": 1,
            "batch_size": 64,
        },
        {
            "name": "d64_lr1em4",
            "lr": 1e-4,
            "d_model": 64,
            "d_ff": 128,
            "n_heads": 8,
            "e_layers": 2,
            "d_layers": 1,
            "batch_size": 64,
        },
        {
            "name": "d128_lr3em4",
            "lr": 3e-4,
            "d_model": 128,
            "d_ff": 256,
            "n_heads": 8,
            "e_layers": 2,
            "d_layers": 1,
            "batch_size": 64,
        },
        {
            "name": "d128_lr1em4",
            "lr": 1e-4,
            "d_model": 128,
            "d_ff": 256,
            "n_heads": 8,
            "e_layers": 2,
            "d_layers": 1,
            "batch_size": 64,
        },
    ],
    "TimeXer": [
        {
            "name": "d64_p16_lr3em4",
            "lr": 3e-4,
            "d_model": 64,
            "d_ff": 128,
            "n_heads": 8,
            "e_layers": 2,
            "patch_len": 16,
            "batch_size": 64,
        },
        {
            "name": "d64_p16_lr1em4",
            "lr": 1e-4,
            "d_model": 64,
            "d_ff": 128,
            "n_heads": 8,
            "e_layers": 2,
            "patch_len": 16,
            "batch_size": 64,
        },
        {
            "name": "d128_p16_lr3em4",
            "lr": 3e-4,
            "d_model": 128,
            "d_ff": 256,
            "n_heads": 8,
            "e_layers": 2,
            "patch_len": 16,
            "batch_size": 64,
        },
        {
            "name": "d128_p24_lr1em4",
            "lr": 1e-4,
            "d_model": 128,
            "d_ff": 256,
            "n_heads": 8,
            "e_layers": 2,
            "patch_len": 24,
            "batch_size": 64,
        },
    ],
    "Koopa": [
        {
            "name": "dyn16_h64_b1_lr1em3",
            "lr": 1e-3,
            "koopa_dynamic_dim": 16,
            "koopa_hidden_dim": 64,
            "koopa_hidden_layers": 2,
            "koopa_num_blocks": 1,
            "batch_size": 32,
        },
        {
            "name": "dyn32_h64_b1_lr3em4",
            "lr": 3e-4,
            "koopa_dynamic_dim": 32,
            "koopa_hidden_dim": 64,
            "koopa_hidden_layers": 2,
            "koopa_num_blocks": 1,
            "batch_size": 24,
        },
        {
            "name": "dyn32_h64_b2_lr3em4",
            "lr": 3e-4,
            "koopa_dynamic_dim": 32,
            "koopa_hidden_dim": 64,
            "koopa_hidden_layers": 2,
            "koopa_num_blocks": 2,
            "batch_size": 16,
        },
        {
            "name": "dyn64_h128_b2_lr1em4",
            "lr": 1e-4,
            "koopa_dynamic_dim": 64,
            "koopa_hidden_dim": 128,
            "koopa_hidden_layers": 2,
            "koopa_num_blocks": 2,
            "batch_size": 8,
        },
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "command",
        choices=["smoke", "tune", "select", "final", "summarize", "status"],
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=MODELS,
    )
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--force",
        action="store_true",
        help="Rerun even when completion marker exists.",
    )
    return p.parse_args()


def trainer_path(root: Path) -> Path:
    return (
        root
        / "experiments"
        / "scripts"
        / "train"
        / "train_global_additional_baselines.py"
    )


def results_root(root: Path) -> Path:
    return root / "experiments" / "results" / "02_global_baselines"


def configs_root(root: Path) -> Path:
    return root / "experiments" / "configs"


def model_slug(model: str) -> str:
    return model.lower()


def model_global_dir(root: Path, model: str) -> Path:
    return results_root(root) / f"1e_global_{model_slug(model)}"


def model_tuning_dir(root: Path, model: str) -> Path:
    return results_root(root) / f"1e_tuning_{model_slug(model)}"


def selected_path(root: Path, model: str) -> Path:
    return configs_root(root) / f"{model_slug(model)}.json"


def done_status(path: Path, validation_only: bool) -> bool:
    status_path = path / "run_status.json"
    if not status_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    expected = "completed_validation_only" if validation_only else "completed"
    return status.get("status") == expected


def add_kv_args(cmd: list[str], config: dict) -> None:
    for key, value in config.items():
        if key == "name":
            continue
        cmd.extend([f"--{key}", str(value)])


def stream_command(cmd: list[str], log_path: Path, dry_run: bool) -> None:
    printable = shlex.join(cmd)
    print("\n" + "=" * 88)
    print(printable)
    print("=" * 88, flush=True)

    if dry_run:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(printable + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def base_command(args: argparse.Namespace, model: str, out_dir: Path) -> list[str]:
    return [
        args.python,
        str(trainer_path(args.root)),
        "--root",
        str(args.root),
        "--model_name",
        model,
        "--variant",
        "city_time",
        "--seed",
        "2026",
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--num_workers",
        str(args.num_workers),
        "--require_cuda",
        "--out_dir",
        str(out_dir),
    ]


def run_smoke(args: argparse.Namespace) -> None:
    smoke_root = results_root(args.root) / "00_smoke_missing4"
    for model in args.models:
        out_dir = smoke_root / f"{model}_city_time_smoke"
        if done_status(out_dir, validation_only=False) and not args.force:
            print(f"[SKIP] smoke completed: {out_dir}")
            continue
        config = CANDIDATES[model][0]
        cmd = base_command(args, model, out_dir)
        cmd[cmd.index("--epochs") + 1] = "1"
        cmd[cmd.index("--patience") + 1] = "1"
        cmd.extend(
            [
                "--max_train_samples",
                "2048",
                "--max_val_samples",
                "512",
                "--max_test_samples",
                "512",
                "--save_arrays",
                "1",
                "--log_every",
                "20",
            ]
        )
        add_kv_args(cmd, config)
        stream_command(
            cmd,
            smoke_root / "logs" / f"{model}_smoke.log",
            args.dry_run,
        )


def run_tune(args: argparse.Namespace) -> None:
    for model in args.models:
        for config in CANDIDATES[model]:
            run_name = f"{model}_city_time_{config['name']}_seed2026"
            out_dir = model_tuning_dir(args.root, model) / run_name
            if done_status(out_dir, validation_only=True) and not args.force:
                print(f"[SKIP] tuning completed: {out_dir}")
                continue
            cmd = base_command(args, model, out_dir)
            cmd.extend(["--skip_test", "--save_arrays", "0"])
            add_kv_args(cmd, config)
            stream_command(
                cmd,
                model_tuning_dir(args.root, model)
                / "logs"
                / f"{run_name}.log",
                args.dry_run,
            )


def read_one_row_csv(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row in {path}, found {len(rows)}")
    return rows[0]


def run_select(args: argparse.Namespace) -> None:
    configs_root(args.root).mkdir(parents=True, exist_ok=True)
    for model in args.models:
        candidates = []
        for config in CANDIDATES[model]:
            run_name = f"{model}_city_time_{config['name']}_seed2026"
            out_dir = model_tuning_dir(args.root, model) / run_name
            summary_path = out_dir / "run_summary.csv"
            status_path = out_dir / "run_status.json"
            if not summary_path.exists() or not status_path.exists():
                raise FileNotFoundError(
                    f"Incomplete tuning candidate: {out_dir}"
                )
            row = read_one_row_csv(summary_path)
            candidates.append(
                {
                    "model_name": model,
                    "candidate_name": config["name"],
                    "source_run": run_name,
                    "source_dir": str(out_dir),
                    "best_val_RMSE": float(row["best_val_RMSE"]),
                    "best_epoch": int(float(row["best_epoch"])),
                    "command_args": {
                        key: value for key, value in config.items() if key != "name"
                    },
                }
            )

        candidates.sort(key=lambda x: x["best_val_RMSE"])
        selected = {
            **candidates[0],
            "variant": "city_time",
            "selected_by": (
                "minimum revision-v2 validation physical RMSE; "
                "test split not constructed during tuning"
            ),
            "all_candidates": candidates,
        }
        path = selected_path(args.root, model)
        path.write_text(
            json.dumps(selected, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"[SELECTED] {model}: {selected['candidate_name']} "
            f"val_RMSE={selected['best_val_RMSE']:.6f}\n"
            f"           {path}"
        )


def run_final(args: argparse.Namespace) -> None:
    for model in args.models:
        selected_file = selected_path(args.root, model)
        if not selected_file.exists():
            raise FileNotFoundError(
                f"Selected config missing: {selected_file}. Run select first."
            )
        selected = json.loads(selected_file.read_text(encoding="utf-8"))
        config = selected["command_args"]

        for seed in SEEDS:
            run_name = f"{model}_city_time_seed{seed}"
            out_dir = model_global_dir(args.root, model) / run_name
            if done_status(out_dir, validation_only=False) and not args.force:
                print(f"[SKIP] final completed: {out_dir}")
                continue

            cmd = base_command(args, model, out_dir)
            cmd[cmd.index("--seed") + 1] = str(seed)
            cmd.extend(["--save_arrays", "1"])
            add_kv_args(cmd, config)
            stream_command(
                cmd,
                model_global_dir(args.root, model)
                / "logs"
                / f"{run_name}.log",
                args.dry_run,
            )


def numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_summarize(args: argparse.Namespace) -> None:
    import pandas as pd

    rows = []
    missing = []
    for model in args.models:
        for seed in SEEDS:
            out_dir = (
                model_global_dir(args.root, model)
                / f"{model}_city_time_seed{seed}"
            )
            metric_path = out_dir / "metrics_overall.csv"
            manifest_path = out_dir / "prediction_manifest.json"
            if not metric_path.exists():
                missing.append(str(out_dir))
                continue
            row = pd.read_csv(metric_path).iloc[0].to_dict()
            row["result_dir"] = str(out_dir)
            row["prediction_manifest_exists"] = manifest_path.exists()
            if manifest_path.exists():
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                row["array_metric_match"] = manifest.get("metric_match")
                row["n_saved_samples"] = manifest.get("n_samples")
            rows.append(row)

    if missing:
        print("Missing final runs:")
        for x in missing:
            print("  ", x)
        raise RuntimeError(f"{len(missing)} final runs are incomplete.")

    df = pd.DataFrame(rows)
    out_root = results_root(args.root) / "1e_missing4_combined"
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        out_root / "all_runs.csv", index=False, encoding="utf-8-sig"
    )

    preferred_metrics = [
        "MAE",
        "RMSE",
        "city_MAE_macro",
        "city_RMSE_macro",
        "station_MAE_macro",
        "station_RMSE_macro",
        "train_seconds",
        "test_seconds",
        "peak_gpu_memory_mb",
        "total_params",
        "trainable_params",
    ]
    agg = {}
    for metric in preferred_metrics:
        if metric in df.columns:
            agg[f"{metric}_mean"] = (metric, "mean")
            agg[f"{metric}_std"] = (metric, "std")
    summary = (
        df.groupby("model_name", as_index=False)
        .agg(**agg)
        .sort_values("MAE_mean")
    )
    summary["n_seeds"] = 5
    summary.to_csv(
        out_root / "summary_by_model.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(summary.to_string(index=False))
    print(f"\nSaved: {out_root}")


def run_status(args: argparse.Namespace) -> None:
    for model in args.models:
        smoke_dir = (
            results_root(args.root)
            / "00_smoke_missing4"
            / f"{model}_city_time_smoke"
        )
        tuning_done = sum(
            done_status(
                model_tuning_dir(args.root, model)
                / f"{model}_city_time_{c['name']}_seed2026",
                validation_only=True,
            )
            for c in CANDIDATES[model]
        )
        final_done = sum(
            done_status(
                model_global_dir(args.root, model)
                / f"{model}_city_time_seed{s}",
                validation_only=False,
            )
            for s in SEEDS
        )
        print(
            f"{model:12s} "
            f"smoke={'OK' if done_status(smoke_dir, False) else '--'} "
            f"tuning={tuning_done}/4 "
            f"selected={'OK' if selected_path(args.root, model).exists() else '--'} "
            f"final={final_done}/5"
        )


def main() -> None:
    args = parse_args()
    args.root = args.root.expanduser().resolve()
    path = trainer_path(args.root)
    if not path.exists() and args.command not in {"status"}:
        raise FileNotFoundError(
            f"Trainer not found: {path}\n"
            "Upload train_global_additional_baselines.py first."
        )

    actions = {
        "smoke": run_smoke,
        "tune": run_tune,
        "select": run_select,
        "final": run_final,
        "summarize": run_summarize,
        "status": run_status,
    }
    actions[args.command](args)


if __name__ == "__main__":
    main()
