#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RTX 5080 same-protocol inference benchmark wrapper.

Purpose
-------
Run the already-frozen Jetson/edge inference suite on an RTX 5080 without
changing the underlying benchmark protocol, then archive and summarize the
results separately.

Default protocol inherited from run_edge_benchmark.py:
  - FP32 native PyTorch
  - Timer / P4 / C4 / QV4 / TSMixer
  - batch-1 latency: warmup=50, timed forwards up to 1000 / target ~30 s
  - throughput: batch=32, 5 rounds x 20 iters
  - 295-station refresh: chunk=295, 5 rounds
  - timing scope inside the frozen suite:
      preconstructed CPU tensors -> H2D -> model forward -> CUDA synchronize

This wrapper DOES NOT:
  - train any model
  - use AMP / FP16 / TensorRT / torch.compile
  - modify checkpoints or runtime code
  - recompute predictive metrics
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import Any


MODELS = ["timer", "p4", "c4", "qv4", "tsmixer"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_text(cmd: list[str]) -> dict[str, Any]:
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return {
            "command": cmd,
            "returncode": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
        }
    except Exception as e:
        return {
            "command": cmd,
            "returncode": -1,
            "stdout": "",
            "stderr": repr(e),
        }


def import_versions() -> dict[str, Any]:
    import numpy as np
    import torch

    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in the current Python environment. "
            "Activate the RTX 5080 CUDA PyTorch environment before running."
        )

    name = torch.cuda.get_device_name(0)
    if "5080" not in name.upper():
        raise RuntimeError(
            f"Expected an RTX 5080 for the formal workstation benchmark, "
            f"but CUDA device 0 is: {name!r}"
        )

    free_b, total_b = torch.cuda.mem_get_info(0)

    env = {
        "captured_at": datetime.now().astimezone().isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "transformers": transformers_version,
        "gpu_name": name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_memory_free_bytes_before_run": int(free_b),
        "cuda_memory_total_bytes": int(total_b),
        "cuda_memory_free_ratio_before_run": float(free_b / total_b),
        "precision_protocol": "FP32 native PyTorch",
        "forbidden_optimizations": [
            "AMP/FP16",
            "BF16",
            "TensorRT",
            "INT8",
            "torch.compile",
        ],
    }

    # Best-effort NVIDIA environment capture. No clocks/power settings are changed.
    queries = {
        "nvidia_smi_gpu": [
            "nvidia-smi",
            "--query-gpu=name,driver_version,pstate,power.limit,"
            "clocks.current.graphics,clocks.current.memory,"
            "temperature.gpu,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        "nvidia_smi_full": ["nvidia-smi"],
    }
    env["system_commands"] = {k: run_text(v) for k, v in queries.items()}
    return env


def find_suite(edge_dir: Path) -> Path:
    candidates = [
        edge_dir / "run_edge_benchmark.py",
        edge_dir / "runtime" / "run_edge_benchmark.py",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Could not find run_edge_benchmark.py. Checked:\n  "
        + "\n  ".join(str(x) for x in candidates)
    )


def stream_process(cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> int:
    print("\n" + "=" * 88)
    print("FORMAL RTX 5080 SAME-PROTOCOL INFERENCE")
    print("=" * 88)
    print("cwd :", cwd)
    print("cmd :", subprocess.list2cmdline(cmd))
    print("log :", log_path)
    print("=" * 88 + "\n")

    with log_path.open("w", encoding="utf-8", newline="") as log:
        log.write("COMMAND:\n")
        log.write(subprocess.list2cmdline(cmd) + "\n\n")
        log.flush()

        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return p.wait()


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def load_benchmarks(raw_outputs: Path) -> dict[str, dict[str, Any]]:
    bench_dir = raw_outputs / "benchmark"
    out: dict[str, dict[str, Any]] = {}
    for m in MODELS:
        p = bench_dir / f"{m}_benchmark.json"
        if not p.is_file():
            raise FileNotFoundError(f"Missing formal benchmark JSON: {p}")
        out[m] = json.loads(p.read_text(encoding="utf-8"))
    return out


def mib(n: int | float) -> float:
    return float(n) / (1024.0 ** 2)


def write_csv(benches: dict[str, dict[str, Any]], path: Path) -> None:
    fields = [
        "model",
        "total_params",
        "b1_n",
        "b1_warmup",
        "b1_mean_ms",
        "b1_std_ms",
        "b1_median_ms",
        "b1_p95_ms",
        "b1_min_ms",
        "b1_max_ms",
        "throughput_batch",
        "throughput_rounds",
        "throughput_iters_per_round",
        "throughput_mean_windows_s",
        "throughput_std_windows_s",
        "throughput_median_windows_s",
        "throughput_min_windows_s",
        "throughput_max_windows_s",
        "refresh_stations",
        "refresh_chunk_size",
        "refresh_rounds",
        "refresh_mean_ms",
        "refresh_std_ms",
        "refresh_median_ms",
        "refresh_p95_ms",
        "refresh_min_ms",
        "refresh_max_ms",
        "refresh_mean_stations_s",
        "peak_cuda_allocated_mib",
        "peak_cuda_reserved_mib",
        "smoke_finite",
        "quantile_crossing_count",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for m in MODELS:
            d = benches[m]
            lat = d["latency"]
            thr = d["throughput"]
            ref = d["network_refresh_295"]
            smoke = d.get("smoke", {})
            w.writerow({
                "model": m,
                "total_params": d["model_parameters"],
                "b1_n": lat["n"],
                "b1_warmup": lat["warmup"],
                "b1_mean_ms": lat["mean_ms"],
                "b1_std_ms": lat["std_ms"],
                "b1_median_ms": lat["median_ms"],
                "b1_p95_ms": lat["p95_ms"],
                "b1_min_ms": lat["min_ms"],
                "b1_max_ms": lat["max_ms"],
                "throughput_batch": thr["batch_size"],
                "throughput_rounds": thr["rounds"],
                "throughput_iters_per_round": thr["iters_per_round"],
                "throughput_mean_windows_s": thr["windows_per_second_mean"],
                "throughput_std_windows_s": thr["windows_per_second_std"],
                "throughput_median_windows_s": thr["windows_per_second_median"],
                "throughput_min_windows_s": thr["windows_per_second_min"],
                "throughput_max_windows_s": thr["windows_per_second_max"],
                "refresh_stations": ref["stations"],
                "refresh_chunk_size": ref["chunk_size"],
                "refresh_rounds": ref["rounds"],
                "refresh_mean_ms": ref["refresh_ms_mean"],
                "refresh_std_ms": ref["refresh_ms_std"],
                "refresh_median_ms": ref["refresh_ms_median"],
                "refresh_p95_ms": ref["refresh_ms_p95"],
                "refresh_min_ms": ref["refresh_ms_min"],
                "refresh_max_ms": ref["refresh_ms_max"],
                "refresh_mean_stations_s": ref["stations_per_second_mean"],
                "peak_cuda_allocated_mib": mib(d["cuda_peak_allocated_bytes"]),
                "peak_cuda_reserved_mib": mib(d["cuda_peak_reserved_bytes"]),
                "smoke_finite": smoke.get("finite"),
                "quantile_crossing_count": smoke.get("quantile_crossing_count"),
            })


def write_summary(benches: dict[str, dict[str, Any]], env_data: dict[str, Any], path: Path) -> None:
    name_map = {
        "timer": "Timer",
        "p4": "P4 / IO-Time-Context",
        "c4": "C4 / Full DualContext",
        "qv4": "QV4 / Quantile-DualContext",
        "tsmixer": "TSMixer",
    }

    L: list[str] = []
    A = L.append

    A("RTX 5080 SAME-PROTOCOL INFERENCE BENCHMARK")
    A("=" * 88)
    A("")
    A(f"Captured at : {env_data['captured_at']}")
    A(f"GPU         : {env_data['gpu_name']}")
    A(f"Python      : {env_data['python'].splitlines()[0]}")
    A(f"PyTorch     : {env_data['torch']}")
    A(f"Torch CUDA  : {env_data['torch_cuda']}")
    A(f"Transformers: {env_data['transformers']}")
    A("")
    A("Formal protocol:")
    A("  Precision             : FP32 native PyTorch")
    A("  Models                : Timer / P4 / C4 / QV4 / TSMixer")
    A("  Batch-1 latency       : warmup 50; frozen suite timed protocol")
    A("  Throughput            : batch 32; 5 rounds x 20 iterations")
    A("  Network refresh       : 295 stations; chunk 295; 5 rounds")
    A("  Timing scope          : preconstructed CPU tensors -> H2D -> forward -> CUDA synchronize")
    A("  Excluded              : model/checkpoint/data loading, metrics, file saving")
    A("  Disabled              : AMP/FP16/BF16/TensorRT/INT8/torch.compile")
    A("")
    A("Core results:")
    A("-" * 88)
    A(
        f"{'Model':28s} {'B1 med ms':>10s} {'B1 P95':>10s} "
        f"{'B32 med win/s':>15s} {'295 med ms':>12s} {'Peak MiB':>10s}"
    )
    A("-" * 88)
    for m in MODELS:
        d = benches[m]
        A(
            f"{name_map[m]:28s} "
            f"{d['latency']['median_ms']:10.3f} "
            f"{d['latency']['p95_ms']:10.3f} "
            f"{d['throughput']['windows_per_second_median']:15.2f} "
            f"{d['network_refresh_295']['refresh_ms_median']:12.3f} "
            f"{mib(d['cuda_peak_allocated_bytes']):10.3f}"
        )
    A("-" * 88)
    A("")
    q = benches["qv4"].get("smoke", {})
    A(f"QV4 quantile crossing count: {q.get('quantile_crossing_count')}")
    A("")
    A("Raw benchmark arrays remain in raw_suite_outputs/benchmark/*.json.")

    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--edge-dir",
        type=Path,
        default=Path(__file__).resolve().parents[0] / "edge_benchmark",
        help="Directory containing the frozen run_edge_benchmark.py.",
    )
    ap.add_argument(
        "--archive-dir",
        type=Path,
        default=Path(__file__).resolve().parents[0] / "rtx5080_inference_benchmark_v1",
        help="Separate directory for the RTX 5080 formal archive.",
    )
    ap.add_argument("--throughput-batch", type=int, default=32)
    ap.add_argument("--refresh-chunk-size", type=int, default=295)
    ap.add_argument(
        "--skip-free-memory-warning",
        action="store_true",
        help="Do not pause on low initial CUDA free-memory ratio.",
    )
    args = ap.parse_args()

    edge_dir = args.edge_dir.resolve()
    archive_dir = args.archive_dir.resolve()
    suite = find_suite(edge_dir)

    archive_dir.mkdir(parents=True, exist_ok=True)
    log_dir = archive_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env_data = import_versions()
    env_path = archive_dir / "rtx5080_environment.json"
    env_path.write_text(
        json.dumps(env_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    free_ratio = env_data["cuda_memory_free_ratio_before_run"]
    print("=" * 88)
    print("RTX 5080 PRE-FLIGHT")
    print("=" * 88)
    print("Python :", env_data["python_executable"])
    print("GPU    :", env_data["gpu_name"])
    print("Torch  :", env_data["torch"])
    print("CUDA   :", env_data["torch_cuda"])
    print(f"Initial CUDA free memory ratio: {free_ratio:.1%}")
    print("Suite  :", suite)
    print("Archive:", archive_dir)
    print("=" * 88)

    # A warning only. This does not inspect or terminate other GPU processes.
    if free_ratio < 0.80 and not args.skip_free_memory_warning:
        print(
            "\nWARNING: less than 80% of CUDA memory is initially free.\n"
            "For formal timing, wait until other RTX 5080 workloads finish.\n"
            "Benchmark aborted before any model run.\n"
            "If this is intentional, rerun with --skip-free-memory-warning."
        )
        return 3

    # Preserve any pre-existing local suite outputs before the formal 5080 run.
    suite_outputs = edge_dir / "outputs"
    if suite_outputs.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = archive_dir / f"preexisting_suite_outputs_{ts}"
        print(f"\nBacking up pre-existing suite outputs -> {backup}")
        copy_tree(suite_outputs, backup)

    # Formal protocol: use the frozen suite unchanged.
    cmd = [
        sys.executable,
        str(suite),
        "--throughput-batch",
        str(args.throughput_batch),
        "--refresh-chunk-size",
        str(args.refresh_chunk_size),
    ]

    child_env = os.environ.copy()
    child_env["TRANSFORMERS_OFFLINE"] = "1"
    child_env["HF_HUB_OFFLINE"] = "1"
    # Do not set autocast or optimization variables.

    start = time.time()
    log_path = log_dir / "formal_benchmark_5080.log"
    rc = stream_process(cmd, edge_dir, child_env, log_path)
    elapsed = time.time() - start

    if rc != 0:
        print(f"\nERROR: frozen suite exited with return code {rc}.")
        print(f"See: {log_path}")
        return rc

    if not suite_outputs.is_dir():
        raise FileNotFoundError(
            f"The frozen suite completed but outputs directory was not found: {suite_outputs}"
        )

    raw_archive = archive_dir / "raw_suite_outputs"
    copy_tree(suite_outputs, raw_archive)

    benches = load_benchmarks(raw_archive)

    csv_path = archive_dir / "rtx5080_inference_summary.csv"
    txt_path = archive_dir / "rtx5080_inference_summary.txt"
    write_csv(benches, csv_path)
    write_summary(benches, env_data, txt_path)

    run_meta = {
        "status": "PASS",
        "wall_seconds_wrapper": elapsed,
        "edge_dir": str(edge_dir),
        "suite": str(suite),
        "python_executable": sys.executable,
        "throughput_batch": args.throughput_batch,
        "refresh_chunk_size": args.refresh_chunk_size,
        "environment_file": env_path.name,
        "formal_log": str(log_path.relative_to(archive_dir)),
        "summary_csv": csv_path.name,
        "summary_txt": txt_path.name,
        "raw_suite_outputs": raw_archive.name,
    }
    (archive_dir / "run_metadata.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Zip the entire formal archive for handoff, then write the ZIP hash
    # as a sibling file so the recorded checksum exactly matches the final ZIP.
    zip_base = archive_dir.parent / archive_dir.name
    zip_path = Path(
        shutil.make_archive(str(zip_base), "zip", root_dir=archive_dir)
    )
    zip_hash = sha256_file(zip_path)
    sha_path = zip_path.with_suffix(zip_path.suffix + ".sha256.txt")
    sha_path.write_text(
        f"{zip_hash}  {zip_path.name}\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("RTX 5080 FORMAL INFERENCE BENCHMARK PASS")
    print("=" * 88)
    print(txt_path.read_text(encoding="utf-8"))
    print("Summary CSV :", csv_path)
    print("Summary TXT :", txt_path)
    print("Environment :", env_path)
    print("Raw outputs :", raw_archive)
    print("ZIP         :", zip_path)
    print("ZIP SHA256  :", zip_hash)
    print("SHA256 file :", sha_path)
    print("=" * 88)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
