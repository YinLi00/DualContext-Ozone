#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from edge_common import default_base_root, environment_report, resolve_device, sync, write_json
from p4_runtime import load_p4
from qv4_runtime import load_qv4
from tsmixer_runtime import call_tsmixer, load_tsmixer

MODELS = ("timer", "p4", "c4", "qv4", "tsmixer")


def import_base_runtime(base_root: Path):
    r = base_root / "runtime"
    if not r.is_dir():
        raise FileNotFoundError(f"Base runtime directory missing: {r}")
    if str(r) not in sys.path:
        sys.path.insert(0, str(r))
    import timer_runtime
    import c4_runtime
    return timer_runtime, c4_runtime


def load_model(name: str, ext_root: Path, base_root: Path, device: torch.device):
    if name == "timer":
        tr, _ = import_base_runtime(base_root)
        return tr.load_timer(base_root, device), None
    if name == "c4":
        _, cr = import_base_runtime(base_root)
        return cr.load_c4(base_root, device), None
    if name == "p4":
        return load_p4(ext_root, base_root, device), None
    if name == "qv4":
        return load_qv4(ext_root, base_root, device), None
    if name == "tsmixer":
        return load_tsmixer(ext_root, device)
    raise ValueError(name)


def input_path(name: str, split: str, ext_root: Path, base_root: Path) -> Path:
    if name == "timer":
        return base_root / "inputs" / split / "timer_inputs.npz"
    if name == "c4":
        return base_root / "inputs" / split / "c4_inputs.npz"
    return ext_root / "inputs" / split / "common_inputs.npz"


def load_cpu_cache(path: Path, name: str, pin_memory: bool) -> Dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=False)
    needed = ["x"] if name == "timer" else ["x", "x_multi", "time_feat", "var_idx", "city_id"]
    cache = {}
    for k in needed:
        if k not in z.files:
            raise KeyError(f"{path}: missing {k}; available={z.files}")
        if k in ("var_idx", "city_id"):
            t = torch.from_numpy(np.asarray(z[k], dtype=np.int64))
        else:
            t = torch.from_numpy(np.asarray(z[k], dtype=np.float32))
        if pin_memory:
            try:
                t = t.pin_memory()
            except Exception:
                pass
        cache[k] = t
    return cache


def cache_len(cache: Dict[str, torch.Tensor]) -> int:
    return int(next(iter(cache.values())).shape[0])


def copy_batch(cache: Dict[str, torch.Tensor], start: int, end: int,
               device: torch.device, non_blocking: bool):
    return {
        k: v[start:end].to(device=device, non_blocking=non_blocking)
        for k, v in cache.items()
    }


def forward_model(name: str, model, extra, batch: dict):
    if name == "timer":
        out = model(batch["x"]).logits
        if out.shape[-1] < 48:
            raise RuntimeError(f"Timer logits too short: {tuple(out.shape)}")
        return out[:, :48].float()
    if name in ("c4", "p4"):
        return model(
            batch["x"], batch["x_multi"], batch["time_feat"],
            batch["var_idx"], batch["city_id"],
        ).float()
    if name == "qv4":
        return model(
            batch["x"], batch["x_multi"], batch["time_feat"],
            batch["var_idx"], batch["city_id"],
        ).float()
    if name == "tsmixer":
        return call_tsmixer(model, batch, int(extra)).float()
    raise ValueError(name)


def check_output(name: str, out: torch.Tensor, batch: int) -> dict:
    expected = (batch, 48, 5) if name == "qv4" else (batch, 48)
    if tuple(out.shape) != expected:
        raise RuntimeError(f"{name}: expected output {expected}, got {tuple(out.shape)}")
    finite = bool(torch.isfinite(out).all().item())
    if not finite:
        raise RuntimeError(f"{name}: non-finite output")
    rep = {"output_shape": list(out.shape), "finite": finite}
    if name == "qv4":
        crossing = int((out[..., :-1] > out[..., 1:]).sum().item())
        rep["quantile_crossing_count"] = crossing
        if crossing != 0:
            raise RuntimeError(f"QV4 crossing_count={crossing}")
    return rep


def stats_ms(values):
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean_ms": float(a.mean()),
        "std_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.quantile(a, 0.95)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def sample_start(i: int, batch: int, n: int) -> int:
    if batch > n:
        raise ValueError(f"batch={batch} > input pool n={n}")
    span = n - batch + 1
    return (i * batch) % span if span > 1 else 0


@torch.inference_mode()
def smoke(name, model, extra, cache, device, non_blocking):
    b = min(2, cache_len(cache))
    batch = copy_batch(cache, 0, b, device, non_blocking)
    out = forward_model(name, model, extra, batch)
    sync(device)
    return check_output(name, out, b)


@torch.inference_mode()
def latency(name, model, extra, cache, device, *, warmup: int,
            min_iters: int, max_iters: int, target_seconds: float,
            non_blocking: bool):
    n = cache_len(cache)
    for i in range(warmup):
        s = sample_start(i, 1, n)
        batch = copy_batch(cache, s, s + 1, device, non_blocking)
        out = forward_model(name, model, extra, batch)
        check_output(name, out, 1)
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    raw = []
    wall0 = time.perf_counter()
    i = 0
    while i < max_iters:
        s = sample_start(i, 1, n)
        sync(device)
        t0 = time.perf_counter()
        batch = copy_batch(cache, s, s + 1, device, non_blocking)
        out = forward_model(name, model, extra, batch)
        sync(device)
        raw.append((time.perf_counter() - t0) * 1000.0)
        i += 1
        if i >= min_iters and (time.perf_counter() - wall0) >= target_seconds:
            break
    rep = stats_ms(raw)
    rep.update({
        "batch_size": 1,
        "warmup": int(warmup),
        "target_seconds": float(target_seconds),
        "max_iters": int(max_iters),
        "timing_scope": "preconstructed CPU tensors -> H2D -> model forward -> CUDA synchronize",
        "raw_latency_ms": [float(x) for x in raw],
    })
    return rep


@torch.inference_mode()
def throughput(name, model, extra, cache, device, *, batch_size: int,
               warmup: int, rounds: int, iters_per_round: int,
               non_blocking: bool):
    n = cache_len(cache)
    if batch_size > n:
        raise ValueError(f"throughput batch {batch_size} > pool {n}")
    for i in range(warmup):
        s = sample_start(i, batch_size, n)
        b = copy_batch(cache, s, s + batch_size, device, non_blocking)
        out = forward_model(name, model, extra, b)
        check_output(name, out, batch_size)
    sync(device)
    vals = []
    ms = []
    global_i = 0
    for r in range(rounds):
        sync(device)
        t0 = time.perf_counter()
        for _ in range(iters_per_round):
            s = sample_start(global_i, batch_size, n)
            b = copy_batch(cache, s, s + batch_size, device, non_blocking)
            forward_model(name, model, extra, b)
            global_i += 1
        sync(device)
        dt = time.perf_counter() - t0
        windows = batch_size * iters_per_round
        vals.append(windows / dt)
        ms.append(dt * 1000.0)
    a = np.asarray(vals, dtype=np.float64)
    return {
        "batch_size": int(batch_size),
        "rounds": int(rounds),
        "iters_per_round": int(iters_per_round),
        "windows_per_second_mean": float(a.mean()),
        "windows_per_second_std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "windows_per_second_median": float(np.median(a)),
        "windows_per_second_min": float(a.min()),
        "windows_per_second_max": float(a.max()),
        "raw_windows_per_second": [float(x) for x in vals],
        "raw_round_ms": [float(x) for x in ms],
        "timing_scope": "preconstructed CPU tensors -> H2D -> model forward -> CUDA synchronize",
    }


@torch.inference_mode()
def refresh(name, model, extra, cache, device, *, chunk_size: int,
            warmup: int, rounds: int, non_blocking: bool):
    n = cache_len(cache)
    if n != 295:
        raise RuntimeError(f"network refresh input must contain 295 stations, got {n}")
    if chunk_size <= 0:
        chunk_size = n
    chunk_size = min(chunk_size, n)

    def one_refresh(check=False):
        last = None
        for s in range(0, n, chunk_size):
            e = min(n, s + chunk_size)
            b = copy_batch(cache, s, e, device, non_blocking)
            last = forward_model(name, model, extra, b)
            if check:
                check_output(name, last, e - s)
        return last

    for _ in range(warmup):
        one_refresh(check=False)
    sync(device)
    raw_ms = []
    for r in range(rounds):
        sync(device)
        t0 = time.perf_counter()
        one_refresh(check=(r == 0))
        sync(device)
        raw_ms.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(raw_ms, dtype=np.float64)
    sps = 295.0 / (a / 1000.0)
    return {
        "stations": 295,
        "forecast_horizon": 48,
        "chunk_size": int(chunk_size),
        "rounds": int(rounds),
        "refresh_ms_mean": float(a.mean()),
        "refresh_ms_std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "refresh_ms_median": float(np.median(a)),
        "refresh_ms_p95": float(np.quantile(a, 0.95)),
        "refresh_ms_min": float(a.min()),
        "refresh_ms_max": float(a.max()),
        "stations_per_second_mean": float(sps.mean()),
        "raw_refresh_ms": [float(x) for x in raw_ms],
        "timing_scope": "entire 295-station refresh, H2D + forward for all chunks + CUDA synchronize",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--mode", choices=["smoke", "latency", "throughput", "refresh", "all"], default="all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--base-root", type=Path, default=None)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--pin-memory", type=int, choices=[0, 1], default=0)
    ap.add_argument("--non-blocking", type=int, choices=[0, 1], default=0)
    ap.add_argument("--latency-warmup", type=int, default=50)
    ap.add_argument("--latency-min-iters", type=int, default=100)
    ap.add_argument("--latency-max-iters", type=int, default=1000)
    ap.add_argument("--latency-target-seconds", type=float, default=30.0)
    ap.add_argument("--throughput-batch", type=int, default=32)
    ap.add_argument("--throughput-warmup", type=int, default=20)
    ap.add_argument("--throughput-rounds", type=int, default=5)
    ap.add_argument("--throughput-iters-per-round", type=int, default=20)
    ap.add_argument("--refresh-chunk-size", type=int, default=295)
    ap.add_argument("--refresh-warmup", type=int, default=3)
    ap.add_argument("--refresh-rounds", type=int, default=5)
    a = ap.parse_args()

    ext_root = Path(__file__).resolve().parents[1]
    base_root = a.base_root.resolve() if a.base_root else default_base_root(ext_root)
    device = resolve_device(a.device)
    if device.type != "cuda":
        raise RuntimeError("Formal edge latency runner requires CUDA; use runtime scripts separately for CPU debugging")

    rep = {
        "model": a.model,
        "mode": a.mode,
        "precision": "fp32",
        "extension_root": str(ext_root),
        "base_root": str(base_root),
        "environment": environment_report(device),
    }
    model, extra = load_model(a.model, ext_root, base_root, device)
    sync(device)
    rep["model_parameters"] = int(sum(p.numel() for p in model.parameters()))
    rep["cuda_allocated_after_model_load_bytes"] = int(torch.cuda.memory_allocated(device))
    rep["cuda_reserved_after_model_load_bytes"] = int(torch.cuda.memory_reserved(device))

    # Always smoke on correctness input first.
    cpath = input_path(a.model, "correctness", ext_root, base_root)
    ccache = load_cpu_cache(cpath, a.model, bool(a.pin_memory))
    rep["smoke"] = smoke(a.model, model, extra, ccache, device, bool(a.non_blocking))

    if a.mode in ("latency", "all"):
        lpath = input_path(a.model, "latency", ext_root, base_root)
        lcache = load_cpu_cache(lpath, a.model, bool(a.pin_memory))
        rep["latency"] = latency(
            a.model, model, extra, lcache, device,
            warmup=a.latency_warmup, min_iters=a.latency_min_iters,
            max_iters=a.latency_max_iters, target_seconds=a.latency_target_seconds,
            non_blocking=bool(a.non_blocking),
        )
    if a.mode in ("throughput", "all"):
        if "lcache" not in locals():
            lpath = input_path(a.model, "latency", ext_root, base_root)
            lcache = load_cpu_cache(lpath, a.model, bool(a.pin_memory))
        rep["throughput"] = throughput(
            a.model, model, extra, lcache, device,
            batch_size=a.throughput_batch, warmup=a.throughput_warmup,
            rounds=a.throughput_rounds, iters_per_round=a.throughput_iters_per_round,
            non_blocking=bool(a.non_blocking),
        )
    if a.mode in ("refresh", "all"):
        rpath = input_path(a.model, "network_refresh_295", ext_root, base_root)
        rcache = load_cpu_cache(rpath, a.model, bool(a.pin_memory))
        rep["network_refresh_295"] = refresh(
            a.model, model, extra, rcache, device,
            chunk_size=a.refresh_chunk_size, warmup=a.refresh_warmup,
            rounds=a.refresh_rounds, non_blocking=bool(a.non_blocking),
        )

    rep["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    rep["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    write_json(a.output_json, rep)
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
