#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch

L = 96
H = 48
QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return dev


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def default_base_root(extension_root: Path) -> Path:
    return extension_root.parent / "inference_runtime"


def load_timer(base_root: Path, device: torch.device) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    model_dir = base_root / "checkpoints" / "timer_base_84m"
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Timer backbone not found: {model_dir}\n"
            "Expected inference_runtime to be a sibling of edge_benchmark."
        )
    timer = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    if hasattr(timer.config, "use_cache"):
        timer.config.use_cache = False
    timer.eval()
    timer.to(device)
    return timer


def npz_len(z) -> int:
    if "x" in z.files:
        return int(len(z["x"]))
    if "x_multi" in z.files:
        return int(len(z["x_multi"]))
    raise KeyError("NPZ contains neither x nor x_multi")


def validate_common_npz(z) -> None:
    required = ["x", "x_multi", "time_feat", "var_idx", "city_id", "mu", "std"]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise KeyError(f"Missing input keys: {missing}; available={z.files}")
    n = len(z["x"])
    expected = {
        "x": (n, 96),
        "x_multi": (n, 96, 8),
        "time_feat": (n, 192),
        "var_idx": (n,),
        "city_id": (n,),
        "mu": (n,),
        "std": (n,),
    }
    for k, shape in expected.items():
        if tuple(z[k].shape) != shape:
            raise ValueError(f"{k}: expected {shape}, got {z[k].shape}")


def cpu_tensor_cache(z) -> Dict[str, torch.Tensor]:
    validate_common_npz(z)
    return {
        "x": torch.from_numpy(np.asarray(z["x"], dtype=np.float32)),
        "x_multi": torch.from_numpy(np.asarray(z["x_multi"], dtype=np.float32)),
        "time_feat": torch.from_numpy(np.asarray(z["time_feat"], dtype=np.float32)),
        "var_idx": torch.from_numpy(np.asarray(z["var_idx"], dtype=np.int64)),
        "city_id": torch.from_numpy(np.asarray(z["city_id"], dtype=np.int64)),
        "mu": torch.from_numpy(np.asarray(z["mu"], dtype=np.float32)),
        "std": torch.from_numpy(np.asarray(z["std"], dtype=np.float32)),
    }


def maybe_pin(cache: Dict[str, torch.Tensor], enabled: bool) -> Dict[str, torch.Tensor]:
    if not enabled:
        return cache
    out = {}
    for k, v in cache.items():
        try:
            out[k] = v.pin_memory()
        except Exception:
            out[k] = v
    return out


def move_slice(
    cache: Dict[str, torch.Tensor],
    start: int,
    end: int,
    device: torch.device,
    non_blocking: bool,
) -> Dict[str, torch.Tensor]:
    keys = ("x", "x_multi", "time_feat", "var_idx", "city_id")
    return {
        k: cache[k][start:end].to(device=device, non_blocking=non_blocking)
        for k in keys
    }


def inverse_point(pred_std: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (pred_std * std[:, None] + mu[:, None]).astype(np.float32, copy=False)


def inverse_quantiles(pred_std: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (pred_std * std[:, None, None] + mu[:, None, None]).astype(np.float32, copy=False)


def environment_report(device: torch.device) -> dict:
    rep = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": torch.version.cuda,
        "device": str(device),
    }
    try:
        import transformers
        rep["transformers"] = transformers.__version__
    except Exception as e:
        rep["transformers_error"] = repr(e)
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        rep.update({
            "gpu_name": torch.cuda.get_device_name(idx),
            "compute_capability": list(torch.cuda.get_device_capability(idx)),
            "total_cuda_memory_bytes": int(props.total_memory),
        })
    return rep


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
