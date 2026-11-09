#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Revision-v2 zero-shot inference and prediction dump for Chronos-Bolt-Base."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from chronos import BaseChronosPipeline

from chronos_common import H, run_inference_archive


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument("--model_path", type=Path, default=Path("models_hf_local/chronos_bolt_base"))
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log_every", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    model_path = args.model_path if args.model_path.is_absolute() else root / args.model_path
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print("Loading Chronos-Bolt-Base...")
    try:
        pipeline = BaseChronosPipeline.from_pretrained(
            str(model_path),
            device_map=device,
            dtype=dtype,
            local_files_only=True,
        )
    except TypeError:
        pipeline = BaseChronosPipeline.from_pretrained(
            str(model_path),
            device_map=device,
            torch_dtype=dtype,
            local_files_only=True,
        )
    print("Pipeline:", type(pipeline))

    @torch.inference_mode()
    def predict_batch(batch: dict[str, torch.Tensor]) -> np.ndarray:
        x = batch["x"]
        quantiles, _mean = pipeline.predict_quantiles(
            inputs=x,
            prediction_length=H,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        q = quantiles.detach().float().cpu().numpy()
        b = x.shape[0]
        if q.shape == (b, H, 3):
            pred = q[:, :, 1]
        elif q.shape == (b, 3, H):
            pred = q[:, 1, :]
        else:
            raise RuntimeError(f"Unexpected Chronos-Bolt quantile shape: {q.shape}")
        return np.asarray(pred, dtype=np.float32)

    run_inference_archive(
        root=root,
        out_dir=args.out_dir,
        model_name="Chronos-Bolt-Base",
        model_path=model_path,
        point_definition="physical input; q0.5 median from predict_quantiles",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        resume=args.resume,
        overwrite=args.overwrite,
        log_every=args.log_every,
        predict_batch=predict_batch,
        run_signature_extra={"device": device, "dtype": args.dtype, "quantile_levels": [0.1, 0.5, 0.9]},
    )


if __name__ == "__main__":
    main()
