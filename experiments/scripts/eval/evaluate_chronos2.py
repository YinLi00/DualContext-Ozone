#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Revision-v2 zero-shot inference and prediction dump for Chronos-2."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

from chronos_common import H, run_inference_archive


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument("--model_path", type=Path, default=Path("models_hf_local/chronos_2"))
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--point_col", choices=["predictions", "0.5"], default="predictions")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log_every", type=int, default=20)
    return p.parse_args()


def make_context_df(x: torch.Tensor, input_time_ns: torch.Tensor) -> pd.DataFrame:
    frames = []
    x_np = x.detach().cpu().numpy()
    t_np = input_time_ns.detach().cpu().numpy().astype(np.int64)
    for i in range(x_np.shape[0]):
        frames.append(
            pd.DataFrame(
                {
                    "id": f"series_{i}",
                    "timestamp": pd.to_datetime(t_np[i]),
                    "target": x_np[i].astype(np.float32, copy=False),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def extract_prediction_array(pred_df: pd.DataFrame, bsz: int, point_col: str) -> np.ndarray:
    df = pred_df.copy()
    if "target_name" in df.columns:
        df = df[df["target_name"] == "target"].copy()
    chosen = point_col
    if chosen not in df.columns:
        if chosen == "predictions" and "0.5" in df.columns:
            chosen = "0.5"
        elif chosen == "0.5" and "predictions" in df.columns:
            chosen = "predictions"
        else:
            raise RuntimeError(f"Point column {point_col!r} absent; columns={df.columns.tolist()}")
    sort_cols = ["id"] + (["timestamp"] if "timestamp" in df.columns else [])
    df = df.sort_values(sort_cols)
    out = []
    for i in range(bsz):
        sid = f"series_{i}"
        arr = df.loc[df["id"] == sid, chosen].to_numpy(dtype=np.float32)
        if len(arr) < H:
            raise RuntimeError(f"Chronos-2 prediction too short for {sid}: {len(arr)}")
        out.append(arr[:H])
    return np.stack(out, axis=0)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    model_path = args.model_path if args.model_path.is_absolute() else root / args.model_path
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    print("Loading Chronos-2...")
    pipeline = Chronos2Pipeline.from_pretrained(
        str(model_path),
        device_map=args.device,
        local_files_only=True,
    )
    print("Pipeline:", type(pipeline))

    def predict_batch(batch: dict[str, torch.Tensor]) -> np.ndarray:
        context_df = make_context_df(batch["x"], batch["input_time_ns"])
        with torch.inference_mode():
            pred_df = pipeline.predict_df(
                context_df,
                prediction_length=H,
                quantile_levels=[0.1, 0.5, 0.9],
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )
        return extract_prediction_array(pred_df, batch["x"].shape[0], args.point_col)

    run_inference_archive(
        root=root,
        out_dir=args.out_dir,
        model_name="Chronos-2",
        model_path=model_path,
        point_definition=f"physical input; {args.point_col} column from predict_df",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        resume=args.resume,
        overwrite=args.overwrite,
        log_every=args.log_every,
        predict_batch=predict_batch,
        run_signature_extra={"device": args.device, "point_col": args.point_col, "quantile_levels": [0.1, 0.5, 0.9]},
    )


if __name__ == "__main__":
    main()
