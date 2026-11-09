#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared revision-v2 data, archival, resume, and metric utilities for Chronos."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

L = 96
H = 48
EXPECTED_FULL_N = 106_200


def atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_root(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def station_id_from_meta(meta: dict[str, Any], aid: int) -> str:
    stations = meta.get("stations")
    if isinstance(stations, list) and 0 <= aid < len(stations):
        row = stations[aid]
        if isinstance(row, dict):
            for key in ("station_id", "station", "site_id", "code", "name"):
                value = row.get(key)
                if value is not None and str(value).strip():
                    return str(value)
    try:
        return Path(meta["array_paths"][aid]).stem
    except Exception:
        return str(aid)


class V2OzoneDataset(Dataset):
    """Physical-space O3 windows in exact revision-v2 test index order."""

    def __init__(
        self,
        root: Path,
        split: str = "test",
        row_offset: int = 0,
        max_samples: int = 0,
    ) -> None:
        self.root = root
        self.index_root = root / "experiments/data_v2/indices"
        self.meta_path = self.index_root / "meta.json"
        self.index_path = self.index_root / f"{split}.npz"
        if not self.meta_path.exists():
            raise FileNotFoundError(self.meta_path)
        if not self.index_path.exists():
            raise FileNotFoundError(self.index_path)

        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.variables = list(self.meta["variables"])
        self.cities = list(self.meta["cities"])
        self.array_paths = list(self.meta["array_paths"])
        self.o3_idx = int(self.meta.get("o3_idx", self.variables.index("O3")))

        with np.load(self.index_path, allow_pickle=False) as z:
            full_array_id = np.asarray(z["array_id"], dtype=np.int64)
            full_city_id = np.asarray(z["city_id"], dtype=np.int64)
            full_start_idx = np.asarray(z["start_idx"], dtype=np.int64)

        n_full = len(full_array_id)
        if split == "test" and n_full != EXPECTED_FULL_N:
            raise RuntimeError(f"Unexpected revision-v2 test size: {n_full:,}")
        if not (0 <= row_offset <= n_full):
            raise ValueError(f"row_offset={row_offset}, n_full={n_full}")

        stop = n_full if max_samples <= 0 else min(n_full, row_offset + max_samples)
        self.global_rows = np.arange(row_offset, stop, dtype=np.int64)
        self.array_id = full_array_id[row_offset:stop]
        self.city_id = full_city_id[row_offset:stop]
        self.start_idx = full_start_idx[row_offset:stop]
        self.n_full = n_full
        self.row_offset = row_offset
        self.cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.array_id)

    def _load(self, aid: int) -> tuple[np.ndarray, np.ndarray]:
        aid = int(aid)
        if aid not in self.cache:
            raw = Path(self.array_paths[aid])
            path = raw if raw.is_absolute() else self.root / raw
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=True) as z:
                values = np.asarray(z["values"], dtype=np.float32)
                if "time_ns" in z.files:
                    time_ns = np.asarray(z["time_ns"], dtype=np.int64)
                elif "timestamps_ns" in z.files:
                    time_ns = np.asarray(z["timestamps_ns"], dtype=np.int64)
                else:
                    raise KeyError(f"No time_ns in {path}; keys={z.files}")
            self.cache[aid] = (values, time_ns)
        return self.cache[aid]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        aid = int(self.array_id[i])
        cid = int(self.city_id[i])
        st = int(self.start_idx[i])
        values, time_ns = self._load(aid)

        x = values[st : st + L, self.o3_idx]
        y = values[st + L : st + L + H, self.o3_idx]
        input_time_ns = time_ns[st : st + L]
        target_time_ns = time_ns[st + L : st + L + H]
        if x.shape != (L,) or y.shape != (H,):
            raise RuntimeError(
                f"Bad window row={int(self.global_rows[i])}, aid={aid}, st={st}, "
                f"x={x.shape}, y={y.shape}"
            )
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise RuntimeError(
                f"Non-finite physical O3 window row={int(self.global_rows[i])}, "
                f"aid={aid}, st={st}"
            )

        return {
            "x": torch.from_numpy(np.asarray(x, dtype=np.float32)),
            "y": torch.from_numpy(np.asarray(y, dtype=np.float32)),
            "input_time_ns": torch.from_numpy(np.asarray(input_time_ns, dtype=np.int64)),
            "target_time_ns": torch.from_numpy(np.asarray(target_time_ns, dtype=np.int64)),
            "sample_index": torch.tensor(int(self.global_rows[i]), dtype=torch.long),
            "array_id": torch.tensor(aid, dtype=torch.long),
            "city_id": torch.tensor(cid, dtype=torch.long),
            "start_idx": torch.tensor(st, dtype=torch.long),
            "forecast_start_time_ns": torch.tensor(int(target_time_ns[0]), dtype=torch.long),
        }


def make_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    kwargs: dict[str, Any] = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


@dataclass
class ArrayPaths:
    pred: Path
    true: Path
    city_id: Path
    array_id: Path
    start_idx: Path
    sample_index: Path
    forecast_start_time_ns: Path


def _array_paths(out_dir: Path, partial: bool) -> ArrayPaths:
    suffix = ".partial.npy" if partial else ".npy"
    return ArrayPaths(
        pred=out_dir / f"pred{suffix}",
        true=out_dir / f"true{suffix}",
        city_id=out_dir / f"city_id{suffix}",
        array_id=out_dir / f"array_id{suffix}",
        start_idx=out_dir / f"start_idx{suffix}",
        sample_index=out_dir / f"sample_index{suffix}",
        forecast_start_time_ns=out_dir / f"forecast_start_time_ns{suffix}",
    )


class PredictionArchive:
    def __init__(
        self,
        out_dir: Path,
        n_samples: int,
        run_signature: dict[str, Any],
        resume: bool,
        overwrite: bool,
    ) -> None:
        self.out_dir = out_dir
        self.n_samples = n_samples
        self.status_path = out_dir / "run_status.json"
        self.partial = _array_paths(out_dir, partial=True)
        self.final = _array_paths(out_dir, partial=False)
        out_dir.mkdir(parents=True, exist_ok=True)

        if overwrite:
            for p in list(vars(self.partial).values()) + list(vars(self.final).values()):
                Path(p).unlink(missing_ok=True)
            for p in [
                self.status_path,
                out_dir / "metrics_overall.csv",
                out_dir / "metrics_by_city.csv",
                out_dir / "metrics_by_horizon.csv",
                out_dir / "metrics_by_station.csv",
                out_dir / "sample_metadata.csv",
                out_dir / "prediction_manifest.json",
            ]:
                p.unlink(missing_ok=True)

        if self.status_path.exists():
            status = json.loads(self.status_path.read_text(encoding="utf-8"))
            if status.get("state") == "complete" and all(
                Path(p).exists() for p in vars(self.final).values()
            ):
                self.already_complete = True
                self.completed = n_samples
                self.status = status
                return
            if not resume:
                raise FileExistsError(
                    f"Incomplete output exists: {out_dir}; use --resume or --overwrite"
                )
            old_sig = status.get("run_signature")
            if old_sig != run_signature:
                raise RuntimeError(
                    "Existing partial run signature differs. Use --overwrite only after checking.\n"
                    f"old={old_sig}\nnew={run_signature}"
                )
            self.completed = int(status.get("completed_samples", 0))
        else:
            self.completed = 0

        self.already_complete = False
        self.status = {
            "state": "running",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "completed_samples": self.completed,
            "n_samples": n_samples,
            "run_signature": run_signature,
        }
        self._open_arrays(create=self.completed == 0)
        self.write_status()

    def _open_arrays(self, create: bool) -> None:
        mode = "w+" if create else "r+"
        specs = {
            "pred": ((self.n_samples, H), np.float32),
            "true": ((self.n_samples, H), np.float32),
            "city_id": ((self.n_samples,), np.int16),
            "array_id": ((self.n_samples,), np.int32),
            "start_idx": ((self.n_samples,), np.int32),
            "sample_index": ((self.n_samples,), np.int64),
            "forecast_start_time_ns": ((self.n_samples,), np.int64),
        }
        self.arrays: dict[str, np.memmap] = {}
        for name, (shape, dtype) in specs.items():
            path = getattr(self.partial, name)
            if not create and not path.exists():
                raise FileNotFoundError(path)
            arr = np.lib.format.open_memmap(path, mode=mode, dtype=dtype, shape=shape)
            self.arrays[name] = arr

    def write_batch(self, start: int, batch: dict[str, np.ndarray]) -> int:
        b = int(batch["pred"].shape[0])
        end = start + b
        if end > self.n_samples:
            raise RuntimeError(f"Write beyond archive: {start}:{end}/{self.n_samples}")
        for name, arr in batch.items():
            self.arrays[name][start:end] = arr
        self.completed = end
        return end

    def flush(self) -> None:
        for arr in self.arrays.values():
            arr.flush()

    def write_status(self, **extra: Any) -> None:
        self.status.update(extra)
        self.status.update(
            {
                "state": self.status.get("state", "running"),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "completed_samples": self.completed,
                "n_samples": self.n_samples,
                "progress_pct": 100.0 * self.completed / max(1, self.n_samples),
            }
        )
        atomic_write_json(self.status_path, self.status)

    def mark_failed(self, exc: BaseException) -> None:
        try:
            self.flush()
        except Exception:
            pass
        self.status["state"] = "failed"
        self.write_status(
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )

    def finalize_files(self) -> None:
        self.flush()
        self.arrays.clear()
        for name in vars(self.partial):
            os.replace(getattr(self.partial, name), getattr(self.final, name))


def streaming_metrics(true_path: Path, pred_path: Path, block: int = 8192) -> dict[str, Any]:
    true = np.load(true_path, mmap_mode="r", allow_pickle=False)
    pred = np.load(pred_path, mmap_mode="r", allow_pickle=False)
    if true.shape != pred.shape:
        raise RuntimeError(f"Shape mismatch: true={true.shape}, pred={pred.shape}")
    abs_sum = 0.0
    sq_sum = 0.0
    count = 0
    n_nonfinite = 0
    for s in range(0, true.shape[0], block):
        e = min(s + block, true.shape[0])
        t = np.asarray(true[s:e], dtype=np.float64)
        p = np.asarray(pred[s:e], dtype=np.float64)
        finite = np.isfinite(t) & np.isfinite(p)
        n_nonfinite += int(finite.size - finite.sum())
        if finite.any():
            d = p[finite] - t[finite]
            abs_sum += float(np.abs(d).sum(dtype=np.float64))
            sq_sum += float(np.square(d).sum(dtype=np.float64))
            count += int(d.size)
    if count == 0:
        raise RuntimeError("No finite prediction/true pairs")
    mse = sq_sum / count
    return {
        "MAE": abs_sum / count,
        "MSE": mse,
        "RMSE": math.sqrt(mse),
        "count": count,
        "nonfinite_count": n_nonfinite,
    }


def _metric_arrays(true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(true) & np.isfinite(pred)
    if not finite.any():
        return {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "count": 0}
    d = pred[finite].astype(np.float64) - true[finite].astype(np.float64)
    mse = float(np.mean(np.square(d), dtype=np.float64))
    return {
        "MAE": float(np.mean(np.abs(d), dtype=np.float64)),
        "MSE": mse,
        "RMSE": math.sqrt(mse),
        "count": int(d.size),
    }


def validate_true_reference(root: Path, true_path: Path, n_samples: int) -> dict[str, Any]:
    ref_path = (
        root
        / "experiments/results/03_zeroshot_baselines_v2_reindexed"
        / "00_test_reference/true.npy"
    )
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Canonical revision-v2 true reference is required: {ref_path}"
        )
    ref = np.load(ref_path, mmap_mode="r", allow_pickle=False)
    got = np.load(true_path, mmap_mode="r", allow_pickle=False)
    expected = ref[:n_samples]
    if got.shape != expected.shape:
        raise RuntimeError(f"Reference shape mismatch: got={got.shape}, expected={expected.shape}")
    exact = bool(np.array_equal(got, expected))
    max_abs = float(np.max(np.abs(got.astype(np.float64) - expected.astype(np.float64))))
    if not exact:
        raise RuntimeError(f"Reconstructed true differs from canonical v2 true; max_abs={max_abs}")
    return {
        "reference_path": str(ref_path),
        "exact_equal": exact,
        "max_abs_diff": max_abs,
        "reference_sha256": sha256_file(ref_path),
        "generated_true_sha256": sha256_file(true_path),
    }


def create_reports(
    root: Path,
    out_dir: Path,
    model_name: str,
    model_path: Path,
    point_definition: str,
    run_signature: dict[str, Any],
    started_at: float,
) -> dict[str, Any]:
    paths = _array_paths(out_dir, partial=False)
    pred = np.load(paths.pred, mmap_mode="r", allow_pickle=False)
    true = np.load(paths.true, mmap_mode="r", allow_pickle=False)
    city_id = np.load(paths.city_id, mmap_mode="r", allow_pickle=False)
    array_id = np.load(paths.array_id, mmap_mode="r", allow_pickle=False)
    start_idx = np.load(paths.start_idx, mmap_mode="r", allow_pickle=False)
    sample_index = np.load(paths.sample_index, mmap_mode="r", allow_pickle=False)
    forecast_ns = np.load(paths.forecast_start_time_ns, mmap_mode="r", allow_pickle=False)

    n = len(pred)
    if pred.shape != (n, H) or true.shape != (n, H):
        raise RuntimeError(f"Unexpected final arrays pred={pred.shape}, true={true.shape}")
    if not np.array_equal(sample_index, np.arange(n, dtype=np.int64)):
        raise RuntimeError("sample_index is not exact 0..N-1 revision-v2 order")

    true_check = validate_true_reference(root, paths.true, n)
    overall = streaming_metrics(paths.true, paths.pred)
    if overall["nonfinite_count"] != 0:
        raise RuntimeError(f"Non-finite final predictions: {overall['nonfinite_count']}")

    meta_path = root / "experiments/data_v2/indices/meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cities = list(meta["cities"])

    overall_row = {
        "model_name": model_name,
        "protocol": "revision-v2",
        "point_definition": point_definition,
        **overall,
    }
    pd.DataFrame([overall_row]).to_csv(
        out_dir / "metrics_overall.csv", index=False, encoding="utf-8-sig"
    )

    city_rows = []
    for cid in sorted(np.unique(city_id).tolist()):
        mask = city_id == cid
        city_rows.append(
            {
                "city_id": int(cid),
                "city": cities[int(cid)],
                "n_samples": int(mask.sum()),
                **_metric_arrays(true[mask], pred[mask]),
            }
        )
    pd.DataFrame(city_rows).to_csv(
        out_dir / "metrics_by_city.csv", index=False, encoding="utf-8-sig"
    )

    horizon_rows = []
    for h in range(H):
        horizon_rows.append({"horizon": h + 1, **_metric_arrays(true[:, h], pred[:, h])})
    pd.DataFrame(horizon_rows).to_csv(
        out_dir / "metrics_by_horizon.csv", index=False, encoding="utf-8-sig"
    )

    station_rows = []
    for aid in sorted(np.unique(array_id).tolist()):
        mask = array_id == aid
        station_rows.append(
            {
                "array_id": int(aid),
                "station_id": station_id_from_meta(meta, int(aid)),
                "city_id": int(city_id[np.flatnonzero(mask)[0]]),
                "city": cities[int(city_id[np.flatnonzero(mask)[0]])],
                "n_samples": int(mask.sum()),
                **_metric_arrays(true[mask], pred[mask]),
            }
        )
    pd.DataFrame(station_rows).to_csv(
        out_dir / "metrics_by_station.csv", index=False, encoding="utf-8-sig"
    )

    metadata = pd.DataFrame(
        {
            "sample_index": sample_index,
            "array_id": array_id,
            "station_id": [station_id_from_meta(meta, int(a)) for a in array_id],
            "city_id": city_id,
            "city": [cities[int(c)] for c in city_id],
            "start_idx": start_idx,
            "forecast_start_time_ns": forecast_ns,
            "forecast_start_time": pd.to_datetime(forecast_ns.astype(np.int64)),
        }
    )
    metadata.to_csv(out_dir / "sample_metadata.csv", index=False, encoding="utf-8-sig")

    config_path = model_path / "config.json"
    manifest = {
        "model_name": model_name,
        "protocol": "revision-v2",
        "input": "physical-space univariate O3, past 96 hourly values",
        "forecast_horizon": 48,
        "point_definition": point_definition,
        "n_samples": n,
        "scalar_count": int(n * H),
        "model_path": str(model_path),
        "model_config_path": str(config_path),
        "model_config_sha256": sha256_file(config_path) if config_path.exists() else None,
        "run_signature": run_signature,
        "true_reference_check": true_check,
        "metrics": overall,
        "array_sha256": {
            name: sha256_file(getattr(paths, name)) for name in vars(paths)
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "elapsed_seconds": time.time() - started_at,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_json(out_dir / "prediction_manifest.json", manifest)
    return manifest


def run_inference_archive(
    *,
    root: Path,
    out_dir: Path,
    model_name: str,
    model_path: Path,
    point_definition: str,
    batch_size: int,
    num_workers: int,
    max_samples: int,
    resume: bool,
    overwrite: bool,
    log_every: int,
    predict_batch: Callable[[dict[str, torch.Tensor]], np.ndarray],
    run_signature_extra: Optional[dict[str, Any]] = None,
) -> None:
    root = resolve_root(root)
    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = root / model_path

    index_path = root / "experiments/data_v2/indices/test.npz"
    with np.load(index_path, allow_pickle=False) as z:
        n_full = len(z["array_id"])
    n_samples = n_full if max_samples <= 0 else min(max_samples, n_full)

    run_signature = {
        "model_name": model_name,
        "protocol": "revision-v2",
        "n_samples": n_samples,
        "seq_len": L,
        "pred_len": H,
        "batch_size": batch_size,
        "point_definition": point_definition,
        "model_path": str(model_path.resolve()),
        **(run_signature_extra or {}),
    }

    archive = PredictionArchive(
        out_dir=out_dir,
        n_samples=n_samples,
        run_signature=run_signature,
        resume=resume,
        overwrite=overwrite,
    )
    if archive.already_complete:
        print(f"[SKIP] Complete output already exists: {out_dir}")
        print(pd.read_csv(out_dir / "metrics_overall.csv").to_string(index=False))
        return

    started_at = time.time()
    completed_before = archive.completed
    remaining = n_samples - completed_before
    dataset = V2OzoneDataset(
        root=root,
        split="test",
        row_offset=completed_before,
        max_samples=remaining,
    )
    loader = make_loader(dataset, batch_size=batch_size, num_workers=num_workers)

    print("=" * 80)
    print(f"Model       : {model_name}")
    print("Protocol    : revision-v2")
    print(f"Samples     : {n_samples:,}")
    print(f"Resume row  : {completed_before:,}")
    print(f"Batch size  : {batch_size}")
    print(f"Workers     : {num_workers}")
    print(f"Model path  : {model_path}")
    print(f"Output      : {out_dir}")
    print(f"Point       : {point_definition}")
    print("=" * 80, flush=True)

    try:
        cursor = completed_before
        n_batches = len(loader)
        for bi, batch in enumerate(loader, start=1):
            pred = np.asarray(predict_batch(batch), dtype=np.float32)
            y = batch["y"].detach().cpu().numpy().astype(np.float32, copy=False)
            b = y.shape[0]
            if pred.shape != (b, H):
                raise RuntimeError(f"Bad prediction shape: {pred.shape}, expected={(b, H)}")
            if not np.isfinite(pred).all():
                bad = int(pred.size - np.isfinite(pred).sum())
                raise RuntimeError(f"Non-finite predictions in batch {bi}: {bad}")

            packed = {
                "pred": pred,
                "true": y,
                "city_id": batch["city_id"].cpu().numpy().astype(np.int16),
                "array_id": batch["array_id"].cpu().numpy().astype(np.int32),
                "start_idx": batch["start_idx"].cpu().numpy().astype(np.int32),
                "sample_index": batch["sample_index"].cpu().numpy().astype(np.int64),
                "forecast_start_time_ns": batch["forecast_start_time_ns"]
                .cpu()
                .numpy()
                .astype(np.int64),
            }
            cursor = archive.write_batch(cursor, packed)

            if bi == 1 or bi % log_every == 0 or bi == n_batches:
                archive.flush()
                elapsed = time.time() - started_at
                processed_now = cursor - completed_before
                rate = processed_now / max(elapsed, 1e-9)
                eta = (n_samples - cursor) / max(rate, 1e-9)
                archive.write_status(
                    state="running",
                    elapsed_this_session_seconds=elapsed,
                    samples_per_second=rate,
                    eta_seconds=eta,
                    current_batch=bi,
                    total_batches=n_batches,
                )
                batch_metric = _metric_arrays(y, pred)
                print(
                    f"[{bi:5d}/{n_batches:5d}] rows={cursor:,}/{n_samples:,} "
                    f"batch_MAE={batch_metric['MAE']:.4f} "
                    f"batch_RMSE={batch_metric['RMSE']:.4f} "
                    f"rate={rate:.2f} samples/s ETA={eta/3600:.2f} h",
                    flush=True,
                )

        if cursor != n_samples:
            raise RuntimeError(f"Incomplete archive: cursor={cursor}, expected={n_samples}")
        archive.finalize_files()
        manifest = create_reports(
            root=root,
            out_dir=out_dir,
            model_name=model_name,
            model_path=model_path,
            point_definition=point_definition,
            run_signature=run_signature,
            started_at=started_at,
        )
        archive.status["state"] = "complete"
        archive.write_status(
            state="complete",
            completed_samples=n_samples,
            metrics=manifest["metrics"],
            manifest_path=str(out_dir / "prediction_manifest.json"),
        )
        print("\n==== Complete ====")
        print(pd.read_csv(out_dir / "metrics_overall.csv").to_string(index=False))
        print("Saved to:", out_dir)
    except BaseException as exc:
        archive.mark_failed(exc)
        raise
