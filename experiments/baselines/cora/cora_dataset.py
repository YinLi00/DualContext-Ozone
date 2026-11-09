#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical revision-v2 multivariate Dataset for the CoRA baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


L = 96
H = 48


class CoRARevisionV2Dataset(Dataset):
    """
    Each canonical O3-index row identifies one station/start window.

    CoRA receives all 8 historical variables and all 8 future variables:
      x_std: [96, 8]
      y_std: [48, 8]

    Normalization uses the existing revision-v2 city x variable moments
    computed from the training split only.
    """

    def __init__(
        self,
        *,
        root: Path,
        split: str,
        max_samples: int = 0,
        seed: int = 2026,
    ):
        self.root = root.resolve()
        self.split = split

        index_root = self.root / "experiments/data_v2/indices"
        meta_path = index_root / "meta.json"
        stats_path = self.root / "experiments/data_v2/city_var_stats_train_only.json"

        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stats = json.loads(stats_path.read_text(encoding="utf-8"))

        self.variables = list(self.meta["variables"])
        self.cities = list(self.meta["cities"])
        self.array_paths = list(self.meta["array_paths"])
        self.stations = list(self.meta["stations"])
        self.o3_idx = int(self.meta["o3_idx"])

        if int(self.meta["seq_len"]) != L or int(self.meta["pred_len"]) != H:
            raise RuntimeError(
                f"Unexpected revision-v2 L/H: "
                f"{self.meta['seq_len']}/{self.meta['pred_len']}"
            )
        if len(self.variables) != 8:
            raise RuntimeError(f"Expected 8 variables, got {len(self.variables)}")
        if self.o3_idx != 4:
            raise RuntimeError(f"Expected O3 index 4, got {self.o3_idx}")
        if stats.get("stats_source") != "training split only":
            raise RuntimeError("Normalization statistics are not train-only")
        if list(stats["variables"]) != self.variables:
            raise RuntimeError("Stats variable order differs from meta.json")

        mu = np.empty((len(self.cities), len(self.variables)), dtype=np.float32)
        sd = np.empty_like(mu)
        for cid, city in enumerate(self.cities):
            for vid, var in enumerate(self.variables):
                s = stats["city_variable_stats"][city][var]
                mu[cid, vid] = np.float32(s["mean"])
                value = np.float32(s["std"])
                if not np.isfinite(value) or value <= 0:
                    raise RuntimeError(f"Bad std for {city}/{var}: {value}")
                sd[cid, vid] = value
        self.city_mu = mu
        self.city_std = sd

        split_path = index_root / f"{split}.npz"
        z = np.load(split_path, allow_pickle=False)
        required = {"array_id", "city_id", "var_idx", "start_idx"}
        if not required.issubset(z.files):
            raise RuntimeError(f"{split_path} missing keys: {required - set(z.files)}")

        array_id = z["array_id"].astype(np.int32)
        city_id = z["city_id"].astype(np.int16)
        var_idx = z["var_idx"].astype(np.int8)
        start_idx = z["start_idx"].astype(np.int32)
        canonical_position = np.arange(len(array_id), dtype=np.int64)

        if not np.all(var_idx == self.o3_idx):
            raise RuntimeError(f"{split}: index rows are not all O3 target rows")

        if max_samples and 0 < max_samples < len(array_id):
            rng = np.random.default_rng(seed)
            selection = rng.choice(len(array_id), size=max_samples, replace=False)
            if split != "train":
                selection.sort()
            array_id = array_id[selection]
            city_id = city_id[selection]
            var_idx = var_idx[selection]
            start_idx = start_idx[selection]
            canonical_position = canonical_position[selection]

        self.array_id = array_id
        self.city_id = city_id
        self.var_idx = var_idx
        self.start_idx = start_idx
        self.canonical_position = canonical_position

        # One cache per Dataset worker. All 295 station arrays are modest in size.
        self.cache: dict[int, np.ndarray] = {}

    def __len__(self):
        return len(self.array_id)

    def _load_values(self, array_id: int) -> np.ndarray:
        aid = int(array_id)
        if aid not in self.cache:
            path = self.root / self.array_paths[aid]
            with np.load(path, allow_pickle=True) as data:
                values = data["values"].astype(np.float32)
            if values.ndim != 2 or values.shape[1] != len(self.variables):
                raise RuntimeError(f"Unexpected values shape {values.shape} at {path}")
            if not np.isfinite(values).all():
                raise RuntimeError(f"Non-finite values in {path}")
            self.cache[aid] = values
        return self.cache[aid]

    def __getitem__(self, i: int):
        aid = int(self.array_id[i])
        cid = int(self.city_id[i])
        start = int(self.start_idx[i])

        values = self._load_values(aid)
        x_phys = values[start:start + L, :]
        y_phys = values[start + L:start + L + H, :]
        if x_phys.shape != (L, len(self.variables)):
            raise RuntimeError(f"Bad x shape for row {i}: {x_phys.shape}")
        if y_phys.shape != (H, len(self.variables)):
            raise RuntimeError(f"Bad y shape for row {i}: {y_phys.shape}")

        mu = self.city_mu[cid]
        sd = self.city_std[cid]
        x_std = (x_phys - mu[None, :]) / sd[None, :]
        y_std = (y_phys - mu[None, :]) / sd[None, :]

        return {
            "x": torch.from_numpy(x_std.astype(np.float32, copy=False)),
            "y": torch.from_numpy(y_std.astype(np.float32, copy=False)),
            "y_o3_phys": torch.from_numpy(
                y_phys[:, self.o3_idx].astype(np.float32, copy=False)
            ),
            "o3_mu": torch.tensor(mu[self.o3_idx], dtype=torch.float32),
            "o3_std": torch.tensor(sd[self.o3_idx], dtype=torch.float32),
            "city_id": torch.tensor(cid, dtype=torch.long),
            "array_id": torch.tensor(aid, dtype=torch.long),
            "start_idx": torch.tensor(start, dtype=torch.long),
            "canonical_position": torch.tensor(
                int(self.canonical_position[i]), dtype=torch.long
            ),
        }
