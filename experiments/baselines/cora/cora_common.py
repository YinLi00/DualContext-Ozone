#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Timer-67M / CoRA bridge for the revision-v2 formal baseline."""

from __future__ import annotations

import importlib.util
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


TIMER_PARAMETER_COUNT = 67_397_728


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Keep benchmarking deterministic where practical.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def install_timer_namespace(cora_root: Path) -> None:
    """
    Make the official Timer package importable without executing the unrelated
    ts_benchmark.baselines.pre_train.__init__ import chain.
    """
    packages = {
        "ts_benchmark": cora_root / "ts_benchmark",
        "ts_benchmark.baselines": cora_root / "ts_benchmark/baselines",
        "ts_benchmark.baselines.pre_train":
            cora_root / "ts_benchmark/baselines/pre_train",
        "ts_benchmark.baselines.pre_train.submodules":
            cora_root / "ts_benchmark/baselines/pre_train/submodules",
        "ts_benchmark.baselines.pre_train.submodules.Timer":
            cora_root / "ts_benchmark/baselines/pre_train/submodules/Timer",
    }
    for name, path in packages.items():
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        mod.__package__ = name
        sys.modules[name] = mod


def import_timer_core(cora_root: Path):
    install_timer_namespace(cora_root)
    from ts_benchmark.baselines.pre_train.submodules.Timer.models.Timer import (
        Model as TimerCore,
    )
    return TimerCore


def load_clean_cora_plugin(plugin_path: Path):
    """
    Load the audited local copy of official plugin_ar.py.

    Required local compatibility edits:
      - remove one unused TinyTimeMixer import;
      - prediction head 96 -> configs.horizon;
      - output_patch_len 96 -> configs.horizon;
      - fuse the FIRST H Timer steps when H < 96.
    """
    if not plugin_path.is_file():
        raise FileNotFoundError(plugin_path)

    text = plugin_path.read_text(encoding="utf-8")
    required = [
        "PredictionHead(False,self.n_vars,self.d_model,configs.horizon",
        "self.output_patch_len = configs.horizon",
        "output = output[:, :self.prediction_length, :]",
        "output_plugin = output_plugin[:, :self.prediction_length, :]",
    ]
    missing = [x for x in required if x not in text]
    if missing:
        raise RuntimeError(
            "Vendored CoRA plugin has not passed the audited H=48 patch. "
            f"Missing markers: {missing}"
        )

    spec = importlib.util.spec_from_file_location("_cora_plugin_ar_v2", plugin_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {plugin_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Timer67MBridge(nn.Module):
    """
    Official Timer forecasting backbone plus the interface expected by CoRA.

    The official Lightning checkpoint is loaded explicitly with
    weights_only=False because PyTorch >=2.6 changed torch.load's default.
    The checkpoint is trusted only because it is the user-downloaded official
    Timer forecasting checkpoint.
    """

    def __init__(self, timer_core_cls, checkpoint: Path):
        super().__init__()
        cfg = SimpleNamespace(
            task_name="forecast",
            ckpt_path="",
            patch_len=96,
            d_model=1024,
            d_ff=2048,
            e_layers=8,
            n_heads=8,
            factor=3,
            dropout=0.1,
            activation="gelu",
            output_attention=False,
        )
        self.model = timer_core_cls(cfg)

        obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict) or "state_dict" not in obj:
            raise RuntimeError("Official Timer checkpoint does not contain state_dict")

        clean = {}
        for key, value in obj["state_dict"].items():
            if not key.startswith("model."):
                raise RuntimeError(f"Unexpected Timer checkpoint key: {key}")
            clean[key[6:]] = value

        # Strong compatibility check.
        self.model.backbone.load_state_dict(clean, strict=True)

        count = sum(p.numel() for p in self.parameters())
        if count != TIMER_PARAMETER_COUNT:
            raise RuntimeError(
                f"Unexpected Timer parameter count: {count:,} "
                f"(expected {TIMER_PARAMETER_COUNT:,})"
            )

    def get_settings(self):
        m = self.model
        patch_num = (96 - m.patch_len) // m.patch_len + 1
        return m.d_model, m.patch_len, m.stride, patch_num

    def forcast_for_plugin_decoder(
        self, inputs, x_mark_enc=None, dec_inp=None, x_mark_dec=None
    ):
        # Official Plugin_AG sends [(B*K), L, 1].
        out, emb = self.model.forcast_for_plugin_decoder(
            inputs, x_mark_enc, dec_inp, x_mark_dec
        )
        # Timer returns [(B*K), P, D]; Plugin_AG expects [(B*K), 1, P, D].
        return out, emb.unsqueeze(1)

    def denorm_for_plugin(self, inputs):
        # CoRA branch is [B, H, K], while Timer stored normalization for
        # [(B*K), H, 1].
        b, h, k = inputs.shape
        flat = rearrange(inputs, "b h k -> (b k) h 1")
        out = self.model.denorm_for_plugin(flat)
        return rearrange(out, "(b k) h 1 -> b h k", b=b, k=k)


def make_cora_config(
    *,
    n_vars: int = 8,
    seq_len: int = 96,
    horizon: int = 48,
    plugin_dim: int = 512,
    dropout: float = 0.2,
    head_dropout: float = 0.1,
    num_before: int = 3,
    num_after: int = 4,
    beta: float = 0.2,
    gamma: float = 0.001,
    K: int = 3,
    de: int = 4,
    threshold: float = 0.3,
    plugin_lr: float = 1e-4,
    backbone_lr: float = 1e-4,
):
    plugin = SimpleNamespace(
        plugin_dim=plugin_dim,
        dropout=dropout,
        head_dropout=head_dropout,
        num_before=num_before,
        num_after=num_after,
        beta=beta,
        gama=gamma,          # official source spelling
        K=K,
        de=de,
        thresold=threshold,  # official source spelling
        M=n_vars // 10 + 1,
        plugin_lr=plugin_lr,
        backbone_lr=backbone_lr,
    )
    return SimpleNamespace(
        enc_in=n_vars,
        seq_len=seq_len,
        horizon=horizon,
        pred_len=horizon,
        is_train=1,
        plugin=plugin,
    )


def build_cora_model(
    *,
    root: Path,
    checkpoint: Path,
    plugin_path: Path,
    config,
):
    cora_root = root / "CoRA-main"
    timer_core_cls = import_timer_core(cora_root)
    plugin_mod = load_clean_cora_plugin(plugin_path)
    fm = Timer67MBridge(timer_core_cls, checkpoint)
    model = plugin_mod.Plugin(backbone=fm, pth="", configs=config)
    return model


def parameter_groups(model: nn.Module):
    backbone = []
    plugin = []
    for name, p in model.named_parameters():
        if name.startswith("fm."):
            backbone.append(p)
        else:
            plugin.append(p)
    return backbone, plugin
