#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal 48-h bridge for the official Timer CoRA Plugin_AG.

This file intentionally imports and reuses the official CoRA implementation.
It does NOT replace DCE/correlation estimation, positive/negative branches,
contrastive loss, projection blocks, normalization, beta fusion, or Timer.

Required adaptation:
- official `plugin_ar.Plugin` constructs PredictionHead(..., forecast_len=96)
- revision-v2 requires H=48 with L=96
- with one 96-h Timer patch, the official head would otherwise emit 96 steps
- this subclass replaces only the two output heads with `forecast_len=H`,
  sets `output_patch_len=H` for the inherited inference loop, and slices the
  backbone output to the last H steps before beta fusion.
"""

from __future__ import annotations

import torch
from einops import rearrange

from ts_benchmark.plugin.plugin_ar import (
    Plugin as OfficialTimerCoRAPlugin,
    PredictionHead,
)


class TimerCoRAPlugin48(OfficialTimerCoRAPlugin):
    """Official Timer CoRA architecture with a task-matched 48-h output head."""

    def __init__(self, backbone=None, pth="", configs=None):
        super().__init__(backbone=backbone, pth=pth, configs=configs)

        horizon = int(configs.horizon)
        if horizon <= 0:
            raise ValueError(f"Invalid horizon={horizon}")

        # Minimal task adaptation: official code hard-codes 96 here.
        self.head = torch.nn.ModuleDict(
            {
                "pos": PredictionHead(
                    False,
                    self.n_vars,
                    self.d_model,
                    horizon,
                    head_dropout=configs.plugin.head_dropout,
                ),
                "neg": PredictionHead(
                    False,
                    self.n_vars,
                    self.d_model,
                    horizon,
                    head_dropout=configs.plugin.head_dropout,
                ),
            }
        )
        self.prediction_length = horizon

        # The inherited inference loop assumes each CoRA forward emits
        # `output_patch_len` steps. For revision-v2 the emitted block is H.
        self.output_patch_len = horizon

    def forcast(
        self,
        input,
        x_mark_enc,
        dec_inp,
        x_mark_dec,
        device=None,
        num_samples=None,
    ):
        """
        Same CoRA computation as official plugin_ar.Plugin.forcast(), except
        both branches are aligned to the configured horizon before fusion.
        """
        output, embedding = self.fm.forcast_for_plugin_decoder(
            input, x_mark_enc, dec_inp, x_mark_dec
        )

        # Official Timer plugin path returns a flattened channel batch.
        embedding = rearrange(
            embedding, "(b k) 1 p d -> b k p d", k=self.n_vars
        )

        embedding = self.dropout(embedding)

        # Preserve official 96-h patch construction for correlation estimation.
        input_patch = rearrange(
            input,
            "(b c) (n p) 1 -> (b n) c p",
            p=self.patch_size,
            c=self.n_vars,
        )

        self.contrastive.cal_corr(input_patch, embedding)

        loss = {}
        enhance = {}
        for polarity in ["pos", "neg"]:
            x = self.adapter[polarity](embedding)
            x_mixer = x
            for mixer in self.projecions_before[polarity]:
                x_mixer = mixer(x_mixer)

            cc_loss, _ = self.contrastive(
                rearrange(x_mixer, "b c p d -> (b p) c d"),
                polarity,
            )
            loss[polarity] = cc_loss

            for mixer in self.projecions_after[polarity]:
                x_mixer = mixer(x_mixer)

            x_mixer = self.norm(x_mixer + x)
            enhance[polarity] = x_mixer

        # Official code uses a common head over summed pos/neg features.
        enhance = self.head["neg"](enhance["neg"] + enhance["pos"])
        enhance = rearrange(enhance, "b p c t -> b (p t) c")
        output_plugin = self.fm.denorm_for_plugin(enhance)

        # Restore Timer channel dimension then align to revision-v2 H.
        output = rearrange(
            output, "(b c) t 1 -> b t c", c=self.n_vars
        )
        output = output[:, -self.prediction_length :, :]
        output_plugin = output_plugin[:, -self.prediction_length :, :]

        if output.shape != output_plugin.shape:
            raise RuntimeError(
                "CoRA/base output mismatch after horizon alignment: "
                f"base={tuple(output.shape)} plugin={tuple(output_plugin.shape)}"
            )

        fused = (
            output_plugin * self.beta
            + output * (1.0 - self.beta)
        )

        cc = self.gama * (loss["neg"] + loss["pos"])

        if self.config.is_train == 0:
            return fused
        if self.training:
            return fused, cc
        return fused
