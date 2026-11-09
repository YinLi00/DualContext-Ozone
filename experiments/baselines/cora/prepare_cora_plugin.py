#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create an auditable local CoRA Timer plugin for the revision-v2 48-h task.

The script DOES NOT modify CoRA-main.

Source:
  CoRA-main/ts_benchmark/plugin/plugin_ar.py

Generated:
  experiments/baselines/cora/vendor/plugin_ar_v2.py
  experiments/baselines/cora/vendor/plugin_ar_v2.patch.txt
  experiments/baselines/cora/vendor/plugin_ar_v2_manifest.json

Changes are intentionally minimal:
1) remove one confirmed-unused TinyTimeMixer import;
2) replace the two hard-coded PredictionHead forecast lengths (96) with configs.horizon;
3) set output_patch_len to configs.horizon;
4) align backbone/plugin outputs to prediction_length before beta fusion.

All replacements are asserted. If the upstream source differs, the script aborts
instead of silently producing a modified method.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path


DEAD_IMPORT = (
    "from ts_benchmark.baselines.pre_train.submodules.TinyTimeMixer."
    "modeling_tinytimemixer import TinyTimeMixerLayer,TinyTimeMixerConfig,"
    "TinyTimeMixerChannelFeatureMixerBlock,FeatureMixerBlock,PatchMixerBlock"
)

HEAD_OLD = (
    "{'pos':PredictionHead(False,self.n_vars,self.d_model,96,head_dropout=head_dropout),\n"
    "             'neg':PredictionHead(False,self.n_vars,self.d_model,96,head_dropout=head_dropout)}"
)

HEAD_NEW = (
    "{'pos':PredictionHead(False,self.n_vars,self.d_model,configs.horizon,head_dropout=head_dropout),\n"
    "             'neg':PredictionHead(False,self.n_vars,self.d_model,configs.horizon,head_dropout=head_dropout)}"
)

PATCHLEN_OLD = "self.output_patch_len = 96"
PATCHLEN_NEW = "self.output_patch_len = configs.horizon"

FUSION_OLD = """        output = rearrange(output, '(b c) t 1 -> b t c' , c = self.n_vars)
        output = (output_plugin*self.beta + output * (1-self.beta))
"""

FUSION_NEW = """        output = rearrange(output, '(b c) t 1 -> b t c' , c = self.n_vars)

        # Revision-v2 task alignment: both branches must predict exactly H.
        output = output[:, -self.prediction_length:, :]
        output_plugin = output_plugin[:, -self.prediction_length:, :]
        if output.shape != output_plugin.shape:
            raise RuntimeError(
                f"CoRA/base horizon mismatch: base={tuple(output.shape)} "
                f"plugin={tuple(output_plugin.shape)}"
            )

        output = (output_plugin*self.beta + output * (1-self.beta))
"""


def sha256_bytes(x: bytes) -> str:
    return hashlib.sha256(x).hexdigest()


def require_count(text: str, needle: str, expected: int, name: str) -> None:
    got = text.count(needle)
    if got != expected:
        raise RuntimeError(
            f"{name}: expected {expected} occurrence(s), found {got}. "
            "Upstream source may differ; aborting."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    args = ap.parse_args()
    root = args.root.resolve()

    src = root / "CoRA-main/ts_benchmark/plugin/plugin_ar.py"
    out_dir = root / "experiments/baselines/cora/vendor"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "plugin_ar_v2.py"

    if not src.is_file():
        raise FileNotFoundError(src)

    raw = src.read_bytes()
    text = raw.decode("utf-8")

    require_count(text, DEAD_IMPORT, 1, "dead TinyTimeMixer import")
    require_count(text, HEAD_OLD, 1, "hard-coded 96-h CoRA heads")
    require_count(text, PATCHLEN_OLD, 1, "output_patch_len=96")
    require_count(text, FUSION_OLD, 1, "beta fusion block")

    patched = text.replace(DEAD_IMPORT + "\n", "")
    patched = patched.replace(HEAD_OLD, HEAD_NEW)
    patched = patched.replace(PATCHLEN_OLD, PATCHLEN_NEW)
    patched = patched.replace(FUSION_OLD, FUSION_NEW)

    # Postconditions.
    if "TinyTimeMixerLayer" in patched:
        raise RuntimeError("Dead TinyTimeMixer symbols remain after patch.")
    if "self.d_model,96,head_dropout" in patched:
        raise RuntimeError("A hard-coded 96-h CoRA head remains after patch.")
    if "self.output_patch_len = 96" in patched:
        raise RuntimeError("output_patch_len remains hard-coded at 96.")

    dst.write_text(patched, encoding="utf-8")

    diff = "".join(
        difflib.unified_diff(
            text.splitlines(True),
            patched.splitlines(True),
            fromfile=str(src),
            tofile=str(dst),
        )
    )
    (out_dir / "plugin_ar_v2.patch.txt").write_text(diff, encoding="utf-8")

    manifest = {
        "source": str(src),
        "generated": str(dst),
        "source_sha256": sha256_bytes(raw),
        "generated_sha256": sha256_bytes(patched.encode("utf-8")),
        "changes": [
            "removed confirmed-unused TinyTimeMixer import",
            "PredictionHead forecast_len: hard-coded 96 -> configs.horizon",
            "output_patch_len: hard-coded 96 -> configs.horizon",
            "slice Timer base and CoRA plugin predictions to prediction_length before beta fusion",
        ],
        "unchanged_core": [
            "Channel_contrastive",
            "Pearson/learned correlation composition",
            "positive/negative polarity branches",
            "contrastive objective",
            "projection blocks",
            "normalization",
            "beta fusion formula",
        ],
    }
    (out_dir / "plugin_ar_v2_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[PASS] clean CoRA plugin generated")
    print("source   :", src)
    print("generated:", dst)
    print("diff     :", out_dir / "plugin_ar_v2.patch.txt")
    print("manifest :", out_dir / "plugin_ar_v2_manifest.json")
    print()
    print(diff)


if __name__ == "__main__":
    main()
