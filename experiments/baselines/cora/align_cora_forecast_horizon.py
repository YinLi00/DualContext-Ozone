#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix the only remaining H=48 semantic issue in the vendored CoRA plugin.

For Timer, one 96-point input patch produces a 96-step forecast patch.
The official Timer inference wrapper truncates H<96 by taking the FIRST H
forecast steps. Therefore the revision-v2 CoRA bridge must fuse horizons
1..48, not horizons 49..96.

This script only edits the local vendored file under experiments.
It does NOT modify CoRA-main.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    args = ap.parse_args()
    root = args.root.resolve()
    p = root / "experiments/baselines/cora/vendor/plugin_ar_v2.py"
    if not p.is_file():
        raise FileNotFoundError(p)

    s = p.read_text(encoding="utf-8")

    old1 = "output = output[:, -self.prediction_length:, :]"
    old2 = "output_plugin = output_plugin[:, -self.prediction_length:, :]"
    new1 = "output = output[:, :self.prediction_length, :]"
    new2 = "output_plugin = output_plugin[:, :self.prediction_length, :]"

    if new1 in s and new2 in s:
        print("[PASS] H=48 alignment is already FIRST-H.")
        return

    if s.count(old1) != 1 or s.count(old2) != 1:
        raise RuntimeError(
            "Expected exactly one LAST-H slice for both base and plugin outputs. "
            "Source differs from the audited version; aborting."
        )

    backup = p.with_suffix(f".py.bak_h48_{int(time.time())}")
    shutil.copy2(p, backup)

    s = s.replace(old1, new1, 1).replace(old2, new2, 1)
    p.write_text(s, encoding="utf-8")

    print("[PASS] patched H=48 alignment: LAST-H -> FIRST-H")
    print("file   :", p)
    print("backup :", backup)
    print("reason : official Timer H<96 inference keeps horizons 1..H")


if __name__ == "__main__":
    main()
