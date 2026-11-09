#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the established Sundial cache script on the revision-v2 Dataset.

The historical Sundial script imports the old Stage1 dataset module by name.
This wrapper injects a compatibility module that exposes the canonical
revision-v2 Dataset with the same constructor, then executes the established
Sundial inference implementation unchanged.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    wrapper_args, old_args = parser.parse_known_args()
    root = wrapper_args.root.expanduser().resolve()

    source_path = (
        root / "experiments/scripts/train/"
        "train_attribution_reference.py"
    )
    source = import_file(source_path, "_sundial_revision_v2_source")
    index_root = root / "experiments/data_v2/indices"
    stats_path = (
        root / "experiments/data_v2/"
        "city_var_stats_train_only.json"
    )
    _, cora = source.load_project_modules(root, index_root, stats_path)
    NativeDataset = cora.Stage1O3CoRAContextDataset

    class RevisionV2Dataset(NativeDataset):
        def __init__(
            self,
            split="test",
            max_samples=0,
            seed=2026,
            time_feature_mode="both",
            *args,
            **kwargs,
        ):
            # The native class uses the same positional contract.
            super().__init__(
                split,
                int(max_samples or 0),
                int(seed),
                time_feature_mode,
            )

    shim_name = "train_timer_context_ablation"
    shim = types.ModuleType(shim_name)
    shim.Stage1O3CoRAContextDataset = RevisionV2Dataset
    sys.modules[shim_name] = shim

    old_script = root / "scripts/eval/cache_sundial_forecasts.py"
    old_module = import_file(
        old_script, "_sundial_established_cache_revision_v2"
    )
    # transformers >= 4.53 routes both greedy and sampling generation
    # through GenerationMixin._sample(). Sundial's time-series-specific
    # patch-token attention-mask handling is implemented in _greedy_search().
    # Restore the intended Sundial generation route for this process only.
    original_from_pretrained = old_module.AutoModelForCausalLM.from_pretrained

    def patched_from_pretrained(*args, **kwargs):
        model = original_from_pretrained(*args, **kwargs)

        if not hasattr(model, "_greedy_search"):
            raise RuntimeError(
                "Loaded Sundial model does not expose its custom "
                "_greedy_search implementation."
            )

        model._sample = model._greedy_search

        print(
            "[Sundial compatibility] transformers _sample "
            "-> Sundial _greedy_search",
            flush=True,
        )
        return model

    old_module.AutoModelForCausalLM.from_pretrained = (
        patched_from_pretrained
    )

    sys.argv = [str(old_script)] + old_args
    old_module.main()


if __name__ == "__main__":
    main()
