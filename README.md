# DualContext for multi-city ozone forecasting

Source code accompanying **Parameter-Efficient TSFM Adaptation for Distributed Monitoring — Multi-City Ozone Forecasting and Edge Inference**.

This source release starts from cleaned hourly station CSV files. It contains the data-index builders, training and evaluation entry points, archived experiment settings, and device benchmark code. The preprocessing and statistical protocols are described in the supplementary material.

## Prepare the environment and data

From the repository root, install a device-compatible PyTorch build and the dependencies in `requirements.txt`. The Python sources were checked with the Python 3.12 parser. Training and benchmark execution require the corresponding data, model files, and device environment.

The study uses air-quality measurements from the China National Environmental Monitoring Center and ERA5 temperature and relative humidity. The manuscript describes data acquisition and preprocessing. [CNEMC](https://air.cnemc.cn:18007/) · [ERA5 catalogue](https://cds.climate.copernicus.eu/datasets)

Place the cleaned files under `datasets/<city>/<station>.csv`. The CSV reader expects the following columns. Use the study's local hourly timestamp convention and its physical units.

| Fields | Format |
|---|---|
| `time` | `YYYYMMDD HH:MM` |
| `PM2.5`, `PM10`, `SO2`, `NO2`, `O3`, `CO`, `temperature`, `relative_humidity` | Numeric observations in the study's input units |

The reference station list, variable order and date ranges are in `configs/data_protocol.json`. The reference protocol retains 295 stations across 22 cities and uses a 96-hour history and 48-hour forecast.

```bash
python scripts/data/prepare_station_data.py
python experiments/scripts/data/build_data_splits.py --root . --exclude_file configs/excluded_stations.txt
```

The revision-v2 builder writes split indices and training-only normalization statistics. It also writes `experiments/data_v2/indices/test_targets.npy` from the physical O3 targets in test-index order. The formal reference has 106,200 test windows. Output and audit scripts can use this locally generated reference when an archived Timer reference is unavailable. Archived references still take precedence for exact consistency checks.

## Main experiments

Place the paper's Timer backbone in `models_hf_local/timer_base_84m/` with its configuration and custom model code, or set `IOTJ_TIMER_MODEL_PATH` to its local directory for the shared Timer trainer. The archived configurations under `configs/paper_runs/` record seeds 2026–2030 for P4, C4 and QV4.

```bash
python experiments/scripts/run/run_adapter_ablation.py formal --root . --models p4_io_time_context
python experiments/scripts/run/run_dualcontext_ablation.py formal --root . --variants c4_full_dualcontext
python experiments/scripts/run/run_monotonic_quantile.py formal --root . --cal_fraction 0.25
python evaluate_signed_cqr.py --root .
```

P4 is the forecast-attribution reference. C4 is the full variant in the DualContext ablations. QV4 splits the November validation period into an earlier model-selection segment and a later conformal-calibration segment, with boundary-straddling windows excluded. The generic December index in the data protocol is a separate split.

| Analysis | Entry point |
|---|---|
| Full adapter/time ablations | `experiments/scripts/run/run_adapter_ablation.py` |
| Full DualContext ablations | `experiments/scripts/run/run_dualcontext_ablation.py` |
| Forecast attribution and interventions | `experiments/scripts/run/run_forecast_attribution.py` and `experiments/scripts/eval/evaluate_forecast_interventions.py` |
| Alternative relevance cues, LoRA and fine-tuning | `experiments/scripts/run/run_adaptation_comparisons.py` |
| Cross-backbone output adaptation | `experiments/scripts/run/run_cross_backbone_adaptation.py` |
| Supervised comparison and training-scope analysis | `experiments/scripts/train/train_global_*_v2.py` and `experiments/scripts/run/run_supervised_scopes.py` |
| Leave-one-city-out evaluation | `experiments/scripts/run/run_dualcontext_loco.py` and `experiments/scripts/run/run_supervised_loco.py` |
| Six-model zero-shot result assembly | `experiments/scripts/summary/build_zero_shot_archive.py` after the aligned native predictions are available |
| CoRA comparison | `experiments/baselines/cora/README.md` |
| Native FP32 device benchmark | `edge_benchmark/run_edge_benchmark.py` |

The entry points expose their commands and arguments through `--help`. Use the archived run configurations to check the paper settings and provide the external assets below.

## External assets for baselines and deployment

| Component | Local input |
|---|---|
| Supervised library baselines | The study's Time-Series-Library checkout in `Time-Series-Library-main/` or a supplied `--tslib_root` for compatible trainers |
| Native pretrained baselines | The original Sundial, TimeMoE-200M, SEMPO and Chronos model assets. Cache scripts expose model paths or use the local paths shown in their argument defaults |
| Chronos zero-shot baselines | A compatible Chronos inference environment and the two `eval_chronos*_v2_dump.py` scripts under `experiments/scripts/eval/` |
| CoRA | The original checkout at `CoRA-main/` and the Timer forecasting checkpoint at `Timer_forecast_1.0.ckpt` or `--checkpoint`. Run `prepare_cora_plugin.py` in the CoRA experiment directory to generate its local plugin before training |
| Edge benchmark | The two sibling folders `inference_runtime/` and `edge_benchmark/`, with their local checkpoint and input folders populated as described below |

For the edge benchmark, provide the original Timer model under `inference_runtime/checkpoints/timer_base_84m/`, the C4 checkpoint `c4_seed2026_best.pt` in the same `checkpoints/` directory, and the regenerated index metadata at `inference_runtime/metadata/meta.json`. Put the P4, QV4 and TSMixer checkpoints in `edge_benchmark/checkpoints/` under the filenames used by the runtime loaders. Provide the corresponding model-ready inputs in `inputs/correctness/`, `inputs/latency/` and `inputs/network_refresh_295/` in the appropriate bundle. The runtime readers specify the expected NPZ fields and dtypes. The measured RTX 5080 and Jetson configurations are described in the supplementary material.

The source distribution includes the project-side code and configurations. Obtain upstream code and weights from their original sources and retain their attribution and licensing. The original data, trained weights, prepared benchmark inputs, and prediction archives are separate assets.
