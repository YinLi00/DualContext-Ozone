# Formal CoRA baseline — revision-v2

## Frozen protocol

- Method: Cheng et al. CoRA correlation-aware adapter/plugin.
- Backbone: official Timer forecasting checkpoint loaded into the official Timer-67M architecture.
- Official checkpoint used on this server: `<project-root>/Timer_forecast_1.0.ckpt`.
- Timer trainable parameters: 67,397,728.
- CoRA parameters in the audited H=48 configuration: 10,578,057.
- Total: 77,975,785.
- Input: 96 h × 8 historical variables.
- Supervision: 48 h × 8 future variables.
- Paper evaluation: O3 only (`o3_idx=4`).
- Data: canonical revision-v2 22-city pooled train/val/test.
- Normalization: existing city × variable statistics computed from the training split only.
- City identity: not supplied to the model.
- No future pollutant or meteorological value is an inference input.
- Model selection: validation physical-unit O3 RMSE.
- Seeds: 2026–2030.

## Audited compatibility edits

The original CoRA/Timer code is preserved except for the local vendored plugin bridge already created in this experiment directory:

1. remove one confirmed-unused TinyTimeMixer import that pulled unrelated `timm/torchvision` dependencies;
2. replace the CoRA prediction head's hard-coded 96 with `configs.horizon=48`;
3. set `output_patch_len=configs.horizon`;
4. when H<96, fuse the **first H** Timer forecast steps, matching official Timer inference;
5. load the trusted official Lightning checkpoint with `weights_only=False` for PyTorch >=2.6 compatibility.

The CoRA correlation mechanism, positive/negative branches, contrastive objective, projection blocks, normalization, and beta fusion formula are unchanged.

## Optimizer note

The active official repository branch named `train_mode='plugin'` calls an optimizer on `self.model.fm.parameters()`, while `fm` is the Timer backbone. This is ambiguous/inconsistent with intended CoRA adaptation.

The primary formal setting here is therefore:

`--optimizer_scope all`

which explicitly steps both Timer and CoRA parameters and logs their parameter groups. The literal repository behavior is retained as an optional sensitivity mode:

`--optimizer_scope repo_literal`

Do **not** mix these modes within the five formal seeds.

## Recommended sequence

### 1. GPU integration smoke on a free GPU

```bash
conda activate cora_v2
cd /path/to/DualContext-Ozone

python experiments/baselines/cora/train_cora.py \
  --seed 2026 \
  --epochs 1 \
  --patience 1 \
  --batch_size 64 \
  --num_workers 2 \
  --max_train_samples 512 \
  --max_val_samples 256 \
  --max_test_samples 256 \
  --out_dir experiments/results/18_cora_baseline_smoke
```

### 2. Formal experiments across five seeds

```bash
bash experiments/baselines/cora/run_cora_experiments.sh
```

The runner audits each seed and summarizes the five runs.

## Result layout

`experiments/results/18_cora_baseline/seedXXXX/`

contains:
- `pred_o3.npy`
- `true_o3.npy`
- `city_ids.npy`
- `array_id.npy`
- `start_idx.npy`
- `canonical_test_positions.npy`
- `metrics_overall.csv`
- `metrics_by_city.csv`
- `metrics_by_horizon.csv`
- `training_log.csv`
- `best.pt`
- `run_manifest.json`
