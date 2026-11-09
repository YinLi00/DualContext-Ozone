#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

for SEED in 2026 2027 2028 2029 2030; do
  echo "================================================================================"
  echo "CoRA formal seed ${SEED}"
  echo "================================================================================"
  python experiments/baselines/cora/train_cora.py \
    --seed "$SEED" \
    --device cuda \
    --batch_size 64 \
    --num_workers 4 \
    --epochs 10 \
    --patience 3 \
    --backbone_lr 1e-4 \
    --plugin_lr 1e-4 \
    --alpha_mae 0.5 \
    --optimizer_scope all \
    --out_dir experiments/results/18_cora_baseline

  python experiments/baselines/cora/verify_cora_results.py \
    --seed "$SEED"
done

python experiments/baselines/cora/summarize_cora.py
