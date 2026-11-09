#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/outputs"; mkdir -p "$OUT"
python "$ROOT/runtime/timer_runtime.py" --input "$ROOT/inputs/correctness/timer_inputs.npz" --output "$OUT/timer_correctness_pred.npy" --batch_size 1
python "$ROOT/runtime/c4_runtime.py" --input "$ROOT/inputs/correctness/c4_inputs.npz" --output "$OUT/c4_correctness_pred.npy" --batch_size 1
python "$ROOT/runtime/benchmark_latency.py" --model timer --input "$ROOT/inputs/latency/timer_inputs.npz" --output_json "$OUT/timer_latency.json"
python "$ROOT/runtime/benchmark_latency.py" --model c4 --input "$ROOT/inputs/latency/c4_inputs.npz" --output_json "$OUT/c4_latency.json"
python "$ROOT/runtime/timer_runtime.py" --input "$ROOT/inputs/network_refresh_295/timer_inputs.npz" --output "$OUT/timer_network_refresh_295_pred.npy" --batch_size 295
python "$ROOT/runtime/c4_runtime.py" --input "$ROOT/inputs/network_refresh_295/c4_inputs.npz" --output "$OUT/c4_network_refresh_295_pred.npy" --batch_size 295
echo "DONE: $OUT"
