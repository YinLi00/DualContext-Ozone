EM&S_2 EDGE EXTENSION V1

Models:
- P4 / IO-Time-Context
- QV4 / Quantile-DualContext
- TSMixer supervised reference

Dependency:
This directory must be placed next to the frozen inference_runtime directory.
Timer backbone is reused from:
  ../inference_runtime/checkpoints/timer_base_84m

Formal Jetson policy:
- FP32 primary only in this v1 runtime.
- No training.
- No labels / true / formal server predictions.
- Jetson only generates fresh predictions and latency/throughput/295-station timing.
- Metrics/statistics are computed after copying outputs back to Windows.

Suggested first command on Jetson:
  python runtime/collect_edge_env.py --output-json outputs/device_env.json --device cuda
  python run_edge_benchmark.py --smoke-only

After all smoke tests pass:
  python run_edge_benchmark.py --throughput-batch 32 --refresh-chunk-size 295

Base package used at build time:
<project-root>\inference_runtime
