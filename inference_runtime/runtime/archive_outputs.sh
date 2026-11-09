#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
zip -r -9 jetson_outputs.zip outputs
echo "Created $ROOT/jetson_outputs.zip"
