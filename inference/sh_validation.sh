#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 inference/run_inference.py \
  --testset prdbench \
  --model_path weights/ICRDrag \
  "$@"
