#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "$ROOT_DIR/run_baseline.py" \
  --task all \
  --split public_val \
  --model microsoft/Phi-4-multimodal-instruct \
  --cache-dir "$ROOT_DIR/models"
