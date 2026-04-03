#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="${1:-Qwen/Qwen3-VL-8B-Instruct}"
OEQ_EVALUATOR_BACKEND="${OEQ_EVALUATOR_BACKEND:-local}"
OEQ_EVALUATOR_MODEL="${OEQ_EVALUATOR_MODEL:-}"
OEQ_EVALUATOR_CONCURRENCY="${OEQ_EVALUATOR_CONCURRENCY:-1}"
EXTRA_ARGS=()

if [[ -n "$OEQ_EVALUATOR_MODEL" ]]; then
  EXTRA_ARGS+=(--oeq-evaluator-model "$OEQ_EVALUATOR_MODEL")
fi
if [[ "$OEQ_EVALUATOR_CONCURRENCY" != "1" ]]; then
  EXTRA_ARGS+=(--oeq-evaluator-concurrency "$OEQ_EVALUATOR_CONCURRENCY")
fi

python3 "$ROOT_DIR/evaluate_predictions.py" \
  --task all \
  --split public_val \
  --predictions-root "$ROOT_DIR/starter_kit_outputs" \
  --model "$MODEL_ID" \
  --oeq-evaluator-backend "$OEQ_EVALUATOR_BACKEND" \
  "${EXTRA_ARGS[@]}"
