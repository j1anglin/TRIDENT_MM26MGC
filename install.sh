#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARTER_DIR="$ROOT_DIR"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
INSTALL_DECORD="${INSTALL_DECORD:-0}"
TORCH_PACKAGES="${TORCH_PACKAGES:-torch torchvision}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-}"

echo "[INFO] Root: $ROOT_DIR"
echo "[INFO] Venv: $VENV_DIR"

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [[ "$INSTALL_TORCH" == "1" ]]; then
  if [[ -n "$PYTORCH_INDEX_URL" ]]; then
    python -m pip install --upgrade $TORCH_PACKAGES --index-url "$PYTORCH_INDEX_URL"
  else
    python -m pip install --upgrade $TORCH_PACKAGES
  fi
fi

python -m pip install --upgrade -r "$STARTER_DIR/requirements.txt"

if [[ "$INSTALL_DECORD" == "1" ]]; then
  if ! python -m pip install --upgrade decord; then
    echo "[WARN] decord installation failed. Video support can still fall back to imageio if available."
  fi
fi

if command -v ffmpeg >/dev/null 2>&1; then
  echo "[INFO] ffmpeg found: $(command -v ffmpeg)"
else
  echo "[WARN] ffmpeg is not installed. Video tasks may fail depending on your decoder backend."
fi

python - <<'PY'
import importlib

modules = [
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "imageio",
    "requests",
    "peft",
    "soundfile",
    "qwen_vl_utils",
]

for name in modules:
    try:
        mod = importlib.import_module(name)
        print(f"[OK] {name} {getattr(mod, '__version__', 'n/a')}")
    except Exception as exc:
        print(f"[WARN] {name} import failed: {exc}")
PY

echo "[INFO] Activation: source \"$VENV_DIR/bin/activate\""
