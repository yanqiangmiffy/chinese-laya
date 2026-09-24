#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${MODEL_PATH:-}" ]]; then
  MODEL_PATH="$ROOT/runs/laya-typed-decisions-zh"
elif [[ "$MODEL_PATH" != /* ]]; then
  MODEL_PATH="$ROOT/$MODEL_PATH"
fi

if [[ ! -f "$MODEL_PATH/model.safetensors" ]]; then
  echo "Checkpoint not found at '$MODEL_PATH'. Set MODEL_PATH to the training output directory." >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not on PATH. Initialize Conda, then run this script again." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base
export MODEL_PATH
export MODEL_NAME="${MODEL_NAME:-laya-typed-decisions-zh}"
export DEVICE="${DEVICE:-cuda}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"
exec python demo-app/run-backend.py
