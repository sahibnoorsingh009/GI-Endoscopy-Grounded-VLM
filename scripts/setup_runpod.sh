#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/pip-cache}"

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"
cd "$REPO_DIR"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py

echo "RunPod setup complete. Start with: python app.py"
