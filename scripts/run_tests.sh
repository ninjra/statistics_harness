#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
  PYTHON_BIN="python3"
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
  fi
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
. "$VENV_DIR/bin/activate"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest -q "$@"
