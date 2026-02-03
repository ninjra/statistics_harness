#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export STAT_HARNESS_SAFE_RENAME=1
export PYTHONPATH="$ROOT_DIR/tools${PYTHONPATH:+:$PYTHONPATH}"

VENV_DIR="$ROOT_DIR/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
  PYTHON_BIN="python3"
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
  fi
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

mkdir -p "$ROOT_DIR/.pip-tmp"
export TMPDIR="$ROOT_DIR/.pip-tmp"

# shellcheck source=/dev/null
. "$VENV_DIR/bin/activate"
python -m pip install -e ".[dev]" --no-build-isolation
