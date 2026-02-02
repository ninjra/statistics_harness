#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export STAT_HARNESS_SAFE_RENAME=1
export PYTHONPATH="$ROOT_DIR/tools${PYTHONPATH:+:$PYTHONPATH}"

pip install -e ".[dev]" --no-build-isolation
