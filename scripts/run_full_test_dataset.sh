#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=/dev/null
. "$ROOT_DIR/.venv/bin/activate"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Enforce the operator decision: run at most 2 analysis plugins at a time on large datasets.
export STAT_HARNESS_MAX_WORKERS_ANALYSIS="2"
export STAT_HARNESS_CLI_PROGRESS="1"
# Integrity checks on a multi-GB SQLite file can dominate startup time; keep it off for the
# "run everything" harness script. Enable manually when you want it.
export STAT_HARNESS_STARTUP_INTEGRITY="off"
# Soft memory governor defaults (operator override via env).
export STAT_HARNESS_MEM_GOVERNOR_STAGES="${STAT_HARNESS_MEM_GOVERNOR_STAGES:-analysis}"
export STAT_HARNESS_MEM_GOVERNOR_MAX_USED_PCT="${STAT_HARNESS_MEM_GOVERNOR_MAX_USED_PCT:-50}"
export STAT_HARNESS_MEM_GOVERNOR_POLL_SECONDS="${STAT_HARNESS_MEM_GOVERNOR_POLL_SECONDS:-5}"
export STAT_HARNESS_MEM_GOVERNOR_LOG_SECONDS="${STAT_HARNESS_MEM_GOVERNOR_LOG_SECONDS:-30}"

DATASET_VERSION_ID="3246cc7cd7d57a317ddc05e80e6f6f5bfe7f50deb0ee7af8db50d04bae180e1a"

python scripts/run_loaded_dataset_full.py --dataset-version-id "$DATASET_VERSION_ID" --plugin-set full --run-seed 123
