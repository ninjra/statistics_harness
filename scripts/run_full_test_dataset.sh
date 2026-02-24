#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=/dev/null
. "$ROOT_DIR/.venv/bin/activate"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

RESOURCE_PROFILE="${STAT_HARNESS_RESOURCE_PROFILE:-interactive}"
if [[ "$RESOURCE_PROFILE" == "interactive" ]]; then
  default_workers_analysis="1"
  default_mem_governor_max_used_pct="20"
  default_mem_governor_min_available_mb="12288"
  default_nice_level="15"
elif [[ "$RESOURCE_PROFILE" == "respectful" ]]; then
  default_workers_analysis="1"
  default_mem_governor_max_used_pct="30"
  default_mem_governor_min_available_mb="4096"
  default_nice_level="12"
elif [[ "$RESOURCE_PROFILE" == "balanced" ]]; then
  default_workers_analysis="2"
  default_mem_governor_max_used_pct="40"
  default_mem_governor_min_available_mb="3072"
  default_nice_level="7"
else
  default_workers_analysis="2"
  default_mem_governor_max_used_pct="55"
  default_mem_governor_min_available_mb="2048"
  default_nice_level="0"
fi
export STAT_HARNESS_RESOURCE_PROFILE="$RESOURCE_PROFILE"
export STAT_HARNESS_MAX_WORKERS_ANALYSIS="${STAT_HARNESS_MAX_WORKERS_ANALYSIS:-$default_workers_analysis}"
export STAT_HARNESS_MAX_WORKERS_TRANSFORM="${STAT_HARNESS_MAX_WORKERS_TRANSFORM:-1}"
export STAT_HARNESS_CLI_PROGRESS="1"
export STAT_HARNESS_REUSE_CACHE="${STAT_HARNESS_REUSE_CACHE:-1}"
# Integrity checks on a multi-GB SQLite file can dominate startup time; keep it off for the
# "run everything" harness script. Enable manually when you want it.
export STAT_HARNESS_STARTUP_INTEGRITY="off"
# Soft memory governor defaults (operator override via env).
export STAT_HARNESS_MEM_GOVERNOR_STAGES="${STAT_HARNESS_MEM_GOVERNOR_STAGES:-analysis}"
export STAT_HARNESS_MEM_GOVERNOR_MAX_USED_PCT="${STAT_HARNESS_MEM_GOVERNOR_MAX_USED_PCT:-$default_mem_governor_max_used_pct}"
export STAT_HARNESS_MEM_GOVERNOR_MIN_AVAILABLE_MB="${STAT_HARNESS_MEM_GOVERNOR_MIN_AVAILABLE_MB:-$default_mem_governor_min_available_mb}"
export STAT_HARNESS_MEM_GOVERNOR_POLL_SECONDS="${STAT_HARNESS_MEM_GOVERNOR_POLL_SECONDS:-3}"
export STAT_HARNESS_MEM_GOVERNOR_LOG_SECONDS="${STAT_HARNESS_MEM_GOVERNOR_LOG_SECONDS:-30}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export STAT_HARNESS_PROCESS_NICE_LEVEL="${STAT_HARNESS_PROCESS_NICE_LEVEL:-$default_nice_level}"

DATASET_VERSION_ID="3246cc7cd7d57a317ddc05e80e6f6f5bfe7f50deb0ee7af8db50d04bae180e1a"

nice -n "${STAT_HARNESS_PROCESS_NICE_LEVEL}" python scripts/run_loaded_dataset_full.py --dataset-version-id "$DATASET_VERSION_ID" --plugin-set full --run-seed 123
