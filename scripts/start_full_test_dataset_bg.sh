#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=/dev/null
. "$ROOT_DIR/.venv/bin/activate"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

RESOURCE_PROFILE="${STAT_HARNESS_RESOURCE_PROFILE:-respectful}"
if [[ "$RESOURCE_PROFILE" == "respectful" ]]; then
  default_workers_analysis="1"
  default_nice_level="10"
elif [[ "$RESOURCE_PROFILE" == "balanced" ]]; then
  default_workers_analysis="2"
  default_nice_level="5"
else
  default_workers_analysis="2"
  default_nice_level="0"
fi
export STAT_HARNESS_RESOURCE_PROFILE="$RESOURCE_PROFILE"
export STAT_HARNESS_MAX_WORKERS_ANALYSIS="${STAT_HARNESS_MAX_WORKERS_ANALYSIS:-$default_workers_analysis}"
export STAT_HARNESS_CLI_PROGRESS="1"
export STAT_HARNESS_STARTUP_INTEGRITY="off"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export STAT_HARNESS_PROCESS_NICE_LEVEL="${STAT_HARNESS_PROCESS_NICE_LEVEL:-$default_nice_level}"

export DATASET_VERSION_ID="3246cc7cd7d57a317ddc05e80e6f6f5bfe7f50deb0ee7af8db50d04bae180e1a"
LOG_PATH="$ROOT_DIR/appdata/full_run_${DATASET_VERSION_ID:0:8}_$(date -u +%Y%m%dT%H%M%SZ).log"

nohup nice -n "${STAT_HARNESS_PROCESS_NICE_LEVEL}" python -u scripts/run_loaded_dataset_full.py --dataset-version-id "$DATASET_VERSION_ID" --plugin-set full --run-seed 123 >"$LOG_PATH" 2>&1 &
PID="$!"

sleep 0.5
if ! kill -0 "$PID" 2>/dev/null; then
  echo "ERROR: runner exited early (pid=$PID)"
  if [[ -f "$LOG_PATH" ]]; then
    tail -n 100 "$LOG_PATH" || true
  fi
  exit 1
fi

RUN_ID=""
for _ in {1..60}; do
  RUN_ID="$(python -c "import os, sqlite3; from pathlib import Path; dvid=os.environ.get('DATASET_VERSION_ID',''); db=Path('appdata/state.sqlite'); con=sqlite3.connect(db); con.row_factory=sqlite3.Row; row=con.execute(\"select run_id from runs where status='running' and dataset_version_id=? order by created_at desc limit 1\", (dvid,)).fetchone(); print(row['run_id'] if row else ''); con.close()")"
  if [[ -n "$RUN_ID" ]]; then
    break
  fi
  sleep 1
done

echo "PID=$PID"
echo "RUN_ID=$RUN_ID"
echo "LOG=$LOG_PATH"
