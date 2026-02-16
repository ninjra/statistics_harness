#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=/dev/null
. "$ROOT_DIR/.venv/bin/activate"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

DATASET_VERSION_ID="${1:-3246cc7cd7d57a317ddc05e80e6f6f5bfe7f50deb0ee7af8db50d04bae180e1a}"
RUN_SEED="${2:-123}"

# Default: 2 analysis workers (user-requested). Override by exporting STAT_HARNESS_MAX_WORKERS_ANALYSIS.
export STAT_HARNESS_MAX_WORKERS_ANALYSIS="${STAT_HARNESS_MAX_WORKERS_ANALYSIS:-2}"
export STAT_HARNESS_CLI_PROGRESS="${STAT_HARNESS_CLI_PROGRESS:-1}"
export STAT_HARNESS_REUSE_CACHE="${STAT_HARNESS_REUSE_CACHE:-1}"
export STAT_HARNESS_STARTUP_INTEGRITY="${STAT_HARNESS_STARTUP_INTEGRITY:-off}"
export STAT_HARNESS_DISCOVERY_TOP_N="${STAT_HARNESS_DISCOVERY_TOP_N:-12}"
export STAT_HARNESS_RECOMMENDATION_MIN_RELEVANCE="${STAT_HARNESS_RECOMMENDATION_MIN_RELEVANCE:-0.0}"
export STAT_HARNESS_MAX_OBVIOUSNESS="${STAT_HARNESS_MAX_OBVIOUSNESS:-0.74}"
export STAT_HARNESS_ALLOW_ACTION_TYPES="${STAT_HARNESS_ALLOW_ACTION_TYPES:-batch_input,batch_group_candidate,batch_or_cache,batch_input_refactor,add_server,tune_schedule,unblock_dependency_chain,reduce_transition_gap}"
# Operator exclusion list: non-adjustable processes should not appear as recommendations.
export STAT_HARNESS_EXCLUDE_PROCESSES="${STAT_HARNESS_EXCLUDE_PROCESSES:-losextchld,losloadcld,jbcreateje,jboachild,jbvalcdblk,jbinvoice,postwkfl,qemail,jbpreproof,rdimpairje}"
# Soft memory governor defaults (operator override via env):
# - if system RAM usage exceeds this, the pipeline will delay starting additional analysis plugins
#   to avoid multi-plugin RAM spikes.
export STAT_HARNESS_MEM_GOVERNOR_STAGES="${STAT_HARNESS_MEM_GOVERNOR_STAGES:-analysis}"
export STAT_HARNESS_MEM_GOVERNOR_MAX_USED_PCT="${STAT_HARNESS_MEM_GOVERNOR_MAX_USED_PCT:-50}"
export STAT_HARNESS_MEM_GOVERNOR_POLL_SECONDS="${STAT_HARNESS_MEM_GOVERNOR_POLL_SECONDS:-5}"
export STAT_HARNESS_MEM_GOVERNOR_LOG_SECONDS="${STAT_HARNESS_MEM_GOVERNOR_LOG_SECONDS:-30}"
# Hard cap each plugin subprocess address space to prevent host-level OOM kills.
export STAT_HARNESS_PLUGIN_RLIMIT_AS_MB="${STAT_HARNESS_PLUGIN_RLIMIT_AS_MB:-4096}"

LOG_PATH="$ROOT_DIR/appdata/full_run_${DATASET_VERSION_ID:0:8}_$(date -u +%Y%m%dT%H%M%SZ).log"

nohup python -u scripts/run_loaded_dataset_full.py --dataset-version-id "$DATASET_VERSION_ID" --plugin-set full --run-seed "$RUN_SEED" >"$LOG_PATH" 2>&1 &
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
  if [[ "${STAT_HARNESS_WATCH_REPAIR:-0}" == "1" ]]; then
    python scripts/repair_stale_running_runs.py >/dev/null 2>&1 || true
  fi
  RUN_ID="$(
  python -c "import sqlite3; from pathlib import Path; db=Path('appdata/state.sqlite'); con=sqlite3.connect(db); con.row_factory=sqlite3.Row; row=con.execute(\"select run_id from runs where status='running' and dataset_version_id=? order by created_at desc limit 1\", ('${DATASET_VERSION_ID}',)).fetchone(); print(row['run_id'] if row else ''); con.close()"
  )"
  if [[ -n "$RUN_ID" ]]; then
    break
  fi
  sleep 1
done

echo "PID=$PID"
echo "RUN_ID=$RUN_ID"
echo "LOG=$LOG_PATH"
