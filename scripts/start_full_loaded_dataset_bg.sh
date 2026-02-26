#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=/dev/null
. "$ROOT_DIR/.venv/bin/activate"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

DATASET_VERSION_ID="${1:-3246cc7cd7d57a317ddc05e80e6f6f5bfe7f50deb0ee7af8db50d04bae180e1a}"
RUN_SEED="${2:-123}"
KNOWN_ISSUES_MODE="${3:-${STAT_HARNESS_KNOWN_ISSUES_MODE:-on}}"
ORCHESTRATOR_MODE="${4:-${STAT_HARNESS_ORCHESTRATOR_MODE:-two_lane_strict}}"
GOLDEN_MODE="${5:-${STAT_HARNESS_GOLDEN_MODE:-strict}}"
RUN_ID="${STAT_HARNESS_RUN_ID:-full_loaded_${DATASET_VERSION_ID:0:8}_$(date -u +%Y%m%dT%H%M%SZ)}"

case "$KNOWN_ISSUES_MODE" in
  on|off) ;;
  *)
    echo "ERROR: invalid KNOWN_ISSUES_MODE=$KNOWN_ISSUES_MODE (expected on|off)"
    exit 2
    ;;
esac
case "$ORCHESTRATOR_MODE" in
  legacy|two_lane_strict) ;;
  *)
    echo "ERROR: invalid ORCHESTRATOR_MODE=$ORCHESTRATOR_MODE (expected legacy|two_lane_strict)"
    exit 2
    ;;
esac
case "$GOLDEN_MODE" in
  off|default|strict) ;;
  *)
    echo "ERROR: invalid GOLDEN_MODE=$GOLDEN_MODE (expected off|default|strict)"
    exit 2
    ;;
esac

RESOURCE_PROFILE="${STAT_HARNESS_RESOURCE_PROFILE:-interactive}"
case "$RESOURCE_PROFILE" in
  interactive|respectful|balanced|performance) ;;
  *)
    echo "ERROR: invalid STAT_HARNESS_RESOURCE_PROFILE=$RESOURCE_PROFILE (expected interactive|respectful|balanced|performance)"
    exit 2
    ;;
esac

default_taskset_cpulist=""
if [[ "$RESOURCE_PROFILE" == "interactive" ]]; then
  default_workers_analysis="1"
  default_mem_governor_max_used_pct="20"
  default_mem_governor_min_available_mb="12288"
  default_plugin_rlimit_mb="2048"
  default_nice_level="15"
  default_ionice_class="3"
  default_ionice_prio="7"
elif [[ "$RESOURCE_PROFILE" == "respectful" ]]; then
  default_workers_analysis="1"
  default_mem_governor_max_used_pct="30"
  default_mem_governor_min_available_mb="4096"
  default_plugin_rlimit_mb="2560"
  default_nice_level="12"
  default_ionice_class="2"
  default_ionice_prio="7"
elif [[ "$RESOURCE_PROFILE" == "balanced" ]]; then
  default_workers_analysis="2"
  default_mem_governor_max_used_pct="40"
  default_mem_governor_min_available_mb="3072"
  default_plugin_rlimit_mb="3072"
  default_nice_level="7"
  default_ionice_class="2"
  default_ionice_prio="7"
else
  default_workers_analysis="2"
  default_mem_governor_max_used_pct="55"
  default_mem_governor_min_available_mb="2048"
  default_plugin_rlimit_mb="4096"
  default_nice_level="0"
  default_ionice_class="2"
  default_ionice_prio="7"
fi
if [[ "$RESOURCE_PROFILE" == "interactive" ]] && command -v nproc >/dev/null 2>&1; then
  cpu_count="$(nproc 2>/dev/null || echo 1)"
  if [[ "$cpu_count" =~ ^[0-9]+$ ]] && (( cpu_count > 1 )); then
    default_taskset_cpulist="$((cpu_count - 1))"
  fi
fi

export STAT_HARNESS_RESOURCE_PROFILE="$RESOURCE_PROFILE"
export STAT_HARNESS_MAX_WORKERS_ANALYSIS="${STAT_HARNESS_MAX_WORKERS_ANALYSIS:-$default_workers_analysis}"
export STAT_HARNESS_MAX_WORKERS_TRANSFORM="${STAT_HARNESS_MAX_WORKERS_TRANSFORM:-1}"
export STAT_HARNESS_CLI_PROGRESS="${STAT_HARNESS_CLI_PROGRESS:-1}"
export STAT_HARNESS_REUSE_CACHE="${STAT_HARNESS_REUSE_CACHE:-1}"
export STAT_HARNESS_STARTUP_INTEGRITY="${STAT_HARNESS_STARTUP_INTEGRITY:-off}"
export STAT_HARNESS_DISCOVERY_TOP_N="${STAT_HARNESS_DISCOVERY_TOP_N:-12}"
export STAT_HARNESS_RECOMMENDATION_MIN_RELEVANCE="${STAT_HARNESS_RECOMMENDATION_MIN_RELEVANCE:-0.0}"
export STAT_HARNESS_MAX_OBVIOUSNESS="${STAT_HARNESS_MAX_OBVIOUSNESS:-0.74}"
export STAT_HARNESS_ALLOW_ACTION_TYPES="${STAT_HARNESS_ALLOW_ACTION_TYPES:-}"
# Operator exclusion list: default empty here; runner resolves dataset-scoped
# exclusions from explicit input + historical settings for that dataset.
export STAT_HARNESS_EXCLUDE_PROCESSES="${STAT_HARNESS_EXCLUDE_PROCESSES:-}"
# Soft memory governor defaults (operator override via env):
# - if system RAM usage exceeds this, the pipeline will delay starting additional analysis plugins
#   to avoid multi-plugin RAM spikes.
export STAT_HARNESS_MEM_GOVERNOR_STAGES="${STAT_HARNESS_MEM_GOVERNOR_STAGES:-analysis}"
export STAT_HARNESS_MEM_GOVERNOR_MAX_USED_PCT="${STAT_HARNESS_MEM_GOVERNOR_MAX_USED_PCT:-$default_mem_governor_max_used_pct}"
export STAT_HARNESS_MEM_GOVERNOR_MIN_AVAILABLE_MB="${STAT_HARNESS_MEM_GOVERNOR_MIN_AVAILABLE_MB:-$default_mem_governor_min_available_mb}"
export STAT_HARNESS_MEM_GOVERNOR_POLL_SECONDS="${STAT_HARNESS_MEM_GOVERNOR_POLL_SECONDS:-3}"
export STAT_HARNESS_MEM_GOVERNOR_LOG_SECONDS="${STAT_HARNESS_MEM_GOVERNOR_LOG_SECONDS:-30}"
# Hard cap each plugin subprocess address space to prevent host-level OOM kills.
export STAT_HARNESS_PLUGIN_RLIMIT_AS_MB="${STAT_HARNESS_PLUGIN_RLIMIT_AS_MB:-$default_plugin_rlimit_mb}"
export STAT_HARNESS_KNOWN_ISSUES_MODE="$KNOWN_ISSUES_MODE"
export STAT_HARNESS_ORCHESTRATOR_MODE="$ORCHESTRATOR_MODE"
export STAT_HARNESS_GOLDEN_MODE="$GOLDEN_MODE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export STAT_HARNESS_PROCESS_NICE_LEVEL="${STAT_HARNESS_PROCESS_NICE_LEVEL:-$default_nice_level}"
export STAT_HARNESS_PROCESS_IONICE_CLASS="${STAT_HARNESS_PROCESS_IONICE_CLASS:-$default_ionice_class}"
export STAT_HARNESS_PROCESS_IONICE_PRIO="${STAT_HARNESS_PROCESS_IONICE_PRIO:-$default_ionice_prio}"
export STAT_HARNESS_PROCESS_TASKSET_CPULIST="${STAT_HARNESS_PROCESS_TASKSET_CPULIST:-$default_taskset_cpulist}"
export STAT_HARNESS_ROUTE_ENABLE="${STAT_HARNESS_ROUTE_ENABLE:-0}"
export STAT_HARNESS_ROUTE_MAX_DEPTH="${STAT_HARNESS_ROUTE_MAX_DEPTH:-3}"
export STAT_HARNESS_ROUTE_BEAM_WIDTH="${STAT_HARNESS_ROUTE_BEAM_WIDTH:-6}"
export STAT_HARNESS_ROUTE_MIN_DELTA_ENERGY="${STAT_HARNESS_ROUTE_MIN_DELTA_ENERGY:-0.0}"
export STAT_HARNESS_ROUTE_MIN_CONFIDENCE="${STAT_HARNESS_ROUTE_MIN_CONFIDENCE:-0.0}"
export STAT_HARNESS_ROUTE_ALLOW_CROSS_TARGET_STEPS="${STAT_HARNESS_ROUTE_ALLOW_CROSS_TARGET_STEPS:-0}"
export STAT_HARNESS_ROUTE_STOP_ENERGY_THRESHOLD="${STAT_HARNESS_ROUTE_STOP_ENERGY_THRESHOLD:-1.0}"
export STAT_HARNESS_ROUTE_CANDIDATE_LIMIT="${STAT_HARNESS_ROUTE_CANDIDATE_LIMIT:-30}"
export STAT_HARNESS_ROUTE_TIME_BUDGET_MS="${STAT_HARNESS_ROUTE_TIME_BUDGET_MS:-2000}"
export STAT_HARNESS_ROUTE_DISALLOWED_LEVER_IDS="${STAT_HARNESS_ROUTE_DISALLOWED_LEVER_IDS:-}"
export STAT_HARNESS_ROUTE_DISALLOWED_ACTION_TYPES="${STAT_HARNESS_ROUTE_DISALLOWED_ACTION_TYPES:-}"

LOG_PATH="$ROOT_DIR/appdata/full_run_${DATASET_VERSION_ID:0:8}_$(date -u +%Y%m%dT%H%M%SZ).log"
route_args=()
if [[ "$STAT_HARNESS_ROUTE_ENABLE" == "1" || "$STAT_HARNESS_ROUTE_ENABLE" == "true" || "$STAT_HARNESS_ROUTE_ENABLE" == "yes" || "$STAT_HARNESS_ROUTE_ENABLE" == "on" ]]; then
  route_args+=(
    --route-enable
    --route-max-depth "$STAT_HARNESS_ROUTE_MAX_DEPTH"
    --route-beam-width "$STAT_HARNESS_ROUTE_BEAM_WIDTH"
    --route-min-delta-energy "$STAT_HARNESS_ROUTE_MIN_DELTA_ENERGY"
    --route-min-confidence "$STAT_HARNESS_ROUTE_MIN_CONFIDENCE"
    --route-stop-energy-threshold "$STAT_HARNESS_ROUTE_STOP_ENERGY_THRESHOLD"
    --route-candidate-limit "$STAT_HARNESS_ROUTE_CANDIDATE_LIMIT"
    --route-time-budget-ms "$STAT_HARNESS_ROUTE_TIME_BUDGET_MS"
  )
  if [[ "$STAT_HARNESS_ROUTE_ALLOW_CROSS_TARGET_STEPS" == "1" || "$STAT_HARNESS_ROUTE_ALLOW_CROSS_TARGET_STEPS" == "true" || "$STAT_HARNESS_ROUTE_ALLOW_CROSS_TARGET_STEPS" == "yes" || "$STAT_HARNESS_ROUTE_ALLOW_CROSS_TARGET_STEPS" == "on" ]]; then
    route_args+=(--route-allow-cross-target-steps)
  fi
  if [[ -n "$STAT_HARNESS_ROUTE_DISALLOWED_LEVER_IDS" ]]; then
    route_args+=(--route-disallowed-lever-ids "$STAT_HARNESS_ROUTE_DISALLOWED_LEVER_IDS")
  fi
  if [[ -n "$STAT_HARNESS_ROUTE_DISALLOWED_ACTION_TYPES" ]]; then
    route_args+=(--route-disallowed-action-types "$STAT_HARNESS_ROUTE_DISALLOWED_ACTION_TYPES")
  fi
fi

taskset_prefix=()
if [[ -n "$STAT_HARNESS_PROCESS_TASKSET_CPULIST" ]] && command -v taskset >/dev/null 2>&1; then
  taskset_prefix=(taskset -c "$STAT_HARNESS_PROCESS_TASKSET_CPULIST")
fi
if command -v ionice >/dev/null 2>&1; then
  nohup "${taskset_prefix[@]}" ionice -c "${STAT_HARNESS_PROCESS_IONICE_CLASS}" -n "${STAT_HARNESS_PROCESS_IONICE_PRIO}" nice -n "${STAT_HARNESS_PROCESS_NICE_LEVEL}" python -u scripts/run_loaded_dataset_full.py --dataset-version-id "$DATASET_VERSION_ID" --plugin-set full --run-seed "$RUN_SEED" --run-id "$RUN_ID" --known-issues-mode "$KNOWN_ISSUES_MODE" --orchestrator-mode "$ORCHESTRATOR_MODE" "${route_args[@]}" >"$LOG_PATH" 2>&1 &
else
  nohup "${taskset_prefix[@]}" nice -n "${STAT_HARNESS_PROCESS_NICE_LEVEL}" python -u scripts/run_loaded_dataset_full.py --dataset-version-id "$DATASET_VERSION_ID" --plugin-set full --run-seed "$RUN_SEED" --run-id "$RUN_ID" --known-issues-mode "$KNOWN_ISSUES_MODE" --orchestrator-mode "$ORCHESTRATOR_MODE" "${route_args[@]}" >"$LOG_PATH" 2>&1 &
fi
PID="$!"

sleep 0.5
if ! kill -0 "$PID" 2>/dev/null; then
  echo "ERROR: runner exited early (pid=$PID)"
  if [[ -f "$LOG_PATH" ]]; then
    tail -n 100 "$LOG_PATH" || true
  fi
  exit 1
fi

for _ in {1..60}; do
  if [[ "${STAT_HARNESS_WATCH_REPAIR:-0}" == "1" ]]; then
    python scripts/repair_stale_running_runs.py >/dev/null 2>&1 || true
  fi
  if python -c "import sqlite3; from pathlib import Path; db=Path('appdata/state.sqlite'); con=sqlite3.connect(db); con.row_factory=sqlite3.Row; row=con.execute(\"select run_id from runs where run_id=?\", ('${RUN_ID}',)).fetchone(); print('1' if row else '0'); con.close()" | grep -q "^1$"; then
    break
  fi
  sleep 1
done

if ! python -c "import sqlite3; from pathlib import Path; db=Path('appdata/state.sqlite'); con=sqlite3.connect(db); con.row_factory=sqlite3.Row; row=con.execute(\"select run_id from runs where run_id=?\", ('${RUN_ID}',)).fetchone(); print('1' if row else '0'); con.close()" | grep -q "^1$"; then
  echo "ERROR: run_id_not_registered run_id=$RUN_ID"
  if [[ -f "$LOG_PATH" ]]; then
    tail -n 100 "$LOG_PATH" || true
  fi
  exit 1
fi

echo "PID=$PID"
echo "RUN_ID=$RUN_ID"
echo "KNOWN_ISSUES_MODE=$KNOWN_ISSUES_MODE"
echo "ORCHESTRATOR_MODE=$ORCHESTRATOR_MODE"
echo "GOLDEN_MODE=$GOLDEN_MODE"
echo "RESOURCE_PROFILE=$RESOURCE_PROFILE"
echo "MAX_WORKERS_ANALYSIS=$STAT_HARNESS_MAX_WORKERS_ANALYSIS"
echo "MEM_GOVERNOR_MAX_USED_PCT=$STAT_HARNESS_MEM_GOVERNOR_MAX_USED_PCT"
echo "MEM_GOVERNOR_MIN_AVAILABLE_MB=$STAT_HARNESS_MEM_GOVERNOR_MIN_AVAILABLE_MB"
echo "PLUGIN_RLIMIT_AS_MB=$STAT_HARNESS_PLUGIN_RLIMIT_AS_MB"
echo "PROCESS_NICE_LEVEL=$STAT_HARNESS_PROCESS_NICE_LEVEL"
echo "PROCESS_IONICE_CLASS=$STAT_HARNESS_PROCESS_IONICE_CLASS"
echo "PROCESS_TASKSET_CPULIST=$STAT_HARNESS_PROCESS_TASKSET_CPULIST"
echo "ROUTE_ENABLE=$STAT_HARNESS_ROUTE_ENABLE"
echo "LOG=$LOG_PATH"
