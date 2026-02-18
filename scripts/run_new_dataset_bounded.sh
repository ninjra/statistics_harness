#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=/dev/null
. "$ROOT_DIR/.venv/bin/activate"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export STAT_HARNESS_STARTUP_INTEGRITY="${STAT_HARNESS_STARTUP_INTEGRITY:-off}"
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
export STAT_HARNESS_CLI_PROGRESS="${STAT_HARNESS_CLI_PROGRESS:-1}"
export STAT_HARNESS_REUSE_CACHE="${STAT_HARNESS_REUSE_CACHE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export STAT_HARNESS_PROCESS_NICE_LEVEL="${STAT_HARNESS_PROCESS_NICE_LEVEL:-$default_nice_level}"

DATASET_VERSION_ID="${1:-de7c1da5a4ea6e8c684872d7857bb608492f63a9c7e0b7ca014fa0f093a88e66}"
RUN_SEED="${2:-123}"

PLANNER_ALLOW="analysis_actionable_ops_levers_v1,analysis_queue_delay_decomposition,analysis_busy_period_segmentation_v2,analysis_close_cycle_contention,analysis_close_cycle_duration_shift,analysis_close_cycle_capacity_model,analysis_close_cycle_capacity_impact,analysis_close_cycle_window_resolver,analysis_close_cycle_revenue_compression,analysis_close_cycle_uplift,analysis_capacity_scaling,analysis_waterfall_summary_v2,analysis_traceability_manifest_v2,analysis_recommendation_dedupe_v2,analysis_issue_cards_v2,analysis_ideaspace_normative_gap,analysis_ideaspace_energy_ebm_v1,analysis_ideaspace_action_planner,analysis_ebm_action_verifier_v1,analysis_concurrency_reconstruction,analysis_chain_makespan"

nice -n "${STAT_HARNESS_PROCESS_NICE_LEVEL}" python scripts/run_loaded_dataset_full.py --dataset-version-id "$DATASET_VERSION_ID" --plugin-set auto --run-seed "$RUN_SEED" --planner-allow "$PLANNER_ALLOW"
