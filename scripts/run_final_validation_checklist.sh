#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=/dev/null
. "$ROOT_DIR/.venv/bin/activate"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

DATASET_VERSION_ID="${1:-3246cc7cd7d57a317ddc05e80e6f6f5bfe7f50deb0ee7af8db50d04bae180e1a}"
RUN_SEED="${2:-1337}"
INTERVAL_SECONDS="${3:-30}"
D_STATE_STREAK_THRESHOLD="${STAT_HARNESS_D_STATE_STREAK_THRESHOLD:-8}"
PYTEST_D_STATE_STREAK_THRESHOLD="${STAT_HARNESS_PYTEST_D_STATE_STREAK_THRESHOLD:-8}"
CHECK_ID="final_validation_$(date -u +%Y%m%dT%H%M%SZ)"
CHECK_DIR="$ROOT_DIR/appdata/final_validation/$CHECK_ID"
mkdir -p "$CHECK_DIR"

echo "CHECK_ID=$CHECK_ID"
echo "CHECK_DIR=$CHECK_DIR"
echo "DATASET_VERSION_ID=$DATASET_VERSION_ID"
echo "RUN_SEED=$RUN_SEED"
echo "INTERVAL_SECONDS=$INTERVAL_SECONDS"
echo "PYTEST_D_STATE_STREAK_THRESHOLD=$PYTEST_D_STATE_STREAK_THRESHOLD"

echo "STEP=verify_docs_and_plugin_matrices"
verify_steps=(
  "scripts/run_repo_improvements_pipeline.py --verify"
  "scripts/binding_implementation_matrix.py --verify"
  "scripts/docs_coverage_matrix.py --verify"
  "scripts/plugin_data_access_matrix.py --verify"
  "scripts/plugins_functionality_matrix.py --verify"
  "scripts/sql_assist_adoption_matrix.py --verify"
  "scripts/redteam_ids_matrix.py --verify"
)
verify_rc=0
verify_fail_step=""
verify_pass_once=0
for verify_cmd in "${verify_steps[@]}"; do
  echo "VERIFY_STEP=$verify_cmd" | tee -a "$CHECK_DIR/verify_docs_and_plugin_matrices.log"
  if ./.venv/bin/python $verify_cmd >> "$CHECK_DIR/verify_docs_and_plugin_matrices.log" 2>&1; then
    :
  else
    verify_rc=$?
    verify_fail_step="$verify_cmd"
    break
  fi
done
if [[ "$verify_rc" -eq 0 ]]; then
  verify_pass_once=1
fi
if [[ "$verify_pass_once" -ne 1 && "${STAT_HARNESS_FINAL_REFRESH_ON_VERIFY_FAIL:-0}" == "1" ]]; then
  echo "VERIFY_RETRY=refresh_all_matrices" | tee -a "$CHECK_DIR/verify_docs_and_plugin_matrices.log"
  bash scripts/refresh_all_matrices.sh >> "$CHECK_DIR/verify_docs_and_plugin_matrices.log" 2>&1
  verify_rc=0
  verify_fail_step=""
  for verify_cmd in "${verify_steps[@]}"; do
    echo "VERIFY_STEP_RETRY=$verify_cmd" | tee -a "$CHECK_DIR/verify_docs_and_plugin_matrices.log"
    if ./.venv/bin/python $verify_cmd >> "$CHECK_DIR/verify_docs_and_plugin_matrices.log" 2>&1; then
      :
    else
      verify_rc=$?
      verify_fail_step="$verify_cmd"
      break
    fi
  done
fi
if [[ "$verify_rc" -ne 0 ]]; then
  echo "VERIFY_FAILED_STEP=$verify_fail_step"
  echo "VERIFY_LOG=$CHECK_DIR/verify_docs_and_plugin_matrices.log"
  tail -n 200 "$CHECK_DIR/verify_docs_and_plugin_matrices.log" || true
  exit "$verify_rc"
fi

echo "STEP=pytest_q"
pytest_log="$CHECK_DIR/pytest_q.log"
pytest_console="$CHECK_DIR/pytest_q_console.log"
: > "$pytest_log"
: > "$pytest_console"
PYTHONUNBUFFERED=1 ./.venv/bin/python -m pytest -q > "$pytest_log" 2>&1 &
pytest_pid=$!
pytest_tick=0
pytest_d_streak=0
pytest_hung=0
while kill -0 "$pytest_pid" >/dev/null 2>&1; do
  pytest_tick=$((pytest_tick + 1))
  pytest_state="unknown"
  if [[ -r "/proc/$pytest_pid/status" ]]; then
    pytest_state="$(awk '/^State:/{print $2}' "/proc/$pytest_pid/status" || true)"
  fi
  if [[ "$pytest_state" == "D" ]]; then
    pytest_d_streak=$((pytest_d_streak + 1))
  else
    pytest_d_streak=0
  fi
  msg="pytest -q running... tick=$pytest_tick pid=$pytest_pid state=${pytest_state:-unknown} d_state_streak=$pytest_d_streak (log=$pytest_log)"
  echo "$msg" | tee -a "$pytest_console"
  if [[ "$pytest_d_streak" -ge "$PYTEST_D_STATE_STREAK_THRESHOLD" ]]; then
    echo "pytest watchdog killing pid=$pytest_pid due to sustained D-state" | tee -a "$pytest_console"
    kill -9 "$pytest_pid" || true
    pytest_hung=1
    break
  fi
  sleep "$INTERVAL_SECONDS"
done
pytest_rc=0
wait "$pytest_pid" || pytest_rc=$?
echo "pytest -q exit_code=$pytest_rc" | tee -a "$pytest_console"
if [[ "$pytest_hung" -ne 0 || "$pytest_rc" -ne 0 ]]; then
  echo "pytest -q failed; tailing log:" | tee -a "$pytest_console"
  tail -n 250 "$pytest_log" | tee -a "$pytest_console" || true
  exit 1
fi

echo "STEP=start_full_loaded_dataset_bg"
START_OUT="$(bash scripts/start_full_loaded_dataset_bg.sh "$DATASET_VERSION_ID" "$RUN_SEED")"
printf '%s\n' "$START_OUT" | tee "$CHECK_DIR/start_full_loaded_dataset_bg.log"
RUN_ID="$(printf '%s\n' "$START_OUT" | awk -F= '/^RUN_ID=/{print $2}' | tail -n 1)"
if [[ -z "$RUN_ID" ]]; then
  echo "FINAL_VALIDATION_OK=false reason=missing_run_id"
  exit 2
fi
echo "$RUN_ID" > "$CHECK_DIR/run_id.txt"

echo "STEP=watch_run_until_done RUN_ID=$RUN_ID"
d_state_streak=0
hang_aborted=0
while true; do
  status_out="$(python scripts/run_run_status.py --run-id "$RUN_ID" || true)"
  printf '%s\n' "$status_out" | tee -a "$CHECK_DIR/run_status.log"
  run_status="$(
    printf '%s\n' "$status_out" | awk '
      /^run_id=/ {
        for (i = 1; i <= NF; i++) {
          if ($i ~ /^status=/) {
            split($i, a, "=");
            print a[2];
            exit;
          }
        }
      }
    '
  )"

  if [[ "$run_status" == "running" ]]; then
    pid="$(
      python -c "import json; from pathlib import Path; p=Path('appdata/runs/$RUN_ID/journal.json'); print((json.loads(p.read_text(encoding='utf-8')).get('pid') if p.exists() else '') or '')"
    )"
    if [[ -n "$pid" && -r "/proc/$pid/status" ]]; then
      state="$(awk '/^State:/{print $2}' "/proc/$pid/status" || true)"
      if [[ "$state" == "D" ]]; then
        d_state_streak=$((d_state_streak + 1))
      else
        d_state_streak=0
      fi
      echo "WATCH pid=$pid state=${state:-unknown} d_state_streak=$d_state_streak" | tee -a "$CHECK_DIR/run_watch.log"
      if [[ "$d_state_streak" -ge "$D_STATE_STREAK_THRESHOLD" ]]; then
        echo "WATCH action=kill reason=kernel_d_state_streak_exceeded pid=$pid threshold=$D_STATE_STREAK_THRESHOLD" | tee -a "$CHECK_DIR/run_watch.log"
        kill -9 "$pid" || true
        hang_aborted=1
        break
      fi
    fi
    sleep "$INTERVAL_SECONDS"
    continue
  fi
  break
done

echo "STEP=show_actionable_results RUN_ID=$RUN_ID"
python scripts/show_actionable_results.py --run-id "$RUN_ID" | tee "$CHECK_DIR/show_actionable_results.log"

echo "STEP=build_summary"
python scripts/build_final_validation_summary.py --run-id "$RUN_ID" --out "$CHECK_DIR/summary.json" | tee "$CHECK_DIR/summary_console.log"

run_status_final="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); print(str(d.get('run_status') or ''))" "$CHECK_DIR/summary.json")"
skip_count="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); c=d.get('plugin_status_counts') or {}; print(int(c.get('skipped',0)))" "$CHECK_DIR/summary.json")"
degraded_count="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); c=d.get('plugin_status_counts') or {}; print(int(c.get('degraded',0)))" "$CHECK_DIR/summary.json")"
error_count="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); c=d.get('plugin_status_counts') or {}; print(int(c.get('error',0))+int(c.get('aborted',0)))" "$CHECK_DIR/summary.json")"
missing_count="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); m=d.get('missing_plugin_results'); print(int(m or 0))" "$CHECK_DIR/summary.json")"
analysis_ok_without_findings_count="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); print(int(d.get('analysis_ok_without_findings_count') or 0))" "$CHECK_DIR/summary.json")"
sql_assist_required_failure_count="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); print(int(d.get('sql_assist_required_failure_count') or 0))" "$CHECK_DIR/summary.json")"

ok=true
if [[ "$hang_aborted" -ne 0 ]]; then ok=false; fi
if [[ "$run_status_final" != "completed" ]]; then ok=false; fi
if [[ "$skip_count" -ne 0 ]]; then ok=false; fi
if [[ "$degraded_count" -ne 0 ]]; then ok=false; fi
if [[ "$error_count" -ne 0 ]]; then ok=false; fi
if [[ "$missing_count" -ne 0 ]]; then ok=false; fi
if [[ "$analysis_ok_without_findings_count" -ne 0 ]]; then ok=false; fi
if [[ "$sql_assist_required_failure_count" -ne 0 ]]; then ok=false; fi

echo "FINAL_VALIDATION_OK=$ok"
echo "FINAL_VALIDATION_RUN_ID=$RUN_ID"
echo "FINAL_VALIDATION_SUMMARY=$CHECK_DIR/summary.json"

if [[ "$ok" != "true" ]]; then
  exit 1
fi
