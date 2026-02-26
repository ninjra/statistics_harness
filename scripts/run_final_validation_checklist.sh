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
KNOWN_ISSUES_MODE="${4:-${STAT_HARNESS_KNOWN_ISSUES_MODE:-on}}"
ORCHESTRATOR_MODE="${5:-${STAT_HARNESS_ORCHESTRATOR_MODE:-two_lane_strict}}"
GOLDEN_MODE="${6:-${STAT_HARNESS_GOLDEN_MODE:-strict}}"
RECALL_TOP_N="${STAT_HARNESS_REQUIRED_LANDMARK_RECALL_TOP_N:-30}"
REQUIRE_LANDMARK_RECALL="${STAT_HARNESS_REQUIRE_LANDMARK_RECALL:-1}"
D_STATE_STREAK_THRESHOLD="${STAT_HARNESS_D_STATE_STREAK_THRESHOLD:-8}"
PYTEST_D_STATE_STREAK_THRESHOLD="${STAT_HARNESS_PYTEST_D_STATE_STREAK_THRESHOLD:-8}"
AUTONOMOUS_NOVELTY_MIN="${STAT_HARNESS_AUTONOMOUS_NOVELTY_MIN:-1}"
AUTONOMOUS_NOVELTY_MAX_JACCARD="${STAT_HARNESS_AUTONOMOUS_NOVELTY_MAX_JACCARD:-0.95}"
STREAMING_POLICY_STRICT="${STAT_HARNESS_STREAMING_POLICY_STRICT:-0}"
REQUIRE_NOVELTY_REFERENCE="${STAT_HARNESS_REQUIRE_NOVELTY_REFERENCE:-1}"
CHECK_ID="final_validation_$(date -u +%Y%m%dT%H%M%SZ)"
CHECK_DIR="$ROOT_DIR/appdata/final_validation/$CHECK_ID"
mkdir -p "$CHECK_DIR"

if [[ "$KNOWN_ISSUES_MODE" == "on" ]]; then
  export STAT_HARNESS_INCLUDE_KNOWN_RECOMMENDATIONS="${STAT_HARNESS_INCLUDE_KNOWN_RECOMMENDATIONS:-1}"
fi

echo "CHECK_ID=$CHECK_ID"
echo "CHECK_DIR=$CHECK_DIR"
echo "DATASET_VERSION_ID=$DATASET_VERSION_ID"
echo "RUN_SEED=$RUN_SEED"
echo "INTERVAL_SECONDS=$INTERVAL_SECONDS"
echo "KNOWN_ISSUES_MODE=$KNOWN_ISSUES_MODE"
echo "ORCHESTRATOR_MODE=$ORCHESTRATOR_MODE"
echo "GOLDEN_MODE=$GOLDEN_MODE"
echo "RECALL_TOP_N=$RECALL_TOP_N"
echo "REQUIRE_LANDMARK_RECALL=$REQUIRE_LANDMARK_RECALL"
echo "PYTEST_D_STATE_STREAK_THRESHOLD=$PYTEST_D_STATE_STREAK_THRESHOLD"
echo "AUTONOMOUS_NOVELTY_MIN=$AUTONOMOUS_NOVELTY_MIN"
echo "AUTONOMOUS_NOVELTY_MAX_JACCARD=$AUTONOMOUS_NOVELTY_MAX_JACCARD"
echo "STREAMING_POLICY_STRICT=$STREAMING_POLICY_STRICT"

if [[ "$KNOWN_ISSUES_MODE" != "on" && "$KNOWN_ISSUES_MODE" != "off" ]]; then
  echo "KNOWN_ISSUES_MODE_INVALID=$KNOWN_ISSUES_MODE (expected on|off)"
  exit 2
fi
if [[ "$ORCHESTRATOR_MODE" != "legacy" && "$ORCHESTRATOR_MODE" != "two_lane_strict" ]]; then
  echo "ORCHESTRATOR_MODE_INVALID=$ORCHESTRATOR_MODE (expected legacy|two_lane_strict)"
  exit 2
fi
if [[ "$GOLDEN_MODE" != "off" && "$GOLDEN_MODE" != "default" && "$GOLDEN_MODE" != "strict" ]]; then
  echo "GOLDEN_MODE_INVALID=$GOLDEN_MODE (expected off|default|strict)"
  exit 2
fi

echo "STEP=verify_docs_and_plugin_matrices"
verify_steps=(
  "scripts/generate_codex_plugin_catalog.py --verify"
  "scripts/run_repo_improvements_pipeline.py --verify"
  "scripts/binding_implementation_matrix.py --verify"
  "scripts/docs_coverage_matrix.py --verify"
  "scripts/plugin_data_access_matrix.py --verify"
  "scripts/plugins_functionality_matrix.py --verify"
  "scripts/sql_assist_adoption_matrix.py --verify"
  "scripts/redteam_ids_matrix.py --verify"
  "scripts/build_plugin_class_actionability_matrix.py --verify"
  "scripts/generate_plugin_example_cards.py --verify"
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
START_OUT="$(bash scripts/start_full_loaded_dataset_bg.sh "$DATASET_VERSION_ID" "$RUN_SEED" "$KNOWN_ISSUES_MODE" "$ORCHESTRATOR_MODE" "$GOLDEN_MODE")"
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
show_cmd=(python scripts/show_actionable_results.py --run-id "$RUN_ID" --recall-top-n "$RECALL_TOP_N")
if [[ "$KNOWN_ISSUES_MODE" == "on" && "$REQUIRE_LANDMARK_RECALL" != "0" ]]; then
  show_cmd+=(--require-landmark-recall)
fi
"${show_cmd[@]}" | tee "$CHECK_DIR/show_actionable_results.log"

echo "STEP=build_summary"
python scripts/build_final_validation_summary.py --run-id "$RUN_ID" --out "$CHECK_DIR/summary.json" | tee "$CHECK_DIR/summary_console.log"

echo "STEP=verify_agent_execution_contract RUN_ID=$RUN_ID"
contract_json="$CHECK_DIR/agent_execution_contract.json"
contract_console="$CHECK_DIR/agent_execution_contract_console.log"
contract_cmd=(python scripts/verify_agent_execution_contract.py --run-id "$RUN_ID" --expected-known-issues-mode "$KNOWN_ISSUES_MODE" --out "$contract_json")
if [[ "$KNOWN_ISSUES_MODE" == "on" && -n "${STAT_HARNESS_REQUIRED_KNOWN_SIGNATURES:-}" ]]; then
  IFS=',' read -r -a required_known_signatures <<<"${STAT_HARNESS_REQUIRED_KNOWN_SIGNATURES}"
  for sig in "${required_known_signatures[@]}"; do
    sig="$(echo "$sig" | xargs)"
    if [[ -n "$sig" ]]; then
      contract_cmd+=(--require-known-signature "$sig")
    fi
  done
fi
contract_rc=0
"${contract_cmd[@]}" > "$contract_console" 2>&1 || contract_rc=$?
cat "$contract_console"
contract_ok="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}; print('1' if bool(d.get('ok')) else '0')" "$contract_json")"
if [[ "$contract_rc" -ne 0 || "$contract_ok" != "1" ]]; then
  echo "AGENT_EXECUTION_CONTRACT_FAILED=1"
fi

run_status_final="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); print(str(d.get('run_status') or ''))" "$CHECK_DIR/summary.json")"
skip_count="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); c=d.get('plugin_status_counts') or {}; print(int(c.get('skipped',0)))" "$CHECK_DIR/summary.json")"
degraded_count="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); c=d.get('plugin_status_counts') or {}; print(int(c.get('degraded',0)))" "$CHECK_DIR/summary.json")"
error_count="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); c=d.get('plugin_status_counts') or {}; print(int(c.get('error',0))+int(c.get('aborted',0)))" "$CHECK_DIR/summary.json")"
missing_count="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); m=d.get('missing_plugin_results'); print(int(m or 0))" "$CHECK_DIR/summary.json")"
analysis_ok_without_findings_count="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); print(int(d.get('analysis_ok_without_findings_count') or 0))" "$CHECK_DIR/summary.json")"
sql_assist_required_failure_count="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); print(int(d.get('sql_assist_required_failure_count') or 0))" "$CHECK_DIR/summary.json")"
unexplained_plugin_count="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); print(int(d.get('unexplained_plugin_count') or 0))" "$CHECK_DIR/summary.json")"
blank_kind_findings_count="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); print(int(d.get('blank_kind_findings_count') or 0))" "$CHECK_DIR/summary.json")"
explanations_missing_plain_text_count="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); print(int(d.get('explanations_missing_plain_text_count') or 0))" "$CHECK_DIR/summary.json")"
non_decision_explanations_missing_downstream_count="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); print(int(d.get('non_decision_explanations_missing_downstream_count') or 0))" "$CHECK_DIR/summary.json")"
summary_known_issues_mode="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); print(str(d.get('known_issues_mode') or 'on'))" "$CHECK_DIR/summary.json")"
recommendation_item_count="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); print(int(d.get('recommendation_item_count') or 0))" "$CHECK_DIR/summary.json")"
discovery_recommendation_count="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); print(int(d.get('discovery_recommendation_count') or 0))" "$CHECK_DIR/summary.json")"
actionable_plugin_count="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); print(int(d.get('actionable_plugin_count') or 0))" "$CHECK_DIR/summary.json")"
runtime_contract_mismatch_count="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); print(int(d.get('runtime_contract_mismatch_count') or 0))" "$CHECK_DIR/summary.json")"
summary_orchestrator_mode="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); print(str(d.get('orchestrator_mode') or ''))" "$CHECK_DIR/summary.json")"
summary_golden_mode="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); print(str(d.get('golden_mode') or ''))" "$CHECK_DIR/summary.json")"
summary_pre_report_filter_mode="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); print(str(d.get('pre_report_filter_mode') or 'unknown'))" "$CHECK_DIR/summary.json")"
summary_pre_report_drop_total="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); print(int(d.get('pre_report_drop_count_total') or 0))" "$CHECK_DIR/summary.json")"
summary_no_pre_report_filter_violation="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); print('1' if bool(d.get('no_pre_report_filter_violation')) else '0')" "$CHECK_DIR/summary.json")"

ok=true
if [[ "$hang_aborted" -ne 0 ]]; then ok=false; fi
if [[ "$run_status_final" != "completed" ]]; then ok=false; fi
if [[ "$skip_count" -ne 0 ]]; then ok=false; fi
if [[ "$degraded_count" -ne 0 ]]; then ok=false; fi
if [[ "$error_count" -ne 0 ]]; then ok=false; fi
if [[ "$missing_count" -ne 0 ]]; then ok=false; fi
if [[ "$analysis_ok_without_findings_count" -ne 0 ]]; then ok=false; fi
if [[ "$sql_assist_required_failure_count" -ne 0 ]]; then ok=false; fi
if [[ "$unexplained_plugin_count" -ne 0 ]]; then ok=false; fi
if [[ "$blank_kind_findings_count" -ne 0 ]]; then ok=false; fi
if [[ "$explanations_missing_plain_text_count" -ne 0 ]]; then ok=false; fi
if [[ "$non_decision_explanations_missing_downstream_count" -ne 0 ]]; then ok=false; fi
if [[ "$contract_rc" -ne 0 || "$contract_ok" != "1" ]]; then ok=false; fi
if [[ "$runtime_contract_mismatch_count" -gt 0 && "$STREAMING_POLICY_STRICT" == "1" ]]; then ok=false; fi
if [[ "$runtime_contract_mismatch_count" -gt 0 && "$STREAMING_POLICY_STRICT" != "1" ]]; then
  echo "STREAMING_POLICY_WARNING runtime_contract_mismatch_count=$runtime_contract_mismatch_count (set STAT_HARNESS_STREAMING_POLICY_STRICT=1 to fail)"
fi
echo "PRE_REPORT_FILTER_CONTRACT mode=$summary_pre_report_filter_mode drop_total=$summary_pre_report_drop_total violation=$summary_no_pre_report_filter_violation"
if [[ "$summary_no_pre_report_filter_violation" != "0" ]]; then ok=false; fi
if [[ -n "$summary_orchestrator_mode" && "$summary_orchestrator_mode" != "$ORCHESTRATOR_MODE" ]]; then ok=false; fi
if [[ -n "$summary_golden_mode" && "$summary_golden_mode" != "$GOLDEN_MODE" ]]; then ok=false; fi
if [[ "$KNOWN_ISSUES_MODE" == "off" || "$summary_known_issues_mode" == "off" ]]; then
  if [[ "$recommendation_item_count" -le 0 ]]; then ok=false; fi
  if [[ "$discovery_recommendation_count" -le 0 ]]; then ok=false; fi
  if [[ "$actionable_plugin_count" -le 0 ]]; then ok=false; fi
  reference_run_id="${STAT_HARNESS_REFERENCE_RUN_ID:-}"
  if [[ -z "$reference_run_id" ]]; then
    reference_run_id="$(python -c "import sqlite3,sys; rid=sys.argv[1]; conn=sqlite3.connect('appdata/state.sqlite'); row=conn.execute('select dataset_version_id,created_at from runs where run_id=?',(rid,)).fetchone(); out=''; out=((conn.execute(\"select run_id from runs where dataset_version_id=? and status='completed' and run_id<>? and created_at<? order by created_at desc limit 1\",(row[0], rid, row[1] or '')).fetchone() or [''])[0] if row and row[0] else ''); print(out); conn.close()" "$RUN_ID")"
  fi
  echo "AUTONOMOUS_REFERENCE_RUN_ID=${reference_run_id:-none}"
  if [[ -n "$reference_run_id" ]]; then
    ./.venv/bin/python scripts/compare_plugin_actionability_runs.py --before-run-id "$reference_run_id" --after-run-id "$RUN_ID" --out "$CHECK_DIR/novelty_compare.json" | tee "$CHECK_DIR/novelty_compare_console.log"
    novelty_new_count="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); print(int((((d.get('novelty') or {}).get('discovery') or {}).get('new_count')) or 0))" "$CHECK_DIR/novelty_compare.json")"
    novelty_jaccard="$(python -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads(p.read_text(encoding='utf-8')); print(float((((d.get('novelty') or {}).get('discovery') or {}).get('jaccard')) or 1.0))" "$CHECK_DIR/novelty_compare.json")"
    novelty_gate_pass="$(python -c "import sys; new_count=float(sys.argv[1]); j=float(sys.argv[2]); min_new=float(sys.argv[3]); max_j=float(sys.argv[4]); print('1' if (new_count>=min_new or j<=max_j) else '0')" "$novelty_new_count" "$novelty_jaccard" "$AUTONOMOUS_NOVELTY_MIN" "$AUTONOMOUS_NOVELTY_MAX_JACCARD")"
    echo "AUTONOMOUS_NOVELTY discovery_new_count=$novelty_new_count discovery_jaccard=$novelty_jaccard gate_pass=$novelty_gate_pass"
    if [[ "$novelty_gate_pass" != "1" ]]; then ok=false; fi
  else
    echo "AUTONOMOUS_NOVELTY reference_run_missing=true"
    if [[ "$REQUIRE_NOVELTY_REFERENCE" == "1" ]]; then ok=false; fi
  fi
fi

echo "STEP=validate_report_schema"
python -c "import json; from pathlib import Path; from jsonschema import validate; rid=Path('$CHECK_DIR/run_id.txt').read_text(encoding='utf-8').strip(); report=Path('appdata/runs')/rid/'report.json'; schema=Path('docs/report.schema.json'); payload=json.loads(report.read_text(encoding='utf-8')); spec=json.loads(schema.read_text(encoding='utf-8')); validate(instance=payload, schema=spec); print('REPORT_SCHEMA_OK=1')"

echo "FINAL_VALIDATION_OK=$ok"
echo "FINAL_VALIDATION_RUN_ID=$RUN_ID"
echo "FINAL_VALIDATION_SUMMARY=$CHECK_DIR/summary.json"

if [[ "$ok" != "true" ]]; then
  exit 1
fi
