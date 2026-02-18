# Operator Playbook: Autonomous Insights

## Scope
Run the full validation gauntlet in both known-issues modes, verify autonomous novelty, and inspect streaming-policy evidence.

## One-Line Commands
- Known issues ON:
  - `bash scripts/run_final_validation_checklist.sh 3246cc7cd7d57a317ddc05e80e6f6f5bfe7f50deb0ee7af8db50d04bae180e1a 1337 30 on`
- Known issues OFF:
  - `bash scripts/run_final_validation_checklist.sh 3246cc7cd7d57a317ddc05e80e6f6f5bfe7f50deb0ee7af8db50d04bae180e1a 1337 30 off`
- Compare two runs:
  - `./.venv/bin/python scripts/compare_plugin_actionability_runs.py --before-run-id <before_run_id> --after-run-id <after_run_id> --out appdata/final_validation/compare_<before>_to_<after>.json`
- Verify execution contract (single run):
  - `./.venv/bin/python scripts/verify_agent_execution_contract.py --run-id <run_id> --expected-known-issues-mode on --require-known-signature analysis_close_cycle_contention:close_cycle_contention --out appdata/final_validation/<check_id>/agent_execution_contract.json`
- Verify execution contract (run comparison):
  - `./.venv/bin/python scripts/verify_agent_execution_contract.py --run-id <run_id_a> --compare-run-id <run_id_b> --expected-known-issues-mode on --require-known-signature analysis_close_cycle_contention:close_cycle_contention --out appdata/final_validation/<check_id>/agent_execution_contract_compare.json`
- Generate plugin class matrix and cards:
  - `./.venv/bin/python scripts/build_plugin_class_actionability_matrix.py --run-id <run_id> && ./.venv/bin/python scripts/generate_plugin_example_cards.py`

## Expected Artifacts
- `appdata/final_validation/<check_id>/summary.json`
- `appdata/final_validation/<check_id>/agent_execution_contract.json`
- `appdata/final_validation/<check_id>/novelty_compare.json` (autonomous mode)
- `appdata/runs/<run_id>/batch_sequence_validation_checklist.json`
- `appdata/runs/<run_id>/batch_sequence_validation_checklist.md`
- `docs/plugin_class_actionability_matrix.json`
- `docs/plugin_example_cards/index.json`

## Hard-Fail Gates
- Any plugin status in `skipped`, `degraded`, `error`, or `aborted`
- Missing plugin results
- `analysis_ok_without_findings_count > 0`
- SQL-assist required plugin failures
- Unexplained plugins
- Blank finding kind count
- Missing explanation text/downstream map
- Agent execution contract failure (`agent_execution_contract.json.ok != true`)
- In known-issues-off mode:
  - no discovery recommendations
  - no actionable plugins
  - novelty gate failure against reference run

## Streaming Policy Gate
- `runtime_contract_mismatch_count` is always reported in summary.
- Warning mode:
  - `STAT_HARNESS_STREAMING_POLICY_STRICT=0`
- Hard-fail mode:
  - `STAT_HARNESS_STREAMING_POLICY_STRICT=1`

## Novelty Gate Controls
- Minimum new discovery signatures:
  - `STAT_HARNESS_AUTONOMOUS_NOVELTY_MIN` (default `1`)
- Maximum allowed similarity before requiring novelty:
  - `STAT_HARNESS_AUTONOMOUS_NOVELTY_MAX_JACCARD` (default `0.95`)
- Require reference run:
  - `STAT_HARNESS_REQUIRE_NOVELTY_REFERENCE=1`
- Optional manual reference:
  - `STAT_HARNESS_REFERENCE_RUN_ID=<run_id>`

## Troubleshooting
- `FINAL_VALIDATION_OK=false` with novelty failure:
  - Check `novelty_compare.json` for unchanged/new/dropped signature counts.
- Streaming mismatch warnings/failures:
  - Check `summary.json.runtime_contract_mismatches`.
  - Inspect per-plugin runtime artifacts under `appdata/runs/<run_id>/artifacts/<plugin_id>/runtime_access.json`.
- Missing sequence checklist:
  - Re-run `python scripts/show_actionable_results.py --run-id <run_id>` to regenerate answer artifacts.
