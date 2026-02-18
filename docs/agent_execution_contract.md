# Agent Execution Contract

## Purpose
Prevent invalid quality claims by enforcing deterministic run-certification checks in code.

## Contract Checklist (Hard-Fail)
1. `run.completed`: run status is `completed`.
2. `run.results_complete`: every scheduled plugin has a terminal result row.
3. `run.no_bad_plugin_statuses`: no `skipped`, `degraded`, `error`, or `aborted`.
4. `run.known_issues_mode`: run matches requested known-issues lane (`on` or `off`).
5. `run.required_known_signatures`: required baseline known signatures are confirmed.
6. `compare.completed`: comparison run is also `completed`.
7. `compare.same_dataset`: both runs use the same `dataset_version_id`.
8. `compare.same_seed`: both runs keep the same seed policy.
9. `compare.same_known_issues_mode`: no mixed-mode run comparisons.
10. `compare.same_expected_plugin_count`: gauntlet size is unchanged between runs.

## Required Baseline Signature
- `analysis_close_cycle_contention:close_cycle_contention`

This signature is mandatory for baseline certification when known issues are enabled.

## Deterministic Verifier
- Script: `scripts/verify_agent_execution_contract.py`
- Output schema: `agent_execution_contract.v1`
- Deterministic output fields:
  - run snapshots
  - required known signatures
  - ordered check list with expected vs actual values
  - top-level `ok` boolean

## One-Line Commands
- Verify a single run in known-issues-on lane:
  - `./.venv/bin/python scripts/verify_agent_execution_contract.py --run-id <run_id> --expected-known-issues-mode on --require-known-signature analysis_close_cycle_contention:close_cycle_contention --out appdata/final_validation/<check_id>/agent_execution_contract.json`
- Verify a run-to-run comparison (must be apples-to-apples):
  - `./.venv/bin/python scripts/verify_agent_execution_contract.py --run-id <run_id_a> --compare-run-id <run_id_b> --expected-known-issues-mode on --require-known-signature analysis_close_cycle_contention:close_cycle_contention --out appdata/final_validation/<check_id>/agent_execution_contract_compare.json`

## Final Validation Integration
`scripts/run_final_validation_checklist.sh` runs this verifier automatically and fails the checklist when contract checks fail.
