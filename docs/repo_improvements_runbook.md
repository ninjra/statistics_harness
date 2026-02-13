# Repo Improvements Runbook

## Purpose
This runbook describes the deterministic local workflow for the repo-improvements catalog and rollout artifacts.

## Commands
1. `./.venv/bin/python scripts/run_repo_improvements_pipeline.py`
2. `./.venv/bin/python scripts/verify_docs_and_plugin_matrices.py`
3. `./.venv/bin/python -m pytest -q`

## Verify-Only
1. `./.venv/bin/python scripts/run_repo_improvements_pipeline.py --verify`
2. `./.venv/bin/python scripts/verify_docs_and_plugin_matrices.py`

## Failure Triage
1. If normalization fails, inspect `docs/repo_improvements_touchpoint_map.json`.
2. If dependency validation fails, inspect `docs/repo_improvements_execution_plan_v1.json`.
3. If status ledger is stale, rerun `scripts/run_repo_improvements_pipeline.py`.
4. If instruction coverage reports missing files, run:
   - `./.venv/bin/python scripts/full_instruction_coverage_report.py`
   - `./.venv/bin/python scripts/full_repo_misses.py`

## Rollback
1. Revert newly generated repo-improvements docs.
2. Re-run matrix generators.
3. Re-run test suite before shipping.
