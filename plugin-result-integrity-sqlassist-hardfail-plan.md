# Plan: Plugin Result Integrity, SQL-Assist Hard-Fail, and No-Findings Remediation

**Generated**: 2026-02-17  
**Estimated Complexity**: High

## Available Skills (Full List)
- `atlas`
- `audit-log-integrity-checker`
- `ccpm-debugging`
- `cloudflare-deploy`
- `config-matrix-validator`
- `deterministic-tests-marshal`
- `develop-web-game`
- `discover-observability`
- `doc`
- `e2e-testing-patterns`
- `evidence-trace-auditor`
- `export-sanitization-verifier`
- `figma`
- `figma-implement-design`
- `find-skills`
- `gh-address-comments`
- `gh-fix-ci`
- `golden-answer-harness`
- `grpc-protocol-profiler`
- `imagegen`
- `jupyter-notebook`
- `linear`
- `logging-best-practices`
- `logging-observability`
- `netlify-deploy`
- `notion-knowledge-capture`
- `notion-meeting-intelligence`
- `notion-research-documentation`
- `notion-spec-to-implementation`
- `observability-engineer`
- `observability-slo-broker`
- `openai-docs`
- `parallel-task`
- `pdf`
- `perf-regression-gate`
- `plan-harder`
- `planner`
- `playwright`
- `policygate-penetration-suite`
- `python-observability`
- `python-testing-patterns`
- `render-deploy`
- `resource-budget-enforcer`
- `screenshot`
- `security-best-practices`
- `security-ownership-map`
- `security-threat-model`
- `security-threats-to-tests`
- `sentry`
- `shell-lint-ps-wsl`
- `sora`
- `source-quality-linter`
- `speech`
- `spreadsheet`
- `state-recovery-simulator`
- `swarm-planner`
- `testing`
- `transcribe`
- `vercel-deploy`
- `webapp-testing`
- `yeet`
- `skill-creator`
- `skill-installer`

## Selected Skills and Rationale by Work Area
- Planning and decomposition: `plan-harder`
  - Build phased, atomic, committable implementation sequence with hard acceptance criteria.
- Status/result contract + plugin remediation tests: `python-testing-patterns`, `testing`
  - Enforce run-fail policy for `ok`+no-findings and add deterministic regression tests for the 7 target plugins.
- Pipeline/run telemetry and proof of correctness: `discover-observability`, `observability-engineer`, `python-observability`
  - Add explicit metrics/logs proving fresh execution vs cache reuse and plugin result quality gates.
- Command safety/consistency during implementation: `shell-lint-ps-wsl`
  - Keep command syntax/linting consistent across PowerShell/WSL execution paths.

## Overview
Implement a strict, auditable result-quality framework so the system can no longer hide non-working behavior behind `ok` statuses or ambiguous empty outputs.

User-confirmed target behavior:
- Plugins should produce `ok` with at least one diagnostic/actionable finding, not empty `ok`.
- Run continues even if a plugin is noncompliant, but overall run status must fail.
- Caching is allowed only when plugin code + transformed data state indicate identical result potential.
- SQL-assist schema wiring is mandatory for SQL-assist-dependent flow; missing schema is a hard error and must fail run.
- `transform_sqlpack_materialize_v1` should be enabled in the mode most optimal for the four pillars.

## Prerequisites
- Python 3.11+ virtual environment active (`.venv`).
- Existing baseline dataset run IDs available in SQLite for before/after validation.
- Full plugin catalog current (`docs/_codex_plugin_catalog.md`) and run manifests available.
- Agreement that strict policy supersedes legacy permissive behavior for empty `ok`.

## Sprint 1: Define Enforceable Result Contract
**Goal**: Make plugin output validity machine-checkable and impossible to misclassify.
**Demo/Validation**:
- Run targeted tests for status normalization and new result-quality contract.
- Demonstrate a synthetic plugin returning `ok` + zero findings causes overall run failure.

### Task 1.1: Add Result-Quality Contract Types
- **Location**: `src/statistic_harness/core/pipeline.py`, `src/statistic_harness/core/plugin_manager.py`, `src/statistic_harness/core/stat_plugins/contract.py`
- **Description**: Introduce contract fields and enforcement helpers:
  - `diagnostic_only` plugin capability flag.
  - `reason_code` for non-actionable outcomes.
  - `fresh_or_reused` execution marker.
- **Complexity**: 6/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - Runtime can classify each plugin result as actionable, diagnostic-only valid, or contract violation.
  - Legacy status aliases still normalize to allowed terminal statuses.
- **Validation**:
  - Unit tests in `tests/test_pipeline_na_status_normalization.py` and new contract tests.

### Task 1.2: Add Run-Level Failure Gate for Empty `ok`
- **Location**: `src/statistic_harness/core/pipeline.py`, `scripts/build_final_validation_summary.py`, `scripts/run_final_validation_checklist.sh`
- **Description**: If any plugin returns `ok` with zero findings and is not `diagnostic_only=true`, mark run failed overall (while continuing execution).
- **Complexity**: 5/10
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Pipeline completes all plugins but run status becomes failed/partial with explicit violation record.
  - Final validation script exits non-zero on this condition.
- **Validation**:
  - New integration test covering continue-execution + failed-overall semantics.

### Task 1.3: Define Deterministic `reason_code` Taxonomy
- **Location**: `docs/result_quality_contract.md` (new), `src/statistic_harness/core/pipeline.py`
- **Description**: Create fixed reason codes for NA/non-actionable paths (example: `INSUFFICIENT_POSITIVE_SAMPLES`, `QUADRATIC_CAP_EXCEEDED`, `NO_ELIGIBLE_SLICE`, `NO_SIGNIFICANT_EFFECT`).
- **Complexity**: 4/10
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Every NA or non-actionable diagnostic result has a deterministic reason code.
  - No free-form-only reason strings in persisted result contracts.
  - Every allowed `reason_code` is documented in `docs/result_quality_contract.md`.
  - Persisted contract schema enforces reason-code enum presence for NA/non-actionable outputs.
- **Validation**:
  - Schema/unit test asserting reason_code presence and valid enum.
  - Enum coverage test that fails if documented codes and schema codes drift.

## Sprint 2: SQL-Assist Hard-Fail and SQLPack Enablement
**Goal**: Remove false confidence in SQL-assist path and enable SQLPack in the 4-pillar-optimal profile.
**Demo/Validation**:
- Demonstrate hard run failure when SQL-assist schema is missing.
- Demonstrate successful SQL-intents + SQLPack materialization when wired.

### Task 2.1: Wire SQL-Assist Schema Snapshot as Mandatory Dependency
- **Location**: `src/statistic_harness/core/sql_assist.py`, `src/statistic_harness/core/pipeline.py`, transform plugin modules
- **Description**: Replace current NA fallback for missing schema snapshot with hard error when SQL-assist-dependent plugins are in selected run profile.
- **Complexity**: 7/10
- **Dependencies**: Sprint 1 complete
- **Acceptance Criteria**:
  - Missing schema snapshot produces hard plugin error and overall run failure.
  - Error message identifies remediation path and expected artifact/source.
  - Profile gating is explicit and tested:
    - SQL-assist-dependent profile selected + missing schema => hard fail.
    - SQL-assist-dependent profile unselected + missing schema => no SQL-assist hard fail.
- **Validation**:
  - Unit/integration tests for missing schema and present schema scenarios.
  - Profile-aware integration test for selected vs unselected SQL-assist-dependent plugin sets.

### Task 2.2: Enable `transform_sqlpack_materialize_v1` by Profile Policy
- **Location**: `src/statistic_harness/core/plugin_manager.py`, plugin settings defaults, run profile config files
- **Description**: Set SQLPack materialization to enabled in the profile that best supports four pillars (traceability, reliability, quality, security) while preserving deterministic behavior.
- **Complexity**: 5/10
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Release/full-validation profile has SQLPack materialization enabled by default.
  - Any disablement is explicit and logged as policy exception.
- **Validation**:
  - Profile tests and final validation checklist assertion.

### Task 2.3: Fix `transform_template` Outcome Semantics
- **Location**: template transform plugin implementation and schema
- **Description**: Convert “Not applicable: no template configured” from ambiguous `ok` into deterministic `na` + reason_code or require template in selected profiles.
- **Complexity**: 4/10
- **Dependencies**: Task 1.3
- **Acceptance Criteria**:
  - No `ok` status for absent template config without diagnostic findings.
- **Validation**:
  - Unit tests for configured vs non-configured template runs.

## Sprint 3: Remediate the 7 Priority No-Findings Plugins
**Goal**: Ensure the 7 priority plugins either emit valid diagnostic findings under `ok`, or deterministic `na`/error when preconditions fail.
**Demo/Validation**:
- Single baseline run with all 7 producing contract-compliant outputs.
- No ambiguous empty-`ok` results for these plugins.

### Task 3.1: Precondition-to-NA/Error Refactor for Priority 7
- **Location**: owning plugin modules (primarily `src/statistic_harness/core/stat_plugins/registry.py` and related analysis modules)
- **Description**: For each priority plugin:
  - Distinguish true compute success from precondition miss.
  - Use `na` + `reason_code` for non-applicability.
  - Use `error` for invariant/contract violations.
- **Complexity**: 8/10
- **Dependencies**: Sprint 1 and Sprint 2
- **Acceptance Criteria**:
  - No priority plugin returns `ok` with empty findings without explicit diagnostic finding.
- **Validation**:
  - Parameterized tests per plugin over baseline-like fixtures.

### Task 3.2: Add Diagnostic Findings for “No Significant Signal”
- **Location**: same as Task 3.1
- **Description**: Where computation executes but no significant effect is detected, emit one low-severity diagnostic finding containing:
  - test/statistic values,
  - threshold used,
  - reason_code `NO_SIGNIFICANT_EFFECT`.
- **Complexity**: 6/10
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - `ok` paths include at least one finding with concrete numeric evidence.
- **Validation**:
  - Unit tests asserting non-empty findings and expected fields.

### Task 3.3: Quadratic-Cap Strategy for Heavy Plugins
- **Location**: affected plugin modules for distance/geometric/RQA methods
- **Description**: Replace silent non-actionable outcomes with bounded fallback:
  - deterministic sampling/approximation,
  - explicit confidence downgrade,
  - emitted diagnostic finding with cap metadata.
- **Complexity**: 7/10
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - No bare “quadratic cap exceeded” `ok` outcome without finding payload.
- **Validation**:
  - Determinism tests with fixed `run_seed` and repeated executions.

## Sprint 4: Cache Correctness and Freshness Proof
**Goal**: Enforce “cache only when unchanged” with auditable proof.
**Demo/Validation**:
- Run with mixed reused/fresh plugins and verify reuse eligibility is mathematically explained.
- Run diff report shows fresh-vs-reused counts and reasons.

### Task 4.1: Tighten Reuse Eligibility Fingerprint
- **Location**: pipeline execution fingerprint logic in `src/statistic_harness/core/pipeline.py` and storage helpers
- **Description**: Require cache reuse only when plugin code hash, config hash, dataset/transformed-state hash, and dependency fingerprints all match.
- **Complexity**: 8/10
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Reused result includes trace fields proving equivalence inputs.
  - Any changed transform/plugin/dependency forces fresh execution.
- **Validation**:
  - Integration tests mutating one dimension at a time and asserting cache invalidation.

### Task 4.2: Add Freshness Audit Report
- **Location**: `scripts/build_final_validation_summary.py` (and possibly new script)
- **Description**: Emit per-run table:
  - plugin_id,
  - executed_fresh bool,
  - reuse_source_run_id,
  - equivalence fingerprint,
  - result-quality compliance.
- **Complexity**: 5/10
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Reviewers can prove what truly executed versus reused.
- **Validation**:
  - Golden-file style assertions for summary JSON fields.

## Sprint 5: Full Gauntlet, Comparison, and Release Gate
**Goal**: Deliver a release-ready gate where noncompliant plugin outputs cannot pass.
**Demo/Validation**:
- Full baseline gauntlet run completes.
- Final checklist passes only when:
  - no skipped/degraded,
  - no hard errors,
  - no empty `ok` violations,
  - SQL-assist dependent chain is wired and passing.

### Task 5.1: Extend Final Validation Checklist
- **Location**: `scripts/run_final_validation_checklist.sh`
- **Description**: Add explicit checks:
  - `ok_empty_violation_count == 0`,
  - `sql_assist_required_failures == 0`,
  - reason-code completeness.
- **Complexity**: 4/10
- **Dependencies**: Sprints 1-4
- **Acceptance Criteria**:
  - Checklist fails decisively for any policy breach.
- **Validation**:
  - Script-level tests or dry-run fixtures simulating each violation.

### Task 5.2: Before/After Comparison on Primary Baseline Dataset
- **Location**: comparison scripts/reports under `docs/release_evidence/`
- **Description**: Produce plain-language comparison between pre-fix and post-fix runs:
  - counts of actionable findings,
  - counts of diagnostic-only findings,
  - zero ambiguous `ok` outputs.
- **Complexity**: 5/10
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - Reproducible comparison artifact committed.
- **Validation**:
  - Repeatability check with fixed run seed.

## Testing Strategy
- Unit tests:
  - status normalization and contract compliance.
  - reason-code enum coverage.
  - SQL-assist hard-fail behavior.
- Integration tests:
  - full pipeline continuation + failed-overall behavior on quality violations.
  - cache reuse eligibility and invalidation matrix.
- Full-system tests:
  - `python -m pytest -q` must pass.
  - final validation checklist must pass with strict gates.

## Potential Risks & Gotchas
- Risk: Over-converting valid “no signal” scientific outcomes into NA.
  - Mitigation: keep `ok` for executed methods, but always emit one diagnostic finding with numeric evidence.
- Risk: Cache policy becomes too strict and hurts runtime.
  - Mitigation: profile-equivalence fingerprint with targeted invalidation dimensions.
- Risk: SQL-assist hard-fail blocks users lacking setup.
  - Mitigation: enforce only when SQL-assist-dependent plugins are included in selected profile; message remediation clearly.
- Risk: Legacy reports expect old statuses/strings.
  - Mitigation: backward-compatible adapters in report/render layers, strict runtime contract in new runs.

## Rollback Plan
- Revert new quality gate checks in `pipeline.py` and checklist script if release-blocking false positives occur.
- Keep reason-code additions and observability instrumentation even if gate strictness is temporarily reduced.
- Restore prior cache behavior behind a temporary feature flag if runtime regression is unacceptable.
