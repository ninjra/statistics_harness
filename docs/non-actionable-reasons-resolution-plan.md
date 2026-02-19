# Plan: Non-Actionable Reasons Resolution

**Generated**: 2026-02-19
**Estimated Complexity**: High

## Overview
Resolve the remaining non-actionable reason buckets by converting ambiguous explanation outcomes into either:
1) concrete, process-targeted recommendations, or
2) deterministic, policy-valid blocker outcomes with explicit remediation metadata.

Primary target is to close the remaining baseline explanation buckets:
- `OBSERVATION_ONLY`
- `NO_ACTIONABLE_FINDING_CLASS`
- `PLUGIN_PRECONDITION_UNMET`
- `NO_DIRECT_PROCESS_TARGET`

This plan keeps the existing fail-closed contract (`skipped/degraded/error/aborted` fail run) and preserves full-gauntlet validation.

## Success Criteria (Hard Gates)
- `NO_DIRECT_PROCESS_TARGET` = `0` for analysis plugins on baseline certification run.
- `NO_ACTIONABLE_FINDING_CLASS` = `0` for analysis plugins on baseline certification run.
- `NO_DECISION_SIGNAL` remains `0` (no regression).
- `PLUGIN_PRECONDITION_UNMET` <= `10` and every row includes explicit `required_inputs` + `missing_inputs` + `next_step`; opaque fallback rows are disallowed.
- `OBSERVATION_ONLY` allowed only for whitelisted `utility_observation` plugins; non-whitelisted count = `0`.
- Contract and full test gates must both pass:
  - `.venv/bin/python -m pytest -q`
  - `scripts/verify_agent_execution_contract.py` on baseline + synthetics.

## Requirement Clarifications (from current repo policy)
- Full gauntlet required for final validation; no subset-only certification.
- `skipped` is failure.
- Baseline dataset remains primary reference for quality gates.
- Non-decision outputs are allowed only when deterministic, explicit, and useful to downstream plugins/users.
- Recommendation quality must stay aligned to modeled time/effort reduction.

## Skills Matrix
- `plan-harder`: structure phased, atomic, sprinted implementation plan.
- `shell-lint-ps-wsl`: enforce command hygiene while executing plan tasks.
- `python-testing-patterns`: add unit/integration contract coverage per reason family.
- `testing`: full-gauntlet and regression verification gates.
- `discover-observability` + `python-observability`: instrument reason metrics, trend deltas, and fail thresholds.
- `config-matrix-validator`: validate plugin-kind-to-routing coverage matrix.
- `deterministic-tests-marshal`: confirm stable reason outputs across repeated runs.

## Prerequisites
- Working baseline run artifacts in `appdata/runs/`.
- Existing contract checker: `scripts/verify_agent_execution_contract.py`.
- Existing recommendation/explanation synthesis in `src/statistic_harness/core/report.py`.
- Full test suite executable via `.venv/bin/python -m pytest -q`.

## Sprint 1: Build Deterministic Coverage Matrix
**Goal**: Create exact, reproducible ownership matrix for all remaining non-actionable rows.
**Skills**: `discover-observability`, `python-observability`, `config-matrix-validator`
**Demo/Validation**:
- Matrix artifact generated for baseline.
- Every row maps to plugin, finding kind, reason code, routing state, and expected owner.

### Task 1.1: Emit Reason Coverage Snapshot
- **Location**: `scripts/build_plugin_class_actionability_matrix.py`, `docs/plugin_class_actionability_matrix.json`
- **Description**: Extend snapshot to include `reason_code`, `finding_kind_preview`, `has_recommendation_text`, `has_process_target`, `action_type`, `is_policy_blocked`.
- **Complexity**: 4/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - Baseline snapshot clearly separates ambiguous vs policy-blocked vs prerequisite rows.
- **Validation**:
  - `pytest` for matrix script tests.
  - Scripted assertion for top offenders with deterministic sorting.
  - Manual spot-check only as supplemental verification.

### Task 1.2: Freeze Baseline Evidence Snapshot (Pre-Change)
- **Location**: `docs/release_evidence/`, `appdata/runs/`
- **Description**: Capture immutable pre-change reason histogram, offender list, and recommendation summary from baseline and synthetics.
- **Complexity**: 3/10
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Baseline snapshot artifacts are written and checksummed before any remediation edits.
- **Validation**:
  - Snapshot script output includes run_id, dataset_version_id, reason buckets, offender plugin lists.

### Task 1.3: Add Reason Threshold Gates
- **Location**: `scripts/verify_agent_execution_contract.py`, `tests/test_verify_agent_execution_contract.py`
- **Description**: Add configurable hard thresholds per reason bucket for analysis plugins (default strict for `NO_DIRECT_PROCESS_TARGET` and `NO_ACTIONABLE_FINDING_CLASS`).
- **Complexity**: 5/10
- **Dependencies**: Task 1.2
- **Acceptance Criteria**:
  - Contract fails when threshold exceeded.
  - Contract output includes actionable offenders list.
- **Validation**:
  - Contract tests with pass/fail fixtures.

### Task 1.4: Finding-Kind Routing Registry Scaffolding
- **Location**: `src/statistic_harness/core/report.py`, `docs/plugin_data_access_matrix.json`/routing registry file
- **Description**: Introduce registry scaffold for finding-kind handlers and required fields to prevent ad-hoc routing logic drift.
- **Complexity**: 6/10
- **Dependencies**: Task 1.3
- **Acceptance Criteria**:
  - Registry exists and is consulted before fallback routing.
  - Unknown kinds fail contract unless explicitly registered.
- **Validation**:
  - Registry unit tests + contract tests.

## Sprint 2: Eliminate `NO_DIRECT_PROCESS_TARGET`
**Goal**: Convert grouped/implicit recommendations into process-targeted recommendation rows.
**Skills**: `python-testing-patterns`, `testing`
**Demo/Validation**:
- Baseline `NO_DIRECT_PROCESS_TARGET` reduced to zero for analysis plugins.

### Task 2.1: Generalized Process Target Expansion
- **Location**: `src/statistic_harness/core/report.py`
- **Description**: Expand targets from `evidence.selected`, `target_process_ids`, `process_ids`, and normalized `process_id` tokens into per-process recommendation rows.
- **Complexity**: 6/10
- **Dependencies**: Task 1.4
- **Acceptance Criteria**:
  - Grouped findings produce process-level candidates where process hints exist.
  - Non-modifiable chain actions remain filtered.
- **Validation**:
  - Unit tests for grouped expansion behavior.

### Task 2.2: Plugin-Specific Target Extractors (24-offender lane)
- **Location**: `src/statistic_harness/core/report.py`, optional per-plugin helper modules
- **Description**: Add plugin-family adapters for remaining no-target offenders (association rules, clustering, leftfield/causal families) using evidence fields and deterministic inference rules.
- **Complexity**: 8/10
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Every recommendation-bearing offender emits at least one process target or is reclassified to explicit non-process reason with downstream explanation.
- **Validation**:
  - Baseline rerun reason diff.
  - Targeted tests for each adapter family.

## Sprint 3: Eliminate `NO_ACTIONABLE_FINDING_CLASS`
**Goal**: Ensure each finding kind either routes to recommendation lane or maps to explicit deterministic non-decision reason with rationale.
**Skills**: `python-testing-patterns`, `config-matrix-validator`
**Demo/Validation**:
- Baseline `NO_ACTIONABLE_FINDING_CLASS` reduced to zero.

### Task 3.1: Adapter Implementations for Current Unknown Kinds
- **Location**: `src/statistic_harness/core/report.py`
- **Description**: Implement routing for current `NO_ACTIONABLE_FINDING_CLASS` offender kinds (e.g., attribution/cluster/changepoint families) to recommendation text + modeled metrics where feasible.
- **Complexity**: 8/10
- **Dependencies**: Task 1.4, Task 2.2
- **Acceptance Criteria**:
  - No baseline rows remain in `NO_ACTIONABLE_FINDING_CLASS`.
- **Validation**:
  - Baseline recompute + reason histogram check.

## Sprint 4: Resolve `PLUGIN_PRECONDITION_UNMET`
**Goal**: Move from opaque precondition fallback to deterministic, repairable, data-backed outcomes.
**Skills**: `testing`, `discover-observability`, `python-observability`
**Demo/Validation**:
- Each precondition case identifies exact missing requirements and recovery path.

### Task 4.1: Prerequisite Contract Schema
- **Location**: `src/statistic_harness/core/actionability_explanations.py`, plugin result schema helpers, tests
- **Description**: Require `required_inputs`, `missing_inputs`, `fallback_basis`, and `next_step` fields for precondition outcomes.
- **Complexity**: 6/10
- **Dependencies**: Task 1.3, Task 1.4
- **Acceptance Criteria**:
  - Precondition explanations are deterministic and non-generic.
- **Validation**:
  - Schema/contract tests.
  - `report.json` validation against `docs/report.schema.json`.

### Task 4.2: Normalization/Column Alias Backfill for Top Offenders
- **Location**: `src/statistic_harness/core/column_inference.py`, plugin input helpers, plugin modules
- **Description**: Add alias/derived fields to satisfy true missing-input gaps for top precondition offenders.
- **Complexity**: 8/10
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Precondition bucket reduced by at least 75% from frozen baseline snapshot.
  - Precondition bucket final count <= 10.
  - Remaining cases are justified and reproducible.
- **Validation**:
  - Baseline + synthetic runs compare pre/post counts.

## Sprint 5: Reduce `OBSERVATION_ONLY` to Decision-Useful Output
**Goal**: Convert high-value observation plugins into recommendation emitters or explicitly classify as non-decision utility plugins.
**Skills**: `python-testing-patterns`, `testing`, `discover-observability`
**Demo/Validation**:
- Observation-only plugin list is explicitly partitioned into decision vs utility classes.

### Task 5.1: Observation Plugin Classification Lane
- **Location**: `docs/plugin_class_taxonomy.yaml`, `src/statistic_harness/core/report.py`
- **Description**: Add explicit class labels: `decision_candidate`, `utility_observation`, `support_only`; enforce routing behavior by class.
- **Complexity**: 5/10
- **Dependencies**: Task 1.4
- **Acceptance Criteria**:
  - Utility observations no longer treated as unresolved actionability debt.
- **Validation**:
  - Taxonomy tests + explanation lane tests.

### Task 5.2: Promote Top Observation Families to Actionable
- **Location**: relevant plugin files under `src/statistic_harness/core/stat_plugins/`
- **Description**: For top 15 observation families ranked by close-window footprint + execution volume, emit process-specific recommendations with modeled delta-hour fields.
- **Complexity**: 9/10
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - Observation-only non-whitelisted count reaches zero.
  - No regression on recommendation quality metrics: `modeled_delta_hours > 0`, `dimensionless_efficiency_gain` present, `ranking_metric` populated.
- **Validation**:
  - Baseline recommendation diff + quality contract checks.

## Sprint 6: Full Certification Pass
**Goal**: Certify reason-resolution changes across baseline + synthetic datasets.
**Skills**: `testing`, `deterministic-tests-marshal`, `python-observability`
**Demo/Validation**:
- Full-gauntlet pass and deterministic reason deltas reported.

### Task 6.1: Full Gauntlet Runs
- **Location**: `scripts/run_loaded_dataset_full.py`, `appdata/runs/*`
- **Description**: Run full gauntlet on baseline and both synthetic datasets with consistent seed and mode settings.
- **Complexity**: 5/10
- **Dependencies**: Sprints 2-5
- **Acceptance Criteria**:
  - No skipped/degraded/error statuses.
  - Reason thresholds meet contract.
  - Repeat run with same `run_seed` reproduces identical reason histogram and offender list.
- **Validation**:
  - `verify_agent_execution_contract.py` passes for each run.

### Task 6.2: Before/After Evidence Bundle
- **Location**: `docs/release_evidence/`, `docs/plugin_class_actionability_matrix.*`
- **Description**: Emit before/after reason histograms, offender lists, and recommendation-quality deltas.
- **Complexity**: 4/10
- **Dependencies**: Task 1.2, Task 6.1
- **Acceptance Criteria**:
  - Required artifacts are present:
    - `docs/release_evidence/non_actionable_reason_before_after.json`
    - `docs/release_evidence/non_actionable_reason_before_after.md`
    - `docs/release_evidence/non_actionable_offenders_before_after.csv`
  - Artifact payload includes run ids, dataset ids, reason deltas, plugin offender deltas, and recommendation quality deltas.
- **Validation**:
  - Evidence files present and reproducible from scripts.

## Testing Strategy
- Unit tests per new helper (`process target extraction`, `reason mapping`, `kind routing`).
- Contract tests for reason bucket thresholds and allowlists.
- Integration tests for recommendation synthesis from grouped/multi-process findings.
- Full repo test gate: `.venv/bin/python -m pytest -q`.
- Full gauntlet run verification for baseline + synthetic datasets.

## Potential Risks & Gotchas
- Over-aggressive inference may create false process targets.
  - Mitigation: require confidence + evidence trace for inferred targets.
- Utility/observational plugins may be forced into weak recommendations.
  - Mitigation: explicit plugin taxonomy with utility class and deterministic rationale.
- Plugin precondition fixes can create hidden normalization coupling.
  - Mitigation: add explicit required/missing input schema and migration tests.
- Long gauntlet runtime can hide hung I/O states.
  - Mitigation: watchdog telemetry + per-stage heartbeat checks.

## Rollback Plan
- Revert routing changes in `report.py` and contract thresholds in `verify_agent_execution_contract.py`.
- Restore previous reason allowlist and baseline report snapshot.
- Re-run full tests and baseline gauntlet on previous commit to confirm restoration.
