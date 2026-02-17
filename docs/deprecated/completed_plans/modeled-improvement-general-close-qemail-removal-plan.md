# Plan: Modeled Improvement Quality + General vs Close Separation + QEMAIL Removal Modeling

**Generated**: 2026-02-12
**Estimated Complexity**: High

## Overview
Improve recommendation modeling so outputs consistently provide useful modeled percent impact, split recommendations into two explicit scopes (`general` and `close_specific`), and add deterministic counterfactual modeling for removing QEMAIL in both scopes.

The plan prioritizes the 4 pillars by design:
- Better modeling accuracy and evidence traceability.
- More useful operator output (general vs close-specific sections).
- Recency-weighted relevance (newest logs matter most).
- Deterministic, testable, fail-closed behavior when modeled % cannot be computed.

## Prerequisites
- Python 3.11+ environment in `.venv`.
- Existing loaded normalized dataset in SQLite.
- Existing recommendation/report pipeline in:
  - `src/statistic_harness/core/report.py`
  - `src/statistic_harness/core/stat_plugins/ideaspace.py`
  - `src/statistic_harness/core/lever_library.py`
  - `scripts/run_loaded_dataset_full.py`
  - `scripts/show_actionable_results.py`
- Existing tests passing before changes:
  - `python -m pytest -q`

## Sprint 1: Modeling Contract + Scope Taxonomy
**Goal**: Define strict output contracts for modeled impact and scope split before algorithm changes.
**Demo/Validation**:
- Report/recommendation schema fields for modeled impact and scope are explicit.
- Every recommendation has either modeled `%` or a deterministic `not_modeled_reason`.

### Task 1.1: Add recommendation scope taxonomy
- **Location**: `src/statistic_harness/core/report.py`, `docs/report.schema.json`
- **Description**: Add required recommendation field `scope_class` with allowed values:
  - `general`
  - `close_specific`
- **Complexity**: 4
- **Dependencies**: None
- **Acceptance Criteria**:
  - Discovery recommendations are tagged with one scope.
  - Rendering can group by scope without guessing.
- **Validation**:
  - Unit test for scope assignment.
  - Schema validation test.

### Task 1.2: Standardize modeled-impact fields
- **Location**: `src/statistic_harness/core/report.py`, `docs/report.schema.json`
- **Description**: Normalize impact fields to always include:
  - `modeled_percent` (float 0..100 or null)
  - `modeled_basis_hours` (baseline hours used for % calc)
  - `modeled_delta_hours` (absolute modeled reduction)
  - `not_modeled_reason` (required when `modeled_percent` is null)
- **Complexity**: 5
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - No recommendation with both missing modeled percent and missing reason.
  - Existing `modeled_delta`/`impact_hours` values are mapped consistently.
- **Validation**:
  - Unit tests for conversion logic and null-handling.

### Task 1.3: Add recency weighting contract (newest-first relevance)
- **Location**: `src/statistic_harness/core/report.py`
- **Description**: Define deterministic recency weighting for general recommendations:
  - Newer records weighted more heavily than older records.
  - Still uses full data (no truncation), but relevance and modeled % use weighted aggregates.
- **Complexity**: 6
- **Dependencies**: Task 1.2
- **Acceptance Criteria**:
  - Recency factor is deterministic and configurable.
  - No month-specific operator tuning required.
- **Validation**:
  - Unit test with synthetic old/new periods verifies newer period dominates weighting.

## Sprint 2: Counterfactual Engine for Modeled %
**Goal**: Make modeled percent robust and available for most high-value recommendations.
**Demo/Validation**:
- Top recommendations contain meaningful modeled `%`.
- Recommendations with missing `%` include deterministic reason.

### Task 2.1: Build shared counterfactual helpers
- **Location**: `src/statistic_harness/core/report.py` (or new helper module under `src/statistic_harness/core/`)
- **Description**: Implement reusable deterministic functions:
  - baseline metric extraction per scope
  - scenario metric extraction per recommendation action
  - percent computation: `100 * delta_hours / baseline_hours`
- **Complexity**: 7
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - A single engine computes modeled percent for close/general scenarios.
  - Division-by-zero and missing-baseline cases handled fail-closed.
- **Validation**:
  - Unit tests for normal, zero, missing, and negative-delta cases.

### Task 2.2: Add action-type model adapters
- **Location**: `src/statistic_harness/core/report.py`
- **Description**: Implement per-action deterministic adapters for:
  - `batch_input`
  - `batch_or_cache`
  - `throttle_or_dedupe`
  - `add_server`
  - `tune_schedule`
  - `orchestrate_macro`
  - `route_process`
  - `batch_group_candidate`
- **Complexity**: 8
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Each action type either returns modeled % or explicit not-modeled reason.
  - High-priority actions (`add_server`, `tune_schedule`, `batch_input`) produce % whenever evidence is sufficient.
- **Validation**:
  - Action-type unit tests with synthetic fixtures.

### Task 2.3: Enforce modeled-impact gate for top-ranked recommendations
- **Location**: `src/statistic_harness/core/report.py`
- **Description**: Add quality gate:
  - Top N recommendations must have modeled `%` unless blocked by explicit `not_modeled_reason`.
  - Optional demotion of recommendations with poor modeling confidence.
- **Complexity**: 6
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - Top recommendations are not opaque.
  - Ranking is explainable and stable.
- **Validation**:
  - Ranking tests with mixed modeled/non-modeled recommendations.

## Sprint 3: General vs Close-Specific Output Separation
**Goal**: Split insights into operator-usable sections with independent modeled baselines.
**Demo/Validation**:
- Final outputs have separate `General Improvements` and `Close-Specific Improvements`.
- Each section has its own baseline and modeled percent context.

### Task 3.1: Add deterministic scope classifier
- **Location**: `src/statistic_harness/core/report.py`
- **Description**: Classify recommendations into:
  - `general`: applies broadly across timeline/workload.
  - `close_specific`: tied to close-cycle windows, close metrics, spillover, close contention.
- **Complexity**: 5
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Close-cycle plugins/metrics always map to `close_specific`.
  - Non-close structural changes map to `general` unless evidence indicates close-only effect.
- **Validation**:
  - Classification tests by plugin kind + evidence fields.

### Task 3.2: Render separated sections in all outputs
- **Location**: `src/statistic_harness/core/report.py`, `scripts/run_loaded_dataset_full.py`, `scripts/show_actionable_results.py`
- **Description**: Update rendering so:
  - `report.md`, `answers_recommendations.md`, and `answers_recommendations_plain.md` show grouped sections.
  - each recommendation line includes modeled `%` and absolute delta hours.
- **Complexity**: 5
- **Dependencies**: Task 3.1, Sprint 2
- **Acceptance Criteria**:
  - Grouping is stable and professional.
  - Plain-language output retains exact % values.
- **Validation**:
  - Snapshot tests for markdown outputs.

## Sprint 4: QEMAIL Removal Modeling (General + Close)
**Goal**: Explicitly model a no-QEMAIL counterfactual and surface both scope impacts.
**Demo/Validation**:
- Known-issue check for QEMAIL is based on modeled impact, not title matching only.
- Output includes:
  - `general_no_qemail_modeled_percent`
  - `close_no_qemail_modeled_percent`

### Task 4.1: Add no-QEMAIL scenario simulator
- **Location**: `src/statistic_harness/core/report.py`, `src/statistic_harness/core/stat_plugins/ideaspace.py`
- **Description**: Build deterministic counterfactual removing QEMAIL rows from modeled scenario metrics:
  - Compute baseline and no-QEMAIL deltas for general scope.
  - Compute baseline and no-QEMAIL deltas for close-specific scope.
- **Complexity**: 8
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - Both scope percentages are computed when QEMAIL evidence is sufficient.
  - Explicit reason when insufficient evidence.
- **Validation**:
  - Unit tests with fixture where QEMAIL removal gives >10% close improvement.

### Task 4.2: Integrate QEMAIL model into known-issue checks
- **Location**: `scripts/show_actionable_results.py`, `src/statistic_harness/core/report.py`
- **Description**: Replace weak presence checks with modeled-scenario checks:
  - Known issue passes only when modeled close and/or general impact crosses configured threshold.
- **Complexity**: 6
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - QEMAIL known issue can pass based on modeled evidence even when wording changes.
  - Output table displays both scope percentages.
- **Validation**:
  - Regression tests for known-issue pass/fail semantics.

## Sprint 5: End-to-End Verification + Baseline Reset Workflow
**Goal**: Validate behavior on loaded dataset and lock deterministic baseline.
**Demo/Validation**:
- Full gauntlet run completes with separated sections and robust modeled values.
- Baseline update workflow remains simple and repeatable.

### Task 5.1: Full run validation against loaded dataset
- **Location**: runtime artifacts under `appdata/runs/<run_id>/`
- **Description**: Execute full gauntlet and verify:
  - top recommendations include modeled percent (or clear reasons)
  - general vs close-specific sections are present
  - QEMAIL no-removal modeled impacts appear in both scopes
- **Complexity**: 4
- **Dependencies**: Sprints 1-4
- **Acceptance Criteria**:
  - Output files generated and internally consistent.
  - Known issue summary uses modeled evidence.
- **Validation**:
  - `python -m pytest -q`
  - full run + `scripts/show_actionable_results.py`

### Task 5.2: Baseline reset script update (if output contracts changed)
- **Location**: `scripts/reset_baseline.sh`, tests/snapshots as needed
- **Description**: Ensure baseline-reset flow captures new output shape and matrix drift.
- **Complexity**: 3
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - One-command baseline reset remains valid.
- **Validation**:
  - Run `scripts/reset_baseline.sh` and confirm pass.

## Testing Strategy
- Unit tests:
  - modeled percent calculation and null-reason enforcement
  - action-type scenario adapters
  - scope classifier (general vs close_specific)
  - QEMAIL removal modeling (general + close)
- Integration tests:
  - recommendation build path includes grouped scopes and modeled fields
  - known-issue evaluation uses modeled evidence
- End-to-end:
  - `python -m pytest -q`
  - full dataset gauntlet run and output verification

## Potential Risks & Gotchas
- Risk: modeled percent remains null too often.
  - Mitigation: enforce top-N quality gate + explicit reasons + adapter coverage.
- Risk: overfitting to one ERP naming pattern.
  - Mitigation: use normalized/process-inference helpers and evidence-driven classification.
- Risk: close-specific split leaks into generic recommendations.
  - Mitigation: deterministic scope classifier tied to plugin kind + metric context.
- Risk: QEMAIL check still fails due to sparse close-window evidence.
  - Mitigation: dual-scope modeling with confidence thresholds and fallback reason.

## Rollback Plan
- Keep prior recommendation rendering path behind a feature flag.
- If regressions appear:
  - disable new modeled gate
  - keep scope grouping optional
  - preserve old recommendation payload shape temporarily
- Re-run baseline tests and gauntlet before re-enabling.

## Plan Review (Self-Check)
- Added explicit no-month tuning requirement: newest data weighted most, but all history included.
- Added explicit QEMAIL removal modeling in both general and close-specific scopes.
- Added required field contract so missing modeled % cannot be silent.
- Added baseline-reset compatibility task to keep test operations practical.
