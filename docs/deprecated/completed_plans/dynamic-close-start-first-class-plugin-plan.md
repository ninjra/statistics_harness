# Plan: Dynamic Close Start First-Class Plugin

**Generated**: 2026-02-13
**Estimated Complexity**: High

## Overview
Add a new first-class analysis plugin that infers close-start dynamically per accounting month by detecting month-roll markers and backtracking from roll time using recurring process signatures. Then wire its outputs into the existing close-cycle resolver and core helpers so downstream plugins can consistently narrow to active close rows.

## Prerequisites
- Existing close-window artifacts and helpers:
  - `plugins/analysis_close_cycle_window_resolver/plugin.py`
  - `src/statistic_harness/core/close_cycle.py`
- Existing dynamic detector patterns:
  - `plugins/analysis_dynamic_close_detection/plugin.py`
- Full plugin execution harness:
  - `scripts/run_loaded_dataset_full.py`

## Sprint 1: Plugin Foundation
**Goal**: Create a first-class plugin that infers close-start dynamically from month-roll + recurring process patterns.
**Demo/Validation**:
- Run plugin directly in a pipeline run and confirm it emits non-empty findings/artifacts for known Quorum datasets.
- Verify artifacts include month-level inferred close-start timestamps and confidence/support metrics.

### Task 1.1: Scaffold plugin
- **Location**: `plugins/analysis_close_cycle_start_backtrack_v1/`
- **Description**: Add `plugin.py`, `plugin.yaml`, `config.schema.json`, and `output.schema.json`.
- **Complexity**: 4/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - Plugin is discoverable by manifest scan.
  - Plugin runs with empty dataset and returns graceful `skipped`.
- **Validation**:
  - Run plugin smoke test in unit fixture.

### Task 1.2: Implement inference algorithm
- **Location**: `plugins/analysis_close_cycle_start_backtrack_v1/plugin.py`
- **Description**:
  - Detect roll points from accounting month markers in params.
  - Build monthly recurring process signatures (frequency and day-offset near roll).
  - Backtrack from each roll to earliest contiguous window where signature coverage exceeds threshold.
  - Emit per-month inferred `close_start_dynamic` and `close_end_dynamic`.
- **Complexity**: 8/10
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Inference works with noisy months (fallback to default where confidence low).
  - Deterministic output for fixed seed/data.
- **Validation**:
  - Unit tests with synthetic month patterns.

### Task 1.3: Emit stable artifacts + metrics
- **Location**: `plugins/analysis_close_cycle_start_backtrack_v1/plugin.py`
- **Description**: Write `close_windows.csv` + JSON artifacts with confidence, signature support, month coverage, and fallback reasons.
- **Complexity**: 5/10
- **Dependencies**: Task 1.2
- **Acceptance Criteria**:
  - Artifact schema is stable and machine-consumable.
  - Findings include explicit reason when backtracking is not reliable.
- **Validation**:
  - Contract tests for expected columns and field types.

### Task 1.4: Enforce deterministic inference behavior
- **Location**: `plugins/analysis_close_cycle_start_backtrack_v1/plugin.py`, `tests/plugins/test_analysis_close_cycle_start_backtrack_v1.py`
- **Description**: Ensure any sampling/tie-break logic uses run-seeded RNG and deterministic ordering.
- **Complexity**: 5/10
- **Dependencies**: Task 1.2
- **Acceptance Criteria**:
  - Same input + `run_seed` yields identical artifacts/findings.
- **Validation**:
  - Repeat-run determinism test with fixed seed.

## Sprint 2: Integration for Downstream Narrowing
**Goal**: Make downstream plugins consume the new inferred windows by default.
**Demo/Validation**:
- Existing close-cycle plugins automatically read backtracked windows when available.
- Row narrowing metrics reflect dynamic windows source.

### Task 2.1: Integrate into close-cycle resolver
- **Location**: `plugins/analysis_close_cycle_window_resolver/plugin.py`, `plugins/analysis_close_cycle_window_resolver/plugin.yaml`
- **Description**:
  - Add dependency on `analysis_close_cycle_start_backtrack_v1`.
  - Load new plugin artifact when present and promote inferred start/end into resolver output.
  - Preserve fallback to current default logic.
- **Complexity**: 7/10
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Resolver outputs include backtrack source markers.
  - No regression when backtrack artifact missing.
- **Validation**:
  - Integration test with and without backtrack artifact.

### Task 2.2: Update core close-cycle loader precedence
- **Location**: `src/statistic_harness/core/close_cycle.py`
- **Description**: Prefer backtrack-informed resolver artifacts for mask generation; keep compatibility with existing CSV format.
- **Complexity**: 5/10
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - `resolve_close_cycle_masks()` uses inferred windows where available.
- **Validation**:
  - Unit test on mask generation using synthetic timestamps.

### Task 2.3: Expose shared helper for downstream narrowing
- **Location**: `src/statistic_harness/core/close_cycle.py`
- **Description**: Add one central helper/path for “active close rows” so downstream plugins use a consistent narrowing source.
- **Complexity**: 6/10
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - Shared helper returns masks + source metadata + fallback reasons.
- **Validation**:
  - Unit tests cover backtrack present/absent/noisy fallback.

### Task 2.4: Broaden downstream consumption
- **Location**: close-window-aware plugins (initially):
  - `plugins/analysis_close_cycle_contention/plugin.py`
  - `plugins/analysis_close_cycle_uplift/plugin.py`
  - `plugins/analysis_queue_delay_decomposition/plugin.py`
  - `plugins/analysis_close_cycle_duration_shift/plugin.py`
  - `plugins/analysis_close_cycle_capacity_model/plugin.py`
  - `plugins/analysis_close_cycle_capacity_impact/plugin.py`
  - `plugins/analysis_close_cycle_revenue_compression/plugin.py`
- **Description**: Ensure metrics explicitly report row narrowing source as backtrack/resolver/default and row counts by source.
- **Complexity**: 7/10
- **Dependencies**: Task 2.3
- **Acceptance Criteria**:
  - Most close-cycle-critical plugins consume narrowed rows from shared helper.
  - Metrics expose `close_cycle_dynamic_available`, `close_cycle_rows_dynamic`, and source marker.
- **Validation**:
  - Existing plugin tests updated to assert source markers and row count behavior.

## Sprint 3: Validation, Full Gauntlet, and Baseline-First Reporting
**Goal**: Prove correctness and release safety with full runs.
**Demo/Validation**:
- Full test suite passes.
- Full gauntlet runs complete for baseline + synthetic datasets.
- Results reported as deltas vs baseline dataset.

### Task 3.1: Add tests
- **Location**:
  - `tests/plugins/test_analysis_close_cycle_start_backtrack_v1.py` (new)
  - `tests/plugins/test_close_cycle_contention.py`
  - `tests/plugins/test_queue_delay_decomposition.py`
  - `tests/plugins/test_close_cycle_capacity_model.py`
  - `tests/plugins/test_close_cycle_capacity_impact.py`
  - `tests/plugins/test_close_cycle_revenue_compression.py`
- **Description**: Add unit + integration tests for inference, fallback, and downstream mask usage.
- **Complexity**: 8/10
- **Dependencies**: Sprints 1-2
- **Acceptance Criteria**:
  - Deterministic tests cover happy path + fallback path + missing marker path.
  - Artifacts include confidence/reason fields in both strong and fallback modes.
  - Generated `report.json` remains schema-valid against `docs/report.schema.json`.
- **Validation**:
  - `python -m pytest -q`

### Task 3.2: Wire plugin into default execution surfaces
- **Location**:
  - `scripts/run_loaded_dataset_full.py`
  - plugin registry/allowlist paths used by release runs
- **Description**: Ensure the new plugin is included in default full gauntlet execution and not omitted by planner/allowlist behavior.
- **Complexity**: 4/10
- **Dependencies**: Task 2.4
- **Acceptance Criteria**:
  - New plugin appears in full run `plugin_executions`.
- **Validation**:
  - DB check for plugin presence in full run.

### Task 3.3: Execute full gauntlet runs
- **Location**: runtime artifacts under `appdata/runs/*`
- **Description**: Run full plugin gauntlet for:
  - Baseline dataset `3246cc7cd7d57a317ddc05e80e6f6f5bfe7f50deb0ee7af8db50d04bae180e1a`
  - Synthetic dataset `de7c1da5a4ea6e8c684872d7857bb608492f63a9c7e0b7ca014fa0f093a88e66`
- **Complexity**: 6/10
- **Dependencies**: Tasks 3.1-3.2
- **Acceptance Criteria**:
  - Runs complete (no abort) and emit report + recommendations.
  - Coverage summary includes ok/skipped/error counts.
  - Dynamic vs default row-narrowing delta checks are reported before merge.
- **Validation**:
  - DB query over `plugin_executions` and `plugin_results_v2`.

### Task 3.4: Baseline-referenced readout
- **Location**: `docs/first_release_execution_readout.md` (update)
- **Description**: Add section showing inferred close-start quality and plugin narrowing impact as deltas vs baseline; include report/schema compatibility confirmation and artifact field notes for downstream readers.
- **Complexity**: 4/10
- **Dependencies**: Task 3.3
- **Acceptance Criteria**:
  - Report explicitly states baseline-first comparison.
- **Validation**:
  - Manual verification of run IDs + metric deltas.

## Testing Strategy
- Unit test inference logic with synthetic month-roll fixtures.
- Integration test resolver + close_cycle helper precedence.
- Regression tests for existing close-cycle plugins to ensure no behavior break.
- Full gauntlet execution on baseline and synthetic datasets.

## Potential Risks & Gotchas
- Risk: Month marker sparsity can make inferred starts unstable.
  - Mitigation: confidence thresholds and explicit fallback to default start day.
- Risk: Over-narrowing rows may hide true issues.
  - Mitigation: emit both dynamic and default row counts + source markers.
- Risk: Artifact schema drift could break existing loaders.
  - Mitigation: keep resolver CSV contract stable and add contract tests.
- Risk: Runtime growth on large datasets.
  - Mitigation: bounded backtracking window and process-signature caps.

## Rollback Plan
- Disable new plugin via registry/allowlist.
- Revert resolver dependency to current logic.
- Keep backward-compatible artifact loading in `close_cycle.py`.
