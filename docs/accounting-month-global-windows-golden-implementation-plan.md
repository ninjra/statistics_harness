# Plan: Accounting-Month Global Windows Golden Implementation

**Generated**: 2026-02-18  
**Estimated Complexity**: High

## Overview
Make `accounting month` (not calendar month) the global time-window contract across the entire repository, and require every recommendation row to carry comparable modeled metrics for:
- full accounting month (dynamically resolved per month roll),
- close static window (configured/static bounds),
- close dynamic window (data-inferred bounds).

This plan prioritizes deterministic, comparable recommendation scoring and hard failure visibility so no plugin silently skips meaningful output.

## Prerequisites
- Python 3.11 environment with repo dependencies installed.
- Existing close-window plugins available:
  - `analysis_close_cycle_window_resolver`
  - `analysis_close_cycle_start_backtrack_v1`
  - `analysis_dynamic_close_detection`
- Baseline + synthetic datasets available in repo docs:
  - `docs/proc_log_synth_custom_issues_6mo_sheet_v2.xlsx`
  - `docs/proc_log_synth_custom_issues_8mo_sheet_v4_2.xlsx`
- Full test suite runnable via `python -m pytest -q`.

## Skills Matrix (Implementation Guidance for Codex CLI)
Use these skills by sprint/section.

1. `plan-harder`
- Why: structure phased, atomic execution and dependencies.
- Where used: this planning artifact and sprint sequencing.

2. `python-testing-patterns`
- Why: enforce unit/integration contract tests for new window + metrics schema.
- Where used: sprints 1-4 test authoring.

3. `testing`
- Why: run full gauntlet, plugin-wide validation, and failure semantics.
- Where used: sprints 4-6 verification.

4. `deterministic-tests-marshal`
- Why: guarantee stable outputs and cache correctness across reruns.
- Where used: sprint 5 determinism/caching checks.

5. `discover-observability`
- Why: identify missing telemetry and failure visibility points quickly.
- Where used: sprint 4 observability gap discovery.

6. `python-observability`
- Why: instrument pipeline/plugin runtime, window-coverage, and hard-fail counters.
- Where used: sprint 4 implementation.

7. `observability-engineer`
- Why: convert instrumentation into release-grade reliability controls and run evidence.
- Where used: sprint 6 release hardening.

8. `config-matrix-validator`
- Why: validate config/env matrix for window modes, fail policies, and known-issues mode.
- Where used: sprint 5 config contract checks.

9. `shell-lint-ps-wsl`
- Why: command hygiene and shell-safe execution in all runbooks/scripts.
- Where used: all sprint command/runbook updates.

## Sprint 1: Canonical Accounting-Month Window Contract
**Goal**: Introduce one shared source of truth for accounting-month, static-close, and dynamic-close windows, derived from month-roll signals and backtracking.
**Demo/Validation**:
- Build canonical window artifact for a run and inspect rows by `accounting_month`.
- Verify static/dynamic/full-accounting fields exist for each month.

### Task 1.1: Create Global Window Contract Module
- **Location**: `src/statistic_harness/core/accounting_windows.py` (new), `src/statistic_harness/core/close_cycle.py`
- **Description**: Add a canonical resolver that returns per-accounting-month records with:
  - `accounting_month`
  - `accounting_month_start_ts`, `accounting_month_end_ts`
  - `close_static_start_ts`, `close_static_end_ts`
  - `close_dynamic_start_ts`, `close_dynamic_end_ts`
  - `source_plugin`, `confidence`, `fallback_reason`
- **Complexity**: 8
- **Dependencies**: None
- **Acceptance Criteria**:
  - Resolver reads preferred plugin artifacts in deterministic order.
  - Full accounting month is dynamically derived from roll events (no calendar-month fallback except explicit `N/A` reason).
  - Provides deterministic fallback record when evidence is insufficient.
- **Validation**:
  - New unit tests for month roll around day ~5 behavior and wrap-around.

### Task 1.2: Persist Canonical Window Artifact Per Run
- **Location**: `src/statistic_harness/core/pipeline.py`, `runs/<run_id>/artifacts/...` generation path
- **Description**: Write a stable run artifact (CSV + JSON) for canonical windows used by all downstream plugins/reports.
- **Complexity**: 6
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Artifact is always present when timestamp data exists.
  - Missing evidence produces deterministic `N/A` + reason, not silent skip.
- **Validation**:
  - Integration test checks artifact existence and schema.

### Task 1.3: Deprecate Calendar-Month Semantics in Core Helpers
- **Location**: `src/statistic_harness/core/close_cycle.py`, `src/statistic_harness/core/report.py`
- **Description**: Replace calendar-month assumptions with accounting-month resolver outputs wherever month cohorts/windows are computed.
- **Complexity**: 7
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - No recommendation/report code uses calendar-month-only slicing for core metrics.
  - Helper names/docs explicitly call out accounting month.
- **Validation**:
  - Search-based guard test asserting no forbidden calendar-month logic patterns in critical metric paths.

## Sprint 2: Global Recommendation Metric Contract (3 Windows, Comparable Everywhere)
**Goal**: Every recommendation row has comparable dimensionless efficiency gain and modeled delta hours for all required windows.
**Demo/Validation**:
- Inspect recommendation JSON row and confirm all required metric columns are present.
- Verify single-process and grouped/sequence recommendations both populate same metric fields.

### Task 2.1: Define Required Per-Window Metric Fields
- **Location**: `docs/4_pillars_scoring_spec.md`, `docs/report.schema.json`
- **Description**: Formalize required fields per recommendation row:
  - `delta_hours_accounting_month`, `efficiency_gain_accounting_month`
  - `delta_hours_close_static`, `efficiency_gain_close_static`
  - `delta_hours_close_dynamic`, `efficiency_gain_close_dynamic`
  - optional companions: manual touches, contention reductions, each window-scoped.
- **Complexity**: 5
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Schema validation fails if any recommendation row lacks required 3-window metric pair.
  - `N/A` only allowed with explicit `na_reason` + `metric_confidence`.
- **Validation**:
  - Schema tests + report contract tests.

### Task 2.2: Implement Unified Metric Calculator
- **Location**: `src/statistic_harness/core/report.py`, `src/statistic_harness/core/report_v2_utils.py`
- **Description**: Central function computes metrics for all recommendation types using identical formulas:
  - `delta_hours_W = modeled saved hours in window W`
  - `efficiency_gain_W = delta_hours_W / baseline_hours_W`
- **Description (required denominator contract)**:
  - Persist `baseline_hours_accounting_month`, `baseline_hours_close_static`, `baseline_hours_close_dynamic` on each row.
  - If `baseline_hours_W` is missing/zero, emit deterministic `N/A` for `efficiency_gain_W` with `na_reason`, never implicit null.
- **Complexity**: 8
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Sequence/group/timing recommendations use same denominator/basis logic as single-process recommendations.
  - Field naming and units consistent across plugin outputs.
- **Validation**:
  - Unit tests for multiple recommendation classes proving identical formula contract.

### Task 2.3: Enforce Ranking Based on Required Comparable Fields
- **Location**: `src/statistic_harness/core/report.py`
- **Description**: Rework ranking to prioritize comparable client-value impact:
  1) `delta_hours_close_dynamic` desc  
  2) `efficiency_gain_close_dynamic` desc  
  3) `delta_hours_accounting_month` desc  
  4) confidence/tie-breakers
- **Complexity**: 6
- **Dependencies**: Task 2.2, Task 3.2
- **Acceptance Criteria**:
  - High-impact close-window items are not buried by class heuristics.
  - Ranking remains deterministic.
- **Validation**:
  - Snapshot tests on baseline run recommendations.

### Task 2.4: Human Report Parity (`report.md`)
- **Location**: `src/statistic_harness/core/plain_report.py`, `src/statistic_harness/core/report.py`, report template/render sections
- **Description**: Ensure human-readable output mirrors JSON contract and shows all 3 windows per recommendation with clear labeling and `N/A` reasons.
- **Complexity**: 5
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - `report.md` includes all required per-window metrics for each surfaced recommendation.
  - Human report and JSON values match for sampled rows.
- **Validation**:
  - Integration test compares rendered markdown metrics vs report JSON payload.

## Sprint 3: Plugin-Wide Adoption (Normalized Layer + Large-Data Safe Access)
**Goal**: Ensure plugin logic reads normalized layer consistently and consumes canonical windows without calendar-month drift.
**Demo/Validation**:
- Plugin matrix report showing each plugin’s data source, window use, and compliance status.

### Task 3.1: Build Plugin Compliance Inventory
- **Location**: `docs/plugin_data_access_matrix.json`, `docs/plugin_data_access_matrix.md`, `scripts/` audit helper
- **Description**: Produce deterministic inventory for each plugin:
  - normalized-layer access Y/N
  - accounting-window contract usage Y/N
  - streaming-safe access Y/N
  - actionable output or explicit plain-English non-actionable reason
- **Complexity**: 7
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Every plugin has an entry and compliance status.
  - No hidden “skipped”; failure or `N/A` with reason only.
- **Validation**:
  - CI check ensures full inventory coverage.

### Task 3.2: Update Non-Compliant Plugins to Canonical Windows
- **Location**: `plugins/**/plugin.py`, `src/statistic_harness/core/stat_plugins/*.py`
- **Description**: Refactor plugin window filtering calls to use canonical accounting-window API.
- **Complexity**: 9
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - All month/window filters point to accounting-month resolver outputs.
  - Legacy calendar month filtering removed from actionable plugin paths.
- **Validation**:
  - Plugin unit tests + compliance inventory diff to zero non-compliant rows.

### Task 3.3: Sequence/Cluster Recommendation Specificity Upgrade
- **Location**: `src/statistic_harness/core/report.py`, sequence-related plugin modules
- **Description**: Ensure grouped recommendations explicitly identify exact process IDs, user/manual-run patterns, and pathway scope while still emitting unified window metrics.
- **Complexity**: 8
- **Dependencies**: Task 3.2
- **Acceptance Criteria**:
  - No generic “improve everything” recommendations.
  - Grouped recs include exact target set and per-window comparable metrics.
- **Validation**:
  - Plain-report tests assert actionable wording and explicit targets.

## Sprint 4: Hard-Fail Reliability + Observability (No Silent Failure)
**Goal**: Detect structural plugin failures before gauntlet and make silent degradation impossible.
**Demo/Validation**:
- Run pipeline with induced schema mismatch and observe explicit hard-fail telemetry.

### Task 4.1: Preflight Structural Gate
- **Location**: `src/statistic_harness/core/pipeline.py`, `src/statistic_harness/core/plugin_runner.py`, preflight script
- **Description**: Add preflight that validates required columns/mappings/window artifacts before plugin execution.
- **Complexity**: 7
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Structural mismatches fail plugin with explicit reason code.
  - Run continues per policy but final run status is failed when any plugin fails.
- **Validation**:
  - Integration tests for missing-column and malformed-window scenarios.

### Task 4.2: Runtime Telemetry for Plugin Health and Lag
- **Location**: `src/statistic_harness/core/plugin_runner.py`, `src/statistic_harness/core/pipeline.py`, evidence scripts/docs
- **Description**: Emit deterministic metrics:
  - plugin runtime, peak memory hints, window coverage %, metric-fill coverage %, failure reason counts.
- **Complexity**: 6
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Silent failure rate = 0 by contract checks.
  - Metrics are written to run artifacts and summarized in release evidence.
- **Validation**:
  - Observability tests and release evidence snapshot checks.

## Sprint 5: Determinism, Caching, and Config Matrix
**Goal**: Preserve deterministic outputs and avoid unnecessary reruns while respecting plugin/dataset/window contract changes.
**Demo/Validation**:
- Re-run unchanged dataset/plugins and show cache reuse.
- Modify one plugin and show targeted invalidation only.

### Task 5.1: Window-Contract Versioned Cache Key
- **Location**: `src/statistic_harness/core/pipeline.py`, `src/statistic_harness/core/dataset_cache.py`
- **Description**: Include accounting-window contract version/hash in cache key derivation.
- **Complexity**: 6
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - Cache hit only when dataset hash + plugin effective code hash + window-contract hash match.
- **Validation**:
  - Determinism/caching tests across repeated runs.

### Task 5.2: Config Matrix Validation
- **Location**: `docs/implementation_matrix.json`, config matrix checks under scripts/tests
- **Description**: Validate matrix for:
  - known issues on/off,
  - strict fail policy,
  - window modes enabled,
  - SQL-assist hard-fail behavior.
- **Complexity**: 5
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - Invalid combinations are blocked with explicit errors.
- **Validation**:
  - Matrix validator test suite.

## Sprint 6: Full Gauntlet, Diff Evidence, and Release Readiness
**Goal**: Produce final evidence that new contract improves comparability and preserves/raises actionable quality.
**Demo/Validation**:
- Full-gauntlet runs on baseline + synthetics complete with no silent failures.
- Before/after diff report clearly shows changed recommendation metrics and ordering.

### Task 6.1: Baseline + Synthetic Full Runs Under New Contract
- **Location**: run scripts + `docs/release_evidence/*`
- **Description**: Execute complete pipeline for:
  - baseline real dataset
  - synthetic 6mo
  - synthetic 8mo
  with strict fail policy and canonical windows.
- **Complexity**: 7
- **Dependencies**: Sprints 1-5
- **Acceptance Criteria**:
  - Every recommendation row has all 3 required metric pairs or deterministic `N/A` reason.
  - No plugin marked skipped as success.
- **Validation**:
  - Contract verifier + schema validator + full test suite pass.

### Task 6.2: New-vs-Old Recommendation Diff Pack
- **Location**: `docs/release_evidence/*.json`, `docs/release_evidence/*.md`
- **Description**: Generate diff pack comparing previous baseline plugin run vs new contract run:
  - ordering changes,
  - metric changes per recommendation,
  - actionable-result quality changes.
- **Complexity**: 6
- **Dependencies**: Task 6.1
- **Acceptance Criteria**:
  - Clear per-plugin and per-recommendation deltas with reason annotations.
- **Validation**:
  - Deterministic regeneration test for diff artifacts.

## Testing Strategy
- Unit:
  - canonical accounting window resolution (roll detection, fallback reasons, confidence).
  - unified metric calculation across recommendation classes.
  - ranking/tie-break determinism.
  - denominator safety tests for `baseline_hours_W` and deterministic `N/A` outputs.
- Integration:
  - full pipeline report generation with schema validation.
  - markdown (`report.md`) and JSON (`report.json`) metric parity checks.
  - plugin preflight failures and explicit failed run state.
  - plugin inventory coverage completeness.
- End-to-end:
  - full gauntlet on baseline + two synthetics.
  - before/after diff evidence reproducibility.
- Determinism:
  - repeated runs with same `run_seed`, same dataset hash, same plugin hash yield stable outputs.

## Potential Risks & Gotchas
1. Accounting-month markers missing or inconsistent in some datasets.
- Mitigation: deterministic `N/A` window row with explicit fallback reason and confidence downgrade; never silent skip.

2. Legacy calendar-month assumptions hidden in SQL snippets/plugins.
- Mitigation: repository-wide lint/check gate for forbidden calendar-month-only filters in core metric paths.

3. Sequence recommendations remain generic.
- Mitigation: hard contract for exact target IDs and manual/user burden metrics in recommendation serializer.

4. Ranking regressions from mixed old/new fields.
- Mitigation: single ranking function keyed only to required per-window metrics + confidence.

5. Runtime overhead and host lag.
- Mitigation: bounded concurrency, telemetry-first profiling, and low-priority execution policy.

## Rollback Plan
1. Keep new window contract behind explicit version flag during rollout.
2. Preserve previous recommendation ranking path for emergency fallback.
3. Revert by feature flag if schema consumers break, while keeping artifact generation for debugging.
4. Maintain migration notes in release evidence so rollback state is auditable.

## Definition of Done
- Global repository behavior treats accounting month as the only month concept for recommendation scoring windows.
- Every recommendation row contains comparable dimensionless + delta-hours metrics for accounting month, close static, and close dynamic windows.
- Full gauntlet passes with explicit fail semantics and no silent failure modes.
- Release evidence includes deterministic before/after baseline comparison under the new contract.
