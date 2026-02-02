# Plan: Queue/Capacity Plugins + Evaluation Parity

**Generated**: 2026-02-01
**Estimated Complexity**: High

## Overview
Implement 10 separate, deterministic analysis plugins that reproduce the working queue/capacity methods on arbitrary ERP event-log data, without pre-known column names. Add dynamic field inference, measured vs modeled labeling, and evaluation parity checks so the harness can validate that plugin outputs match expected results for Quorum Upstream (and later Enertia). All plugins must be independent, skip gracefully when required fields cannot be inferred, and emit explicit "not applicable" + "error detail" findings instead of hard-failing.

## Decisions (from user)
- Role inference: auto-select best match and proceed (no manual override required to run).
- Evaluation strictness: fail on unexpected findings by default.
- ERP default: "unknown" with its own known-issues set; can be updated later.

## Prerequisites
- Current schema + plugin system (src/ + plugins/ layouts)
- Existing dynamic dataset ingestion into SQLite
- Ability to run tests: `python -m pytest -q`
- Sample event-log fixture(s) to validate outputs (Quorum Upstream test file)

## Sprint 1: Field Inference + Evaluation Enhancements
**Goal**: Infer event-log roles from arbitrary headers and extend evaluation so numeric outputs can be checked for parity.
**Demo/Validation**:
- Run the new inference plugin on a fixture and see inferred roles with confidence.
- Run evaluation on a report using a new expected-metrics payload and see pass/fail reasons.

### Task 1.1: Add event-log field inference plugin
- **Location**: `plugins/profile_eventlog/`, `src/statistic_harness/core/storage.py`, `src/statistic_harness/core/migrations.py`
- **Description**: Create a profile plugin that scans dataset columns and stores ranked role candidates (queue time, start time, end time, process id/name, module code, user id, dependency id, master/sequence id, host/worker id, status). Use name heuristics + value pattern checks (timestamp parse rate, cardinality, monotonicity hints) and write results to a new `dataset_role_candidates` table plus set `dataset_columns.role` for the best-scoring role per field.
- **Complexity**: 8
- **Dependencies**: None
- **Acceptance Criteria**:
  - Plugin outputs a deterministic role map with scores + reasons.
  - If roles are ambiguous, output includes multiple candidates and marks confidence below threshold.
- **Validation**:
  - Unit tests with synthetic columns; assert deterministic role assignments.

### Task 1.2: Add ERP type metadata + known-issues scoping
- **Location**: `src/statistic_harness/core/storage.py`, `src/statistic_harness/core/migrations.py`, `src/statistic_harness/ui/server.py`, `src/statistic_harness/ui/templates/known_issues.html`, `src/statistic_harness/ui/templates/wizard.html`
- **Description**: Add `erp_type` to projects and allow known-issues sets to be scoped by ERP type (default "unknown"). When evaluating, prefer known-issues for the project's ERP type; fall back to upload hash if ERP type is missing.
- **Complexity**: 6
- **Dependencies**: None
- **Acceptance Criteria**:
  - Known issues can be saved/loaded by ERP type and project.
  - Evaluation picks ERP-scoped issues automatically when available.
- **Validation**:
  - Storage tests for ERP-scoped sets; UI smoke test shows ERP type.

### Task 1.3: Extend evaluation with numeric expected metrics
- **Location**: `src/statistic_harness/core/evaluation.py`, `docs/evaluation.md`, `docs/report.schema.json`
- **Description**: Add `expected_metrics` to ground truth: list of {plugin_id, metric, value, tolerance}. Support numeric tolerance (abs/rel). Validate plugin metrics by exact key path (e.g., `eligible_wait.p95`). Default evaluation behavior is strict: unexpected findings fail unless explicitly disabled in ground truth.
- **Complexity**: 5
- **Dependencies**: None
- **Acceptance Criteria**:
  - Evaluation fails when a metric is missing or outside tolerance.
  - Clear messages identify plugin/metric mismatches.
- **Validation**:
  - Unit tests for metric comparisons and tolerance behavior.

### Task 1.4: Introduce measured vs modeled labeling contract
- **Location**: `docs/report.schema.json`, `src/statistic_harness/core/report.py`, `plugins/*/output.schema.json`
- **Description**: Standardize a `measurement_type` field for findings/metrics (values: `measured`, `modeled`, `not_applicable`, `error`). Require each plugin to label outputs accordingly.
- **Complexity**: 4
- **Dependencies**: None
- **Acceptance Criteria**:
  - Schema validates all plugin outputs with measurement labels.
- **Validation**:
  - Schema tests + plugin unit tests with both measured and modeled cases.

### Task 1.5: Add manual role overrides (optional per project)
- **Location**: `src/statistic_harness/core/storage.py`, `src/statistic_harness/core/migrations.py`, `src/statistic_harness/ui/templates/`, `src/statistic_harness/ui/server.py`
- **Description**: Provide a per-project override table so you can explicitly map key roles (queue/start/end/process/dependency/master/user/module/host) when inference is ambiguous. Overrides must be deterministic and preserved in DB.
- **Complexity**: 6
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - UI can save overrides and they are used by analysis plugins.
- **Validation**:
  - Unit test: overrides take precedence over inferred roles.

## Sprint 2: Core Queue/Delay Plugins (1-5, 9, 10)
**Goal**: Implement the first seven plugins and verify deterministic parity on the Quorum test file.
**Demo/Validation**:
- Run the plugin suite on fixture; evaluation passes with expected metrics/findings.

### Task 2.1: Queue delay decomposition plugin
- **Location**: `plugins/analysis_queue_delay_decomposition/`
- **Description**: Compute total wait (START-QUEUE), split into prereq wait (DEP_END-QUEUE) and eligible wait (START-DEP_END). Emit measured findings per process/module/user and tail stats.
- **Complexity**: 7
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Produces eligible/prereq buckets with deterministic counts and percentiles.
  - Emits not_applicable when dependency or timestamp roles missing.
- **Validation**:
  - Unit test fixture with known splits.

### Task 2.2: Dependency-resolution join plugin
- **Location**: `plugins/analysis_dependency_resolution_join/`
- **Description**: Infer dependency key, join to dependency end time, compute START-DEP_END. Report near-zero starts to show dependency dominance.
- **Complexity**: 6
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Emits distribution of dependency lag and proportion <=2s.
- **Validation**:
  - Synthetic fixture test verifying lag distribution.

### Task 2.3: Standalone vs sequence classification plugin
- **Location**: `plugins/analysis_sequence_classification/`
- **Description**: Classify rows by presence of dependency/master pointers. Output counts and eligible-wait stats per class.
- **Complexity**: 5
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Emits counts/percentiles per class with evidence.
- **Validation**:
  - Unit test on synthetic rows with/without deps.

### Task 2.4: Threshold-based tail isolation plugin
- **Location**: `plugins/analysis_tail_isolation/`
- **Description**: Filter eligible-wait > configurable threshold (default 60s). Attribute tail to process/module/user/sequence.
- **Complexity**: 5
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Emits tail segments with evidence row IDs.
- **Validation**:
  - Fixture test verifying threshold effect.

### Task 2.5: Percentile analysis plugin
- **Location**: `plugins/analysis_percentile_analysis/`
- **Description**: Compute p50/p95/p99 for eligible wait and completion time per process/module.
- **Complexity**: 4
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Emits percentiles + dominant tail contributors.
- **Validation**:
  - Numeric tolerance tests in evaluation.

### Task 2.6: Attribution analysis plugin
- **Location**: `plugins/analysis_attribution/`
- **Description**: Aggregate eligible wait and tail counts by PROCESS_ID, MODULE_CD, USER_ID with deterministic ranking.
- **Complexity**: 4
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Deterministic ranking with evidence slices.
- **Validation**:
  - Fixture test ensures top contributor is stable.

### Task 2.7: Determinism discipline plugin
- **Location**: `plugins/analysis_determinism_discipline/`
- **Description**: Inspect all plugin outputs and emit a summary finding validating measurement_type usage. Flags any modeled output lacking explicit assumptions or any missing label.
- **Complexity**: 5
- **Dependencies**: Task 1.4
- **Acceptance Criteria**:
  - Emits one summary finding and error findings for violations.
- **Validation**:
  - Unit test with mocked plugin outputs.

## Sprint 3: Sequence + Concurrency + Capacity Modeling (6-8)
**Goal**: Implement the remaining plugins and ensure modeled outputs are clearly separated from measured facts.
**Demo/Validation**:
- Run all plugins on the Quorum fixture and pass evaluation.

### Task 3.1: Chain makespan analysis plugin
- **Location**: `plugins/analysis_chain_makespan/`
- **Description**: Use MASTER_PROCESS_QUEUE_ID (or inferred sequence id) to compute makespan, runtime sum, idle gaps, and critical path effects. Emit evidence rows.
- **Complexity**: 7
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Outputs per-sequence makespan and idle-gap metrics.
- **Validation**:
  - Synthetic fixture with known makespan.

### Task 3.2: Concurrency reconstruction plugin
- **Location**: `plugins/analysis_concurrency_reconstruction/`
- **Description**: Reconstruct concurrent running counts using START/END overlap per host/QPEC. Output observed concurrency distribution vs configured caps if present.
- **Complexity**: 7
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Emits concurrency time series summary and peak concurrency.
- **Validation**:
  - Fixture with known overlaps.

### Task 3.3: Capacity-scaling scenario modeling plugin
- **Location**: `plugins/analysis_capacity_scaling/`
- **Description**: Apply proportional scaling to the eligible-wait bucket only (modeled). Support configurable scale factor; default derived from host count detected. Output modeled reduction stats.
- **Complexity**: 6
- **Dependencies**: Task 2.1, Task 3.2
- **Acceptance Criteria**:
  - Modeled outputs are labeled and include explicit assumptions.
- **Validation**:
  - Evaluation checks modeled metrics with tolerance.

### Task 3.4: Evaluation parity suite for Quorum Upstream
- **Location**: `tests/fixtures/`, `tests/test_evaluation.py`, `tests/plugins/` for new plugin tests
- **Description**: Add a Quorum-style fixture and expected metrics/findings that represent the known results (qemail close-cycle contention, dependency waits, etc.). Ensure evaluation passes only when plugins reproduce them.
- **Complexity**: 6
- **Dependencies**: Sprint 2, Sprint 3
- **Acceptance Criteria**:
  - Evaluation passes with expected metrics and fails when metrics drift.
- **Validation**:
  - `python -m pytest -q` passes with new fixtures.

## Testing Strategy
- Unit tests per new plugin using synthetic fixtures.
- Integration test running the full pipeline on a Quorum-style fixture and validating report schema + evaluation parity.
- Migration tests for ERP-scoped known issues and role candidates tables.

## Potential Risks & Gotchas
- **Ambiguous column inference** could cause mis-classification. Mitigation: store multiple candidates with confidence, auto-select best match but emit a warning finding when confidence is below threshold.
- **Mixed timezones/invalid timestamps** could distort waits. Mitigation: parse failures tracked; emit a warning finding with counts; allow per-project override later.
- **Huge datasets** could make overlap reconstruction expensive. Mitigation: chunked SQL with indexes; report when sampling is used.

## Rollback Plan
- Disable new plugins by removing them from the autoplanner/selection list and leave existing pipeline intact.
- Revert migration by leaving new tables unused (no destructive changes to existing tables).
