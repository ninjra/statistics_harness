# Plan: Four Pillars Accuracy + Streaming + Actionability

**Generated**: 2026-02-26  
**Estimated Complexity**: High

## Overview
Implement a strict, deterministic plugin pipeline that improves all four pillars:

- `performant`: bounded memory and predictable runtime on large normalized data
- `accurate`: correct targeting/windowing and valid modeled deltas
- `secure`: fail-closed behavior, no silent skips, strict dependency gates
- `citable`: evidence-linked outputs with traceable recommendation lineage

The execution path is contract-first, then runtime telemetry, then recommendation
quality and ranking. Final release output is blocked unless all contracts and tests
pass, including full `pytest -q`.

## Prerequisites
- Repo entrypoint artifacts:
  - `REPO_TOC.md`
  - `docs/4_pillars_scoring_spec.md`
  - `docs/optimal_4pillars_execution_path.md`
  - `docs/plugin_data_access_matrix.md`
  - `docs/plugins_functionality_matrix.md`
- Existing contracts:
  - `docs/plugin_validation_contract.schema.json`
  - `docs/report.schema.json`
  - `docs/frozen_plugin_surfaces.contract.json`
- Test gate:
  - `python -m pytest -q`

## Skills and Usage Plan
- `plan-harder`: sprint and task decomposition, sequencing, and risk coverage.
- `python-testing-patterns` + `testing`: deterministic unit/integration/contract
  coverage for every changed plugin path.
- `deterministic-tests-marshal`: chunk-size invariance and repeated-run drift checks.
- `discover-observability` + `observability-engineer` + `python-observability`:
  plugin runtime telemetry and run-level evidence instrumentation.
- `resource-budget-enforcer`: CPU/RAM contracts and bounded-memory assertions.
- `config-matrix-validator`: normalized schema alignment and cross-dataset mapping.
- `evidence-trace-auditor`: recommendation-to-evidence trace chain validation.
- `shell-lint-ps-wsl`: one-line, shell-safe command discipline.

## Sprint 1: Contracts and Frozen Surfaces
**Goal**: Eliminate ambiguous plugin states and lock stable interfaces.  
**Demo/Validation**:
- Contract validation succeeds for all plugin outputs.
- No plugin emits `skip` or empty result payload.

### Task 1.1: Enforce canonical plugin outcome contract
- **Location**: `src/statistic_harness/core/pipeline.py`,
  `src/statistic_harness/core/models.py`,
  `docs/plugin_validation_contract.schema.json`
- **Description**: Require every plugin to emit one of:
  `pass_actionable`, `pass_valid_low_signal`, `na_with_reason`, `failed`.
- **Complexity**: 7
- **Dependencies**: None
- **Acceptance Criteria**:
  - `skip` status removed from executable flow.
  - Canonical enum only: `pass_actionable`, `pass_valid_low_signal`,
    `na_with_reason`, `failed`.
  - `na_with_reason` must include deterministic reason code from a closed enum
    plus fallback result payload.
- **Validation**:
  - Schema tests with pass/fail fixtures.
  - Integration test proving pipeline completes with plugin failures and still
    emits `report.md`, `report.json`, and deterministic error summaries.

### Task 1.2: Freeze stable plugin surfaces
- **Location**: `docs/frozen_plugin_surfaces.contract.json`,
  `docs/frozen_surfaces_contract.md`,
  `tests/test_frozen_surfaces_contract.py`
- **Description**: Lock plugin input/output contracts for stable plugins and detect
  unapproved interface drift.
- **Complexity**: 6
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Contract includes hash/check metadata for stable surfaces.
  - CI fails on unexpected surface changes.
- **Validation**:
  - Contract drift test.

### Task 1.3: Hard-fail missing core dependencies
- **Location**: `src/statistic_harness/core/sql_assist.py`,
  `src/statistic_harness/core/plugin_runtime.py`,
  `tests/core/test_fail_closed_dependency_wiring.py`
- **Description**: Convert schema/sql-assist missing states to explicit failures.
- **Complexity**: 5
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - No silent fallback when required dependency is missing.
  - Run summary records dependency failure count.
- **Validation**:
  - Dependency-missing integration tests.

## Sprint 2: Normalized-Layer Streaming Correctness
**Goal**: Prove plugins read from normalized data correctly and stream safely.  
**Demo/Validation**:
- Streaming contract audit file generated per run.
- Chunk-invariance checks pass for changed plugins.

### Task 2.1: Shared normalized streaming reader contract
- **Location**: `src/statistic_harness/core/normalized_reader.py`,
  `src/statistic_harness/core/stat_plugins/base.py`
- **Description**: Route plugin data access through a single chunked iterator API.
- **Complexity**: 8
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Plugins cannot directly materialize full normalized table without declared
    exception.
  - Runtime records chunk count and rows read.
- **Validation**:
  - Unit tests for iterator behavior.
  - Contract tests for illegal full-load access path.

### Task 2.2: Chunk-size invariance harness
- **Location**: `tests/contracts/test_chunk_invariance.py`,
  `scripts/run_chunk_invariance.py`
- **Description**: Run plugins at chunk sizes `1k`, `10k`, `50k` and assert same
  result envelope.
- **Complexity**: 7
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Numeric tolerances are fixed in
    `config/chunk_invariance_tolerances.yaml` and checked in CI.
  - Default tolerances: hours `<= 0.01`, percent `<= 0.001`,
    counts `== 0` delta.
- **Validation**:
  - Repeated-run deterministic harness in CI.

### Task 2.3: Runtime access matrix reconciliation
- **Location**: `scripts/plugin_data_access_matrix.py`,
  `scripts/audit_plugin_streaming_contract.py`,
  `docs/plugin_data_access_matrix.json`
- **Description**: Compare static declared access vs runtime observed access.
- **Complexity**: 6
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Every plugin labeled `contract_match` or `contract_mismatch`.
  - Mismatches fail strict mode.
- **Validation**:
  - Matrix reconciliation tests.

### Task 2.4: Resource budget enforcement
- **Location**: `tests/perf/test_plugin_resource_budget.py`,
  `docs/perf_hotspots.md`
- **Description**: Assert per-plugin RAM/CPU budgets and hotspot ranking.
- **Complexity**: 6
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Budget breaches are deterministic failures.
  - Hotspot rows must include `plugin_id`, `rss_peak_mb`, `cpu_ms`,
    `rows_read`, and deterministic sort key
    (`rss_peak_mb desc`, then `cpu_ms desc`, then `plugin_id asc`).
- **Validation**:
  - Perf gate tests with synthetic large fixtures.

### Task 2.5: Deterministic `run_seed` propagation
- **Location**: `src/statistic_harness/core/run_context.py`,
  `src/statistic_harness/core/pipeline.py`,
  `tests/core/test_seed_propagation.py`
- **Description**: Enforce per-run seed propagation from pipeline to every plugin
  and helper RNG usage path.
- **Complexity**: 6
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - All randomness is sourced from run context seeded RNG.
  - Any direct global RNG usage fails lint/tests.
- **Validation**:
  - Determinism test with repeated same-seed runs.
  - Negative test for forbidden unseeded RNG path.

## Sprint 3: Windowing + Targeting Accuracy
**Goal**: Ensure all plugins target the correct accounting month and close windows.  
**Demo/Validation**:
- Dynamic accounting-month and close-window coverage report.
- No plugin-local calendar-month logic remains.

### Task 3.1: Global accounting-month service
- **Location**: `src/statistic_harness/core/windowing.py`,
  `src/statistic_harness/core/stat_plugins/registry.py`
- **Description**: Replace plugin-local window calculations with shared service.
- **Complexity**: 8
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - Supports full accounting month, close static, close dynamic windows.
  - Window metadata included in each finding.
- **Validation**:
  - Unit tests with shifted month boundaries.
  - Integration test for window selection parity across plugins.

### Task 3.2: Targeting telemetry and audits
- **Location**: `src/statistic_harness/core/pipeline.py`,
  `scripts/audit_plugin_targeting_windows.py`,
  `scripts/audit_plugin_process_targeting.py`
- **Description**: Persist and validate per-plugin target row/process/window usage.
- **Complexity**: 7
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Every plugin records `input_rows`, `target_rows`, `window_source`.
  - Missing telemetry becomes failure.
- **Validation**:
  - Contract tests for telemetry parity with run manifest.

### Task 3.3: Parent-child feasibility and chain constraints
- **Location**: `src/statistic_harness/core/stat_plugins/actionability_contract.py`,
  `src/statistic_harness/core/recommendation_filters.py`
- **Description**: Encode non-modifiable process-chain constraints and disallow
  impossible child-process recommendations.
- **Complexity**: 6
- **Dependencies**: Task 3.2
- **Acceptance Criteria**:
  - Recommendations restricted to modifiable process-level actions.
  - Non-actionable chains explain blocked reason and affected downstream plugins.
- **Validation**:
  - Unit fixtures for parent/child and master-process-id behavior.

## Sprint 4: Recommendation Quality and Ranking Logic
**Goal**: Improve ranking to reflect client value, not arbitrary plugin type bias.  
**Demo/Validation**:
- Freshman report includes consistent modeled fields per row.
- Baseline rediscovers known landmarks independently.

### Task 4.1: Unified ranking objective function
- **Location**: `src/statistic_harness/core/ranking.py`,
  `config/recommendation_weights.yaml`
- **Description**: Rank on modeled delta-hours, close-window impact, manual effort
  reduction, and contention relief with configurable weights.
- **Complexity**: 8
- **Dependencies**: Tasks 3.2, 4.2
- **Acceptance Criteria**:
  - All recommendation classes emit comparable dimensionless gain and deltas.
  - No hard-coded class bonus.
- **Validation**:
  - Ranking invariance tests against fixed fixtures.

### Task 4.2: Manual-effort and user-load model
- **Location**: `src/statistic_harness/core/user_effort_model.py`,
  `tests/core/test_user_effort_model.py`
- **Description**: Quantify repeated manual launches by user/process/params and
  convert to modeled user-touch savings per close/accounting month.
- **Complexity**: 7
- **Dependencies**: Task 3.3
- **Acceptance Criteria**:
  - Outputs include `touch_reduction_count` and user-facing effort metrics.
  - Repeated-run scenarios (e.g., report extraction loops) surface clearly.
- **Validation**:
  - Unit tests with repeated-launch fixtures.

### Task 4.3: Known-issue rediscovery gate
- **Location**: `tests/integration/test_known_issue_rediscovery_baseline.py`,
  `docs/release_evidence/known_issue_detection_matrix.json`
- **Description**: Baseline run must independently rediscover known landmark issues
  without injection/pinning.
- **Complexity**: 6
- **Dependencies**: Tasks 4.1, 4.2
- **Acceptance Criteria**:
  - Rediscovery thresholds pass:
    `min_recall=1.00` for required landmark set,
    `min_precision>=0.60` for surfaced top set.
  - Landmark issues present with modeled metrics and evidence links using:
    `run://<run_id>/plugin/<plugin_id>/artifact/<artifact_name>#<row_id>`.
  - Gate fails on regression.
- **Validation**:
  - Deterministic integration gate in CI.

### Task 4.4: Recommendation explanation contract
- **Location**: `docs/schemas/actionable_recommendation_contract_v2.json`,
  `src/statistic_harness/core/reporting/freshman_formatter.py`
- **Description**: Require plain-English recommendation narrative plus quantitative
  fields on each row.
- **Complexity**: 5
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - `top-N` is controlled only by `config/reporting.yaml::recommendation_top_n`.
  - Every recommendation row includes required fields:
    `rank`, `idea_id`, `process_id`, `action`, `modeled_delta_hours`,
    `modeled_delta_hours_close_dynamic`, `dimensionless_gain`,
    `manual_touch_reduction_count`, `evidence_links`, `freshman_explanation`.
- **Validation**:
  - JSON schema validation + formatter snapshot tests.

## Sprint 5: Observability, Release Gates, and Proof
**Goal**: Make quality measurable and release-safe by default.  
**Demo/Validation**:
- Single command outputs run bundle with plugin validity summary.
- Full gauntlet + proof artifacts required for release.

### Task 5.1: Per-plugin observability bundle
- **Location**: `scripts/build_post_run_bundle.py`,
  `docs/release_evidence/plugin_runtime_matrix.json`
- **Description**: Emit rows-read/chunks/runtime/rss/evidence counts per plugin.
- **Complexity**: 6
- **Dependencies**: Sprint 2, Sprint 3
- **Acceptance Criteria**:
  - One runtime row for each plugin in run manifest.
  - Missing rows cause failed bundle generation.
- **Validation**:
  - Bundle parity tests.

### Task 5.2: Evaluator harness and ground-truth checks
- **Location**: `scripts/evaluator_harness.py`,
  `tests/fixtures/ground_truth.yaml`,
  `tests/integration/test_evaluator_harness.py`
- **Description**: Validate expected hidden attributes and recommendation markers
  against ground truth with configured tolerances.
- **Complexity**: 6
- **Dependencies**: Tasks 2.5, 4.3
- **Acceptance Criteria**:
  - Harness runs in CI and fails on tolerance breaches.
  - Ground-truth assertions are versioned and dataset-scoped.
- **Validation**:
  - Pass and fail fixture tests.

### Task 5.3: Report artifact generation and schema gate
- **Location**: `src/statistic_harness/core/reporting/`,
  `docs/report.schema.json`,
  `tests/integration/test_report_artifacts_contract.py`
- **Description**: Enforce generation of `report.md` and `report.json` for every
  run, then validate `report.json` against `docs/report.schema.json`.
- **Complexity**: 5
- **Dependencies**: Tasks 1.1, 5.1
- **Acceptance Criteria**:
  - Missing report artifact fails run.
  - Invalid `report.json` schema fails run.
- **Validation**:
  - Contract tests for missing/invalid artifacts.

### Task 5.4: Full gauntlet strict release gate
- **Location**: `scripts/run_release_gate.py`,
  `scripts/finalize_optimal_4pillars.py`
- **Description**: Block release unless:
  - `python -m pytest -q` passes
  - no runtime network usage is detected in analysis/report stages
  - contract audits pass
  - known-issue rediscovery gate passes
  - evaluator harness passes
  - report artifacts and schema gate pass
  - scorecard balance constraints pass
- **Complexity**: 6
- **Dependencies**: Tasks 1.2, 2.2, 2.3, 2.4, 2.5, 3.2, 4.3, 4.4, 5.1, 5.2, 5.3
- **Acceptance Criteria**:
  - Non-zero exit on any gate breach.
  - Release evidence written with deterministic reason codes.
  - Pipeline fail-closed behavior verified: plugin failures do not crash run and
    still produce both report artifacts plus deterministic error summary.
- **Validation**:
  - Gate integration test for pass and fail scenarios.

### Task 5.5: Cross-dataset comparison support
- **Location**: `scripts/compare_dataset_runs.py`,
  `tests/integration/test_compare_dataset_runs.py`
- **Description**: Deterministic diff for dataset-vs-dataset and pluginset-vs-pluginset.
- **Complexity**: 5
- **Dependencies**: Tasks 5.1, 5.3
- **Acceptance Criteria**:
  - Diffs include deltas for accuracy/performance/actionability metrics.
- **Validation**:
  - Snapshot tests on baseline vs synthetic fixtures.

## Testing Strategy
- Unit tests for each new helper/model/contract.
- Integration tests for end-to-end pipeline and known-issue rediscovery.
- Contract tests for schemas and run-manifest parity.
- Determinism tests:
  - fixed `run_seed`
  - repeated execution equality checks
  - chunk-size invariance checks
- Performance/resource tests for streaming plugins.
- Final gate: `python -m pytest -q` must pass before shipping.

## Potential Risks and Gotchas
- False confidence from low-signal plugins mislabeled as failures.
  - Mitigation: explicit `na_with_reason` taxonomy and reason-code QA.
- Window drift across ERPs with different month rollover behavior.
  - Mitigation: global window service plus dataset-level window diagnostics.
- Runtime overhead from additional telemetry.
  - Mitigation: lightweight counters and optional debug-detail mode.
- Recommendation ranking instability from weight changes.
  - Mitigation: locked defaults + fixtures + snapshot tests.
- Cross-dataset schema variance breaking plugin logic.
  - Mitigation: normalized alias mapping and strict matrix validation.

## Rollback Plan
- Keep previous ranking and windowing logic behind feature flags.
- Revert gate strictness to warn-only mode if emergency unblock is needed.
- Preserve prior report format in compatibility mode while migrating.
- If regressions appear, rollback by sprint boundary:
  1. Disable new ranking path
  2. Disable strict streaming enforcement
  3. Re-enable stable prior contracts
