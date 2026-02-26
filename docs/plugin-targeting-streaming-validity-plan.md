# Plan: Plugin Targeting + Streaming + Validity Audit

**Generated**: 2026-02-26  
**Estimated Complexity**: High

## Overview
Build a deterministic audit pipeline that classifies every plugin into:

- `PASS_ACTIONABLE`: valid logic and materially actionable in this dataset
- `PASS_VALID_LOW_SIGNAL`: valid logic, modeled impact below threshold
- `PASS_VALID_MARKER_ABSENT`: valid logic, required marker not present
- `FAIL_TARGETING`: bad process/window/row targeting
- `FAIL_STREAMING_CONTRACT`: loader/streaming contract violation
- `FAIL_LOGIC`: invalid or contradictory findings

### Decision Precedence + Tie-Break
- Precedence order is strict: `FAIL_*` > `PASS_*`.
- If multiple `FAIL_*` states trigger, keep highest severity:
  `FAIL_LOGIC` > `FAIL_TARGETING` > `FAIL_STREAMING_CONTRACT`.
- Pass-side tie-break:
  `PASS_ACTIONABLE` > `PASS_VALID_LOW_SIGNAL` > `PASS_VALID_MARKER_ABSENT`.

### Materiality / Low-Signal Formula
- Primary metric: `modeled_delta_hours_close_dynamic`.
- Secondary metric fallback: `modeled_delta_hours`.
- `PASS_ACTIONABLE` threshold: `>= 1.00h` close-dynamic delta.
- `PASS_VALID_LOW_SIGNAL`: `> 0.00h` and `< 1.00h`.
- `PASS_VALID_MARKER_ABSENT`: marker catalog says prerequisite marker missing.
- All hour metrics rounded to 2 decimals; percentages to 3 decimals.

This directly answers whether only ~18 plugins matter because of dataset signal
distribution or because other plugins are malfunctioning.

## Prerequisites
- Baseline run artifacts in `appdata/runs/<run_id>/`
- SQLite state DB at `appdata/state.sqlite`
- Existing matrices:
  - `docs/plugin_data_access_matrix.json`
  - `docs/plugin_class_actionability_matrix.json`
  - `docs/release_evidence/contract_*.json`
- Canonical run-manifest source-of-truth (to add):
  - `docs/plugin_run_manifest.schema.json`
  - `appdata/runs/<run_id>/plugin_run_manifest.json`
- Full test gate remains `python -m pytest -q`

## Skills By Sprint
- Sprint 1-2: `config-matrix-validator`, `observability-engineer`
- Sprint 2-3: `python-testing-patterns`, `deterministic-tests-marshal`
- Sprint 3-4: `resource-budget-enforcer`, `python-observability`
- Sprint 4-5: `evidence-trace-auditor`, `testing`
- All sprints: `shell-lint-ps-wsl`

## Sprint 1: Audit Contract + Classification Layer
**Goal**: Define a single per-plugin audit contract and deterministic outcomes.  
**Demo/Validation**:
- Run contract generator and produce one JSON row per plugin.
- Verify all plugins map to exactly one classification state.

### Task 1.1: Define audit schema
- **Location**: `docs/plugin_validation_contract.schema.json`
- **Description**: Add required fields for targeting, streaming, logic validity,
  marker presence, and final classification.
- **Complexity**: 5
- **Dependencies**: None
- **Acceptance Criteria**:
  - Schema has strict required keys and enums for all 6 outcome states.
  - Supports per-window metrics for accounting/close-static/close-dynamic.
  - Includes evidence fingerprint fields:
    `run_seed`, `dataset_hash`, `plugin_registry_hash`, `schema_version`,
    `git_commit`, `serialized_output_hash`.
- **Validation**:
  - JSON schema validation unit test with valid and invalid fixtures.

### Task 1.2: Build classification mapper
- **Location**: `scripts/classify_plugin_validity.py`
- **Description**: Ingest run report + plugin artifacts and emit one deterministic
  row per plugin with `state` and `reason_code`.
- **Complexity**: 7
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - 1:1 mapping across all plugin IDs in run scope.
  - No ambiguous/multi-state assignments.
- **Validation**:
  - Test for full plugin count parity vs run manifest.

### Task 1.4: Add canonical run-manifest contract
- **Location**: `docs/plugin_run_manifest.schema.json`,
  `scripts/build_plugin_run_manifest.py`
- **Description**: Create canonical plugin universe manifest for a run and use it
  as source-of-truth for all parity checks.
- **Complexity**: 5
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Manifest lists all expected plugins with stable ordering.
  - All audit scripts read this manifest for parity assertions.
- **Validation**:
  - Contract test for parity across manifest, report plugins, and audit output.

### Task 1.5: Fail-closed plugin error mapping
- **Location**: `src/statistic_harness/core/pipeline.py`,
  `scripts/classify_plugin_validity.py`
- **Description**: Ensure plugin exceptions map to deterministic `FAIL_*` rows
  while pipeline still completes and reports failures.
- **Complexity**: 6
- **Dependencies**: Task 1.2
- **Acceptance Criteria**:
  - No silent plugin exception paths.
  - Pipeline report always includes error summaries.
- **Validation**:
  - Integration test with injected plugin exception.

### Task 1.3: Add downstream usage tracing
- **Location**: `src/statistic_harness/core/report.py`,
  `scripts/build_plugin_class_actionability_matrix.py`
- **Description**: Record which downstream stages consumed each plugin output.
- **Complexity**: 6
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Every non-decision plugin lists downstream consumers or explicit none.
- **Validation**:
  - Contract test that downstream map exists for all plugins.

## Sprint 2: Targeting Correctness Verification
**Goal**: Prove each plugin targets the intended rows/processes/windows correctly.  
**Demo/Validation**:
- Produce `targeting_audit_<run_id>.json` with pass/fail per plugin.
- Include mismatch deltas for expected vs actual targeted rows.

### Task 2.1: Add targeting telemetry capture
- **Location**: `src/statistic_harness/core/pipeline.py`,
  `src/statistic_harness/core/stat_plugins/registry.py`
- **Description**: Persist per-plugin targeting telemetry:
  `input_rows`, `target_rows`, `process_scope`, `window_source`.
- **Complexity**: 8
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Telemetry emitted for every plugin execution.
  - Missing telemetry is a hard `FAIL_TARGETING`.
- **Validation**:
  - Integration test asserts telemetry coverage = plugin count.

### Task 2.2: Window-consistency checks
- **Location**: `scripts/audit_plugin_targeting_windows.py`
- **Description**: Verify plugins align with dynamic accounting month and close
  windows where applicable.
- **Complexity**: 7
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Plugins flagged if using default/fallback window when dynamic exists.
  - Per-plugin `window_alignment_status` emitted.
- **Validation**:
  - Fixtures with shifted month boundary and expected detection behavior.

### Task 2.3: Process-target integrity checks
- **Location**: `scripts/audit_plugin_process_targeting.py`
- **Description**: Ensure process-targeted plugins reference normalized process
  IDs and avoid queue-ID confusion.
- **Complexity**: 6
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Process-targeted findings always carry resolved process identifiers.
- **Validation**:
  - Unit tests with numeric queue IDs and expected process resolution.

## Sprint 3: Streaming + Resource Contract Validation
**Goal**: Prove plugins obey dataset access contracts under large data.  
**Demo/Validation**:
- Produce `streaming_contract_audit_<run_id>.json`.
- Fail run when strict policy is enabled and mismatches exist.

### Task 3.1: Static vs runtime access reconciliation
- **Location**: `scripts/audit_plugin_streaming_contract.py`
- **Description**: Compare static matrix (`plugin_data_access_matrix`) with
  runtime artifacts (`runtime_access.json`) per plugin.
- **Complexity**: 7
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Every plugin labeled `contract_match` or `contract_mismatch`.
- **Validation**:
  - Tests for false positive prevention and true mismatch detection.

### Task 3.2: Bounded-memory enforcement tests
- **Location**: `tests/test_streaming_contract_enforcement.py`
- **Description**: Add large-fixture tests proving bounded behavior or explicit
  documented unbounded allowance with evidence.
- **Complexity**: 8
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - No silent full-load for plugins expected to stream.
  - Budget breaches produce deterministic failure.
- **Validation**:
  - Repeat runs with fixed seed and RSS thresholds.

### Task 3.3: Offender prioritization feed
- **Location**: `scripts/rank_streaming_offenders.py`,
  `docs/streaming_offenders_ranked.md`
- **Description**: Extend ranking output with contract mismatch severity and
  remediation priority.
- **Complexity**: 4
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Offenders are ranked by impact + contract risk.
- **Validation**:
  - Snapshot test on generated ranking markdown/json.

## Sprint 4: Logic Validity vs Marker-Absence Proof
**Goal**: Distinguish "working but low-signal" from "logic bug".  
**Demo/Validation**:
- Produce `logic_validity_audit_<run_id>.json`.
- Show explicit counts by final classification bucket.

### Task 4.1: Marker catalog per plugin family
- **Location**: `docs/plugin_marker_catalog.yaml`
- **Description**: Define required marker signals per plugin/family and expected
  fallback behavior when absent.
- **Complexity**: 6
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Every plugin has marker expectation or explicit `non_decision`.
- **Validation**:
  - Lint test requiring 100% plugin coverage in marker catalog.

### Task 4.2: Marker-present / marker-absent A/B fixtures
- **Location**: `tests/fixtures/plugin_validity_ab/`,
  `tests/test_plugin_marker_ab_validation.py`
- **Description**: For each family, run A/B fixtures to prove plugin detects
  when markers exist and returns deterministic low-signal when absent.
- **Complexity**: 9
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Marker-present case yields expected finding class.
  - Marker-absent case yields deterministic non-actionable reason.
- **Validation**:
  - Parameterized tests per plugin family.

### Task 4.3: Classification report generator
- **Location**: `scripts/build_plugin_validity_summary.py`
- **Description**: Aggregate all audits and output:
  - counts by classification bucket
  - per-plugin reason
  - evidence links to runtime artifacts/findings
- **Complexity**: 6
- **Dependencies**: Tasks 1.2, 2.3, 3.3, 4.1, 4.2
- **Acceptance Criteria**:
  - One row per plugin with unambiguous `why`.
- **Validation**:
  - Contract test for row count parity and required columns.

### Task 4.4: Evaluator harness for tolerance checks
- **Location**: `scripts/evaluate_plugin_validity_ground_truth.py`,
  `tests/fixtures/ground_truth.yaml`
- **Description**: Add evaluator harness that checks expected plugin outcomes
  against ground truth with configured tolerances.
- **Complexity**: 7
- **Dependencies**: Task 4.3
- **Acceptance Criteria**:
  - Ground-truth checks run in CI and fail on tolerance breaches.
- **Validation**:
  - Unit test with pass and fail fixtures.

## Sprint 5: Baseline Decision Package + Gates
**Goal**: Produce decision-ready evidence for keep/optimize/deprecate choices.  
**Demo/Validation**:
- Output `docs/release_evidence/plugin_validity_decision_<run_id>.json`.
- Gate fails when unresolved `FAIL_*` items exist.

### Task 5.1: Keep/optimize/deprecate recommendations
- **Location**: `scripts/recommend_plugin_lane_assignment.py`
- **Description**: Assign each plugin to:
  `core_lane`, `discovery_lane`, `fix_first`, or `deprecate_candidate`.
- **Complexity**: 6
- **Dependencies**: Tasks 2.3, 3.3, 4.3
- **Acceptance Criteria**:
  - Assignments backed by objective metrics and evidence references.
- **Validation**:
  - Deterministic output test with fixed run snapshot.

### Task 5.2: Add hard CI gate for unresolved fails
- **Location**: `scripts/run_release_gate.py`,
  `tests/test_plugin_validity_release_gate.py`
- **Description**: Block release if any plugin remains in `FAIL_TARGETING`,
  `FAIL_STREAMING_CONTRACT`, or `FAIL_LOGIC`.
- **Complexity**: 5
- **Dependencies**: Task 4.3
- **Acceptance Criteria**:
  - Gate blocks on unresolved fail states.
  - Gate passes when only pass states remain.
- **Validation**:
  - Gate simulation tests with synthetic matrices.

### Task 5.3: Integrate into required final reports
- **Location**: `src/statistic_harness/core/report.py`,
  `docs/report.schema.json`,
  `tests/test_report_schema_plugin_validity_section.py`
- **Description**: Add audit summary into `report.md` and `report.json` and
  ensure schema validation includes the new section.
- **Complexity**: 6
- **Dependencies**: Tasks 4.3, 4.4
- **Acceptance Criteria**:
  - Report outputs include plugin validity summary and pass schema validation.
- **Validation**:
  - Report schema test + end-to-end integration test.

## Testing Strategy
- Unit: schema, classifiers, and reason-code assignment.
- Integration: one full baseline run with all audit outputs generated.
- Determinism:
  - Canonical JSON serialization: sorted keys, stable row ordering by
    `plugin_id`, normalized timestamps excluded from content hash.
  - Repeat audit commands 3x and compare `serialized_output_hash`.
- Resource: streaming/RSS checks on large fixture with strict mode on.
- Contract: row-count parity across run manifest, plugin matrix, and audit output.
- Evaluator harness: `ground_truth.yaml` tolerance checks must pass.

## Potential Risks & Gotchas
- Risk: false fails from telemetry gaps, not plugin defects.
  - Mitigation: explicit `telemetry_missing` reason code and bootstrap checks.
- Risk: marker catalog drift as plugins evolve.
  - Mitigation: CI check enforcing coverage + plugin ID sync.
- Risk: high runtime for all-family A/B fixtures.
  - Mitigation: stratified family fixtures + nightly exhaustive sweep.
- Risk: classification bias toward direct-action plugins.
  - Mitigation: require evidence-backed distinction between low-signal and fail.

## Rollback Plan
- Audit layer touches `scripts/`, `docs/`, and core files:
  `src/statistic_harness/core/pipeline.py`,
  `src/statistic_harness/core/report.py`.
- Revert by removing audit scripts/docs and disabling gate/report flags:
  `STAT_HARNESS_PLUGIN_VALIDITY_GATE=0`.
- Disable report integration flag (to add):
  `STAT_HARNESS_PLUGIN_VALIDITY_REPORT_SECTION=0`.
- Keep prior recommendation path operational while audit layer is disabled.
