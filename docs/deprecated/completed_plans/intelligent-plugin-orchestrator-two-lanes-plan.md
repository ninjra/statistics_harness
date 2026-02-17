# Plan: Intelligent Plugin Orchestrator Two Lanes

**Generated**: 2026-02-17  
**Estimated Complexity**: High

## Overview
Implement a deterministic, fail-closed plugin orchestrator that optimizes two lanes:
- `decision lane`: plugins expected to produce actionable recommendations.
- `explanation lane`: plugins that provide diagnostic/context/supporting output.

The orchestrator must preserve full-run integrity (no silent skips), enforce strict result semantics (`ok` / `failed` / deterministic `na` with reason), and improve throughput via dependency-aware scheduling while keeping recommendation quality measurable.

## Prerequisites
- Existing plugin manifests under `plugins/*/plugin.yaml`.
- Existing pipeline and storage contracts in:
  - `src/statistic_harness/core/pipeline.py`
  - `src/statistic_harness/core/storage.py`
  - `src/statistic_harness/core/report.py`
- Existing run-quality and comparison scripts in `scripts/`.
- Test runner in project venv (`.venv/bin/python -m pytest -q`).

## Implementation Skills Matrix (What The Build Must Use)
This section prescribes skills for implementation, not planning.

### Sprint-to-Skill Assignment
- **Sprint 1 (lane contract + deterministic DAG)**:
  - Primary: `testing`, `python-testing-patterns`
  - Secondary: `config-matrix-validator`
  - Why: schema/contract correctness and deterministic scheduler behavior.
- **Sprint 2 (failure semantics + cache + preflight)**:
  - Primary: `testing`, `python-testing-patterns`, `deterministic-tests-marshal`
  - Secondary: `policygate-penetration-suite`
  - Why: fail-closed runtime behavior with deterministic cache outcomes and boundary safety.
- **Sprint 3 (observability + evidence gates)**:
  - Primary: `discover-observability`, `observability-engineer`, `python-observability`, `logging-observability`
  - Secondary: `evidence-trace-auditor`, `audit-log-integrity-checker`
  - Why: detect silent failures, prove traceability, and enforce evidence integrity.
- **Sprint 4 (plugin fleet alignment + normalized-layer enforcement)**:
  - Primary: `config-matrix-validator`, `testing`
  - Secondary: `python-testing-patterns`
  - Why: matrix-driven compliance across all plugins and deterministic enforcement.
- **Sprint 5 (full-gauntlet + before/after insight deltas)**:
  - Primary: `deterministic-tests-marshal`, `perf-regression-gate`, `resource-budget-enforcer`
  - Secondary: `state-recovery-simulator`, `golden-answer-harness`
  - Why: prove deterministic stability, performance bounds, and measurable recommendation deltas.

### Mandatory Cross-Cutting Skills
- `shell-lint-ps-wsl`: lint every command before execution.
- `testing` + `python-testing-patterns`: any orchestrator code change must ship with tests.
- `discover-observability` + `observability-engineer`: no orchestration change without metric/trace visibility.

### Skill Activation Rules During Implementation
- Before touching scheduler/runtime code:
  - Activate: `testing`, `python-testing-patterns`.
- Before touching run summaries/report quality gates:
  - Activate: `discover-observability`, `observability-engineer`, `python-observability`, `evidence-trace-auditor`.
- Before full-gauntlet reruns:
  - Activate: `deterministic-tests-marshal`, `perf-regression-gate`, `resource-budget-enforcer`, `state-recovery-simulator`.
- Before plugin-matrix refactors:
  - Activate: `config-matrix-validator`.

## Sprint 1: Lane Contract and Orchestrator Core
**Goal**: Introduce first-class two-lane orchestration contract and deterministic scheduler.
**Demo/Validation**:
- Build lane-aware DAG from manifests.
- Show deterministic topological order with fixed seed.
- Show blocked execution when dependency/failure policy requires it.

### Task 1.1: Define Lane and Result-State Contract
- **Location**:
  - `docs/golden_release_runtime_policy.md`
  - `docs/result_quality_contract.md`
  - `docs/insight_quality_contract.md`
- **Description**:
  - Define lane semantics (`decision`, `explanation`) and allowed transitions.
  - Define strict result-state policy: no implicit skip; only `ok`, `failed`, `na` with deterministic reason code.
  - Define overall run status rule: any `failed` plugin => run `failed` even if run continues.
- **Complexity**: 4
- **Dependencies**: None
- **Acceptance Criteria**:
  - Contract docs specify lane rules, result taxonomy, and fail-closed behavior.
  - Contract includes plain-English explanation requirements for `na`.
- **Validation**:
  - Doc consistency checks (`scripts/verify_docs_and_plugin_matrices.py`).
- **Implementation Skills**:
  - `config-matrix-validator`, `testing`

### Task 1.2: Extend Plugin Manifest Schema for Lane Metadata
- **Location**:
  - `plugins/*/plugin.yaml`
  - `src/statistic_harness/core/plugin_manager.py` (or equivalent loader)
  - `src/statistic_harness/core/migrations.py` (if DB persistence required)
- **Description**:
  - Add lane metadata and role flags (`decision_capable`, `diagnostic_only`, `requires_downstream_mapping`).
  - Validate manifests at load time; malformed lane metadata hard-fails preflight.
- **Complexity**: 6
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Every plugin resolves to exactly one lane.
  - Validation errors are explicit and block run startup.
- **Validation**:
  - Unit tests for manifest parsing and schema errors.
- **Implementation Skills**:
  - `testing`, `python-testing-patterns`, `config-matrix-validator`

### Task 1.3: Implement Deterministic Lane-Aware DAG Builder
- **Location**:
  - `src/statistic_harness/core/pipeline.py`
  - `src/statistic_harness/core/orchestrator.py` (new, if needed)
- **Description**:
  - Build dependency graph with stable ordering key.
  - Schedule by lane policy:
    - decision-lane priority for actionable generation.
    - explanation lane runs in dependency-safe parallel windows.
- **Complexity**: 8
- **Dependencies**: Task 1.2
- **Acceptance Criteria**:
  - Same input + same seed => identical execution plan ordering.
  - Dependency cycles produce deterministic hard error with trace.
- **Validation**:
  - Unit tests for graph build/order/cycle detection.
- **Implementation Skills**:
  - `testing`, `python-testing-patterns`, `deterministic-tests-marshal`

### Task 1.4: Add Orchestrator Mode Feature Flag (`legacy` vs `two_lane_strict`)
- **Location**:
  - `src/statistic_harness/core/pipeline.py`
  - `src/statistic_harness/core/known_issues_mode.py` (or runtime settings path)
  - `scripts/run_loaded_dataset_full.py`
- **Description**:
  - Add explicit runtime switch for orchestration mode with deterministic default.
  - Implement backward-compatible path for `legacy` mode and strict path for `two_lane_strict`.
  - Emit selected mode in run manifest/report lineage.
- **Complexity**: 6
- **Dependencies**: Task 1.1, Task 1.3
- **Acceptance Criteria**:
  - Both modes are callable and produce valid run outputs.
  - Mode choice is visible in run artifacts.
- **Validation**:
  - Integration tests for mode switching and parity checks where expected.
- **Implementation Skills**:
  - `testing`, `python-testing-patterns`, `config-matrix-validator`

## Sprint 2: Execution Policy, Failure Semantics, and Caching
**Goal**: Make runtime behavior strict, observable, and efficient.
**Demo/Validation**:
- Full run continues after plugin failure but marks overall run failed.
- No plugin can silently skip.
- Cache reuse works only when fingerprint compatibility is exact.

### Task 2.1: Enforce No-Silent-Skip Runtime Policy
- **Location**:
  - `src/statistic_harness/core/pipeline.py`
  - `src/statistic_harness/core/storage.py`
  - `src/statistic_harness/core/report.py`
- **Description**:
  - Replace permissive skip/degraded pathways with explicit `failed` or deterministic `na`.
  - Require deterministic reason code and plain-English explanation for `na`.
- **Complexity**: 8
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - `skipped` status is not emitted by orchestrated runs.
  - All `na` rows include reason code + human explanation + downstream list (if non-decision plugin).
- **Validation**:
  - Integration tests checking run summaries and plugin rows.
- **Implementation Skills**:
  - `testing`, `python-testing-patterns`, `policygate-penetration-suite`

### Task 2.2: Preflight Structural Validation Gate
- **Location**:
  - `src/statistic_harness/core/pipeline.py`
  - `scripts/run_loaded_dataset_full.py`
- **Description**:
  - Add preflight scan for required columns/schema availability before scheduling.
  - Fail plugin preflight explicitly instead of late runtime structural crashes.
- **Complexity**: 7
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Missing-column and structural mismatches surface as explicit plugin failures with reason.
  - Run report includes preflight failure section.
- **Validation**:
  - Targeted integration tests with synthetic missing-column fixtures.
- **Implementation Skills**:
  - `testing`, `python-testing-patterns`, `ccpm-debugging`

### Task 2.3: Fingerprint-Safe Cache Policy
- **Location**:
  - `src/statistic_harness/core/storage.py`
  - `src/statistic_harness/core/pipeline.py`
  - `scripts/backfill_stat_plugin_cache_keys.py`
- **Description**:
  - Enforce cache hits only when plugin code hash, settings hash, dataset hash, and lane-policy version match.
  - Record cache hit/miss reason in execution telemetry.
- **Complexity**: 6
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - No stale-cache false positives.
  - Deterministic cache behavior across reruns.
- **Validation**:
  - Unit tests for cache key computation and invalidation cases.
- **Implementation Skills**:
  - `deterministic-tests-marshal`, `testing`

## Sprint 3: Observability and Evidence Integrity
**Goal**: Ensure silent failure detection is auditable and actionable.
**Demo/Validation**:
- Lane-level telemetry dashboards/reports exist in run artifacts.
- Every plugin has explicit execution outcome and evidence trail.

### Task 3.1: Lane-Level Runtime Metrics and Tracing
- **Location**:
  - `src/statistic_harness/core/pipeline.py`
  - `src/statistic_harness/core/report.py`
  - `scripts/runtime_stats.py`
- **Description**:
  - Emit per-lane metrics: duration, fail count, actionable yield, NA count, downstream utilization.
  - Add deterministic run summary fields for silent-failure checks.
- **Complexity**: 7
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - Final run summary includes lane SLA metrics.
  - Metrics are available for baseline and synthetic runs.
- **Validation**:
  - Script-level assertions for metric presence and non-null values.
- **Implementation Skills**:
  - `discover-observability`, `observability-engineer`, `python-observability`, `logging-observability`

### Task 3.2: Evidence and Explanation Coverage Gate
- **Location**:
  - `scripts/build_final_validation_summary.py`
  - `scripts/validate_plugin_finding_quality.py`
  - `docs/release_evidence/` outputs
- **Description**:
  - Add hard checks for:
    - unexplained plugin outputs,
    - blank finding kinds,
    - non-decision plugins missing downstream mapping,
    - missing plain-English NA explanation.
- **Complexity**: 6
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Quality scripts fail when any silent/non-actionable condition is detected.
  - Report contains explicit offending plugin list.
- **Validation**:
  - Regression tests on known-bad historical runs.
- **Implementation Skills**:
  - `evidence-trace-auditor`, `audit-log-integrity-checker`, `testing`

## Sprint 4: Plugin-Wide Alignment and Normalized-Layer Enforcement
**Goal**: Bring all plugins under orchestrator contract and normalized-data access requirements.
**Demo/Validation**:
- Matrix proving each plugin has lane classification and normalized-access contract.
- Non-compliant plugins fail with deterministic reasons.

### Task 4.1: Plugin Lane Matrix and Actionability Mapping
- **Location**:
  - `docs/plugins_functionality_matrix.json`
  - `docs/plugin_class_actionability_matrix.json`
  - `scripts/plugins_functionality_matrix.py`
  - `scripts/build_plugin_class_actionability_matrix.py`
- **Description**:
  - Regenerate plugin matrix to include lane, decision/actionability class, and explanation obligations.
  - Ensure all plugins have documented role.
- **Complexity**: 6
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - 100% plugin coverage in lane/actionability matrix.
  - No unclassified plugins.
- **Validation**:
  - Matrix verification script with fail-on-missing rows.
- **Implementation Skills**:
  - `config-matrix-validator`, `testing`

### Task 4.2: Normalized Layer and Streaming Access Guardrails
- **Location**:
  - `docs/plugin_data_access_matrix.json`
  - `scripts/plugin_data_access_matrix.py`
  - `src/statistic_harness/core/pipeline.py`
- **Description**:
  - Enforce data-access contract per plugin (normalized-layer path, bounded loader behavior, SQL path declarations).
  - Flag runtime/static mismatches as failures.
  - Define concrete threshold for large dataset mode: `dataset_row_count > 1_000_000` (unless overridden by explicit config).
  - Enforce fail rule in large dataset mode:
    - if plugin uses unbounded dataset loader and has no SQL/batch fallback => plugin `failed`.
    - if plugin static contract says batched access but runtime shows unbounded loader => plugin `failed`.
- **Complexity**: 8
- **Dependencies**: Sprint 3
- **Acceptance Criteria**:
  - Runtime access mismatches become explicit failures.
  - Large dataset mode enforces bounded/streaming policy.
- **Validation**:
  - Add tests for bounded/unbounded loader policy and contract mismatch detection.
- **Implementation Skills**:
  - `config-matrix-validator`, `python-testing-patterns`, `testing`

## Sprint 5: Full-Gauntlet Verification and Comparative Insight Validation
**Goal**: Prove orchestrator improves reliability and insight quality across baseline + synthetic datasets.
**Demo/Validation**:
- Deterministic full-gauntlet results for all three datasets.
- Before/after comparison artifacts produced and reviewable.

### Task 5.1: Deterministic Full-Gauntlet Execution Matrix
- **Location**:
  - `scripts/run_loaded_dataset_full.py`
  - `scripts/run_final_validation_checklist.sh`
  - `docs/release_evidence/`
- **Description**:
  - Run baseline real dataset first, then synthetic datasets with same plugin set and seed policy.
  - Capture per-run quality summaries and enforcement outputs.
- **Complexity**: 7
- **Dependencies**: Sprint 4
- **Acceptance Criteria**:
  - No silent failure categories in quality summary.
  - Run status and plugin outcomes are deterministic per seed.
- **Validation**:
  - Repeat-run determinism checks.
  - Verify report outputs exist and validate:
    - `report.md` exists for every completed run.
    - `report.json` exists and validates against `docs/report.schema.json`.
- **Implementation Skills**:
  - `deterministic-tests-marshal`, `state-recovery-simulator`, `resource-budget-enforcer`

### Task 5.2: Insight Delta Comparison (Before vs After Orchestrator)
- **Location**:
  - `scripts/compare_run_outputs.py`
  - `scripts/compare_dataset_runs.py`
  - `docs/release_evidence/*.json`
- **Description**:
  - Compare baseline pre-orchestrator and post-orchestrator runs.
  - Compare synthetic datasets against baseline using same orchestrator policy.
  - Rank modeled improvements descending and include confidence/evidence flags.
- **Complexity**: 6
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - Diff artifacts enumerate actionable deltas, status deltas, and explanation deltas.
  - Output includes plain-English summary for stakeholders.
- **Validation**:
  - Automated diff script checks and spot verification on top deltas.
- **Implementation Skills**:
  - `golden-answer-harness`, `perf-regression-gate`, `testing`

## Testing Strategy
- Unit tests:
  - DAG order determinism, lane routing, manifest validation, cache key logic.
- Integration tests:
  - failure semantics (`failed` vs `na`), preflight structural checks, normalized-layer enforcement.
- System tests:
  - full gauntlet baseline + synthetic datasets with deterministic seeds.
- Quality gates:
  - `analysis_ok_without_findings_count == 0` for decision plugins.
  - `unexplained_plugin_count == 0`.
  - `blank_kind_findings_count == 0`.
  - no `skipped` statuses in orchestrated runs.
  - `report.md` and schema-valid `report.json` are mandatory outputs.
- Gate source-of-truth and enforcement locus:
  - Runtime gates emitted in `run_manifest.json` and `report.json` summary blocks.
  - Post-run enforcement via:
    - `scripts/build_final_validation_summary.py`
    - `scripts/validate_plugin_finding_quality.py`
    - final checklist runner (`scripts/run_final_validation_checklist.sh`).
- Observability gate:
  - lane metrics must be present in runtime summaries and parseable by `scripts/runtime_stats.py`.

## Potential Risks & Gotchas
- Lane misclassification can suppress valid recommendations.
  - Mitigation: start with conservative mapping + matrix review gate.
- Cache over-reuse can mask plugin behavior changes.
  - Mitigation: strict fingerprint include lane-policy version.
- Historical run artifacts may not satisfy new strict contracts.
  - Mitigation: mark legacy runs as pre-contract; enforce strict mode on new runs.
- WSL `/mnt/*` I/O stalls can appear as plugin hangs near report stage.
  - Mitigation: add hang watchdog, heartbeat checks, and fail-fast stale-run reconciliation.
- Large plugin fleet raises migration risk.
  - Mitigation: staged rollout with per-sprint demoable gates and regression snapshots.

## Rollback Plan
- Feature-flag orchestrator by mode (`legacy` vs `two_lane_strict`) in runtime settings.
- Keep old scheduler path callable for emergency rollback.
- If strict mode introduces blocking regressions:
  - revert to legacy mode,
  - preserve generated artifacts,
  - run targeted plugin-fix sprint,
  - re-enable strict mode only after full quality gate pass.
- Rollback validation checklist:
  - switch from `two_lane_strict` to `legacy` in one config change,
  - rerun baseline dataset with same seed,
  - verify both report outputs are present and schema-valid,
  - verify rollback run is explicitly tagged `orchestrator_mode=legacy`.

## One Round of Clarifying Questions
1. For launch scoring, should optimization priority order be:
   1) no silent failures, 2) recommendation quality, 3) runtime speed, 4) resource usage?
2. Do you want strict mode to hard-fail run completion when **any** plugin is `failed`, even if all decision plugins succeed?
3. Should lane assignment live only in plugin manifests, or also allow runtime overrides from project settings for controlled experiments?
