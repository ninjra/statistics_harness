# Plan: Plugin Actionability Recommended-Next-Step Implementation

**Generated**: 2026-02-19  
**Estimated Complexity**: High

## Overview
Implement every `recommended_next_step` from `docs/release_evidence/plugin_actionability_audit_full_loaded_debug_20260219T1700Z.csv` so each plugin is either:
1. actionable with ranked, modeled recommendations, or
2. deterministic `not_applicable` with a plain-English reason and downstream impact context.

Approach:
1. Build a deterministic plugin-by-plugin execution matrix from the CSV.
2. Execute remediation by next-step cluster (not ad hoc plugin edits).
3. Add hard gates so no plugin can silently regress to non-actionable/no-decision behavior.

## Skills By Section
1. Planning and scope control: `plan-harder`.
Why: enforce phased sprints, atomic tasks, and review gates.
2. Coverage/matrix enforcement: `config-matrix-validator`.
Why: validate `plugin_id -> recommended_next_step -> implementation_state` coverage with no gaps.
3. Code/test implementation quality: `python-testing-patterns`, `testing`.
Why: unit + integration + full-gauntlet checks per remediation lane.
4. Observability and failure transparency: `discover-observability`, `observability-engineer`, `python-observability`, `logging-best-practices`.
Why: per-plugin progress metrics, reason-code drift detection, and hard-fail visibility.
5. Determinism/stability gates: `deterministic-tests-marshal`, `resource-budget-enforcer`, `state-recovery-simulator`.
Why: repeatable results across runs, bounded resource behavior, and crash-safe continuation.

## Prerequisites
1. Source audit artifact: `docs/release_evidence/plugin_actionability_audit_full_loaded_debug_20260219T1700Z.csv`.
2. Existing actionability/report logic:
`src/statistic_harness/core/report.py`, `src/statistic_harness/core/actionability_explanations.py`, `scripts/audit_plugin_actionability.py`.
3. Existing pipeline/run orchestration:
`src/statistic_harness/core/pipeline.py`, `scripts/run_loaded_dataset_full.py`, `scripts/verify_agent_execution_contract.py`.
4. Baseline validation commands available:
`python -m pytest -q` and strict actionability audit CLI.

## Sprint 1: Deterministic Coverage Matrix
**Goal**: Convert CSV rows into a deterministic, testable work contract that covers all plugin IDs.
**Skills**: `config-matrix-validator`, `python-testing-patterns`.
**Demo/Validation**:
1. Matrix summary reports exact counts by next-step cluster and equals CSV row count.
2. Hard gate fails if any plugin row has blank/unknown implementation state.

### Task 1.1: Build Plugin Next-Step Work Contract
- **Location**: `docs/release_evidence/plugin_actionability_audit_full_loaded_debug_20260219T1700Z.csv`, `scripts/audit_plugin_actionability.py`.
- **Description**: Define canonical cluster mapping for each next-step string and assign every plugin row to one remediation lane.
- **Complexity**: 5
- **Dependencies**: None
- **Acceptance Criteria**:
  - 254/254 plugin IDs assigned.
  - No empty next-step mapping.
  - Cluster totals are deterministic across reruns.
  - Each plugin ID has an explicit target state transition:
    `current_reason_code -> implemented_next_step -> expected_post_state`.
- **Validation**:
  - Add/extend tests near existing actionability audit script coverage.

### Task 1.2: Add Coverage Integrity Gate
- **Location**: `scripts/audit_plugin_actionability.py`, `scripts/verify_agent_execution_contract.py`.
- **Description**: Add strict mode that fails when any plugin lacks mapped remediation lane or has stale next-step token.
- **Complexity**: 4
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Hard-fail for unmapped plugins.
  - Machine-readable summary for CI.
- **Validation**:
  - Unit tests + one integration test with intentionally bad mapping fixture.

## Sprint 2: Adapter Expansion Lane (151 plugins)
**Goal**: Implement/extend recommendation adapters so observational/statistical findings become actionable output where valid.
**Skills**: `python-testing-patterns`, `testing`, `discover-observability`.
**Demo/Validation**:
1. Count for “Add or extend recommendation adapters...” decreases materially.
2. New recommendations contain modeled delta fields and process/user targeting where possible.

### Task 2.1: Normalize Finding-Family Adapter Registry
- **Location**: `src/statistic_harness/core/report.py`, `src/statistic_harness/core/actionability_explanations.py`.
- **Description**: Group finding families into reusable adapter handlers (observation-only, statistical-signal, precondition-dependent), with deterministic fallbacks.
- **Complexity**: 7
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Every mapped family resolves to one handler path.
  - No implicit drop/ignore path.
- **Validation**:
  - Expand adapter-focused tests around recommendation synthesis.

### Task 2.2: Add Plain-English Non-Actionable Explanations with Downstream Impact
- **Location**: `src/statistic_harness/core/actionability_explanations.py`, `src/statistic_harness/core/report.py`.
- **Description**: Ensure remaining non-actionable outputs include concrete reason, required input, and downstream consumer list.
- **Complexity**: 5
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - No generic “no action” text.
  - Explanations include next operator move.
- **Validation**:
  - `tests/test_report_explanations_lane.py` plus new targeted fixtures.

### Task 2.3: Resolve Special Next-Step Rows (Capacity + Downstream-Review)
- **Location**: `src/statistic_harness/core/report.py`, `src/statistic_harness/core/actionability_explanations.py`, `scripts/audit_plugin_actionability.py`.
- **Description**: Implement explicit handling for special rows from the CSV, including:
  - capacity-impact expansion (`analysis_close_cycle_capacity_impact`),
  - downstream review contracts (`profile_basic`, `transform_normalize_mixed`, `transform_sql_intents_pack_v1`),
  - standalone/downstream decision enforcement (`transform_template`).
- **Complexity**: 6
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - These plugin IDs have deterministic post-remediation states and test coverage.
  - No unresolved special-row next-step remains in strict audit output.
- **Validation**:
  - Add plugin-specific fixture tests for each special-row branch.

## Sprint 3: Direct-Action Finding Contract Lane (65 plugins)
**Goal**: Enforce direct-action finding payload standards so plugins can produce decision-grade recommendations.
**Skills**: `python-testing-patterns`, `testing`, `config-matrix-validator`.
**Demo/Validation**:
1. “NO_ACTIONABLE_FINDING_CLASS” count drops.
2. New recommendations include required modeled and targeting fields.

### Task 3.1: Enforce Direct-Action Finding Schema
- **Location**: `src/statistic_harness/core/report.py`.
- **Description**: Add explicit required fields for `actionable_ops_lever`, `ideaspace_action`, and `verified_action` paths, with deterministic `not_applicable` fallback if requirements fail.
- **Complexity**: 8
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Missing required fields never silently pass.
  - Fallback includes reason + corrective hint.
- **Validation**:
  - Unit tests for positive/negative finding payloads.

### Task 3.2: Retrofit Plugin Emitters to Contract
- **Location**: `plugins/`, `src/statistic_harness/core/report.py`.
- **Description**: Update each plugin in this lane to emit target process/user/module and modeled delta measures compatible with ranking output.
- **Complexity**: 9
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Every plugin in this lane emits contract-compliant findings or deterministic `not_applicable`.
- **Validation**:
  - Plugin-specific tests + smoke gauntlet sample.

## Sprint 4: Process Targeting and Chain-Bound Promotion Lane (24 + 3 plugins)
**Goal**: Resolve “no process target” and “policy-blocked target” by promoting to nearest modifiable parent process deterministically.
**Skills**: `python-testing-patterns`, `testing`.
**Demo/Validation**:
1. “NO_DIRECT_PROCESS_TARGET” and “EXCLUDED_BY_PROCESS_POLICY” decline.
2. Recommendations never target unmodifiable child/chain nodes.

### Task 4.1: Add Parent/Child Promotion Resolver
- **Location**: `src/statistic_harness/core/process_matcher.py`, `src/statistic_harness/core/report.py`.
- **Description**: Implement deterministic parent-promotion resolver using available process hierarchy fields and policy constraints.
- **Complexity**: 8
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Chain-bound child targets map to valid modifiable parent targets.
  - Resolver explains promotion path.
  - Plugins flagged for process target emission (for example `analysis_association_rules_apriori_v1`) emit `process_norm` or `target_process_ids`, or deterministic fallback reason.
- **Validation**:
  - Unit tests with parent/child/chain fixtures.

### Task 4.2: Propagate Resolved Targets into Recommendation Ranking
- **Location**: `src/statistic_harness/core/report.py`.
- **Description**: Ensure ranked recommendation records consume resolved parent targets and preserve modeled impact attribution.
- **Complexity**: 6
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Ranked items include resolved process target fields.
  - No child-only recommendation survives final ranking.
- **Validation**:
  - Integration tests on known baseline fixtures.

## Sprint 5: Snapshot/Serialization and Non-Analysis Plugin Completion Lane
**Goal**: Eliminate `REPORT_SNAPSHOT_OMISSION` and ensure non-analysis plugins produce explicit, coherent outputs.
**Skills**: `python-testing-patterns`, `testing`, `logging-best-practices`.
**Demo/Validation**:
1. `REPORT_SNAPSHOT_OMISSION` reaches zero.
2. Report/LLM/transform/profile plugins appear consistently in report payload with explicit state.

### Task 5.1: Include Executed Report/LLM Results in Report Snapshot
- **Location**: `src/statistic_harness/core/report.py`, `src/statistic_harness/core/pipeline.py`.
- **Description**: Ensure executed plugin results are serialized in `report.plugins` for all plugin types.
- **Complexity**: 6
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - `llm_prompt_builder`, `llm_text2sql_local_generate_v1`, `report_bundle`, `report_plain_english_v1` no longer omitted.
- **Validation**:
  - Integration test asserting presence of all executed plugin IDs.

### Task 5.2: Downstream-Only Plugin Classification Pass
- **Location**: `src/statistic_harness/core/actionability_explanations.py`, `src/statistic_harness/core/report.py`.
- **Description**: For downstream/supporting plugins, emit deterministic non-actionable reason with downstream dependency chain and never return ambiguous status.
- **Complexity**: 5
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - No ambiguous “unknown/no output” states.
  - `na` status carries clear prerequisite or design rationale.
- **Validation**:
  - Existing explanation-lane tests + new downstream-chain tests.

## Sprint 6: Observability, Determinism, and Full-Gauntlet Quality Gates
**Goal**: Make improvements measurable and durable across baseline + synthetic runs.
**Skills**: `discover-observability`, `observability-engineer`, `python-observability`, `deterministic-tests-marshal`, `resource-budget-enforcer`, `state-recovery-simulator`.
**Demo/Validation**:
1. Full gauntlet runs complete with explicit pass/fail plus per-plugin actionability telemetry.
2. Repeat runs produce stable reason-code and actionable-count distributions within tolerance.

### Task 6.1: Actionability Telemetry Contract
- **Location**: `src/statistic_harness/core/pipeline.py`, `scripts/verify_agent_execution_contract.py`, `scripts/show_actionable_results.py`.
- **Description**: Track per-run metrics: actionable count, non-actionable reason histogram, omission count, policy-blocked count, and remediation lane completion.
- **Complexity**: 7
- **Dependencies**: Sprints 2-5
- **Acceptance Criteria**:
  - Every full run emits complete actionability metric bundle.
  - Contract gate fails on missing metrics.
  - Contract includes per-plugin compliance records proving each CSV `plugin_id` had its `recommended_next_step` handled.
- **Validation**:
  - Contract verification tests and CLI smoke checks.

### Task 6.2: Determinism and Resource-Stability Gate
- **Location**: `scripts/run_loaded_dataset_full.py`, `src/statistic_harness/core/pipeline.py`.
- **Description**: Add rerun consistency checks and resource watchdog thresholds to catch regressions/hangs before release.
- **Complexity**: 7
- **Dependencies**: Task 6.1
- **Acceptance Criteria**:
  - Repeatability checks pass on baseline and synthetic datasets.
  - Hung-run detection and stale-state repair are automatic and auditable.
- **Validation**:
  - Repeated full-gauntlet pass with stable summary deltas.

## Testing Strategy
1. Unit tests:
`tests/test_report_explanations_lane.py` and actionability synthesis tests for each remediation lane.
2. Integration tests:
report snapshot integrity, parent-promotion behavior, adapter conversion behavior.
3. Compliance tests:
per-plugin assertions that every CSV `plugin_id` has a resolved post-state tied to its original `recommended_next_step`.
4. Full gauntlet:
run baseline + synthetic datasets with strict actionability audit gates.
5. Regression checks:
compare reason-code histogram and actionable counts before/after each sprint.

## Potential Risks & Gotchas
1. Risk: Large adapter lane causes inconsistent recommendation quality.
Mitigation: enforce schema-level required fields and lane-specific contract tests before merge.
2. Risk: Parent-promotion logic introduces false target mapping.
Mitigation: deterministic resolver with explainable trace and fixture-based chain tests.
3. Risk: Report snapshot drift for non-analysis plugins.
Mitigation: explicit “executed plugin IDs must be present in report payload” gate.
4. Risk: WSL I/O stalls during long runs.
Mitigation: stale-run detection, retry/recovery path, and resource watchdog telemetry.
5. Risk: Hidden regressions in non-actionable explanations.
Mitigation: strict reason-code and plain-English explanation tests.

## Rollback Plan
1. Keep each sprint in isolated commits with clear migration notes.
2. If recommendation quality regresses, revert the affected lane commit and keep coverage gate active.
3. If full-gauntlet stability regresses, disable only new lane routing logic behind config flag while preserving telemetry and strict failure reporting.

## Definition of Done
1. Every plugin ID from the CSV is mapped to a completed remediation lane.
2. Full gauntlet passes with:
   - no `NON_DECISION_PLUGIN`,
   - no snapshot omissions,
   - explicit actionable or deterministic non-actionable output per plugin.
3. `python -m pytest -q` passes.
4. Actionability telemetry and comparison summaries are produced for baseline + synthetics.
