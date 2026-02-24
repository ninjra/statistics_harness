# Plan: Each Plugin Delivering Actionable Results

**Generated**: 2026-02-21  
**Estimated Complexity**: High

## Overview
Bring the harness to a strict state where every plugin returns decision-grade, actionable output on full gauntlet runs.

Current baseline run evidence (`baseline_optimal_20260221T0210Z`):
- 254 executable plugins produced result rows.
- 252 `ok`, 2 `error` (timeouts): `analysis_param_near_duplicate_minhash_v1`, `analysis_param_near_duplicate_simhash_v1`.
- Recommendations include 111 actionable items and 190 explanation items.
- Explanation reasons are still dominated by non-actionable classes (`OBSERVATION_ONLY`, `NO_ACTIONABLE_FINDING_CLASS`, `NO_STATISTICAL_SIGNAL`, precondition buckets).

Approach:
1. Eliminate execution errors first.
2. Convert every non-actionable plugin outcome into actionable recommendation generation.
3. Enforce hard run contracts: no plugin may end without actionable output.
4. Prove deterministically with per-plugin metrics and full-gauntlet evidence.

## Skills By Section
1. Planning and sequencing: `plan-harder`.
Why: phased, sprinted execution with atomic and committable tasks.
2. Root-cause and failure elimination: `ccpm-debugging`.
Why: remove timeout and logic defects without symptom patching.
3. Coverage and matrix integrity: `config-matrix-validator`.
Why: guarantee full plugin-by-plugin coverage with no gaps.
4. Test enforcement: `python-testing-patterns`, `testing`.
Why: lock behavior with unit/integration/contract tests and prevent regressions.
5. Observability and run governance: `discover-observability`, `observability-engineer`, `python-observability`, `logging-best-practices`.
Why: make per-plugin actionability measurable and auditable.
6. Determinism/resource controls: `deterministic-tests-marshal`, `resource-budget-enforcer`, `state-recovery-simulator`.
Why: ensure stability under full load and prevent hidden hangs.

## Prerequisites
- Latest baseline report artifacts:
  - `appdata/runs/baseline_optimal_20260221T0210Z/report.json`
  - `appdata/state.sqlite`
- Frozen baseline artifact lock (must be created before any remediation):
  - `docs/release_evidence/contract_baseline_full_with_runbook30_probe.json`
  - includes file hashes for report, answers summary, and source DB snapshot metadata.
- Actionability engine files:
  - `src/statistic_harness/core/report.py`
  - `src/statistic_harness/core/actionability_explanations.py`
  - `scripts/audit_plugin_actionability.py`
  - `scripts/verify_agent_execution_contract.py`
- Pipeline orchestration:
  - `scripts/run_loaded_dataset_full.py`
  - `scripts/finalize_optimal_4pillars.py`

## Assumptions and Clarifications
- In scope: all plugin types (`analysis`, `profile`, `transform`, `report`, `llm`, `planner`).
- Required end-state: every plugin produces actionable output; explanation-only end states are not acceptable for release.
- If a plugin historically provided observational-only output, it must be upgraded to emit direct decision payloads (process target + modeled impact).
- “Valid actionable” means the recommendation can be ranked against all others using the same key metrics (dimensionless efficiency gain + modeled delta hours + windowed metrics).

## Sprint 1: Stabilize Full Execution
**Goal**: Achieve `completed` run with zero plugin execution errors.  
**Demo/Validation**:
- Full run ends `completed` (not `partial`).
- `plugin_results_v2` status counts: `error=0`, `skipped=0`, `degraded=0`, `aborted=0`.

### Task 1.1: Fix Minhash/Simhash Timeout Failures
- **Location**: `src/statistic_harness/core/top20_plugins.py`, `plugins/analysis_param_near_duplicate_minhash_v1/*`, `plugins/analysis_param_near_duplicate_simhash_v1/*`.
- **Description**: Profile runtime path and introduce bounded, deterministic execution envelopes that always return result payloads.
- **Complexity**: 7
- **Dependencies**: None
- **Acceptance Criteria**:
  - Both plugins finish under full baseline load.
  - No timeout status.
  - Findings include modeled impact fields.
- **Validation**:
  - Targeted plugin tests + full baseline rerun.

### Task 1.2: Add Timeout Regression Tests
- **Location**: `tests/test_top20_simhash_plugin_smoke.py`, `tests/test_top20_hawkes_dtw_smoke.py`, new minhash/simhash runtime guard tests.
- **Description**: Add bounded-runtime tests with deterministic fixture sizes.
- **Complexity**: 5
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Tests fail if runtime envelope is exceeded.
  - Tests fail if plugin returns empty actionable payload.
- **Validation**:
  - `python -m pytest -q` includes these tests.

## Sprint 2: Build Plugin-by-Plugin Actionability Matrix (Hard Coverage)
**Goal**: Create deterministic map from every plugin ID to concrete actionability implementation path.  
**Demo/Validation**:
- Matrix row count equals executable plugin count.
- No plugin marked “unknown/missing lane.”

### Task 2.1: Generate Current-State Matrix from Latest Baseline
- **Location**: `docs/release_evidence/`, `scripts/audit_plugin_actionability.py`.
- **Description**: Emit machine-readable matrix containing plugin_id, reason_code, finding_kinds, current_actionability_state, required remediation lane.
- **Complexity**: 4
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - 1 row per executable plugin.
  - Explicit lane assignment for each plugin.
- **Validation**:
  - Matrix diff test against run plugin counts.

### Task 2.1a: Baseline Artifact Locking
- **Location**: `scripts/`, `docs/release_evidence/`.
- **Description**: Create a deterministic baseline lock artifact with hashes and run metadata before generating matrix outputs.
- **Complexity**: 3
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Lock file is generated once and reused for all remediation comparisons.
  - Matrix generation fails if lock artifact is missing or hash-mismatched.
- **Validation**:
  - Unit test for lock read/validate behavior.

### Task 2.2: Add “No Plugin Left Behind” Contract Gate
- **Location**: `scripts/verify_agent_execution_contract.py`.
- **Description**: Introduce strict check `run.every_plugin_actionable` that fails if any plugin lands in explanation lane.
- **Complexity**: 6
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Contract fails on any non-actionable plugin.
  - Gate appears in finalizer artifacts.
- **Validation**:
  - Contract unit tests with positive/negative fixtures.

## Sprint 3: Convert Observation-Only Plugins into Action Emitters
**Goal**: Remove `OBSERVATION_ONLY` outcomes by emitting actionable recommendations directly from statistical outputs.  
**Demo/Validation**:
- `OBSERVATION_ONLY` count reaches zero.
- Every converted plugin emits at least one ranked recommendation.

### Task 3.1: Observation-to-Action Adapter Library
- **Location**: `src/statistic_harness/core/report.py`.
- **Description**: Add canonical adapters for observational finding families (e.g., topology, PCA/factor, survival, concurrency, drift summaries) to map signals into process-targeted action candidates.
- **Complexity**: 8
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - Adapter coverage exists for every observation-only finding kind.
  - All adapter outputs include action_type, target, modeled delta fields.
- **Validation**:
  - Adapter unit tests by finding kind.

### Task 3.1a: Observation Reason-Code Catalog and Coverage Gate
- **Location**: `docs/release_evidence/`, `src/statistic_harness/core/report.py`, `scripts/verify_agent_execution_contract.py`.
- **Description**: Build explicit catalog `observation_reason_code -> finding_kind -> adapter_handler` and fail hard when a new observation code lacks adapter coverage.
- **Complexity**: 5
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Every observation reason code in baseline lock has mapped adapter.
  - New unmapped observation code causes contract failure.
- **Validation**:
  - Contract fixture with intentionally unmapped code.

### Task 3.2: Process Target Inference for Observation Families
- **Location**: `src/statistic_harness/core/report.py`, `src/statistic_harness/core/process_matcher.py`.
- **Description**: Enforce deterministic process target resolution from finding evidence/columns/row_ids.
- **Complexity**: 7
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - No converted recommendation lacks process target.
  - Child/non-modifiable targets are promoted to nearest modifiable parent.
- **Validation**:
  - Parent/child policy tests.

## Sprint 4: Convert NO_ACTIONABLE_FINDING_CLASS Plugins
**Goal**: Eliminate non-decision classification by enforcing direct-action finding contracts in plugin emitters.  
**Demo/Validation**:
- `NO_ACTIONABLE_FINDING_CLASS` reaches zero.
- Previously affected plugins generate valid actionable items.

### Task 4.1: Direct-Action Output Contract in Plugin Emitters
- **Location**: targeted plugin folders under `plugins/` and stat registry emitters in `src/statistic_harness/core/stat_plugins/registry.py`.
- **Description**: Retrofit plugin outputs so each emits one of `actionable_ops_lever`, `ideaspace_action`, `verified_action` with complete fields.
- **Complexity**: 9
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - No plugin returns only summary kinds without action payload.
  - Contract fields always populated or deterministic fallback action generated.
- **Validation**:
  - Plugin-specific unit tests for each retrofitted family.

### Task 4.2: Enforce Action Payload Schema in Router
- **Location**: `src/statistic_harness/core/report.py`.
- **Description**: Hard-fail malformed action payloads into deterministic remediation diagnostics during test runs, not silent explanation fallback.
- **Complexity**: 6
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Malformed action payloads are test-detectable.
  - Production path still returns deterministic output with explicit failure signature.
- **Validation**:
  - Router schema tests.

## Sprint 5: Resolve Statistical/Prerequisite Non-Actionable Buckets
**Goal**: Convert `NO_STATISTICAL_SIGNAL`, `PLUGIN_PRECONDITION_UNMET`, and related buckets into actionable pathways.  
**Demo/Validation**:
- No plugin ends in these reason buckets for baseline run.

### Task 5.1: Minimal-Impact Action Synthesis for Low-Signal Cases
- **Location**: `src/statistic_harness/core/report.py`.
- **Description**: For low-signal methods, emit low-confidence but explicit micro-actions (small modeled delta allowed) instead of non-actionable outcomes.
- **Complexity**: 7
- **Dependencies**: Sprint 3
- **Acceptance Criteria**:
  - Low-signal plugins still produce ranked actionable recommendations.
  - Confidence and modeled deltas clearly reported.
- **Validation**:
  - Tests for low-signal fallback generation.

### Task 5.2: Prerequisite Self-Recovery Paths
- **Location**: affected plugins + normalization hooks in `transform_normalize_mixed`.
- **Description**: Add internal fallback derivations for commonly missing prerequisites so plugins can still compute.
- **Complexity**: 8
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - Missing prerequisite reason buckets collapse to near-zero.
  - Plugin still computes statistical method output and emits action.
- **Validation**:
  - Fixture variants with intentionally missing columns/roles.

## Sprint 6: Non-Analysis Plugin Actionability (Profile/Transform/Report/LLM/Planner)
**Goal**: Make all non-analysis plugins emit actionable outputs instead of non-decision classifications.  
**Demo/Validation**:
- `NON_DECISION_PLUGIN` and standalone explanation-only outcomes reach zero.

### Task 6.1: Decision Hooks for Non-Analysis Stages
- **Location**: `plugins/profile_*`, `plugins/transform_*`, `plugins/report_*`, `plugins/llm_*`, `plugins/planner_basic`.
- **Description**: Add decision artifacts that map stage-specific diagnostics to concrete action records (schema fix, data quality fix, prompt fix, report quality fix, transform enablement action).
- **Complexity**: 8
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - Each non-analysis plugin contributes at least one actionable item or deterministic micro-action.
- **Validation**:
  - Unit tests per plugin type.

### Task 6.2: Downstream-Dependency to Action Promotion
- **Location**: `src/statistic_harness/core/report.py`.
- **Description**: When a supporting plugin has downstream consumers, emit explicit action to improve downstream quality/coverage with modeled effect proxy.
- **Complexity**: 6
- **Dependencies**: Task 6.1
- **Acceptance Criteria**:
  - No downstream-only plugin ends without a recommendation record.
- **Validation**:
  - Integration tests with synthetic dependency graphs.

### Task 6.3: Early Metric Envelope for New Action Types
- **Location**: `src/statistic_harness/core/report.py`, `docs/report.schema.json`.
- **Description**: Add required placeholder/initial metric fields for all newly introduced action classes before Sprint 7 so every emitted recommendation is schema-complete.
- **Complexity**: 6
- **Dependencies**: Tasks 6.1-6.2
- **Acceptance Criteria**:
  - No new action type emits missing ranking metric fields.
  - Placeholder policy is deterministic and documented.
- **Validation**:
  - Schema tests for each new action class.

## Sprint 7: Unified Ranking Metrics Across All Recommendations
**Goal**: Ensure every recommendation is directly comparable and client-meaningful.  
**Demo/Validation**:
- Every recommendation row has full metric columns:
  - `efficiency_gain_accounting_month`
  - `efficiency_gain_close_static`
  - `efficiency_gain_close_dynamic`
  - `delta_hours_accounting_month`
  - `delta_hours_close_static`
  - `delta_hours_close_dynamic`
  - user-touch/run-reduction metrics

### Task 7.1: Metric Completion for All Action Types
- **Location**: `src/statistic_harness/core/report.py`.
- **Description**: Backfill modeled deltas for all action classes (not only capacity/batch) with deterministic formulas.
- **Complexity**: 7
- **Dependencies**: Sprints 3-6 (including Task 6.3)
- **Acceptance Criteria**:
  - No recommendation missing required comparison fields.
- **Validation**:
  - Recommendation schema conformance tests.

### Task 7.2: Hard Gate for Comparable Ranking Fields
- **Location**: `scripts/verify_agent_execution_contract.py`, `scripts/finalize_optimal_4pillars.py`.
- **Description**: Contract check fails if any recommendation lacks mandatory ranking metrics.
- **Complexity**: 5
- **Dependencies**: Task 7.1
- **Acceptance Criteria**:
  - Release artifact includes pass/fail metric completeness section.
- **Validation**:
  - Contract checker unit/integration tests.

## Sprint 8: Full-Gauntlet Certification and Evidence
**Goal**: Certify repo as release-ready under strict “all plugins actionable” rule.  
**Demo/Validation**:
- Full gauntlet run status `completed`.
- Contract checks all pass including `run.every_plugin_actionable`.
- Before/after evidence and freshman-ready summary generated.

### Task 8.1: Run and Certify Baseline
- **Location**: `appdata/runs/*`, `docs/release_evidence/*`.
- **Description**: Execute full baseline gauntlet with strict gates and produce final evidence bundle.
- **Complexity**: 4
- **Dependencies**: Sprints 1-7
- **Acceptance Criteria**:
  - `error=0`, `skipped=0`, `explanation_count=0` (or zero non-actionable lane under strict policy).
- **Validation**:
  - Contract artifact + run summary.

### Task 8.2: Validate Against Synthetic Datasets
- **Location**: `docs/release_evidence/*`.
- **Description**: Re-run strict certification on synthetic datasets and compare actionable lift deltas vs baseline.
- **Complexity**: 4
- **Dependencies**: Task 8.1
- **Acceptance Criteria**:
  - Same contract guarantees pass on synthetic runs.
  - Diff report highlights meaningful new insights.
- **Validation**:
  - Compare scripts and contract artifacts.

## Testing Strategy
- Unit:
  - Per-plugin emitter contract tests.
  - Router adapter mapping tests by finding kind.
  - Ranking metric completeness tests.
- Integration:
  - Full recommendation build from report payload.
  - Dependency/parent-promotion and window-sliced scoring behavior.
- End-to-end:
  - Full gauntlet baseline and synthetic.
  - Contract verification with `--recompute-recommendations always`.
- Determinism:
  - Repeat run with same `run_seed` and compare actionability histogram + top-N recommendation identity.
  - Per-sprint test checklist includes fixed seed injection and artifact seed assertion.

## Seed Governance Checklist
1. All new tests must set explicit deterministic `run_seed`.
2. All sprint validation runs must log `run_seed` in emitted artifacts.
3. Final certification diffs are valid only when before/after seeds match exactly.

## Potential Risks & Gotchas
- Risk: forcing actionability on weak-signal plugins may produce low-value noise.
  - Mitigation: enforce confidence-weighted low-impact actions and clear confidence fields.
- Risk: broad adapter rules can over-generalize process targeting.
  - Mitigation: strict target-resolution tests + policy-bound parent promotion.
- Risk: runtime blowups during new action synthesis.
  - Mitigation: bounded complexity caps and resource watchdog checks.
- Risk: contract gates become too strict for non-analysis plugins.
  - Mitigation: plugin-type specific actionable templates with deterministic output guarantees.

## Rollback Plan
1. Keep each sprint in isolated commits.
2. If recommendation quality regresses, revert only affected adapter/emitter sprint commits.
3. Preserve strict telemetry and failure reporting even during rollback.

## Definition of Done
1. Every executable plugin returns actionable recommendation output in full gauntlet runs.
2. No plugin ends in error/skipped/degraded/aborted.
3. No explanation-only plugin outcomes remain under strict certification mode.
4. All recommendations include comparable modeled metrics for accounting month, close static, and close dynamic windows.
5. Baseline + synthetic strict certification runs pass with deterministic evidence artifacts.
