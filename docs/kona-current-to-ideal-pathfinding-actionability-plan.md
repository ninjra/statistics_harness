# Plan: Kona Current-to-Ideal Pathfinding + Actionable Output

**Generated**: 2026-02-21  
**Estimated Complexity**: High

## Overview
Current Kona flow computes:
- current vs ideal vectors,
- per-action modeled delta energy,
- sorted single-step recommendations.

It does not compute a true multi-step route from current state to ideal state.

This plan adds a deterministic pathfinding layer that:
1. Builds a transition graph from verified actions.
2. Solves an ordered action sequence (route) under constraints.
3. Emits actionable route steps with process targets, modeled impact, and validation steps.
4. Integrates route output into recommendation ranking without regressing existing plugin contracts.

## Skill Assignments (What + Why)
- `plan-harder`: structure full phased execution plan with atomic, committable tasks.
- `python-testing-patterns`: define deterministic algorithm tests, fixture-driven route cases, regression tests.
- `testing`: enforce full gate criteria, schema checks, and non-regression of existing Kona outputs.
- `discover-observability`: identify missing route metrics/traces needed for deterministic run evidence.
- `python-observability`: specify concrete instrumentation fields/events for path expansion, pruning, and final route confidence.
- `shell-lint-ps-wsl`: keep run/validation commands shell-safe and consistent with repo command policy.

## Prerequisites
- Existing Kona plugins remain enabled:
  - `analysis_ideaspace_energy_ebm_v1`
  - `analysis_ideaspace_action_planner`
  - `analysis_ebm_action_verifier_v1`
- Existing report and recommendation pipeline remains the integration target.
- Determinism constraints remain mandatory (seeded, stable tie-break ordering).

## Sprint 1: Route Contract + Data Model
**Goal**: Define stable route artifact and algorithm contract before coding search logic.
**Demo/Validation**:
- `route_plan.schema.json` exists and validates sample payload.
- `analysis_ebm_action_verifier_v1` artifacts can be mapped into normalized route candidates deterministically.

### Task 1.1: Add Route Schema Contract
- **Location**: `docs/schemas/kona_route_plan.schema.json` (new)
- **Description**: Define schema for route plan artifact with:
  - route metadata (`run_id`, `dataset_version_id`, `ideal_mode`, seed),
  - ordered `steps`,
  - per-step modeled metrics,
  - aggregate route metrics (`total_delta_energy`, `route_confidence`, `estimated_delta_hours_*`, `efficiency_gain_*`),
  - `not_applicable` reason contract when no route is possible.
- **Complexity**: 4
- **Dependencies**: None
- **Acceptance Criteria**:
  - Schema validates both success and N/A route payloads.
  - Schema disallows empty ambiguous responses.
- **Validation**:
  - Add unit test for schema validation pass/fail fixtures.

### Task 1.2: Define Candidate Transition Record
- **Location**: `src/statistic_harness/core/stat_plugins/ideaspace.py`
- **Description**: Normalize verifier outputs into deterministic transition candidates:
  - `lever_id`, `action_type`, `target_process_ids`, `delta_energy`, `confidence`,
  - preconditions, incompatibilities, and max-use rules.
- **Complexity**: 6
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Candidate ordering deterministic for identical inputs.
  - Unknown/partial rows return structured `N/A` candidate reason.
- **Validation**:
  - New unit test for deterministic normalization order.

### Task 1.3: Add Route Policy Config Surface
- **Location**:
  - `plugins/analysis_ebm_action_verifier_v1/config.schema.json`
  - `plugins/analysis_ebm_action_verifier_v1/plugin.yaml`
- **Description**: Add route controls:
  - `route_max_depth`,
  - `route_beam_width`,
  - `route_min_delta_energy`,
  - `route_min_confidence`,
  - `route_allow_cross_target_steps`,
  - `route_stop_energy_threshold`.
- **Complexity**: 3
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Config defaults preserve current behavior when route mode disabled.
- **Validation**:
  - Config schema tests pass for defaults and invalid values.

## Sprint 2: Deterministic Pathfinding Engine
**Goal**: Implement real multi-step route solving.
**Demo/Validation**:
- Route search emits a multi-step route where sequence outperforms best single step.
- Solver is deterministic with same run seed and input.

### Task 2.1: Implement Route Search Core
- **Location**: `src/statistic_harness/core/ideaspace_route.py` (new)
- **Description**: Implement deterministic beam/A* hybrid:
  - State: projected metrics + accumulated actions.
  - Transition: apply candidate action transform.
  - Score: objective combines energy reduction, confidence, feasibility penalties.
  - Tie-break: stable by score, then lever_id/action tuple.
- **Complexity**: 8
- **Dependencies**: Sprint 1 complete
- **Acceptance Criteria**:
  - Supports depth-limited multi-step plans.
  - Honors incompatibility and duplicate-step constraints.
  - Returns explicit `N/A` with reason when no valid path exists.
- **Validation**:
  - Unit tests: monotonicity, deterministic ties, constraint pruning, no-path behavior.

### Task 2.2: Integrate Solver into EBM Verifier Plugin
- **Location**: `src/statistic_harness/core/stat_plugins/ideaspace.py`
- **Description**:
  - Keep existing per-action scoring.
  - Add route solve stage after candidate scoring.
  - Emit `artifacts/analysis_ebm_action_verifier_v1/route_plan.json`.
  - Emit new finding kind: `verified_route_action_plan`.
- **Complexity**: 7
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Existing `verified_action` findings still emitted.
  - New route finding includes ordered steps and cumulative modeled impact.
- **Validation**:
  - Plugin integration test with crafted 3-step improvement scenario.

### Task 2.3: Add Route Feasibility Guards
- **Location**:
  - `src/statistic_harness/core/stat_plugins/ideaspace.py`
  - `src/statistic_harness/core/actionability_explanations.py`
- **Description**:
  - Block chain-bound/non-modifiable process-only steps.
  - Require direct actionable target (or deterministic N/A reason).
  - Enforce no generic “improve everything” routes.
- **Complexity**: 6
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - Route only contains actionable, modifiable steps.
  - Non-actionable route attempts return clear plain-English reason + fallback.
- **Validation**:
  - Tests for child/parent unmodifiable cases and fallback behavior.

## Sprint 3: Actionable Recommendation Integration
**Goal**: Make route output drive user-facing actionable recommendations.
**Demo/Validation**:
- Freshman output includes route block with step-by-step actions and comparable metrics across windows.

### Task 3.1: Add Route-Aware Recommendation Mapper
- **Location**: `src/statistic_harness/core/report.py`
- **Description**:
  - Map `verified_route_action_plan` into recommendation rows.
  - Include:
    - `delta_hours_accounting_month/close_static/close_dynamic`,
    - `efficiency_gain_*` (dimensionless),
    - per-step user-touch reduction and contention reduction.
- **Complexity**: 7
- **Dependencies**: Sprint 2 complete
- **Acceptance Criteria**:
  - Route recommendations are comparable to single-step recommendations with same metric columns.
- **Validation**:
  - Report tests asserting required fields on each route row.

### Task 3.2: Enhance Plain/Freshman Rendering
- **Location**:
  - `scripts/run_loaded_dataset_full.py`
  - `scripts/show_actionable_results.py`
- **Description**:
  - Render “Current -> Step 1 -> Step N -> Ideal” with clear timescale labels.
  - Show route-level and step-level modeled savings.
  - Keep compact line width and stable field ordering.
- **Complexity**: 5
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Route output readable and non-overflowing in CLI format.
  - Field glossary appears before rows.
- **Validation**:
  - Snapshot tests for formatted output.

### Task 3.3: Dataset-Scoped Ignore Compliance in Route
- **Location**:
  - `src/statistic_harness/core/report.py`
  - `src/statistic_harness/core/report_v2_utils.py`
  - `scripts/run_loaded_dataset_full.py`
- **Description**:
  - Ensure route candidates and route steps honor dataset-specific ignores (e.g., LOS for baseline dataset).
  - Preserve ability for system levers only when policy allows.
- **Complexity**: 4
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Ignored process families never appear as direct route targets for that dataset.
- **Validation**:
  - Dataset-specific unit/integration test for ignore behavior.

## Sprint 4: Observability + Failure Semantics
**Goal**: Make route solving transparent, debuggable, and fail-closed.
**Demo/Validation**:
- Route diagnostics artifact explains why each candidate/branch was pruned or selected.

### Task 4.1: Add Route Search Telemetry
- **Location**: `src/statistic_harness/core/stat_plugins/ideaspace.py`
- **Description**:
  - Emit metrics:
    - `route_candidates_total`,
    - `route_nodes_expanded`,
    - `route_branches_pruned`,
    - `route_search_ms`,
    - `route_solution_depth`,
    - `route_solution_delta_energy`.
  - Emit diagnostic artifact `route_search_trace.json`.
- **Complexity**: 5
- **Dependencies**: Sprint 2 complete
- **Acceptance Criteria**:
  - Trace artifact deterministic and present for `ok` and `na`.
- **Validation**:
  - Observability tests assert trace fields and reason codes.

### Task 4.2: Hard-Fail Contract on Malformed Route Data
- **Location**:
  - `src/statistic_harness/core/stat_plugins/ideaspace.py`
  - `src/statistic_harness/core/plugin_runner.py`
- **Description**:
  - If prerequisites exist but route payload is malformed, return `failed` (not skipped).
  - If prerequisites absent/insufficient, return deterministic `N/A` reason.
- **Complexity**: 4
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - No silent skip for malformed route stage.
- **Validation**:
  - Negative tests for malformed action/energy artifacts.

## Sprint 5: End-to-End Validation + Rollout
**Goal**: Prove route planner generates actionable results on baseline and synthetics.
**Demo/Validation**:
- Full gauntlet completes with route artifacts and route recommendations present.
- Comparison report includes before vs after route quality deltas.

### Task 5.1: Add Route Regression Fixture Set
- **Location**:
  - `tests/fixtures/`
  - `tests/test_kona_energy_ideaspace_architecture.py`
  - `tests/test_report_discovery_recommendations.py`
- **Description**:
  - Add fixtures for:
    - single-step best action,
    - multi-step route superior to any single action,
    - blocked route with deterministic N/A.
- **Complexity**: 6
- **Dependencies**: Sprint 3 complete
- **Acceptance Criteria**:
  - Route solver test coverage includes success, tie, and no-path.
- **Validation**:
  - Dedicated route pytest module passes deterministically.

### Task 5.2: Baseline + Synthetic Route Comparison Script
- **Location**: `scripts/compare_kona_route_quality.py` (new)
- **Description**:
  - Compare runs by dataset:
    - route depth,
    - cumulative delta energy,
    - modeled delta hours by window,
    - user-touch reduction.
- **Complexity**: 5
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - Produces machine-readable JSON and human-readable markdown summary.
- **Validation**:
  - Script test with fixture run artifacts.

### Task 5.3: Full Gate + Evidence Pack
- **Location**:
  - `docs/release_evidence/`
  - existing gauntlet scripts
- **Description**:
  - Run full gauntlet only after plugin-level route tests pass.
  - Store route evidence artifacts and before/after comparisons.
- **Complexity**: 4
- **Dependencies**: Sprint 5 prior tasks
- **Acceptance Criteria**:
  - Route outputs present in final evidence and recommendation bundle.
- **Validation**:
  - `python -m pytest -q` pass.
  - Evidence files generated and schema-valid.

## Testing Strategy
- Unit tests:
  - route state transitions,
  - pruning/constraints,
  - deterministic ordering.
- Plugin integration tests:
  - route artifact generation from verifier prerequisites.
- Report integration tests:
  - route recommendation fields and rendering.
- End-to-end:
  - baseline + synthetic dataset route comparisons.
- Hard gate:
  - Full `python -m pytest -q` must pass before gauntlet rollout.

## Potential Risks & Gotchas
- **Action interaction overestimation**:
  - Mitigation: cap compounding effects and require confidence decay per step.
- **No explicit dependency metadata in lever library**:
  - Mitigation: introduce deterministic rule table in route engine (preconditions/incompatibilities).
- **Route explosion**:
  - Mitigation: depth/beam caps + admissible heuristic + pruning trace.
- **Non-actionable route suggestions**:
  - Mitigation: enforce modifiable-target gate and deterministic N/A fallback.
- **Metric mismatch with client priorities**:
  - Mitigation: require route output to include delta hours + efficiency gains across accounting month/close windows.

## Rollback Plan
- Feature flag route planner:
  - `route_mode=off` falls back to current single-step verifier ranking.
- Keep existing `verified_action` artifact unchanged for backward compatibility.
- If route stage fails in production run:
  - preserve per-action outputs,
  - emit route `N/A` with reason,
  - mark route stage failed in diagnostics.

