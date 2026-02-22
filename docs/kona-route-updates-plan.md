# Plan: Kona Route Updates (Merged Two-Doc Execution Plan)

**Generated**: 2026-02-21  
**Estimated Complexity**: High

## Overview
This plan merges requirements from:
1. `codex_kona_route_updates.md` (hard implementation contract).
2. `docs/kona-current-to-ideal-pathfinding-actionability-plan.md` (phased delivery + actionability focus).

Objective:
- Add deterministic Kona current-to-ideal multi-step pathfinding.
- Keep route planning OFF by default.
- Preserve existing single-step behavior/artifacts.
- Emit route outputs that are actionable and report-integrated.

## Skill Plan (What + Why)
- `plan-harder`: primary phased planning structure and atomic task breakdown.
- `shell-lint-ps-wsl`: command policy compliance for every run/verification command.
- `python-testing-patterns`: deterministic algorithm and schema test design.
- `testing`: full gate and regression strategy for plugin/report/pipeline behavior.
- `discover-observability`: identify required route diagnostics and run evidence metrics.
- `python-observability`: define concrete telemetry fields, trace artifacts, and failure semantics.

## Assumptions
- No blocking ambiguity remains; both source docs are explicit enough to plan full scope.
- Existing ideaspace/EBM plugin contracts remain backward-compatible unless route is explicitly enabled.

## Prerequisites
- Existing files/components:
  - `src/statistic_harness/core/stat_plugins/ideaspace.py`
  - `plugins/analysis_ebm_action_verifier_v1/*`
  - `src/statistic_harness/core/report.py`
  - `scripts/run_loaded_dataset_full.py`
  - `src/statistic_harness/core/stat_plugins/code_hash.py`
- Existing tests pass before new route implementation branch work starts.

## Sprint 1: Contract Lockdown + Preflight
**Goal**: Freeze exact contract and avoid duplicate logic/schema creation.  
**Demo/Validation**:
- Preflight report listing existing route/schema/code surfaces.
- Confirmed implementation map showing where each requirement lands.

### Task 1.1: Repo Preflight Duplicate Scan
- **Location**: repo-wide scan notes in `docs/release_evidence/` (new summary artifact).
- **Description**: Search for existing route schema, route finding kinds, config keys, and helper overlaps before edits.
- **Complexity**: 2
- **Dependencies**: None
- **Acceptance Criteria**:
  - One preflight checklist artifact generated with pass/fail for duplicate-risk items.
- **Validation**:
  - Unit-free; deterministic grep/result snapshot in evidence.

### Task 1.2: Requirement Mapping Matrix
- **Location**: `docs/release_evidence/kona_route_requirement_map.json` (new).
- **Description**: Map each clause from `codex_kona_route_updates.md` to exact target file/function.
- **Complexity**: 3
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - 100% clause coverage, including optional cache/hash requirement.
- **Validation**:
  - Manual plus JSON schema consistency checks (if schema used).

### Task 1.3: Skills-to-Sprint Execution Guide
- **Location**: `docs/release_evidence/kona_route_skill_rationale.md` (new).
- **Description**: Document which skill governs each sprint/task category.
- **Complexity**: 1
- **Dependencies**: Task 1.2
- **Acceptance Criteria**:
  - Every sprint has explicit skill rationale.
- **Validation**:
  - File presence and lint pass.

## Sprint 2: Schema + Core Route Engine
**Goal**: Implement deterministic route contract and pure route solver module.  
**Demo/Validation**:
- Schema validates modeled and not_applicable payloads.
- Route solver returns deterministic results for fixed fixtures.

### Task 2.1: Add `kona_route_plan` Artifact Schema
- **Location**: `docs/schemas/kona_route_plan.schema.json`
- **Description**: Add schema exactly per contract in `codex_kona_route_updates.md` (decision gating, steps constraints, totals).
- **Complexity**: 4
- **Dependencies**: Sprint 1 complete
- **Acceptance Criteria**:
  - Schema content matches contract fields/conditionals.
- **Validation**:
  - New tests in Sprint 5 schema contract suite.

### Task 2.2: Add Pure Route Solver Module
- **Location**: `src/statistic_harness/core/ideaspace_route.py`
- **Description**: Implement required dataclasses/API signatures and deterministic beam/A*-style route search semantics.
- **Complexity**: 8
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - API signatures exactly match contract.
  - No `statistic_harness.core.stat_plugins.*` imports in this module.
  - Route disabled behavior returns deterministic `not_applicable` with `ROUTE_DISABLED`.
- **Validation**:
  - Unit tests for deterministic ordering, stop conditions, no-path return.

### Task 2.3: Add Solver-Specific Guardrails
- **Location**: `src/statistic_harness/core/ideaspace_route.py`
- **Description**: Implement candidate filters, target-signature locking, time-budget behavior, and stable tie-break rules exactly as specified.
- **Complexity**: 6
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - Behavior matches documented semantics for:
    - candidate filtering/sorting,
    - route depth/beam limits,
    - threshold/time/max-depth stop reasons.
- **Validation**:
  - Focused unit tests with strict expected outputs.

## Sprint 3: Verifier Plugin Integration (No Regression by Default)
**Goal**: Wire pathfinding into verifier without breaking existing outputs when route disabled.  
**Demo/Validation**:
- Default config produces unchanged single-step behavior.
- Enabled config emits `route_plan.json` and route finding.

### Task 3.1: Extend Verifier Config Surface
- **Location**:
  - `plugins/analysis_ebm_action_verifier_v1/config.schema.json`
  - `plugins/analysis_ebm_action_verifier_v1/plugin.yaml`
- **Description**: Add route keys/defaults exactly:
  - `route_max_depth=0`
  - `route_beam_width=0`
  - `route_min_delta_energy=0.0`
  - `route_min_confidence=0.0`
  - `route_allow_cross_target_steps=false`
  - `route_stop_energy_threshold=1.0`
  - `route_candidate_limit=50`
  - `route_time_budget_ms=0`
  - `route_disallowed_lever_ids=[]`
  - `route_disallowed_action_types=[]`
- **Complexity**: 3
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - Defaults keep routing disabled and preserve existing behavior.
- **Validation**:
  - Config schema tests for defaults + type bounds.

### Task 3.2: Normalize Route Candidates from Verified Actions
- **Location**: `src/statistic_harness/core/stat_plugins/ideaspace.py`
- **Description**: Convert existing verified action records into `RouteCandidate` with canonical action_type/targets and deterministic parsing.
- **Complexity**: 5
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Canonical mapping implemented for required lever IDs.
  - Target parsing from existing `target` string deterministic.
- **Validation**:
  - Plugin unit tests for mapping and target extraction.

### Task 3.3: Call Route Solver When Enabled
- **Location**: `src/statistic_harness/core/stat_plugins/ideaspace.py`
- **Description**:
  - Build `RouteConfig`,
  - call `solve_kona_route_plan(...)`,
  - write `route_plan.json`,
  - emit finding kind `verified_route_action_plan` for modeled decision.
- **Complexity**: 7
- **Dependencies**: Task 3.2
- **Acceptance Criteria**:
  - Route artifacts/findings only when enabled.
  - Existing `verified_actions.json` unchanged.
- **Validation**:
  - Regression test comparing disabled-mode output before/after.

### Task 3.4: Reuse Existing Energy/Constraint Logic
- **Location**: `src/statistic_harness/core/stat_plugins/ideaspace.py`
- **Description**: Refactor minimal helpers so verifier and route callbacks share logic (no duplicated transform or energy equations).
- **Complexity**: 6
- **Dependencies**: Task 3.3
- **Acceptance Criteria**:
  - No duplicated large logic blocks for lever transforms or energy computation.
- **Validation**:
  - Diff review and unit tests for helper equivalence.

## Sprint 4: Reporting + Runner + Actionability Surface
**Goal**: Surface route as first-class actionable recommendations and script outputs.  
**Demo/Validation**:
- Report includes route recommendation item with ordered steps.
- Loader script prefers real route artifact over fabricated fallback route.

### Task 4.1: Report Integration for `verified_route_action_plan`
- **Location**: `src/statistic_harness/core/report.py`
- **Description**: Extend discovery recommendation synthesis to include route finding with:
  - `kind="verified_route_action_plan"`
  - `action_type="route_process"`
  - modeled percent hint from `total_delta_energy/energy_before` where possible.
- **Complexity**: 6
- **Dependencies**: Sprint 3
- **Acceptance Criteria**:
  - Route recommendation text renders numbered steps and summary metrics.
- **Validation**:
  - Report unit tests for route recommendation inclusion and formatting.

### Task 4.2: Route-Aware Dataset Runner Extraction
- **Location**: `scripts/run_loaded_dataset_full.py`
- **Description**:
  - Prefer verifier `route_plan.json` for route map extraction when modeled+steps available.
  - Fall back to existing top-verified-actions behavior otherwise.
- **Complexity**: 4
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Script output uses true route when present; legacy behavior preserved otherwise.
- **Validation**:
  - Script-level unit test for route preference/fallback branches.

### Task 4.3: Actionability Guard Alignment
- **Location**:
  - `src/statistic_harness/core/report.py`
  - `src/statistic_harness/core/actionability_explanations.py`
- **Description**: Ensure route finding yields actionable targets or deterministic N/A reason; no generic/non-modifiable chain advice.
- **Complexity**: 5
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Non-actionable route results map to explicit reason code + fallback guidance.
- **Validation**:
  - Actionability tests on route findings and reasons.

## Sprint 5: Tests, Observability, and Cache Safety
**Goal**: Make route feature robust, traceable, and regression-safe.  
**Demo/Validation**:
- Dedicated schema + integration tests pass.
- Cache invalidation includes new route module.
- Route diagnostics available for debugging.

### Task 5.1: Add Schema Contract Test Suite
- **Location**: `tests/test_kona_route_plan_schema.py` (new)
- **Description**: Add modeled/not_applicable valid cases and invalid cases from contract.
- **Complexity**: 4
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - All schema behavior checks pass deterministically.
- **Validation**:
  - `pytest` target file passes.

### Task 5.2: Add Route Integration Tests
- **Location**:
  - `tests/test_kona_energy_ideaspace_architecture.py` (extend)
  - optional dedicated route integration test module.
- **Description**:
  - Enable route settings in fixture,
  - assert `route_plan.json` emitted and schema-valid,
  - assert `verified_route_action_plan` finding exists and has steps.
- **Complexity**: 6
- **Dependencies**: Sprint 3
- **Acceptance Criteria**:
  - Deterministic route plan generation verified.
- **Validation**:
  - Repeated test runs stable (no order drift).

### Task 5.3: Add Route Telemetry + Diagnostics
- **Location**: `src/statistic_harness/core/stat_plugins/ideaspace.py`
- **Description**:
  - Emit route metrics (`expanded_states`, search timing, depth, stop reason),
  - write `route_search_trace.json`,
  - include deterministic debug reasoning path.
- **Complexity**: 5
- **Dependencies**: Sprint 3
- **Acceptance Criteria**:
  - Diagnostics artifact present for modeled and not_applicable route decisions.
- **Validation**:
  - Observability tests asserting required fields.

### Task 5.4: Ensure Cache Hash Includes New Module
- **Location**: `src/statistic_harness/core/stat_plugins/code_hash.py`
- **Description**: Include `src/statistic_harness/core/ideaspace_route.py` in relevant plugin hash inputs without duplicating hash logic.
- **Complexity**: 3
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - Route solver code changes invalidate cached results correctly.
- **Validation**:
  - Cache key unit/regression test for route module modification signal.

## Sprint 6: Full Validation + Rollout Readiness
**Goal**: Prove end-to-end quality and recommendation usefulness against real + synthetic datasets.  
**Demo/Validation**:
- Full gauntlet run after plugin fixes.
- Route recommendations appear in freshman output and compare cleanly with baseline.

### Task 6.1: Baseline + Synthetic Route Comparison Script
- **Location**: `scripts/compare_kona_route_quality.py` (new)
- **Description**: Compare route depth, total delta energy, modeled hours by window, and user-touch reduction across runs.
- **Complexity**: 5
- **Dependencies**: Sprint 5
- **Acceptance Criteria**:
  - JSON + markdown output for release evidence.
- **Validation**:
  - Script test with fixture run dirs.

### Task 6.2: Full Test Gate + Route Evidence Pack
- **Location**:
  - `docs/release_evidence/`
  - existing gauntlet scripts
- **Description**: Run full test gate and gauntlet; capture route plan evidence and before/after recommendation deltas.
- **Complexity**: 4
- **Dependencies**: Task 6.1
- **Acceptance Criteria**:
  - `python -m pytest -q` passes.
  - Route artifacts and route recommendation rows present in evidence.
- **Validation**:
  - Release evidence files generated, reviewed, and deterministic.

## Testing Strategy
- Unit:
  - route solver semantics (filters, scoring, tie-breaks, stop conditions),
  - helper purity and determinism.
- Plugin integration:
  - verifier route path enabled/disabled behavior,
  - route artifact emission correctness.
- Report integration:
  - route recommendation rendering + ranking fields.
- Script integration:
  - route_plan preference in `run_loaded_dataset_full.py`.
- Full gate:
  - `python -m pytest -q` mandatory before gauntlet.

## Potential Risks & Gotchas
- **Route explosion**:
  - Mitigation: strict beam/depth/time limits, deterministic pruning.
- **Multi-step overestimation**:
  - Mitigation: bounded transforms, confidence aggregation, conservative stop threshold.
- **Behavior regression with defaults**:
  - Mitigation: explicit route OFF defaults and regression tests for unchanged outputs.
- **Duplicate logic drift**:
  - Mitigation: extract shared helper(s), forbid duplicated transform/energy blocks.
- **Non-actionable generic route output**:
  - Mitigation: route feasibility gates + deterministic reasoned N/A fallback.
- **Cache stale outputs**:
  - Mitigation: hash include `ideaspace_route.py`.

## Rollback Plan
- Keep route feature gated by config defaults (`route_max_depth=0`, `route_beam_width=0`).
- If route stage fails:
  - continue existing verified single-action flow,
  - mark route decision `not_applicable` with reason,
  - preserve existing artifacts/findings.
- If needed, disable route in config only; no schema removal required.

