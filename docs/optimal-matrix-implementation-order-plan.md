# Plan: Optimal Implementation Order for Remaining Matrix Misses

**Generated**: February 12, 2026  
**Estimated Complexity**: High

## Overview
Current matrix state is structurally complete:
- Hard gaps: `0`
  - no missing normative/binding references
  - no missing redteam required IDs
  - no unclassified plugin data-access contracts
  - no missing plugin dependency edges
- Remaining misses are soft/adoption items:
  - `required_sql_not_using`: `0`
  - `recommended_sql_not_using`: `127`
  - `optional_sql_not_using`: `1`

The optimal path is **not** to blindly add SQL to 127 plugins. That would create overlapping implementations and overwrite risk.  
Instead, implement a supersession-safe sequence:
1) reclassify intent where SQL is not the right tool,  
2) add shared SQL capability in common execution layers,  
3) only then patch individual plugins that remain true gaps.

## Scope Rules (Supersession Guardrails)
- Never implement per-plugin SQL changes before shared-layer upgrades are complete.
- Any plugin that is orchestration/artifact-only remains `not_applicable`.
- SQL-enabled paths must be optional/read-only and deterministic.
- No change may degrade any 4 pillar to improve another.

## Prerequisites
- Matrices refreshed by `scripts/refresh_all_matrices.sh`.
- Baseline artifacts available:
  - `docs/full_repo_misses.json`
  - `docs/full_instruction_coverage_report.json`
  - `docs/plugin_data_access_matrix.json`
  - `docs/sql_assist_adoption_matrix.json`

## Sprint 1: Freeze Baseline + Candidate Partition
**Goal**: Lock a stable baseline and split the 127 recommended candidates into non-overlapping work groups.
**Demo/Validation**:
- A partition artifact exists where every plugin maps to exactly one group.
- No plugin appears in multiple implementation tracks.

### Task 1.1: Produce Candidate Partition Matrix
- **Location**: `docs/sql_adoption_partition_matrix.json` (new), `scripts/full_repo_misses.py`
- **Description**: Partition recommended plugins into:
  - `group_A_shared_layer_covered` (will be solved by registry/helper changes)
  - `group_B_direct_sql_benefit` (needs plugin-local SQL path)
  - `group_C_not_applicable_reclassify` (should not use SQL)
  - `group_D_optional_defer` (low value / high risk for now)
- **Complexity**: 6
- **Dependencies**: None
- **Acceptance Criteria**:
  - Coverage: 127/127 classified.
  - Exclusivity: each plugin in exactly one group.
- **Validation**:
  - Validation script fails on overlaps or unclassified items.

### Task 1.2: Add Supersession Metadata
- **Location**: `docs/sql_assist_intent_overrides.json`
- **Description**: Add `superseded_by` notes for plugins expected to be resolved via shared changes so no duplicate implementation occurs.
- **Complexity**: 4
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Every Group A plugin references the shared change that supersedes local patching.
- **Validation**:
  - Lint check confirms `superseded_by` target exists.

## Sprint 2: Shared-Layer SQL Enablement (Highest ROI)
**Goal**: Implement common SQL retrieval support in shared execution paths to reduce per-plugin duplicate work.
**Demo/Validation**:
- Group A plugins show adoption impact without per-plugin SQL rewrites.

### Task 2.1: Enhance Shared Stat Plugin Execution Layer
- **Location**: `src/statistic_harness/core/stat_plugins/registry.py`, shared helpers under `src/statistic_harness/core/stat_plugins/`
- **Description**: Add optional SQL-backed retrieval APIs (read-only) for stat plugin wrappers where equivalent semantics can be preserved.
- **Complexity**: 8
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Shared layer supports deterministic SQL retrieval with fallback to existing behavior.
  - No output schema regressions.
- **Validation**:
  - Wrapper plugin regression tests pass unchanged expectations.

### Task 2.2: Update Detection Tooling to Recognize Shared Adoption
- **Location**: `scripts/plugin_data_access_matrix.py`, `scripts/sql_assist_adoption_matrix.py`
- **Description**: Ensure matrix logic credits shared-layer SQL enablement and avoids false “not using SQL” for superseded plugins.
- **Complexity**: 6
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Post-refresh, Group A plugins no longer counted as misses.
- **Validation**:
  - Matrix verify checks pass with reduced recommended misses.

## Sprint 3: Targeted Plugin-Local SQL Adoption
**Goal**: Implement SQL only where shared-layer changes do not close the gap.
**Demo/Validation**:
- Group B plugins each have explicit SQL path + evidence of benefit.

### Task 3.1: Rank Group B by 4-Pillar Impact
- **Location**: `docs/sql_adoption_execution_order.md` (new), `scripts/run_hotspots_report.py`
- **Description**: Rank plugins by expected impact (runtime, evidence quality, determinism, security).
- **Complexity**: 5
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - Ordered implementation queue with rationale per plugin.
- **Validation**:
  - Ordering artifact generated and committed.

### Task 3.2: Implement Batch 1 (Top 10 High-Impact)
- **Location**: selected `plugins/*/plugin.py`
- **Description**: Add optional SQL retrieval for top-ranked Group B plugins.
- **Complexity**: 8
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Each plugin keeps deterministic behavior and explicit fallback.
  - No write access to normalized layer.
- **Validation**:
  - Unit + integration tests for each changed plugin.

### Task 3.3: Reclassify Group C/D After Evidence
- **Location**: `docs/sql_assist_intent_overrides.json`
- **Description**: Move plugins to `not_applicable` or `optional` where SQL is demonstrably not beneficial.
- **Complexity**: 4
- **Dependencies**: Sprint 3.2
- **Acceptance Criteria**:
  - Remaining misses reflect true gaps, not classification debt.
- **Validation**:
  - `docs/full_repo_misses.json` shows shrinking recommended miss count with rationale.

## Sprint 4: Enforcement + Drift Prevention
**Goal**: Ensure future changes stay matrix-first and overwrite-safe.
**Demo/Validation**:
- CI/test gates reject unplanned overlap and stale matrices.

### Task 4.1: Add “No Overlap” Gate
- **Location**: `tests/test_sql_adoption_partition_matrix.py` (new), `scripts/full_repo_misses.py`
- **Description**: Enforce that each recommended plugin belongs to exactly one partition group and has a clear execution path.
- **Complexity**: 5
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Test fails when overlap/unclassified entries exist.
- **Validation**:
  - Targeted pytest passes.

### Task 4.2: Matrix-First Preflight Script
- **Location**: `scripts/refresh_all_matrices.sh`, `scripts/full_repo_misses.py`
- **Description**: Add single preflight command that refreshes all matrices and prints hard/soft misses before implementation.
- **Complexity**: 3
- **Dependencies**: None
- **Acceptance Criteria**:
  - One command yields authoritative miss state.
- **Validation**:
  - Script completes and outputs deterministic counts.

## Testing Strategy
- Per sprint:
  - matrix refresh + verify (`docs`, `binding`, `redteam`, `data_access`, `sql_adoption`, `full_instruction`, `full_repo_misses`)
  - targeted tests for changed scripts/plugins
- Final gate (only after implementation complete):
  - `python -m pytest -q`

## Potential Risks & Gotchas
- Shared-layer SQL changes may accidentally alter plugin semantics.
  - Mitigation: keep read-only fallback + snapshot regressions.
- Overeager SQL adoption can reduce accuracy/citeability.
  - Mitigation: intent + partition model with explicit reclassification path.
- Duplicate local + shared implementations can diverge.
  - Mitigation: `superseded_by` metadata and no-overlap test gate.

## Rollback Plan
- Revert shared SQL layer changes first if regressions appear.
- Keep partition/intent artifacts (documentation-only) for quick re-planning.
- Re-run matrix refresh to confirm rollback state is coherent.
