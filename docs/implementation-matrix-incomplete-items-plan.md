# Plan: Implementation Matrix Incomplete Items Closure

**Generated**: February 12, 2026  
**Estimated Complexity**: High

## Overview
Matrix verification is currently healthy for reference integrity, but capability coverage is incomplete:
- `docs/implementation_matrix.json`: no missing normative references.
- `docs/binding_implementation_matrix.json`: no missing binding references.
- `docs/redteam_ids_matrix.json`: no missing required IDs.
- `docs/plugin_data_access_matrix.json`: `36` plugins have no detected read path contract (`dataset loader`, `iter batches`, `sql assist`, or `sql direct`).
- `docs/sql_assist_adoption_matrix.json`: only `3/171` plugins currently register SQL usage.

The goal is to close true implementation gaps, not force SQL where it is inappropriate. We will formalize per-plugin access contracts first, then expand SQL-assist adoption where it improves the 4 pillars.

## Prerequisites
- Existing matrix generators:
  - `scripts/docs_coverage_matrix.py`
  - `scripts/binding_implementation_matrix.py`
  - `scripts/plugins_functionality_matrix.py`
  - `scripts/plugin_data_access_matrix.py`
  - `scripts/sql_assist_adoption_matrix.py`
- Existing test gates:
  - `tests/test_docs_implementation_matrix.py`
  - `tests/test_binding_implementation_matrix.py`
  - `tests/test_plugins_functionality_matrix.py`
  - `tests/test_redteam_ids_matrix.py`

## Sprint 1: Contract Baseline
**Goal**: Define what “complete” means for plugin data access and SQL-assist adoption.
**Demo/Validation**:
- Matrix output includes explicit contract categories per plugin.
- No plugin is left in an ambiguous “unknown access path” state.

### Task 1.1: Add Access Contract Taxonomy
- **Location**: `scripts/plugin_data_access_matrix.py`, `docs/plugin_data_access_matrix.md`
- **Description**: Add explicit `access_contract` classification:
  - `dataset_loader`
  - `iter_batches`
  - `sql_assist`
  - `sql_direct`
  - `artifact_only` (valid no-read plugins)
  - `orchestration_only` (planner/report/meta plugins)
- **Complexity**: 6
- **Dependencies**: None
- **Acceptance Criteria**:
  - Every plugin maps to one or more valid contracts.
  - “No detected access path” is reduced to zero by explicit classification.
- **Validation**:
  - Matrix generation succeeds and a new test asserts no unclassified plugins.

### Task 1.2: Add SQL-Adoption Intent Field
- **Location**: `scripts/sql_assist_adoption_matrix.py`, `docs/sql_assist_adoption_matrix.md`
- **Description**: Add per-plugin intent field:
  - `required`
  - `recommended`
  - `optional`
  - `not_applicable`
- **Complexity**: 5
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Low SQL usage is not treated as failure for `not_applicable` plugins.
  - SQL-adoption completeness is measured against intent, not raw count.
- **Validation**:
  - New matrix summary reports `% complete by intent`.

## Sprint 2: Close the 36 Access-Contract Gaps
**Goal**: Eliminate ambiguous read-path detection for all plugins currently flagged.
**Demo/Validation**:
- `docs/plugin_data_access_matrix.json` reports zero ambiguous plugins.

### Task 2.1: Normalize Detector Coverage for Shared Helpers
- **Location**: `scripts/plugin_data_access_matrix.py`
- **Description**: Update static detector to recognize shared read helpers and wrapper APIs used by analysis/report/planner plugins.
- **Complexity**: 7
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Detector correctly identifies access in helper-layer calls.
  - False negatives are minimized for plugins using abstractions.
- **Validation**:
  - Unit tests with representative plugin snippets.

### Task 2.2: Add Explicit Metadata Overrides for Edge Plugins
- **Location**: `docs/plugin_data_access_overrides.json` (new), `scripts/plugin_data_access_matrix.py`
- **Description**: Add reviewable overrides for plugins that are intentionally artifact-only or orchestration-only.
- **Complexity**: 4
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Overrides are minimal, documented, and deterministic.
  - No hidden fallback behavior.
- **Validation**:
  - Test ensures every override references an existing plugin and valid contract type.

## Sprint 3: SQL-Assist Expansion by Intent
**Goal**: Increase SQL-assist adoption where it materially improves throughput, citable evidence, and determinism.
**Demo/Validation**:
- SQL-adoption matrix shows intent-based completion improvement.
- No regression in pytest gauntlet.

### Task 3.1: Prioritize Candidate Plugins
- **Location**: `docs/sql_assist_adoption_matrix.json`, `docs/plugins_functionality_matrix.json`
- **Description**: Create ranked rollout list for `required` and `recommended` plugins, prioritizing:
  - high runtime impact,
  - high evidence-value findings,
  - repeated data-scan hotspots.
- **Complexity**: 5
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Top rollout batch is explicit and traceable.
  - Candidate rationale is recorded per plugin.
- **Validation**:
  - Checklist artifact generated with ranking inputs.

### Task 3.2: Implement SQL-Assist Adapters for Batch 1
- **Location**: plugin-specific `plugins/*/plugin.py`, shared SQL helper modules under `src/statistic_harness/core/`
- **Description**: Add SQL-assist read paths for the first priority batch, keeping plugin behavior deterministic and read-only.
- **Complexity**: 8
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Plugins preserve outputs and schema contracts.
  - SQL paths are optional and safe to bypass.
- **Validation**:
  - Plugin unit tests + integration tests updated.

## Sprint 4: Gates and Governance
**Goal**: Prevent matrix drift and incomplete classification from returning.
**Demo/Validation**:
- CI-level verify commands fail on stale/incomplete matrix state.

### Task 4.1: Add Matrix Completeness Test Gates
- **Location**: `tests/test_plugin_data_access_matrix.py` (new), `tests/test_sql_assist_adoption_matrix.py` (new)
- **Description**: Add tests enforcing:
  - no ambiguous access contracts,
  - SQL intent coverage metrics within configured thresholds.
- **Complexity**: 6
- **Dependencies**: Sprints 1–3
- **Acceptance Criteria**:
  - Fails closed if matrix contracts regress.
- **Validation**:
  - `python -m pytest -q` passes with new tests enabled.

### Task 4.2: Add One-Command Matrix Refresh Workflow
- **Location**: `scripts/refresh_all_matrices.sh` (new), `docs/implementation_matrix.md`
- **Description**: Single command refreshes all matrices before development and before merge.
- **Complexity**: 3
- **Dependencies**: None
- **Acceptance Criteria**:
  - Workflow is deterministic and documented.
- **Validation**:
  - Script run produces no diff when repository is already current.

## Testing Strategy
- Matrix verify gates:
  - `python scripts/docs_coverage_matrix.py --verify`
  - `python scripts/binding_implementation_matrix.py --extra-doc topo-tda-addon-pack-plan.md --verify`
  - `python scripts/redteam_ids_matrix.py --verify`
  - `python scripts/plugin_data_access_matrix.py --verify`
  - `python scripts/sql_assist_adoption_matrix.py --verify`
- Unit tests for detector logic and override validity.
- Full gauntlet: `python -m pytest -q` before shipping.

## Potential Risks & Gotchas
- Static detection can miss indirect helper calls.
  - Mitigation: add explicit override metadata with review rules.
- Forcing SQL assist on non-candidate plugins can reduce clarity.
  - Mitigation: intent-based policy (`required/recommended/optional/not_applicable`).
- Contract labels can drift if plugin patterns change.
  - Mitigation: CI gates plus one-command matrix refresh.

## Rollback Plan
- Keep existing matrix schemas backward compatible.
- If detector changes cause false positives:
  - revert to previous detector behavior,
  - retain override file for explicit contracts,
  - re-run matrix generators and tests.
