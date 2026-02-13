# Plan: Repo Improvements Catalog V3 Implementation Path

**Generated**: 2026-02-13
**Estimated Complexity**: High

## Overview
This plan turns `docs/repo_improvements_catalog_v3.json` into an executable, low-risk improvement program for this repository. The catalog currently contains 120 templated items, references a non-existent core file (`src/statistic_harness/core/plugin_registry.py`), and proposes 120 plugin IDs that do not exist under `plugins/`. The implementation path therefore starts with catalog normalization and mapping, then executes improvements in controlled waves with full pytest gates and determinism/security constraints.

## Assumptions
- In scope: implementing the catalog as a planning/execution artifact and delivering improvements in phased waves.
- Out of scope for first wave: shipping all 120 new plugins at once.
- Existing plugin architecture remains: manifest-driven plugins in `plugins/*`, handlers in `src/statistic_harness/core/stat_plugins/registry.py`, orchestration via `src/statistic_harness/core/pipeline.py`.
- Release gate remains mandatory: `python -m pytest -q` must pass before shipping.

## Prerequisites
- Python environment with repo dev dependencies installed.
- Existing matrix/document generator scripts remain source of truth.
- Agreement to execute improvements by prioritized tranche, not all-at-once.

## Sprint 0: Catalog Canonicalization
**Goal**: De-template the 120 repetitive catalog rows into canonical, implementation-meaningful units before scoring/execution.
**Demo/Validation**:
- Produce canonicalized catalog with cluster assignments and merge rationale.
- Show reduction report (`raw_items`, `canonical_items`, reduction_percent).

### Task 0.1: Build Canonicalization Pass
- **Location**: `scripts/canonicalize_repo_improvements_catalog.py`
- **Description**: Collapse near-duplicate rows into canonical items by category + method intent + acceptance profile, and emit `canonical_item_id` and `cluster_id`.
- **Complexity**: 6/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - Canonical output is deterministic and traceable back to source IDs.
  - Merge rationale included per canonical item.
- **Validation**:
  - Golden fixture test for canonicalization output.

### Task 0.2: Add Canonicalization Reduction Report
- **Location**: `docs/repo_improvements_catalog_v3.reduction_report.json`
- **Description**: Persist summary metrics and source-to-canonical mapping for auditability.
- **Complexity**: 3/10
- **Dependencies**: Task 0.1
- **Acceptance Criteria**:
  - Every source item maps to exactly one canonical item.
- **Validation**:
  - Mapping integrity test (`no_orphans`, `no_duplicates`).

## Sprint 1: Catalog Normalization Foundation
**Goal**: Make the catalog structurally valid and repo-aligned so it can drive implementation safely.
**Demo/Validation**:
- Run normalization script and produce normalized catalog artifact.
- Verify no invalid touchpoints remain and all items have valid execution status metadata.

### Task 1.1: Add Catalog Schema
- **Location**: `docs/repo_improvements_catalog.raw.schema.json`, `docs/repo_improvements_catalog.normalized.schema.json`
- **Description**: Define versioned schemas for raw and normalized catalogs (IDs, categories, plugin proposals, touchpoints, dependencies, acceptance criteria, four-pillars impact, canonicalization metadata).
- **Complexity**: 4/10
- **Dependencies**: Sprint 0
- **Acceptance Criteria**:
  - Raw schema validates source catalog format.
  - Normalized schema validates transformed format.
  - Rejects missing required fields and malformed impact payloads.
- **Validation**:
  - Add tests using `jsonschema` for valid/invalid samples.

### Task 1.2: Build Normalizer for V3 Catalog
- **Location**: `scripts/normalize_repo_improvements_catalog.py`, `docs/repo_improvements_touchpoint_map.json`
- **Description**: Read canonicalized catalog, apply strict touchpoint allowlist mapping (`plugin_registry.py` -> real integration files), add derived fields (`normalized_touchpoints`, `implementation_status`, `priority_score_inputs`), and emit normalized output.
- **Complexity**: 6/10
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Produces `docs/repo_improvements_catalog_v3.normalized.json` deterministically.
  - Fails closed on unresolved or ambiguous touchpoints.
  - Fails closed on unknown categories unless explicitly mapped to `deferred`.
- **Validation**:
  - Script idempotence test (same output hash across repeated runs).

### Task 1.3: Add Catalog Validation Test Harness
- **Location**: `tests/test_repo_improvements_catalog.py`
- **Description**: Add tests to enforce schema validity, deterministic normalization, and zero unresolved touchpoints in normalized output.
- **Complexity**: 5/10
- **Dependencies**: Task 1.1, Task 1.2
- **Acceptance Criteria**:
  - Tests fail if catalog drifts from schema or normalization rules.
- **Validation**:
  - `python -m pytest -q tests/test_repo_improvements_catalog.py`

## Sprint 2: Mapping and Prioritization Engine
**Goal**: Convert normalized catalog items into an actionable implementation queue mapped to real repo components.
**Demo/Validation**:
- Generate ranked execution plan with tranche assignments.
- Confirm each item is mapped to either existing plugin enhancement or new plugin scaffold candidate.

### Task 2.1: Implement Capability Mapping Rules
- **Location**: `scripts/map_repo_improvements_to_capabilities.py`
- **Description**: Map each catalog item/category to one of:
  - existing plugin enhancement,
  - new plugin scaffold target,
  - deferred/no-go (with local-only/offline reason).
- **Complexity**: 7/10
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - 100% of items classified.
  - Mapping output includes rationale and target paths.
- **Validation**:
  - Unit tests for mapping rules and fallback behavior.

### Task 2.2: Add Prioritization Scoring
- **Location**: `scripts/plan_repo_improvements_rollout.py`
- **Description**: Compute priority score from four-pillars expected impact, dependency depth, implementation risk, and estimated effort; assign sprint wave (`wave_1`, `wave_2`, `wave_3`) with deterministic tie-break sort `(-priority_score, dependency_depth, canonical_item_id)`.
- **Complexity**: 6/10
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Every item has deterministic score and wave.
  - Highest-priority wave is balanced across categories.
- **Validation**:
  - Determinism test and golden-output fixture test.

### Task 2.3: Emit Human + Machine Execution Plan
- **Location**: `docs/repo_improvements_execution_plan_v1.json`, `docs/repo_improvements_execution_plan_v1.md`
- **Description**: Generate machine-readable and markdown execution plans linked to real files/tests and acceptance checks.
- **Complexity**: 4/10
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - JSON + Markdown outputs generated from same source.
  - All wave-1 items are independently committable.
- **Validation**:
  - Snapshot tests for both outputs.

### Task 2.4: Dependency and Cycle Integrity Checks
- **Location**: `scripts/validate_repo_improvement_dependencies.py`, `tests/test_repo_improvement_dependencies.py`
- **Description**: Validate dependency graph completeness, detect cycles, flag unknown plugin targets, and fail planning if dependency integrity is broken.
- **Complexity**: 5/10
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - No cycles in execution graph.
  - Unknown prerequisites are explicitly deferred with reasons.
- **Validation**:
  - dependency-graph unit tests + synthetic cycle fixtures.

## Sprint 3: Wave-1 Implementation (Pilot)
**Goal**: Deliver a safe, demoable first tranche proving the catalog pipeline and implementation model.
**Demo/Validation**:
- Ship 10-15 high-priority items with green tests and report impact artifacts.
- Demonstrate at least one completed item per major category.

### Task 3.1: Wave-1 Scaffold Preflight
- **Location**: `scripts/scaffold_repo_improvement_plugins.py`, `plugins/analysis_plugin_*_v1/`
- **Description**: Create minimal manifests/wrappers for selected wave-1 missing plugin IDs so handler work is not blocked by discovery failures.
- **Complexity**: 6/10
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - Selected wave-1 missing IDs discover successfully.
  - No scaffolded plugin violates `sandbox.no_network` contract.
- **Validation**:
  - `tests/test_plugin_manifest_schema.py`
  - `tests/test_plugin_discovery.py`

### Task 3.2: Execute Existing-Plugin Enhancements First
- **Location**: plugin-specific files under `src/statistic_harness/core/stat_plugins/` and corresponding `plugins/*` wrappers
- **Description**: Prioritize wave-1 items that improve existing handlers (low integration risk) before introducing new plugin IDs.
- **Complexity**: 8/10
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Each item has linked commit, tests, and acceptance result.
  - No regression in determinism/offline/security guardrails.
- **Validation**:
  - Per-item unit/integration tests.
  - `tests/test_offline.py`, `tests/test_security_paths.py`, `tests/test_missing_plugins.py`.

### Task 3.3: Expand Missing Plugin IDs for Approved Wave-1 Gaps
- **Location**: `scripts/scaffold_repo_improvement_plugins.py`, `plugins/analysis_plugin_*_v1/`
- **Description**: Scaffold only approved wave-1 missing plugins using existing manifest conventions and registry handler routing patterns.
- **Complexity**: 7/10
- **Dependencies**: Task 3.2
- **Acceptance Criteria**:
  - New plugin manifests validate via `PluginManager.discover()`.
  - New handlers return structured, citable outputs.
- **Validation**:
  - `tests/test_plugin_manifest_schema.py`
  - plugin-specific tests under `tests/plugins/`.

### Task 3.4: Add Wave-1 Evaluation Harness Hooks
- **Location**: `tests/test_evaluator.py`, `tests/fixtures/ground_truth_*.yaml`, optional new `tests/fixtures/ground_truth_repo_improvements_wave1.yaml`
- **Description**: Extend evaluator checks so wave-1 claims are asserted in `report.json` with numeric tolerances against explicit baseline fixtures.
- **Complexity**: 6/10
- **Dependencies**: Task 3.2, Task 3.3
- **Acceptance Criteria**:
  - Wave-1 improvements produce measurable evaluator outcomes with predeclared threshold table.
- **Validation**:
  - evaluator test subset + `python -m pytest -q`.

## Sprint 4: Scale-Out Automation and Governance
**Goal**: Make wave-2/3 execution repeatable with strict quality, resource, and evidence controls.
**Demo/Validation**:
- One command regenerates catalog normalization, mapping, rollout docs, and verification checks.
- Governance report shows status per improvement item.

### Task 4.1: Unified Improvement Pipeline Command
- **Location**: `scripts/run_repo_improvements_pipeline.py`
- **Description**: Compose existing update/verify script chain plus repo-improvements steps into one deterministic pipeline command (avoid duplicating existing matrix orchestration).
- **Complexity**: 5/10
- **Dependencies**: Sprints 1-3
- **Acceptance Criteria**:
  - Single command updates all improvement artifacts.
  - Non-zero exit on any validation failure.
- **Validation**:
  - CI/local smoke test for pipeline script.

### Task 4.2: Integrate with Existing Matrix Verification
- **Location**: `scripts/update_docs_and_plugin_matrices.py`, `scripts/verify_docs_and_plugin_matrices.py`, plus new improvement verify script if needed
- **Description**: Extend doc/matrix update + verify flow to include repo improvements artifacts.
- **Complexity**: 5/10
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Improvement artifacts participate in existing verify workflow.
- **Validation**:
  - dedicated tests + existing matrix tests.

### Task 4.3: Add Governance Status Ledger
- **Location**: `docs/repo_improvements_status.json`, `docs/repo_improvements_status.md`
- **Description**: Track status (`todo`, `in_progress`, `done`, `deferred`), owner, linked PR, validation status, measured impact, and traceability (`catalog_item -> code_path -> test_ids -> report_keys`).
- **Complexity**: 4/10
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Every catalog item has lifecycle status and traceability metadata.
- **Validation**:
  - schema + snapshot tests.

## Sprint 5: Release Hardening and Adoption
**Goal**: Operationalize the improvements program as a maintainable release discipline.
**Demo/Validation**:
- Release checklist includes improvements pipeline, full test pass, and documentation handoff.

### Task 5.1: CI Gate for Improvement Artifacts
- **Location**: project CI workflow and verification scripts
- **Description**: Ensure PRs that touch catalog/mapping/plugins fail if normalization, mapping, or status artifacts are stale.
- **Complexity**: 5/10
- **Dependencies**: Sprint 4
- **Acceptance Criteria**:
  - CI catches drift before merge.
- **Validation**:
  - intentionally stale artifact test in CI branch.

### Task 5.2: Operator Runbook and Contribution Guide
- **Location**: `docs/repo_improvements_runbook.md`
- **Description**: Document commands, decision rules, wave selection, rollback steps, and acceptance evidence requirements.
- **Complexity**: 3/10
- **Dependencies**: Sprint 4
- **Acceptance Criteria**:
  - New contributor can add one improvement item end-to-end without tribal knowledge.
- **Validation**:
  - dry-run by a second engineer.

## Testing Strategy
- Unit tests for canonicalization, schema, normalization, mapping, prioritization, dependency integrity, and status ledger.
- Integration tests for improvement pipeline orchestration and artifact generation.
- Plugin-level tests for each new/modified wave item.
- Per-wave mandatory offline/security/fail-closed checks:
  - `tests/test_offline.py`
  - `tests/test_security_paths.py`
  - `tests/test_missing_plugins.py`
- Mandatory full-suite gate before shipping:
  - `python -m pytest -q`
- Determinism checks:
  - stable output hash for canonicalization + normalization + rollout artifacts.
  - stable ordering via explicit tie-break keys.

## Potential Risks & Gotchas
- Catalog-template risk: highly repetitive entries can create noisy or low-value implementation churn.
  - Mitigation: map to capabilities first; enforce wave quotas and acceptance evidence.
- Wrong integration points: catalog references non-existent `plugin_registry.py` path.
  - Mitigation: normalization rules + failing test on unresolved touchpoints.
- Scale risk: implementing all 120 directly can destabilize release cadence.
  - Mitigation: wave-based rollout with strict stop/go gates after each wave.
- Evidence inflation: “improvement” claims without measured deltas.
  - Mitigation: baseline fixtures + evaluator threshold table + status ledger fields for measured outcomes.
- Resource-policy drift: new plugins may violate CPU/RAM constraints.
  - Mitigation: include governance checks and runtime budget validation in acceptance tests.
- LLM/agentic scope conflict with local-only policy.
  - Mitigation: no-go rubric in mapping stage for proposals requiring network/external services.

## Rollback Plan
- Keep generated artifacts deterministic and reproducible from source catalog.
- If wave introduces regressions:
  - revert wave commit set,
  - regenerate normalization/mapping/status artifacts,
  - rerun full pytest gate.
- Maintain improvement status ledger to mark reverted items and reasons.
