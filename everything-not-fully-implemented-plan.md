# Plan: Everything Not Fully Implemented

**Generated**: 2026-02-13
**Estimated Complexity**: High

## Overview
This plan closes the repository's current "not fully implemented" surface in a deterministic, test-gated way. It covers:
- all currently missing hard-gap artifacts reported by `docs/full_repo_misses.json` (40 missing items, 22 unique missing paths),
- the unimplemented repo-improvements execution toolchain referenced by `repo-improvements-catalog-v3-implementation-path-plan.md`,
- remaining blueprint-deliverable feature plugins referenced by `docs/codex_statistics_harness_blueprint.md`.

Approach: implement in vertical slices that each produce demoable outputs, keep generated artifacts in sync, and preserve non-negotiables (offline/local-only, deterministic runs, fail-closed behavior, full `pytest -q` pass).

## Assumptions
- "Everything not fully implemented" includes both:
  - current hard-gap artifacts in `full_repo_misses`, and
  - blueprint feature plugins still absent from `plugins/`.
- If release scope must be narrower, Sprint 6 can be explicitly deferred after Sprint 5 while still closing hard-gap artifact debt.

## Current Gap Inventory
- Missing hard-gap files from instruction coverage:
  - `docs/_codex_repo_manifest.txt` (blueprint-required; currently missing even if not yet flagged as hard-gap)
  - `docs/_codex_plugin_catalog.md`
  - `docs/schemas/changepoints.schema.json`
  - `docs/schemas/process_mining.schema.json`
  - `docs/schemas/causal.schema.json`
  - `docs/repo_improvements_catalog.raw.schema.json`
  - `docs/repo_improvements_catalog.normalized.schema.json`
  - `docs/repo_improvements_touchpoint_map.json`
  - `docs/repo_improvements_catalog_v3.normalized.json`
  - `docs/repo_improvements_execution_plan_v1.json`
  - `docs/repo_improvements_execution_plan_v1.md`
  - `docs/repo_improvements_status.json`
  - `docs/repo_improvements_status.md`
  - `docs/repo_improvements_runbook.md`
  - `scripts/normalize_repo_improvements_catalog.py`
  - `scripts/map_repo_improvements_to_capabilities.py`
  - `scripts/plan_repo_improvements_rollout.py`
  - `scripts/validate_repo_improvement_dependencies.py`
  - `scripts/scaffold_repo_improvement_plugins.py`
  - `scripts/run_repo_improvements_pipeline.py`
  - `tests/test_repo_improvements_catalog.py`
  - `tests/test_repo_improvement_dependencies.py`
  - `tests/fixtures/ground_truth_repo_improvements_wave1.yaml`
- Additional blueprint implementation targets not yet present:
  - `plugins/changepoint_detection_v1/*`
  - `plugins/process_mining_discovery_v1/*`
  - `plugins/process_mining_conformance_v1/*`
  - `plugins/causal_recommendations_v1/*`

## Prerequisites
- Python env and local dependencies available (`.venv` expected).
- Existing matrix generators remain source-of-truth:
  - `scripts/docs_coverage_matrix.py`
  - `scripts/binding_implementation_matrix.py`
  - `scripts/full_instruction_coverage_report.py`
  - `scripts/full_repo_misses.py`
- Existing plugin contracts remain authoritative:
  - `src/statistic_harness/core/plugin_manager.py`
  - `src/statistic_harness/core/pipeline.py`
  - `src/statistic_harness/core/stat_plugins/registry.py`

## Sprint 1: Eliminate Hard-Gap Artifact Misses
**Goal**: Reduce `full_repo_misses` hard gaps to zero for missing paths.
**Demo/Validation**:
- `docs/full_repo_misses.json` reports no missing file paths in `instruction_coverage_items`.
- `scripts/full_instruction_coverage_report.py --verify` and `scripts/full_repo_misses.py --verify` pass.

### Task 1.1: Add Blueprint Contract Artifacts
- **Location**: `docs/_codex_repo_manifest.txt`, `docs/_codex_plugin_catalog.md`, `docs/schemas/changepoints.schema.json`, `docs/schemas/process_mining.schema.json`, `docs/schemas/causal.schema.json`
- **Description**: Create the missing blueprint contract docs/schemas in minimal valid form with deterministic formatting.
- **Complexity**: 4/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - Manifest/catalog/schema files exist and are valid text/JSON artifacts.
  - Schema files define required top-level structure and version marker.
- **Validation**:
  - JSON lint + targeted tests reading schema files.

### Task 1.2: Add Plugin Catalog Generator and Output
- **Location**: `scripts/generate_codex_repo_manifest.py` (new), `scripts/generate_codex_plugin_catalog.py` (new), `docs/_codex_repo_manifest.txt`, `docs/_codex_plugin_catalog.md`
- **Description**: Generate repository manifest (`git ls-files` equivalent, deterministic sort) and plugin catalog from existing `plugins/*/plugin.yaml` manifests (do not require future plugin directories).
- **Complexity**: 5/10
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Catalog is generated, not hand-maintained.
  - `--verify` mode fails on stale output.
- **Validation**:
  - New test: `tests/test_generate_codex_plugin_catalog.py`.

### Task 1.3: Prevent Coverage False Positives (`report_keys`)
- **Location**: `docs/docs_non_plugin_tokens.json`
- **Description**: Add coverage-safe non-plugin token exceptions (including `report_keys`) for instruction coverage parsing.
- **Complexity**: 2/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - No unresolved token false positives for non-plugin traceability strings.
- **Validation**:
  - `scripts/full_instruction_coverage_report.py` output contains no unresolved `report_*` pseudo-tokens.

## Sprint 2: Repo-Improvement Catalog Foundations
**Goal**: Deliver schema-validated normalized catalog artifacts from canonical input.
**Demo/Validation**:
- `docs/repo_improvements_catalog_v3.normalized.json` exists and validates against normalized schema.
- Deterministic regeneration produces zero diff.

### Task 2.0: Canonicalization as Formal Pipeline Input
- **Location**: `scripts/canonicalize_repo_improvements_catalog.py`, `docs/repo_improvements_catalog_v3.canonical.json`, `docs/repo_improvements_catalog_v3.reduction_report.json`
- **Description**: Promote canonicalization outputs from ad hoc artifact to explicit prerequisite stage for normalization.
- **Complexity**: 3/10
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Canonical and reduction artifacts are generated by script and verified via `--verify`.
  - Normalizer consumes canonical output, not raw templated catalog.
- **Validation**:
  - `tests/test_repo_improvements_catalog_canonicalization.py` + script verify mode.

### Task 2.1: Add Raw + Normalized Catalog Schemas
- **Location**: `docs/repo_improvements_catalog.raw.schema.json`, `docs/repo_improvements_catalog.normalized.schema.json`
- **Description**: Define strict required fields for raw/canonical/normalized forms and validation constraints.
- **Complexity**: 5/10
- **Dependencies**: Task 2.0
- **Acceptance Criteria**:
  - Valid sample passes; malformed sample fails.
- **Validation**:
  - New tests in `tests/test_repo_improvements_catalog.py`.

### Task 2.2: Add Touchpoint Allowlist Map
- **Location**: `docs/repo_improvements_touchpoint_map.json`
- **Description**: Map invalid/legacy touchpoints (for example `src/statistic_harness/core/plugin_registry.py`) to actual repo integration points (`src/statistic_harness/core/stat_plugins/registry.py`, plugin wrappers, tests paths).
- **Complexity**: 4/10
- **Dependencies**: Task 2.0, Task 2.1
- **Acceptance Criteria**:
  - Every raw touchpoint maps deterministically to a known path category.
- **Validation**:
  - Mapping integrity assertions in `tests/test_repo_improvements_catalog.py`.

### Task 2.3: Implement Normalizer Script
- **Location**: `scripts/normalize_repo_improvements_catalog.py`, `docs/repo_improvements_catalog_v3.normalized.json`
- **Description**: Transform canonical catalog into normalized execution-ready catalog with resolved touchpoints, stable IDs, and computed scoring inputs.
- **Complexity**: 7/10
- **Dependencies**: Task 2.0, Task 2.1, Task 2.2
- **Acceptance Criteria**:
  - `--verify` mode supported.
  - Output sorting is deterministic.
  - Fails closed on unmapped touchpoints/unknown category.
- **Validation**:
  - Determinism + schema validation tests in `tests/test_repo_improvements_catalog.py`.

## Sprint 3: Capability Mapping, Rollout Plan, Dependency Integrity
**Goal**: Produce actionable, dependency-safe execution plan artifacts.
**Demo/Validation**:
- Execution plan JSON/Markdown generated from normalized catalog.
- Dependency validator returns zero cycles or explicitly deferred unresolved nodes.

### Task 3.1: Implement Capability Mapper
- **Location**: `scripts/map_repo_improvements_to_capabilities.py`
- **Description**: Classify each normalized item into existing-plugin enhancement, new-plugin scaffold, or deferred.
- **Complexity**: 7/10
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - 100% normalized items classified with rationale.
  - Output includes target code paths.
- **Validation**:
  - Mapping unit tests in `tests/test_repo_improvements_catalog.py`.

### Task 3.2: Implement Rollout Planner
- **Location**: `scripts/plan_repo_improvements_rollout.py`, `docs/repo_improvements_execution_plan_v1.json`, `docs/repo_improvements_execution_plan_v1.md`
- **Description**: Score and rank items into wave buckets using deterministic tie-breakers and dependency depth.
- **Complexity**: 6/10
- **Dependencies**: Task 2.3, Task 3.1
- **Acceptance Criteria**:
  - Deterministic plan generation and stable wave assignments.
  - Human-readable markdown includes rationale and validation commands.
  - Planner fails closed if normalized catalog is missing or stale.
- **Validation**:
  - Snapshot-style tests for generated JSON/MD.

### Task 3.3: Implement Dependency Validator
- **Location**: `scripts/validate_repo_improvement_dependencies.py`, `tests/test_repo_improvement_dependencies.py`
- **Description**: Validate improvement graph (node existence, cycle detection, missing predecessor handling).
- **Complexity**: 6/10
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Detects cycles and unknown dependencies with precise diagnostics.
  - Supports `--verify` for CI-style use.
- **Validation**:
  - Unit fixtures for acyclic, cyclic, and unknown-node cases.

## Sprint 4: Governance, Status Ledger, and One-Command Pipeline
**Goal**: Make the repo-improvements stream operational and repeatable.
**Demo/Validation**:
- One command updates and verifies all repo-improvement artifacts.
- Status ledger tracks lifecycle and traceability for each canonical item.

### Task 4.1: Build Repo-Improvement Pipeline Runner
- **Location**: `scripts/run_repo_improvements_pipeline.py`
- **Description**: Orchestrate canonicalize -> normalize -> map -> plan -> validate -> status update, composed with existing matrix refresh/verify scripts.
- **Complexity**: 6/10
- **Dependencies**: Sprint 2, Sprint 3
- **Acceptance Criteria**:
  - Non-zero on any failed phase.
  - Deterministic output order and logs.
- **Validation**:
  - End-to-end smoke test invoking script twice and asserting no diff.

### Task 4.2: Add Status Ledger Artifacts
- **Location**: `docs/repo_improvements_status.json`, `docs/repo_improvements_status.md`
- **Description**: Track per-item status (`todo`, `in_progress`, `done`, `deferred`), owner, evidence links, report keys, and test links.
- **Complexity**: 4/10
- **Dependencies**: Task 3.2
- **Acceptance Criteria**:
  - Every canonical item has exactly one status record.
  - Markdown view derives from JSON source (no manual drift).
- **Validation**:
  - Ledger consistency test in `tests/test_repo_improvements_catalog.py`.

### Task 4.3: Add Operator Runbook
- **Location**: `docs/repo_improvements_runbook.md`
- **Description**: Document lifecycle, commands, expected outputs, failure triage, and rollback steps.
- **Complexity**: 3/10
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - New engineer can execute pipeline and interpret outputs.
- **Validation**:
  - Runbook command checklist cross-checked in CI docs lint/test.

## Sprint 5: Wave-1 Scaffolding + Evaluator Hooks
**Goal**: Ship the first implementable tranche, proving end-to-end execution.
**Demo/Validation**:
- Wave-1 selected plugin scaffolds are discoverable and testable.
- Evaluator fixture for wave-1 exists and is exercised.

### Task 5.1: Add Scaffold Generator for Selected Missing Plugin IDs
- **Location**: `scripts/scaffold_repo_improvement_plugins.py`, `plugins/analysis_plugin_*_v1/`
- **Description**: Scaffold only wave-1 IDs from execution plan; use existing plugin manifest conventions and no-network sandbox defaults.
- **Complexity**: 7/10
- **Dependencies**: Task 3.2
- **Acceptance Criteria**:
  - Generated plugins pass manifest schema and discovery.
  - Scaffolder supports `--ids` and `--verify`.
- **Validation**:
  - `tests/test_plugin_manifest_schema.py`
  - `tests/test_plugin_discovery.py`

### Task 5.2: Add Wave-1 Evaluator Fixture + Tests
- **Location**: `tests/fixtures/ground_truth_repo_improvements_wave1.yaml`, `tests/test_evaluator.py` (extend)
- **Description**: Add deterministic evaluation expectations for wave-1 outputs.
- **Complexity**: 5/10
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - Evaluator asserts declared wave-1 findings/metrics with tolerances.
- **Validation**:
  - `python -m pytest -q tests/test_evaluator.py`

## Sprint 6: Complete Blueprint Feature Plugins
**Goal**: Implement remaining blueprint feature plugins that are still absent.
**Demo/Validation**:
- Plugins appear in `list-plugins`, run in pipeline, emit artifacts, and evaluate successfully.

### Task 6.1: Implement `changepoint_detection_v1`
- **Location**: `plugins/changepoint_detection_v1/`, `src/statistic_harness/core/stat_plugins/registry.py`
- **Description**: Add deterministic changepoint plugin with seeded tests and bounded processing.
- **Complexity**: 8/10
- **Dependencies**: Sprint 1 schema files
- **Acceptance Criteria**:
  - Emits changepoint findings compatible with evaluator.
- **Validation**:
  - New plugin test module under `tests/plugins/`.

### Task 6.2: Implement Process Mining Plugins
- **Location**: `plugins/process_mining_discovery_v1/`, `plugins/process_mining_conformance_v1/`, shared helpers under `src/statistic_harness/core/`
- **Description**: Build event-log abstraction and two plugin paths (discovery + conformance) with deterministic outputs.
- **Complexity**: 8/10
- **Dependencies**: Task 6.1
- **Acceptance Criteria**:
  - Artifacts validated against process mining schema.
- **Validation**:
  - Dedicated tests in `tests/plugins/` with synthetic event logs.

### Task 6.3: Implement `causal_recommendations_v1`
- **Location**: `plugins/causal_recommendations_v1/`, optional helper module under `src/statistic_harness/core/`
- **Description**: Add assumption-explicit causal recommendation plugin with refutation reporting.
- **Complexity**: 9/10
- **Dependencies**: Sprint 1 causal schema
- **Acceptance Criteria**:
  - Emits effects or explicit non-identification with assumptions.
- **Validation**:
  - Plugin tests and evaluator expectations.

## Sprint 7: Repository Hardening and Release Gate
**Goal**: Verify completion and prevent regressions.
**Demo/Validation**:
- Hard-gaps removed, pipeline and matrices up to date, full tests green.

### Task 7.1: Integrate New Scripts into Existing Update/Verify Flows
- **Location**: `scripts/update_docs_and_plugin_matrices.py`, `scripts/verify_docs_and_plugin_matrices.py`, `scripts/refresh_all_matrices.sh`
- **Description**: Include repo-improvement scripts and new generated artifacts in update/verify chain.
- **Complexity**: 5/10
- **Dependencies**: Sprints 2-5
- **Acceptance Criteria**:
  - One update command and one verify command cover all new artifacts.
- **Validation**:
  - Verify scripts run clean on fresh clone and no-op rerun.

### Task 7.2: Final Completion Assertions
- **Location**: `tests/test_repo_improvements_catalog.py` (expand), `docs/full_instruction_coverage_report.json`, `docs/full_repo_misses.json`
- **Description**: Add assertions that no missing-path hard gaps remain and that catalog pipeline artifacts are current.
- **Complexity**: 4/10
- **Dependencies**: Task 7.1
- **Acceptance Criteria**:
  - `docs/full_repo_misses.json` reports `has_hard_gaps: false`.
  - `python -m pytest -q` passes.
- **Validation**:
  - full-suite run + regenerate/verify scripts.

## Testing Strategy
- Per-sprint targeted tests for each new script/artifact.
- Determinism checks for generated JSON/MD (`--verify` or snapshot tests).
- Plugin discovery/schema tests for all scaffolded/new plugins.
- Evaluator checks with fixed seeds and ground-truth fixtures.
- Full release gate:
  - `python -m pytest -q`
  - matrix verify scripts
  - repo-improvements pipeline verify pass.

## Potential Risks & Gotchas
- Recursive coverage noise from generated reports (`full_repo_misses` referencing itself).
  - Mitigation: keep generation order strict; exclude generated-only docs where appropriate; classify non-plugin tokens.
- Planning artifacts can create false hard gaps before implementation files exist.
  - Mitigation: complete foundational artifacts in Sprint 1-3 before refreshing coverage reports.
- Scaffolding too many placeholder plugins can inflate maintenance burden.
  - Mitigation: scaffold only wave-1 plugin IDs selected by rollout plan.
- Touchpoint mapping drift if core integration files change.
  - Mitigation: maintain single source map (`docs/repo_improvements_touchpoint_map.json`) and validate in tests.

## Rollback Plan
- Revert repo-improvement pipeline additions in reverse dependency order:
  1. Remove new verify/update hooks.
  2. Revert rollout/dependency scripts.
  3. Revert normalization schemas/scripts.
  4. Revert blueprint feature plugins.
- Keep canonicalization artifacts (`docs/repo_improvements_catalog_v3.canonical.json`, `docs/repo_improvements_catalog_v3.reduction_report.json`) as baseline reference even if later phases roll back.
