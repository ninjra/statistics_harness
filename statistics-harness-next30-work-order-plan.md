# Plan: Statistics Harness Next30 Work Order

**Generated**: 2026-02-13  
**Estimated Complexity**: High

## Overview
Implement the full Next30 analysis plugin pack from `docs/statistics_harness_next30_work_order.md` using the stat-plugin architecture, with deterministic behavior, offline-only execution, schema-valid outputs, and full test coverage. The implementation path prioritizes one shared handler module (`next30_addon.py`), thin plugin wrappers, strong gating/skip behavior, and repeatable run-level validation.

## Prerequisites
- Existing stat-plugin execution path in:
  - `src/statistic_harness/core/stat_plugins/registry.py`
  - `src/statistic_harness/core/types.py`
  - `src/statistic_harness/core/plugin_runner.py`
- Existing helper patterns in:
  - `src/statistic_harness/core/stat_plugins/erp_next_wave.py`
  - `src/statistic_harness/core/stat_plugins/ideaspace.py`
- Plugin manifest/output schema contracts:
  - `docs/plugin_manifest.schema.json`
  - `docs/report.schema.json`
- Test harness and pipeline entry points:
  - `tests/conftest.py`
  - `src/statistic_harness/core/pipeline.py`
  - `scripts/run_loaded_dataset_full.py`

## Sprint 1: Foundation and Scaffolding
**Goal**: Add skeletons for all 30 plugins and wire the new addon module without algorithm implementation risk.
**Demo/Validation**:
- All 30 plugin directories are discoverable with valid manifests and schemas.
- Registry imports and recognizes `NEXT30_HANDLERS`.
- Smoke execution returns `skipped` or `ok`, never import/discovery failure.

### Task 1.1: Create addon module scaffold
- **Location**: `src/statistic_harness/core/stat_plugins/next30_addon.py`
- **Description**: Add shared helper primitives (`_artifact_json`, `_basic_metrics`, `_make_finding`, `_skip`, logging helpers, deterministic tie-break helpers, optional SciPy/sklearn guards) and placeholder handlers for 30 IDs.
- **Complexity**: 5/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - Module imports cleanly in a minimal environment.
  - `HANDLERS` dictionary contains all 30 exact IDs.
- **Validation**:
  - Targeted import test via `pytest` or direct module import.

### Task 1.2: Wire registry integration
- **Location**: `src/statistic_harness/core/stat_plugins/registry.py`
- **Description**: Import and merge `NEXT30_HANDLERS` after existing handler maps; preserve deterministic order and no handler key collisions.
- **Complexity**: 3/10
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - `run_plugin(<next30_id>, ctx)` resolves to the new handler.
- **Validation**:
  - Unit test or quick handler lookup assertion in test code.

### Task 1.3: Scaffold 30 plugin wrappers and schemas
- **Location**: `plugins/analysis_*_v1/` for all 30 IDs from the work order
- **Description**: Create `plugin.py`, `plugin.yaml`, `config.schema.json`, `output.schema.json` for each ID using repo-compatible templates.
- **Complexity**: 6/10
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - `PluginManager.discover()` reports no errors.
  - All manifests satisfy schema and sandbox constraints.
- **Validation**:
  - Discovery/manifest validation tests.

### Task 1.4: Fix and lock plugin YAML hygiene
- **Location**: `plugins/*/plugin.yaml`
- **Description**: Add a guard script/test to detect malformed YAML list indentation and invalid `depends_on` structures to prevent preflight failures.
- **Complexity**: 4/10
- **Dependencies**: Task 1.3
- **Acceptance Criteria**:
  - YAML parse and manifest checks fail fast in tests.
- **Validation**:
  - New test and/or CI pre-check script execution.

### Task 1.5: Enforce wrapper template and no-network compliance
- **Location**: `plugins/analysis_*_v1/plugin.yaml`, `plugins/analysis_*_v1/plugin.py`, validation script under `scripts/`
- **Description**: Add a structural compliance check for all 30 wrappers against work-order template expectations, including `depends_on`, settings defaults, schema paths, and `sandbox.no_network: true`.
- **Complexity**: 5/10
- **Dependencies**: Task 1.3
- **Acceptance Criteria**:
  - Validation fails if any Next30 wrapper drifts from required template structure.
  - All Next30 plugin manifests explicitly enforce offline/no-network execution.
- **Validation**:
  - Run compliance script/test before algorithm implementation sprints.

## Sprint 2: Implement A/B/C Families (1-15)
**Goal**: Deliver robust, deterministic implementations for time-series, changepoint, and robust/distribution diagnostics families.
**Demo/Validation**:
- IDs 1-15 produce meaningful `ok` findings on synthetic fixtures where applicable.
- Inapplicable inputs yield explicit `skipped` with gating reasons.

### Task 2.1: Implement A-family handlers (1-5)
- **Location**: `src/statistic_harness/core/stat_plugins/next30_addon.py`
- **Description**: Implement algorithms A1-A5 as specified (counterfactual, STL-like decomposition, Holt-Winters residuals, Lomb-Scargle fallback correlation grid, volatility shift ratio).
- **Complexity**: 8/10
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Artifacts written with required filenames.
  - Severity thresholds and effect metrics match spec defaults.
- **Validation**:
  - Unit tests per handler on synthetic time-series fixtures.

### Task 2.2: Implement B-family handlers (6-10)
- **Location**: `src/statistic_harness/core/stat_plugins/next30_addon.py`
- **Description**: Implement robust changepoint score variants and consensus merge logic, including deterministic interval sampling for WBS.
- **Complexity**: 8/10
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Deterministic changepoint indices for fixed seed and data.
  - Consensus output includes votes/tolerance merge metadata.
- **Validation**:
  - Repeat-run determinism tests for changepoint handlers.

### Task 2.3: Implement C-family handlers (11-15)
- **Location**: `src/statistic_harness/core/stat_plugins/next30_addon.py`
- **Description**: Implement Benford, geometric median, Marchenko-Pastur denoising, Cook’s distance, and Hill tail index with gating/caps.
- **Complexity**: 8/10
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - Quadratic workloads obey cap-and-skip policy.
  - Findings include stable IDs, severity, confidence, and evidence.
- **Validation**:
  - Unit tests with targeted datasets and expected metric ranges.

## Sprint 3: Implement D/E/F Families (16-30)
**Goal**: Complete nonlinear modeling, latent factor separation, and count/categorical complexity handlers.
**Demo/Validation**:
- IDs 16-30 all resolve in registry and return `ok|skipped` with valid artifacts.
- Optional dependency paths degrade gracefully.

### Task 3.1: Implement D-family handlers (16-20)
- **Location**: `src/statistic_harness/core/stat_plugins/next30_addon.py`
- **Description**: Add distance correlation, GAM spline, quantile boosting, quantile-regression-forest approximation, sparse PCA.
- **Complexity**: 9/10
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - sklearn-dependent paths guarded with fallback/skip messages.
  - Feature ranking outputs are deterministic and stable-sorted.
- **Validation**:
  - Unit tests with controlled numeric matrices and fixed seeds.

### Task 3.2: Implement E-family handlers (21-25)
- **Location**: `src/statistic_harness/core/stat_plugins/next30_addon.py`
- **Description**: Add ICA, CCA, varimax rotation, Oja subspace tracking, VIF screen.
- **Complexity**: 8/10
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Principal-angle and multicollinearity thresholds are applied per spec.
  - Artifacts carry interpretable component/loadings payloads.
- **Validation**:
  - Synthetic fixture tests for subspace shift and collinearity triggers.

### Task 3.3: Implement F-family handlers (26-30)
- **Location**: `src/statistic_harness/core/stat_plugins/next30_addon.py`
- **Description**: Add ZIP-EM, NB overdispersion, Dirichlet-multinomial dispersion proxy, Fisher enrichment, RQA metrics.
- **Complexity**: 9/10
- **Dependencies**: Task 3.2
- **Acceptance Criteria**:
  - Fisher exact uses deterministic log-space computation and BH correction.
  - Recurrence metrics respect quadratic caps and do not silently overrun.
- **Validation**:
  - Unit tests with integer-like, categorical, and embedded series fixtures.

## Sprint 4: Cross-Cutting Contracts and Test Harness
**Goal**: Enforce logging, determinism, budgets, output schema, and smoke/determinism test requirements across all Next30 plugins.
**Demo/Validation**:
- Every Next30 plugin logs START/SKIP/END into `run_dir/logs/<plugin_id>.log`.
- Smoke suite and determinism suite pass for all 30 plugins.

### Task 4.1: Add shared contract assertions in handler helpers
- **Location**: `src/statistic_harness/core/stat_plugins/next30_addon.py`
- **Description**: Centralize logging and finding construction so each handler emits required fields and uses consistent measurement metadata.
- **Complexity**: 6/10
- **Dependencies**: Sprint 3
- **Acceptance Criteria**:
  - No handler bypasses mandatory logging and finding envelope.
- **Validation**:
  - Tests assert log file presence and non-empty content.

### Task 4.1a: Enforce `BudgetTimer` and quadratic cap behavior centrally
- **Location**: `src/statistic_harness/core/stat_plugins/next30_addon.py`
- **Description**: Add shared guard helpers to enforce `time_budget_ms`, `max_points_for_quadratic`, and required skip behavior when limits are exceeded (unless `allow_row_sampling=True`).
- **Complexity**: 6/10
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - All quadratic handlers call the shared cap guard.
  - Limit breaches return deterministic `skipped` payloads with gating reasons.
- **Validation**:
  - Focused unit tests forcing over-cap and timeout-like paths.

### Task 4.2: Implement smoke + determinism test file
- **Location**: `tests/test_next30_plugins_smoke.py`
- **Description**: Build fixture generator and tests that run each plugin twice with identical seeds and compare payload equality when status is `ok`.
- **Complexity**: 7/10
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - All 30 plugins execute via `Pipeline.run(...)` with `status in {"ok","skipped"}`.
  - Determinism assertions hold for `ok` runs.
  - Log assertions verify mandated `START`, `SKIP` (when applicable), and `END` entries.
- **Validation**:
  - `python -m pytest -q tests/test_next30_plugins_smoke.py`

### Task 4.3: Add focused unit tests for algorithmic edge cases
- **Location**: `tests/plugins/test_next30_*` (split by family for maintainability)
- **Description**: Add atomic tests for gating, tie-break ordering, cap behavior, optional dependency fallback, and artifact contract.
- **Complexity**: 8/10
- **Dependencies**: Tasks 4.1-4.2
- **Acceptance Criteria**:
  - Each family has at least one edge-case regression test.
  - Artifact contract is validated: deterministic JSON content, expected filename, and non-empty `PluginArtifact` metadata.
- **Validation**:
  - `python -m pytest -q` full suite gate.

### Task 4.4: Add action audit log for Codex execution state
- **Location**: `.codex/STATE.md`
- **Description**: Record changed files, rationale, commands, and any skipped local installs/tests per work-order requirement.
- **Complexity**: 3/10
- **Dependencies**: Sprint 4 implementation progress
- **Acceptance Criteria**:
  - State log updated with reproducible command history and outcomes.
- **Validation**:
  - Manual review of `.codex/STATE.md`.

## Sprint 5: Full Lifecycle Validation and Release Readiness
**Goal**: Validate integration at pipeline scale and confirm ship criteria without breaking existing plugin ecosystem.
**Demo/Validation**:
- Full `pytest -q` passes.
- Discovery + execution matrices remain healthy.
- Next30 plugins run under full gauntlet without network usage.

### Task 5.1: Discovery and registry verification pass
- **Location**: `scripts/plugins_functionality_matrix.py`, `scripts/plugin_data_access_matrix.py` (and related existing validation scripts)
- **Description**: Run/update matrix checks to ensure new IDs are present and discoverable.
- **Complexity**: 4/10
- **Dependencies**: Sprints 1-4
- **Acceptance Criteria**:
  - No plugin manifest/discovery errors.
- **Validation**:
  - Existing repo matrix scripts and pipeline plugin listing checks.

### Task 5.2: Integration run on synthetic fixture set
- **Location**: runtime artifacts under `appdata/runs/*`
- **Description**: Execute bounded and full plugin runs to verify stability/performance and ensure no `error` status on fixture datasets for Next30 IDs.
- **Complexity**: 6/10
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - Next30 plugins produce expected artifacts/logs under pipeline orchestration.
- **Validation**:
  - `scripts/run_loaded_dataset_full.py` plus DB inspection of `plugin_executions`.

### Task 5.3: Final release gate
- **Location**: repository root (`pytest.ini`, CI workflow context)
- **Description**: Execute and document final `python -m pytest -q` result; resolve any collection/environment issues without loosening test quality.
- **Complexity**: 5/10
- **Dependencies**: Task 5.2
- **Acceptance Criteria**:
  - Release gate is green with deterministic tests.
- **Validation**:
  - Successful full test run output captured in `.codex/STATE.md`.

## Testing Strategy
- Unit tests per algorithm family with deterministic synthetic fixtures.
- Pipeline smoke tests through `Pipeline.run(...)` for all 30 plugin IDs.
- Determinism tests with same seed/input and exact payload equality for `ok` outcomes.
- Manifest/discovery validation for plugin scaffolding.
- Full repository gate: `python -m pytest -q`.

## Potential Risks & Gotchas
- Risk: Manifest drift or malformed YAML can halt entire runs early.
  - Mitigation: add explicit YAML parse/manifest validation checks and keep wrappers templated.
- Risk: Quadratic algorithms can overwhelm runtime on large datasets.
  - Mitigation: enforce `max_points_for_quadratic` cap with explicit `skipped` behavior unless sampling is enabled.
- Risk: Optional dependency behavior (`scipy`, `sklearn`) can diverge across environments.
  - Mitigation: explicit guarded imports and deterministic fallback implementations with tests.
- Risk: Duplicate test module basenames can break collection.
  - Mitigation: preserve/importlib collection mode and unique test module naming.
- Risk: Thin-wrapper dependency edits can accidentally introduce missing edges.
  - Mitigation: keep `depends_on` changes minimal and validate DAG before full runs.
- Risk: Existing plugin ecosystem regression from registry merge order.
  - Mitigation: append-only handler merge and collision assertion tests.

## Rollback Plan
- Revert `next30_addon` import/update in `registry.py`.
- Remove Next30 plugin directories in one commit.
- Remove Next30-specific tests and `.codex/STATE.md` entries for this work order.
- Re-run manifest/discovery and full pytest gate to confirm baseline restoration.

## Assumptions
- The "implement exactly" algorithm baselines in the work order are authoritative over convenience substitutions.
- No new third-party dependencies are required beyond currently declared project deps.
- Because interactive clarification tooling is unavailable in current mode, unresolved ambiguities are handled with explicit skip/gating behavior and documented defaults.
