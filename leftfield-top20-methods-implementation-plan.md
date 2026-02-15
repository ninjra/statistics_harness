# Plan: Leftfield Top20 Methods Plugin Implementation and Baseline Comparison

**Generated**: 2026-02-14
**Estimated Complexity**: High

## Overview
Implement the 20 additional analysis plugins listed in `docs/statistics_harness_top20_leftfield_methods.md` as first-class, runnable harness plugins (no stubs), wire them into the existing plugin execution path, add deterministic tests, run the full gauntlet, then compare post-change baseline results against the pre-change baseline run using the existing run/dataset comparison tooling.

## Skills Selected (and Why)
- `plan-harder`: required by user request; drives phased planning, risks, and review workflow.
- `python-testing-patterns`: add deterministic unit tests and handler-level assertions for each plugin family.
- `testing`: execute and validate full test gauntlet (`.venv/bin/python -m pytest -q`) before shipping.
- `discover-observability`: keep metric outputs/diagnostic artifacts consistent and measurable for before/after evaluation.
- `shell-lint-ps-wsl`: ensure command hygiene for all shell execution and reproducibility.

## Prerequisites
- Existing plugin wrapper pattern via `run_top20_plugin` in `src/statistic_harness/core/top20_plugins.py`.
- Existing schema/manifest conventions under `plugins/analysis_*`.
- Local baseline runs available for dataset `3246cc7cd7d57a317ddc05e80e6f6f5bfe7f50deb0ee7af8db50d04bae180e1a`.
- Existing comparator script: `scripts/compare_run_outputs.py`.
- Dependency policy: prefer existing repo dependencies (`numpy/pandas/sklearn/networkx/igraph` etc.) and avoid adding network/model-download requirements for heavy methods.

## Sprint 1: Scaffold and Wire New Plugins
**Goal**: Register all 20 plugin IDs as runnable first-class plugins.
**Demo/Validation**:
- `plugins/<plugin_id>/plugin.yaml` exists for each of the 20 IDs.
- Each wrapper calls `run_top20_plugin(<plugin_id>, ctx)`.
- Plugin manager discovery includes all 20.

### Task 1.1: Build Leftfield Plugin Spec Scaffold
- **Location**: `scripts/scaffold_leftfield_top20_plugins.py`
- **Description**: Add a scaffold generator analogous to existing top20 scripts with defaults/depends_on/capabilities from doc.
- **Complexity**: 4
- **Dependencies**: none
- **Acceptance Criteria**:
  - Script can generate manifests/wrappers/config/output schemas for all 20 IDs.
  - Defaults are explicit and deterministic.
- **Validation**:
  - Run scaffold script and verify created plugin directories.

### Task 1.2: Generate Plugin Wrapper Directories
- **Location**: `plugins/analysis_*_v1/`
- **Description**: Materialize plugin wrappers/manifests/schemas for all 20 leftfield IDs.
- **Complexity**: 3
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - All 20 plugin folders created with required files.
- **Validation**:
  - `rg --files plugins | rg 'analysis_.*_v1/plugin.yaml'` includes all target IDs.

### Task 1.3: Lock Scaffold Outputs with Verification
- **Location**: `scripts/scaffold_leftfield_top20_plugins.py`, generated `plugins/*`, docs/matrix tests
- **Description**: Add or reuse verification to ensure generated wrappers remain in-sync (no manual drift) before handler implementation starts.
- **Complexity**: 3
- **Dependencies**: Tasks 1.1 and 1.2
- **Acceptance Criteria**:
  - Regeneration is deterministic and CI-visible through existing freshness tests.
- **Validation**:
  - Run targeted matrix/manifest tests after generation.

## Sprint 2: Implement Leftfield Handler Algorithms
**Goal**: Implement concrete algorithms and artifacts for all 20 plugin IDs.
**Demo/Validation**:
- `HANDLERS` map contains all 20 plugin IDs.
- Each plugin returns `ok` or `degraded` with findings/metrics/artifacts (never missing handler).

### Task 2.1: Add Shared Utility Primitives
- **Location**: `src/statistic_harness/core/top20_plugins.py`
- **Description**: Add reusable helpers for numeric matrix extraction, time-window segmentation, stable covariance/pairwise operations, deterministic ranking, and fallback-safe calculations.
- **Complexity**: 6
- **Dependencies**: Sprint 1 complete
- **Acceptance Criteria**:
  - Helper functions are deterministic and bounded by config caps.
- **Validation**:
  - New helper-focused tests pass.

### Task 2.2: Implement Rank 1-10 Leftfield Methods
- **Location**: `src/statistic_harness/core/top20_plugins.py`
- **Description**: Add concrete handlers for:
  - SSA decomposition changepoint
  - CUR decomposition explain
  - HSIC screen
  - ICP-style invariance screen
  - LiNGAM-style ICA causal proxy
  - Frequent Directions covariance sketch
  - DMD / Koopman modes
  - Diffusion maps manifold
  - Sinkhorn OT drift
  - kNN graph two-sample FR-style test
- **Complexity**: 9
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Each emits method-specific metrics + artifact JSON.
  - No unhandled exceptions on baseline dataset.
- **Validation**:
  - Unit/smoke tests for each handler.

### Task 2.3: Implement Rank 11-20 Leftfield Methods
- **Location**: `src/statistic_harness/core/top20_plugins.py`
- **Description**: Add concrete handlers for:
  - KSD anomaly
  - PC-style causal skeleton
  - GES-style score-based causal
  - PHATE-like trajectory embedding approximation
  - Node2Vec-like role embedding approximation
  - Tensor CP/PARAFAC decomposition
  - Symbolic-regression GP approximation
  - Normalizing-flow-like density approximation
  - TabPFN-like few-shot baseline approximation
  - NAM-like additive nonlinear effect approximation
- **Complexity**: 10
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - All 20 handlers registered and runnable without external network/model downloads.
  - Heavy-method placeholders are avoided; each has concrete deterministic computation.
- **Validation**:
  - Unit/smoke tests confirm metrics/findings structure and non-error status.

### Task 2.4: Register Handler Map and Execution Routing
- **Location**: `src/statistic_harness/core/top20_plugins.py`
- **Description**: Extend `HANDLERS` map and ensure `run_top20_plugin` resolves all new IDs.
- **Complexity**: 3
- **Dependencies**: Tasks 2.2 and 2.3
- **Acceptance Criteria**:
  - No “missing handler” for any leftfield plugin.
- **Validation**:
  - Targeted plugin invocation tests.

## Sprint 3: Tests, Matrices, and Observability
**Goal**: Ensure reliability, determinism, and measurable outputs.
**Demo/Validation**:
- Dedicated tests pass for new plugins.
- Plugin docs/matrices are regenerated and up to date.

### Task 3.1: Add Plugin Smoke + Determinism Tests
- **Location**: `tests/plugins/`, `tests/test_*`
- **Description**: Add tests covering execution, schema shape, deterministic outputs for repeated seeds.
- **Complexity**: 7
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - New tests fail on missing handlers and pass with implementation.
- **Validation**:
  - `.venv/bin/python -m pytest -q <new test modules>`.

### Task 3.2: Update Matrix/Manifest Docs
- **Location**: `docs/_codex_repo_manifest.txt`, `docs/redteam_ids_matrix.json`, `docs/redteam_ids_matrix.md`, relevant plugin matrices
- **Description**: Regenerate artifact-tracked docs impacted by new plugin files.
- **Complexity**: 3
- **Dependencies**: Sprint 1/2
- **Acceptance Criteria**:
  - Artifact freshness tests pass.
- **Validation**:
  - Targeted test modules for matrix/manifest.

### Task 3.3: Metric Contract for Comparison
- **Location**: `scripts/compare_run_outputs.py` usage + run artifacts
- **Description**: Ensure comparison captures status deltas, material payload deltas, and top changed plugins.
- **Complexity**: 2
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - Comparable JSON outputs exist for before/after.
- **Validation**:
  - Run comparator in `run_to_run` and `dataset_to_dataset` modes.

### Task 3.4: Report Output and Schema Contract Validation
- **Location**: run artifacts under `appdata/runs/<run_id>/`, `docs/report.schema.json`
- **Description**: Explicitly validate that full-run outputs include both `report.md` and schema-valid `report.json` after adding leftfield plugins.
- **Complexity**: 2
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - `report.md` exists and non-empty.
  - `report.json` validates against `docs/report.schema.json`.
- **Validation**:
  - Run existing integration/schema tests and targeted assertions on produced run.

## Sprint 4: Full Run and Baseline Comparison
**Goal**: Execute new plugin set on baseline dataset and compare against known pre-change baseline run.
**Demo/Validation**:
- A new completed run exists for baseline dataset using full plugin set.
- Comparison JSON and plain-English summary include meaningful-improvement indicators.

### Task 4.1: Execute Full Baseline Run (No Subset)
- **Location**: `scripts/run_loaded_dataset_full.py` (or direct pipeline run)
- **Description**: Run full harness on baseline dataset with deterministic seed, capturing run id and artifacts.
- **Complexity**: 4
- **Dependencies**: Sprint 3
- **Acceptance Criteria**:
  - Run status is completed/partial with explicit plugin outcomes recorded.
- **Validation**:
  - Verify DB rows and run artifacts exist.

### Task 4.2: Compare Against Pre-change Baseline Run
- **Location**: `/tmp/*diff*.json` generated by comparator
- **Description**: Compare new run vs known pre-change run (`db3c8c76cf70401cb1592471a056e071`) and summarize material improvements.
- **Complexity**: 3
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Delta report includes status changes, material payload changes, and recommendation-impact summary.
- **Validation**:
  - Comparator JSON present and parsed in summary output.

### Task 4.3: Baseline Comparison Preflight Integrity Check
- **Location**: `appdata/state.sqlite`, baseline run dirs
- **Description**: Verify pre-change baseline run (`db3c8c76cf70401cb1592471a056e071`) and comparison reference run artifacts still exist and are readable before computing deltas.
- **Complexity**: 2
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Baseline run metadata and `report.json` are present.
  - Comparison run IDs are explicitly captured in final output.
- **Validation**:
  - Preflight check command/script exits cleanly and lists resolved run IDs.

## Testing Strategy
- Unit and smoke tests for new handlers, deterministic seeds, and plugin registration.
- Targeted artifact freshness tests for generated docs.
- Report schema + output existence validation for new full run outputs.
- Full repo gate: `.venv/bin/python -m pytest -q`.
- Runtime verification by executing a full baseline run and comparing to baseline pre-change run.

## Potential Risks & Gotchas
- Heavy-method IDs (flow/TabPFN/NAM) may imply unavailable deps; mitigation: implement deterministic approximations using existing local dependencies and clearly expose approximation metadata in metrics/debug.
- Performance risk on large baseline dataset; mitigation: enforce row caps/sample caps from settings with deterministic sample seeds.
- False “improvement” due summary/debug churn; mitigation: use material component comparison (`metrics/findings/artifacts/error`) for claims.
- Plugin skip risk; mitigation: return `ok`/`degraded` with explicit gating reasons and measurable outputs instead of skip where feasible.

## Rollback Plan
- Revert newly added plugin directories and handler map entries.
- Remove scaffold script and tests tied to leftfield IDs.
- Regenerate docs matrices/manifests to prior state.
- Re-run full test suite to verify repo stability after rollback.
