# Plan: Statistics Harness Next30B Ranked Work Order

**Generated**: 2026-02-13  
**Estimated Complexity**: High

## Overview
Implement the Next30B plugin pack as first-class analysis plugins with no stubs, no placeholder handlers, and deterministic behavior under the existing harness contracts. Reuse existing stat-plugin architecture (`registry.run_plugin`) and shared helpers, enforce hard caps for expensive methods, add references mapping, and ship a comprehensive smoke/determinism suite plus full repo gauntlet.

## Scope
- Implement `src/statistic_harness/core/stat_plugins/next30b_addon.py` with complete handlers for all Batch-2 plugin IDs.
- Add or update plugin wrappers (`plugins/<id>/plugin.py`, `plugin.yaml`, schemas).
- Wire into `src/statistic_harness/core/stat_plugins/registry.py`.
- Extend `src/statistic_harness/core/stat_plugins/references.py`.
- Add `tests/test_next30b_plugins_smoke.py` with deterministic run checks.
- Regenerate and commit matrix artifacts required by repo guardrails.
- Run full test gauntlet and fix all regressions.

## Non-Negotiable Contracts
- No network at runtime for plugins.
- No plugin returns `error` on smoke datasets.
- Deterministic outputs for `status == "ok"` under identical seed/input.
- `pytest -q` passes before shipping.
- `.codex/STATE.md` updated with actions, commands, and outcomes.

## Implementation Notes (Critical)
- Duplicate ID exists: `analysis_negative_binomial_overdispersion_v1` is already implemented in Next30.  
  Plan handles this explicitly by reconciling behavior to one canonical handler path and one plugin directory.
- Logging must remain non-empty per plugin run (`START`, `SKIP` when applicable, `END`).
- Expensive handlers must hard-cap rows/resamples and respect `BudgetTimer`.

## Sprint 1: Baseline, Reconciliation, and Scaffolding
**Goal**: Establish exact implementation targets and generate all required plugin wrappers cleanly.  
**Demo/Validation**:
- `next30b_addon.py` exists with utility scaffolding and `HANDLERS` map skeleton.
- All plugin directories/files exist and are discoverable.

### Task 1.1: Reconcile plugin ID inventory
- **Location**: `docs/statistics_harness_next30b_ranked_work_order.md`, `plugins/`
- **Description**: Build final authoritative ID list and flag duplicates against existing plugins (especially NB overdispersion).
- **Complexity**: 3
- **Dependencies**: None
- **Acceptance Criteria**:
  - Canonical list of 30 IDs documented in plan/state.
  - Duplicate-ID strategy explicitly chosen and recorded.
- **Validation**:
  - Local script/check confirms exactly one wrapper directory per final ID.

### Task 1.2: Generate/normalize wrappers for Next30B IDs
- **Location**: `plugins/<id>/plugin.py`, `plugins/<id>/plugin.yaml`, `plugins/<id>/config.schema.json`, `plugins/<id>/output.schema.json`
- **Description**: Create missing wrappers and normalize manifests/schemas to repo conventions.
- **Complexity**: 5
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Every ID has complete wrapper files.
  - Manifests declare offline sandbox and deterministic defaults.
- **Validation**:
  - `tests/test_plugin_discovery.py`
  - `tests/test_plugin_manifest_schema.py`
  - `tests/test_cli_plugins_validate.py`

### Task 1.3: Wire addon into registry
- **Location**: `src/statistic_harness/core/stat_plugins/registry.py`
- **Description**: Import `NEXT30B_HANDLERS` and merge into `HANDLERS`.
- **Complexity**: 2
- **Dependencies**: Task 1.2
- **Acceptance Criteria**:
  - All Next30B plugin wrappers execute through registry path.
- **Validation**:
  - Direct plugin smoke invocation for 2-3 representative IDs.

## Sprint 2: Core Utilities and Shared Safety Guards
**Goal**: Build robust shared helper layer used by all 30 handlers.  
**Demo/Validation**:
- Utility functions implemented once and used consistently.
- Quadratic/resampling/time-budget limits enforced centrally.

### Task 2.1: Implement deterministic artifact/finding/metrics utilities
- **Location**: `src/statistic_harness/core/stat_plugins/next30b_addon.py`
- **Description**: Add `_artifact_json`, `_basic_metrics`, `_make_finding`, deterministic rounding/serialization helpers.
- **Complexity**: 4
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - All handlers can return consistent `PluginResult` contracts.
- **Validation**:
  - Unit smoke for helper behavior.

### Task 2.2: Implement capping, sampling, split, and logging primitives
- **Location**: `src/statistic_harness/core/stat_plugins/next30b_addon.py`
- **Description**: Add `_cap_quadratic`, `_cap_resamples`, `_split_pre_post`, `_log_start/_log_skip/_log_end`.
- **Complexity**: 5
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Expensive operations cannot exceed configured caps/time budget.
  - All handlers emit non-empty logs.
- **Validation**:
  - Targeted tests for caps and deterministic sampling.

### Task 2.3: Build reusable feature/column selectors
- **Location**: `src/statistic_harness/core/stat_plugins/next30b_addon.py`
- **Description**: Add stable selectors for time/count/duration/binary/categorical/numeric blocks and graph constructors.
- **Complexity**: 5
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Handlers avoid copy-paste selectors and converge on one selection policy.
- **Validation**:
  - Selector tests on synthetic mixed-schema datasets.

## Sprint 3: Implement Priority A/B Plugins (1-12)
**Goal**: Ship highest-value and general-purpose handlers with full artifacts/findings.  
**Demo/Validation**:
- IDs 1-12 return `ok` on smoke datasets and produce artifact JSON.

### Task 3.1: Implement A-tier methods (1-4)
- **Location**: `src/statistic_harness/core/stat_plugins/next30b_addon.py`
- **Description**: Implement beta-binomial overdispersion, circular drift, Mann-Kendall, quantile-mapping drift.
- **Complexity**: 7
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - Metrics/artifact schemas align with work order.
  - Deterministic findings for fixed seed.
- **Validation**:
  - New smoke tests for each handler.

### Task 3.2: Implement B-tier methods (5-12)
- **Location**: `src/statistic_harness/core/stat_plugins/next30b_addon.py`
- **Description**: Implement constraints detector, NB overdispersion reconciliation, partial-corr shift, piecewise trend, Poisson drivers, P2 quantiles, robust regression, state-space smoother.
- **Complexity**: 8
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - No placeholder implementations.
  - All methods guard edge cases and insufficient data gracefully.
- **Validation**:
  - Plugin-level deterministic tests and no-error smoke execution.

## Sprint 4: Implement C/D Plugins (13-23)
**Goal**: Complete medium/domain-sensitive methods with conservative assumptions and robust gating.  
**Demo/Validation**:
- IDs 13-23 run deterministically and avoid `error`.

### Task 4.1: Survival/entropy/wavelet group (13-17)
- **Location**: `src/statistic_harness/core/stat_plugins/next30b_addon.py`
- **Description**: Implement AFT lognormal proxy, CIF approximation, Haar transient detector, Hurst, permutation entropy drift.
- **Complexity**: 7
- **Dependencies**: Sprint 3
- **Acceptance Criteria**:
  - Clear assumptions surfaced in summary/debug for approximate methods.
- **Validation**:
  - Synthetic datasets with known shifts produce expected signal metrics.

### Task 4.2: Capacity/graph/spectral group (18-23)
- **Location**: `src/statistic_harness/core/stat_plugins/next30b_addon.py`
- **Description**: Implement frontier envelope, assortativity shift, pagerank hotspots, Higuchi FD, marked intensity, spectral radius stability.
- **Complexity**: 8
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Graph construction fallback paths deterministic and capped.
- **Validation**:
  - Graph-focused fixtures and deterministic comparisons.

## Sprint 5: Implement Expensive E Plugins (24-30) with Hard Caps
**Goal**: Complete all expensive methods safely under strict resource controls.  
**Demo/Validation**:
- IDs 24-30 execute with bounded runtime and no uncapped quadratic loops.

### Task 5.1: Bootstrap/permutation class (24-26)
- **Location**: `src/statistic_harness/core/stat_plugins/next30b_addon.py`
- **Description**: Implement bootstrap CI effect sizes, energy distance, randomization median shift with `max_resamples` + timer checks.
- **Complexity**: 8
- **Dependencies**: Sprint 4
- **Acceptance Criteria**:
  - Resampling count never exceeds config cap.
  - Timer enforced inside loops.
- **Validation**:
  - Stress tests with high row counts and low time budget.

### Task 5.2: Quadratic/entropy-heavy class (27-30)
- **Location**: `src/statistic_harness/core/stat_plugins/next30b_addon.py`
- **Description**: Implement distance covariance, triad motifs shift, MSE, sample entropy with quadratic caps and deterministic sampling.
- **Complexity**: 8
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - Quadratic operations are always capped.
  - No handler returns `error` on supported smoke data.
- **Validation**:
  - Cap-focused tests with forced-over-limit synthetic input.

## Sprint 6: References, Tests, Matrix Regeneration, and Full Gauntlet
**Goal**: Close all repo contracts and release-readiness checks.  
**Demo/Validation**:
- Full suite green and matrices/docs up to date.

### Task 6.1: Extend references mapping
- **Location**: `src/statistic_harness/core/stat_plugins/references.py`
- **Description**: Add new reference keys and plugin-to-reference mappings from work order section 7.
- **Complexity**: 4
- **Dependencies**: Sprint 3+
- **Acceptance Criteria**:
  - Every Next30B plugin gets a relevant default reference set.
- **Validation**:
  - Existing reference tests plus Next30B-specific assertions.

### Task 6.2: Add Next30B smoke and determinism suite
- **Location**: `tests/test_next30b_plugins_smoke.py`
- **Description**: Create parameterized tests for all 30 IDs using two synthetic datasets; require non-empty logs; deterministic equality for `ok` runs.
- **Complexity**: 6
- **Dependencies**: Sprint 5
- **Acceptance Criteria**:
  - No `error` statuses.
  - Determinism checks pass.
- **Validation**:
  - `pytest -q tests/test_next30b_plugins_smoke.py`

### Task 6.3: Regenerate matrices/contracts that gate CI
- **Location**: `docs/*.json` generated matrix files
- **Description**: Run matrix generators impacted by plugin count changes and commit updated outputs.
- **Complexity**: 5
- **Dependencies**: Task 6.2
- **Acceptance Criteria**:
  - Matrix tests match checked-in files.
- **Validation**:
  - `tests/test_plugins_functionality_matrix.py`
  - `tests/test_docs_implementation_matrix.py`
  - `tests/test_binding_implementation_matrix.py`

### Task 6.4: Full gauntlet and state log
- **Location**: repo root, `.codex/STATE.md`
- **Description**: Run complete `pytest -q`; fix failures; update state log with commands and outcomes.
- **Complexity**: 7
- **Dependencies**: Task 6.3
- **Acceptance Criteria**:
  - `pytest -q` passes.
  - `.codex/STATE.md` updated with final evidence.
- **Validation**:
  - Full suite run artifact/output captured in state log.

## Testing Strategy
- Layered validation:
  - Fast targeted tests per sprint.
  - Next30B dedicated smoke/determinism suite.
  - Existing repository contract tests (discovery/schema/matrices/references).
  - Full `pytest -q` gauntlet at end.
- Determinism approach:
  - Fixed seed, single-worker env for analysis-heavy tests.
  - Canonicalize volatile fields only where contract allows (for example runtime).

## Risks and Gotchas
- Duplicate plugin ID (`analysis_negative_binomial_overdispersion_v1`) may create handler ownership ambiguity.
- Matrix/coverage tests can fail if docs are not regenerated after adding plugin manifests.
- Expensive methods can silently violate budget without loop-level timer checks.
- Graph methods can become unstable with sparse/noisy categorical data unless graph fallback path is deterministic.
- Entropy/fractal methods can produce undefined values on low-variance series; must degrade gracefully.

## Mitigations
- Lock one canonical handler path per plugin ID and test explicit registry ownership.
- Add explicit cap/time guard assertions in tests.
- Include fallback summaries with structured `gating_reason` debug values when data is insufficient.
- Keep artifact payloads aggregate-only and deterministic.

## Rollback Plan
- Revert `next30b_addon.py` and wrapper plugin directories in one commit if regressions are severe.
- Keep registry wiring isolated to one import/update block for easy rollback.
- Maintain incremental commits by sprint to bisect quickly.
