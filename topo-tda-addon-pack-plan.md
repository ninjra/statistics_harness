# Plan: Topo/TDA Add-On Pack + CEO-Grade Ideaspace Recommendations

**Generated**: 2026-02-05
**Estimated Complexity**: High

## Overview
Implement the plugin families specified in `docs/deprecated/codex_topo_tda_plugin_spec_addon.md` (Families A-E) and add an "ideaspace recommendation" layer that produces **CEO-grade, process_id-targeted actions** (Family F) for unknown ERP process logs.

Key outcome: after running `stat-harness ... run --plugins auto`, the system produces:
- `report.json` + `report.md` with **discovery-only recommendations** (known-issue landmarks excluded).
- At least 3 concrete recommendations that identify `process_id` (or `process_norm`) + action + quantified expected delta (seconds/hours/%), when the dataset contains enough signal.

## Prerequisites
- Python 3.11+ (repo currently uses venv + pytest)
- Keep runtime local-only: no network calls, no subprocess, no shell-outs in plugins.
- Optional dependencies must remain optional (plugins degrade gracefully without them) unless user explicitly approves promoting them to core deps:
  - TDA: `ripser` or `gudhi`
  - Time series STL: `statsmodels`
  - Some tests may optionally use `scipy`

## Clarifying Questions (One Round)
1. Should optional dependencies (`ripser/gudhi`, `statsmodels`, `scipy`) remain optional extras, or are you OK making any of them core `pyproject.toml` dependencies? make core
2. What is the canonical CEO output format: max 5 bullets, max 10 bullets, or a single "Top 3 actions" table? max 20. they have to be specific to a process or sequence of processes, no generic or global changes. we can only recommend procedure or scheduling improvement, we cannot change the code that is running, only some variables.
3. For “process_id”, do you want strict use of the detected process column (e.g. `PROCESS_NAME`) or allow derived IDs like `PROCESS_NAME + PARAMS` (e.g. `qpec:batch=close`) when that yields better actions? derive wehatever you like, it may be different every time so make sure it is all stored in a learning encyclopedia or db or osomething  so each one we do learns from the last ones
4. Is it acceptable for “CEO recommendations” to include one instrumentation action when it unlocks large savings (e.g. “add TRACE_ID”), or should those be strictly separated into an “Engineering actions” section? acceptable, but not as a replacement for the others.
5. Any hard exclusions besides known-issue landmarks (e.g. do not recommend actions about `qemail` or `qpec` at all, even if they’re top drivers)? i know at least one more issue exists that is actionable to a specific process or sequence of processes, and needs no more information than what is in the log. you need to mutate our methods or cross apply, or use the internet to look up more statistical, mathematical,topographical and etc methods to add as plugins. not finding any results is absolutely unaccpetable, especially when I know that one exists and you have yet look for that type of situation otherwise you would likely recommend more than what i found.

## Sprint 1: Core Scaffolding + Shared Utilities
**Goal**: Provide reusable building blocks (profiling, selection, redaction, budgets, citations) so all new plugins are consistent and safe.
**Demo/Validation**:
- `python -m pytest -q`
- Run `statistic_harness.cli run` on `tests/fixtures/quorum_close_cycle.csv` and confirm new plugins can be discovered (even if most skip).

### Task 1.1: Extend Planner Feature Detection (Capabilities)
- **Location**: `src/statistic_harness/core/planner.py`
- **Description**: Add features required by A-E:
  - `has_coords` (x/y[/z] detection)
  - `has_text`
  - `has_groupable` (categorical columns with reasonable cardinality)
  - `has_epoch` (time/epoch column distinct from event timestamps)
  - `has_point_id` (entity id that repeats)
- **Complexity**: 6
- **Dependencies**: none
- **Acceptance Criteria**:
  - New capability tokens can be expressed in `plugins/*/plugin.yaml` and are selectable under `--plugins auto`.
- **Validation**:
  - Unit tests for `_infer_dataset_features` on synthetic frames.

### Task 1.2: Shared “Ideaspace” Utilities Module
- **Location**: `src/statistic_harness/core/stat_plugins/ideaspace.py` (new)
- **Description**: Implement utilities required by multiple plugin families:
  - Group/feature selection helpers (bounded cardinality, k-min)
  - Stable seeded RNG per-run
  - BH/FDR helper (reuse existing)
  - Redaction helpers for labels and exemplars
  - Budget guardrails (max_rows, max_pairs, max_points)
  - Citation helpers: `references=[...]` always present for ok/degraded
- **Complexity**: 7
- **Dependencies**: existing stat_plugins utilities
- **Acceptance Criteria**:
  - No plugin needs to re-implement selection/redaction/budget logic.
- **Validation**:
  - Unit tests: determinism, redaction masking, budget caps.

### Task 1.3: Add Cross-Cutting Test Harness for New Plugins
- **Location**: `tests/plugins/test_crosscutting_new_plugins.py` (new)
- **Description**: Add parametrized tests enforcing the doc’s cross-cutting requirements for all new plugin IDs (A-E, F):
  - determinism under fixed `run_seed`
  - “skip with gating_reason” when inapplicable
  - citations presence when `status in {ok,degraded}`
  - redaction: no raw emails/uuids in findings
- **Complexity**: 6
- **Dependencies**: Task 1.2
- **Acceptance Criteria**:
  - New plugins cannot regress determinism/security without test failures.

## Sprint 2: Family D (Classic Auto-Scan Suite)
**Goal**: Implement broad, low-risk scanners that produce non-trivial, specific findings on unknown datasets.
**Demo/Validation**:
- Run on `tests/fixtures/synth_shift_corr.csv`, `tests/fixtures/synth_timeseries.csv` and assert at least 1 actionable finding per plugin where applicable.

### Task 2.1: `analysis_ttests_auto`
- **Location**: `plugins/analysis_ttests_auto/*` (new plugin dir: `plugin.yaml`, `plugin.py`, `config.schema.json`, `output.schema.json`)
- **Description**: Welch tests + effect sizes, FDR, k-min grouping.
- **Complexity**: 5
- **Acceptance Criteria**:
  - Emits findings with `process_norm` or `group_id` plus measured delta and a concrete action template (e.g., reroute group, fix outlier).
- **Validation**:
  - Synthetic fixture test with known mean shift.

### Task 2.2: `analysis_chi_square_association`
- **Location**: `plugins/analysis_chi_square_association/*`
- **Description**: Cat-cat association scan with Cramer’s V; caps pairs; FDR.
- **Complexity**: 5

### Task 2.3: `analysis_regression_auto`
- **Location**: `plugins/analysis_regression_auto/*`
- **Description**: Choose target (duration-like or highest variance), fit Lasso/Huber (degrade if unavailable), bootstrap stability.
- **Complexity**: 7

### Task 2.4: `analysis_time_series_analysis_auto`
- **Location**: `plugins/analysis_time_series_analysis_auto/*`
- **Description**: Time inference, trend/seasonality (optional STL), residual anomalies, periodicity peaks.
- **Complexity**: 7

### Task 2.5: `analysis_cluster_analysis_auto` + `analysis_pca_auto`
- **Location**: `plugins/analysis_cluster_analysis_auto/*`, `plugins/analysis_pca_auto/*`
- **Description**: Cluster profiles, outlier clusters, PCA loadings; map back via redacted exemplars.
- **Complexity**: 6

## Sprint 3: Family B (Topographic Similarity + Permutation Map Tests)
**Goal**: Detect pattern differences between groups/time windows that translate into concrete operational levers (server routing, batching, throttling).
**Demo/Validation**:
- Synthetic “scale-only vs angle-only” tests.
- Ensure outputs do not leak raw labels; enforce k-min.

### Task 3.1: `analysis_topographic_similarity_angle_projection`
- **Location**: `plugins/analysis_topographic_similarity_angle_projection/*`
- **Description**: Build maps over numeric features for candidate groupings; cosine similarity + projection; report most divergent groups.
- **Complexity**: 7

### Task 3.2: `analysis_topographic_angle_dynamics`
- **Location**: `plugins/analysis_topographic_angle_dynamics/*`
- **Description**: Sliding windows over time; detect emergence of a pattern; bootstrap confidence.
- **Complexity**: 6

### Task 3.3: `analysis_topographic_tanova_permutation`
- **Location**: `plugins/analysis_topographic_tanova_permutation/*`
- **Description**: Permutation test over map differences; FDR across candidate group cols.
- **Complexity**: 8

### Task 3.4: `analysis_map_permutation_test_karniski`
- **Location**: `plugins/analysis_map_permutation_test_karniski/*`
- **Description**: Distribution-free permutation test (implemented from first principles), high-dim low-n safe; feature contribution diagnostics.
- **Complexity**: 9

## Sprint 4: Family C (Surface/Terrain Complexity)
**Goal**: Provide a surface adapter and conservative surface metrics that can identify “hot regions” and stable hotspots without relying on GIS-specific inputs.
**Demo/Validation**:
- SurfaceBuilder determinism tests.
- Synthetic surfaces for RI/TPI/SSO/hydrology basins.

### Task 4.1: SurfaceBuilder
- **Location**: `src/statistic_harness/core/stat_plugins/surface.py` (new)
- **Description**: `build_surface(...)` with coords detection or PCA embedding fallback; deterministic binning + optional IDW fill.
- **Complexity**: 9

### Task 4.2: Surface Metrics Plugins (C1-C6)
- **Location**: `plugins/analysis_surface_*/*` (6 plugins)
- **Description**: Implement curvature (FFT conv), variogram FD, rugosity, TPI, SSO eigen fabric, D8 flow/watersheds (bounded grid).
- **Complexity**: 10
- **Acceptance Criteria**:
  - Each plugin emits at least one stable, citable metric and a hotspot summary with safe references.

## Sprint 5: Family A (TDA Suite)
**Goal**: Add high-insight structural drift detectors with strict budgets and graceful degraded fallbacks.
**Demo/Validation**:
- Circle/two-cluster synthetic tests.
- Windowed drift synthetic changepoint test.

### Task 5.1: `analysis_tda_persistent_homology` (with degraded fallback)
- **Location**: `plugins/analysis_tda_persistent_homology/*`
- **Description**: Use `ripser`/`gudhi` when available; otherwise MST + cycle proxy; output scalar + windowed summaries.
- **Complexity**: 9

### Task 5.2: Landscapes, Mapper, Betti Curve Changepoint (A2-A4)
- **Location**: `plugins/analysis_tda_*/*` (3 plugins)
- **Description**: Implement with tight caps and deterministic approximations.
- **Complexity**: 10

## Sprint 6: Family E (Bayesian + Uncertainty)
**Goal**: Provide uncertainty-aware displacement and surface stability reporting.
**Demo/Validation**:
- Synthetic displacement: posterior P(||Δ||>ε) high for injected subset.
- Monte Carlo stability scores behave as expected.

### Task 6.1: `analysis_bayesian_point_displacement`
- **Location**: `plugins/analysis_bayesian_point_displacement/*`
- **Description**: Conjugate NIW-ish closed-form + seeded sampling; degrade to group-centroids if no explicit point ids.
- **Complexity**: 9

### Task 6.2: `analysis_monte_carlo_surface_uncertainty`
- **Location**: `plugins/analysis_monte_carlo_surface_uncertainty/*`
- **Description**: Correlated noise via FFT; recompute chosen surface metric(s); report stability.
- **Complexity**: 10

### Task 6.3: `analysis_surface_roughness_metrics`
- **Location**: `plugins/analysis_surface_roughness_metrics/*`
- **Description**: 1D + 2D roughness, PSD slope, anisotropy direction.
- **Complexity**: 7

## Sprint 7: Family F (CEO-Grade Ideaspace Recommendations)
**Goal**: Generate recommendations that are specific (process_id + lever + expected delta) and not generic “reduce wait time”.
**Demo/Validation**:
- On a multi-process synthetic ERP-like dataset fixture, produce at least:
  - 1 routing recommendation (server imbalance/outlier)
  - 1 batching/frequency recommendation (burst + low param diversity)
  - 1 scheduling recommendation (close-window contention not tied to known issues)

### Task 7.1: `analysis_actionable_ops_levers_v1`
- **Location**: `plugins/analysis_actionable_ops_levers_v1/*`
- **Description**: Convert statistical findings into a bounded set of actionable levers, with templates like:
  - `route_process(process_id, from=server_a, to=server_b)` with expected median delta
  - `batch_or_cache(process_id, key=param_subset)` when param diversity is low and volume is high
  - `reschedule(process_id, window=after_hours/weekend/close_window)` when time-bucket effects dominate
  - `throttle_or_dedupe(process_id, trigger=burst)` when bursty arrivals correlate with downstream slowdown
  - `add_dependency(process_id, depends_on=process_b, key=param_overlap)` when overlap is high
- **Complexity**: 10
- **Acceptance Criteria**:
  - Each recommendation includes:
    - `process_norm` (or derived `process_id`)
    - `action_type`
    - `expected_delta_{seconds|hours|percent}`
    - `confidence` + `assumptions`
    - evidence pointers (artifact paths, row_ids/column_ids)

### Task 7.2: Report Integration (CEO Section)
- **Location**: `src/statistic_harness/core/report.py`
- **Description**: Add a top section in `report.md` that prints “Top Actions” from `analysis_actionable_ops_levers_v1` (discovery-only, excluding known-issue landmarks).
- **Complexity**: 6

### Task 7.3: Ideaspace Index Expansion
- **Location**: `src/statistic_harness/core/report.py`
- **Description**: Extend `report.json["ideaspace"]` to include:
  - detected roles (time/process/server/user/params/case_id)
  - normalization decisions
  - which idea-families were applicable / skipped and why
- **Complexity**: 6

## Testing Strategy
- Unit tests per plugin with synthetic fixtures that encode one clear “ground truth” signal.
- Integration test that runs pipeline end-to-end and asserts:
  - `report.md` and `report.json` exist
  - `report.json` validates `docs/report.schema.json`
  - Deterministic outputs under fixed `run_seed`
- Evaluator harness: add ground-truth fixtures for at least one non-quorum, multi-process ERP-like dataset so CEO recommendations are validated (process_id + action type + quantified delta within tolerance).

## Potential Risks & Gotchas
- Plugin explosion: A-E is large; strict sprinting and optional deps are required to avoid destabilizing the repo.
- Performance: surface + MC + mapper can go quadratic; must hard-cap and degrade.
- Recommendation quality: turning statistical findings into actions requires careful templates and guardrails; avoid recommending on k<k_min groups or tiny samples.
- Known issues should remain as landmarks only; do not let them suppress novel discovery elsewhere (except by explicit filtering).

## Rollback Plan
- All new plugins gated behind `--plugins all` initially; keep `auto` conservative until validated.
- Each plugin is independently removable by deleting its `plugins/analysis_*` directory and removing planner selection (no core coupling).
- Keep optional deps behind extras; removing extras does not break runtime.
