# Plan: Support ~2,000,000 Row Datasets (Harness-First, 4 Pillars)

**Generated**: 2026-02-08  
**Estimated Complexity**: High

## Overview
Most plugins currently call `ctx.dataset_loader()` which loads a pandas DataFrame (optionally limited by `budget.row_limit`). At ~2,000,000 rows, naive full-DataFrame loads per plugin are too slow and risk OOM, especially with parallel analysis.

This plan makes the harness scale to ~2M rows while optimizing for the 4 pillars:
- **Performant**: budget caps, streaming, caching, and safe concurrency.
- **Accurate**: full-dataset scans via batching/streaming; deterministic sketches/diagnostics where needed.
- **Secure**: keep sandbox/no-network, avoid new attack surface (no shelling out), keep artifacts under `appdata/`.
- **Citable**: preserve references; ensure report captures budget and provenance.

Key strategy:
1. **Streaming first**: add first-class batch APIs to `PluginContext` and migrate plugins off full-DataFrame loads.
2. **Policy-driven complexity budgets (not row sampling)**: for large datasets, bound algorithmic complexity (max columns/pairs/groups/windows) while still scanning all rows in batches.
3. **One-time dataset tuning**: indexes (already done for `row_index`) + optional local caches.
4. **Observability/ETA**: record per-plugin durations/RSS, estimate time-to-completion, and auto-tune concurrency.

## Prerequisites
- No runtime network calls (existing sandbox guard must remain).
- Python 3.11+.
- Must keep plugin IDs and plugin discovery/manifest behavior stable (no changing which plugins load; only harness behavior).
- “Do not ship unless” gate: `python -m pytest -q` must pass.

## Locked Requirement (From You)
- **No row sampling**. Plugins must support large datasets by intelligently **batching/streaming and scanning** (multi-pass ok). Long runtimes (even ~38 hours) are acceptable as long as results are actionable and specific.

Notes:
- Streaming sketches (quantiles/heavy hitters/entropy) are acceptable if they scan all rows and are deterministic; they are not “row sampling”.

## Sprint 1: Streaming Contract + Safety Gates
**Goal**: Make 2M-row runs safe by default: no accidental full-DF loads, deterministic behavior, and explicit streaming APIs, without changing plugin IDs.

**Demo/Validation**:
- Run a “large dataset policy” integration test that fakes row_count >= 2,000,000 and asserts budgets/timeouts are applied.
- `bash scripts/run_gauntlet.sh`

### Task 1.0: Generate Plugin Data-Access Matrix (Avoid Duplicate Work)
- **Location**: new `scripts/plugin_data_access_matrix.py`, outputs `docs/plugin_data_access_matrix.md` + `docs/plugin_data_access_matrix.json`
- **Description**: Generate and keep updated a matrix that records, per plugin:
  - whether it calls `ctx.dataset_loader()` unbounded
  - whether it supports streaming via `ctx.dataset_iter_batches()`
  - primary access pattern (full DF, batched, SQL aggregation, cached)
  - migration status: `not_started|in_progress|streaming_ok`
- **Complexity**: 5/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - Matrix includes all plugin IDs under `plugins/`.
  - Matrix is generated deterministically and is used to drive subsequent migration work.
- **Validation**:
  - Unit test: matrix generation includes all plugins and has stable ordering.

### Task 1.1: Add Large-Dataset Policy Module (Complexity Budgets)
- **Location**: `src/statistic_harness/core/large_dataset_policy.py`
- **Description**: Implement a deterministic policy that decides complexity/resource caps (not row sampling) based on:
  - dataset row_count/column_count
  - plugin type (`profile|planner|transform|analysis|report|llm`)
  - plugin_id patterns (e.g., `analysis_matrix_profile_*`, `analysis_*prefixspan*`, `analysis_*lda*`)
  - optional user overrides from env or a policy file.
  Output should be a dict merged into plugin settings and/or `ctx.budget`, e.g.:
  - `budget.time_limit_ms`, `budget.cpu_limit_ms`
  - `budget.batch_size`
  - `budget.max_cols`, `budget.max_pairs`, `budget.max_groups`, `budget.max_windows`, `budget.max_findings`
- **Complexity**: 6/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - Policy is deterministic given same inputs.
  - Given `row_count>=2_000_000`, policy provides non-null `batch_size` and complexity caps for heavy plugins.
- **Validation**:
  - Unit tests for policy decisions.

### Task 1.2: Apply Policy in Pipeline Before Subprocess Execution
- **Location**: `src/statistic_harness/core/pipeline.py`
- **Description**: In `run_spec()`, merge computed policy budget into plugin config:
  - If plugin config explicitly sets a cap, keep it as an override.
  - Otherwise, inject policy caps.
  - Ensure `budget.sampled` remains `false` by default.
- **Complexity**: 5/10
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - For large datasets, plugins have access to `budget.batch_size` + complexity caps.
  - Subprocess timeout uses `budget.time_limit_ms`.
- **Validation**:
  - Integration test with a tiny fixture but mocked dataset metadata.

### Task 1.3: Enforce “No Full-DF Load” For Large Datasets
- **Location**: `src/statistic_harness/core/dataset_io.py`, `src/statistic_harness/core/plugin_runner.py`
- **Description**: For datasets above a threshold, fail closed if a plugin attempts `ctx.dataset_loader()` without a `row_limit`, with an actionable error telling plugin authors/operators to use `ctx.dataset_iter_batches()` (or explicitly override to allow full loads).
- **Complexity**: 4/10
- **Dependencies**: Task 1.2
- **Acceptance Criteria**:
  - On 2M rows, unbounded full-DF loads fail closed with actionable guidance to use batch iteration.
- **Validation**:
  - Unit test that large row_count triggers the refusal unless explicitly allowed.

## Sprint 2: Streaming API in PluginContext + First Migrations
**Goal**: Enable full-dataset correctness where feasible by streaming batches, and reduce reliance on full in-memory DataFrames.

**Demo/Validation**:
- Add one streaming-first plugin path (e.g., `profile_basic` or a simple analysis) and verify outputs remain deterministic.
- `bash scripts/run_gauntlet.sh`

### Task 2.1: Add `dataset_iter_batches()` to PluginContext
- **Location**: `src/statistic_harness/core/types.py`, `src/statistic_harness/core/plugin_runner.py`, `src/statistic_harness/core/dataset_io.py`
- **Description**:
  - Extend `PluginContext` with `dataset_iter_batches(columns=None, batch_size=..., row_limit=None)`.
  - Implement it in `plugin_runner.py` to call `DatasetAccessor.iter_batches()`, defaulting `batch_size` from `budget.batch_size`.
- **Complexity**: 5/10
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Existing plugins unchanged (backwards compatible).
  - New/updated plugins can stream data deterministically.
- **Validation**:
  - Unit test that iter_batches yields stable ordering and respects `row_limit`.

### Task 2.2: Streaming Profile for Large Datasets
- **Location**: `plugins/profile_basic/plugin.py`, `plugins/profile_eventlog/plugin.py`
- **Description**: For large datasets, compute core profile stats via batches (counts, null rates, type inference via bounded scans, basic percentiles via deterministic streaming sketches or two-pass bounded approaches).
- **Complexity**: 7/10
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Profile plugins do not require loading the full DataFrame for 2M rows.
  - Output remains schema-valid and deterministic.
- **Validation**:
  - Unit tests with synthetic larger-than-threshold row_count (small actual data ok) verifying streaming path is chosen.

### Task 2.3: Identify and Migrate Top “Worst Offenders”
- **Location**: `plugins/*/plugin.py`, new doc `docs/perf_hotspots.md`
- **Description**:
  - Use `plugin_executions.max_rss` and `duration_ms` to rank slowest plugins on real runs.
  - For the top 10, either:
    - implement streaming algorithm; or
    - replace full-DF algorithms with streaming/multi-pass alternatives; if infeasible, mark `skipped` with a specific reason and a concrete follow-up recommendation.
- **Complexity**: 8/10
- **Dependencies**: Sprint 1, Task 2.1
- **Acceptance Criteria**:
  - No single plugin should OOM or run unbounded by default on 2M rows.
- **Validation**:
  - Integration run on a large-ish synthetic dataset (or downscaled) verifying no plugin attempts a full load.

## Sprint 3: Dataset Materialization Cache (Optional but High Leverage)
**Goal**: Avoid repeated SQLite scans across many plugin subprocesses.

**Demo/Validation**:
- Create a cache for the dataset once; subsequent plugin loads should be faster.
- `bash scripts/run_gauntlet.sh`

### Task 3.1: Add Columnar Cache Format (No New Heavy Deps)
- **Location**: `src/statistic_harness/core/dataset_cache.py` (new), `src/statistic_harness/core/dataset_io.py`
- **Description**:
  - Implement a cache keyed by `dataset_version_id` + `data_hash` + schema hash.
  - Store columns as numpy `.npy` (optionally memory-mapped) under `appdata/cache/datasets/<key>/`.
  - `DatasetAccessor.iter_batches(...)` should prefer the cache if present; otherwise scan SQLite once and populate cache (optional knob) while streaming out batches.
  - `DatasetAccessor.load(...)` may use cache for small datasets, but for large datasets we will enforce streaming APIs instead of full loads.
- **Complexity**: 8/10
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Cache is deterministic and safe to reuse.
  - Cache writes stay under `appdata/` and are traversal-safe.
- **Validation**:
  - Unit tests for keying, cache hit/miss behavior.

### Task 3.2: CLI Helpers to Build/Inspect Cache
- **Location**: `scripts/materialize_dataset_cache.py`, `scripts/inspect_dataset_cache.py`
- **Description**: Provide one-line operator tooling to precompute caches for a dataset version.
- **Complexity**: 4/10
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Running materialization twice is idempotent.
- **Validation**:
  - Smoke test in CI (small dataset).

## Sprint 4: Auto-Tuned Concurrency + Better ETA
**Goal**: Use observed resource usage to safely increase throughput on large datasets without OOM.

**Demo/Validation**:
- Run with `STAT_HARNESS_MAX_WORKERS_ANALYSIS=auto` and confirm it selects a safe worker count.

### Task 4.1: Persist/Use Execution Telemetry for Scheduling
- **Location**: `src/statistic_harness/core/pipeline.py`, `src/statistic_harness/core/storage.py`
- **Description**:
  - Use recorded `max_rss`/`duration_ms` to:
    - compute a safe concurrency for future layers,
    - optionally serialize known memory hogs even when others can parallelize.
- **Complexity**: 7/10
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - No layer starts more workers than budgeted.
  - Memory hog plugins do not run concurrently by default.
- **Validation**:
  - Unit test with synthetic telemetry driving scheduler decisions.

### Task 4.2: Improve Run ETA Reporting
- **Location**: `scripts/run_run_status.py`
- **Description**:
  - Estimate ETA from median/EMA of completed plugin durations, not “plugins/sec since start”.
  - Report `expected_plugins_executable`, `done`, `running`, and `eta_minutes`.
- **Complexity**: 4/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - ETA stabilizes after ~10 plugins.
- **Validation**:
  - Unit test on a seeded fake timeline.

## Testing Strategy
- Unit tests:
  - Large-dataset policy decisions, deterministic keys, cache behavior.
  - Streaming batch ordering/limits.
  - Scheduler decisions from telemetry.
- Integration tests:
  - Full pipeline on fixture dataset still produces `report.md` and `report.json`.
  - Large-dataset policy test: simulate `row_count>=2_000_000` via DB row metadata and assert no unbounded full-DF loads occur by default.
- Always gate with `bash scripts/run_gauntlet.sh`.

## Potential Risks & Gotchas
- “Run all plugins against the entire dataset” conflicts with feasibility for O(n²) methods.
  - Mitigation: streaming/multi-pass implementations + strict complexity caps (pairs/groups/windows) that still scan all rows; if infeasible, `skipped` with an actionable recommendation.
- Cache correctness:
  - Mitigation: strict keying by data_hash + schema; include SHA256 in manifest; never trust external paths.
- SQLite locking/IO contention with higher parallelism:
  - Mitigation: keep WAL, add indexes, cap workers, optionally stagger heavy readers.
- Determinism vs speed:
  - Mitigation: deterministic batch ordering + deterministic sketches; deterministic ordering in reports.

## Rollback Plan
- Keep all new behavior behind env/policy toggles:
  - `STAT_HARNESS_LARGE_DATASET_POLICY=off` to disable injected policy caps.
  - `STAT_HARNESS_DATASET_CACHE=off` to disable cache reads/writes.
  - `STAT_HARNESS_MAX_WORKERS_ANALYSIS` to force fixed parallelism.
- Revert to current behavior by disabling toggles; no schema-breaking changes.
