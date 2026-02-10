# Plan: Repo Update Recommendations (Ops-Focused, Payout-Style Opportunities)

**Generated**: 2026-02-10  
**Estimated Complexity**: High

## Overview
Implement the changes requested in `docs/statistics_repo_update_recommendations.md` to make the harness produce **non-generic, action-oriented recommendations** (especially “batch/multi-input” opportunities like payout fan-out), while supporting:
- baseline close window analysis: **20th → 5th** (wrap-around window)
- target close window: **20th → EOM/31** (no wrap)
- pattern-aware process exclusions (glob/SQL-like/regex)
- a parameter-aware batching detector using the normalization-layer parameter tables (`parameter_kv`, `row_parameter_link`), without hardcoding process names

Key principle: keep the “4 pillars” (Performant, Accurate, Secure, Citable), deterministic outputs, no network, and schema-valid plugin outputs.

## Prerequisites
- Repo at `/mnt/d/projects/statistics_harness/statistics_harness`
- Tests must pass: `.venv/bin/python -m pytest -q`
- Confirm the normalization layer includes a stable `row_index` that can join to `parameter_kv.row_index`.

## Sprint 1: Shared Matchers + Exclusion Engine (P0)
**Goal**: Exclusions support wildcards/patterns and are applied consistently across analysis and report rendering.

**Demo/Validation**:
- Unit tests for matcher semantics: glob, SQL-like, exact, and optional regex.
- Regression: an exclude pattern like `LOS*` suppresses `LOS...` recommendations everywhere.

### Task 1.1: Implement Pattern-Aware Process Matcher
- **Location**: `src/statistic_harness/core/` (new module `process_matcher.py`)
- **Description**:
  - Add `compile_patterns(patterns: list[str]) -> Callable[[str], bool]`.
  - Support:
    - Exact: `POSTWKFL`
    - Glob: `LOS*`, `JEPOST*`
    - SQL-like: `JEPOST%` and `_` as single-char
    - Optional regex prefix: `re:^JEPOST.*`
  - Case-insensitive matching.
  - Deterministic compilation (stable ordering; explicit precedence: regex > glob/sql > exact).
- **Complexity**: 6/10
- **Dependencies**: none
- **Acceptance Criteria**:
  - Correct matches for all pattern kinds.
  - No catastrophic backtracking from user-provided regex (bound by a conservative allowlist or a hard-disable unless explicitly enabled).
- **Validation**:
  - New unit tests under `tests/` (new file `test_process_matcher.py`).

### Task 1.2: Wire Matcher Into Exclusion Sources (Known Issues + Env)
- **Location**: `src/statistic_harness/core/report.py`
- **Description**:
  - Update `_explicit_excluded_processes()` to return the raw configured strings (not just exact normalized values).
  - Update `_recommendation_has_excluded_process()` to use matcher instead of exact-set membership.
  - Preserve existing exact-match behavior as a subset of patterns.
- **Complexity**: 5/10
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Any recommendation referencing a process matching an exclusion pattern is suppressed.
- **Validation**:
  - Unit test covering `where.process_norm`, `contains.process_norm`, list values.

### Task 1.3: Apply Matcher Inside Plugins That Exclude Processes
- **Location**:
  - `plugins/analysis_queue_delay_decomposition/plugin.py`
  - `plugins/analysis_close_cycle_duration_shift/plugin.py` (and other close-cycle plugins that have exclusions)
  - `src/statistic_harness/core/stat_plugins/topo_tda_addon.py` (`analysis_actionable_ops_levers_v1`)
- **Description**:
  - Replace `.isin(exclude_processes)` / `value in excluded_set` with matcher predicate.
  - Keep current config keys (`exclude_processes`) but interpret as patterns.
  - Add default exclusions (overrideable): `LOS*`, `POSTWKFL`, `POSTWJFL`, `BKRV*`, `JEPOST*` (and accept SQL-like `JEPOST%`).
- **Complexity**: 6/10
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Exclusions affect both intermediate stats and final recommendations.
- **Validation**:
  - Plugin-level unit tests that confirm excluded patterns remove those process rows from computations.

## Sprint 2: Close-Window Wrap Fix + Target Close Window Spillover (P0)
**Goal**: Fix wrap-around close window day filtering and add target close window spillover metrics (baseline vs target).

**Demo/Validation**:
- Synthetic dataset demonstrating wrap-around window (20..5) correctness.
- Report contains “spillover past EOM” metrics.

### Task 2.1: Replace SQL `BETWEEN` Day Filtering With Wrap-Aware Predicates
- **Location**: `src/statistic_harness/core/report.py` (and any other SQL helper paths)
- **Description**:
  - Introduce helper: `sql_day_of_month_window(expr, start_day, end_day) -> (sql_fragment, params)`:
    - if `start_day <= end_day`: `day BETWEEN ? AND ?`
    - else: `(day >= ?) OR (day <= ?)`
  - Ensure day-of-month extraction is numeric-safe (avoid string comparisons).
  - Replace the existing `SUBSTR(..., 9, 2) BETWEEN ? AND ?` logic.
- **Complexity**: 5/10
- **Dependencies**: none
- **Acceptance Criteria**:
  - Wrap-around windows include days 20-31 plus 1-5.
- **Validation**:
  - Unit test under `tests/` (new file `test_close_window_sql_wrap.py`).

### Task 2.2: Add “Target Close End Day” Fields and Spillover Computation
- **Location**:
  - Close-cycle plugins: `plugins/analysis_close_cycle_capacity_model/plugin.py`, `plugins/analysis_close_cycle_duration_shift/plugin.py` (or centralize in a shared helper module)
  - Shared helper: `src/statistic_harness/core/close_cycle.py` (preferred)
- **Description**:
  - Add settings:
    - `baseline_close_start_day` default `20`
    - `baseline_close_end_day` default `5`
    - `target_close_end_day` default `31` (interpret as EOM)
  - Compute per `close_month` cohort:
    - baseline mask: `day>=20 OR day<=5`
    - target mask: `day>=20 AND day<=target_end`
    - spillover: baseline AND NOT target
  - Emit a finding with:
    - spillover job count
    - spillover runtime hours and queue-wait hours (where available)
    - top spillover processes by hours (with exclusion matcher applied before ranking)
    - spillover by day histogram (optional artifact)
- **Complexity**: 8/10
- **Dependencies**: Task 1.1, 2.1
- **Acceptance Criteria**:
  - Spillover rows are counted correctly for e.g. “Jan 5 belongs to Dec close_month”.
  - Excluded patterns are removed before spillover totals and top lists.
- **Validation**:
  - Synthetic fixture test under `tests/plugins/` (new file `test_close_cycle_spillover_target_window.py`).

## Sprint 3: Parameter-Sweep “Batch Input” Detector (P0)
**Goal**: Extend `analysis_actionable_ops_levers_v1` to detect payout-style fan-out as a **batch_input** recommendation, based on normalized parameter tables, not a single param column.

**Demo/Validation**:
- On a synthetic “payout-like” fixture:
  - process repeats many times in a close-month cohort
  - `Payout ID` (or equivalent key) is near 1:1 with runs
  - other keys are stable
  - emits `action_type="batch_input"` with key evidence.

### Task 3.1: Add SQL-Backed Parameter Statistics Query Helpers
- **Location**: `src/statistic_harness/core/sql_assist.py` or `src/statistic_harness/core/sql_intents.py`
- **Description**:
  - Add a read-only helper that returns, for each (process_norm, key):
    - `runs_with_key` (count distinct `row_index`)
    - `unique_values` (count distinct `value`)
    - `coverage` (runs_with_key / total runs for process)
    - `unique_ratio` (unique_values / runs_with_key)
  - Support cohorting:
    - by `close_month` and/or close-window days using baseline window masks derived from timestamp columns in normalized table.
  - Guardrails:
    - hard cap number of keys evaluated per process (top-N by coverage)
    - ignore key blacklist (`run id`, `queue id`, `seq`, etc.) via matcher-like logic on key names.
- **Complexity**: 8/10
- **Dependencies**: Sprint 1, Sprint 2 (close_month derivation)
- **Acceptance Criteria**:
  - Works even when the dataset has no obvious single “param column” because it uses `parameter_kv`.
  - Emits SQL evidence (`query`, `row_ranges`/`row_ids` where feasible) for citation.
- **Validation**:
  - Unit tests against the fixture SQLite DB in `tests/fixtures/db/`.

### Task 3.2: Extend `analysis_actionable_ops_levers_v1` With `batch_input`
- **Location**: `src/statistic_harness/core/stat_plugins/topo_tda_addon.py` (`_actionable_ops_levers_v1`)
- **Description**:
  - Keep existing levers: `route_process`, `unblock_dependency_chain`, `reschedule`, `batch_or_cache`, `throttle_or_dedupe`.
  - Add a second batching detector:
    - `action_type="batch_input"`
    - Condition examples (tunable):
      - `coverage >= 0.9`
      - `unique_ratio >= 0.8` inside baseline close window (and/or within close_month cohorts)
      - cohort peaks: in at least one close_month, `runs >= k_min` and `unique_ratio >= threshold`
  - Emit recommendation text in plain English:
    - “Modify job to accept a list of {key} values; run once per close-month cohort rather than once per key.”
  - Artifacts:
    - `batch_input_candidates.json` with per-process per-key cohort stats + sample values.
- **Complexity**: 9/10
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Detects the payout-style pattern without hardcoding process names.
  - Recommendation includes:
    - process_norm
    - key
    - best_close_month + runs + unique_values
    - sample values (redacted if privacy enabled)
    - clear “upper bound” vs “estimate” labeling
- **Validation**:
  - New plugin test under `tests/plugins/` (new file `test_actionable_ops_batch_input.py`).

## Sprint 4: Recommendation De-Genericization (P1)
**Goal**: Prevent “non-starter” recommendations from dominating the Top-N while preserving useful targeted schedule and lever recommendations.

**Demo/Validation**:
- `scripts/show_actionable_results.py` output surfaces structural levers first and keeps breadth.

### Task 4.1: Add Action-Type Prioritization + Per-Action Caps
- **Location**: `src/statistic_harness/core/report.py`, `scripts/show_actionable_results.py`
- **Description**:
  - Add config knobs:
    - `suppress_action_types: []`
    - `max_per_action_type: { "batch_input": 5, "batch_or_cache": 5, "reschedule": 3 }`
  - Scoring tiers:
    - Tier 1: structural (`batch_input`, `batch_or_cache`, dedupe, chain batching)
    - Tier 2: targeted schedule changes (only if concentrated in spillover or top driver)
    - Tier 3: generic reductions (suppress unless extremely high impact)
- **Complexity**: 6/10
- **Dependencies**: Sprint 3
- **Acceptance Criteria**:
  - Top-N list is not dominated by generic items when structural items exist.
- **Validation**:
  - Unit test under `tests/` (new file `test_recommendation_prioritization_tiers.py`).

## Sprint 5: Integration + Regression Runbook
**Goal**: Ensure changes are reflected in matrices/docs and can be validated on the real dataset.

**Demo/Validation**:
- Full run against loaded dataset completes with `status=completed`.
- `answers_recommendations.md` contains at least one `batch_input` recommendation when the signal exists.

### Task 5.1: Update Matrices + Docs
- **Location**: `scripts/update_docs_and_plugin_matrices.py`, `docs/plugins_functionality_matrix.md`
- **Description**:
  - Add/confirm matrix rows for:
    - exclusion matcher adoption (which plugins use patterns)
    - spillover target close window metrics
    - actionable ops: `batch_input` support
- **Complexity**: 3/10
- **Dependencies**: prior sprints
- **Validation**:
  - `scripts/verify_docs_and_plugin_matrices.py` (or `python -m pytest -q` if tests cover).

### Task 5.2: Full Gauntlet
- **Validation**:
  - `.venv/bin/python -m pytest -q`
  - Full run command (repo standard) on the loaded dataset, then `./s` to inspect recommendations.

## Testing Strategy
- Unit tests for `process_matcher` + wrap-around SQL helper.
- Plugin tests for:
  - close-cycle spillover baseline vs target
  - actionable ops `batch_input` detection using `parameter_kv`
- Integration/regression:
  - ensure no `partial` runs due to report ordering
  - ensure exclusions suppress outputs consistently across report + analysis

## Potential Risks & Gotchas
- Parameter tables may be missing or sparsely populated for some ERPs:
  - Mitigation: degrade to existing param-column heuristic path with a clear gating reason.
- Regex patterns can be abused:
  - Mitigation: default-disable regex (`re:`) unless explicitly enabled; or apply strict pattern length limits.
- Close-month derivation depends on reliable timestamps:
  - Mitigation: use the best available time column (queue/start) and report which column was used; if missing, skip spillover metrics with a clear reason.
- Performance on 2M rows:
  - Mitigation: compute key stats via SQL aggregation (SQLite) instead of loading into pandas; cap candidate keys per process.

## Rollback Plan
- Keep new behavior behind config defaults:
  - If `target_close_end_day` not provided, spillover metrics can be suppressed.
  - If parameter tables absent, `batch_input` path skips cleanly and the plugin continues to emit other levers.
- Revert matcher adoption in one place at a time (central helper makes this straightforward).
