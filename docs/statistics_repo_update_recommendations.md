# Statistics_Harness repo update recommendations (ops-focused)

This document focuses on changes that make the harness produce **non-generic, action-oriented recommendations** from the processor-log dataset (e.g., “batch/multi-input opportunities” like the payout pipeline), while also supporting a **target close window of 20th → end-of-month (EOM / day 31)** and a broader **process exclude pattern list**.

---

## 1) What is currently preventing the harness from finding “payout-style” opportunities

### 1.1 Exclusions are exact-match only (no wildcards)
Multiple places treat the exclude list as **exact strings** (e.g., `value in excluded_set`), so patterns like `JEPOST%` or `LOS*` are not honored. This causes the harness to spend recommendation budget on “already-accounted-for” jobs.

**Fix:** add a shared *pattern-aware* exclude engine (glob/SQL-like/regex) and use it everywhere exclusions are applied (analysis + recommendation rendering).

### 1.2 Close-cycle day filtering has a wrap-around bug in at least one SQL path
There is at least one SQL query path that filters close-window rows using `BETWEEN start_day AND end_day` on day-of-month text. This fails when `start_day > end_day` (e.g., `20…5`). That can silently drop close-cycle rows from certain analyses that depend on that query.

**Fix:** replace `BETWEEN` logic with wrap-aware day predicates: `(day >= start) OR (day <= end)` when `start > end`.

### 1.3 “Batch/cache” lever currently only fires when the varying key repeats (cache-like), not when it sweeps (batch-like)
The existing “batch_or_cache” recommendation language and evidence is based on **repeat frequency** (same key appearing many times). That misses “sweep” patterns where the key is *mostly unique within a close cycle* (e.g., many distinct Payout IDs), even though it’s exactly the scenario where a multi-input/batch interface is valuable.

**Fix:** extend the actionable ops plugin to detect **parameter sweeps** (high unique ratio within close cycle / close-month cohorts) and emit a **batch_input** recommendation.

### 1.4 Parameters exist, but the lever logic doesn’t fully exploit them
The repo already supports parameter normalization (`parameter_entities`, `parameter_kv`, and `row_parameter_link`). The actionable ops lever should query those tables rather than relying on a single param column.

**Fix:** join actionable ops lever logic against `parameter_kv` to detect stable vs varying keys.

---

## 2) Dataset evidence: payout pipeline is a clear “batch input” opportunity

From `proc log 1-14-26.csv`, the close-window (20th–5th) contains a payout “fan-out” where the same jobs run hundreds of times with different `Payout ID` values.

### 2.1 Payout processes (close window only)
| Process    |   Runs_in_close_20_5 |   Unique_payout_ids_in_close |   Runs_on_days_1_5 | Peak_close_month   |   Peak_month_runs |   Peak_month_unique_payout_ids |
|:-----------|---------------------:|-----------------------------:|-------------------:|:-------------------|------------------:|-------------------------------:|
| RPT_POR002 |                  651 |                          542 |                215 | 2025-10            |               418 |                            416 |
| POEXTRPRVN |                  612 |                          535 |                195 | 2025-10            |               415 |                            415 |
| POGNRTRPT  |                  608 |                          533 |                192 | 2025-10            |               414 |                            414 |
| POEXTRPEXP |                  609 |                          532 |                194 | 2025-10            |               413 |                            413 |

Key signal: in the peak close month, each of these processes runs **~400+ times** with **~400+ distinct Payout IDs** (nearly 1:1), i.e., classic “single-item inputs that should be batched”.

### 2.2 Example of spillover past month-end
For December close (close_month=`2025-12`), `RPT_POR002` ran **190 times** and all of them occurred on **day 5** (Jan 5). That is exactly the “20th→EOM” target gap.

---

## 3) Recommended repo changes (what to tell Codex CLI)

### P0 — Implement wildcard/pattern exclusions (glob + SQL-like)
**Goal:** A single exclude list should support:
- Prefix patterns: `LOS*`
- SQL-like patterns: `JEPOST%`
- Exact names: `POSTWKFL`
- Optional regex: `re:^JEPOST.*` (nice-to-have)

**Implementation**
1. Add a shared helper under `src/statistic_harness/core/` (new module `process_matcher.py`):
   - `compile_patterns(patterns: list[str]) -> callable(str)->bool`
   - Convert SQL `%` → glob `*`, `_` → `?`
   - Case-insensitive matching
2. Update all exclude applications to use this helper (examples):
    - `analysis_queue_delay_decomposition` (currently uses `.isin(exclude_processes)`)
    - `analysis_close_cycle_duration_shift`
    - `analysis_actionable_ops_levers_v1` (and any other place filtering processes)
   - `report_v2_utils._explicit_excluded_processes()` and `_recommendation_has_excluded_process()`

**Defaults**
Set default exclude patterns (can be overridden by settings/env):
- `LOS*`
- `POSTWKFL` (and `POSTWJFL` alias)
- `BKRV*`
- `JEPOST*` (and/or `JEPOST%`)

**Tests**
- Unit tests for pattern matching edge cases (`LOS*`, `JEPOST%`, case-insensitive).
- Regression test: when exclusions include `LOS*`, recommendations do not target LOS jobs.

---

### P0 — Add a “target close window” concept (baseline vs target) and compute spillover
**Goal:** keep baseline close analysis (e.g., 20th→5th) **but** introduce a *target* close end day (EOM/31). Then compute:
- spillover job count on days 1–5
- spillover runtime / queue-wait totals
- top spillover processes (excluding excluded patterns)

**Implementation approach**
1. Add config fields to close-cycle plugins:
   - `baseline_close_start_day` (default 20)
   - `baseline_close_end_day` (default 5)
   - `target_close_end_day` (default 31; interpret as EOM)
2. Add a helper:
   - `close_month = month(queue_dt) - 1 if day<=baseline_end else month(queue_dt)`
3. For each close_month cohort:
   - baseline window mask = day>=baseline_start OR day<=baseline_end
   - target window mask = day>=baseline_start AND day<=target_end (no wrap)
   - spillover = baseline_mask AND NOT target_mask
4. Emit a new finding (or extend an existing close-cycle finding) with:
   - total spillover jobs/hours
   - top spillover processes/hours
   - spillover-by-day histogram (optional)

**Where to surface**
- Include in `analysis_close_cycle_capacity_model` and/or `analysis_close_cycle_duration_shift` output JSON.
- Have the report include “Spillover past EOM” as a top-level metric.

**Tests**
- Synthetic dataset: jobs on day 2 should count as spillover when target_end=31.
- Ensure excluded patterns are removed before computing spillover totals.

---

### P0 — Rewrite actionable ops lever batching logic to detect parameter sweeps (multi-input candidates)
**Goal:** automatically detect “batch/multi-input” opportunities like the payout pipeline **without hardcoding process names**.

**Core algorithm (generic)**
For each process `P` in (baseline close window) and not excluded:
1. Parse params via `parameter_kv` (not raw strings).
2. For each key `K`:
   - `runs_with_key`
   - `unique_values`
   - `unique_ratio = unique_values / runs_with_key`
3. Prefer keys that:
   - appear in most runs (`coverage >= 0.9`)
   - have high unique ratio in baseline close window (`unique_ratio >= 0.8`)
   - are not in a key blacklist (`*Run ID*`, `*Queue ID*`, `*Seq*`, `Slice*`, etc.)
4. Score by “batch payoff”:
   - compute close_month cohorts; within each cohort compute runs and unique values for `K`
   - estimate calls reducible as `sum(max(0, runs - cohort_count))` (cohort_count often 1 if other keys constant)
5. Emit `action_type: "batch_input"` recommendation with:
   - `process`
   - `key`
   - `best_close_month` / `runs` / `unique_values`
   - sample values (first 5–10)
   - suggested change: “modify job to accept a list of K values” (or “use MULTIPLE mode” if present)

**Where to implement**
- Extend `analysis_actionable_ops_levers_v1` handler (in core plugin registry) to add a second batching detector path:
  - existing: `batch_or_cache` (repeat-key/caching style)
  - new: `batch_input` (sweep-key batching style)

**Expected effect**
- The payout pipeline processes will be detected because `Payout ID` is near 1:1 with runs inside close-month cohorts.

**Tests**
- Add a small fixture dataset (10–30 rows) with a single process repeated over many unique `Payout ID` values and constant other keys; assert a `batch_input` rec is emitted.

---

### P0 — Fix wrap-around close day logic in SQL helpers
**Goal:** any SQL that filters by close day range must support wrap-around windows.

**Implementation**
- Replace `BETWEEN start_day AND end_day` with wrap-aware clauses when `start > end`.
- Create a reusable helper: `sql_day_of_month_window(col, start_day, end_day) -> (sql, params)`.

---

### P1 — De-genericize recommendation output (prioritization + suppression knobs)
**Goal:** reduce “non-starter” recommendations without losing the good ones (QEMAIL scheduling + processor+1).

**Implementation**
1. Add a scoring tier:
   - Tier 1: structural changes (`batch_input`, `batch_or_cache`, `dedupe_reruns`, “accept multi-input”)
   - Tier 2: targeted schedule changes (only when concentrated in spillover days or top driver)
   - Tier 3: generic (“make faster”, “reduce wait”) — suppress unless modeled savings is very high
2. Add config knobs to recommendation stage:
   - `suppress_action_types: []`
   - `max_per_action_type: { "batch_input": 5, "batch_or_cache": 5, "schedule_shift": 3 }`

---

## 4) Additional plugin ideas worth adding (optional but aligned with your goals)

These are **not** generic “make faster” plugins; they discover structural change opportunities:

1. **Duplicate rerun detector**
   - Detect same `(process, normalized_param_entity_id)` repeated within short windows.
   - Recommendation: dedupe, cache, or avoid re-running identical work units.

2. **Fan-out chain batching**
   - Identify A→B→C chains repeated for the same key set (e.g., payout id).
   - Recommend adding a “batch driver” that accepts N keys and runs the chain internally once.

3. **Parameterized loop clustering**
   - Cluster runs by high similarity of parameter sets.
   - For each cluster, identify minimal varying keys ⇒ batch candidate.

---

## Appendix A — Top batch-input candidates found in the close window (after simple key/process filtering)

These are examples of what the improved lever would surface automatically:

| process    | key                  |   runs_with_key |   unique_values |   unique_ratio |
|:-----------|:---------------------|----------------:|----------------:|---------------:|
| WRKSPC     | Workspace Group      |            1372 |            1353 |       0.986152 |
| DOINTXWRK  | Workspace Group      |            1299 |            1253 |       0.964588 |
| DVDVERIF   | Workspace Group      |            1288 |            1249 |       0.96972  |
| COPY_DVD_W | Base URL             |            1233 |            1204 |       0.97648  |
| COPY_DVD_W | Workspace Group      |            1233 |            1204 |       0.97648  |
| RPT_POR002 | Payout ID            |             651 |             542 |       0.832565 |
| RPT_POR002 | Report Email Subject |             651 |             539 |       0.827957 |
| POEXTRPRVN | Payout ID            |             612 |             535 |       0.874183 |
| POEXTRPEXP | Payout ID            |             609 |             532 |       0.873563 |
| POGNRTRPT  | Payout ID            |             608 |             533 |       0.876645 |
| DOI_COPY   | Job No               |             572 |             572 |       1        |
| CIPROC     | Check Group          |             524 |             479 |       0.914122 |
| GRPDEL     | Workspace Group      |             152 |             152 |       1        |
| UNDOPROC   | Revenue Run          |             146 |             138 |       0.945205 |
| ADPIMGVALD | SYS_PROCESS_QUEUE_ID |             104 |             100 |       0.961538 |
