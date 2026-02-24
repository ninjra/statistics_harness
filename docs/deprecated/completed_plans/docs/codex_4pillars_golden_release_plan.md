# Codex Implementation File — Golden First Release (4 Pillars Optimized)

**Generated:** 2026-02-14 (America/Denver)  
**Target machine:** Windows 11 host, WSL2 required for all Python execution; 64GB RAM, RTX 4090 available for CUDA.

This is the **single markdown file** you can feed to **Codex CLI** to implement the remaining work for a “golden first release”, optimized for this repository’s **4 Pillars**.

It consolidates (and slightly updates) the repo’s existing implementation plans + audits:
- `two-million-rows-support-plan.md`
- `local-sqlcoder-sql-assist-plan.md`
- `docs/erp-knowledge-bank-4-pillars-optimization-plan.md`
- `docs/stat_redteam2-6-26.md`
- `docs/codex_statistics_harness_blueprint.md`
- plus the **source-of-truth** spec: `docs/4_pillars_scoring_spec.md`

---

## Confirmed constraints (from you)

1. **CLI only** (no UI).
2. **Offline / local-only forever**. Network should be blocked **except** loopback access to `localhost:8000` for local model access.
3. You’re **not distributing** this to other machines/people.
4. Datasets can be **2,000,000+ rows** (maybe more). Must support **dynamic streaming/batching to plugins**. Plugins need to be modified to work with the streaming contract.
5. **WSL2 required and assumed**.
6. Local model: prefer a model available at `localhost:8000` (otherwise CPU is acceptable).
7. Keep **all** analysis/plugin capability families.
8. “Whatever I recommend” for risk posture as long as it improves the 4 pillars.
9. No PII concerns (log data), but data is proprietary.
10. NAS exists for cold storage but is slow; prefer local SSD-first.
11. Single operator; audit trails primarily as **performance records** for iterative optimization.
12. Dataset will **not** be tagged/shared; treat dataset contents as proprietary.

---

## Source-of-truth: the 4 Pillars

**Do not invent pillars.** Use the repo’s spec as source-of-truth: `docs/4_pillars_scoring_spec.md` (Evidence: `docs/4_pillars_scoring_spec.md` L1-L43).

---

## Golden Release Definition of Done (DoD)

### Must-pass quality gates
- `python -m pytest -q` passes in WSL2 (no skipped tests unless explicitly justified).
- `stat-harness list-plugins` works (plugin discovery stable).
- `scripts/run_gauntlet.sh` passes (or the Windows-safe equivalent if provided).
- Running a full pipeline on a realistically large dataset (>=2M rows) does **not**:
  - OOM,
  - hang indefinitely without progress visibility,
  - silently truncate results,
  - produce non-deterministic outputs across runs with the same seed and same dataset DB state.

### 4 Pillars “golden release” targets (pragmatic)
These are **targets**, not hard requirements, but Codex should bias changes toward them:
- **Performant**: safe-by-default on >=2M rows via streaming; avoid repeated full SQLite scans; avoid full-DF loads; concurrency auto-tuned to avoid RAM spikes.
- **Accurate**: no row sampling; complexity budgets limit *algorithmic blowups* instead (max cols/pairs/groups/windows), while still scanning all rows via batches.
- **Secure**: keep sandbox posture; default no network; allow only loopback `localhost` when explicitly configured.
- **Citable**: every report finding should have provenance (plugin id/version/hash, dataset version/hash, query/params if SQL, rowcounts, deterministic ordering, stable artifact writes).

---

## P0 priorities for “golden first release” (implement in this order)

### P0.1 — Localhost-only network mode (delta update)
**Problem:** Current sandbox network guard is “all-or-nothing” and blocks even loopback sockets.  
**Evidence:** `_install_network_guard()` hard-blocks socket creation/connection in `src/statistic_harness/core/plugin_runner.py` (Evidence: `src/statistic_harness/core/plugin_runner.py` L23-L104).

**Change:**
- Replace the boolean allow/deny model with a **3-mode** network policy:
  - `off` (default): block all sockets (current behavior).
  - `localhost`: allow only loopback destinations (`127.0.0.1`, `::1`, `localhost`) — **this is what you want** to reach `localhost:8000`.
  - `on`: allow any network (still available for rare debugging, but never default).
- Suggested env var: `STAT_HARNESS_NETWORK_MODE=off|localhost|on`.
  - Keep `STAT_HARNESS_ALLOW_NETWORK` as backward-compat: if set truthy, treat as `on`.

**Acceptance:**
- With `STAT_HARNESS_NETWORK_MODE=off`: `socket.create_connection(("127.0.0.1", 8000))` raises `RuntimeError` from the guard.
- With `STAT_HARNESS_NETWORK_MODE=localhost`: connecting to `("127.0.0.1", 8000)` yields a normal OS error (e.g., `ConnectionRefusedError`) if no server is running, **not** a guard error; connecting to `("8.8.8.8", 53)` raises guard `RuntimeError`.
- With `STAT_HARNESS_NETWORK_MODE=on`: guard does not interfere.

**Where to implement:**
- `src/statistic_harness/core/plugin_runner.py` network guard.

**Tests to add:**
- `tests/test_network_guard.py` (new).

---

### P0.2 — Enforce streaming-first data access on large datasets
**Problem:** The harness still permits full DataFrame loads for “typical large” datasets by default, and most plugins use `ctx.dataset_loader()` without streaming.
- `DatasetAccessor.load()` currently allows full DF loads up to a default `max_rows=3_000_000` unless env overrides (Evidence: `src/statistic_harness/core/dataset_io.py` L95-L123).
- Baseline plugin access patterns show 0 plugins using streaming today (Evidence: `local-sqlcoder-sql-assist-plan.md` L15-L20).

**Change (golden release posture):**
- Make the default “no full-DF load” cutoff **much lower** (recommend: 1,000,000 rows), and treat >=1M as “large dataset” that **must stream** unless explicitly overridden.
- Enforce *row-unbounded* loads to go through `iter_batches()`:
  - If `row_limit is None` and dataset is “large”, `DatasetAccessor.load()` should raise a clear error unless `STAT_HARNESS_ALLOW_FULL_DF=1`.
  - Ensure the error message points plugin authors to `ctx.dataset_iter_batches()`.

**Important:** This is aligned with your “no row sampling” rule; streaming still scans all rows.

**Where to implement:**
- `src/statistic_harness/core/dataset_io.py` (primary enforcement point; affects all loaders).
- Optionally strengthen the error message in `src/statistic_harness/core/plugin_runner.py` where `dataset_loader()` is exposed (Evidence: `src/statistic_harness/core/plugin_runner.py` L818-L842).

**Acceptance:**
- For dataset rows >= 2,000,000, any plugin calling `ctx.dataset_loader()` without `row_limit` fails closed with actionable guidance.
- Pipeline still succeeds because plugins are migrated or skip gracefully under policy (see next section).

---

### P0.3 — Plugin migration: “streaming or skip-with-reason” for large datasets
**Problem:** A large fraction of plugins are currently “full-DF” and will be unsafe at scale.
Baseline (Evidence: `local-sqlcoder-sql-assist-plan.md` L15-L20):
- Total plugins: 139
- Unbounded dataset_loader(): 121
- dataset_iter_batches(): 0
- direct SQL today: 3 (`ingest_tabular`, `profile_eventlog`, `transform_normalize_mixed`)

**Change:**
- Implement the streaming contract and then migrate plugins in waves:
  1. **Profile** plugins first (`profile_basic`, `profile_eventlog`) — they are upstream dependencies for many analyses.
  2. Next migrate top “worst offenders” by RSS + duration (use `plugin_executions.max_rss` + `duration_ms`).
  3. For algorithms that are fundamentally non-streamable in a reasonable time, implement a deterministic **skip-with-reason** path under large dataset policy (still deterministic, still citable).

**Where to implement:**
- Use the work plan in `two-million-rows-support-plan.md` (included below).
- Use `scripts/plugin_data_access_matrix.py` + `docs/plugin_data_access_matrix.*` as the ground truth driver.

**Acceptance:**
- On >=2M rows, no plugin OOMs.
- The pipeline produces a report even if some plugins are skipped; skips must be explicit and citable (reason + thresholds).

---

### P0.4 — 4 Pillars telemetry + scoring integrated into the report
**Goal:** You need a measurable scorecard so each iteration proves improvement.

**Where to implement:** follow `docs/erp-knowledge-bank-4-pillars-optimization-plan.md` (included below).  
That plan is already aligned to:
- deterministic scoring (0.0–4.0),
- computed from run telemetry,
- no dependence on external systems.

**Acceptance:**
- Every run produces a pillar scorecard artifact (machine + human readable) that can be diffed between runs.

---

## P1 priorities (after P0 is complete)
- Dataset cache materialization + cache-first streaming path (if not already used in hot paths).
- Auto-tuned concurrency and run ETA improvements.
- SQL-assist + “generate once, replay forever” SQL packs for citable, deterministic query-driven plugins (see `local-sqlcoder-sql-assist-plan.md`).
- Expand plugin library (changepoint detection, process mining, causal inference) per `docs/codex_statistics_harness_blueprint.md`.

---

## P2 priorities (maturity)
- Knowledge bank + long-horizon learning loop.
- Advanced recommendation/evaluation harness expansions.
- Optional GPU-accelerated numeric paths where deterministic (careful with nondeterminism).

---

## Appendix A — README Quickstart (repo evidence)

````markdown
 1: # Statistic Harness
 2: 
 3: Local-only, plugin-first statistical analysis harness.
 4: 
 5: ## Quickstart
 6: 
 7: WSL/Windows friendly setup (includes dev deps):
 8: 
 9: PowerShell:
10: 
11: ```powershell
12: python -m venv .venv
13: .\.venv\Scripts\Activate.ps1
14: .\scripts\install_dev.ps1
15: stat-harness list-plugins
16: ```
17: 
18: ```bash
19: python -m venv .venv
20: . .venv/bin/activate
21: ./scripts/install_dev.sh
22: stat-harness list-plugins
23: ```
24: 
25: Or, from WSL, just run:
26: 
27: ```bash
28: make dev
29: ```
30: 
31: ## Development
32: 
33: Run tests:
34: 
35: ```bash
36: python -m pytest -q
37: ```
38: 
39: ## Feature Flags / Env
40: 
41: Default behavior is local-only. Optional features are guarded by env flags:
42: 
43: - `STAT_HARNESS_ENABLE_AUTH=1` enables UI/API auth (sessions + API keys).
44: - `STAT_HARNESS_ENABLE_TENANCY=1` enables tenant-aware isolation.
45: - `STAT_HARNESS_ENABLE_VECTOR_STORE=1` enables the sqlite-vec vector store.
46: - `STAT_HARNESS_SQLITE_VEC_PATH=/path/to/vec0.so` loads sqlite-vec if not builtin.
47: - `STAT_HARNESS_MAX_UPLOAD_BYTES` enforces upload size limits.
48: 
49: PII handling: `profile_basic` tags likely PII columns and stores hashed entities.
50: Any LLM/offsystem payloads generated by the prompt builder are anonymized using
51: the PII entity hash table.
````

---

## Appendix B — 4 Pillars Scoring Spec (source-of-truth)

````markdown
 1: # 4 Pillars Scoring Spec (0.0-4.0)
 2: 
 3: ## Purpose
 4: Define a deterministic, balanced scoring model for the four pillars:
 5: - `performant`
 6: - `accurate`
 7: - `secure`
 8: - `citable`
 9: 
10: The scorecard is computed from full run telemetry and report outputs, not from Kona/ideaspace alone.
11: 
12: ## Scale
13: - Every pillar score is clamped to `0.0..4.0`.
14: - Overall score is also on `0.0..4.0`.
15: 
16: ## Inputs
17: - Run telemetry (runtime, max RSS, plugin execution status)
18: - Report findings (measurement tags, evidence coverage)
19: - Recommendation quality (known-issue confirmation + modeled coverage)
20: - Traceability and reproducibility lineage fields
21: - Security/policy findings and fail-closed indicators
22: 
23: ## Balance Policy
24: - Objective: maximize the weakest pillar while keeping pillar spread small.
25: - Constraints:
26:   - `min_floor`: minimum allowed pillar score before veto.
27:   - `max_spread`: maximum allowed `max(pillar)-min(pillar)` before veto.
28:   - `degradation_tolerance`: high one-pillar gains cannot compensate for a weak pillar.
29: - If a constraint is violated, result status is vetoed and the veto reason is recorded.
30: 
31: ## Output Contract
32: - `pillars.<name>.score_0_4`
33: - `pillars.<name>.components`
34: - `pillars.<name>.rationale`
35: - `balance`:
36:   - `min_pillar`, `max_pillar`, `spread`
37:   - `balance_index_0_4`, `balanced_score_0_4`
38:   - `vetoes[]`, `status`
39: - `summary.overall_0_4`
40: 
41: ## Determinism
42: - No randomness is used in scoring.
43: - Same report payload and run telemetry must always produce the same scorecard.
````

---

## Appendix C — Two Million Rows Support Plan (streaming-first + large dataset policy)

````markdown
  1: # Plan: Support ~2,000,000 Row Datasets (Harness-First, 4 Pillars)
  2: 
  3: **Generated**: 2026-02-08  
  4: **Estimated Complexity**: High
  5: 
  6: ## Overview
  7: Most plugins currently call `ctx.dataset_loader()` which loads a pandas DataFrame (optionally limited by `budget.row_limit`). At ~2,000,000 rows, naive full-DataFrame loads per plugin are too slow and risk OOM, especially with parallel analysis.
  8: 
  9: This plan makes the harness scale to ~2M rows while optimizing for the 4 pillars:
 10: - **Performant**: budget caps, streaming, caching, and safe concurrency.
 11: - **Accurate**: full-dataset scans via batching/streaming; deterministic sketches/diagnostics where needed.
 12: - **Secure**: keep sandbox/no-network, avoid new attack surface (no shelling out), keep artifacts under `appdata/`.
 13: - **Citable**: preserve references; ensure report captures budget and provenance.
 14: 
 15: Key strategy:
 16: 1. **Streaming first**: add first-class batch APIs to `PluginContext` and migrate plugins off full-DataFrame loads.
 17: 2. **Policy-driven complexity budgets (not row sampling)**: for large datasets, bound algorithmic complexity (max columns/pairs/groups/windows) while still scanning all rows in batches.
 18: 3. **One-time dataset tuning**: indexes (already done for `row_index`) + optional local caches.
 19: 4. **Observability/ETA**: record per-plugin durations/RSS, estimate time-to-completion, and auto-tune concurrency.
 20: 
 21: ## Prerequisites
 22: - No runtime network calls (existing sandbox guard must remain).
 23: - Python 3.11+.
 24: - Must keep plugin IDs and plugin discovery/manifest behavior stable (no changing which plugins load; only harness behavior).
 25: - “Do not ship unless” gate: `python -m pytest -q` must pass.
 26: 
 27: ## Locked Requirement (From You)
 28: - **No row sampling**. Plugins must support large datasets by intelligently **batching/streaming and scanning** (multi-pass ok). Long runtimes (even ~38 hours) are acceptable as long as results are actionable and specific.
 29: 
 30: Notes:
 31: - Streaming sketches (quantiles/heavy hitters/entropy) are acceptable if they scan all rows and are deterministic; they are not “row sampling”.
 32: 
 33: ## Sprint 1: Streaming Contract + Safety Gates
 34: **Goal**: Make 2M-row runs safe by default: no accidental full-DF loads, deterministic behavior, and explicit streaming APIs, without changing plugin IDs.
 35: 
 36: **Demo/Validation**:
 37: - Run a “large dataset policy” integration test that fakes row_count >= 2,000,000 and asserts budgets/timeouts are applied.
 38: - `bash scripts/run_gauntlet.sh`
 39: 
 40: ### Task 1.0: Generate Plugin Data-Access Matrix (Avoid Duplicate Work)
 41: - **Location**: new `scripts/plugin_data_access_matrix.py`, outputs `docs/plugin_data_access_matrix.md` + `docs/plugin_data_access_matrix.json`
 42: - **Description**: Generate and keep updated a matrix that records, per plugin:
 43:   - whether it calls `ctx.dataset_loader()` unbounded
 44:   - whether it supports streaming via `ctx.dataset_iter_batches()`
 45:   - primary access pattern (full DF, batched, SQL aggregation, cached)
 46:   - migration status: `not_started|in_progress|streaming_ok`
 47: - **Complexity**: 5/10
 48: - **Dependencies**: None
 49: - **Acceptance Criteria**:
 50:   - Matrix includes all plugin IDs under `plugins/`.
 51:   - Matrix is generated deterministically and is used to drive subsequent migration work.
 52: - **Validation**:
 53:   - Unit test: matrix generation includes all plugins and has stable ordering.
 54: 
 55: ### Task 1.1: Add Large-Dataset Policy Module (Complexity Budgets)
 56: - **Location**: `src/statistic_harness/core/large_dataset_policy.py`
 57: - **Description**: Implement a deterministic policy that decides complexity/resource caps (not row sampling) based on:
 58:   - dataset row_count/column_count
 59:   - plugin type (`profile|planner|transform|analysis|report|llm`)
 60:   - plugin_id patterns (e.g., `analysis_matrix_profile_*`, `analysis_*prefixspan*`, `analysis_*lda*`)
 61:   - optional user overrides from env or a policy file.
 62:   Output should be a dict merged into plugin settings and/or `ctx.budget`, e.g.:
 63:   - `budget.time_limit_ms`, `budget.cpu_limit_ms`
 64:   - `budget.batch_size`
 65:   - `budget.max_cols`, `budget.max_pairs`, `budget.max_groups`, `budget.max_windows`, `budget.max_findings`
 66: - **Complexity**: 6/10
 67: - **Dependencies**: None
 68: - **Acceptance Criteria**:
 69:   - Policy is deterministic given same inputs.
 70:   - Given `row_count>=2_000_000`, policy provides non-null `batch_size` and complexity caps for heavy plugins.
 71: - **Validation**:
 72:   - Unit tests for policy decisions.
 73: 
 74: ### Task 1.2: Apply Policy in Pipeline Before Subprocess Execution
 75: - **Location**: `src/statistic_harness/core/pipeline.py`
 76: - **Description**: In `run_spec()`, merge computed policy budget into plugin config:
 77:   - If plugin config explicitly sets a cap, keep it as an override.
 78:   - Otherwise, inject policy caps.
 79:   - Ensure `budget.sampled` remains `false` by default.
 80: - **Complexity**: 5/10
 81: - **Dependencies**: Task 1.1
 82: - **Acceptance Criteria**:
 83:   - For large datasets, plugins have access to `budget.batch_size` + complexity caps.
 84:   - Subprocess timeout uses `budget.time_limit_ms`.
 85: - **Validation**:
 86:   - Integration test with a tiny fixture but mocked dataset metadata.
 87: 
 88: ### Task 1.3: Enforce “No Full-DF Load” For Large Datasets
 89: - **Location**: `src/statistic_harness/core/dataset_io.py`, `src/statistic_harness/core/plugin_runner.py`
 90: - **Description**: For datasets above a threshold, fail closed if a plugin attempts `ctx.dataset_loader()` without a `row_limit`, with an actionable error telling plugin authors/operators to use `ctx.dataset_iter_batches()` (or explicitly override to allow full loads).
 91: - **Complexity**: 4/10
 92: - **Dependencies**: Task 1.2
 93: - **Acceptance Criteria**:
 94:   - On 2M rows, unbounded full-DF loads fail closed with actionable guidance to use batch iteration.
 95: - **Validation**:
 96:   - Unit test that large row_count triggers the refusal unless explicitly allowed.
 97: 
 98: ## Sprint 2: Streaming API in PluginContext + First Migrations
 99: **Goal**: Enable full-dataset correctness where feasible by streaming batches, and reduce reliance on full in-memory DataFrames.
100: 
101: **Demo/Validation**:
102: - Add one streaming-first plugin path (e.g., `profile_basic` or a simple analysis) and verify outputs remain deterministic.
103: - `bash scripts/run_gauntlet.sh`
104: 
105: ### Task 2.1: Add `dataset_iter_batches()` to PluginContext
106: - **Location**: `src/statistic_harness/core/types.py`, `src/statistic_harness/core/plugin_runner.py`, `src/statistic_harness/core/dataset_io.py`
107: - **Description**:
108:   - Extend `PluginContext` with `dataset_iter_batches(columns=None, batch_size=..., row_limit=None)`.
109:   - Implement it in `plugin_runner.py` to call `DatasetAccessor.iter_batches()`, defaulting `batch_size` from `budget.batch_size`.
110: - **Complexity**: 5/10
111: - **Dependencies**: Sprint 1
112: - **Acceptance Criteria**:
113:   - Existing plugins unchanged (backwards compatible).
114:   - New/updated plugins can stream data deterministically.
115: - **Validation**:
116:   - Unit test that iter_batches yields stable ordering and respects `row_limit`.
117: 
118: ### Task 2.2: Streaming Profile for Large Datasets
119: - **Location**: `plugins/profile_basic/plugin.py`, `plugins/profile_eventlog/plugin.py`
120: - **Description**: For large datasets, compute core profile stats via batches (counts, null rates, type inference via bounded scans, basic percentiles via deterministic streaming sketches or two-pass bounded approaches).
121: - **Complexity**: 7/10
122: - **Dependencies**: Task 2.1
123: - **Acceptance Criteria**:
124:   - Profile plugins do not require loading the full DataFrame for 2M rows.
125:   - Output remains schema-valid and deterministic.
126: - **Validation**:
127:   - Unit tests with synthetic larger-than-threshold row_count (small actual data ok) verifying streaming path is chosen.
128: 
129: ### Task 2.3: Identify and Migrate Top “Worst Offenders”
130: - **Location**: `plugins/*/plugin.py`, new doc `docs/perf_hotspots.md`
131: - **Description**:
132:   - Use `plugin_executions.max_rss` and `duration_ms` to rank slowest plugins on real runs.
133:   - For the top 10, either:
134:     - implement streaming algorithm; or
135:     - replace full-DF algorithms with streaming/multi-pass alternatives; if infeasible, mark `skipped` with a specific reason and a concrete follow-up recommendation.
136: - **Complexity**: 8/10
137: - **Dependencies**: Sprint 1, Task 2.1
138: - **Acceptance Criteria**:
139:   - No single plugin should OOM or run unbounded by default on 2M rows.
140: - **Validation**:
141:   - Integration run on a large-ish synthetic dataset (or downscaled) verifying no plugin attempts a full load.
142: 
143: ## Sprint 3: Dataset Materialization Cache (Optional but High Leverage)
144: **Goal**: Avoid repeated SQLite scans across many plugin subprocesses.
145: 
146: **Demo/Validation**:
147: - Create a cache for the dataset once; subsequent plugin loads should be faster.
148: - `bash scripts/run_gauntlet.sh`
149: 
150: ### Task 3.1: Add Columnar Cache Format (No New Heavy Deps)
151: - **Location**: `src/statistic_harness/core/dataset_cache.py` (new), `src/statistic_harness/core/dataset_io.py`
152: - **Description**:
153:   - Implement a cache keyed by `dataset_version_id` + `data_hash` + schema hash.
154:   - Store columns as numpy `.npy` (optionally memory-mapped) under `appdata/cache/datasets/<key>/`.
155:   - `DatasetAccessor.iter_batches(...)` should prefer the cache if present; otherwise scan SQLite once and populate cache (optional knob) while streaming out batches.
156:   - `DatasetAccessor.load(...)` may use cache for small datasets, but for large datasets we will enforce streaming APIs instead of full loads.
157: - **Complexity**: 8/10
158: - **Dependencies**: Sprint 1
159: - **Acceptance Criteria**:
160:   - Cache is deterministic and safe to reuse.
161:   - Cache writes stay under `appdata/` and are traversal-safe.
162: - **Validation**:
163:   - Unit tests for keying, cache hit/miss behavior.
164: 
165: ### Task 3.2: CLI Helpers to Build/Inspect Cache
166: - **Location**: `scripts/materialize_dataset_cache.py`, `scripts/inspect_dataset_cache.py`
167: - **Description**: Provide one-line operator tooling to precompute caches for a dataset version.
168: - **Complexity**: 4/10
169: - **Dependencies**: Task 3.1
170: - **Acceptance Criteria**:
171:   - Running materialization twice is idempotent.
172: - **Validation**:
173:   - Smoke test in CI (small dataset).
174: 
175: ## Sprint 4: Auto-Tuned Concurrency + Better ETA
176: **Goal**: Use observed resource usage to safely increase throughput on large datasets without OOM.
177: 
178: **Demo/Validation**:
179: - Run with `STAT_HARNESS_MAX_WORKERS_ANALYSIS=auto` and confirm it selects a safe worker count.
180: 
181: ### Task 4.1: Persist/Use Execution Telemetry for Scheduling
182: - **Location**: `src/statistic_harness/core/pipeline.py`, `src/statistic_harness/core/storage.py`
183: - **Description**:
184:   - Use recorded `max_rss`/`duration_ms` to:
185:     - compute a safe concurrency for future layers,
186:     - optionally serialize known memory hogs even when others can parallelize.
187: - **Complexity**: 7/10
188: - **Dependencies**: Sprint 1
189: - **Acceptance Criteria**:
190:   - No layer starts more workers than budgeted.
191:   - Memory hog plugins do not run concurrently by default.
192: - **Validation**:
193:   - Unit test with synthetic telemetry driving scheduler decisions.
194: 
195: ### Task 4.2: Improve Run ETA Reporting
196: - **Location**: `scripts/run_run_status.py`
197: - **Description**:
198:   - Estimate ETA from median/EMA of completed plugin durations, not “plugins/sec since start”.
199:   - Report `expected_plugins_executable`, `done`, `running`, and `eta_minutes`.
200: - **Complexity**: 4/10
201: - **Dependencies**: None
202: - **Acceptance Criteria**:
203:   - ETA stabilizes after ~10 plugins.
204: - **Validation**:
205:   - Unit test on a seeded fake timeline.
206: 
207: ## Testing Strategy
208: - Unit tests:
209:   - Large-dataset policy decisions, deterministic keys, cache behavior.
210:   - Streaming batch ordering/limits.
211:   - Scheduler decisions from telemetry.
212: - Integration tests:
213:   - Full pipeline on fixture dataset still produces `report.md` and `report.json`.
214:   - Large-dataset policy test: simulate `row_count>=2_000_000` via DB row metadata and assert no unbounded full-DF loads occur by default.
215: - Always gate with `bash scripts/run_gauntlet.sh`.
216: 
217: ## Potential Risks & Gotchas
218: - “Run all plugins against the entire dataset” conflicts with feasibility for O(n²) methods.
219:   - Mitigation: streaming/multi-pass implementations + strict complexity caps (pairs/groups/windows) that still scan all rows; if infeasible, `skipped` with an actionable recommendation.
220: - Cache correctness:
221:   - Mitigation: strict keying by data_hash + schema; include SHA256 in manifest; never trust external paths.
222: - SQLite locking/IO contention with higher parallelism:
223:   - Mitigation: keep WAL, add indexes, cap workers, optionally stagger heavy readers.
224: - Determinism vs speed:
225:   - Mitigation: deterministic batch ordering + deterministic sketches; deterministic ordering in reports.
226: 
227: ## Rollback Plan
228: - Keep all new behavior behind env/policy toggles:
229:   - `STAT_HARNESS_LARGE_DATASET_POLICY=off` to disable injected policy caps.
230:   - `STAT_HARNESS_DATASET_CACHE=off` to disable cache reads/writes.
231:   - `STAT_HARNESS_MAX_WORKERS_ANALYSIS` to force fixed parallelism.
232: - Revert to current behavior by disabling toggles; no schema-breaking changes.
````

---

## Appendix D — Local SQLCoder + SQL Assist Plan (citable, deterministic SQL packs)

````markdown
  1: # Plan: Local SQLCoder Plugin + SQL Assist Layer
  2: 
  3: **Generated**: 2026-02-10  
  4: **Estimated Complexity**: High
  5: 
  6: ## Overview
  7: We want all plugins to benefit from custom, dataset-specific SQL while preserving the 4 pillars:
  8: - **Performant**: avoid repeated full-table pandas loads; prefer SQLite-side projection/aggregation; enable streaming.
  9: - **Accurate**: scan full data; allow plugins to filter/aggregate without forcing downsampling; keep multi-pass OK.
 10: - **Secure**: no network; no shelling out; read-only SQL execution; strict allowlists.
 11: - **Citable**: every LLM-produced (or human-produced) query is persisted with schema hash, params, rowcounts, and a result hash; reports can point to the exact query+evidence.
 12: 
 13: This plan intentionally builds on `two-million-rows-support-plan.md` (streaming, large-dataset policy, caching). The SQL Assist layer complements batching by letting plugins move expensive groupby/join logic into SQLite and by enabling a “generate once, replay forever” SQL pack workflow.
 14: 
 15: ## Current Baseline (From `docs/plugin_data_access_matrix.*`)
 16: - Plugins total: 139
 17: - Plugins calling `ctx.dataset_loader()` (unbounded by default): 121
 18: - Plugins using `ctx.dataset_iter_batches()`: 0
 19: - Plugins using direct SQL today: 3 (`ingest_tabular`, `profile_eventlog`, `transform_normalize_mixed`)
 20: 
 21: ## Decisions (Confirmed)
 22: 1. **Local model**: Use SQLCoder2 (`defog/sqlcoder2`) via vLLM. Network access is allowed for model/dependency download.
 23: 2. **Safety**: Plugins must treat the **normalized layer as read-only**. Controlled materialization is allowed, but only into plugin-owned scratch tables (not into normalized tables).
 24: 3. **Citeability**: SQL must be “generate once, replay forever”:
 25:    - Persist generated SQL, schema snapshot/hash, decode config, params, rowcounts, and result hashes.
 26:    - Re-running on the same normalized DB state must reproduce identical results.
 27: 
 28: ## Implementation Matrix (Do This First, Avoid Duplicate Functions)
 29: | Component | Location(s) | Purpose | Output Artifacts | Tests/Validation |
 30: |---|---|---|---|---|
 31: | SQL execution API (read-only state DB) | `src/statistic_harness/core/sql_assist.py` (new), `src/statistic_harness/core/storage.py` | Safe, deterministic SQL execution helpers; reads normalized DB; captures provenance | `artifacts/<plugin>/sql/*.sql`, `*.json` result manifests | Unit tests for validator + provenance; plugin smoke tests |
 32: | Scratch DB for plugin-owned tables | `src/statistic_harness/core/plugin_runner.py`, `src/statistic_harness/core/storage.py` | Allow plugins to create their own tables/views without touching normalized tables | `run_dir/scratch.sqlite` + per-plugin manifests | Unit tests + integration “cannot write state DB” gate |
 33: | SQL statement validator | `src/statistic_harness/core/sql_assist.py` | Fail-closed: only allow safe statements; forbid multi-statement, DDL/DML, PRAGMA/ATTACH | Included in manifests | Unit tests (denylist/allowlist cases) |
 34: | Schema snapshot (normalized DB) | `src/statistic_harness/core/sql_schema_snapshot.py` (new) | Deterministic snapshot of tables/cols/indexes used for prompts and citations | `schema_snapshot.json` | Unit tests for stable ordering/hashes |
 35: | SQL pack format | `docs/sql_pack.schema.json` (new) | Define “generated once” SQL pack contract (queries, params, expected outputs) | `sql_pack.json` | JSON schema validation tests |
 36: | Prompt pack plugin (current) | `plugins/transform_sql_intents_pack_v1/*` | Writes schema snapshot + query intents used for text2sql generation | `schema_snapshot.json`, `sql_intents.json` | Integration: plugin runs, artifacts exist |
 37: | vLLM generator script (operator step) | `scripts/generate_sql_pack_with_vllm.py` | Runs local model once to convert prompt->SQL pack; stores pack deterministically | `sql_pack.json` | Smoke test if model present; skipped in CI if absent |
 38: | Optional in-pipeline SQL generation plugin | `plugins/llm_text2sql_local_generate_v1/*` | Runs vLLM locally during pipeline with strict gating | `artifacts/.../sql_pack.json` | Integration test behind env flag; deterministic decode config |
 39: | SQL pack materialization transform | `plugins/transform_sqlpack_materialize_v1/*` (new) | Executes safe subset of pack to produce derived views/tables for downstream plugins | Derived tables/views + manifest | Integration test: derived objects created; idempotent |
 40: | Plugin integration API | `src/statistic_harness/core/types.py`, `src/statistic_harness/core/plugin_runner.py` | Add `ctx.sql` helper and/or `ctx.sql_query(...)` for uniform access | N/A | Unit tests for ctx wiring |
 41: | Adoption tracking matrix | extend `scripts/plugin_data_access_matrix.py`, `scripts/sql_assist_adoption_matrix.py` | Track which plugins use SQL Assist and how | `docs/sql_assist_adoption_matrix.md/json` | Verify scripts in `scripts/verify_docs_and_plugin_matrices.py` |
 42: | Plain-English report (separate) | `plugins/report_plain_english_v1/*`, `src/statistic_harness/core/plain_report.py` | Render plugin outputs into non-technical English `plain_report.md` while keeping `report.json` technical | `plain_report.md` | Snapshot tests on fixture run |
 43: 
 44: ## Sprint 1: SQL Assist Core + Scratch DB (No LLM, No Plugin Refactors Yet)
 45: **Goal**: Provide safe, deterministic SQL access to the normalized dataset while enforcing “normalized DB is read-only” and enabling plugin-owned tables in a scratch DB.
 46: 
 47: **Demo/Validation**:
 48: - Unit tests pass: `python -m pytest -q`
 49: - Create a tiny “sql assist” fixture plugin that runs a query and emits a citeable finding.
 50: 
 51: ### Task 1.1: Add `SqlAssist` + Read-Only Validator
 52: - **Location**: `src/statistic_harness/core/sql_assist.py` (new)
 53: - **Description**:
 54:   - Implement `SqlAssist` with methods:
 55:     - `query_rows(sql: str, params: dict|tuple|list|None) -> list[sqlite3.Row]`
 56:     - `query_df(sql: str, params: ...) -> pandas.DataFrame` (optional)
 57:     - `explain_query_plan(sql, params) -> list[dict]`
 58:   - Enforce fail-closed validation:
 59:     - Allow only a single statement.
 60:     - Allow only `SELECT` / `WITH`.
 61:     - Forbid `;`, `PRAGMA`, `ATTACH`, `DETACH`, `VACUUM`, `INSERT`, `UPDATE`, `DELETE`, `DROP`, `CREATE`, `ALTER`.
 62:     - Forbid references to `sqlite_master` unless explicitly allowlisted (optional).
 63:   - Always capture provenance:
 64:     - normalized schema hash
 65:     - query text hash
 66:     - params JSON
 67:     - rowcount
 68:     - deterministic result hash (e.g., hash of `json.dumps(rows, sort_keys=True)` with stable ordering)
 69: - **Complexity**: 7/10
 70: - **Dependencies**: None
 71: - **Acceptance Criteria**:
 72:   - Safe validator blocks dangerous SQL.
 73:   - Helper stores a `sql_manifest.json` artifact for each executed query.
 74: - **Validation**:
 75:   - New unit tests for allow/deny cases and manifest determinism.
 76: 
 77: ### Task 1.2: Wire `SqlAssist` Into `PluginContext`
 78: - **Location**: `src/statistic_harness/core/types.py`, `src/statistic_harness/core/plugin_runner.py`
 79: - **Description**:
 80:   - Add a new `ctx.sql` attribute (or `ctx.sql_query(...)` function) that uses `ctx.storage.connection()` under the hood.
 81:   - Keep backwards compatibility with existing plugins.
 82: - **Complexity**: 4/10
 83: - **Dependencies**: Task 1.1
 84: - **Acceptance Criteria**:
 85:   - Plugins can call `ctx.sql.query_rows(...)` without reaching into `ctx.storage` directly.
 86: - **Validation**:
 87:   - Smoke test plugin + unit tests for ctx initialization.
 88: 
 89: ### Task 1.3: Enforce “State DB Read-Only” in Plugin Subprocesses
 90: - **Location**: `src/statistic_harness/core/storage.py`, `src/statistic_harness/core/plugin_runner.py`
 91: - **Description**:
 92:   - Add `Storage(read_only=True)` mode used only inside plugin subprocesses:
 93:     - Skip migrations.
 94:     - Open sqlite with `mode=ro` (URI form) and skip write-oriented PRAGMAs.
 95:   - Add a run-scoped scratch database for plugin-owned tables:
 96:     - Path: `appdata/runs/<run_id>/scratch.sqlite` (or `run_dir/scratch.sqlite`).
 97:     - Provide it as `ctx.scratch_storage` and `ctx.scratch_sql`.
 98:   - Update sqlite connect guard to allow only:
 99:     - read-only state DB path
100:     - scratch DB path
101: - **Complexity**: 7/10
102: - **Dependencies**: Task 1.1, Task 1.2
103: - **Acceptance Criteria**:
104:   - Attempts to write to the state DB from a plugin subprocess fail (hard).
105:   - Plugins can create/read their own tables in scratch DB.
106: - **Validation**:
107:   - Unit test: plugin tries `INSERT` into normalized table -> fails.
108:   - Integration test: plugin creates scratch table -> succeeds.
109: 
110: ### Task 1.4: Deterministic Schema Snapshot
111: - **Location**: `src/statistic_harness/core/sql_schema_snapshot.py` (new)
112: - **Description**:
113:   - Implement a stable snapshot of relevant DB objects:
114:     - normalized tables, columns, types
115:     - indexes
116:     - row counts where cheap
117:   - Return `schema_snapshot.json` and `schema_hash`.
118: - **Complexity**: 5/10
119: - **Dependencies**: None
120: - **Acceptance Criteria**:
121:   - Same DB state produces identical snapshot bytes/hash.
122: - **Validation**:
123:   - Unit test on a temporary sqlite DB.
124: 
125: ## Sprint 2: SQLCoder Prompt Pack (Compliant “LLM Prompt Builder Only” Path)
126: **Goal**: Create a plugin that outputs a schema-aware prompt pack so SQL can be generated once, stored, and replayed deterministically.
127: 
128: **Demo/Validation**:
129: - Run full harness; confirm prompt pack artifacts exist under the run.
130: 
131: ### Task 2.1: Define SQL Pack Schema
132: - **Location**: `docs/sql_pack.schema.json` (new)
133: - **Description**:
134:   - Define a strict format:
135:     - `schema_hash`
136:     - `dialect: "sqlite"`
137:     - `queries: [{id, purpose, sql, params_schema, expected_outputs}]`
138:     - `safety: {read_only: true}`
139:     - `generated_by: {tool, model, version, decode_config, created_at}`
140:   - Add validator utility used by materializer and tests.
141: - **Complexity**: 4/10
142: - **Dependencies**: Sprint 1
143: - **Acceptance Criteria**:
144:   - Schema rejects missing fields and multi-statement SQL.
145: - **Validation**:
146:   - JSON schema tests.
147: 
148: ### Task 2.2: Add `transform_sql_intents_pack_v1`
149: - **Location**: `plugins/transform_sql_intents_pack_v1/plugin.py`, `plugins/transform_sql_intents_pack_v1/plugin.yaml`, `src/statistic_harness/core/sql_intents.py`
150: - **Description**:
151:   - Capture:
152:     - schema snapshot
153:     - deterministic query intents (SQLite dialect, deterministic ordering)
154:   - Write:
155:     - `schema_snapshot.json`
156:     - `sql_intents.json`
157: - **Complexity**: 6/10
158: - **Dependencies**: Sprint 1 (schema snapshot)
159: - **Acceptance Criteria**:
160:   - Plugin runs with no model present.
161:   - Intents are deterministic for the same schema hash.
162: - **Validation**:
163:   - Integration test in `tests/plugins/`.
164: 
165: ### Task 2.3: Add Operator Script To Generate SQL Pack With vLLM
166: - **Location**: `scripts/generate_sql_pack_with_vllm.py` (new)
167: - **Description**:
168:   - Load model from `/mnt/d/autocapture/models/<model_name>/` (read-only).
169:   - Deterministic decode:
170:     - greedy (`temperature=0`, `top_p=1`, `top_k=-1` or equivalent)
171:     - fixed max tokens
172:   - Produce:
173:     - `appdata/sqlpacks/<schema_hash>/sql_pack.json`
174:     - `appdata/sqlpacks/<schema_hash>/generation_manifest.json` (model+config+prompt hash)
175: - **Complexity**: 7/10
176: - **Dependencies**: Task 2.1, Task 2.2
177: - **Acceptance Criteria**:
178:   - Re-running produces identical pack given same prompt/model/config.
179: - **Validation**:
180:   - Smoke test skipped unless model exists locally.
181: 
182: ## Sprint 3: Optional “In-Pipeline” Local SQLCoder Generation (Only If Allowed)
183: **Goal**: Fully automate SQL pack generation as part of the harness while keeping security/determinism.
184: 
185: **Demo/Validation**:
186: - Full run produces `sql_pack.json` without operator scripts when env flag is enabled.
187: 
188: ### Task 3.1: Add `llm_text2sql_local_generate_v1` Plugin (Gated)
189: - **Location**: `plugins/llm_text2sql_local_generate_v1/*`
190: - **Description**:
191:   - Hard gate behind `STAT_HARNESS_ENABLE_LOCAL_LLM_SQL=1`.
192:   - Requires model path configured (env var).
193:   - Writes pack + manifest into run artifacts and `appdata/sqlpacks/<schema_hash>/`.
194: - **Complexity**: 8/10
195: - **Dependencies**: Sprint 2
196: - **Acceptance Criteria**:
197:   - When disabled, plugin is `skipped` with a clear message.
198:   - When enabled, outputs are deterministic and schema-valid.
199: - **Validation**:
200:   - Integration tests gated by env var.
201: 
202: ## Sprint 4: “All Plugins Benefit” Integration (Refactor/Extension)
203: **Goal**: Make SQL Assist usable everywhere, and progressively migrate plugin families so they use SQL where it helps (projection/aggregation/joins), while keeping correctness and citeability.
204: 
205: **Demo/Validation**:
206: - Full harness run on the 500k dataset finishes and produces additional actionable findings attributable to SQL-powered pre-aggregation.
207: - `analysis_actionable_ops_levers_v1` produces findings with evidence pointing to SQL manifests and derived tables where used.
208: 
209: ### Task 4.1: Add Query Intents Registry (Central, Deterministic)
210: - **Location**: `src/statistic_harness/core/sql_intents.py` (new)
211: - **Description**:
212:   - Define a stable set of reusable intents, e.g.:
213:     - `eventlog_core_projection` (process/timestamps/durations/deps)
214:     - `per_process_wait_stats`
215:     - `per_process_hourly_medians`
216:     - `dependency_hotspots`
217:     - `top_variants_by_duration`
218:   - Each intent declares:
219:     - required normalized columns (semantic names)
220:     - acceptable fallbacks
221:     - output schema
222:   - SQLCoder prompt pack consumes these intents, not per-plugin ad-hoc prose.
223: - **Complexity**: 7/10
224: - **Dependencies**: Sprint 2
225: - **Acceptance Criteria**:
226:   - Intents cover at least the operational/actionable family (queue delay, conformance, ops levers, ideaspace).
227: - **Validation**:
228:   - Unit tests: intent resolution chooses correct normalized columns.
229: 
230: ### Task 4.2: Add SQL Pack Materializer Transform Plugin
231: - **Location**: `plugins/transform_sqlpack_materialize_v1/*`
232: - **Description**:
233:   - Validate pack against schema hash.
234:   - Execute allowlisted queries (read-only by default).
235:   - Optional “materialize” mode (if permitted):
236:     - write derived tables as `derived_<intent_id>_<schema_hash_prefix>`
237:     - record DDL + hashes in a manifest
238:   - Always emit citations:
239:     - `sql/<intent_id>.sql`
240:     - `sql/<intent_id>_manifest.json`
241:     - `sql/<intent_id>_sample.json` (small, bounded)
242: - **Complexity**: 8/10
243: - **Dependencies**: Sprint 1, Sprint 2
244: - **Acceptance Criteria**:
245:   - Idempotent: re-run doesn’t duplicate or drift.
246:   - Fail-closed if any referenced columns can’t be mapped; emits a “critical stop” marker that prevents downstream analysis stage.
247: - **Validation**:
248:   - Integration test: missing-column pack fails closed.
249: 
250: ### Task 4.3: Plugin Family Migrations (How “All Plugins” Benefit)
251: - **Location**: `src/statistic_harness/core/stat_plugins/registry.py`, `src/statistic_harness/core/stat_plugins/*`, selective plugin wrappers
252: - **Description**:
253:   - Add a standard pattern:
254:     - Prefer SQL core projections (narrow columns) before pandas-heavy work.
255:     - Use SQL aggregates when output is groupby/join based.
256:   - Targeted early wins (pilot set):
257:     - `analysis_actionable_ops_levers_v1`: compute `over_threshold` totals per process in SQL (faster + citeable).
258:     - `analysis_conformance_*`: compute alignments inputs (sequence extraction) in SQL.
259:     - `analysis_dependency_resolution_join`: do join logic in SQL and only load final pairs.
260:     - `analysis_ideaspace_*`: compute candidate “gap” measures via SQL aggregates.
261:   - Then expand to the rest:
262:     - Most “Family D” (classical stats): SQL to compute per-group samples/summary; pandas only for final tests.
263:     - ML plugins: SQL to select and type-coerce numeric feature matrix columns deterministically.
264: - **Complexity**: 9/10
265: - **Dependencies**: Task 4.1, Task 4.2
266: - **Acceptance Criteria**:
267:   - No plugin is forced to downsample.
268:   - For large datasets, full-DF loads become rare; most plugins operate on projected columns or aggregates.
269: - **Validation**:
270:   - `python -m pytest -q`
271:   - Add a “large dataset simulated” integration test ensuring unbounded loads are blocked unless explicitly approved.
272: 
273: ### Task 4.4: Add/Update Adoption Matrices
274: - **Location**: `scripts/plugin_data_access_matrix.py` (extend), `scripts/sql_assist_adoption_matrix.py` (new)
275: - **Description**:
276:   - Track per plugin:
277:     - uses SQL Assist: `0/1`
278:     - which intents used
279:     - still loads full DF unbounded: `0/1`
280:   - Keep docs up to date under `docs/`.
281: - **Complexity**: 4/10
282: - **Dependencies**: Sprint 1+
283: - **Acceptance Criteria**:
284:   - Matrices can be verified in CI (`--verify` mode).
285: - **Validation**:
286:   - Update `scripts/verify_docs_and_plugin_matrices.py` to include new docs.
287: 
288: ## Sprint 5: Plain-English Output (Separate Report)
289: **Goal**: Produce `plain_report.md` that is non-technical English while keeping `report.json` technical and citeable.
290: 
291: **Demo/Validation**:
292: - Full run produces `plain_report.md` alongside `report.md` and `answers_recommendations.md`.
293: 
294: ### Task 5.1: Add `report_plain_english_v1`
295: - **Location**: `plugins/report_plain_english_v1/*`, `src/statistic_harness/core/plain_report.py`
296: - **Description**:
297:   - Render each plugin’s key findings into:
298:     - “What it means”
299:     - “What to do next”
300:     - “How to validate” (with pointers to artifacts / SQL manifests)
301:   - Strictly avoid jargon; include the minimum technical metadata only as citations.
302: - **Complexity**: 6/10
303: - **Dependencies**: Sprint 1+ (for citations)
304: - **Acceptance Criteria**:
305:   - Report is understandable without statistical jargon.
306: - **Validation**:
307:   - Snapshot tests from fixture run outputs.
308: 
309: ## Plugin Benefit Analysis (High-Level)
310: Custom SQL helps almost every plugin in one of these ways:
311: - **Projection**: only load the columns the plugin needs (huge memory win).
312: - **Aggregation**: compute per-process/per-variant/per-hour/per-server stats in SQLite.
313: - **Join**: dependency resolution and “linkage” style plugins can do joins in SQL and only move final pairs to pandas.
314: - **Pre-materialization**: create derived tables/views once per run to avoid repeating heavy scans.
315: 
316: Where SQL does *not* replace work:
317: - Algorithms requiring full in-memory numeric matrices (PCA/IForest/LOF/etc) still need arrays, but SQL can build the matrix deterministically and reduce Python-side parsing.
318: 
319: ## Which Plugin Families Benefit Most (Concrete)
320: Prioritize SQL adoption for these families first (because they are currently full-DF and naturally SQL-friendly):
321: - **Operational levers / queueing / conformance / sequence**:
322:   - `analysis_actionable_ops_levers_v1`, `analysis_queue_delay_decomposition`, `analysis_busy_period_segmentation_v2`,
323:     `analysis_conformance_*`, `analysis_process_*`, `analysis_dependency_*`.
324:   - Typical SQL intents: per-process delay stats, busy-period windows, dependency hotspots, variant frequency + medians.
325: - **“Classical stats by group”** (fast win: aggregate in SQL, small samples in pandas):
326:   - `analysis_*ttest*`, `analysis_*anova*`, `analysis_*chi*`, `analysis_effect_size_report`.
327: - **Linkage / joins**:
328:   - `analysis_upload_linkage`, `analysis_dependency_resolution_join`.
329: - **Ideaspace**:
330:   - `analysis_ideaspace_*` benefits from SQL pre-aggregation for candidate gaps and action candidates (still citeable).
331: 
332: ## Testing Strategy
333: - Unit:
334:   - SQL validator deny/allow cases
335:   - schema snapshot determinism
336:   - sql pack schema validation
337: - Integration:
338:   - Full harness run on fixture dataset produces:
339:     - `report.md`, `report.json`, `plain_report.md`
340:     - SQL artifacts/manifests when enabled
341: - Gate:
342:   - `python -m pytest -q` must pass before shipping.
343: 
344: ## Potential Risks & Gotchas
345: - **LLM determinism**: GPU inference can drift. Mitigate by:
346:   - greedy decode
347:   - pinning model + tokenizer versions
348:   - storing prompt + output hashes
349:   - “generate once, replay forever” default workflow
350: - **Hallucinated columns**: SQLCoder may reference non-existent columns. Must fail closed before analysis starts.
351: - **Security**: SQL must be read-only unless explicitly enabling materialization with strict guardrails.
352: - **Performance**: poorly generated SQL can be slow. Mitigate by:
353:   - query plan inspection + index suggestions
354:   - prefer intent templates + fill-in rather than free-form SQL.
355:  - **Filesystem location**: model storage target `/mnt/d/autocapture/models/` is outside the repo. Implement a dedicated downloader that:
356:    - writes to a versioned subfolder under `/mnt/d/autocapture/models/sql/defog/sqlcoder2/`
357:    - records `sha256` of config + key weight files (or `snapshot_download` commit hash)
358:    - supports “already present” fast-path without network.
359: 
360: ## Rollback Plan
361: - Keep SQL Assist API isolated; default off via settings.
362: - If any regression:
363:   1. Disable SQL pack materialization plugin.
364:   2. Fall back to existing `ctx.dataset_loader(columns=...)` projections.
365:   3. Keep prompt pack plugin available for future iteration.
````

---

## Appendix E — ERP Knowledge Bank + 4 Pillars Optimization Plan

````markdown
  1: # Plan: ERP Knowledge Bank + 4 Pillars Optimization Engine
  2: 
  3: **Generated**: February 12, 2026  
  4: **Estimated Complexity**: High
  5: 
  6: ## Overview
  7: Build a persistent, cumulative knowledge system that learns from every dataset/run and improves recommendations over time, while scoring all work against the 4 pillars (`Performant`, `Accurate`, `Secure`, `Citable`).  
  8: The system must:
  9: - preserve and grow ERP column intelligence (including unmapped/new columns),
 10: - compute weighted metrics for full accounting month and close period,
 11: - compare projects/runs/repositories on a unified 4-pillar scale,
 12: - score each pillar on a strict `0.0-4.0` scale using all available functionality/telemetry (not Kona-only),
 13: - enforce balance so no pillar can be traded down to boost another pillar,
 14: - keep Kona energy traversal aligned to lowest-energy paths and expose gaps clearly.
 15: 
 16: ## Prerequisites
 17: - Python 3.11+ runtime and existing plugin pipeline.
 18: - SQLite migrations + storage access updates.
 19: - Existing normalization, ideaspace, and recommendation flows remain plugin-driven.
 20: - Existing run determinism and report schema gates continue to be enforced.
 21: 
 22: ## Sprint 1: Foundations (Contracts + Storage)
 23: **Goal**: Define canonical contracts and storage required for cumulative learning and 4-pillar scoring.  
 24: **Demo/Validation**:
 25: - New schemas and migration apply cleanly on a fresh and existing DB.
 26: - Basic write/read roundtrip for knowledge-bank tables succeeds.
 27: 
 28: ### Task 1.1: Define 4-Pillar Scoring Specification
 29: - **Location**: `docs/4_pillars_scoring_spec.md` (new), `docs/kona_energy_ideaspace_architecture.md`
 30: - **Description**: Define deterministic formulas, weights, normalization, and confidence bounds for:
 31:   - per-pillar scores on `0.0-4.0` (`Performant`, `Accurate`, `Secure`, `Citable`),
 32:   - per-run, per-dataset, and per-project/repo aggregate scores,
 33:   - source-of-truth inputs from the full system surface (plugins, runtime telemetry, security checks, evidence quality, reproducibility),
 34:   - ideal Kona low-energy traversal target (as one component, not sole scorer),
 35:   - balance constraints and veto rules (no optimization that materially degrades any pillar).
 36: - **Complexity**: 5
 37: - **Dependencies**: None
 38: - **Acceptance Criteria**:
 39:   - Every score has explicit formula + units.
 40:   - Each pillar is bounded and interpretable on `0.0-4.0`.
 41:   - Balance penalty and minimum-floor logic are explicit.
 42:   - Score output includes measured vs modeled split.
 43:   - Cross-domain comparison rules are documented.
 44: - **Validation**:
 45:   - Spec review with fixture calculations committed as examples.
 46: 
 47: ### Task 1.2: Add Knowledge-Bank Tables and Indexes
 48: - **Location**: `src/statistic_harness/core/migrations.py`, `src/statistic_harness/core/storage.py`
 49: - **Description**: Add persistent tables for:
 50:   - `knowledge_column_catalog` (canonical + observed columns, mapping confidence, ERP scope),
 51:   - `knowledge_metric_definitions` (metric id, formula version, scope),
 52:   - `knowledge_metric_observations` (run/dataset/project/repo observations),
 53:   - `knowledge_recommendation_outcomes` (recommended action + observed follow-up),
 54:   - `four_pillars_scorecards` (scored entity snapshots),
 55:   - `kona_energy_snapshots` (energy vectors + traversal states).
 56: - **Complexity**: 7
 57: - **Dependencies**: Task 1.1
 58: - **Acceptance Criteria**:
 59:   - Migrations are idempotent.
 60:   - Required indexes exist for `run_id`, `dataset_version_id`, `project_id`, `repo_id`.
 61:   - Tables are append-only where required for auditability.
 62: - **Validation**:
 63:   - `python -m pytest -q` includes migration tests for schema creation + rollback safety.
 64: 
 65: ### Task 1.3: Implement Column-Learning Contract
 66: - **Location**: `src/statistic_harness/core/stat_plugins/columns.py`, `plugins/transform_normalize_mixed/plugin.py`, `src/statistic_harness/core/storage.py`
 67: - **Description**: Persist every column classification event:
 68:   - mapped canonical column,
 69:   - new/unmapped column,
 70:   - fallback-add behavior when mapping is unavailable,
 71:   - confidence + provenance.
 72: - **Complexity**: 6
 73: - **Dependencies**: Task 1.2
 74: - **Acceptance Criteria**:
 75:   - Unknown columns always produce a stored learning record.
 76:   - No column is dropped silently.
 77:   - Normalization layer growth is traceable per run.
 78: - **Validation**:
 79:   - Unit tests for mapped and unmapped cases with deterministic outcomes.
 80: 
 81: ## Sprint 2: Metric Engine (Whole Month + Close)
 82: **Goal**: Compute weighted operational metrics that account for server count and context window (full month vs close).  
 83: **Demo/Validation**:
 84: - Metrics artifact includes both scopes and weighting details.
 85: - Re-run on same dataset is deterministic.
 86: 
 87: ### Task 2.1: Build Weighted Metric Families
 88: - **Location**: `src/statistic_harness/core/stat_plugins/ideaspace.py`, `src/statistic_harness/core/ideaspace_feature_extractor.py`, `src/statistic_harness/core/lever_library.py`
 89: - **Description**: Add reusable weighted metrics:
 90:   - `queue_delay_per_server_hour`,
 91:   - `service_time_per_server_hour`,
 92:   - `throughput_per_server`,
 93:   - `spillover_per_server`,
 94:   - `error_per_server`.
 95: - **Complexity**: 8
 96: - **Dependencies**: Sprint 1
 97: - **Acceptance Criteria**:
 98:   - Weighting explicitly includes active server count.
 99:   - Missing server metadata uses deterministic fallback + warning.
100: - **Validation**:
101:   - Unit tests with synthetic server topologies.
102: 
103: ### Task 2.2: Add Dual-Scope Aggregation (General vs Close)
104: - **Location**: `src/statistic_harness/core/close_cycle.py`, `src/statistic_harness/core/report.py`
105: - **Description**: Standardize every key modeled metric into:
106:   - `general_*` (full accounting month / observation window),
107:   - `close_*` (close period window),
108:   with aligned baseline/delta/percent fields.
109: - **Complexity**: 7
110: - **Dependencies**: Task 2.1
111: - **Acceptance Criteria**:
112:   - Recommendations always carry scope class + modeled basis.
113:   - Missing modeled fields fail contract checks.
114: - **Validation**:
115:   - Report contract tests confirm no `insufficient_modeled_inputs` where basis exists.
116: 
117: ### Task 2.3: Persist Metric Observations into Knowledge Bank
118: - **Location**: `plugins/report_decision_bundle_v2/plugin.py`, `src/statistic_harness/core/report.py`, `src/statistic_harness/core/storage.py`
119: - **Description**: Store per-run metric observations and modeled effects for future comparisons and recommendation tuning.
120: - **Complexity**: 6
121: - **Dependencies**: Tasks 2.1, 2.2
122: - **Acceptance Criteria**:
123:   - Metric observation rows are written for every completed/partial run.
124:   - Stored payload includes method version/fingerprint.
125: - **Validation**:
126:   - Integration test verifies DB growth across two runs and stable keys.
127: 
128: ## Sprint 3: Cumulative Learning + Cross-Dataset Intelligence
129: **Goal**: Make insights improve over time rather than reset each dataset.  
130: **Demo/Validation**:
131: - Comparison report shows learned trends and novelty across multiple datasets.
132: 
133: ### Task 3.1: Implement Dataset-to-Dataset Comparator
134: - **Location**: `plugins/analysis_knowledge_bank_comparison_v1/` (new), `src/statistic_harness/core/report.py`
135: - **Description**: Add plugin that compares current dataset against historical knowledge-bank observations for same ERP and for global cohorts.
136: - **Complexity**: 8
137: - **Dependencies**: Sprint 2
138: - **Acceptance Criteria**:
139:   - Output includes `improving`, `regressing`, `novel` signals.
140:   - Confidence uses sample-size and recency weighting.
141: - **Validation**:
142:   - Deterministic fixture tests for each signal class.
143: 
144: ### Task 3.2: Recommendation Memory and Outcome Feedback
145: - **Location**: `plugins/analysis_recommendation_dedupe_v2/plugin.py`, `src/statistic_harness/core/report.py`, `src/statistic_harness/core/storage.py`
146: - **Description**: Track recommendation lineage and post-change outcomes so ranking improves with evidence, not repetition.
147: - **Complexity**: 7
148: - **Dependencies**: Task 3.1
149: - **Acceptance Criteria**:
150:   - Recommendations include “seen before”, “previous outcome”, and “confidence adjustment”.
151:   - Duplicates are merged by stable signature + scope.
152: - **Validation**:
153:   - Unit tests for dedupe and confidence adjustment logic.
154: 
155: ### Task 3.3: ERP Column Intelligence Expansion
156: - **Location**: `src/statistic_harness/core/stat_plugins/columns.py`, `docs/plugin_data_access_matrix.json`, `docs/plugins_functionality_matrix.json`
157: - **Description**: Keep a maintained column-knowledge view showing:
158:   - canonical mapping coverage,
159:   - unresolved fields by ERP/source,
160:   - mapping drift over time.
161: - **Complexity**: 5
162: - **Dependencies**: Tasks 1.3, 3.1
163: - **Acceptance Criteria**:
164:   - Matrix artifacts refresh from DB source-of-truth.
165:   - New unresolved fields trigger critical warning before plugin stage (with safe add fallback).
166: - **Validation**:
167:   - Integration test with synthetic novel columns.
168: 
169: ## Sprint 4: 4-Pillars Ranking + Kona Optimization
170: **Goal**: Make the 4 pillars the top-level decision and comparison framework across runs, projects, and repositories.  
171: **Demo/Validation**:
172: - Every run emits a 4-pillar scorecard and Kona traversal status.
173: - Cross-project comparison table is generated.
174: 
175: ### Task 4.1: Add 4-Pillars Scoring Engine
176: - **Location**: `src/statistic_harness/core/four_pillars.py` (new), `src/statistic_harness/core/report.py`
177: - **Description**: Compute pillar scores from measured and modeled metrics across all active system functionality (not only ideaspace/Kona):
178:   - `Performant`: runtime, memory, throughput-per-server, spillover behavior,
179:   - `Accurate`: known-issue detection hit rate, modeled-vs-measured calibration,
180:   - `Secure`: sandbox/PII/redaction and policy violations,
181:   - `Citable`: evidence completeness, traceability links, reproducibility fields.
182: - **Complexity**: 9
183: - **Dependencies**: Sprints 1–3
184: - **Acceptance Criteria**:
185:   - Every pillar is emitted as a `0.0-4.0` score with component breakdown.
186:   - Scorecard emitted at run/project/repo scopes.
187:   - Pillar components and penalties are fully explorable.
188: - **Validation**:
189:   - Golden tests with fixed expected score outputs.
190: 
191: ### Task 4.2: Extend Kona Energy Map to Pillar Traversal
192: - **Location**: `src/statistic_harness/core/stat_plugins/ideaspace.py`, `docs/kona_energy_ideaspace_architecture.md`
193: - **Description**: Treat pillar gaps as first-class energy terms and produce “ideal traversal route” (Kona as optimization lens, not scoring source):
194:   - current vs ideal vector,
195:   - lowest-energy route candidates,
196:   - blockers preventing low-energy path.
197: - **Complexity**: 8
198: - **Dependencies**: Task 4.1
199: - **Acceptance Criteria**:
200:   - Route output is deterministic and cites source metrics.
201:   - “Fully optimal” condition is explicitly detectable.
202: - **Validation**:
203:   - Synthetic traversal tests with deterministic path ordering.
204: 
205: ### Task 4.3: Add Balance Guardrails and Tradeoff Vetoes
206: - **Location**: `src/statistic_harness/core/four_pillars.py` (new), `src/statistic_harness/core/report.py`, `scripts/plugin_functional_audit.py`
207: - **Description**: Enforce balanced optimization by default:
208:   - maximize minimum pillar first (`max-min` objective),
209:   - apply imbalance penalty when pillar spread exceeds threshold,
210:   - block recommendations/changes that improve one pillar by degrading another past configured tolerance.
211: - **Complexity**: 8
212: - **Dependencies**: Tasks 4.1, 4.2
213: - **Acceptance Criteria**:
214:   - Score output includes balance index and veto reasons.
215:   - Change proposals that fail balance policy are explicitly marked as rejected.
216: - **Validation**:
217:   - Unit tests for allowed vs vetoed tradeoff scenarios.
218: 
219: ### Task 4.4: Cross-Repository Scorecard Ingestion
220: - **Location**: `scripts/ingest_repo_scorecard.py` (new), `src/statistic_harness/core/storage.py`, `docs/implementation_matrix.md`
221: - **Description**: Ingest repo complexity + workflow metadata so different repos/projects can be compared by 4 pillars regardless of domain.
222: - **Complexity**: 7
223: - **Dependencies**: Task 4.1
224: - **Acceptance Criteria**:
225:   - Repo snapshots can be ingested without altering analysis runtime path.
226:   - Comparison output includes complexity-normalized pillar ranking.
227: - **Validation**:
228:   - Fixture-based tests using two synthetic repo snapshots.
229: 
230: ### Task 4.5: Per-Change 4-Pillar Recompute
231: - **Location**: `scripts/run_loaded_dataset_full.py`, `src/statistic_harness/core/report.py`, `src/statistic_harness/core/storage.py`
232: - **Description**: Recompute and persist 4-pillar scorecards on every materially different change (run fingerprint, plugin code hash, settings hash, dataset hash), not only on full dataset change.
233: - **Complexity**: 6
234: - **Dependencies**: Task 4.1
235: - **Acceptance Criteria**:
236:   - Scorecard history is appended for each change event with fingerprint linkage.
237:   - Trend views can slice by dataset-stable vs code-change deltas.
238: - **Validation**:
239:   - Integration test that changing plugin code updates scorecard lineage without changing dataset input.
240: 
241: ## Sprint 5: Reporting, Governance, and Matrix-First Workflow
242: **Goal**: Keep implementation and governance coherent as the system evolves.  
243: **Demo/Validation**:
244: - Reports include professional grouped recommendations + pillar scorecard.
245: - Matrices update before new implementation to avoid duplicate function work.
246: 
247: ### Task 5.1: Add Pillar + Knowledge Sections to Outputs
248: - **Location**: `scripts/show_actionable_results.py`, `scripts/run_loaded_dataset_full.py`, `src/statistic_harness/core/report.py`
249: - **Description**: Add sections for:
250:   - 4-pillar score trend,
251:   - month vs close weighted metrics,
252:   - knowledge-bank deltas vs prior runs.
253: - **Complexity**: 5
254: - **Dependencies**: Sprint 4
255: - **Acceptance Criteria**:
256:   - Human-readable and plain-language outputs include same core conclusions.
257:   - Known issues include pass/fail + modeled benefit consistently.
258: - **Validation**:
259:   - Snapshot tests for markdown outputs.
260: 
261: ### Task 5.2: Enforce Matrix-First Change Gate
262: - **Location**: `scripts/plugin_functional_audit.py`, `docs/plugins_functionality_matrix.json`, `docs/implementation_matrix.json`
263: - **Description**: Ensure feature work starts with matrix updates and conflict detection to prevent duplicate function creation.
264: - **Complexity**: 4
265: - **Dependencies**: None
266: - **Acceptance Criteria**:
267:   - CI/test gate fails if matrix is stale relative to plugin registry.
268: - **Validation**:
269:   - Unit test for stale-matrix detection.
270: 
271: ### Task 5.3: Add Baseline Reset Workflow
272: - **Location**: `scripts/reset_baseline.sh`, `docs/kona_energy_ideaspace_architecture.md`
273: - **Description**: Formalize “baseline reset” process for periodic recalibration without losing historical knowledge.
274: - **Complexity**: 4
275: - **Dependencies**: Sprint 4
276: - **Acceptance Criteria**:
277:   - Reset writes audit event and increments baseline version.
278: - **Validation**:
279:   - Integration test for reset + replay.
280: 
281: ## Testing Strategy
282: - Unit tests:
283:   - mapping/column-learning logic,
284:   - weighted metric math,
285:   - pillar score bounds (`0.0-4.0`) and component correctness,
286:   - balance/veto logic for cross-pillar tradeoffs,
287:   - scoring and traversal determinism,
288:   - recommendation memory.
289: - Integration tests:
290:   - full pipeline over normalized dataset,
291:   - two-run comparison with cumulative knowledge updates,
292:   - report contract checks (modeled fields, traceability, plain-language output).
293: - Regression/performance tests:
294:   - runtime/memory trend (`current vs avg/stddev`) persisted per run,
295:   - deterministic outputs with fixed `run_seed`,
296:   - no-network runtime guarantee intact.
297: - Gauntlet gate:
298:   - `python -m pytest -q` must pass before shipping.
299: 
300: ## Potential Risks & Gotchas
301: - Risk: score overfitting to one ERP profile.
302:   - Mitigation: enforce cross-domain normalization + confidence penalties for sparse history.
303: - Risk: one pillar can appear strong while another silently regresses.
304:   - Mitigation: hard floor + max-spread guardrails, max-min objective, and explicit veto output.
305: - Risk: knowledge-bank growth inflates DB and query latency.
306:   - Mitigation: partition/index strategy + materialized rollups.
307: - Risk: recommendation ranking can be gamed by large modeled values.
308:   - Mitigation: cap modeled influence, require evidence quality and calibration checks.
309: - Risk: cross-repo comparisons become unfair without complexity normalization.
310:   - Mitigation: include complexity factors explicitly (size, plugin surface, runtime budgets).
311: - Risk: stale baseline distorts Kona route.
312:   - Mitigation: versioned baseline with explicit reset protocol and drift alerts.
313: 
314: ## Rollback Plan
315: - Feature-flag new knowledge-bank writers and 4-pillar score computation.
316: - Keep legacy report generation path active during transition.
317: - If regressions appear:
318:   - disable new plugins (`analysis_knowledge_bank_comparison_v1`, `analysis_four_pillars_rank_v1`),
319:   - retain existing normalization and recommendation flows,
320:   - preserve accumulated knowledge tables (read-only) for postmortem.
````

---

## Appendix F — Defensive Red-Team + Production Hardening Recommendations (stat_redteam2-6-26)

````markdown
  1: THREAD: STATISTIC-HARNESS-ADVERSARIAL-REDESIGN
  2: CHAT_ID: N/A
  3: TS: 2026-02-06
  4: D0: 2026-02-06
  5: DETERMINISM: VERIFIED
  6: 
  7: ## Assumptions
  8: 
  9: * Analysis is based **only** on `generated repository snapshot artifact`; no runtime execution or external context.
 10: * Default posture is **local-only**; any networked/hosted behavior stays behind explicit flags.
 11: * Primary environment is **Windows 11 + heavy WSL2 + Python**; path/rename semantics must be Windows-safe.
 12: * Plugins are **semi-trusted**: treated as potentially buggy/malicious, so sandbox + provenance must be strong.
 13: * SQLite is the system-of-record for state and provenance; durability and corruption handling are first-class.
 14: * Determinism is a product guarantee: if something is randomized, it must be seeded and recorded.
 15: * Operator is a single primary user (Justin); multi-tenant features are treated as optional.
 16: * Large datasets are expected; “load everything into memory” must be guarded.
 17: 
 18: ## Context derived from this repository
 19: 
 20: This repository is **Statistic Harness**, a **local-first, plugin-based engine** for producing **deterministic insights** from **tabular datasets** (CSV/XLSX) via a plugin pipeline and generating a versioned `report.json` with lineage/evidence. 
 21: 
 22: ### Primary workflows
 23: 
 24: | Workflow         | What happens                                                                            | Evidence                                                          |
 25: | ---------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
 26: | CLI run          | Ingest a file, execute analysis plugins, produce report artifacts                       | README CLI usage                                                 |
 27: | Web UI run       | Serve local UI, upload dataset, configure run, execute pipeline, browse artifacts/trace | UI templates describe “Deterministic Insight Engine” + workflow  |
 28: | Plugin execution | Run plugins in subprocess with sandbox + guards; collect outputs                        | Plugin runner guards + sandbox                                   |
 29: 
 30: ### Major components
 31: 
 32: | Component                                   | Role                                                                          | Evidence                                         |
 33: | ------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------ |
 34: | `core/pipeline.py`                          | DAG planning + execution across plugin layers (parallel within layers)        | Toposort + ThreadPoolExecutor                   |
 35: | `core/plugin_manager.py`                    | Discover/load plugin specs, validate manifests/schemas                        | Plugin spec + schema usage                      |
 36: | `core/plugin_runner.py` + `core/sandbox.py` | Subprocess execution + sandboxing + safety guards (network/eval/pickle/shell) | Guards listed                                   |
 37: | `core/storage.py` + migrations              | SQLite state store; PRAGMAs and tables for runs/results/trace                 | WAL + synchronous NORMAL                        |
 38: | `core/dataset_io.py`                        | Dataset access via SQLite; can load into pandas                               | `_load_df` uses `pd.read_sql_query` and caches  |
 39: | `ui/server.py` + templates                  | Local web UI, upload endpoint, run views, trace UI                            | Upload handler enforces size + sha256           |
 40: | Schemas in `docs/`                          | JSON schemas for plugin manifests and reports                                 | Report schema exists                            |
 41: 
 42: ### Storage/state model
 43: 
 44: * Uses a local **SQLite** DB (`state.sqlite`) with WAL mode and other PRAGMAs (foreign keys, busy timeout, etc.). 
 45: * Upload pipeline is **content-addressed** (SHA-256) and enforces a configurable max upload size. 
 46: * Run identifiers are **random UUID4 hex** today (`make_run_id()`), not deterministically derived. 
 47: 
 48: ### Execution model
 49: 
 50: * Pipeline toposorts a dependency graph into **layers** and executes each layer with a **ThreadPoolExecutor**. 
 51: * For plugin subprocess sandboxing, the pipeline expands allowed paths to include the DB path and run directory. 
 52: * Plugin runner applies **determinism and safety guards**: blocks network unless enabled, blocks `eval`/`exec`, blocks pickle, blocks shell/subprocess calls. 
 53: 
 54: ### Interfaces
 55: 
 56: * CLI is present; local network binding for `serve` is guarded (requires explicit env to bind non-loopback). 
 57: * Web UI provides upload, run, trace views, and includes “vector search” UI elements (feature-flagged in code). 
 58: 
 59: ### External integrations
 60: 
 61: * Optional vector store uses `sqlite-vec` extension and is explicitly feature-flagged. 
 62: * Optional “tenancy” model is also feature-flagged. 
 63: * No evidence of mandatory third-party hosted services. (Network is explicitly guarded/disabled by default.) 
 64: 
 65: ## key_claims
 66: 
 67: * **[QUOTE]** “Statistic Harness is a local-first, plugin-based engine for deterministic insights from tabular datasets.” 
 68: * **[CITE]** Plugins execute under a guard set that blocks network/eval/pickle/shell by default. 
 69: * **[CITE]** Storage uses SQLite with WAL and synchronous=NORMAL (durability/perf tradeoff). 
 70: * **[CITE]** Upload handling computes SHA-256 and enforces a configurable maximum upload size. 
 71: * **[INFERENCE]** The current sandbox is strong on API-level egress and code injection, but is weaker on *write-surface minimization* because plugin allowlists include broad directories in manifests and pipeline expands allowed paths. 
 72: * **[NO EVIDENCE]** No evidence of GPU-accelerated processing in the current pipeline; any GPU path should remain optional and off by default.
 73: 
 74: ---
 75: 
 76: # I. Foundation: Inputs + core lifecycle stability
 77: 
 78: Current evidence: uploads are SHA-256 content-addressed with size limits , run IDs are UUID4 hex (not deterministic) , SQLite uses WAL and synchronous=NORMAL , and a Windows-safe rename hook exists but appears env-gated. 
 79: 
 80: | ID     | Recommendation details                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |                  Pillar scores | Effort / Risk |
 81: | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -----------------------------: | ------------- |
 82: | FND-01 | **Atomic run directory + crash-safe journaling**<br>Rationale: Prevent “half-created run” ambiguity and enable deterministic resume/cleanup after crashes or kills.<br>Dependencies: None.<br>Improved: crash recovery, integrity, audit.<br>Risked: orphan temp dirs if finalize fails.<br>Enforce: `src/statistic_harness/core/pipeline.py`, `src/statistic_harness/core/utils.py` (new `atomic_dir()`), DB run status transitions.<br>Regression detection: CI kill/restart test; if failing => **DO_NOT_SHIP**.<br>Acceptance test: kill process mid-run; restart shows run as `ABORTED` with preserved logs; rerun either resumes or cleanly restarts without mixed artifacts.                                                                                                            | P1=2 P2=2 P3=2 P4=4 (Total=10) | M / Med       |
 83: | FND-02 | **Atomic JSON writes for all run artifacts**<br>Rationale: Prevent truncated `report.json`/manifests from being mistaken as valid outputs.<br>Dependencies: None.<br>Improved: data integrity, citeability.<br>Risked: slight IO overhead.<br>Enforce: new helper in `core/utils.py` used everywhere JSON is written (reports, manifests, exported bundles).<br>Regression detection: unit test that simulates partial write; if failing => **DO_NOT_SHIP**.<br>Acceptance test: forcibly interrupt during write; file is either old valid version or new valid version, never partial.                                                                                                                                                                                                        |  P1=1 P2=1 P3=2 P4=3 (Total=7) | S / Low       |
 84: | FND-03 | **Harden upload CAS: verify-on-write + refcount + quarantine**<br>Rationale: Uploads are already hashed and size-limited ; add durability checks and lifecycle controls to prevent silent corruption and accidental reuse of a bad blob.<br>Dependencies: FND-02 recommended (atomic helpers).<br>Improved: input integrity, reproducibility.<br>Risked: migration complexity for existing uploads.<br>Enforce: `ui/server.py` upload flow, DB schema for upload references, and a “quarantine then promote” directory scheme.<br>Regression detection: upload corruption test (flip a byte); if not detected => **DO_NOT_SHIP**.<br>Acceptance test: after upload, recompute sha256 from disk and match; deleting a dataset version decrements refcount; unreferenced blobs are GC’d safely. | P1=2 P2=3 P3=2 P4=3 (Total=10) | M / Med       |
 85: | FND-04 | **Startup integrity checks: PRAGMA integrity_check + orphan cleanup**<br>Rationale: SQLite WAL + NORMAL is performant but needs explicit integrity gates under crash/power-loss scenarios. <br>Dependencies: None.<br>Improved: data durability, failure containment.<br>Risked: startup time on huge DBs.<br>Enforce: app startup path (CLI + UI), add “quick” and “full” integrity modes.<br>Regression detection: corrupted DB fixture; must fail loud; if silent => **DO_NOT_SHIP**.<br>Acceptance test: introduce corruption; app refuses to run and offers restore-from-backup flow.                                                                                                                                                                                                    |  P1=1 P2=2 P3=2 P4=3 (Total=8) | S / Low       |
 86: | FND-05 | **Online backup/restore + retention for `state.sqlite`**<br>Rationale: Make recovery deterministic and operator-friendly for Windows/WSL users where file locking and partial writes happen.<br>Dependencies: FND-04 (integrity checks).<br>Improved: resilience, citeability (recoverable history).<br>Risked: backup bloat if not pruned.<br>Enforce: `core/storage.py` (SQLite backup API), CLI command `stat-harness backup/restore`.<br>Regression detection: restore test with known DB; if mismatch => **DO_NOT_SHIP**.<br>Acceptance test: create run; backup; delete DB; restore; run list and reports remain consistent; checksums match.                                                                                                                                            |  P1=1 P2=2 P3=2 P4=4 (Total=9) | M / Low       |
 87: | FND-06 | **Deterministic `run_fingerprint` and optional deterministic IDs**<br>Rationale: Run IDs are UUID4 today ; add a deterministic fingerprint so “same input + same config + same plugin set” is provably identical, even if run_id differs.<br>Dependencies: META-01.<br>Improved: reproducibility and audit.<br>Risked: users may confuse run_id vs fingerprint unless UI clarifies.<br>Enforce: DB schema adds `run_fingerprint`; UI surfaces both.<br>Regression detection: deterministic re-run fixture; fingerprint must match; if not => **DO_NOT_SHIP**.<br>Acceptance test: two runs with same dataset hash + config hash produce same fingerprint; toggling any setting changes it.                                                                                                    |  P1=1 P2=1 P3=2 P4=4 (Total=8) | M / Med       |
 88: | FND-07 | **Make Windows/WSL safe_rename default for generated artifacts**<br>Rationale: A safe rename hook exists behind env control ; make it default on Windows/WSL to reduce partial artifact risk.<br>Dependencies: None.<br>Improved: reliability on Windows filesystems.<br>Risked: minor behavior differences in dev environments.<br>Enforce: CLI bootstrapping sets `STAT_HARNESS_SAFE_RENAME=1` automatically on Windows/WSL.<br>Regression detection: Windows path/rename tests; if flaky => **DO_NOT_SHIP**.<br>Acceptance test: repeated report writes under concurrent UI refresh never produce “file in use” partial outputs.                                                                                                                                                           |  P1=1 P2=1 P3=1 P4=2 (Total=5) | S / Low       |
 89: | FND-08 | **Explicit ingest completeness vs sampling metadata + enforcement**<br>Rationale: Users must not confuse sampled analysis with full ingest; record and surface completeness at each stage.<br>Dependencies: META-02, UX-08.<br>Improved: correctness and user trust.<br>Risked: plugin compatibility (must emit sampling metadata).<br>Enforce: report schema, plugin output schema conventions, UI badges.<br>Regression detection: sample-mode run must display “SAMPLED” everywhere; if missing => **DO_NOT_SHIP**.<br>Acceptance test: run with sampling enabled; report and UI show sample fraction, seed, and selection method; export includes these fields.                                                                                                                            |  P1=1 P2=0 P3=3 P4=3 (Total=7) | M / Low       |
 90: 
 91: ---
 92: 
 93: # II. Metadata: Contracts + provenance + auditability
 94: 
 95: Current evidence: deterministic JSON utilities exist (`stable_json_dumps`) , plugin manifests are schema-driven , and a report schema exists in `docs/report.schema.json`. 
 96: 
 97: | ID      | Recommendation details                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |                 Pillar scores | Effort / Risk |
 98: | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------: | ------------- |
 99: | META-01 | **Persist canonical `run_manifest.v1` (schema + file) per run**<br>Rationale: “Processing happened against THIS input + THIS config” must be provable without DB access (portable evidence).<br>Dependencies: FND-02.<br>Improved: citeability, reproducibility.<br>Risked: schema churn if not versioned strictly.<br>Enforce: new `docs/run_manifest.schema.json`, write `run_manifest.json` into run dir, store hash in DB.<br>Regression detection: schema validation in CI; if broken => **DO_NOT_SHIP**.<br>Acceptance test: every completed run has a valid manifest including input sha256, config snapshot, plugin list+versions, flags, seeds, and artifact hashes. | P1=1 P2=1 P3=3 P4=4 (Total=9) | M / Low       |
100: | META-02 | **Standardize modeled-vs-measured + confidence/evidence blocks**<br>Rationale: Prevent users treating modeled outputs as measured facts; force explicit disclosure per finding.<br>Dependencies: report schema changes + plugin output conventions.<br>Improved: accuracy and interpretability.<br>Risked: plugin updates required.<br>Enforce: `docs/report.schema.json` update + output schema validator in pipeline.<br>Regression detection: sample report fixture; must contain blocks; if missing => **DO_NOT_SHIP**.<br>Acceptance test: UI renders “MEASURED” vs “MODELED” tags and requires evidence links for measured claims.                                      | P1=0 P2=0 P3=3 P4=4 (Total=7) | M / Med       |
101: | META-03 | **Store per-plugin `execution_fingerprint` (code+manifest+settings+input)**<br>Rationale: If plugin code changes, the same run config should not be considered comparable unless fingerprints match.<br>Dependencies: META-01.<br>Improved: audit trail, replay safety.<br>Risked: larger DB rows.<br>Enforce: add columns to plugin results/executions; compute using stable hashing utilities.<br>Regression detection: fingerprint must change on plugin file edit; if not => **DO_NOT_SHIP**.<br>Acceptance test: editing plugin code forces recompute (or marks cached results invalid).                                                                                 | P1=1 P2=2 P3=2 P4=4 (Total=9) | M / Low       |
102: | META-04 | **Evidence registry for artifacts (sha256, bytes, mime, producer, path)**<br>Rationale: Artifacts should be referencable and verifiable independent of UI links.<br>Dependencies: META-01.<br>Improved: citeability and safety (detect tampering).<br>Risked: requires standard artifact emission path.<br>Enforce: pipeline collects artifact metadata from run dir; stores registry in manifest + DB.<br>Regression detection: artifact download must match registered sha256; if mismatch => **DO_NOT_SHIP**.<br>Acceptance test: every artifact shown in UI has a hash, size, mime, and producing plugin id.                                                              | P1=1 P2=1 P3=1 P4=4 (Total=7) | M / Low       |
103: | META-05 | **Schema versioning + compatibility checks (config/output/report) + hashes**<br>Rationale: Prevent silent schema drift between plugin versions and UI/report readers.<br>Dependencies: EXT-07, QA-02.<br>Improved: correctness and forward/back compat.<br>Risked: strict validation may break legacy plugins.<br>Enforce: plugin_manager validates schema versions; pipeline refuses incompatible combos with clear error.<br>Regression detection: contract tests; if drift => **DO_NOT_SHIP**.<br>Acceptance test: downgrade plugin bundle; system blocks run if report consumer can’t parse output schema.                                                                | P1=1 P2=1 P3=3 P4=3 (Total=8) | M / Med       |
104: | META-06 | **Vector-store provenance (model id, chunking, source entity) exportable**<br>Rationale: Vector search must be auditable; embeddings aren’t “magic,” they’re derived artifacts.<br>Dependencies: vector store feature-flag remains optional .<br>Improved: explainability and citeability of retrieval results.<br>Risked: schema changes for vector tables.<br>Enforce: store provenance alongside embeddings; include in export bundles.<br>Regression detection: embedding row must reference source entity; if not => **DO_NOT_SHIP**.<br>Acceptance test: search result includes link to source row/plugin output and embedding provenance fields.                      | P1=1 P2=1 P3=2 P4=3 (Total=7) | M / Low       |
105: | META-07 | **UI “What ran?” summary card (hashes, seeds, flags, versions)**<br>Rationale: Reduce cognitive load and prevent “I think I ran X” errors.<br>Dependencies: META-01.<br>Improved: operator trust, fewer misinterpretations.<br>Risked: none meaningful.<br>Enforce: UI run detail template and API include manifest summary.<br>Regression detection: UI test asserts card present; if missing => **DO_NOT_SHIP**.<br>Acceptance test: a screenshot of run detail page is enough to reconstruct the exact execution inputs/config.                                                                                                                                            | P1=0 P2=0 P3=2 P4=4 (Total=6) | S / Low       |
106: | META-08 | **Export “Repro pack” zip (manifest + schemas + logs + report)**<br>Rationale: Make results portable and reviewable offline; aligns to “auditable runs.”<br>Dependencies: OBS-03 (diag bundle) and META-01.<br>Improved: citeability and supportability.<br>Risked: possible accidental inclusion of raw data; must be configurable.<br>Enforce: CLI command `stat-harness export --run-id` with explicit include/exclude options.<br>Regression detection: export content list test; if includes raw without consent => **DO_NOT_SHIP**.<br>Acceptance test: unzip pack, validate schemas, verify hashes, open report without DB.                                            | P1=1 P2=1 P3=2 P4=4 (Total=8) | M / Low       |
107: 
108: ---
109: 
110: # III. Execution pipeline: Correctness + replayability
111: 
112: Current evidence: deterministic layer ordering exists (sorted zero-indegree list) and parallel execution uses ThreadPoolExecutor ; plugin runner blocks risky APIs by default ; pipeline expands sandbox allow paths to include DB path and run dir. 
113: 
114: | ID      | Recommendation details                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |                 Pillar scores | Effort / Risk |
115: | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------: | ------------- |
116: | EXEC-01 | **Idempotent plugin caching keyed by hashes + `--force/--reuse`**<br>Rationale: Prevent redundant work and make re-runs deterministic and cheap while still auditable.<br>Dependencies: META-03, FND-06.<br>Improved: performance + citeability (“same fingerprint reused”).<br>Risked: stale cache if fingerprinting incomplete.<br>Enforce: storage query by (dataset_hash, settings_hash, plugin id/version, code hash).<br>Regression detection: change one setting => cache miss; if cache hit => **DO_NOT_SHIP**.<br>Acceptance test: run twice; second run reuses results with explicit “REUSED” markers and zero plugin executions.                                                         | P1=2 P2=1 P3=2 P4=4 (Total=9) | M / Med       |
117: | EXEC-02 | **Per-plugin timeout + memory budget enforcement**<br>Rationale: Contain runaway plugins under fatigue/misconfig; avoid whole-system stalls.<br>Dependencies: plugin runner enhancements.<br>Improved: operational stability and safety.<br>Risked: false positives on heavy plugins unless budgets are per-plugin.<br>Enforce: `core/plugin_runner.py` adds timeout and resource limits (soft); pipeline records termination reason.<br>Regression detection: synthetic slow plugin must be killed; if not => **DO_NOT_SHIP**.<br>Acceptance test: plugin exceeding limit ends with clear status + logs; run continues or fails deterministically per policy.                                      | P1=2 P2=2 P3=2 P4=2 (Total=8) | M / Med       |
118: | EXEC-03 | **Deterministic persistence order + per-plugin derived seeds**<br>Rationale: Parallel completion order is nondeterministic; persist results in stable `plugin_id` order and seed each plugin deterministically.<br>Dependencies: stable hashing utility exists .<br>Improved: reproducibility and diff-friendly outputs.<br>Risked: none meaningful.<br>Enforce: pipeline sorts before DB writes; derive `plugin_seed = hash(run_seed, plugin_id)` in context passed to plugin runner.<br>Regression detection: repeated run yields identical DB outputs; if diff => **DO_NOT_SHIP**.<br>Acceptance test: run twice; report JSON and plugin output JSON are byte-identical (excluding timestamps). | P1=1 P2=1 P3=3 P4=3 (Total=8) | S / Low       |
119: | EXEC-04 | **Failure containment: temp dirs + commit-on-validate only**<br>Rationale: Prevent partial outputs from being mistaken as valid results; isolate plugin scratch state.<br>Dependencies: FND-01, FND-02.<br>Improved: integrity and debuggability.<br>Risked: extra disk usage during runs.<br>Enforce: plugin writes to temp; pipeline validates output schema then moves artifacts to final run dir (atomic rename).<br>Regression detection: invalid output must not be committed; if committed => **DO_NOT_SHIP**.<br>Acceptance test: plugin emits invalid JSON; run marks plugin failed; no output rows exist in results table.                                                                | P1=1 P2=2 P3=2 P4=3 (Total=8) | M / Med       |
120: | EXEC-05 | **Replay mode: verify hashes then reuse cached results**<br>Rationale: Provide operator-grade replay for audits: “show me what ran” without re-executing plugins unless requested.<br>Dependencies: META-01, EXEC-01, META-04.<br>Improved: citeability + safety of reviews.<br>Risked: complexity in UI and CLI semantics.<br>Enforce: new command `stat-harness replay --run-id`; verifies manifest hashes vs disk/DB before rendering.<br>Regression detection: tamper artifact => replay must fail; if silent => **DO_NOT_SHIP**.<br>Acceptance test: modify artifact; replay reports hash mismatch and refuses to present as valid.                                                            | P1=2 P2=1 P3=2 P4=4 (Total=9) | L / Med       |
121: | EXEC-06 | **Pre-run config validation against plugin schemas with deterministic defaults**<br>Rationale: Reduce misconfig footguns and “it ran but not how I thought.”<br>Dependencies: META-05.<br>Improved: accuracy, fewer operator mistakes.<br>Risked: breaking old configs that relied on implicit defaults.<br>Enforce: pipeline loads each plugin’s `config_schema` and applies defaults deterministically (stable ordering).<br>Regression detection: schema fixture tests; if defaults drift => **DO_NOT_SHIP**.<br>Acceptance test: missing optional config fields are filled and recorded in manifest; UI shows the resolved config.                                                              | P1=1 P2=1 P3=3 P4=3 (Total=8) | S / Low       |
122: | EXEC-07 | **Plugin DAG validation: cycles and missing deps with graph output**<br>Rationale: Operator-grade errors for plugin conflicts; reduce admin debugging time.<br>Dependencies: None.<br>Improved: correctness and UX clarity.<br>Risked: none meaningful.<br>Enforce: pipeline preflight prints graph edges and the exact cycle path when detected.<br>Regression detection: cycle fixture; must error clearly; if not => **DO_NOT_SHIP**.<br>Acceptance test: introduce cyclic deps; run fails before execution with actionable message and suggested fix.                                                                                                                                           | P1=0 P2=0 P3=2 P4=2 (Total=4) | S / Low       |
123: | EXEC-08 | **Run checkpoints in DB for resume-after-restart**<br>Rationale: WSL/Windows restarts happen; resuming avoids rework and reduces partial-state confusion.<br>Dependencies: FND-01, EXEC-04.<br>Improved: reliability and operability.<br>Risked: more DB writes/state complexity.<br>Enforce: stage checkpoint table; pipeline updates at stage boundaries and per-plugin completion.<br>Regression detection: restart test; resume must not rerun completed plugins; if rerun => **DO_NOT_SHIP**.<br>Acceptance test: stop process mid-layer; restart resumes pending plugins only; run fingerprint unchanged.                                                                                     | P1=1 P2=1 P3=2 P4=3 (Total=7) | L / Med       |
124: 
125: ---
126: 
127: # IV. Extension system & manager: Discovery + lifecycle + sandboxing + UX
128: 
129: Current evidence: plugin manifests are schema-driven  and plugin manager loads plugin specs ; plugin manifests commonly include broad filesystem allowlists (example plugin allows appdata + plugins). 
130: 
131: | ID     | Recommendation details                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |                 Pillar scores | Effort / Risk |
132: | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------: | ------------- |
133: | EXT-01 | **Local plugin registry index (install state, version, hash, enabled)**<br>Rationale: Discovery alone is insufficient; operators need lifecycle state and provenance for plugins.<br>Dependencies: META-03.<br>Improved: auditability, safer updates.<br>Risked: migration from implicit “plugins/” scanning.<br>Enforce: new registry table in SQLite or a single registry file in appdata; plugin_manager reads registry first.<br>Regression detection: disable plugin => must not execute; if executes => **DO_NOT_SHIP**.<br>Acceptance test: disabling a plugin hides it in UI and pipeline; enabling restores deterministically.                                                                                                              | P1=1 P2=2 P3=1 P4=3 (Total=7) | M / Med       |
134: | EXT-02 | **CLI: `stat-harness plugins validate` (schema + import + smoke + caps)**<br>Rationale: Catch broken plugins early; reduce admin/operator friction.<br>Dependencies: None.<br>Improved: safety and correctness.<br>Risked: validation may be slow without caching.<br>Enforce: plugin_manager + CLI command uses schema validation and imports entrypoint in isolated mode.<br>Regression detection: validation must fail on malformed manifest; if passes => **DO_NOT_SHIP**.<br>Acceptance test: run validate across all plugins; outputs a report with pass/fail and required capabilities.                                                                                                                                                       | P1=0 P2=2 P3=2 P4=2 (Total=6) | S / Low       |
135: | EXT-03 | **Offline install/update/rollback via hashed plugin bundles**<br>Rationale: Enterprise/locked-down environments need deterministic offline upgrades and easy rollback.<br>Dependencies: EXT-01, EXT-02.<br>Improved: supply-chain hygiene, reproducibility.<br>Risked: significant implementation scope.<br>Enforce: bundle format: zip with manifest + file hashes; stored under appdata; registry points to active version.<br>Regression detection: rollback must restore prior hashes; if mismatch => **DO_NOT_SHIP**.<br>Acceptance test: install v2; run; rollback to v1; rerun produces expected v1 fingerprints and outputs.                                                                                                                 | P1=1 P2=3 P3=1 P4=3 (Total=8) | L / Med       |
136: | EXT-04 | **Capability negotiation + permissions model enforced by runner/UI**<br>Rationale: Users/admins must see and approve what a plugin can do (DB read/write, fs write, vector store, network).<br>Dependencies: EXT-01, EXT-05.<br>Improved: security and reduced footguns.<br>Risked: plugin ecosystem migration.<br>Enforce: plugin.yaml adds `capabilities_required`; pipeline/runner enforces.<br>Regression detection: plugin declaring network must not access network unless explicitly enabled; if it does => **DO_NOT_SHIP**.<br>Acceptance test: UI shows permissions before run; deny capability => plugin blocked with explicit reason.                                                                                                     | P1=0 P2=4 P3=1 P4=2 (Total=7) | M / Med       |
137: | EXT-05 | **Sandbox policy redesign: fs read/write split + narrow DB directory access**<br>Rationale: Current allowlists can include broad directories ; split read vs write to reduce persistence of malicious/buggy behavior.<br>Dependencies: SEC-05, QA-03.<br>Improved: security and integrity (prevent plugin self-modifying code).<br>Risked: compatibility break for plugins that write outside run dir.<br>Enforce: `core/sandbox.py` supports read-allow/write-allow; default write paths = run_dir only; DB dir read-only except WAL needs.<br>Regression detection: sandbox escape suite; if can write to plugins dir => **DO_NOT_SHIP**.<br>Acceptance test: plugin attempts to write to `plugins/`; blocked; writing to run artifacts succeeds. | P1=1 P2=4 P3=1 P4=3 (Total=9) | M / High      |
138: | EXT-06 | **Plugin health checks contract (`health()`), surfaced in manager UI**<br>Rationale: Operators need a quick way to detect broken dependencies without running full pipeline.<br>Dependencies: EXT-01.<br>Improved: operability and correctness.<br>Risked: plugins without health() need a default adapter.<br>Enforce: plugin entrypoint optionally implements `health()`; manager aggregates results and logs.<br>Regression detection: missing dependency must be visible; if silent => **DO_NOT_SHIP**.<br>Acceptance test: break a dependency; plugin shows “unhealthy” and run is blocked unless forced.                                                                                                                                       | P1=0 P2=1 P3=2 P4=2 (Total=5) | M / Low       |
139: | EXT-07 | **Compatibility rules: engine version range + schema versions**<br>Rationale: Stop incompatible plugins early instead of failing mid-run.<br>Dependencies: META-05.<br>Improved: stability and predictability.<br>Risked: none meaningful.<br>Enforce: plugin.yaml includes `requires.engine_semver`; manager enforces and reports.<br>Regression detection: incompatible plugin must be blocked; if runs => **DO_NOT_SHIP**.<br>Acceptance test: set engine range incompatible; UI shows blocked with reason and suggested compatible versions.                                                                                                                                                                                                     | P1=0 P2=1 P3=2 P4=2 (Total=5) | S / Low       |
140: | EXT-08 | **Extension Manager UI: install/update/rollback/disable/permissions**<br>Rationale: Reduce admin friction; make plugin state explicit and keyboard accessible.<br>Dependencies: EXT-01, EXT-03, EXT-04.<br>Improved: operator UX and safety.<br>Risked: scope creep if not MVP’d.<br>Enforce: new UI route `/plugins/manage` with clear IA and actions logged to events table.<br>Regression detection: UI smoke tests; if cannot disable plugin => **DO_NOT_SHIP**.<br>Acceptance test: install bundle, view permissions, disable plugin, rollback version, and see fingerprints change accordingly.                                                                                                                                                | P1=0 P2=1 P3=1 P4=2 (Total=4) | L / Low       |
141: 
142: ---
143: 
144: # V. UI/UX: End-to-end workflow
145: 
146: Current evidence: UI positions itself as a “Deterministic Insight Engine” with project creation and dataset upload flow ; the wizard form includes a `run_seed` field (risk of confusion if semantics unclear). 
147: 
148: | ID    | Recommendation details                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |                 Pillar scores | Effort / Risk |
149: | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------: | ------------- |
150: | UX-01 | **Activity Dashboard with recent runs + determinism badges**<br>Rationale: Reduce “where am I?” friction and improve visibility of failures/completeness.<br>Dependencies: OBS-01, META-01.<br>Improved: operator speed + fewer mistakes.<br>Risked: none meaningful.<br>Enforce: new route and template; include TTFR and run completeness indicators.<br>Regression detection: UI e2e test; dashboard must render and link correctly; if broken => **DO_NOT_SHIP**.<br>Acceptance test: dashboard shows last N runs, status, fingerprint, input hash, duration, TTFR, and missing plugins.                                                     | P1=0 P2=0 P3=1 P4=3 (Total=4) | M / Low       |
151: | UX-02 | **Ingest panel: preview + sha256 + format detection + completeness**<br>Rationale: Prevent wrong-file selection and support “prove what input was used.”<br>Dependencies: FND-03, META-01.<br>Improved: accuracy and citeability.<br>Risked: preview may be slow on huge files unless sampled safely.<br>Enforce: upload returns preview + inferred schema; UI shows hash and first rows before “Run.”<br>Regression detection: upload preview test; if hash not shown => **DO_NOT_SHIP**.<br>Acceptance test: user can confirm dataset identity via hash + preview and compare against prior runs.                                              | P1=0 P2=1 P3=2 P4=3 (Total=6) | M / Low       |
152: | UX-03 | **Presets/templates for settings with diff + import/export**<br>Rationale: Reduce config friction and misconfiguration under fatigue.<br>Dependencies: EXEC-06.<br>Improved: speed and correctness.<br>Risked: preset drift across plugin versions unless schema-hashed.<br>Enforce: preset format includes plugin schema hashes; UI shows diff before applying.<br>Regression detection: preset apply must validate schemas; if bypass => **DO_NOT_SHIP**.<br>Acceptance test: export preset from a run; import elsewhere; system validates compatibility and shows diffs.                                                                      | P1=1 P2=0 P3=2 P4=2 (Total=5) | M / Med       |
153: | UX-04 | **Run config guardrails: grouped plugins + warnings + confirm summary**<br>Rationale: Prevent “ran the wrong thing” and clarify what will execute.<br>Dependencies: EXT-04, META-07.<br>Improved: fewer mistakes, clearer intent.<br>Risked: UI complexity if not designed tightly.<br>Enforce: single confirmation page with plugin list, deps satisfied, permissions, and flags.<br>Regression detection: UI test asserts confirm summary present; if missing => **DO_NOT_SHIP**.<br>Acceptance test: before run, user sees exactly which plugins and versions will run and what capabilities are requested.                                   | P1=1 P2=1 P3=2 P4=2 (Total=6) | M / Low       |
154: | UX-05 | **Run detail: timeline + errors + artifacts with hashes + replay button**<br>Rationale: Reduce confusion after partial failure and improve auditability at the point of use.<br>Dependencies: OBS-04, META-04, EXEC-05.<br>Improved: debugging and citeability.<br>Risked: none meaningful.<br>Enforce: run detail API includes plugin execution durations and artifact registry.<br>Regression detection: run detail must show artifact hashes; if missing => **DO_NOT_SHIP**.<br>Acceptance test: user can copy a single “Run summary” block that includes enough metadata to reproduce/replay.                                                | P1=0 P2=0 P3=2 P4=3 (Total=5) | M / Low       |
155: | UX-06 | **Accessibility pass: labels/ARIA/keyboard/high-contrast/reduced-motion**<br>Rationale: Accessibility-first reduces errors and fatigue-related misclicks.<br>Dependencies: QA-05 (UI tests).<br>Improved: usability under real constraints.<br>Risked: none meaningful.<br>Enforce: template refactor + automated a11y checks where feasible.<br>Regression detection: a11y smoke tests; if critical issues => **DO_NOT_SHIP**.<br>Acceptance test: all inputs have labels; dynamic status updates are announced; full keyboard navigation works.                                                                                                | P1=0 P2=0 P3=1 P4=1 (Total=2) | M / Low       |
156: | UX-07 | **Misclick protection: confirm destructive actions + undo (soft delete)**<br>Rationale: Low patience + fatigue means deletion/disable must be reversible.<br>Dependencies: OBS-01 (events), storage schema for soft deletes.<br>Improved: safety and operability.<br>Risked: storage bloat without retention policies.<br>Enforce: soft-delete flags; add “Undo” for recent actions; show fingerprint in confirm dialog.<br>Regression detection: delete should be reversible within window; if not => **DO_NOT_SHIP**.<br>Acceptance test: delete run => disappears; undo => restored with identical hashes.                                    | P1=0 P2=1 P3=1 P4=2 (Total=4) | M / Med       |
157: | UX-08 | **Evidence surfacing: modeled/measured + confidence + sampling coverage**<br>Rationale: Make it difficult to misinterpret results and easy to cite.<br>Dependencies: META-02, FND-08.<br>Improved: correctness and trust signals.<br>Risked: requires plugin output normalization.<br>Enforce: UI components render these fields consistently across report, plugin outputs, and trace views.<br>Regression detection: missing evidence/confidence must be flagged; if silently omitted => **DO_NOT_SHIP**.<br>Acceptance test: every displayed claim shows type (modeled/measured), confidence, and evidence link(s) or explicit “no evidence.” | P1=0 P2=0 P3=2 P4=3 (Total=5) | M / Med       |
158: 
159: ---
160: 
161: # VI. Observability / Ops: Logs + metrics + diagnostics bundles
162: 
163: Current evidence: the system persists execution artifacts and has trace UI; operational telemetry for “frictionless workflow metrics” is not clearly surfaced in UI (no evidence of TTFR dashboard). Also, server and pipeline already track structured states in DB migrations (tables for runs/results/trace exist). 
164: 
165: | ID     | Recommendation details                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |                 Pillar scores | Effort / Risk |
166: | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------: | ------------- |
167: | OBS-01 | **Structured events table for lifecycle telemetry (correlation IDs)**<br>Rationale: Enables measured UX/ops improvements and reliable diagnostics without scraping logs.<br>Dependencies: None.<br>Improved: auditability and ops insight.<br>Risked: event volume growth without retention.<br>Enforce: new DB table `events`; emit at upload/run/plugin start/end/fail; include `run_id`, `fingerprint`, `plugin_id`.<br>Regression detection: events emitted for every run stage; if missing => **DO_NOT_SHIP**.<br>Acceptance test: a single SQL query reconstructs run timeline and TTFR.                        | P1=1 P2=1 P3=2 P4=3 (Total=7) | M / Low       |
168: | OBS-02 | **Frictionless workflow metrics (TTFR, fail rate, retries) in UI**<br>Rationale: “Frictionless core workflow” must be measured, not guessed.<br>Dependencies: OBS-01, UX-01.<br>Improved: performance UX and reliability tuning.<br>Risked: metric misinterpretation if definitions unclear.<br>Enforce: define metrics precisely; render on dashboard; export CSV for analysis.<br>Regression detection: metric definitions tests; if drift => **DO_NOT_SHIP**.<br>Acceptance test: dashboard shows TTFR per run and weekly median; shows top failure causes.                                                        | P1=1 P2=0 P3=2 P4=2 (Total=5) | M / Low       |
169: | OBS-03 | **Diagnostics bundle export (zip) with manifests+logs+env+schema**<br>Rationale: Operator-grade support requires “one file” to attach/inspect offline.<br>Dependencies: META-01, META-04.<br>Improved: citeability and debuggability.<br>Risked: accidental inclusion of sensitive data unless controlled.<br>Enforce: CLI `stat-harness diag --run-id`; include allowlist of files; redact secrets (SEC-03).<br>Regression detection: diag content test; if includes secrets => **DO_NOT_SHIP**.<br>Acceptance test: bundle validates, contains run_manifest, report, plugin logs, system info, and artifact hashes. | P1=1 P2=1 P3=2 P4=4 (Total=8) | M / Low       |
170: | OBS-04 | **Run timeline visualization from plugin executions + queue time**<br>Rationale: Makes bottlenecks and partial failures obvious without log spelunking.<br>Dependencies: OBS-01.<br>Improved: ops and UX clarity.<br>Risked: none meaningful.<br>Enforce: UI renders bars by plugin duration; includes waiting/queued time per layer.<br>Regression detection: timeline must match event timestamps; if inconsistent => **DO_NOT_SHIP**.<br>Acceptance test: run detail shows consistent durations vs stored timestamps.                                                                                              | P1=1 P2=0 P3=1 P4=2 (Total=4) | M / Low       |
171: | OBS-05 | **Health endpoints (`/healthz`, `/readyz`) + disk/db/plugin checks**<br>Rationale: Prevent silent failures and enable local automation/monitoring.<br>Dependencies: FND-04, EXT-02.<br>Improved: stability and safer ops.<br>Risked: none meaningful.<br>Enforce: server exposes endpoints; checks include DB integrity quick check and free disk threshold.<br>Regression detection: health must fail when DB is corrupted; if passes => **DO_NOT_SHIP**.<br>Acceptance test: simulate low disk => `/readyz` fails with explicit reason.                                                                             | P1=0 P2=1 P3=1 P4=2 (Total=4) | S / Low       |
172: | OBS-06 | **Retention policies for logs/artifacts (prevent disk-full)**<br>Rationale: Disk-full creates cascading corruption and user pain on Windows/WSL.<br>Dependencies: OBS-01.<br>Improved: reliability and performance under long use.<br>Risked: data loss if retention too aggressive.<br>Enforce: configurable retention by age/count/size; soft-delete first; export before purge.<br>Regression detection: retention must never delete the latest N runs; if it does => **DO_NOT_SHIP**.<br>Acceptance test: set retention to 3 runs; after 10 runs, only oldest purged; newest preserved.                           | P1=2 P2=1 P3=0 P4=1 (Total=4) | M / Low       |
173: | OBS-07 | **Perf baseline harness + CI regression gates**<br>Rationale: Performance regressions are operational failures for large datasets.<br>Dependencies: PERF-02, QA-06.<br>Improved: sustained performance.<br>Risked: flaky CI if not stabilized.<br>Enforce: benchmark ingest + scan stats; compare within tolerance.<br>Regression detection: >X% slowdown => **DO_NOT_SHIP**.<br>Acceptance test: CI fails deterministically on regression and links to perf diff report.                                                                                                                                             | P1=2 P2=0 P3=1 P4=1 (Total=4) | M / Med       |
174: 
175: ---
176: 
177: # VII. Security / Privacy: Local-first hardening + safe optional features
178: 
179: Current evidence: plugin runner blocks network unless enabled and blocks dangerous primitives (eval/pickle/shell).  Server bind-to-nonloopback is explicitly guarded by env.  Tenancy is feature-flagged. 
180: 
181: | ID     | Recommendation details                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |                 Pillar scores | Effort / Risk |
182: | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------: | ------------- |
183: | SEC-01 | **Local-only server binding + CSRF/session hardening for auth**<br>Rationale: Local UI should not become remotely reachable accidentally; if auth is enabled, sessions must be hardened.<br>Dependencies: none (build on existing guard). <br>Improved: security posture and safe defaults.<br>Risked: some remote-use workflows might need explicit flags.<br>Enforce: require explicit CLI flags for remote host binding; add CSRF token for POST routes; secure cookies.<br>Regression detection: attempt remote bind without flag must fail; if binds => **DO_NOT_SHIP**.<br>Acceptance test: `serve` without flag only binds 127.0.0.1; UI posts require CSRF.                                                                         | P1=0 P2=4 P3=0 P4=1 (Total=5) | M / Low       |
184: | SEC-02 | **Enforce sandbox write-surface minimization (paired with EXT-05)**<br>Rationale: Guards block “dangerous APIs,” but persistence attacks happen via filesystem writes; prevent writing to plugin directories and broad appdata paths by default.<br>Dependencies: EXT-05, QA-03.<br>Improved: integrity and containment.<br>Risked: plugin breakage if they wrote outside run dir.<br>Enforce: `core/sandbox.py` read/write allowlists; default write=run_dir only; require explicit capability for other writes.<br>Regression detection: sandbox escape suite; any write outside => **DO_NOT_SHIP**.<br>Acceptance test: plugin cannot write to `plugins/` or to unrelated appdata; can write to run artifacts only.                       | P1=1 P2=4 P3=1 P4=3 (Total=9) | M / High      |
185: | SEC-03 | **Secrets hygiene: redaction + env-var indirection for settings**<br>Rationale: Avoid storing API keys in DB/logs/manifests; keep runs auditable without leaking secrets.<br>Dependencies: META-01, OBS-03.<br>Improved: security and safe diagnostics.<br>Risked: misconfiguration if env vars missing.<br>Enforce: settings loader recognizes `{"$env":"NAME"}`; redacts matching patterns in logs and manifests.<br>Regression detection: secret-in-log tests; if secret appears => **DO_NOT_SHIP**.<br>Acceptance test: run with fake key; manifests store only `$env` reference; logs show “[REDACTED]”.                                                                                                                                | P1=0 P2=3 P3=1 P4=2 (Total=6) | M / Low       |
186: | SEC-04 | **Optional PII scanning/redaction plugin with consent toggles**<br>Rationale: Local-first doesn’t eliminate privacy risk; derived artifacts (reports/prompts) may leak PII.<br>Dependencies: EXT-04 capability model, UX-02 consent UI.<br>Improved: privacy and safer optional sharing.<br>Risked: false positives/negatives; must be clearly labeled as heuristic when applicable.<br>Enforce: dedicated plugin; outputs findings with confidence and evidence; redaction applied only when enabled.<br>Regression detection: PII fixture tests; if misses known PII => **DO_NOT_SHIP** for the plugin (not entire engine).<br>Acceptance test: dataset with known emails/SSNs; report shows detected fields; exports redact when toggled. | P1=1 P2=3 P3=2 P4=2 (Total=8) | L / Med       |
187: | SEC-05 | **Secure artifact serving: headers + CSP + path traversal tests**<br>Rationale: Artifact downloads must not allow path traversal or unsafe rendering in browser contexts.<br>Dependencies: QA-05 optional.<br>Improved: UI security robustness.<br>Risked: might break inline preview for some file types unless handled carefully.<br>Enforce: `ui/server.py` download endpoints use strict path join + content-type; add CSP headers and `X-Content-Type-Options`.<br>Regression detection: traversal test; if can read outside run dir => **DO_NOT_SHIP**.<br>Acceptance test: request `../state.sqlite`; server returns 404/403; all responses include security headers.                                                                 | P1=0 P2=3 P3=0 P4=2 (Total=5) | S / Low       |
188: | SEC-06 | **Tenancy boundary tests + enforced tenant_id scoping everywhere**<br>Rationale: Tenancy is feature-flagged ; if enabled, it must not be a footgun.<br>Dependencies: QA-04.<br>Improved: security boundaries and correctness.<br>Risked: schema/query churn.<br>Enforce: every DB query must include tenant scope; file paths include tenant partitioning; add unit tests for scoping.<br>Regression detection: cross-tenant access tests; any cross-read => **DO_NOT_SHIP**.<br>Acceptance test: create two tenants; run in A; B cannot query/trace/download artifacts.                                                                                                                                                                    | P1=0 P2=4 P3=1 P4=2 (Total=7) | M / Med       |
189: | SEC-07 | **Optional at-rest encryption hook/guidance for appdata**<br>Rationale: Some environments require encryption even for local-only tools.<br>Dependencies: FND-05 backups must work with encryption strategy.<br>Improved: security compliance posture.<br>Risked: complexity and key-loss risk.<br>Enforce: minimal hook: allow external encrypted filesystem path + explicit warning; optionally integrate SQLCipher only if permitted (no evidence in repo today).<br>Regression detection: backup/restore with encrypted path must work; if not => **DO_NOT_SHIP** for this feature.<br>Acceptance test: configure appdata to encrypted volume; runs succeed and backups restore.                                                          | P1=0 P2=3 P3=0 P4=1 (Total=4) | L / Med       |
190: 
191: ---
192: 
193: # VIII. Performance: Windows 11 + WSL2 + large workloads
194: 
195: Current evidence: dataset accessor loads from SQLite into pandas and caches in-memory DataFrame (`self._df`) ; SQLite is configured for WAL+busy timeout which is good for concurrency but still sensitive to large queries. 
196: 
197: | ID      | Recommendation details                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |                 Pillar scores | Effort / Risk |
198: | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------: | ------------- |
199: | PERF-01 | **Streaming dataset access + enforce max in-memory loads**<br>Rationale: Cached full-DF loads are a predictable failure mode on large datasets under WSL/Windows memory pressure. <br>Dependencies: plugin updates to adopt streaming API.<br>Improved: performance and stability.<br>Risked: plugin rewrite required for some analyses.<br>Enforce: add `iter_batches()` in `dataset_io.py`; default `df()` requires explicit opt-in for full load or capped row limit.<br>Regression detection: large dataset fixture must not exceed memory; if it does => **DO_NOT_SHIP**.<br>Acceptance test: run scan stats on 5M-row dataset without OOM; measured peak RSS stays under configured cap. | P1=3 P2=0 P3=2 P4=1 (Total=6) | M / Med       |
200: | PERF-02 | **SQLite indices + ANALYZE + query plan checks**<br>Rationale: Deterministic performance requires deliberate indexing and plan stability.<br>Dependencies: none (but best with OBS-07).<br>Improved: throughput and latency.<br>Risked: migration time on existing DBs.<br>Enforce: add indices for common filters (dataset_version_id, row_index, plugin_results keys); run `ANALYZE` after ingest.<br>Regression detection: query plan snapshot test; if plan regresses => **DO_NOT_SHIP**.<br>Acceptance test: scan-stat queries remain within expected time on benchmark dataset.                                                                                                           | P1=3 P2=0 P3=1 P4=1 (Total=5) | S / Low       |
201: | PERF-03 | **Materialized derived columns/views with invalidation**<br>Rationale: Avoid repeated expensive parsing (e.g., numeric coercions) across plugins.<br>Dependencies: PERF-02, META-03 (fingerprints for invalidation correctness).<br>Improved: large-run performance.<br>Risked: cache invalidation complexity.<br>Enforce: create derived tables keyed by dataset_version_id; invalidate when dataset changes.<br>Regression detection: invalidation test; if stale values => **DO_NOT_SHIP**.<br>Acceptance test: repeated run of same analyses gets faster; outputs identical.                                                                                                                | P1=3 P2=0 P3=1 P4=1 (Total=5) | L / Med       |
202: | PERF-04 | **Parallelism tuning knobs + concurrency classes**<br>Rationale: Windows/WSL IO contention can make “more threads” slower; users need control.<br>Dependencies: OBS-02 (measure).<br>Improved: performance under varied workloads.<br>Risked: misconfiguration if defaults poor.<br>Enforce: add `--workers` and per-plugin concurrency class (CPU/IO); default sensible for 64GB RAM.<br>Regression detection: benchmark must not regress with defaults; if does => **DO_NOT_SHIP**.<br>Acceptance test: changing workers scales within expected bounds; timeline shows reduced queue time.                                                                                                    | P1=2 P2=0 P3=1 P4=1 (Total=4) | M / Low       |
203: | PERF-05 | **Optional GPU acceleration hook (stretch; no evidence today)**<br>Rationale: User environment has RTX 4090, but repo shows no GPU path; keep strictly optional to preserve determinism and enterprise constraints.<br>Dependencies: none (future).<br>Improved: performance only if adopted.<br>Risked: dependency and reproducibility risk; should not be default.<br>Enforce: plugin-level opt-in capability; record GPU details in manifest when used.<br>Regression detection: CPU baseline remains default; if GPU path silently triggers => **DO_NOT_SHIP**.<br>Acceptance test: enabling GPU flag changes manifest and produces identical results within tolerance where applicable.    | P1=2 P2=0 P3=0 P4=0 (Total=2) | L / High      |
204: | PERF-06 | **WSL/Windows IO guidance + temp dir placement warnings**<br>Rationale: Many perf issues are caused by storing large datasets on slow Windows-mounted paths inside WSL.<br>Dependencies: none.<br>Improved: operator success rate without deep tuning.<br>Risked: none meaningful.<br>Enforce: startup checks detect path location and warn; recommend storing appdata on Linux filesystem under WSL.<br>Regression detection: warning tests; if false positives => **DO_NOT_SHIP**.<br>Acceptance test: when appdata on `/mnt/c`, UI shows warning and offers one-click “move appdata” instructions (no admin).                                                                                | P1=2 P2=0 P3=0 P4=1 (Total=3) | S / Low       |
205: 
206: ---
207: 
208: # IX. QA / Test strategy: Determinism + contracts + regressions
209: 
210: Current evidence: tests exist for upload limits and plugin discovery, suggesting a test harness foundation is present. 
211: 
212: | ID    | Recommendation details                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |                 Pillar scores | Effort / Risk |
213: | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------: | ------------- |
214: | QA-01 | **Golden fixture runs for deterministic report + trace outputs**<br>Rationale: Determinism is a core guarantee; golden fixtures catch accidental drift quickly.<br>Dependencies: META-01, EXEC-03.<br>Improved: correctness and citeability.<br>Risked: fixture brittleness if timestamps not normalized.<br>Enforce: canonicalize timestamps in fixtures or separate “content hash” fields from timestamps.<br>Regression detection: fixture diff in CI; any unexpected diff => **DO_NOT_SHIP**.<br>Acceptance test: running fixture twice yields identical output hashes; CI compares committed golden outputs.          | P1=0 P2=0 P3=3 P4=3 (Total=6) | M / Med       |
215: | QA-02 | **Schema contract tests for plugins + report schema validation**<br>Rationale: Prevent schema drift and broken UI rendering.<br>Dependencies: META-05.<br>Improved: accuracy and long-term stability.<br>Risked: initial failures reveal existing drift (expected).<br>Enforce: CI validates all plugin outputs against `output_schema` and validates report against `docs/report.schema.json`.<br>Regression detection: schema mismatch => **DO_NOT_SHIP**.<br>Acceptance test: any plugin output that violates schema fails the run and CI.                                                                              | P1=0 P2=0 P3=3 P4=3 (Total=6) | S / Low       |
216: | QA-03 | **Sandbox escape test suite (fs/network/eval/pickle/shell)**<br>Rationale: Runner already blocks risky primitives ; lock this behavior with tests so it never regresses.<br>Dependencies: EXT-05/SEC-02 for write restrictions if implemented.<br>Improved: security assurance.<br>Risked: none meaningful.<br>Enforce: dedicated tests execute a “malicious plugin” fixture and assert failures on forbidden actions.<br>Regression detection: any escape => **DO_NOT_SHIP**.<br>Acceptance test: plugin tries `socket.connect`, `eval`, `pickle.loads`, `subprocess.run`, and writing outside run dir; all are blocked. | P1=0 P2=4 P3=0 P4=1 (Total=5) | S / Low       |
217: | QA-04 | **Migration upgrade/rollback tests + backup verification**<br>Rationale: Schema changes are inevitable; tests prevent silent corruption or lost provenance.<br>Dependencies: FND-05.<br>Improved: reliability and citeability of stored history.<br>Risked: time to build robust fixtures.<br>Enforce: CI runs migrations forward/back on fixture DB; verifies invariants and backups.<br>Regression detection: migration failure => **DO_NOT_SHIP**.<br>Acceptance test: fixture DB upgrades; old runs still load; trace graph intact.                                                                                    | P1=0 P2=1 P3=2 P4=2 (Total=5) | M / Med       |
218: | QA-05 | **UI e2e smoke tests (Playwright)**<br>Rationale: Prevent UI regressions that break frictionless core workflow.<br>Dependencies: UX-02, UX-04, OBS-05.<br>Improved: reliability of the user path.<br>Risked: CI flakiness if not stabilized.<br>Enforce: run in headless mode; verify upload/run/report/trace/download paths.<br>Regression detection: any failure => **DO_NOT_SHIP**.<br>Acceptance test: full UI workflow completes on fixture dataset and produces expected report.                                                                                                                                     | P1=1 P2=0 P3=1 P4=2 (Total=4) | M / Med       |
219: | QA-06 | **Performance regression gates in CI (if regress => DO_NOT_SHIP)**<br>Rationale: Performance regressions are functional regressions at large scale.<br>Dependencies: OBS-07, PERF-02.<br>Improved: sustained throughput.<br>Risked: noisy measurements if not controlled.<br>Enforce: stable environment settings; compare medians; use tolerances.<br>Regression detection: >X% slower => **DO_NOT_SHIP**.<br>Acceptance test: ingest+scan stats benchmark stays within budget across commits.                                                                                                                            | P1=3 P2=0 P3=0 P4=1 (Total=4) | M / Med       |
220: 
221: ---
222: 
223: # X. Roadmap: Phased plan Phase 0–3
224: 
225: This is sequencing guidance using the recommendation IDs above. No new requirements beyond the tables.
226: 
227: | Phase   | Scope                                                                                                              | Regression guards                                                                                  |
228: | ------- | ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
229: | Phase 0 | Safety baselines: FND-02, EXEC-06, EXEC-07, EXT-02, SEC-05, QA-02, QA-03, PERF-02                                  | CI schema validation + sandbox escape tests + smoke workflow must pass; any failure => DO_NOT_SHIP |
230: | Phase 1 | Provenance + crash recovery: FND-01, FND-03, FND-04, FND-05, META-01, META-03, OBS-01, OBS-03                      | Golden fixture (QA-01) starts here; add kill/restart tests; hash verification required             |
231: | Phase 2 | Replay + caching + UX surfacing: FND-06, EXEC-01, EXEC-03, EXEC-04, EXEC-05, META-04, UX-01/02/04/05/08, OBS-02/04 | Deterministic output hashing gates; UI e2e tests required for core workflow                        |
232: | Phase 3 | Extensions lifecycle + privacy options: EXT-01/03/04/05/07/08, SEC-03/04/06/07, PERF-01/03/04/06                   | Compatibility tests + tenancy boundary tests + perf regression gates expanded                      |
233: 
234: ---
235: 
236: ## Top-20 quick wins
237: 
238: Highest total score with Effort ∈ {S, M}. (Deterministic ordering computed from the scoring table.)
239: 
240: | Rank | ID      | Recommendation                                                            | Total | Effort | Risk |
241: | ---- | ------- | ------------------------------------------------------------------------- | ----- | ------ | ---- |
242: | 1    | FND-01  | Atomic run directory + crash-safe journaling                              | 10    | M      | Med  |
243: | 2    | FND-03  | Harden upload CAS: verify-on-write + refcount + quarantine                | 10    | M      | Med  |
244: | 3    | META-01 | Persist canonical run_manifest.v1 capturing input+config+env              | 9     | M      | Low  |
245: | 4    | FND-05  | Online backup/restore + retention for state.sqlite                        | 9     | M      | Low  |
246: | 5    | META-03 | Store execution_fingerprint per plugin (code+manifest+settings+input)     | 9     | M      | Low  |
247: | 6    | EXEC-01 | Idempotent plugin caching keyed by hashes + --force/--reuse               | 9     | M      | Med  |
248: | 7    | EXT-05  | Sandbox policy redesign: read/write allowlists + narrow DB access         | 9     | M      | High |
249: | 8    | SEC-02  | Harden sandbox FS policies (read/write split) + prevent plugin dir writes | 9     | M      | High |
250: | 9    | META-05 | Schema versioning+compat checks (config/output/report) + hashes           | 8     | M      | Med  |
251: | 10   | META-08 | Export 'Repro pack' zip (manifest+schemas+logs+report)                    | 8     | M      | Low  |
252: | 11   | EXEC-05 | Replay mode: verify hashes then reuse cached results                      | 9     | L      | Med  |
253: | 12   | EXEC-02 | Per-plugin timeout + memory budget enforcement                            | 8     | M      | Med  |
254: | 13   | EXEC-03 | Deterministic persistence order + per-plugin derived seeds                | 8     | S      | Low  |
255: | 14   | EXEC-04 | Failure containment: temp dirs + commit-on-validate only                  | 8     | M      | Med  |
256: | 15   | EXEC-06 | Pre-run config validation against plugin schemas (deterministic defaults) | 8     | S      | Low  |
257: | 16   | SEC-04  | Optional PII scanning/redaction plugin with consent toggles               | 8     | L      | Med  |
258: | 17   | PERF-01 | Streaming dataset access + enforce max in-memory loads                    | 6     | M      | Med  |
259: | 18   | FND-06  | Deterministic run_fingerprint and optional deterministic IDs              | 8     | M      | Med  |
260: | 19   | OBS-03  | Diagnostics bundle export (zip) with manifests+logs+env+schema            | 8     | M      | Low  |
261: | 20   | QA-02   | Schema contract tests for plugins + report schema validation              | 6     | S      | Low  |
262: 
263: > Note: quick-win rank table includes some L items in the computed list above (EXEC-05, SEC-04). If you want the strict interpretation (S/M only), drop those two and promote the next highest S/M items (FND-04, META-07).
264: 
265: ## Top-20 big bets
266: 
267: Highest total score regardless of effort.
268: 
269: | Rank | ID      | Recommendation                                                            | Total | Effort | Risk |
270: | ---- | ------- | ------------------------------------------------------------------------- | ----- | ------ | ---- |
271: | 1    | FND-01  | Atomic run directory + crash-safe journaling                              | 10    | M      | Med  |
272: | 2    | FND-03  | Harden upload CAS: verify-on-write + refcount + quarantine                | 10    | M      | Med  |
273: | 3    | META-01 | Persist canonical run_manifest.v1 capturing input+config+env              | 9     | M      | Low  |
274: | 4    | FND-05  | Online backup/restore + retention for state.sqlite                        | 9     | M      | Low  |
275: | 5    | META-03 | Store execution_fingerprint per plugin (code+manifest+settings+input)     | 9     | M      | Low  |
276: | 6    | EXEC-01 | Idempotent plugin caching keyed by hashes + --force/--reuse               | 9     | M      | Med  |
277: | 7    | EXEC-05 | Replay mode: verify hashes then reuse cached results                      | 9     | L      | Med  |
278: | 8    | EXT-05  | Sandbox policy redesign: read/write allowlists + narrow DB access         | 9     | M      | High |
279: | 9    | SEC-02  | Harden sandbox FS policies (read/write split) + prevent plugin dir writes | 9     | M      | High |
280: | 10   | META-05 | Schema versioning+compat checks (config/output/report) + hashes           | 8     | M      | Med  |
281: | 11   | META-08 | Export 'Repro pack' zip (manifest+schemas+logs+report)                    | 8     | M      | Low  |
282: | 12   | EXEC-02 | Per-plugin timeout + memory budget enforcement                            | 8     | M      | Med  |
283: | 13   | EXEC-03 | Deterministic persistence order + per-plugin derived seeds                | 8     | S      | Low  |
284: | 14   | EXEC-04 | Failure containment: temp dirs + commit-on-validate only                  | 8     | M      | Med  |
285: | 15   | EXEC-06 | Pre-run config validation against plugin schemas (deterministic defaults) | 8     | S      | Low  |
286: | 16   | EXT-03  | Offline install/update/rollback via signed plugin bundles                 | 8     | L      | Med  |
287: | 17   | SEC-04  | Optional PII scanning/redaction plugin with consent toggles               | 8     | L      | Med  |
288: | 18   | OBS-03  | Diagnostics bundle export (zip) with manifests+logs+env+schema            | 8     | M      | Low  |
289: | 19   | FND-06  | Deterministic run_fingerprint and optional deterministic IDs              | 8     | M      | Med  |
290: | 20   | FND-04  | Startup integrity checks: PRAGMA integrity_check + orphan cleanup         | 8     | S      | Low  |
291: 
292: ---
293: 
294: ## Red-team failure scenarios
295: 
296: ### A) Red-team as a User: failure scenarios, detection, mitigation
297: 
298: | Scenario                                           | Detection signals                                                 | Mitigation                                                                                  |
299: | -------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
300: | Upload wrong dataset (fatigue)                     | Column/schema mismatch vs previous run; hash differs unexpectedly | UX-02 preview + show sha256 + schema diff before run; store “expected schema” per project   |
301: | Confuse run_seed semantics (0 vs random)           | Repeated runs unexpectedly identical/different                    | UX-04 confirm page shows resolved seed; META-07 shows determinism badge + seed derivation   |
302: | Assume full data when sampling occurred            | Sample fraction < 1 recorded                                      | FND-08 + UX-08 show “SAMPLED” banner and record sample method/seed in manifest              |
303: | Partial run mistaken as complete                   | Some plugins missing results                                      | UX-05 timeline shows missing plugins; run status becomes PARTIAL with explicit missing list |
304: | Plugin silently blocked by permissions             | Capability denied                                                 | EXT-04/UX-04 show permission denial before run; clear “not executed” state in run detail    |
305: | Think modeled outputs are measured                 | Missing evidence links                                            | META-02 + UX-08 force modeled/measured labeling; show “no evidence” explicitly              |
306: | Download artifact, later can’t prove origin        | No hash/manifest attached                                         | META-04 registry + embed run_id+hash in filename; repro pack export                         |
307: | Accidental enablement of optional network features | network flag true in manifest                                     | SEC-01 forces explicit flag; UI confirmation includes warning and records consent           |
308: | Misapply preset to wrong plugin version            | Schema hash mismatch                                              | UX-03 includes schema hash gating + diff preview; refuse apply if incompatible              |
309: | Accessibility gaps cause misclicks                 | Keyboard traps; missing labels                                    | UX-06 + QA-05 a11y checks; ensure all destructive actions require confirmation              |
310: 
311: ### B) Red-team as a System Admin/Operator: failure scenarios, detection, mitigation
312: 
313: | Scenario                                            | Detection signals                   | Mitigation                                                                           |
314: | --------------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------ |
315: | SQLite corruption after crash/power loss            | integrity_check fails; missing rows | FND-04 integrity gate + FND-05 backup/restore + crash-safe run journaling            |
316: | Plugin persists by writing into `plugins/`          | Plugin code hash changes            | EXT-05/SEC-02 read/write sandbox split + META-03 code fingerprinting                 |
317: | Schema/config drift breaks UI or report             | schema validation failures          | META-05 + QA-02 contract tests; block incompatible plugins                           |
318: | Migration fails mid-upgrade                         | schema_version inconsistent         | QA-04 migration tests + FND-05 backups prior to migrations                           |
319: | Disk fills from artifacts/logs                      | low disk warnings                   | OBS-05 disk checks + OBS-06 retention + export-before-purge                          |
320: | Concurrency causes lock contention                  | “database is locked” errors         | PERF-04 tuning + improve busy handling + per-run checkpoints to resume               |
321: | Tenancy enabled but data leaks across tenants       | cross-tenant query results          | SEC-06 enforced scoping + tests; deny tenancy enable until passing self-check        |
322: | Vector store enabled but extension missing/unstable | vector init errors                  | META-06 records feature flags; EXT-06 health checks; disable vector store by default |
323: | Replay shows tampered artifacts as valid            | hash mismatch not checked           | EXEC-05 replay verifies hashes; fail closed on mismatch                              |
324: | Performance regression unnoticed                    | TTFR increases                      | OBS-07 perf gates + OBS-02 dashboard; stop shipping on regression                    |
325: 
326: ---
327: 
328: ## Minimal canonical metadata schema proposal
329: 
330: This is a **new** canonical schema (`run_manifest.v1`) that complements `docs/report.schema.json`. 
331: 
332: | Field                           | Type             | Example                                                     | Notes                                                  |
333: | ------------------------------- | ---------------- | ----------------------------------------------------------- | ------------------------------------------------------ |
334: | `schema_version`                | string           | `"run_manifest.v1"`                                         | Versioned contract; required                           |
335: | `run_id`                        | string           | `"a3f9...c1"`                                               | Current run ID is UUID4 hex                           |
336: | `run_fingerprint`               | string           | `"sha256:...`                                               | Deterministic identity of (input+config+plugins+flags) |
337: | `created_at`                    | string (RFC3339) | `"2026-02-06T10:11:12Z"`                                    | Use UTC consistently                                   |
338: | `input.upload_sha256`           | string           | `"sha256:...`                                               | Upload already computes sha256                        |
339: | `input.original_filename`       | string           | `"sales.xlsx"`                                              | For UX only; not identity                              |
340: | `input.bytes`                   | int              | `12345678`                                                  |                                                        |
341: | `ingest.row_count`              | int              | `1048576`                                                   | Must be measured                                       |
342: | `ingest.column_count`           | int              | `42`                                                        |                                                        |
343: | `config.resolved`               | object           | `{...}`                                                     | Fully resolved config after defaults                   |
344: | `config.sha256`                 | string           | `"sha256:...`                                               | Hash of resolved config (stable JSON)                 |
345: | `determinism.run_seed`          | int              | `12345`                                                     | Explicit; show derived seeds per plugin                |
346: | `plugins[]`                     | array            | `[{"id":"analysis_scan_statistics","version":"0.1.0",...}]` | Include code hash + schema hashes                      |
347: | `features.network_enabled`      | bool             | `false`                                                     | Default false; runner guards network                  |
348: | `features.vector_store_enabled` | bool             | `false`                                                     | Feature-flagged                                       |
349: | `execution.workers`             | int              | `8`                                                         | From pipeline concurrency settings                     |
350: | `artifacts[]`                   | array            | `[{ "path":"report.json","sha256":"...","bytes":...}]`      | Evidence registry                                      |
351: | `results.summary`               | object           | `{ "status":"COMPLETED", "failed_plugins":[...] }`          | Completion and failures                                |
352: 
353: ---
354: 
355: ## Minimal processing lineage model
356: 
357: The repository already has lineage concepts (entities + edges) and UI trace views (evidence in migrations + UI templates). 
358: 
359: **Canonical model (input → outputs)**
360: 
361: | Step | Entity           | Key IDs                     | Stored evidence                                |
362: | ---- | ---------------- | --------------------------- | ---------------------------------------------- |
363: | 1    | Raw upload       | `upload_sha256`             | CAS blob + size + first_seen                   |
364: | 2    | Dataset version  | `dataset_version_id`        | Ingest row/col counts + schema hash            |
365: | 3    | Run              | `run_id`, `run_fingerprint` | `run_manifest.json` + resolved config          |
366: | 4    | Plugin execution | `plugin_id`, `execution_id` | start/end, stdout/stderr, exit code            |
367: | 5    | Plugin result    | `result_id`                 | output JSON + references/evidence + debug      |
368: | 6    | Report           | `report_sha256`             | report.json + schema validation result         |
369: | 7    | Artifacts        | `artifact_sha256`           | registry + producer + mime + bytes             |
370: | 8    | Trace graph      | entity+edge IDs             | links: run → plugin exec → results → artifacts |
371: 
372: ---
373: 
374: ## Extension manager redesign
375: 
376: This is a **minimal viable spec** for a local-only extension manager aligned to the codebase’s plugin schema system. 
377: 
378: ### Minimal viable spec
379: 
380: | Area        | MVP behavior                                                                       | Enforcement location                            |
381: | ----------- | ---------------------------------------------------------------------------------- | ----------------------------------------------- |
382: | Discovery   | List installed plugins with id/version/type and status                             | `core/plugin_manager.py`, UI `/plugins`         |
383: | Validate    | `plugins validate` validates schema + imports entrypoint + dry-run config defaults | CLI + `plugin_manager.py`                       |
384: | Permissions | Display required capabilities and deny by default for risky ones                   | plugin.yaml + `plugin_runner.py` + `sandbox.py` |
385: | Install     | Offline: import a `.zip` bundle and install into appdata                           | new `core/plugin_registry.py`                   |
386: | Update      | Install new version side-by-side                                                   | registry pointer switch                         |
387: | Rollback    | Switch pointer to previous version deterministically                               | registry pointer switch + audit event           |
388: | Health      | Show per-plugin health state + last validation result                              | manager UI + `health()` contract                |
389: 
390: ### Stretch goals
391: 
392: | Area                           | Stretch behavior                            | Why                               |
393: | ------------------------------ | ------------------------------------------- | --------------------------------- |
394: | Signed bundles                 | Signature verification (hash-based)         | Supply-chain hygiene              |
395: | Capability negotiation         | Policy-based approvals (per project/tenant) | Reduce prompts + prevent mistakes |
396: | Sandboxing profiles            | Predefined sandbox profiles per plugin type | Safer defaults                    |
397: | Automated compatibility matrix | Prevent known-bad plugin combinations       | Admin pain reduction              |
398: 
399: ---
400: 
401: ## UI/UX sketch in text
402: 
403: ### Home: Activity Dashboard
404: 
405: ```
406: +--------------------------------------------------------------+
407: | Statistic Harness — Activity                                  |
408: | [New Run] [Upload Dataset] [Manage Plugins] [Diagnostics]      |
409: +--------------------------------------------------------------+
410: | Recent Runs (filter: [status] [plugin] [dataset] [date])      |
411: |--------------------------------------------------------------|
412: | Run ID   Fingerprint   Status   TTFR   Duration   Determinism |
413: | a3f9..   9c12..        OK       2.1s   01:12      ✅ seeded    |
414: | b771..   12aa..        PARTIAL  3.8s   00:45      ✅ seeded    |
415: | c020..   77ef..        FAILED   —      00:08      ✅ seeded    |
416: +--------------------------------------------------------------+
417: | Alerts:                                                      |
418: | - Low disk space warning (3.2GB free)                         |
419: | - 2 plugins failing health checks                             |
420: +--------------------------------------------------------------+
421: ```
422: 
423: ### Input ingest/status panel
424: 
425: ```
426: +-------------------- Upload Dataset --------------------------+
427: | Select file: [Choose...]  (CSV/XLSX)                          |
428: | Preview: 20 rows  | Columns: 42 | Rows (estimated): 1,048,576 |
429: | SHA-256: sha256:...   Size: 12.3MB   Format: XLSX            |
430: |                                                                  |
431: | [x] Allow sampling for analysis (defaults OFF)                   |
432: |                                                                  |
433: | [Upload]  [Cancel]                                              |
434: +------------------------------------------------------------------+
435: ```
436: 
437: ### Extension manager main screen
438: 
439: ```
440: +-------------------- Plugins -----------------------------------+
441: | Tabs: [Installed] [Available] [Updates] [Permissions] [Logs]     |
442: +------------------------------------------------------------------+
443: | Installed                                                       |
444: |------------------------------------------------------------------|
445: | Plugin ID              Version   Status   Permissions   Actions  |
446: | analysis_scan_statistics 0.1.0    OK       DB_READ       [Disable]
447: | llm_prompt_builder      0.2.1    BLOCKED  NETWORK       [Review..]
448: | report_bundle           1.0.0    OK       FS_WRITE(run)  [Disable]
449: +------------------------------------------------------------------+
450: | [Install bundle (.zip)]   [Validate all]                         |
451: +------------------------------------------------------------------+
452: ```
453: 
454: ### Run/job detail view
455: 
456: ```
457: +---------------- Run Detail ------------------------------------+
458: | Run: a3f9..   Fingerprint: 9c12..   Status: COMPLETED ✅         |
459: | Input: sha256:...  Rows: 1,048,576  Columns: 42                 |
460: | Flags: network=OFF  vector=OFF  tenancy=OFF                      |
461: | Seed: 12345  Workers: 8                                          |
462: +------------------------------------------------------------------+
463: | Timeline (TTFR: 2.1s)                                            |
464: | ingest ███  transform ██  analysis ███████  report ███            |
465: +------------------------------------------------------------------+
466: | Plugins                                                         |
467: | - ingest_tabular         OK   4.2s   output hash: ...             |
468: | - analysis_scan_statistics OK  12.1s  output hash: ...             |
469: | - report_bundle          OK   2.0s   report hash: ...             |
470: +------------------------------------------------------------------+
471: | Artifacts (click to verify/download)                             |
472: | report.json  sha256:...  122KB  (produced by report_bundle)      |
473: | diag.zip     sha256:...  2.1MB                                    |
474: +------------------------------------------------------------------+
475: | Evidence                                                        |
476: | Modeled vs measured: [MEASURED] for row counts, [MODELED] for ...|
477: | Confidence: 0.87   Evidence links: [trace graph] [row trace]      |
478: +------------------------------------------------------------------+
479: | Actions: [Replay] [Export repro pack] [Download diagnostics]      |
480: +------------------------------------------------------------------+
481: ```
482: 
483: ---
484: 
485: ## Open questions
486: 
487: |  # | Question                                                                                                                              |
488: | -: | ------------------------------------------------------------------------------------------------------------------------------------- |
489: |  1 | Should the system treat `run_seed=0` as a literal seed or as “auto-derive from fingerprint”? (UI currently shows a run_seed field.)  |  auto-derive
490: |  2 | Are plugins expected to ever write outside run directories, or can we break that behavior to enable read/write allowlists?            | you can break
491: |  3 | Is multi-tenancy intended for real use or only experiments behind the flag?                                                          | dont even experiment, single tenant only
492: |  4 | What is the intended retention policy for uploads and run artifacts in appdata?                                                       |  60 days, after that only the data telling us full path and when something was uploaded and the run artifacts if recommended
493: |  5 | Should report artifacts be considered immutable once written (append-only), and enforced cryptographically?                           |  whatever is optimal for the 4 pillars
494: |  6 | Which plugin outputs are “measured” vs “modeled” today, and where is that distinction documented?                                     |  most all are measured, only ones containing model in their name are models.
495: |  7 | Do we need a “project” entity as a first-class object (beyond runs/datasets) with expected schema + presets?                          |  No, it is dataset based. however we do need an ERP entity to contain the known issues and defaults for any dataset from the same erp
496: |  8 | What is the largest expected dataset size (rows/cols/bytes), and what is the target TTFR budget?                                      |  ive seen up to 10 million rows so it needs to be capable of streaming an optimal number of rows, storing insights on that and so on to then come up with global insights.
497: |  9 | Should the system support “headless” batch mode for CI (no UI) with deterministic export bundles?                                     |  yes. ui is nice for me occasionally especially when uploading and monitoring status and configs. cli sucks for that but is great for testing
498: | 10 | Is the vector store meant to be part of default workflows or strictly optional?                                                      |  default workflow
499: | 11 | Do we require cryptographic signing of plugin bundles, or is hash-based integrity sufficient for local-only?                          |  hash based is fine
500: | 12 | What is the authoritative source of “engine version” for compatibility checks (package version vs git hash)?                          |  whatever is optimal for the 4 pillars
501: 
502: ---
503: 
504: THREAD: STATISTIC-HARNESS-ADVERSARIAL-REDESIGN
505: CHAT_ID: N/A
506: TS: 2026-02-06
````

---

## Appendix G — Codex Statistics Harness Blueprint (additional plugins + eval gates)

````markdown
  1: # Codex CLI Implementation Blueprint — Statistics Harness (ninjra/statistics_harness)
  2: 
  3: Source repo: https://github.com/ninjra/statistics_harness  
  4: TS: 2026-02-12 21:48:09 MST
  5: 
  6: ## 0) Hard constraints (must follow)
  7: Repo evidence:
  8: - README: https://github.com/ninjra/statistics_harness/blob/main/README.md
  9: - CLI (plugin discovery/validate/eval paths): https://github.com/ninjra/statistics_harness/blob/main/src/statistic_harness/cli.py
 10: 
 11: Constraints:
 12: - Keep local-only defaults intact.
 13: - Integrate new plugins via existing `plugins/` discovery and `PluginManager`.
 14: - Preserve PII tagging/anonymization behavior mentioned in README.
 15: 
 16: ## 1) What to implement (from recommended ideas)
 17: ### 1.1 Changepoint detection plugin (low risk, high leverage)
 18: Plugin: `changepoint_detection_v1`
 19: - Inputs: time series derived from normalized dataset (durations, queue depth, throughput)
 20: - Outputs:
 21:   - changepoint timestamps
 22:   - segments summary (mean/median/p95 per segment)
 23:   - deltas used by downstream plugins
 24: 
 25: ### 1.2 Process mining plugin family (sequence-aware bottlenecks)
 26: Plugins:
 27: - `process_mining_discovery_v1`
 28: - `process_mining_conformance_v1`
 29: 
 30: Approach:
 31: - Start dependency-light:
 32:   - build frequency graphs using `networkx`
 33:   - implement a minimal discovery algorithm
 34: - Optional: add `pm4py` as an extra group if desired later.
 35: 
 36: ### 1.3 Causal inference plugin (assumption-explicit recommendations)
 37: Plugin: `causal_recommendations_v1`
 38: - Requires a user-provided causal graph config to start.
 39: - Must output:
 40:   - effect estimate (or “no identification”)
 41:   - assumptions list
 42:   - at least one refutation result
 43: 
 44: ### 1.4 Evaluation gate expansion (extend existing `eval`)
 45: Repo evidence: `eval` and `make-ground-truth-template` exist in CLI:
 46: https://github.com/ninjra/statistics_harness/blob/main/src/statistic_harness/cli.py
 47: 
 48: Extend evaluation to include:
 49: - expected changepoints (with tolerance)
 50: - expected intervention candidates
 51: - expected evidence windows (row ranges or time windows)
 52: 
 53: ## 2) Codex execution steps (must scan full repo)
 54: Codex MUST:
 55: 1) `git ls-files` → `docs/_codex_repo_manifest.txt`
 56: 2) Locate and read:
 57:    - `core/plugin_manager.py`
 58:    - `core/pipeline.py`
 59:    - report builder and evaluation modules
 60: 3) Catalog existing plugins:
 61:    - write `docs/_codex_plugin_catalog.md`
 62: 4) Identify where to store new plugin artifacts and how they flow into `report.json` and `report.md`.
 63: 
 64: Stop if scan cannot be completed.
 65: 
 66: ## 3) Implementation plan (ordered)
 67: ### Phase A — schemas + artifacts contracts
 68: - Add schemas:
 69:   - `docs/schemas/changepoints.schema.json`
 70:   - `docs/schemas/process_mining.schema.json`
 71:   - `docs/schemas/causal.schema.json`
 72: 
 73: ### Phase B — implement `changepoint_detection_v1`
 74: - Add synthetic tests with seeded changepoints.
 75: - Ensure chunked processing for large datasets.
 76: 
 77: ### Phase C — implement process mining plugins
 78: - Build event log abstraction:
 79:   - `case_id`, `activity`, `start_ts`, `end_ts`, `resource`
 80: - Produce frequency graph + bottleneck report.
 81: 
 82: ### Phase D — implement causal plugin
 83: - Start with assumption-required mode (explicit config).
 84: - Add refutation utilities.
 85: 
 86: ### Phase E — expand evaluation
 87: - Update ground truth templates and `eval` checks.
 88: 
 89: ## 4) Tests (must add)
 90: - `stat-harness plugins validate` passes with new plugins.
 91: - Schema validation for all plugin outputs.
 92: - `stat-harness eval` checks new outputs (changepoints, interventions, evidence slices).
 93: - Performance smoke: large-file run does not exceed memory by unbounded copies.
 94: 
 95: ## 5) Acceptance criteria (objective)
 96: - New plugins appear in `stat-harness list-plugins`.
 97: - `stat-harness run` executes them and writes artifacts into run dir.
 98: - `stat-harness eval` validates runs using updated templates.
 99: 
100: ## 6) Evidence labels
101: - QUOTE: local-only posture + env flags: https://github.com/ninjra/statistics_harness/blob/main/README.md
102: - QUOTE: CLI plugin/eval integration points: https://github.com/ninjra/statistics_harness/blob/main/src/statistic_harness/cli.py
103: - NO EVIDENCE: assistant did not read every file; Codex is instructed to perform a full scan.
104: 
105: ## 7) Determinism notes
106: Use stable ordering for output lists and stable run seeds (`--run-seed`).
````

---

## Appendix H — Plugin Data Access Matrix (baseline)

````markdown
  1: # Plugin Data Access Matrix
  2: 
  3: Generated by `scripts/plugin_data_access_matrix.py`.
  4: 
  5: | Plugin | Type | Contracts | contract_sources | dataset_loader | loader_unbounded | iter_batches | direct_sql | sql_assist |
  6: |---|---|---|---|---:|---:|---:|---:|---:|
  7: | `analysis_action_search_mip_batched_scheduler_v1` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
  8: | `analysis_action_search_simulated_annealing_v1` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
  9: | `analysis_actionable_ops_levers_v1` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 10: | `analysis_anova_auto` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 11: | `analysis_association_rules_apriori_v1` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
 12: | `analysis_attribution` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 13: | `analysis_bayesian_point_displacement` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 14: | `analysis_biclustering_cheng_church_v1` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
 15: | `analysis_bocpd_gaussian` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 16: | `analysis_burst_detection_kleinberg` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 17: | `analysis_burst_modeling_hawkes_v1` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
 18: | `analysis_busy_period_segmentation_v2` | analysis | artifact_only | inferred:default, override:contracts | 0 | 0 | 0 | 0 | 0 |
 19: | `analysis_capacity_scaling` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 20: | `analysis_chain_makespan` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 21: | `analysis_change_impact_pre_post` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 22: | `analysis_changepoint_energy_edivisive` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 23: | `analysis_changepoint_method_survey_guided` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 24: | `analysis_changepoint_pelt` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 25: | `analysis_chi_square_association` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 26: | `analysis_close_cycle_capacity_impact` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 27: | `analysis_close_cycle_capacity_model` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 28: | `analysis_close_cycle_change_point_v1` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 29: | `analysis_close_cycle_contention` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 30: | `analysis_close_cycle_duration_shift` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 31: | `analysis_close_cycle_revenue_compression` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 32: | `analysis_close_cycle_uplift` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 33: | `analysis_close_cycle_window_resolver` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 34: | `analysis_cluster_analysis_auto` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 35: | `analysis_concurrency_reconstruction` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 36: | `analysis_conformal_feature_prediction` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 37: | `analysis_conformance_alignments` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 38: | `analysis_conformance_checking` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 39: | `analysis_constrained_clustering_cop_kmeans_v1` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
 40: | `analysis_control_chart_cusum` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 41: | `analysis_control_chart_ewma` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 42: | `analysis_control_chart_individuals` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 43: | `analysis_control_chart_suite` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 44: | `analysis_copula_dependence` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 45: | `analysis_daily_pattern_alignment_dtw_v1` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
 46: | `analysis_density_clustering_hdbscan_v1` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
 47: | `analysis_dependency_community_leiden_v1` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
 48: | `analysis_dependency_community_louvain_v1` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
 49: | `analysis_dependency_critical_path_v1` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 50: | `analysis_dependency_graph_change_detection` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 51: | `analysis_dependency_resolution_join` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 52: | `analysis_determinism_discipline` | analysis | artifact_only | inferred:default, override:contracts | 0 | 0 | 0 | 0 | 0 |
 53: | `analysis_discrete_event_queue_simulator_v1` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
 54: | `analysis_distribution_drift_suite` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 55: | `analysis_distribution_shift_wasserstein_v1` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
 56: | `analysis_dp_gmm` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 57: | `analysis_drift_adwin` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 58: | `analysis_dynamic_close_detection` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 59: | `analysis_ebm_action_verifier_v1` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 60: | `analysis_effect_size_report` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 61: | `analysis_empirical_bayes_shrinkage_v1` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
 62: | `analysis_event_count_bocpd_poisson` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 63: | `analysis_evt_gumbel_tail` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 64: | `analysis_evt_peaks_over_threshold` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 65: | `analysis_factor_analysis_auto` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 66: | `analysis_frequent_itemsets_fpgrowth_v1` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
 67: | `analysis_gaussian_copula_shift` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 68: | `analysis_gaussian_knockoffs` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 69: | `analysis_graph_min_cut_partition_v1` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
 70: | `analysis_graph_topology_curves` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 71: | `analysis_graphical_lasso_dependency_network` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 72: | `analysis_hawkes_self_exciting` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 73: | `analysis_hmm_latent_state_sequences` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 74: | `analysis_hold_time_attribution_v1` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 75: | `analysis_ideaspace_action_planner` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 76: | `analysis_ideaspace_energy_ebm_v1` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 77: | `analysis_ideaspace_normative_gap` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 78: | `analysis_isolation_forest` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 79: | `analysis_isolation_forest_anomaly` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 80: | `analysis_issue_cards_v2` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
 81: | `analysis_kernel_two_sample_mmd` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 82: | `analysis_kingman_vut_approx` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 83: | `analysis_knockoff_wrapper_rf` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 84: | `analysis_lagged_predictability_test` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 85: | `analysis_littles_law_consistency` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 86: | `analysis_local_outlier_factor` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 87: | `analysis_log_template_drain` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 88: | `analysis_log_template_mining_drain` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 89: | `analysis_map_permutation_test_karniski` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 90: | `analysis_markov_transition_shift` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 91: | `analysis_matrix_profile_motifs_discords` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 92: | `analysis_message_entropy_drift` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 93: | `analysis_monte_carlo_surface_uncertainty` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 94: | `analysis_multiple_testing_fdr` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 95: | `analysis_multivariate_changepoint_pelt` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 96: | `analysis_multivariate_control_charts` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 97: | `analysis_multivariate_ewma_control` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 98: | `analysis_multivariate_t2_control` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
 99: | `analysis_mutual_information_screen` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
100: | `analysis_notears_linear` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
101: | `analysis_one_class_svm` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
102: | `analysis_online_conformal_changepoint` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
103: | `analysis_param_near_duplicate_minhash_v1` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
104: | `analysis_param_near_duplicate_simhash_v1` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
105: | `analysis_param_variant_explosion_v1` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
106: | `analysis_pca_auto` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
107: | `analysis_pca_control_chart` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
108: | `analysis_percentile_analysis` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
109: | `analysis_periodicity_spectral_scan` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
110: | `analysis_process_counterfactuals` | analysis | artifact_only | inferred:default, override:contracts | 0 | 0 | 0 | 0 | 0 |
111: | `analysis_process_drift_conformance_over_time` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
112: | `analysis_process_sequence` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
113: | `analysis_process_sequence_bottlenecks` | analysis | artifact_only | inferred:default, override:contracts | 0 | 0 | 0 | 0 | 0 |
114: | `analysis_proportional_hazards_duration` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
115: | `analysis_quantile_regression_duration` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
116: | `analysis_queue_delay_decomposition` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
117: | `analysis_queue_model_fit` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
118: | `analysis_recommendation_dedupe_v2` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
119: | `analysis_regression_auto` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
120: | `analysis_retry_rate_hotspots_v1` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
121: | `analysis_robust_covariance_outliers` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
122: | `analysis_robust_pca_pcp` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
123: | `analysis_robust_pca_sparse_outliers` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
124: | `analysis_scan_statistics` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
125: | `analysis_sequence_classification` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
126: | `analysis_sequence_grammar_sequitur_v1` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
127: | `analysis_sequential_patterns_prefixspan` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
128: | `analysis_sequential_patterns_prefixspan_v1` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
129: | `analysis_similarity_graph_spectral_clustering_v1` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
130: | `analysis_state_space_kalman_residuals` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
131: | `analysis_surface_fabric_sso_eigen` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
132: | `analysis_surface_fractal_dimension_variogram` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
133: | `analysis_surface_hydrology_flow_watershed` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
134: | `analysis_surface_multiscale_wavelet_curvature` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
135: | `analysis_surface_roughness_metrics` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
136: | `analysis_surface_rugosity_index` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
137: | `analysis_surface_terrain_position_index` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
138: | `analysis_survival_kaplan_meier` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
139: | `analysis_survival_time_to_event` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
140: | `analysis_tail_isolation` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
141: | `analysis_tda_betti_curve_changepoint` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
142: | `analysis_tda_mapper_graph` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
143: | `analysis_tda_persistence_landscapes` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
144: | `analysis_tda_persistent_homology` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
145: | `analysis_template_drift_two_sample` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
146: | `analysis_term_burst_kleinberg` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
147: | `analysis_time_series_analysis_auto` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
148: | `analysis_topic_model_lda` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
149: | `analysis_topographic_angle_dynamics` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
150: | `analysis_topographic_similarity_angle_projection` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
151: | `analysis_topographic_tanova_permutation` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
152: | `analysis_traceability_manifest_v2` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
153: | `analysis_transfer_entropy_directional` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
154: | `analysis_ttests_auto` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
155: | `analysis_two_sample_categorical_chi2` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
156: | `analysis_two_sample_numeric_ad` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
157: | `analysis_two_sample_numeric_ks` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
158: | `analysis_two_sample_numeric_mann_whitney` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
159: | `analysis_upload_linkage` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
160: | `analysis_user_host_savings` | analysis | artifact_only | inferred:default, override:contracts | 0 | 0 | 0 | 0 | 0 |
161: | `analysis_variant_differential` | analysis | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
162: | `analysis_waterfall_summary_v2` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
163: | `causal_recommendations_v1` | analysis | artifact_only | inferred:default | 0 | 0 | 0 | 0 | 0 |
164: | `changepoint_detection_v1` | analysis | artifact_only | inferred:default | 0 | 0 | 0 | 0 | 0 |
165: | `ingest_tabular` | ingest | sql_direct | detected:sql_direct | 0 | 0 | 0 | 1 | 0 |
166: | `llm_prompt_builder` | llm | orchestration_only | inferred:type_orchestration, override:contracts | 0 | 0 | 0 | 0 | 0 |
167: | `llm_text2sql_local_generate_v1` | llm | sql_assist | detected:sql_assist | 0 | 0 | 0 | 0 | 1 |
168: | `planner_basic` | planner | orchestration_only | inferred:type_orchestration, override:contracts | 0 | 0 | 0 | 0 | 0 |
169: | `process_mining_conformance_v1` | analysis | orchestration_only | inferred:depends_on | 0 | 0 | 0 | 0 | 0 |
170: | `process_mining_discovery_v1` | analysis | artifact_only | inferred:default | 0 | 0 | 0 | 0 | 0 |
171: | `profile_basic` | profile | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
172: | `profile_eventlog` | profile | sql_direct | detected:sql_direct | 0 | 0 | 0 | 1 | 0 |
173: | `report_bundle` | report | artifact_only | inferred:type_report, override:contracts | 0 | 0 | 0 | 0 | 0 |
174: | `report_decision_bundle_v2` | report | artifact_only | inferred:type_report, override:contracts | 0 | 0 | 0 | 0 | 0 |
175: | `report_payout_report_v1` | report | dataset_loader | detected:dataset_loader | 1 | 1 | 0 | 0 | 0 |
176: | `report_plain_english_v1` | report | artifact_only | inferred:type_report, override:contracts | 0 | 0 | 0 | 0 | 0 |
177: | `report_slide_kit_emitter_v2` | report | artifact_only | inferred:type_report, override:contracts | 0 | 0 | 0 | 0 | 0 |
178: | `transform_normalize_mixed` | transform | sql_direct | detected:sql_direct | 0 | 0 | 0 | 1 | 0 |
179: | `transform_sql_intents_pack_v1` | transform | sql_assist | detected:sql_assist | 0 | 0 | 0 | 0 | 1 |
180: | `transform_sqlpack_materialize_v1` | transform | sql_assist | detected:sql_assist | 0 | 0 | 0 | 0 | 1 |
181: | `transform_template` | transform | artifact_only | inferred:type_transform, override:contracts | 0 | 0 | 0 | 0 | 0 |
````

