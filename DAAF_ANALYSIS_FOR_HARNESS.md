# DAAF Repo — Transferable Patterns for the Statistics Harness (v3)

**D0:** 2026-03-04  
**Source:** DAAF (Data Analyst Augmentation Framework) — open-source Claude Code orchestration framework  
**Harness context:** Fully deterministic Python pipeline. 256+ plugins. Toposorted layers with `depends_on`. Two-lane orchestration (decision → explanation). Parallel execution within layers via `ThreadPoolExecutor`. Subprocess isolation per plugin. Shared state via SQLite `Storage` + artifacts directory. Deterministic seeding. Budget/timeout system. Result quality enforcement. Memory governor. Fingerprint-based cache reuse.  
**IP Status:** INTERNAL analysis. No code taken.

---

## What the Harness Already Has

After reading the actual codebase, most patterns I previously recommended are already implemented:

| Pattern | Harness Implementation |
|---------|----------------------|
| Topological dependency resolution | `_toposort_layers()` with cycle detection |
| Parallel execution within layers | `ThreadPoolExecutor` with row-count-based worker caps |
| Two-lane orchestration | Decision plugins first, then explanation; fallback to mixed if cross-lane deps |
| Deterministic seeding | `_seed_runtime()` + `_deterministic_env()` (PYTHONHASHSEED, numpy, random) |
| Budget/timeout per plugin | `BudgetTimer` + `time_limit_ms` + `mem_limit_mb` via RLIMIT_AS |
| Result quality enforcement | `finalize_result_contract()` — normalizes status, autofills finding kinds, validates output schema |
| Status normalization | `normalize_result_status()` — maps `na`/`skipped`/`degraded` → `ok` with observation finding |
| Memory governor | `_memory_governor_wait()` — polls `/proc/meminfo`, blocks new work under pressure |
| Cache reuse | `execution_fingerprint` (code_hash + settings_hash + dataset_hash) → skip re-run |
| Frozen surfaces | Contract-based code+settings drift detection |
| Subprocess isolation | `run_plugin_subprocess()` with network guard, eval guard |

**Bottom line:** The harness is architecturally mature. DAAF's orchestration patterns are largely redundant. What DAAF offers that the harness does NOT already have falls into four areas.

---

## Pattern 1: Typed Error Classification (NEW — not in harness)

**What DAAF does:** Classifies errors into types with different recovery strategies. Data errors, code errors, access errors, and validation failures each have different retry counts and escalation paths.

**Current harness gap:** `PluginError` has `type: str` and `message: str` but the type field is freeform. The pipeline's `normalize_result_status()` handles status normalization but doesn't classify error *causes*. When 15 plugins fail, the operator sees 15 independent errors instead of "all 15 failed because `econml` isn't installed."

**Adaptation (pure Python, ~30 lines in pipeline.py):**

Add a deterministic error classifier that runs inside `finalize_result_contract()`:

```python
ERROR_CATEGORIES = {
    "ImportError": "dependency_missing",
    "ModuleNotFoundError": "dependency_missing",
    "TimeoutError": "budget_exceeded",
    "MemoryError": "memory_exceeded",
    "numpy.linalg.LinAlgError": "numerical_failure",
    "ValueError": "input_invalid",
    "KeyError": "column_missing",
    "ConvergenceWarning": "convergence_failure",
}

def classify_error(error: PluginError) -> str:
    for pattern, category in ERROR_CATEGORIES.items():
        if pattern in (error.type or "") or pattern in (error.traceback or ""):
            return category
    return "unknown"
```

Then in the run manifest `summary`, group error counts by category:

```json
{"error_categories": {"dependency_missing": 15, "numerical_failure": 3, "budget_exceeded": 2}}
```

**Impact:** Operator immediately sees "install econml to fix 15 plugins" instead of reading 15 tracebacks. The report generator can also use this to suppress "noise" errors (dependency_missing is a setup issue, not a data issue).

---

## Pattern 2: Per-Plugin Validation Log / Run Record (NEW — partially in harness)

**What DAAF does:** Every transformation captures pre-state and post-state. The before/after pair is stored inline, creating an auditable trace of what the data looked like at each step.

**Current harness state:** The pipeline records `plugin_started` and `plugin_completed` events in the events table, with timing and execution fingerprint. `Storage.save_plugin_result()` persists the full `PluginResult`. But there's no record of the *input data characteristics* at the moment each plugin consumed it.

**What's missing:** If two analysis plugins in the same layer produce contradictory findings, there's no proof they saw the same data shape. If a dataset_version is updated between runs, the `cache_dataset_hash` handles invalidation — but within a single run, there's no per-plugin input snapshot.

**Adaptation (~25 lines in `run_spec()`):**

Before calling `run_plugin_subprocess()`, capture a lightweight input fingerprint:

```python
input_snapshot = {
    "row_count": dataset_row_count,
    "column_count": dataset_column_count,
    "dataset_hash": cache_dataset_hash,
    "template_status": str((dataset_template or {}).get("status", "")),
}
```

Store it in the `plugin_started` event payload. This is already almost there — the event payload has `execution_fingerprint` and `plugin_seed`. Adding `input_snapshot` is 5 lines.

**Impact:** Low effort, moderate value. Primarily useful for debugging contradictory results between plugins and for the new gap-analysis plugins (causal plugins are sensitive to input shape — a DML plugin seeing 50 columns vs. 5 columns will produce very different results).

---

## Pattern 3: Cross-Plugin Plausibility Checks (NEW — not in harness)

**What DAAF does:** Code-reviewer runs "adversarial" checks after each script. This is LLM-based in DAAF.

**Deterministic equivalent for the harness:** When multiple plugins analyze the same treatment/outcome relationship, their results should be cross-checked. This is especially relevant for the new gap-analysis plugins: DML (Plugin 1), IPW (Plugin 6), PSM (Plugin 29), and DiD (Plugin 3) all estimate treatment effects — they should roughly agree. If DML says +2.3 and DiD says -1.8, something is wrong.

**Current harness state:** Each plugin produces independent findings. The report aggregates them but doesn't cross-validate.

**Adaptation — two options:**

**Option A: Dedicated cross-validation plugin (~200 lines, new plugin)**

A new `analysis_causal_cross_validation_v1` plugin with `depends_on: [analysis_double_ml_ate_v1, analysis_diff_in_diff_v1, ...]` that:
1. Reads upstream causal plugin results from Storage
2. Extracts effect_size estimates from each
3. Flags contradictions (sign disagreement, magnitude > 3× ratio)
4. Produces a synthesis finding: "3/4 causal methods agree on positive treatment effect of 1.8–2.5 units; DiD disagrees (parallel trends test failed)"

This fits the existing architecture perfectly — it's just another plugin with `depends_on`.

**Option B: Post-analysis validation pass in pipeline.py (~80 lines)**

After analysis layers complete but before report stage, add a function that reads all analysis results and runs plausibility checks:

| Check | Rule | Applies When |
|-------|------|-------------|
| Sign agreement | All causal plugins estimating the same effect should agree on direction | ≥2 causal plugins returned `ok` |
| Magnitude agreement | Effect sizes within 3× of each other | ≥2 causal plugins returned `ok` |
| Anomaly detection agreement | Anomaly count within 50% across methods | ≥2 anomaly plugins returned `ok` |
| Changepoint agreement | Detected changepoints within ±5 time steps | ≥2 changepoint plugins returned `ok` |

Emit warnings as findings in the run manifest, not as hard failures.

**Recommendation:** Option A. It's cleaner, fits the plugin architecture, and downstream report plugins can consume the cross-validation findings naturally.

---

## Pattern 4: Structured Learning Capture (NEW — not in harness, NO LLM needed)

**What DAAF does:** `LEARNINGS.md` captures data surprises, methodology decisions, and system improvement items. In DAAF this is LLM-driven.

**Deterministic equivalent for the harness:** Not a narrative document, but a **structured signals table** appended to the run manifest. Each plugin can optionally emit a `learning_signal` in its result, and the pipeline aggregates them.

**Current harness state:** Results have `debug: dict[str, Any]` which is freeform. Some plugins use it for diagnostic info, but there's no structured "this is something the operator should know for future runs" field.

**Adaptation (~15 lines across types.py + pipeline.py):**

In the run manifest summary, add a `signals` section that the pipeline auto-generates from result patterns:

```python
signals = []
# Auto-detect: plugin ran on insufficient data
for pid, result in results.items():
    if result.status == "ok" and result.budget.get("sampled"):
        signals.append({"type": "sampled_input", "plugin_id": pid, 
                        "note": "Result based on sampled data, not full dataset"})
    # Auto-detect: near-threshold sample size
    n = (result.metrics or {}).get("n_observations")
    if isinstance(n, (int, float)) and 30 <= n <= 50:
        signals.append({"type": "marginal_sample_size", "plugin_id": pid, 
                        "note": f"n={n}, just above minimum"})
    # Auto-detect: convergence warnings in debug
    if "convergence" in str(result.debug).lower():
        signals.append({"type": "convergence_concern", "plugin_id": pid})
```

No new field on `PluginResult` needed — this is computed post-hoc from existing data.

**Impact:** Operators scanning the run manifest see "12 plugins ran on sampled data" and "3 plugins had convergence concerns" without reading 256 individual results.

---

## Patterns That Do NOT Transfer (Confirmed After Repo Review)

| DAAF Pattern | Why It Doesn't Transfer |
|---|---|
| Gate-gated stage progression | Harness already has this via `_toposort_layers()` + two-lane orchestration |
| Standardized validation checkpoints | Harness already has `finalize_result_contract()` + schema validation |
| Parallel execution with dependency resolution | Harness already has `ThreadPoolExecutor` + toposort layers |
| Session state / recovery | Harness uses SQLite events table + journal.json + crash recovery on startup |
| Deterministic seeding | Harness already has `_seed_runtime()` + `_deterministic_env()` |
| Budget/timeout management | Harness already has `BudgetTimer` + RLIMIT_AS + time_limit_ms |
| Status normalization | Harness already has `normalize_result_status()` |
| Memory governor | Harness already has `_memory_governor_wait()` |
| Truth hierarchy | Development-time concern, not runtime; not a code pattern |
| Engagement mode classification | Harness always runs all plugins — correct for its use case |
| Phase Status Updates / user gates | No interactive user at runtime |
| Subagent context isolation | Harness uses subprocess isolation — already solved differently |

---

## Where LLMs Are NOT Necessary

All four patterns above are pure deterministic Python. No LLM is needed anywhere in the analytical pipeline.

The one area where an LLM *could* add value that deterministic code cannot:

**Natural-language narrative synthesis.** The harness produces structured JSON findings. Converting "DML ATE=2.3 (p=0.003), IPW ATE=2.1 (p=0.008), DiD estimate invalid (parallel trends violated)" into a coherent paragraph for a non-technical stakeholder requires either an LLM or an elaborate template engine.

**Recommendation:** If this is desired, implement it as a `report_plain_english_v1` plugin (which already exists in the repo) that runs in the report stage and reads analysis findings from Storage. The LLM call happens in the report plugin, not in the analytical engine. The analytical pipeline stays 100% deterministic.

---

## Implementation Priority

| Rank | Pattern | Where | LOC | Impact |
|------|---------|-------|-----|--------|
| 1 | Error classification (#1) | `finalize_result_contract()` + manifest summary | ~30 | Groups 256 failures into actionable categories |
| 2 | Cross-plugin plausibility (#3) | New plugin: `analysis_causal_cross_validation_v1` | ~200 | Catches contradictory causal estimates |
| 3 | Run-level signals aggregation (#4) | Manifest generation in pipeline.py | ~15 | Surfaces data quality concerns without reading 256 results |
| 4 | Per-plugin input snapshot (#2) | `run_spec()` event payload | ~5 | Debugging aid for contradictory results |

Total: ~250 lines. Pattern 2 (cross-validation plugin) is the most valuable and should be built alongside the Tier 1 causal plugins from the gap analysis.

---

*End of Analysis*
