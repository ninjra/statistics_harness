# Statistical Plugin Spec Pack (Codex CLI) — Unknown Process Logs
Version: 0.1  
Goal: Provide **complete, self-contained** specifications for Codex CLI to implement new statistical
analysis plugins that can run against **unknown datasets** (unknown schema, content, source) and still
produce **useful, safe, deterministic, citable** insights.

This document covers:
- **Additional plugin families** (high-yield gaps)
- **Priority recommendations** (implement first)
- **Backlog categories 1–8** (full list of additional plugins)

---

## 0) Non-negotiable invariants (4 pillars)

### Performant
- Must complete within a configurable time budget.
- Must cap runtime and memory via deterministic sampling and dimension limits.
- Must avoid O(n²) by default; allow “expensive mode” only when explicitly enabled.

### Accurate
- Must include statistical calibration and diagnostics (false positive controls, sensitivity).
- Must prefer robust statistics (median/MAD, trimmed means, shrinkage covariances) when schema is unknown.
- Must produce **effect sizes** and confidence proxies, not just p-values.

### Secure
- Must not leak PII or proprietary raw text.
- Must redact/mask suspected PII before any textual modeling (template mining, topic modeling).
- Must emit only aggregated evidence (counts, hashes, exemplars capped and redacted).

### Citable
- Each plugin result must include a `references` field with canonical method references
  (DOIs/arXiv/author PDFs). Do not claim links are live; treat them as pointers.

---

## 1) Standard plugin contract (required)

### 1.1 Files to generate per plugin
- `plugins/<plugin_id>.py`
- `plugins/<plugin_id>_schema.json` (config schema; optional but preferred)
- `tests/test_<plugin_id>.py`
- `docs/<plugin_id>.md` (brief usage + interpretation, optional if content in this spec is enough)

Where `<plugin_id>` matches the harness naming convention, e.g. `analysis_control_chart_suite`.

### 1.2 Entry point signature
Implement:
```python
def run(df: "pd.DataFrame", config: dict, *, context: dict | None = None) -> dict:
    ...
```

Inputs:
- `df`: pandas DataFrame loaded from unknown source.
- `config`: dict with plugin-specific options (must have defaults).
- `context`: optional metadata from harness (client/project id, run id, global budgets, redaction rules).

Output: a JSON-serializable dict:
```json
{
  "plugin_id": "analysis_xxx",
  "version": "0.1.0",
  "status": "ok|skipped|error",
  "summary": "human-readable 1–3 sentences",
  "findings": [
    {
      "id": "stable_id_string",
      "severity": "info|warn|critical",
      "confidence": 0.0,
      "title": "short",
      "what": "what happened",
      "why": "why we think so",
      "evidence": { "metrics": {}, "tables": [], "series": [], "params": {} },
      "where": { "column": "...", "group": {"k":"v"}, "window": {"start":"...","end":"..."} },
      "recommendation": "next step"
    }
  ],
  "artifacts": [
    { "type": "table|series|graph", "name": "string", "data": {} }
  ],
  "metrics": { "runtime_ms": 0, "rows_seen": 0, "rows_used": 0, "cols_used": 0 },
  "references": [
    { "title": "...", "url": "...", "doi": "...", "notes": "optional" }
  ],
  "debug": { "warnings": [], "skips": [], "column_inference": {} }
}
```

### 1.3 Hard output rules
- **Determinism:** Same `df` + same `config` + same `seed` ⇒ identical output (ordering and values within tolerance).
- **Stability:** Findings must have stable `id` computed from deterministic inputs
  (e.g., hash of plugin_id + column + group + window + finding type).
- **Privacy:** Never output raw message strings by default. When needed, output:
  - template hash
  - redacted exemplar snippet (max 120 chars) and only if policy allows
  - counts and rates, not raw rows

### 1.4 Common config (must be supported by every plugin)
Every plugin must accept these keys (with defaults), even if it ignores some:

```yaml
seed: 1337
time_budget_ms: 25000
max_rows: 200000
max_cols: 80
max_groups: 30
max_findings: 30

# Column selection
time_column: "auto"        # or explicit column
group_by: "auto"           # list[str] or "auto" or []
value_columns: "auto"      # list[str] or "auto" or []

# Focus policy (client/project priorities change)
focus:
  mode: "full_scan"        # "full_scan" | "focused_plus_scan"
  windows: []              # list of {"name": "...", "start": "...", "end": "..."} OR {"name":"close_cycle","rules":{...}}
  # If windows empty, run whole dataset
  include_full_scan: True  # when focused_plus_scan

# Privacy / redaction
privacy:
  enable_redaction: True
  redact_patterns: ["email", "ip", "uuid", "credit_card", "phone"]
  max_exemplars: 3
  allow_exemplar_snippets: False

# Output verbosity
verbosity: "normal"        # "low" | "normal" | "high"
```

---

## 2) Dataset profiling & column inference (shared utility)

Implement a shared helper (or inline consistently) to profile the dataset:

### 2.1 Column typing heuristics
- Numeric: pandas numeric dtype, excluding:
  - “id-like” columns (unique_ratio > 0.98 and integer-like)
- Timestamp:
  - datetime dtype OR parseable strings with high success ratio
  - name hints: `time`, `ts`, `date`, `start`, `end`, `queued`, `eligible`, `created`, `updated`
- Categorical:
  - object/string with unique_ratio <= 0.2 OR <= 500 unique (configurable)
- Text/message:
  - object/string with median length >= 30 and high unique_ratio
  - name hints: `message`, `log`, `exception`, `error`, `stack`, `trace`
- Group-by candidates:
  - categorical columns with moderate cardinality (2..max_groups) and non-null coverage

### 2.2 Time axis selection (auto)
If `time_column="auto"`:
1. Prefer an explicit timestamp-like column with best parse rate and broad coverage.
2. If multiple candidates, choose:
   - the one with widest span (max - min)
   - then the one with highest non-null ratio
3. If none, plugin must still run in “no-time” mode where applicable.

### 2.3 Deterministic sampling (required)
If `len(df) > max_rows`:
- Select rows deterministically by stable hashing:
  - require a stable row key if possible (`id`, `case_id`, etc.)
  - else hash of concatenated selected columns with fixed salt + seed
- Never use `df.sample()` without fixed random_state and stable seed.

### 2.4 Numeric matrix construction
When a plugin needs a multivariate numeric matrix:
- Select top `max_cols` numeric columns by:
  - non-null coverage
  - variance (post-robust scaling)
  - low missingness
- Impute missing values deterministically:
  - median per column (robust)
- Standardize with robust scale:
  - `x' = (x - median) / MAD` (fallback MAD->IQR->std)

---

## 3) Common test harness (every plugin must pass)

Create shared pytest helpers, then apply to all plugins.

### 3.1 Contract tests
- `test_contract_fields_present`: output has required top-level keys.
- `test_json_serializable`: `json.dumps(result)` succeeds.
- `test_status_is_valid`: status is in {ok, skipped, error}.
- `test_no_exception_on_empty_df`: empty df returns `skipped` with reason, not crash.
- `test_no_exception_on_weird_types`: mixed dtypes, NaNs, inf, strings.

### 3.2 Determinism tests
- `test_deterministic_same_seed`: run twice ⇒ identical result (or within numeric tolerance).
- `test_nondeterministic_different_seed_allowed`: if algorithm uses randomness, different seed may differ,
  but must still satisfy contract and not explode findings.

### 3.3 Privacy tests (Secure)
- `test_no_raw_text_leakage_by_default`:
  - create a df with a message column containing emails/IPs
  - ensure result does not contain those literals unless `allow_exemplar_snippets=True`
- `test_redaction_applied_when_enabled`:
  - if exemplars allowed, ensure patterns are redacted/masked.

### 3.4 Performance tests (bounded)
- `test_runtime_budget_respected_smoke`:
  - run with time_budget_ms small; plugin should either return quickly or `skipped` gracefully.
- `test_sampling_applied_when_large`:
  - df with > max_rows; ensure `rows_used <= max_rows`.

### 3.5 Findings quality tests
- `test_findings_capped`: `len(findings) <= max_findings`.
- `test_findings_have_required_fields`: every finding has id, severity, confidence, title, what, why.

---

## 4) Additional statistical plugin families (the “why” behind the backlog)

Implementing the backlog should close these gaps:
1. Control charts (univariate + multivariate) for early drift detection
2. Distribution drift tooling with effect sizes + multiple testing control
3. Diverse anomaly detectors for different geometry (global, local density, sparse outliers, subsequence discord)
4. Text → structure (template mining, topic/burst) for unknown log content
5. Process conformance + drift (behavioral deviation vs expected flow)
6. Dependence/directionality (dependency networks, time-lag directional signals)
7. Duration & SLA risk (survival/quantile) + queueing sanity checks

---

# 5) PRIORITY IMPLEMENTATION SET (implement first)

These 8 plugins are the “first wave” because they:
- are broadly applicable to unknown datasets
- complement your existing set
- produce high-value insights with manageable complexity

Each spec below includes: purpose, inference, algorithm, outputs, configs, and tests.

---

## 5.1 analysis_control_chart_suite (CUSUM + EWMA + Individuals)

### Purpose
Early detection of small/gradual shifts in numeric signals, per metric and per slice.

### Column inference
- `value_columns="auto"`: pick numeric columns (exclude id-like).
- `group_by="auto"`: choose up to `max_groups` categorical candidates.
- `time_column="auto"`: optional; if present, sort by time; else use row order.

### Algorithm (default)
For each selected `value` and each group (including “ALL”):
1. Sort by time if available; drop NaNs.
2. Compute baseline center/scale using robust stats:
   - center = median
   - scale = MAD (fallback IQR/1.349, then std)
3. Individuals chart:
   - flag points with |z_robust| > z_thresh (default 4.0)
4. EWMA:
   - EWMA_t = λ x_t + (1-λ) EWMA_{t-1}
   - signal when |EWMA - center| > L * scale_ewma
5. CUSUM (two-sided):
   - C⁺_t = max(0, C⁺_{t-1} + (x_t - (center + k)))
   - C⁻_t = max(0, C⁻_{t-1} + ((center - k) - x_t))
   - signal when C⁺ or C⁻ exceeds h
6. Emit top findings ranked by:
   - recency (if time exists)
   - magnitude (effect size)
   - persistence (run length)

### Config
```yaml
control_chart:
  z_thresh: 4.0
  ewma_lambda: 0.2
  ewma_L: 3.0
  cusum_k: 0.5      # in sigma units
  cusum_h: 5.0      # in sigma units
  min_points: 50
  handle_autocorr: "off"   # "off" | "difference1" | "residual_ar1"
  max_series_points: 5000  # deterministic downsample for artifacts
```

### Outputs
- Findings: per (value, group) alarm summary:
  - last alarm timestamp
  - shift direction
  - estimated shift size in robust sigma
  - run length (how persistent)
- Artifacts:
  - EWMA series (downsampled)
  - CUSUM series (downsampled)
  - table of top alarms

### Edge cases
- If min_points not met for a group: skip that group with debug note.
- If constant series: skip.
- If time column exists but non-monotonic: sort and continue.

### Required tests (in addition to common tests)
- `test_detects_mean_shift`:
  - synthetic normal series with mean shift at t=200; expect a finding.
- `test_no_false_alarm_stationary_smoke`:
  - stationary noise; expect <= small number of findings (or severity=info).
- `test_group_by_slices`:
  - create 2 groups; shift only one; ensure finding references correct group.

References (include in plugin output):
- EWMA/CUSUM/control chart general refs (NIST handbook pages are acceptable pointers).

---

## 5.2 analysis_multivariate_control_charts (T² + MEWMA + PCA residuals)

### Purpose
Detect coordinated multi-metric drift that is invisible in any single metric.

### Column inference
- Build numeric matrix X from robustly standardized numeric columns (Section 2.4).
- Optional grouping and time sorting (same as 5.1).

### Algorithm
Per group:
1. Select up to `d = max_cols` numeric columns with best coverage.
2. Robust standardize.
3. Compute covariance with shrinkage:
   - Ledoit–Wolf shrinkage if sklearn available; else diagonal shrinkage.
4. Hotelling T²:
   - T²_t = (x_t - μ)ᵀ Σ^{-1} (x_t - μ)
   - threshold via χ² approximation with df=d (heuristic) or empirical quantiles.
5. MEWMA:
   - Z_t = λ x_t + (1-λ) Z_{t-1}
   - monitor statistic based on Σ_Z
6. PCA chart:
   - fit PCA on baseline segment (first p% of time or random deterministic split)
   - monitor Q-residual (SPE) and T² in PC space

### Config
```yaml
mv_control:
  min_points: 200
  baseline_fraction: 0.3
  shrinkage: "auto"    # "auto" | "lw" | "diag"
  pca_components: "auto"  # or int
  mewma_lambda: 0.2
  threshold_quantile: 0.995
```

### Outputs
- Findings: spikes in T² / Q-residual with implicated top contributing columns
  (contribution via standardized residual magnitude).
- Artifacts: time series of multivariate statistic, top contributing features table.

### Required tests
- `test_detects_joint_shift`:
  - simulate correlated 5D normal; shift 2 dims slightly; expect finding.
- `test_invariant_to_column_scaling`:
  - scale one column; robust standardization should yield similar flags (within tolerance).
- `test_shrinkage_fallback`:
  - run without sklearn; ensure diag shrinkage path works.

---

## 5.3 analysis_multivariate_changepoint_pelt

### Purpose
Segment a multivariate numeric stream into regimes; identify regime boundaries and the metrics that changed.

### Column inference
- Use time axis if available; else row order.
- Numeric matrix from Section 2.4; optionally per group.

### Algorithm
1. Reduce to <= `max_points` rows deterministically if huge (e.g., sample by time buckets).
2. Define cost:
   - default: L2 cost with per-segment mean (sum of squared errors) on standardized X.
3. Run PELT:
   - exact linear-time under assumptions; implement in pure python + numpy.
   - or optionally use `ruptures` if present.
4. Choose penalty:
   - default: `pen = beta * log(n) * d`, with beta default 2.0
5. For each changepoint:
   - compare window pre vs post with effect sizes per column
   - rank top-changed columns

### Config
```yaml
pelt:
  max_points: 20000
  penalty_beta: 2.0
  min_segment_size: 50
  max_changepoints: 20
```

### Outputs
- Findings: changepoint locations (time or index), regime summaries
- Artifacts: table of changepoints and top changed metrics

### Required tests
- `test_recovers_two_changepoints`:
  - generate 2 regime changes in 3D; ensure cp count ~2 and locations near truth.
- `test_penalty_controls_oversegmentation`:
  - with higher beta, fewer cps.
- `test_min_segment_enforced`.

Reference pointers:
- Killick et al. (PELT).

---

## 5.4 analysis_distribution_drift_suite (numeric + categorical + multivariate)

### Purpose
Answer: “What changed, where, and by how much?” across time windows and cohorts, with false discovery control.

### Column inference
- Determine analysis axis:
  - if time exists: compare rolling windows and/or pre/post focus windows
  - else: compare cohorts by group columns (top splits)
- Numeric columns: KS + AD + effect sizes (Cliff’s delta or standardized median diff)
- Categorical: χ² / G-test; effect size Cramér’s V
- Multivariate (optional): MMD (RBF kernel) with permutation p-value (capped)

### Algorithm
1. Define “reference” vs “candidate” windows:
   - default: earliest 30% vs latest 30% (time-based)
   - plus any `focus.windows` if provided
2. Run tests per column and per group slice (capped).
3. Compute effect sizes.
4. Apply BH-FDR across the family of tests.
5. Summarize only discoveries passing q <= alpha (default 0.05), capped.

### Config
```yaml
drift:
  alpha: 0.05
  window_strategy: "early_vs_late"  # + "rolling"
  rolling_window_count: 6
  min_group_size: 200
  mmd:
    enabled: True
    max_points: 5000
    permutations: 200
```

### Outputs
- Findings: drifted columns with:
  - p-value, q-value, effect size
  - direction (higher/lower or distribution change)
  - group slice and window definition
- Artifacts: top drift table, distribution summaries (quantiles), category shift table

### Required tests
- `test_numeric_drift_detected`:
  - early N(0,1) vs late N(0.5,1); expect flag with q<=alpha.
- `test_categorical_drift_detected`:
  - category proportions changed; expect flag.
- `test_fdr_reduces_false_positives_smoke`:
  - many null columns; ensure flagged count is bounded.

Reference pointers:
- KS/AD tests; Benjamini–Hochberg; MMD.

---

## 5.5 analysis_isolation_forest_anomaly

### Purpose
Multivariate anomaly scoring across numeric telemetry to identify rare regimes/outliers.

### Column inference
- Numeric matrix; optional grouping; time sorting optional.

### Algorithm
- If sklearn available:
  - fit IsolationForest on sampled rows; score all rows or sample.
- Else fallback:
  - robust z-score aggregation:
    - anomaly_score = max_i |z_i| or sum_i |z_i|
- Report top-k anomalies and the features contributing most.

### Config
```yaml
iforest:
  enabled: True
  sample_size: 10000
  contamination: "auto"   # or float
  n_estimators: 200
  score_top_k: 50
  fallback: "robust_z"
```

### Outputs
- Findings: top anomaly clusters or points, with contributing features.
- Artifacts: table of anomaly exemplars (numeric-only by default).

### Required tests
- `test_outliers_ranked_high`:
  - inject a few extreme points; ensure they appear in top-k.
- `test_seeded_determinism`:
  - same seed yields same top-k order (within stable ties).
- `test_fallback_path_without_sklearn`:
  - monkeypatch import to fail; ensure robust_z path runs.

Reference pointers:
- Isolation Forest paper.

---

## 5.6 analysis_robust_pca_sparse_outliers (Robust PCA / PCP)

### Purpose
Separate low-rank structure (normal behavior) from sparse corruptions/outliers across multiple metrics.

### Column inference
- Numeric matrix, robust standardized.

### Algorithm
Preferred (if feasible within budget):
- Inexact Augmented Lagrange Multiplier (IALM) for Principal Component Pursuit:
  - minimize ||L||_* + λ||S||₁ subject to X = L + S
- If too large or time_budget small:
  - fallback: PCA on robust scaled X; use reconstruction residuals to flag outliers.

Outlier scoring:
- row_score = ||S_row||₁ or ||residual||₂
- feature contributions: largest |S_ij| entries.

### Config
```yaml
rpca:
  mode: "auto"           # "auto" | "ialm" | "pca_residual"
  max_points: 5000
  max_iters: 200
  tol: 1e-6
  lambda_scale: "auto"
  outlier_top_k: 50
```

### Outputs
- Findings: sparse-outlier episodes, implicated metrics.
- Artifacts: singular value summary; outlier table.

### Required tests
- `test_recovers_sparse_outliers`:
  - generate X = low_rank + sparse; ensure injected sparse rows flagged.
- `test_auto_falls_back_on_large`:
  - n > max_points triggers fallback.
- `test_numerical_stability`:
  - ensure no NaNs/inf in outputs.

Reference pointers:
- Candes et al. Robust PCA.

---

## 5.7 analysis_log_template_mining_drain

### Purpose
Turn raw log message text into structured templates and parameters; enable downstream analysis
(template drift, burst detection, conformance on events).

### Column inference
- Detect message column(s) via heuristics (Section 2.1).
- Optional timestamp and grouping.

### Algorithm (Drain)
1. Redact/mask PII tokens BEFORE parsing (emails/IPs/UUIDs/paths).
2. Tokenize message into tokens.
3. Build parse tree keyed by token count and token positions.
4. Produce:
   - `template_id` (hash of template string)
   - `template` (optional; only if policy allows; else store masked template)
   - `params` extracted per line (not emitted by default)

### Config
```yaml
drain:
  max_depth: 4
  sim_thresh: 0.4
  max_children: 100
  redact_before_parse: True
  store_template_text: False
  max_templates: 5000
```

### Outputs
- Findings:
  - top templates by frequency
  - newly emergent templates in late window vs early window
  - templates associated with extreme latency/duration (if duration columns exist)
- Artifacts:
  - template frequency table (template_id, count, first_seen, last_seen)
  - template drift table (delta share)

### Required tests
- `test_templates_stable_across_runs`:
  - same messages ⇒ identical template_ids and counts.
- `test_redaction_masks_pii`:
  - message contains email/ip; ensure output has no literal.
- `test_new_template_detected`:
  - add new message pattern in late segment; ensure drift finding.

Reference pointers:
- Drain log parsing.

---

## 5.8 analysis_conformance_checking (process model vs observed)

### Purpose
Detect behavioral deviation: skipped steps, unexpected loops, reordered activities, new edges.

### Column inference
Try to infer event log schema:
- Case identifier candidates: `case_id`, `trace_id`, `request_id`, `session_id`, `sequence`, `chain_id`
- Activity candidates: `activity`, `event`, `name`, `step`, `operation`, `process`, `module`
- Timestamp: from Section 2.2

If cannot infer, `skipped` with actionable debug.

### Algorithm (two-tier)
Tier A (no external deps; default):
1. Build directly-follows graph (DFG) from sequences per case:
   - edges (a→b) counts
2. Build baseline DFG from early window; compare with late window:
   - new edges, removed edges, edge weight shifts
3. Conformance proxy:
   - compute fraction of late traces whose edges are subset of baseline edges
   - identify common violating edges and variants

Tier B (optional, if pm4py available and budget allows):
- Discover process model (e.g., inductive miner) on baseline
- Run alignments/token replay fitness on late window
- Summarize deviation types

### Config
```yaml
conformance:
  min_cases: 50
  max_cases: 50000
  baseline_fraction: 0.3
  variant_top_k: 20
  use_pm4py_if_available: False
```

### Outputs
- Findings:
  - conformance score drop (late vs baseline)
  - top new/changed edges
  - top violating variants (represented as activity sequences, optionally hashed)
- Artifacts:
  - edge tables, variant frequency tables

### Required tests
- `test_detects_new_edge`:
  - baseline: A→B→C; late: A→D→C appears; expect new edge finding.
- `test_conformance_proxy_scores`:
  - introduce many deviating traces; score decreases.
- `test_schema_inference`:
  - use generic column names; plugin should infer.

Reference pointers:
- Process Mining: Data Science in Action (conformance).

---

# 6) BACKLOG CATEGORY 1 — Drift + changepoints (online/offline)

Below are additional plugins beyond the priority set. Some overlap with priority items; treat those as already implemented,
and implement the rest as separate plugins only if the harness benefits from modularity.

For each plugin: implement the standard contract, common config, and common tests. Add the plugin-specific tests below.

---

## 6.1 analysis_control_chart_cusum (univariate)

If `analysis_control_chart_suite` exists, this plugin can be a thin wrapper that only runs CUSUM and emits richer diagnostics.

Plugin-specific tests:
- `test_detects_shift_cusum_only`
- `test_handles_missing_values`

---

## 6.2 analysis_control_chart_ewma (univariate)

Thin wrapper around EWMA with parameter sweeps (optional).

Plugin-specific tests:
- `test_detects_small_shift_ewma`

---

## 6.3 analysis_control_chart_individuals

Individuals chart + robust sigma with heavy-tail tolerance.

Plugin-specific tests:
- `test_flags_extreme_points_only`

---

## 6.4 analysis_multivariate_t2_control

If `analysis_multivariate_control_charts` exists, this can be a wrapper focusing on T² only.

Plugin-specific tests:
- `test_joint_outlier_spike`

---

## 6.5 analysis_multivariate_ewma_control (MEWMA)

Wrapper focusing on MEWMA only.

Plugin-specific tests:
- `test_detects_gradual_multivariate_shift`

---

## 6.6 analysis_pca_control_chart

Wrapper focusing on PCA residual charts.

Plugin-specific tests:
- `test_detects_subspace_shift`

---

## 6.7 analysis_changepoint_pelt (univariate or modular)

If you already implement `analysis_multivariate_changepoint_pelt`, you may also implement a univariate version that:
- runs per numeric column and emits a cross-column changepoint “consensus” view.

Plugin-specific tests:
- `test_consensus_changepoints_align`

---

## 6.8 analysis_changepoint_energy_edivisive (nonparametric multivariate)

### Purpose
Detect changepoints without assuming Gaussianity, robust under heavy tails.

### Algorithm (bounded)
- Use energy distance statistics; implement a capped variant:
  - sample max_points
  - greedy divisive search up to max_changepoints
- If budget too small: `skipped` with message.

Config:
```yaml
edivisive:
  max_points: 8000
  max_changepoints: 10
  min_segment_size: 80
```

Tests:
- `test_detects_heavy_tail_shift`
- `test_budget_skip`

Reference pointers:
- Matteson & James energy statistics changepoints.

---

## 6.9 analysis_drift_adwin (streaming drift detector)

### Purpose
Streaming drift detection using adaptive windows.

### Algorithm
- Apply ADWIN to 1D streams (numeric columns), optionally per group.
- Emit drift points and drift rate.

Config:
```yaml
adwin:
  delta: 0.002
  min_points: 200
  max_points: 20000
```

Tests:
- `test_adwin_detects_shift`
- `test_adwin_no_shift_stationary`

Reference pointers:
- Bifet & Gavalda.

---

## 6.10 analysis_changepoint_method_survey_guided (orchestration meta-plugin)

### Purpose
Auto-select among changepoint methods based on data properties.
Primarily produces a “method recommendation” and runs a default safe method.

Rules:
- If n < 500 and d <= 10: run PELT
- If heavy tails suspected (kurtosis high): run energy method
- If time missing: run cohort drift (Category 2) instead

Tests:
- `test_selects_energy_on_heavy_tails`
- `test_selects_pelt_on_gaussian`

---

# 7) BACKLOG CATEGORY 2 — Windowed cohort comparisons (“what changed, where”)

This category overlaps with `analysis_distribution_drift_suite`. You can implement these as separate, smaller plugins if useful
for modularity, or keep them as submodules under the drift suite.

---

## 7.1 analysis_two_sample_numeric_ks

- Compare distributions between two windows/cohorts using KS.
- Emit effect size + direction proxy via median difference.

Config:
```yaml
ks:
  alpha: 0.01
  min_n: 200
```

Tests:
- `test_ks_detects_shift`
- `test_ks_handles_ties`

---

## 7.2 analysis_two_sample_numeric_ad

- Anderson–Darling for tails.

Config:
```yaml
ad:
  alpha: 0.01
  min_n: 200
```

Tests:
- `test_ad_detects_tail_change`

---

## 7.3 analysis_two_sample_numeric_mann_whitney

- Mann–Whitney U; report Cliff’s delta (or rank-biserial) effect size.

Config:
```yaml
mw:
  alpha: 0.01
  min_n: 200
```

Tests:
- `test_mw_detects_location_shift`

---

## 7.4 analysis_two_sample_categorical_chi2

- χ² test with category pooling when expected counts low.

Config:
```yaml
chi2:
  alpha: 0.01
  min_expected: 5
  max_categories: 200
```

Tests:
- `test_chi2_detects_proportion_change`
- `test_category_pooling`

---

## 7.5 analysis_kernel_two_sample_mmd

- MMD with RBF kernel; approximate p-value via permutations.

Config:
```yaml
mmd:
  alpha: 0.01
  max_points: 4000
  permutations: 200
  bandwidth: "median_heuristic"
```

Tests:
- `test_mmd_detects_multivariate_shift`
- `test_mmd_budget_capped`

---

## 7.6 analysis_effect_size_report

- Pure reporting plugin:
  - standardized median diff
  - Cliff’s delta
  - Cramér’s V
  - bootstrap CI (capped)

Tests:
- `test_effect_sizes_reasonable_values`

---

## 7.7 analysis_multiple_testing_fdr

- Implement BH procedure; usable by other plugins.

Tests:
- `test_bh_monotone_qvalues`
- `test_controls_false_discoveries_smoke`

---

## 7.8 analysis_change_impact_pre_post

- Given a changepoint boundary (from config or discovered), compute pre/post impact.

Config:
```yaml
impact:
  boundary: "auto"   # or timestamp
  alpha: 0.05
```

Tests:
- `test_pre_post_reports_top_drivers`

---

# 8) BACKLOG CATEGORY 3 — Anomaly detection + tail risk

---

## 8.1 analysis_isolation_forest

If already implemented as `analysis_isolation_forest_anomaly`, treat as same plugin.
Otherwise implement here.

---

## 8.2 analysis_local_outlier_factor

### Purpose
Detect local-density anomalies.

Algorithm:
- robust scale numeric matrix
- fit LOF on sample; score sample or all (bounded)
- report top-k

Config:
```yaml
lof:
  n_neighbors: 35
  max_points: 20000
  top_k: 50
```

Tests:
- `test_lof_detects_local_cluster_outliers`

Reference pointers:
- Breunig et al. LOF.

---

## 8.3 analysis_one_class_svm

Purpose:
- novelty detection when anomalies form a boundary.

Config:
```yaml
ocsvm:
  nu: 0.02
  kernel: "rbf"
  max_points: 10000
  top_k: 50
```

Tests:
- `test_ocsvm_detects_outliers`
- `test_fallback_if_sklearn_missing`

Reference pointers:
- One-class SVM novelty detection.

---

## 8.4 analysis_robust_covariance_outliers

Purpose:
- robust Mahalanobis distance using robust covariance estimator (MCD or shrinkage).

Config:
```yaml
robust_cov:
  max_points: 20000
  top_k: 50
  method: "auto"     # "mcd" | "shrinkage" | "diag"
```

Tests:
- `test_mahalanobis_flags_outliers`

---

## 8.5 analysis_robust_pca_pcp

Same as priority robust PCA plugin; may be alias.

---

## 8.6 analysis_evt_gumbel_tail

Purpose:
- model distribution of maxima/minima (extremes) for tail risk.

Algorithm:
- choose numeric columns with enough points
- block maxima by time bucket if time exists; else global maxima sampling
- fit Gumbel (or GEV if implemented) via MLE or scipy if available
- compute return levels and tail probabilities

Config:
```yaml
evt_gumbel:
  min_points: 500
  block: "day"     # "hour" | "day" | "week" | "none"
  top_cols: 20
```

Tests:
- `test_evt_fits_without_nan`
- `test_tail_probability_monotone`

Reference pointers:
- EVT NIST pages / standard EVT texts.

---

## 8.7 analysis_evt_peaks_over_threshold (POT / GPD)

Purpose:
- characterize p99/p999 tail via generalized Pareto.

Algorithm:
- choose threshold via quantile (e.g., 0.95) or mean residual life (optional)
- fit GPD; compute tail quantiles

Config:
```yaml
evt_pot:
  threshold_quantile: 0.95
  min_exceedances: 200
  top_cols: 20
```

Tests:
- `test_pot_detects_heavier_tail`
- `test_min_exceedances_enforced`

---

## 8.8 analysis_matrix_profile_motifs_discords

Purpose:
- detect discord subsequences (time-series anomalies) and motifs.

Algorithm (bounded):
- require time series with enough points
- compute matrix profile approx (STOMP/STAMP-like) or use a fast library if present
- else fallback: sliding-window nearest neighbor distance on downsample

Config:
```yaml
matrix_profile:
  window_size: 50
  max_points: 10000
  top_k_discords: 10
```

Tests:
- `test_discovers_injected_discord`

Reference pointers:
- Matrix Profile papers.

---

# 9) BACKLOG CATEGORY 4 — Event-rate / burst / self-exciting behavior

---

## 9.1 analysis_burst_detection_kleinberg

Purpose:
- detect bursts in event streams (counts of templates/errors per time bucket).

Algorithm:
- choose an event key:
  - template_id from Drain if available
  - else categorical column with name hint `error`, `status`, `type`
- bucket by time; run Kleinberg burst model
- emit bursts with start/end and burst weight

Config:
```yaml
burst:
  bucket: "hour"
  s: 2.0
  gamma: 1.0
  top_k: 20
```

Tests:
- `test_burst_detects_spike_interval`

Reference pointers:
- Kleinberg burst detection.

---

## 9.2 analysis_event_count_bocpd_poisson

Purpose:
- online changepoints for event counts.

Algorithm:
- bucket counts; run BOCPD with Poisson likelihood (conjugate Gamma prior)
- emit changepoints with posterior run-length stats

Config:
```yaml
bocpd_poisson:
  hazard: 0.01
  bucket: "hour"
  max_points: 5000
```

Tests:
- `test_bocpd_poisson_detects_rate_change`

---

## 9.3 analysis_hawkes_self_exciting

Purpose:
- detect cascades / aftershocks (self-excitation) in event arrivals.

Algorithm:
- build timestamps for a chosen event type
- fit simple exponential Hawkes (MLE bounded); or use method-of-moments approximation
- report branching ratio and periods of elevated intensity

Config:
```yaml
hawkes:
  max_events: 20000
  kernel: "exp"
  top_k: 10
```

Tests:
- `test_hawkes_branching_ratio_reasonable` (on simulated Hawkes vs Poisson)

Reference pointers:
- Hawkes 1971.

---

## 9.4 analysis_periodicity_spectral_scan

Purpose:
- detect periodic components (cron jobs, batch effects) in metrics or event counts.

Algorithm:
- for candidate series, compute periodogram/FFT
- find strong peaks above noise floor
- report likely period(s)

Config:
```yaml
spectral:
  max_points: 20000
  top_k_periods: 5
```

Tests:
- `test_detects_injected_periodicity`

---

## 9.5 analysis_state_space_kalman_residuals

Purpose:
- smooth time series and detect anomalies via residuals.

Algorithm:
- simple local-level Kalman filter (can be implemented without heavy deps)
- compute standardized residuals; flag spikes and level shifts

Config:
```yaml
kalman:
  max_points: 20000
  residual_z: 4.0
```

Tests:
- `test_kalman_residual_flags_spike`

---

# 10) BACKLOG CATEGORY 5 — Process mining + sequence intelligence

---

## 10.1 analysis_conformance_alignments

If pm4py available:
- compute alignments fitness; otherwise `skipped` (or use proxy already in 5.8).

Tests:
- `test_alignment_fitness_drop_on_deviation` (pm4py optional marker)

---

## 10.2 analysis_process_drift_conformance_over_time

Purpose:
- rolling conformance score to detect drift timing.

Algorithm:
- compute conformance proxy per rolling window; then changepoint on that score.

Config:
```yaml
conformance_drift:
  windows: 8
  min_cases_per_window: 30
```

Tests:
- `test_conformance_drift_detects_change_point`

---

## 10.3 analysis_variant_differential

Purpose:
- compare “fast vs slow” or “success vs failure” traces.

Algorithm:
- define outcome partition:
  - if duration exists: top 20% slow vs bottom 20% fast
  - else: status/error column if present
- compute transition deltas; test significance with permutation (capped)

Config:
```yaml
variant_diff:
  outcome: "auto"
  top_k: 20
  permutations: 200
```

Tests:
- `test_variant_diff_identifies_discriminative_transition`

---

## 10.4 analysis_markov_transition_shift

Purpose:
- detect significant shifts in transition probabilities over time.

Algorithm:
- build Markov transition matrices per window
- distance metric (JS divergence) and/or χ² tests on transitions
- summarize top changed transitions

Config:
```yaml
markov_shift:
  windows: 6
  min_transitions: 200
```

Tests:
- `test_markov_shift_detects_transition_change`

---

## 10.5 analysis_sequential_patterns_prefixspan

Purpose:
- mine frequent sequential patterns and rare-but-costly patterns.

Algorithm:
- implement PrefixSpan (bounded by max_patterns and max_len)
- associate patterns with outcomes (duration/error)

Config:
```yaml
prefixspan:
  max_patterns: 200
  max_len: 6
  min_support: 0.01
```

Tests:
- `test_prefixspan_finds_known_pattern`

Reference pointers:
- PrefixSpan paper.

---

## 10.6 analysis_hmm_latent_state_sequences

Purpose:
- infer latent states and abnormal state transitions.

Algorithm:
- choose observation symbols (activities/templates)
- fit HMM (if hmmlearn available; else simple Markov baseline)
- report states with high error/duration association

Config:
```yaml
hmm:
  n_states: 6
  max_events: 50000
  fallback: "markov"
```

Tests:
- `test_hmm_fallback_to_markov`
- `test_hmm_states_reasonable` (optional)

Reference pointers:
- Rabiner HMM tutorial.

---

## 10.7 analysis_dependency_graph_change_detection

Purpose:
- build dependency graph (from ids/parent ids) and detect topology changes over time.

Algorithm:
- infer edges from columns like parent_id→id, dependency→id
- compute graph metrics per window:
  - node/edge counts, in-degree/out-degree distributions, SCC count
- changepoint on those curves (reuse PELT/univariate)
- report major shifts and implicated subgraphs (by hashed node ids)

Config:
```yaml
graph_change:
  windows: 6
  max_nodes: 50000
  max_edges: 200000
```

Tests:
- `test_graph_metric_shift_detected`

---

# 11) BACKLOG CATEGORY 6 — Text/message fields (unknown content → structure)

---

## 11.1 analysis_log_template_drain

Same as 5.7 (priority).

---

## 11.2 analysis_topic_model_lda

Purpose:
- surface latent themes in message content/templates.

Secure-by-default:
- Prefer modeling on templates (from Drain) rather than raw messages.
- If no templates, tokenize redacted text and model on hashed tokens.

Algorithm:
- LDA with small K (e.g., 10–30)
- produce top tokens per topic (masked) and topic prevalence drift over time

Config:
```yaml
lda:
  topics: 15
  max_docs: 50000
  max_vocab: 20000
  use_templates_if_present: True
```

Tests:
- `test_lda_runs_on_templates`
- `test_no_raw_pii_tokens_in_topics`

Reference pointers:
- Blei et al. LDA.

---

## 11.3 analysis_term_burst_kleinberg

Same burst algorithm, but on token frequencies (masked).

Config:
```yaml
term_burst:
  bucket: "day"
  top_k_terms: 50
```

Tests:
- `test_term_burst_detects_emergent_token`

---

## 11.4 analysis_message_entropy_drift

Purpose:
- detect “message regime changes” even when templates are not stable.

Algorithm:
- compute per-window entropy of:
  - template_id distribution if available
  - else token distribution (masked)
- detect changepoints in entropy; report windows and likely driver tokens/templates.

Config:
```yaml
entropy:
  windows: 8
  min_docs_per_window: 200
```

Tests:
- `test_entropy_changes_when_new_templates_appear`

---

## 11.5 analysis_template_drift_two_sample

Purpose:
- treat templates as categorical distribution; run χ²/JS divergence over windows.

Config:
```yaml
template_drift:
  alpha: 0.05
  windows: 2   # early vs late
```

Tests:
- `test_template_distribution_shift_detected`

---

# 12) BACKLOG CATEGORY 7 — Dependency + causality (directional signals)

These methods can be expensive and fragile on unknown data. Implement with strict caps and safe fallbacks.

---

## 12.1 analysis_graphical_lasso_dependency_network

Purpose:
- infer conditional dependence graph among numeric metrics.

Algorithm:
- robust standardize
- fit sparse precision matrix via graphical lasso (sklearn if available; else skip)
- emit top edges (i,j) with partial correlation magnitude

Config:
```yaml
glasso:
  max_points: 5000
  max_cols: 40
  alpha: 0.01
  require_sklearn: True
```

Tests:
- `test_glasso_recovers_sparse_structure` (synthetic)
- `test_skip_if_dependency_missing`

Reference pointers:
- Friedman et al. graphical lasso.

---

## 12.2 analysis_mutual_information_screen

Purpose:
- detect nonlinear associations (metric pairs) that correlation misses.

Algorithm:
- choose top numeric cols (<= 30)
- estimate MI with kNN or binning estimator (bounded)
- report top pairs with MI and a simple sanity plot artifact (optional)

Config:
```yaml
mi:
  max_cols: 30
  max_points: 8000
  estimator: "bins"   # "bins" is safer than kNN without dependencies
  top_k: 30
```

Tests:
- `test_mi_detects_nonlinear_relation` (e.g., y=x^2)
- `test_mi_budget_capped`

---

## 12.3 analysis_transfer_entropy_directional

Purpose:
- directional dependence between time series beyond correlation.

Algorithm (bounded):
- require time column and enough points
- discretize series into bins
- compute transfer entropy TE(X→Y) conditioning on history length 1 (default)
- permutation test for significance (capped)
- report top directed pairs

Config:
```yaml
te:
  max_cols: 15
  max_points: 5000
  bins: 5
  permutations: 200
```

Tests:
- `test_te_detects_directionality_in_simulated_causal_pair`
- `test_te_skips_without_time`

Reference pointers:
- Schreiber transfer entropy.

---

## 12.4 analysis_lagged_predictability_test (Granger-style)

Purpose:
- test if past of X improves prediction of Y.

Algorithm:
- build lagged regressions (OLS) with small max_lag
- compare via F-test / AIC (bounded)
- report top pairs and best lag

Config:
```yaml
granger:
  max_cols: 20
  max_points: 8000
  max_lag: 3
```

Tests:
- `test_granger_detects_causal_lag`
- `test_granger_handles_missing`

Reference pointers:
- Granger causality.

---

## 12.5 analysis_copula_dependence

Purpose:
- model dependence structure for non-Gaussian metrics.

Algorithm:
- rank-transform marginals to uniform
- fit simple copula family (Gaussian copula via correlation is simplest)
- report tail dependence proxies if implemented

Config:
```yaml
copula:
  max_cols: 20
  max_points: 10000
  family: "gaussian"
```

Tests:
- `test_copula_rank_invariance`

Reference pointers:
- Copula intro chapter.

---

# 13) BACKLOG CATEGORY 8 — Duration + SLA risk + queueing sanity checks

These are particularly valuable for process logs.

---

## 13.1 analysis_survival_kaplan_meier

Purpose:
- survival curve for time-to-complete with censoring.

Inference:
- detect start/end timestamps; compute duration
- censoring:
  - if end missing but start present ⇒ right-censored at data_end_time

Config:
```yaml
km:
  min_events: 200
  duration_col: "auto"   # or computed
```

Outputs:
- survival curve points (time, S(t))
- median survival (if defined)
- group comparisons (log-rank optional; bounded)

Tests:
- `test_km_monotone_survival`
- `test_censoring_handled`

Reference pointers:
- Kaplan–Meier.

---

## 13.2 analysis_proportional_hazards_duration (Cox PH)

Purpose:
- identify covariates that affect completion hazard (SLA breach risk).

Algorithm:
- if lifelines library available: fit CoxPH
- else fallback: stratified log-rank on top categorical drivers, or quantile regression plugin

Config:
```yaml
cox:
  max_covariates: 30
  require_lifelines: False
  fallback: "stratified_logrank"
```

Tests:
- `test_cox_identifies_known_driver` (synthetic)
- `test_fallback_runs_without_lifelines`

Reference pointers:
- Cox 1972.

---

## 13.3 analysis_quantile_regression_duration

Purpose:
- drivers of p90/p99 durations, not just mean.

Algorithm:
- if statsmodels available: quantile regression
- else fallback: conditional quantile via binning + medians
- report coefficients or directional impacts

Config:
```yaml
qreg:
  quantiles: [0.5, 0.9, 0.99]
  max_covariates: 30
```

Tests:
- `test_qreg_detects_tail_driver`

Reference pointers:
- Koenker & Bassett.

---

## 13.4 analysis_littles_law_consistency

Purpose:
- sanity-check WIP, throughput, lead time consistency per segment.

Inference:
- arrivals from start/queued timestamp
- completions from end timestamp
- WIP approximated via concurrency reconstruction if available

Algorithm:
- compute Little’s Law: L ≈ λ W
- quantify residual; flag segments/time windows with large inconsistencies

Config:
```yaml
littles_law:
  bucket: "day"
  min_bucket_events: 200
  tolerance_ratio: 0.25
```

Tests:
- `test_littles_law_on_simulated_queue`
- `test_flags_inconsistent_bucket`

Reference pointers:
- Little’s law notes.

---

## 13.5 analysis_kingman_vut_approx

Purpose:
- waiting time approximation for G/G/1 to support what-if scenarios.

Algorithm:
- estimate utilization ρ, coefficients of variation Ca, Cs from interarrival and service times
- Kingman: Wq ≈ (ρ/(1-ρ)) * (Ca² + Cs²)/2 * E[S]
- report implied sensitivity to utilization, highlight bottleneck segments

Config:
```yaml
kingman:
  min_events: 500
  bucket: "week"
```

Tests:
- `test_kingman_increases_with_utilization`
- `test_skips_without_service_time`

Reference pointers:
- Heavy traffic / queueing results (Kingman approximation).

---

## 13.6 analysis_queue_model_fit

Purpose:
- fit simple queue models (M/M/1, M/M/c) for rough capacity insights.

Algorithm:
- estimate arrival rate λ and service rate μ from timestamps
- compute implied utilization and expected wait; compare to observed
- select best-fit model by error metric

Config:
```yaml
queue_fit:
  models: ["mm1", "mmc"]
  max_c: 10
  bucket: "day"
```

Tests:
- `test_queue_fit_picks_mm1_on_mm1_sim`
- `test_queue_fit_handles_sparse_data`

---

# 14) Codex CLI “build instructions” (how to generate code from this spec)

For each plugin:
1. Create the module file in `plugins/`.
2. Implement the `run(df, config, context=None)` entry point.
3. Implement `infer_columns(df, config)` and reuse shared utilities if present.
4. Implement bounded compute paths:
   - enforce `time_budget_ms`
   - enforce `max_rows`/`max_cols`
   - deterministically sample
5. Implement safe outputs:
   - cap findings and artifacts
   - redact text by default
6. Add tests:
   - common tests
   - plugin-specific tests from this spec

Minimum done criteria:
- All tests pass.
- Plugin returns `ok` or `skipped` without exceptions for unknown df.
- Deterministic under fixed seed.
- No raw PII leakage in default mode.

---

## 15) Appendix — reference pointers list (for plugin `references` field)

Use these as non-verified pointers in the plugin output:
- PELT: Killick et al. “Optimal detection of changepoints…” (2012), arXiv:1101.1438
- Isolation Forest: Liu et al. (2008)
- Robust PCA / PCP: Candès et al. (2011)
- Drain log parsing: He et al. (ICWS 2017)
- Process Mining conformance: van der Aalst (book)
- MMD: Gretton et al. (JMLR 2012)
- BH FDR: Benjamini & Hochberg (1995)
- Hawkes: Hawkes (1971)
- LDA: Blei et al. (2003)
- HMM tutorial: Rabiner (1989)
- Transfer entropy: Schreiber (2000)
- Granger causality: Granger (1969)
- Kaplan–Meier: Kaplan & Meier (1958)
- Cox PH: Cox (1972)
- Quantile regression: Koenker & Bassett (1978)

End of spec.
