# Statistics Harness — Next30 (Batch 2) Ranked Plugin Pack
Offline, self-contained work order for Codex CLI.

> **Internet note:** This file includes **reference links as pointers** (URLs/DOIs) for humans and for the plugin `references` field.  
> These external links were **not fetched/validated in-session**. Treat them as **NO EVIDENCE** until you verify independently.

---

## 0) Non-negotiable repo contracts (do not break)

### 0.1 Execution model
- Plugins are discovered from `plugins/*/plugin.yaml` (manifested plugins).
- Plugins execute in a sandboxed subprocess; `ctx.logger(...)` writes to `run_dir/logs/<plugin_id>.log`.
- Stat-plugins are routed through `statistic_harness.core.stat_plugins.registry.run_plugin(...)`.

### 0.2 Available helpers you should reuse
From `statistic_harness.core.stat_plugins` (already in repo):
- `BudgetTimer`, `infer_columns`, `merge_config`, `deterministic_sample`
- robust stats: `robust_center_scale`, `robust_zscores`, `standardized_median_diff`, `cliffs_delta`, `cramers_v`
- multiple testing: `bh_fdr`
- ids: `stable_id`
- redaction: `build_redactor`, `redact_text`, `redact_series`

---

## 1) The 30 new plugin IDs (Batch 2)

### A) Highest priority (best worst-pillar scores)
1. `analysis_beta_binomial_overdispersion_v1`
2. `analysis_circular_time_of_day_drift_v1`
3. `analysis_mann_kendall_trend_test_v1`
4. `analysis_quantile_mapping_drift_qq_v1`

### B) Strong, generally useful
5. `analysis_constraints_violation_detector_v1`
6. `analysis_negative_binomial_overdispersion_v1`
7. `analysis_partial_correlation_network_shift_v1`
8. `analysis_piecewise_linear_trend_changepoints_v1`
9. `analysis_poisson_regression_rate_drivers_v1`
10. `analysis_quantile_sketch_p2_streaming_v1`
11. `analysis_robust_regression_huber_ransac_v1`
12. `analysis_state_space_smoother_level_shift_v1`

### C) Medium priority (still good; more assumptions)
13. `analysis_aft_survival_lognormal_v1`
14. `analysis_competing_risks_cif_v1`
15. `analysis_haar_wavelet_transient_detector_v1`
16. `analysis_hurst_exponent_long_memory_v1`
17. `analysis_permutation_entropy_drift_v1`

### D) Useful, but more domain-sensitive
18. `analysis_capacity_frontier_envelope_v1`
19. `analysis_graph_assortativity_shift_v1`
20. `analysis_graph_pagerank_hotspots_v1`
21. `analysis_higuchi_fractal_dimension_v1`
22. `analysis_marked_point_process_intensity_v1`
23. `analysis_spectral_radius_stability_v1`

### E) Expensive (keep last; strict caps)
24. `analysis_bootstrap_ci_effect_sizes_v1`
25. `analysis_energy_distance_two_sample_v1`
26. `analysis_randomization_test_median_shift_v1`
27. `analysis_distance_covariance_dependence_v1`
28. `analysis_graph_motif_triads_shift_v1`
29. `analysis_multiscale_entropy_mse_v1`
30. `analysis_sample_entropy_irregularity_v1`

---

## 2) Pillars ranking (0–3 per pillar; leximin order)

### 2.1 Scoring rubric (INFERENCE)
- Performant (P): 3 = linear-ish / capped; 2 = moderate model fit; 1 = expensive (quadratic / bootstrap) but capped.
- Accurate (A): 3 = robust & standard; 2 = useful but more heuristic/assumption-heavy; 1 = fragile.
- Secure (S): 3 = aggregate-only outputs (this pack assumes 3 everywhere).
- Citable (C): 3 = interpretable metrics + artifacts; 2 = interpretable but weaker direct evidence.

### 2.2 Ranked table
|Rank|Plugin ID|P|A|S|C|Worst pillar(s)|
|---:|---|---:|---:|---:|---:|---|
|1|analysis_beta_binomial_overdispersion_v1|3|3|3|3|P,A,S,C=3|
|2|analysis_circular_time_of_day_drift_v1|3|3|3|3|P,A,S,C=3|
|3|analysis_mann_kendall_trend_test_v1|3|3|3|3|P,A,S,C=3|
|4|analysis_quantile_mapping_drift_qq_v1|3|3|3|3|P,A,S,C=3|
|5|analysis_constraints_violation_detector_v1|3|2|3|3|A=2|
|6|analysis_negative_binomial_overdispersion_v1|3|2|3|3|A=2|
|7|analysis_partial_correlation_network_shift_v1|2|3|3|3|P=2|
|8|analysis_piecewise_linear_trend_changepoints_v1|2|3|3|3|P=2|
|9|analysis_poisson_regression_rate_drivers_v1|2|3|3|3|P=2|
|10|analysis_quantile_sketch_p2_streaming_v1|3|2|3|3|A=2|
|11|analysis_robust_regression_huber_ransac_v1|2|3|3|3|P=2|
|12|analysis_state_space_smoother_level_shift_v1|2|3|3|3|P=2|
|13|analysis_aft_survival_lognormal_v1|2|2|3|3|P,A=2|
|14|analysis_competing_risks_cif_v1|2|2|3|3|P,A=2|
|15|analysis_haar_wavelet_transient_detector_v1|3|2|3|2|A,C=2|
|16|analysis_hurst_exponent_long_memory_v1|3|2|3|2|A,C=2|
|17|analysis_permutation_entropy_drift_v1|3|2|3|2|A,C=2|
|18|analysis_capacity_frontier_envelope_v1|2|2|3|2|P,A,C=2|
|19|analysis_graph_assortativity_shift_v1|2|2|3|2|P,A,C=2|
|20|analysis_graph_pagerank_hotspots_v1|2|2|3|2|P,A,C=2|
|21|analysis_higuchi_fractal_dimension_v1|2|2|3|2|P,A,C=2|
|22|analysis_marked_point_process_intensity_v1|2|2|3|2|P,A,C=2|
|23|analysis_spectral_radius_stability_v1|2|2|3|2|P,A,C=2|
|24|analysis_bootstrap_ci_effect_sizes_v1|1|3|3|3|P=1|
|25|analysis_energy_distance_two_sample_v1|1|3|3|3|P=1|
|26|analysis_randomization_test_median_shift_v1|1|3|3|3|P=1|
|27|analysis_distance_covariance_dependence_v1|1|3|3|2|P=1|
|28|analysis_graph_motif_triads_shift_v1|1|2|3|2|P=1|
|29|analysis_multiscale_entropy_mse_v1|1|2|3|2|P=1|
|30|analysis_sample_entropy_irregularity_v1|1|2|3|2|P=1|

---

## 3) Architecture to implement as plugins (Codex instructions)

### 3.1 Implement as stat-plugin handlers (one module)
Create:
- `src/statistic_harness/core/stat_plugins/next30b_addon.py`

It must export:
- `HANDLERS: dict[str, Callable[..., PluginResult]]`

Wire into:
- `src/statistic_harness/core/stat_plugins/registry.py`
  - import and `HANDLERS.update(NEXT30B_HANDLERS)` near the bottom.

### 3.2 Add wrapper plugin directories (30)
For each plugin id `<plugin_id>`:
- `plugins/<plugin_id>/plugin.yaml`
- `plugins/<plugin_id>/plugin.py` (thin wrapper: `run_plugin("<plugin_id>", ctx)`)
- `plugins/<plugin_id>/config.schema.json`
- `plugins/<plugin_id>/output.schema.json`

### 3.3 Logging requirement (hard)
Each handler must:
- `ctx.logger("START ...")`
- `ctx.logger("SKIP ...")` if skipped
- `ctx.logger("END ...")`

Also write one artifact JSON per plugin to `ctx.artifacts_dir(PID)`.

### 3.4 Capping rules (hard)
- Any quadratic step must enforce `plugin.max_points_for_quadratic` (default 2000).
- Any bootstrap/permutation must enforce:
  - `plugin.max_resamples` (default 200)
  - `plugin.max_points_for_quadratic` cap
  - time budget via `BudgetTimer`.

---

## 4) Standard templates (copy/paste)

### 4.1 `plugins/<plugin_id>/plugin.py`
```python
from __future__ import annotations

from statistic_harness.core.stat_plugins.registry import run_plugin


class Plugin:
    def run(self, ctx):
        return run_plugin("<plugin_id>", ctx)
```

### 4.2 `plugins/<plugin_id>/plugin.yaml`
```yaml
id: <plugin_id>
name: "<plugin_id>"
version: "0.1.0"
type: analysis
entrypoint: "plugin.py:Plugin"
depends_on: []
capabilities: [analysis]
config_schema: "config.schema.json"
output_schema: "output.schema.json"
settings:
  defaults:
    seed: 1337
    time_budget_ms: 25000
    max_rows: 200000
    max_cols: 80
    max_pairs: 2000
    max_windows: 64
    max_findings: 30
    allow_row_sampling: false
    privacy:
      enable_redaction: true
      redact_patterns: ["email","ip","uuid","credit_card","phone"]
      max_exemplars: 3
      allow_exemplar_snippets: false
    verbosity: "normal"
    plugin:
      max_points_for_quadratic: 2000
      max_resamples: 200
sandbox:
  no_network: true
  fs_allowlist: [run_dir]
```

### 4.3 `config.schema.json` and `output.schema.json`
Reuse the same base schemas used for the first Next30 pack (no new schema required).  
Only add optional plugin-specific keys under `settings.defaults.plugin`.

---

## 5) Shared handler utilities (required)

In `next30b_addon.py`, implement these helpers to keep 30 handlers consistent:

1. `_artifact_json(ctx, plugin_id, filename, payload, description)`
   - writes deterministic JSON (sort keys)
   - returns `PluginArtifact` with relative path

2. `_basic_metrics(df, sample_meta)` using `rows_seen`, `rows_used`, `cols_used`

3. `_make_finding(...)` returning a dict with stable `id` via `stable_id(...)`

4. `_cap_quadratic(df, config, timer, reason)`:
   - if `len(df) > max_points_for_quadratic` and `allow_row_sampling` is False → return `skipped`
   - else deterministically sample rows (seeded) to that cap

5. `_split_pre_post(df, time_col)`:
   - pre = first half, post = second half (time-sorted if available)

---

## 6) Plugin-by-plugin specs (design, metrics, artifacts, references)

> Conventions:
> - Use `inferred = infer_columns(df, config)`
> - `time_col = inferred.get("time_column")`
> - `numeric_cols = inferred.get("numeric_columns") or []`
> - `cat_cols = inferred.get("categorical_columns") or []`
> - All outputs are aggregate-only (no raw text).

Each spec includes:
- **Metrics**: fields to put in `PluginResult.metrics`
- **Artifact**: JSON file name + payload outline
- **References**: keys to add to `references.py` (see §7)

---

### 6.1 analysis_beta_binomial_overdispersion_v1
**Purpose:** detect overdispersion in binary outcomes vs binomial expectation.  
**Requires:** a binary-like column (0/1 or boolean) and optional group/time.

Algorithm:
- Find candidate binary columns: numeric with unique subset {0,1} or boolean dtype.
- Partition into windows (pre/post if time exists; else by top group column if available; else entire dataset).
- For each partition:
  - compute `p = mean(x)`, `n = count`
- Overdispersion proxy:
  - compute variance of `p` across partitions vs binomial variance proxy `p(1-p)/n̄`
  - emit `dispersion_ratio`
- Findings: top column(s) with highest dispersion ratio.

Metrics:
- `binary_columns_scanned`, `partitions`, `dispersion_ratio_max`

Artifact: `beta_binomial_overdispersion.json`
- `{column, partitions: [...], dispersion_ratio, p_values: [...]}`

References: `beta_binomial`

---

### 6.2 analysis_circular_time_of_day_drift_v1
**Purpose:** drift in time-of-day distribution (circular stats).  
**Requires:** time column.

Algorithm:
- Parse time; extract hour-of-day as angle θ = 2π*(hour/24).
- Pre/post split (time-sorted).
- Compute mean resultant vector:
  - `C = mean(cos θ)`, `S = mean(sin θ)`, `R = sqrt(C^2 + S^2)`
  - mean direction `mu = atan2(S,C)`
- Drift metrics:
  - `delta_mu` (circular distance), `delta_R`
- Warn if `delta_mu` > threshold (e.g., 1 rad) or `delta_R` > 0.2.

Metrics:
- `delta_mu_rad`, `delta_R`, `pre_R`, `post_R`

Artifact: `circular_time_drift.json`

References: `circular_stats`

---

### 6.3 analysis_mann_kendall_trend_test_v1
**Purpose:** nonparametric monotone trend detection.  
**Requires:** time + numeric series.

Algorithm:
- Choose one numeric column (top variance).
- Time-sort; compute Mann-Kendall S statistic:
  - S = Σ_{i<j} sign(x_j - x_i)
- Normal approximation with tie correction (bounded; use cap n<=5000 else sample by time buckets).
- Compute p-value approximation; also Sen slope estimate (median of pairwise slopes; cap pairs).
- Emit finding when p < 0.05 and |slope| is meaningful.

Metrics:
- `S`, `z`, `p_value`, `sen_slope`

Artifact: `mann_kendall.json`

References: `mann_kendall`

---

### 6.4 analysis_quantile_mapping_drift_qq_v1
**Purpose:** distribution drift via Q-Q distance between pre/post.  
**Requires:** numeric columns; time optional.

Algorithm:
- Split pre/post (time if available else first/second half).
- For each numeric col (cap max_cols):
  - compute quantiles at fixed grid (e.g., 0.05..0.95 step 0.05)
  - compute `qq_l1 = mean(|q_pre - q_post| / scale)` where scale = MAD(pre)
- Rank columns by qq_l1.

Metrics:
- `cols_scanned`, `top_qq_l1`

Artifact: `qq_drift.json` with quantile tables.

References: `qq_plot`

---

### 6.5 analysis_constraints_violation_detector_v1
**Purpose:** detect and quantify invariant violations (range/monotonic/nonnegativity) and their drift.  
**Requires:** any numeric; optional time.

Algorithm:
- For each numeric col:
  - infer simple constraints:
    - if col name contains `count`, `qty`, `num` → nonnegative integer-like
    - if 99% values >=0 → nonnegative
  - compute violation rate pre and post
- Emit findings where violation rate increases or is high.

Metrics:
- `constraints_checked`, `violations_total`, `max_violation_rate`

Artifact: `constraint_violations.json`

References: `data_quality` (new)

---

### 6.6 analysis_negative_binomial_overdispersion_v1
**Purpose:** detect overdispersion in counts; estimate NB parameters.  
**Requires:** count-like numeric column (>=0 integers).

Algorithm:
- Identify count columns (>=90% integers, >=0).
- For each:
  - mean m, var v
  - overdispersion ratio r = v / max(m, eps)
  - NB size k ≈ m^2 / max(v - m, eps)
- Emit finding for high ratio.

Metrics:
- `count_cols_scanned`, `overdispersion_ratio_max`

Artifact: `nb_overdispersion.json`

References: `negative_binomial`

---

### 6.7 analysis_partial_correlation_network_shift_v1
**Purpose:** conditional dependency shift via partial correlations.  
**Requires:** >=3 numeric cols.

Algorithm:
- Choose top p cols by variance (cap p<=25).
- Pre/post split.
- Estimate precision matrix:
  - use sklearn `LedoitWolf` covariance then invert (or pseudo-inverse)
- Partial corr:
  - ρ_ij = -P_ij / sqrt(P_ii P_jj)
- Compare pre vs post: max |Δρ| edge.

Metrics:
- `p_cols`, `max_delta_partial_corr`

Artifact: `partial_corr_shift.json` with top edges.

References: `partial_correlation`, `ledoit_wolf`

---

### 6.8 analysis_piecewise_linear_trend_changepoints_v1
**Purpose:** segmented trend (piecewise linear) and changepoints.  
**Requires:** time + numeric.

Algorithm:
- Bucket time series.
- Use dynamic programming with K segments (K<=5) minimizing SSE with linear fits, OR greedy split until improvement < penalty.
- Emit changepoints and segment slopes.

Metrics:
- `segments`, `changepoints`, `best_sse`

Artifact: `piecewise_trend.json`

References: `piecewise_linear`, `prophet_2017` (pointer)

---

### 6.9 analysis_poisson_regression_rate_drivers_v1
**Purpose:** drivers of event-rate/count column using Poisson regression.  
**Requires:** count target + predictors.

Algorithm:
- Choose target count column (most count-like).
- Choose predictors: numeric + one-hot top categorical (cap).
- Fit GLM-like Poisson via IRLS (implement minimal) or sklearn `PoissonRegressor` if available.
- Output top coefficients.

Metrics:
- `target`, `n_features`, `deviance`

Artifact: `poisson_regression.json`

References: `poisson_regression`, `glm_nelder_wedderburn`

---

### 6.10 analysis_quantile_sketch_p2_streaming_v1
**Purpose:** fast streaming quantiles for huge datasets (avoid full sort).  
**Requires:** numeric column(s).

Algorithm:
- Implement P² algorithm (Jain & Chlamtac) for q in {0.5,0.9,0.99}.
- For each numeric col (cap):
  - stream through values, update markers
- Output approximate quantiles.

Metrics:
- `cols_processed`, `quantiles`

Artifact: `p2_quantiles.json`

References: `p2_quantile`

---

### 6.11 analysis_robust_regression_huber_ransac_v1
**Purpose:** robust relationship discovery; outlier identification.  
**Requires:** numeric pair(s).

Algorithm:
- Select candidate y and x (top variance).
- Fit Huber regression (iteratively reweighted least squares) and RANSAC line fit (deterministic sampling via seed).
- Compare residual distributions; emit outlier count and robust slope.

Metrics:
- `slope_huber`, `slope_ransac`, `outliers`

Artifact: `robust_regression.json`

References: `huber_1964`, `ransac_1981`

---

### 6.12 analysis_state_space_smoother_level_shift_v1
**Purpose:** level shift detection via smoothing (Kalman-style).  
**Requires:** time + numeric.

Algorithm:
- Use local level model:
  - forward filter (as in existing kalman residual plugin patterns)
  - backward RTS smoother (implement minimal)
- Identify point where smoothed level changes most.

Metrics:
- `max_level_shift`, `index`

Artifact: `state_space_level_shift.json`

References: `kalman_1960`, `rts_smoother`

---

### 6.13 analysis_aft_survival_lognormal_v1
**Purpose:** duration drivers using AFT (lognormal) model.  
**Requires:** duration-like numeric; covariates optional.

Algorithm:
- Select duration column by name hints.
- Fit log(duration) ~ X via OLS/ridge (AFT proxy).
- Emit top coefficients and predicted p90 shift across groups.

Metrics:
- `duration_col`, `n_covariates`, `r2`

Artifact: `aft_lognormal.json`

References: `aft_survival`

---

### 6.14 analysis_competing_risks_cif_v1
**Purpose:** competing risks cumulative incidence (CIF).  
**Requires:** event type categorical + duration/time-to-event.

Algorithm:
- Identify event type column (categorical with name hints `status`, `type`, `outcome`).
- Identify duration column.
- Estimate CIF per event type using cumulative hazard approximation over time bins.
- Emit top event types and CIF at t50/t90.

Metrics:
- `event_types`, `cif_at_t`

Artifact: `competing_risks.json`

References: `competing_risks`, `fine_gray_1999` (pointer)

---

### 6.15 analysis_haar_wavelet_transient_detector_v1
**Purpose:** multiscale transient detection (step/spike).  
**Requires:** numeric series; time optional.

Algorithm:
- Cap n (<= max_points_for_quadratic not required; this is O(n)).
- Compute Haar wavelet coefficients at scales 1,2,4,... up to 2^k.
- Flag indices with large coefficient magnitude relative to MAD.

Metrics:
- `max_coeff_z`, `scale`

Artifact: `haar_transients.json`

References: `haar_1910`, `wavelets_intro`

---

### 6.16 analysis_hurst_exponent_long_memory_v1
**Purpose:** long memory / persistence detection.  
**Requires:** numeric time series.

Algorithm:
- Use rescaled range (R/S) approximation across window sizes.
- Fit log(R/S) vs log(window) slope = H.
- Warn if H > 0.7 (persistent) or < 0.3 (anti-persistent).

Metrics:
- `hurst_H`, `fit_r2`

Artifact: `hurst.json`

References: `hurst_1951`

---

### 6.17 analysis_permutation_entropy_drift_v1
**Purpose:** complexity drift via permutation entropy.  
**Requires:** numeric time series.

Algorithm:
- Choose embedding dimension m=3, delay=1.
- Count ordinal patterns; compute normalized entropy.
- Compare pre vs post; warn if |Δ| > 0.1.

Metrics:
- `H_pre`, `H_post`, `delta`

Artifact: `permutation_entropy.json`

References: `bandt_pompe_2002`

---

### 6.18 analysis_capacity_frontier_envelope_v1
**Purpose:** envelope/efficiency frontier (DEA-style) for capacity analysis.  
**Requires:** numeric “input” and “output” columns.

Algorithm:
- Choose two columns:
  - output = throughput-like (name hints `completed`, `processed`, `count`)
  - input = resource/time-like (name hints `cpu`, `duration`, `time`, `server`)
- Compute convex hull frontier in 2D (upper envelope).
- For each point: efficiency = y / y_frontier(x).
- Emit worst-efficiency segments/groups.

Metrics:
- `frontier_points`, `min_efficiency`

Artifact: `capacity_frontier.json`

References: `dea_ccr_1978`

---

### 6.19 analysis_graph_assortativity_shift_v1
**Purpose:** mixing pattern drift in graphs over time windows.  
**Requires:** ability to infer a graph edge list OR build co-occurrence graph.

Algorithm:
- Build graph:
  - prefer columns like `parent_id->id`, `from->to`, `src->dst`
  - else build co-occurrence graph from categorical pairs in same row (cap).
- Compute assortativity (numeric or categorical) for pre and post windows.
- Emit Δ assortativity.

Metrics:
- `assort_pre`, `assort_post`, `delta`

Artifact: `graph_assortativity.json`

References: `newman_assortativity_2002`

---

### 6.20 analysis_graph_pagerank_hotspots_v1
**Purpose:** identify central nodes as hotspots (risk/bottleneck) in dependency graphs.  
**Requires:** graph.

Algorithm:
- Build graph as above.
- Compute PageRank (networkx).
- Emit top-k nodes (hashed ids), plus concentration metric (top10 share).

Metrics:
- `nodes`, `edges`, `top10_share`

Artifact: `pagerank_hotspots.json`

References: `pagerank_1998`

---

### 6.21 analysis_higuchi_fractal_dimension_v1
**Purpose:** complexity of time series via Higuchi fractal dimension.  
**Requires:** numeric series.

Algorithm:
- Implement Higuchi FD for k_max (default 10).
- Compute FD; compare pre/post.

Metrics:
- `fd`, `k_max`

Artifact: `higuchi_fd.json`

References: `higuchi_1988`

---

### 6.22 analysis_marked_point_process_intensity_v1
**Purpose:** event intensity over time with marks (severity/size).  
**Requires:** time column and mark column (numeric/categorical).

Algorithm:
- Bucket time to hours/days.
- Intensity λ(t) = counts per bucket.
- Mark summary per bucket (mean/entropy).
- Detect changepoint in λ(t) via rolling mean shift.

Metrics:
- `lambda_mean`, `lambda_p95`, `max_shift`

Artifact: `marked_intensity.json`

References: `point_process_ogata_1988` (pointer)

---

### 6.23 analysis_spectral_radius_stability_v1
**Purpose:** stability proxy via leading eigenvalue of correlation/adjacency matrix.  
**Requires:** numeric block or graph.

Algorithm:
- If numeric:
  - corr matrix on top p cols; compute largest eigenvalue λ_max
  - compare pre vs post
- If graph:
  - adjacency; compute spectral radius via power iteration
- Emit Δλ.

Metrics:
- `lambda_pre`, `lambda_post`, `delta`

Artifact: `spectral_radius.json`

References: `perron_frobenius` (pointer)

---

### 6.24 analysis_bootstrap_ci_effect_sizes_v1 (expensive)
**Purpose:** uncertainty quantification for effect sizes (bootstrap CI).  
**Requires:** numeric + group split.

Algorithm:
- Define group split:
  - pre/post if time, else top category vs rest.
- For top numeric cols (cap):
  - effect = standardized median diff
  - bootstrap B resamples (B<=max_resamples) for CI [2.5%,97.5%]
- Emit only if CI excludes 0.

Metrics:
- `B`, `effects_with_ci`

Artifact: `bootstrap_effect_ci.json`

References: `efron_1979_bootstrap`

---

### 6.25 analysis_energy_distance_two_sample_v1 (expensive)
**Purpose:** robust two-sample test using energy distance.  
**Requires:** numeric vector or multivariate block.

Algorithm:
- Cap rows to max_points_for_quadratic.
- Energy distance:
  - E = 2 E|X-Y| - E|X-X'| - E|Y-Y'|
- Permutation p-value with B<=max_resamples.
- Emit if p <= 0.05.

Metrics:
- `energy_stat`, `p_value`, `B`

Artifact: `energy_distance.json`

References: `energy_distance_szekely_2004`

---

### 6.26 analysis_randomization_test_median_shift_v1 (expensive)
**Purpose:** permutation test for median shift (robust).  
**Requires:** numeric + split.

Algorithm:
- Split pre/post.
- Observed Δ = median(post) - median(pre)
- Permute labels B times (<=max_resamples) to get p-value.
- Emit if p <= 0.05.

Metrics:
- `delta_median`, `p_value`, `B`

Artifact: `median_randomization.json`

References: `permutation_tests_fisher_1935` (pointer)

---

### 6.27 analysis_distance_covariance_dependence_v1 (expensive)
**Purpose:** dependence test via distance covariance (nonlinear).  
**Requires:** numeric pair (x,y). Quadratic.

Algorithm:
- Cap rows to max_points_for_quadratic.
- Compute distance covariance; optional permutation p-value.
- Emit top dependent pairs.

Metrics:
- `dcov`, `dcor`, `p_value`

Artifact: `distance_covariance.json`

References: `distance_covariance_szekely_2007`

---

### 6.28 analysis_graph_motif_triads_shift_v1 (expensive)
**Purpose:** triad motif frequency drift in directed graphs.  
**Requires:** directed graph.

Algorithm:
- Build directed graph (cap nodes/edges).
- Count triad motifs (use networkx triadic census if available).
- Compare pre/post motif distributions (L1 distance).
- Emit if drift high.

Metrics:
- `triads_total`, `l1_drift`

Artifact: `triad_motifs.json`

References: `network_motifs_milo_2002`

---

### 6.29 analysis_multiscale_entropy_mse_v1 (expensive)
**Purpose:** multiscale entropy (MSE) for complexity changes.  
**Requires:** numeric time series.

Algorithm:
- Cap n to max_points_for_quadratic.
- For scales s=1..S (S<=10):
  - coarse-grain by averaging in non-overlapping windows
  - compute sample entropy (calls internal sample entropy)
- Compare pre/post average entropy.

Metrics:
- `scales`, `mse_pre`, `mse_post`, `delta`

Artifact: `mse.json`

References: `mse_costa_2002`

---

### 6.30 analysis_sample_entropy_irregularity_v1 (expensive)
**Purpose:** sample entropy as irregularity metric.  
**Requires:** numeric time series.

Algorithm:
- Cap n to max_points_for_quadratic.
- Sample entropy:
  - m=2, r=0.2*std
  - count matches of length m and m+1
- Compute SampEn = -log(A/B).
- Compare pre/post.

Metrics:
- `sampen_pre`, `sampen_post`, `delta`

Artifact: `sample_entropy.json`

References: `sampen_richman_moorman_2000`

---

## 7) Reference links to embed (NO EVIDENCE pointers)

### 7.1 Implementation requirement
Update `src/statistic_harness/core/stat_plugins/references.py`:
- Add new entries under `REFERENCE_LIBRARY` for keys listed below.
- Update `default_references_for_plugin(plugin_id)` to include these keys based on plugin_id matching.

### 7.2 New reference entries (suggested)
Add these keys and URLs (pointers only):

- `beta_binomial`:
  - title: Beta-binomial distribution (overview)
  - url: https://en.wikipedia.org/wiki/Beta-binomial_distribution

- `circular_stats`:
  - title: Fisher (1993) Statistical Analysis of Circular Data (overview)
  - url: https://en.wikipedia.org/wiki/Circular_statistics

- `mann_kendall`:
  - title: Mann–Kendall trend test (overview)
  - url: https://en.wikipedia.org/wiki/Mann%E2%80%93Kendall_trend_test

- `qq_plot`:
  - title: Q–Q plot (overview)
  - url: https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot

- `data_quality`:
  - title: Data quality dimensions (overview)
  - url: https://en.wikipedia.org/wiki/Data_quality

- `negative_binomial`:
  - title: Negative binomial distribution (overview)
  - url: https://en.wikipedia.org/wiki/Negative_binomial_distribution

- `partial_correlation`:
  - title: Partial correlation (overview)
  - url: https://en.wikipedia.org/wiki/Partial_correlation

- `ledoit_wolf`:
  - title: Ledoit & Wolf (2004) covariance shrinkage (overview)
  - url: https://en.wikipedia.org/wiki/Ledoit%E2%80%93Wolf_shrinkage

- `prophet_2017`:
  - title: Taylor & Letham (2017) Forecasting at scale (Prophet)
  - url: https://peerj.com/preprints/3190/

- `poisson_regression`:
  - title: Poisson regression (overview)
  - url: https://en.wikipedia.org/wiki/Poisson_regression

- `glm_nelder_wedderburn`:
  - title: Nelder & Wedderburn (1972) Generalized Linear Models (overview)
  - url: https://en.wikipedia.org/wiki/Generalized_linear_model

- `p2_quantile`:
  - title: Jain & Chlamtac (1985) P² algorithm (overview)
  - url: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Approximate_quantiles

- `huber_1964`:
  - title: Huber loss / robust estimation (overview)
  - url: https://en.wikipedia.org/wiki/Huber_loss

- `ransac_1981`:
  - title: RANSAC (overview)
  - url: https://en.wikipedia.org/wiki/Random_sample_consensus

- `rts_smoother`:
  - title: Rauch–Tung–Striebel smoother (overview)
  - url: https://en.wikipedia.org/wiki/Kalman_filter#Rauch%E2%80%93Tung%E2%80%93Striebel

- `aft_survival`:
  - title: Accelerated failure time model (overview)
  - url: https://en.wikipedia.org/wiki/Accelerated_failure_time_model

- `competing_risks`:
  - title: Competing risks (overview)
  - url: https://en.wikipedia.org/wiki/Competing_risks

- `fine_gray_1999`:
  - title: Fine & Gray (1999) subdistribution hazards (overview)
  - url: https://en.wikipedia.org/wiki/Competing_risks#Fine%E2%80%93Gray_model

- `haar_1910`:
  - title: Haar wavelet (overview)
  - url: https://en.wikipedia.org/wiki/Haar_wavelet

- `wavelets_intro`:
  - title: Wavelet transform (overview)
  - url: https://en.wikipedia.org/wiki/Wavelet_transform

- `hurst_1951`:
  - title: Hurst exponent (overview)
  - url: https://en.wikipedia.org/wiki/Hurst_exponent

- `bandt_pompe_2002`:
  - title: Permutation entropy (overview)
  - url: https://en.wikipedia.org/wiki/Permutation_entropy

- `dea_ccr_1978`:
  - title: Data envelopment analysis (CCR model) (overview)
  - url: https://en.wikipedia.org/wiki/Data_envelopment_analysis

- `newman_assortativity_2002`:
  - title: Newman (2002) assortative mixing (overview)
  - url: https://en.wikipedia.org/wiki/Assortativity

- `pagerank_1998`:
  - title: PageRank (overview)
  - url: https://en.wikipedia.org/wiki/PageRank

- `higuchi_1988`:
  - title: Higuchi fractal dimension (overview)
  - url: https://en.wikipedia.org/wiki/Fractal_dimension#Time_series

- `point_process_ogata_1988`:
  - title: Point process intensity estimation (overview)
  - url: https://en.wikipedia.org/wiki/Point_process

- `perron_frobenius`:
  - title: Spectral radius / Perron–Frobenius (overview)
  - url: https://en.wikipedia.org/wiki/Spectral_radius

- `efron_1979_bootstrap`:
  - title: Bootstrap (overview)
  - url: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)

- `energy_distance_szekely_2004`:
  - title: Energy distance (overview)
  - url: https://en.wikipedia.org/wiki/Energy_distance

- `permutation_tests_fisher_1935`:
  - title: Permutation test (overview)
  - url: https://en.wikipedia.org/wiki/Permutation_test

- `distance_covariance_szekely_2007`:
  - title: Distance covariance (overview)
  - url: https://en.wikipedia.org/wiki/Distance_correlation

- `network_motifs_milo_2002`:
  - title: Network motifs (overview)
  - url: https://en.wikipedia.org/wiki/Network_motif

- `mse_costa_2002`:
  - title: Multiscale entropy (overview)
  - url: https://en.wikipedia.org/wiki/Multiscale_entropy

- `sampen_richman_moorman_2000`:
  - title: Sample entropy (overview)
  - url: https://en.wikipedia.org/wiki/Sample_entropy

### 7.3 Plugin → reference keys mapping (suggested)
In `default_references_for_plugin`, add string matches:
- `beta_binomial_overdispersion` → `beta_binomial`
- `circular_time_of_day` → `circular_stats`
- `mann_kendall` → `mann_kendall`
- `quantile_mapping` or `qq` → `qq_plot`
- `constraints_violation` → `data_quality`
- `negative_binomial` → `negative_binomial`
- `partial_correlation` → `partial_correlation`, `ledoit_wolf`
- `piecewise_linear_trend` → `prophet_2017`
- `poisson_regression` → `poisson_regression`, `glm_nelder_wedderburn`
- `p2_streaming` → `p2_quantile`
- `huber` → `huber_1964`
- `ransac` → `ransac_1981`
- `rts` or `smoother` → `rts_smoother`, `kalman`
- `aft_survival` → `aft_survival`
- `competing_risks` or `cif` → `competing_risks`, `fine_gray_1999`
- `haar_wavelet` → `haar_1910`, `wavelets_intro`
- `hurst_exponent` → `hurst_1951`
- `permutation_entropy` → `bandt_pompe_2002`
- `capacity_frontier` or `dea` → `dea_ccr_1978`
- `assortativity` → `newman_assortativity_2002`
- `pagerank` → `pagerank_1998`
- `higuchi` → `higuchi_1988`
- `marked_point_process` → `point_process_ogata_1988`
- `spectral_radius` → `perron_frobenius`
- `bootstrap_ci` → `efron_1979_bootstrap`
- `energy_distance` → `energy_distance_szekely_2004`
- `randomization_test` → `permutation_tests_fisher_1935`
- `distance_covariance` → `distance_covariance_szekely_2007`
- `graph_motif` or `triads` → `network_motifs_milo_2002`
- `multiscale_entropy` → `mse_costa_2002`
- `sample_entropy` → `sampen_richman_moorman_2000`

---

## 8) Test plan (Codex must implement)

Create `tests/test_next30b_plugins_smoke.py` (parameterized):
- Build two small datasets (<=2000 rows):
  - `ts.csv`: time + numeric + count + categorical
  - `ts_shift.csv`: same but with clear shift
- For each plugin_id:
  - run `Pipeline.run(csv, [plugin_id], {}, seed)`
  - assert status in `{ok, skipped}` (never error)
  - assert log file exists and non-empty: `run_dir/logs/<plugin_id>.log`
- Determinism (only when status == ok):
  - run twice with same seed and assert stored `metrics/findings/summary/status` match exactly.

Environment in tests:
- `STAT_HARNESS_RETENTION_ENABLED=0`
- `STAT_HARNESS_MAX_WORKERS_ANALYSIS=1`

---

## 9) Codex CLI work instructions (offline)

Codex must create/append `.codex/STATE.md` including:
- files added/changed
- brief why
- commands run and outputs

Commands:
```bash
poetry install --with dev
pytest -q
```

If installs are unavailable in the execution environment:
- skip execution and rely on CI
- do not claim tests failing unless they actually ran with deps installed

---

## 10) Acceptance criteria
- 30 plugin directories exist and are discoverable.
- `next30b_addon.py` implements all handlers and is registered in `registry.py`.
- Each plugin writes non-empty `run_dir/logs/<plugin_id>.log`.
- No plugin returns `error` on the smoke datasets.
- Deterministic outputs for `status=="ok"` across identical runs.
- `pytest` passes.

END
