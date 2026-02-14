# Statistics Harness — Next30 Plugin Pack (Implementation Work Order)
**Scope:** Implement 30 new `analysis_*` plugins in `ninjra/statistics_harness` with **offline-only** Codex execution.  
**Design constraint:** Codex must not require the internet; all method specs, schemas, and templates needed are in this file.

---

## 1) Repository contracts you must not break

### 1.1 Plugin discovery and execution (read before coding)
- Plugins are discovered by scanning `plugins/*/plugin.yaml` at runtime.
- Each plugin runs in a **subprocess** via `statistic_harness.core.plugin_runner`, under:
  - deterministic env (`PYTHONHASHSEED`, `TZ=UTC`, `LC_ALL=C`, thread caps)
  - no-network guard (unless env permits)
  - file sandbox (read/write allowlists)
  - optional resource limits (CPU/memory)
  - a `PluginContext` that provides `ctx.logger(msg)` which writes to `run_dir/logs/<plugin_id>.log`.

### 1.2 Pipeline staging
The pipeline runs (in order): `ingest_tabular` → `transform_normalize_mixed` (required) → optional `transform_template` → profile/planner (optional) → analysis → report.

### 1.3 Stat-plugin architecture (preferred)
Many analysis plugins are thin wrappers that call:
`statistic_harness.core.stat_plugins.registry.run_plugin(plugin_id, ctx)`

That registry:
- loads the dataset via `ctx.dataset_loader()`
- applies deterministic sampling if configured
- runs a handler from `HANDLERS[plugin_id]`
- attaches defaults, references, and debug metadata.

**For this Next30 pack: implement handlers in one module** and keep wrappers trivial.

---

## 2) Goal: 30 new plugin IDs

Implement these plugin IDs exactly (all new; do not rename existing plugins):

### A) Time-series decomposition & forecasting residuals
1. `analysis_bsts_intervention_counterfactual_v1`
2. `analysis_stl_seasonal_decompose_v1`
3. `analysis_seasonal_holt_winters_forecast_residuals_v1`
4. `analysis_lomb_scargle_periodogram_v1`
5. `analysis_garch_volatility_shift_v1`

### B) Changepoint variants
6. `analysis_bayesian_online_changepoint_studentt_v1`
7. `analysis_wild_binary_segmentation_v1`
8. `analysis_fused_lasso_trend_filtering_v1`
9. `analysis_cusum_on_model_residuals_v1`
10. `analysis_change_score_consensus_v1`

### C) Robust statistics & distribution diagnostics
11. `analysis_benfords_law_anomaly_v1`
12. `analysis_geometric_median_multivariate_location_v1`
13. `analysis_random_matrix_marchenko_pastur_denoise_v1`
14. `analysis_outlier_influence_cooks_distance_v1`
15. `analysis_heavy_tail_index_hill_v1`

### D) Nonlinear association & tabular modeling
16. `analysis_distance_correlation_screen_v1`
17. `analysis_gam_spline_regression_v1`
18. `analysis_quantile_loss_boosting_v1`
19. `analysis_quantile_regression_forest_v1`
20. `analysis_sparse_pca_interpretable_components_v1`

### E) Latent factor separation across columns
21. `analysis_ica_source_separation_v1`
22. `analysis_cca_crossblock_association_v1`
23. `analysis_factor_rotation_varimax_v1`
24. `analysis_subspace_tracking_oja_v1`
25. `analysis_multicollinearity_vif_screen_v1`

### F) Discrete/count/categorical & complexity measures
26. `analysis_zero_inflated_count_model_v1`
27. `analysis_negative_binomial_overdispersion_v1`
28. `analysis_dirichlet_multinomial_categorical_overdispersion_v1`
29. `analysis_fisher_exact_enrichment_v1`
30. `analysis_recurrence_quantification_rqa_v1`

---

## 3) Implementation plan (architecture)

### 3.1 Add a handler module
Create:
- `src/statistic_harness/core/stat_plugins/next30_addon.py`

It must define:
- `HANDLERS: dict[str, Callable[..., PluginResult]]`

Handler signature must match other stat-plugin packs:
```python
def handler(
    plugin_id: str,
    ctx,
    df: "pd.DataFrame",
    config: dict,
    inferred: dict,
    timer: "BudgetTimer",
    sample_meta: dict,
) -> "PluginResult":
    ...
```

### 3.2 Wire into the registry
Edit:
- `src/statistic_harness/core/stat_plugins/registry.py`

Add:
```python
from statistic_harness.core.stat_plugins.next30_addon import (
    HANDLERS as NEXT30_HANDLERS,
)
...
HANDLERS.update(NEXT30_HANDLERS)
```

### 3.3 Add plugin wrappers (30 directories)
For each plugin ID:
- `plugins/<plugin_id>/plugin.yaml`
- `plugins/<plugin_id>/plugin.py`
- `plugins/<plugin_id>/config.schema.json`
- `plugins/<plugin_id>/output.schema.json`

Use templates in §4.

### 3.4 Tests
Add:
- `tests/test_next30_plugins_smoke.py`

It must:
- generate small synthetic datasets on the fly
- run each plugin via `Pipeline.run(...)`
- assert `status in {"ok","skipped"}` and log file exists and non-empty
- determinism check for `status=="ok"`: identical plugin-result payload across two runs with same seed.

---

## 4) File templates (copy/paste)

### 4.1 `plugins/<plugin_id>/plugin.py`
```python
from __future__ import annotations

from statistic_harness.core.stat_plugins.registry import run_plugin


class Plugin:
    def run(self, ctx):
        return run_plugin("<plugin_id>", ctx)
```

### 4.2 `plugins/<plugin_id>/plugin.yaml`
Use this for every plugin (only change `id` and `name`):

```yaml
id: <plugin_id>
name: "<plugin_id>"
version: "0.1.0"
type: analysis
entrypoint: "plugin.py:Plugin"
depends_on: []
capabilities:
  - analysis
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
      redact_patterns: ["email", "ip", "uuid", "credit_card", "phone"]
      max_exemplars: 3
      allow_exemplar_snippets: false
    verbosity: "normal"
    plugin:
      max_points_for_quadratic: 2000
sandbox:
  no_network: true
  fs_allowlist:
    - run_dir
```

### 4.3 `plugins/<plugin_id>/config.schema.json`
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "stat_plugin_config",
  "type": "object",
  "additionalProperties": true,
  "properties": {
    "seed": {"type": "integer", "default": 1337},
    "time_budget_ms": {"type": "integer", "default": 25000},
    "max_rows": {"type": ["integer", "null"], "default": 200000},
    "max_cols": {"type": ["integer", "null"], "default": 80},
    "max_pairs": {"type": ["integer", "null"], "default": 2000},
    "max_windows": {"type": ["integer", "null"], "default": 64},
    "max_findings": {"type": ["integer", "null"], "default": 30},
    "allow_row_sampling": {"type": "boolean", "default": false},
    "privacy": {
      "type": "object",
      "additionalProperties": true,
      "properties": {
        "enable_redaction": {"type": "boolean", "default": true},
        "redact_patterns": {
          "type": "array",
          "items": {"type": "string"},
          "default": ["email", "ip", "uuid", "credit_card", "phone"]
        },
        "max_exemplars": {"type": "integer", "default": 3},
        "allow_exemplar_snippets": {"type": "boolean", "default": false}
      },
      "default": {}
    },
    "verbosity": {"type": "string", "enum": ["low", "normal", "high"], "default": "normal"},
    "plugin": {
      "type": "object",
      "additionalProperties": true,
      "default": {}
    }
  }
}
```

### 4.4 `plugins/<plugin_id>/output.schema.json`
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "plugin_output",
  "type": "object",
  "additionalProperties": true,
  "properties": {
    "status": {"type": "string"},
    "summary": {"type": "string"},
    "metrics": {"type": "object"},
    "findings": {"type": "array"},
    "artifacts": {"type": "array"},
    "budget": {"type": ["object", "null"]},
    "error": {"type": ["object", "null"]},
    "references": {"type": "array"},
    "debug": {"type": "object"}
  },
  "required": ["status", "summary", "metrics", "findings", "artifacts", "error"]
}
```

---

## 5) Cross-cutting requirements for every handler

### 5.1 Determinism
- Use `seed = int(config.get("seed", ctx.run_seed))`.
- Any sampling must be deterministic (use existing `deterministic_sample` helper if imported; else stable hashing).
- Sort all “top-k” selections deterministically.

### 5.2 Logging (mandatory)
Every handler must call `ctx.logger(...)`:
- at start: `START plugin_id=... rows=... cols=... seed=...`
- on gating/skips: `SKIP reason=...`
- at end: `END runtime_ms=... findings=...`

### 5.3 Budgets and caps
- Use `BudgetTimer` to enforce `time_budget_ms`.
- For any quadratic computation, cap rows to:
  - `max_points_for_quadratic = int(config.get("plugin", {}).get("max_points_for_quadratic", 2000))`
  - if data larger: **skip** (do not silently sample unless `allow_row_sampling=True`).

### 5.4 Findings format
Use stable ids:
- `id = stable_id(f"{plugin_id}:{key}")` if you import `stable_id` from stat_plugins helpers
- else fallback: `hashlib.sha256(...).hexdigest()[:16]`

Each finding dict must include:
- `id`, `severity`, `confidence`, `title`, `what`, `why`
- `evidence` (dict)
- `where` (dict)
- `recommendation` (string)
- `measurement_type` (string; default `"measured"`)

### 5.5 Artifact writing
Write JSON artifacts under `ctx.artifacts_dir(plugin_id)`:
- use deterministic JSON dumps (sort keys)
- return `PluginArtifact(path=..., type="json", description=...)`

---

## 6) Detailed per-plugin minimum algorithms (implement exactly these baselines)

> All algorithms must be implemented without requiring new third-party dependencies beyond what the repo already declares.
> Where `sklearn` is available, you may use it. SciPy is optional in this repo; treat SciPy as optional and provide fallback logic.

### A1) analysis_bsts_intervention_counterfactual_v1
**Requires:** time column + numeric series  
**Goal:** estimate counterfactual post-intervention mean and quantify effect size.

Algorithm:
1. Infer `time_column`; if missing/unparseable → `skipped`.
2. Pick numeric `value` column: highest variance among inferred numeric columns (cap to `max_cols`).
3. Bucket by day (or if median delta < 1h, bucket by hour).
4. Determine intervention index `t0`:
   - compute rolling mean with window `W = max(7, n//20)`; find index with max absolute diff of rolling mean derivative.
5. Fit pre-intervention baseline with deterministic local-level recursion:
   - `level_0 = median(pre)`; `level_t = level_{t-1} + alpha*(y_t - level_{t-1})`, alpha = 0.2
6. Forecast post by carrying the last level forward (or applying same recursion to predicted only).
7. Effect size:
   - `delta = mean(post_actual) - mean(post_pred)`
   - `scale = MAD(pre_actual)` (fallback to std)
   - `effect = delta / max(scale, 1e-6)`
8. Finding severity:
   - warn if |effect| >= 0.5, critical if >= 1.0.

Artifact `bsts_counterfactual.json` includes:
- bucket info
- t0
- pre/post means (actual/pred)
- effect size

### A2) analysis_stl_seasonal_decompose_v1
**Requires:** time + numeric  
Algorithm:
1. Bucket time series.
2. Trend: centered moving median with window `trend_window` (default 7).
3. Detrend: `d = y - trend`.
4. Seasonality: choose period `p` (default 7); seasonal[phase] = mean(d[phase]).
5. Residual: `r = y - trend - seasonal(phase)`.
6. Flag top residual spikes by robust z-score > `residual_z` (default 4.0).

Artifact `stl_components.json` includes summary stats + top spikes.

### A3) analysis_seasonal_holt_winters_forecast_residuals_v1
**Requires:** time + numeric  
Algorithm:
- Additive HW:
  - init level = y0
  - init trend = (y[p]-y0)/p if possible else 0
  - init season[0..p-1] from first period deviations
  - update with alpha/beta/gamma defaults (0.2/0.05/0.1)
- Compute one-step-ahead forecast errors; flag spikes with robust z > `residual_z`.

Artifact `holt_winters.json`.

### A4) analysis_lomb_scargle_periodogram_v1
**Requires:** time + numeric  
Algorithm:
1. Convert times to seconds since start; drop NaNs.
2. If time deltas are near-regular (CV < 0.1): fallback to FFT peak/median ratio on resampled series.
3. Else:
   - frequency grid size `F = plugin.max_freqs` (default 256)
   - frequencies in [1/T, F/(2*T)] where T = duration seconds
   - compute correlation score with sin/cos basis:
     - score(f) = (corr(y, sin(2πft))^2 + corr(y, cos(2πft))^2)
4. Return top 3 periods (1/f) and peak/median score ratio.

Artifact `lomb_scargle.json`.

### A5) analysis_garch_volatility_shift_v1
**Requires:** numeric  
Algorithm:
- returns = diff(y)
- rolling variance windows (window default 50)
- variance ratio = var(last_window)/var(first_window)
- warn if ratio > 2.0, critical if > 4.0

Artifact `volatility_shift.json`.

---

### B6) analysis_bayesian_online_changepoint_studentt_v1
**Requires:** numeric  
Goal: robust CP score (heavy tails) without SciPy.

Algorithm:
- Use sliding window size `W` (default 50).
- For each t in [W..n-W):
  - pre = y[t-W:t], post = y[t:t+W]
  - compute robust mean/scale for pre and post
  - score = |median(post)-median(pre)| / max(MAD(pre),1e-6)
- CP at max score; warn if score > 0.8, critical if > 1.5

Artifact `bocpd_studentt.json`.

### B7) analysis_wild_binary_segmentation_v1
**Requires:** numeric, n>=200  
Algorithm:
- Deterministically generate K intervals (default 128) from seed:
  - use RNG with seed; sample (l,r) with min length 2*min_seg.
- For each interval:
  - find best split k maximizing SSE reduction
- Aggregate candidate ks; cluster within tolerance 5; keep top `max_changepoints` (default 10)

Artifact `wbs_changepoints.json`.

### B8) analysis_fused_lasso_trend_filtering_v1
**Requires:** numeric  
Algorithm:
- Approximate fused-lasso on differences:
  - start with y
  - iterative (max 50):
    - d = diff(level)
    - d = soft_threshold(d, lam)
    - reconstruct level by cumulative sum with fixed start
- breakpoints = indices where |d| > eps

Artifact `trend_filtering.json`.

### B9) analysis_cusum_on_model_residuals_v1
**Requires:** numeric  
Algorithm:
- detrend by rolling median (window 25)
- residual r = y - trend
- compute robust center/scale on r
- CUSUM with k/h in sigma units (defaults: k=0.5, h=5.0)
- emit last alarm index, direction, magnitude

Artifact `cusum_residuals.json`.

### B10) analysis_change_score_consensus_v1
**Requires:** numeric  
Algorithm:
- Compute three CP candidates:
  - Student-t score (B6)
  - WBS (B7)
  - simple rolling-mean derivative max
- Merge within tolerance 10 indices; output consensus points with vote count.

Artifact `changepoint_consensus.json`.

---

### C11) analysis_benfords_law_anomaly_v1
**Requires:** numeric  
Algorithm:
- For each numeric column:
  - extract first non-zero digit of abs(x) as int 1..9
  - empirical distribution p̂(d)
  - benford p(d)=log10(1+1/d)
  - chi2 = Σ (n*(p̂-p)^2/p)
- report top columns by chi2.

Artifact `benford.json`.

### C12) analysis_geometric_median_multivariate_location_v1
**Requires:** >=2 numeric cols  
Algorithm:
- Build X (rows capped by `max_points_for_quadratic` unless allow sampling)
- Weiszfeld iterations (max 50):
  - w_i = 1/max(||x_i - m||, eps)
  - m = Σ w_i x_i / Σ w_i
- distances = ||x_i - m||
- outliers = top-k distances.

Artifact `geomed_outliers.json`.

### C13) analysis_random_matrix_marchenko_pastur_denoise_v1
**Requires:** >=2 numeric cols  
Algorithm:
- X standardized, n rows, p cols (cap p<=40, n>=p+5)
- corr = (XᵀX)/n
- eigenvalues λ
- q=p/n
- λ_max = (1+sqrt(q))^2
- effective_dim = count(λ > λ_max)
- produce denoised corr by zeroing eigenvalues below λ_max.

Artifact `mp_denoise.json`.

### C14) analysis_outlier_influence_cooks_distance_v1
**Requires:** numeric pair  
Algorithm:
- pick x and y as top-variance pair (or y=largest variance, x=next)
- OLS beta = cov(x,y)/var(x)
- yhat = a + b x
- residuals e
- leverage h_i = 1/n + (x_i-x̄)^2/Σ(x-x̄)^2
- MSE = Σ e^2/(n-2)
- Cook D_i = e_i^2/(p*MSE) * h_i/(1-h_i)^2 (p=2)
- report top points.

Artifact `cooks_distance.json`.

### C15) analysis_heavy_tail_index_hill_v1
**Requires:** numeric  
Algorithm:
- choose column with name hint (duration, wait, latency, elapsed) else top variance
- sort descending: x(1)>=...>=x(n)
- choose k = min(200, n//10) with k>=20
- hill = (1/k) Σ_{i=1..k} log(x(i)/x(k))
- tail_index_alpha = 1/hill
- warn if alpha < 2.5, critical if < 1.5

Artifact `hill_tail_index.json`.

---

### D16) analysis_distance_correlation_screen_v1
**Requires:** >=2 numeric cols; quadratic capped  
Algorithm:
- choose up to `p=10` cols by variance
- cap rows to `max_points_for_quadratic` unless allow sampling
- distance correlation:
  - compute pairwise distance matrices A,B
  - double center: Ā = A - rowmean - colmean + grandmean (same for B)
  - dCov² = mean(Ā*B̄)
  - dVarX = mean(Ā*Ā), dVarY similarly
  - dCor = dCov/sqrt(dVarX*dVarY)
- report top pairs by dCor.

Artifact `distance_correlation.json`.

### D17) analysis_gam_spline_regression_v1
**Requires:** numeric target + predictors  
Algorithm:
- Choose target y: top variance
- Choose predictors X: next top cols up to 6
- Fit sklearn `SplineTransformer` + `Ridge` with fixed random_state=seed
- Report:
  - R² on train (bounded)
  - top coefficients by L2 norm per original feature
  - partial response samples for each feature (10 points)

Artifact `gam_spline.json`.

### D18) analysis_quantile_loss_boosting_v1
**Requires:** numeric target + predictors  
Algorithm:
- Fit sklearn `GradientBoostingRegressor(loss="quantile", alpha=q)` for q in {0.5,0.9}
- report feature importances
- if time exists: compare predicted q90 early vs late; emit drift if |Δ|/scale>0.5

Artifact `quantile_boosting.json`.

### D19) analysis_quantile_regression_forest_v1
**Requires:** numeric target + predictors  
Algorithm:
- Fit sklearn RF with fixed random_state
- For each sample, collect per-tree predictions; estimate q90 as percentile
- Report:
  - feature importances
  - top q90 residual points.

Artifact `qrf.json`.

### D20) analysis_sparse_pca_interpretable_components_v1
**Requires:** numeric matrix  
Algorithm:
- Use sklearn `SparsePCA` if available; else PCA then threshold small loadings
- Output top 5 components with top 8 loading columns.

Artifact `sparse_pca.json`.

---

### E21) analysis_ica_source_separation_v1
**Requires:** numeric matrix  
Algorithm:
- Use sklearn `FastICA` if available; else PCA fallback
- Output:
  - component kurtosis
  - top loading columns per component

Artifact `ica_sources.json`.

### E22) analysis_cca_crossblock_association_v1
**Requires:** numeric matrix p>=4  
Algorithm:
- Split cols into two blocks A/B:
  - if name hints: block A contains cols with ("queue","wait","eligible"), block B contains ("duration","service","runtime"), else half/half
- Use sklearn `CCA(n_components=2)`; fallback to corr of first PCs
- Output canonical correlations and top contributing cols.

Artifact `cca.json`.

### E23) analysis_factor_rotation_varimax_v1
**Requires:** numeric matrix  
Algorithm:
- Compute loadings L via PCA components (k=3) as proxy if factor-analysis not already available
- Varimax rotation:
  - iterative update maximizing variance of squared loadings (max 50 iters)
- Output rotated loadings and sparsity metric.

Artifact `varimax.json`.

### E24) analysis_subspace_tracking_oja_v1
**Requires:** time + numeric matrix  
Algorithm:
- Online Oja update for top k=3 components:
  - w = w + η * x * (xᵀw); normalize
- Fit on first half and second half separately; compute principal angles via SVD of W1ᵀW2
- Warn if max angle > 20°, critical if > 35°

Artifact `oja_subspace.json`.

### E25) analysis_multicollinearity_vif_screen_v1
**Requires:** numeric predictors p>=3  
Algorithm:
- Choose up to p=8 numeric columns by variance
- For each Xi:
  - regress Xi on others using ridge (closed form) → R²
  - VIF = 1/(1-R²)
- Warn if VIF>5, critical if >10.

Artifact `vif.json`.

---

### F26) analysis_zero_inflated_count_model_v1
**Requires:** integer-like count series  
Algorithm (EM, bounded):
- Identify count column: numeric where >90% values are integers and >=0
- Model:
  - P(X=0)=pi + (1-pi)*Poisson(0;λ)
  - P(X=k>0)=(1-pi)*Poisson(k;λ)
- EM iterations (max 50):
  - E-step: compute posterior z_i for zeros
  - M-step:
    - pi = mean(z_i)
    - λ = mean(x_i) / max(1-pi, eps)
- Output pi, λ, and whether pi differs early vs late (if time).

Artifact `zip_em.json`.

### F27) analysis_negative_binomial_overdispersion_v1
**Requires:** count column  
Algorithm:
- mean m, var v
- overdispersion ratio = v / max(m, eps)
- NB method-of-moments:
  - r = m^2 / max(v-m, eps)
- Warn if ratio>2, critical if >5

Artifact `nb_overdispersion.json`.

### F28) analysis_dirichlet_multinomial_categorical_overdispersion_v1
**Requires:** categorical + group or time  
Algorithm:
- Select categorical col with moderate cardinality <=50
- Partition data into windows (early/late) or groups (top 3 categories of group col)
- For each partition: compute category proportions vector p
- Dispersion score:
  - mean squared deviation of p across partitions (L2 variance)
- Warn if dispersion > threshold (default 0.01)

Artifact `dirichlet_multinomial.json`.

### F29) analysis_fisher_exact_enrichment_v1
**Requires:** categorical + a case/control split  
Algorithm:
- Define case/control:
  - if time exists: late = case, early = control
  - else: most frequent group value = case, rest = control (if group column exists)
- For each category value (cap 100):
  - 2x2 table counts
  - Fisher exact p-value:
    - implement in log-space with log-factorials and hypergeometric tail
- Apply BH-FDR using existing helper (or implement minimal BH).
- Emit top enrichments with q<=0.1.

Artifact `fisher_enrichment.json`.

### F30) analysis_recurrence_quantification_rqa_v1
**Requires:** numeric series; quadratic capped  
Algorithm:
- Cap n to `max_points_for_quadratic` unless allow sampling.
- Embed m=2, delay=1: vectors v_t = [x_t, x_{t+1}]
- Recurrence matrix R(i,j)=1 if ||v_i-v_j|| <= eps
  - eps = 0.1 * median pairwise distance (bounded)
- RQA metrics:
  - recurrence rate = mean(R)
  - determinism = fraction of recurrence points on diagonal lines length>=2
  - laminarity = fraction on vertical lines length>=2
- Compare metrics early vs late; warn if change > 0.1

Artifact `rqa.json`.

---

## 7) Code skeleton for `next30_addon.py` (include and fill in)

Create the file and implement:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import math
import json
import hashlib

import numpy as np
import pandas as pd

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json

# Import these if available in your repo; otherwise implement minimal versions.
from statistic_harness.core.stat_plugins import (
    BudgetTimer,
    infer_columns,
    robust_center_scale,
    robust_zscores,
    stable_id,
)

def _artifact_json(ctx, plugin_id: str, filename: str, payload: Any, description: str) -> PluginArtifact:
    artifacts_dir = ctx.artifacts_dir(plugin_id)
    path = artifacts_dir / filename
    # Deterministic JSON
    path.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return PluginArtifact(path=str(path.relative_to(ctx.run_dir)), type="json", description=description)

def _basic_metrics(df: pd.DataFrame, sample_meta: dict[str, Any]) -> dict[str, Any]:
    m = {
        "rows_seen": int(sample_meta.get("rows_total", len(df))),
        "rows_used": int(sample_meta.get("rows_used", len(df))),
        "cols_used": int(len(df.columns)),
    }
    m.update(sample_meta or {})
    return m

def _make_finding(plugin_id: str, key: str, title: str, what: str, why: str, evidence: dict[str, Any], *,
                  where: dict[str, Any] | None = None, recommendation: str = "", severity: str = "info",
                  confidence: float = 0.5, measurement_type: str = "measured") -> dict[str, Any]:
    try:
        fid = stable_id(f"{plugin_id}:{key}")
    except Exception:
        fid = hashlib.sha256(f"{plugin_id}:{key}".encode("utf-8")).hexdigest()[:16]
    return {
        "id": fid,
        "severity": severity,
        "confidence": float(max(0.0, min(1.0, confidence))),
        "title": title,
        "what": what,
        "why": why,
        "evidence": evidence,
        "where": where or {},
        "recommendation": recommendation,
        "measurement_type": measurement_type,
    }

def _skip(plugin_id: str, reason: str, df: pd.DataFrame, sample_meta: dict[str, Any], *, debug: dict[str, Any] | None = None) -> PluginResult:
    dbg = dict(debug or {})
    dbg.setdefault("gating_reason", reason)
    return PluginResult("skipped", reason, _basic_metrics(df, sample_meta), [], [], None, references=None, debug=dbg)

def _log_start(ctx, plugin_id: str, df: pd.DataFrame, config: dict[str, Any], inferred: dict[str, Any]) -> None:
    ctx.logger(f"START plugin={plugin_id} rows={len(df)} cols={len(df.columns)} seed={config.get('seed')} time_budget_ms={config.get('time_budget_ms')}")
    ctx.logger(f"INFER time={inferred.get('time_column')} numeric={len(inferred.get('numeric_columns') or [])} cat={len(inferred.get('categorical_columns') or [])}")

def _log_end(ctx, plugin_id: str, findings: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    ctx.logger(f"END plugin={plugin_id} findings={len(findings)} metrics_keys={sorted(list(metrics.keys()))[:10]}")

# Implement 30 handler functions here; each returns PluginResult
def _handler_example(...): ...
HANDLERS: dict[str, Callable[..., PluginResult]] = {
    # "analysis_bsts_intervention_counterfactual_v1": _bsts_handler,
}
```

Then implement each handler per §6 and populate `HANDLERS`.

---

## 8) Test file skeleton (must implement)

Create `tests/test_next30_plugins_smoke.py`:

```python
import json
from pathlib import Path

import pandas as pd

from statistic_harness.core.pipeline import Pipeline


NEXT30 = [
  "analysis_bsts_intervention_counterfactual_v1",
  "analysis_stl_seasonal_decompose_v1",
  "analysis_seasonal_holt_winters_forecast_residuals_v1",
  "analysis_lomb_scargle_periodogram_v1",
  "analysis_garch_volatility_shift_v1",
  "analysis_bayesian_online_changepoint_studentt_v1",
  "analysis_wild_binary_segmentation_v1",
  "analysis_fused_lasso_trend_filtering_v1",
  "analysis_cusum_on_model_residuals_v1",
  "analysis_change_score_consensus_v1",
  "analysis_benfords_law_anomaly_v1",
  "analysis_geometric_median_multivariate_location_v1",
  "analysis_random_matrix_marchenko_pastur_denoise_v1",
  "analysis_outlier_influence_cooks_distance_v1",
  "analysis_heavy_tail_index_hill_v1",
  "analysis_distance_correlation_screen_v1",
  "analysis_gam_spline_regression_v1",
  "analysis_quantile_loss_boosting_v1",
  "analysis_quantile_regression_forest_v1",
  "analysis_sparse_pca_interpretable_components_v1",
  "analysis_ica_source_separation_v1",
  "analysis_cca_crossblock_association_v1",
  "analysis_factor_rotation_varimax_v1",
  "analysis_subspace_tracking_oja_v1",
  "analysis_multicollinearity_vif_screen_v1",
  "analysis_zero_inflated_count_model_v1",
  "analysis_negative_binomial_overdispersion_v1",
  "analysis_dirichlet_multinomial_categorical_overdispersion_v1",
  "analysis_fisher_exact_enrichment_v1",
  "analysis_recurrence_quantification_rqa_v1",
]

def _make_dataset(path: Path, *, shift: bool) -> None:
    n = 800
    ts = pd.date_range("2026-01-01", periods=n, freq="H")
    value = pd.Series(range(n), dtype=float) * 0.01
    if shift:
        value.iloc[n//2:] += 5.0
    noise = pd.Series(((pd.Series(range(n))*1103515245 + 12345) % 997) / 997.0)
    value = value + noise
    count = (noise * 3).round().astype(int)
    cat = ["A" if i % 3 else "B" for i in range(n)]
    df = pd.DataFrame({"ts": ts.astype(str), "value": value, "count": count, "cat": cat})
    df.to_csv(path, index=False)

def _run_once(tmp_path: Path, csv_path: Path, plugin_id: str, seed: int) -> tuple[Path, dict]:
    appdata = tmp_path / f"appdata_{plugin_id}_{seed}"
    pipeline = Pipeline(appdata, Path("plugins"))
    run_id = pipeline.run(csv_path, [plugin_id], {}, seed)
    run_dir = appdata / "runs" / run_id
    results = pipeline.storage.fetch_plugin_results(run_id)
    row = next(r for r in results if r["plugin_id"] == plugin_id)
    return run_dir, row

def test_next30_plugins_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("STAT_HARNESS_RETENTION_ENABLED", "0")
    monkeypatch.setenv("STAT_HARNESS_MAX_WORKERS_ANALYSIS", "1")

    csv_a = tmp_path / "ts_small.csv"
    csv_b = tmp_path / "ts_shift.csv"
    _make_dataset(csv_a, shift=False)
    _make_dataset(csv_b, shift=True)

    for plugin_id in NEXT30:
        for csv_path in (csv_a, csv_b):
            run_dir, row = _run_once(tmp_path, csv_path, plugin_id, 123)
            assert row["status"] in {"ok", "skipped"}
            log_path = run_dir / "logs" / f"{plugin_id}.log"
            assert log_path.exists()
            assert log_path.read_text(encoding="utf-8").strip()

def test_next30_ok_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setenv("STAT_HARNESS_RETENTION_ENABLED", "0")
    monkeypatch.setenv("STAT_HARNESS_MAX_WORKERS_ANALYSIS", "1")

    csv_path = tmp_path / "ts_shift.csv"
    _make_dataset(csv_path, shift=True)

    for plugin_id in NEXT30:
        run_dir1, row1 = _run_once(tmp_path, csv_path, plugin_id, 777)
        run_dir2, row2 = _run_once(tmp_path, csv_path, plugin_id, 777)
        if row1["status"] != "ok" or row2["status"] != "ok":
            continue
        # Compare stored JSON fields (exact match)
        for key in ("metrics", "findings", "summary", "status"):
            assert row1[key] == row2[key]
```

---

## 9) Codex execution and logging requirements

### 9.1 Codex must log its own actions
Create/append: `.codex/STATE.md` with:
- files changed
- brief rationale
- commands run and outputs
- if dependency install unavailable: record that tests were skipped and rely on CI

### 9.2 Commands
```bash
poetry install --with dev
pytest -q
```

If installs are unavailable in the sandbox:
- skip execution and rely on CI
- do not claim tests failing unless they actually ran with deps installed

---

## 10) Acceptance criteria
- All 30 plugin directories exist and discover properly.
- `next30_addon.py` implements all 30 handlers and is registered in `registry.py`.
- Each plugin writes non-empty `run_dir/logs/<plugin_id>.log`.
- No plugin returns `error` on the test datasets.
- Deterministic outputs for `status=="ok"` across identical runs.
- `pytest` passes.

END OF WORK ORDER
