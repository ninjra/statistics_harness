# Topographical + Topological Statistical Plugin Add‑On Pack (Codex CLI Spec)

**THREAD:** stat-plugins  
**CHAT_ID:** topo-tda-addon-20260205  
**TS (America/Denver):** 2026-02-05T15:43:27-07:00

## Scope
This document specifies **new statistical plugins** for a log-analysis harness that runs against **unknown tabular datasets** (process logs of unknown source/content) and must still produce useful insights. It focuses on:

- **Topological Data Analysis (TDA)**: persistent homology, persistence landscapes, mapper graphs.
- **Topographic similarity statistics** from EEG/MEG “sensor topographies” (angle, projection, angle-dynamics, TANOVA-style permutation tests) generalized to arbitrary multivariate “maps”.
- **Surface/terrain/topographic complexity metrics** (multiscale wavelet curvature, fractal dimension, rugosity, TPI) inspired by *pyTopoComplexity*.
- **DEM-style hydrology/geomorphology operators** inspired by *TopoToolbox* (flow direction/accumulation/watersheds) generalized to synthetic “surfaces” derived from logs.
- **Classic statistical methods suite** (t-test, chi-square, ANOVA, regression, time series, survival, factor analysis, clustering, PCA) for fully automatic “scan & report”.

### Primary sources (visited)
- Karniski et al. (1994) permutation-based distribution-free test for comparing “topographic maps” (Springer preview): https://link.springer.com/article/10.1007/BF01187710
- TopoToolbox (Schwanghart & Kuhn, 2010) overview & function list (ScienceDirect preview): https://www.sciencedirect.com/science/article/abs/pii/S1364815209003053
- TopoToolbox 2 performance claims (Schwanghart & Scherler, 2014, ESurf): https://esurf.copernicus.org/articles/2/1/2014/esurf-2-1-2014.html
- pyTopoComplexity paper (Lai et al., 2025, ESurf): https://esurf.copernicus.org/articles/13/417/2025/
- pyTopoComplexity repository (AGPL‑3.0; module descriptions): https://github.com/GeoLarryLai/pyTopoComplexity
- SSO/terrain-fabric algorithm description (Guth, 1999 HTML): https://www.geog.leeds.ac.uk/groups/geocomp/1999/096/gc_096.htm
- TopoToolbox for EEG/MEG topographies (Tian et al., 2011; full text via Semantic Scholar PDF): https://pdfs.semanticscholar.org/6ba7/d89d6c19125fbba1a3a4bf8b845a819d112e.pdf
- Dovetail “10 statistical analysis methods” page (note: numbering skips from 4 → 6 in the page HTML): https://dovetail.com/research/key-statistical-analysis-methods-explained/

### Important note about PMC link
The provided PMC URL (https://pmc.ncbi.nlm.nih.gov/articles/PMC3090718/) returned HTTP 403 in this environment. The design below uses alternative accessible sources (TopoToolbox paper & figure, Springer preview, etc.) to implement the same *angle/projection similarity* concepts.

## Design goals (optimize for the 4 pillars)
### Performant
- Default to **O(N)** or **O(N log N)** scans.
- Any **O(N^2)** method must:
  - cap N via `max_points`
  - and/or use approximate neighbors.
- All plugins must implement `resource_budget` (time, memory) and self-throttle.

### Accurate
- Prefer **distribution-free** inference when assumptions are unknown (permutation / bootstrap / conformal).
- Use multiple-comparison control (BH/FDR) whenever scanning many column pairs/groups.

### Secure
- No network calls, no subprocess calls, no shelling out.
- Redact potential PII in outputs:
  - do not print raw IDs/emails/full usernames
  - do not output free-text columns verbatim
  - aggregate to k-anonymity threshold (`k_min`) for any grouping output.

### Citable
- Each plugin includes a `citations` field with:
  - method citation(s)
  - implementation reference(s) if relevant (but avoid copying AGPL code into proprietary codebases; see licensing notes).

---

## Harness contract assumed
(Adjust names/types to your actual harness; this is intentionally explicit for Codex.)

### Minimal plugin API
```python
PLUGIN_ID: str

def applicable(df: pd.DataFrame, ctx: dict) -> tuple[bool, str]:
    """Return (True, "") if plugin can run, else (False, reason)."""

def run(df: pd.DataFrame, cfg: dict, ctx: dict) -> dict:
    """Return a JSON-serializable PluginResult (see schema below)."""
```

### Standard `PluginResult` schema
```json
{
  "plugin_id": "analysis_*",
  "status": "ok|skipped|degraded|error",
  "summary": "1-3 sentence human summary",
  "findings": [
    {
      "title": "string",
      "severity": "info|warn|high",
      "evidence": { "metrics": {}, "tables": [], "plots": [] },
      "recommendations": ["actionable text..."],
      "limitations": ["assumption/coverage notes..."]
    }
  ],
  "artifacts": [
    {
      "type": "json|csv|png|md",
      "path": "relative/path",
      "description": "string"
    }
  ],
  "metrics": {},
  "diagnostics": {
    "chosen_columns": [],
    "row_count": 0,
    "col_count": 0,
    "runtime_ms": 0,
    "random_seed": 0,
    "gating_reason": ""
  },
  "citations": [
    {
      "label": "Karniski1994",
      "url": "https://link.springer.com/article/10.1007/BF01187710"
    }
  ]
}
```

### Shared utilities each plugin may implement internally (or import from a safe local utils module)
- `profile_dataframe(df)` → types, missingness, numeric/cat/time candidates, uniqueness, monotonicity
- `choose_time_column(df)` → best timestamp col or None
- `choose_duration_columns(df)` → start/end pairs, delta columns
- `choose_group_columns(df)` → categorical columns with `2 <= nunique <= max_cardinality`
- `safe_label(value)` → redaction/hashing/truncation rules
- `stable_hash_seed(df, plugin_id)` → deterministic seed based on schema+head sample (no PII leakage)
- `bh_fdr(p_values)` for multiple comparisons

---

# Plugin Family A: Topological Data Analysis (TDA)

## A1. `analysis_tda_persistent_homology`
### What it does
Compute **persistent homology** summaries (H0/H1 at minimum) on a point cloud built from numeric columns; report:
- “shape” complexity (total persistence)
- connected components merging scale (H0)
- loopiness/holes proxy (H1)
- **topological drift over time** via sliding windows (if a time column exists)

### Data adaptation (unknown logs)
1. Build point cloud `X`:
   - numeric columns only
   - robust scale (median/IQR)
   - optionally PCA to `pca_dims` (default 3–10) for stability
2. If time column `t` exists:
   - sort by t
   - compute persistence summaries per window (size `window_rows`, step `step_rows`)
   - detect changepoints in topological metrics (CUSUM on total persistence)

### Dependencies
- Optional: `ripser` or `gudhi`.
- Baseline fallback if missing:
  - compute MST edge-length distribution + kNN graph cycle counts as a weak proxy; mark `status="degraded"`.

### Outputs
- Topological signature table:
  - window_id, t_start, t_end
  - H0_total_persistence, H1_total_persistence
  - max_persistence_H0/H1, betti0@eps_grid, betti1@eps_grid
- Candidate “regime shift windows” when metrics exceed robust z-threshold.

### Performance controls
- `max_points` (default 5,000)
- `max_windows` (default 200)
- `max_dim` (default 1)
- `eps_grid_size` (default 32)

### Tests
- **Determinism**: same df + seed ⇒ identical metrics.
- **Correctness**:
  - circle dataset ⇒ strong H1 persistence peak
  - two clusters ⇒ H0 merges late; H1 low
- **Time drift**: synthetic dataset with topological change at midpoint ⇒ changepoint flagged.
- **Budget**: runtime < configured threshold at max_points.
- **Security**: findings do not include raw row values.

## A2. `analysis_tda_persistence_landscapes`
### What it does
Convert persistence diagrams to **persistence landscapes** (vector summaries) to enable:
- statistical comparisons between groups/time windows
- distance-based anomaly detection in “shape space”

### Data adaptation
Reuse diagrams from A1 if available in ctx cache; else recompute quickly.

### Outputs
- landscape vectors per window
- pairwise distance matrix (capped)
- top anomalous windows (largest distance to median landscape)

### Tests
- landscape distance symmetry and zero diagonal
- known anomaly window is top-ranked

## A3. `analysis_tda_mapper_graph`
### What it does
Run the **Mapper algorithm** to create a graph summarizing high-dimensional structure; report:
- number of connected components/branches
- small isolated nodes (outlier regimes)
- feature enrichments per node (which columns differ)

### Data adaptation
- Choose filter functions:
  - PC1, density estimate, and/or time
- Cover with `n_intervals`, `overlap`
- Cluster within each interval (k-means baseline; optional HDBSCAN)

### Outputs
- Graph in JSON (nodes, edges, node membership)
- Node summary table (size, centroid stats)
- “Interesting branches” = sequences of nodes with monotone drift in selected metric.

### Tests
- synthetic two-moons ⇒ mapper yields branched graph
- stable node count under fixed seed
- no node outputs raw IDs (redaction enforced)

## A4. `analysis_tda_betti_curve_changepoint`
### What it does
Compute **Betti curves** (Betti numbers as a function of ε) for windows and run changepoint detection in curve space.

### Why separate from A1
A1 focuses on scalar summaries; this plugin focuses on **functional summaries** (curves), improving sensitivity to subtle structural changes.

### Tests
- curve-based changepoint catches change that scalar summaries miss (synthetic).

---

# Plugin Family B: Topographic Similarity + Permutation Map Tests (generalized)

This family generalizes EEG/MEG “topographies” to **any multivariate map**:
- A “map” = vector of feature values across a fixed feature set (like sensors).
- A “condition” = grouping label (process/module/user/time bucket/etc).

Angle/projection methods are described in the TopoToolbox paper (Tian et al., 2011), including angle similarity and projection onto a template pattern for magnitude comparison.

## B1. `analysis_topographic_similarity_angle_projection`
### What it does
Automatically build “maps” for candidate conditions and compute:
- **Angle similarity** (cosine similarity) between condition maps
- **Projection magnitude** of each condition onto a chosen template map

### Evidence basis
TopoToolbox describes angle between condition vectors as similarity, and projection onto a template as “how much” template produced (Figure 1 caption).

### Data adaptation (unknown logs)
1. Choose candidate grouping columns `G`:
   - categorical, `2 <= nunique <= max_cardinality`
2. Choose feature set `F`:
   - numeric columns
   - plus optionally one-hot encodings of low-cardinality categoricals
3. For each g in top-K group columns:
   - build map vector per group value = robust mean of F
4. Template selection:
   - default = global mean map
   - alt = best-separated group map (largest norm)

### Outputs
- For each g:
  - top pairwise dissimilarities (smallest cosine)
  - groups with highest/lowest projection onto template
- “Pattern shift” warnings when:
  - group maps are nearly orthogonal to global baseline
  - projection magnitudes differ strongly but angles remain similar (magnitude-only effect)

### Tests
- synthetic: two groups differ only by scale ⇒ angle similar, projection differs (expected)
- synthetic: two groups differ by feature permutation ⇒ angle differs (expected)
- redaction test on group labels

## B2. `analysis_topographic_angle_dynamics`
### What it does
“Angle dynamics test”: detect **when** a response pattern emerges by comparing template map to time-sliced maps over time.

### Data adaptation
- Requires time column OR can create pseudo-time by row order.
- Build maps over sliding time windows.
- Compute angle(template, window_map) vs time.

### Outputs
- time intervals when similarity crosses threshold (emergence)
- optional confidence via bootstrap over rows within windows

### Tests
- synthetic: injected pattern emerges at t0 ⇒ detected near t0

## B3. `analysis_topographic_tanova_permutation`
### What it does
Run a **permutation test** for map differences (“TANOVA”-style) using cosine distance or RMS difference, suitable when covariance assumptions are unknown.

### Outputs
- p-values per candidate grouping column (FDR corrected)
- effect sizes (median cosine distance between groups vs permuted)

### Tests
- synthetic: no difference ⇒ p uniform
- synthetic: known difference ⇒ significant after FDR

## B4. `analysis_map_permutation_test_karniski`
### What it does
Implement a generalized **distribution-free permutation test** for differences between multivariate maps, motivated by Karniski et al. (1994), which emphasizes:
- exact, distribution-free testing
- no sphericity assumption
- works when features (“electrodes”) exceed sample size

Important: implement from first principles; do not copy paywalled text or third-party code.

### Data adaptation
- Select 2–K conditions from candidate groupings.
- Within each condition, compute subject/instance maps (per entity id if exists; else bootstrap pseudo-subjects).

### Outputs
- test statistic distribution, exact/Monte Carlo p-value (depending on permutation count)
- “localized change” diagnostics: per-feature contribution to statistic

### Tests
- high-dimension low-n synthetic scenario still computes and detects difference
- permutation p-value reproducible under fixed seed

---

# Plugin Family C: Surface / Terrain / Topographic Complexity (pyTopoComplexity-inspired)

pyTopoComplexity describes four methods for topographic complexity: 2D-CWT Mexican-hat wavelet curvature, fractal dimension estimation, rugosity index, and terrain position index.

Licensing note: pyTopoComplexity is AGPL‑3.0. Do not copy code from the repository into non‑AGPL projects. Implement algorithms independently.

## Shared component: SurfaceBuilder (required for all Family C plugins)
Most process logs are not DEMs. To make surface methods broadly useful, implement a reusable adapter:

### `build_surface(df, x_spec, y_spec, z_spec) -> Surface`
- If df has explicit coordinate columns:
  - `x_col` in {x, lon, longitude, easting, ...}
  - `y_col` in {y, lat, latitude, northing, ...}
  - `z_col` numeric (elevation/value)
  - bin to grid with `grid_size` or `n_bins`
- Else:
  - compute 2D embedding of numeric features (PCA, deterministic)
  - use embedding as (x,y)
  - choose `z` as:
    - density (counts per bin) OR
    - chosen metric column (top variance numeric) aggregated by bin
- Missing bins:
  - fill via deterministic IDW interpolation OR leave NaN with mask.

Surface is:
```python
Surface = dict(
  Z=np.ndarray,           # shape (H,W) float
  mask=np.ndarray,        # valid cells
  x_edges=np.ndarray,
  y_edges=np.ndarray,
  meta={"x_col":..., "y_col":..., "z_col":..., "built_from":"coords|embedding"}
)
```

### Tests for SurfaceBuilder
- deterministic embedding given fixed seed
- interpolation does not extrapolate beyond convex hull unless allowed
- mask coverage reported accurately

## C1. `analysis_surface_multiscale_wavelet_curvature`
### What it does
Compute multiscale curvature/roughness via **2D continuous wavelet transform** with Mexican-hat wavelet curvature.

### Implementation sketch
- For each scale s in `scales`:
  - convolve Z with Mexican-hat kernel at scale s (FFT-based)
  - curvature proxy = |wavelet_coeff|
- Summaries:
  - per-scale mean/median curvature
  - characteristic scale = argmax of curvature variance (morphological scale)

### Outputs
- per-scale metrics table
- optional heatmap artifact (png) with curvature at best scale

### Tests
- synthetic surface with known bump size ⇒ characteristic scale near bump size
- runtime bounded by grid size cap

## C2. `analysis_surface_fractal_dimension_variogram`
### What it does
Estimate local **fractal dimension** via a variogram-based method; record reliability metrics (R^2, SE).

### Implementation sketch
- For windows over surface:
  - sample profiles in 4 directions
  - compute variogram γ(h) over lags h
  - fit log(γ) ~ α log(h) (robust regression)
  - derive FD from slope α (document formula used; cite)
  - record R^2, SE

### Outputs
- FD distribution (median, IQR)
- maps of FD and R^2 (optional)

### Tests
- fractional Brownian surface (synthetic) yields expected FD range
- R^2 threshold gating

## C3. `analysis_surface_rugosity_index`
### What it does
Compute **Rugosity Index (RI)**: surface area / planar area in windows; include optional slope-corrected variant (Arc‑Chord ratio).

### Outputs
- RI summary + hotspots (top 1% windows)
- if built_from embedding: interpret hotspots as “high structural complexity regions” in data distribution

### Tests
- flat surface ⇒ RI approx 1
- rough surface ⇒ RI > 1

## C4. `analysis_surface_terrain_position_index`
### What it does
Compute **Terrain Position Index (TPI)**: z(cell) − mean(z(neighborhood)); positive=ridges, negative=valleys.

### Outputs
- TPI distribution + ridge/valley hotspot coordinates in surface space
- if embedding-based: map hotspots back to representative rows (only via redacted identifiers)

### Tests
- synthetic ridge surface ⇒ positive TPI on ridge line

## C5. `analysis_surface_fabric_sso_eigen`
### What it does
Implement **Statistical Slope Orientation (SSO) diagram** + eigenvector fabric metrics (Guth algorithm), including:
- compute slope & aspect
- represent surface normals on lower-hemisphere equal-area projection
- build 3x3 cross-product matrix of direction cosines
- eigenvalues S1>=S2>=S3 -> flatness ln(S1/S2), organization ln(S2/S3), strength c=ln(S1/S3), and dominant direction from eigenvector S3.

### Outputs
- flatness, organization, strength, dominant_direction_deg
- if explicit geo coords: interpret as terrain fabric; else interpret as drift direction in embedding space.

### Tests
- planar sloped surface ⇒ strong organization, stable direction
- isotropic noise surface ⇒ low organization

## C6. `analysis_surface_hydrology_flow_watershed`
### What it does
Implement a minimal Python analogue to TopoToolbox-style operators:
- fill sinks in Z
- compute D8 flow direction
- compute flow accumulation
- delineate basins/catchments (watersheds)
- compute curvature and slope (finite differences)

### Outputs
- number of basins; basin sizes
- top basins by mean z (e.g., delay hotspots)
- ridge lines coordinates

### Tests
- synthetic bowl surface ⇒ single basin
- two-basin surface ⇒ 2 basins detected

---

# Plugin Family D: Classic Statistical Methods (Auto-Scan Suite)

The Dovetail page lists common methods (t-test, chi-square, ANOVA, regression, time series, survival, factor analysis, cluster analysis, PCA), though the numbering skips from 4 to 6 in the page HTML.

These plugins must be **self-contained scanners** that:
- infer targets, groups, and features
- control multiple testing
- produce “top-N insights” with clear evidence

## D1. `analysis_ttests_auto`
- For each numeric metric and each binary group column:
  - Welch's t-test + effect size (Cohen's d)
  - BH-FDR correction across all tests
- Output top significant differences (k-anonymized).

## D2. `analysis_chi_square_association`
- For each pair of categorical columns (bounded by max_pairs):
  - chi-square test of independence
  - effect size: Cramer's V
  - BH-FDR correction
- Output strongest associations.

## D3. `analysis_anova_auto`
- For each numeric metric and each categorical group col with 3+ levels:
  - one-way ANOVA (or Kruskal-Wallis if non-normal; deterministic rule)
  - post-hoc pairwise (Tukey or Dunn) for top effects (capped)
- Output top effects.

## D4. `analysis_regression_auto`
- Auto-select target:
  - numeric with highest variance (unless cfg sets target)
- Fit:
  - robust linear regression (Huber) and/or Lasso
  - report standardized coefficients + stability via bootstrap
- Output top drivers.

## D5. `analysis_time_series_analysis_auto`
- Detect time column; regularize to interval if possible.
- Compute:
  - trend/seasonality via STL (if statsmodels available)
  - autocorrelation peaks
  - outlier bursts (robust z on residuals)
- Output periodicities, drifts, anomalies.

## D6. `analysis_survival_time_to_event`
- Detect start/end + event indicator (or infer censoring).
- Compute:
  - Kaplan-Meier curves by group (top 1-2 group cols)
  - log-rank test (permutation fallback)
- Output groups with significantly different “time to completion/failure”.

## D7. `analysis_factor_analysis_auto`
- Run FA (or PCA fallback) on numeric matrix.
- Output:
  - number of factors by explained variance threshold
  - top-loading features per factor
  - factor scores correlation with key outcomes (duration, revenue proxy)

## D8. `analysis_cluster_analysis_auto`
- Cluster rows on numeric features:
  - k-means baseline with k chosen by silhouette (bounded)
  - optional GMM if available
- Output cluster profiles and anomaly clusters (small clusters).

## D9. `analysis_pca_auto`
- PCA on numeric columns; report:
  - explained variance
  - strongest loading columns
  - projection extremes (outlier rows, redacted)

---

---

# Plugin Family E: Spatial + Bayesian Uncertainty Modeling (Point/Surface Displacement & Interpolation)

This family targets datasets that contain *spatial coordinates* (x/y[/z], lat/lon[/elev]) or any situation where we can construct a surface via SurfaceBuilder and want **uncertainty-aware** conclusions.

Evidence basis (examples):
- NHESS describes repeated geodetic observations across epochs and Bayesian parameter estimation for deformation monitoring networks (Tanir et al., 2008). https://nhess.copernicus.org/articles/8/335/2008/
- Monte Carlo simulation is widely used to propagate DEM vertical error into derivative uncertainty (e.g., slope/aspect/watersheds) by sampling error fields and re-running analyses (Oksanen & Sarjakoski, 2001). https://icaci.org/files/documents/ICC_proceedings/ICC2001/icc2001/file/f20006.pdf

## E1. `analysis_bayesian_point_displacement`
### What it does
Estimate **point displacement** between two epochs (pre/post) with uncertainty, using a simple Bayesian model:
- For each point id, observe coordinates at two (or more) epochs.
- Model measurement noise and infer posterior displacement vectors.
- Flag points/regions with posterior probability of displacement > threshold.

### Gating / data requirements
- Requires either:
  - explicit `epoch`/`time` column AND `point_id` column AND coordinate columns (x/y or x/y/z), OR
  - two datasets provided via ctx (baseline + follow-up) with a join key.

If no explicit points exist, degrade to **group-level displacement**:
- treat each group (process/module) centroid in embedding space as a “point” and compare epochs.

### Implementation sketch (conjugate, no heavy MCMC)
- For each point, let Δ = (x_post - x_pre, y_post - y_pre[, z_post - z_pre]).
- Assume Δ ~ Normal(μ, Σ) with NIW prior (μ, Σ).
- Closed-form posterior for μ and Σ gives:
  - posterior mean displacement
  - credible intervals
  - P(||Δ|| > ε) approximated from posterior samples (small N_samp, seeded)

### Outputs
- Top displaced points/groups (redacted ids)
- Posterior displacement summaries:
  - mean vector, ||mean||, 95% CI, probability above ε
- Spatial clustering of displacements (optional: DBSCAN on displacement vectors)

### Tests
- synthetic: injected displacement on subset ⇒ detected with high posterior probability
- synthetic: pure noise ⇒ low probabilities, controlled false positives
- determinism: seeded posterior sampling stable

## E2. `analysis_monte_carlo_surface_uncertainty`
### What it does
Quantify **uncertainty of derived surface metrics** (slope, curvature, watershed/basins, rugosity, TPI) by Monte Carlo perturbation:
1) Build baseline surface Z.
2) Define an error model (default: Gaussian noise with spatial autocorrelation length L).
3) Generate N realizations Z_i = Z + ε_i (seeded).
4) Recompute chosen surface metric(s) on each Z_i.
5) Report uncertainty bands and stability of “hotspots”.

### Why this matters for unknown datasets
Surface-based insights are vulnerable to sampling noise and binning artifacts. This plugin reports which conclusions are stable.

### Outputs
- For each metric:
  - per-cell mean and std (optional artifacts)
  - summary stability score: fraction of simulations where hotspot remains in top p%
- Confidence labels: high/medium/low stability

### Performance controls
- `mc_samples` default 64 (cap 256)
- `grid_max_hw` cap (e.g., 256×256)
- use FFT to generate correlated noise efficiently

### Tests
- synthetic: stable ridge remains stable under perturbations
- synthetic: random hotspots show low stability
- budget: respects `mc_samples` and grid cap

## E3. `analysis_surface_roughness_metrics`
### What it does
Compute **surface roughness** on 1D and 2D representations:
- 1D (time series): RMS roughness, autocorrelation length, PSD slope
- 2D (surface): RMS height, skewness/kurtosis, structure function slope, anisotropy of roughness

### Data adaptation
- If time column exists:
  - choose top numeric metrics and compute 1D roughness over time
- If surface exists:
  - compute 2D roughness on Z

### Outputs
- Ranked list of metrics with unusually high roughness (potential instability)
- If anisotropy detected, report dominant roughness direction (links to SSO fabric)

### Tests
- white noise series has PSD slope near 0
- smooth sinusoid shows low roughness vs noise
- deterministic outputs under fixed seed


# Cross-cutting tests (ALL new plugins)
1. **No-network test**: monkeypatch socket/requests; plugin must not attempt outbound calls.
2. **Determinism test**: fixed seed ⇒ identical result JSON (ignoring runtime_ms).
3. **Budget test**: with large synthetic df, plugin respects `max_points/max_cols/max_pairs`.
4. **Schema-agnostic test**: plugin returns `status="skipped"` with a clear gating reason when requirements aren’t met.
5. **Redaction test**: any string value matching email/uuid patterns must be masked in outputs.
6. **Numerical stability test**: NaNs/infs handled; no crashes; diagnostics record imputation/row drops.
7. **Citations presence test**: plugin returns at least 1 citation when `status in {ok,degraded}`.

---

# Licensing / compliance notes (must be enforced in code review)
- **pyTopoComplexity is AGPL-3.0** (see repo). If your harness/plugins are not AGPL-compatible, implement methods from scratch and do not copy code. Use only the idea/method and cite the paper/repo.
- Springer full text is subscription; implement based on general permutation test theory + their preview summary (do not copy paywalled text).
- TopoToolbox is MATLAB; implement your own Python analogue; cite papers.

---

## Implementation priority (recommended)
1) Family B (topographic similarity + permutation): high leverage for arbitrary logs  
2) Family D (auto-scan classical stats): broad coverage, low risk  
3) Family A (TDA): high insight, but needs careful performance gating  
4) Family C (surface complexity): highest value when dataset can be embedded into a meaningful surface; keep conservative defaults
