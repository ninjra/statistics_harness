# Claude Code Implementation Task: Kona 3D Visualization + Kohonen SOM + Per-Dataset Energy Weight Learning

## 0) Your role & non-negotiables

You are running inside the `statistics_harness` repository. Implement the changes below by **scanning the repo first** and **updating existing code in-place**. Do **not** introduce duplicate schemas, duplicate helper functions, or duplicate logic blocks.

### Hard constraints
- **No external references required**: everything you need is in this file + the repo you are scanning.
- **Deterministic outputs**: identical inputs + same seed ⇒ identical artifacts, findings, and recommendation ordering.
- **No duplicate info**: if the repo already has an equivalent schema/helper/logic, reuse it; only add what's missing.
- **Preserve existing behavior by default**: new capabilities must be **disabled by default** so existing runs are unchanged unless explicitly enabled via config.
- **Keep existing artifacts stable**: especially `verified_actions.json`, `route_plan.json`, `energy_breakdown.json`, and `energy_state_vector.json` produced by existing Kona plugins.
- **Dependencies**: only add packages already in the project's dependency list (`numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`) plus `minisom` (MIT, pure Python, pip-installable). Do NOT add plotly, torch, or any other heavy dependency for this task.

---

## 1) Context: What already exists (DO NOT recreate)

### 1.1 Kona EBM pipeline (fully implemented)

The Kona energy-based model pipeline already computes:
- **Observed metric vectors** per entity: `{duration_p95, queue_delay_p95, error_rate, rate_per_min, background_overhead_per_min}`
- **Ideal metric vectors**: via signed baseline OR in-dataset Pareto frontier OR synthetic target
- **Energy scores**: `E_total = E_gap + E_constraints` where `E_gap = Σ w_i * gap_i²`
- **Verified actions**: ranked by `delta_energy` (energy reduction)
- **Route plans**: multi-step beam search from current→ideal (when `route_max_depth >= 2`)

Key files (scan these before writing any code):
```
src/statistic_harness/core/stat_plugins/ideaspace.py    # Energy EBM + action verifier handlers
src/statistic_harness/core/ideaspace_route.py            # Route solver (RouteCandidate, RouteConfig, solve_kona_route_plan)
src/statistic_harness/core/ideaspace_feature_extractor.py # Column inference, entity slicing, KPI summary
src/statistic_harness/core/lever_library.py              # Lever definitions + evidence gates
src/statistic_harness/core/baselines.py                  # Signed baseline loading
docs/schemas/kona_route_plan.schema.json                 # Route plan artifact schema
docs/kona_energy_ideaspace_architecture.md               # Architecture spec
```

### 1.2 Existing energy keys and weights

```python
MINIMIZE_KEYS = ("duration_p95", "queue_delay_p95", "error_rate")
MAXIMIZE_KEYS = ("rate_per_min",)
ENERGY_MINIMIZE_KEYS = MINIMIZE_KEYS + ("background_overhead_per_min",)
ENERGY_MAXIMIZE_KEYS = MAXIMIZE_KEYS
```

Default weights are defined in `_default_energy_weights()` in `ideaspace.py`. The energy gap formula is:
- Minimize metric: `gap = max(0, (x - x*) / max(|x*|, eps))`
- Maximize metric: `gap = max(0, (x* - x) / max(|x|, eps))`
- `E_gap = Σ w_k * gap_k²`

### 1.3 Plugin system

Plugins live in `plugins/<plugin_id>/` with:
- `plugin.yaml` (id, type, depends_on, sandbox)
- `plugin.py` (thin wrapper calling `run_plugin("<id>", ctx)` OR full implementation)
- `config.schema.json` + `output.schema.json`

Stat plugin handlers are registered in:
- `src/statistic_harness/core/stat_plugins/registry.py` (HANDLERS dict)
- Handler functions live in `ideaspace.py` or addon files

### 1.4 Entity model

Entities are slices of the dataset (e.g., ALL, per-PROCESS_ID, per-HOST). Each entity has:
- `entity_key` (string, e.g., "ALL", "QEMAIL", "LOSEXTCHLD")
- Observed metrics dict
- Ideal metrics dict
- Energy terms, energy_gap, energy_constraints, energy_total

---

## 2) What you will add

Three new capabilities, each as a new plugin:

| Plugin ID | Type | Purpose | Depends on |
|-----------|------|---------|------------|
| `analysis_kona_3d_landscape_v1` | analysis | 3D energy landscape visualization artifact | `analysis_ideaspace_energy_ebm_v1` |
| `analysis_kohonen_energy_map_v1` | analysis | Kohonen SOM topology-preserving map of entity energy space | `analysis_ideaspace_energy_ebm_v1` |
| `analysis_kona_weight_learner_v1` | analysis | Per-dataset optimal energy weight discovery via L-BFGS-B | `analysis_ideaspace_energy_ebm_v1`, `analysis_ebm_action_verifier_v1` |

All three are **disabled by default** — they produce empty/NA results unless their config enables them.

---

## 3) Plugin 1: `analysis_kona_3d_landscape_v1` — 3D Energy Landscape

### 3.1 Purpose

Generate a 3D representation of the Kona energy landscape showing:
1. **Existing state**: where each entity currently sits in metric space, with energy as the Z-axis (height = energy = bad)
2. **Ideal state**: the target region (local minima in the energy landscape)
3. **Route path**: if a route plan exists, the multi-step path from current to ideal overlaid on the landscape

This is the "3D Kona map" — an energy surface where **local minima are optimal activities** and the route planner navigates downhill.

### 3.2 Algorithm

**Step 1 — Load energy state vector**
```
Read: artifacts/analysis_ideaspace_energy_ebm_v1/energy_state_vector.json
Extract: entities[], energy_keys[], weights{}
```

**Step 2 — Build feature matrix**
For each entity with valid observed metrics, build a row vector using `energy_keys`. Normalize each column to [0, 1] range using min-max scaling across all entities. Store the scaler parameters for inverse mapping.

**Step 3 — Reduce to 2D surface coordinates**
Use PCA (from sklearn.decomposition) to project the N-dimensional observed metric space to 2 principal components. These become the X and Y axes of the landscape.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(feature_matrix)  # shape: (n_entities, n_features)
pca = PCA(n_components=2, random_state=int(run_seed))
X_2d = pca.fit_transform(X_scaled)  # shape: (n_entities, 2)
```

**Step 4 — Compute energy surface**
The Z-axis is the entity's `energy_total`. This creates a 3D point cloud: `(pc1, pc2, energy_total)` per entity.

To generate a continuous surface (for visualization), create a grid over the PC1/PC2 range and interpolate energy using scipy's griddata:

```python
from scipy.interpolate import griddata
import numpy as np

grid_x = np.linspace(X_2d[:, 0].min() - margin, X_2d[:, 0].max() + margin, grid_resolution)
grid_y = np.linspace(X_2d[:, 1].min() - margin, X_2d[:, 1].max() + margin, grid_resolution)
grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
grid_z = griddata(X_2d, energies, (grid_xx, grid_yy), method='cubic', fill_value=np.nanmax(energies))
```

**Step 5 — Mark ideal points**
Project ideal vectors through the same PCA transform. If baseline-derived, the ideal is a single point. If frontier-derived, it's a set of points. Mark these on the surface as the target minima.

```python
ideal_matrix = build_ideal_matrix(entities, energy_keys)  # same shape as feature_matrix
ideal_scaled = scaler.transform(ideal_matrix)
ideal_2d = pca.transform(ideal_scaled)
```

**Step 6 — Overlay route path (optional)**
If `artifacts/analysis_ebm_action_verifier_v1/route_plan.json` exists with `decision == "modeled"`:
- Extract `modeled_metrics_after` from each step
- Project each intermediate state through scaler → PCA
- Compute energy at each step
- Emit as an ordered polyline on the 3D surface

**Step 7 — Emit artifacts**

Produce two artifacts:

1. `kona_3d_landscape.json` — machine-readable:
```json
{
  "schema_version": "kona_3d_landscape.v1",
  "pca_components": [[...], [...]],
  "pca_explained_variance_ratio": [0.65, 0.22],
  "scaler_min": [...],
  "scaler_scale": [...],
  "grid": {
    "x": [...],
    "y": [...],
    "z": [[...], ...]
  },
  "entities": [
    {
      "entity_key": "ALL",
      "pc1": 0.42,
      "pc2": -0.18,
      "energy_total": 3.72,
      "label": "ALL"
    }
  ],
  "ideal_points": [
    {"pc1": -0.55, "pc2": 0.10, "energy_total": 0.0, "label": "IDEAL"}
  ],
  "route_path": [
    {"step_index": 0, "pc1": 0.42, "pc2": -0.18, "energy": 3.72, "label": "current"},
    {"step_index": 1, "pc1": 0.20, "pc2": -0.05, "energy": 2.10, "label": "tune_schedule_qemail_frequency_v1"},
    {"step_index": 2, "pc1": -0.30, "pc2": 0.08, "energy": 0.85, "label": "add_qpec_capacity_plus_one_v1"}
  ]
}
```

2. `kona_3d_landscape.html` — self-contained HTML visualization using inline JavaScript + canvas (no external CDN dependencies):
   - Render the energy surface as a wireframe/heatmap on a 2D canvas using isometric projection
   - Mark entity points as colored circles (color = energy, green=low, red=high)
   - Mark ideal points as blue stars
   - Draw route path as a thick green line with step numbers
   - Include axis labels: "PC1 (X% variance)", "PC2 (Y% variance)", "Energy"
   - Include a legend
   - This must be a **single self-contained HTML file** with no external dependencies

### 3.3 Config schema (`plugins/analysis_kona_3d_landscape_v1/config.schema.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "additionalProperties": true,
  "properties": {
    "enabled": {"type": "boolean", "default": false},
    "grid_resolution": {"type": "integer", "minimum": 10, "maximum": 200, "default": 50},
    "grid_margin_pct": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.15},
    "include_route_overlay": {"type": "boolean", "default": true},
    "min_entities_for_surface": {"type": "integer", "minimum": 3, "default": 4}
  }
}
```

### 3.4 Gating

- If `enabled` is false (default): return `status="na"`, `summary="3D landscape disabled"`, `gating_reason="disabled_by_config"`.
- If fewer than `min_entities_for_surface` entities have valid energy scores: return NA with `gating_reason="insufficient_entities"`.
- If energy state vector artifact is missing: return NA with `gating_reason="missing_energy_artifact"`.

### 3.5 Finding

```python
{
    "kind": "kona_3d_landscape",
    "measurement_type": "measured",
    "pca_explained_variance_total": float,  # sum of explained_variance_ratio
    "entity_count": int,
    "ideal_count": int,
    "route_steps": int,  # 0 if no route
    "grid_resolution": int,
}
```

### 3.6 Performance

- PCA + griddata + HTML generation: O(n_entities * n_features) for PCA, O(grid_resolution²) for interpolation.
- Budget: honor `timer.exceeded()` checks before expensive operations.

---

## 4) Plugin 2: `analysis_kohonen_energy_map_v1` — Kohonen Self-Organizing Map

### 4.1 Purpose

Add an alternative topology-preserving dimensionality reduction using a Kohonen Self-Organizing Map (SOM). Unlike PCA (linear), SOM preserves nonlinear neighborhood relationships — entities that are similar in the full metric space remain neighbors on the map.

This produces a 2D grid where each cell is a "neuron" that learned to represent a region of the energy space. Entities are mapped to their Best Matching Unit (BMU). The U-Matrix (distance between neighboring neurons) reveals cluster boundaries.

### 4.2 Dependency

```
pip install minisom
```

`minisom` is a pure Python MIT-licensed SOM implementation with numpy backend. It has no heavy dependencies. Add it to `pyproject.toml` under `[project.optional-dependencies]` as `som = ["minisom>=2.3"]` — NOT as a hard dependency. The plugin must gracefully degrade if minisom is not installed.

### 4.3 Algorithm

**Step 1 — Load energy state vector** (same as Plugin 1)

**Step 2 — Build and normalize feature matrix** (same as Plugin 1)

**Step 3 — Determine SOM grid size**

```python
# Heuristic: sqrt(5 * sqrt(n_entities)) per side, clamped to [3, 20]
import math
n = len(entities)
side = max(3, min(20, int(math.sqrt(5 * math.sqrt(n)))))
som_shape = (side, side)
```

**Step 4 — Train SOM**

```python
from minisom import MiniSom

som = MiniSom(
    x=som_shape[0],
    y=som_shape[1],
    input_len=n_features,
    sigma=max(1.0, side / 2.0),
    learning_rate=0.5,
    neighborhood_function='gaussian',
    random_seed=int(run_seed),
)
som.pca_weights_init(X_scaled)  # PCA-based initialization for reproducibility
som.train(X_scaled, num_iteration=min(500 * n, 50000), random_order=False)  # deterministic order
```

CRITICAL: `random_order=False` is mandatory for determinism.

**Step 5 — Map entities to BMUs**

```python
entity_bmus = []
for i, entity in enumerate(entities):
    bmu = som.winner(X_scaled[i])
    entity_bmus.append({
        "entity_key": entity["entity_key"],
        "bmu_x": int(bmu[0]),
        "bmu_y": int(bmu[1]),
        "energy_total": float(entity.get("energy_total", 0.0)),
        "quantization_error": float(np.linalg.norm(X_scaled[i] - som.get_weights()[bmu[0], bmu[1]])),
    })
```

**Step 6 — Map ideal to BMU**

```python
ideal_bmus = []
for ideal_vec in ideal_vectors_scaled:
    bmu = som.winner(ideal_vec)
    ideal_bmus.append({"bmu_x": int(bmu[0]), "bmu_y": int(bmu[1]), "label": "IDEAL"})
```

**Step 7 — Compute U-Matrix**

```python
umatrix = som.distance_map()  # shape: (som_shape[0], som_shape[1])
```

The U-Matrix shows average distance between each neuron and its neighbors. High values = cluster boundaries. Low values = homogeneous regions. Local minima in the U-Matrix correspond to cluster centers — these are the "optimal activity basins" in the Kona analogy.

**Step 8 — Compute energy heatmap on SOM grid**

For each neuron, compute the energy of its weight vector relative to the ideal:

```python
weights = som.get_weights()  # shape: (grid_x, grid_y, n_features)
energy_grid = np.zeros(som_shape)
for i in range(som_shape[0]):
    for j in range(som_shape[1]):
        neuron_metrics = scaler.inverse_transform(weights[i, j].reshape(1, -1))[0]
        # Compute energy using the same formula as the EBM plugin
        energy_grid[i, j] = compute_energy_from_metrics(neuron_metrics, ideal_metrics, energy_weights)
```

**Step 9 — Overlay route path on SOM**

If route plan exists, project each step's `modeled_metrics_after` through the scaler → find BMU → trace path on grid.

**Step 10 — Emit artifacts**

1. `kohonen_energy_map.json` — machine-readable:
```json
{
  "schema_version": "kohonen_energy_map.v1",
  "som_shape": [10, 10],
  "sigma": 5.0,
  "learning_rate": 0.5,
  "iterations": 5000,
  "quantization_error": 0.142,
  "topographic_error": 0.08,
  "umatrix": [[...], ...],
  "energy_grid": [[...], ...],
  "entity_bmus": [...],
  "ideal_bmus": [...],
  "route_path_bmus": [...],
  "feature_names": ["duration_p95", "queue_delay_p95", ...]
}
```

2. `kohonen_energy_map.html` — self-contained HTML:
   - Render U-Matrix as a heatmap (dark = boundary, light = basin)
   - Overlay entity markers at their BMU positions (colored by energy)
   - Mark ideal BMU(s) with distinct marker
   - Draw route path as connected line through BMU positions
   - Include hover tooltip showing entity_key + energy_total
   - Legend explaining U-Matrix coloring

### 4.4 Config schema (`plugins/analysis_kohonen_energy_map_v1/config.schema.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "additionalProperties": true,
  "properties": {
    "enabled": {"type": "boolean", "default": false},
    "som_max_side": {"type": "integer", "minimum": 3, "maximum": 30, "default": 20},
    "som_max_iterations": {"type": "integer", "minimum": 100, "maximum": 100000, "default": 50000},
    "include_route_overlay": {"type": "boolean", "default": true},
    "min_entities": {"type": "integer", "minimum": 3, "default": 5}
  }
}
```

### 4.5 Gating

- If `enabled` is false (default): return NA.
- If minisom not installed: return NA with `gating_reason="minisom_not_installed"`.
- If fewer than `min_entities` entities: return NA with `gating_reason="insufficient_entities"`.

### 4.6 Finding

```python
{
    "kind": "kohonen_energy_map",
    "measurement_type": "measured",
    "som_shape": [int, int],
    "quantization_error": float,
    "topographic_error": float,
    "entity_count": int,
    "ideal_bmu_count": int,
    "route_steps_mapped": int,
}
```

### 4.7 How Kohonen complements PCA (Plugin 1)

PCA gives a linear projection — fast and interpretable but loses nonlinear structure. Kohonen SOM preserves topology: if two processes have similar operational profiles, they land on adjacent neurons even if PCA would scatter them. For datasets with clusters of process types (close-cycle vs. batch vs. scheduled), SOM reveals the cluster structure that PCA flattens.

Both plugins read the same `energy_state_vector.json` and use the same scaler. They produce complementary views.

---

## 5) Plugin 3: `analysis_kona_weight_learner_v1` — Per-Dataset Energy Weight Optimization

### 5.1 Purpose

The existing EBM uses hardcoded default weights for energy metrics. This plugin learns **dataset-specific optimal weights** that maximize the correlation between energy reduction and actual operational impact (measured by existing lever evidence).

This replaces the original "train a tiny model per dataset" idea. Instead of training a neural network (wrong architecture for this problem), we solve a **constrained convex optimization** that finds weights maximizing the energy model's ability to rank actions by their true impact.

### 5.2 Algorithm

**Step 1 — Load verified actions with evidence metrics**

```
Read: artifacts/analysis_ebm_action_verifier_v1/verified_actions.json
Read: artifacts/analysis_ideaspace_energy_ebm_v1/energy_state_vector.json
```

Extract for each verified action:
- `lever_id`, `delta_energy`, `confidence`
- Evidence metrics (e.g., `eligible_wait_gt_hours`, `qemail_rows`, `qpec_host_count`)

**Step 2 — Build calibration targets**

For each action with evidence, compute a "ground truth impact score" from evidence:
```python
# Impact is derived from evidence metrics, not from the energy model
# This gives us an independent signal to calibrate weights against
impact_score = confidence * estimated_improvement_pct / 100.0
```

If fewer than 3 actions have evidence with `estimated_improvement_pct`, return NA — not enough signal to learn from.

**Step 3 — Optimize weights via L-BFGS-B**

Find weights `w*` that maximize rank correlation (Spearman) between energy-based ranking and evidence-based ranking:

```python
from scipy.optimize import minimize
from scipy.stats import spearmanr

def objective(w_raw):
    # Ensure weights are positive and sum-normalized
    w = np.exp(w_raw)
    w = w / w.sum()
    
    # Recompute energy deltas with candidate weights
    deltas = []
    for action in calibration_actions:
        observed = action["observed_metrics"]
        ideal = action["ideal_metrics"]
        modeled = action["modeled_metrics_after"]
        
        e_before = sum(w[k] * gap(observed[k], ideal[k])**2 for k in energy_keys)
        e_after = sum(w[k] * gap(modeled[k], ideal[k])**2 for k in energy_keys)
        deltas.append(e_before - e_after)
    
    # Maximize Spearman rank correlation with ground truth impacts
    if len(set(deltas)) < 2 or len(set(impacts)) < 2:
        return 0.0  # degenerate — no gradient
    corr, _ = spearmanr(deltas, impacts)
    return -corr  # minimize negative correlation

# Initial weights from defaults
w0 = np.log(np.array([default_weights[k] for k in energy_keys]))

result = minimize(
    objective,
    w0,
    method='L-BFGS-B',
    options={'maxiter': 200, 'ftol': 1e-8},
)

learned_w_raw = np.exp(result.x)
learned_weights = {k: float(v / learned_w_raw.sum()) for k, v in zip(energy_keys, learned_w_raw)}
```

**Step 4 — Validate learned weights**

Compare default-weight ranking vs learned-weight ranking:
```python
default_corr = spearmanr(default_deltas, impacts).statistic
learned_corr = spearmanr(learned_deltas, impacts).statistic
improvement = learned_corr - default_corr
```

If `improvement <= 0.0`, learned weights are not better — emit the result but flag `improved=false`.

**Step 5 — Emit artifacts**

1. `learned_energy_weights.json`:
```json
{
  "schema_version": "kona_weight_learner.v1",
  "default_weights": {"duration_p95": 1.0, "queue_delay_p95": 1.2, ...},
  "learned_weights": {"duration_p95": 0.85, "queue_delay_p95": 1.55, ...},
  "calibration_actions": 7,
  "default_rank_correlation": 0.62,
  "learned_rank_correlation": 0.89,
  "improvement": 0.27,
  "improved": true,
  "optimization_converged": true,
  "optimization_iterations": 42
}
```

2. `weight_comparison.json`:
```json
{
  "per_action_comparison": [
    {
      "lever_id": "tune_schedule_qemail_frequency_v1",
      "default_delta_energy": 1.82,
      "learned_delta_energy": 2.15,
      "evidence_impact": 0.45,
      "default_rank": 2,
      "learned_rank": 1
    }
  ]
}
```

### 5.3 Config schema (`plugins/analysis_kona_weight_learner_v1/config.schema.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "additionalProperties": true,
  "properties": {
    "enabled": {"type": "boolean", "default": false},
    "min_calibration_actions": {"type": "integer", "minimum": 2, "default": 3},
    "max_optimizer_iterations": {"type": "integer", "minimum": 10, "maximum": 1000, "default": 200},
    "apply_learned_weights": {"type": "boolean", "default": false}
  }
}
```

Note: `apply_learned_weights` is a future hook — when true, the learned weights would be written to a location that the EBM plugin reads on subsequent runs. For now, this is ALWAYS treated as false; the plugin only reports what it learned, it does not modify other plugins' behavior.

### 5.4 Gating

- If `enabled` is false (default): return NA.
- If fewer than `min_calibration_actions` verified actions have evidence with `estimated_improvement_pct`: return NA with `gating_reason="insufficient_calibration_data"`.

### 5.5 Finding

```python
{
    "kind": "kona_weight_learner",
    "measurement_type": "modeled",
    "calibration_actions": int,
    "default_rank_correlation": float,
    "learned_rank_correlation": float,
    "improvement": float,
    "improved": bool,
    "top_weight_change": {"metric": str, "default": float, "learned": float},
}
```

---

## 6) Implementation checklist

### 6.1 New plugin directories

Create these directories and files:

```
plugins/analysis_kona_3d_landscape_v1/
  plugin.yaml
  plugin.py
  config.schema.json
  output.schema.json

plugins/analysis_kohonen_energy_map_v1/
  plugin.yaml
  plugin.py
  config.schema.json
  output.schema.json

plugins/analysis_kona_weight_learner_v1/
  plugin.yaml
  plugin.py
  config.schema.json
  output.schema.json
```

### 6.2 plugin.yaml template

Each plugin.yaml follows this pattern:
```yaml
id: analysis_kona_3d_landscape_v1
version: "1.0.0"
type: analysis
depends_on:
  - analysis_ideaspace_energy_ebm_v1
sandbox:
  network: false
  filesystem: read_artifacts
```

For the weight learner, add `analysis_ebm_action_verifier_v1` to `depends_on`.

### 6.3 plugin.py pattern

**For plugins 1 and 3** (no external optional dependency):
```python
from __future__ import annotations
from statistic_harness.core.stat_plugins.registry import run_plugin

class Plugin:
    def run(self, ctx):
        return run_plugin("analysis_kona_3d_landscape_v1", ctx)
```

**For plugin 2** (optional minisom dependency):
```python
from __future__ import annotations
from statistic_harness.core.stat_plugins.registry import run_plugin

class Plugin:
    def run(self, ctx):
        return run_plugin("analysis_kohonen_energy_map_v1", ctx)
```

### 6.4 Register handlers

**Option A (preferred)**: Create a new handler module:
```
src/statistic_harness/core/stat_plugins/kona_visualization.py
```

This module contains all three handler functions:
- `_kona_3d_landscape_v1(plugin_id, ctx, df, config, inferred, timer, sample_meta) -> PluginResult`
- `_kohonen_energy_map_v1(plugin_id, ctx, df, config, inferred, timer, sample_meta) -> PluginResult`
- `_kona_weight_learner_v1(plugin_id, ctx, df, config, inferred, timer, sample_meta) -> PluginResult`

Export a HANDLERS dict:
```python
KONA_VIS_HANDLERS: dict[str, Callable[..., PluginResult]] = {
    "analysis_kona_3d_landscape_v1": _kona_3d_landscape_v1,
    "analysis_kohonen_energy_map_v1": _kohonen_energy_map_v1,
    "analysis_kona_weight_learner_v1": _kona_weight_learner_v1,
}
```

**Then** in `registry.py`, import and merge:
```python
from statistic_harness.core.stat_plugins.kona_visualization import KONA_VIS_HANDLERS
HANDLERS.update(KONA_VIS_HANDLERS)
```

Scan `registry.py` first to find where other addon HANDLERS are merged — follow the exact same pattern.

### 6.5 output.schema.json

Use the standard output schema (same as `analysis_kona_signal_coverage_audit_v1/output.schema.json` — copy it).

### 6.6 Shared helpers

The handler module should import from `ideaspace.py`:
```python
from statistic_harness.core.stat_plugins.ideaspace import (
    _default_energy_weights,
    _energy_terms,
    _energy_gap_for_metrics,
    ENERGY_MINIMIZE_KEYS,
    ENERGY_MAXIMIZE_KEYS,
)
```

Also import from the stat_plugins utilities:
```python
from statistic_harness.core.stat_plugins import BudgetTimer, stable_id, deterministic_sample
from statistic_harness.core.types import PluginResult, PluginArtifact
from statistic_harness.core.utils import read_json, write_json
```

### 6.7 HTML generation

For the self-contained HTML visualizations, use Python string templating (NOT Jinja2). Generate the HTML as a single string with inline `<style>` and `<script>` blocks. Use `<canvas>` for rendering. No external CDN, no fetch calls, no iframes.

The HTML must render correctly when opened directly in a browser from the local filesystem.

---

## 7) Tests

### 7.1 Required test file

Create: `tests/plugins/test_kona_visualization.py`

### 7.2 Test cases

```python
def test_3d_landscape_disabled_by_default():
    """Plugin returns NA when config.enabled is false."""

def test_3d_landscape_insufficient_entities():
    """Plugin returns NA when fewer than min_entities_for_surface entities."""

def test_3d_landscape_produces_artifacts():
    """With enabled=true and sufficient entities, produces kona_3d_landscape.json and .html."""

def test_3d_landscape_deterministic():
    """Same input + same seed → identical artifact content."""

def test_3d_landscape_route_overlay():
    """When route_plan.json exists with modeled decision, route path appears in artifact."""

def test_kohonen_disabled_by_default():
    """Plugin returns NA when config.enabled is false."""

def test_kohonen_graceful_no_minisom():
    """Plugin returns NA with gating_reason when minisom not installed."""

def test_kohonen_produces_artifacts():
    """With enabled=true and minisom installed, produces kohonen_energy_map.json and .html."""

def test_kohonen_deterministic():
    """Same input + same seed → identical SOM weights and BMU assignments."""

def test_weight_learner_disabled_by_default():
    """Plugin returns NA when config.enabled is false."""

def test_weight_learner_insufficient_actions():
    """Returns NA when fewer than min_calibration_actions have evidence."""

def test_weight_learner_produces_learned_weights():
    """With sufficient calibration data, produces learned_energy_weights.json."""

def test_weight_learner_no_regression_when_no_improvement():
    """When learned weights don't improve ranking, improved=false is set."""
```

### 7.3 Test fixtures

Create synthetic fixtures that exercise the plugins without needing real ERP data:

```
tests/fixtures/kona_vis/
  energy_state_vector_5_entities.json    # 5 entities with varied metrics
  verified_actions_4_levers.json         # 4 actions with evidence + estimated_improvement_pct
  route_plan_3_steps.json                # modeled route with 3 steps
```

Build these by extracting the minimum required fields from the schemas documented above.

---

## 8) Preflight scan commands (run these before writing code)

```bash
rg -n "analysis_kona_3d_landscape" -S
rg -n "analysis_kohonen_energy_map" -S
rg -n "analysis_kona_weight_learner" -S
rg -n "minisom" -S
rg -n "KONA_VIS_HANDLERS" -S
rg -n "from.*ideaspace import" src/statistic_harness/core/stat_plugins/ -S
```

If any of these return results, **extend** the existing code rather than creating duplicates.

---

## 9) Relationship to microjpt

The microjpt repo (100-line Julia GPT) was evaluated for this task and **rejected** because:
- GPT architecture (causal token prediction) is architecturally wrong for energy landscape navigation
- The harness needs topology-preserving dimensionality reduction (Kohonen) and convex optimization (L-BFGS-B), not sequence generation
- microjpt's training loop (manual backprop on character sequences) cannot be repurposed for metric vector optimization without a complete rewrite

The "train something small per dataset" instinct is captured by Plugin 3 (weight learner) — which learns dataset-specific energy weights using constrained optimization rather than a neural network. This is faster, deterministic, interpretable, and auditable under the four pillars framework.

---

## 10) Summary of what's new vs what's reused

| Component | Status |
|-----------|--------|
| Energy computation (`_energy_terms`, `_energy_gap_for_metrics`) | **REUSE** from `ideaspace.py` |
| Entity slicing and feature extraction | **REUSE** from `ideaspace_feature_extractor.py` |
| Route plan loading and parsing | **REUSE** existing artifact read pattern |
| PCA dimensionality reduction | **NEW** — sklearn.decomposition.PCA |
| Energy surface interpolation | **NEW** — scipy.interpolate.griddata |
| Kohonen SOM training | **NEW** — minisom (optional dependency) |
| Energy weight optimization | **NEW** — scipy.optimize.minimize (L-BFGS-B) |
| 3D landscape HTML rendering | **NEW** — inline canvas + isometric projection |
| SOM heatmap HTML rendering | **NEW** — inline canvas + grid coloring |
| Plugin registration pattern | **REUSE** existing HANDLERS merge pattern |
| Plugin directory structure | **REUSE** existing plugin.yaml + plugin.py + schemas pattern |

---

## DETERMINISM CONTRACT

All three plugins must satisfy:
1. Same `run_seed` + same input data → identical artifacts byte-for-byte
2. No random calls without explicit seed (PCA: `random_state=seed`, SOM: `random_seed=seed`, optimizer: deterministic by default)
3. All orderings use stable tie-breaks (sort by energy desc, then entity_key asc)
4. HTML output must be deterministic (no timestamps, no random IDs)
5. All floating point values rounded to 6 decimal places in JSON artifacts
