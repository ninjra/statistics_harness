# Statistics Harness — Cross-Domain Plugin Implementation Guide (182 Plugins)

**D0:** 2026-03-04
**Purpose:** Self-contained instructions for Claude Code to implement 182 cross-domain analysis plugins
**IP Status:** INTERNAL — Statistics Harness is personal IP (Justin Ram / Infoil LLC)
**Source:** Cross-domain brainstorm mapping 40+ knowledge domains to batch-process log analysis

---

## Table of Contents

1. [Pre-Flight Checklist](#1-pre-flight-checklist)
2. [Architecture & Conventions](#2-architecture--conventions)
3. [Dependency Management](#3-dependency-management)
4. [Plugin Scaffold Template](#4-plugin-scaffold-template)
5. [Handler Signature & Return Contract](#5-handler-signature--return-contract)
6. [Addon Module Pattern](#6-addon-module-pattern)
7. [Implementation Order](#7-implementation-order)
8. [Plugin Specifications (1–182)](#8-plugin-specifications-1182)
9. [Post-Plugin Checklist](#9-post-plugin-checklist)
10. [Testing Protocol](#10-testing-protocol)
11. [Session Log Protocol](#11-session-log-protocol)

---

## 1. Pre-Flight Checklist

Before starting ANY plugin work:

```bash
# 1. Sync repo
git pull origin main

# 2. Verify existing test baseline
python -m pytest -q
# Record pass/fail count as your baseline

# 3. Confirm plugin protocol location
cat src/statistic_harness/core/types.py
# The Plugin protocol: run(self, ctx: PluginContext) -> PluginResult

# 4. Review existing addon for pattern reference
cat src/statistic_harness/core/stat_plugins/next30b_addon.py | head -100

# 5. Check current plugin count
grep -c '^  analysis_' config/plugin_kind_map.yaml
```

---

## 2. Architecture & Conventions

### Critical Architecture Facts (from repo analysis)

1. **Thin wrapper pattern**: Each plugin lives in `plugins/<plugin_id>/plugin.py` and delegates to the central registry:
```python
from __future__ import annotations
from statistic_harness.core.stat_plugins.registry import run_plugin

class Plugin:
    def run(self, ctx):
        return run_plugin("<plugin_id>", ctx)
```

2. **Central handler dispatch**: `src/statistic_harness/core/stat_plugins/registry.py` contains a `HANDLERS` dict mapping plugin_id → handler function. Addon modules export their own `HANDLERS` dict which gets merged via `HANDLERS.update(...)`.

3. **Addon module pattern**: Large batches of handlers live in separate addon files:
   - `registry.py` — base handlers (~50 plugins)
   - `topo_tda_addon.py` — TDA/topology pack
   - `ideaspace.py` — Kona energy/ideaspace plugins
   - `erp_next_wave.py` — ERP-specific plugins
   - `next30_addon.py` — first expansion wave
   - `next30b_addon.py` — second expansion wave
   - **NEW: `cross_domain_addon.py`** — this implementation (182 plugins)

4. **Handler signature** (exact — do not deviate):
```python
def _my_handler(
    plugin_id: str,
    ctx,                        # PluginContext
    df: pd.DataFrame,           # already loaded, sampled, filtered
    config: dict[str, Any],     # merged settings + budget
    inferred: dict[str, Any],   # column inference results
    timer: BudgetTimer,         # time budget enforcement
    sample_meta: dict[str, Any] # sampling metadata
) -> PluginResult:
```

5. **PluginResult dataclass** (from `src/statistic_harness/core/types.py`):
```python
@dataclass
class PluginResult:
    status: str                          # "ok" | "skipped" | "degraded" | "error" | "na"
    summary: str
    metrics: dict[str, Any]              # must include rows_seen, rows_used, cols_used
    findings: list[dict[str, Any]]       # each must have: id, severity, confidence, title, what, why
    artifacts: list[PluginArtifact]
    error: PluginError | None = None
    references: list[dict[str, Any]] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = ...
```

6. **Finding contract** (required fields per finding dict):
```python
{
    "id": str,               # stable_id(plugin_id, key)
    "severity": str,         # "info" | "warn" | "critical"
    "confidence": float,     # 0.0–1.0
    "title": str,
    "what": str,             # what was detected
    "why": str,              # why it matters
    "evidence": dict,        # supporting data
    "where": dict,           # optional location context
    "recommendation": str,   # actionable next step
    "measurement_type": str, # "measured" | "inferred" | "modeled"
    "kind": str | None,      # auto-filled from plugin_kind_map if missing
}
```

7. **Column inference** (`inferred` dict keys):
   - `time_column`: str | None
   - `numeric_columns`: list[str]
   - `categorical_columns`: list[str]
   - `duration_column`: str | None (heuristic — look for "duration", "latency", "elapsed" etc.)
   - `group_columns`: list[str]

8. **DAG discovery**: Parent-child relationships are in a column identifiable by heuristic (contains "parent", "ppid", "parent_id", "parent_process"). The addon must dynamically discover this column per-dataset.

9. **plugin_kind_map.yaml format**:
```yaml
version: 1
mappings:
  # ── <kind_category> ──────
  analysis_plugin_id_v1: kind_category
```

### New Kind Categories for Cross-Domain Plugins

In addition to existing kinds (`anomaly`, `changepoint`, `correlation`, `distribution`, `time_series`, `survival`, `regression`, `graph`, `causal`, `counterfactual`, `cluster`, `process_variant`, `chi_square_association`, `tail_isolation`, `sequence_classification`, `percentile_stats`, `chain_makespan`, `dependence_shift`, `graph_edge`, `tda_betti_curve_changepoint`, `bayesian_point_displacement`, `ideaspace_gap`, `role_inference`, `capacity_scale_model`, `close_cycle_capacity_model`), add these new kinds:

```yaml
  # ── bioinformatics ──────
  # ── ecology ──────
  # ── epidemiology ──────
  # ── pharmacokinetics ──────
  # ── physics ──────
  # ── fluid_dynamics ──────
  # ── thermodynamics ──────
  # ── materials_science ──────
  # ── reliability ──────
  # ── control_theory ──────
  # ── topology ──────  (extends existing tda_betti_curve_changepoint)
  # ── chaos ──────
  # ── combinatorial_optimization ──────
  # ── information_theory ──────
  # ── graph_advanced ──────
  # ── sabermetrics ──────
  # ── linguistics ──────
  # ── psychophysics ──────
  # ── psychometrics ──────
  # ── social_network ──────
  # ── game_theory ──────
  # ── forensic_accounting ──────
  # ── astronomy ──────
  # ── oceanography ──────
  # ── seismology ──────
  # ── geospatial ──────
  # ── military_or ──────
  # ── actuarial ──────
  # ── climatology ──────
  # ── signal_processing ──────
  # ── music_theory ──────
  # ── queueing_advanced ──────
  # ── telecom ──────
  # ── compiler_optimization ──────
  # ── archaeology ──────
  # ── evolutionary ──────
  # ── cognitive ──────
  # ── supply_chain ──────
  # ── semiconductor ──────
  # ── library_science ──────
  # ── demography ──────
  # ── voting_theory ──────
  # ── sports_physiology ──────
  # ── computational_geometry ──────
```

---

## 3. Dependency Management

### Already in venv (confirmed from repo imports)
`numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `networkx`, `pyyaml`, `jsonschema`, `matplotlib`

### Packages to install (check first with `pip show <pkg>`)

**Tier 1 — Lightweight / stdlib-only**: No new packages needed. Most Tier 1 plugins use only numpy/scipy/pandas/networkx.

**Tier 2 — Moderate packages**:
```bash
pip install astropy --break-system-packages        # Lomb-Scargle, BLS periodograms
pip install giotto-tda --break-system-packages      # persistent homology, Mapper (may already exist)
pip install kmapper --break-system-packages         # Mapper algorithm
pip install ripser --break-system-packages          # fast persistent homology
pip install gudhi --break-system-packages           # Euler characteristic, simplicial
pip install cdlib --break-system-packages           # community detection
pip install leidenalg --break-system-packages       # Leiden community detection
pip install chainladder --break-system-packages     # actuarial triangles
pip install hrvanalysis --break-system-packages     # HRV-style interval analysis
pip install py-irt --break-system-packages          # item response theory
pip install nolds --break-system-packages           # Lyapunov exponents, fractal dimension
pip install hmmlearn --break-system-packages        # HMMs (may already exist)
pip install ssqueezepy --break-system-packages      # synchrosqueezing transform
pip install dit --break-system-packages             # discrete information theory
pip install nashpy --break-system-packages          # Nash equilibrium
pip install pyextremes --break-system-packages      # extreme value theory
pip install reliability --break-system-packages     # Weibull, bathtub, FMEA
pip install pingouin --break-system-packages        # Cronbach's alpha
pip install esda --break-system-packages            # spatial autocorrelation (Moran's I)
pip install pykrige --break-system-packages         # kriging
pip install benford-py --break-system-packages      # Benford's law (may overlap existing)
pip install powerlaw --break-system-packages        # power law fitting
pip install librosa --break-system-packages         # onset detection, tempo, SSM
pip install python-Levenshtein --break-system-packages  # edit distance
pip install rapidfuzz --break-system-packages       # fast fuzzy matching
pip install python-control --break-system-packages  # control theory
pip install ortools --break-system-packages         # combinatorial optimization
pip install ciw --break-system-packages             # queueing simulation
pip install pygsp --break-system-packages           # graph signal processing
pip install deap --break-system-packages            # evolutionary algorithms
pip install pymoo --break-system-packages           # multi-objective optimization
pip install utide --break-system-packages           # tidal harmonic analysis
pip install lifelines --break-system-packages       # life tables (may already exist)
```

**Install strategy**: Check before installing. Wrap all imports in try/except with `HAS_<PKG>` flags, exactly as existing addons do. If a package is unavailable, the handler returns `status="na"` with a gating reason.

---

## 4. Plugin Scaffold Template

For each plugin_id, create two files:

### `plugins/<plugin_id>/plugin.py`
```python
from __future__ import annotations

from statistic_harness.core.stat_plugins.registry import run_plugin


class Plugin:
    def run(self, ctx):
        return run_plugin("<plugin_id>", ctx)
```

### `plugins/<plugin_id>/plugin.yaml` (optional but recommended)
```yaml
id: <plugin_id>
version: "0.1.0"
kind: <kind_category>
description: "<One-line description>"
source_domain: "<e.g., bioinformatics, ecology, sabermetrics>"
data_requirements:
  needs_dag: <true|false>
  needs_time_series: <true|false>
  needs_numeric: <true|false>
  min_rows: <integer>
dependencies:
  - <package_name>
references:
  - title: "<Paper or doc title>"
    url: "<URL>"
```

---

## 5. Handler Signature & Return Contract

Every handler function MUST follow this exact pattern. Copy from `next30b_addon.py`:

```python
def _my_handler(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)

    # 1. Gate: check prerequisites
    num_cols = _numeric_columns(df, inferred, max_cols=80)
    if not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_columns")

    _ensure_budget(timer)  # call periodically

    # 2. Compute
    # ... your analysis logic ...

    _ensure_budget(timer)

    # 3. Build findings
    findings = []
    if some_condition:
        findings.append(_make_finding(
            plugin_id=plugin_id,
            key="my_finding_key",
            title="Human-readable title",
            what="What was detected",
            why="Why it matters for process optimization",
            evidence={"metric": value, "threshold": threshold},
            recommendation="Specific actionable recommendation",
            severity="warn",  # "info" | "warn" | "critical"
            confidence=0.75,
            measurement_type="measured",
            kind="<kind_category>",
        ))

    # 4. Return
    return _finalize(
        plugin_id, ctx, df, sample_meta,
        summary=f"Analyzed {len(num_cols)} columns with <method_name>",
        findings=findings,
        artifacts=[],  # or list of PluginArtifact
        extra_metrics={"runtime_ms": _runtime_ms(timer), "custom_metric": value},
    )
```

### Helper functions (copy from next30b_addon.py)

The addon module must include these helper functions (copy them exactly from `next30b_addon.py`):

- `_safe_id(plugin_id, key)` — stable finding IDs
- `_artifact_json(ctx, plugin_id, filename, payload, description)` — write JSON artifacts
- `_basic_metrics(df, sample_meta)` — standard metrics dict
- `_make_finding(...)` — contract-compliant finding builder
- `_ok_with_reason(...)` — graceful skip/gate result
- `_finalize(...)` — standard result builder
- `_log_start(ctx, plugin_id, df, config, inferred)` — logging
- `_ensure_budget(timer)` — time budget check
- `_runtime_ms(timer)` — elapsed time
- `_numeric_columns(df, inferred, max_cols)` — numeric column discovery
- `_categorical_columns(df, inferred, max_cols)` — categorical column discovery
- `_time_series(df, inferred)` — time column discovery
- `_split_pre_post(df, inferred)` — temporal splitting
- `_duration_column(df, inferred)` — duration column heuristic
- `_variance_sorted_numeric(df, inferred, limit)` — variance-ranked columns
- `_safe_float(value, default)` — safe numeric conversion
- `_find_graph_columns(df, inferred)` — DAG column discovery (src/dst or parent/child)

---

## 6. Addon Module Pattern

Create the new addon file at: `src/statistic_harness/core/stat_plugins/cross_domain_addon.py`

**IMPORTANT**: Due to the size (182 handlers), split into multiple addon files:

```
src/statistic_harness/core/stat_plugins/
  cross_domain_life_sciences.py        # Plugins 1–24  (bioinformatics, ecology, epidemiology, pharmacokinetics, agriculture)
  cross_domain_physics_engineering.py   # Plugins 25–47  (stat mech, fluid dynamics, thermodynamics, materials, reliability, control)
  cross_domain_pure_math.py            # Plugins 48–72  (TDA, algebra, chaos, combinatorial opt, info theory, graph theory)
  cross_domain_social_sports.py        # Plugins 73–94  (sabermetrics, linguistics, psychophysics, psychometrics, sociology, game theory, forensic)
  cross_domain_earth_space.py          # Plugins 95–121 (astronomy, oceanography, seismology, geospatial, military OR, actuarial, climatology)
  cross_domain_signal_music_network.py # Plugins 122–148 (signal processing, music theory, network science, queueing, telecom, compiler/CS)
  cross_domain_unconventional.py       # Plugins 149–182 (archaeology, evolutionary, cognitive, supply chain, semiconductor, library science, demography, voting, sports physiology, computational geometry)
```

Each file exports a `HANDLERS` dict. Wire them in `registry.py`:

```python
# In registry.py, add imports:
from statistic_harness.core.stat_plugins.cross_domain_life_sciences import (
    HANDLERS as CROSS_DOMAIN_LIFE_HANDLERS,
)
from statistic_harness.core.stat_plugins.cross_domain_physics_engineering import (
    HANDLERS as CROSS_DOMAIN_PHYSICS_HANDLERS,
)
from statistic_harness.core.stat_plugins.cross_domain_pure_math import (
    HANDLERS as CROSS_DOMAIN_MATH_HANDLERS,
)
from statistic_harness.core.stat_plugins.cross_domain_social_sports import (
    HANDLERS as CROSS_DOMAIN_SOCIAL_HANDLERS,
)
from statistic_harness.core.stat_plugins.cross_domain_earth_space import (
    HANDLERS as CROSS_DOMAIN_EARTH_HANDLERS,
)
from statistic_harness.core.stat_plugins.cross_domain_signal_music_network import (
    HANDLERS as CROSS_DOMAIN_SIGNAL_HANDLERS,
)
from statistic_harness.core.stat_plugins.cross_domain_unconventional import (
    HANDLERS as CROSS_DOMAIN_UNCONVENTIONAL_HANDLERS,
)

# At bottom of registry.py, after existing .update() calls:
HANDLERS.update(CROSS_DOMAIN_LIFE_HANDLERS)
HANDLERS.update(CROSS_DOMAIN_PHYSICS_HANDLERS)
HANDLERS.update(CROSS_DOMAIN_MATH_HANDLERS)
HANDLERS.update(CROSS_DOMAIN_SOCIAL_HANDLERS)
HANDLERS.update(CROSS_DOMAIN_EARTH_HANDLERS)
HANDLERS.update(CROSS_DOMAIN_SIGNAL_HANDLERS)
HANDLERS.update(CROSS_DOMAIN_UNCONVENTIONAL_HANDLERS)
```

### Addon file template

```python
"""Cross-domain plugins: <domain_group> (plugins N–M)."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any, Callable

import numpy as np
import pandas as pd

from statistic_harness.core.stat_plugins import (
    BudgetTimer,
    deterministic_sample,
    robust_center_scale,
    stable_id,
    standardized_median_diff,
)
from statistic_harness.core.types import PluginArtifact, PluginResult

# Optional dependencies — always try/except
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except Exception:
    scipy_stats = None
    HAS_SCIPY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except Exception:
    nx = None
    HAS_NETWORKX = False

# ... more optional imports as needed ...


# ── Helpers (copy from next30b_addon.py) ──────────────────────────────────
# Copy ALL helper functions listed in Section 5 here.
# They are module-private and must be duplicated per addon file.
# DO NOT import them from next30b_addon — keep each addon self-contained.


# ── Handlers ──────────────────────────────────────────────────────────────

def _handler_name(plugin_id, ctx, df, config, inferred, timer, sample_meta):
    """<Description>."""
    _log_start(ctx, plugin_id, df, config, inferred)
    # ... implementation ...
    return _finalize(plugin_id, ctx, df, sample_meta, "summary", [], [])


# ── Export ────────────────────────────────────────────────────────────────

HANDLERS: dict[str, Callable[..., PluginResult]] = {
    "analysis_plugin_id_v1": _handler_name,
    # ...
}
```

---

## 7. Implementation Order

### Tier 1: Quick Wins (numpy/scipy only — ~50 plugins)

These require NO new dependencies. Implement first for immediate breadth expansion.

| # | plugin_id | Domain | Method |
|---|-----------|--------|--------|
| 5 | `analysis_shannon_diversity_index_v1` | ecology | Shannon H' per host/step |
| 8 | `analysis_niche_overlap_pianka_v1` | ecology | Pianka's workload similarity |
| 10 | `analysis_cascade_r0_reproduction_v1` | epidemiology | R₀ for failure cascades |
| 12 | `analysis_epicurve_classification_v1` | epidemiology | Anomaly onset shape classification |
| 13 | `analysis_hardy_weinberg_param_coupling_v1` | pop. genetics | Non-random parameter pairing |
| 14 | `analysis_linkage_disequilibrium_steps_v1` | pop. genetics | Non-random step co-failure |
| 18 | `analysis_half_life_utilization_v1` | pharmacokinetics | Resource utilization decay |
| 20 | `analysis_michaelis_menten_saturation_v1` | pharmacokinetics | Queue depth saturation point |
| 24 | `analysis_liebig_minimum_constraint_v1` | agriculture | Binding constraint identification |
| 29 | `analysis_reynolds_number_analogue_v1` | fluid dynamics | Laminar vs turbulent regime |
| 32 | `analysis_entropy_production_rate_v1` | thermodynamics | Degradation early warning |
| 33 | `analysis_carnot_efficiency_bound_v1` | thermodynamics | Theoretical efficiency ceiling |
| 34 | `analysis_exergy_waste_analysis_v1` | thermodynamics | Useful work vs waste decomposition |
| 36 | `analysis_periodic_schedule_fft_v1` | materials science | Hidden periodicities in timestamps |
| 37 | `analysis_defect_density_classification_v1` | materials science | Point/line/planar anomaly types |
| 38 | `analysis_stress_strain_capacity_v1` | materials science | Load-latency yield point |
| 65 | `analysis_kolmogorov_complexity_ncd_v1` | info theory | Compression-based anomaly |
| 66 | `analysis_renyi_entropy_spectrum_v1` | info theory | Parameterized entropy family |
| 73 | `analysis_war_value_above_replacement_v1` | sabermetrics | Component value vs baseline |
| 74 | `analysis_win_probability_added_v1` | sabermetrics | Per-step success contribution |
| 75 | `analysis_leverage_index_dag_v1` | sabermetrics | High-stakes step identification |
| 76 | `analysis_pythagorean_expectation_v1` | sabermetrics | Expected vs actual success rate |
| 77 | `analysis_zipf_law_frequency_v1` | linguistics | Power-law frequency test |
| 78 | `analysis_entropy_rate_step_sequences_v1` | linguistics | Workflow predictability |
| 81 | `analysis_weber_fechner_threshold_v1` | psychophysics | Perceptual alerting thresholds |
| 82 | `analysis_stevens_power_law_severity_v1` | psychophysics | Unified severity scoring |
| 83 | `analysis_signal_detection_dprime_v1` | psychophysics | Alert quality evaluation |
| 93 | `analysis_benford_first_digit_v1` | forensic accounting | First-digit anomaly (if not already fully covered) |
| 94 | `analysis_benford_second_digit_v1` | forensic accounting | Second-digit distribution |
| 137 | `analysis_kingman_vut_decomposition_v1` | queueing advanced | Delay component decomposition |
| 142 | `analysis_erlang_bc_blocking_v1` | telecom | Machine pool sizing |
| 143 | `analysis_cdma_interference_sir_v1` | telecom | Concurrency interference |
| 147 | `analysis_amdahl_law_serial_fraction_v1` | compiler/CS | Scaling diminishing returns |
| 156 | `analysis_hick_law_pool_sizing_v1` | cognitive | Optimal host pool size |
| 158 | `analysis_cognitive_load_concurrency_v1` | cognitive | Concurrency degradation threshold |
| 159 | `analysis_bullwhip_effect_detection_v1` | supply chain | Variance amplification |
| 160 | `analysis_eoq_batch_sizing_v1` | supply chain | Optimal batch size |
| 163 | `analysis_spc_cpk_capability_v1` | semiconductor | Process capability index |
| 166 | `analysis_r2r_ewma_control_v1` | semiconductor | Run-to-run parameter tuning |
| 167 | `analysis_tfidf_step_importance_v1` | library science | Step uniqueness ranking |
| 168 | `analysis_bradford_law_core_hosts_v1` | library science | Core vs peripheral hosts |
| 169 | `analysis_lotka_law_concentration_v1` | library science | Workload concentration |
| 172 | `analysis_population_pyramid_shape_v1` | demography | Queue age distribution shape |
| 174 | `analysis_condorcet_host_ranking_v1` | voting theory | Paradox-resistant ranking |
| 175 | `analysis_borda_count_multicriteria_v1` | voting theory | Multi-criteria ranking |
| 177 | `analysis_hrv_intercompletion_v1` | sports physiology | Inter-completion variability |
| 182 | `analysis_convex_hull_operating_envelope_v1` | comp geometry | Configuration safety boundary |

### Tier 2: Moderate Effort (mature packages — ~60 plugins)

| # | plugin_id | Domain | Package |
|---|-----------|--------|---------|
| 1 | `analysis_smith_waterman_workflow_align_v1` | bioinformatics | `swalign` or numpy DP |
| 2 | `analysis_meme_motif_discovery_v1` | bioinformatics | custom EM or `pymemesuite` |
| 3 | `analysis_hmm_workflow_state_v1` | bioinformatics | `hmmlearn` |
| 4 | `analysis_phylogenetic_workflow_tree_v1` | bioinformatics | `scipy.cluster.hierarchy` |
| 6 | `analysis_rarefaction_coverage_v1` | ecology | `scikit-bio` or numpy |
| 7 | `analysis_lotka_volterra_competition_v1` | ecology | `scipy.integrate` |
| 9 | `analysis_sir_failure_cascade_v1` | epidemiology | `scipy.integrate` + networkx |
| 11 | `analysis_contact_tracing_superspreader_v1` | epidemiology | `networkx` |
| 15 | `analysis_fst_machine_differentiation_v1` | pop. genetics | `numpy` |
| 16 | `analysis_wright_fisher_config_drift_v1` | pop. genetics | `numpy` |
| 17 | `analysis_compartment_redistribution_v1` | pharmacokinetics | `scipy.integrate` |
| 19 | `analysis_hill_equation_dose_response_v1` | pharmacokinetics | `scipy.optimize` |
| 21 | `analysis_response_surface_interaction_v1` | agriculture | `statsmodels` |
| 23 | `analysis_soil_depletion_resource_v1` | agriculture | `scipy.integrate` |
| 25 | `analysis_ising_model_correlation_v1` | stat mech | `numpy` + `numba` |
| 26 | `analysis_percolation_fragility_v1` | stat mech | `networkx` |
| 28 | `analysis_partition_function_landscape_v1` | stat mech | `numpy` |
| 30 | `analysis_navier_stokes_backpressure_v1` | fluid dynamics | `networkx` + `scipy` |
| 39 | `analysis_creep_fatigue_prediction_v1` | materials science | `reliability` |
| 40 | `analysis_bathtub_curve_lifecycle_v1` | reliability | `reliability` |
| 41 | `analysis_fmea_rpn_scoring_v1` | reliability | `pandas` + `networkx` |
| 42 | `analysis_fault_tree_minimal_cuts_v1` | reliability | `reliability` |
| 43 | `analysis_reliability_block_diagram_v1` | reliability | `networkx` |
| 44 | `analysis_pid_controller_analogue_v1` | control theory | `python-control` |
| 45 | `analysis_lyapunov_stability_v1` | control theory | `python-control` |
| 46 | `analysis_bode_frequency_response_v1` | control theory | `scipy.signal` |
| 48 | `analysis_persistent_homology_regimes_v1` | topology | `giotto-tda` or `ripser` |
| 49 | `analysis_mapper_subpopulations_v1` | topology | `kmapper` |
| 52 | `analysis_euler_characteristic_curve_v1` | topology | `gudhi` |
| 57 | `analysis_lyapunov_exponent_chaos_v1` | chaos | `nolds` |
| 58 | `analysis_rqa_recurrence_v1` | chaos | custom numpy (PyRQA optional) |
| 59 | `analysis_fractal_dimension_attractor_v1` | chaos | `nolds` |
| 61 | `analysis_job_shop_scheduling_bound_v1` | combinatorial opt | `ortools` |
| 62 | `analysis_bin_packing_utilization_v1` | combinatorial opt | `ortools` |
| 64 | `analysis_min_cost_flow_throughput_v1` | combinatorial opt | `networkx` |
| 67 | `analysis_pid_synergy_redundancy_v1` | info theory | `dit` |
| 68 | `analysis_excess_entropy_memory_v1` | info theory | `dit` |
| 70 | `analysis_spectral_graph_fiedler_v1` | graph advanced | `scipy.sparse.linalg` |
| 72 | `analysis_cheeger_bottleneck_v1` | graph advanced | `networkx` + `scipy` |
| 79 | `analysis_levenshtein_workflow_dist_v1` | linguistics | `python-Levenshtein` |
| 80 | `analysis_ngram_step_transitions_v1` | linguistics | `collections.Counter` |
| 84 | `analysis_irt_step_difficulty_v1` | psychometrics | `py-irt` |
| 86 | `analysis_cronbach_alpha_consistency_v1` | psychometrics | `pingouin` |
| 87 | `analysis_structural_holes_broker_v1` | social network | `networkx` |
| 90 | `analysis_nash_equilibrium_contention_v1` | game theory | `nashpy` |
| 95 | `analysis_lomb_scargle_periodogram_v2` | astronomy | `astropy` (if v1 exists, extend) |
| 96 | `analysis_bls_periodic_outage_v1` | astronomy | `astropy` |
| 99 | `analysis_tidal_harmonic_cycles_v1` | oceanography | `utide` |
| 103 | `analysis_gutenberg_richter_bvalue_v1` | seismology | `scipy.optimize` |
| 104 | `analysis_omori_aftershock_decay_v1` | seismology | `scipy.optimize` |
| 107 | `analysis_moran_i_autocorrelation_v1` | geospatial | `esda` |
| 109 | `analysis_ripley_k_temporal_clustering_v1` | geospatial | `numpy` |
| 115 | `analysis_chain_ladder_ibnr_v1` | actuarial | `chainladder` |
| 116 | `analysis_buhlmann_credibility_v1` | actuarial | `numpy` |
| 117 | `analysis_lee_carter_trend_decomp_v1` | actuarial | `numpy.linalg.svd` |
| 120 | `analysis_return_period_evt_v1` | climatology | `pyextremes` |
| 122 | `analysis_cepstral_decomposition_v1` | signal processing | `scipy.signal` |
| 123 | `analysis_synchrosqueezing_drift_v1` | signal processing | `ssqueezepy` |
| 125 | `analysis_spectral_coherence_coupling_v1` | signal processing | `scipy.signal` |
| 126 | `analysis_matched_filter_anomaly_v1` | signal processing | `scipy.signal` |
| 127 | `analysis_onset_detection_rhythm_v1` | music theory | `librosa` |
| 128 | `analysis_tempo_tracking_cadence_v1` | music theory | `librosa` |
| 129 | `analysis_self_similarity_matrix_v1` | music theory | `librosa` |
| 131 | `analysis_polyrhythm_contention_v1` | music theory | `librosa` |
| 132 | `analysis_community_detection_dag_v1` | network science | `cdlib` or `leidenalg` |
| 133 | `analysis_network_motif_triads_v2` | network science | custom or extend existing |
| 134 | `analysis_network_controllability_v1` | network science | `networkx` |
| 136 | `analysis_kshell_decomposition_v1` | network science | `networkx` |
| 138 | `analysis_fork_join_straggler_v1` | queueing advanced | `scipy.stats` |
| 144 | `analysis_critical_path_slack_v1` | compiler/CS | `networkx` |
| 145 | `analysis_heft_list_scheduling_v1` | compiler/CS | `networkx` + `heapq` |
| 146 | `analysis_dataflow_dead_output_v1` | compiler/CS | `networkx` |
| 148 | `analysis_graph_coloring_resources_v1` | compiler/CS | `networkx` |
| 152 | `analysis_fitness_landscape_mapping_v1` | evolutionary | `deap` or numpy |
| 153 | `analysis_punctuated_equilibrium_v1` | evolutionary | `ruptures` (may exist) |
| 161 | `analysis_toc_binding_constraint_v1` | supply chain | `networkx` |
| 162 | `analysis_safety_stock_buffer_v1` | supply chain | `scipy.stats` |
| 170 | `analysis_pagerank_dag_influence_v1` | library science | `networkx` (if not already exists) |
| 171 | `analysis_life_table_hazard_v1` | demography | `lifelines` |
| 179 | `analysis_banister_fitness_fatigue_v1` | sports physiology | `scipy.optimize` |
| 180 | `analysis_voronoi_resource_partition_v1` | comp geometry | `scipy.spatial` |
| 181 | `analysis_delaunay_interpolation_v1` | comp geometry | `scipy.spatial` |

### Tier 3: Significant Engineering (~40 plugins)

| # | plugin_id | Domain |
|---|-----------|--------|
| 22 | `analysis_job_sequence_cache_rotation_v1` | agriculture |
| 27 | `analysis_renormalization_multiscale_v1` | stat mech |
| 31 | `analysis_vorticity_thrashing_v1` | fluid dynamics |
| 35 | `analysis_maxwell_demon_monitoring_v1` | thermodynamics |
| 47 | `analysis_controllability_gramian_v1` | control theory |
| 50 | `analysis_persistence_landscape_features_v1` | topology |
| 51 | `analysis_vietoris_rips_higher_order_v1` | topology |
| 53 | `analysis_sheaf_consistency_v1` | algebra |
| 60 | `analysis_takens_embedding_v1` | chaos |
| 63 | `analysis_csp_binding_rules_v1` | combinatorial opt |
| 69 | `analysis_channel_capacity_bound_v1` | info theory |
| 71 | `analysis_graph_wavelet_multiscale_v1` | graph advanced |
| 85 | `analysis_rasch_step_host_scale_v1` | psychometrics |
| 88 | `analysis_weak_ties_bridge_v1` | social network |
| 89 | `analysis_assortativity_scheduling_bias_v1` | social network |
| 91 | `analysis_mechanism_design_scheduling_v1` | game theory |
| 92 | `analysis_vickrey_priority_truth_v1` | game theory |
| 97 | `analysis_pulsar_timing_residual_v1` | astronomy |
| 98 | `analysis_hr_diagram_classification_v1` | astronomy |
| 100 | `analysis_wave_directional_spectra_v1` | oceanography |
| 101 | `analysis_thermohaline_circulation_v1` | oceanography |
| 102 | `analysis_ekman_transport_indirect_v1` | oceanography |
| 105 | `analysis_etas_self_exciting_cascade_v1` | seismology |
| 106 | `analysis_psha_probabilistic_risk_v1` | seismology |
| 108 | `analysis_kriging_missing_data_v1` | geospatial |
| 110 | `analysis_getis_ord_hotspot_v1` | geospatial |
| 111 | `analysis_lanchester_concentration_v1` | military OR |
| 112 | `analysis_salvo_burst_threshold_v1` | military OR |
| 113 | `analysis_koopman_search_allocation_v1` | military OR |
| 114 | `analysis_ooda_loop_bottleneck_v1` | military OR |
| 118 | `analysis_experience_rating_score_v1` | actuarial |
| 119 | `analysis_teleconnection_lag_v1` | climatology |
| 121 | `analysis_climate_attribution_far_v1` | climatology |
| 124 | `analysis_wigner_ville_coupling_v1` | signal processing |
| 130 | `analysis_markov_harmonic_progression_v1` | music theory |
| 135 | `analysis_temporal_network_bottleneck_v1` | network science |
| 139 | `analysis_fluid_model_backlog_v1` | queueing advanced |
| 140 | `analysis_diffusion_approx_heavy_traffic_v1` | queueing advanced |
| 141 | `analysis_phase_type_multimodal_v1` | queueing advanced |
| 149 | `analysis_seriation_ordering_v1` | archaeology |
| 150 | `analysis_stratigraphic_layers_v1` | archaeology |
| 151 | `analysis_harris_matrix_ordering_v1` | archaeology |
| 154 | `analysis_neutral_drift_noise_v1` | evolutionary |
| 155 | `analysis_red_queen_arms_race_v1` | evolutionary |
| 157 | `analysis_fitts_law_precision_v1` | cognitive |
| 164 | `analysis_poisson_yield_complexity_v1` | semiconductor |
| 165 | `analysis_virtual_metrology_early_v1` | semiconductor |
| 173 | `analysis_demographic_transition_v1` | demography |
| 176 | `analysis_arrow_impossibility_pareto_v1` | voting theory |
| 178 | `analysis_vo2max_capacity_ceiling_v1` | sports physiology |

### Tier 4: Research Frontier (~30 plugins)

| # | plugin_id | Domain |
|---|-----------|--------|
| 54 | `analysis_functorial_migration_v1` | algebra |
| 55 | `analysis_operad_composition_v1` | algebra |
| 56 | `analysis_grothendieck_coverage_v1` | algebra |

These are included for completeness. Implement if time permits or as stretch goals.

---

## 8. Plugin Specifications (1–182)

### Key for all specifications

- **plugin_id**: The exact string to use everywhere
- **kind**: The value in plugin_kind_map.yaml
- **data_req**: `DAG` = needs parent_process_id, `TS` = needs time column, `XS` = cross-sectional only
- **deps**: Python packages beyond numpy/scipy/pandas
- **core_logic**: Pseudocode or key computation
- **novel_insight**: What this catches that conventional stats miss
- **finding_template**: Example finding output

### Cluster A: Life Sciences (Plugins 1–24)

#### 1. Smith-Waterman Workflow Alignment
- **plugin_id:** `analysis_smith_waterman_workflow_align_v1`
- **kind:** `bioinformatics`
- **data_req:** DAG
- **deps:** `swalign` or pure numpy DP implementation
- **core_logic:**
  1. Encode each job's step sequence as a symbolic string (step_name tokens)
  2. For all job-type pairs, compute local alignment score via Smith-Waterman
  3. Identify shared sub-workflows scoring above threshold
  4. Report consolidation candidates
- **novel_insight:** Hidden shared sub-processes across dissimilar job types that could be consolidated
- **finding_template:** "Jobs of type A and type B share a 5-step sub-workflow (steps X→Y→Z→W→V) with alignment score 0.87. Consider extracting this as a shared module."

#### 2. MEME Motif Discovery
- **plugin_id:** `analysis_meme_motif_discovery_v1`
- **kind:** `bioinformatics`
- **data_req:** DAG
- **deps:** custom EM implementation (~100 lines) or `pymemesuite`
- **core_logic:**
  1. Encode step sequences as symbolic strings
  2. Run EM for position weight matrices finding enriched 3–8 step motifs
  3. Correlate motif presence with duration/failure outcomes
- **novel_insight:** Recurring step patterns correlated with bottlenecks — a 4-step subsequence always preceding a slowdown

#### 3. HMM Workflow State Detection
- **plugin_id:** `analysis_hmm_workflow_state_v1`
- **kind:** `bioinformatics`
- **data_req:** TS
- **deps:** `hmmlearn`
- **core_logic:**
  1. Features = (step_name encoded, duration, parameter hash)
  2. Fit Gaussian HMM with 3–5 hidden states via BIC selection
  3. Viterbi decode → label each row with latent state
  4. Report state transitions and durations per state
- **novel_insight:** Detects latent regime changes invisible from raw metrics — "pre-failure" state 5 steps before threshold breach

#### 4. Phylogenetic Workflow Tree
- **plugin_id:** `analysis_phylogenetic_workflow_tree_v1`
- **kind:** `bioinformatics`
- **data_req:** DAG
- **deps:** `scipy.cluster.hierarchy`
- **core_logic:**
  1. Compute pairwise edit distances between job-type step sequences
  2. Build UPGMA dendrogram
  3. Identify convergent/divergent workflow clusters over time windows
- **novel_insight:** Shows whether independent workflows converged on similar solutions or diverged into regressions

#### 5. Shannon Diversity Index
- **plugin_id:** `analysis_shannon_diversity_index_v1`
- **kind:** `ecology`
- **data_req:** XS
- **deps:** none (numpy)
- **core_logic:**
  ```python
  # Per host: step_name as "species", proportional duration as p_i
  for host in hosts:
      counts = df[df[host_col] == host][step_col].value_counts(normalize=True)
      H = -np.sum(counts * np.log(counts + 1e-12))
      # H' near 0 = dangerous specialization
      # H' > 2.0 = healthy diversification
  ```
- **novel_insight:** Quantifies workload heterogeneity per machine — H'=0.2 is dangerously specialized

#### 6. Rarefaction Coverage
- **plugin_id:** `analysis_rarefaction_coverage_v1`
- **kind:** `ecology`
- **data_req:** XS
- **deps:** `scikit-bio` or custom numpy
- **core_logic:**
  1. Per machine, plot unique error/step types discovered vs. job runs sampled
  2. Fit rarefaction curve; report whether plateau reached
- **novel_insight:** Determines whether enough runs observed to characterize a machine's full failure repertoire

#### 7. Lotka-Volterra Competition
- **plugin_id:** `analysis_lotka_volterra_competition_v1`
- **kind:** `ecology`
- **data_req:** TS
- **deps:** `scipy.integrate.solve_ivp`
- **core_logic:**
  1. Identify top 2–3 competing job types sharing resources
  2. Estimate N₁, N₂ (concurrent counts), K (capacity), α (cross-impact)
  3. Solve ODEs; classify equilibrium (coexistence, exclusion, unstable)
- **novel_insight:** Predicts whether two job types can stably coexist or one will starve the other

#### 8. Pianka Niche Overlap
- **plugin_id:** `analysis_niche_overlap_pianka_v1`
- **kind:** `ecology`
- **data_req:** XS
- **deps:** none (numpy)
- **core_logic:**
  ```python
  # O_jk = sum(p_ij * p_ik) / sqrt(sum(p_ij^2) * sum(p_ik^2))
  # For each machine pair, compute proportional time on each step type
  ```
- **novel_insight:** Identifies redundant machines (consolidation) and unique specialists (SPOF)

#### 9. SIR Failure Cascade
- **plugin_id:** `analysis_sir_failure_cascade_v1`
- **kind:** `epidemiology`
- **data_req:** DAG + TS
- **deps:** `scipy.integrate`, `networkx`
- **core_logic:**
  1. Build DAG from parent_process_id
  2. Model β = P(failed parent → child delay), γ = recovery rate
  3. Solve SIR on DAG; produce epidemic curve prediction
- **novel_insight:** Predicts epidemic curve of failure cascades

#### 10. R₀ Reproduction Number
- **plugin_id:** `analysis_cascade_r0_reproduction_v1`
- **kind:** `epidemiology`
- **data_req:** DAG
- **deps:** `networkx`
- **core_logic:**
  1. For each "failed" step (duration > threshold), count child steps also delayed
  2. R₀ = mean(secondary_failures per primary_failure)
  3. R₀ > 1 = exponential cascade; R₀ < 1 = self-containing
- **novel_insight:** Single most actionable metric for containment investment

#### 11. Contact Tracing / Super-spreader
- **plugin_id:** `analysis_contact_tracing_superspreader_v1`
- **kind:** `epidemiology`
- **data_req:** DAG + XS
- **deps:** `networkx`
- **core_logic:**
  1. Build extended contact graph: DAG edges + shared-resource edges
  2. Trace failures backward through both dependency and resource-mediated connections
  3. Rank nodes by out-degree of failure propagation
- **novel_insight:** Reveals "super-spreader" machines amplifying failures through resource contention

#### 12. Epi-Curve Classification
- **plugin_id:** `analysis_epicurve_classification_v1`
- **kind:** `epidemiology`
- **data_req:** TS
- **deps:** `scipy.signal.find_peaks`
- **core_logic:**
  1. Histogram anomaly onset times
  2. Classify shape: point-source (single peak), continuous (plateau), propagated (waves)
- **novel_insight:** Shape is diagnostic of root cause category without investigation

#### 13–16. Population Genetics (Hardy-Weinberg, Linkage Disequilibrium, FST, Wright-Fisher)

- **13** `analysis_hardy_weinberg_param_coupling_v1` — kind: `ecology`, XS. Chi-square test on parameter combination frequencies vs independence.
- **14** `analysis_linkage_disequilibrium_steps_v1` — kind: `ecology`, DAG. D = freq(AB) − freq(A)×freq(B) for step-anomaly pairs.
- **15** `analysis_fst_machine_differentiation_v1` — kind: `ecology`, XS. Between-group vs total variance of step outcomes across machine groups.
- **16** `analysis_wright_fisher_config_drift_v1` — kind: `ecology`, TS. Track parameter value frequencies across run generations.

#### 17–20. Pharmacokinetics

- **17** `analysis_compartment_redistribution_v1` — kind: `pharmacokinetics`, TS. Multi-compartment ODE model of workload redistribution.
- **18** `analysis_half_life_utilization_v1` — kind: `pharmacokinetics`, TS. t₁/₂ = ln(2)/kₑ for post-peak utilization decay per host.
- **19** `analysis_hill_equation_dose_response_v1` — kind: `pharmacokinetics`, XS. 4-parameter log-logistic fit: parameter → duration.
- **20** `analysis_michaelis_menten_saturation_v1` — kind: `pharmacokinetics`, XS. v = Vmax×[S]/(Km+[S]) for queue_depth → completion_rate.

#### 21–24. Agricultural Science

- **21** `analysis_response_surface_interaction_v1` — kind: `regression`, XS. Quadratic model y = β₀ + Σβᵢxᵢ + Σβᵢᵢxᵢ² + ΣΣβᵢⱼxᵢxⱼ.
- **22** `analysis_job_sequence_cache_rotation_v1` — kind: `chain_makespan`, DAG+TS. Test if job-type ordering on same machine affects duration.
- **23** `analysis_soil_depletion_resource_v1` — kind: `time_series`, TS. First-order decay: dN/dt = −kN + I(t) for resource capacity.
- **24** `analysis_liebig_minimum_constraint_v1` — kind: `capacity_scale_model`, XS. Identify binding constraint resource via Mitscherlich/Liebig.

### Cluster B: Physics & Engineering (Plugins 25–47)

- **25** `analysis_ising_model_correlation_v1` — kind: `physics`, DAG+XS. Binary state interactions on DAG lattice; critical temperature.
- **26** `analysis_percolation_fragility_v1` — kind: `physics`, DAG. Critical step-failure probability for end-to-end breakdown.
- **27** `analysis_renormalization_multiscale_v1` — kind: `physics`, DAG+TS. Coarse-grain DAG; track relevant vs irrelevant parameters per scale.
- **28** `analysis_partition_function_landscape_v1` — kind: `physics`, XS. Enumerate configurations weighted by Boltzmann factors.
- **29** `analysis_reynolds_number_analogue_v1` — kind: `fluid_dynamics`, TS+XS. Re = (throughput × pipeline_depth) / resource_capacity.
- **30** `analysis_navier_stokes_backpressure_v1` — kind: `fluid_dynamics`, DAG+TS. Job flow as fluid; identify backpressure propagation.
- **31** `analysis_vorticity_thrashing_v1` — kind: `fluid_dynamics`, DAG+TS. Detect circular flow patterns (retry storms, thrashing).
- **32** `analysis_entropy_production_rate_v1` — kind: `thermodynamics`, TS. dH/dt of duration distribution; rising = early degradation warning.
- **33** `analysis_carnot_efficiency_bound_v1` — kind: `thermodynamics`, XS. η = 1 − T_cold/T_hot (idle/active time ratio).
- **34** `analysis_exergy_waste_analysis_v1` — kind: `thermodynamics`, DAG+XS. Decompose into useful compute vs waste (retries, idle, context switch).
- **35** `analysis_maxwell_demon_monitoring_v1` — kind: `thermodynamics`, XS+TS. Information cost of scheduling vs entropy reduction.
- **36** `analysis_periodic_schedule_fft_v1` — kind: `materials_science`, TS. FFT on start timestamps; detect hidden periodicities and their defects.
- **37** `analysis_defect_density_classification_v1` — kind: `materials_science`, DAG+XS. Classify anomalies as point/line/planar defects.
- **38** `analysis_stress_strain_capacity_v1` — kind: `materials_science`, TS+XS. Load vs latency curve; identify elastic limit, yield, fracture.
- **39** `analysis_creep_fatigue_prediction_v1` — kind: `materials_science`, TS. Norton's creep + S-N fatigue prediction.
- **40** `analysis_bathtub_curve_lifecycle_v1` — kind: `reliability`, TS. Three-phase hazard model.
- **41** `analysis_fmea_rpn_scoring_v1` — kind: `reliability`, DAG+XS. RPN = Severity × Occurrence × Detection.
- **42** `analysis_fault_tree_minimal_cuts_v1` — kind: `reliability`, DAG. AND/OR fault tree; minimal cut sets.
- **43** `analysis_reliability_block_diagram_v1` — kind: `reliability`, DAG. Series/parallel reliability computation.
- **44** `analysis_pid_controller_analogue_v1` — kind: `control_theory`, TS. P/I/D decomposition of SLA error signal.
- **45** `analysis_lyapunov_stability_v1` — kind: `control_theory`, TS. Stability margin under load perturbation.
- **46** `analysis_bode_frequency_response_v1` — kind: `control_theory`, TS. Transfer function from arrival rate to duration; resonance detection.
- **47** `analysis_controllability_gramian_v1` — kind: `control_theory`, TS+XS. Identify "dark states" not observable from logs.

### Cluster C: Pure Mathematics & Topology (Plugins 48–72)

- **48** `analysis_persistent_homology_regimes_v1` — kind: `topology`, XS/TS. Birth-death pairs across filtration scales.
- **49** `analysis_mapper_subpopulations_v1` — kind: `topology`, XS. Compressed topological graph via lens function.
- **50** `analysis_persistence_landscape_features_v1` — kind: `topology`, XS. ML-compatible vector from persistence diagrams.
- **51** `analysis_vietoris_rips_higher_order_v1` — kind: `topology`, XS/DAG. Higher-order dependencies via simplicial complexes.
- **52** `analysis_euler_characteristic_curve_v1` — kind: `topology`, XS. Cheap topological fingerprint per batch.
- **53** `analysis_sheaf_consistency_v1` — kind: `topology`, DAG. Consistency radius for data across DAG.
- **54** `analysis_functorial_migration_v1` — kind: `topology`, DAG. Schema-preserving data migration.
- **55** `analysis_operad_composition_v1` — kind: `topology`, DAG. Compositional process equivalence.
- **56** `analysis_grothendieck_coverage_v1` — kind: `topology`, DAG. Minimal monitoring configuration.
- **57** `analysis_lyapunov_exponent_chaos_v1` — kind: `chaos`, TS. Positive = chaotic sensitivity.
- **58** `analysis_rqa_recurrence_v1` — kind: `chaos`, TS. Determinism, laminarity, trapping time.
- **59** `analysis_fractal_dimension_attractor_v1` — kind: `chaos`, TS. Correlation dimension of attractor.
- **60** `analysis_takens_embedding_v1` — kind: `chaos`, TS. State space reconstruction from single series.
- **61** `analysis_job_shop_scheduling_bound_v1` — kind: `combinatorial_optimization`, DAG+XS. Theoretical minimum makespan.
- **62** `analysis_bin_packing_utilization_v1` — kind: `combinatorial_optimization`, XS. Optimal job-to-host packing.
- **63** `analysis_csp_binding_rules_v1` — kind: `combinatorial_optimization`, DAG+XS. Identify binding constraint rules.
- **64** `analysis_min_cost_flow_throughput_v1` — kind: `combinatorial_optimization`, DAG. Min-cut = throughput limit.
- **65** `analysis_kolmogorov_complexity_ncd_v1` — kind: `information_theory`, TS/XS. Compression-based anomaly.
- **66** `analysis_renyi_entropy_spectrum_v1` — kind: `information_theory`, XS. Parameterized entropy for avg vs worst case.
- **67** `analysis_pid_synergy_redundancy_v1` — kind: `information_theory`, XS. Unique/redundant/synergistic decomposition.
- **68** `analysis_excess_entropy_memory_v1` — kind: `information_theory`, TS. How far back to look for prediction.
- **69** `analysis_channel_capacity_bound_v1` — kind: `information_theory`, XS. Information-theoretic throughput floor.
- **70** `analysis_spectral_graph_fiedler_v1` — kind: `graph_advanced`, DAG. Algebraic connectivity; bottleneck location.
- **71** `analysis_graph_wavelet_multiscale_v1` — kind: `graph_advanced`, DAG+XS. Multi-scale anomaly on DAG.
- **72** `analysis_cheeger_bottleneck_v1` — kind: `graph_advanced`, DAG. Isoperimetric bottleneck number.

### Cluster D: Social Sciences, Sports & Linguistics (Plugins 73–94)

- **73** `analysis_war_value_above_replacement_v1` — kind: `sabermetrics`, XS.
- **74** `analysis_win_probability_added_v1` — kind: `sabermetrics`, DAG.
- **75** `analysis_leverage_index_dag_v1` — kind: `sabermetrics`, DAG.
- **76** `analysis_pythagorean_expectation_v1` — kind: `sabermetrics`, XS.
- **77** `analysis_zipf_law_frequency_v1` — kind: `linguistics`, XS.
- **78** `analysis_entropy_rate_step_sequences_v1` — kind: `linguistics`, TS.
- **79** `analysis_levenshtein_workflow_dist_v1` — kind: `linguistics`, DAG.
- **80** `analysis_ngram_step_transitions_v1` — kind: `linguistics`, DAG/TS.
- **81** `analysis_weber_fechner_threshold_v1` — kind: `psychophysics`, TS.
- **82** `analysis_stevens_power_law_severity_v1` — kind: `psychophysics`, XS.
- **83** `analysis_signal_detection_dprime_v1` — kind: `psychophysics`, XS.
- **84** `analysis_irt_step_difficulty_v1` — kind: `psychometrics`, XS.
- **85** `analysis_rasch_step_host_scale_v1` — kind: `psychometrics`, XS.
- **86** `analysis_cronbach_alpha_consistency_v1` — kind: `psychometrics`, XS.
- **87** `analysis_structural_holes_broker_v1` — kind: `social_network`, DAG.
- **88** `analysis_weak_ties_bridge_v1` — kind: `social_network`, DAG.
- **89** `analysis_assortativity_scheduling_bias_v1` — kind: `social_network`, DAG.
- **90** `analysis_nash_equilibrium_contention_v1` — kind: `game_theory`, XS.
- **91** `analysis_mechanism_design_scheduling_v1` — kind: `game_theory`, XS.
- **92** `analysis_vickrey_priority_truth_v1` — kind: `game_theory`, XS.
- **93** `analysis_benford_first_digit_v1` — kind: `forensic_accounting`, XS. (Check if overlaps with existing `analysis_benfords_law_anomaly_v1` — if so, skip or extend)
- **94** `analysis_benford_second_digit_v1` — kind: `forensic_accounting`, XS.

### Cluster E: Earth & Space Sciences (Plugins 95–121)

- **95–98**: Astronomy (Lomb-Scargle v2, BLS, pulsar residuals, HR diagram)
- **99–102**: Oceanography (tidal harmonics, directional spectra, thermohaline, Ekman)
- **103–106**: Seismology (Gutenberg-Richter, Omori, ETAS, PSHA)
- **107–110**: Geospatial (Moran's I, kriging, Ripley's K, Getis-Ord)
- **111–114**: Military OR (Lanchester, salvo, Koopman search, OODA)
- **115–118**: Actuarial (chain-ladder, Bühlmann, Lee-Carter, experience rating)
- **119–121**: Climatology (teleconnection, return period, attribution)

### Cluster F: Signal/Music/Network/Queueing (Plugins 122–148)

- **122–126**: Signal processing (cepstral, SST, Wigner-Ville, coherence, matched filter)
- **127–131**: Music theory (onset, tempo, SSM, Markov, polyrhythm)
- **132–136**: Network science (community, motifs, controllability, temporal, k-shell)
- **137–143**: Queueing/telecom (VUT decomposition, fork-join, fluid, diffusion, phase-type, Erlang, CDMA)
- **144–148**: Compiler/CS (CPM, HEFT, dataflow, Amdahl, graph coloring)

### Cluster G: Unconventional Domains (Plugins 149–182)

- **149–151**: Archaeology (seriation, stratigraphy, Harris matrix)
- **152–155**: Evolutionary biology (fitness landscape, punctuated equilibrium, neutral drift, Red Queen)
- **156–158**: Cognitive science (Hick's law, Fitts's law, cognitive load)
- **159–162**: Supply chain (bullwhip, EOQ, TOC, safety stock)
- **163–166**: Semiconductor (Cp/Cpk, Poisson yield, virtual metrology, R2R)
- **167–170**: Library science (TF-IDF, Bradford, Lotka, PageRank)
- **171–173**: Demography (life tables, population pyramids, demographic transition)
- **174–176**: Voting theory (Condorcet, Borda, Arrow/Pareto)
- **177–179**: Sports physiology (HRV, VO₂max, Banister)
- **180–182**: Computational geometry (Voronoi, Delaunay, convex hull)

---

## 9. Post-Plugin Checklist

After implementing each batch of plugins:

```bash
# 1. Run full test suite
python -m pytest -q

# 2. Verify plugin_kind_map.yaml is updated
python -c "
import yaml
from pathlib import Path
data = yaml.safe_load(Path('config/plugin_kind_map.yaml').read_text())
mappings = data.get('mappings', {})
print(f'Total plugins in kind_map: {len(mappings)}')
"

# 3. Verify all thin wrappers exist
for plugin_id in $(grep -oP 'analysis_\w+' config/plugin_kind_map.yaml); do
    if [ ! -f "plugins/${plugin_id}/plugin.py" ]; then
        echo "MISSING: plugins/${plugin_id}/plugin.py"
    fi
done

# 4. Verify all handler IDs are registered
python -c "
from statistic_harness.core.stat_plugins.registry import HANDLERS
print(f'Total handlers registered: {len(HANDLERS)}')
"

# 5. Smoke test a single new plugin
python -c "
from statistic_harness.core.stat_plugins.registry import HANDLERS
assert 'analysis_shannon_diversity_index_v1' in HANDLERS, 'Handler not registered'
print('OK: handler registered')
"
```

---

## 10. Testing Protocol

### Minimal test per addon module

Create `tests/stat_plugins/test_cross_domain_<cluster>.py`:

```python
"""Smoke tests for cross-domain <cluster> plugins."""
import pandas as pd
import numpy as np
from statistic_harness.core.stat_plugins.cross_domain_<cluster> import HANDLERS


def _make_ctx(df):
    """Minimal mock context for handler testing."""
    from pathlib import Path
    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class MockCtx:
        run_id: str = "test"
        run_dir: Path = Path("/tmp/test_run")
        settings: dict = field(default_factory=dict)
        run_seed: int = 42
        budget: dict = field(default_factory=lambda: {
            "row_limit": None, "sampled": False,
            "time_limit_ms": 30000, "cpu_limit_ms": None, "batch_size": None,
        })
        _df: Any = None

        def logger(self, msg): pass
        def dataset_loader(self): return self._df
        def artifacts_dir(self, plugin_id):
            p = self.run_dir / "artifacts" / plugin_id
            p.mkdir(parents=True, exist_ok=True)
            return p

    return MockCtx(_df=df)


def _make_df(n=200):
    """Generate synthetic process log data."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "step_name": rng.choice(["STEP_A", "STEP_B", "STEP_C", "STEP_D"], n),
        "host": rng.choice(["HOST1", "HOST2", "HOST3"], n),
        "duration_sec": rng.exponential(60, n),
        "parent_process_id": [None] + list(range(n - 1)),
        "param_x": rng.uniform(0, 100, n),
        "param_y": rng.choice([0, 1], n),
        "status": rng.choice(["OK", "FAIL"], n, p=[0.9, 0.1]),
    })


def test_all_handlers_registered():
    assert len(HANDLERS) > 0


def test_handlers_return_plugin_result():
    from statistic_harness.core.stat_plugins import BudgetTimer, merge_config, infer_columns
    from statistic_harness.core.stat_plugins.sampling import deterministic_sample

    df = _make_df()
    ctx = _make_ctx(df)
    config = merge_config(ctx.settings)
    inferred = infer_columns(df, config)
    timer = BudgetTimer(30000)
    sample_meta = {"rows_total": len(df), "rows_used": len(df), "sampled": False}

    for plugin_id, handler in list(HANDLERS.items())[:5]:  # test first 5
        result = handler(plugin_id, ctx, df, config, inferred, timer, sample_meta)
        assert result.status in ("ok", "skipped", "degraded", "error", "na"), f"{plugin_id}: bad status {result.status}"
        assert isinstance(result.findings, list), f"{plugin_id}: findings not list"
        assert isinstance(result.metrics, dict), f"{plugin_id}: metrics not dict"
```

---

## 11. Session Log Protocol

At the end of every session that modifies code:

```
### YYYY-MM-DD — [branch-name]
- Added: `src/statistic_harness/core/stat_plugins/cross_domain_<cluster>.py` (N handlers)
- Added: `plugins/analysis_<id>/plugin.py` (thin wrappers × N)
- Modified: `config/plugin_kind_map.yaml` (added N mappings)
- Modified: `src/statistic_harness/core/stat_plugins/registry.py` (imported addon)
- Tests: NNN/NNN passed
- TODO: Next cluster to implement
```

---

## 12. DAG Discovery Pattern

Since every dataset is zero-shot, the addon modules need a robust DAG column discovery function:

```python
def _find_parent_column(df: pd.DataFrame) -> str | None:
    """Dynamically discover the parent process ID column."""
    hints = ("parent", "ppid", "parent_id", "parent_process", "parent_pid",
             "caller", "upstream", "predecessor", "depends_on")
    for col in df.columns:
        col_lower = str(col).lower().replace(" ", "_")
        if any(h in col_lower for h in hints):
            # Verify: should have some nulls (root processes) and some valid refs
            null_frac = float(df[col].isna().mean())
            if 0.01 < null_frac < 0.99:  # has both roots and children
                return str(col)
    return None


def _build_dag(df: pd.DataFrame, parent_col: str, id_col: str | None = None) -> "nx.DiGraph":
    """Build a DAG from parent-child column."""
    import networkx as nx
    G = nx.DiGraph()
    if id_col is None:
        id_col = df.index.name or "index"
        ids = df.index.tolist()
    else:
        ids = df[id_col].tolist()

    for idx, parent in zip(ids, df[parent_col].tolist()):
        G.add_node(idx)
        if pd.notna(parent):
            G.add_edge(parent, idx)
    return G
```

Use this in any DAG-dependent plugin. If `_find_parent_column` returns None, the handler should return `_ok_with_reason(...)` with `"no_parent_column_detected"`.

---

## 13. Deduplication Check

Before implementing, verify these plugins don't overlap with existing ones:

- **#93 Benford's law** — `analysis_benfords_law_anomaly_v1` already exists. Skip #93 or make #94 (second-digit) only.
- **#95 Lomb-Scargle** — `analysis_lomb_scargle_periodogram_v1` already exists. Skip or name v2 for enhanced version.
- **#58 RQA** — `analysis_recurrence_quantification_rqa_v1` already exists. Skip.
- **#133 Network motifs** — `analysis_graph_motif_triads_shift_v1` already exists. Skip or extend.
- **#170 PageRank** — `analysis_graph_pagerank_hotspots_v1` already exists. Skip.
- **#144 Critical path** — check if `analysis_chain_makespan` covers this. If so, extend.

After deduplication, the actual count will be ~175 net new plugins.

---

## 14. Commit Strategy

1. **One addon file per commit** (not one plugin per commit — too many)
2. Each commit includes: addon file + thin wrappers + kind_map update + registry import
3. Run `pytest -q` before each commit
4. Branch naming: `feat/cross-domain-<cluster>` (e.g., `feat/cross-domain-life-sciences`)
5. Merge to main after each cluster passes tests

---

## END OF GUIDE
