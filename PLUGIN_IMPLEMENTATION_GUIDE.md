# Statistics Harness — Plugin Implementation Guide (30 New Plugins)

**D0:** 2026-03-04  
**Purpose:** Self-contained instructions for Claude Code to implement all 30 gap-analysis plugins  
**Source:** `new_plugin_gap_analysis.md`  
**IP Status:** INTERNAL — Statistics Harness is personal IP (Justin Ram / Infoil LLC)

---

## Table of Contents

1. [Pre-Flight Checklist](#1-pre-flight-checklist)
2. [Architecture & Conventions](#2-architecture--conventions)
3. [Dependency Management](#3-dependency-management)
4. [Plugin Scaffold Template](#4-plugin-scaffold-template)
5. [Plugin YAML Template](#5-plugin-yaml-template)
6. [Implementation Order](#6-implementation-order)
7. [Plugin Specifications (1–30)](#7-plugin-specifications-130)
8. [Post-Plugin Checklist](#8-post-plugin-checklist)
9. [Testing Protocol](#9-testing-protocol)
10. [Session Log Protocol](#10-session-log-protocol)

---

## 1. Pre-Flight Checklist

Before starting ANY plugin work:

```bash
# 1. Sync repo — Claude Code updates frequently
git pull origin main

# 2. Verify existing test baseline
python -m pytest -q
# Record pass/fail count as your baseline

# 3. Confirm plugin protocol location
cat src/statistic_harness/core/plugin.py  # or wherever Plugin protocol lives
# You need the exact signature: run(self, ctx: PluginContext) -> PluginResult

# 4. Review existing plugin for pattern reference
ls plugins/ | head -5
# Pick one and read its plugin.py + plugin.yaml as your template

# 5. Check current plugin_kind_map
cat config/plugin_kind_map.yaml | wc -l
# Should show 256+ entries

# 6. Check what's already installed
pip list --format=freeze | grep -iE "econml|causalml|causalpy|mapie|shap|pysindy|scikit-fda|reservoirpy|POT|doubleml|rdrobust|xgboost|river|torch|pytorch-forecasting|bayesian-optimization|aif360|fairlearn|linearmodels|torchdiffeq|shapiq|flexcode|pymatch|compositional"
```

**CRITICAL:** Read `CLAUDE.md` and `AGENTS.md` before starting. They may contain repo-specific conventions that override this guide.

---

## 2. Architecture & Conventions

### File Structure Per Plugin

```
plugins/
└── <plugin_id>/
    ├── plugin.py        # Implementation — must implement Plugin protocol
    ├── plugin.yaml      # Metadata — kind, description, dependencies, column requirements
    └── tests/
        └── test_plugin.py  # At minimum: smoke test with synthetic data
```

### Plugin Protocol (expected signature)

```python
from statistic_harness.core.plugin import Plugin, PluginContext, PluginResult

class AnalysisXxxV1(Plugin):
    """Docstring: what it does, assumptions, references."""

    def run(self, ctx: PluginContext) -> PluginResult:
        # 1. Extract data from ctx
        # 2. Validate inputs
        # 3. Run analysis
        # 4. Return PluginResult
        ...
```

**IMPORTANT:** Before writing any plugin, verify the exact `Plugin`, `PluginContext`, and `PluginResult` interfaces by reading the source. The above is the expected pattern — adjust to match the actual codebase.

### Thin-Wrapper Pattern

Where the method has a well-maintained library (e.g., `doubleml`, `econml`), the plugin should be a **thin wrapper**:

```python
def run(self, ctx: PluginContext) -> PluginResult:
    # Extract
    df = ctx.dataframe
    treatment_col = ctx.params["treatment_col"]
    outcome_col = ctx.params["outcome_col"]
    covariate_cols = ctx.params.get("covariate_cols", self._auto_covariates(df, treatment_col, outcome_col))

    # Validate
    self._validate_columns(df, [treatment_col, outcome_col] + covariate_cols)
    self._validate_no_nulls(df[[treatment_col, outcome_col]])

    # Run (thin wrapper around library)
    <library_call>

    # Return
    return PluginResult(
        plugin_id=self.plugin_id,
        summary={...},
        detail={...},
        recommendations=[...],
        metadata={...}
    )
```

### Naming Conventions

- **Class name:** PascalCase, matches plugin_id. `analysis_double_ml_ate_v1` → `AnalysisDoubleMlAteV1`
- **plugin_id:** Exactly as specified in this guide (snake_case, `_v1` suffix)
- **kind:** Exactly as specified — must match existing kind taxonomy in `plugin_kind_map.yaml`

### Error Handling

Every plugin MUST:

```python
import logging

logger = logging.getLogger(__name__)

class AnalysisXxxV1(Plugin):
    def run(self, ctx: PluginContext) -> PluginResult:
        try:
            # ... core logic ...
        except ValueError as e:
            logger.warning(f"[{self.plugin_id}] Input validation failed: {e}")
            return PluginResult(
                plugin_id=self.plugin_id,
                summary={"status": "skipped", "reason": str(e)},
                detail={},
                recommendations=[],
                metadata={"error": str(e)}
            )
        except Exception as e:
            logger.error(f"[{self.plugin_id}] Unexpected error: {e}", exc_info=True)
            return PluginResult(
                plugin_id=self.plugin_id,
                summary={"status": "error", "reason": str(e)},
                detail={},
                recommendations=[],
                metadata={"error": str(e), "traceback": traceback.format_exc()}
            )
```

### Output Standards

Every `PluginResult.summary` MUST include:

- `status`: `"ok"` | `"skipped"` | `"error"`
- `method`: Human-readable method name
- `n_observations`: Row count used
- `effect_size` or `statistic` (where applicable): The primary numeric result
- `confidence_level` or `alpha`: Significance threshold used
- `interpretation`: One-sentence plain-English summary

Every `PluginResult.detail` SHOULD include:

- Raw numeric outputs (coefficients, p-values, intervals, etc.)
- Any diagnostic statistics (e.g., parallel trends test for DiD)
- Visualization-ready data (x/y arrays for plots)

Every `PluginResult.recommendations` SHOULD include:

- Actionable statements derived from the analysis
- Each recommendation tagged with confidence: `high` / `medium` / `low`

---

## 3. Dependency Management

### New Packages Required

Install in this order (grouped by dependency tree to minimize conflicts):

```bash
# Group 1: Core causal inference (Tier 1 + 2 plugins)
pip install doubleml
pip install econml
pip install causalpy
pip install rdrobust

# Group 2: Explainability + uncertainty
pip install shap
pip install shapiq
pip install mapie

# Group 3: Experiment analysis
pip install causalml

# Group 4: Time series + dynamics
pip install pysindy
pip install scikit-fda
pip install reservoirpy
pip install pytorch-forecasting  # pulls torch if not present

# Group 5: Distribution + drift
pip install POT
pip install river
pip install compositional

# Group 6: Optimization + fairness
pip install bayesian-optimization
pip install fairlearn
pip install xgboost  # likely already installed

# Group 7: Advanced (only if Tier 4 reached)
pip install torchdiffeq
pip install linearmodels
```

### Pre-Install Checks

Before installing any package:

```bash
# Check if already available
python -c "import <package_name>" 2>/dev/null && echo "INSTALLED" || echo "MISSING"
```

### Version Pinning

After installing, record versions:

```bash
pip freeze | grep -iE "doubleml|econml|causalpy|rdrobust|shap|shapiq|mapie|causalml|pysindy|scikit-fda|reservoirpy|pytorch-forecasting|POT|river|compositional|bayesian-optimization|fairlearn|xgboost|torchdiffeq|linearmodels" >> requirements-plugins.txt
```

---

## 4. Plugin Scaffold Template

Use this as the starting point for every plugin. Copy, rename, fill in.

```python
"""
Plugin: <PLUGIN_ID>
Kind: <KIND>
Method: <METHOD_NAME>

<One-paragraph description of what this plugin does.>

References:
    - <Paper/docs link>

Dependencies:
    - <package_name> >= <version>
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

import numpy as np
import pandas as pd

# Library-specific imports
# from <library> import <class>

from statistic_harness.core.plugin import Plugin, PluginContext, PluginResult

logger = logging.getLogger(__name__)


class <ClassName>(Plugin):
    """<Docstring: method, assumptions, limitations.>"""

    plugin_id = "<plugin_id>"
    kind = "<kind>"
    version = "1"

    # --- Configuration defaults ---
    DEFAULT_PARAMS: dict[str, Any] = {
        # "treatment_col": None,   # required — no default
        # "outcome_col": None,     # required — no default
        # "alpha": 0.05,
    }

    # --- Validation helpers ---

    def _validate_inputs(self, ctx: PluginContext) -> dict[str, Any]:
        """Extract and validate all required parameters. Raises ValueError on failure."""
        params = {**self.DEFAULT_PARAMS, **(ctx.params or {})}

        # Example: require treatment_col and outcome_col
        for required in ["treatment_col", "outcome_col"]:
            if not params.get(required):
                raise ValueError(f"Missing required parameter: {required}")

        df = ctx.dataframe
        if df is None or df.empty:
            raise ValueError("Input dataframe is None or empty")

        # Validate columns exist
        for col in [params["treatment_col"], params["outcome_col"]]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe. Available: {list(df.columns)}")

        return params

    def _auto_covariates(self, df: pd.DataFrame, exclude: list[str]) -> list[str]:
        """Auto-select numeric covariate columns, excluding treatment/outcome."""
        return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    # --- Core logic ---

    def run(self, ctx: PluginContext) -> PluginResult:
        try:
            params = self._validate_inputs(ctx)
            df = ctx.dataframe.copy()

            # ── Extract columns ──
            treatment_col = params["treatment_col"]
            outcome_col = params["outcome_col"]
            alpha = params.get("alpha", 0.05)

            # ── Auto-detect covariates if not specified ──
            covariate_cols = params.get("covariate_cols") or self._auto_covariates(
                df, exclude=[treatment_col, outcome_col]
            )

            # ── Drop rows with nulls in analysis columns ──
            analysis_cols = [treatment_col, outcome_col] + covariate_cols
            df_clean = df[analysis_cols].dropna()
            n_dropped = len(df) - len(df_clean)
            if n_dropped > 0:
                logger.info(f"[{self.plugin_id}] Dropped {n_dropped} rows with nulls")

            if len(df_clean) < 30:  # Minimum sample size guard
                raise ValueError(f"Insufficient data after cleaning: {len(df_clean)} rows (need >= 30)")

            # ══════════════════════════════════════════════
            # IMPLEMENTATION GOES HERE
            # Replace this block with library-specific logic
            # ══════════════════════════════════════════════

            result_summary = {}
            result_detail = {}
            recommendations = []

            # ══════════════════════════════════════════════

            return PluginResult(
                plugin_id=self.plugin_id,
                summary={
                    "status": "ok",
                    "method": "<METHOD_NAME>",
                    "n_observations": len(df_clean),
                    "n_dropped": n_dropped,
                    **result_summary,
                },
                detail=result_detail,
                recommendations=recommendations,
                metadata={
                    "params_used": params,
                    "covariate_cols": covariate_cols,
                    "alpha": alpha,
                },
            )

        except ValueError as e:
            logger.warning(f"[{self.plugin_id}] Validation: {e}")
            return PluginResult(
                plugin_id=self.plugin_id,
                summary={"status": "skipped", "reason": str(e)},
                detail={},
                recommendations=[],
                metadata={"error": str(e)},
            )
        except Exception as e:
            logger.error(f"[{self.plugin_id}] Error: {e}", exc_info=True)
            return PluginResult(
                plugin_id=self.plugin_id,
                summary={"status": "error", "reason": str(e)},
                detail={},
                recommendations=[],
                metadata={"error": str(e), "traceback": traceback.format_exc()},
            )
```

---

## 5. Plugin YAML Template

```yaml
# plugins/<plugin_id>/plugin.yaml
plugin_id: <plugin_id>
kind: <kind>
version: "1"
name: "<Human-Readable Method Name>"
description: >
  <One paragraph: what it does, when it applies, key assumptions.>

# Column requirements — used by the planner to match plugins to datasets
requires:
  treatment_col:
    type: binary_or_numeric
    description: "Column indicating treatment/intervention assignment"
  outcome_col:
    type: numeric
    description: "Column containing the outcome metric"

optional:
  covariate_cols:
    type: list[numeric]
    description: "Confounders/controls. Auto-detected if omitted."
  alpha:
    type: float
    default: 0.05
    description: "Significance level"

# Dependencies
dependencies:
  - <package_name>

# Minimum data requirements
constraints:
  min_rows: 30
  requires_panel: false      # true for DiD, synthetic control, etc.
  requires_time_index: false  # true for ITS, time-series plugins

# Tags for planner matching
tags:
  - <tag1>
  - <tag2>
```

---

## 6. Implementation Order

Work in tiers. Do NOT start a new tier until all plugins in the current tier pass tests.

### Tier 1: Immediate (7 plugins)

These directly extend actionability of existing recommendations.

| Order | # | plugin_id | Package | Complexity |
|-------|---|-----------|---------|------------|
| 1.1 | 9 | `analysis_shap_feature_attribution_v1` | `shap` | Low — well-documented API |
| 1.2 | 8 | `analysis_conformal_prediction_interval_v1` | `mapie` | Low — wraps existing models |
| 1.3 | 15 | `analysis_interrupted_time_series_v1` | `causalpy` or `statsmodels` | Medium |
| 1.4 | 1 | `analysis_double_ml_ate_v1` | `doubleml` | Medium |
| 1.5 | 3 | `analysis_diff_in_diff_v1` | `causalpy` | Medium |
| 1.6 | 4 | `analysis_synthetic_control_v1` | `causalpy` | Medium |
| 1.7 | 5 | `analysis_regression_discontinuity_v1` | `causalpy` or `rdrobust` | Medium |

**Checkpoint:** Run full test suite. Update `plugin_kind_map.yaml`. Commit.

### Tier 2: High (7 plugins)

| Order | # | plugin_id | Package | Complexity |
|-------|---|-----------|---------|------------|
| 2.1 | 23 | `analysis_gradient_boosting_importance_v1` | `xgboost` | Low |
| 2.2 | 13 | `analysis_bayesian_ab_test_v1` | `scipy.stats` | Low |
| 2.3 | 14 | `analysis_cuped_variance_reduction_v1` | `numpy`/`scipy` | Low — ~50 lines |
| 2.4 | 29 | `analysis_propensity_score_matching_v1` | `causalml` | Medium |
| 2.5 | 6 | `analysis_inverse_propensity_weighting_v1` | `econml` | Medium |
| 2.6 | 7 | `analysis_meta_learner_cate_v1` | `causalml` | Medium |
| 2.7 | 2 | `analysis_causal_forest_hte_v1` | `econml` | Medium |

**Checkpoint:** Run full test suite. Update `plugin_kind_map.yaml`. Commit.

### Tier 3: Medium (9 plugins)

| Order | # | plugin_id | Package | Complexity |
|-------|---|-----------|---------|------------|
| 3.1 | 12 | `analysis_concept_drift_ddm_v1` | `river` | Low |
| 3.2 | 25 | `analysis_sliced_wasserstein_drift_v1` | `POT` | Low |
| 3.3 | 11 | `analysis_compositional_logratio_v1` | `compositional` | Low |
| 3.4 | 27 | `analysis_instrumental_variables_2sls_v1` | `statsmodels` | Medium |
| 3.5 | 10 | `analysis_shapley_interactions_v1` | `shapiq` | Medium |
| 3.6 | 24 | `analysis_fairness_bias_detection_v1` | `fairlearn` | Medium |
| 3.7 | 17 | `analysis_temporal_fusion_transformer_v1` | `pytorch-forecasting` | High |
| 3.8 | 18 | `analysis_sindy_dynamics_discovery_v1` | `pysindy` | Medium |
| 3.9 | 19 | `analysis_functional_data_analysis_v1` | `scikit-fda` | Medium |

**Checkpoint:** Run full test suite. Update `plugin_kind_map.yaml`. Commit.

### Tier 4: Specialized (7 plugins)

| Order | # | plugin_id | Package | Complexity |
|-------|---|-----------|---------|------------|
| 4.1 | 22 | `analysis_thompson_sampling_bandit_v1` | `scipy.stats` | Low |
| 4.2 | 21 | `analysis_bayesian_optimization_v1` | `bayesian-optimization` | Medium |
| 4.3 | 28 | `analysis_information_bottleneck_v1` | `scipy`/`sklearn` | Medium |
| 4.4 | 30 | `analysis_conditional_density_estimation_v1` | `sklearn` KDE | Medium |
| 4.5 | 20 | `analysis_reservoir_computing_esn_v1` | `reservoirpy` | Medium |
| 4.6 | 16 | `analysis_autoencoder_anomaly_v1` | `torch` | High |
| 4.7 | 26 | `analysis_neural_ode_dynamics_v1` | `torchdiffeq` | High |

**Checkpoint:** Run full test suite. Update `plugin_kind_map.yaml`. Commit.

---

## 7. Plugin Specifications (1–30)

Each spec contains: the exact core logic to insert into the scaffold template, parameter requirements, and test data generation.

---

### Plugin 1: Double/Debiased Machine Learning (DML)

**plugin_id:** `analysis_double_ml_ate_v1`  
**kind:** `causal`  
**Class:** `AnalysisDoubleMlAteV1`  
**Package:** `doubleml`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `treatment_col` | str | Yes | — | Binary treatment column |
| `outcome_col` | str | Yes | — | Numeric outcome column |
| `covariate_cols` | list[str] | No | auto-detect | Confounders |
| `ml_learner` | str | No | `"lasso"` | Base learner: `"lasso"`, `"random_forest"`, `"xgboost"` |
| `n_folds` | int | No | `5` | Number of cross-fitting folds |
| `alpha` | float | No | `0.05` | Significance level |

#### Core Logic

```python
import doubleml as dml
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor

# Map learner name to sklearn estimator
LEARNERS = {
    "lasso": lambda: LassoCV(),
    "random_forest": lambda: RandomForestRegressor(n_estimators=100, max_depth=5, n_jobs=-1),
}

# Build DoubleML data object
dml_data = dml.DoubleMLData(
    df_clean,
    y_col=outcome_col,
    d_cols=treatment_col,
    x_cols=covariate_cols,
)

# Select learner
learner_factory = LEARNERS.get(params.get("ml_learner", "lasso"), LEARNERS["lasso"])
ml_l = learner_factory()  # nuisance: outcome model
ml_m = learner_factory()  # nuisance: treatment model

# Fit Partially Linear Regression (PLR) model
dml_plr = dml.DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m, n_folds=params.get("n_folds", 5))
dml_plr.fit()

# Extract results
coef = dml_plr.coef[0]
se = dml_plr.se[0]
ci = dml_plr.confint(level=1 - alpha).iloc[0]
pval = dml_plr.pval[0]

result_summary = {
    "effect_size": float(coef),
    "std_error": float(se),
    "ci_lower": float(ci["2.5 %"]),
    "ci_upper": float(ci["97.5 %"]),
    "p_value": float(pval),
    "significant": bool(pval < alpha),
    "interpretation": (
        f"Estimated ATE of {treatment_col} on {outcome_col}: {coef:.4f} "
        f"(95% CI: [{ci['2.5 %']:.4f}, {ci['97.5 %']:.4f}], p={pval:.4f}). "
        f"{'Statistically significant.' if pval < alpha else 'Not statistically significant.'}"
    ),
}

result_detail = {
    "coefficient": float(coef),
    "std_error": float(se),
    "t_statistic": float(dml_plr.t_stat[0]),
    "p_value": float(pval),
    "ci_lower": float(ci["2.5 %"]),
    "ci_upper": float(ci["97.5 %"]),
    "n_folds": params.get("n_folds", 5),
    "ml_learner": params.get("ml_learner", "lasso"),
}

recommendations = []
if pval < alpha:
    direction = "increases" if coef > 0 else "decreases"
    recommendations.append({
        "text": f"{treatment_col} {direction} {outcome_col} by ~{abs(coef):.4f} units (causal estimate, DML).",
        "confidence": "high" if pval < 0.01 else "medium",
    })
```

#### Test Data Generator

```python
def make_dml_test_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    T = (rng.normal(0.5 * X1 + 0.3 * X2, 1) > 0).astype(int)
    Y = 2.0 * T + 1.5 * X1 - 0.8 * X2 + rng.normal(0, 1, n)
    return pd.DataFrame({"treatment": T, "outcome": Y, "x1": X1, "x2": X2})
# True ATE = 2.0
```

---

### Plugin 2: Causal Forest (Heterogeneous Treatment Effects)

**plugin_id:** `analysis_causal_forest_hte_v1`  
**kind:** `causal`  
**Class:** `AnalysisCausalForestHteV1`  
**Package:** `econml`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `treatment_col` | str | Yes | — | Binary treatment column |
| `outcome_col` | str | Yes | — | Numeric outcome column |
| `covariate_cols` | list[str] | No | auto-detect | Effect modifiers |
| `n_estimators` | int | No | `200` | Number of trees |
| `alpha` | float | No | `0.05` | For confidence intervals |

#### Core Logic

```python
from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor

Y = df_clean[outcome_col].values
T = df_clean[treatment_col].values
X = df_clean[covariate_cols].values

cf = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100, max_depth=5, n_jobs=-1),
    model_t=LassoCV(),
    n_estimators=params.get("n_estimators", 200),
    random_state=42,
)
cf.fit(Y, T, X=X)

# CATE predictions
cate = cf.effect(X)
cate_intervals = cf.effect_interval(X, alpha=alpha)

# ATE (average of CATE)
ate = float(np.mean(cate))
ate_se = float(np.std(cate) / np.sqrt(len(cate)))

# Feature importance for heterogeneity
feat_imp = cf.feature_importances_

result_summary = {
    "ate": ate,
    "ate_std_error": ate_se,
    "cate_mean": float(np.mean(cate)),
    "cate_median": float(np.median(cate)),
    "cate_std": float(np.std(cate)),
    "cate_min": float(np.min(cate)),
    "cate_max": float(np.max(cate)),
    "interpretation": (
        f"Average treatment effect: {ate:.4f}. "
        f"CATE ranges from {np.min(cate):.4f} to {np.max(cate):.4f}, "
        f"indicating {'substantial' if np.std(cate) > 0.5 * abs(ate) else 'modest'} heterogeneity."
    ),
}

result_detail = {
    "cate_values": cate.tolist(),
    "cate_ci_lower": cate_intervals[0].flatten().tolist(),
    "cate_ci_upper": cate_intervals[1].flatten().tolist(),
    "feature_importances": dict(zip(covariate_cols, feat_imp.tolist())),
    "cate_percentiles": {
        "p10": float(np.percentile(cate, 10)),
        "p25": float(np.percentile(cate, 25)),
        "p50": float(np.percentile(cate, 50)),
        "p75": float(np.percentile(cate, 75)),
        "p90": float(np.percentile(cate, 90)),
    },
}

# Top heterogeneity driver
top_feature = covariate_cols[np.argmax(feat_imp)]
recommendations = [{
    "text": (
        f"Treatment effect varies most along '{top_feature}' "
        f"(importance: {feat_imp[np.argmax(feat_imp)]:.3f}). "
        f"Consider segmenting process changes by this variable."
    ),
    "confidence": "medium",
}]
```

#### Test Data

```python
def make_causal_forest_test_data(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    X1 = rng.normal(0, 1, n)
    X2 = rng.uniform(-1, 1, n)
    T = rng.binomial(1, 0.5, n)
    # Heterogeneous effect: treatment helps when X1 > 0
    tau = 2.0 * (X1 > 0).astype(float) + 0.5
    Y = tau * T + X1 + 0.5 * X2 + rng.normal(0, 0.5, n)
    return pd.DataFrame({"treatment": T, "outcome": Y, "x1": X1, "x2": X2})
```

---

### Plugin 3: Difference-in-Differences (DiD)

**plugin_id:** `analysis_diff_in_diff_v1`  
**kind:** `counterfactual`  
**Class:** `AnalysisDiffInDiffV1`  
**Package:** `causalpy`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `outcome_col` | str | Yes | — | Numeric outcome |
| `time_col` | str | Yes | — | Time period column |
| `group_col` | str | Yes | — | Treated vs control group indicator |
| `intervention_time` | any | Yes | — | Value in `time_col` when treatment starts |
| `covariate_cols` | list[str] | No | `[]` | Additional covariates |

#### Core Logic

```python
import causalpy as cp

# Build formula
formula_parts = [outcome_col, "~", "1"]
# CausalPy expects specific data format — check their DiD API
# Alternative: manual OLS implementation for maximum control

# --- Manual DiD (statsmodels fallback) ---
import statsmodels.formula.api as smf

df_clean["post"] = (df_clean[time_col] >= intervention_time).astype(int)
df_clean["treated"] = df_clean[group_col].astype(int)
df_clean["did_interaction"] = df_clean["post"] * df_clean["treated"]

formula = f"{outcome_col} ~ treated + post + did_interaction"
if covariate_cols:
    formula += " + " + " + ".join(covariate_cols)

model = smf.ols(formula, data=df_clean).fit()

did_coef = model.params["did_interaction"]
did_se = model.bse["did_interaction"]
did_pval = model.pvalues["did_interaction"]
did_ci = model.conf_int(alpha=alpha).loc["did_interaction"]

result_summary = {
    "effect_size": float(did_coef),
    "std_error": float(did_se),
    "p_value": float(did_pval),
    "ci_lower": float(did_ci[0]),
    "ci_upper": float(did_ci[1]),
    "significant": bool(did_pval < alpha),
    "interpretation": (
        f"DiD estimate: {did_coef:.4f} (p={did_pval:.4f}). "
        f"The intervention {'significantly' if did_pval < alpha else 'did not significantly'} "
        f"change {outcome_col}."
    ),
}

result_detail = {
    "full_model_summary": model.summary2().tables[1].to_dict(),
    "r_squared": float(model.rsquared),
    "n_treated_pre": int(((df_clean["treated"] == 1) & (df_clean["post"] == 0)).sum()),
    "n_treated_post": int(((df_clean["treated"] == 1) & (df_clean["post"] == 1)).sum()),
    "n_control_pre": int(((df_clean["treated"] == 0) & (df_clean["post"] == 0)).sum()),
    "n_control_post": int(((df_clean["treated"] == 0) & (df_clean["post"] == 1)).sum()),
    "group_means": {
        "treated_pre": float(df_clean.loc[(df_clean["treated"] == 1) & (df_clean["post"] == 0), outcome_col].mean()),
        "treated_post": float(df_clean.loc[(df_clean["treated"] == 1) & (df_clean["post"] == 1), outcome_col].mean()),
        "control_pre": float(df_clean.loc[(df_clean["treated"] == 0) & (df_clean["post"] == 0), outcome_col].mean()),
        "control_post": float(df_clean.loc[(df_clean["treated"] == 0) & (df_clean["post"] == 1), outcome_col].mean()),
    },
}

recommendations = []
if did_pval < alpha:
    direction = "increased" if did_coef > 0 else "decreased"
    recommendations.append({
        "text": f"The intervention {direction} {outcome_col} by {abs(did_coef):.4f} units relative to controls.",
        "confidence": "high" if did_pval < 0.01 else "medium",
    })
```

#### Special: Parallel Trends Diagnostic

```python
# Add to result_detail: pre-treatment parallel trends test
pre_data = df_clean[df_clean["post"] == 0]
if time_col in pre_data.columns and pre_data[time_col].nunique() > 2:
    pre_formula = f"{outcome_col} ~ C({time_col}) * treated"
    pre_model = smf.ols(pre_formula, data=pre_data).fit()
    interaction_terms = [k for k in pre_model.params.index if ":" in k and "treated" in k]
    parallel_trends_pvals = {k: float(pre_model.pvalues[k]) for k in interaction_terms}
    parallel_trends_pass = all(p > 0.05 for p in parallel_trends_pvals.values())
    result_detail["parallel_trends_test"] = {
        "interaction_pvalues": parallel_trends_pvals,
        "pass": parallel_trends_pass,
    }
    if not parallel_trends_pass:
        recommendations.append({
            "text": "WARNING: Parallel trends assumption may be violated. DiD estimate should be interpreted with caution.",
            "confidence": "low",
        })
```

---

### Plugin 4: Synthetic Control Method

**plugin_id:** `analysis_synthetic_control_v1`  
**kind:** `counterfactual`  
**Class:** `AnalysisSyntheticControlV1`  
**Package:** `causalpy`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `outcome_col` | str | Yes | — | Numeric outcome |
| `time_col` | str | Yes | — | Time index |
| `unit_col` | str | Yes | — | Unit identifier |
| `treated_unit` | any | Yes | — | Value in `unit_col` for treated unit |
| `intervention_time` | any | Yes | — | Time of intervention |

#### Core Logic

```python
import causalpy as cp

# CausalPy synthetic control
# Pivot data to wide format: rows=time, cols=units
wide = df_clean.pivot(index=time_col, columns=unit_col, values=outcome_col)
treated_series = wide[treated_unit]
control_units = [c for c in wide.columns if c != treated_unit]
control_matrix = wide[control_units]

# Pre/post split
pre_mask = wide.index < intervention_time
post_mask = wide.index >= intervention_time

# Fit: find weights that minimize pre-period MSE
from scipy.optimize import minimize

def synth_loss(weights, treated_pre, control_pre):
    synthetic = control_pre @ weights
    return float(np.sum((treated_pre - synthetic) ** 2))

n_controls = len(control_units)
x0 = np.ones(n_controls) / n_controls
bounds = [(0, 1)] * n_controls
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt = minimize(
    synth_loss,
    x0,
    args=(treated_series[pre_mask].values, control_matrix[pre_mask].values),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)

weights = opt.x
synthetic_series = control_matrix.values @ weights

# Effects
pre_fit_mse = float(np.mean((treated_series[pre_mask].values - synthetic_series[pre_mask.values]) ** 2))
post_gaps = treated_series[post_mask].values - synthetic_series[post_mask.values]
avg_effect = float(np.mean(post_gaps))

result_summary = {
    "effect_size": avg_effect,
    "pre_fit_mse": pre_fit_mse,
    "pre_fit_rmse": float(np.sqrt(pre_fit_mse)),
    "post_gap_mean": avg_effect,
    "post_gap_std": float(np.std(post_gaps)),
    "interpretation": (
        f"Synthetic control estimate: average post-intervention gap = {avg_effect:.4f}. "
        f"Pre-period fit RMSE = {np.sqrt(pre_fit_mse):.4f}."
    ),
}

result_detail = {
    "weights": dict(zip(control_units, weights.tolist())),
    "treated_series": treated_series.tolist(),
    "synthetic_series": synthetic_series.tolist(),
    "time_index": wide.index.tolist(),
    "post_gaps": post_gaps.tolist(),
}

# Recommendation if gap is large relative to pre-period noise
if abs(avg_effect) > 2 * np.sqrt(pre_fit_mse):
    direction = "increased" if avg_effect > 0 else "decreased"
    recommendations.append({
        "text": f"Post-intervention, {treated_unit} {direction} by ~{abs(avg_effect):.4f} vs synthetic counterfactual.",
        "confidence": "medium",
    })
```

---

### Plugin 5: Regression Discontinuity Design (RDD)

**plugin_id:** `analysis_regression_discontinuity_v1`  
**kind:** `counterfactual`  
**Class:** `AnalysisRegressionDiscontinuityV1`  
**Package:** `rdrobust` (preferred) or `statsmodels`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `outcome_col` | str | Yes | — | Numeric outcome |
| `running_col` | str | Yes | — | Running/forcing variable |
| `cutoff` | float | No | `0.0` | Cutoff value |
| `bandwidth` | str | No | `"auto"` | `"auto"` uses MSE-optimal; or float |

#### Core Logic

```python
from rdrobust import rdrobust

Y = df_clean[outcome_col].values
X = df_clean[running_col].values
c = params.get("cutoff", 0.0)

rdd_result = rdrobust(Y, X, c=c)

coef = float(rdd_result.coef.iloc[0])  # Conventional estimate
se = float(rdd_result.se.iloc[0])
pval = float(rdd_result.pv.iloc[0])
ci = (float(rdd_result.ci.iloc[0, 0]), float(rdd_result.ci.iloc[0, 1]))
bw = float(rdd_result.bws.iloc[0, 0])  # bandwidth used

result_summary = {
    "effect_size": coef,
    "std_error": se,
    "p_value": pval,
    "ci_lower": ci[0],
    "ci_upper": ci[1],
    "bandwidth": bw,
    "significant": bool(pval < alpha),
    "interpretation": (
        f"RDD estimate at cutoff {c}: effect = {coef:.4f} "
        f"(SE={se:.4f}, p={pval:.4f}). "
        f"Bandwidth: {bw:.4f}."
    ),
}

result_detail = {
    "conventional_estimate": coef,
    "bias_corrected_estimate": float(rdd_result.coef.iloc[1]) if len(rdd_result.coef) > 1 else coef,
    "robust_estimate": float(rdd_result.coef.iloc[2]) if len(rdd_result.coef) > 2 else coef,
    "bandwidth_left": float(rdd_result.bws.iloc[0, 0]),
    "bandwidth_right": float(rdd_result.bws.iloc[1, 0]) if len(rdd_result.bws) > 1 else bw,
    "n_left": int(rdd_result.N_h.iloc[0]),
    "n_right": int(rdd_result.N_h.iloc[1]) if len(rdd_result.N_h) > 1 else 0,
}
```

**Note on `rdrobust`:** The Python API may differ slightly from R. If import fails, fall back to manual local linear regression:

```python
# Fallback: manual RDD via local linear regression
from sklearn.linear_model import LinearRegression

bw = params.get("bandwidth", np.std(X) * 0.5)  # crude default
mask = np.abs(X - c) <= bw
X_local = X[mask].reshape(-1, 1)
Y_local = Y[mask]
T_local = (X[mask] >= c).astype(float).reshape(-1, 1)
X_centered = (X[mask] - c).reshape(-1, 1)

design = np.hstack([T_local, X_centered, T_local * X_centered])
model = LinearRegression().fit(design, Y_local)
coef = float(model.coef_[0])  # treatment effect at cutoff
```

---

### Plugin 6: Inverse Propensity Weighting (IPW)

**plugin_id:** `analysis_inverse_propensity_weighting_v1`  
**kind:** `causal`  
**Class:** `AnalysisInversePropensityWeightingV1`  
**Package:** `econml` (DRLearner)

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `treatment_col` | str | Yes | — | Binary treatment |
| `outcome_col` | str | Yes | — | Numeric outcome |
| `covariate_cols` | list[str] | No | auto-detect | Confounders |
| `trim_threshold` | float | No | `0.05` | Trim extreme propensity scores below this / above 1-this |

#### Core Logic

```python
from sklearn.linear_model import LogisticRegressionCV

Y = df_clean[outcome_col].values
T = df_clean[treatment_col].values.astype(int)
X = df_clean[covariate_cols].values

# Estimate propensity scores
ps_model = LogisticRegressionCV(cv=5, max_iter=1000)
ps_model.fit(X, T)
ps = ps_model.predict_proba(X)[:, 1]

# Trim extreme propensity scores
trim = params.get("trim_threshold", 0.05)
ps_clipped = np.clip(ps, trim, 1 - trim)
n_trimmed = int(np.sum((ps < trim) | (ps > 1 - trim)))

# IPW estimate
weights_treated = T / ps_clipped
weights_control = (1 - T) / (1 - ps_clipped)
ate_ipw = float(np.mean(weights_treated * Y) - np.mean(weights_control * Y))

# Bootstrap SE
n_boot = 500
rng = np.random.default_rng(42)
boot_ates = []
for _ in range(n_boot):
    idx = rng.choice(len(Y), len(Y), replace=True)
    w_t = T[idx] / ps_clipped[idx]
    w_c = (1 - T[idx]) / (1 - ps_clipped[idx])
    boot_ates.append(np.mean(w_t * Y[idx]) - np.mean(w_c * Y[idx]))
boot_se = float(np.std(boot_ates))
ci_lower = float(np.percentile(boot_ates, 100 * alpha / 2))
ci_upper = float(np.percentile(boot_ates, 100 * (1 - alpha / 2)))

result_summary = {
    "effect_size": ate_ipw,
    "std_error": boot_se,
    "ci_lower": ci_lower,
    "ci_upper": ci_upper,
    "n_trimmed": n_trimmed,
    "interpretation": f"IPW ATE estimate: {ate_ipw:.4f} (bootstrap SE: {boot_se:.4f}).",
}

result_detail = {
    "propensity_scores_summary": {
        "mean": float(np.mean(ps)),
        "std": float(np.std(ps)),
        "min": float(np.min(ps)),
        "max": float(np.max(ps)),
        "p5": float(np.percentile(ps, 5)),
        "p95": float(np.percentile(ps, 95)),
    },
    "n_trimmed": n_trimmed,
    "trim_threshold": trim,
    "bootstrap_n": n_boot,
}
```

---

### Plugin 7: Meta-Learners (T/S/X/R-Learner)

**plugin_id:** `analysis_meta_learner_cate_v1`  
**kind:** `causal`  
**Class:** `AnalysisMetaLearnerCateV1`  
**Package:** `causalml`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `treatment_col` | str | Yes | — | Binary treatment |
| `outcome_col` | str | Yes | — | Numeric outcome |
| `covariate_cols` | list[str] | No | auto-detect | Features |
| `learner_type` | str | No | `"x"` | One of: `"t"`, `"s"`, `"x"`, `"r"` |

#### Core Logic

```python
from causalml.inference.meta import (
    BaseTClassifier, BaseSClassifier, BaseXClassifier, BaseRClassifier,
    BaseTRegressor, BaseSRegressor, BaseXRegressor, BaseRRegressor,
)
from sklearn.ensemble import GradientBoostingRegressor

Y = df_clean[outcome_col].values
T = df_clean[treatment_col].values
X_df = df_clean[covariate_cols]

learner_type = params.get("learner_type", "x").lower()
base_model = GradientBoostingRegressor(n_estimators=100, max_depth=4)

REGRESSORS = {
    "t": BaseTRegressor,
    "s": BaseSRegressor,
    "x": BaseXRegressor,
    "r": BaseRRegressor,
}

learner_cls = REGRESSORS.get(learner_type, BaseXRegressor)
learner = learner_cls(learner=base_model)

# Fit and estimate CATE
cate = learner.fit_predict(X_df, treatment=T, y=Y)
# cate is array of shape (n,) or (n, 1)
cate = cate.flatten()

ate = float(np.mean(cate))
result_summary = {
    "ate": ate,
    "cate_std": float(np.std(cate)),
    "cate_min": float(np.min(cate)),
    "cate_max": float(np.max(cate)),
    "learner_type": learner_type,
    "interpretation": f"{learner_type.upper()}-Learner ATE: {ate:.4f}, CATE std: {np.std(cate):.4f}.",
}

result_detail = {
    "cate_values": cate.tolist(),
    "cate_percentiles": {
        f"p{p}": float(np.percentile(cate, p)) for p in [5, 25, 50, 75, 95]
    },
}
```

---

### Plugin 8: Conformal Prediction Intervals (MAPIE)

**plugin_id:** `analysis_conformal_prediction_interval_v1`  
**kind:** `distribution`  
**Class:** `AnalysisConformalPredictionIntervalV1`  
**Package:** `mapie`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `target_col` | str | Yes | — | Numeric target to predict |
| `feature_cols` | list[str] | No | auto-detect | Predictor columns |
| `alpha` | float | No | `0.1` | Miscoverage rate (0.1 = 90% intervals) |
| `method` | str | No | `"plus"` | MAPIE method: `"naive"`, `"base"`, `"plus"`, `"minmax"` |

#### Core Logic

```python
from mapie.regression import MapieRegressor
from sklearn.ensemble import GradientBoostingRegressor

target_col = params["target_col"]
feature_cols = params.get("feature_cols") or self._auto_covariates(df, exclude=[target_col])

X = df_clean[feature_cols].values
y = df_clean[target_col].values

base_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
mapie = MapieRegressor(estimator=base_model, method=params.get("method", "plus"), cv=5)
mapie.fit(X, y)

y_pred, y_intervals = mapie.predict(X, alpha=alpha)
# y_intervals shape: (n, 2, 1) → squeeze
ci_lower = y_intervals[:, 0, 0]
ci_upper = y_intervals[:, 1, 0]
interval_widths = ci_upper - ci_lower

# Empirical coverage
coverage = float(np.mean((y >= ci_lower) & (y <= ci_upper)))

result_summary = {
    "empirical_coverage": coverage,
    "target_coverage": 1 - alpha,
    "mean_interval_width": float(np.mean(interval_widths)),
    "median_interval_width": float(np.median(interval_widths)),
    "interpretation": (
        f"Conformal intervals at {(1-alpha)*100:.0f}% level. "
        f"Empirical coverage: {coverage:.3f}. "
        f"Mean interval width: {np.mean(interval_widths):.4f}."
    ),
}

result_detail = {
    "predictions": y_pred.tolist(),
    "ci_lower": ci_lower.tolist(),
    "ci_upper": ci_upper.tolist(),
    "interval_widths": interval_widths.tolist(),
    "coverage": coverage,
}
```

**Note:** This plugin differs from most — it takes `target_col` + `feature_cols` rather than treatment/outcome. Adjust the scaffold `_validate_inputs` accordingly.

---

### Plugin 9: SHAP Feature Attribution

**plugin_id:** `analysis_shap_feature_attribution_v1`  
**kind:** `role_inference`  
**Class:** `AnalysisShapFeatureAttributionV1`  
**Package:** `shap`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `target_col` | str | Yes | — | Outcome to explain |
| `feature_cols` | list[str] | No | auto-detect | Feature columns |
| `max_samples` | int | No | `500` | Background dataset size for SHAP |

#### Core Logic

```python
import shap
from sklearn.ensemble import GradientBoostingRegressor

target_col = params["target_col"]
feature_cols = params.get("feature_cols") or self._auto_covariates(df, exclude=[target_col])

X = df_clean[feature_cols]
y = df_clean[target_col].values

# Train model
model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
model.fit(X, y)

# SHAP values
max_samples = min(params.get("max_samples", 500), len(X))
explainer = shap.Explainer(model, X.sample(min(100, len(X)), random_state=42))
shap_values = explainer(X.iloc[:max_samples])

# Global feature importance (mean |SHAP|)
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
importance_ranking = sorted(
    zip(feature_cols, mean_abs_shap.tolist()),
    key=lambda x: x[1],
    reverse=True,
)

result_summary = {
    "top_features": [{"feature": f, "mean_abs_shap": v} for f, v in importance_ranking[:10]],
    "model_r2": float(model.score(X, y)),
    "n_explained": max_samples,
    "interpretation": (
        f"Top driver: '{importance_ranking[0][0]}' (mean |SHAP| = {importance_ranking[0][1]:.4f}). "
        f"Model R² = {model.score(X, y):.3f}."
    ),
}

result_detail = {
    "shap_values": shap_values.values.tolist(),
    "base_value": float(shap_values.base_values.mean()),
    "feature_importance_ranking": importance_ranking,
    "feature_cols": feature_cols,
}

recommendations = [{
    "text": f"'{importance_ranking[0][0]}' is the strongest driver of {target_col}. Investigate process controls on this variable first.",
    "confidence": "high" if model.score(X, y) > 0.5 else "medium",
}]
```

---

### Plugin 10: Shapley Interaction Indices

**plugin_id:** `analysis_shapley_interactions_v1`  
**kind:** `role_inference`  
**Class:** `AnalysisShapleyInteractionsV1`  
**Package:** `shapiq`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `target_col` | str | Yes | — | Outcome to explain |
| `feature_cols` | list[str] | No | auto-detect | Feature columns |
| `max_order` | int | No | `2` | Max interaction order |
| `budget` | int | No | `2048` | Evaluation budget |

#### Core Logic

```python
import shapiq
from sklearn.ensemble import GradientBoostingRegressor

target_col = params["target_col"]
feature_cols = params.get("feature_cols") or self._auto_covariates(df, exclude=[target_col])

X = df_clean[feature_cols].values
y = df_clean[target_col].values

model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
model.fit(X, y)

# shapiq explainer
explainer = shapiq.TabularExplainer(
    model=model,
    data=X,
    index="k-SII",  # k-Shapley Interaction Index
    max_order=params.get("max_order", 2),
)

# Explain a representative sample
n_explain = min(100, len(X))
interaction_values = explainer.explain(X[:n_explain], budget=params.get("budget", 2048))

# Extract top pairwise interactions
# interaction_values is an InteractionValues object
# Access depends on shapiq version — inspect attributes
top_interactions = []
# Iterate over interaction indices of order 2
if hasattr(interaction_values, "get_n_order"):
    order2 = interaction_values.get_n_order(2)
    for key, val in sorted(order2.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
        i, j = key
        top_interactions.append({
            "features": [feature_cols[i], feature_cols[j]],
            "interaction_value": float(val),
        })

result_summary = {
    "top_interactions": top_interactions[:5],
    "interpretation": (
        f"Strongest interaction: {top_interactions[0]['features'][0]} × {top_interactions[0]['features'][1]} "
        f"(value: {top_interactions[0]['interaction_value']:.4f})" if top_interactions else "No significant interactions detected."
    ),
}

result_detail = {
    "all_interactions": top_interactions,
    "max_order": params.get("max_order", 2),
}
```

**Note:** The `shapiq` API may change between versions. Read the installed version's API before implementing. Fallback: use `shap.TreeExplainer` with `interaction_contribs=True`.

---

### Plugin 11: Compositional Data Analysis (CoDA / ILR)

**plugin_id:** `analysis_compositional_logratio_v1`  
**kind:** `distribution`  
**Class:** `AnalysisCompositionalLogratioV1`  
**Package:** `compositional`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `component_cols` | list[str] | Yes | — | Columns forming the composition (must sum to ~constant) |
| `transform` | str | No | `"ilr"` | `"ilr"`, `"clr"`, or `"alr"` |
| `outcome_col` | str | No | None | Optional outcome for regression in transformed space |

#### Core Logic

```python
# compositional package or manual implementation
try:
    from compositional import ilr, clr, alr, closure
except ImportError:
    # Manual implementation
    from scipy.special import log_softmax

    def closure(X):
        return X / X.sum(axis=1, keepdims=True)

    def clr(X):
        log_X = np.log(X)
        return log_X - log_X.mean(axis=1, keepdims=True)

    def ilr(X):
        # Simplified ILR using default Helmert basis
        D = X.shape[1]
        clr_X = clr(X)
        # Helmert sub-matrix (D-1 x D)
        basis = np.zeros((D - 1, D))
        for i in range(D - 1):
            basis[i, :i+1] = 1.0 / (i + 1)
            basis[i, i+1] = -(i + 1.0) / (i + 1)
            basis[i] *= np.sqrt((i + 1.0) / (i + 2.0))
        return clr_X @ basis.T

component_cols = params["component_cols"]
transform_type = params.get("transform", "ilr")

comp_data = df_clean[component_cols].values
# Replace zeros with small value (multiplicative replacement)
comp_data = np.where(comp_data == 0, 1e-6, comp_data)
comp_closed = comp_data / comp_data.sum(axis=1, keepdims=True)

TRANSFORMS = {"ilr": ilr, "clr": clr, "alr": lambda X: np.log(X[:, :-1] / X[:, -1:]) }
transformed = TRANSFORMS[transform_type](comp_closed)

result_summary = {
    "transform": transform_type,
    "n_components": len(component_cols),
    "n_transformed_dims": transformed.shape[1],
    "closure_check_max_deviation": float(np.max(np.abs(comp_closed.sum(axis=1) - 1))),
    "interpretation": f"Applied {transform_type.upper()} transform to {len(component_cols)} compositional columns.",
}

result_detail = {
    "transformed_data": transformed.tolist(),
    "component_means": dict(zip(component_cols, comp_closed.mean(axis=0).tolist())),
    "component_stds": dict(zip(component_cols, comp_closed.std(axis=0).tolist())),
}

# If outcome_col provided, do regression in transformed space
outcome_col = params.get("outcome_col")
if outcome_col and outcome_col in df_clean.columns:
    import statsmodels.api as sm
    y = df_clean[outcome_col].values
    X_reg = sm.add_constant(transformed)
    ols = sm.OLS(y, X_reg).fit()
    result_detail["regression_summary"] = {
        "r_squared": float(ols.rsquared),
        "f_pvalue": float(ols.f_pvalue),
        "coefficients": ols.params.tolist(),
    }
```

---

### Plugin 12: Concept Drift Detection (DDM / EDDM)

**plugin_id:** `analysis_concept_drift_ddm_v1`  
**kind:** `changepoint`  
**Class:** `AnalysisConceptDriftDdmV1`  
**Package:** `river`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `error_col` | str | Yes | — | Column of binary errors (1=error, 0=correct) or continuous error magnitudes |
| `method` | str | No | `"ddm"` | `"ddm"` or `"eddm"` |
| `warning_level` | float | No | `2.0` | Std devs for warning zone |
| `drift_level` | float | No | `3.0` | Std devs for drift detection |

#### Core Logic

```python
from river.drift import DDM, EDDM

errors = df_clean[params["error_col"]].values
method = params.get("method", "ddm").lower()

detector = DDM() if method == "ddm" else EDDM()

warnings = []
drifts = []

for i, err in enumerate(errors):
    detector.update(int(err > 0) if not np.issubdtype(type(err), np.integer) else int(err))
    if detector.drift_detected:
        drifts.append(i)
    elif hasattr(detector, "warning_detected") and detector.warning_detected:
        warnings.append(i)

result_summary = {
    "n_drift_points": len(drifts),
    "n_warning_points": len(warnings),
    "first_drift_index": int(drifts[0]) if drifts else None,
    "method": method.upper(),
    "interpretation": (
        f"{method.upper()} detected {len(drifts)} drift point(s) and {len(warnings)} warning(s) "
        f"across {len(errors)} observations."
    ),
}

result_detail = {
    "drift_indices": drifts,
    "warning_indices": warnings,
}

if drifts:
    recommendations.append({
        "text": f"Concept drift detected at index {drifts[0]}. Re-train or re-calibrate models using data from this point forward.",
        "confidence": "high",
    })
```

---

### Plugin 13: Bayesian A/B Testing

**plugin_id:** `analysis_bayesian_ab_test_v1`  
**kind:** `counterfactual`  
**Class:** `AnalysisBayesianAbTestV1`  
**Package:** `scipy.stats` (no extra dependency)

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `outcome_col` | str | Yes | — | Numeric outcome |
| `group_col` | str | Yes | — | A/B group indicator (binary or categorical with 2 levels) |
| `metric_type` | str | No | `"continuous"` | `"continuous"` (Normal) or `"binary"` (Beta-Binomial) |
| `n_samples` | int | No | `10000` | Posterior samples |
| `rope` | float | No | `0.0` | Region of practical equivalence half-width |

#### Core Logic

```python
from scipy import stats

group_col = params["group_col"]
outcome_col = params["outcome_col"]
metric_type = params.get("metric_type", "continuous")
n_samples = params.get("n_samples", 10000)
rope = params.get("rope", 0.0)

groups = df_clean[group_col].unique()
if len(groups) != 2:
    raise ValueError(f"Expected 2 groups, got {len(groups)}: {groups}")

g0, g1 = sorted(groups)
y0 = df_clean.loc[df_clean[group_col] == g0, outcome_col].values
y1 = df_clean.loc[df_clean[group_col] == g1, outcome_col].values

rng = np.random.default_rng(42)

if metric_type == "binary":
    # Beta-Binomial
    s0, n0 = y0.sum(), len(y0)
    s1, n1 = y1.sum(), len(y1)
    post0 = rng.beta(1 + s0, 1 + n0 - s0, n_samples)
    post1 = rng.beta(1 + s1, 1 + n1 - s1, n_samples)
else:
    # Normal-Normal (known variance approximation)
    m0, s0, n0 = y0.mean(), y0.std(ddof=1), len(y0)
    m1, s1, n1 = y1.mean(), y1.std(ddof=1), len(y1)
    post0 = rng.normal(m0, s0 / np.sqrt(n0), n_samples)
    post1 = rng.normal(m1, s1 / np.sqrt(n1), n_samples)

diff = post1 - post0
prob_b_better = float(np.mean(diff > rope))
prob_a_better = float(np.mean(diff < -rope))
expected_loss_b = float(np.mean(np.maximum(-diff, 0)))
expected_loss_a = float(np.mean(np.maximum(diff, 0)))

result_summary = {
    "prob_b_better": prob_b_better,
    "prob_a_better": prob_a_better,
    "expected_loss_choosing_b": expected_loss_b,
    "expected_loss_choosing_a": expected_loss_a,
    "mean_difference": float(np.mean(diff)),
    "ci_95_lower": float(np.percentile(diff, 2.5)),
    "ci_95_upper": float(np.percentile(diff, 97.5)),
    "interpretation": (
        f"P({g1} > {g0}) = {prob_b_better:.3f}. "
        f"Expected loss choosing {g1}: {expected_loss_b:.4f}. "
        f"Mean diff: {np.mean(diff):.4f}."
    ),
}

result_detail = {
    "posterior_diff_samples": diff.tolist()[:1000],  # truncate for storage
    "group_stats": {
        str(g0): {"mean": float(y0.mean()), "std": float(y0.std()), "n": len(y0)},
        str(g1): {"mean": float(y1.mean()), "std": float(y1.std()), "n": len(y1)},
    },
    "rope": rope,
}

winner = g1 if prob_b_better > 0.95 else (g0 if prob_a_better > 0.95 else "inconclusive")
recommendations.append({
    "text": f"Bayesian A/B result: {'choose ' + str(winner) if winner != 'inconclusive' else 'No clear winner — continue collecting data.'}",
    "confidence": "high" if max(prob_b_better, prob_a_better) > 0.95 else "low",
})
```

---

### Plugin 14: CUPED (Controlled Using Pre-Experiment Data)

**plugin_id:** `analysis_cuped_variance_reduction_v1`  
**kind:** `counterfactual`  
**Class:** `AnalysisCupedVarianceReductionV1`  
**Package:** None (numpy/scipy only — ~50 lines)

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `outcome_col` | str | Yes | — | Post-experiment outcome |
| `pre_outcome_col` | str | Yes | — | Pre-experiment outcome (same metric, earlier period) |
| `group_col` | str | Yes | — | A/B group indicator |

#### Core Logic

```python
group_col = params["group_col"]
outcome_col = params["outcome_col"]
pre_col = params["pre_outcome_col"]

Y_post = df_clean[outcome_col].values
Y_pre = df_clean[pre_col].values
T = df_clean[group_col].values

# CUPED: Y_adjusted = Y_post - theta * (Y_pre - mean(Y_pre))
# theta = Cov(Y_post, Y_pre) / Var(Y_pre)
theta = float(np.cov(Y_post, Y_pre)[0, 1] / np.var(Y_pre, ddof=1))
Y_adj = Y_post - theta * (Y_pre - np.mean(Y_pre))

groups = sorted(np.unique(T))
g0, g1 = groups[0], groups[1]

# Unadjusted
mean_diff_raw = float(Y_post[T == g1].mean() - Y_post[T == g0].mean())
se_raw = float(np.sqrt(Y_post[T == g1].var(ddof=1)/np.sum(T == g1) + Y_post[T == g0].var(ddof=1)/np.sum(T == g0)))

# CUPED-adjusted
mean_diff_adj = float(Y_adj[T == g1].mean() - Y_adj[T == g0].mean())
se_adj = float(np.sqrt(Y_adj[T == g1].var(ddof=1)/np.sum(T == g1) + Y_adj[T == g0].var(ddof=1)/np.sum(T == g0)))

variance_reduction = 1 - (se_adj / se_raw) ** 2 if se_raw > 0 else 0.0

result_summary = {
    "effect_size_raw": mean_diff_raw,
    "se_raw": se_raw,
    "effect_size_cuped": mean_diff_adj,
    "se_cuped": se_adj,
    "variance_reduction_pct": float(variance_reduction * 100),
    "theta": theta,
    "interpretation": (
        f"CUPED reduced SE from {se_raw:.4f} to {se_adj:.4f} "
        f"({variance_reduction*100:.1f}% variance reduction). "
        f"Adjusted effect: {mean_diff_adj:.4f}."
    ),
}

result_detail = {
    "theta": theta,
    "correlation_pre_post": float(np.corrcoef(Y_pre, Y_post)[0, 1]),
    "adjusted_values": Y_adj.tolist(),
}
```

---

### Plugin 15: Interrupted Time Series (ITS)

**plugin_id:** `analysis_interrupted_time_series_v1`  
**kind:** `counterfactual`  
**Class:** `AnalysisInterruptedTimeSeriesV1`  
**Package:** `statsmodels`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `outcome_col` | str | Yes | — | Metric time series |
| `time_col` | str | Yes | — | Time index (numeric or datetime) |
| `intervention_time` | any | Yes | — | When intervention occurred |

#### Core Logic

```python
import statsmodels.formula.api as smf

outcome_col = params["outcome_col"]
time_col = params["time_col"]
intervention_time = params["intervention_time"]

# Create ITS variables
df_its = df_clean.sort_values(time_col).copy()
# Numeric time
if pd.api.types.is_datetime64_any_dtype(df_its[time_col]):
    t0 = df_its[time_col].min()
    df_its["_time"] = (df_its[time_col] - t0).dt.total_seconds() / 86400  # days
    intervention_numeric = (pd.Timestamp(intervention_time) - t0).total_seconds() / 86400
else:
    df_its["_time"] = df_its[time_col].astype(float)
    intervention_numeric = float(intervention_time)

df_its["_post"] = (df_its["_time"] >= intervention_numeric).astype(int)
df_its["_time_since_intervention"] = np.maximum(df_its["_time"] - intervention_numeric, 0)

# Segmented regression: Y = b0 + b1*time + b2*post + b3*time_since_intervention
formula = f"{outcome_col} ~ _time + _post + _time_since_intervention"
model = smf.ols(formula, data=df_its).fit()

level_change = float(model.params["_post"])
slope_change = float(model.params["_time_since_intervention"])
level_pval = float(model.pvalues["_post"])
slope_pval = float(model.pvalues["_time_since_intervention"])

result_summary = {
    "level_change": level_change,
    "level_change_pval": level_pval,
    "slope_change": slope_change,
    "slope_change_pval": slope_pval,
    "significant_level": bool(level_pval < alpha),
    "significant_slope": bool(slope_pval < alpha),
    "interpretation": (
        f"ITS: Level change = {level_change:.4f} (p={level_pval:.4f}), "
        f"slope change = {slope_change:.6f} (p={slope_pval:.4f})."
    ),
}

result_detail = {
    "model_params": model.params.to_dict(),
    "model_pvalues": model.pvalues.to_dict(),
    "r_squared": float(model.rsquared),
    "fitted_values": model.fittedvalues.tolist(),
    "residuals": model.resid.tolist(),
    "durbin_watson": float(model.diagn.get("dw", -1)) if hasattr(model, "diagn") else None,
}
```

---

### Plugin 16: Autoencoder Anomaly Detection

**plugin_id:** `analysis_autoencoder_anomaly_v1`  
**kind:** `anomaly`  
**Class:** `AnalysisAutoencoderAnomalyV1`  
**Package:** `torch`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `feature_cols` | list[str] | No | auto-detect numeric | Columns to encode |
| `encoding_dim` | int | No | auto (`max(2, n_features//4)`) | Bottleneck dimension |
| `epochs` | int | No | `50` | Training epochs |
| `threshold_percentile` | float | No | `95` | Reconstruction error percentile for anomaly cutoff |

#### Core Logic

```python
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

feature_cols = params.get("feature_cols") or [c for c in df_clean.select_dtypes(include=[np.number]).columns]
X = df_clean[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.FloatTensor(X_scaled)

n_features = X_scaled.shape[1]
encoding_dim = params.get("encoding_dim", max(2, n_features // 4))
epochs = params.get("epochs", 50)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super().__init__()
        mid = max(encoding_dim + 1, (input_dim + encoding_dim) // 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid), nn.ReLU(),
            nn.Linear(mid, encoding_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, mid), nn.ReLU(),
            nn.Linear(mid, input_dim),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder(n_features, encoding_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = loss_fn(output, X_tensor)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    reconstructed = model(X_tensor).numpy()

recon_errors = np.mean((X_scaled - reconstructed) ** 2, axis=1)
threshold = np.percentile(recon_errors, params.get("threshold_percentile", 95))
anomalies = recon_errors > threshold

result_summary = {
    "n_anomalies": int(anomalies.sum()),
    "anomaly_rate": float(anomalies.mean()),
    "threshold": float(threshold),
    "mean_recon_error": float(recon_errors.mean()),
    "interpretation": (
        f"Autoencoder detected {anomalies.sum()} anomalies ({anomalies.mean()*100:.1f}%) "
        f"at {params.get('threshold_percentile', 95)}th percentile threshold."
    ),
}

result_detail = {
    "reconstruction_errors": recon_errors.tolist(),
    "anomaly_mask": anomalies.tolist(),
    "anomaly_indices": np.where(anomalies)[0].tolist(),
    "threshold": float(threshold),
    "training_loss_final": float(loss.item()),
}
```

---

### Plugin 17: Temporal Fusion Transformer (TFT) Residuals

**plugin_id:** `analysis_temporal_fusion_transformer_v1`  
**kind:** `time_series`  
**Class:** `AnalysisTemporalFusionTransformerV1`  
**Package:** `pytorch-forecasting`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `target_col` | str | Yes | — | Target time series column |
| `time_col` | str | Yes | — | Time index column |
| `group_col` | str | No | None | Group/entity identifier (for panel data) |
| `known_reals` | list[str] | No | `[]` | Known future real covariates |
| `max_encoder_length` | int | No | `30` | Lookback window |
| `max_prediction_length` | int | No | `7` | Forecast horizon |
| `max_epochs` | int | No | `30` | Training epochs |

#### Core Logic

**NOTE:** This is the most complex plugin. `pytorch-forecasting` has specific data format requirements. Read their docs carefully before implementing.

```python
import pytorch_forecasting as ptf
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
import pytorch_lightning as pl

# Prepare data in TimeSeriesDataSet format
# This requires: time_idx (int), group_id, target, and static/dynamic covariates
# Refer to pytorch-forecasting docs for exact formatting

# Simplified skeleton:
df_tft = df_clean.copy()
df_tft["time_idx"] = range(len(df_tft))  # simplification — adjust for real data

training_cutoff = int(len(df_tft) * 0.8)

training = TimeSeriesDataSet(
    df_tft[:training_cutoff],
    time_idx="time_idx",
    target=target_col,
    group_ids=[group_col] if group_col else ["_dummy_group"],
    max_encoder_length=params.get("max_encoder_length", 30),
    max_prediction_length=params.get("max_prediction_length", 7),
    # ... additional config
)

# Train TFT
trainer = pl.Trainer(max_epochs=params.get("max_epochs", 30), accelerator="cpu")
tft = TemporalFusionTransformer.from_dataset(training, learning_rate=0.03)
train_dataloader = training.to_dataloader(train=True, batch_size=64)
trainer.fit(tft, train_dataloaders=train_dataloader)

# Extract variable importance
interpretation = tft.interpret_output(...)
# interpretation contains encoder_variables, decoder_variables importances

# Residual analysis for anomaly flagging
# predictions vs actuals → flag large residuals
```

**CAUTION:** This plugin has heavy dependencies and may fail on limited hardware. Wrap the entire implementation in a try/except that gracefully degrades. Consider making this plugin optional with a `pytorch_forecasting_available` flag.

---

### Plugin 18: SINDy (Sparse Identification of Nonlinear Dynamics)

**plugin_id:** `analysis_sindy_dynamics_discovery_v1`  
**kind:** `time_series`  
**Class:** `AnalysisSindyDynamicsDiscoveryV1`  
**Package:** `pysindy`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `state_cols` | list[str] | Yes | — | Columns representing the system state |
| `time_col` | str | No | None | Time column (if None, assumes uniform spacing) |
| `dt` | float | No | `1.0` | Time step (if `time_col` is None) |
| `threshold` | float | No | `0.1` | Sparsity threshold for STLSQ optimizer |
| `poly_degree` | int | No | `2` | Max polynomial degree in library |

#### Core Logic

```python
import pysindy as ps

state_cols = params["state_cols"]
X = df_clean[state_cols].values

time_col = params.get("time_col")
if time_col and time_col in df_clean.columns:
    t = df_clean[time_col].values.astype(float)
else:
    dt = params.get("dt", 1.0)
    t = np.arange(len(X)) * dt

optimizer = ps.STLSQ(threshold=params.get("threshold", 0.1))
library = ps.PolynomialLibrary(degree=params.get("poly_degree", 2))

model = ps.SINDy(optimizer=optimizer, feature_library=library, feature_names=state_cols)
model.fit(X, t=t)

# Extract discovered equations
equations = model.equations()
coefficients = model.coefficients()
feature_names_lib = model.get_feature_names()

# Model score (R² on derivative prediction)
score = model.score(X, t=t)

result_summary = {
    "equations": equations,
    "model_score_r2": float(score),
    "n_nonzero_terms": int(np.count_nonzero(coefficients)),
    "n_total_terms": int(coefficients.size),
    "sparsity": float(1 - np.count_nonzero(coefficients) / coefficients.size),
    "interpretation": f"Discovered {len(equations)} governing equations with R² = {score:.4f}.",
}

result_detail = {
    "coefficients": coefficients.tolist(),
    "feature_names": feature_names_lib,
    "equations_text": equations,
}

if score > 0.7:
    recommendations.append({
        "text": f"SINDy discovered interpretable dynamics (R²={score:.3f}). Equations: {'; '.join(equations[:3])}",
        "confidence": "medium",
    })
```

---

### Plugin 19: Functional Data Analysis (FDA)

**plugin_id:** `analysis_functional_data_analysis_v1`  
**kind:** `distribution`  
**Class:** `AnalysisFunctionalDataAnalysisV1`  
**Package:** `scikit-fda`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `curve_cols` | list[str] | Yes | — | Ordered columns representing curve values (e.g., `t1, t2, ..., tN`) |
| `n_components` | int | No | `3` | Number of FPCA components |
| `n_clusters` | int | No | `3` | Number of curve clusters |

#### Core Logic

```python
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA
from skfda.ml.clustering import KMeans as FDAKMeans

curve_cols = params["curve_cols"]
curve_data = df_clean[curve_cols].values  # shape: (n_curves, n_points)

# Build FDataGrid
grid_points = np.linspace(0, 1, len(curve_cols))
fd = FDataGrid(data_matrix=curve_data, grid_points=grid_points)

# Functional PCA
n_components = min(params.get("n_components", 3), len(curve_cols) - 1, len(curve_data) - 1)
fpca = FPCA(n_components=n_components)
scores = fpca.fit_transform(fd)
variance_explained = fpca.explained_variance_ratio_

# Functional clustering
n_clusters = params.get("n_clusters", 3)
fda_kmeans = FDAKMeans(n_clusters=n_clusters)
cluster_labels = fda_kmeans.fit_predict(fd)

result_summary = {
    "n_curves": int(curve_data.shape[0]),
    "n_points_per_curve": int(curve_data.shape[1]),
    "fpca_variance_explained": variance_explained.tolist(),
    "fpca_cumulative_variance": float(np.sum(variance_explained)),
    "n_clusters": n_clusters,
    "cluster_sizes": {str(i): int(np.sum(cluster_labels == i)) for i in range(n_clusters)},
    "interpretation": (
        f"FPCA: {n_components} components explain {np.sum(variance_explained)*100:.1f}% of curve variation. "
        f"Clustered into {n_clusters} shape groups."
    ),
}

result_detail = {
    "fpca_scores": scores.tolist(),
    "fpca_variance_ratios": variance_explained.tolist(),
    "cluster_labels": cluster_labels.tolist(),
    "curve_means_by_cluster": {
        str(i): curve_data[cluster_labels == i].mean(axis=0).tolist()
        for i in range(n_clusters)
    },
}
```

---

### Plugin 20: Reservoir Computing / Echo State Network

**plugin_id:** `analysis_reservoir_computing_esn_v1`  
**kind:** `time_series`  
**Class:** `AnalysisReservoirComputingEsnV1`  
**Package:** `reservoirpy`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `target_col` | str | Yes | — | Time series to forecast |
| `feature_cols` | list[str] | No | `[]` | Optional exogenous inputs |
| `reservoir_size` | int | No | `300` | Number of reservoir neurons |
| `spectral_radius` | float | No | `0.9` | Spectral radius of reservoir |
| `forecast_horizon` | int | No | `10` | Steps ahead to predict |
| `train_fraction` | float | No | `0.8` | Train/test split |

#### Core Logic

```python
from reservoirpy.nodes import Reservoir, Ridge

target_col = params["target_col"]
y_full = df_clean[target_col].values.reshape(-1, 1)

reservoir = Reservoir(
    units=params.get("reservoir_size", 300),
    sr=params.get("spectral_radius", 0.9),
    input_scaling=1.0,
    seed=42,
)
readout = Ridge(ridge=1e-6)
esn = reservoir >> readout

# Train/test split
split = int(len(y_full) * params.get("train_fraction", 0.8))
y_train = y_full[:split]
y_test = y_full[split:]

# Fit (teacher forcing)
esn = esn.fit(y_train[:-1], y_train[1:])

# Predict
y_pred = esn.run(y_test[:-1])

# Metrics
mse = float(np.mean((y_test[1:] - y_pred) ** 2))
mae = float(np.mean(np.abs(y_test[1:] - y_pred)))
residuals = (y_test[1:] - y_pred).flatten()

result_summary = {
    "mse": mse,
    "rmse": float(np.sqrt(mse)),
    "mae": mae,
    "n_train": split,
    "n_test": len(y_test),
    "interpretation": f"ESN forecast RMSE: {np.sqrt(mse):.4f}, MAE: {mae:.4f} on {len(y_test)} test points.",
}

result_detail = {
    "predictions": y_pred.flatten().tolist(),
    "actuals": y_test[1:].flatten().tolist(),
    "residuals": residuals.tolist(),
}

# Flag large residuals as anomalies
threshold = 3 * np.std(residuals)
anomaly_mask = np.abs(residuals) > threshold
if anomaly_mask.any():
    recommendations.append({
        "text": f"{anomaly_mask.sum()} forecast violations detected (> 3σ). Investigate these time points for process disruptions.",
        "confidence": "medium",
    })
```

---

### Plugin 21: Bayesian Optimization

**plugin_id:** `analysis_bayesian_optimization_v1`  
**kind:** `chain_makespan`  
**Class:** `AnalysisBayesianOptimizationV1`  
**Package:** `bayesian-optimization`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `objective_col` | str | Yes | — | Column to optimize (maximize by default) |
| `param_cols` | list[str] | Yes | — | Columns to optimize over |
| `param_bounds` | dict | No | auto (min/max) | Bounds per param: `{"col": [low, high]}` |
| `n_iter` | int | No | `25` | Optimization iterations |
| `maximize` | bool | No | `True` | Maximize or minimize |

#### Core Logic

```python
from bayes_opt import BayesianOptimization as BayesOpt
from sklearn.ensemble import GradientBoostingRegressor

objective_col = params["objective_col"]
param_cols = params["param_cols"]
maximize = params.get("maximize", True)

X = df_clean[param_cols].values
y = df_clean[objective_col].values

# Build surrogate from existing data
surrogate = GradientBoostingRegressor(n_estimators=100, random_state=42)
surrogate.fit(X, y)

# Bounds
param_bounds = params.get("param_bounds", {})
pbounds = {}
for i, col in enumerate(param_cols):
    if col in param_bounds:
        pbounds[col] = tuple(param_bounds[col])
    else:
        pbounds[col] = (float(X[:, i].min()), float(X[:, i].max()))

def objective_fn(**kwargs):
    x = np.array([[kwargs[c] for c in param_cols]])
    pred = surrogate.predict(x)[0]
    return pred if maximize else -pred

optimizer = BayesOpt(f=objective_fn, pbounds=pbounds, random_state=42, verbose=0)
# Seed with existing observations (subsample)
n_seed = min(20, len(X))
for i in range(n_seed):
    try:
        optimizer.probe(params={col: float(X[i, j]) for j, col in enumerate(param_cols)}, lazy=True)
    except Exception:
        pass
optimizer.maximize(init_points=5, n_iter=params.get("n_iter", 25))

best = optimizer.max
best_params = best["params"]
best_value = best["target"] if maximize else -best["target"]

result_summary = {
    "best_value": float(best_value),
    "best_params": {k: float(v) for k, v in best_params.items()},
    "n_iterations": len(optimizer.res),
    "interpretation": f"Optimal {objective_col}: {best_value:.4f} at params {best_params}.",
}

result_detail = {
    "optimization_history": [{"params": r["params"], "target": float(r["target"])} for r in optimizer.res],
    "bounds": pbounds,
}

recommendations.append({
    "text": f"Recommended parameter settings: {', '.join(f'{k}={v:.3f}' for k, v in best_params.items())}.",
    "confidence": "medium",
})
```

---

### Plugin 22: Multi-Armed Bandit / Thompson Sampling

**plugin_id:** `analysis_thompson_sampling_bandit_v1`  
**kind:** `chain_makespan`  
**Class:** `AnalysisThompsonSamplingBanditV1`  
**Package:** `scipy.stats` (no extra deps)

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `outcome_col` | str | Yes | — | Binary success/failure or continuous reward |
| `arm_col` | str | Yes | — | Column identifying which arm/variant was played |
| `metric_type` | str | No | `"binary"` | `"binary"` or `"continuous"` |
| `n_simulations` | int | No | `10000` | Thompson sampling simulations |

#### Core Logic

```python
from scipy import stats

arm_col = params["arm_col"]
outcome_col = params["outcome_col"]
metric_type = params.get("metric_type", "binary")
n_sim = params.get("n_simulations", 10000)

arms = df_clean[arm_col].unique()
rng = np.random.default_rng(42)

arm_stats = {}
for arm in arms:
    arm_data = df_clean.loc[df_clean[arm_col] == arm, outcome_col].values
    if metric_type == "binary":
        successes = int(arm_data.sum())
        failures = int(len(arm_data) - successes)
        arm_stats[str(arm)] = {"alpha": 1 + successes, "beta": 1 + failures, "n": len(arm_data)}
    else:
        arm_stats[str(arm)] = {"mean": float(arm_data.mean()), "std": float(arm_data.std(ddof=1)), "n": len(arm_data)}

# Thompson sampling simulation
best_arm_counts = {str(arm): 0 for arm in arms}
for _ in range(n_sim):
    samples = {}
    for arm in arms:
        s = arm_stats[str(arm)]
        if metric_type == "binary":
            samples[str(arm)] = rng.beta(s["alpha"], s["beta"])
        else:
            samples[str(arm)] = rng.normal(s["mean"], s["std"] / np.sqrt(s["n"]))
    best = max(samples, key=samples.get)
    best_arm_counts[best] += 1

prob_best = {k: v / n_sim for k, v in best_arm_counts.items()}
recommended_arm = max(prob_best, key=prob_best.get)

result_summary = {
    "probability_best_arm": prob_best,
    "recommended_arm": recommended_arm,
    "recommendation_confidence": float(prob_best[recommended_arm]),
    "interpretation": f"Arm '{recommended_arm}' is best with probability {prob_best[recommended_arm]:.3f}.",
}

result_detail = {
    "arm_statistics": arm_stats,
    "simulation_counts": best_arm_counts,
    "n_simulations": n_sim,
}

recommendations.append({
    "text": f"Allocate more volume to arm '{recommended_arm}' (P(best) = {prob_best[recommended_arm]:.3f}).",
    "confidence": "high" if prob_best[recommended_arm] > 0.9 else "medium",
})
```

---

### Plugin 23: XGBoost Feature Importance Ensemble

**plugin_id:** `analysis_gradient_boosting_importance_v1`  
**kind:** `role_inference`  
**Class:** `AnalysisGradientBoostingImportanceV1`  
**Package:** `xgboost`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `target_col` | str | Yes | — | Outcome column |
| `feature_cols` | list[str] | No | auto-detect | Feature columns |
| `importance_type` | str | No | `"gain"` | `"gain"`, `"weight"`, `"cover"` |

#### Core Logic

```python
import xgboost as xgb

target_col = params["target_col"]
feature_cols = params.get("feature_cols") or self._auto_covariates(df, exclude=[target_col])

X = df_clean[feature_cols]
y = df_clean[target_col]

model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
model.fit(X, y)

importance_type = params.get("importance_type", "gain")
importance = model.get_booster().get_score(importance_type=importance_type)
# Map feature names (xgboost uses f0, f1, ... by default)
importance_mapped = {}
for k, v in importance.items():
    idx = int(k.replace("f", "")) if k.startswith("f") and k[1:].isdigit() else None
    if idx is not None and idx < len(feature_cols):
        importance_mapped[feature_cols[idx]] = v
    else:
        importance_mapped[k] = v

ranking = sorted(importance_mapped.items(), key=lambda x: x[1], reverse=True)

result_summary = {
    "top_features": [{"feature": f, importance_type: float(v)} for f, v in ranking[:10]],
    "model_r2": float(model.score(X, y)),
    "importance_type": importance_type,
    "interpretation": f"Top driver: '{ranking[0][0]}' ({importance_type}={ranking[0][1]:.2f}). Model R²={model.score(X, y):.3f}.",
}

result_detail = {
    "full_importance_ranking": ranking,
    "n_features": len(feature_cols),
}
```

---

### Plugin 24: Fairness / Bias Detection

**plugin_id:** `analysis_fairness_bias_detection_v1`  
**kind:** `role_inference`  
**Class:** `AnalysisFairnessBiasDetectionV1`  
**Package:** `fairlearn`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `outcome_col` | str | Yes | — | Binary or numeric outcome |
| `sensitive_col` | str | Yes | — | Protected attribute column |
| `prediction_col` | str | No | None | Model predictions (if evaluating a model) |

#### Core Logic

```python
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame,
)
from sklearn.metrics import accuracy_score, mean_absolute_error

outcome_col = params["outcome_col"]
sensitive_col = params["sensitive_col"]

y_true = df_clean[outcome_col].values
sensitive = df_clean[sensitive_col].values

# If no prediction_col, use outcome itself (outcome bias)
if params.get("prediction_col") and params["prediction_col"] in df_clean.columns:
    y_pred = df_clean[params["prediction_col"]].values
else:
    y_pred = y_true  # Measuring outcome disparity

is_binary = set(np.unique(y_true)).issubset({0, 1})

metrics = {}
if is_binary:
    metrics["demographic_parity_diff"] = float(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive))
    try:
        metrics["equalized_odds_diff"] = float(equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive))
    except Exception:
        metrics["equalized_odds_diff"] = None

# Group-level stats
groups = np.unique(sensitive)
group_stats = {}
for g in groups:
    mask = sensitive == g
    group_stats[str(g)] = {
        "n": int(mask.sum()),
        "mean_outcome": float(y_true[mask].mean()),
        "std_outcome": float(y_true[mask].std()),
    }

# Disparate impact ratio
means = [group_stats[str(g)]["mean_outcome"] for g in groups]
if max(means) > 0:
    disparate_impact = float(min(means) / max(means))
else:
    disparate_impact = None

metrics["disparate_impact_ratio"] = disparate_impact

result_summary = {
    **metrics,
    "group_stats": group_stats,
    "interpretation": (
        f"Disparate impact ratio: {disparate_impact:.3f} "
        f"({'PASS (≥0.8)' if disparate_impact and disparate_impact >= 0.8 else 'FAIL (<0.8)'}). "
        if disparate_impact else "Could not compute disparate impact."
    ),
}

result_detail = {
    "metrics": metrics,
    "group_statistics": group_stats,
    "n_groups": len(groups),
    "groups": [str(g) for g in groups],
}

if disparate_impact and disparate_impact < 0.8:
    recommendations.append({
        "text": f"Disparate impact detected (ratio={disparate_impact:.3f} < 0.8). Investigate outcome differences across {sensitive_col} groups.",
        "confidence": "high",
    })
```

---

### Plugin 25: Sliced-Wasserstein Distance

**plugin_id:** `analysis_sliced_wasserstein_drift_v1`  
**kind:** `distribution`  
**Class:** `AnalysisSlicedWassersteinDriftV1`  
**Package:** `POT`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `feature_cols` | list[str] | No | auto-detect numeric | Columns to compare |
| `split_col` | str | No | None | Column to split into two distributions |
| `split_value` | any | No | None | Value to split on (if split_col provided) |
| `reference_fraction` | float | No | `0.5` | Fraction of data as reference (if no split_col) |
| `n_projections` | int | No | `100` | Number of random projections |

#### Core Logic

```python
import ot

feature_cols = params.get("feature_cols") or [c for c in df_clean.select_dtypes(include=[np.number]).columns]

if params.get("split_col") and params.get("split_value") is not None:
    mask = df_clean[params["split_col"]] == params["split_value"]
    X_ref = df_clean.loc[~mask, feature_cols].values
    X_test = df_clean.loc[mask, feature_cols].values
else:
    split = int(len(df_clean) * params.get("reference_fraction", 0.5))
    X_ref = df_clean.iloc[:split][feature_cols].values
    X_test = df_clean.iloc[split:][feature_cols].values

n_proj = params.get("n_projections", 100)

# Sliced Wasserstein
swd = ot.sliced_wasserstein_distance(X_ref, X_test, n_projections=n_proj)

# Per-feature marginal Wasserstein (1D) for interpretability
marginal_distances = {}
for i, col in enumerate(feature_cols):
    w1 = float(ot.wasserstein_1d(X_ref[:, i], X_test[:, i]))
    marginal_distances[col] = w1

result_summary = {
    "sliced_wasserstein_distance": float(swd),
    "n_projections": n_proj,
    "n_reference": len(X_ref),
    "n_test": len(X_test),
    "top_drifting_features": sorted(marginal_distances.items(), key=lambda x: x[1], reverse=True)[:5],
    "interpretation": f"Sliced-Wasserstein distance: {swd:.4f} across {len(feature_cols)} features.",
}

result_detail = {
    "marginal_wasserstein_1d": marginal_distances,
    "sliced_wasserstein": float(swd),
}
```

---

### Plugin 26: Neural ODE for Time Series

**plugin_id:** `analysis_neural_ode_dynamics_v1`  
**kind:** `time_series`  
**Class:** `AnalysisNeuralOdeDynamicsV1`  
**Package:** `torchdiffeq`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `state_cols` | list[str] | Yes | — | State variables |
| `time_col` | str | No | None | Time column (irregular spacing OK) |
| `hidden_dim` | int | No | `64` | Hidden layer dimension |
| `epochs` | int | No | `100` | Training epochs |

#### Core Logic

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint

state_cols = params["state_cols"]
X = df_clean[state_cols].values

if params.get("time_col") and params["time_col"] in df_clean.columns:
    t = torch.FloatTensor(df_clean[params["time_col"]].values.astype(float))
else:
    t = torch.linspace(0, 1, len(X))

X_tensor = torch.FloatTensor(X)
dim = X.shape[1]
hidden_dim = params.get("hidden_dim", 64)

class ODEFunc(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )
    def forward(self, t, y):
        return self.net(y)

func = ODEFunc(dim, hidden_dim)
optimizer = torch.optim.Adam(func.parameters(), lr=1e-3)

# Training loop
epochs = params.get("epochs", 100)
for epoch in range(epochs):
    optimizer.zero_grad()
    pred = odeint(func, X_tensor[0], t)
    loss = torch.mean((pred - X_tensor) ** 2)
    loss.backward()
    optimizer.step()

# Final predictions
with torch.no_grad():
    pred_final = odeint(func, X_tensor[0], t).numpy()

mse = float(np.mean((X - pred_final) ** 2))
residuals = X - pred_final

result_summary = {
    "mse": mse,
    "rmse": float(np.sqrt(mse)),
    "training_loss_final": float(loss.item()),
    "interpretation": f"Neural ODE fit RMSE: {np.sqrt(mse):.4f} across {dim} state variables.",
}

result_detail = {
    "predictions": pred_final.tolist(),
    "residuals": residuals.tolist(),
    "per_variable_rmse": {col: float(np.sqrt(np.mean(residuals[:, i]**2))) for i, col in enumerate(state_cols)},
}
```

---

### Plugin 27: Instrumental Variables / 2SLS

**plugin_id:** `analysis_instrumental_variables_2sls_v1`  
**kind:** `causal`  
**Class:** `AnalysisInstrumentalVariables2slsV1`  
**Package:** `statsmodels` (already installed) or `linearmodels`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `outcome_col` | str | Yes | — | Dependent variable |
| `treatment_col` | str | Yes | — | Endogenous regressor |
| `instrument_cols` | list[str] | Yes | — | Instruments |
| `covariate_cols` | list[str] | No | `[]` | Exogenous controls |

#### Core Logic

```python
try:
    from linearmodels.iv import IV2SLS as LM_IV2SLS
    use_linearmodels = True
except ImportError:
    use_linearmodels = False

import statsmodels.api as sm

outcome_col = params["outcome_col"]
treatment_col = params["treatment_col"]
instrument_cols = params["instrument_cols"]
covariate_cols = params.get("covariate_cols", [])

y = df_clean[outcome_col]
endog = df_clean[[treatment_col]]
exog = sm.add_constant(df_clean[covariate_cols]) if covariate_cols else sm.add_constant(pd.DataFrame(index=df_clean.index))
instruments = df_clean[instrument_cols]

if use_linearmodels:
    model = LM_IV2SLS(y, exog, endog, instruments).fit()
    coef = float(model.params[treatment_col])
    se = float(model.std_errors[treatment_col])
    pval = float(model.pvalues[treatment_col])
    ci = (float(model.conf_int().loc[treatment_col, "lower"]), float(model.conf_int().loc[treatment_col, "upper"]))
    first_stage_f = float(model.first_stage.diagnostics["f.stat"].iloc[0]) if hasattr(model, "first_stage") else None
else:
    # Statsmodels IV2SLS
    from statsmodels.sandbox.regression.gmm import IV2SLS
    instruments_full = pd.concat([exog, instruments], axis=1)
    endog_with_controls = pd.concat([endog, exog], axis=1)
    model = IV2SLS(y, endog_with_controls, instruments_full).fit()
    coef = float(model.params.iloc[0])
    se = float(model.bse.iloc[0])
    pval = float(model.pvalues.iloc[0])
    ci = (float(model.conf_int().iloc[0, 0]), float(model.conf_int().iloc[0, 1]))
    first_stage_f = None

result_summary = {
    "effect_size": coef,
    "std_error": se,
    "p_value": pval,
    "ci_lower": ci[0],
    "ci_upper": ci[1],
    "first_stage_f_stat": first_stage_f,
    "weak_instrument_warning": bool(first_stage_f is not None and first_stage_f < 10),
    "interpretation": (
        f"IV/2SLS estimate: {coef:.4f} (SE={se:.4f}, p={pval:.4f}). "
        f"{'WEAK INSTRUMENT WARNING (F < 10). ' if first_stage_f and first_stage_f < 10 else ''}"
    ),
}

if first_stage_f and first_stage_f < 10:
    recommendations.append({
        "text": "Weak instrument detected (F < 10). IV estimates may be unreliable. Consider stronger instruments.",
        "confidence": "low",
    })
```

---

### Plugin 28: Information Bottleneck Method

**plugin_id:** `analysis_information_bottleneck_v1`  
**kind:** `cluster`  
**Class:** `AnalysisInformationBottleneckV1`  
**Package:** `sklearn` + `scipy` (no extra deps)

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `target_col` | str | Yes | — | Target variable |
| `feature_cols` | list[str] | No | auto-detect | Feature columns |
| `n_clusters` | int | No | `5` | Number of compressed clusters |
| `beta` | float | No | `1.0` | Trade-off parameter (higher = more information retained) |

#### Core Logic

```python
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer

target_col = params["target_col"]
feature_cols = params.get("feature_cols") or self._auto_covariates(df, exclude=[target_col])
n_clusters = params.get("n_clusters", 5)

X = df_clean[feature_cols].values
y = df_clean[target_col].values

# Discretize target for MI computation
kbd = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
y_disc = kbd.fit_transform(y.reshape(-1, 1)).flatten().astype(int)

# Cluster features into compressed representation
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
T = kmeans.fit_predict(X)  # compressed representation

# I(T; Y) — information about target retained
mi_ty = normalized_mutual_info_score(T, y_disc)

# I(T; X) — compression level (via proxy)
# Approximate as cluster assignment entropy
from collections import Counter
counts = Counter(T)
probs = np.array([counts[i] / len(T) for i in range(n_clusters)])
entropy_T = float(-np.sum(probs * np.log(probs + 1e-10)))

# Per-feature MI with target
feature_mi = {}
for i, col in enumerate(feature_cols):
    x_disc = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile").fit_transform(X[:, i:i+1]).flatten().astype(int)
    feature_mi[col] = float(normalized_mutual_info_score(x_disc, y_disc))

feature_mi_ranked = sorted(feature_mi.items(), key=lambda x: x[1], reverse=True)

result_summary = {
    "compression_mi_ty": float(mi_ty),
    "compression_entropy": entropy_T,
    "n_clusters": n_clusters,
    "top_informative_features": feature_mi_ranked[:10],
    "interpretation": (
        f"Information bottleneck: {n_clusters} clusters retain NMI={mi_ty:.3f} about {target_col}. "
        f"Top feature: '{feature_mi_ranked[0][0]}' (NMI={feature_mi_ranked[0][1]:.3f})."
    ),
}

result_detail = {
    "cluster_labels": T.tolist(),
    "feature_mutual_information": feature_mi_ranked,
    "cluster_sizes": dict(counts),
}
```

---

### Plugin 29: Propensity Score Matching

**plugin_id:** `analysis_propensity_score_matching_v1`  
**kind:** `causal`  
**Class:** `AnalysisPropensityScoreMatchingV1`  
**Package:** `causalml`

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `treatment_col` | str | Yes | — | Binary treatment |
| `outcome_col` | str | Yes | — | Numeric outcome |
| `covariate_cols` | list[str] | No | auto-detect | Matching covariates |
| `n_neighbors` | int | No | `1` | Number of matches per treated unit |
| `caliper` | float | No | `0.2` | Max propensity score distance for matches |

#### Core Logic

```python
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import NearestNeighbors

treatment_col = params["treatment_col"]
outcome_col = params["outcome_col"]
covariate_cols = params.get("covariate_cols") or self._auto_covariates(df, exclude=[treatment_col, outcome_col])

T = df_clean[treatment_col].values.astype(int)
Y = df_clean[outcome_col].values
X = df_clean[covariate_cols].values

# Propensity score
ps_model = LogisticRegressionCV(cv=5, max_iter=1000)
ps_model.fit(X, T)
ps = ps_model.predict_proba(X)[:, 1]

# Match treated to control
n_neighbors = params.get("n_neighbors", 1)
caliper = params.get("caliper", 0.2)

treated_idx = np.where(T == 1)[0]
control_idx = np.where(T == 0)[0]

nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
nn.fit(ps[control_idx].reshape(-1, 1))
distances, indices = nn.kneighbors(ps[treated_idx].reshape(-1, 1))

# Apply caliper
valid_matches = distances[:, 0] <= caliper * np.std(ps)
matched_treated = treated_idx[valid_matches]
matched_control = control_idx[indices[valid_matches, 0]]

n_matched = len(matched_treated)
n_dropped = len(treated_idx) - n_matched

# ATT on matched sample
att = float(Y[matched_treated].mean() - Y[matched_control].mean())
# SE via bootstrap
rng = np.random.default_rng(42)
boot_atts = []
for _ in range(500):
    idx = rng.choice(n_matched, n_matched, replace=True)
    boot_atts.append(Y[matched_treated[idx]].mean() - Y[matched_control[idx]].mean())
se = float(np.std(boot_atts))

result_summary = {
    "att": att,
    "std_error": se,
    "ci_lower": float(np.percentile(boot_atts, 2.5)),
    "ci_upper": float(np.percentile(boot_atts, 97.5)),
    "n_matched": n_matched,
    "n_dropped_caliper": n_dropped,
    "interpretation": f"PSM ATT: {att:.4f} (SE={se:.4f}) on {n_matched} matched pairs.",
}

result_detail = {
    "propensity_score_stats": {
        "treated_mean": float(ps[T == 1].mean()),
        "control_mean": float(ps[T == 0].mean()),
        "overlap": float(np.mean((ps > 0.1) & (ps < 0.9))),
    },
    "balance_after_matching": {},  # could compute standardized mean differences
}
```

---

### Plugin 30: Conditional Density Estimation

**plugin_id:** `analysis_conditional_density_estimation_v1`  
**kind:** `distribution`  
**Class:** `AnalysisConditionalDensityEstimationV1`  
**Package:** `sklearn` (KDE wrapper — no extra deps needed for fallback)

#### Parameters

| Param | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `target_col` | str | Yes | — | Response variable |
| `feature_cols` | list[str] | No | auto-detect | Conditioning variables |
| `n_grid` | int | No | `100` | Grid points for density evaluation |
| `bandwidth` | str | No | `"auto"` | KDE bandwidth: `"auto"`, `"scott"`, or float |

#### Core Logic

```python
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

target_col = params["target_col"]
feature_cols = params.get("feature_cols") or self._auto_covariates(df, exclude=[target_col])

y = df_clean[target_col].values
X = df_clean[feature_cols].values

# Grid for density evaluation
y_grid = np.linspace(y.min(), y.max(), params.get("n_grid", 100))

# Conditional density via KDE on (X, Y) jointly, then normalize
# Simplified approach: bin X, compute KDE of Y within each bin
from sklearn.cluster import KMeans

n_bins = min(10, len(X) // 20)
kmeans = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
x_bins = kmeans.fit_predict(X)

conditional_densities = {}
for b in range(n_bins):
    y_bin = y[x_bins == b]
    if len(y_bin) < 5:
        continue
    bw = params.get("bandwidth", "auto")
    if bw == "auto" or bw == "scott":
        bw_val = 1.06 * np.std(y_bin) * len(y_bin) ** (-1/5)
    else:
        bw_val = float(bw)
    kde = KernelDensity(bandwidth=max(bw_val, 0.01), kernel="gaussian")
    kde.fit(y_bin.reshape(-1, 1))
    log_density = kde.score_samples(y_grid.reshape(-1, 1))
    conditional_densities[int(b)] = {
        "density": np.exp(log_density).tolist(),
        "mean": float(y_bin.mean()),
        "std": float(y_bin.std()),
        "n": int(len(y_bin)),
        "percentiles": {
            "p5": float(np.percentile(y_bin, 5)),
            "p25": float(np.percentile(y_bin, 25)),
            "p50": float(np.percentile(y_bin, 50)),
            "p75": float(np.percentile(y_bin, 75)),
            "p95": float(np.percentile(y_bin, 95)),
        },
    }

# Unconditional density
kde_full = KernelDensity(bandwidth=1.06 * np.std(y) * len(y) ** (-1/5), kernel="gaussian")
kde_full.fit(y.reshape(-1, 1))
unconditional_density = np.exp(kde_full.score_samples(y_grid.reshape(-1, 1)))

result_summary = {
    "n_conditional_bins": len(conditional_densities),
    "unconditional_mean": float(y.mean()),
    "unconditional_std": float(y.std()),
    "max_conditional_std": max(d["std"] for d in conditional_densities.values()),
    "min_conditional_std": min(d["std"] for d in conditional_densities.values()),
    "interpretation": (
        f"Conditional density of {target_col} estimated across {len(conditional_densities)} feature bins. "
        f"Conditional std ranges from {min(d['std'] for d in conditional_densities.values()):.4f} to "
        f"{max(d['std'] for d in conditional_densities.values()):.4f} (unconditional: {y.std():.4f})."
    ),
}

result_detail = {
    "y_grid": y_grid.tolist(),
    "unconditional_density": unconditional_density.tolist(),
    "conditional_densities": conditional_densities,
}
```

---

## 8. Post-Plugin Checklist

After implementing EACH plugin:

```bash
# 1. Run the plugin's own test
python -m pytest plugins/<plugin_id>/tests/test_plugin.py -v

# 2. Run the full suite
python -m pytest -q

# 3. Update plugin_kind_map.yaml
echo "  <plugin_id>: <kind>" >> config/plugin_kind_map.yaml

# 4. Verify YAML is valid
python -c "import yaml; yaml.safe_load(open('config/plugin_kind_map.yaml'))"

# 5. Verify plugin can be loaded
python -c "from plugins.<plugin_id>.plugin import <ClassName>; print('OK')"
```

After implementing EACH TIER:

```bash
# 1. Full test suite
python -m pytest -q

# 2. Verify plugin_kind_map count increased
python -c "import yaml; d=yaml.safe_load(open('config/plugin_kind_map.yaml')); print(f'{len(d)} plugins registered')"

# 3. Commit with session log
git add -A
git commit -m "feat: add Tier N plugins (<list plugin_ids>)"
```

---

## 9. Testing Protocol

### Minimum Test Per Plugin

```python
"""tests/test_<plugin_id>.py"""
import numpy as np
import pandas as pd
import pytest

from plugins.<plugin_id>.plugin import <ClassName>


class TestPluginSmoke:
    """Smoke test: runs on synthetic data, returns valid PluginResult."""

    def setup_method(self):
        self.plugin = <ClassName>()

    def test_basic_run(self):
        # Generate synthetic data (use the test data generator from the spec)
        df = <make_test_data_function>(n=200, seed=42)
        ctx = PluginContext(dataframe=df, params={
            "treatment_col": "treatment",
            "outcome_col": "outcome",
            # ... required params
        })
        result = self.plugin.run(ctx)
        assert result.summary["status"] == "ok"
        assert result.summary["n_observations"] > 0

    def test_missing_required_param(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ctx = PluginContext(dataframe=df, params={})
        result = self.plugin.run(ctx)
        assert result.summary["status"] == "skipped"

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        ctx = PluginContext(dataframe=df, params={"treatment_col": "t", "outcome_col": "y"})
        result = self.plugin.run(ctx)
        assert result.summary["status"] in ("skipped", "error")

    def test_nulls_handled(self):
        df = <make_test_data_function>(n=200, seed=42)
        df.iloc[0:10, 0] = np.nan  # inject nulls
        ctx = PluginContext(dataframe=df, params={
            "treatment_col": "treatment",
            "outcome_col": "outcome",
        })
        result = self.plugin.run(ctx)
        # Should either skip or run on clean subset
        assert result.summary["status"] in ("ok", "skipped")
```

---

## 10. Session Log Protocol

At the end of every session that modifies code, tests, config, or docs, append to `CLAUDE.md`:

```markdown
### YYYY-MM-DD — [branch-name]
- Added: `plugins/<plugin_id>/` (new plugin)
- Modified: `config/plugin_kind_map.yaml` (added mapping)
- Tests: NNN/NNN passed
- TODO: <any carry-forward items>
```

---

## Appendix: plugin_kind_map.yaml Additions

Add these entries to `config/plugin_kind_map.yaml`:

```yaml
# === New Plugins (Gap Analysis 2026-03-04) ===

# Tier 1: Immediate
analysis_double_ml_ate_v1: causal
analysis_diff_in_diff_v1: counterfactual
analysis_synthetic_control_v1: counterfactual
analysis_regression_discontinuity_v1: counterfactual
analysis_conformal_prediction_interval_v1: distribution
analysis_shap_feature_attribution_v1: role_inference
analysis_interrupted_time_series_v1: counterfactual

# Tier 2: High
analysis_causal_forest_hte_v1: causal
analysis_inverse_propensity_weighting_v1: causal
analysis_meta_learner_cate_v1: causal
analysis_bayesian_ab_test_v1: counterfactual
analysis_cuped_variance_reduction_v1: counterfactual
analysis_gradient_boosting_importance_v1: role_inference
analysis_propensity_score_matching_v1: causal

# Tier 3: Medium
analysis_shapley_interactions_v1: role_inference
analysis_compositional_logratio_v1: distribution
analysis_concept_drift_ddm_v1: changepoint
analysis_temporal_fusion_transformer_v1: time_series
analysis_sindy_dynamics_discovery_v1: time_series
analysis_functional_data_analysis_v1: distribution
analysis_fairness_bias_detection_v1: role_inference
analysis_sliced_wasserstein_drift_v1: distribution
analysis_instrumental_variables_2sls_v1: causal

# Tier 4: Specialized
analysis_autoencoder_anomaly_v1: anomaly
analysis_reservoir_computing_esn_v1: time_series
analysis_bayesian_optimization_v1: chain_makespan
analysis_thompson_sampling_bandit_v1: chain_makespan
analysis_neural_ode_dynamics_v1: time_series
analysis_information_bottleneck_v1: cluster
analysis_conditional_density_estimation_v1: distribution
```

---

*End of Implementation Guide*
