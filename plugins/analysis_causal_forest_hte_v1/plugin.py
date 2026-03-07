from __future__ import annotations
import logging
import traceback
import numpy as np
import pandas as pd
from statistic_harness.core.types import PluginResult, PluginArtifact, PluginError

logger = logging.getLogger(__name__)


class Plugin:
    def run(self, ctx) -> PluginResult:
        try:
            df = ctx.dataset_loader()
            if df.empty:
                return PluginResult("na", "Empty dataset", {}, [], [], None)

            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric_cols) < 3:
                return PluginResult("na", "Need at least 3 numeric columns", {}, [], [], None)

            treatment_col = ctx.settings.get("treatment_column")
            outcome_col = ctx.settings.get("outcome_column")

            binary_cols = [c for c in numeric_cols if df[c].dropna().isin([0, 1, 0.0, 1.0]).all()]
            non_binary = [c for c in numeric_cols if c not in binary_cols]

            if not treatment_col:
                if not binary_cols:
                    return PluginResult("na", "No binary treatment column found", {}, [], [], None)
                treatment_col = binary_cols[0]
            if not outcome_col:
                if not non_binary:
                    return PluginResult("na", "No continuous outcome column found", {}, [], [], None)
                outcome_col = non_binary[0]

            covariate_cols = [c for c in numeric_cols if c not in (treatment_col, outcome_col)]
            if len(covariate_cols) < 1:
                return PluginResult("na", "Need at least 1 covariate", {}, [], [], None)

            needed = [outcome_col, treatment_col] + covariate_cols
            work = df[needed].dropna()
            if len(work) < 50:
                return PluginResult("na", f"Insufficient rows ({len(work)})", {}, [], [], None)

            max_rows = int(ctx.budget.get("row_limit") or 5000)
            rng = np.random.RandomState(ctx.run_seed)
            if len(work) > max_rows:
                work = work.iloc[rng.choice(len(work), max_rows, replace=False)].copy()

            T = work[treatment_col].values
            Y = work[outcome_col].values
            X = work[covariate_cols].values

            from econml.dml import CausalForestDML
            from sklearn.linear_model import LogisticRegression, Lasso

            est = CausalForestDML(
                model_y=Lasso(alpha=0.01, random_state=ctx.run_seed),
                model_t=LogisticRegression(max_iter=1000, random_state=ctx.run_seed),
                discrete_treatment=True,
                n_estimators=100,
                min_samples_leaf=5,
                random_state=ctx.run_seed,
            )
            est.fit(Y, T, X=X)

            # ATE
            ate = float(est.ate(X))
            ci = est.ate_interval(X, alpha=0.05)
            ci_low = float(ci[0])
            ci_high = float(ci[1])
            ate_se = (ci_high - ci_low) / (2 * 1.96)

            # Heterogeneous treatment effects
            hte = est.effect(X).flatten()
            hte_std = float(np.std(hte))
            hte_iqr = float(np.percentile(hte, 75) - np.percentile(hte, 25))
            hte_min = float(np.min(hte))
            hte_max = float(np.max(hte))

            # Feature importance for treatment heterogeneity
            try:
                importances = est.feature_importances_
                feature_importance = {
                    col: round(float(imp), 6)
                    for col, imp in zip(covariate_cols, importances)
                }
            except Exception:
                feature_importance = {}

            # p-value from z-test on ATE
            z = ate / ate_se if ate_se > 0 else 0.0
            from scipy.stats import norm
            pvalue = float(2 * (1 - norm.cdf(abs(z))))

            findings = [{
                "kind": "causal",
                "measurement_type": "measured",
                "method": "Causal Forest (CausalForestDML)",
                "ate": round(ate, 6),
                "std_error": round(ate_se, 6),
                "p_value": round(pvalue, 6),
                "ci_95_lower": round(ci_low, 6),
                "ci_95_upper": round(ci_high, 6),
                "significant": pvalue < 0.05,
                "hte_std": round(hte_std, 6),
                "hte_iqr": round(hte_iqr, 6),
                "hte_min": round(hte_min, 6),
                "hte_max": round(hte_max, 6),
                "feature_importance": feature_importance,
                "treatment_column": treatment_col,
                "outcome_column": outcome_col,
                "n_covariates": len(covariate_cols),
                "interpretation": (
                    f"Causal Forest ATE: {ate:+.4f} (p={pvalue:.4f}), "
                    f"HTE std={hte_std:.4f}, IQR={hte_iqr:.4f}"
                ),
            }]

            return PluginResult(
                status="ok",
                summary=f"Causal Forest ATE: {ate:+.4f}, HTE std={hte_std:.4f}",
                metrics={
                    "n_observations": len(work),
                    "ate": round(ate, 6),
                    "std_error": round(ate_se, 6),
                    "p_value": round(pvalue, 6),
                    "hte_std": round(hte_std, 6),
                    "hte_iqr": round(hte_iqr, 6),
                    "n_covariates": len(covariate_cols),
                    "n_estimators": 100,
                    "method": "CausalForestDML (econml)",
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except ImportError as e:
            return PluginResult("na", f"econml not available: {e}", {}, [], [], None)
        except Exception as e:
            logger.error(f"Causal forest HTE analysis failed: {e}", exc_info=True)
            return PluginResult("error", f"Causal forest HTE analysis failed: {e}", {}, [], [],
                PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()))
