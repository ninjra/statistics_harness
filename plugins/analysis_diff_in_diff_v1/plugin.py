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

            # Need: outcome (numeric), treatment (binary), post (binary)
            outcome_col = ctx.settings.get("outcome_column")
            treatment_col = ctx.settings.get("treatment_column")
            post_col = ctx.settings.get("post_column")

            # Auto-detect if not specified
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            binary_cols = [c for c in numeric_cols if df[c].dropna().isin([0, 1, 0.0, 1.0]).all()]

            if not outcome_col:
                non_binary = [c for c in numeric_cols if c not in binary_cols]
                if not non_binary:
                    return PluginResult("na", "No continuous outcome column found", {}, [], [], None)
                outcome_col = non_binary[0]
            if not treatment_col and len(binary_cols) >= 1:
                treatment_col = binary_cols[0]
            if not post_col and len(binary_cols) >= 2:
                post_col = binary_cols[1]

            if not all([outcome_col, treatment_col, post_col]):
                return PluginResult("na", "Missing required columns (outcome, treatment, post)", {}, [], [], None)

            needed = [outcome_col, treatment_col, post_col]
            for c in needed:
                if c not in df.columns:
                    return PluginResult("na", f"Column {c} not found", {}, [], [], None)

            work = df[needed].dropna()
            if len(work) < 20:
                return PluginResult("na", f"Insufficient rows ({len(work)})", {}, [], [], None)

            import statsmodels.api as sm

            y = work[outcome_col].values.astype(float)
            treat = work[treatment_col].values.astype(float)
            post = work[post_col].values.astype(float)
            interaction = treat * post

            X = np.column_stack([treat, post, interaction])
            X = sm.add_constant(X)

            model = sm.OLS(y, X).fit()

            did_estimate = float(model.params[3])
            did_se = float(model.bse[3])
            did_pvalue = float(model.pvalues[3])
            ci_low = float(model.conf_int()[3, 0])
            ci_high = float(model.conf_int()[3, 1])

            findings = [{
                "kind": "counterfactual",
                "measurement_type": "measured",
                "method": "Difference-in-Differences",
                "did_estimate": round(did_estimate, 6),
                "std_error": round(did_se, 6),
                "p_value": round(did_pvalue, 6),
                "ci_95_lower": round(ci_low, 6),
                "ci_95_upper": round(ci_high, 6),
                "significant": did_pvalue < 0.05,
                "outcome_column": outcome_col,
                "treatment_column": treatment_col,
                "post_column": post_col,
                "interpretation": f"DiD estimate: {did_estimate:+.4f} (p={did_pvalue:.4f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}])",
            }]

            return PluginResult(
                status="ok",
                summary=f"DiD: treatment effect = {did_estimate:+.4f} (p={did_pvalue:.4f})",
                metrics={
                    "n_observations": len(work),
                    "did_estimate": round(did_estimate, 6),
                    "did_std_error": round(did_se, 6),
                    "did_pvalue": round(did_pvalue, 6),
                    "r_squared": round(float(model.rsquared), 4),
                    "method": "OLS Difference-in-Differences",
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"DiD analysis failed: {e}", exc_info=True)
            return PluginResult("error", f"DiD analysis failed: {e}", {}, [], [],
                PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()))
