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
            if len(numeric_cols) < 1:
                return PluginResult("na", "No numeric columns", {}, [], [], None)

            # Use first numeric column as outcome, intervention at midpoint
            outcome_col = ctx.settings.get("outcome_column") or numeric_cols[0]
            if outcome_col not in df.columns:
                return PluginResult("na", f"Column {outcome_col} not found", {}, [], [], None)

            y = df[outcome_col].dropna().values.astype(float)
            if len(y) < 20:
                return PluginResult("na", f"Insufficient data points ({len(y)})", {}, [], [], None)

            # Intervention point: user setting or midpoint
            intervention_idx = int(ctx.settings.get("intervention_index", len(y) // 2))
            intervention_idx = max(5, min(len(y) - 5, intervention_idx))

            import statsmodels.api as sm

            n = len(y)
            time = np.arange(n, dtype=float)
            post = (time >= intervention_idx).astype(float)
            time_after = np.where(post, time - intervention_idx, 0.0)

            X = np.column_stack([time, post, time_after])
            X = sm.add_constant(X)

            model = sm.OLS(y, X).fit()

            # Coefficients: const, trend, level_change, slope_change
            level_change = float(model.params[2])
            slope_change = float(model.params[3])
            level_pvalue = float(model.pvalues[2])
            slope_pvalue = float(model.pvalues[3])

            significant = level_pvalue < 0.05 or slope_pvalue < 0.05

            findings = [{
                "kind": "counterfactual",
                "measurement_type": "measured",
                "method": "Interrupted Time Series (segmented regression)",
                "intervention_index": intervention_idx,
                "level_change": round(level_change, 6),
                "level_change_pvalue": round(level_pvalue, 6),
                "slope_change": round(slope_change, 6),
                "slope_change_pvalue": round(slope_pvalue, 6),
                "r_squared": round(float(model.rsquared), 4),
                "significant": significant,
                "interpretation": (
                    f"Level change: {level_change:+.3f} (p={level_pvalue:.4f}), "
                    f"slope change: {slope_change:+.3f} (p={slope_pvalue:.4f})"
                ),
            }]

            return PluginResult(
                status="ok",
                summary=f"ITS: level change={level_change:+.3f} (p={level_pvalue:.4f}), slope change={slope_change:+.3f} (p={slope_pvalue:.4f})",
                metrics={
                    "n_observations": n,
                    "intervention_index": intervention_idx,
                    "level_change": round(level_change, 6),
                    "slope_change": round(slope_change, 6),
                    "r_squared": round(float(model.rsquared), 4),
                    "method": "OLS segmented regression",
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"ITS analysis failed: {e}", exc_info=True)
            return PluginResult("error", f"ITS analysis failed: {e}", {}, [], [],
                PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()))
