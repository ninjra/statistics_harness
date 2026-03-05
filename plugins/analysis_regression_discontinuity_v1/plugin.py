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
                return PluginResult("skipped", "Empty dataset", {}, [], [], None)

            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric_cols) < 2:
                return PluginResult("skipped", "Need at least 2 numeric columns (running + outcome)", {}, [], [], None)

            running_col = ctx.settings.get("running_column") or numeric_cols[0]
            outcome_col = ctx.settings.get("outcome_column") or numeric_cols[1] if len(numeric_cols) > 1 else None

            if not outcome_col or running_col not in df.columns or outcome_col not in df.columns:
                return PluginResult("skipped", "Required columns not found", {}, [], [], None)

            work = df[[running_col, outcome_col]].dropna()
            if len(work) < 30:
                return PluginResult("skipped", f"Insufficient rows ({len(work)})", {}, [], [], None)

            cutoff = float(ctx.settings.get("cutoff", work[running_col].median()))

            x = work[running_col].values.astype(float)
            y = work[outcome_col].values.astype(float)

            try:
                from rdrobust import rdrobust as rd_estimate
                result_rd = rd_estimate(y, x, c=cutoff)

                coef = result_rd.coef
                rd_effect = float(coef.iloc[0] if hasattr(coef, 'iloc') else coef)
                se_val = result_rd.se
                rd_se = float(se_val.iloc[0] if hasattr(se_val, 'iloc') else se_val)
                pv_val = result_rd.pv
                rd_pvalue = float(pv_val.iloc[0] if hasattr(pv_val, 'iloc') else pv_val)
                ci_val = result_rd.ci
                ci_low = float(ci_val.iloc[0, 0] if hasattr(ci_val, 'iloc') else ci_val[0])
                ci_high = float(ci_val.iloc[0, 1] if hasattr(ci_val, 'iloc') else ci_val[1])
                bws_val = result_rd.bws
                bandwidth = float(bws_val.iloc[0, 0] if hasattr(bws_val, 'iloc') else bws_val[0])
                method_used = "rdrobust (local polynomial)"
            except Exception:
                # Fallback: simple linear RD
                import statsmodels.api as sm

                above = (x >= cutoff).astype(float)
                x_centered = x - cutoff
                X_rd = np.column_stack([x_centered, above, x_centered * above])
                X_rd = sm.add_constant(X_rd)
                model = sm.OLS(y, X_rd).fit()

                rd_effect = float(model.params[2])
                rd_se = float(model.bse[2])
                rd_pvalue = float(model.pvalues[2])
                ci = model.conf_int()
                ci_low = float(ci[2, 0])
                ci_high = float(ci[2, 1])
                bandwidth = float(np.std(x))
                method_used = "OLS linear RD (fallback)"

            findings = [{
                "kind": "counterfactual",
                "measurement_type": "measured",
                "method": "Regression Discontinuity",
                "rd_estimate": round(rd_effect, 6),
                "std_error": round(rd_se, 6),
                "p_value": round(rd_pvalue, 6),
                "ci_95_lower": round(ci_low, 6),
                "ci_95_upper": round(ci_high, 6),
                "cutoff": round(cutoff, 6),
                "bandwidth": round(bandwidth, 6),
                "significant": rd_pvalue < 0.05,
                "method_detail": method_used,
                "interpretation": f"RD estimate at cutoff {cutoff:.2f}: {rd_effect:+.4f} (p={rd_pvalue:.4f})",
            }]

            return PluginResult(
                status="ok",
                summary=f"RD: effect={rd_effect:+.4f} at cutoff={cutoff:.2f} (p={rd_pvalue:.4f})",
                metrics={
                    "n_observations": len(work),
                    "rd_estimate": round(rd_effect, 6),
                    "std_error": round(rd_se, 6),
                    "p_value": round(rd_pvalue, 6),
                    "cutoff": round(cutoff, 6),
                    "bandwidth": round(bandwidth, 6),
                    "method": method_used,
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"RD analysis failed: {e}", exc_info=True)
            return PluginResult("error", f"RD analysis failed: {e}", {}, [], [],
                PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()))
