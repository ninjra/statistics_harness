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
            if len(numeric_cols) < 2:
                return PluginResult("na", "Need at least 2 numeric columns (treated + donors)", {}, [], [], None)

            # First column is the treated unit, rest are donors
            treated_col = ctx.settings.get("treated_column") or numeric_cols[0]
            donor_cols = ctx.settings.get("donor_columns") or [c for c in numeric_cols if c != treated_col]

            if not donor_cols:
                return PluginResult("na", "Need at least 1 donor column", {}, [], [], None)

            needed = [treated_col] + donor_cols
            work = df[needed].dropna()
            if len(work) < 20:
                return PluginResult("na", f"Insufficient rows ({len(work)})", {}, [], [], None)

            intervention_idx = int(ctx.settings.get("intervention_index", len(work) // 2))
            intervention_idx = max(5, min(len(work) - 5, intervention_idx))

            from scipy.optimize import minimize

            y_treated = work[treated_col].values.astype(float)
            X_donors = work[donor_cols].values.astype(float)

            # Pre-period
            y_pre = y_treated[:intervention_idx]
            X_pre = X_donors[:intervention_idx]

            # Constrained optimization: weights sum to 1, non-negative
            n_donors = X_pre.shape[1]

            def objective(w):
                synthetic = X_pre @ w
                return float(np.sum((y_pre - synthetic) ** 2))

            from scipy.optimize import LinearConstraint

            # Weights sum to 1
            constraint = LinearConstraint(np.ones(n_donors), lb=1.0, ub=1.0)
            bounds = [(0.0, 1.0)] * n_donors
            w0 = np.ones(n_donors) / n_donors

            result_opt = minimize(objective, w0, method="SLSQP", bounds=bounds,
                                  constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0})

            weights = result_opt.x
            synthetic = X_donors @ weights

            # Treatment effect
            effect = y_treated[intervention_idx:] - synthetic[intervention_idx:]
            mean_effect = float(np.mean(effect))
            pre_rmse = float(np.sqrt(np.mean((y_pre - X_pre @ weights) ** 2)))

            findings = [{
                "kind": "counterfactual",
                "measurement_type": "measured",
                "method": "Synthetic Control",
                "mean_treatment_effect": round(mean_effect, 6),
                "pre_intervention_rmse": round(pre_rmse, 6),
                "intervention_index": intervention_idx,
                "n_donors": n_donors,
                "donor_weights": {col: round(float(w), 4) for col, w in zip(donor_cols, weights) if w > 0.01},
                "interpretation": f"Synthetic control estimate: mean post-intervention effect = {mean_effect:+.4f}",
            }]

            return PluginResult(
                status="ok",
                summary=f"Synthetic control: mean effect = {mean_effect:+.4f}, pre-RMSE = {pre_rmse:.4f}",
                metrics={
                    "n_observations": len(work),
                    "intervention_index": intervention_idx,
                    "mean_treatment_effect": round(mean_effect, 6),
                    "pre_intervention_rmse": round(pre_rmse, 6),
                    "n_donors": n_donors,
                    "method": "Synthetic Control (constrained OLS)",
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"Synthetic control failed: {e}", exc_info=True)
            return PluginResult("error", f"Synthetic control failed: {e}", {}, [], [],
                PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()))
