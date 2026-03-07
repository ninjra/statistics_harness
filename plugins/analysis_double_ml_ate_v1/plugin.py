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

            # Identify treatment (binary), outcome (continuous), covariates (rest)
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
            if len(work) > max_rows:
                rng = np.random.RandomState(ctx.run_seed)
                work = work.iloc[rng.choice(len(work), max_rows, replace=False)].copy()

            from doubleml import DoubleMLPLR, DoubleMLData
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

            dml_data = DoubleMLData(
                work, y_col=outcome_col, d_cols=treatment_col,
                x_cols=covariate_cols
            )

            ml_l = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=ctx.run_seed, n_jobs=1)
            ml_m = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=ctx.run_seed, n_jobs=1)

            dml_plr = DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=3)
            dml_plr.fit()

            ate = float(dml_plr.coef[0])
            se = float(dml_plr.se[0])
            pvalue = float(dml_plr.pval[0])
            ci = dml_plr.confint(level=0.95)
            ci_low = float(ci.iloc[0, 0])
            ci_high = float(ci.iloc[0, 1])

            findings = [{
                "kind": "causal",
                "measurement_type": "measured",
                "method": "Double Machine Learning (PLR)",
                "ate": round(ate, 6),
                "std_error": round(se, 6),
                "p_value": round(pvalue, 6),
                "ci_95_lower": round(ci_low, 6),
                "ci_95_upper": round(ci_high, 6),
                "significant": pvalue < 0.05,
                "treatment_column": treatment_col,
                "outcome_column": outcome_col,
                "n_covariates": len(covariate_cols),
                "interpretation": f"DML ATE: {ate:+.4f} (p={pvalue:.4f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}])",
            }]

            return PluginResult(
                status="ok",
                summary=f"Double ML ATE: {ate:+.4f} (p={pvalue:.4f})",
                metrics={
                    "n_observations": len(work),
                    "ate": round(ate, 6),
                    "std_error": round(se, 6),
                    "p_value": round(pvalue, 6),
                    "n_covariates": len(covariate_cols),
                    "method": "DoubleML PLR (RandomForest)",
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"DML analysis failed: {e}", exc_info=True)
            return PluginResult("error", f"DML analysis failed: {e}", {}, [], [],
                PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()))
