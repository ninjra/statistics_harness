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
                return PluginResult("skipped", "Need at least 2 numeric columns", {}, [], [], None)

            # Use last numeric column as target, rest as features
            target_col = numeric_cols[-1]
            feature_cols = numeric_cols[:-1]

            work = df[feature_cols + [target_col]].dropna()
            if len(work) < 30:
                return PluginResult("skipped", f"Insufficient rows ({len(work)})", {}, [], [], None)

            # Budget: cap rows
            max_rows = int(ctx.budget.get("row_limit") or 5000)
            if len(work) > max_rows:
                rng = np.random.RandomState(ctx.run_seed)
                work = work.iloc[rng.choice(len(work), max_rows, replace=False)].copy()

            X = work[feature_cols].values.astype(float)
            y = work[target_col].values.astype(float)

            from sklearn.ensemble import RandomForestRegressor
            import shap

            rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=ctx.run_seed, n_jobs=1)
            rf.fit(X, y)

            explainer = shap.TreeExplainer(rf)
            # Use a background sample for efficiency
            bg_size = min(100, len(X))
            shap_values = explainer.shap_values(X[:bg_size])

            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            feature_importance = sorted(
                zip(feature_cols, mean_abs_shap.tolist()),
                key=lambda x: -x[1]
            )

            top_n = min(10, len(feature_importance))
            findings = []
            for rank, (col, importance) in enumerate(feature_importance[:top_n], 1):
                findings.append({
                    "kind": "role_inference",
                    "measurement_type": "measured",
                    "feature": col,
                    "mean_abs_shap": round(float(importance), 6),
                    "rank": rank,
                    "target": target_col,
                    "method": "SHAP TreeExplainer",
                })

            return PluginResult(
                status="ok",
                summary=f"SHAP feature attribution: top feature is {feature_importance[0][0]} (mean|SHAP|={feature_importance[0][1]:.4f})",
                metrics={
                    "n_observations": len(work),
                    "n_features": len(feature_cols),
                    "target_column": target_col,
                    "top_feature": feature_importance[0][0],
                    "top_shap_value": round(float(feature_importance[0][1]), 6),
                    "method": "SHAP TreeExplainer (RandomForest)",
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"SHAP feature attribution failed: {e}", exc_info=True)
            return PluginResult(
                status="error",
                summary=f"SHAP analysis failed: {e}",
                metrics={},
                findings=[],
                artifacts=[],
                error=PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()),
            )
