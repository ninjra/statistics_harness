from __future__ import annotations

import logging
import traceback

import numpy as np
import pandas as pd

from statistic_harness.core.types import PluginArtifact, PluginError, PluginResult

logger = logging.getLogger(__name__)


class Plugin:
    def run(self, ctx) -> PluginResult:
        try:
            df = ctx.dataset_loader()
            if df.empty:
                return PluginResult("na", "Empty dataset", {}, [], [], None)

            target_col = ctx.settings.get("target_column")
            numeric_cols = [
                c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
            ]

            if target_col and target_col not in df.columns:
                return PluginResult(
                    "na",
                    f"Column '{target_col}' not found",
                    {},
                    [],
                    [],
                    None,
                )

            if not target_col:
                if len(numeric_cols) < 3:
                    return PluginResult(
                        "na",
                        "Need at least 3 numeric columns",
                        {},
                        [],
                        [],
                        None,
                    )
                target_col = numeric_cols[-1]

            feature_cols = [c for c in numeric_cols if c != target_col]
            if len(feature_cols) < 2:
                return PluginResult(
                    "na",
                    "Need at least 2 feature columns for interactions",
                    {},
                    [],
                    [],
                    None,
                )

            max_features = int(ctx.settings.get("max_features", 10))
            feature_cols = feature_cols[:max_features]

            work = df[feature_cols + [target_col]].dropna()
            if len(work) < 30:
                return PluginResult(
                    "na",
                    f"Insufficient rows ({len(work)})",
                    {},
                    [],
                    [],
                    None,
                )

            max_rows = int(ctx.settings.get("max_rows", 2000))
            seed = int(getattr(ctx, "run_seed", 42) or 42)
            if len(work) > max_rows:
                rng = np.random.RandomState(seed)
                work = work.iloc[rng.choice(len(work), max_rows, replace=False)].copy()

            X = work[feature_cols].values
            y = work[target_col].values

            from sklearn.ensemble import RandomForestRegressor
            import shapiq

            rf = RandomForestRegressor(
                n_estimators=50, max_depth=5, random_state=seed, n_jobs=1
            )
            rf.fit(X, y)

            explainer = shapiq.TreeExplainer(
                model=rf, max_order=2, index="k-SII"
            )

            # Compute interactions on a sample
            n_explain = min(50, len(X))
            rng = np.random.RandomState(seed)
            explain_idx = rng.choice(len(X), n_explain, replace=False)
            X_explain = X[explain_idx]

            # Aggregate pairwise interactions across explanation samples
            pair_sums: dict[tuple[int, int], float] = {}
            pair_counts: dict[tuple[int, int], int] = {}

            n_feat = len(feature_cols)
            for row in X_explain:
                iv = explainer.explain(row)
                # get_n_order_values(2) returns an n_feat x n_feat matrix
                interaction_matrix = iv.get_n_order_values(2)
                for i in range(n_feat):
                    for j in range(i + 1, n_feat):
                        val = float(interaction_matrix[i, j])
                        pair = (i, j)
                        pair_sums[pair] = pair_sums.get(pair, 0.0) + abs(val)
                        pair_counts[pair] = pair_counts.get(pair, 0) + 1

            # Average and rank
            top_n = int(ctx.settings.get("top_n", 10))
            ranked = sorted(
                pair_sums.items(),
                key=lambda item: item[1] / max(pair_counts.get(item[0], 1), 1),
                reverse=True,
            )[:top_n]

            interactions = []
            for (i, j), total in ranked:
                count = pair_counts.get((i, j), 1)
                interactions.append(
                    {
                        "feature_a": feature_cols[i],
                        "feature_b": feature_cols[j],
                        "mean_abs_interaction": round(total / count, 6),
                    }
                )

            findings = [
                {
                    "kind": "role_inference",
                    "measurement_type": "measured",
                    "method": "Shapley Interaction Values (k-SII)",
                    "target_column": target_col,
                    "n_features": len(feature_cols),
                    "n_observations": len(work),
                    "n_explain_samples": n_explain,
                    "top_interactions": interactions,
                }
            ]

            return PluginResult(
                status="ok",
                summary=(
                    f"Shapley interactions: top pair is "
                    f"({interactions[0]['feature_a']}, {interactions[0]['feature_b']}) "
                    f"with mean |interaction|={interactions[0]['mean_abs_interaction']:.4f}"
                    if interactions
                    else "No pairwise interactions found"
                ),
                metrics={
                    "n_features": len(feature_cols),
                    "n_observations": len(work),
                    "n_interactions_reported": len(interactions),
                    "target_column": target_col,
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(
                f"Shapley interactions analysis failed: {e}", exc_info=True
            )
            return PluginResult(
                "error",
                f"Shapley interactions analysis failed: {e}",
                {},
                [],
                [],
                PluginError(
                    type=type(e).__name__,
                    message=str(e),
                    traceback=traceback.format_exc(),
                ),
            )
