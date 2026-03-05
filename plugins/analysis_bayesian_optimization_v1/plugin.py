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

            settings = ctx.settings or {}
            objective_col = settings.get("objective_column")
            param_cols = settings.get("param_columns")
            n_iter = int(settings.get("n_iter", 5))
            init_points = int(settings.get("init_points", 3))

            # Auto-detect: numeric columns; last numeric column is objective
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric_cols) < 2:
                return PluginResult(
                    "skipped",
                    f"Need at least 2 numeric columns, found {len(numeric_cols)}",
                    {}, [], [], None,
                )

            if not objective_col:
                objective_col = numeric_cols[-1]
            if not param_cols:
                param_cols = [c for c in numeric_cols if c != objective_col]

            if not param_cols:
                return PluginResult("skipped", "No parameter columns identified", {}, [], [], None)

            work = df[param_cols + [objective_col]].dropna()
            if len(work) < 5:
                return PluginResult("skipped", f"Insufficient rows ({len(work)})", {}, [], [], None)

            # Build parameter bounds from data
            pbounds = {}
            for col in param_cols:
                col_min = float(work[col].min())
                col_max = float(work[col].max())
                if col_min == col_max:
                    col_max = col_min + 1.0
                pbounds[col] = (col_min, col_max)

            from bayes_opt import BayesianOptimization

            # Build a lookup function using nearest-neighbor interpolation
            from sklearn.neighbors import KNeighborsRegressor

            X = work[param_cols].values
            y = work[objective_col].values
            knn = KNeighborsRegressor(n_neighbors=min(5, len(work)))
            knn.fit(X, y)

            def black_box(**kwargs):
                point = np.array([[kwargs[c] for c in param_cols]])
                return float(knn.predict(point)[0])

            optimizer = BayesianOptimization(
                f=black_box,
                pbounds=pbounds,
                random_state=ctx.run_seed,
                verbose=0,
                allow_duplicate_points=True,
            )

            optimizer.maximize(init_points=init_points, n_iter=n_iter)

            best_params = optimizer.max["params"]
            best_target = float(optimizer.max["target"])

            # Expected improvement over observed mean
            observed_mean = float(y.mean())
            expected_improvement = best_target - observed_mean

            findings = [{
                "kind": "distribution",
                "measurement_type": "measured",
                "objective_column": objective_col,
                "param_columns": param_cols,
                "best_params": {k: round(v, 6) for k, v in best_params.items()},
                "best_target": round(best_target, 6),
                "observed_mean": round(observed_mean, 6),
                "expected_improvement": round(expected_improvement, 6),
                "n_iterations": n_iter,
                "method": "Bayesian Optimization (GP surrogate)",
            }]

            return PluginResult(
                status="ok",
                summary=(
                    f"Bayesian optimization: best_target={best_target:.4f}, "
                    f"improvement over mean={expected_improvement:.4f}, "
                    f"{len(param_cols)} params"
                ),
                metrics={
                    "n_observations": len(work),
                    "n_params": len(param_cols),
                    "best_target": round(best_target, 6),
                    "observed_mean": round(observed_mean, 6),
                    "expected_improvement": round(expected_improvement, 6),
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}", exc_info=True)
            return PluginResult(
                status="error",
                summary=f"Bayesian optimization failed: {e}",
                metrics={},
                findings=[],
                artifacts=[],
                error=PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()),
            )
