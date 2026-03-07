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

            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if len(numeric_cols) < 2:
                return PluginResult(
                    "na",
                    "Need at least 2 numeric columns for functional data",
                    {}, [], [], None,
                )

            work = df[numeric_cols].dropna()
            if len(work) < 5:
                return PluginResult(
                    "na", f"Insufficient rows ({len(work)}) for FDA", {}, [], [], None,
                )

            try:
                import skfda
                from skfda import FDataGrid
                from skfda.exploratory.depth import ModifiedBandDepth
                from skfda.exploratory.stats import mean as fda_mean, var as fda_var
            except ImportError:
                return PluginResult("na", "skfda not installed", {}, [], [], None)

            data_matrix = work.values.astype(float)
            grid_points = np.linspace(0, 1, data_matrix.shape[1])
            fd = FDataGrid(data_matrix=data_matrix, grid_points=grid_points)

            # Functional mean and variance
            func_mean = fda_mean(fd)
            func_var = fda_var(fd)

            mean_values = func_mean.data_matrix[0, :, 0].tolist()
            var_values = func_var.data_matrix[0, :, 0].tolist()

            # Outlier detection via modified band depth
            depth = ModifiedBandDepth()
            depth_values = depth(fd).tolist()

            # Flag outliers as those below 25th percentile of depth
            depth_arr = np.array(depth_values)
            q25 = float(np.percentile(depth_arr, 25))
            outlier_indices = [int(i) for i in np.where(depth_arr < q25)[0]]
            n_outliers = len(outlier_indices)

            findings = [{
                "kind": "distribution",
                "measurement_type": "measured",
                "numeric_columns": numeric_cols,
                "n_functions": len(work),
                "n_grid_points": len(grid_points),
                "functional_mean": [round(v, 6) for v in mean_values],
                "functional_variance": [round(v, 6) for v in var_values],
                "depth_quartile_25": round(q25, 6),
                "n_outliers": n_outliers,
                "outlier_indices": outlier_indices[:20],  # cap for readability
                "method": "Functional Data Analysis (scikit-fda, ModifiedBandDepth)",
            }]

            return PluginResult(
                status="ok",
                summary=(
                    f"FDA on {len(work)} functions over {len(numeric_cols)} grid points: "
                    f"{n_outliers} depth-based outliers detected"
                ),
                metrics={
                    "n_functions": len(work),
                    "n_grid_points": len(grid_points),
                    "n_outliers": n_outliers,
                    "depth_q25": round(q25, 6),
                    "mean_depth": round(float(np.mean(depth_arr)), 6),
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"Functional data analysis failed: {e}", exc_info=True)
            return PluginResult(
                status="error",
                summary=f"Functional data analysis failed: {e}",
                metrics={},
                findings=[],
                artifacts=[],
                error=PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()),
            )
