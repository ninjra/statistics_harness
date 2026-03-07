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

            numeric_cols = [
                c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
            ]
            if not numeric_cols:
                return PluginResult(
                    "na", "No numeric columns", {}, [], [], None
                )

            target_col = ctx.settings.get("target_column", numeric_cols[0])
            if target_col not in df.columns:
                return PluginResult(
                    "na",
                    f"Column '{target_col}' not found",
                    {},
                    [],
                    [],
                    None,
                )

            series = df[target_col].dropna().values.astype(np.float64)
            if len(series) < 20:
                return PluginResult(
                    "na",
                    f"Insufficient data ({len(series)} rows)",
                    {},
                    [],
                    [],
                    None,
                )

            mid = len(series) // 2
            first_half = series[:mid].reshape(-1, 1)
            second_half = series[mid:].reshape(-1, 1)

            seed = int(getattr(ctx, "run_seed", 42) or 42)
            n_projections = int(ctx.settings.get("n_projections", 50))

            import ot

            swd = float(
                ot.sliced_wasserstein_distance(
                    first_half, second_half, n_projections=n_projections, seed=seed
                )
            )

            findings = [
                {
                    "kind": "distribution",
                    "measurement_type": "measured",
                    "method": "Sliced Wasserstein Distance",
                    "column": target_col,
                    "sliced_wasserstein_distance": round(swd, 6),
                    "n_projections": n_projections,
                    "first_half_n": len(first_half),
                    "second_half_n": len(second_half),
                    "interpretation": (
                        f"SWD={swd:.4f} between first and second half of '{target_col}'"
                    ),
                }
            ]

            return PluginResult(
                status="ok",
                summary=f"Sliced Wasserstein distance: {swd:.4f} on '{target_col}'",
                metrics={
                    "sliced_wasserstein_distance": round(swd, 6),
                    "n_observations": len(series),
                    "column": target_col,
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(
                f"Sliced Wasserstein drift analysis failed: {e}", exc_info=True
            )
            return PluginResult(
                "error",
                f"Sliced Wasserstein drift analysis failed: {e}",
                {},
                [],
                [],
                PluginError(
                    type=type(e).__name__,
                    message=str(e),
                    traceback=traceback.format_exc(),
                ),
            )
