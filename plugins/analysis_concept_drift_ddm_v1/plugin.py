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

            series = df[target_col].dropna().reset_index(drop=True)
            if len(series) < 30:
                return PluginResult(
                    "na",
                    f"Insufficient data ({len(series)} rows)",
                    {},
                    [],
                    [],
                    None,
                )

            from river.drift.binary import DDM

            detector = DDM()
            median_val = float(series.median())
            drift_points: list[int] = []

            for i, val in enumerate(series):
                error = 1 if float(val) > median_val else 0
                detector.update(error)
                if detector.drift_detected:
                    drift_points.append(i)

            findings = [
                {
                    "kind": "changepoint",
                    "measurement_type": "measured",
                    "method": "DDM (Drift Detection Method)",
                    "column": target_col,
                    "drift_point_indices": drift_points,
                    "n_drifts_detected": len(drift_points),
                    "median_threshold": round(median_val, 6),
                    "n_observations": len(series),
                }
            ]

            return PluginResult(
                status="ok",
                summary=f"DDM detected {len(drift_points)} drift point(s) in '{target_col}'",
                metrics={
                    "n_drift_points": len(drift_points),
                    "n_observations": len(series),
                    "column": target_col,
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"DDM concept drift analysis failed: {e}", exc_info=True)
            return PluginResult(
                "error",
                f"DDM concept drift analysis failed: {e}",
                {},
                [],
                [],
                PluginError(
                    type=type(e).__name__,
                    message=str(e),
                    traceback=traceback.format_exc(),
                ),
            )
