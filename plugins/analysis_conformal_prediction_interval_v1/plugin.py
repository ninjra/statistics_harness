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
                return PluginResult("skipped", "Empty dataset", {}, [], [], None)

            numeric_cols = [
                c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
            ]
            if len(numeric_cols) < 2:
                return PluginResult(
                    "skipped", "Need at least 2 numeric columns", {}, [], [], None
                )

            target_col = numeric_cols[-1]
            feature_cols = numeric_cols[:-1]

            work = df[feature_cols + [target_col]].dropna()
            if len(work) < 30:
                return PluginResult(
                    "skipped", f"Insufficient rows ({len(work)})", {}, [], [], None
                )

            max_rows = int(ctx.budget.get("row_limit") or 5000)
            if len(work) > max_rows:
                rng = np.random.RandomState(ctx.run_seed)
                work = work.iloc[
                    rng.choice(len(work), max_rows, replace=False)
                ].copy()

            X = work[feature_cols].values.astype(float)
            y = work[target_col].values.astype(float)

            from mapie.regression import SplitConformalRegressor
            from sklearn.ensemble import RandomForestRegressor

            alpha = float(ctx.settings.get("alpha", 0.1))
            confidence_level = 1.0 - alpha

            base_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=ctx.run_seed,
                n_jobs=1,
            )
            base_model.fit(X, y)
            mapie = SplitConformalRegressor(
                base_model, prefit=True, confidence_level=confidence_level
            )
            mapie.conformalize(X, y)

            y_pred, y_pis = mapie.predict_interval(X)
            # y_pis shape: (n, 2, 1) — squeeze last dim
            y_pis = y_pis[:, :, 0]

            interval_widths = y_pis[:, 1] - y_pis[:, 0]
            coverage = float(
                np.mean((y >= y_pis[:, 0]) & (y <= y_pis[:, 1]))
            )
            mean_width = float(np.mean(interval_widths))
            median_width = float(np.median(interval_widths))

            findings = [
                {
                    "kind": "distribution",
                    "measurement_type": "measured",
                    "method": "MAPIE conformal prediction",
                    "confidence_level": 1.0 - alpha,
                    "empirical_coverage": round(coverage, 4),
                    "mean_interval_width": round(mean_width, 4),
                    "median_interval_width": round(median_width, 4),
                    "target": target_col,
                    "n_features": len(feature_cols),
                    "interpretation": (
                        f"Conformal intervals at {1 - alpha:.0%} level cover "
                        f"{coverage:.1%} of observations with mean width "
                        f"{mean_width:.2f}"
                    ),
                }
            ]

            # Flag features with widest intervals
            wide_mask = interval_widths > np.percentile(interval_widths, 90)
            if wide_mask.sum() > 0:
                findings.append(
                    {
                        "kind": "distribution",
                        "measurement_type": "measured",
                        "type": "high_uncertainty_region",
                        "n_high_uncertainty": int(wide_mask.sum()),
                        "pct_high_uncertainty": round(
                            float(wide_mask.mean()) * 100, 2
                        ),
                        "p90_width": round(
                            float(np.percentile(interval_widths, 90)), 4
                        ),
                    }
                )

            return PluginResult(
                status="ok",
                summary=(
                    f"Conformal prediction: {1 - alpha:.0%} intervals, "
                    f"coverage={coverage:.1%}, mean width={mean_width:.2f}"
                ),
                metrics={
                    "n_observations": len(work),
                    "n_features": len(feature_cols),
                    "target_column": target_col,
                    "confidence_level": 1.0 - alpha,
                    "empirical_coverage": round(coverage, 4),
                    "mean_interval_width": round(mean_width, 4),
                    "method": "MAPIE MapieRegressor (plus)",
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error("Conformal prediction failed: %s", e, exc_info=True)
            return PluginResult(
                "error",
                f"Conformal prediction failed: {e}",
                {},
                [],
                [],
                PluginError(
                    type=type(e).__name__,
                    message=str(e),
                    traceback=traceback.format_exc(),
                ),
            )
