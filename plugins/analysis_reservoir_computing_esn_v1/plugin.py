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
            col = settings.get("column")

            # Auto-detect: pick the first numeric column with enough data
            if not col:
                for c in df.select_dtypes(include="number").columns:
                    if df[c].dropna().shape[0] >= 30:
                        col = c
                        break

            if not col or col not in df.columns:
                return PluginResult(
                    "skipped",
                    "No suitable numeric column found (need >= 30 non-null values)",
                    {}, [], [], None,
                )

            series = df[col].dropna().values.astype(float)
            if len(series) < 30:
                return PluginResult(
                    "skipped",
                    f"Insufficient data points ({len(series)}), need >= 30",
                    {}, [], [], None,
                )

            # Normalize to [0, 1]
            s_min, s_max = float(np.min(series)), float(np.max(series))
            if s_max - s_min < 1e-12:
                return PluginResult(
                    "skipped", "Constant series, nothing to predict",
                    {}, [], [], None,
                )
            normed = (series - s_min) / (s_max - s_min)

            # Reshape for reservoirpy: (timesteps, features)
            X = normed[:-1].reshape(-1, 1)
            y = normed[1:].reshape(-1, 1)

            split = int(len(X) * 0.8)
            X_train, y_train = X[:split], y[:split]
            X_test, y_test = X[split:], y[split:]

            if len(X_test) < 2:
                return PluginResult(
                    "skipped", "Test set too small after 80/20 split",
                    {}, [], [], None,
                )

            from reservoirpy.nodes import Reservoir, Ridge

            reservoir = Reservoir(100, sr=0.9, seed=ctx.run_seed)
            readout = Ridge(ridge=1e-6)
            esn = reservoir >> readout

            esn.fit(X_train, y_train)
            y_pred = esn.run(X_test)

            y_pred_arr = np.array(y_pred).flatten()
            y_test_arr = y_test.flatten()

            # Denormalize for reporting
            y_pred_denorm = y_pred_arr * (s_max - s_min) + s_min
            y_test_denorm = y_test_arr * (s_max - s_min) + s_min

            rmse = float(np.sqrt(np.mean((y_pred_denorm - y_test_denorm) ** 2)))
            mae = float(np.mean(np.abs(y_pred_denorm - y_test_denorm)))
            series_std = float(np.std(series))
            captures_dynamics = rmse < series_std

            findings = [{
                "kind": "time_series",
                "measurement_type": "measured",
                "column": col,
                "rmse": round(rmse, 6),
                "mae": round(mae, 6),
                "series_std": round(series_std, 6),
                "captures_dynamics": captures_dynamics,
                "train_size": split,
                "test_size": len(X_test),
                "method": "Echo State Network (reservoirpy)",
            }]

            return PluginResult(
                status="ok",
                summary=(
                    f"ESN prediction on '{col}': RMSE={rmse:.4f}, MAE={mae:.4f}, "
                    f"captures_dynamics={captures_dynamics}"
                ),
                metrics={
                    "column": col,
                    "n_total": len(series),
                    "train_size": split,
                    "test_size": len(X_test),
                    "rmse": round(rmse, 6),
                    "mae": round(mae, 6),
                    "captures_dynamics": captures_dynamics,
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"Reservoir computing ESN failed: {e}", exc_info=True)
            return PluginResult(
                status="error",
                summary=f"Reservoir computing ESN failed: {e}",
                metrics={},
                findings=[],
                artifacts=[],
                error=PluginError(
                    type=type(e).__name__,
                    message=str(e),
                    traceback=traceback.format_exc(),
                ),
            )
