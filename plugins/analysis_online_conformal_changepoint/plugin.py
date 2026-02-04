from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


class Plugin:
    def run(self, ctx) -> PluginResult:
        df = ctx.dataset_loader()
        value_col = ctx.settings.get("value_column")
        value_cols = ctx.settings.get("value_columns")
        columns: list[str]
        if value_col:
            columns = [value_col] if value_col in df.columns else []
        elif isinstance(value_cols, list):
            columns = [str(col) for col in value_cols if col in df.columns]
        else:
            numeric = df.select_dtypes(include="number")
            if numeric.empty:
                return PluginResult("skipped", "No numeric columns", {}, [], [], None)
            columns = list(numeric.columns)
        if not columns:
            return PluginResult("skipped", "No value columns", {}, [], [], None)

        forecast_window = int(ctx.settings.get("forecast_window", 5))
        calib_window = int(ctx.settings.get("calib_window", 10))
        alpha = float(ctx.settings.get("alpha", 0.1))
        alarm_rate_window = int(ctx.settings.get("alarm_rate_window", 10))
        alarm_rate_threshold = float(ctx.settings.get("alarm_rate_threshold", 0.4))

        changepoints = []
        for col in columns:
            series = df[col].dropna().to_numpy()
            n = len(series)
            if n < max(forecast_window + 1, calib_window + 1):
                continue
            if not np.isfinite(np.nanstd(series)) or np.nanstd(series) == 0:
                continue
            scores = []
            anomalies = []
            for i in range(forecast_window, n):
                forecast = series[i - forecast_window : i].mean()
                score = abs(series[i] - forecast)
                scores.append(score)
                if i >= calib_window:
                    calib_scores = np.array(scores[-calib_window:])
                    thresh = np.quantile(calib_scores, 1 - alpha)
                    anomalies.append(score > thresh)
                else:
                    anomalies.append(False)

            for i in range(alarm_rate_window, len(anomalies)):
                window = anomalies[i - alarm_rate_window : i]
                rate = sum(window) / alarm_rate_window
                if rate > alarm_rate_threshold:
                    changepoints.append(
                        {
                            "kind": "changepoint",
                            "column": col,
                            "index": int(i),
                            "time": int(i),
                            "score": float(rate),
                        }
                    )
                    break

        artifacts_dir = ctx.artifacts_dir("analysis_online_conformal_changepoint")
        alerts_path = artifacts_dir / "alerts.json"
        write_json(alerts_path, changepoints)
        artifacts = [
            PluginArtifact(
                path=str(alerts_path.relative_to(ctx.run_dir)),
                type="json",
                description="Changepoints",
            )
        ]
        return PluginResult(
            "ok",
            "Detected changepoints",
            {
                "count": len(changepoints),
                "columns_scanned": len(columns),
                "columns_with_findings": len({c["column"] for c in changepoints}),
            },
            changepoints,
            artifacts,
            None,
        )
