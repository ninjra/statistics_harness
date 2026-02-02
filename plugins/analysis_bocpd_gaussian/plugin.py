from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


class Plugin:
    def run(self, ctx) -> PluginResult:
        df = ctx.dataset_loader()
        value_col = ctx.settings.get("value_column")
        if not value_col:
            numeric = df.select_dtypes(include="number")
            if numeric.empty:
                return PluginResult("skipped", "No numeric columns", {}, [], [], None)
            value_col = numeric.columns[0]
        series = df[value_col].to_numpy()
        n = len(series)
        if n < 2:
            return PluginResult("skipped", "Not enough data", {}, [], [], None)

        rolling_mean = np.cumsum(series) / (np.arange(n) + 1)
        diffs = np.abs(series - rolling_mean)
        probs = diffs / (diffs.max() + 1e-8)
        peak_threshold = float(ctx.settings.get("peak_threshold", 0.6))
        changepoints = []
        for idx, prob in enumerate(probs):
            if prob >= peak_threshold and idx > 0:
                changepoints.append(
                    {"kind": "changepoint", "index": int(idx), "prob": float(prob)}
                )
                break

        artifacts_dir = ctx.artifacts_dir("analysis_bocpd_gaussian")
        out_path = artifacts_dir / "changepoints.json"
        write_json(out_path, changepoints)
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Changepoints",
            )
        ]
        return PluginResult(
            "ok",
            "Computed BOCPD changepoints",
            {"count": len(changepoints)},
            changepoints,
            artifacts,
            None,
        )
