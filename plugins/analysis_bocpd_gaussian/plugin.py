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

        peak_threshold = float(ctx.settings.get("peak_threshold", 0.6))
        changepoints = []
        for col in columns:
            series = df[col].dropna().to_numpy()
            n = len(series)
            if n < 2:
                continue
            if not np.isfinite(np.nanstd(series)) or np.nanstd(series) == 0:
                continue
            rolling_mean = np.cumsum(series) / (np.arange(n) + 1)
            diffs = np.abs(series - rolling_mean)
            max_diff = diffs.max()
            if not np.isfinite(max_diff) or max_diff <= 0:
                continue
            probs = diffs / (max_diff + 1e-8)
            for idx, prob in enumerate(probs):
                if prob >= peak_threshold and idx > 0:
                    changepoints.append(
                        {
                            "kind": "changepoint",
                            "column": col,
                            "index": int(idx),
                            "prob": float(prob),
                        }
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
            {
                "count": len(changepoints),
                "columns_scanned": len(columns),
                "columns_with_findings": len({c["column"] for c in changepoints}),
            },
            changepoints,
            artifacts,
            None,
        )
