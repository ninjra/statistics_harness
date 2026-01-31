from __future__ import annotations

import pandas as pd

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


class Plugin:
    def run(self, ctx) -> PluginResult:
        df = ctx.dataset_loader()
        columns = []
        numeric = df.select_dtypes(include="number")
        for col in df.columns:
            series = df[col]
            entry = {
                "name": col,
                "dtype": str(series.dtype),
                "missing_pct": float(series.isna().mean()),
                "unique": int(series.nunique()),
            }
            if pd.api.types.is_numeric_dtype(series):
                entry.update(
                    {
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "mean": float(series.mean()),
                        "std": float(series.std(ddof=0)),
                    }
                )
            columns.append(entry)

        artifacts = []
        artifacts_dir = ctx.artifacts_dir("profile_basic")
        columns_path = artifacts_dir / "columns.json"
        write_json(columns_path, columns)
        artifacts.append(
            PluginArtifact(path=str(columns_path.relative_to(ctx.run_dir)), type="json", description="Column stats")
        )

        max_cols = int(ctx.settings.get("max_corr_cols", 10))
        if numeric.shape[1] > 1 and numeric.shape[1] <= max_cols:
            corr = numeric.corr()
            corr_path = artifacts_dir / "correlation.csv"
            corr.to_csv(corr_path)
            artifacts.append(
                PluginArtifact(path=str(corr_path.relative_to(ctx.run_dir)), type="csv", description="Correlation")
            )

        return PluginResult(
            status="ok",
            summary="Profiled dataset",
            metrics={"columns": len(columns)},
            findings=[],
            artifacts=artifacts,
            error=None,
        )
