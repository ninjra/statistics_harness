from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


def _pick_numeric_column(df: pd.DataFrame, preferred: str | None) -> str | None:
    if preferred and preferred in df.columns and pd.api.types.is_numeric_dtype(df[preferred]):
        return preferred
    numeric = [str(col) for col in df.select_dtypes(include="number").columns]
    return numeric[0] if numeric else None


class Plugin:
    def run(self, ctx) -> PluginResult:
        loader = getattr(ctx, "dataset_loader", None)
        if not callable(loader):
            return PluginResult(
                status="skipped",
                summary="dataset_loader unavailable",
                metrics={},
                findings=[],
                artifacts=[],
                references=[],
                debug={"reason": "missing_dataset_loader"},
            )
        df = loader()
        if df is None or len(df) == 0:
            return PluginResult(
                status="skipped",
                summary="Empty dataset",
                metrics={"rows": 0},
                findings=[],
                artifacts=[],
                references=[],
                debug={},
            )

        value_column = _pick_numeric_column(df, ctx.settings.get("value_column"))
        if not value_column:
            return PluginResult(
                status="skipped",
                summary="No numeric column for changepoint detection",
                metrics={"rows": int(len(df)), "columns": int(len(df.columns))},
                findings=[],
                artifacts=[],
                references=[],
                debug={},
            )

        series = pd.to_numeric(df[value_column], errors="coerce").dropna().to_numpy(dtype=float)
        min_points = int(ctx.settings.get("min_points", 24))
        window = int(ctx.settings.get("window", 8))
        if len(series) < max(min_points, (2 * window) + 1):
            return PluginResult(
                status="skipped",
                summary="Not enough points for changepoint detection",
                metrics={"rows_used": int(len(series))},
                findings=[],
                artifacts=[],
                references=[],
                debug={"value_column": value_column},
            )

        deltas: list[tuple[int, float]] = []
        for idx in range(window, len(series) - window):
            left = float(np.mean(series[idx - window : idx]))
            right = float(np.mean(series[idx : idx + window]))
            deltas.append((idx, abs(right - left)))
        best_idx, best_delta = max(deltas, key=lambda x: x[1])
        denom = max(1e-9, max(delta for _, delta in deltas))
        score = float(best_delta / denom)

        finding = {
            "kind": "changepoint",
            "column": value_column,
            "index": int(best_idx),
            "prob": float(round(score, 6)),
        }
        artifact_payload = {
            "schema_version": "changepoints.v1",
            "plugin_id": "changepoint_detection_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "series": value_column,
            "changepoints": [
                {
                    "index": int(best_idx),
                    "score": float(round(score, 6)),
                    "segment_before_mean": float(np.mean(series[max(0, best_idx - window) : best_idx])),
                    "segment_after_mean": float(np.mean(series[best_idx : best_idx + window])),
                }
            ],
        }
        artifacts_dir = ctx.artifacts_dir("changepoint_detection_v1")
        out_path = artifacts_dir / "changepoints.json"
        write_json(out_path, artifact_payload)
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="detected changepoints",
            )
        ]
        return PluginResult(
            status="ok",
            summary="Computed deterministic changepoint candidate",
            metrics={
                "rows_used": int(len(series)),
                "value_column": value_column,
                "changepoints": 1,
            },
            findings=[finding],
            artifacts=artifacts,
            references=[],
            debug={"value_column": value_column, "window": window},
        )
