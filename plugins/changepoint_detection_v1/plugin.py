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

        all_delta_vals = np.array([d for _, d in deltas], dtype=float)
        mean_delta = float(np.mean(all_delta_vals))
        std_delta = float(np.std(all_delta_vals))
        z_threshold = float(ctx.settings.get("z_threshold", 2.0))

        significant_changepoints: list[dict] = []
        for idx, delta in sorted(deltas, key=lambda x: x[1], reverse=True):
            if std_delta > 1e-9:
                z_score = (delta - mean_delta) / std_delta
            else:
                z_score = 0.0
            if z_score >= z_threshold:
                significant_changepoints.append({
                    "index": int(idx),
                    "z_score": float(round(z_score, 6)),
                    "delta": float(round(delta, 6)),
                    "segment_before_mean": float(np.mean(series[max(0, idx - window) : idx])),
                    "segment_after_mean": float(np.mean(series[idx : idx + window])),
                })

        max_changepoints = int(ctx.settings.get("max_changepoints", 10))
        significant_changepoints = significant_changepoints[:max_changepoints]

        findings = []
        for cp in significant_changepoints:
            findings.append({
                "kind": "changepoint",
                "column": value_column,
                "index": cp["index"],
                "prob": float(round(min(1.0, cp["z_score"] / 5.0), 6)),
                "z_score": cp["z_score"],
            })

        artifact_payload = {
            "schema_version": "changepoints.v1",
            "plugin_id": "changepoint_detection_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "series": value_column,
            "changepoints": significant_changepoints,
            "null_distribution": {
                "mean_delta": float(round(mean_delta, 6)),
                "std_delta": float(round(std_delta, 6)),
                "z_threshold": z_threshold,
            },
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
            summary=f"Detected {len(significant_changepoints)} significant changepoint(s)",
            metrics={
                "rows_used": int(len(series)),
                "value_column": value_column,
                "changepoints": len(significant_changepoints),
            },
            findings=findings,
            artifacts=artifacts,
            references=[],
            debug={"value_column": value_column, "window": window, "z_threshold": z_threshold},
        )
