from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


def _pick_two_numeric(df: pd.DataFrame) -> tuple[str | None, str | None]:
    cols = [str(col) for col in df.select_dtypes(include="number").columns]
    if len(cols) < 2:
        return None, None
    return cols[0], cols[1]


class Plugin:
    def run(self, ctx) -> PluginResult:
        loader = getattr(ctx, "dataset_loader", None)
        if not callable(loader):
            return PluginResult("skipped", "dataset_loader unavailable", {}, [], [], None)
        df = loader()
        if df is None or len(df) == 0:
            return PluginResult("skipped", "Empty dataset", {"rows": 0}, [], [], None)

        t_col = ctx.settings.get("treatment_column")
        y_col = ctx.settings.get("outcome_column")
        if not (isinstance(t_col, str) and t_col in df.columns and isinstance(y_col, str) and y_col in df.columns):
            t_guess, y_guess = _pick_two_numeric(df)
            t_col = t_col if isinstance(t_col, str) and t_col in df.columns else t_guess
            y_col = y_col if isinstance(y_col, str) and y_col in df.columns else y_guess
        if not t_col or not y_col:
            return PluginResult(
                "skipped",
                "Need treatment and outcome numeric columns",
                {"rows": int(len(df)), "columns": int(len(df.columns))},
                [],
                [],
                None,
            )

        t = pd.to_numeric(df[t_col], errors="coerce")
        y = pd.to_numeric(df[y_col], errors="coerce")
        valid = t.notna() & y.notna()
        t = t[valid]
        y = y[valid]
        if len(t) < 8:
            return PluginResult(
                "skipped",
                "Not enough valid rows for causal approximation",
                {"rows_used": int(len(t))},
                [],
                [],
                None,
            )

        threshold = ctx.settings.get("treatment_threshold")
        if threshold is None:
            threshold = float(t.median())
        treated = t > float(threshold)
        if int(treated.sum()) == 0 or int((~treated).sum()) == 0:
            return PluginResult(
                "degraded",
                "Treatment split degenerate; effect not identified",
                {"rows_used": int(len(t))},
                [
                    {
                        "kind": "recommendation",
                        "status": "not_identified",
                        "reason": "degenerate_treatment_split",
                    }
                ],
                [],
                None,
            )

        effect = float(y[treated].mean() - y[~treated].mean())
        rng = np.random.default_rng(int(getattr(ctx, "run_seed", 0) or 0))
        shuffled = pd.Series(rng.permutation(treated.to_numpy()))
        placebo = float(y[shuffled].mean() - y[~shuffled].mean())
        assumptions = [
            "No unmeasured confounding (approximation).",
            "Stable treatment assignment threshold.",
            "Outcome column reflects downstream impact signal.",
        ]
        refutations = [
            {
                "name": "placebo_shuffle",
                "status": "pass" if abs(placebo) < abs(effect) else "warn",
                "detail": f"placebo_effect={placebo:.6f}",
            }
        ]
        findings = [
            {
                "kind": "recommendation",
                "treatment_column": t_col,
                "outcome_column": y_col,
                "effect_estimate": round(effect, 6),
                "placebo_effect": round(placebo, 6),
                "assumptions": assumptions,
            }
        ]
        artifact_payload = {
            "schema_version": "causal.v1",
            "plugin_id": "causal_recommendations_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "assumptions": assumptions,
            "effect_estimate": {
                "status": "identified",
                "value": round(effect, 6),
                "method": "difference_in_means_threshold_split",
            },
            "refutations": refutations,
        }
        out_dir = ctx.artifacts_dir("causal_recommendations_v1")
        out_path = out_dir / "causal_recommendations.json"
        write_json(out_path, artifact_payload)
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="causal recommendation artifact",
            )
        ]
        return PluginResult(
            status="ok",
            summary="Computed assumption-explicit causal recommendation",
            metrics={
                "rows_used": int(len(t)),
                "effect_estimate": round(effect, 6),
                "placebo_effect": round(placebo, 6),
            },
            findings=findings,
            artifacts=artifacts,
            references=[],
            debug={"treatment_column": t_col, "outcome_column": y_col},
        )
