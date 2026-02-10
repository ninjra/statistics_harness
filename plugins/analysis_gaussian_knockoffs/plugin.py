from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


class Plugin:
    def run(self, ctx) -> PluginResult:
        df = ctx.dataset_loader()
        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            return PluginResult(
                "skipped", "Not enough numeric columns", {}, [], [], None
            )
        target_column = ctx.settings.get("target_column")
        target_columns = [target_column] if target_column else list(numeric.columns)

        rng = np.random.default_rng(ctx.run_seed)
        findings = []
        for target in target_columns:
            if target not in numeric.columns:
                continue
            frame = numeric.dropna(axis=0, how="any")
            if frame.empty:
                continue
            y = frame[target]
            X = frame.drop(columns=[target])
            if X.empty:
                continue

            scores = {}
            for col in X.columns:
                corr = np.corrcoef(X[col], y)[0, 1]
                knockoff = rng.permutation(X[col])
                knock_corr = np.corrcoef(knockoff, y)[0, 1]
                if not np.isfinite(corr) or not np.isfinite(knock_corr):
                    scores[col] = 0.0
                else:
                    scores[col] = abs(corr) - abs(knock_corr)

            if not scores:
                continue
            score_values = np.array(list(scores.values()))
            threshold = np.quantile(
                score_values, 1 - float(ctx.settings.get("fdr_q", 0.1))
            )
            for feature, score in scores.items():
                selected = bool(score >= threshold)
                findings.append(
                    {
                        "kind": "feature_discovery",
                        "target": target,
                        "feature": feature,
                        "score": float(score),
                        "selected": selected,
                    }
                )

        artifacts_dir = ctx.artifacts_dir("analysis_gaussian_knockoffs")
        selection_path = artifacts_dir / "selection.json"
        write_json(selection_path, findings)
        artifacts = [
            PluginArtifact(
                path=str(selection_path.relative_to(ctx.run_dir)),
                type="json",
                description="Selection",
            )
        ]
        selected_count = sum(1 for f in findings if f["selected"])
        return PluginResult(
            "ok",
            "Computed knockoff selection",
            {
                "selected": selected_count,
            },
            findings,
            artifacts,
            None,
            debug={"targets_scanned": len(target_columns)},
        )
