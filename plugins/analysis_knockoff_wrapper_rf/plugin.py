from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

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
        numeric = numeric.dropna(axis=0, how="any")
        if numeric.empty:
            return PluginResult(
                "skipped", "No complete numeric rows", {}, [], [], None
            )
        target_column = ctx.settings.get("target_column")
        target_columns = [target_column] if target_column else list(numeric.columns)
        rng = np.random.default_rng(ctx.run_seed)
        findings = []
        for target in target_columns:
            if target not in numeric.columns:
                continue
            y = numeric[target].to_numpy()
            X = numeric.drop(columns=[target])
            if X.empty:
                continue
            rf = RandomForestRegressor(
                n_estimators=int(ctx.settings.get("n_estimators", 50)),
                random_state=ctx.run_seed,
            )
            rf.fit(X, y)
            importances = rf.feature_importances_
            knockoff = X.apply(lambda col: rng.permutation(col.to_numpy()))
            rf_knock = RandomForestRegressor(
                n_estimators=int(ctx.settings.get("n_estimators", 50)),
                random_state=ctx.run_seed,
            )
            rf_knock.fit(knockoff, y)
            knock_imp = rf_knock.feature_importances_

            scores = importances - knock_imp
            threshold = np.quantile(scores, 1 - float(ctx.settings.get("fdr_q", 0.1)))
            for feature, score in zip(X.columns, scores):
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

        artifacts_dir = ctx.artifacts_dir("analysis_knockoff_wrapper_rf")
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
            "Computed RF knockoff selection",
            {
                "selected": selected_count,
                "targets_scanned": len(target_columns),
            },
            findings,
            artifacts,
            None,
        )
