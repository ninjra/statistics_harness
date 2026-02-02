from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.stat_controls import benjamini_hochberg, confidence_from_p
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
        min_window = int(ctx.settings.get("min_window", 5))
        max_window = int(ctx.settings.get("max_window", 20))
        n_perm = int(ctx.settings.get("n_permutations", 100))
        rng = np.random.default_rng(ctx.run_seed)

        best = None
        for window in range(min_window, min(max_window, n) + 1):
            for start in range(0, n - window + 1):
                end = start + window
                segment = series[start:end]
                score = (segment.mean() - series.mean()) / (series.std() + 1e-8)
                if best is None or score > best["score"]:
                    best = {"start": start, "end": end, "score": float(score)}

        p_value = 1.0
        if best:
            perm_scores = []
            for _ in range(n_perm):
                perm = rng.permutation(series)
                segment = perm[best["start"] : best["end"]]
                perm_score = (segment.mean() - perm.mean()) / (perm.std() + 1e-8)
                perm_scores.append(perm_score)
            p_value = float((np.array(perm_scores) >= best["score"]).mean())

        results = []
        if best:
            results.append(
                {
                    "kind": "cluster",
                    "start": int(best["start"]),
                    "end": int(best["end"]),
                    "score": float(best["score"]),
                    "p_value": p_value,
                }
            )
        if results:
            q_values = benjamini_hochberg([r["p_value"] for r in results])
            for finding, q_value in zip(results, q_values):
                finding["q_value"] = q_value
                finding["confidence"] = confidence_from_p(q_value)

        artifacts_dir = ctx.artifacts_dir("analysis_scan_statistics")
        out_path = artifacts_dir / "results.json"
        write_json(out_path, results)
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Scan results",
            )
        ]
        return PluginResult(
            "ok",
            "Computed scan statistics",
            {"count": len(results)},
            results,
            artifacts,
            None,
        )
