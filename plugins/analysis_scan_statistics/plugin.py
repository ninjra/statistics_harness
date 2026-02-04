from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.stat_controls import benjamini_hochberg, confidence_from_p
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

        max_rows = int(ctx.settings.get("max_rows", 5000))
        min_window = int(ctx.settings.get("min_window", 5))
        max_window = int(ctx.settings.get("max_window", 20))
        n_perm = int(ctx.settings.get("n_permutations", 100))
        max_p_value = float(ctx.settings.get("max_p_value", 0.05))
        rng = np.random.default_rng(ctx.run_seed)

        results = []
        sampled_columns = 0
        for col in columns:
            series = df[col].dropna().to_numpy()
            n = len(series)
            if n < min_window:
                continue
            if not np.isfinite(np.nanstd(series)) or np.nanstd(series) == 0:
                continue
            sampled = False
            if n > max_rows > 0:
                idx = np.linspace(0, n - 1, max_rows, dtype=int)
                series = series[idx]
                n = len(series)
                sampled = True
            if sampled:
                sampled_columns += 1

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

            if best and p_value <= max_p_value:
                finding = {
                    "kind": "cluster",
                    "column": col,
                    "start": int(best["start"]),
                    "end": int(best["end"]),
                    "score": float(best["score"]),
                    "p_value": p_value,
                }
                q_values = benjamini_hochberg([finding["p_value"]])
                if q_values:
                    finding["q_value"] = q_values[0]
                    finding["confidence"] = confidence_from_p(q_values[0])
                results.append(finding)

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
            {
                "count": len(results),
                "columns_scanned": len(columns),
                "columns_with_findings": len({r["column"] for r in results}),
                "sampled_columns": sampled_columns,
            },
            results,
            artifacts,
            None,
        )
