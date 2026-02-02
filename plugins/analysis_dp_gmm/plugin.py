from __future__ import annotations

import pandas as pd

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


class Plugin:
    def run(self, ctx) -> PluginResult:
        df = ctx.dataset_loader()
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            return PluginResult("skipped", "No numeric columns", {}, [], [], None)
        values = numeric.to_numpy()
        mean = values.mean(axis=0)
        labels = (values[:, 0] > mean[0]).astype(int)
        assignments = pd.DataFrame({"cluster_id": labels})

        clusters = []
        findings = []
        for cluster_id in sorted(set(labels)):
            cluster_points = values[labels == cluster_id]
            clusters.append(
                {
                    "cluster_id": int(cluster_id),
                    "size": int(cluster_points.shape[0]),
                    "mean": cluster_points.mean(axis=0).tolist(),
                }
            )
            findings.append(
                {
                    "kind": "cluster",
                    "cluster_id": int(cluster_id),
                    "size": int(cluster_points.shape[0]),
                }
            )

        artifacts_dir = ctx.artifacts_dir("analysis_dp_gmm")
        assignments_path = artifacts_dir / "assignments.csv"
        assignments.to_csv(assignments_path, index=False)
        summary_path = artifacts_dir / "summary.json"
        write_json(summary_path, {"clusters": clusters})
        artifacts = [
            PluginArtifact(
                path=str(assignments_path.relative_to(ctx.run_dir)),
                type="csv",
                description="Assignments",
            ),
            PluginArtifact(
                path=str(summary_path.relative_to(ctx.run_dir)),
                type="json",
                description="Summary",
            ),
        ]
        return PluginResult(
            "ok",
            "Computed DP GMM clusters",
            {"clusters": len(clusters)},
            findings,
            artifacts,
            None,
        )
