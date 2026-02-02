from __future__ import annotations


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
        max_cols = int(ctx.settings.get("max_cols", 20))
        numeric = numeric.iloc[:, :max_cols]
        corr = numeric.corr().fillna(0.0).to_numpy()
        nodes = list(numeric.columns)
        edges = []
        edges_compact = []
        threshold = float(ctx.settings.get("weight_threshold", 0.2))
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    continue
                weight = corr[i, j]
                if abs(weight) >= threshold:
                    edges.append(
                        {
                            "source": nodes[i],
                            "target": nodes[j],
                            "weight": float(weight),
                        }
                    )
                    edges_compact.append([i, j, float(weight)])
        graph = {"nodes": nodes, "edges": edges, "edges_compact": edges_compact}
        artifacts_dir = ctx.artifacts_dir("analysis_notears_linear")
        graph_path = artifacts_dir / "graph.json"
        write_json(graph_path, graph)
        findings = [
            {
                "kind": "graph_edge",
                "source": e["source"],
                "target": e["target"],
                "weight": e["weight"],
            }
            for e in edges
        ]
        artifacts = [
            PluginArtifact(
                path=str(graph_path.relative_to(ctx.run_dir)),
                type="json",
                description="Graph",
            )
        ]
        return PluginResult(
            "ok", "Estimated graph", {"edges": len(edges)}, findings, artifacts, None
        )
