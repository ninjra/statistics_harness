from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def count_components(self) -> int:
        return len({self.find(i) for i in range(len(self.parent))})


class Plugin:
    def run(self, ctx) -> PluginResult:
        df = ctx.dataset_loader()
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            return PluginResult("skipped", "No numeric columns", {}, [], [], None)
        max_points = ctx.settings.get("max_points")
        if not isinstance(max_points, int) or max_points <= 0:
            max_points = int(len(numeric))
        n_thresholds = int(ctx.settings.get("n_thresholds", 10))
        sample = numeric.head(max_points).to_numpy()
        n = sample.shape[0]
        if n < 2:
            return PluginResult("skipped", "Not enough points", {}, [], [], None)

        dists = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1)
        max_dist = float(dists.max())
        eps_values = np.linspace(0.0, max_dist, n_thresholds)
        beta0 = []
        beta1 = []

        for eps in eps_values:
            uf = UnionFind(n)
            edges = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if dists[i, j] <= eps:
                        uf.union(i, j)
                        edges += 1
            components = uf.count_components()
            beta0.append(int(components))
            beta1.append(int(edges - n + components))

        curves = {"eps": [float(e) for e in eps_values], "beta0": beta0, "beta1": beta1}
        artifacts_dir = ctx.artifacts_dir("analysis_graph_topology_curves")
        out_path = artifacts_dir / "curves.json"
        write_json(out_path, curves)
        findings = []
        if beta1:
            findings.append(
                {"kind": "topology", "metric": "beta1_peak", "value": float(max(beta1))}
            )
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Topology curves",
            )
        ]
        return PluginResult(
            "ok",
            "Computed topology curves",
            {"beta1_peak": max(beta1)},
            findings,
            artifacts,
            None,
        )
