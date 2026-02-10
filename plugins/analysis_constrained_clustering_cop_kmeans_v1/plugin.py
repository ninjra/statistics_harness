from __future__ import annotations

from statistic_harness.core.top20_plugins import run_top20_plugin


class Plugin:
    def run(self, ctx):
        return run_top20_plugin('analysis_constrained_clustering_cop_kmeans_v1', ctx)
