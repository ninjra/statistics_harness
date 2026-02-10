from __future__ import annotations

from statistic_harness.core.top20_plugins import run_top20_plugin


class Plugin:
    def run(self, ctx):
        return run_top20_plugin('analysis_similarity_graph_spectral_clustering_v1', ctx)
