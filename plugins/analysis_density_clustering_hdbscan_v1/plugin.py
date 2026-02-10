from __future__ import annotations

from statistic_harness.core.top20_plugins import run_top20_plugin


class Plugin:
    def run(self, ctx):
        return run_top20_plugin('analysis_density_clustering_hdbscan_v1', ctx)
