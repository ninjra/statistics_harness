from __future__ import annotations

from statistic_harness.core.leftfield_top20.analysis_pc_algorithm_causal_graph_v1 import run


class Plugin:
    def run(self, ctx):
        return run(ctx)
