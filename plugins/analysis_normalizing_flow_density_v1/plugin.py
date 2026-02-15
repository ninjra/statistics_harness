from __future__ import annotations

from statistic_harness.core.leftfield_top20.analysis_normalizing_flow_density_v1 import run


class Plugin:
    def run(self, ctx):
        return run(ctx)
