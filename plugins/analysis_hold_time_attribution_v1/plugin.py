from __future__ import annotations

from statistic_harness.core.stat_plugins.registry import run_plugin


class Plugin:
    def run(self, ctx):
        return run_plugin("analysis_hold_time_attribution_v1", ctx)

