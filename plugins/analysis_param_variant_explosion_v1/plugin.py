from __future__ import annotations

from statistic_harness.core.stat_plugins.registry import run_plugin


class Plugin:
    def run(self, ctx):
        return run_plugin("analysis_param_variant_explosion_v1", ctx)

