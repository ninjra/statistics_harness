from __future__ import annotations

from statistic_harness.core.stat_plugins.registry import run_plugin


class Plugin:
    def run(self, ctx):
        return run_plugin("analysis_state_space_kalman_residuals", ctx)
