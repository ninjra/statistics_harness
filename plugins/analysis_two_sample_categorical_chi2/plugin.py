from __future__ import annotations

from statistic_harness.core.stat_plugins.registry import run_plugin


class Plugin:
    def run(self, ctx):
        return run_plugin("analysis_two_sample_categorical_chi2", ctx)
