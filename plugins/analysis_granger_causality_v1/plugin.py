from __future__ import annotations

from statistic_harness.core.stat_plugins.runbook30_surrogates import run_surrogate


class Plugin:
    def run(self, ctx):
        return run_surrogate("analysis_granger_causality_v1", ctx)
