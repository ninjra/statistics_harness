from __future__ import annotations

from statistic_harness.core.stat_plugins.runbook30_surrogates import run_surrogate


class Plugin:
    def run(self, ctx):
        return run_surrogate("analysis_elastic_net_regularized_glm_v1", ctx)
