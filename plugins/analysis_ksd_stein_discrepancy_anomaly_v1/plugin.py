from __future__ import annotations

from statistic_harness.core.leftfield_top20.analysis_ksd_stein_discrepancy_anomaly_v1 import run


class Plugin:
    def run(self, ctx):
        return run(ctx)
