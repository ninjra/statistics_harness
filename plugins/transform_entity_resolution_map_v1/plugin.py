from __future__ import annotations

from statistic_harness.core.openplanter_pack import run_openplanter_plugin


class Plugin:
    def run(self, ctx):
        return run_openplanter_plugin('transform_entity_resolution_map_v1', ctx)
