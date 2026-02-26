from __future__ import annotations

from plugins.transform_entity_resolution_map_v1.plugin import Plugin
from tests.plugins.openplanter_pack_test_utils import make_openplanter_context


def test_transform_entity_resolution_map_v1_smoke(run_dir) -> None:
    ctx = make_openplanter_context(run_dir)
    ctx.settings = {}
    result = Plugin().run(ctx)
    assert result.status == "ok"

