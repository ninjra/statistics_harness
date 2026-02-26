from __future__ import annotations

from plugins.transform_cross_dataset_link_graph_v1.plugin import Plugin
from tests.plugins.openplanter_pack_test_utils import make_openplanter_context


def test_transform_cross_dataset_link_graph_v1_smoke(run_dir) -> None:
    ctx = make_openplanter_context(run_dir)
    ctx.settings = {}
    result = Plugin().run(ctx)
    assert result.status == "ok"

