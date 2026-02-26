from __future__ import annotations

from plugins.transform_cross_dataset_link_graph_v1.plugin import Plugin
from tests.plugins.openplanter_pack_test_utils import make_openplanter_context


def test_cross_dataset_pack_resource_budget_smoke(run_dir) -> None:
    ctx = make_openplanter_context(run_dir)
    ctx.settings = {"batch_size": 1000}
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert isinstance(result.budget, dict)

