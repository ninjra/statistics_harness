from __future__ import annotations

from plugins.transform_cross_dataset_link_graph_v1.plugin import Plugin as LinkGraphPlugin
from plugins.transform_entity_resolution_map_v1.plugin import Plugin as EntityMapPlugin
from tests.plugins.openplanter_pack_test_utils import make_openplanter_context


def test_cross_dataset_transforms_fail_closed_with_strict_prereq(run_dir) -> None:
    ctx = make_openplanter_context(run_dir)
    ctx.settings = {
        "strict_prerequisites": True,
        "datasets": {},
        "fields": [{"role": "contracts", "field": "vendor_name1"}],
    }
    assert EntityMapPlugin().run(ctx).status == "error"

    ctx.settings = {
        "strict_prerequisites": True,
        "datasets": {},
        "edges": [
            {
                "left": {"role": "contracts", "field": "vendor_name1"},
                "right": {"role": "contributions", "field": "employer"},
            }
        ],
    }
    assert LinkGraphPlugin().run(ctx).status == "error"

