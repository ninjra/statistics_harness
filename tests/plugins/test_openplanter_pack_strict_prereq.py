from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests.conftest import make_context
from plugins.transform_entity_resolution_map_v1.plugin import Plugin as EntityMapPlugin
from plugins.transform_cross_dataset_link_graph_v1.plugin import Plugin as LinkGraphPlugin


def test_openplanter_transforms_fail_closed_when_strict_prerequisites_enabled(run_dir: Path) -> None:
    df = pd.DataFrame([{"x": 1}])
    ctx = make_context(run_dir, df, settings={}, run_seed=9)

    ctx.settings = {"strict_prerequisites": True, "datasets": {}, "fields": [{"role": "contracts", "field": "vendor_name1"}]}
    result = EntityMapPlugin().run(ctx)
    assert result.status == "error"

    ctx.settings = {"strict_prerequisites": True, "datasets": {}, "edges": [{"left": {"role": "contracts", "field": "vendor_name1"}, "right": {"role": "contributions", "field": "employer"}}]}
    result = LinkGraphPlugin().run(ctx)
    assert result.status == "error"

