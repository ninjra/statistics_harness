from __future__ import annotations

from plugins.analysis_vendor_influence_breadth_v1.plugin import Plugin
from tests.plugins.openplanter_pack_test_utils import make_openplanter_context, run_openplanter_transforms


def test_analysis_vendor_influence_breadth_v1_smoke(run_dir) -> None:
    ctx = make_openplanter_context(run_dir)
    run_openplanter_transforms(ctx)
    ctx.settings = {}
    result = Plugin().run(ctx)
    assert result.status == "ok"
