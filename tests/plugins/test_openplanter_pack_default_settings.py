from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests.conftest import make_context

from plugins.analysis_bundled_donations_v1.plugin import Plugin as BundlingPlugin
from plugins.analysis_contribution_limit_flags_v1.plugin import Plugin as LimitPlugin
from plugins.analysis_red_flags_refined_v1.plugin import Plugin as RedFlagsPlugin
from plugins.analysis_vendor_influence_breadth_v1.plugin import Plugin as BreadthPlugin
from plugins.analysis_vendor_politician_timing_permutation_v1.plugin import Plugin as TimingPlugin
from plugins.report_evidence_index_v1.plugin import Plugin as EvidencePlugin
from plugins.transform_cross_dataset_link_graph_v1.plugin import Plugin as LinkGraphPlugin
from plugins.transform_entity_resolution_map_v1.plugin import Plugin as EntityMapPlugin


def test_openplanter_plugins_default_to_ok_without_cross_dataset_config(run_dir: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "vendor_name1": "Acme Inc",
                "employer": "Acme Incorporated",
                "candidate_id": "C001",
                "donation_date": "2025-01-05",
                "award_date": "2025-01-10",
                "amount": 100.0,
            }
        ]
    )
    ctx = make_context(run_dir, df, settings={}, run_seed=9)

    for plugin in (
        EntityMapPlugin(),
        LinkGraphPlugin(),
        BundlingPlugin(),
        LimitPlugin(),
        BreadthPlugin(),
        TimingPlugin(),
        RedFlagsPlugin(),
        EvidencePlugin(),
    ):
        ctx.settings = {}
        result = plugin.run(ctx)
        assert result.status == "ok"

