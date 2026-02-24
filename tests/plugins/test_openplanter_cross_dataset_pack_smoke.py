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


def test_openplanter_pack_smoke(run_dir: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "vendor_name1": "Acme, Inc.",
                "employer": "Acme Inc",
                "candidate_id": "C001",
                "donation_date": "2025-01-05",
                "amount": 250.0,
                "award_date": "2025-01-10",
            },
            {
                "vendor_name1": "Acme Inc",
                "employer": "Acme Incorporated",
                "candidate_id": "C001",
                "donation_date": "2025-01-05",
                "amount": 260.0,
                "award_date": "2025-01-12",
            },
            {
                "vendor_name1": "Beta LLC",
                "employer": "Beta LLC",
                "candidate_id": "C002",
                "donation_date": "2025-01-11",
                "amount": 110.0,
                "award_date": "2025-01-15",
            },
            {
                "vendor_name1": "Acme Inc",
                "employer": "Acme Inc",
                "candidate_id": "C001",
                "donation_date": "2025-01-05",
                "amount": 270.0,
                "award_date": "2025-01-11",
            },
        ]
    )
    dataset_id = "test_dataset"
    datasets = {
        "contracts": {"dataset_version_id": dataset_id},
        "contributions": {"dataset_version_id": dataset_id},
    }
    ctx = make_context(run_dir, df, settings={}, run_seed=7)

    ctx.settings = {
        "datasets": datasets,
        "fields": [
            {"role": "contracts", "field": "vendor_name1", "entity_type": "org", "key": "vendor"},
            {"role": "contributions", "field": "employer", "entity_type": "org", "key": "employer"},
        ],
    }
    result = EntityMapPlugin().run(ctx)
    assert result.status == "ok"
    assert any(a.path.endswith("entity_map.json") for a in result.artifacts)

    ctx.settings = {
        "datasets": datasets,
        "edges": [
            {
                "left": {"role": "contracts", "field": "vendor_name1"},
                "right": {"role": "contributions", "field": "employer"},
                "relation": "vendor_employer",
            }
        ],
    }
    result = LinkGraphPlugin().run(ctx)
    assert result.status == "ok"
    assert any(a.path.endswith("cross_links.csv") for a in result.artifacts)

    ctx.settings = {
        "contributions_dataset_version_id": dataset_id,
        "employer": "employer",
        "candidate_id": "candidate_id",
        "donation_date": "donation_date",
        "amount": "amount",
        "min_donors": 2,
    }
    result = BundlingPlugin().run(ctx)
    assert result.status == "ok"
    assert any(a.path.endswith("bundling_events.csv") for a in result.artifacts)

    ctx.settings = {
        "contributions_dataset_version_id": dataset_id,
        "donor_id_fields": ["employer", "candidate_id"],
        "amount_field": "amount",
        "date_field": "donation_date",
        "annual_limit": 400.0,
    }
    result = LimitPlugin().run(ctx)
    assert result.status == "ok"
    assert any(a.path.endswith("contribution_limit_flags.csv") for a in result.artifacts)

    ctx.settings = {}
    result = BreadthPlugin().run(ctx)
    assert result.status == "ok"

    ctx.settings = {
        "contracts_dataset_version_id": dataset_id,
        "contributions_dataset_version_id": dataset_id,
        "vendor_field": "vendor_name1",
        "award_date_field": "award_date",
        "candidate_id_field": "candidate_id",
        "donation_date_field": "donation_date",
        "amount_field": "amount",
        "min_donations": 2,
        "n_permutations": 100,
    }
    result = TimingPlugin().run(ctx)
    assert result.status == "ok"
    assert any(a.path.endswith("vendor_politician_timing.csv") for a in result.artifacts)

    ctx.settings = {"max_p_value_for_timing_flag": 0.5, "min_effect_size_for_timing_flag": 0.0}
    result = RedFlagsPlugin().run(ctx)
    assert result.status == "ok"
    assert any(a.path.endswith("red_flags_refined.csv") for a in result.artifacts)

    ctx.settings = {}
    result = EvidencePlugin().run(ctx)
    assert result.status == "ok"
    assert any(a.path.endswith("evidence_index.json") for a in result.artifacts)

