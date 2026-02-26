from __future__ import annotations

from plugins.analysis_bundled_donations_v1.plugin import Plugin as BundledPlugin
from plugins.analysis_contribution_limit_flags_v1.plugin import Plugin as LimitPlugin
from plugins.analysis_red_flags_refined_v1.plugin import Plugin as RedFlagPlugin
from plugins.analysis_vendor_politician_timing_permutation_v1.plugin import Plugin as TimingPlugin
from plugins.report_evidence_index_v1.plugin import Plugin
from tests.plugins.openplanter_pack_test_utils import make_openplanter_context, run_openplanter_transforms


def test_report_evidence_index_v1_smoke(run_dir) -> None:
    ctx = make_openplanter_context(run_dir, run_seed=13)
    run_openplanter_transforms(ctx)
    ctx.settings = {}
    assert BundledPlugin().run(ctx).status == "ok"
    assert LimitPlugin().run(ctx).status == "ok"
    ctx.settings = {
        "contracts_dataset_version_id": "test_dataset",
        "contributions_dataset_version_id": "test_dataset",
        "vendor_field": "vendor_name1",
        "award_date_field": "award_date",
        "candidate_id_field": "candidate_id",
        "donation_date_field": "donation_date",
        "amount_field": "amount",
        "min_donations": 1,
        "n_permutations": 200,
    }
    assert TimingPlugin().run(ctx).status == "ok"
    ctx.settings = {}
    assert RedFlagPlugin().run(ctx).status == "ok"
    ctx.settings = {}
    result = Plugin().run(ctx)
    assert result.status == "ok"

