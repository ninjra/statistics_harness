from __future__ import annotations

import json
from pathlib import Path

from statistic_harness.core.pipeline import Pipeline


def test_openplanter_cross_dataset_pack_pipeline_integration(tmp_path, monkeypatch) -> None:
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata))
    fixture = Path("tests/fixtures/synth_linear.csv")
    plugin_ids = [
        "profile_basic",
        "transform_entity_resolution_map_v1",
        "transform_cross_dataset_link_graph_v1",
        "analysis_bundled_donations_v1",
        "analysis_contribution_limit_flags_v1",
        "analysis_vendor_influence_breadth_v1",
        "analysis_vendor_politician_timing_permutation_v1",
        "analysis_red_flags_refined_v1",
        "report_evidence_index_v1",
        "report_bundle",
    ]
    pipeline = Pipeline(appdata, Path("plugins"))
    run_id = pipeline.run(fixture, plugin_ids, settings={}, run_seed=5150)
    run_dir = appdata / "runs" / run_id
    report_json = run_dir / "report.json"
    report_md = run_dir / "report.md"
    assert report_json.exists()
    assert report_md.exists()
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert isinstance(report.get("plugins"), dict)
    assert "report_evidence_index_v1" in report["plugins"]

