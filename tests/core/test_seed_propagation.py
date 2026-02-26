from __future__ import annotations

import json
from pathlib import Path

from statistic_harness.core.pipeline import Pipeline


def test_seed_propagation_openplanter_pack_is_deterministic(tmp_path, monkeypatch):
    appdata_1 = tmp_path / "appdata_1"
    appdata_2 = tmp_path / "appdata_2"
    fixture = Path("tests/fixtures/synth_linear.csv")
    plugin_ids = [
        "profile_basic",
        "transform_entity_resolution_map_v1",
        "transform_cross_dataset_link_graph_v1",
        "analysis_vendor_politician_timing_permutation_v1",
        "report_bundle",
    ]

    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata_1))
    run_id_1 = Pipeline(appdata_1, Path("plugins")).run(fixture, plugin_ids, {}, 4242)
    manifest_1 = json.loads((appdata_1 / "runs" / run_id_1 / "run_manifest.json").read_text(encoding="utf-8"))

    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata_2))
    run_id_2 = Pipeline(appdata_2, Path("plugins")).run(fixture, plugin_ids, {}, 4242)
    manifest_2 = json.loads((appdata_2 / "runs" / run_id_2 / "run_manifest.json").read_text(encoding="utf-8"))

    assert manifest_1["run_seed"] == 4242
    assert manifest_2["run_seed"] == 4242
    assert manifest_1["plugins"] == manifest_2["plugins"]

