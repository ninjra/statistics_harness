from pathlib import Path

from statistic_harness.core.plugin_manager import PluginManager


def test_plugin_discovery():
    manager = PluginManager(Path("plugins"))
    specs = manager.discover()
    ids = {spec.plugin_id for spec in specs}
    expected = {
        "ingest_tabular",
        "profile_basic",
        "profile_eventlog",
        "planner_basic",
        "transform_template",
        "transform_normalize_mixed",
        "analysis_conformal_feature_prediction",
        "analysis_online_conformal_changepoint",
        "analysis_gaussian_knockoffs",
        "analysis_knockoff_wrapper_rf",
        "analysis_notears_linear",
        "analysis_bocpd_gaussian",
        "analysis_scan_statistics",
        "analysis_graph_topology_curves",
        "analysis_dp_gmm",
        "analysis_gaussian_copula_shift",
        "analysis_close_cycle_contention",
        "analysis_process_sequence",
        "report_bundle",
        "llm_prompt_builder",
    }
    assert expected.issubset(ids)
    for spec in specs:
        assert spec.config_schema.exists()
        assert spec.output_schema.exists()
        assert spec.sandbox
