from pathlib import Path

from statistic_harness.core.plugin_manager import PluginManager


def test_plugin_discovery():
    manager = PluginManager(Path("plugins"))
    specs = manager.discover()
    ids = {spec.plugin_id for spec in specs}
    expected = {
        "ingest_tabular",
        "profile_basic",
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
        "report_bundle",
        "llm_prompt_builder",
    }
    assert expected.issubset(ids)
