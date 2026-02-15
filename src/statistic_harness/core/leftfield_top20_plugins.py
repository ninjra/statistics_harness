from __future__ import annotations

from typing import Callable

from statistic_harness.core.types import PluginResult

from .leftfield_top20 import analysis_cur_decomposition_explain_v1
from .leftfield_top20 import analysis_diffusion_maps_manifold_v1
from .leftfield_top20 import analysis_dmd_koopman_modes_v1
from .leftfield_top20 import analysis_frequent_directions_cov_sketch_v1
from .leftfield_top20 import analysis_ges_score_based_causal_v1
from .leftfield_top20 import analysis_hsic_independence_screen_v1
from .leftfield_top20 import analysis_icp_invariant_causal_prediction_v1
from .leftfield_top20 import analysis_knn_graph_two_sample_test_v1
from .leftfield_top20 import analysis_ksd_stein_discrepancy_anomaly_v1
from .leftfield_top20 import analysis_lingam_causal_discovery_v1
from .leftfield_top20 import analysis_neural_additive_model_nam_v1
from .leftfield_top20 import analysis_node2vec_graph_embedding_drift_v1
from .leftfield_top20 import analysis_normalizing_flow_density_v1
from .leftfield_top20 import analysis_pc_algorithm_causal_graph_v1
from .leftfield_top20 import analysis_phate_trajectory_embedding_v1
from .leftfield_top20 import analysis_sinkhorn_ot_drift_v1
from .leftfield_top20 import analysis_ssa_decomposition_changepoint_v1
from .leftfield_top20 import analysis_symbolic_regression_gp_v1
from .leftfield_top20 import analysis_tabpfn_foundation_tabular_v1
from .leftfield_top20 import analysis_tensor_cp_parafac_decomp_v1

HANDLERS: dict[str, Callable[..., PluginResult]] = {
    "analysis_ssa_decomposition_changepoint_v1": analysis_ssa_decomposition_changepoint_v1.run,
    "analysis_cur_decomposition_explain_v1": analysis_cur_decomposition_explain_v1.run,
    "analysis_hsic_independence_screen_v1": analysis_hsic_independence_screen_v1.run,
    "analysis_icp_invariant_causal_prediction_v1": analysis_icp_invariant_causal_prediction_v1.run,
    "analysis_lingam_causal_discovery_v1": analysis_lingam_causal_discovery_v1.run,
    "analysis_frequent_directions_cov_sketch_v1": analysis_frequent_directions_cov_sketch_v1.run,
    "analysis_dmd_koopman_modes_v1": analysis_dmd_koopman_modes_v1.run,
    "analysis_diffusion_maps_manifold_v1": analysis_diffusion_maps_manifold_v1.run,
    "analysis_sinkhorn_ot_drift_v1": analysis_sinkhorn_ot_drift_v1.run,
    "analysis_knn_graph_two_sample_test_v1": analysis_knn_graph_two_sample_test_v1.run,
    "analysis_ksd_stein_discrepancy_anomaly_v1": analysis_ksd_stein_discrepancy_anomaly_v1.run,
    "analysis_pc_algorithm_causal_graph_v1": analysis_pc_algorithm_causal_graph_v1.run,
    "analysis_ges_score_based_causal_v1": analysis_ges_score_based_causal_v1.run,
    "analysis_phate_trajectory_embedding_v1": analysis_phate_trajectory_embedding_v1.run,
    "analysis_node2vec_graph_embedding_drift_v1": analysis_node2vec_graph_embedding_drift_v1.run,
    "analysis_tensor_cp_parafac_decomp_v1": analysis_tensor_cp_parafac_decomp_v1.run,
    "analysis_symbolic_regression_gp_v1": analysis_symbolic_regression_gp_v1.run,
    "analysis_normalizing_flow_density_v1": analysis_normalizing_flow_density_v1.run,
    "analysis_tabpfn_foundation_tabular_v1": analysis_tabpfn_foundation_tabular_v1.run,
    "analysis_neural_additive_model_nam_v1": analysis_neural_additive_model_nam_v1.run,
}


def _run_by_id(plugin_id: str, ctx) -> PluginResult:
    handler = HANDLERS.get(plugin_id)
    if handler is None:
        return PluginResult("error", f"Unknown leftfield plugin_id: {plugin_id}", {}, [], [], None)
    return handler(ctx)


def run_analysis_ssa_decomposition_changepoint_v1(ctx) -> PluginResult:
    return _run_by_id("analysis_ssa_decomposition_changepoint_v1", ctx)


def run_analysis_cur_decomposition_explain_v1(ctx) -> PluginResult:
    return _run_by_id("analysis_cur_decomposition_explain_v1", ctx)


def run_analysis_hsic_independence_screen_v1(ctx) -> PluginResult:
    return _run_by_id("analysis_hsic_independence_screen_v1", ctx)


def run_analysis_icp_invariant_causal_prediction_v1(ctx) -> PluginResult:
    return _run_by_id("analysis_icp_invariant_causal_prediction_v1", ctx)


def run_analysis_lingam_causal_discovery_v1(ctx) -> PluginResult:
    return _run_by_id("analysis_lingam_causal_discovery_v1", ctx)


def run_analysis_frequent_directions_cov_sketch_v1(ctx) -> PluginResult:
    return _run_by_id("analysis_frequent_directions_cov_sketch_v1", ctx)


def run_analysis_dmd_koopman_modes_v1(ctx) -> PluginResult:
    return _run_by_id("analysis_dmd_koopman_modes_v1", ctx)


def run_analysis_diffusion_maps_manifold_v1(ctx) -> PluginResult:
    return _run_by_id("analysis_diffusion_maps_manifold_v1", ctx)


def run_analysis_sinkhorn_ot_drift_v1(ctx) -> PluginResult:
    return _run_by_id("analysis_sinkhorn_ot_drift_v1", ctx)


def run_analysis_knn_graph_two_sample_test_v1(ctx) -> PluginResult:
    return _run_by_id("analysis_knn_graph_two_sample_test_v1", ctx)


def run_analysis_ksd_stein_discrepancy_anomaly_v1(ctx) -> PluginResult:
    return _run_by_id("analysis_ksd_stein_discrepancy_anomaly_v1", ctx)


def run_analysis_pc_algorithm_causal_graph_v1(ctx) -> PluginResult:
    return _run_by_id("analysis_pc_algorithm_causal_graph_v1", ctx)


def run_analysis_ges_score_based_causal_v1(ctx) -> PluginResult:
    return _run_by_id("analysis_ges_score_based_causal_v1", ctx)


def run_analysis_phate_trajectory_embedding_v1(ctx) -> PluginResult:
    return _run_by_id("analysis_phate_trajectory_embedding_v1", ctx)


def run_analysis_node2vec_graph_embedding_drift_v1(ctx) -> PluginResult:
    return _run_by_id("analysis_node2vec_graph_embedding_drift_v1", ctx)


def run_analysis_tensor_cp_parafac_decomp_v1(ctx) -> PluginResult:
    return _run_by_id("analysis_tensor_cp_parafac_decomp_v1", ctx)


def run_analysis_symbolic_regression_gp_v1(ctx) -> PluginResult:
    return _run_by_id("analysis_symbolic_regression_gp_v1", ctx)


def run_analysis_normalizing_flow_density_v1(ctx) -> PluginResult:
    return _run_by_id("analysis_normalizing_flow_density_v1", ctx)


def run_analysis_tabpfn_foundation_tabular_v1(ctx) -> PluginResult:
    return _run_by_id("analysis_tabpfn_foundation_tabular_v1", ctx)


def run_analysis_neural_additive_model_nam_v1(ctx) -> PluginResult:
    return _run_by_id("analysis_neural_additive_model_nam_v1", ctx)
