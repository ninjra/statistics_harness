#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGINS_DIR = REPO_ROOT / "plugins"


@dataclass(frozen=True)
class PluginSpec:
    plugin_id: str
    name: str
    description: str
    defaults: dict[str, Any]
    depends_on: list[str]
    capabilities: list[str]


LEFTFIELD_TOP20: list[PluginSpec] = [
    PluginSpec(
        "analysis_ssa_decomposition_changepoint_v1",
        "SSA Decomposition and Changepoint",
        "Singular spectrum analysis decomposition with changepoint candidates over dominant temporal components.",
        {"max_rows": 15000, "window": 48, "components": 5, "top_k": 10},
        ["profile_eventlog", "transform_normalize_mixed"],
        ["needs_eventlog", "needs_numeric", "needs_timestamp"],
    ),
    PluginSpec(
        "analysis_cur_decomposition_explain_v1",
        "CUR Decomposition Explainability",
        "CUR-style row/column subset selection to explain low-rank structure with interpretable exemplars.",
        {"max_rows": 12000, "max_cols": 24, "rank": 6, "top_k": 10},
        ["profile_basic", "profile_eventlog", "transform_normalize_mixed"],
        ["needs_eventlog", "needs_numeric"],
    ),
    PluginSpec(
        "analysis_hsic_independence_screen_v1",
        "HSIC Nonlinear Dependence Screen",
        "Kernel HSIC dependence screening across numeric feature pairs for nonlinear signal discovery.",
        {"max_rows": 8000, "max_features": 16, "top_k": 20},
        ["profile_basic", "profile_eventlog", "transform_normalize_mixed"],
        ["needs_numeric"],
    ),
    PluginSpec(
        "analysis_icp_invariant_causal_prediction_v1",
        "Invariant Causal Prediction Screen",
        "Environment-invariance screening to identify candidate causal parent variables with stable effects.",
        {"max_rows": 12000, "max_features": 14, "target_index": 0, "min_env_rows": 100, "top_k": 10},
        ["profile_eventlog", "transform_normalize_mixed"],
        ["needs_numeric", "needs_timestamp"],
    ),
    PluginSpec(
        "analysis_lingam_causal_discovery_v1",
        "LiNGAM-style Causal Discovery",
        "ICA-based linear non-Gaussian causal proxy graph with ranked directional edges.",
        {"max_rows": 10000, "max_features": 12, "top_k": 12},
        ["profile_eventlog", "transform_normalize_mixed"],
        ["needs_numeric"],
    ),
    PluginSpec(
        "analysis_frequent_directions_cov_sketch_v1",
        "Frequent Directions Covariance Sketch",
        "Streaming covariance sketch to track dominant covariance directions with bounded memory.",
        {"max_rows": 50000, "max_features": 20, "sketch_size": 8, "top_k": 10},
        ["profile_basic", "profile_eventlog", "transform_normalize_mixed"],
        ["needs_numeric"],
    ),
    PluginSpec(
        "analysis_dmd_koopman_modes_v1",
        "Dynamic Mode Decomposition Koopman Modes",
        "DMD decomposition over multivariate trajectories to expose dominant temporal modes and drift rates.",
        {"max_rows": 12000, "max_features": 12, "rank": 6, "top_k": 8},
        ["profile_eventlog", "transform_normalize_mixed"],
        ["needs_eventlog", "needs_numeric", "needs_timestamp"],
    ),
    PluginSpec(
        "analysis_diffusion_maps_manifold_v1",
        "Diffusion Maps Manifold Drift",
        "Diffusion map embedding and eigengap diagnostics for manifold drift and structure changes.",
        {"max_rows": 5000, "max_features": 12, "n_components": 4, "top_k": 10},
        ["profile_basic", "profile_eventlog", "transform_normalize_mixed"],
        ["needs_numeric"],
    ),
    PluginSpec(
        "analysis_sinkhorn_ot_drift_v1",
        "Sinkhorn OT Drift",
        "Entropic OT/Sinkhorn divergence between temporal windows to quantify distribution shift.",
        {"max_rows": 8000, "max_features": 8, "bins": 32, "iterations": 80, "epsilon": 0.05, "top_k": 10},
        ["profile_eventlog", "transform_normalize_mixed"],
        ["needs_numeric", "needs_timestamp"],
    ),
    PluginSpec(
        "analysis_knn_graph_two_sample_test_v1",
        "kNN Graph Two-sample Test",
        "Friedman-Rafsky-style kNN graph test across temporal cohorts with permutation p-values.",
        {"max_rows": 6000, "max_features": 12, "k": 7, "permutations": 200},
        ["profile_eventlog", "transform_normalize_mixed"],
        ["needs_numeric", "needs_timestamp"],
    ),
    PluginSpec(
        "analysis_ksd_stein_discrepancy_anomaly_v1",
        "KSD Stein Discrepancy Anomaly",
        "Kernelized Stein discrepancy over baseline vs recent samples for fit/anomaly diagnostics.",
        {"max_rows": 6000, "max_features": 10, "top_k": 8},
        ["profile_eventlog", "transform_normalize_mixed"],
        ["needs_numeric", "needs_timestamp"],
    ),
    PluginSpec(
        "analysis_pc_algorithm_causal_graph_v1",
        "PC Algorithm Causal Skeleton",
        "Constraint-based causal skeleton approximation using correlation and partial-correlation pruning.",
        {"max_rows": 9000, "max_features": 10, "alpha": 0.05, "top_k": 12},
        ["profile_eventlog", "transform_normalize_mixed"],
        ["needs_numeric"],
    ),
    PluginSpec(
        "analysis_ges_score_based_causal_v1",
        "GES-style Score-based Causal Graph",
        "Greedy score-based DAG edge search using BIC-like improvement criterion.",
        {"max_rows": 9000, "max_features": 10, "max_edges": 18, "top_k": 12},
        ["profile_eventlog", "transform_normalize_mixed"],
        ["needs_numeric"],
    ),
    PluginSpec(
        "analysis_phate_trajectory_embedding_v1",
        "PHATE-like Trajectory Embedding",
        "Diffusion-potential trajectory embedding approximation for process-state progression analysis.",
        {"max_rows": 5000, "max_features": 10, "n_components": 3, "top_k": 8},
        ["profile_eventlog", "transform_normalize_mixed"],
        ["needs_eventlog", "needs_numeric", "needs_timestamp"],
    ),
    PluginSpec(
        "analysis_node2vec_graph_embedding_drift_v1",
        "Node2Vec-style Graph Embedding Drift",
        "Random-walk graph embedding approximation to detect role and hotspot drift in process transition graphs.",
        {"max_nodes": 120, "walk_length": 12, "walks_per_node": 16, "embedding_dim": 8, "top_k": 12},
        ["profile_eventlog", "transform_normalize_mixed"],
        ["needs_eventlog", "needs_timestamp"],
    ),
    PluginSpec(
        "analysis_tensor_cp_parafac_decomp_v1",
        "Tensor CP PARAFAC Decomposition",
        "CP/PARAFAC tensor decomposition over time-process-feature cube for latent factor diagnostics.",
        {"max_rows": 12000, "rank": 3, "iterations": 40, "top_k": 10},
        ["profile_eventlog", "transform_normalize_mixed"],
        ["needs_eventlog", "needs_numeric", "needs_timestamp"],
    ),
    PluginSpec(
        "analysis_symbolic_regression_gp_v1",
        "Symbolic Regression GP Approximation",
        "Symbolic expression search approximation for interpretable formula candidates over key targets.",
        {"max_rows": 8000, "max_features": 8, "top_k": 8},
        ["profile_eventlog", "transform_normalize_mixed"],
        ["needs_numeric"],
    ),
    PluginSpec(
        "analysis_normalizing_flow_density_v1",
        "Normalizing Flow Density Approximation",
        "Flow-like monotonic transform and density scoring approximation for anomaly ranking.",
        {"max_rows": 8000, "max_features": 8, "quantiles": 32, "top_k": 12},
        ["profile_eventlog", "transform_normalize_mixed"],
        ["needs_numeric"],
    ),
    PluginSpec(
        "analysis_tabpfn_foundation_tabular_v1",
        "TabPFN-style Few-shot Tabular Baseline",
        "Foundation-style few-shot tabular prediction proxy using deterministic ensemble baselines and uncertainty spread.",
        {"max_rows": 7000, "max_features": 12, "n_bootstrap": 12, "top_k": 10},
        ["profile_eventlog", "transform_normalize_mixed"],
        ["needs_numeric"],
    ),
    PluginSpec(
        "analysis_neural_additive_model_nam_v1",
        "Neural Additive Model Approximation",
        "Additive nonlinear effect modeling approximation with per-feature contribution curves.",
        {"max_rows": 7000, "max_features": 10, "bins": 24, "top_k": 10},
        ["profile_eventlog", "transform_normalize_mixed"],
        ["needs_numeric"],
    ),
]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    _write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_yaml(path: Path, payload: Any) -> None:
    _write(path, yaml.safe_dump(payload, sort_keys=False))


def _config_schema(spec: PluginSpec) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for key, value in spec.defaults.items():
        if isinstance(value, bool):
            t = "boolean"
        elif isinstance(value, int):
            t = "integer"
        elif isinstance(value, float):
            t = "number"
        elif isinstance(value, list):
            t = "array"
        else:
            t = "string"
        props[key] = {"type": [t, "null"], "default": value}
    props.setdefault(
        "exclude_processes",
        {"type": ["array", "null"], "items": {"type": "string"}, "default": None},
    )
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": props,
        "additionalProperties": True,
    }


def _load_output_schema_template() -> dict[str, Any]:
    src = PLUGINS_DIR / "analysis_busy_period_segmentation_v2" / "output.schema.json"
    return json.loads(src.read_text(encoding="utf-8"))


def main() -> int:
    output_schema = _load_output_schema_template()
    for spec in LEFTFIELD_TOP20:
        plugin_dir = PLUGINS_DIR / spec.plugin_id
        plugin_dir.mkdir(parents=True, exist_ok=True)
        _write_yaml(
            plugin_dir / "plugin.yaml",
            {
                "id": spec.plugin_id,
                "name": spec.name,
                "version": "0.1.0",
                "type": "analysis",
                "entrypoint": "plugin.py:Plugin",
                "depends_on": spec.depends_on,
                "settings": {
                    "description": spec.description,
                    "defaults": spec.defaults,
                },
                "capabilities": list(spec.capabilities),
                "config_schema": "config.schema.json",
                "output_schema": "output.schema.json",
                "sandbox": {"no_network": True, "fs_allowlist": ["appdata", "plugins", "run_dir"]},
            },
        )
        _write(
            plugin_dir / "plugin.py",
            (
                "from __future__ import annotations\n\n"
                f"from statistic_harness.core.leftfield_top20.{spec.plugin_id} import run\n\n\n"
                "class Plugin:\n"
                "    def run(self, ctx):\n"
                "        return run(ctx)\n"
            ),
        )
        _write_json(plugin_dir / "config.schema.json", _config_schema(spec))
        _write_json(plugin_dir / "output.schema.json", output_schema)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
