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


TOP20: list[PluginSpec] = [
    PluginSpec(
        "analysis_param_near_duplicate_minhash_v1",
        "Param Near-Duplicate Detection (MinHash/LSH)",
        "Find near-duplicate executions by MinHash+LSH over tokenized params to propose batch/multi-input refactors.",
        {
            "max_processes": 60,
            "max_entities_per_process": 1500,
            "num_perm": 128,
            "lsh_threshold": 0.85,
            "min_cluster_size": 5,
            "ignore_param_keys_regex": "(run id|queue id|seq|slice|timestamp)",
        },
        depends_on=["profile_basic", "profile_eventlog", "transform_normalize_mixed"],
        capabilities=["needs_eventlog", "needs_text"],
    ),
    PluginSpec(
        "analysis_param_near_duplicate_simhash_v1",
        "Param Near-Duplicate Detection (SimHash)",
        "Find near-duplicate executions using SimHash over tokenized params/text; propose batch, cache, or dedupe changes.",
        {
            "max_processes": 60,
            "max_entities_per_process": 3000,
            "fingerprint_bits": 64,
            "max_hamming_distance": 3,
            "min_cluster_size": 5,
            "ignore_param_keys_regex": "(run id|queue id|seq|slice|timestamp)",
        },
        depends_on=["profile_basic", "profile_eventlog", "transform_normalize_mixed"],
        capabilities=["needs_eventlog", "needs_text"],
    ),
    PluginSpec(
        "analysis_frequent_itemsets_fpgrowth_v1",
        "Frequent Param Sets (FP-Growth)",
        "Mine frequent parameter itemsets to identify repeatable recipes and consolidation opportunities.",
        {"min_support": 0.02, "max_itemset_size": 6, "max_item_count": 200},
        depends_on=["profile_basic", "profile_eventlog", "transform_normalize_mixed"],
        capabilities=["needs_eventlog", "needs_text"],
    ),
    PluginSpec(
        "analysis_association_rules_apriori_v1",
        "Association Rules (Apriori)",
        "Generate actionable conditional rules over parameters to simplify variants and remove redundant runs.",
        {"min_support": 0.02, "min_confidence": 0.6, "min_lift": 1.1, "max_item_count": 200, "max_rules": 200},
        depends_on=["profile_basic", "profile_eventlog", "transform_normalize_mixed"],
        capabilities=["needs_eventlog", "needs_text"],
    ),
    PluginSpec(
        "analysis_sequential_patterns_prefixspan_v1",
        "Sequential Patterns (PrefixSpan)",
        "Mine frequent process sequences (case/trace scoped) to propose orchestration macros.",
        {"min_support": 5, "max_pattern_len": 6, "max_cases": 5000, "top_k": 25},
        depends_on=["profile_eventlog", "transform_normalize_mixed"],
        capabilities=["needs_eventlog", "needs_timestamp"],
    ),
    PluginSpec(
        "analysis_sequence_grammar_sequitur_v1",
        "Sequence Grammar Inference (SEQUITUR)",
        "Discover repeated subsequences (“macros”) via grammar inference and propose consolidation into fewer steps.",
        {"max_cases": 2000, "min_rule_uses": 5, "top_k": 25},
        depends_on=["profile_eventlog", "transform_normalize_mixed"],
        capabilities=["needs_eventlog", "needs_timestamp"],
    ),
    PluginSpec(
        "analysis_biclustering_cheng_church_v1",
        "Biclustering (Process×Param Coherence)",
        "Find coherent process×param blocks that suggest shared batch endpoints or shared caches.",
        {"n_clusters": 8, "max_items": 200, "top_k": 20},
        depends_on=["profile_basic", "profile_eventlog", "transform_normalize_mixed"],
        capabilities=["needs_eventlog", "needs_text"],
    ),
    PluginSpec(
        "analysis_density_clustering_hdbscan_v1",
        "Density Clustering (HDBSCAN)",
        "Cluster executions into variable-density groups to propose batchable sets without choosing k.",
        {"max_entities": 5000, "min_cluster_size": 10, "min_samples": 5, "top_k": 20},
        depends_on=["profile_basic", "profile_eventlog", "transform_normalize_mixed"],
        capabilities=["needs_eventlog", "needs_text"],
    ),
    PluginSpec(
        "analysis_constrained_clustering_cop_kmeans_v1",
        "Constrained Clustering (COP-KMeans)",
        "Apply constraints-aware clustering to propose safe groupings that respect exclude lists and domain constraints.",
        {"k": 8, "max_entities": 5000, "top_k": 20},
        depends_on=["profile_basic", "profile_eventlog", "transform_normalize_mixed"],
        capabilities=["needs_eventlog", "needs_text"],
    ),
    PluginSpec(
        "analysis_dependency_community_louvain_v1",
        "Dependency Community Detection (Louvain)",
        "Detect dependency communities to identify decoupling boundaries and refactor modules.",
        {"top_k": 20, "min_edge_weight": 5},
        depends_on=["profile_eventlog", "transform_normalize_mixed"],
        capabilities=["needs_eventlog"],
    ),
    PluginSpec(
        "analysis_dependency_community_leiden_v1",
        "Dependency Community Detection (Leiden)",
        "Detect higher-quality dependency communities (Leiden) to identify stable module boundaries.",
        {"top_k": 20, "min_edge_weight": 5},
        depends_on=["profile_eventlog", "transform_normalize_mixed"],
        capabilities=["needs_eventlog"],
    ),
    PluginSpec(
        "analysis_similarity_graph_spectral_clustering_v1",
        "Similarity Graph Spectral Clustering",
        "Cluster a similarity graph of processes/params into coherent groups for batching and consolidation.",
        {"n_clusters": 8, "top_k": 20, "min_edge_weight": 5},
        depends_on=["profile_basic", "profile_eventlog", "transform_normalize_mixed"],
        capabilities=["needs_eventlog", "needs_text"],
    ),
    PluginSpec(
        "analysis_graph_min_cut_partition_v1",
        "Graph Minimum Cut Partition",
        "Find minimal cut edges between subsystems to propose where to decouple or queue-separate.",
        {"min_edge_weight": 5},
        depends_on=["profile_eventlog", "transform_normalize_mixed"],
        capabilities=["needs_eventlog"],
    ),
    PluginSpec(
        "analysis_distribution_shift_wasserstein_v1",
        "Distribution Shift (Wasserstein/EMD)",
        "Quantify close vs open workload/latency distribution shift to target concrete levers.",
        {"top_k": 25, "bins": 60},
        depends_on=["profile_eventlog", "transform_normalize_mixed"],
        capabilities=["needs_eventlog", "needs_numeric", "needs_timestamp"],
    ),
    PluginSpec(
        "analysis_burst_modeling_hawkes_v1",
        "Burst Modeling (Hawkes-Style)",
        "Model bursts as self-exciting patterns to identify trigger processes and dampening interventions.",
        {"top_k": 20, "bucket_minutes": 60},
        depends_on=["profile_eventlog", "transform_normalize_mixed"],
        capabilities=["needs_eventlog", "needs_timestamp"],
    ),
    PluginSpec(
        "analysis_daily_pattern_alignment_dtw_v1",
        "Daily Pattern Alignment (DTW)",
        "Align daily patterns (close vs non-close) to detect shape anomalies and close-cycle drivers.",
        {"top_k": 20, "bucket_minutes": 60},
        depends_on=["profile_eventlog", "transform_normalize_mixed"],
        capabilities=["needs_eventlog", "needs_timestamp"],
    ),
    PluginSpec(
        "analysis_action_search_simulated_annealing_v1",
        "Action Search (Simulated Annealing)",
        "Search combinations of actions to hit target close window under constraints.",
        {"max_actions": 8, "iterations": 4000, "top_k": 10},
        depends_on=[
            "analysis_actionable_ops_levers_v1",
            "analysis_process_sequence_bottlenecks",
            "analysis_close_cycle_capacity_model",
        ],
        capabilities=["needs_eventlog"],
    ),
    PluginSpec(
        "analysis_action_search_mip_batched_scheduler_v1",
        "Action Search (MIP Batched Scheduler)",
        "Solve a constrained selection of actions with MIP/CP-SAT to produce a concrete plan.",
        {"max_actions": 8, "top_k": 10},
        depends_on=[
            "analysis_actionable_ops_levers_v1",
            "analysis_process_sequence_bottlenecks",
            "analysis_close_cycle_capacity_model",
        ],
        capabilities=["needs_eventlog"],
    ),
    PluginSpec(
        "analysis_discrete_event_queue_simulator_v1",
        "Discrete-Event Queue Simulator",
        "Stress-test action plans via discrete-event simulation and estimate deltas to hit the close-window target.",
        {"sim_hours": 72, "replications": 5},
        depends_on=[
            "analysis_action_search_mip_batched_scheduler_v1",
            "analysis_close_cycle_capacity_model",
        ],
        capabilities=["needs_eventlog"],
    ),
    PluginSpec(
        "analysis_empirical_bayes_shrinkage_v1",
        "Empirical Bayes Shrinkage Rankings",
        "Stable ranking of true worst drivers using shrinkage (James–Stein style de-noising).",
        {"top_k": 25},
        depends_on=["profile_eventlog", "transform_normalize_mixed"],
        capabilities=["needs_eventlog", "needs_numeric"],
    ),
]


def _load_output_schema_template() -> dict[str, Any]:
    src = PLUGINS_DIR / "analysis_busy_period_segmentation_v2" / "output.schema.json"
    return json.loads(src.read_text(encoding="utf-8"))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    _write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_yaml(path: Path, payload: Any) -> None:
    _write(path, yaml.safe_dump(payload, sort_keys=False))


def _config_schema(spec: PluginSpec) -> dict[str, Any]:
    props = {}
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
    props.setdefault("exclude_processes", {"type": ["array", "null"], "items": {"type": "string"}, "default": None})
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": props,
        "additionalProperties": True,
    }


def main() -> int:
    out_schema = _load_output_schema_template()
    for spec in TOP20:
        pdir = PLUGINS_DIR / spec.plugin_id
        pdir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "id": spec.plugin_id,
            "name": spec.name,
            "version": "0.1.0",
            "type": "analysis",
            "entrypoint": "plugin.py:Plugin",
            "depends_on": spec.depends_on,
            "settings": {"description": spec.description, "defaults": spec.defaults},
            "capabilities": list(spec.capabilities),
            "config_schema": "config.schema.json",
            "output_schema": "output.schema.json",
            "sandbox": {"no_network": True, "fs_allowlist": ["appdata", "plugins", "run_dir"]},
        }
        _write_yaml(pdir / "plugin.yaml", manifest)
        _write(
            pdir / "plugin.py",
            (
                "from __future__ import annotations\n\n"
                "from statistic_harness.core.top20_plugins import run_top20_plugin\n\n\n"
                "class Plugin:\n"
                "    def run(self, ctx):\n"
                f"        return run_top20_plugin({spec.plugin_id!r}, ctx)\n"
            ),
        )
        _write_json(pdir / "config.schema.json", _config_schema(spec))
        _write_json(pdir / "output.schema.json", out_schema)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
