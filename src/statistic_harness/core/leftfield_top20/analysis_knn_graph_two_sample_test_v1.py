from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginResult

from .common import HAS_SKLEARN, NearestNeighbors, artifact, build_config, degraded, finding, prepare_data, rng

PLUGIN_ID = "analysis_knn_graph_two_sample_test_v1"


def run(ctx) -> PluginResult:
    config = build_config(ctx)
    prepared = prepare_data(ctx, config)
    if isinstance(prepared, PluginResult):
        prepared.summary = f"{PLUGIN_ID}: {prepared.summary}"
        return prepared
    if not HAS_SKLEARN or NearestNeighbors is None:
        return degraded(PLUGIN_ID, "scikit-learn unavailable for kNN graph test")
    x = prepared.matrix
    n = x.shape[0]
    if n < 40:
        return degraded(PLUGIN_ID, "Need >=40 rows for kNN graph test", {"rows_used": int(n)})
    split = n // 2
    labels = np.zeros(n, dtype=int)
    labels[split:] = 1
    k = max(2, min(int(config.get("k") or 7), n - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(x)
    _, idx = nbrs.kneighbors(x)
    edges = []
    for i in range(n):
        for j in idx[i, 1:]:
            a, b = int(min(i, int(j))), int(max(i, int(j)))
            edges.append((a, b))
    edge_set = sorted(set(edges))
    obs_cross = int(sum(1 for i, j in edge_set if labels[i] != labels[j]))
    perms = int(config.get("permutations") or 200)
    local_rng = rng(ctx, config)
    ge = 1
    for _ in range(perms):
        perm = labels.copy()
        local_rng.shuffle(perm)
        cross = sum(1 for i, j in edge_set if perm[i] != perm[j])
        if cross <= obs_cross:
            ge += 1
    p_value = float(ge / float(perms + 1))
    artifacts = [artifact(ctx, PLUGIN_ID, "knn_graph_two_sample_test.json", {"edge_count": len(edge_set), "observed_cross_edges": obs_cross, "permutations": perms, "p_value": p_value})]
    findings = [finding(PLUGIN_ID, "Graph two-sample shift detected" if p_value < 0.05 else "Graph two-sample shift not significant", "If significant, inspect process cluster boundaries and re-route high-friction transitions.", 1.0 - p_value, {"p_value": p_value, "observed_cross_edges": obs_cross})]
    return PluginResult("ok", "Completed kNN graph two-sample test", {"p_value": p_value, "edge_count": len(edge_set)}, findings, artifacts, None)
