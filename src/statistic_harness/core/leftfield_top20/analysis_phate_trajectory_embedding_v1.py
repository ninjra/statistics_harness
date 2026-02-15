from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginResult

from .common import artifact, build_config, finding, prepare_data, rbf_kernel, safe_svd

PLUGIN_ID = "analysis_phate_trajectory_embedding_v1"


def run(ctx) -> PluginResult:
    config = build_config(ctx)
    prepared = prepare_data(ctx, config)
    if isinstance(prepared, PluginResult):
        prepared.summary = f"{PLUGIN_ID}: {prepared.summary}"
        return prepared
    x = prepared.matrix
    k = rbf_kernel(x)
    p = k / np.maximum(1e-9, k.sum(axis=1, keepdims=True))
    p2 = p @ p
    potential = -np.log(np.maximum(1e-9, p2))
    centered = potential - potential.mean(axis=0, keepdims=True)
    u, s, _ = safe_svd(centered)
    n_components = max(2, min(int(config.get("n_components") or 3), u.shape[1]))
    emb = u[:, :n_components] * s[:n_components]
    steps = np.linalg.norm(np.diff(emb, axis=0), axis=1) if emb.shape[0] > 1 else np.array([0.0])
    volatility = float(np.mean(steps))
    artifacts = [artifact(ctx, PLUGIN_ID, "phate_trajectory_embedding_proxy.json", {"volatility": volatility, "embedding_sample": emb[:200].tolist()})]
    findings = [finding(PLUGIN_ID, "Trajectory embedding indicates transition volatility", "Higher trajectory volatility indicates unstable process progression; focus on transition smoothing and dependency readiness.", volatility, {"volatility": volatility})]
    return PluginResult("ok", "Computed PHATE-like trajectory embedding proxy", {"trajectory_volatility": volatility}, findings, artifacts, None)
