from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginResult

from .common import artifact, build_config, degraded, finding, prepare_data, rbf_kernel

PLUGIN_ID = "analysis_diffusion_maps_manifold_v1"


def run(ctx) -> PluginResult:
    config = build_config(ctx)
    prepared = prepare_data(ctx, config)
    if isinstance(prepared, PluginResult):
        prepared.summary = f"{PLUGIN_ID}: {prepared.summary}"
        return prepared
    x = prepared.matrix
    if x.shape[0] < 20:
        return degraded(PLUGIN_ID, "Need >=20 rows for diffusion map", {"rows_used": int(x.shape[0])})
    cap_rows = int(config.get("max_rows_for_diffusion") or 450)
    sampled_rows = int(x.shape[0])
    if x.shape[0] > cap_rows:
        seed = int(getattr(ctx, "run_seed", 0) or 0)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(x.shape[0], size=cap_rows, replace=False))
        x = x[idx, :]
        sampled_rows = int(x.shape[0])
    k = rbf_kernel(x)
    p = k / np.maximum(1e-9, k.sum(axis=1, keepdims=True))
    eigvals, eigvecs = np.linalg.eig(p)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    n_components = max(2, min(int(config.get("n_components") or 4), eigvecs.shape[1]))
    emb = eigvecs[:, 1 : n_components + 1]
    eigengap = float(eigvals[1] - eigvals[2]) if eigvals.size > 2 else 0.0
    artifacts = [artifact(ctx, PLUGIN_ID, "diffusion_maps_embedding.json", {"eigenvalues": eigvals[:10].tolist(), "eigengap": eigengap, "embedding_sample": emb[:200].tolist(), "rows_sampled_for_kernel": sampled_rows})]
    findings = [finding(PLUGIN_ID, "Diffusion manifold structure extracted", "Large eigengap suggests stable operating manifolds; monitor manifold drift for early warning.", eigengap, {"eigengap": eigengap})]
    return PluginResult("ok", "Computed diffusion maps manifold embedding", {"rows_used": int(x.shape[0]), "rows_sampled_for_kernel": sampled_rows, "eigengap": eigengap}, findings, artifacts, None)
