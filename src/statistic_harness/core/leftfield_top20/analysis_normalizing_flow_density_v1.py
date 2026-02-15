from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginResult

from .common import artifact, build_config, degraded, finding, prepare_data

PLUGIN_ID = "analysis_normalizing_flow_density_v1"


def run(ctx) -> PluginResult:
    config = build_config(ctx)
    prepared = prepare_data(ctx, config)
    if isinstance(prepared, PluginResult):
        prepared.summary = f"{PLUGIN_ID}: {prepared.summary}"
        return prepared
    x = prepared.matrix
    n, m = x.shape
    if n < 30:
        return degraded(PLUGIN_ID, "Need >=30 rows for density scoring", {"rows_used": int(n)})
    z = np.zeros_like(x)
    for j in range(m):
        order = np.argsort(x[:, j])
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(n, dtype=float)
        u = (ranks + 0.5) / float(n + 1)
        u = np.clip(u, 1e-6, 1 - 1e-6)
        z[:, j] = np.log(u / (1.0 - u))
    mu = z.mean(axis=0)
    cov = np.cov(z.T) + np.eye(m) * 1e-5
    inv_cov = np.linalg.pinv(cov)
    centered = z - mu
    mahal = np.sum((centered @ inv_cov) * centered, axis=1)
    logp = -0.5 * mahal
    cutoff = float(np.quantile(logp, 0.05))
    out_idx = np.where(logp <= cutoff)[0]
    artifacts = [artifact(ctx, PLUGIN_ID, "normalizing_flow_density_proxy.json", {"logp_cutoff_p05": cutoff, "outlier_row_indices": [int(i) for i in out_idx[:200]], "outlier_count": int(len(out_idx))})]
    findings = [finding(PLUGIN_ID, "Flow-like density outliers identified", "Review low-density rows to uncover unusual process/value combinations affecting close reliability.", float(len(out_idx)), {"outlier_count": int(len(out_idx)), "cutoff": cutoff})]
    return PluginResult("ok", "Computed normalizing-flow-like density proxy", {"outlier_count": int(len(out_idx)), "logp_cutoff_p05": cutoff}, findings, artifacts, None)
