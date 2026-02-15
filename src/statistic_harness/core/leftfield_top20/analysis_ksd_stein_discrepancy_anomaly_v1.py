from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginResult

from .common import artifact, build_config, degraded, finding, prepare_data, rbf_kernel, split_early_late

PLUGIN_ID = "analysis_ksd_stein_discrepancy_anomaly_v1"


def run(ctx) -> PluginResult:
    config = build_config(ctx)
    prepared = prepare_data(ctx, config)
    if isinstance(prepared, PluginResult):
        prepared.summary = f"{PLUGIN_ID}: {prepared.summary}"
        return prepared
    x = prepared.matrix
    a, b = split_early_late(x)
    if a.shape[0] < 20 or b.shape[0] < 20:
        return degraded(PLUGIN_ID, "Need larger windows for KSD", {"rows_used": int(x.shape[0])})
    mu = a.mean(axis=0)
    cov = np.cov(a.T) + np.eye(a.shape[1]) * 1e-6
    inv_cov = np.linalg.pinv(cov)
    s_a = (a - mu) @ inv_cov
    s_b = (b - mu) @ inv_cov
    k_a = rbf_kernel(a)
    k_b = rbf_kernel(b)
    ksd_a = float(np.mean(np.sum(s_a * s_a, axis=1)) * np.mean(k_a))
    ksd_b = float(np.mean(np.sum(s_b * s_b, axis=1)) * np.mean(k_b))
    ratio = float(ksd_b / max(1e-9, ksd_a))
    artifacts = [artifact(ctx, PLUGIN_ID, "ksd_anomaly_scores.json", {"baseline_ksd": ksd_a, "recent_ksd": ksd_b, "ratio": ratio})]
    findings = [finding(PLUGIN_ID, "Recent distribution fit worsened" if ratio > 1.0 else "Recent distribution fit stable", "When KSD ratio rises, focus on recent process/config changes that altered core numeric behavior.", ratio, {"ksd_ratio": ratio})]
    return PluginResult("ok", "Computed KSD anomaly ratio", {"ksd_ratio": ratio, "baseline_ksd": ksd_a, "recent_ksd": ksd_b}, findings, artifacts, None)
