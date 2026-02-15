from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginResult

from .common import artifact, build_config, degraded, finding, prepare_data

PLUGIN_ID = "analysis_neural_additive_model_nam_v1"


def run(ctx) -> PluginResult:
    config = build_config(ctx)
    prepared = prepare_data(ctx, config)
    if isinstance(prepared, PluginResult):
        prepared.summary = f"{PLUGIN_ID}: {prepared.summary}"
        return prepared
    x = prepared.matrix
    if x.shape[1] < 2:
        return degraded(PLUGIN_ID, "Need >=2 features for NAM proxy", {"cols_used": int(x.shape[1])})
    y = x[:, 0]
    feats = x[:, 1:]
    cols = prepared.numeric_cols[1:]
    bins = max(6, int(config.get("bins") or 24))
    baseline = float(np.mean(y))
    contrib = np.zeros_like(feats)
    feature_ranges = []
    for j in range(feats.shape[1]):
        v = feats[:, j]
        qs = np.quantile(v, np.linspace(0, 1, bins + 1))
        qs = np.unique(qs)
        if len(qs) <= 2:
            continue
        bin_id = np.digitize(v, qs[1:-1], right=False)
        means = {}
        for b in np.unique(bin_id):
            means[int(b)] = float(np.mean(y[bin_id == b]) - baseline)
        contrib[:, j] = np.array([means.get(int(b), 0.0) for b in bin_id], dtype=float)
        feature_ranges.append({"feature": cols[j], "effect_range": float(np.max(contrib[:, j]) - np.min(contrib[:, j]))})
    pred = baseline + np.sum(contrib, axis=1) / max(1, feats.shape[1])
    ss_tot = float(np.sum((y - baseline) ** 2))
    ss_res = float(np.sum((y - pred) ** 2))
    r2 = float(1.0 - (ss_res / max(1e-9, ss_tot)))
    feature_ranges.sort(key=lambda r: float(r["effect_range"]), reverse=True)
    top = feature_ranges[: int(config.get("top_k") or 10)]
    artifacts = [artifact(ctx, PLUGIN_ID, "nam_proxy_effect_curves.json", {"r2_proxy": r2, "feature_effect_ranges": top})]
    findings = []
    if top:
        t0 = top[0]
        findings.append(finding(PLUGIN_ID, f"Strong additive effect feature: {t0['feature']}", "This feature has the largest additive effect range; evaluate thresholding/policy changes here first.", float(t0["effect_range"]), t0))
    return PluginResult("ok", "Computed NAM-style additive effect proxy", {"r2_proxy": r2, "features_modeled": len(top)}, findings, artifacts, None)
