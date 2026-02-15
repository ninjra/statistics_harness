from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginResult

from .common import artifact, build_config, degraded, finding, prepare_data

PLUGIN_ID = "analysis_icp_invariant_causal_prediction_v1"


def run(ctx) -> PluginResult:
    config = build_config(ctx)
    prepared = prepare_data(ctx, config)
    if isinstance(prepared, PluginResult):
        prepared.summary = f"{PLUGIN_ID}: {prepared.summary}"
        return prepared
    x = prepared.matrix
    n, m = x.shape
    if m < 2 or n < 60:
        return degraded(PLUGIN_ID, "Need >=60 rows and >=2 numeric columns", {"rows_used": int(n), "cols_used": int(m)})
    target_idx = int(config.get("target_index") or 0) % m
    y = x[:, target_idx]
    env_count = 4
    env_size = max(1, n // env_count)
    rows = []
    for j in range(m):
        if j == target_idx:
            continue
        corrs = []
        for e in range(env_count):
            lo = e * env_size
            hi = n if e == env_count - 1 else min(n, (e + 1) * env_size)
            if hi - lo < int(config.get("min_env_rows") or 50):
                continue
            corr = float(np.corrcoef(y[lo:hi], x[lo:hi, j])[0, 1])
            if np.isfinite(corr):
                corrs.append(corr)
        if len(corrs) < 2:
            continue
        mean_abs = float(abs(np.mean(corrs)))
        var = float(np.std(corrs))
        invariance = float(mean_abs / max(1e-9, 1.0 + var))
        rows.append({"feature": prepared.numeric_cols[j], "corr_mean_abs": mean_abs, "corr_std": var, "invariance_score": invariance})
    rows.sort(key=lambda r: float(r["invariance_score"]), reverse=True)
    top_k = int(config.get("top_k") or 10)
    rows = rows[:top_k]
    artifacts = [artifact(ctx, PLUGIN_ID, "icp_invariance_screen.json", {"target": prepared.numeric_cols[target_idx], "candidates": rows})]
    findings = []
    if rows:
        r0 = rows[0]
        findings.append(finding(PLUGIN_ID, f"Invariant predictor candidate: {r0['feature']}", "Prioritize this stable driver for controls because it behaves consistently across operating environments.", float(r0["invariance_score"]), r0))
    return PluginResult("ok", f"Evaluated invariance for {len(rows)} features", {"target": prepared.numeric_cols[target_idx], "candidates": len(rows)}, findings, artifacts, None)
