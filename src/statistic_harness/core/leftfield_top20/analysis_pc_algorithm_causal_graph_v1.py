from __future__ import annotations

import math

import numpy as np

from statistic_harness.core.types import PluginResult

from .common import artifact, build_config, degraded, finding, prepare_data

PLUGIN_ID = "analysis_pc_algorithm_causal_graph_v1"


def run(ctx) -> PluginResult:
    config = build_config(ctx)
    prepared = prepare_data(ctx, config)
    if isinstance(prepared, PluginResult):
        prepared.summary = f"{PLUGIN_ID}: {prepared.summary}"
        return prepared
    x = prepared.matrix
    n, m = x.shape
    if m < 3:
        return degraded(PLUGIN_ID, "Need >=3 features for PC skeleton", {"cols_used": int(m)})
    corr = np.corrcoef(x, rowvar=False)
    alpha = float(config.get("alpha") or 0.05)
    thresh = min(0.8, max(0.1, 0.35 - 0.15 * math.log10(max(10.0, float(n)))))
    edges = {(i, j) for i in range(m) for j in range(i + 1, m) if abs(float(corr[i, j])) >= thresh}
    drop = set()
    for i, j in list(edges):
        for k in range(m):
            if k in {i, j}:
                continue
            num = corr[i, j] - corr[i, k] * corr[j, k]
            den = math.sqrt(max(1e-9, (1 - corr[i, k] ** 2) * (1 - corr[j, k] ** 2)))
            pcor = float(num / den)
            if abs(pcor) < thresh * (1.0 - alpha):
                drop.add((i, j))
                break
    edges = sorted(edges - drop)
    edge_rows = [{"left": prepared.numeric_cols[i], "right": prepared.numeric_cols[j], "corr": float(corr[i, j])} for i, j in edges]
    edge_rows.sort(key=lambda r: abs(float(r["corr"])), reverse=True)
    edge_rows = edge_rows[: int(config.get("top_k") or 12)]
    artifacts = [artifact(ctx, PLUGIN_ID, "pc_causal_skeleton_proxy.json", {"edge_count": len(edge_rows), "edges": edge_rows})]
    findings = []
    if edge_rows:
        e0 = edge_rows[0]
        findings.append(finding(PLUGIN_ID, f"Strong skeleton edge: {e0['left']} - {e0['right']}", "Use this dependency in root-cause triage and intervention design; validate with controlled changes.", abs(float(e0["corr"])), e0))
    return PluginResult("ok", f"Built PC-style skeleton with {len(edge_rows)} edges", {"edge_count": len(edge_rows)}, findings, artifacts, None)
