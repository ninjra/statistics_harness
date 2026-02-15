from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginResult

from .common import artifact, build_config, degraded, finding, prepare_data

PLUGIN_ID = "analysis_lingam_causal_discovery_v1"


def run(ctx) -> PluginResult:
    config = build_config(ctx)
    prepared = prepare_data(ctx, config)
    if isinstance(prepared, PluginResult):
        prepared.summary = f"{PLUGIN_ID}: {prepared.summary}"
        return prepared
    x = prepared.matrix
    n, m = x.shape
    if n < 80 or m < 2:
        return degraded(PLUGIN_ID, "Need >=80 rows and >=2 numeric columns", {"rows_used": int(n), "cols_used": int(m)})
    z = (x - x.mean(axis=0, keepdims=True)) / np.maximum(1e-9, x.std(axis=0, keepdims=True))
    kurt = np.mean(z**4, axis=0) - 3.0
    order = list(np.argsort(-np.abs(kurt)))
    edges = []
    for pos, dst in enumerate(order):
        parents = order[:pos]
        if not parents:
            continue
        xp = z[:, parents]
        yp = z[:, dst]
        try:
            coef, *_ = np.linalg.lstsq(xp, yp, rcond=None)
        except Exception:
            continue
        for pidx, c in zip(parents, coef):
            w = float(abs(c))
            if w > 0.05:
                edges.append({"src": prepared.numeric_cols[pidx], "dst": prepared.numeric_cols[dst], "weight": w})
    edges.sort(key=lambda r: float(r["weight"]), reverse=True)
    edges = edges[: int(config.get("top_k") or 12)]
    artifacts = [artifact(ctx, PLUGIN_ID, "lingam_proxy_edges.json", {"ordering": [prepared.numeric_cols[i] for i in order], "edges": edges})]
    findings = []
    if edges:
        e0 = edges[0]
        findings.append(finding(PLUGIN_ID, f"Directional edge candidate: {e0['src']} -> {e0['dst']}", "Treat this direction as a causal hypothesis and validate by targeted interventions or controlled sequencing.", float(e0["weight"]), e0))
    return PluginResult("ok", f"Estimated LiNGAM-style edge hypotheses ({len(edges)} edges)", {"edges": len(edges)}, findings, artifacts, None)
