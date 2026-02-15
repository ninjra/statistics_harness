from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginResult

from .common import artifact, build_config, degraded, finding, prepare_data, split_early_late

PLUGIN_ID = "analysis_sinkhorn_ot_drift_v1"


def _sinkhorn_1d(p: np.ndarray, q: np.ndarray, eps: float, iters: int) -> float:
    n = int(len(p))
    idx = np.arange(n, dtype=float)
    c = np.abs(idx.reshape(-1, 1) - idx.reshape(1, -1))
    k = np.exp(-c / max(1e-6, eps))
    u = np.ones(n, dtype=float)
    v = np.ones(n, dtype=float)
    for _ in range(max(10, iters)):
        u = p / np.maximum(1e-9, k @ v)
        v = q / np.maximum(1e-9, k.T @ u)
    t = np.outer(u, v) * k
    return float(np.sum(t * c))


def run(ctx) -> PluginResult:
    config = build_config(ctx)
    prepared = prepare_data(ctx, config)
    if isinstance(prepared, PluginResult):
        prepared.summary = f"{PLUGIN_ID}: {prepared.summary}"
        return prepared
    x = prepared.matrix
    a, b = split_early_late(x)
    if a.size == 0 or b.size == 0:
        return degraded(PLUGIN_ID, "Need early/late windows for OT drift", {"rows_used": int(x.shape[0])})
    bins = int(config.get("bins") or 32)
    eps = float(config.get("epsilon") or 0.05)
    iters = int(config.get("iterations") or 80)
    rows = []
    for j, col in enumerate(prepared.numeric_cols):
        lo = float(min(a[:, j].min(), b[:, j].min()))
        hi = float(max(a[:, j].max(), b[:, j].max()))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        pa, edges = np.histogram(a[:, j], bins=bins, range=(lo, hi), density=False)
        pb, _ = np.histogram(b[:, j], bins=edges, density=False)
        p = pa.astype(float)
        q = pb.astype(float)
        p = p / max(1.0, p.sum())
        q = q / max(1.0, q.sum())
        cost = _sinkhorn_1d(p, q, eps, iters)
        rows.append({"feature": col, "sinkhorn_cost": float(cost)})
    rows.sort(key=lambda r: float(r["sinkhorn_cost"]), reverse=True)
    top_k = int(config.get("top_k") or 10)
    rows = rows[:top_k]
    artifacts = [artifact(ctx, PLUGIN_ID, "sinkhorn_ot_drift.json", {"feature_costs": rows})]
    findings = []
    if rows:
        r0 = rows[0]
        findings.append(finding(PLUGIN_ID, f"Highest OT drift feature: {r0['feature']}", "Prioritize this feature’s process drivers for stabilization because its distribution changed the most.", float(r0["sinkhorn_cost"]), r0))
    return PluginResult("ok", f"Computed Sinkhorn OT drift for {len(rows)} features", {"features_scored": len(rows)}, findings, artifacts, None)
