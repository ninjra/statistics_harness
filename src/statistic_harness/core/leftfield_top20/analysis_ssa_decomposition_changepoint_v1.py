from __future__ import annotations

from typing import Any

import numpy as np

from statistic_harness.core.types import PluginResult

from .common import artifact, build_config, degraded, finding, prepare_data, safe_svd, ssa_reconstruct

PLUGIN_ID = "analysis_ssa_decomposition_changepoint_v1"


def run(ctx) -> PluginResult:
    config = build_config(ctx)
    prepared = prepare_data(ctx, config)
    if isinstance(prepared, PluginResult):
        prepared.summary = f"{PLUGIN_ID}: {prepared.summary}"
        return prepared
    x = prepared.matrix
    if x.shape[0] < 40:
        return degraded(PLUGIN_ID, "Too few rows for SSA", {"rows_used": int(x.shape[0])})
    centered = x - x.mean(axis=0, keepdims=True)
    u, s, _ = safe_svd(centered)
    pc1 = u[:, 0] * s[0]
    recon, svals = ssa_reconstruct(pc1, int(config.get("window") or 48), int(config.get("components") or 5))
    delta = np.abs(np.diff(recon))
    if delta.size == 0:
        return degraded(PLUGIN_ID, "SSA failed to produce changepoints", {"rows_used": int(x.shape[0])})
    top_k = int(config.get("top_k") or 10)
    idx = np.argsort(delta)[::-1][:top_k]
    cp = [{"index": int(i + 1), "magnitude": float(delta[i])} for i in idx]
    artifacts = [artifact(ctx, PLUGIN_ID, "ssa_components_changepoints.json", {"cp_candidates": cp, "singular_values": svals[:12].tolist()})]
    findings = [
        finding(
            PLUGIN_ID,
            f"SSA regime shift near row {cp[0]['index']}",
            "Inspect process mix and queue pressure around this boundary; route/control changes near this regime can improve close stability.",
            cp[0]["magnitude"],
            cp[0],
        )
    ]
    return PluginResult("ok", f"Computed SSA changepoints over {x.shape[0]} rows", {"rows_used": int(x.shape[0]), "cp_candidates": len(cp)}, findings, artifacts, None)
