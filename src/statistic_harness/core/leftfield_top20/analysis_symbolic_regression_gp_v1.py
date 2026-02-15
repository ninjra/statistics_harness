from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginResult

from .common import artifact, build_config, degraded, finding, prepare_data

PLUGIN_ID = "analysis_symbolic_regression_gp_v1"


def run(ctx) -> PluginResult:
    config = build_config(ctx)
    prepared = prepare_data(ctx, config)
    if isinstance(prepared, PluginResult):
        prepared.summary = f"{PLUGIN_ID}: {prepared.summary}"
        return prepared
    x = prepared.matrix
    if x.shape[1] < 2:
        return degraded(PLUGIN_ID, "Need >=2 features for symbolic regression", {"cols_used": int(x.shape[1])})
    y = x[:, 0]
    feats = x[:, 1:]
    cols = prepared.numeric_cols[1:]
    candidates: list[tuple[str, np.ndarray]] = []
    for j, col in enumerate(cols):
        v = feats[:, j]
        candidates.append((col, v))
        candidates.append((f"{col}^2", v * v))
        candidates.append((f"log1p(|{col}|)", np.log1p(np.abs(v))))
    if feats.shape[1] >= 2:
        v0 = feats[:, 0]
        v1 = feats[:, 1]
        candidates.append((f"{cols[0]}*{cols[1]}", v0 * v1))
    best = None
    for expr, vec in candidates:
        design = np.column_stack([np.ones_like(vec), vec])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ coef
        mse = float(np.mean((y - pred) ** 2))
        if best is None or mse < best["mse"]:
            best = {"expression": expr, "coef": [float(c) for c in coef], "mse": mse}
    assert best is not None
    artifacts = [artifact(ctx, PLUGIN_ID, "symbolic_regression_proxy.json", {"best_expression": best, "candidate_count": len(candidates)})]
    findings = [finding(PLUGIN_ID, f"Best symbolic expression proxy: {best['expression']}", "Use this interpretable expression as a compact hypothesis for target behavior and validate on holdout windows.", 1.0 / max(1e-9, best["mse"]), best)]
    return PluginResult("ok", "Selected symbolic-regression proxy expression", {"candidate_count": len(candidates), "best_mse": float(best["mse"])}, findings, artifacts, None)
