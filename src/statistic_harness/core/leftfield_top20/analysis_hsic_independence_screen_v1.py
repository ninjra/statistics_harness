from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginResult

from .common import artifact, build_config, degraded, finding, hsic_score, prepare_data

PLUGIN_ID = "analysis_hsic_independence_screen_v1"


def run(ctx) -> PluginResult:
    config = build_config(ctx)
    prepared = prepare_data(ctx, config)
    if isinstance(prepared, PluginResult):
        prepared.summary = f"{PLUGIN_ID}: {prepared.summary}"
        return prepared
    x = prepared.matrix
    max_rows = int(config.get("max_rows_for_hsic") or 320)
    if x.shape[0] > max_rows:
        seed = int(getattr(ctx, "run_seed", 0) or 0)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(x.shape[0], size=max_rows, replace=False))
        x = x[idx, :]
    m = x.shape[1]
    if m < 2:
        return degraded(PLUGIN_ID, "Need at least two numeric columns", {"cols_used": int(m)})
    max_cols = int(config.get("max_cols_for_hsic") or 14)
    if m > max_cols:
        # Prefer high-variance columns to keep dependence scan informative and bounded.
        var = np.var(x, axis=0)
        keep = np.argsort(var)[::-1][:max_cols]
        keep = np.sort(keep)
        x = x[:, keep]
        cols = [prepared.numeric_cols[int(i)] for i in keep]
    else:
        cols = list(prepared.numeric_cols)
    m = x.shape[1]
    top_k = int(config.get("top_k") or 20)
    max_pairs = int(config.get("max_pairs_for_hsic") or 120)
    pairs = []
    for i in range(m):
        xi = x[:, i : i + 1]
        for j in range(i + 1, m):
            score = hsic_score(xi, x[:, j : j + 1])
            pairs.append({"left": cols[i], "right": cols[j], "hsic": float(score)})
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break
    pairs.sort(key=lambda r: float(r["hsic"]), reverse=True)
    winners = pairs[:top_k]
    artifacts = [artifact(ctx, PLUGIN_ID, "hsic_dependence_pairs.json", {"pairs": winners, "pairs_scored_total": len(pairs), "rows_used_hsic": int(x.shape[0]), "cols_used_hsic": int(x.shape[1])})]
    findings = []
    if winners:
        w = winners[0]
        findings.append(
            finding(
                PLUGIN_ID,
                f"Strong nonlinear dependence: {w['left']} ~ {w['right']}",
                "Use this pair to segment operations and check if policy/sequence changes jointly reduce close volatility.",
                float(w["hsic"]),
                w,
            )
        )
    return PluginResult("ok", f"Scored HSIC for {len(pairs)} feature pairs", {"pairs_scored": len(pairs), "rows_used_hsic": int(x.shape[0]), "cols_used_hsic": int(x.shape[1]), "top_hsic": float(winners[0]["hsic"]) if winners else 0.0}, findings, artifacts, None)
