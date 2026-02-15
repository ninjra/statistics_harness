from __future__ import annotations

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
    m = x.shape[1]
    if m < 2:
        return degraded(PLUGIN_ID, "Need at least two numeric columns", {"cols_used": int(m)})
    top_k = int(config.get("top_k") or 20)
    pairs = []
    for i in range(m):
        xi = x[:, i : i + 1]
        for j in range(i + 1, m):
            score = hsic_score(xi, x[:, j : j + 1])
            pairs.append({"left": prepared.numeric_cols[i], "right": prepared.numeric_cols[j], "hsic": float(score)})
    pairs.sort(key=lambda r: float(r["hsic"]), reverse=True)
    winners = pairs[:top_k]
    artifacts = [artifact(ctx, PLUGIN_ID, "hsic_dependence_pairs.json", {"pairs": winners})]
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
    return PluginResult("ok", f"Scored HSIC for {len(pairs)} feature pairs", {"pairs_scored": len(pairs), "top_hsic": float(winners[0]["hsic"]) if winners else 0.0}, findings, artifacts, None)
