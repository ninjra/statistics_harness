from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from statistic_harness.core.types import PluginResult

from .common import artifact, build_config, degraded, finding, prepare_data

PLUGIN_ID = "analysis_ges_score_based_causal_v1"


def _bic_score(y: np.ndarray, x: np.ndarray) -> float:
    n = max(1, y.shape[0])
    if x.size == 0:
        resid = y - np.mean(y)
        k = 1
    else:
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ coef
        k = x.shape[1] + 1
    sigma2 = float(np.mean(resid * resid))
    return float(n * math.log(max(1e-9, sigma2)) + k * math.log(max(2, n)))


def _has_cycle(parents: dict[int, list[int]], m: int) -> bool:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * m

    def _dfs(node: int) -> bool:
        color[node] = GRAY
        for child in range(m):
            if node in parents.get(child, []):
                if color[child] == GRAY:
                    return True
                if color[child] == WHITE and _dfs(child):
                    return True
        color[node] = BLACK
        return False

    for v in range(m):
        if color[v] == WHITE:
            if _dfs(v):
                return True
    return False


def run(ctx) -> PluginResult:
    config = build_config(ctx)
    prepared = prepare_data(ctx, config)
    if isinstance(prepared, PluginResult):
        prepared.summary = f"{PLUGIN_ID}: {prepared.summary}"
        return prepared
    x = prepared.matrix
    m = x.shape[1]
    if m < 2:
        return degraded(PLUGIN_ID, "Need >=2 features for GES", {"cols_used": int(m)})
    parents: dict[int, list[int]] = defaultdict(list)
    edges: list[tuple[int, int, float]] = []
    max_edges = max(1, int(config.get("max_edges") or 18))

    # Forward phase: greedily add edges that improve BIC
    for _ in range(max_edges):
        best: tuple[int, int, float] | None = None
        for i in range(m):
            for j in range(m):
                if i == j or i in parents[j]:
                    continue
                if j in parents[i]:
                    continue
                # Check if adding i->j would create a cycle
                parents[j].append(i)
                would_cycle = _has_cycle(parents, m)
                parents[j].remove(i)
                if would_cycle:
                    continue
                y_vec = x[:, j]
                x_old = x[:, parents[j]] if parents[j] else np.empty((x.shape[0], 0))
                old_bic = _bic_score(y_vec, x_old)
                cand_par = parents[j] + [i]
                x_new = x[:, cand_par]
                new_bic = _bic_score(y_vec, x_new)
                gain = old_bic - new_bic
                if gain > 0 and (best is None or gain > best[2]):
                    best = (i, j, float(gain))
        if best is None:
            break
        parents[best[1]].append(best[0])
        edges.append(best)

    # Backward phase: remove edges that improve BIC
    changed = True
    while changed:
        changed = False
        worst: tuple[int, int, float] | None = None
        for j in range(m):
            for i in list(parents[j]):
                y_vec = x[:, j]
                x_cur = x[:, parents[j]] if parents[j] else np.empty((x.shape[0], 0))
                cur_bic = _bic_score(y_vec, x_cur)
                reduced = [p for p in parents[j] if p != i]
                x_red = x[:, reduced] if reduced else np.empty((x.shape[0], 0))
                red_bic = _bic_score(y_vec, x_red)
                gain = cur_bic - red_bic
                if gain > 0 and (worst is None or gain > worst[2]):
                    worst = (i, j, float(gain))
        if worst is not None:
            parents[worst[1]].remove(worst[0])
            edges = [(i, j, g) for i, j, g in edges if not (i == worst[0] and j == worst[1])]
            changed = True

    rows = [{"src": prepared.numeric_cols[i], "dst": prepared.numeric_cols[j], "score_gain": g} for i, j, g in edges]
    artifacts = [artifact(ctx, PLUGIN_ID, "ges_score_graph_proxy.json", {"edges": rows})]
    findings = []
    if rows:
        r0 = rows[0]
        findings.append(finding(PLUGIN_ID, f"Top score-based edge: {r0['src']} -> {r0['dst']}", "Candidate directional dependency with strongest score improvement; prioritize in causal hypothesis testing.", float(r0["score_gain"]), r0))
    return PluginResult("ok", f"Built GES-style graph with {len(rows)} edges", {"edge_count": len(rows)}, findings, artifacts, None)
