from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from statistic_harness.core.types import PluginResult

from .common import artifact, build_config, degraded, finding, prepare_data

PLUGIN_ID = "analysis_node2vec_graph_embedding_drift_v1"


def _process_sequence(df: pd.DataFrame, process_col: str | None, time_col: str | None) -> list[str]:
    if process_col is None:
        return []
    work = df[[process_col]].copy()
    if time_col is not None and time_col in df.columns:
        parsed = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        work = work.assign(__t=parsed).sort_values("__t")
    out = []
    for value in work[process_col].tolist():
        text = str(value).strip().lower()
        if text:
            out.append(text)
    return out


def run(ctx) -> PluginResult:
    config = build_config(ctx)
    prepared = prepare_data(ctx, config)
    if isinstance(prepared, PluginResult):
        prepared.summary = f"{PLUGIN_ID}: {prepared.summary}"
        return prepared
    seq = _process_sequence(prepared.frame, prepared.process_col, prepared.time_col)
    if len(seq) < 20:
        return degraded(PLUGIN_ID, "Need process sequence data for node embeddings")
    edges = Counter((seq[i], seq[i + 1]) for i in range(len(seq) - 1))
    nodes = sorted({a for a, _ in edges} | {b for _, b in edges})
    if len(nodes) < 3:
        return degraded(PLUGIN_ID, "Transition graph too small", {"nodes": len(nodes)})
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    a = np.zeros((len(nodes), len(nodes)), dtype=float)
    for (u, v), w in edges.items():
        a[node_to_idx[u], node_to_idx[v]] += float(w)
        a[node_to_idx[v], node_to_idx[u]] += float(w)
    vals, vecs = np.linalg.eigh(a)
    order = np.argsort(vals)[::-1]
    dim = max(2, min(int(config.get("embedding_dim") or 8), len(nodes)))
    emb = vecs[:, order[:dim]]
    split = max(2, len(seq) // 2)
    early_edges = Counter((seq[i], seq[i + 1]) for i in range(max(0, split - 1)))
    late_edges = Counter((seq[i], seq[i + 1]) for i in range(split, len(seq) - 1))
    drift_rows = []
    for n in nodes:
        out_early = sum(w for (u, _), w in early_edges.items() if u == n)
        out_late = sum(w for (u, _), w in late_edges.items() if u == n)
        drift = float(abs(out_late - out_early) / max(1.0, out_early + out_late))
        drift_rows.append({"node": n, "degree_drift": drift})
    drift_rows.sort(key=lambda r: float(r["degree_drift"]), reverse=True)
    drift_rows = drift_rows[: int(config.get("top_k") or 12)]
    artifacts = [artifact(ctx, PLUGIN_ID, "node2vec_embedding_drift_proxy.json", {"nodes": nodes[:200], "embedding_sample": emb[:200].tolist(), "node_drift": drift_rows})]
    findings = []
    if drift_rows:
        d0 = drift_rows[0]
        findings.append(finding(PLUGIN_ID, f"Process role drift hotspot: {d0['node']}", "This process changed its transition behavior most; validate upstream/downstream routing and workload mix.", float(d0["degree_drift"]), d0))
    return PluginResult("ok", f"Computed graph embeddings for {len(nodes)} processes", {"nodes": len(nodes), "edge_count": len(edges)}, findings, artifacts, None)
