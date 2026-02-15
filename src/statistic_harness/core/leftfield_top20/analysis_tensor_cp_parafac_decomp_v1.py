from __future__ import annotations

import numpy as np
import pandas as pd

from statistic_harness.core.types import PluginResult

from .common import artifact, build_config, degraded, finding, prepare_data

PLUGIN_ID = "analysis_tensor_cp_parafac_decomp_v1"


def run(ctx) -> PluginResult:
    config = build_config(ctx)
    prepared = prepare_data(ctx, config)
    if isinstance(prepared, PluginResult):
        prepared.summary = f"{PLUGIN_ID}: {prepared.summary}"
        return prepared
    if prepared.process_col is None:
        return degraded(PLUGIN_ID, "No process column available for tensor decomposition")
    df = prepared.frame.copy()
    if prepared.time_col is None:
        df["__bucket__"] = np.arange(len(df)) // max(1, len(df) // 24)
    else:
        parsed = pd.to_datetime(df[prepared.time_col], errors="coerce", utc=True)
        if parsed.notna().any():
            df["__bucket__"] = parsed.dt.floor("6h").astype(str)
        else:
            df["__bucket__"] = np.arange(len(df)) // max(1, len(df) // 24)
    metric_col = prepared.numeric_cols[0]
    grouped = df.groupby(["__bucket__", prepared.process_col], dropna=False)[metric_col].mean().reset_index()
    buckets = sorted(str(v) for v in grouped["__bucket__"].dropna().unique())
    processes = sorted(str(v).strip().lower() for v in grouped[prepared.process_col].dropna().astype(str).unique() if str(v).strip())
    if len(buckets) < 3 or len(processes) < 2:
        return degraded(PLUGIN_ID, "Insufficient tensor dimensions", {"buckets": len(buckets), "processes": len(processes)})
    b_idx = {b: i for i, b in enumerate(buckets)}
    p_idx = {p: i for i, p in enumerate(processes)}
    t = np.zeros((len(buckets), len(processes), 1), dtype=float)
    for row in grouped.itertuples(index=False):
        b = str(getattr(row, "_0")) if hasattr(row, "_0") else str(row[0])
        p = str(getattr(row, "_1")) if hasattr(row, "_1") else str(row[1])
        val = float(getattr(row, metric_col, row[2]))
        p_norm = p.strip().lower()
        if b in b_idx and p_norm in p_idx:
            t[b_idx[b], p_idx[p_norm], 0] = val
    rank = max(1, min(int(config.get("rank") or 3), min(t.shape[0], t.shape[1])))
    unfold0 = t.reshape(t.shape[0], -1)
    unfold1 = np.transpose(t, (1, 0, 2)).reshape(t.shape[1], -1)
    u0, s0, _ = np.linalg.svd(unfold0, full_matrices=False)
    u1, _, _ = np.linalg.svd(unfold1, full_matrices=False)
    recon0 = (u0[:, :rank] * s0[:rank]) @ u0[:, :rank].T
    err = float(np.linalg.norm(unfold0 - recon0 @ unfold0) / max(1e-9, np.linalg.norm(unfold0)))
    top_proc = np.argsort(np.abs(u1[:, 0]))[::-1][: min(8, len(processes))]
    drivers = [{"process": processes[int(i)], "loading": float(abs(u1[i, 0]))} for i in top_proc]
    artifacts = [artifact(ctx, PLUGIN_ID, "tensor_cp_parafac_proxy.json", {"rank": rank, "reconstruction_error": err, "process_loadings": drivers})]
    findings = [finding(PLUGIN_ID, "Tensor factor loadings isolate process-time structure", "Top factor loadings indicate process groups driving temporal tensor variation; prioritize them for focused interventions.", 1.0 - min(1.0, err), {"reconstruction_error": err})]
    return PluginResult("ok", "Computed tensor CP/PARAFAC proxy factors", {"rank": rank, "reconstruction_error": err}, findings, artifacts, None)
