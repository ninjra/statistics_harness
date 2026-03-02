from __future__ import annotations

import numpy as np
import pandas as pd

from statistic_harness.core.types import PluginResult

from .common import artifact, build_config, degraded, finding, prepare_data, rng as _make_rng

PLUGIN_ID = "analysis_tensor_cp_parafac_decomp_v1"


def _als_cp(tensor: np.ndarray, rank: int, max_iter: int = 50, seed: int = 0) -> tuple[list[np.ndarray], float]:
    dims = tensor.shape
    gen = np.random.default_rng(seed)
    factors = [gen.standard_normal((d, rank)) for d in dims]
    for _ in range(max_iter):
        for mode in range(len(dims)):
            unfold = np.reshape(np.moveaxis(tensor, mode, 0), (dims[mode], -1))
            khatri_rao = factors[(mode + 1) % len(dims)]
            for m in range(2, len(dims)):
                other = factors[(mode + m) % len(dims)]
                kr_new = np.zeros((khatri_rao.shape[0] * other.shape[0], rank))
                for r in range(rank):
                    kr_new[:, r] = np.kron(khatri_rao[:, r], other[:, r])
                khatri_rao = kr_new
            factors[mode], _, _, _ = np.linalg.lstsq(khatri_rao.T @ khatri_rao, khatri_rao.T @ unfold.T, rcond=None)
            factors[mode] = factors[mode].T
    recon = np.zeros(dims)
    for r in range(rank):
        component = factors[0][:, r]
        for mode in range(1, len(dims)):
            component = np.tensordot(component, factors[mode][:, r], axes=0)
        recon += component
    err = float(np.linalg.norm(tensor - recon) / max(1e-9, np.linalg.norm(tensor)))
    return factors, err


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

    metric_cols = prepared.numeric_cols[:min(10, len(prepared.numeric_cols))]
    if len(metric_cols) < 1:
        return degraded(PLUGIN_ID, "No numeric columns for tensor", {"cols": 0})

    buckets = sorted(str(v) for v in df["__bucket__"].dropna().unique())
    processes = sorted(str(v).strip().lower() for v in df[prepared.process_col].dropna().astype(str).unique() if str(v).strip())
    if len(buckets) < 3 or len(processes) < 2:
        return degraded(PLUGIN_ID, "Insufficient tensor dimensions", {"buckets": len(buckets), "processes": len(processes)})

    b_idx = {b: i for i, b in enumerate(buckets)}
    p_idx = {p: i for i, p in enumerate(processes)}
    n_metrics = len(metric_cols)
    t = np.zeros((len(buckets), len(processes), n_metrics), dtype=float)

    for mc_i, mc in enumerate(metric_cols):
        grouped = df.groupby(["__bucket__", prepared.process_col], dropna=False)[mc].mean().reset_index()
        for row in grouped.itertuples(index=False):
            b = str(row[0])
            p = str(row[1]).strip().lower()
            val = float(row[2]) if pd.notna(row[2]) else 0.0
            if b in b_idx and p in p_idx:
                t[b_idx[b], p_idx[p], mc_i] = val

    rank = max(1, min(int(config.get("rank") or 3), min(t.shape[0], t.shape[1])))
    local_rng = _make_rng(ctx, config)
    seed_val = int(local_rng.integers(0, 2**31))

    if n_metrics > 1 and rank > 1:
        factors, err = _als_cp(t, rank, seed=seed_val)
        u1 = factors[1]
    else:
        unfold0 = t.reshape(t.shape[0], -1)
        unfold1 = np.transpose(t, (1, 0, 2)).reshape(t.shape[1], -1)
        u0, s0, _ = np.linalg.svd(unfold0, full_matrices=False)
        u1, _, _ = np.linalg.svd(unfold1, full_matrices=False)
        recon0 = (u0[:, :rank] * s0[:rank]) @ u0[:, :rank].T
        err = float(np.linalg.norm(unfold0 - recon0 @ unfold0) / max(1e-9, np.linalg.norm(unfold0)))

    top_proc = np.argsort(np.abs(u1[:, 0]))[::-1][: min(8, len(processes))]
    drivers = [{"process": processes[int(i)], "loading": float(u1[i, 0])} for i in top_proc]
    artifacts = [artifact(ctx, PLUGIN_ID, "tensor_cp_parafac_proxy.json", {"rank": rank, "reconstruction_error": err, "process_loadings": drivers, "metrics_used": metric_cols, "tensor_shape": list(t.shape)})]
    findings = [finding(PLUGIN_ID, "Tensor factor loadings isolate process-time structure", "Top factor loadings indicate process groups driving temporal tensor variation; prioritize them for focused interventions.", 1.0 - min(1.0, err), {"reconstruction_error": err, "tensor_dimensions": list(t.shape)})]
    return PluginResult("ok", f"Computed tensor CP/PARAFAC with {n_metrics} metrics", {"rank": rank, "reconstruction_error": err, "metrics_used": n_metrics}, findings, artifacts, None)
