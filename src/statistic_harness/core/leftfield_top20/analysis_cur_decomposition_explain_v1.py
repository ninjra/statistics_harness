from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginResult

from .common import artifact, build_config, finding, prepare_data, safe_svd

PLUGIN_ID = "analysis_cur_decomposition_explain_v1"


def run(ctx) -> PluginResult:
    config = build_config(ctx)
    prepared = prepare_data(ctx, config)
    if isinstance(prepared, PluginResult):
        prepared.summary = f"{PLUGIN_ID}: {prepared.summary}"
        return prepared
    x = prepared.matrix
    r = max(2, min(int(config.get("rank") or 6), min(x.shape)))
    u, s, vt = safe_svd(x - x.mean(axis=0, keepdims=True))
    v_r = vt[:r, :].T
    u_r = u[:, :r]
    col_lev = np.sum(v_r * v_r, axis=1) / float(r)
    row_lev = np.sum(u_r * u_r, axis=1) / float(r)
    top_cols_idx = np.argsort(col_lev)[::-1][: min(8, len(col_lev))]
    top_rows_idx = np.argsort(row_lev)[::-1][: min(8, len(row_lev))]
    c = x[:, top_cols_idx]
    rmat = x[top_rows_idx, :]
    try:
        u_mid = np.linalg.pinv(c) @ x @ np.linalg.pinv(rmat)
        recon = c @ u_mid @ rmat
        err = float(np.linalg.norm(x - recon) / max(1e-9, np.linalg.norm(x)))
    except Exception:
        err = 1.0
    top_cols = [{"column": prepared.numeric_cols[int(i)], "score": float(col_lev[i])} for i in top_cols_idx]
    artifacts = [artifact(ctx, PLUGIN_ID, "cur_explainability.json", {"top_columns": top_cols, "top_rows": [int(i) for i in top_rows_idx], "relative_reconstruction_error": err})]
    findings = [
        finding(
            PLUGIN_ID,
            f"CUR identified {len(top_cols)} influential columns",
            "Prioritize these high-leverage fields when simplifying controls, data quality checks, and recommendation explainability.",
            float(top_cols[0]["score"]) if top_cols else None,
            top_cols[0] if top_cols else None,
        )
    ]
    return PluginResult("ok", f"Computed CUR decomposition (rank={r})", {"rows_used": int(x.shape[0]), "cols_used": int(x.shape[1]), "relative_error": err}, findings, artifacts, None)
