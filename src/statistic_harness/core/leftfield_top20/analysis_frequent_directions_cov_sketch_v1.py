from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginResult

from .common import artifact, build_config, finding, prepare_data

PLUGIN_ID = "analysis_frequent_directions_cov_sketch_v1"


def run(ctx) -> PluginResult:
    config = build_config(ctx)
    prepared = prepare_data(ctx, config)
    if isinstance(prepared, PluginResult):
        prepared.summary = f"{PLUGIN_ID}: {prepared.summary}"
        return prepared
    x = prepared.matrix
    sketch_size = max(2, min(int(config.get("sketch_size") or 8), x.shape[1]))
    b = np.zeros((sketch_size, x.shape[1]), dtype=float)
    next_row = 0
    for i in range(x.shape[0]):
        b[next_row, :] = x[i, :]
        next_row += 1
        if next_row < sketch_size:
            continue
        _, s, vt = np.linalg.svd(b, full_matrices=False)
        delta = s[-1] ** 2
        s2 = np.sqrt(np.maximum(0.0, s**2 - delta))
        b = np.diag(s2) @ vt
        next_row = sketch_size - 1
    cov_true = (x.T @ x) / max(1, x.shape[0])
    cov_approx = (b.T @ b) / max(1, x.shape[0])
    err = float(np.linalg.norm(cov_true - cov_approx) / max(1e-9, np.linalg.norm(cov_true)))
    eigvals = np.linalg.eigvalsh(cov_approx)
    top = np.argsort(eigvals)[::-1][: min(6, len(eigvals))]
    top_modes = [{"feature": prepared.numeric_cols[int(i)], "eig_approx": float(eigvals[i])} for i in top]
    artifacts = [artifact(ctx, PLUGIN_ID, "frequent_directions_sketch.json", {"sketch_size": sketch_size, "relative_cov_error": err, "top_modes": top_modes})]
    findings = [finding(PLUGIN_ID, "Frequent Directions covariance sketch completed", "Use the sketch modes to monitor multivariate drift at scale without full covariance recomputation.", 1.0 - min(1.0, err), {"relative_cov_error": err})]
    return PluginResult("ok", "Built covariance sketch with Frequent Directions", {"rows_used": int(x.shape[0]), "relative_cov_error": err}, findings, artifacts, None)
