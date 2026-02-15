from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginResult

from .common import artifact, build_config, degraded, finding, prepare_data

PLUGIN_ID = "analysis_dmd_koopman_modes_v1"


def run(ctx) -> PluginResult:
    config = build_config(ctx)
    prepared = prepare_data(ctx, config)
    if isinstance(prepared, PluginResult):
        prepared.summary = f"{PLUGIN_ID}: {prepared.summary}"
        return prepared
    x = prepared.matrix
    if x.shape[0] < 30:
        return degraded(PLUGIN_ID, "Need >=30 rows for DMD", {"rows_used": int(x.shape[0])})
    x1 = x[:-1, :].T
    x2 = x[1:, :].T
    a = x2 @ np.linalg.pinv(x1)
    eigvals, _ = np.linalg.eig(a)
    radius = np.abs(eigvals)
    drift = float(np.sum(np.maximum(0.0, radius - 1.0)))
    lead = sorted([float(v) for v in eigvals[: min(10, len(eigvals))]], key=lambda z: abs(z), reverse=True)
    artifacts = [artifact(ctx, PLUGIN_ID, "dmd_koopman_modes.json", {"eigenvalues": lead, "drift_score": drift})]
    findings = [finding(PLUGIN_ID, "Koopman mode drift detected" if drift > 0.0 else "Koopman modes stable", "High drift score indicates unstable dynamics; test batching/scheduling actions that dampen oscillatory modes.", drift, {"drift_score": drift})]
    return PluginResult("ok", f"Computed DMD over {x.shape[0]} observations", {"drift_score": drift, "mode_count": int(len(eigvals))}, findings, artifacts, None)
