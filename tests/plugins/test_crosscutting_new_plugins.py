from __future__ import annotations

import json
import re
from dataclasses import asdict
from importlib import import_module

import pandas as pd
import pytest

from tests.conftest import make_context


NEW_PLUGIN_IDS = [
    # Family A
    "analysis_tda_persistent_homology",
    "analysis_tda_persistence_landscapes",
    "analysis_tda_mapper_graph",
    "analysis_tda_betti_curve_changepoint",
    # Family B
    "analysis_topographic_similarity_angle_projection",
    "analysis_topographic_angle_dynamics",
    "analysis_topographic_tanova_permutation",
    "analysis_map_permutation_test_karniski",
    # Family C
    "analysis_surface_multiscale_wavelet_curvature",
    "analysis_surface_fractal_dimension_variogram",
    "analysis_surface_rugosity_index",
    "analysis_surface_terrain_position_index",
    "analysis_surface_fabric_sso_eigen",
    "analysis_surface_hydrology_flow_watershed",
    "analysis_surface_roughness_metrics",
    "analysis_monte_carlo_surface_uncertainty",
    # Family D
    "analysis_ttests_auto",
    "analysis_chi_square_association",
    "analysis_regression_auto",
    "analysis_time_series_analysis_auto",
    "analysis_cluster_analysis_auto",
    "analysis_pca_auto",
    # Family E
    "analysis_bayesian_point_displacement",
    # Family F
    "analysis_actionable_ops_levers_v1",
]


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")


def _plugin_result_stable_view(result) -> dict:
    payload = asdict(result)
    # Ignore timings if any plugin wrote them.
    if isinstance(payload.get("metrics"), dict):
        for key in list(payload["metrics"].keys()):
            if "runtime" in key or "time" == key or key.endswith("_ms"):
                payload["metrics"].pop(key, None)
    return {
        "status": payload.get("status"),
        "summary": payload.get("summary"),
        "findings": payload.get("findings"),
        "references": payload.get("references"),
    }


@pytest.mark.parametrize("plugin_id", NEW_PLUGIN_IDS)
def test_new_plugins_crosscutting_contracts(run_dir, plugin_id: str):
    # Small synthetic eventlog-like dataset satisfying most gates.
    df = pd.DataFrame(
        {
            "case_id": ["c1", "c1", "c2", "c3", "c3", "c4"] * 40,
            "activity": ["p1", "p2", "p1", "p3", "p2", "p1"] * 40,
            "host": ["h1", "h2", "h1", "h2", "h1", "h2"] * 40,
            "group": ["g1", "g2", "g1", "g2", "g1", "g2"] * 40,
            # Storage fixture inserts rows via sqlite bindings; keep timestamps as strings.
            "created_at": pd.date_range("2026-01-01", periods=240, freq="min").astype(str).tolist(),
            "duration_seconds": [float(i % 17) for i in range(240)],
            "x_coord": list(range(240)),
            "y_coord": list(range(240))[::-1],
            "message": ["this is a fairly long free-text message " + str(i) for i in range(240)],
        }
    )
    settings = {
        "seed": 123,
        "max_rows": 300,
        # Heavy plugins should be runnable in tests, but they may still choose to degrade
        # due to optional deps. This ensures we don't permanently lock them in placeholder mode.
        "enable_heavy": True,
        "privacy": {"enable_redaction": True, "redact_patterns": ["email", "uuid"]},
    }
    ctx = make_context(run_dir, df, settings=settings, run_seed=123)
    module = import_module(f"plugins.{plugin_id}.plugin")
    plugin = module.Plugin()
    res1 = plugin.run(ctx)
    res2 = plugin.run(ctx)

    # Determinism: stable outputs under fixed seed.
    assert _plugin_result_stable_view(res1) == _plugin_result_stable_view(res2)

    # Skip gating reason is present.
    if res1.status in {"skipped", "degraded"}:
        assert isinstance(res1.debug, dict)
        assert isinstance(res1.debug.get("gating_reason"), str)
        assert res1.debug.get("gating_reason")

    # Citations/references present when plugin claims ok or degraded work.
    if res1.status in {"ok", "degraded"}:
        assert isinstance(res1.references, list)
        assert len(res1.references) >= 1

    # Redaction: findings do not contain obvious emails or UUIDs.
    blob = json.dumps(res1.findings, sort_keys=True)
    assert EMAIL_RE.search(blob) is None
    assert UUID_RE.search(blob) is None
