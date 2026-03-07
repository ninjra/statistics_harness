"""Smoke tests for the 61 exempted plugins.

These plugins were previously validated only through integration/matrix
contracts.  This file provides direct coverage so that
``verify_plugin_test_coverage.py`` detects them (it searches for literal
plugin-ID strings in test files).

The 53 thin-wrapper plugins are already exercised by
``test_next30_plugins_smoke.py`` and ``test_next30b_plugins_smoke.py`` via
imported ID tuples — but the coverage tool cannot see those references.
The parametrized test below re-runs them with explicit IDs.

The 4 full-implementation plugins get dedicated smoke tests.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_context

# ---------------------------------------------------------------------------
# All 61 exempted plugin IDs — listed as literal strings so the coverage
# checker can detect them.
# ---------------------------------------------------------------------------

# -- Thin-wrapper: NEXT30 group (27) --
NEXT30_EXEMPTED = [
    "analysis_bayesian_online_changepoint_studentt_v1",
    "analysis_benfords_law_anomaly_v1",
    "analysis_cca_crossblock_association_v1",
    "analysis_cusum_on_model_residuals_v1",
    "analysis_dirichlet_multinomial_categorical_overdispersion_v1",
    "analysis_distance_correlation_screen_v1",
    "analysis_factor_rotation_varimax_v1",
    "analysis_fisher_exact_enrichment_v1",
    "analysis_fused_lasso_trend_filtering_v1",
    "analysis_gam_spline_regression_v1",
    "analysis_garch_volatility_shift_v1",
    "analysis_geometric_median_multivariate_location_v1",
    "analysis_heavy_tail_index_hill_v1",
    "analysis_ica_source_separation_v1",
    "analysis_lomb_scargle_periodogram_v1",
    "analysis_multicollinearity_vif_screen_v1",
    "analysis_outlier_influence_cooks_distance_v1",
    "analysis_quantile_loss_boosting_v1",
    "analysis_quantile_regression_forest_v1",
    "analysis_random_matrix_marchenko_pastur_denoise_v1",
    "analysis_recurrence_quantification_rqa_v1",
    "analysis_seasonal_holt_winters_forecast_residuals_v1",
    "analysis_sparse_pca_interpretable_components_v1",
    "analysis_stl_seasonal_decompose_v1",
    "analysis_subspace_tracking_oja_v1",
    "analysis_zero_inflated_count_model_v1",
    "analysis_negative_binomial_overdispersion_v1",
]

# -- Thin-wrapper: NEXT30B group (26) --
NEXT30B_EXEMPTED = [
    "analysis_aft_survival_lognormal_v1",
    "analysis_beta_binomial_overdispersion_v1",
    "analysis_capacity_frontier_envelope_v1",
    "analysis_circular_time_of_day_drift_v1",
    "analysis_competing_risks_cif_v1",
    "analysis_constraints_violation_detector_v1",
    "analysis_distance_covariance_dependence_v1",
    "analysis_energy_distance_two_sample_v1",
    "analysis_graph_assortativity_shift_v1",
    "analysis_haar_wavelet_transient_detector_v1",
    "analysis_higuchi_fractal_dimension_v1",
    "analysis_hurst_exponent_long_memory_v1",
    "analysis_mann_kendall_trend_test_v1",
    "analysis_marked_point_process_intensity_v1",
    "analysis_multiscale_entropy_mse_v1",
    "analysis_partial_correlation_network_shift_v1",
    "analysis_permutation_entropy_drift_v1",
    "analysis_piecewise_linear_trend_changepoints_v1",
    "analysis_poisson_regression_rate_drivers_v1",
    "analysis_quantile_mapping_drift_qq_v1",
    "analysis_quantile_sketch_p2_streaming_v1",
    "analysis_randomization_test_median_shift_v1",
    "analysis_robust_regression_huber_ransac_v1",
    "analysis_sample_entropy_irregularity_v1",
    "analysis_spectral_radius_stability_v1",
    "analysis_state_space_smoother_level_shift_v1",
]

# -- Full-implementation plugins (4) --
FULL_IMPL_EXEMPTED = [
    "analysis_issue_cards_v2",
    "analysis_upload_linkage",
    "llm_text2sql_local_generate_v1",
    "report_plain_english_v1",
]

ALL_THIN_WRAPPER_EXEMPTED = sorted(set(NEXT30_EXEMPTED + NEXT30B_EXEMPTED))


# ---------------------------------------------------------------------------
# Shared dataset builder
# ---------------------------------------------------------------------------

def _rich_dataset(rows: int = 600) -> pd.DataFrame:
    """Dataset with time series, categoricals, counts, and graph edges."""
    rng = np.random.default_rng(20260226)
    base = datetime(2025, 1, 1, 0, 0, 0)
    idx = np.arange(rows)
    half = rows // 2
    shift = (idx >= half).astype(float)
    seasonal = np.sin((2.0 * np.pi * idx) / 24.0)
    noise = rng.normal(0.0, 0.5, size=rows)

    jitter = rng.integers(0, 40, size=rows, dtype=np.int64)
    event_ts = [base + timedelta(minutes=int(i * 45 + jitter[i])) for i in idx]

    queue_wait = 15.0 + 3.0 * seasonal + 6.5 * shift + noise
    duration = 35.0 + 2.4 * queue_wait + 4.0 * shift + rng.normal(0.0, 1.8, size=rows)
    service_runtime = 20.0 + 1.4 * queue_wait + rng.normal(0.0, 1.2, size=rows)
    metric_x = 10.0 + 0.6 * seasonal + rng.normal(0.0, 0.6, size=rows)
    metric_y = 8.0 + 0.7 * metric_x + rng.normal(0.0, 0.35, size=rows)
    metric_z = 4.0 + 0.3 * metric_x - 0.2 * metric_y + rng.normal(0.0, 0.25, size=rows)

    raw_count = rng.poisson(np.where(shift > 0.5, 7.0, 3.0), size=rows)
    structural_zero = rng.random(rows) < 0.30
    event_count = np.where(structural_zero, 0, raw_count)
    binary_outcome = (rng.random(rows) < np.where(shift > 0.5, 0.62, 0.38)).astype(int)

    process = np.where(
        (idx % 4) == 0, "close_a",
        np.where((idx % 4) == 1, "close_b",
                 np.where((idx % 4) == 2, "recon", "posting")),
    )
    team = np.where(
        shift > 0.5,
        np.where((idx % 3) == 0, "ops", "fin"),
        np.where((idx % 3) == 0, "fin", "shared"),
    )

    return pd.DataFrame(
        {
            "event_ts": [v.isoformat() for v in event_ts],
            "queue_wait_mins": queue_wait,
            "duration_mins": duration,
            "service_runtime_mins": service_runtime,
            "metric_x": metric_x,
            "metric_y": metric_y,
            "metric_z": metric_z,
            "event_count": event_count.astype(int),
            "binary_outcome": binary_outcome,
            "process_id": process,
            "team": team,
        }
    )


def _run_plugin(tmp_path: Path, plugin_id: str, df: pd.DataFrame):
    run_dir = tmp_path / plugin_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = make_context(
        run_dir, df,
        settings={"seed": 1337, "allow_row_sampling": False},
        run_seed=1337,
    )
    module = import_module(f"plugins.{plugin_id}.plugin")
    return module.Plugin().run(ctx)


# ---------------------------------------------------------------------------
# Thin-wrapper smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plugin_id", ALL_THIN_WRAPPER_EXEMPTED)
def test_exempted_thin_wrapper_smoke(tmp_path: Path, plugin_id: str) -> None:
    """Each exempted thin-wrapper plugin must produce status ok or warn."""
    df = _rich_dataset()
    result = _run_plugin(tmp_path, plugin_id, df)
    assert result.status in {"ok", "warn"}, (
        f"{plugin_id} -> {result.status}: {result.summary}"
    )
    assert isinstance(result.metrics, dict)


# ---------------------------------------------------------------------------
# Full-implementation plugin tests
# ---------------------------------------------------------------------------

def test_analysis_issue_cards_v2_smoke(tmp_path: Path) -> None:
    """Issue cards plugin must run without error on a context with findings."""
    df = _rich_dataset(200)
    run_dir = tmp_path / "analysis_issue_cards_v2"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = make_context(run_dir, df, settings={}, run_seed=42)
    # Pre-populate upstream plugin results so issue cards has something to read
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "plugins": {
                    "analysis_percentile_analysis": {
                        "status": "ok",
                        "findings": [
                            {
                                "kind": "percentile_outlier",
                                "title": "High queue wait",
                                "what": "queue_wait_mins p99 exceeds threshold",
                                "why": "Potential bottleneck",
                                "severity": "medium",
                                "confidence": 0.85,
                                "id": "f:1",
                            }
                        ],
                        "metrics": {"rows_seen": 200},
                    }
                },
                "recommendations": {"items": []},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    module = import_module("plugins.analysis_issue_cards_v2.plugin")
    result = module.Plugin().run(ctx)
    # Issue cards may produce "ok" or "warn" depending on upstream data
    assert result.status in {"ok", "warn", "na"}, (
        f"analysis_issue_cards_v2 -> {result.status}: {result.summary}"
    )


def test_analysis_upload_linkage_smoke(tmp_path: Path) -> None:
    """Upload linkage plugin must run without error."""
    df = _rich_dataset(200)
    result = _run_plugin(tmp_path, "analysis_upload_linkage", df)
    assert result.status in {"ok", "warn", "na"}, (
        f"analysis_upload_linkage -> {result.status}: {result.summary}"
    )


def test_llm_text2sql_local_generate_v1_smoke(tmp_path: Path) -> None:
    """LLM text2sql plugin must not crash (may skip if model not available)."""
    df = _rich_dataset(100)
    result = _run_plugin(tmp_path, "llm_text2sql_local_generate_v1", df)
    # Likely "na" in CI where model dirs are absent
    assert result.status in {"ok", "warn", "na"}, (
        f"llm_text2sql_local_generate_v1 -> {result.status}: {result.summary}"
    )


def test_report_plain_english_v1_smoke(tmp_path: Path) -> None:
    """Plain English report plugin must run (may skip without upstream report)."""
    df = _rich_dataset(100)
    run_dir = tmp_path / "report_plain_english_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = make_context(run_dir, df, settings={}, run_seed=42)
    # Provide minimal report.json for the report plugin to consume
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "plugins": {},
                "recommendations": {
                    "items": [
                        {
                            "title": "Reduce queue wait",
                            "what": "Queue wait exceeds threshold",
                            "why": "Bottleneck in close_a",
                            "modeled_delta_hours": 5.0,
                        }
                    ]
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    module = import_module("plugins.report_plain_english_v1.plugin")
    result = module.Plugin().run(ctx)
    assert result.status in {"ok", "warn", "na"}, (
        f"report_plain_english_v1 -> {result.status}: {result.summary}"
    )
