#!/usr/bin/env python3
"""Verify statistical correctness of all registry handlers.

Iterates over the HANDLERS dict in registry.py, routes each handler to an
appropriate synthetic dataset, runs verification checks, and writes results
to appdata/verification/handler_results.json.

Usage:
    .venv/bin/python scripts/verify_registry_handlers.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plugin_verifier_lib.check_runners import (
    DEFAULT_CHECKS,
    CheckResult,
    VerificationCase,
    VerificationResult,
    check_metric_in_range,
    run_verification,
)
from plugin_verifier_lib.issue_classifier import classify_verification_failure
from plugin_verifier_lib.minimal_context import make_minimal_context
from plugin_verifier_lib.report_builder import build_report
from plugin_verifier_lib.synthetic_datasets import (
    ds_benford,
    ds_causal_dag,
    ds_categorical_assoc,
    ds_changepoint_known,
    ds_heavy_tail,
    ds_known_clusters,
    ds_known_correlation,
    ds_known_normal,
    ds_minimal,
    ds_no_changepoint,
    ds_process_log,
    ds_seasonal,
    ds_survival,
    ds_two_pop_shift,
)

from statistic_harness.core.stat_plugins.registry import HANDLERS, run_plugin

# ---------------------------------------------------------------------------
# Handler-to-dataset routing
# ---------------------------------------------------------------------------

# Keywords in handler names -> (dataset_fn, dataset_name, extra_checks)
CATEGORY_ROUTES: list[tuple[list[str], callable, str, list]] = [
    # Changepoint detection
    (
        ["changepoint", "pelt", "edivisive", "adwin", "survey_guided",
         "binary_segmentation", "fused_lasso", "cusum_on_model",
         "change_score", "piecewise_linear", "level_shift",
         "close_cycle_change"],
        ds_changepoint_known, "ds_changepoint_known", [],
    ),
    # Control charts
    (
        ["control_chart", "multivariate_t2", "multivariate_ewma", "pca_control"],
        ds_changepoint_known, "ds_changepoint_known", [],
    ),
    # Two-sample numeric
    (
        ["two_sample_numeric", "kernel_two_sample", "energy_distance",
         "randomization_test_median", "quantile_mapping_drift"],
        ds_two_pop_shift, "ds_two_pop_shift", [],
    ),
    # Effect size / impact
    (
        ["effect_size", "change_impact", "bootstrap_ci_effect"],
        ds_two_pop_shift, "ds_two_pop_shift", [],
    ),
    # Categorical / association
    (
        ["categorical_chi2", "chi_square_association", "fisher_exact",
         "zero_inflated", "negative_binomial", "dirichlet_multinomial",
         "beta_binomial"],
        ds_categorical_assoc, "ds_categorical_assoc", [],
    ),
    # Anomaly / outlier detection
    (
        ["outlier", "local_outlier", "one_class_svm", "robust_covariance",
         "isolation_forest", "cooks_distance"],
        ds_known_clusters, "ds_known_clusters", [],
    ),
    # Extreme value theory / heavy tail
    (
        ["evt_gumbel", "evt_peaks", "heavy_tail", "hill"],
        ds_heavy_tail, "ds_heavy_tail", [],
    ),
    # Correlation / dependency / mutual information
    (
        ["mutual_information", "transfer_entropy", "lagged_predictability",
         "copula_dependence", "graphical_lasso", "dependency_graph",
         "distance_correlation", "distance_covariance",
         "partial_correlation", "multicollinearity", "cca_crossblock"],
        ds_known_correlation, "ds_known_correlation", [],
    ),
    # Survival
    (
        ["survival", "kaplan_meier", "proportional_hazards", "cox",
         "aft_survival", "competing_risks", "time_to_event"],
        ds_survival, "ds_survival", [],
    ),
    # Seasonality / periodicity / spectral
    (
        ["periodicity", "spectral", "seasonal", "stl_seasonal",
         "holt_winters", "lomb_scargle", "circular_time",
         "garch_volatility"],
        ds_seasonal, "ds_seasonal", [],
    ),
    # Benford / fraud
    (
        ["benford"],
        ds_benford, "ds_benford", [],
    ),
    # Process mining / conformance
    (
        ["conformance", "variant_differential", "process_drift",
         "markov_transition", "sequential_patterns", "hmm_latent"],
        ds_process_log, "ds_process_log", [],
    ),
    # Causal
    (
        ["causal", "copula", "granger"],
        ds_causal_dag, "ds_causal_dag", [],
    ),
    # Queueing
    (
        ["queue_model", "littles_law", "kingman"],
        ds_process_log, "ds_process_log", [],
    ),
    # Text / log / topic / burst / entropy drift
    (
        ["template_drift", "message_entropy", "topic_model", "burst"],
        ds_process_log, "ds_process_log", [],
    ),
    # Time series generic
    (
        ["matrix_profile", "kalman", "hawkes", "event_count_bocpd",
         "haar_wavelet", "hurst_exponent", "permutation_entropy",
         "recurrence_quantification", "multiscale_entropy",
         "sample_entropy", "higuchi_fractal", "marked_point_process",
         "spectral_radius"],
        ds_changepoint_known, "ds_changepoint_known", [],
    ),
    # Dimensionality / decomposition / regression
    (
        ["pca_auto", "factor_analysis", "cluster_analysis",
         "sparse_pca", "ica_source", "factor_rotation",
         "subspace_tracking", "random_matrix"],
        ds_known_correlation, "ds_known_correlation", [],
    ),
    # Statistical tests
    (
        ["ttests_auto", "anova_auto", "regression_auto",
         "time_series_analysis_auto"],
        ds_two_pop_shift, "ds_two_pop_shift", [],
    ),
    # Regression models
    (
        ["gam_spline", "quantile_loss_boosting", "quantile_regression_forest",
         "robust_regression", "poisson_regression"],
        ds_known_correlation, "ds_known_correlation", [],
    ),
    # Multiple testing
    (
        ["multiple_testing_fdr"],
        ds_two_pop_shift, "ds_two_pop_shift", [],
    ),
    # Trend
    (
        ["mann_kendall", "constraints_violation"],
        ds_seasonal, "ds_seasonal", [],
    ),
    # Topographic / surface
    (
        ["topographic", "surface_", "map_permutation"],
        ds_known_clusters, "ds_known_clusters", [],
    ),
    # TDA
    (
        ["tda_persistent", "tda_persistence", "tda_mapper", "tda_betti"],
        ds_known_clusters, "ds_known_clusters", [],
    ),
    # Graph
    (
        ["graph_assortativity", "graph_pagerank", "graph_motif"],
        ds_known_correlation, "ds_known_correlation", [],
    ),
    # Location / geometric
    (
        ["geometric_median", "bayesian_point", "monte_carlo_surface",
         "surface_roughness"],
        ds_known_clusters, "ds_known_clusters", [],
    ),
    # Capacity / operational
    (
        ["capacity_frontier", "hold_time", "retry_rate",
         "dependency_critical_path", "param_variant"],
        ds_process_log, "ds_process_log", [],
    ),
    # Ideaspace / EBM / action
    (
        ["ideaspace", "ebm_action", "actionable_ops"],
        ds_two_pop_shift, "ds_two_pop_shift", [],
    ),
    # BSTS / intervention
    (
        ["bsts_intervention"],
        ds_changepoint_known, "ds_changepoint_known", [],
    ),
    # Quantile sketch / streaming
    (
        ["quantile_sketch"],
        ds_known_normal, "ds_known_normal", [],
    ),
]


def _route_handler(handler_id: str) -> tuple[callable, str, list] | None:
    """Find the best dataset for a handler based on name keywords."""
    lower = handler_id.lower()
    for keywords, ds_fn, ds_name, extra_checks in CATEGORY_ROUTES:
        for kw in keywords:
            if kw in lower:
                return ds_fn, ds_name, extra_checks
    return None


def _build_null_case(handler_id: str, ds_fn, ds_name) -> VerificationCase | None:
    """Build a null-dataset verification case for false-positive checking."""
    # Only changepoint handlers get null-data testing
    lower = handler_id.lower()
    is_changepoint = any(
        kw in lower
        for kw in ["changepoint", "pelt", "edivisive", "adwin", "control_chart"]
    )
    if not is_changepoint:
        return None
    df_null, known_null = ds_no_changepoint()
    run_dir = Path(tempfile.mkdtemp(prefix=f"verify_{handler_id}_null_"))
    ctx = make_minimal_context(df_null, run_dir=run_dir)
    return VerificationCase(
        plugin_id=handler_id,
        dataset_name="ds_no_changepoint",
        run_fn=lambda _ctx=ctx, _hid=handler_id: run_plugin(_hid, _ctx),
        checks=list(DEFAULT_CHECKS),
        known_answers=known_null,
    )


def main() -> None:
    output_dir = ROOT / "appdata" / "verification"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_handler_ids = sorted(HANDLERS.keys())
    print(f"Found {len(all_handler_ids)} registry handlers")

    results: list[VerificationResult] = []
    issues = []
    skipped_ids = []

    for i, handler_id in enumerate(all_handler_ids):
        route = _route_handler(handler_id)
        if route is None:
            # Fallback to ds_known_normal
            route = (ds_known_normal, "ds_known_normal", [])
        ds_fn, ds_name, extra_checks = route

        print(f"  [{i + 1}/{len(all_handler_ids)}] {handler_id} -> {ds_name} ... ", end="", flush=True)

        df, known = ds_fn()
        run_dir = Path(tempfile.mkdtemp(prefix=f"verify_{handler_id}_"))
        ctx = make_minimal_context(df, run_dir=run_dir)

        case = VerificationCase(
            plugin_id=handler_id,
            dataset_name=ds_name,
            run_fn=lambda _ctx=ctx, _hid=handler_id: run_plugin(_hid, _ctx),
            checks=list(DEFAULT_CHECKS) + extra_checks,
            known_answers=known,
        )

        vr = run_verification(case)
        results.append(vr)
        print(f"{vr.status} ({vr.duration_ms:.0f}ms)")

        if vr.status in ("FAIL", "ERROR"):
            classified = classify_verification_failure(vr)
            issues.extend(classified)

        # Null-data test for changepoint handlers
        null_case = _build_null_case(handler_id, ds_fn, ds_name)
        if null_case:
            null_vr = run_verification(null_case)
            results.append(null_vr)
            if null_vr.status in ("FAIL", "ERROR"):
                classified = classify_verification_failure(null_vr)
                issues.extend(classified)

    # Write detailed results
    results_path = output_dir / "handler_results.json"
    results_data = [
        {
            "plugin_id": r.plugin_id,
            "dataset_name": r.dataset_name,
            "status": r.status,
            "plugin_status": r.plugin_status,
            "duration_ms": round(r.duration_ms, 1),
            "checks": [
                {"name": c.name, "passed": c.passed, "message": c.message}
                for c in r.check_results
            ],
            "error": r.error,
        }
        for r in results
    ]
    results_path.write_text(json.dumps(results_data, indent=2, default=str), encoding="utf-8")

    # Build report
    json_path, md_path = build_report(
        results, issues, output_dir,
        report_name="handler_verification_report",
        extra_meta={"phase": "registry_handlers", "total_handlers": len(all_handler_ids)},
    )

    # Summary
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    errored = sum(1 for r in results if r.status == "ERROR")
    print(f"\n{'='*60}")
    print(f"Registry Handler Verification Complete")
    print(f"  Handlers tested: {len(all_handler_ids)}")
    print(f"  Total verifications: {len(results)}")
    print(f"  Passed: {passed}  Failed: {failed}  Errors: {errored}")
    print(f"  Issues found: {len(issues)}")
    print(f"  Results: {results_path}")
    print(f"  Report:  {md_path}")


if __name__ == "__main__":
    main()
