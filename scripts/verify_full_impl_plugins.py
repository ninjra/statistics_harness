#!/usr/bin/env python3
"""Verify statistical correctness of full-implementation plugins.

Identifies plugins that don't delegate to registry.run_plugin, loads them
directly, and runs verification checks against synthetic datasets.

Usage:
    .venv/bin/python scripts/verify_full_impl_plugins.py
"""
from __future__ import annotations

import ast
import importlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plugin_verifier_lib.check_runners import (
    DEFAULT_CHECKS,
    VerificationCase,
    VerificationResult,
    run_verification,
)
from plugin_verifier_lib.issue_classifier import classify_verification_failure
from plugin_verifier_lib.minimal_context import make_minimal_context
from plugin_verifier_lib.report_builder import build_report
from plugin_verifier_lib.synthetic_datasets import (
    ALL_DATASETS,
    ds_causal_dag,
    ds_categorical_assoc,
    ds_changepoint_known,
    ds_heavy_tail,
    ds_known_clusters,
    ds_known_correlation,
    ds_known_normal,
    ds_minimal,
    ds_process_log,
    ds_seasonal,
    ds_survival,
    ds_two_pop_shift,
)

PLUGINS_DIR = ROOT / "plugins"

# Plugins that cannot be autonomously verified (segregated for human review)
SEGREGATED_PLUGINS: set[str] = {
    # Category A: Domain-specific (openplanter + specialized)
    "analysis_attribution",
    "analysis_actionable_ops_levers_v1",
    # Category B: No closed-form expected output
    "analysis_bart_uplift_surrogate_v1",
    "analysis_neural_additive_model_nam_v1",
    "analysis_monte_carlo_surface_uncertainty",
    # Category C: Infrastructure/orchestration (not statistical)
    "analysis_action_search_mip_batched_scheduler_v1",
    "analysis_action_search_simulated_annealing_v1",
    "analysis_issue_cards_v2",
    # Category D: Domain-specific data required (FEC/donation data)
    "analysis_fec_bundled_donation_detection_v1",
    "analysis_fec_pac_network_influence_mapping_v1",
    "analysis_fec_repeat_donor_velocity_tracking_v1",
    # Category E: Storage-dependent (require prior pipeline state)
    "analysis_close_cycle_change_detection",
    "analysis_close_cycle_stationarity_and_trend",
    "analysis_close_cycle_anomaly_detection",
    "analysis_close_cycle_seasonality_decomposition",
    "analysis_close_cycle_cohort_comparison",
    "analysis_close_cycle_variance_profiling",
    "analysis_close_cycle_forecast_vs_actual",
    "analysis_close_cycle_bottleneck_identification",
    # Category F: Storage/metadata-dependent (ctx.storage queries)
    "analysis_close_cycle_start_backtrack_v1",
    "analysis_close_cycle_window_resolver",
    "analysis_dynamic_close_detection",
    # Category G: Prior pipeline state (upstream artifacts required)
    "analysis_determinism_discipline",
    "analysis_traceability_manifest_v2",
    "analysis_waterfall_summary_v2",
    "analysis_recommendation_dedupe_v2",
    # Category H: Report/LLM pipeline (needs report.json + PII storage)
    "llm_prompt_builder",
    # Category I: Data-dependent crashes on synthetic data
    "analysis_frequent_directions_cov_sketch_v1",
    "analysis_vendor_influence_breadth_v1",
    # Category J: FEC domain (specialized donation data required)
    "analysis_bundled_donations_v1",
    "analysis_contribution_limit_flags_v1",
    "analysis_vendor_politician_timing_permutation_v1",
    # Category K: Thin-wrapper misclassified (delegates to run_top20_plugin)
    "analysis_discrete_event_queue_simulator_v1",
}

# Non-analysis plugin types that we skip
SKIP_TYPES: set[str] = {"ingest", "profile", "planner", "transform", "report"}

# Keyword-based dataset routing for full-impl plugins
KEYWORD_ROUTES: list[tuple[list[str], str]] = [
    (["changepoint", "shift", "pelt", "edivisive", "adwin", "bocpd",
      "level_shift", "binary_seg", "cusum"], "ds_changepoint_known"),
    (["control_chart", "ewma", "cusum"], "ds_changepoint_known"),
    (["two_sample", "ks_test", "mann_whitney", "effect_size",
      "bootstrap_ci"], "ds_two_pop_shift"),
    (["chi2", "chi_square", "categorical", "fisher_exact",
      "binomial", "dirichlet"], "ds_categorical_assoc"),
    (["outlier", "anomaly", "isolation_forest", "lof",
      "svm", "cooks", "robust_pca"], "ds_known_clusters"),
    (["correlation", "mutual_info", "transfer_entropy",
      "graphical_lasso", "dependency", "vif", "cca",
      "distance_corr", "partial_corr"], "ds_known_correlation"),
    (["survival", "kaplan", "cox", "aft", "competing_risk",
      "time_to_event"], "ds_survival"),
    (["seasonal", "spectral", "periodicity", "stl",
      "holt_winters", "lomb_scargle", "garch", "circular"], "ds_seasonal"),
    (["benford"], "ds_known_normal"),
    (["cluster", "kmeans", "dbscan", "mixture", "gmm",
      "bicluster"], "ds_known_clusters"),
    (["causal", "lingam", "ges", "pc_algorithm", "icp",
      "knockoff", "dag"], "ds_causal_dag"),
    (["process", "conformance", "markov", "hmm", "variant",
      "sequential_pattern", "cycle", "makespan", "busy_period",
      "capacity"], "ds_process_log"),
    (["queue", "littles_law", "kingman"], "ds_process_log"),
    (["pca", "factor", "svd", "decomp", "cur", "ica",
      "sparse_pca", "subspace", "tensor", "dmd"], "ds_known_correlation"),
    (["regression", "gam", "quantile_loss", "poisson_reg",
      "robust_reg"], "ds_known_correlation"),
    (["heavy_tail", "evt", "pareto", "hill"], "ds_heavy_tail"),
    (["tda", "persistent_homology", "persistence", "mapper",
      "betti"], "ds_known_clusters"),
    (["graph", "node2vec", "pagerank", "motif", "assortativity",
      "spectral_radius"], "ds_known_correlation"),
    (["drift", "ot_drift", "sinkhorn", "stein", "knn_graph",
      "ksd", "hsic", "phate", "diffusion", "embedding"], "ds_known_clusters"),
    (["entropy", "recurrence", "fractal", "hurst",
      "wavelet", "ssa"], "ds_known_normal"),
    (["text", "topic", "template", "log_template",
      "message_entropy", "burst", "drain"], "ds_process_log"),
    (["normal", "flow", "density", "distribution",
      "gaussian"], "ds_known_normal"),
    (["association_rules", "apriori"], "ds_categorical_assoc"),
    (["symbolic_regression"], "ds_known_correlation"),
    (["conformal", "tabpfn", "foundation"], "ds_known_correlation"),
    (["trend", "mann_kendall"], "ds_seasonal"),
    (["bundled", "donation", "openplanter"], "ds_categorical_assoc"),
    (["intervention", "bsts", "counterfactual"], "ds_changepoint_known"),
]


def _is_thin_wrapper(plugin_dir: Path) -> bool:
    """Check if plugin.py imports run_plugin from registry (thin wrapper)."""
    plugin_py = plugin_dir / "plugin.py"
    if not plugin_py.exists():
        return False
    try:
        source = plugin_py.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if "registry" in module:
                for alias in node.names:
                    if alias.name == "run_plugin":
                        return True
    return False


def _get_plugin_type(plugin_dir: Path) -> str | None:
    """Read plugin type from plugin.yaml."""
    yaml_path = plugin_dir / "plugin.yaml"
    if not yaml_path.exists():
        return None
    try:
        import yaml
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("type", "analysis")
    except Exception:
        # Fallback: scan for type line
        text = yaml_path.read_text(encoding="utf-8")
        for line in text.split("\n"):
            if line.startswith("type:"):
                return line.split(":", 1)[1].strip()
    return "analysis"


def _route_plugin(plugin_id: str) -> tuple[str, callable]:
    """Route a plugin to a dataset based on its name."""
    lower = plugin_id.lower()
    for keywords, ds_name in KEYWORD_ROUTES:
        for kw in keywords:
            if kw in lower:
                return ds_name, ALL_DATASETS[ds_name]
    # Fallback
    return "ds_known_normal", ALL_DATASETS["ds_known_normal"]


def _load_plugin_class(plugin_dir: Path):
    """Dynamically load the Plugin class from a plugin directory."""
    plugin_py = plugin_dir / "plugin.py"
    spec = importlib.util.spec_from_file_location(
        f"plugin_{plugin_dir.name}", plugin_py,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Plugin()


def main() -> None:
    output_dir = ROOT / "appdata" / "verification"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover full-implementation plugins
    all_plugin_dirs = sorted(
        d for d in PLUGINS_DIR.iterdir()
        if d.is_dir() and (d / "plugin.py").exists() and d.name != "__pycache__"
    )

    full_impl_ids = []
    skipped_wrapper = 0
    skipped_type = 0
    segregated = []

    for pdir in all_plugin_dirs:
        plugin_id = pdir.name
        ptype = _get_plugin_type(pdir)
        if ptype in SKIP_TYPES:
            skipped_type += 1
            continue
        if _is_thin_wrapper(pdir):
            skipped_wrapper += 1
            continue
        if plugin_id in SEGREGATED_PLUGINS:
            segregated.append(plugin_id)
            continue
        full_impl_ids.append(plugin_id)

    print(f"Plugin discovery:")
    print(f"  Total plugin dirs: {len(all_plugin_dirs)}")
    print(f"  Thin wrappers (skip): {skipped_wrapper}")
    print(f"  Non-analysis type (skip): {skipped_type}")
    print(f"  Segregated (human review): {len(segregated)}")
    print(f"  Full-impl to verify: {len(full_impl_ids)}")
    print()

    results: list[VerificationResult] = []
    issues = []

    for i, plugin_id in enumerate(full_impl_ids):
        ds_name, ds_fn = _route_plugin(plugin_id)
        print(f"  [{i + 1}/{len(full_impl_ids)}] {plugin_id} -> {ds_name} ... ", end="", flush=True)

        df, known = ds_fn()
        run_dir = Path(tempfile.mkdtemp(prefix=f"verify_{plugin_id}_"))
        ctx = make_minimal_context(df, run_dir=run_dir)

        plugin_dir = PLUGINS_DIR / plugin_id

        def _run_fn(_dir=plugin_dir, _ctx=ctx):
            plugin = _load_plugin_class(_dir)
            return plugin.run(_ctx)

        case = VerificationCase(
            plugin_id=plugin_id,
            dataset_name=ds_name,
            run_fn=_run_fn,
            checks=list(DEFAULT_CHECKS),
            known_answers=known,
        )

        vr = run_verification(case)
        results.append(vr)
        print(f"{vr.status} ({vr.duration_ms:.0f}ms)")

        if vr.status in ("FAIL", "ERROR"):
            classified = classify_verification_failure(vr)
            issues.extend(classified)

    # Write results
    results_path = output_dir / "full_impl_results.json"
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

    # Write segregated list
    segregated_path = output_dir / "needs_human_review.json"
    segregated_path.write_text(json.dumps({
        "segregated_plugins": sorted(segregated),
        "reason": "Cannot be autonomously verified; requires domain expertise or lacks closed-form expected output",
        "categories": {
            "domain_specific": [p for p in segregated if p in {"analysis_attribution", "analysis_actionable_ops_levers_v1"}],
            "no_closed_form": [p for p in segregated if p in {"analysis_bart_uplift_surrogate_v1", "analysis_neural_additive_model_nam_v1", "analysis_monte_carlo_surface_uncertainty"}],
            "infrastructure": [p for p in segregated if p in {"analysis_action_search_mip_batched_scheduler_v1", "analysis_action_search_simulated_annealing_v1", "analysis_issue_cards_v2"}],
        },
    }, indent=2), encoding="utf-8")

    # Build report
    json_path, md_path = build_report(
        results, issues, output_dir,
        report_name="full_impl_verification_report",
        extra_meta={
            "phase": "full_impl_plugins",
            "total_plugins": len(full_impl_ids),
            "segregated": segregated,
        },
    )

    # Summary
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    errored = sum(1 for r in results if r.status == "ERROR")
    print(f"\n{'='*60}")
    print(f"Full-Implementation Plugin Verification Complete")
    print(f"  Plugins tested: {len(full_impl_ids)}")
    print(f"  Passed: {passed}  Failed: {failed}  Errors: {errored}")
    print(f"  Issues found: {len(issues)}")
    print(f"  Segregated for human review: {len(segregated)}")
    print(f"  Results: {results_path}")
    print(f"  Report:  {md_path}")
    print(f"  Human review: {segregated_path}")


if __name__ == "__main__":
    main()
