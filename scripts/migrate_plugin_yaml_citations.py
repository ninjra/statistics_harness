#!/usr/bin/env python3
"""Add ``citation:`` blocks to all plugin.yaml files.

Cross-references:
- ``config/plugin_kind_map.yaml`` for canonical ``finding_kind``
- ``stat_plugins/references.py::default_references_for_plugin()`` for ``reference_key``
- Plugin ID naming conventions for ``method`` and ``paper``

Usage:
    python scripts/migrate_plugin_yaml_citations.py [--dry-run]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

PLUGINS_DIR = Path(__file__).resolve().parent.parent / "plugins"
KIND_MAP_PATH = Path(__file__).resolve().parent.parent / "config" / "plugin_kind_map.yaml"


# ------------------------------------------------------------------
# Method name derivation from plugin_id
# ------------------------------------------------------------------

_METHOD_OVERRIDES: dict[str, str] = {
    "analysis_anova_auto": "ANOVA",
    "analysis_pca_auto": "Principal Component Analysis",
    "analysis_pca_control_chart": "PCA Control Chart",
    "analysis_ttests_auto": "t-Tests",
    "analysis_regression_auto": "Auto Regression",
    "analysis_cluster_analysis_auto": "Auto Clustering",
    "analysis_time_series_analysis_auto": "Auto Time-Series",
    "analysis_factor_analysis_auto": "Auto Factor Analysis",
    "analysis_dp_gmm": "Dirichlet Process Gaussian Mixture",
    "analysis_hmm_latent_state_sequences": "Hidden Markov Model",
    "analysis_local_outlier_factor": "Local Outlier Factor",
    "analysis_one_class_svm": "One-Class SVM",
    "analysis_isolation_forest": "Isolation Forest",
    "analysis_isolation_forest_anomaly": "Isolation Forest Anomaly",
    "analysis_chi_square_association": "Chi-Square Association",
    "analysis_tail_isolation": "Tail Isolation",
    "analysis_chain_makespan": "Chain Makespan",
    "analysis_percentile_analysis": "Percentile Analysis",
    "analysis_sequence_classification": "Sequence Classification",
    "analysis_bayesian_point_displacement": "Bayesian Point Displacement",
    "analysis_tda_betti_curve_changepoint": "TDA Betti Curve Changepoint",
    "analysis_tda_mapper_graph": "TDA Mapper Graph",
    "analysis_tda_persistence_landscapes": "TDA Persistence Landscapes",
    "analysis_tda_persistent_homology": "TDA Persistent Homology",
    "analysis_changepoint_pelt": "PELT Changepoint",
    "analysis_drift_adwin": "ADWIN Drift Detection",
    "analysis_bocpd_gaussian": "Bayesian Online Changepoint (Gaussian)",
    "analysis_topic_model_lda": "Latent Dirichlet Allocation",
    "analysis_survival_kaplan_meier": "Kaplan-Meier Survival",
    "analysis_survival_time_to_event": "Survival Time-to-Event",
    "analysis_proportional_hazards_duration": "Cox Proportional Hazards",
    "analysis_evt_gumbel_tail": "Extreme Value Theory (Gumbel)",
    "analysis_evt_peaks_over_threshold": "Peaks Over Threshold (EVT)",
    "analysis_mutual_information_screen": "Mutual Information Screen",
    "analysis_transfer_entropy_directional": "Transfer Entropy",
    "analysis_copula_dependence": "Copula Dependence",
    "analysis_gaussian_copula_shift": "Gaussian Copula Shift",
    "analysis_graphical_lasso_dependency_network": "Graphical Lasso",
    "analysis_effect_size_report": "Effect Size Report",
    "analysis_multiple_testing_fdr": "FDR Multiple Testing",
    "analysis_control_chart_cusum": "CUSUM Control Chart",
    "analysis_control_chart_ewma": "EWMA Control Chart",
    "analysis_control_chart_individuals": "Individuals Control Chart",
    "analysis_control_chart_suite": "Control Chart Suite",
    "analysis_matrix_profile_motifs_discords": "Matrix Profile",
    "analysis_two_sample_numeric_ks": "Kolmogorov-Smirnov Two-Sample",
    "analysis_two_sample_numeric_ad": "Anderson-Darling Two-Sample",
    "analysis_two_sample_numeric_mann_whitney": "Mann-Whitney U Two-Sample",
    "analysis_two_sample_categorical_chi2": "Chi-Square Two-Sample (Categorical)",
    "analysis_kernel_two_sample_mmd": "MMD Kernel Two-Sample",
    "analysis_notears_linear": "NOTEARS Linear",
    "analysis_process_sequence": "Process Sequence Analysis",
    "analysis_process_sequence_bottlenecks": "Process Bottleneck Analysis",
    "analysis_conformance_alignments": "Conformance Alignment",
    "analysis_conformance_checking": "Conformance Checking",
    "analysis_littles_law_consistency": "Little's Law Consistency",
    "analysis_attribution": "Attribution Analysis",
    "analysis_quantile_regression_duration": "Quantile Regression (Duration)",
    "analysis_queue_model_fit": "Queue Model Fit",
    "analysis_queue_delay_decomposition": "Queue Delay Decomposition",
    "analysis_kingman_vut_approx": "Kingman VUT Approximation",
}

_PAPER_OVERRIDES: dict[str, str] = {
    "analysis_changepoint_pelt": "Killick et al. (2012)",
    "analysis_drift_adwin": "Bifet & Gavalda (2007)",
    "analysis_bocpd_gaussian": "Adams & MacKay (2007)",
    "analysis_local_outlier_factor": "Breunig et al. (2000)",
    "analysis_isolation_forest": "Liu et al. (2008)",
    "analysis_isolation_forest_anomaly": "Liu et al. (2008)",
    "analysis_survival_kaplan_meier": "Kaplan & Meier (1958)",
    "analysis_proportional_hazards_duration": "Cox (1972)",
    "analysis_transfer_entropy_directional": "Schreiber (2000)",
    "analysis_graphical_lasso_dependency_network": "Friedman et al. (2008)",
    "analysis_topic_model_lda": "Blei et al. (2003)",
    "analysis_kernel_two_sample_mmd": "Gretton et al. (2012)",
    "analysis_multiple_testing_fdr": "Benjamini & Hochberg (1995)",
    "analysis_matrix_profile_motifs_discords": "Yeh et al. (2016)",
    "analysis_notears_linear": "Zheng et al. (2018)",
    "analysis_bayesian_point_displacement": "Tanir et al. (2008)",
    "analysis_tda_persistent_homology": "Edelsbrunner et al. (2002)",
    "analysis_littles_law_consistency": "Little (1961)",
    "analysis_kingman_vut_approx": "Kingman (1961)",
    "analysis_evt_gumbel_tail": "Fisher & Tippett (1928)",
    "analysis_control_chart_ewma": "Roberts (1959)",
    "analysis_control_chart_cusum": "Page (1954)",
    "analysis_hawkes_self_exciting": "Hawkes (1971)",
    "analysis_hmm_latent_state_sequences": "Baum et al. (1970)",
    "analysis_dp_gmm": "Blei & Jordan (2006)",
    "analysis_chi_square_association": "Pearson (1900)",
}


def _method_from_id(plugin_id: str) -> str:
    """Derive a human-readable method name from plugin_id."""
    if plugin_id in _METHOD_OVERRIDES:
        return _METHOD_OVERRIDES[plugin_id]
    name = plugin_id.removeprefix("analysis_")
    name = re.sub(r"_v\d+$", "", name)
    parts = name.split("_")
    return " ".join(w.capitalize() for w in parts)


def _reference_key_from_id(plugin_id: str) -> str:
    """Derive a reference key from plugin_id for cross-referencing with references.py."""
    pid = plugin_id.lower()
    if "pelt" in pid or "changepoint" in pid:
        return "pelt"
    if "adwin" in pid:
        return "adwin"
    if "isolation_forest" in pid:
        return "isolation_forest"
    if "local_outlier_factor" in pid or "lof" in pid:
        return "lof"
    if "one_class_svm" in pid:
        return "ocsvm"
    if "control_chart" in pid:
        return "control_chart"
    if "robust_pca" in pid or "pcp" in pid:
        return "robust_pca"
    if "mmd" in pid or "kernel_two_sample" in pid:
        return "mmd"
    if "fdr" in pid:
        return "bh_fdr"
    if "drain" in pid or "template_mining" in pid:
        return "drain"
    if "conformance" in pid or "process_sequence" in pid:
        return "process_mining"
    if "evt" in pid or "gumbel" in pid or "peaks_over" in pid:
        return "evt"
    if "matrix_profile" in pid:
        return "matrix_profile"
    if "burst" in pid or "kleinberg" in pid:
        return "kleinberg_burst"
    if "hawkes" in pid:
        return "hawkes"
    if "kalman" in pid or "state_space" in pid:
        return "kalman"
    if "lda" in pid or "topic_model" in pid:
        return "lda"
    if "hmm" in pid:
        return "hmm"
    if "transfer_entropy" in pid:
        return "transfer_entropy"
    if "granger" in pid or "lagged_predictability" in pid:
        return "granger"
    if "copula" in pid:
        return "copula"
    if "graphical_lasso" in pid:
        return "glasso"
    if "kaplan_meier" in pid or "survival" in pid:
        return "kaplan_meier"
    if "proportional_hazards" in pid or "cox" in pid:
        return "cox"
    if "quantile_regression" in pid:
        return "quantile_regression"
    if "tda" in pid:
        return "tda_survey"
    if "notears" in pid:
        return "notears"
    if "little" in pid:
        return "little"
    if "kingman" in pid:
        return "kingman"
    name = plugin_id.removeprefix("analysis_")
    name = re.sub(r"_v\d+$", "", name)
    return name


def migrate_plugin(plugin_dir: Path, kind: str, dry_run: bool) -> bool:
    """Add citation block to plugin.yaml. Returns True if modified."""
    yaml_path = plugin_dir / "plugin.yaml"
    if not yaml_path.exists():
        return False

    content = yaml_path.read_text(encoding="utf-8")

    if "citation:" in content:
        return False  # already has citation

    plugin_id = plugin_dir.name
    method = _method_from_id(plugin_id)
    reference_key = _reference_key_from_id(plugin_id)
    paper = _PAPER_OVERRIDES.get(plugin_id, "")

    citation_block = f"""
citation:
  method: "{method}"
  reference_key: "{reference_key}"
  paper: "{paper}"
  finding_kind: "{kind}"
"""

    if not dry_run:
        yaml_path.write_text(content.rstrip() + "\n" + citation_block, encoding="utf-8")

    return True


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    kind_map = yaml.safe_load(KIND_MAP_PATH.read_text(encoding="utf-8"))
    mappings = kind_map.get("mappings", {})

    migrated = 0
    skipped = 0
    no_kind = 0

    for plugin_dir in sorted(PLUGINS_DIR.iterdir()):
        if not plugin_dir.is_dir():
            continue
        yaml_path = plugin_dir / "plugin.yaml"
        if not yaml_path.exists():
            continue

        kind = mappings.get(plugin_dir.name, "")
        if not kind and plugin_dir.name.startswith("analysis_"):
            no_kind += 1
            continue

        if not kind:
            # Non-analysis plugins: use their category prefix as kind
            for prefix in ("profile_", "transform_", "report_", "llm_", "ingest_", "planner_"):
                if plugin_dir.name.startswith(prefix):
                    kind = prefix.rstrip("_")
                    break
            if not kind:
                kind = "unknown"

        if migrate_plugin(plugin_dir, kind, dry_run):
            migrated += 1
        else:
            skipped += 1

    prefix = "[DRY RUN] " if dry_run else ""
    print(f"{prefix}Migrated: {migrated}")
    print(f"{prefix}Skipped (already has citation): {skipped}")
    if no_kind:
        print(f"{prefix}Analysis plugins without kind mapping: {no_kind}")


if __name__ == "__main__":
    main()
