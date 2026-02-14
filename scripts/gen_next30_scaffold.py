from __future__ import annotations

import json
from pathlib import Path


NEXT30_PLUGIN_IDS = [
    "analysis_bsts_intervention_counterfactual_v1",
    "analysis_stl_seasonal_decompose_v1",
    "analysis_seasonal_holt_winters_forecast_residuals_v1",
    "analysis_lomb_scargle_periodogram_v1",
    "analysis_garch_volatility_shift_v1",
    "analysis_bayesian_online_changepoint_studentt_v1",
    "analysis_wild_binary_segmentation_v1",
    "analysis_fused_lasso_trend_filtering_v1",
    "analysis_cusum_on_model_residuals_v1",
    "analysis_change_score_consensus_v1",
    "analysis_benfords_law_anomaly_v1",
    "analysis_geometric_median_multivariate_location_v1",
    "analysis_random_matrix_marchenko_pastur_denoise_v1",
    "analysis_outlier_influence_cooks_distance_v1",
    "analysis_heavy_tail_index_hill_v1",
    "analysis_distance_correlation_screen_v1",
    "analysis_gam_spline_regression_v1",
    "analysis_quantile_loss_boosting_v1",
    "analysis_quantile_regression_forest_v1",
    "analysis_sparse_pca_interpretable_components_v1",
    "analysis_ica_source_separation_v1",
    "analysis_cca_crossblock_association_v1",
    "analysis_factor_rotation_varimax_v1",
    "analysis_subspace_tracking_oja_v1",
    "analysis_multicollinearity_vif_screen_v1",
    "analysis_zero_inflated_count_model_v1",
    "analysis_negative_binomial_overdispersion_v1",
    "analysis_dirichlet_multinomial_categorical_overdispersion_v1",
    "analysis_fisher_exact_enrichment_v1",
    "analysis_recurrence_quantification_rqa_v1",
]


CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "stat_plugin_config",
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "seed": {"type": "integer", "default": 1337},
        "time_budget_ms": {"type": "integer", "default": 25000},
        "max_rows": {"type": ["integer", "null"], "default": 200000},
        "max_cols": {"type": ["integer", "null"], "default": 80},
        "max_pairs": {"type": ["integer", "null"], "default": 2000},
        "max_windows": {"type": ["integer", "null"], "default": 64},
        "max_findings": {"type": ["integer", "null"], "default": 30},
        "allow_row_sampling": {"type": "boolean", "default": False},
        "privacy": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "enable_redaction": {"type": "boolean", "default": True},
                "redact_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["email", "ip", "uuid", "credit_card", "phone"],
                },
                "max_exemplars": {"type": "integer", "default": 3},
                "allow_exemplar_snippets": {"type": "boolean", "default": False},
            },
            "default": {},
        },
        "verbosity": {
            "type": "string",
            "enum": ["low", "normal", "high"],
            "default": "normal",
        },
        "plugin": {
            "type": "object",
            "additionalProperties": True,
            "default": {},
        },
    },
}


OUTPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "plugin_output",
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "status": {"type": "string"},
        "summary": {"type": "string"},
        "metrics": {"type": "object"},
        "findings": {"type": "array"},
        "artifacts": {"type": "array"},
        "budget": {"type": ["object", "null"]},
        "error": {"type": ["object", "null"]},
        "references": {"type": "array"},
        "debug": {"type": "object"},
    },
    "required": ["status", "summary", "metrics", "findings", "artifacts", "error"],
}


def _plugin_py(plugin_id: str) -> str:
    return (
        "from __future__ import annotations\n\n"
        "from statistic_harness.core.stat_plugins.registry import run_plugin\n\n\n"
        "class Plugin:\n"
        "    def run(self, ctx):\n"
        f"        return run_plugin(\"{plugin_id}\", ctx)\n"
    )


def _plugin_yaml(plugin_id: str) -> str:
    return (
        f"id: {plugin_id}\n"
        f"name: \"{plugin_id}\"\n"
        "version: \"0.1.0\"\n"
        "type: analysis\n"
        "entrypoint: \"plugin.py:Plugin\"\n"
        "depends_on: []\n"
        "capabilities:\n"
        "  - analysis\n"
        "config_schema: \"config.schema.json\"\n"
        "output_schema: \"output.schema.json\"\n"
        "settings:\n"
        f"  description: \"{plugin_id}\"\n"
        "  defaults:\n"
        "    seed: 1337\n"
        "    time_budget_ms: 25000\n"
        "    max_rows: 200000\n"
        "    max_cols: 80\n"
        "    max_pairs: 2000\n"
        "    max_windows: 64\n"
        "    max_findings: 30\n"
        "    allow_row_sampling: false\n"
        "    privacy:\n"
        "      enable_redaction: true\n"
        "      redact_patterns: [\"email\", \"ip\", \"uuid\", \"credit_card\", \"phone\"]\n"
        "      max_exemplars: 3\n"
        "      allow_exemplar_snippets: false\n"
        "    verbosity: \"normal\"\n"
        "    plugin:\n"
        "      max_points_for_quadratic: 2000\n"
        "sandbox:\n"
        "  no_network: true\n"
        "  fs_allowlist:\n"
        "    - run_dir\n"
    )


def main() -> None:
    root = Path("plugins")
    for plugin_id in NEXT30_PLUGIN_IDS:
        out = root / plugin_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "plugin.py").write_text(_plugin_py(plugin_id), encoding="utf-8")
        (out / "plugin.yaml").write_text(_plugin_yaml(plugin_id), encoding="utf-8")
        (out / "config.schema.json").write_text(
            json.dumps(CONFIG_SCHEMA, indent=2) + "\n", encoding="utf-8"
        )
        (out / "output.schema.json").write_text(
            json.dumps(OUTPUT_SCHEMA, indent=2) + "\n", encoding="utf-8"
        )
    print(f"generated={len(NEXT30_PLUGIN_IDS)}")


if __name__ == "__main__":
    main()
