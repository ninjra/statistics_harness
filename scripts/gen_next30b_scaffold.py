from __future__ import annotations

import json
from pathlib import Path


NEXT30B_PLUGIN_IDS = [
    "analysis_beta_binomial_overdispersion_v1",
    "analysis_circular_time_of_day_drift_v1",
    "analysis_mann_kendall_trend_test_v1",
    "analysis_quantile_mapping_drift_qq_v1",
    "analysis_constraints_violation_detector_v1",
    "analysis_negative_binomial_overdispersion_v1",
    "analysis_partial_correlation_network_shift_v1",
    "analysis_piecewise_linear_trend_changepoints_v1",
    "analysis_poisson_regression_rate_drivers_v1",
    "analysis_quantile_sketch_p2_streaming_v1",
    "analysis_robust_regression_huber_ransac_v1",
    "analysis_state_space_smoother_level_shift_v1",
    "analysis_aft_survival_lognormal_v1",
    "analysis_competing_risks_cif_v1",
    "analysis_haar_wavelet_transient_detector_v1",
    "analysis_hurst_exponent_long_memory_v1",
    "analysis_permutation_entropy_drift_v1",
    "analysis_capacity_frontier_envelope_v1",
    "analysis_graph_assortativity_shift_v1",
    "analysis_graph_pagerank_hotspots_v1",
    "analysis_higuchi_fractal_dimension_v1",
    "analysis_marked_point_process_intensity_v1",
    "analysis_spectral_radius_stability_v1",
    "analysis_bootstrap_ci_effect_sizes_v1",
    "analysis_energy_distance_two_sample_v1",
    "analysis_randomization_test_median_shift_v1",
    "analysis_distance_covariance_dependence_v1",
    "analysis_graph_motif_triads_shift_v1",
    "analysis_multiscale_entropy_mse_v1",
    "analysis_sample_entropy_irregularity_v1",
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
        "capabilities: [analysis]\n"
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
        "      redact_patterns: [\"email\",\"ip\",\"uuid\",\"credit_card\",\"phone\"]\n"
        "      max_exemplars: 3\n"
        "      allow_exemplar_snippets: false\n"
        "    verbosity: \"normal\"\n"
        "    plugin:\n"
        "      max_points_for_quadratic: 2000\n"
        "      max_resamples: 200\n"
        "sandbox:\n"
        "  no_network: true\n"
        "  fs_allowlist: [run_dir]\n"
    )


def main() -> None:
    root = Path("plugins")
    for plugin_id in NEXT30B_PLUGIN_IDS:
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
    print(f"generated={len(NEXT30B_PLUGIN_IDS)}")


if __name__ == "__main__":
    main()
