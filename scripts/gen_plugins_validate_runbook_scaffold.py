from __future__ import annotations

import json
from pathlib import Path


MISSING_PLUGIN_IDS = [
    "analysis_elastic_net_regularized_glm_v1",
    "analysis_minimum_covariance_determinant_v1",
    "analysis_gaussian_process_regression_v1",
    "analysis_mixed_effects_hierarchical_v1",
    "analysis_bart_uplift_surrogate_v1",
    "analysis_granger_causality_v1",
    "analysis_nonnegative_matrix_factorization_v1",
    "analysis_tsne_embedding_v1",
    "analysis_umap_embedding_v1",
    "analysis_mice_imputation_chained_equations_v1",
]


CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "plugins_validate_runbook_config",
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "max_rows": {"type": ["integer", "null"], "default": 3000},
        "max_cols": {"type": ["integer", "null"], "default": 12},
        "seed": {"type": "integer", "default": 0},
    },
}


OUTPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "plugins_validate_runbook_output",
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "status": {"type": "string", "enum": ["ok", "na", "error", "skipped"]},
        "summary": {"type": "string"},
        "metrics": {"type": "object"},
        "findings": {"type": "array"},
        "artifacts": {"type": "array"},
        "budget": {"type": ["object", "null"]},
        "error": {"type": ["object", "null"]},
        "references": {"type": "array"},
        "debug": {"type": "object"},
    },
    "required": ["status", "summary", "metrics", "findings", "artifacts", "references", "debug"],
}


def _plugin_py(plugin_id: str) -> str:
    return (
        "from __future__ import annotations\n\n"
        "from statistic_harness.core.stat_plugins.runbook30_surrogates import run_surrogate\n\n\n"
        "class Plugin:\n"
        "    def run(self, ctx):\n"
        f"        return run_surrogate(\"{plugin_id}\", ctx)\n"
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
        "    max_rows: 3000\n"
        "    max_cols: 12\n"
        "    seed: 0\n"
        "sandbox:\n"
        "  no_network: true\n"
        "  fs_allowlist:\n"
        "    - run_dir\n"
    )


def main() -> None:
    plugins_root = Path("plugins")
    created = 0
    for plugin_id in MISSING_PLUGIN_IDS:
        out = plugins_root / plugin_id
        if not out.exists():
            created += 1
        out.mkdir(parents=True, exist_ok=True)
        (out / "plugin.py").write_text(_plugin_py(plugin_id), encoding="utf-8")
        (out / "plugin.yaml").write_text(_plugin_yaml(plugin_id), encoding="utf-8")
        (out / "config.schema.json").write_text(
            json.dumps(CONFIG_SCHEMA, indent=2) + "\n", encoding="utf-8"
        )
        (out / "output.schema.json").write_text(
            json.dumps(OUTPUT_SCHEMA, indent=2) + "\n", encoding="utf-8"
        )
    print(f"generated={len(MISSING_PLUGIN_IDS)} created={created}")


if __name__ == "__main__":
    main()
