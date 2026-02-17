#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=/dev/null
. "$ROOT_DIR/.venv/bin/activate"

.venv/bin/python scripts/generate_codex_plugin_catalog.py
.venv/bin/python scripts/plugin_data_access_matrix.py
.venv/bin/python scripts/redteam_ids_matrix.py
.venv/bin/python scripts/sql_assist_adoption_matrix.py
.venv/bin/python scripts/sql_adoption_partition_matrix.py
.venv/bin/python scripts/sql_adoption_execution_order.py
.venv/bin/python scripts/run_repo_improvements_pipeline.py
.venv/bin/python scripts/docs_coverage_matrix.py
.venv/bin/python scripts/binding_implementation_matrix.py --extra-doc topo-tda-addon-pack-plan.md
.venv/bin/python scripts/plugins_functionality_matrix.py
.venv/bin/python scripts/build_plugin_class_actionability_matrix.py
.venv/bin/python scripts/generate_plugin_example_cards.py
.venv/bin/python scripts/top20_methods_matrix.py
.venv/bin/python scripts/full_instruction_coverage_report.py
.venv/bin/python scripts/full_repo_misses.py
