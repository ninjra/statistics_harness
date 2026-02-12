#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
. .venv/bin/activate
python scripts/update_docs_and_plugin_matrices.py
python scripts/full_instruction_coverage_report.py
python scripts/full_repo_misses.py
python -m pytest -q
