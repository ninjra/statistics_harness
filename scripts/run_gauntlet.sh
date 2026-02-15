#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Ensure `python -m pytest -q` works as required by project policy.
source .venv/bin/activate

stat-harness list-plugins >/dev/null
python scripts/verify_docs_and_plugin_matrices.py
python -m pytest -q
