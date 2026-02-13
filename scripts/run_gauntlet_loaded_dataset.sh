#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=/dev/null
. "$ROOT_DIR/.venv/bin/activate"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# "Full gauntlet" for the loaded 500k+ row test dataset:
# 1) docs/matrix verification
# 2) pytest gate (project policy)
# 3) full plugin run (profile+planner+transform+analysis+report+llm) on the loaded dataset
python scripts/verify_docs_and_plugin_matrices.py
python -m pytest -q

# Args are forwarded as:
#   <dataset-version-id> <run-seed> <interval-seconds>
bash scripts/run_full_and_show.sh "$@"

