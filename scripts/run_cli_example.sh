#!/usr/bin/env bash
set -euo pipefail
. .venv/bin/activate
stat-harness run --file tests/fixtures/synth_linear.csv --plugins ingest_tabular,profile_basic,report_bundle --run-seed 42
