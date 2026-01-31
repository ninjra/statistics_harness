#!/usr/bin/env bash
set -euo pipefail
. .venv/bin/activate
stat-harness serve --host 127.0.0.1 --port 8000
