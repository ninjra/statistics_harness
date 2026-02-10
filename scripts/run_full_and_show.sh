#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT="$(bash scripts/start_full_loaded_dataset_bg.sh "$@")"
echo "$OUT"

RUN_ID="$(echo "$OUT" | awk -F= '/^RUN_ID=/{print $2}' | tail -n 1)"
if [[ -z "$RUN_ID" ]]; then
  echo "ERROR: failed to detect RUN_ID (see LOG line above)"
  exit 1
fi

INTERVAL_SECONDS="${3:-30}"
bash scripts/watch_run_until_done.sh "$RUN_ID" "$INTERVAL_SECONDS"
.venv/bin/python scripts/show_actionable_results.py --run-id "$RUN_ID"

