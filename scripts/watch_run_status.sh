#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=/dev/null
. "$ROOT_DIR/.venv/bin/activate"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

RUN_ID="${1:-}"
INTERVAL_SECONDS="${2:-30}"

if [[ -z "$RUN_ID" ]]; then
  echo "usage: $0 <run-id> [interval-seconds]"
  exit 2
fi

while true; do
  echo "----- $(date -Is) -----"
  python scripts/repair_stale_running_runs.py >/dev/null 2>&1 || true
  python scripts/run_run_status.py --run-id "$RUN_ID" || true
  sleep "$INTERVAL_SECONDS"
done
