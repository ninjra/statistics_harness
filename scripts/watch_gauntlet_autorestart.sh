#!/usr/bin/env bash
set -euo pipefail

LOG_PATH="/mnt/d/projects/statistics_harness/statistics_harness/appdata/gauntlet_last.log"
PROC_PATTERN="stat_harness_run_gauntlet.py"
INTERVAL_SECONDS=30
RUN_CMD="/usr/bin/pwsh -NoProfile -ExecutionPolicy Bypass -File /mnt/d/projects/statistics_harness/statistics_harness/scripts/run_gauntlet_latest.ps1"

while true; do
  echo "----- $(date -Is) -----"
  pid="$(pgrep -f "$PROC_PATTERN" | head -n 1 || true)"
  if [ -n "$pid" ]; then
    etime="$(ps -o etime= -p "$pid" | tr -d ' ')"
    echo "PID=$pid ELAPSED=$etime"
  else
    echo "PID=none ELAPSED=not_running"
    nohup bash -lc "$RUN_CMD" >/tmp/gauntlet_autorestart.out 2>&1 &
    echo "RESTARTED=1"
  fi
  if [ -f "$LOG_PATH" ]; then
    tail -n 10 "$LOG_PATH"
  else
    echo "LOG_MISSING=$LOG_PATH"
  fi
  sleep "$INTERVAL_SECONDS"
done
