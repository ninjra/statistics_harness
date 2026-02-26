#!/usr/bin/env bash
set -euo pipefail

PATTERN='torch/_inductor/compile_worker'
MAX_WORKERS="${STAT_HARNESS_CW_MAX_WORKERS:-8}"
MAX_RSS_MB="${STAT_HARNESS_CW_MAX_RSS_MB:-3072}"
INTERVAL_SEC=20
RUN_ONCE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --once)
      RUN_ONCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --interval-sec)
      INTERVAL_SEC="${2:-20}"
      shift 2
      ;;
    --max-workers)
      MAX_WORKERS="${2:-8}"
      shift 2
      ;;
    --max-rss-mb)
      MAX_RSS_MB="${2:-3072}"
      shift 2
      ;;
    *)
      echo "unknown_arg=$1" >&2
      exit 2
      ;;
  esac
done

if ! [[ "$MAX_WORKERS" =~ ^[0-9]+$ ]]; then
  echo "invalid_max_workers=$MAX_WORKERS" >&2
  exit 2
fi
if ! [[ "$MAX_RSS_MB" =~ ^[0-9]+$ ]]; then
  echo "invalid_max_rss_mb=$MAX_RSS_MB" >&2
  exit 2
fi
if ! [[ "$INTERVAL_SEC" =~ ^[0-9]+$ ]]; then
  echo "invalid_interval_sec=$INTERVAL_SEC" >&2
  exit 2
fi

sample_once() {
  local pids count rss_kb rss_mb action ts
  mapfile -t pids < <(pgrep -f "$PATTERN" || true)
  count="${#pids[@]}"
  rss_kb=0
  if (( count > 0 )); then
    rss_kb="$(ps -o rss= -p "${pids[@]}" 2>/dev/null | awk '{s+=$1} END{print s+0}')"
  fi
  rss_mb=$((rss_kb / 1024))
  action="ok"
  if (( count > MAX_WORKERS || rss_mb > MAX_RSS_MB )); then
    action="reap_needed"
    if (( DRY_RUN == 0 )); then
      pkill -f "$PATTERN" || true
      action="reaped"
    fi
  fi
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'ts=%s;workers=%d;rss_mb=%d;max_workers=%d;max_rss_mb=%d;action=%s\n' \
    "$ts" "$count" "$rss_mb" "$MAX_WORKERS" "$MAX_RSS_MB" "$action"
}

if (( RUN_ONCE == 1 )); then
  sample_once
  exit 0
fi

while true; do
  sample_once
  sleep "$INTERVAL_SEC"
done

