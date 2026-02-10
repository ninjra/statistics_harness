#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LINTER="/home/justi/.codex/skills/shell-lint-ps-wsl/scripts/shell-lint.mjs"
CMD="bash scripts/watch_latest_run.sh"
printf '%s\n' "$CMD" | node "$LINTER" --shell bash >/dev/null

exec bash scripts/watch_latest_run.sh "$@"

