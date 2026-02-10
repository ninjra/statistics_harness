#!/usr/bin/env bash
set -euo pipefail

# Lint (via the Codex shell-lint-ps-wsl skill) before executing any bash command.
# Usage:
#   scripts/run_linted_bash.sh 'cd ... && rg ...'
#
# Note: this is a developer UX helper; it is not used at runtime.

if [[ "${1:-}" == "" ]]; then
  echo "usage: $0 '<bash command string>'" >&2
  exit 2
fi

cmd="$1"

LINTER="/home/justi/.codex/skills/shell-lint-ps-wsl/scripts/shell-lint.mjs"
if [[ ! -f "$LINTER" ]]; then
  echo "shell linter not found at: $LINTER" >&2
  exit 2
fi

printf '%s\n' "$cmd" | node "$LINTER" --shell bash >/dev/null

bash -lc "$cmd"

