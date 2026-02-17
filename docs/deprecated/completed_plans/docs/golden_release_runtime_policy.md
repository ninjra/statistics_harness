# Golden Release Runtime Policy

## Purpose
Define deterministic runtime policy switches used for golden-release verification runs.

## Network Policy
- `STAT_HARNESS_NETWORK_MODE=off|localhost|on`
  - `off` (default): block all network socket creation/connection in plugin subprocesses.
  - `localhost`: allow loopback-only connections (`127.0.0.1`, `::1`, `localhost`).
  - `on`: disable network guard.
- Backward compatibility:
  - `STAT_HARNESS_ALLOW_NETWORK=1` maps to `STAT_HARNESS_NETWORK_MODE=on` when explicit mode is not set.

## Large Dataset Policy
- `STAT_HARNESS_MAX_FULL_DF_ROWS` controls refusal threshold for full dataframe loads.
  - Default: `1000000`.
- `STAT_HARNESS_ALLOW_FULL_DF=1` bypasses full-load guardrails.
- For row counts above threshold, plugin code must use `ctx.dataset_iter_batches(...)`.

## Golden Execution Modes
- `STAT_HARNESS_GOLDEN_MODE=off|default|strict`
  - `off`: standard behavior.
  - `default`: allows deterministic and citable `skipped` plugin results.
  - `strict`: any `skipped` plugin result fails the run with a policy violation.

## Required Gates for Golden Runs
- `python -m pytest -q`
- `stat-harness list-plugins`
- `scripts/run_gauntlet.sh` (or Windows-safe equivalent)

