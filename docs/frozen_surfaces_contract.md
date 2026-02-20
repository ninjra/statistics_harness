# Frozen Plugin Surfaces Contract

## Goal
Lock known-good plugin surfaces so stable plugins are guarded from silent drift.

## Surface Definition
A plugin surface hash is computed from:
- plugin id/version/type/entrypoint
- effective code hash (stat-wrapper aware)
- resolved default settings hash
- `plugin.yaml` hash
- `config.schema.json` hash
- `output.schema.json` hash

## Contract File
- Path: `docs/frozen_plugin_surfaces.contract.json`
- Schema id: `frozen_surfaces_contract.v1`

## Workflow
1. Run a known-good gauntlet.
2. Freeze working plugins from that run:
   - `python3 scripts/freeze_working_plugin_surfaces.py --run-id <run_id>`
3. Verify lock integrity anytime:
   - `python3 scripts/verify_frozen_plugin_surfaces.py`
4. Enforce at runtime:
   - `STAT_HARNESS_FROZEN_SURFACES_MODE=enforce`
   - Optional custom contract path:
     - `STAT_HARNESS_FROZEN_SURFACES_PATH=/abs/path/to/contract.json`

## Runtime Modes
- `off`: contract ignored.
- `warn`: mismatches are recorded in run events/debug, plugin still executes.
- `enforce`: mismatch fails that plugin with `FROZEN_SURFACE_MISMATCH`.

## Why this helps
- Prevents accidental plugin drift after a stable release.
- Keeps recompute policy deterministic and auditable.
- Makes “working plugin stays working” an explicit, testable contract.

