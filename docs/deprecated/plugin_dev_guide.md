# Plugin Development Guide

Plugins live in `plugins/<plugin_id>` and provide a `plugin.yaml` manifest with
an entrypoint class that implements `run(ctx)`.

## Manifest + Schemas

Each plugin must ship:

- `plugin.yaml` (validated against `docs/plugin_manifest.schema.json`)
- `config.schema.json` (defaults validated on load)
- `output.schema.json` (validated after execution)

Manifests define `type`, `entrypoint`, `depends_on`, `capabilities`, and `sandbox`
(`no_network` + `fs_allowlist`).

## Output Requirements

Findings must include:

- `measurement_type`: `measured | modeled | not_applicable | error`
- `evidence`: dataset_id, dataset_version_id, row_ids, column_ids, and query

All plugins should emit a `budget` object (row_limit/sample flags/time/CPU budgets)
even when defaults are “unlimited”.
