# Plugins Validate Runbook Contract

`plugins validate` must emit deterministic machine-readable output and explicit
failure semantics for schema/import/smoke/capability checks.

## Contract Sources
- `docs/schemas/actionable_recommendation_contract_v2.json`
- `docs/release_evidence/plugins_validate_schema.json`
- `tests/test_cli_plugins_validate.py`
- `tests/test_actionability_contract_v2.py`

## Required Behaviors
- Deterministic output ordering for plugin rows and capability fields.
- Non-zero exit for malformed manifests, import failures, or health-check failures.
- No silent `skip` acceptance in strict validation lanes.
- Actionable and non-actionable envelopes must include window triplet fields and dependency map.

