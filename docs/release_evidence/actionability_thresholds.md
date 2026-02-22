# Actionability Thresholds (Runbook Plugins)

Source of truth: `config/actionability_thresholds.yaml`

- `delta_hours_accounting_month >= 0.25`
- `eff_pct_accounting_month >= 2.0`
- `eff_idx_accounting_month >= 0.2`
- `confidence >= 0.15`

Scoring policy:

- `require_all = true` (all four checks must pass)
- `fallback_status = na`
- `fallback_reason_code = BELOW_ACTIONABILITY_THRESHOLD`

Code path:

- Loader: `src/statistic_harness/core/actionability_thresholds.py`
- First adopter: `src/statistic_harness/core/stat_plugins/runbook30_surrogates.py`
