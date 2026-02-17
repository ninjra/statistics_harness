# Result Quality Contract

## Purpose
Define deterministic output rules so plugin runs cannot silently look successful when they are non-actionable.

## Terminal Status Policy
- Allowed terminal statuses at runtime: `ok`, `error`, `aborted`, `na`.
- Legacy statuses (`skipped`, `degraded`, `not_applicable`) are normalized to `na`.

## `ok` Finding Requirement
- For analysis plugins, `ok` requires at least one finding unless the plugin advertises capability `diagnostic_only`.
- If an analysis plugin returns `ok` with zero findings, the pipeline records a policy violation and the run fails overall.

## Deterministic `reason_code` Enum
- `SQL_ASSIST_SCHEMA_UNAVAILABLE`
- `FEATURE_DISABLED`
- `QUADRATIC_CAP_EXCEEDED`
- `INSUFFICIENT_POSITIVE_SAMPLES`
- `NO_ELIGIBLE_SLICE`
- `NO_FEATURES_ELIGIBLE`
- `NO_SIGNIFICANT_EFFECT`
- `MISSING_PREREQUISITE`
- `NOT_APPLICABLE`
- `NO_ACTIONABLE_RESULT`
- `SQL_ASSIST_READY`
- `SQL_PACK_BOOTSTRAPPED`
- `SQLPACK_MATERIALIZATION_FAILED`
- `TEMPLATE_ID_MISSING`
- `TEMPLATE_MAPPING_APPLIED`

## SQL-Assist Hard-Fail Policy
- Plugins marked with capability `sql_assist_required` must fail with `error` if SQL assist prerequisites are unavailable.
- This is profile-scoped by plugin selection: if no `sql_assist_required` plugin is selected, no SQL-assist hard-fail check is triggered.
