# Technique Spec 21: Kalman filter

- Plugin ID: `analysis_state_space_kalman_residuals`
- Implemented: `Y`
- Deterministic seed: `0`
- Required output contract: actionability metrics triplet (`delta_h`, `eff_%`, `eff_idx`)

## Inputs
- Required: dataset rows in normalized or source-compatible tabular form.
- Preferred signals: process identifier, timestamp columns, and numeric covariates.

## Acceptance
- Returns `ok` with actionable finding, or `na` with deterministic reason code.
- Must include plain-English recommendation text.
- Must include modeled windows for accounting-month, close-static, close-dynamic.

## References
- https://doi.org/10.1115/1.3662552
