# Technique Spec 09: Gradient Boosting Machine

- Plugin ID: `analysis_quantile_loss_boosting_v1`
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
- https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation--A-gradient-boosting-machine/10.1214/aos/1013203451.full
