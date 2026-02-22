# Technique Spec 01: Conformal prediction

- Plugin ID: `analysis_conformal_feature_prediction`
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
- https://arxiv.org/abs/0706.3188
