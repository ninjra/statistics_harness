# Technique Spec 19: Dirichlet Process (nonparametric Bayes)

- Plugin ID: `analysis_dirichlet_multinomial_categorical_overdispersion_v1`
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
- https://projecteuclid.org/journals/annals-of-statistics/volume-1/issue-2/A-Bayesian-analysis-of-some-nonparametric-problems/10.1214/aos/1176342360.full
