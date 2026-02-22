# Technique Spec 26: Non-negative Matrix Factorization (NMF)

- Plugin ID: `analysis_nonnegative_matrix_factorization_v1`
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
- https://doi.org/10.1038/44565
