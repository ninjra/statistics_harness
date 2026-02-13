# First Release Execution Readout

This note captures the run evidence used for the first GitHub pin/release decision.

## Runs Used

- Baseline real dataset (Quorum ERP):
  - dataset_version_id: `3246cc7cd7d57a317ddc05e80e6f6f5bfe7f50deb0ee7af8db50d04bae180e1a`
  - run_id: `eb325f419aa64de392bc941d09d16f02`
- Synthetic comparison dataset (Quorum ERP):
  - dataset_version_id: `de7c1da5a4ea6e8c684872d7857bb608492f63a9c7e0b7ca014fa0f093a88e66`
  - run_id: `d925e0f58fc349fb8fa485a1a6a7e308`

## Plugin Execution Coverage

- `eb325f419aa64de392bc941d09d16f02`:
  - plugin executions: `169`
  - status breakdown: `140 ok`, `29 skipped`, `0 errors`
  - outputs present: `125` plugins with findings, `141` with metrics, `169` with summary text
- `d925e0f58fc349fb8fa485a1a6a7e308`:
  - plugin executions: `26`
  - status breakdown: `25 ok`, `1 skipped`, `0 errors`
  - outputs present: `21` plugins with findings, `24` with metrics, `26` with summary text

Interpretation:
- The synthetic run above was a targeted planner-allowlist pass (close-cycle and ideaspace stack), not an all-175-manifest sweep.
- "Below threshold" is about recommendation impact ranking, not plugin failure.
- Plugins can run successfully and still surface low/no modeled improvement for a specific dataset.

## Modeled Improvement Threshold Snapshot

- Synthetic targeted run (`d925...`):
  - recommendation items: `26`
  - `>20%`: `1` item
  - `0-20%`: `19` items
  - `0%`: `3` items
  - `None/unmodeled`: `3` items
  - top `>20%`: `22.839%` (batch/cache `ledger_post`)

## ERP Baseline Matching Check

For synthetic `de7c1da5...`, baseline comparison resolves to the real Quorum dataset `3246cc7c...` with:
- `same_erp=true`
- `same_schema_signature=true`
- row delta: `+64,384` (`1.1478x` target/baseline)

