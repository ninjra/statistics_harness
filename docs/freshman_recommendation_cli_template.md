# Freshman Recommendation CLI Template

Use this template for the first-screen recommendation output in Codex CLI.

## Hard Rules

- Always include independently detected landmark issues in Top-N if present:
  - qemail contention/scheduling
  - qpec +1 capacity
  - payout batch chain (rpt_por002 family)
- Always show one compact freshman table before verbose grouped details.
- Keep wording simple and action-first.

## Freshman Table Columns

| Column | Meaning |
|---|---|
| `#` | Rank in displayed Top-N |
| `process` | Primary process target |
| `action` | Plain-English recommendation label |
| `class` | `direct` or `bundle` |
| `Δh acct/static/dyn` | Modeled hour savings triplet |
| `eff_idx acct/static/dyn` | Dimensionless efficiency index triplet |
| `close Δh` | Close-window modeled delta hours |
| `close eff%` | Close-window modeled percent gain |
| `human runs` | Estimated manual run reductions |
| `human touches` | Estimated touch/click reductions |

## Class Rules

- `direct`: specific change request the client can ticket directly.
- `bundle`: multi-change optimizer package or simulation bundle.

## Required Source

- Values come from `report.json` recommendation fields only.
- If a value is unavailable, print `N/A` (never blank).

