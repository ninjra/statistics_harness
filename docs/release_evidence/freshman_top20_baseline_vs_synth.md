# Freshman Top-20: Baseline vs Synthetic

This report is a plain-language overlay for recommendation deltas between the
baseline run and synthetic comparison runs.

## Inputs
- Baseline comparison artifacts:
  - `docs/release_evidence/compare_outputs_baseline_vs_synth6mo.json`
  - `docs/release_evidence/compare_outputs_baseline_vs_synth8mo.json`
- Actionability comparison artifacts:
  - `docs/release_evidence/compare_actionability_baseline_vs_synth6mo.json`
  - `docs/release_evidence/compare_actionability_baseline_vs_synth8mo.json`

## Reading Guide
- Rank by modeled close-window impact first.
- Use efficiency index (`eff_idx`) to compare unlike recommendation types.
- Treat synthetic-only spikes as investigation leads, not final baseline actions.

## Current Status
- Baseline-first recommendation set is authoritative for client-facing output.
- Synthetic comparisons are diagnostic and troubleshooting support only.

