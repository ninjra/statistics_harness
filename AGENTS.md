# Statistic Harness — Codex / Agent Instructions

## Non-negotiables (do not ship if violated)
- Do not ship unless: `python -m pytest -q` passes.
- Phase 1 is local-only: NO network calls at runtime (UI is local web only).
- All pipeline steps are plugins/modules:
  - ingest (file parsing)
  - profile/validate
  - analysis (statistical techniques)
  - report (human + machine output)
  - llm (OFFLINE prompt builder only)
- Everything stored is self-contained:
  - SQLite (state + results) + filesystem artifacts under `./appdata/`
  - `./appdata/` must be gitignored.
- Outputs must be BOTH:
  - human readable: `report.md`
  - machine readable: `report.json` that validates against `docs/report.schema.json`
- Determinism:
  - Every run has a `run_seed`
  - All randomness uses a per-run RNG seeded from `run_seed`
  - Tests must set `run_seed` deterministically.
- Security:
  - Disallow path traversal in downloads and artifact serving
  - No `pickle` for untrusted data
  - Validate file types and sizes
  - Avoid `eval` and shelling out during analysis.

## Testing requirements
- Unit tests per plugin (synthetic fixtures).
- Integration test runs the full pipeline and asserts report outputs exist and validate.
- Add an evaluator harness:
  - takes a `ground_truth.yaml` describing known hidden attributes
  - asserts they appear in `report.json` within configured tolerances
- If any test fails: do not ship.

## Style
- Python 3.11+.
- `src/` layout.
- Type hints and clear docstrings.
- Fail closed:
  - plugin errors do not crash the pipeline
  - pipeline still generates a report that includes error summaries.
