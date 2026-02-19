# Statistic Harness â€” Codex / Agent Instructions

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
Always provide a single short one-line command with no line breaks and keep it under terminal width; create any needed script/cmd to implement the command and provide a short full-path command to run that script.

## Mandatory Workflow Gates
- `MUST`: Use short one-line commands only (no multi-line shell commands).
- `MUST`: Before any file/code change, output the full available skills list and the selected-skill rationale for the current task.

## Output Formatting (Hard Gate)
- For Y/N status lists, render `Y` and `N` with raw ANSI escapes (not escaped text, not HTML):
  - Green `Y`: `[32mY[0m`
  - Red `N`: `[31mN[0m`
- Status line format is strict: `<colored Y|N>:<identifier>`
- Do not output literal `\x1b[...]` sequences; output real ANSI control characters.

## Codex CLI Theme (Hard Gate)
- Applies to all assistant prose in Codex CLI for this repo (not only report files).
- Use soft cyberpunk ANSI palette:
  - Header: `38;5;177`
  - Label/key: `38;5;111`
  - Value: `38;5;150`
  - Accounting-month value: `38;5;117`
  - Close-static value: `38;5;81`
  - Close-dynamic value: `38;5;183`
  - Dim/supporting text: `90`
- Separators must be bright white (`97`) and visually emphasized:
  - `/` in triplets (`acct/static/dyn`)
  - `;` between metadata items
  - `=` in key/value pairs (`x=y`)
- `x=y` must render with key and value in different colors; `=` must be bright white.
- If `NO_COLOR` is set or output is non-TTY, fall back to plain text while preserving the same structure.

