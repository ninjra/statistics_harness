# Optimal 4-Pillars Execution Path

This is the default operating path for getting accurate, actionable, and fast outcomes without wasting full-gauntlet cycles.

## Goals

1. Maximize actionable insight quality.
2. Minimize runtime and memory waste.
3. Preserve determinism and reproducibility.
4. Keep reporting coherent and comparable across runs.

## Workflow

1. **Baseline checkpoint run**
   - Run full gauntlet once (`--plugin-set full --force`) to establish current truth.
   - Treat this as the comparison anchor.

2. **Deterministic triage lane (plugin-level)**
   - Generate priority queue from live run evidence:
     - hard failures (`error` / `aborted`)
     - longest runtimes
     - highest RSS
   - Command:
     - `python scripts/optimal_4pillars_path.py --run-id <run_id>`
   - Output:
     - `docs/release_evidence/optimal_4pillars_triage_<run_id>.json`
     - `docs/release_evidence/optimal_4pillars_triage_<run_id>.md`

3. **Remediation lane**
   - Fix plugins in queue order:
     - failure correctness first
     - then runtime/memory hotspots
   - Every fix must include targeted regression tests.

4. **Verification lane**
   - Run targeted tests for changed plugins.
   - Run full `pytest -q` gate before final gauntlet.

5. **Final validation gauntlet**
   - Run full gauntlet once on patched code.
   - Compare before/after:
     - `python scripts/compare_run_outputs.py --run-before <before> --run-after <after> --output-json <path>`
     - `python scripts/compare_plugin_actionability_runs.py --before-run-id <before> --after-run-id <after> --out <path>`

6. **Decision output**
  - Publish:
    - pass/fail by plugin
    - new/changed actionable recommendations
    - runtime + memory delta
    - confidence/coverage caveats

## Automation Helper

- `scripts/finalize_optimal_4pillars.py` waits for a target run to finish and automatically writes:
  - run output diff (`compare_run_outputs.py`)
  - plugin actionability diff (`compare_plugin_actionability_runs.py`)
  - runtime/memory hotspots (`run_hotspots_report.py`)

## Hard Rules

- No partial final validation: full gauntlet is required for release decisions.
- A plugin must return deterministic output (`ok` / `error` / explanatory fallback), never silent no-op.
- If a plugin cannot make a decision, it must emit explicit reason and fallback result.
- Document every optimization that changes runtime behavior.
