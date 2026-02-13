# Codex CLI Implementation Blueprint — Statistics Harness (ninjra/statistics_harness)

Source repo: https://github.com/ninjra/statistics_harness  
TS: 2026-02-12 21:48:09 MST

## 0) Hard constraints (must follow)
Repo evidence:
- README: https://github.com/ninjra/statistics_harness/blob/main/README.md
- CLI (plugin discovery/validate/eval paths): https://github.com/ninjra/statistics_harness/blob/main/src/statistic_harness/cli.py

Constraints:
- Keep local-only defaults intact.
- Integrate new plugins via existing `plugins/` discovery and `PluginManager`.
- Preserve PII tagging/anonymization behavior mentioned in README.

## 1) What to implement (from recommended ideas)
### 1.1 Changepoint detection plugin (low risk, high leverage)
Plugin: `changepoint_detection_v1`
- Inputs: time series derived from normalized dataset (durations, queue depth, throughput)
- Outputs:
  - changepoint timestamps
  - segments summary (mean/median/p95 per segment)
  - deltas used by downstream plugins

### 1.2 Process mining plugin family (sequence-aware bottlenecks)
Plugins:
- `process_mining_discovery_v1`
- `process_mining_conformance_v1`

Approach:
- Start dependency-light:
  - build frequency graphs using `networkx`
  - implement a minimal discovery algorithm
- Optional: add `pm4py` as an extra group if desired later.

### 1.3 Causal inference plugin (assumption-explicit recommendations)
Plugin: `causal_recommendations_v1`
- Requires a user-provided causal graph config to start.
- Must output:
  - effect estimate (or “no identification”)
  - assumptions list
  - at least one refutation result

### 1.4 Evaluation gate expansion (extend existing `eval`)
Repo evidence: `eval` and `make-ground-truth-template` exist in CLI:
https://github.com/ninjra/statistics_harness/blob/main/src/statistic_harness/cli.py

Extend evaluation to include:
- expected changepoints (with tolerance)
- expected intervention candidates
- expected evidence windows (row ranges or time windows)

## 2) Codex execution steps (must scan full repo)
Codex MUST:
1) `git ls-files` → `docs/_codex_repo_manifest.txt`
2) Locate and read:
   - `core/plugin_manager.py`
   - `core/pipeline.py`
   - report builder and evaluation modules
3) Catalog existing plugins:
   - write `docs/_codex_plugin_catalog.md`
4) Identify where to store new plugin artifacts and how they flow into `report.json` and `report.md`.

Stop if scan cannot be completed.

## 3) Implementation plan (ordered)
### Phase A — schemas + artifacts contracts
- Add schemas:
  - `docs/schemas/changepoints.schema.json`
  - `docs/schemas/process_mining.schema.json`
  - `docs/schemas/causal.schema.json`

### Phase B — implement `changepoint_detection_v1`
- Add synthetic tests with seeded changepoints.
- Ensure chunked processing for large datasets.

### Phase C — implement process mining plugins
- Build event log abstraction:
  - `case_id`, `activity`, `start_ts`, `end_ts`, `resource`
- Produce frequency graph + bottleneck report.

### Phase D — implement causal plugin
- Start with assumption-required mode (explicit config).
- Add refutation utilities.

### Phase E — expand evaluation
- Update ground truth templates and `eval` checks.

## 4) Tests (must add)
- `stat-harness plugins validate` passes with new plugins.
- Schema validation for all plugin outputs.
- `stat-harness eval` checks new outputs (changepoints, interventions, evidence slices).
- Performance smoke: large-file run does not exceed memory by unbounded copies.

## 5) Acceptance criteria (objective)
- New plugins appear in `stat-harness list-plugins`.
- `stat-harness run` executes them and writes artifacts into run dir.
- `stat-harness eval` validates runs using updated templates.

## 6) Evidence labels
- QUOTE: local-only posture + env flags: https://github.com/ninjra/statistics_harness/blob/main/README.md
- QUOTE: CLI plugin/eval integration points: https://github.com/ninjra/statistics_harness/blob/main/src/statistic_harness/cli.py
- NO EVIDENCE: assistant did not read every file; Codex is instructed to perform a full scan.

## 7) Determinism notes
Use stable ordering for output lists and stable run seeds (`--run-seed`).
