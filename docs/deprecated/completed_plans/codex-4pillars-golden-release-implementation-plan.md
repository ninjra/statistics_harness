# Plan: Codex 4 Pillars Golden Release Implementation

**Generated**: 2026-02-15  
**Estimated Complexity**: High

## Overview
This plan optimally implements `docs/codex_4pillars_golden_release_plan.md` by prioritizing only remaining deltas, preserving already-implemented components, and enforcing hard release gates.

Optimization principle:
1. Close the smallest number of high-risk gaps first.
2. Reuse existing infrastructure already present in repo (`large_dataset_policy`, `sql_assist`, `four_pillars`, cache/materialization scripts, matrix generators).
3. Gate every sprint with deterministic, full-suite validation.

Current observed baseline (from repo state):
- Already present: `src/statistic_harness/core/large_dataset_policy.py`, `src/statistic_harness/core/sql_assist.py`, `src/statistic_harness/core/sql_schema_snapshot.py`, `src/statistic_harness/core/four_pillars.py`, `src/statistic_harness/core/dataset_cache.py`, SQL/gauntlet scripts, 4-pillars report wiring.
- Remaining high-value deltas:
  - Network guard still uses binary `STAT_HARNESS_ALLOW_NETWORK`, not tri-mode localhost policy.
  - Full-DF protection default in `src/statistic_harness/core/dataset_io.py` is still permissive (`3_000_000` default).
  - Streaming adoption is still limited by plugin behavior (`docs/plugin_data_access_matrix.md` shows many unbounded loader patterns).
  - Golden-release needs a strict, repeatable full-gauntlet evidence pack (source-compatible deterministic skip-with-reason, plus optional strict no-skip mode).

## Prerequisites
- WSL2 Python execution path is available.
- Deterministic test seed conventions remain unchanged.
- Existing docs/matrix generators are authoritative and must be regenerated after code changes.
- Hard gate remains: `python -m pytest -q` must pass before release.

## Sprint 1: Lock Golden Scope and Gap Baseline
**Goal**: Create a deterministic implementation baseline so all changes are measured against known deltas only.
**Demo/Validation**:
- A checked-in delta report exists with explicit `implemented|partial|missing` status for each P0/P1 item from `docs/codex_4pillars_golden_release_plan.md`.
- Baseline run identifiers and dataset identifiers are pinned for before/after comparison.

### Task 1.1: Build Golden Delta Mapping
- **Location**: `scripts/golden_release_delta_map.py` (new), `docs/golden_release_delta_map.json` (generated)
- **Description**: Parse plan sections from `docs/codex_4pillars_golden_release_plan.md` and map each requirement to concrete repo paths + status (`implemented|partial|missing`) using file/symbol heuristics.
- **Complexity**: 4/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - Every P0 and P1 requirement has exactly one status entry and owning file path.
  - Output ordering is deterministic.
- **Validation**:
  - Unit test for stable output and required-key coverage.

### Task 1.2: Pin Baseline Comparison Inputs
- **Location**: `docs/golden_release_baseline_inputs.md` (new)
- **Description**: Pin run IDs, dataset version IDs, and plugin set identifiers used for before/after release comparison.
- **Complexity**: 2/10
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Baseline references include one real Quorum dataset baseline and synthetic comparison datasets.
  - Includes command IDs/scripts used to reproduce comparison.
- **Validation**:
  - Manual verification that referenced run IDs exist in SQLite state.

### Task 1.3: Define Golden Execution Policy Modes
- **Location**: `docs/golden_release_runtime_policy.md` (new), `src/statistic_harness/core/pipeline.py`
- **Description**:
  - Define two explicit release modes:
    - `golden_default`: source-plan compatible (`skip-with-reason` allowed, deterministic and citable).
    - `golden_strict`: treats plugin `skipped` as release failure.
  - Add policy toggles and report annotations so mode is always visible in artifacts.
- **Complexity**: 4/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - Mode is explicit and deterministic for each run.
  - `golden_strict` exits non-zero if any plugin is skipped.
  - `golden_default` allows only deterministic, citable skips with explicit thresholds/reasons.
- **Validation**:
  - Integration test with a forced-skipping test plugin.

## Sprint 2: Security Delta — Localhost-Only Network Mode
**Goal**: Implement tri-mode network policy (`off|localhost|on`) with backward compatibility.
**Demo/Validation**:
- Network guard behavior matches acceptance criteria in `docs/codex_4pillars_golden_release_plan.md`.

### Task 2.1: Add Network Mode Resolver
- **Location**: `src/statistic_harness/core/plugin_runner.py`
- **Description**:
  - Introduce `STAT_HARNESS_NETWORK_MODE=off|localhost|on`.
  - Preserve `STAT_HARNESS_ALLOW_NETWORK` behavior as backward-compatible alias to `on`.
  - Default remains secure (`off`).
- **Complexity**: 5/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - Resolver deterministically maps env to one of three modes.
  - Invalid values fail closed to `off`.
- **Validation**:
  - Unit tests for mode resolution matrix.

### Task 2.2: Implement Loopback-Only Socket Guard
- **Location**: `src/statistic_harness/core/plugin_runner.py`
- **Description**:
  - In `localhost` mode, permit connections only to `127.0.0.1`, `::1`, and `localhost`.
  - Non-loopback destinations raise guard `RuntimeError`.
- **Complexity**: 6/10
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Loopback requests are not blocked by policy guard.
  - External destinations are blocked deterministically.
- **Validation**:
  - Add/update tests in `tests/test_network_guard.py`, `tests/test_offline.py`, `tests/test_sandbox.py`.

### Task 2.3: Document Network Runtime Contract
- **Location**: `README.md`, `docs/golden_release_runtime_policy.md`
- **Description**: Document exact network policy semantics and compatibility behavior.
- **Complexity**: 2/10
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - Docs reflect runtime defaults and accepted env values.
- **Validation**:
  - Matrix/docs regeneration tests pass.

## Sprint 3: Large-Dataset Hardening Delta
**Goal**: Enforce streaming-first behavior for large datasets with explicit operator override.
**Demo/Validation**:
- For row count >= 1,000,000, unbounded `dataset_loader()` fails closed unless explicitly overridden.

### Task 3.1: Tighten Full-DF Threshold Defaults
- **Location**: `src/statistic_harness/core/dataset_io.py`
- **Description**:
  - Lower default `STAT_HARNESS_MAX_FULL_DF_ROWS` fallback from `3_000_000` to `1_000_000`.
  - Ensure row-limit and column-limited paths still work for controlled loads.
- **Complexity**: 3/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - Default full-load refusal threshold is 1,000,000 rows.
  - Explicit override via `STAT_HARNESS_ALLOW_FULL_DF=1` remains available.
- **Validation**:
  - Unit tests for threshold behavior.

### Task 3.2: Improve Fail-Closed Guidance and Telemetry
- **Location**: `src/statistic_harness/core/dataset_io.py`, `src/statistic_harness/core/plugin_runner.py`
- **Description**:
  - Enrich refusal errors with plugin ID, dataset row count, and recommended streaming API usage.
  - Emit structured telemetry marker for policy refusal to support observability and triage.
- **Complexity**: 4/10
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Error text is actionable and deterministic.
  - Telemetry includes machine-readable refusal reason.
- **Validation**:
  - Unit tests and one integration test asserting structured refusal output.

### Task 3.3: Large Dataset Policy Coverage Expansion
- **Location**: `src/statistic_harness/core/large_dataset_policy.py`, `tests/test_large_dataset_policy.py` (new or update)
- **Description**: Expand policy caps for known high-complexity plugin families while keeping no-row-sampling guarantee.
- **Complexity**: 5/10
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Policy remains deterministic and row-sampling flag remains `False`.
  - Heavy plugin families get non-null complexity caps.
- **Validation**:
  - Deterministic policy test matrix.

## Sprint 4: Streaming Adoption Wave (Top-Offender First)
**Goal**: Reduce unbounded full-DF plugin behavior through deterministic streaming migration without reducing analysis coverage.
**Demo/Validation**:
- Top-offender waves migrated and validated under full gauntlet with deterministic/citable outcomes.

### Task 4.1: Rank Offenders from Actual Telemetry
- **Location**: `scripts/rank_streaming_offenders.py` (new), `docs/streaming_offenders_ranked.md` (generated)
- **Description**: Rank plugins by combined `duration_ms`, `max_rss`, and unbounded-loader detection from matrix + execution telemetry.
- **Complexity**: 4/10
- **Dependencies**: Sprint 1 baseline inputs
- **Acceptance Criteria**:
  - Deterministic top-N offender list generated.
  - List includes owning plugin paths and recommended migration strategy.
- **Validation**:
  - Unit test with seeded synthetic telemetry payload.

### Task 4.2: Migrate Foundational Upstream Plugins
- **Location**: `plugins/profile_basic/plugin.py`, `plugins/profile_eventlog/plugin.py`, `plugins/transform_normalize_mixed/plugin.py`
- **Description**:
  - Ensure foundational plugins use `ctx.dataset_iter_batches()` path for large datasets.
  - Preserve schema and deterministic outputs.
- **Complexity**: 8/10
- **Dependencies**: Sprint 2, Sprint 3
- **Acceptance Criteria**:
  - Foundational stage completes on large dataset policy without unbounded load.
  - Output compatibility retained.
- **Validation**:
  - Plugin unit tests + integration large-dataset simulation test.

### Task 4.3a: Migrate Wave A Offenders (Operational/Conformance/Queueing)
- **Location**: top-ranked Wave A plugin files under `plugins/` + `src/statistic_harness/core/stat_plugins/`
- **Description**: Convert Wave A offenders to streaming/multi-pass patterns and remove unbounded full-loader behavior.
- **Complexity**: 8/10
- **Dependencies**: Task 4.1, Task 4.2
- **Acceptance Criteria**:
  - Wave A plugins avoid unbounded full-load behavior in large-dataset mode.
  - Any unavoidable skip is deterministic, citable, and threshold-backed.
- **Validation**:
  - Wave-specific integration test + gauntlet subset for impacted families.

### Task 4.3b: Migrate Wave B Offenders (Distribution/Changepoint/Classical Stats)
- **Location**: top-ranked Wave B plugin files under `plugins/` + `src/statistic_harness/core/stat_plugins/`
- **Description**: Apply deterministic streaming rewrites or bounded multi-pass logic for Wave B.
- **Complexity**: 9/10
- **Dependencies**: Task 4.3a
- **Acceptance Criteria**:
  - Wave B plugins execute under large-dataset policy without unbounded loader.
  - Output schema and deterministic ordering remain stable.
- **Validation**:
  - Plugin-level regression tests + targeted performance regression checks.

### Task 4.3c: Migrate Wave C Offenders (Specialty/High-Memory Tail)
- **Location**: top-ranked Wave C plugin files under `plugins/` + `src/statistic_harness/core/stat_plugins/`
- **Description**: Finish migration of remaining high-memory offenders and unify fallback behavior.
- **Complexity**: 10/10
- **Dependencies**: Task 4.3b
- **Acceptance Criteria**:
  - Top-20 offender set has no unbounded full-load behavior in large-dataset mode.
  - Golden policy mode behavior is honored (`golden_default` vs `golden_strict`).
- **Validation**:
  - Full gauntlet and deterministic rerun checks.

## Sprint 5: 4 Pillars + Citation Quality Gate
**Goal**: Ensure scorecard and citation outputs are first-class release blockers, not post-hoc artifacts.
**Demo/Validation**:
- Every release run produces machine/human 4-pillars outputs with deterministic lineage and recommendation traceability.

### Task 5.1: Promote 4 Pillars Scorecard to Release Gate
- **Location**: `src/statistic_harness/core/report.py`, `src/statistic_harness/core/four_pillars.py`, `scripts/run_gauntlet*.sh|ps1`
- **Description**:
  - Add explicit gate checks: scorecard presence, schema completeness, and deterministic rerun consistency.
  - Enforce balance/veto fields in output contract.
- **Complexity**: 5/10
- **Dependencies**: Existing 4-pillars code path
- **Acceptance Criteria**:
  - Missing or malformed scorecard fails golden gauntlet.
  - Rerun with same seed/data preserves scorecard bytes except timestamp metadata.
- **Validation**:
  - Unit + integration tests for required keys and deterministic values.

### Task 5.2: Enforce Evidence Completeness in Findings
- **Location**: `src/statistic_harness/core/pipeline.py`, `src/statistic_harness/core/report.py`, plugin helper modules
- **Description**: Add validation to ensure findings contain consistent measurement/evidence payloads and reject malformed evidence at pipeline boundary.
- **Complexity**: 6/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - Evidence shape violations are surfaced deterministically.
  - Citable coverage metrics in report remain stable.
- **Validation**:
  - Unit tests for evidence normalization and failure cases.

### Task 5.3: Baseline-vs-Current Comparator Standardization
- **Location**: `scripts/compare_run_outputs.py` (extend), `scripts/compare_dataset_runs.py` (new)
- **Description**:
  - Standardize before/after comparisons for:
    - same dataset across plugin revisions,
    - two datasets under same plugin set.
  - Standardize required inputs:
    - `--run-before`, `--run-after`, `--dataset-before`, `--dataset-after`, `--plugin-set`, `--seed`.
  - Emit deterministic artifacts:
    - `comparison_summary.json`,
    - `comparison_recommendation_deltas.json`,
    - `comparison_report.md`.
  - Include sortable recommendation deltas and modeled-improvement ranking.
- **Complexity**: 6/10
- **Dependencies**: Sprint 1 baseline inputs
- **Acceptance Criteria**:
  - Single command generates deterministic diff artifact bundle.
  - Deltas are ordered by modeled `% improvement` descending when no >20% threshold hits.
- **Validation**:
  - Golden fixture comparison tests.

## Sprint 6: Release Hardening and Freeze
**Goal**: Produce a pin-worthy first release candidate with reproducible evidence pack.
**Demo/Validation**:
- Golden run package contains tests, gauntlet output, comparison report, and matrix/citation artifacts.

### Task 6.1: Full Gauntlet on Baseline + Synthetic Datasets
- **Location**: `scripts/run_gauntlet_loaded_dataset.sh`, `scripts/run_loaded_dataset_full.py`, `docs/release_evidence/` (new folder)
- **Description**:
  - Run full gauntlet against:
    - baseline real Quorum dataset (>=2,000,000 rows),
    - synthetic 6-month dataset,
    - synthetic 8-month dataset.
  - Capture run IDs and produce reproducibility manifest.
- **Complexity**: 7/10
- **Dependencies**: Sprints 2-5
- **Acceptance Criteria**:
  - `golden_default`: all plugins execute or produce deterministic/citable skip-with-reason outcomes.
  - `golden_strict`: all plugins execute with zero skips.
  - Reports and scorecards are produced for all three datasets.
  - For >=2M baseline run, evidence explicitly confirms:
    - no OOM,
    - no indefinite hang without progress visibility,
    - no silent truncation,
    - deterministic rerun stability with same seed and dataset DB state.
- **Validation**:
  - Automated assertion script over run DB + artifacts.

### Task 6.2: Determinism and Regression Pack
- **Location**: `scripts/build_golden_release_evidence_pack.py` (new), `docs/release_evidence/golden_release_summary.md`
- **Description**:
  - Re-run baseline with same seed and compare fingerprints, recommendations, and four-pillar scorecards.
  - Publish deterministic/variance checks and regression outcomes.
- **Complexity**: 5/10
- **Dependencies**: Task 6.1
- **Acceptance Criteria**:
  - Determinism criteria pass or explicit, bounded exception list is recorded.
  - Evidence pack includes command provenance and hashes.
- **Validation**:
  - Scripted checks in CI/local gate.

### Task 6.3: Release Candidate Checklist and Roll-forward/rollback
- **Location**: `docs/golden_release_checklist.md` (new)
- **Description**:
  - Final pre-release checklist including tests, gauntlet, matrix refresh, evidence pack, and rollback toggles.
- **Complexity**: 3/10
- **Dependencies**: Task 6.2
- **Acceptance Criteria**:
  - Checklist is executable and references exact one-line commands.
- **Validation**:
  - Dry-run checklist completion.

## Sprint 7: P1 Optimization Closure
**Goal**: Complete post-P0 P1 items explicitly required by the source plan: cache-first performance, safer concurrency/ETA, and deterministic SQL-pack replay.
**Demo/Validation**:
- P1 evidence bundle shows measurable runtime/RSS improvement without weakening determinism or citable lineage.

### Task 7.1: Cache-First Streaming Validation and Gap Closure
- **Location**: `src/statistic_harness/core/dataset_cache.py`, `src/statistic_harness/core/dataset_io.py`, `scripts/materialize_dataset_cache.py`, `scripts/inspect_dataset_cache.py`
- **Description**:
  - Audit and harden cache-hit behavior for large datasets.
  - Add deterministic cache manifest integrity checks and explicit invalidation reason reporting.
- **Complexity**: 6/10
- **Dependencies**: Sprint 3, Sprint 6
- **Acceptance Criteria**:
  - Cache hit/miss reason is explicit in logs/artifacts.
  - Repeated runs on same dataset version show cache reuse and reduced elapsed time.
- **Validation**:
  - Integration benchmarks with/without cache and deterministic-output equivalence checks.

### Task 7.2: Auto-Concurrency + ETA Stabilization
- **Location**: `src/statistic_harness/core/pipeline.py`, `scripts/run_run_status.py`
- **Description**:
  - Implement or harden telemetry-driven worker tuning and memory-hog serialization.
  - Replace simplistic progress estimates with median/EMA ETA models.
- **Complexity**: 6/10
- **Dependencies**: Sprint 6 telemetry outputs
- **Acceptance Criteria**:
  - `auto` worker mode remains within memory budget under large datasets.
  - ETA stabilizes after sufficient completed-plugin samples.
- **Validation**:
  - Unit tests for scheduler/ETA models + integration stress run.

### Task 7.3: SQL Assist “Generate Once, Replay Forever” Gate
- **Location**: `scripts/generate_sql_pack_with_vllm.py`, `plugins/transform_sql_intents_pack_v1/`, `plugins/transform_sqlpack_materialize_v1/`, `plugins/llm_text2sql_local_generate_v1/`, `docs/sql_pack.schema.json`
- **Description**:
  - Add explicit replay gate proving same schema hash + config yields identical SQL pack outputs.
  - Ensure SQL manifest lineage is attached to downstream findings where SQL-derived evidence is used.
- **Complexity**: 7/10
- **Dependencies**: Sprint 5 citation gate
- **Acceptance Criteria**:
  - Replay on identical inputs yields identical `sql_pack.json` and manifests.
  - Schema mismatch fails closed before downstream analysis.
- **Validation**:
  - Deterministic replay tests + integration failure-path test.

## Sprint 8: P2 Maturity and Continuous Improvement
**Goal**: Track and operationalize long-horizon improvements from the source plan without blocking first release.
**Demo/Validation**:
- P2 backlog is executable, measurable, and tied to 4-pillars score improvements over time.

### Task 8.1: Knowledge-Bank Readiness Audit
- **Location**: `src/statistic_harness/core/storage.py`, `src/statistic_harness/core/migrations.py`, `docs/erp-knowledge-bank-4-pillars-optimization-plan.md`
- **Description**:
  - Validate schema/table readiness for cumulative learning and scorecard persistence.
  - Produce missing-index or migration hardening actions as a tracked backlog.
- **Complexity**: 4/10
- **Dependencies**: Sprint 6
- **Acceptance Criteria**:
  - Readiness report enumerates complete/partial/missing items with file ownership.
- **Validation**:
  - Migration smoke tests on fresh and existing DB fixtures.

### Task 8.2: Recommendation Evaluator Expansion Plan
- **Location**: `tests/`, `scripts/`, `docs/`
- **Description**:
  - Define next-phase evaluator harness expansions for recommendation quality and modeled-vs-measured follow-through.
  - Keep artifacts deterministic and citable.
- **Complexity**: 5/10
- **Dependencies**: Sprint 5, Sprint 6
- **Acceptance Criteria**:
  - Backlog includes atomic tasks with explicit metrics and datasets.
- **Validation**:
  - Reviewable plan document + fixture-based test skeletons.

## Testing Strategy
- Unit:
  - Network mode resolution and socket guard behavior.
  - Dataset full-load threshold policy.
  - Large dataset policy determinism.
  - Evidence normalization contract.
  - Comparator determinism and ranking logic.
- Integration:
  - Large dataset policy enforcement with mocked high row count.
  - Full pipeline report generation with 4-pillars scorecard.
  - Golden-mode no-skip enforcement.
- End-to-end:
  - Full gauntlet on baseline + synthetic datasets.
  - Reproducibility rerun on baseline.
  - Cache-hit/cold-start equivalence and SQL-pack replay determinism.
- Release gate:
  - `python -m pytest -q`
  - `stat-harness list-plugins`
  - `scripts/run_gauntlet.sh` (or Windows-safe equivalent)
  - matrix regeneration and verification scripts.

## Potential Risks & Gotchas
- Conflict between “full plugin coverage” and expensive O(n^2) algorithms on >2M rows.
  - Mitigation: deterministic complexity caps + stream/multi-pass rewrites; no silent skipping in golden mode.
- Localhost model dependencies can blur secure default posture.
  - Mitigation: explicit tri-mode network policy, secure default `off`, and documented override path.
- Matrix/catalog drift can repeatedly fail tests late.
  - Mitigation: regenerate docs/matrices as part of every sprint completion checklist.
- Streaming rewrites can alter numerics.
  - Mitigation: plugin-level tolerance assertions and deterministic reference snapshots.
- Long-running gauntlets can hide stalled plugins.
  - Mitigation: progress heartbeat and runtime watchdogs tied to policy limits.

## Rollback Plan
- Keep major behavior behind explicit toggles:
  - `STAT_HARNESS_NETWORK_MODE`
  - `STAT_HARNESS_ALLOW_FULL_DF`
  - `STAT_HARNESS_MAX_FULL_DF_ROWS`
  - `STAT_HARNESS_LARGE_DATASET_POLICY`
  - golden-mode no-skip enforcement flag
- If regression appears:
  1. Disable only the newest toggle.
  2. Re-run baseline deterministic checks.
  3. Re-open offender migration task for targeted fix.
