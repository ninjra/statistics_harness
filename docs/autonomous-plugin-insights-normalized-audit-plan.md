# Plan: Autonomous Plugin Insights + Normalized/Streaming Audit

**Generated**: 2026-02-17
**Estimated Complexity**: High

## Overview
This plan addresses three goals end-to-end:
1. Surface batch-input conversion clusters as explicit, actionable sequence/process recommendations (not broad generic text).
2. Audit and enforce that every plugin uses the normalized layer correctly and handles large data via streaming or approved bounded methods.
3. Build a plugin-class matrix with modeled actionable examples per class/plugin so we can validate that plugins are being used correctly and are producing meaningful analysis.

The plan is intentionally fail-closed: when known issues are disabled, the system must still produce autonomous actionable insights, or fail the run.

## Skills By Section (With Rationale)
- **Planning + decomposition**: `plan-harder`
  - Rationale: phased sprints, atomic tasks, testable increments.
- **Root-cause and correctness hardening**: `ccpm-debugging`
  - Rationale: diagnose “many plugins, same insights” by tracing data path, feature extraction, and recommendation routing.
- **Matrix design + validation**: `config-matrix-validator`
  - Rationale: formal plugin grouping, ownership, and coverage criteria.
- **Test enforcement**: `testing`, `python-testing-patterns`, `deterministic-tests-marshal`
  - Rationale: deterministic and repeatable gate checks for autonomous insight generation.
- **Observability and metrics**: `discover-observability`, `observability-engineer`, `python-observability`
  - Rationale: track insight quality, data access contract compliance, and regression signals over time.
- **Performance/resource safety**: `resource-budget-enforcer`
  - Rationale: validate large normalized dataset behavior and enforce bounded resource use.

## Prerequisites
- `.venv` available and pytest passing baseline.
- Existing matrices available:
  - `docs/plugin_data_access_matrix.json`
  - `docs/plugins_functionality_matrix.json`
  - `docs/sql_assist_adoption_matrix.json`
- Full gauntlet scripts available:
  - `scripts/run_final_validation_checklist.sh`
  - `scripts/run_loaded_dataset_full.py`
- Baseline dataset run IDs retained for before/after comparison.

## Global Definitions (Used By Gates)
- **Large dataset mode**: `rows > 1_000_000` (aligned with `DatasetAccessor.load` safety guard).
- **Approved bounded access contracts**:
  - `iter_batches`
  - `dataset_loader` with explicit `row_limit`
  - `sql_assist` with explicit bounded query (row/time bounded).
- **Unapproved in strict mode**:
  - unbounded `dataset_loader()` for large datasets unless plugin is explicitly allowlisted with written justification.
- **Autonomous novelty metrics**:
  - recommendation signature key:
    - `plugin_id`, `kind`, `action_type`, `where`, normalized recommendation text hash.
  - novelty score:
    - `new_count`, `dropped_count`, `unchanged_count`
    - optional similarity: Jaccard on signature sets.
- **Reference run selection**:
  - latest successful run for same dataset + `known_issues_mode=off` + same plugin set + same seed policy.

## Sprint 1: Define Insight Contract + Baseline Evidence
**Goal**: Lock down what counts as a “new autonomous actionable insight,” and create before-state evidence.
**Skills**: `plan-harder`, `ccpm-debugging`, `python-observability`
**Demo/Validation**:
- Run comparator against current baseline on/off known-issues runs.
- Produce a single JSON evidence file with current counts and deltas.

### Task 1.1: Formalize Autonomous Insight Definition
- **Location**: `docs/insight_quality_contract.md`
- **Description**: Define canonical criteria for “actionable insight,” “new insight,” and “not meaningful.”
- **Complexity**: 3
- **Dependencies**: None
- **Acceptance Criteria**:
  - Defines required fields: `plugin_id`, `kind`, `recommendation`, `where/targets`, `modeled_percent or reason`.
  - Defines “new vs prior run” signature and de-duplication key.
  - Defines fail conditions for autonomous mode.
- **Validation**:
  - Review checklist in markdown with explicit pass/fail examples.

### Task 1.2: Capture Current Baseline Evidence Snapshot
- **Location**: `docs/release_evidence/autonomous_insight_baseline_snapshot_20260217.json`
- **Description**: Store current before-state metrics (known on/off, counts by source plugin, top modeled impacts, unchanged sets).
- **Complexity**: 2
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Contains run IDs and summary fields for reproducibility.
  - Includes discovered insight signatures for diffing.
- **Validation**:
  - Recompute snapshot and assert deterministic equality.

## Sprint 2: Batch Input Cluster Surfacing (Sequence + Process IDs)
**Goal**: Convert generic batch recommendations into precise sequence-level and process-level instructions.
**Skills**: `ccpm-debugging`, `testing`, `python-testing-patterns`, `python-observability`
**Demo/Validation**:
- `answers_recommendations.md` contains explicit sequence groups with process lists and exact execution order suggestions.
- Artifact includes per-sequence cohort evidence and candidate input keys.

### Task 2.1: Emit Sequence-Level Batch Cluster Artifact
- **Location**: `src/statistic_harness/core/stat_plugins/topo_tda_addon.py`
- **Description**: Extend `analysis_actionable_ops_levers_v1` output to emit `batch_group_candidate` records with:
  - `sequence_id`
  - `target_process_ids` (ordered)
  - `close_month`
  - `key`
  - `estimated_calls_reduced`
  - `estimated_delta_seconds_upper`
- **Complexity**: 7
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - At least one artifact row has deterministic `sequence_id`.
  - Recommendations reference exact process IDs, not generic “split batches everywhere”.
- **Validation**:
  - New unit test with synthetic chain fixture verifies process ordering and IDs.

### Task 2.1b: Wire Sequence Artifact Into Report Schema + Pipeline
- **Location**: `docs/report.schema.json`, `src/statistic_harness/core/report.py`, `scripts/run_loaded_dataset_full.py`
- **Description**: Ensure new sequence-level payload is represented in `report.json` and rendered in `report.md`/answers outputs, with schema validation.
- **Complexity**: 5
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - `report.json` validates with sequence fields present.
  - Human and plain reports both show sequence/process-level details.
- **Validation**:
  - `python -m pytest -q tests/test_report_schema.py tests/test_report_outputs.py`

### Task 2.2: Add Sequence-Aware Recommendation Rendering
- **Location**: `src/statistic_harness/core/report.py`, `scripts/run_loaded_dataset_full.py`
- **Description**: Render sequence groups in both technical and plain outputs with “where to change” instructions:
  - exact process IDs
  - key field
  - close-month cohort
  - suggested batched call boundary
- **Complexity**: 5
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Recommendation text includes explicit ordered process list and key.
  - Plain-language report preserves exact IDs while simplifying explanation.
- **Validation**:
  - Snapshot tests for markdown sections.

### Task 2.3: Sequence-to-Validation Checklist Generator
- **Location**: `scripts/generate_batch_sequence_validation_checklist.py` (new)
- **Description**: Generate per-sequence runbook: “process IDs to change,” “expected evidence after change,” “how to verify delta.”
- **Complexity**: 4
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - Produces deterministic checklist markdown/json from run artifact.
- **Validation**:
  - Unit test compares generated checklist against fixture.
  - Include invocation path from `scripts/show_actionable_results.py` for operator visibility.

## Sprint 3: Full Plugin Normalized-Layer + Streaming Audit
**Goal**: Verify every plugin’s data-access contract and enforce compliant behavior for large normalized datasets.
**Skills**: `config-matrix-validator`, `ccpm-debugging`, `resource-budget-enforcer`, `testing`
**Demo/Validation**:
- New audit output listing each plugin as `compliant`, `needs_streaming`, `approved_bounded_loader`, or `violation`.
- Checklist fails if violations exist.

### Task 3.1: Upgrade Data Access Matrix Detection
- **Location**: `scripts/plugin_data_access_matrix.py`
- **Description**: Improve static detection for registry-driven plugins and classify:
  - `dataset_loader_unbounded`
  - `dataset_loader_bounded`
  - `iter_batches`
  - `sql_assist`
  - `orchestration_only`
- **Complexity**: 6
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Matrix differentiates bounded vs unbounded loader usage.
  - No `unclassified` plugins.
- **Validation**:
  - Matrix regeneration + verification tests.

### Task 3.1b: Plugin Catalog Sync Precheck
- **Location**: `scripts/generate_codex_plugin_catalog.py`, `docs/_codex_plugin_catalog.md`, `scripts/plugin_data_access_matrix.py`
- **Description**: Add pre-step requiring plugin catalog regeneration before any access/matrix audit to prevent unmapped plugins.
- **Complexity**: 3
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Matrix generation fails if plugin catalog and discovered plugin set diverge.
- **Validation**:
  - Regression test for mismatch failure.

### Task 3.2: Add Runtime Access/Memory Evidence per Plugin
- **Location**: `src/statistic_harness/core/plugin_runner.py`, `scripts/build_final_validation_summary.py`
- **Description**: Record runtime hints:
  - row count loaded
  - whether batched iteration used
  - peak RSS
  - SQL rows scanned estimate (if available)
- **Complexity**: 8
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Summary includes top offenders and contract mismatch list.
  - Runtime evidence persisted in `plugin_executions`-derived summaries consumed by final checklist.
- **Validation**:
  - New tests for summary fields.

### Task 3.3: Enforce Streaming/Bounded Policy Gates
- **Location**: `scripts/run_final_validation_checklist.sh`, `scripts/build_final_validation_summary.py`
- **Description**: Fail run when large-dataset policies are violated (plugin contract vs runtime behavior mismatch).
- **Complexity**: 6
- **Dependencies**: Task 3.2
- **Acceptance Criteria**:
  - Hard-fail on any non-approved unbounded loader in large dataset mode.
  - Policy exceptions require explicit allowlist with reason.
- **Validation**:
  - Integration test with synthetic violation plugin.
  - Stage gate rollout:
    - phase A: warning-only with metrics
    - phase B: hard-fail once metrics coverage is complete.

## Sprint 4: Plugin Class Matrix + Modeled Actionable Example Library
**Goal**: Group plugins coherently and prove each class can produce at least one modeled actionable example.
**Skills**: `config-matrix-validator`, `testing`, `python-observability`
**Demo/Validation**:
- Matrix artifacts generated and versioned.
- Each actionable class has at least one modeled example from real/synthetic runs.

### Task 4.1: Define Plugin Class Taxonomy
- **Location**: `docs/plugin_class_taxonomy.yaml` (new)
- **Description**: Classify plugins into classes such as:
  - direct_action_generators
  - supporting_signal_detectors
  - synthesis/verification
  - reporting/llm/post-processing
- **Complexity**: 4
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Every plugin mapped to one class.
  - Rationale field included per class.
- **Validation**:
  - Schema validation + no-unmapped assertion.

### Task 4.2: Build Plugin-Class Actionability Matrix
- **Location**: `scripts/build_plugin_class_actionability_matrix.py` (new), `docs/plugin_class_actionability_matrix.json`, `docs/plugin_class_actionability_matrix.md`
- **Description**: For each plugin and class, record:
  - expected actionable output type
  - whether modeled/measured output is supported
  - last run example reference (or deterministic N/A with reason)
- **Complexity**: 7
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Matrix includes all plugins.
  - Explicit examples for each class.
- **Validation**:
  - Deterministic regeneration test.
  - Matrix validation rejects unknown/unmapped plugins.

### Task 4.3: Modeled Example Card Generator
- **Location**: `scripts/generate_plugin_example_cards.py` (new), `docs/plugin_example_cards/`
- **Description**: Generate one modeled actionable example card per plugin class with source run and evidence pointers.
- **Complexity**: 5
- **Dependencies**: Task 4.2
- **Acceptance Criteria**:
  - Each class has at least one card.
  - Cards include traceability: plugin -> finding -> recommendation.
- **Validation**:
  - Unit tests for card schema and missing-source behavior.

## Sprint 5: Autonomous Novelty Enforcement
**Goal**: Guarantee the system can find autonomous insights and report novelty versus prior runs.
**Skills**: `ccpm-debugging`, `deterministic-tests-marshal`, `testing`, `observability-engineer`
**Demo/Validation**:
- Known-issues-off run must pass existing hard gates and novelty gates.
- Delta report clearly lists new/removed/unchanged autonomous insights.

### Task 5.1: Add Novelty Delta Comparator (Run-to-Run)
- **Location**: `scripts/compare_plugin_actionability_runs.py` (extend)
- **Description**: Compare autonomous insight signatures between runs and compute:
  - new insights
  - dropped insights
  - unchanged insights
  - score deltas
- **Complexity**: 5
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Comparator works for same dataset and cross-dataset modes.
  - Comparator stores signature-level diff and Jaccard similarity in JSON.
- **Validation**:
  - Deterministic fixture tests.

### Task 5.2: Add Autonomous Novelty Gate to Final Checklist
- **Location**: `scripts/run_final_validation_checklist.sh`, `scripts/build_final_validation_summary.py`
- **Description**: In `KNOWN_ISSUES_MODE=off`, enforce:
  - discovery recommendations > 0
  - actionable plugins > 0
  - novelty policy threshold (default: `new_count >= 1` OR `jaccard <= 0.95`) relative to reference run
- **Complexity**: 6
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - Checklist exits non-zero when autonomous novelty contract fails.
- **Validation**:
  - Failing and passing integration tests.

## Sprint 6: Full Gauntlet + Evidence Pack + Rollout
**Goal**: Produce release-grade evidence that plugins are coherent, normalized-access compliant, and generating autonomous insights.
**Skills**: `observability-engineer`, `python-observability`, `testing`
**Demo/Validation**:
- End-to-end run pack includes:
  - on/off known-issues runs
  - novelty diff
  - normalized/streaming compliance report
  - plugin class/actionability matrix

### Task 6.1: Execute Full Evidence Pipeline
- **Location**: `scripts/run_final_validation_checklist.sh`, `docs/release_evidence/`
- **Description**: Run full gauntlet on baseline + synthetic datasets and generate consolidated report artifacts.
- **Complexity**: 4
- **Dependencies**: Sprints 1-5
- **Acceptance Criteria**:
  - All hard gates green.
  - Autonomous mode report includes no known-issue injections.
- **Validation**:
  - Repeat run and compare evidence hashes.
  - Explicit run commands:
    - `bash scripts/run_final_validation_checklist.sh <dataset_id> <seed> <interval> on`
    - `bash scripts/run_final_validation_checklist.sh <dataset_id> <seed> <interval> off`
    - `./.venv/bin/python scripts/compare_plugin_actionability_runs.py --before-run-id <on_run> --after-run-id <off_run>`

### Task 6.2: Publish Operator Playbook
- **Location**: `docs/operator_playbook_autonomous_insights.md` (new)
- **Description**: Document exact commands, toggles, expected outputs, and failure handling.
- **Complexity**: 3
- **Dependencies**: Task 6.1
- **Acceptance Criteria**:
  - Includes one-line commands and troubleshooting matrix.
  - Includes strict/relaxed gate toggles and expected evidence outputs in each mode.
- **Validation**:
  - Dry-run playbook commands from clean session.

## Testing Strategy
- Mandatory per-sprint gate: `python -m pytest -q` must pass before sprint sign-off.
- Unit tests:
  - batch-sequence extraction and rendering.
  - known-issues mode behavior.
  - data-access matrix classification.
  - comparator signature determinism.
- Integration tests:
  - full gauntlet in `KNOWN_ISSUES_MODE=off`.
  - fail-closed checks for autonomous insight gate and streaming policy gate.
- Determinism:
  - fixed `run_seed`
  - stable sort/signature hashing
  - repeated run comparisons for drift detection.

## Potential Risks & Gotchas
- **Risk**: “No new insights” despite many plugins.
  - **Mitigation**: novelty comparator + autonomous novelty gate; explicit fail.
- **Risk**: registry-based plugins look compliant statically but load unbounded data at runtime.
  - **Mitigation**: runtime telemetry and contract mismatch gate.
- **Risk**: generic recommendation text remains non-actionable.
  - **Mitigation**: sequence/process-id rendering contract tests.
- **Risk**: known-issue landmarks leak into autonomous output.
  - **Mitigation**: mode-aware output suppression + tests.
- **Risk**: large dataset memory pressure masks correctness issues.
  - **Mitigation**: resource-budget checks and offender ranking in CI.
- **Risk**: strict gates fail due missing telemetry, not real behavior.
  - **Mitigation**: staged rollout (warn -> fail) with explicit telemetry-coverage readiness checklist.

## Rollback Plan
- Keep all new gates behind explicit env toggles for staged rollout:
  - `STAT_HARNESS_KNOWN_ISSUES_MODE`
  - `STAT_HARNESS_AUTONOMOUS_NOVELTY_MIN`
  - `STAT_HARNESS_STREAMING_POLICY_STRICT`
- If rollout blocks operations:
  - disable strict gates,
  - keep telemetry and matrix generation enabled,
  - re-enable gates once offending plugins are remediated.
  - toggles:
    - `STAT_HARNESS_STREAMING_POLICY_STRICT=0`
    - `STAT_HARNESS_AUTONOMOUS_NOVELTY_MIN=0`
    - `STAT_HARNESS_KNOWN_ISSUES_MODE=on` (temporary operational fallback)

## Plan Review Notes (Post-Gotchas)
- Added explicit novelty gate so “no new autonomous insights” becomes a hard failure condition.
- Added runtime-vs-static contract checks to prevent false confidence from static code scanning alone.
- Added sequence-level checklist generator to close the “too generic to act on” gap for batch cluster recommendations.
