# Plan: Rich Useful Plugin + Full Plugin Functional Verification

**Generated**: 2026-02-11
**Estimated Complexity**: High

## Overview
Upgrade the Kona/ideaspace layer from a basic scorer into a rich, deterministic decision plugin (with explicit current-vs-ideal traversal and actionable route evidence), then run a matrix-first, repo-wide plugin verification program to ensure every plugin is implemented, wired correctly, and functioning as expected on the normalized dataset path.

The plan is optimized for the existing 4-pillar operating model by prioritizing:
- Actionability quality (specific, non-obvious recommendations)
- Functional correctness and determinism
- Performance/resource control on large datasets
- Evidence traceability and operator trust

## Prerequisites
- Python environment active in `.venv`.
- Local dataset version available in `appdata/state.sqlite`.
- Existing matrix generators are present and runnable:
  - `scripts/plugins_functionality_matrix.py`
  - `scripts/docs_coverage_matrix.py`
  - `scripts/binding_implementation_matrix.py`
  - `scripts/redteam_ids_matrix.py`
  - `scripts/verify_docs_and_plugin_matrices.py`
- Existing full-run entrypoint:
  - `scripts/run_loaded_dataset_full.py`
- Existing report surface:
  - `src/statistic_harness/core/report.py`
  - `scripts/show_actionable_results.py`

## Sprint 1: Matrix-First Baseline and Scope Lock
**Goal**: Establish a single source of truth for plugin inventory, dependency order, and verification scope before editing runtime logic.
**Demo/Validation**:
- Matrix docs regenerate cleanly.
- Explicit list of all executable plugins and expected run behavior is available.
- Kona enrichment scope is mapped to concrete files and artifacts.

### Task 1.1: Regenerate and lock implementation matrices first
- **Location**: `scripts/update_docs_and_plugin_matrices.py`, `docs/plugins_functionality_matrix.md`, `docs/implementation_matrix.md`, `docs/redteam_ids_matrix.md`
- **Description**: Regenerate all matrix artifacts and freeze them as the baseline for this effort.
- **Complexity**: 3
- **Dependencies**: None
- **Acceptance Criteria**:
  - Matrix generation scripts run without unresolved refs.
  - Plugin count/order and doc-coverage state are current.
- **Validation**:
  - Run `scripts/verify_docs_and_plugin_matrices.py` and assert pass.

### Task 1.2: Define plugin verification catalog and expected outcomes
- **Location**: `docs/plugins_functionality_matrix.md`, `docs/implementation_matrix.md`, new section in `docs/kona_energy_ideaspace_architecture.md`
- **Description**: Add a verification catalog for each plugin class (`transform/profile/planner/analysis/report/llm`) including expected status (`ok/skipped` conditions), required artifacts, and minimum finding quality constraints.
- **Complexity**: 5
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Every executable plugin has an expected behavior contract.
  - Skip/degrade reasons are explicitly bounded and auditable.
- **Validation**:
  - Manual spot-check + schema consistency check against plugin manifests.

### Task 1.3: Formalize Kona "rich usefulness" contract
- **Location**: `docs/kona_energy_ideaspace_architecture.md`
- **Description**: Add strict output requirements for Kona plugin usefulness:
  - Non-degenerate ideal/current traversal,
  - Ranked route steps,
  - Explicit delta model,
  - Known-issue landmark detection and pass/fail semantics.
- **Complexity**: 4
- **Dependencies**: Task 1.2
- **Acceptance Criteria**:
  - Contract is unambiguous and testable.
- **Validation**:
  - Contract checklist added and linked from implementation matrix.

## Sprint 2: Kona Plugin Enrichment (Current vs Ideal Traversal)
**Goal**: Make Kona produce rich, specific, trustworthy route outputs instead of weak/degenerate gap summaries.
**Demo/Validation**:
- Kona artifacts include traversal graph/route and non-trivial deltas.
- Recommendations cite route steps and modeled impacts.
- Known issues are detected with explicit evidence and benefit modeling.

### Task 2.1: Harden entity/process selection for ideaspace state construction
- **Location**: `src/statistic_harness/core/ideaspace_feature_extractor.py`, `src/statistic_harness/core/lever_library.py`, `src/statistic_harness/core/stat_plugins/ideaspace.py`
- **Description**: Ensure semantic process/activity columns are selected over queue/surrogate IDs for all Kona computations.
- **Complexity**: 6
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Process identity in Kona state aligns with normalized business process columns.
  - Queue ID dominance regressions are blocked by tests.
- **Validation**:
  - Unit tests with adversarial column sets.

### Task 2.2: Implement explicit ideal-traversal artifacts
- **Location**: `src/statistic_harness/core/stat_plugins/ideaspace.py`, `src/statistic_harness/core/report.py`, `plugins/analysis_ideaspace_normative_gap/output.schema.json` (if needed)
- **Description**: Add artifacts and findings for:
  - Current state vector,
  - Ideal state vector,
  - Route candidates,
  - Selected traversal path with step deltas.
- **Complexity**: 8
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - `analysis_ideaspace_normative_gap` no longer emits all-zero/degenerate-only summaries when actionable variance exists.
  - Output includes route-level metrics (`delta_energy`, `delta_hours`, confidence).
- **Validation**:
  - Golden fixture where known non-zero gap must produce non-zero traversal steps.

### Task 2.3: Tie action planner to traversal and verifier
- **Location**: `src/statistic_harness/core/stat_plugins/ideaspace.py`, `src/statistic_harness/core/report.py`, `scripts/show_actionable_results.py`
- **Description**: Ensure ideaspace actions are scored and surfaced through route context (why this step now, expected sequence impact, evidence).
- **Complexity**: 7
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - Action planner outputs include route references and target specificity.
  - Report surfaces action chain in professional grouped format.
- **Validation**:
  - Integration test for grouped top-N with route linkage present.

### Task 2.4: Strengthen known-issue landmark evaluator logic (generic)
- **Location**: `scripts/show_actionable_results.py`, `src/statistic_harness/core/report.py`
- **Description**: Make known-issue checks robust but generic:
  - Merge equivalent checks,
  - Tolerate evidence from equivalent plugin kinds,
  - Preserve strict process-level targeting.
- **Complexity**: 6
- **Dependencies**: Task 2.3
- **Acceptance Criteria**:
  - Known issue table reports stable pass/fail without duplicate contradictions.
  - Modeled benefit is shown when derivable.
- **Validation**:
  - Unit tests for qemail/qpec/payout landmark scenarios.

### Task 2.5: Add anti-degeneracy and freshness gates for ideaspace outputs
- **Location**: `src/statistic_harness/core/stat_plugins/ideaspace.py`, `src/statistic_harness/core/pipeline.py`, `scripts/run_loaded_dataset_full.py`
- **Description**: Add explicit guards so ideaspace results cannot silently pass when they are degenerate or stale:
  - Detect all-zero/near-zero gap outputs and emit structured `degraded` with reason when variance exists upstream.
  - Add no-cache verification mode for Kona stage in audit/full verification flows.
  - Emit freshness metadata (`source_run_id`, `reused_from_run_id`, `cache_key_fingerprint`) in artifacts.
- **Complexity**: 7
- **Dependencies**: Task 2.3
- **Acceptance Criteria**:
  - Degenerate traversal is visible and actionable, never silently treated as rich output.
  - Operators can force fresh Kona recompute in one deterministic run mode.
- **Validation**:
  - Regression tests for stale-cache reuse and anti-degeneracy gating behavior.

## Sprint 3: Full Plugin Functional Completeness Harness
**Goal**: Verify all executable plugins are implemented and functioning as expected, not just present in manifests.
**Demo/Validation**:
- One harness run yields per-plugin status quality report.
- Failures categorized into implementation, config, dependency, data-coverage, or expected-skip.

### Task 3.1: Build per-plugin functional audit runner
- **Location**: New `scripts/plugin_functional_audit.py` (or equivalent), `docs/plugins_functionality_matrix.md`
- **Description**: Add an audit runner that executes all executable plugins (or replays from full run) and scores each plugin against expected contract.
- **Complexity**: 7
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Audit output includes `implemented`, `executed`, `status`, `artifact_presence`, `finding_quality`.
- **Validation**:
  - Harness run over full plugin set with machine-readable report.

### Task 3.2: Add contract tests for plugin classes
- **Location**: `tests/` (new suites for class-level contracts)
- **Description**: Add reusable tests for all plugin types:
  - Transform: normalized outputs exist and schema-valid,
  - Analysis: non-crash + typed findings,
  - Report: required report outputs present,
  - LLM: deterministic/offline constraints enforced.
- **Complexity**: 8
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Contract tests fail on missing artifacts, malformed findings, bad status semantics.
- **Validation**:
  - `python -m pytest -q` passes with new contract suites.

### Task 3.3: Classify and close gaps from audit output
- **Location**: `docs/implementation_matrix.md`, `docs/binding_implementation_matrix.md`
- **Description**: Resolve or explicitly document all plugin gaps found by the audit (no silent unknowns).
- **Complexity**: 6
- **Dependencies**: Task 3.2
- **Acceptance Criteria**:
  - No unresolved plugin marked "expected functioning" without evidence.
- **Validation**:
  - Updated matrix shows closed/open status with rationale.

### Task 3.4: Add report-stage contract gate and failure taxonomy
- **Location**: `tests/`, `src/statistic_harness/core/report.py`, `plugins/report_decision_bundle_v2/`
- **Description**: Add explicit report-stage contract validation so late report failures are caught as first-class gate failures:
  - enforce claim-to-evidence mapping completeness,
  - enforce required report artifacts per run type,
  - classify failure cause (`mapping`, `schema`, `traceability`, `runtime`) in audit output.
- **Complexity**: 6
- **Dependencies**: Task 3.2
- **Acceptance Criteria**:
  - Report plugin failures cannot hide behind otherwise successful analysis stages.
  - Failure reason is deterministic and machine-readable.
- **Validation**:
  - New tests for missing claim mapping and artifact contract violations.

## Sprint 4: Large-Dataset Readiness and Determinism
**Goal**: Keep enriched Kona + full plugin set stable on 500k to 2M rows without losing fidelity.
**Demo/Validation**:
- Full run completes within governed resource envelope.
- Deterministic recommendations stable across repeated seeds/runs.

### Task 4.1: Add deterministic replay checks for recommendation stability
- **Location**: `tests/` + `scripts/run_loaded_dataset_full.py`
- **Description**: Add repeated-run checks comparing ranked recommendations and known-issue statuses.
- **Complexity**: 6
- **Dependencies**: Sprint 2, Sprint 3
- **Acceptance Criteria**:
  - Rank and key recommendation identity are stable within tolerance.
- **Validation**:
  - Determinism test report saved as artifact.

### Task 4.2: Tighten resource governor hooks for audit/full runs
- **Location**: `src/statistic_harness/core/pipeline.py`, `scripts/start_full_loaded_dataset_bg.sh`
- **Description**: Ensure memory/concurrency governors work with enriched ideaspace and full plugin fan-out.
- **Complexity**: 5
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - No runaway RAM spikes above configured soft ceiling.
- **Validation**:
  - Hotspot + run-status trend checks.

## Sprint 5: Final Verification and Signoff
**Goal**: Deliver a verified rich Kona experience plus full plugin confidence report.
**Demo/Validation**:
- Full gauntlet run produces report outputs and actionable recommendations with known-issue table.
- Matrices and audit outputs are in sync.

### Task 5.1: Execute full gauntlet and capture outcome pack
- **Location**: `appdata/runs/<run_id>/...`, `docs/implementation_matrix.md`
- **Description**: Run full pipeline on loaded dataset and capture:
  - Grouped top recommendations,
  - Known-issue detection,
  - Kona traversal artifacts,
  - Plugin functional audit summary.
- **Complexity**: 4
- **Dependencies**: Sprints 2-4
- **Acceptance Criteria**:
  - Required outputs exist: `report.md`, `report.json`, `answers_summary.json`, `answers_recommendations.md`, `answers_recommendations_plain.md`.
  - Kona outputs contain route-level evidence, not only high-level suggestions.
- **Validation**:
  - `scripts/show_actionable_results.py` + audit report checks.

### Task 5.2: Final matrix reconciliation and ship gate
- **Location**: `docs/plugins_functionality_matrix.md`, `docs/implementation_matrix.md`, `docs/redteam_ids_matrix.md`
- **Description**: Reconcile docs/matrices with final state and enforce zero unresolved required items.
- **Complexity**: 3
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - Matrix verification scripts pass.
  - No ambiguous "implemented but unverified" entries for executable plugins.
- **Validation**:
  - `scripts/verify_docs_and_plugin_matrices.py` passes.

## Testing Strategy
- Unit tests:
  - Ideaspace process-column scoring,
  - Traversal path construction,
  - Landmark detection merge logic,
  - Recommendation grouping and obviousness ranking.
- Integration tests:
  - End-to-end Kona plugins (`normative_gap`, `energy_ebm`, `action_planner`, `ebm_action_verifier`) on deterministic fixtures.
- Full-system tests:
  - `python -m pytest -q` must pass.
  - Full loaded-dataset run must complete and emit all required outputs.
- Determinism checks:
  - Repeated run comparison for recommendation identity/order and known-issue statuses.

## Potential Risks & Gotchas
- Degenerate ideal/frontier selection can yield zero-gap artifacts even when operational pain exists.
  - Mitigation: enforce fallback ideal selection strategy and non-triviality checks.
- Cached reused results can hide regressions or fixes.
  - Mitigation: add explicit no-cache verification phase in audit workflow.
- Process identity drift (queue IDs vs process IDs) can invalidate Kona landmarks.
  - Mitigation: semantic column scoring + regression tests.
- Overly broad known-issue matching can create false confirmations.
  - Mitigation: require process-level or lever-level structured evidence gates.
- Report plugins may fail late and hide otherwise successful analysis.
  - Mitigation: dedicated report-stage contract tests and failure categorization.

## Rollback Plan
- Revert Kona enrichment changes by plugin boundary only:
  - `src/statistic_harness/core/stat_plugins/ideaspace.py`
  - `src/statistic_harness/core/lever_library.py`
  - `src/statistic_harness/core/report.py`
  - `scripts/show_actionable_results.py`
- Keep matrix/audit tooling changes if they remain useful independently.
- Re-run full gauntlet on prior stable commit and compare run artifacts to ensure operational continuity.
