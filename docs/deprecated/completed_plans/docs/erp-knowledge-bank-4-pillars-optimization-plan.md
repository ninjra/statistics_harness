# Plan: ERP Knowledge Bank + 4 Pillars Optimization Engine

**Generated**: February 12, 2026  
**Estimated Complexity**: High

## Overview
Build a persistent, cumulative knowledge system that learns from every dataset/run and improves recommendations over time, while scoring all work against the 4 pillars (`Performant`, `Accurate`, `Secure`, `Citable`).  
The system must:
- preserve and grow ERP column intelligence (including unmapped/new columns),
- compute weighted metrics for full accounting month and close period,
- compare projects/runs/repositories on a unified 4-pillar scale,
- score each pillar on a strict `0.0-4.0` scale using all available functionality/telemetry (not Kona-only),
- enforce balance so no pillar can be traded down to boost another pillar,
- keep Kona energy traversal aligned to lowest-energy paths and expose gaps clearly.

## Prerequisites
- Python 3.11+ runtime and existing plugin pipeline.
- SQLite migrations + storage access updates.
- Existing normalization, ideaspace, and recommendation flows remain plugin-driven.
- Existing run determinism and report schema gates continue to be enforced.

## Sprint 1: Foundations (Contracts + Storage)
**Goal**: Define canonical contracts and storage required for cumulative learning and 4-pillar scoring.  
**Demo/Validation**:
- New schemas and migration apply cleanly on a fresh and existing DB.
- Basic write/read roundtrip for knowledge-bank tables succeeds.

### Task 1.1: Define 4-Pillar Scoring Specification
- **Location**: `docs/4_pillars_scoring_spec.md` (new), `docs/kona_energy_ideaspace_architecture.md`
- **Description**: Define deterministic formulas, weights, normalization, and confidence bounds for:
  - per-pillar scores on `0.0-4.0` (`Performant`, `Accurate`, `Secure`, `Citable`),
  - per-run, per-dataset, and per-project/repo aggregate scores,
  - source-of-truth inputs from the full system surface (plugins, runtime telemetry, security checks, evidence quality, reproducibility),
  - ideal Kona low-energy traversal target (as one component, not sole scorer),
  - balance constraints and veto rules (no optimization that materially degrades any pillar).
- **Complexity**: 5
- **Dependencies**: None
- **Acceptance Criteria**:
  - Every score has explicit formula + units.
  - Each pillar is bounded and interpretable on `0.0-4.0`.
  - Balance penalty and minimum-floor logic are explicit.
  - Score output includes measured vs modeled split.
  - Cross-domain comparison rules are documented.
- **Validation**:
  - Spec review with fixture calculations committed as examples.

### Task 1.2: Add Knowledge-Bank Tables and Indexes
- **Location**: `src/statistic_harness/core/migrations.py`, `src/statistic_harness/core/storage.py`
- **Description**: Add persistent tables for:
  - `knowledge_column_catalog` (canonical + observed columns, mapping confidence, ERP scope),
  - `knowledge_metric_definitions` (metric id, formula version, scope),
  - `knowledge_metric_observations` (run/dataset/project/repo observations),
  - `knowledge_recommendation_outcomes` (recommended action + observed follow-up),
  - `four_pillars_scorecards` (scored entity snapshots),
  - `kona_energy_snapshots` (energy vectors + traversal states).
- **Complexity**: 7
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Migrations are idempotent.
  - Required indexes exist for `run_id`, `dataset_version_id`, `project_id`, `repo_id`.
  - Tables are append-only where required for auditability.
- **Validation**:
  - `python -m pytest -q` includes migration tests for schema creation + rollback safety.

### Task 1.3: Implement Column-Learning Contract
- **Location**: `src/statistic_harness/core/stat_plugins/columns.py`, `plugins/transform_normalize_mixed/plugin.py`, `src/statistic_harness/core/storage.py`
- **Description**: Persist every column classification event:
  - mapped canonical column,
  - new/unmapped column,
  - fallback-add behavior when mapping is unavailable,
  - confidence + provenance.
- **Complexity**: 6
- **Dependencies**: Task 1.2
- **Acceptance Criteria**:
  - Unknown columns always produce a stored learning record.
  - No column is dropped silently.
  - Normalization layer growth is traceable per run.
- **Validation**:
  - Unit tests for mapped and unmapped cases with deterministic outcomes.

## Sprint 2: Metric Engine (Whole Month + Close)
**Goal**: Compute weighted operational metrics that account for server count and context window (full month vs close).  
**Demo/Validation**:
- Metrics artifact includes both scopes and weighting details.
- Re-run on same dataset is deterministic.

### Task 2.1: Build Weighted Metric Families
- **Location**: `src/statistic_harness/core/stat_plugins/ideaspace.py`, `src/statistic_harness/core/ideaspace_feature_extractor.py`, `src/statistic_harness/core/lever_library.py`
- **Description**: Add reusable weighted metrics:
  - `queue_delay_per_server_hour`,
  - `service_time_per_server_hour`,
  - `throughput_per_server`,
  - `spillover_per_server`,
  - `error_per_server`.
- **Complexity**: 8
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Weighting explicitly includes active server count.
  - Missing server metadata uses deterministic fallback + warning.
- **Validation**:
  - Unit tests with synthetic server topologies.

### Task 2.2: Add Dual-Scope Aggregation (General vs Close)
- **Location**: `src/statistic_harness/core/close_cycle.py`, `src/statistic_harness/core/report.py`
- **Description**: Standardize every key modeled metric into:
  - `general_*` (full accounting month / observation window),
  - `close_*` (close period window),
  with aligned baseline/delta/percent fields.
- **Complexity**: 7
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Recommendations always carry scope class + modeled basis.
  - Missing modeled fields fail contract checks.
- **Validation**:
  - Report contract tests confirm no `insufficient_modeled_inputs` where basis exists.

### Task 2.3: Persist Metric Observations into Knowledge Bank
- **Location**: `plugins/report_decision_bundle_v2/plugin.py`, `src/statistic_harness/core/report.py`, `src/statistic_harness/core/storage.py`
- **Description**: Store per-run metric observations and modeled effects for future comparisons and recommendation tuning.
- **Complexity**: 6
- **Dependencies**: Tasks 2.1, 2.2
- **Acceptance Criteria**:
  - Metric observation rows are written for every completed/partial run.
  - Stored payload includes method version/fingerprint.
- **Validation**:
  - Integration test verifies DB growth across two runs and stable keys.

## Sprint 3: Cumulative Learning + Cross-Dataset Intelligence
**Goal**: Make insights improve over time rather than reset each dataset.  
**Demo/Validation**:
- Comparison report shows learned trends and novelty across multiple datasets.

### Task 3.1: Implement Dataset-to-Dataset Comparator
- **Location**: `plugins/analysis_knowledge_bank_comparison_v1/` (new), `src/statistic_harness/core/report.py`
- **Description**: Add plugin that compares current dataset against historical knowledge-bank observations for same ERP and for global cohorts.
- **Complexity**: 8
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - Output includes `improving`, `regressing`, `novel` signals.
  - Confidence uses sample-size and recency weighting.
- **Validation**:
  - Deterministic fixture tests for each signal class.

### Task 3.2: Recommendation Memory and Outcome Feedback
- **Location**: `plugins/analysis_recommendation_dedupe_v2/plugin.py`, `src/statistic_harness/core/report.py`, `src/statistic_harness/core/storage.py`
- **Description**: Track recommendation lineage and post-change outcomes so ranking improves with evidence, not repetition.
- **Complexity**: 7
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Recommendations include “seen before”, “previous outcome”, and “confidence adjustment”.
  - Duplicates are merged by stable signature + scope.
- **Validation**:
  - Unit tests for dedupe and confidence adjustment logic.

### Task 3.3: ERP Column Intelligence Expansion
- **Location**: `src/statistic_harness/core/stat_plugins/columns.py`, `docs/plugin_data_access_matrix.json`, `docs/plugins_functionality_matrix.json`
- **Description**: Keep a maintained column-knowledge view showing:
  - canonical mapping coverage,
  - unresolved fields by ERP/source,
  - mapping drift over time.
- **Complexity**: 5
- **Dependencies**: Tasks 1.3, 3.1
- **Acceptance Criteria**:
  - Matrix artifacts refresh from DB source-of-truth.
  - New unresolved fields trigger critical warning before plugin stage (with safe add fallback).
- **Validation**:
  - Integration test with synthetic novel columns.

## Sprint 4: 4-Pillars Ranking + Kona Optimization
**Goal**: Make the 4 pillars the top-level decision and comparison framework across runs, projects, and repositories.  
**Demo/Validation**:
- Every run emits a 4-pillar scorecard and Kona traversal status.
- Cross-project comparison table is generated.

### Task 4.1: Add 4-Pillars Scoring Engine
- **Location**: `src/statistic_harness/core/four_pillars.py` (new), `src/statistic_harness/core/report.py`
- **Description**: Compute pillar scores from measured and modeled metrics across all active system functionality (not only ideaspace/Kona):
  - `Performant`: runtime, memory, throughput-per-server, spillover behavior,
  - `Accurate`: known-issue detection hit rate, modeled-vs-measured calibration,
  - `Secure`: sandbox/PII/redaction and policy violations,
  - `Citable`: evidence completeness, traceability links, reproducibility fields.
- **Complexity**: 9
- **Dependencies**: Sprints 1–3
- **Acceptance Criteria**:
  - Every pillar is emitted as a `0.0-4.0` score with component breakdown.
  - Scorecard emitted at run/project/repo scopes.
  - Pillar components and penalties are fully explorable.
- **Validation**:
  - Golden tests with fixed expected score outputs.

### Task 4.2: Extend Kona Energy Map to Pillar Traversal
- **Location**: `src/statistic_harness/core/stat_plugins/ideaspace.py`, `docs/kona_energy_ideaspace_architecture.md`
- **Description**: Treat pillar gaps as first-class energy terms and produce “ideal traversal route” (Kona as optimization lens, not scoring source):
  - current vs ideal vector,
  - lowest-energy route candidates,
  - blockers preventing low-energy path.
- **Complexity**: 8
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Route output is deterministic and cites source metrics.
  - “Fully optimal” condition is explicitly detectable.
- **Validation**:
  - Synthetic traversal tests with deterministic path ordering.

### Task 4.3: Add Balance Guardrails and Tradeoff Vetoes
- **Location**: `src/statistic_harness/core/four_pillars.py` (new), `src/statistic_harness/core/report.py`, `scripts/plugin_functional_audit.py`
- **Description**: Enforce balanced optimization by default:
  - maximize minimum pillar first (`max-min` objective),
  - apply imbalance penalty when pillar spread exceeds threshold,
  - block recommendations/changes that improve one pillar by degrading another past configured tolerance.
- **Complexity**: 8
- **Dependencies**: Tasks 4.1, 4.2
- **Acceptance Criteria**:
  - Score output includes balance index and veto reasons.
  - Change proposals that fail balance policy are explicitly marked as rejected.
- **Validation**:
  - Unit tests for allowed vs vetoed tradeoff scenarios.

### Task 4.4: Cross-Repository Scorecard Ingestion
- **Location**: `scripts/ingest_repo_scorecard.py` (new), `src/statistic_harness/core/storage.py`, `docs/implementation_matrix.md`
- **Description**: Ingest repo complexity + workflow metadata so different repos/projects can be compared by 4 pillars regardless of domain.
- **Complexity**: 7
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Repo snapshots can be ingested without altering analysis runtime path.
  - Comparison output includes complexity-normalized pillar ranking.
- **Validation**:
  - Fixture-based tests using two synthetic repo snapshots.

### Task 4.5: Per-Change 4-Pillar Recompute
- **Location**: `scripts/run_loaded_dataset_full.py`, `src/statistic_harness/core/report.py`, `src/statistic_harness/core/storage.py`
- **Description**: Recompute and persist 4-pillar scorecards on every materially different change (run fingerprint, plugin code hash, settings hash, dataset hash), not only on full dataset change.
- **Complexity**: 6
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Scorecard history is appended for each change event with fingerprint linkage.
  - Trend views can slice by dataset-stable vs code-change deltas.
- **Validation**:
  - Integration test that changing plugin code updates scorecard lineage without changing dataset input.

## Sprint 5: Reporting, Governance, and Matrix-First Workflow
**Goal**: Keep implementation and governance coherent as the system evolves.  
**Demo/Validation**:
- Reports include professional grouped recommendations + pillar scorecard.
- Matrices update before new implementation to avoid duplicate function work.

### Task 5.1: Add Pillar + Knowledge Sections to Outputs
- **Location**: `scripts/show_actionable_results.py`, `scripts/run_loaded_dataset_full.py`, `src/statistic_harness/core/report.py`
- **Description**: Add sections for:
  - 4-pillar score trend,
  - month vs close weighted metrics,
  - knowledge-bank deltas vs prior runs.
- **Complexity**: 5
- **Dependencies**: Sprint 4
- **Acceptance Criteria**:
  - Human-readable and plain-language outputs include same core conclusions.
  - Known issues include pass/fail + modeled benefit consistently.
- **Validation**:
  - Snapshot tests for markdown outputs.

### Task 5.2: Enforce Matrix-First Change Gate
- **Location**: `scripts/plugin_functional_audit.py`, `docs/plugins_functionality_matrix.json`, `docs/implementation_matrix.json`
- **Description**: Ensure feature work starts with matrix updates and conflict detection to prevent duplicate function creation.
- **Complexity**: 4
- **Dependencies**: None
- **Acceptance Criteria**:
  - CI/test gate fails if matrix is stale relative to plugin registry.
- **Validation**:
  - Unit test for stale-matrix detection.

### Task 5.3: Add Baseline Reset Workflow
- **Location**: `scripts/reset_baseline.sh`, `docs/kona_energy_ideaspace_architecture.md`
- **Description**: Formalize “baseline reset” process for periodic recalibration without losing historical knowledge.
- **Complexity**: 4
- **Dependencies**: Sprint 4
- **Acceptance Criteria**:
  - Reset writes audit event and increments baseline version.
- **Validation**:
  - Integration test for reset + replay.

## Testing Strategy
- Unit tests:
  - mapping/column-learning logic,
  - weighted metric math,
  - pillar score bounds (`0.0-4.0`) and component correctness,
  - balance/veto logic for cross-pillar tradeoffs,
  - scoring and traversal determinism,
  - recommendation memory.
- Integration tests:
  - full pipeline over normalized dataset,
  - two-run comparison with cumulative knowledge updates,
  - report contract checks (modeled fields, traceability, plain-language output).
- Regression/performance tests:
  - runtime/memory trend (`current vs avg/stddev`) persisted per run,
  - deterministic outputs with fixed `run_seed`,
  - no-network runtime guarantee intact.
- Gauntlet gate:
  - `python -m pytest -q` must pass before shipping.

## Potential Risks & Gotchas
- Risk: score overfitting to one ERP profile.
  - Mitigation: enforce cross-domain normalization + confidence penalties for sparse history.
- Risk: one pillar can appear strong while another silently regresses.
  - Mitigation: hard floor + max-spread guardrails, max-min objective, and explicit veto output.
- Risk: knowledge-bank growth inflates DB and query latency.
  - Mitigation: partition/index strategy + materialized rollups.
- Risk: recommendation ranking can be gamed by large modeled values.
  - Mitigation: cap modeled influence, require evidence quality and calibration checks.
- Risk: cross-repo comparisons become unfair without complexity normalization.
  - Mitigation: include complexity factors explicitly (size, plugin surface, runtime budgets).
- Risk: stale baseline distorts Kona route.
  - Mitigation: versioned baseline with explicit reset protocol and drift alerts.

## Rollback Plan
- Feature-flag new knowledge-bank writers and 4-pillar score computation.
- Keep legacy report generation path active during transition.
- If regressions appear:
  - disable new plugins (`analysis_knowledge_bank_comparison_v1`, `analysis_four_pillars_rank_v1`),
  - retain existing normalization and recommendation flows,
  - preserve accumulated knowledge tables (read-only) for postmortem.
