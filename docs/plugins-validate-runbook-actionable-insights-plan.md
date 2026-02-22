# Plan: Plugins Validate Runbook Actionable Insights Pack

**Generated**: 2026-02-22  
**Estimated Complexity**: High

## Overview
Implement the full `docs/plugins_validate_runbook.md` scope with two parallel goals:
1. Harden `stat-harness plugins validate` (EXT-02) to be deterministic, isolated, and testable.
2. Implement all 30 new statistical techniques as individual first-class analysis plugins that emit actionable recommendations (or deterministic non-actionable explanations with dependency maps).

This plan keeps each new technique as a separate plugin. Combination is only allowed for shared utility primitives (scoring/windowing/contract helpers) to avoid logic drift and duplicated defects.

## Prerequisites
- Python environment and test dependencies installed (`.venv`).
- Existing plugin architecture intact:
  - `plugins/<plugin_id>/plugin.yaml|plugin.py|config.schema.json|output.schema.json`
  - `src/statistic_harness/core/stat_plugins/registry.py`
- Existing runbook and acceptance context:
  - `docs/plugins_validate_runbook.md`
  - `docs/plugin_manifest.schema.json`
- Baseline dataset marked golden and used as first acceptance gate.
- Existing full-gauntlet runner scripts available.

## Skills To Use During Implementation
- `shell-lint-ps-wsl`: Normalize all shell commands and avoid cross-shell mistakes.
- `plan-harder`: Maintain phased, atomic, testable sprint structure.
- `python-testing-patterns`: Build per-plugin unit tests and deterministic assertions.
- `testing`: Integrate unit + integration + full-gauntlet validation.
- `discover-observability`: Discover existing metrics/telemetry touchpoints.
- `observability-engineer`: Add run-level and plugin-level reliability metrics.
- `deterministic-tests-marshal`: Guard deterministic behavior and flake prevention.
- `resource-budget-enforcer`: Keep CPU/RAM respectful under full gauntlet.
- `config-matrix-validator`: Validate plugin contract matrix and coverage.

## Scope Contract
- In scope:
  - EXT-02 upgrades for `plugins validate`.
  - 30 new techniques as separate plugins.
  - Actionability contract for every new plugin.
  - Baseline-first acceptance, then synthetic datasets.
- Out of scope:
  - Combining separate techniques into one plugin unless required by shared utility primitives.
  - New external network dependencies.

## Non-Negotiable Acceptance Rules
- Every new plugin must return one of:
  - Actionable recommendation payload with modeled metrics.
  - Deterministic non-actionable response with plain-English reason and downstream dependency map.
- No silent skip/no-op behavior.
- Actionable output must include:
  - `delta_h` for accounting month / close-static / close-dynamic.
  - `eff_%` for accounting month / close-static / close-dynamic.
  - `eff_idx` for accounting month / close-static / close-dynamic.
- Baseline dataset passes first before synthetic rollout.

## Sprint 1: Lock Contracts and Map 30 Techniques
**Goal**: Establish exact plugin IDs, acceptance contract, and dependency matrix before code expansion.  
**Demo/Validation**:
- Produced mapping table with 30 unique plugin IDs.
- Contract doc and matrix validated by tests.

### Task 1.1: Build Technique-to-Plugin ID Matrix (30/30)
- **Location**: `docs/release_evidence/plugins_validate_runbook_technique_map.md`, `docs/release_evidence/plugins_validate_runbook_technique_map.json`
- **Description**: Map each runbook section-8 technique to a unique `analysis_*_v1` plugin ID; detect collisions with existing plugin IDs and resolve deterministically.
- **Complexity**: 4
- **Dependencies**: None
- **Acceptance Criteria**:
  - Exactly 30 entries.
  - Zero duplicate IDs.
  - Zero overlap ambiguity with existing plugin intent.
- **Validation**:
  - Matrix lint test and unique-ID assertion.

### Task 1.2: Define Shared Actionability Contract
- **Location**: `docs/schemas/actionable_recommendation_contract_v2.json`, `docs/release_evidence/plugins_validate_runbook_contract.md`
- **Description**: Define required fields for actionable and deterministic non-actionable responses, including window triplets and downstream dependency map.
- **Complexity**: 5
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Contract schema includes required modeled metrics and reason/dependency fields.
  - Existing recommendation formatter compatibility documented.
- **Validation**:
  - Contract schema validation tests.

### Task 1.3: Build Plugin Dependency and Consumer Matrix
- **Location**: `docs/release_evidence/plugins_validate_runbook_dependency_matrix.csv`
- **Description**: Track which downstream plugins consume each new plugin’s signals; required for non-actionable fallback reasoning.
- **Complexity**: 4
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Every new plugin has producer/consumer rows.
  - Empty consumer sets explicitly marked.
- **Validation**:
  - Matrix completeness test (30/30 plugin rows).

### Task 1.4: Codify Actionability Threshold Constants
- **Location**: `config/actionability_thresholds.yaml`, `src/statistic_harness/core/actionability_thresholds.py`, `docs/release_evidence/actionability_thresholds.md`
- **Description**: Define shared minimum thresholds for actionable classification (`delta_h`, `eff_%`, `eff_idx`, confidence/significance), with deterministic defaults and override policy.
- **Complexity**: 5
- **Dependencies**: Task 1.2
- **Acceptance Criteria**:
  - Single source of truth for thresholds used by all new plugins and ranking logic.
  - Threshold provenance and change-control owner documented.
- **Validation**:
  - Threshold schema test + cross-module import/use assertions.

### Task 1.5: Technique Spec Pack (30 Individual Specs)
- **Location**: `docs/release_evidence/plugins_validate_runbook_technique_specs/*.md`
- **Description**: Create one spec per technique containing runbook reference, plugin ID, expected input columns/signals, deterministic seed, tolerances, and acceptance checks.
- **Complexity**: 6
- **Dependencies**: Task 1.1, Task 1.4
- **Acceptance Criteria**:
  - 30 spec files present and linked from the mapping matrix.
  - Every plugin task references its own spec file.
- **Validation**:
  - Spec index completeness check (30/30).

## Sprint 2: EXT-02 `plugins validate` Hardening
**Goal**: Make `stat-harness plugins validate` complete vs EXT-02 (schema + import + smoke + caps + report, deterministic).  
**Demo/Validation**:
- `stat-harness plugins validate` supports deterministic JSON report and failure semantics.
- Isolated import/health path working.

### Task 2.1: Apply Schema Defaults During Validation
- **Location**: `src/statistic_harness/cli.py`, `src/statistic_harness/core/plugin_manager.py`
- **Description**: Replace direct default validation with `resolve_config` flow so schema defaults are materialized before validation.
- **Complexity**: 3
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Required+default schema fields validate correctly.
  - Regression test added for default-materialization pass/fail case.
- **Validation**:
  - `tests/test_cli_plugins_validate.py`, `tests/test_config_defaults.py`.

### Task 2.2: Add Isolated Validation Mode
- **Location**: `src/statistic_harness/core/plugin_runner.py`, `src/statistic_harness/core/pipeline.py`, `src/statistic_harness/cli.py`
- **Description**: Add isolated subprocess validation action for import + optional health check.
- **Complexity**: 7
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - In-process fallback disabled for strict mode.
  - Import and health failures surface with plugin-specific error messages.
- **Validation**:
  - New CLI tests for isolated import failure and unhealthy health response.

### Task 2.3: Add `--smoke`, `--caps`, and `--json`
- **Location**: `src/statistic_harness/cli.py`, `docs/plugins_validate_runbook.md`, `docs/release_evidence/plugins_validate_schema.json`
- **Description**: Implement smoke run mode (tiny synthetic context), capability report, and deterministic JSON artifact output.
- **Complexity**: 7
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - JSON output stable order.
  - Smoke mode validates `run()` output schema.
  - Caps section includes manifest capabilities/sandbox.
- **Validation**:
  - Snapshot-style JSON regression tests.

### Task 2.4: Hard Fail Rules + DO_NOT_SHIP Gates
- **Location**: `tests/test_cli_plugins_validate.py`, `tests/test_plugin_manifest_schema.py`
- **Description**: Encode all DO_NOT_SHIP scenarios from runbook section 6.4.
- **Complexity**: 5
- **Dependencies**: Task 2.3
- **Acceptance Criteria**:
  - Malformed manifest/import/health failures always exit non-zero.
  - Deterministic output order verified.
- **Validation**:
  - Full plugin validate CI target.

## Sprint 3: Shared Utility Substrate (Only Necessary Combination)
**Goal**: Create shared primitives to keep 30 plugins separate but consistent.  
**Why combining here is necessary**:
- Without shared scoring/windowing/contract code, 30 plugins would drift in output semantics, violating determinism and comparability.
**Demo/Validation**:
- Shared utility module consumed by multiple plugins with identical contract output.

### Task 3.1: Window Triplet Metrics Helper
- **Location**: `src/statistic_harness/core/stat_plugins/actionability_metrics.py`
- **Description**: Centralize accounting-month/close-static/close-dynamic metric computation.
- **Complexity**: 6
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Single helper API returns `delta_h`, `eff_%`, `eff_idx` triplets.
- **Validation**:
  - Unit tests for edge windows and zero-denominator behavior.

### Task 3.2: Deterministic Non-Actionable Envelope
- **Location**: `src/statistic_harness/core/stat_plugins/actionability_envelope.py`
- **Description**: Standard builder for non-actionable reason + downstream dependency map.
- **Complexity**: 5
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Standard reason codes and plain-English explanation shape.
- **Validation**:
  - Schema + formatter tests.

### Task 3.3: Shared Recommendation Contract Validator
- **Location**: `src/statistic_harness/core/stat_plugins/actionability_contract.py`, `tests/test_actionability_contract_v2.py`
- **Description**: Validate plugin outputs against actionability contract before returning.
- **Complexity**: 6
- **Dependencies**: Task 3.2
- **Acceptance Criteria**:
  - Contract violations fail plugin with explicit actionable error.
- **Validation**:
  - Contract negative tests.

## Sprint 4: Scaffold 30 Plugins Individually (No Combined Plugins)
**Goal**: Generate first-class plugin shells/tests for each new technique.  
**Demo/Validation**:
- 30 plugin directories + manifests + schemas + smoke tests present and discoverable.

### Task 4.1: Create Scaffolding Script for Runbook 30
- **Location**: `scripts/gen_plugins_validate_runbook_scaffold.py`
- **Description**: Generate 30 plugin folders from mapping matrix with standard manifest/schema/wrapper.
- **Complexity**: 5
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Script is idempotent.
  - Emits all 30 plugin dirs.
- **Validation**:
  - Script dry-run + generated-file consistency test.

### Task 4.2 - 4.31: Scaffold Each Plugin (30 Individual Tasks)
- **Location**: `plugins/<plugin_id>/...`, `tests/plugins/test_<plugin_id>.py`
- **Description**: For each mapped technique plugin ID, add:
  - `plugin.yaml`
  - `plugin.py`
  - `config.schema.json`
  - `output.schema.json`
  - dedicated smoke test file
- **Complexity**: 2 each
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Each plugin independently discoverable and loadable.
  - Each plugin has its own dedicated test file.
- **Validation**:
  - Per-plugin smoke tests and `plugins validate` pass.

## Sprint 5: Implement 30 Technique Plugins Individually
**Goal**: Add full method logic and actionable insight mapping for each technique plugin, one plugin per method.  
**Demo/Validation**:
- Every plugin returns actionable or deterministic non-actionable with dependency map.

### Task 5.0: Deterministic Data-Readiness Packs For 30 Plugins
- **Location**: `tests/fixtures/plugins_validate_runbook/<plugin_id>/`, `docs/release_evidence/plugins_validate_runbook_data_readiness_matrix.csv`
- **Description**: Build deterministic fixture packs and seed/tolerance declarations per plugin before method implementation.
- **Complexity**: 7
- **Dependencies**: Sprint 4, Task 1.5
- **Acceptance Criteria**:
  - Each plugin has fixed fixture(s), required columns list, and deterministic seed.
  - Fixture matrix marks baseline-first and synthetic follow-up coverage.
- **Validation**:
  - Fixture coverage test (30/30) and seed determinism check.

### Technique Plugin List (Individual Implementations)
1. Conformal Prediction  
2. FDR Benjamini-Hochberg  
3. BCa Bootstrap CI  
4. Knockoff Filter  
5. Quantile Regression  
6. LASSO  
7. Elastic Net  
8. Random Forest  
9. Gradient Boosting  
10. Isolation Forest  
11. Local Outlier Factor  
12. Minimum Covariance Determinant  
13. Huber M-estimator  
14. Gaussian Process Regression  
15. Generalized Additive Models  
16. Mixed Effects Model  
17. Cox Proportional Hazards  
18. BART  
19. Dirichlet Process Mixture  
20. Hidden Markov Models  
21. Kalman Filter  
22. Granger Causality  
23. Transfer Entropy  
24. kNN Mutual Information  
25. Independent Component Analysis  
26. Non-negative Matrix Factorization  
27. t-SNE  
28. UMAP  
29. NOTEARS  
30. MICE Imputation

### Task Pattern (Apply To Each of 30 Plugins)
- **Location**: `src/statistic_harness/core/stat_plugins/<new_module_or_addon>.py`, `plugins/<plugin_id>/`, `tests/plugins/test_<plugin_id>.py`
- **Description**:
  - Implement method-specific computation.
  - Load thresholds from shared constants module (no plugin-local hardcoded thresholds).
  - Map method signal to explicit process-level recommendation.
  - Emit required triplet metrics (`delta_h`, `eff_%`, `eff_idx`).
  - Emit deterministic non-actionable reason + downstream dependency map when actionability threshold not met.
- **Complexity**: 6 each
- **Dependencies**: Sprint 3 + Sprint 4
- **Acceptance Criteria**:
  - No plugin returns silent/empty result.
  - Method statistics are present and deterministic under fixed seed.
  - Output contract passes validator.
- **Validation**:
  - Plugin-specific unit tests (happy path + edge + fallback).
  - Determinism tests (repeat runs same seed same output).

## Sprint 6: Recommendation Integration and Ranking Coherence
**Goal**: Ensure new plugins improve actionable recommendation quality without breaking existing ranking semantics.  
**Demo/Validation**:
- Top-N recommendations include new-plugin insights where justified.
- Dimensionless ranking comparable across recommendation types.

### Task 6.1: Integrate New Plugins Into Discovery/Recommendation Flow
- **Location**: `src/statistic_harness/core/report.py`, `src/statistic_harness/core/report_v2_utils.py`
- **Description**: Ensure all 30 plugin outputs flow into dedupe/ranking with consistent score fields.
- **Complexity**: 7
- **Dependencies**: Sprint 5
- **Acceptance Criteria**:
  - New plugin findings visible in recommendation bundle.
  - Ranking uses uniform dimensionless efficiency fields.
- **Validation**:
  - Recommendation diff tests.

### Task 6.2: Non-Actionable Reason Taxonomy + Downstream Trace
- **Location**: `src/statistic_harness/core/report.py`, `scripts/audit_plugin_actionability.py`
- **Description**: Normalize reason taxonomy and show dependency chain for each non-actionable return.
- **Complexity**: 6
- **Dependencies**: Task 6.1
- **Acceptance Criteria**:
  - No opaque reason buckets.
  - Every non-actionable includes downstream list.
- **Validation**:
  - Actionability audit tests.

## Sprint 7: Baseline-First Validation, Then Synthetic Expansion
**Goal**: Enforce baseline golden acceptance first, then synthetic comparisons.  
**Demo/Validation**:
- Baseline gauntlet passes with all new plugins.
- Synthetic comparisons generated and published.

### Task 7.1: Baseline Golden Full Gauntlet
- **Location**: run artifacts under `appdata/runs/*`, evidence in `docs/release_evidence/`
- **Description**: Run full gauntlet on baseline only and enforce pass criteria.
- **Complexity**: 5
- **Dependencies**: Sprint 6
- **Acceptance Criteria**:
  - All plugins terminal (ok/error/na allowed per policy, no silent skip).
  - `plugins validate` strict mode passes.
- **Validation**:
  - `python -m pytest -q`
  - Full gauntlet run status checks.

### Task 7.2: Synthetic 6mo + 8mo Full Gauntlet
- **Location**: run artifacts + `docs/release_evidence/compare_*`
- **Description**: Run full gauntlet on both synthetics and generate baseline-vs-synthetic diffs.
- **Complexity**: 5
- **Dependencies**: Task 7.1
- **Acceptance Criteria**:
  - Comparison artifacts for outputs, actionability, and Kona route quality.
- **Validation**:
  - `scripts/compare_run_outputs.py`
  - `scripts/compare_plugin_actionability_runs.py`
  - `scripts/compare_kona_route_quality.py`

### Task 7.3: Freshman Top-20 Delivery
- **Location**: `docs/release_evidence/freshman_top20_baseline_vs_synth.md`
- **Description**: Publish plain-English top-20 recommendations with modeled gains and user/server impact.
- **Complexity**: 4
- **Dependencies**: Task 7.2
- **Acceptance Criteria**:
  - Includes time-frame and modeled savings context.
  - Includes dimensionless comparability fields.
- **Validation**:
  - Manual review + consistency check against JSON artifacts.

## Sprint 8: Observability, Performance, and Freeze Workflow
**Goal**: Keep runs reliable and respectful while locking stable surfaces.  
**Demo/Validation**:
- Deterministic metrics dashboard and frozen-surface checks pass.

### Task 8.1: Plugin-Level Runtime and Failure Metrics
- **Location**: `src/statistic_harness/core/pipeline.py`, `scripts/runtime_stats.py`, `docs/release_evidence/plugin_runtime_matrix.json`
- **Description**: Capture runtime, memory budget events, and failure taxonomies per plugin.
- **Complexity**: 6
- **Dependencies**: Sprint 7
- **Acceptance Criteria**:
  - Metrics available for all plugins in run.
- **Validation**:
  - Runtime stats script checks.

### Task 8.2: Resource Budget Enforcement for Full Gauntlet
- **Location**: `scripts/start_full_loaded_dataset_bg.sh`, budget tests in `tests/plugins/`
- **Description**: Keep CPU/RAM limits deterministic and host-friendly under full runs.
- **Complexity**: 5
- **Dependencies**: Task 8.1
- **Acceptance Criteria**:
  - No uncontrolled spikes; run remains stable.
- **Validation**:
  - Resource budget tests + repeated run checks.

### Task 8.3: Freeze Stable Plugin Surfaces
- **Location**: `scripts/freeze_working_plugin_surfaces.py`, `scripts/verify_frozen_plugin_surfaces.py`
- **Description**: Lock stable plugin contracts and detect regressions before merges.
- **Complexity**: 4
- **Dependencies**: Task 8.2
- **Acceptance Criteria**:
  - Stable plugins frozen and verified in CI.
- **Validation**:
  - Freeze/verify scripts.

### Task 8.4: Cross-Shell Execution Parity (WSL + PowerShell)
- **Location**: `scripts/*.sh`, `scripts/*.ps1`, `docs/runbooks/cross_shell_execution.md`
- **Description**: Ensure every required gauntlet/validation command has a short one-line Bash and PowerShell equivalent with matching behavior and evidence outputs.
- **Complexity**: 5
- **Dependencies**: Task 8.2
- **Acceptance Criteria**:
  - Required commands can run in both WSL and PowerShell without behavior drift.
  - CI docs include platform-specific invocation guidance.
- **Validation**:
  - Command parity tests and smoke runs on both shells.

## Testing Strategy
- Unit:
  - 30 per-plugin tests with deterministic seeds and edge guards.
  - Contract tests for actionable/non-actionable envelopes.
- Integration:
  - `plugins validate` strict suite (schema/import/smoke/caps/json).
  - Recommendation aggregation and ranking consistency tests.
- Full-system:
  - Baseline-first full gauntlet gate.
  - Synthetic 6mo and 8mo gauntlets after baseline pass.
- Determinism:
  - Repeat-run signature tests for plugin outputs.
  - Stable ordering checks for CLI and report artifacts.

## Potential Risks & Gotchas
- Existing plugin overlap with new 30 techniques may cause duplicated recommendations.
  - Mitigation: explicit technique-to-plugin mapping and dedupe rule tests.
- `report_bundle` and host FS latency may cause timeout-driven partial runs.
  - Mitigation: bounded timeout policy with deterministic retry lane and post-run state repair rules.
- Non-actionable reason inflation can hide weak plugin logic.
  - Mitigation: mandatory downstream dependency map and audit gates.
- Large plugin count increases runtime and memory pressure.
  - Mitigation: resource-budget enforcer, respectful profile defaults, plugin-level caps.
- Drift in recommendation scoring semantics across plugins.
  - Mitigation: shared actionability contract validator and centralized triplet metric helper.

## Rollback Plan
- Keep implementation in incremental commits by sprint and plugin slice.
- If a sprint regresses core behavior:
  - Revert sprint-level commits only.
  - Preserve mapping/contracts artifacts for audit trace.
- For plugin-level regressions:
  - Disable only affected plugin IDs via run config allow/deny list while retaining others.
- If EXT-02 strict mode blocks release unexpectedly:
  - Retain baseline mode command path behind explicit strict flag until fixes land.

## Definition of Done
- EXT-02 hardened command passes strict regression suite.
- All 30 techniques implemented as separate plugins with independent tests.
- Every new plugin returns actionable insight or deterministic non-actionable explanation with downstream map.
- Baseline golden run passes first; synthetic runs pass afterward.
- Comparison artifacts and freshman top-20 report are published and reproducible.
