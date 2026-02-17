# Plan: Codex Stat Plugins Spec Pack Implementation

**Generated**: 2026-02-05  
**Estimated Complexity**: High

## Overview
Implement every statistical plugin described in `docs/codex_stat_plugins_spec_pack.md`, ensuring the
4 pillars (Performant, Accurate, Secure, Citable) across all outputs. The work includes shared
profiling utilities, deterministic sampling, stable finding IDs, schema-compliant outputs, test
coverage, and a stepwise gauntlet rerun to produce actionable recommendations on unknown datasets.

## Prerequisites
- Python 3.11+ virtual environment for this repo.
- Optional dependencies (network install allowed): `scipy`, `scikit-learn`, `statsmodels`,
  `lifelines`, `networkx`, `ruptures`, `hmmlearn`, `pm4py`.
- Local-only runtime (no network calls) enforced by plugin sandbox settings.

## Sprint 1: Foundations & Spec Alignment
**Goal**: Establish shared utilities, schema compliance, and scaffolding to implement all missing plugins.
**Demo/Validation**:
- Confirm plugin discovery succeeds for new skeletons.
- Run contract tests for at least one new skeleton plugin.

### Task 1.1: Spec Gap Inventory
- **Location**: `scripts/` (new), `plugins/`, `docs/codex_stat_plugins_spec_pack.md`
- **Description**: Write a small local script that parses the spec pack and compares against `plugins/*`
  to produce a deterministic list of missing plugin IDs (and alias wrappers).
- **Complexity**: 4
- **Dependencies**: None
- **Acceptance Criteria**:
  - Outputs a stable ordered list of missing plugin IDs.
  - Recognizes aliases (e.g., isolation forest wrapper) as wrappers rather than full implementations.
- **Validation**: Run the script and confirm list matches expected missing IDs.

### Task 1.2: Shared Profiling Utilities (4 Pillars)
- **Location**: `src/statistic_harness/core/stat_plugins/` (existing modules)
- **Description**: Ensure shared helpers implement spec requirements:
  column typing heuristics, time axis selection, deterministic sampling, robust numeric matrix,
  stable IDs, and privacy-aware exemplars. Extend or add helpers where missing.
- **Complexity**: 6
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Utilities cover numeric, categorical, timestamp, text detection as spec.
  - Deterministic sampling uses stable hashing with seed.
  - Robust scaling uses MAD with deterministic fallback.
- **Validation**: Unit tests for utility functions in `tests/stat_plugins/`.

### Task 1.3: Plugin Skeleton Generator
- **Location**: `scripts/` (new), `plugins/`
- **Description**: Create a generator that creates plugin folders with
  `plugin.yaml`, `config.schema.json`, `output.schema.json`, `__init__.py`, `plugin.py`
  using shared defaults and schema contract.
- **Complexity**: 5
- **Dependencies**: Task 1.1, 1.2
- **Acceptance Criteria**:
  - Skeletons match `docs/plugin_manifest.schema.json`.
  - Output schema includes required `references` and `debug`.
- **Validation**: `tests/test_plugin_discovery.py` passes after generating skeletons.

### Task 1.4: Shared Contract + Privacy Tests
- **Location**: `tests/stat_plugins/`, `tests/plugins/`, `tests/conftest.py`
- **Description**: Add shared pytest helpers to validate contract fields, determinism,
  and privacy redaction rules for all plugins.
- **Complexity**: 6
- **Dependencies**: Task 1.2
- **Acceptance Criteria**:
  - Contract tests reusable across all new plugins.
  - Privacy test ensures no raw text leakage by default.
- **Validation**: Run test module on one skeleton and one existing plugin.

## Sprint 2: Priority Plugin Set (Spec “implement first”)
**Goal**: Implement the highest-yield plugins first to unlock novel recommendations.
**Demo/Validation**:
- At least 5 priority plugins produce findings on synthetic fixtures.

### Task 2.1: Changepoint & Drift Priority
- **Location**: `plugins/analysis_*` (new)
- **Description**: Implement priority changepoint/drift plugins (e.g., PELT/EDivisive,
  ADWIN, BOCPD Poisson) using shared utilities and deterministic sampling.
- **Complexity**: 7
- **Dependencies**: Sprint 1 complete
- **Acceptance Criteria**:
  - Each plugin emits stable IDs, metrics, references, and debug warnings.
  - Budget/time limits enforced.
- **Validation**: New tests in `tests/plugins/` for each plugin.

### Task 2.2: Control Charts & Effect Sizes
- **Location**: `plugins/analysis_control_chart_*`, `plugins/analysis_effect_size_report`
- **Description**: Implement CUSUM, EWMA, Individuals, multivariate T2/EWMA, PCA chart,
  and effect size summary plugin.
- **Complexity**: 7
- **Dependencies**: Sprint 1 complete
- **Acceptance Criteria**:
  - Statistical thresholds configurable; outputs include effect sizes and confidence proxies.
- **Validation**: Contract tests + deterministic tests on fixtures.

## Sprint 3: Backlog Categories 1–4 (Core Analytics)
**Goal**: Implement backlog categories that cover general drift, dependency, and temporal analysis.
**Demo/Validation**:
- Each category has at least one plugin producing findings on synthetic data.

### Task 3.1: Two-Sample & Distributional Tests
- **Location**: `plugins/analysis_two_sample_*`, `plugins/analysis_kernel_two_sample_mmd`
- **Description**: Add KS/AD/Mann-Whitney/Chi2/MMD plugins with FDR control integration.
- **Complexity**: 6
- **Dependencies**: Sprint 1 complete
- **Acceptance Criteria**:
  - Outputs include p-value + effect size + FDR-adjusted indicator.
- **Validation**: Tests with known synthetic shifts.

### Task 3.2: Dependency & Information Flow
- **Location**: `plugins/analysis_mutual_information_screen`, `analysis_transfer_entropy_directional`,
  `analysis_copula_dependence`, `analysis_graphical_lasso_dependency_network`
- **Description**: Implement dependency discovery and directional influence.
- **Complexity**: 7
- **Dependencies**: Sprint 1 complete
- **Acceptance Criteria**:
  - Stable rankings, bounded graph sizes, and clear recommendations.
- **Validation**: Deterministic tests on synthetic correlated data.

### Task 3.3: Temporal/Sequence Predictability
- **Location**: `plugins/analysis_lagged_predictability_test`,
  `analysis_markov_transition_shift`, `analysis_sequence_classification` (if needed)
- **Description**: Implement lagged predictability and transition shift tests.
- **Complexity**: 6
- **Dependencies**: Sprint 1 complete
- **Acceptance Criteria**:
  - Handles missing time column via no-time mode or graceful skip.
- **Validation**: Contract + determinism tests.

### Task 3.4: Queueing & Duration Models
- **Location**: `plugins/analysis_queue_model_fit`, `analysis_kingman_vut_approx`,
  `analysis_littles_law_consistency`, `analysis_survival_kaplan_meier`,
  `analysis_proportional_hazards_duration`, `analysis_quantile_regression_duration`
- **Description**: Implement queueing/duration plugins using robust estimates.
- **Complexity**: 8
- **Dependencies**: Sprint 1 complete
- **Acceptance Criteria**:
  - Finds bottlenecks with actionable “expected savings” estimates.
- **Validation**: Synthetic wait-time fixtures + integration test.

## Sprint 4: Backlog Categories 5–8 (Process Mining + Advanced)
**Goal**: Implement remaining advanced plugins including process mining, anomalies, templates.
**Demo/Validation**:
- New plugins emit findings and artifacts; all pass contract tests.

### Task 4.1: Process Mining & Conformance
- **Location**: `plugins/analysis_conformance_alignments`,
  `analysis_process_drift_conformance_over_time`, `analysis_sequential_patterns_prefixspan`
- **Description**: Implement conformance and sequence mining; enforce redaction.
- **Complexity**: 8
- **Dependencies**: Sprint 1 complete
- **Acceptance Criteria**:
  - Redaction enforced for text or template evidence.
- **Validation**: Privacy tests + synthetic event log fixtures.

### Task 4.2: Anomaly/Outlier & EVT
- **Location**: `plugins/analysis_evt_*`, `analysis_local_outlier_factor`,
  `analysis_one_class_svm`, `analysis_robust_covariance_outliers`
- **Description**: Implement EVT tail and outlier detection with capped artifacts.
- **Complexity**: 7
- **Dependencies**: Sprint 1 complete
- **Acceptance Criteria**:
  - Findings limited to `max_findings` and `max_exemplars`.
- **Validation**: Determinism + privacy tests.

### Task 4.3: Template/Topic/Entropy Drift
- **Location**: `plugins/analysis_template_drift_two_sample`,
  `analysis_term_burst_kleinberg`, `analysis_topic_model_lda`, `analysis_message_entropy_drift`
- **Description**: Implement text/template drift using redacted exemplars only.
- **Complexity**: 7
- **Dependencies**: Sprint 1 complete
- **Acceptance Criteria**:
  - No raw text leaked; only hashed or redacted exemplars.
- **Validation**: Privacy tests on sample messages.

### Task 4.4: Alias Wrappers
- **Location**: `plugins/analysis_isolation_forest`, `analysis_log_template_drain`,
  `analysis_robust_pca_pcp`
- **Description**: Implement thin wrappers that call existing base plugins.
- **Complexity**: 3
- **Dependencies**: Base plugin implementations
- **Acceptance Criteria**:
  - Wrapper outputs mirror base plugin outputs.
- **Validation**: Wrapper unit tests.

## Sprint 5: Integration + Gauntlet Rerun
**Goal**: Ensure stability, determinism, and actionable output across the full pipeline.
**Demo/Validation**:
- Full `python -m pytest -q` passes.
- Gauntlet run produces `report.md`, `report.json`, and new actionable recommendations.

### Task 5.1: Plugin Ordering & Time Budgets
- **Location**: `src/statistic_harness/core/planner.py`, `plugins/planner_basic/`
- **Description**: Ensure priority plugins run early and honor time budgets; enforce
  close-cycle windows (default + dynamic).
- **Complexity**: 5
- **Dependencies**: Sprints 1–4
- **Acceptance Criteria**:
  - Planner orders plugins by priority and dependency.
- **Validation**: Integration test + planner snapshot test.

### Task 5.2: Stepwise Gauntlet Rerun
- **Location**: `scripts/` (new runner), `appdata/`
- **Description**: Provide a stepwise runner script that executes each stage (ingest, profile,
  analysis, report) in-order with visible output and no background execution.
- **Complexity**: 5
- **Dependencies**: Sprints 1–4
- **Acceptance Criteria**:
  - All stages emit logs to stdout.
  - Reprocessing same dataset resets metadata deterministically.
- **Validation**: Manual run + validate output artifacts exist.

## Testing Strategy
- Contract tests for every new plugin.
- Determinism tests on selected plugins using fixed seed.
- Privacy tests to ensure no raw text leakage by default.
- Integration test running full pipeline and report schema validation.

## Potential Risks & Gotchas
- Optional dependency install failures or version conflicts.
- Algorithms exceeding time budgets on large datasets.
- False positives from drift/anomaly detection without FDR control.
- Privacy leakage from template/topic plugins if redaction is not strict.

## Rollback Plan
- Remove newly added plugin directories.
- Revert shared utility changes.
- Restore prior test snapshots.
