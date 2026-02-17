# Plan: Codex Unimplemented Specs (Decision Report v2 + Full Stat Plugin Pack)

**Generated**: 2026-02-04
**Estimated Complexity**: High

## Overview
Implement everything described in the three `unimplemented/codex_*.md` specs: the Decision Report v2 system (issue cards, busy periods, waterfall, dedupe, traceability, slide kit, report bundle) plus the full statistical plugin spec pack (priority set and backlog categories 1–8). The plan emphasizes the 4 pillars (Performant, Accurate, Secure, Citable), deterministic behavior, and a final full pipeline rerun that captures time-to-completion metrics. Redaction is disabled by default for dev runs, but redaction capabilities remain available and toggleable for client report generation.

## Prerequisites
- Python 3.11+ environment with project dependencies installed.
- Optional libraries for advanced methods (sklearn, ruptures, lifelines, statsmodels, hmmlearn, pm4py). If missing, plugins must fall back or skip.
- Ability to run full pipeline locally and capture timing metrics from `plugin_executions`.

## Dependency Install Matrix
| Capability | Python Package | Required For | Fallback if Missing |
| --- | --- | --- | --- |
| Advanced ML (IsolationForest, LOF, OCSVM, Graphical Lasso) | `scikit-learn` | Anomaly + dependency plugins | Robust z-score or skip with reason |
| Changepoints (PELT optional) | `ruptures` | Multivariate/univariate changepoints | Pure python PELT fallback |
| Survival/Cox PH | `lifelines` | Cox PH duration modeling | Stratified log-rank or skip |
| Quantile regression | `statsmodels` | Quantile regression duration | Binning + median fallback |
| HMM | `hmmlearn` | Latent state sequences | Markov baseline fallback |
| Process mining | `pm4py` | Conformance alignments | Proxy DFG conformance |

## Sprint 1: Foundations & Shared Utilities
**Goal**: Establish shared utilities and tests to enforce the common plugin contract, determinism, and budgets.
**Demo/Validation**:
- Run unit tests for shared utilities and contract tests.
- Verify a minimal plugin can consume the shared utilities and pass determinism tests.

### Task 1.1: Shared stat plugin utilities
- **Location**: `src/statistic_harness/core/stat_plugins/` (new package)
- **Description**: Implement reusable helpers for column inference, deterministic sampling, robust scaling, effect sizes, FDR (BH), time-budget enforcement, PII redaction toggle, and stable ID hashing. Include default config handling aligned to the spec pack.
- **Complexity**: 7
- **Dependencies**: None
- **Acceptance Criteria**:
  - Utilities cover all common config fields from the spec pack.
  - Deterministic sampling uses stable hashing with seed.
- **Validation**:
  - Unit tests in `tests/stat_plugins/test_utils.py`.

### Task 1.2: Common plugin contract tests
- **Location**: `tests/stat_plugins/test_contract.py`
- **Description**: Implement the common test suite (contract, determinism, privacy toggle, performance budget, findings cap).
- **Complexity**: 6
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - All new plugins can import and reuse these tests.
- **Validation**:
  - `python -m pytest -q tests/stat_plugins/test_contract.py`

### Task 1.3: Shared config schema + references schema
- **Location**: `docs/stat_plugin_config.schema.json`, `docs/stat_plugin_references.schema.json`
- **Description**: Create JSON schemas for common config keys and references payloads for plugins; reference from new plugin schemas.
- **Complexity**: 5
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - New plugin schemas validate defaults and references arrays.
- **Validation**:
  - Schema validation in plugin manager.

## Sprint 2: Decision Report v2 Core Plugins
**Goal**: Implement the Decision Report v2 pipeline, slide kit outputs, issue cards, waterfall, traceability, and dedupe.
**Demo/Validation**:
- Generate `business_summary.md`, `engineering_summary.md`, `appendix_raw.md`.
- Produce slide kit CSV/JSON artifacts with stable ordering.

### Task 2.1: analysis_issue_cards_v2
- **Location**: `plugins/analysis_issue_cards_v2/`
- **Description**: Build issue cards from known issues + checks with PASS/FAIL logic and evidence pointers.
- **Complexity**: 6
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Output includes all required fields and fails if predicates/values missing.
- **Validation**:
  - Unit test for PASS/FAIL with missing predicate.

### Task 2.2: analysis_busy_period_segmentation_v2
- **Location**: `plugins/analysis_busy_period_segmentation_v2/`
- **Description**: Compute busy periods from over-threshold wait intervals and emit CSV + JSON artifacts.
- **Complexity**: 7
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Busy periods merge gaps <= tolerance and are sorted as specified.
- **Validation**:
  - Unit test with synthetic intervals.

### Task 2.3: analysis_waterfall_summary_v2
- **Location**: `plugins/analysis_waterfall_summary_v2/`
- **Description**: Build waterfall summary with exact arithmetic in seconds and rounding rules in hours.
- **Complexity**: 6
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - total = top_driver + remainder within 1 second tolerance.
- **Validation**:
  - Unit test for arithmetic reconciliation.

### Task 2.4: analysis_recommendation_dedupe_v2
- **Location**: `plugins/analysis_recommendation_dedupe_v2/`
- **Description**: Deduplicate recommendations by action + target + scenario + delta signature; fail on conflicting deltas.
- **Complexity**: 6
- **Dependencies**: Task 2.3
- **Acceptance Criteria**:
  - Conflicting deltas fail the run.
- **Validation**:
  - Unit test with duplicate/conflict cases.

### Task 2.5: analysis_traceability_manifest_v2
- **Location**: `plugins/analysis_traceability_manifest_v2/`
- **Description**: Build claim registry and enforce that all printed numbers map to claims.
- **Complexity**: 7
- **Dependencies**: Task 2.1–2.4
- **Acceptance Criteria**:
  - Missing claim mapping fails the run.
- **Validation**:
  - Unit test for missing claim failure.

### Task 2.6: report_decision_bundle_v2 + report_slide_kit_emitter_v2
- **Location**: `plugins/report_decision_bundle_v2/`, `plugins/report_slide_kit_emitter_v2/`, `src/statistic_harness/core/report.py`
- **Description**: Render decision vs appendix outputs and emit slide kit CSVs with stable ordering and row limits.
- **Complexity**: 8
- **Dependencies**: Task 2.1–2.5
- **Acceptance Criteria**:
  - Business summary <= 2 pages equivalent.
  - Engineering summary includes issue cards + checks + traceability.
- **Validation**:
  - Integration test with golden dataset fixture.

### Task 2.7: Check runner enhancements
- **Location**: `src/statistic_harness/core/report.py`, `src/statistic_harness/core/storage.py`
- **Description**: Ensure checks output predicate text, computed/target values, decisions, and evidence.
- **Complexity**: 5
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - FAIL checks include predicate + computed values.
- **Validation**:
  - Unit test with failing check output.

## Sprint 3: Priority Statistical Plugins (Spec Pack Section 5)
**Goal**: Implement the eight highest-priority statistical plugins.
**Demo/Validation**:
- Each plugin runs on a synthetic fixture and emits deterministic findings.

### Task 3.1: analysis_control_chart_suite
- **Location**: `plugins/analysis_control_chart_suite/`
- **Validation**: tests for shift detection and group slicing.

### Task 3.2: analysis_multivariate_control_charts
- **Location**: `plugins/analysis_multivariate_control_charts/`
- **Validation**: tests for joint shift and shrinkage fallback.

### Task 3.3: analysis_multivariate_changepoint_pelt
- **Location**: `plugins/analysis_multivariate_changepoint_pelt/`
- **Validation**: tests for changepoint recovery and penalty control.

### Task 3.4: analysis_distribution_drift_suite
- **Location**: `plugins/analysis_distribution_drift_suite/`
- **Validation**: tests for numeric/categorical drift and FDR control.

### Task 3.5: analysis_isolation_forest_anomaly
- **Location**: `plugins/analysis_isolation_forest_anomaly/`
- **Validation**: tests for outlier ranking + fallback.

### Task 3.6: analysis_robust_pca_sparse_outliers
- **Location**: `plugins/analysis_robust_pca_sparse_outliers/`
- **Validation**: tests for sparse recovery and fallback.

### Task 3.7: analysis_log_template_mining_drain
- **Location**: `plugins/analysis_log_template_mining_drain/`
- **Validation**: tests for template stability and redaction toggle.

### Task 3.8: analysis_conformance_checking
- **Location**: `plugins/analysis_conformance_checking/`
- **Validation**: tests for new edge detection and conformance proxy.

## Sprint 4: Backlog Category 1 & 2 (Drift/Changepoints + Cohorts)
**Goal**: Implement drift/changepoint wrappers and cohort comparison plugins.
**Demo/Validation**:
- Run each plugin on synthetic drift fixtures and validate deterministic output.

### Task 4.1: Control chart wrappers (CUSUM/EWMA/Individuals)
- **Location**: `plugins/analysis_control_chart_cusum/`, `plugins/analysis_control_chart_ewma/`, `plugins/analysis_control_chart_individuals/`

### Task 4.2: Multivariate wrappers (T²/MEWMA/PCA)
- **Location**: `plugins/analysis_multivariate_t2_control/`, `plugins/analysis_multivariate_ewma_control/`, `plugins/analysis_pca_control_chart/`

### Task 4.3: Changepoint variants (univariate, energy, ADWIN)
- **Location**: `plugins/analysis_changepoint_pelt/`, `plugins/analysis_changepoint_energy_edivisive/`, `plugins/analysis_drift_adwin/`

### Task 4.4: Cohort comparison plugins
- **Location**: `plugins/analysis_two_sample_numeric_ks/`, `analysis_two_sample_numeric_ad/`, `analysis_two_sample_numeric_mann_whitney/`, `analysis_two_sample_categorical_chi2/`, `analysis_kernel_two_sample_mmd/`

### Task 4.5: Effect sizes and FDR helpers as plugins
- **Location**: `plugins/analysis_effect_size_report/`, `plugins/analysis_multiple_testing_fdr/`

### Task 4.6: Pre/post impact plugin
- **Location**: `plugins/analysis_change_impact_pre_post/`

## Sprint 5: Backlog Category 3 (Anomaly + Tail Risk)
**Goal**: Implement anomaly detectors and tail risk estimators.
**Demo/Validation**:
- Validate detection on synthetic outlier/tail fixtures.

### Task 5.1: LOF, One-Class SVM, Robust Covariance
- **Location**: `plugins/analysis_local_outlier_factor/`, `analysis_one_class_svm/`, `analysis_robust_covariance_outliers/`

### Task 5.2: EVT Gumbel + POT
- **Location**: `plugins/analysis_evt_gumbel_tail/`, `analysis_evt_peaks_over_threshold/`

### Task 5.3: Matrix profile motifs/discords
- **Location**: `plugins/analysis_matrix_profile_motifs_discords/`

## Sprint 6: Backlog Category 4 (Burst + Event-Rate)
**Goal**: Implement burst detection, BOCPD, Hawkes, spectral periodicity, and Kalman residuals.
**Demo/Validation**:
- Detect synthetic spikes, rate changes, and periodic signals.

### Task 6.1: Burst detection + BOCPD Poisson
- **Location**: `plugins/analysis_burst_detection_kleinberg/`, `analysis_event_count_bocpd_poisson/`

### Task 6.2: Hawkes self-exciting
- **Location**: `plugins/analysis_hawkes_self_exciting/`

### Task 6.3: Periodicity + Kalman residuals
- **Location**: `plugins/analysis_periodicity_spectral_scan/`, `analysis_state_space_kalman_residuals/`

## Sprint 7: Backlog Category 5 (Process Mining + Sequences)
**Goal**: Add process mining and sequence intelligence plugins.
**Demo/Validation**:
- Detect new edges, conformance drift, and discriminative variants on synthetic logs.

### Task 7.1: Conformance alignments + drift
- **Location**: `plugins/analysis_conformance_alignments/`, `analysis_process_drift_conformance_over_time/`

### Task 7.2: Variant differential + Markov shift
- **Location**: `plugins/analysis_variant_differential/`, `analysis_markov_transition_shift/`

### Task 7.3: PrefixSpan + HMM latent states
- **Location**: `plugins/analysis_sequential_patterns_prefixspan/`, `analysis_hmm_latent_state_sequences/`

### Task 7.4: Dependency graph change detection
- **Location**: `plugins/analysis_dependency_graph_change_detection/`

## Sprint 8: Backlog Category 6 (Text/Message Field Intelligence)
**Goal**: Add LDA, term burst, entropy drift, template distribution drift.
**Demo/Validation**:
- Detect emergent tokens/templates on synthetic text fixtures.

### Task 8.1: Topic modeling LDA
- **Location**: `plugins/analysis_topic_model_lda/`

### Task 8.2: Term burst + message entropy drift
- **Location**: `plugins/analysis_term_burst_kleinberg/`, `analysis_message_entropy_drift/`

### Task 8.3: Template drift (two-sample)
- **Location**: `plugins/analysis_template_drift_two_sample/`

## Sprint 9: Backlog Categories 7–8 (Dependency/Causality + Duration/SLA)
**Goal**: Implement causality proxies and duration/SLA/queueing checks.
**Demo/Validation**:
- Detect causal lags and SLA risk in synthetic fixtures.

### Task 9.1: Dependency & causality
- **Location**: `plugins/analysis_graphical_lasso_dependency_network/`, `analysis_mutual_information_screen/`, `analysis_transfer_entropy_directional/`, `analysis_lagged_predictability_test/`, `analysis_copula_dependence/`

### Task 9.2: Duration & SLA risk
- **Location**: `plugins/analysis_survival_kaplan_meier/`, `analysis_proportional_hazards_duration/`, `analysis_quantile_regression_duration/`

### Task 9.3: Queueing sanity checks
- **Location**: `plugins/analysis_littles_law_consistency/`, `analysis_kingman_vut_approx/`, `analysis_queue_model_fit/`

## Sprint 10: Integration, Planner, and Full Rerun
**Goal**: Wire all plugins into the planner, run the full dataset, and capture time-to-completion.
**Demo/Validation**:
- Full pipeline run with all plugins enabled and runtime recorded.

### Task 10.1: Planner integration
- **Location**: `src/statistic_harness/core/planner.py`, `plugins/*/plugin.yaml`
- **Description**: Ensure new plugins are discoverable and selectable with appropriate capabilities.
- **Validation**: CLI lists all plugins and planner selects expected sets.

### Task 10.2: Runtime capture + reporting
- **Location**: `src/statistic_harness/core/report.py`, `slide_kit/artifacts_manifest.json`
- **Description**: Add end-to-end runtime metrics in engineering summary and slide kit.
- **Validation**: Confirm runtime metrics present and traceable.

### Task 10.3: Full rerun (time to completion)
- **Location**: CLI + run outputs in `appdata/runs/`
- **Description**: Run dataset with all plugins; record total run time and per-plugin durations.
- **Validation**:
  - `python -m pytest -q` passes.
  - New reports and slide_kit are generated.

## Testing Strategy
- Unit tests per plugin (per spec pack + Decision Report v2 tests).
- Integration tests for Decision Report v2 and slide_kit outputs.
- Determinism tests for all plugins.
- Performance budget tests for heavy plugins and sampling.

## Potential Risks & Gotchas
- Scope size is very large; consider staged delivery if runtime becomes excessive.
- Optional dependencies may not be installed; plugins must have robust fallbacks or skip paths.
- Enforcing “no redaction” for dev must not break client-report redaction later. Implement toggles with explicit report mode.
- Traceability enforcement can block report generation if any number is emitted without a claim. Ensure claim registry coverage.
- Time-budget enforcement must prevent runaway O(n^2) algorithms.

## Rollback Plan
- Revert plugin integration changes by removing new plugin directories and references from planner.
- Restore previous report generator as the default `report_bundle` until v2 is stable.
- Use git revert to back out specific plugin batches if needed.
