# Plan: Gauntlet Workflow + Novel Recommendation Discovery

**Generated**: 2026-02-04  
**Estimated Complexity**: High

## Overview
Build a deterministic gauntlet workflow that runs the full plugin stack against the test dataset and adds targeted analysis plugins to produce **top‑10 actionable recommendations** outside excluded processes. The workflow must satisfy the 4 pillars (Performant, Accurate, Secure, Citable) and output a clear, evidence‑linked answer to the known‑issue question.

Assumptions (based on your answers):
- Primary KPI for time savings: **Busy‑period over‑threshold wait‑to‑start hours** (BP_OT_WTS_HOURS).
- Exclusions list: **LOS, POSTWKFL, BKRVNU, QEMAIL, QPEC** are excluded **as recommendation targets**, but can appear as context or sequence steps.
- Output: **Top 10 recommendations** (ranked), each with modeled hours saved + evidence + validation steps.

If any assumption is wrong, I will revise the plan before implementation.

## Prerequisites
- Dataset path (WSL): `/mnt/d/projects/statistics_harness/statistics_harness/appdata/uploads/e9c3e32292cf42f2a36624ce44c0d7c2/proc log 1-14-26.csv`
- Python venv: `/mnt/d/projects/statistics_harness/statistics_harness/.venv_wsl/bin/python`
- Existing gauntlet runner: `scripts/run_gauntlet.ps1`
- Reports v2 plugins already present (Decision + Slide Kit)

## Sprint 1: Gauntlet Workflow + Dataset Harness
**Goal**: Ensure the dataset flows through the entire gauntlet with deterministic settings and produces all artifacts needed by the new recommendation plugins.

**Demo/Validation**:
- Run gauntlet end‑to‑end on the dataset and confirm `report.json`, `business_summary.md`, `engineering_summary.md`, and `slide_kit/*` appear.
- Validate run status is `completed` and all plugin executions are recorded.

### Task 1.1: Add a Gauntlet Entry Point for the Test Dataset
- **Location**: `scripts/run_gauntlet.ps1`, `scripts/run_latest_dataset.ps1`
- **Description**: Add a dedicated “full gauntlet” entry point that runs the pipeline with `plugins=all` (analysis) and outputs the run_id + report paths. Ensure it uses deterministic seed and logs runtime per plugin.
- **Complexity**: 4
- **Dependencies**: None
- **Acceptance Criteria**:
  - Single command runs the gauntlet on the test dataset.
  - Output includes run_id and report paths.
- **Validation**:
  - Run on dataset and confirm generated artifacts.

### Task 1.2: Add Time-to-Completion Metrics to the Shared Profile
- **Location**: `plugins/profile_eventlog/`, `plugins/profile_basic/`, `src/statistic_harness/core/report_v2_utils.py`
- **Description**: Add a deterministic “time-to-completion” metric derived from start/end timestamps (or inferred close cycle) and include it in profile outputs and report KPI definitions.
- **Complexity**: 4
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Profile outputs include time-to-completion distributions.
  - Report sections can reference this metric with claims.
- **Validation**:
  - Unit tests using synthetic start/end timestamps.

### Task 1.3: Enforce Plugin Ordering for Close‑Window Detection
- **Location**: `plugins/analysis_dynamic_close_window/*`, `plugins/*/plugin.yaml`
- **Description**: Ensure the dynamic close‑window detection plugin runs **first** among analysis plugins and exports a canonical close‑window artifact used by downstream plugins. Use `depends_on` to enforce ordering.
- **Complexity**: 5
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Dependency graph ensures close‑window plugin completes before other analysis plugins.
  - Downstream plugins read the dynamic + default windows.
- **Validation**:
  - Inspect plugin_executions order.
  - Confirm downstream artifacts reference both windows.

### Task 1.4: Standardize KPI Definition Across Reports and Plugins
- **Location**: `src/statistic_harness/core/report_v2_utils.py`, `plugins/analysis_*`
- **Description**: Define and reuse a single KPI helper for BP_OT_WTS_HOURS (threshold, population filters, busy‑period logic).
- **Complexity**: 3
- **Dependencies**: Task 1.3
- **Acceptance Criteria**:
  - KPI definition printed consistently in business/engineering summaries.
  - All plugins referencing KPI use identical thresholds + filters.
- **Validation**:
  - Compare KPI fields across artifacts for consistency.

## Sprint 2: Novel Recommendation Discovery (New Plugins)
**Goal**: Add **data‑driven** recommendation plugins that identify the top 10 actionable changes outside the excluded processes, with modeled hours saved and evidence.

**Demo/Validation**:
- New plugins produce recommendations not limited to known issues.
- At least one recommendation is novel relative to known‑issues list.
- Recommendations are deduped, ranked, and traced.

### Task 2.1: Plugin — Process Impact Counterfactuals
- **Location**: `plugins/analysis_process_counterfactuals/`
- **Description**: Compute per‑process marginal contribution to BP_OT_WTS_HOURS and model counterfactual savings if the process is shifted, throttled, or scheduled outside busy windows. Exclude targets in the blacklist, but allow them as context.
- **Complexity**: 7
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Produces top‑10 candidate actions with modeled hours saved.
  - Includes scope, assumptions, and measurement_type=modeled.
- **Validation**:
  - Unit test with synthetic data where a single process dominates wait.

### Task 2.2: Plugin — Process Dependency + Sequence Bottleneck Detection
- **Location**: `plugins/analysis_process_sequence_bottlenecks/`
- **Description**: Detect sequences where a downstream process consistently inflates wait‑to‑start following a specific upstream process (e.g., process A → process B). Quantify time impact and propose actionable change (schedule shift, batching, throttling).
- **Complexity**: 7
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Emits findings with effect size and confidence.
  - Generates recommendations with evidence + validation steps.
- **Validation**:
  - Test on synthetic sequence with known delay pattern.

### Task 2.3: Plugin — User/Host Cohort Savings Analysis
- **Location**: `plugins/analysis_user_host_savings/`
- **Description**: Identify user or host cohorts with systematically higher wait‑to‑start, and model savings by redistributing work across under‑utilized hosts/users.
- **Complexity**: 6
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Produces ranked cohort‑level recommendations.
  - Reports modeled hours saved with assumptions.
- **Validation**:
  - Test using synthetic dataset with known skew.

### Task 2.4: Recommendation Integration + Deduplication
- **Location**: `plugins/analysis_recommendation_dedupe_v2/`
- **Description**: Extend dedupe logic to merge new recommendation sources and compute top‑10 final list after exclusions.
- **Complexity**: 4
- **Dependencies**: Tasks 2.1–2.3
- **Acceptance Criteria**:
  - Deduped recommendations include at most 10 items.
  - Excluded targets never appear as primary action.
- **Validation**:
  - Unit tests for dedupe collisions and exclusions.

### Task 2.5: Recommendation Novelty Guard
- **Location**: `plugins/analysis_recommendation_dedupe_v2/`, `plugins/analysis_issue_cards_v2/`
- **Description**: Enforce that final recommendations are not duplicates of known‑issues actions (text + delta signature). If the top‑10 list is entirely known‑issues, fail fast with a specific error.
- **Complexity**: 4
- **Dependencies**: Task 2.4
- **Acceptance Criteria**:
  - Known‑issue recommendations are excluded or downgraded.
  - Fail‑fast if all recommendations are known issues.
- **Validation**:
  - Unit test with synthetic known‑issues overlap.

## Sprint 3: Traceability + Answer Generation
**Goal**: Ensure every recommendation has traceable evidence and produce a definitive answer to the known‑issue question.

**Demo/Validation**:
- `business_summary.md` includes top‑10 recommendations with modeled hours saved.
- `engineering_summary.md` includes evidence pointers and assumptions.
- `traceability_manifest.json` maps every printed number to a claim.

### Task 3.1: Extend Traceability Claims for Recommendations
- **Location**: `plugins/analysis_traceability_manifest_v2/`
- **Description**: Add claim objects for each recommendation, including data source, artifact path, and modeling assumptions.
- **Complexity**: 4
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - No number printed without a claim.
- **Validation**:
  - Fail-fast test when claim missing.

### Task 3.2: Update Decision Bundle Renderer
- **Location**: `plugins/report_decision_bundle_v2/`
- **Description**: Render the top‑10 recommendations section with explicit KPI definition and time‑savings table. Enforce exclusions and limit recommendations to novel findings.
- **Complexity**: 4
- **Dependencies**: Sprint 2–3.1
- **Acceptance Criteria**:
  - Output clearly answers the “largest time savings” question.
  - Recommendation list is novel vs known‑issues list.
- **Validation**:
  - Snapshot test with controlled fixtures.

### Task 3.3: Evidence‑First Answer Output
- **Location**: `plugins/analysis_issue_cards_v2/`, `plugins/analysis_waterfall_summary_v2/`
- **Description**: Ensure modeled fields (scope + assumptions) are enforced and that recommendations cite their artifact files.
- **Complexity**: 3
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - Schema validation fails if modeled fields are missing.
- **Validation**:
  - Unit tests for schema enforcement.

## Sprint 4: Full Gauntlet Execution + Known‑Issue Answer
**Goal**: Run the full dataset and provide the best‑shot answer to the known‑issue question.

**Demo/Validation**:
- Gauntlet run completes.
- Top‑10 recommendations appear in business_summary.md and are traceable.
- Answer extracted: “Best targeted change + modeled hours saved”.

### Task 4.1: Run Full Gauntlet
- **Location**: `scripts/run_gauntlet.ps1`
- **Description**: Execute the gauntlet with the test dataset and capture run outputs.
- **Complexity**: 2
- **Dependencies**: Sprints 1–3
- **Acceptance Criteria**:
  - Run completes without errors.
- **Validation**:
  - Inspect run status + report files.

### Task 4.2: Extract Final Answer
- **Location**: `business_summary.md`, `engineering_summary.md`
- **Description**: Identify the top recommendation and report modeled hours saved with supporting evidence.
- **Complexity**: 2
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - A single best‑shot answer is produced with citations to artifacts.
- **Validation**:
  - Traceability manifest contains claim entries for the recommendation.

## Testing Strategy
- Unit tests for each new plugin with synthetic fixtures.
- Integration test: full pipeline produces report artifacts.
- Regression tests for dedupe and exclusions.
- Determinism tests for recommendation ranking.

## Potential Risks & Gotchas
- **Runtime blowups** on 500k+ rows: mitigate with deterministic sampling + caps.
- **False positives**: require effect size thresholds and confidence weighting.
- **Exclusion leakage**: ensure excluded targets never appear as primary recommendation.
- **Missing claims**: enforce fail‑fast if any number lacks traceability.
- **Dynamic window mismatch**: ensure both default and dynamic windows are used.
- **Path quoting issues in WSL**: enforce robust quoting for filenames with spaces (PS1 + bash).
- **Parallel layer output noise**: suppress progress output for parallel analysis layers to avoid interleaving.
- **Time‑to‑completion ambiguity**: define start/end columns and fallback logic explicitly to avoid misinterpretation.

## Rollback Plan
- Revert new plugins and restore prior report bundle.
- Disable new recommendation plugins via planner allow/deny settings.
