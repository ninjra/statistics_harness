# Codex Implementation Spec: Decision Report v2 + Slide Kit Artifacts

## 1. Objective

Transform the current “Statistic Harness Report” (plugin-dump oriented) into a decision-oriented output that is:
1. Readable by non-technical stakeholders (business_summary.md).
2. Technically sufficient for engineers to validate and reproduce claims (engineering_summary.md).
3. Fully inspectable and complete (appendix_raw.md retains raw plugin dumps).
4. Mechanically convertible into a PPTX (slide_kit/ CSV/JSON bundle).

This spec defines new plugins and report-renderer functionality to implement the following features:
1. Issue Cards (per known issue): metric definition, baseline, target, observed, PASS/FAIL reason, artifact pointers.
2. Waterfall Summary: total busy-period over-threshold wait → top driver (e.g., QEMAIL) share → remainder → modeled deltas.
3. Decision vs Appendix split with relevance scoring and strict row limits in summaries.
4. Recommendation deduplication/merging (identical action+delta must merge).
5. Model field standardization (scale_factor conventions) + schema validation (fail-fast).
6. Traceability manifest mapping each summary claim to source plugin/kind/artifact/query.
7. Slide Kit artifact emission (CSV tables usable to recreate the PPTX tables).

Hard requirement: ANY_REGRESS => DO_NOT_SHIP.
If any regression guardrail triggers (see Section 9), the build must fail.

## 2. Inputs and Assumptions

### 2.1 Inputs
1. Normalized dataset (existing harness input) with timestamp columns (e.g., START_DT, END_DT), PROCESS_ID, LOCAL_MACHINE_ID, plus wait-related columns used by existing plugins.
2. Existing plugin artifacts (already produced by current harness), especially:
   1. analysis_queue_delay_decomposition/results.json
   2. analysis_close_cycle_capacity_model/results.json
   3. analysis_close_cycle_capacity_impact/results.json
   4. analysis_concurrency_reconstruction/concurrency_summary.json
   5. analysis_percentile_analysis/percentiles.json

### 2.2 “Primary KPI” Definition Requirement
The report MUST pick one primary KPI and define it explicitly in the summaries:
1. Name: Busy-period over-threshold wait-to-start hours (BP_OT_WTS_HOURS)
2. Unit: hours
3. Population: runs included after scope filters (e.g., erp_type=Quorum).
4. Threshold: configurable; default 60 seconds (over-threshold means max(wait_to_start_sec - threshold_sec, 0)).
5. Busy period definition: contiguous union of “over-threshold waiting intervals” (see Section 5).

If the harness already defines “eligible_wait_gt_hours_total” in analysis_queue_delay_decomposition, the plugin must state:
1. The relationship between eligible_wait_gt_hours_total and BP_OT_WTS_HOURS (identical vs derived).
2. The exact threshold and eligible basis (e.g., eligible_basis: queue).

If the exact definitions differ from the above defaults, the summaries must print the true definitions.

## 3. Output Deliverables

### 3.1 New Report Outputs
The report bundle v2 must produce:
1. business_summary.md (<= 2 pages equivalent)
2. engineering_summary.md
3. appendix_raw.md

### 3.2 Slide Kit Artifacts
Emit a folder slide_kit/ containing:
1. slide_kit/scenario_summary.csv
2. slide_kit/waterfall_summary.csv
3. slide_kit/busy_periods.csv
4. slide_kit/top_process_contributors.csv
5. slide_kit/issue_cards.json
6. slide_kit/traceability_manifest.json

All CSVs must have stable column order, stable row sorting, and versioned headers.

### 3.3 Artifact Manifest (required)
Emit slide_kit/artifacts_manifest.json listing all generated files with:
1. path
2. sha256
3. schema_version
4. source_plugins (list)
5. created_at_utc

## 4. New Plugin List (and Responsibilities)

Implement the following new plugins (or functionality modules if the harness doesn’t require strict plugin form). Each plugin must:
1. Write a results.json with schema_version.
2. Write CSVs where specified.
3. Return a small “findings” list for summaries (max 20).

### 4.1 analysis_issue_cards_v2 (NEW)
Purpose: Convert known issues + checks into slide-ready “Issue Cards”.

Inputs:
1. Known issues config (existing harness checks list).
2. Check evaluation results (PASS/FAIL).
3. Underlying metric values (must be printed even when FAIL).

Outputs:
1. slide_kit/issue_cards.json
2. artifacts/analysis_issue_cards_v2/issue_cards.md (optional human-readable)

Each Issue Card MUST include:
1. issue_id (stable)
2. title
3. scope (filters applied)
4. metric_name
5. metric_definition (plain-English + formula)
6. unit
7. baseline_value
8. target_expression (e.g., modeled_delta_hours >= 4.0)
9. observed_value
10. decision (PASS/FAIL)
11. failure_reason (required when FAIL)
12. evidence:
   1. plugin
   2. kind
   3. measurement_type
   4. artifact_path
   5. query_or_grouping
13. recommended_actions (list; ties to deduped recs)

Fail-fast: If any FAIL check does not include predicate + computed values, fail the run.

### 4.2 analysis_busy_period_segmentation_v2 (NEW)
Purpose: Make “busy periods” a first-class artifact (not implied).

Algorithm:
1. For each run, compute over_threshold_wait_sec = max(wait_to_start_sec - threshold_sec, 0).
2. Define a waiting interval for each run with over_threshold_wait_sec > 0:
   1. interval_start = eligible_start_ts (or the timestamp that marks when it became eligible/queued)
   2. interval_end = interval_start + over_threshold_wait_sec
3. Busy periods are connected components of the union of these intervals, merging gaps <= gap_tolerance_sec (default 60 sec).
4. For each busy period:
   1. start_ts, end_ts, duration_sec
   2. total_over_threshold_wait_sec (sum of per-run over_threshold_wait_sec within the busy period)
   3. runs_over_threshold_count
   4. top_process_by_wait (process_id + wait_sec)
   5. per_process_over_threshold_wait_sec (top N=10 only in CSV; full in JSON)
   6. per_host_over_threshold_wait_sec (top N=10 only in CSV; full in JSON)

Outputs:
1. slide_kit/busy_periods.csv (top 50 by total_over_threshold_wait_sec)
2. artifacts/analysis_busy_period_segmentation_v2/busy_periods.json (full)
3. artifacts/analysis_busy_period_segmentation_v2/definition.json (prints threshold, eligible timestamp selection, and gap tolerance)

Sorting:
1. Primary: total_over_threshold_wait_sec desc
2. Secondary: start_ts asc
3. Tertiary: busy_period_id asc

### 4.3 analysis_waterfall_summary_v2 (NEW)
Purpose: Explicitly render the implied waterfall that currently exists only implicitly.

Waterfall rows (minimum):
1. total_bp_over_threshold_wait_hours (measured)
2. top_driver_over_threshold_wait_hours (measured) and top_driver_id (e.g., qemail)
3. remainder_without_top_driver_hours = total - top_driver
4. modeled_remainder_after_capacity_hours (modeled) for scenario “add one server” (must declare baseline population)
5. modeled_remainder_after_both_hours (modeled) if scenario is available (remove top driver + add one server)
6. deltas for each modeled row

Rules:
1. All arithmetic must be performed in seconds internally, converted to hours at render time.
2. Rounding: hours printed to 2 decimals; seconds kept full precision in JSON.
3. If a modeled baseline does not exactly match remainder_without_top_driver, the summary must print:
   1. baseline_source (which plugin)
   2. baseline_population (what’s excluded/included)
   3. reconciliation_diff_hours

Outputs:
1. slide_kit/waterfall_summary.csv
2. artifacts/analysis_waterfall_summary_v2/waterfall_summary.json

### 4.4 analysis_recommendation_dedupe_v2 (NEW)
Purpose: Merge repeated recommendations and attach evidence.

Inputs:
1. Existing recommendation list.
2. Modeled scenarios from analysis_waterfall_summary_v2.

Dedup key:
1. action_type (e.g., “add_server”, “remove_process”, “tune_schedule”)
2. target (e.g., “qemail”, “overall”)
3. scenario_id (e.g., “add_1_server”, “remove_qemail+add_1_server”)
4. delta_signature (rounded delta in seconds, not text)

Output:
1. artifacts/analysis_recommendation_dedupe_v2/recommendations.json
2. A reduced list for summaries (max 9), each containing:
   1. action
   2. expected_impact (hours, percent)
   3. evidence pointers (plugins/artifacts)
   4. confidence_tag (MEASURED / MODELED / MIXED)
   5. validation_steps (templated)

Fail-fast: If two recommendations share the dedup key but produce conflicting deltas, fail.

### 4.5 analysis_traceability_manifest_v2 (NEW)
Purpose: Every number printed in business_summary.md must have a machine link back to data.

Mechanism:
1. Define a “claim” object:
   1. claim_id (stable)
   2. label (MEASURED / MODELED / INFERENCE)
   3. summary_text
   4. value (typed)
   5. unit
   6. population_scope (filters)
   7. source:
      1. plugin
      2. kind
      3. measurement_type
      4. artifact_path
      5. query_or_grouping
      6. row_keys (if applicable)
   8. render_targets (business_summary, engineering_summary, slide_kit)
2. The renderer is not allowed to print a number unless a corresponding claim exists.

Outputs:
1. slide_kit/traceability_manifest.json
2. artifacts/analysis_traceability_manifest_v2/manifest.md (optional)

Fail-fast: Missing claim mapping for any printed number => fail.

### 4.6 report_decision_bundle_v2 (NEW)
Purpose: Create the three-tier report layout + slide kit references.

Outputs:
1. business_summary.md
2. engineering_summary.md
3. appendix_raw.md
4. slide_kit/ (from other plugins)

Rendering constraints:
1. business_summary.md:
   1. Must define KPI in 3 lines: name, unit, threshold, population.
   2. Must include scenario_summary table (max 6 rows).
   3. Must include top 10 busy periods (from busy_periods.csv).
   4. Must include top 5 process drivers (from top_process_contributors.csv).
   5. Must include up to 3 recommendations, with “How to validate” steps.
2. engineering_summary.md:
   1. Glossary of terms (eligible wait, threshold, busy period, close cycle).
   2. Issue Cards section (rendered from issue_cards.json).
   3. Check Explainer section: print predicate/value pairs.
   4. Traceability section: link each business claim to claim_id + artifact path.
3. appendix_raw.md:
   1. Raw plugin dumps may be included, but high-volume lists MUST be collapsed:
      1. Print count + top N=10 examples
      2. Provide artifact pointer for full JSON

### 4.7 report_slide_kit_emitter_v2 (NEW)
Purpose: Materialize slide tables used in a deck.

Outputs:
1. slide_kit/scenario_summary.csv
2. slide_kit/top_process_contributors.csv

scenario_summary.csv required columns:
1. scenario_id
2. scenario_name
3. evidence_type (MEASURED/MODELED/MIXED)
4. bp_over_threshold_wait_hours
5. delta_hours_vs_current
6. delta_percent_vs_current
7. notes (plain English; max 120 chars)
8. claim_id (traceability)

top_process_contributors.csv required columns:
1. process_id
2. process_name_normalized
3. bp_over_threshold_wait_hours
4. share_percent
5. close_cycle_slowdown_ratio (if available)
6. claim_id

## 5. Core Definitions and Formatting (must be printed in summaries)

### 5.1 Over-threshold wait
over_threshold_wait_sec = max(wait_to_start_sec - threshold_sec, 0)

### 5.2 Busy period
Busy period is a time interval representing continuous user-visible slowness:
1. Construct waiting intervals from over-threshold runs.
2. Merge intervals with gaps <= gap_tolerance_sec.
3. The merged interval is a busy period.

The report must print:
1. threshold_sec
2. gap_tolerance_sec
3. the timestamp column(s) used

### 5.3 Close cycle window
If using close-cycle start/end day logic, print:
1. close_cycle_start_day
2. close_cycle_end_day
3. timezone assumptions
4. which timestamps define “close” vs “open”

## 6. Standardization and Schema Validation

### 6.1 scale_factor convention (single standard)
Adopt: scale_factor = new_host_count / baseline_host_count
Example: 2 -> 3 hosts => scale_factor = 1.5

If any existing plugin uses inverse scaling, implement conversion and store both:
1. scale_factor_standard
2. scale_factor_original
3. scale_factor_original_definition (string)

### 6.2 Modeled finding required fields
Any finding with measurement_type == “modeled” must include:
1. modeled_assumptions (list)
2. modeled_scope (filters + population definition)
3. baseline_host_count
4. modeled_host_count
5. baseline_value
6. modeled_value
7. delta_value
8. unit

Fail-fast: missing any required field => fail.

### 6.3 Redaction requirements
Before writing business_summary.md and slide_kit CSVs:
1. Remove or hash:
   1. hostnames (LOCAL_MACHINE_ID values)
   2. user identifiers
   3. any free-text parameter strings that can contain sensitive data
2. Keep stable pseudonyms if needed:
   1. host_01, host_02, …

Fail-fast: if forbidden columns appear in slide_kit outputs => fail.

## 7. Check Runner Enhancements (FAIL must be actionable)

Modify the check runner output to always emit:
1. check_id
2. description
3. predicate_text (human readable)
4. computed_values (numbers with units)
5. target_values (numbers with units)
6. decision (PASS/FAIL)
7. failure_reason when FAIL
8. evidence pointers (artifact paths)

Engineering summary must include a “Checks” section that prints this.

## 8. Testing and Regression Guardrails

### 8.1 Deterministic output rules
1. Stable sorting for all tables.
2. Stable rounding rules (hours to 2 decimals, seconds retained in JSON).
3. Avoid nondeterministic iteration over dicts/sets.

### 8.2 Unit tests (required)
1. Busy period segmentation:
   1. Known fixture with intervals → expected busy period boundaries.
2. Waterfall math:
   1. total == top_driver + remainder (within 1 second tolerance).
3. Recommendation dedupe:
   1. Duplicate inputs → single output; conflicting deltas must fail.
4. Traceability:
   1. Attempt to render summary with missing claim → must fail.

### 8.3 Integration tests (required)
1. Golden dataset fixture run produces identical:
   1. business_summary.md
   2. slide_kit/scenario_summary.csv
   3. issue_cards.json
2. If any KPI changes by > configured tolerance without an explicit “expected change” update, fail.

### 8.4 ANY_REGRESS => DO_NOT_SHIP guardrails
Fail the run if:
1. business_summary.md prints a number without claim_id mapping.
2. waterfall reconciliation diff exceeds tolerance without explanation.
3. modeled findings are missing required fields.
4. slide_kit outputs contain forbidden columns.
5. recommendation dedupe detects conflicting deltas under the same key.

## 9. Definition of Done

Done when:
1. business_summary.md is <= 2 pages equivalent and includes:
   1. KPI definition
   2. scenario_summary table
   3. top busy periods
   4. top process drivers
   5. <= 3 recommendations with validation steps
2. engineering_summary.md includes:
   1. Issue Cards with PASS/FAIL predicates and values
   2. traceability table and claim ids
3. appendix_raw.md retains full detail without overwhelming summaries.
4. slide_kit/ contains stable CSVs that can recreate deck tables.
5. All tests pass and guardrails enforce fail-fast behavior.
