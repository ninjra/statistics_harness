# Plan: Full Plugin Actionability Remediation

**Generated**: February 17, 2026  
**Estimated Complexity**: High

## Overview
Current release gates are green for execution integrity, but actionability coverage is incomplete: most analysis plugins run successfully yet do not surface actionable recommendations, and many findings use weak structure (`kind=""`) that blocks reliable routing.  
This plan remediates that by enforcing a strict actionability contract: every plugin must produce either:
- an actionable output, or
- a deterministic plain-English non-actionable explanation with reason, evidence, and next-step hint.

No plugin is allowed to silently produce non-actionable/no-result outputs.

## Scope and Success Criteria
- In scope:
  - Plugin output contract hardening.
  - Recommendation-routing expansion.
  - Per-plugin explainability fallback generation.
  - Run-level gating and observability for actionability.
- Out of scope:
  - External services or network-dependent enrichment.
- Success criteria:
  - `python -m pytest -q` passes.
  - Full gauntlet passes.
  - For each executable plugin in a run, one of:
    - actionable recommendation surfaced, or
    - plain-English non-actionable explanation surfaced.
  - Zero plugins with empty/opaque finding kinds in final actionable/explanation routing path.
  - Deterministic output under fixed `run_seed` for actionability routing and explanation text.

## Prerequisites
- Latest baseline run artifacts and state DB available.
- Existing gate scripts:
  - `scripts/run_final_validation_checklist.sh`
  - `scripts/build_final_validation_summary.py`
- Existing key modules:
  - `src/statistic_harness/core/pipeline.py`
  - `src/statistic_harness/core/report.py`
  - `src/statistic_harness/core/top20_plugins.py`
  - `src/statistic_harness/core/leftfield_top20/`

## Skills by Section
- Contract and planning sections: `plan-harder`, `ccpm-debugging`
  - Why: convert failure signals into deterministic contract and rollout plan.
- Output/routing implementation sections: `python-testing-patterns`, `testing`
  - Why: strict schema-first outputs and regression tests.
- Metrics/gates sections: `discover-observability`, `observability-engineer`, `python-observability`
  - Why: expose actionability regressions as hard, inspectable run metrics.
- Command execution sections: `shell-lint-ps-wsl`
  - Why: command correctness and cross-shell safety.

## Sprint 1: Define and Enforce Actionability Contract
**Goal**: Make “actionable or explained” a first-class, testable runtime contract.  
**Skills**: `plan-harder`, `ccpm-debugging`, `python-testing-patterns`  
**Demo/Validation**:
- Run contract unit tests.
- Verify run manifest contains explicit plugin actionability accounting.

### Task 1.1: Add Canonical Actionability Outcome Schema
- **Location**: `src/statistic_harness/core/types.py`, `docs/result_quality_contract.md`, `docs/report.schema.json`
- **Description**: Define canonical outcome fields:
  - `actionability_status`: `actionable | explained_non_actionable`
  - `actionability_reason_code`
  - `plain_english_explanation`
  - `evidence_summary`
  - `recommended_next_step` (nullable)
  - ownership model:
    - plugin-native may emit these fields directly;
    - pipeline synthesizer must backfill any missing fields deterministically from findings/error/debug.
- **Complexity**: 6
- **Dependencies**: None
- **Acceptance Criteria**:
  - Schema supports both actionable and explained outputs.
  - Missing actionability metadata fails validation.
- **Validation**: Unit tests for serialization/validation of both outcome types.

### Task 1.2: Enforce Contract in Pipeline Finalization
- **Location**: `src/statistic_harness/core/pipeline.py`
- **Description**: Extend result finalization to fail any plugin result that has neither actionable findings nor plain-English explanation payload.
- **Complexity**: 7
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - No plugin can end run with silent non-actionable output.
  - Existing legacy no-action diagnostic fallback is replaced or upgraded to compliant explanation payload.
- **Validation**: New tests for failure and pass paths.

### Task 1.4: Add Fail-Safe Explanation Synthesis for Malformed/Failed Plugin Results
- **Location**: `src/statistic_harness/core/pipeline.py`
- **Description**: Before contract hard-fail, synthesize deterministic explanation payload for malformed plugin outputs and execution failures so every plugin still yields an auditable outcome.
- **Complexity**: 7
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Plugin exceptions/malformed outputs become `explained_non_actionable` entries with reason/evidence/next-step text.
  - Contract failure reserved for truly unrecoverable serialization/pathology.
- **Validation**: Unit tests with synthetic malformed plugin results.

### Task 1.3: Add Run-Level Actionability Metrics
- **Location**: `src/statistic_harness/core/pipeline.py`, `scripts/build_final_validation_summary.py`
- **Description**: Emit counts/lists:
  - actionable plugins
  - explained_non_actionable plugins
  - contract violations
- **Complexity**: 5
- **Dependencies**: Task 1.2
- **Acceptance Criteria**:
  - Summary JSON exposes all three counts and plugin lists.
- **Validation**: Summary builder tests + fixture run.

## Sprint 2: Normalize Finding Quality and Remove Opaque Outputs
**Goal**: Eliminate structurally weak findings that block routing (`kind=""` and equivalent).  
**Skills**: `ccpm-debugging`, `python-testing-patterns`, `testing`  
**Demo/Validation**:
- Run a linter/test that fails on blank finding kinds.

### Task 2.1: Build Finding-Quality Linter
- **Location**: `scripts/validate_plugin_finding_quality.py`, `tests/`
- **Description**: Add deterministic checks:
  - no blank `kind`
  - mandatory fields for actionability/explanation families
- **Complexity**: 6
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Script emits plugin-level violations with fix hints.
  - CI/test gate fails on violations.
- **Validation**: Unit tests for known bad and known good fixtures.

### Task 2.2: Patch High-Volume Opaque Plugins First
- **Location**: affected plugin files from linter output (starting with control-chart/drift/anomaly families)
- **Description**: Replace blank `kind` with explicit, stable taxonomies and ensure each finding maps to actionability or explanation route.
- **Complexity**: 8
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Top offending plugins produce no blank `kind`.
- **Validation**: Plugin unit tests + linter pass.

### Task 2.3: Complete Repo-Wide Finding Normalization
- **Location**: all remaining violating plugins
- **Description**: Finish linter-driven fixes for every plugin flagged.
- **Complexity**: 9
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - `blank_kind_findings_total == 0` on baseline run.
- **Validation**: Full gauntlet + quality script.

## Sprint 3: Expand Recommendation Routing Beyond Current 4 Plugins
**Goal**: Route actionable findings from all capable plugins into final recommendations.  
**Skills**: `python-testing-patterns`, `testing`, `ccpm-debugging`  
**Demo/Validation**:
- `answers_summary.json` includes actionable contributions from broad plugin families (not only current 4).

### Task 3.1: Add Generic Recommendation Adapter Layer
- **Location**: `src/statistic_harness/core/report.py`
- **Description**: Implement adapter registry for common finding families:
  - anomaly, changepoint, drift, bottleneck, capacity, dependency, sequence, leftfield signals.
- **Complexity**: 8
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - Findings with recognized families produce recommendation records with deterministic structure.
- **Validation**: Adapter unit tests by family.

### Task 3.2: Add Plugin-Specific Adapters for Top20 and Leftfield
- **Location**: `src/statistic_harness/core/report.py`, `src/statistic_harness/core/top20_plugins.py`, `src/statistic_harness/core/leftfield_top20/`
- **Description**: Define explicit recommendation/explanation mappers for top20 + leftfield plugins so each plugin is represented in final output.
- **Complexity**: 9
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Top20 plugins: each yields recommendation or explanation entry.
  - Leftfield plugins: each yields recommendation or explanation entry.
- **Validation**: Targeted integration tests on baseline run fixture.

### Task 3.3: Add De-duplication and Ranking Policy for Expanded Set
- **Location**: `src/statistic_harness/core/report.py`
- **Description**: Preserve relevance while scaling plugin coverage:
  - stronger dedupe keys
  - score tie-breakers
  - retain per-plugin representation in explanation lane even if deduped from top recommendations.
  - keep ranking deterministic under fixed `run_seed`.
- **Complexity**: 7
- **Dependencies**: Task 3.2
- **Acceptance Criteria**:
  - No plugin disappears entirely from final surfaced outputs.
- **Validation**: Snapshot tests over `answers_summary.json`.

## Sprint 4: Plain-English Explanation Lane for Non-Actionable Results
**Goal**: Every non-actionable plugin yields human-useful, deterministic explanation text.  
**Skills**: `python-testing-patterns`, `testing`  
**Demo/Validation**:
- Final outputs include explanation entries for all non-actionable plugins.

### Task 4.1: Add Explanation Generator Library
- **Location**: `src/statistic_harness/core/report_v2_utils.py` or new `src/statistic_harness/core/actionability_explanations.py`
- **Description**: Build deterministic template-based plain-English explanations using:
  - plugin type
  - reason code
  - key evidence metrics
  - recommended next data/action step
  - fixed template family per reason code + plugin type (seed-stable ordering/selection)
- **Complexity**: 7
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Explanations are non-empty, plain-English, and specific.
- **Validation**: Unit tests for template coverage.

### Task 4.2: Surface Explanation Lane in Outputs
- **Location**: `src/statistic_harness/core/report.py`, `appdata/runs/*` output writers
- **Description**: Add explicit section in `answers_summary.json`, `answers_recommendations.md`, and plain report for `explained_non_actionable`.
- **Complexity**: 6
- **Dependencies**: Task 4.1, Sprint 3
- **Acceptance Criteria**:
  - Every non-actionable plugin appears in explanation lane with reason/evidence/next step.
- **Validation**: End-to-end report snapshot tests.

### Task 4.4: Add Per-Plugin Explanation Coverage Test
- **Location**: `tests/`, `src/statistic_harness/core/report.py`
- **Description**: Add integration test asserting every non-actionable plugin in a fixture run has a corresponding explanation entry in final outputs.
- **Complexity**: 6
- **Dependencies**: Tasks 4.1-4.2
- **Acceptance Criteria**:
  - Test fails if any non-actionable plugin is missing explanation output.
- **Validation**: New integration fixture with mixed actionable/non-actionable plugins.

### Task 4.3: Replace/Retire Legacy Diagnostic Placeholder
- **Location**: `src/statistic_harness/core/pipeline.py`
- **Description**: Remove generic “no actionable signal” fallback as final state; require explanation object with structured fields.
- **Complexity**: 5
- **Dependencies**: Tasks 4.1-4.2
- **Acceptance Criteria**:
  - Zero legacy placeholder-only results in baseline run.
- **Validation**: Contract tests + run summary assertions.

## Sprint 5: Observability and Hard Release Gates
**Goal**: Make actionability regressions impossible to miss and impossible to ship.  
**Skills**: `discover-observability`, `observability-engineer`, `python-observability`, `testing`  
**Demo/Validation**:
- Final checklist fails immediately for actionability contract violations.

### Task 5.1: Add Actionability SLI Metrics
- **Location**: `src/statistic_harness/core/pipeline.py`, `scripts/build_final_validation_summary.py`
- **Description**:
  - `actionable_plugin_rate`
  - `explained_non_actionable_rate`
  - `unexplained_plugin_count`
  - `blank_kind_findings_count`
- **Complexity**: 6
- **Dependencies**: Sprints 1-4
- **Acceptance Criteria**:
  - Metrics present in summary and run manifest.
- **Validation**: Metric extraction tests.

### Task 5.2: Enforce Checklist Gates
- **Location**: `scripts/run_final_validation_checklist.sh`
- **Description**: Add hard fail conditions:
  - unexplained plugin count > 0
  - blank kind count > 0
  - missing explanation lane entries for non-actionable plugins
  - staged rollout flags:
    - warn-only in first pass,
    - hard-fail after baseline + synthetic certification.
- **Complexity**: 5
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - Checklist exits non-zero on any contract breach.
- **Validation**: Synthetic failing fixtures.

### Task 5.3: Add Plugin-Level Actionability Diff Report
- **Location**: new script `scripts/compare_plugin_actionability_runs.py`
- **Description**: Compare before/after run coverage:
  - actionable vs explained counts per plugin
  - recommendation surfaced changes
  - regressions/newly fixed status
- **Complexity**: 7
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - Deterministic diff output consumable for release evidence.
- **Validation**: Golden snapshot tests.
  - plus synthetic diff-case tests with known expected deltas.

## Sprint 6: Full Validation and Release Evidence
**Goal**: Prove the repo is complete against your acceptance standard on baseline + synthetic datasets.  
**Skills**: `testing`, `python-testing-patterns`, `observability-engineer`  
**Demo/Validation**:
- Full gauntlet runs with complete plugin representation.

### Task 6.1: Baseline Full Gauntlet Certification
- **Location**: runtime + `appdata/final_validation/*`
- **Description**: Run baseline dataset through full validation and confirm:
  - every plugin actionable or explained
  - no opaque results
- **Complexity**: 6
- **Dependencies**: Sprints 1-5
- **Acceptance Criteria**:
  - Certification artifact produced.
- **Validation**: checklist + summary + actionability diff.

### Task 6.2: Synthetic Dataset Certification
- **Location**: same as Task 6.1 (for synthetic inputs)
- **Description**: Repeat certification on both synthetic datasets, ensuring synthetic/system-derived markers are preserved in explanation/recommendation outputs.
- **Complexity**: 6
- **Dependencies**: Task 6.1
- **Acceptance Criteria**:
  - Same contract pass on synthetic runs.
- **Validation**: checklist + summary + per-dataset diff.

### Task 6.3: Publish Consolidated Evidence
- **Location**: `docs/release_evidence/`
- **Description**: Add one consolidated report:
  - plugin actionability coverage table
  - actionable recommendation coverage by plugin family
  - non-actionable explanation quality audit
- **Complexity**: 5
- **Dependencies**: Tasks 6.1-6.2
- **Acceptance Criteria**:
  - Single source of truth for release decision.
- **Validation**: schema/check script over evidence bundle.

## Testing Strategy
- Unit:
  - Contract validation and reason-code completeness.
  - Explanation generator deterministic outputs.
  - Recommendation adapter mappings.
- Integration:
  - Pipeline finalization actionability enforcement.
  - Report generation includes both actionable and explanation lanes.
- End-to-end:
  - `python -m pytest -q`
  - `scripts/run_final_validation_checklist.sh <baseline_dataset_version_id> 1337 10`
- Regression:
  - Run-to-run actionability diff snapshots.
  - Determinism checks: repeated fixed-seed runs produce identical actionability/explanation coverage.

## Potential Risks & Gotchas
- Risk: overfitting adapters to one dataset.
  - Mitigation: enforce generic family adapters + dataset-agnostic tests.
- Risk: recommendation bloat after routing expansion.
  - Mitigation: strict ranking + dedupe policy with guaranteed plugin representation in explanation lane.
- Risk: false “actionable” classification from weak heuristics.
  - Mitigation: explicit contract fields and required evidence payloads.
- Risk: plugin performance regression.
  - Mitigation: preserve caching and add runtime trend checks in evidence.

## Rollback Plan
- Keep changes behind incremental contract flags for one sprint.
- If routing expansion destabilizes outputs:
  - revert adapter layer commit(s),
  - keep contract+explanation gate intact,
  - fall back to explanation lane only until adapters are fixed.
- Use actionability diff script to verify rollback impact before merge.
