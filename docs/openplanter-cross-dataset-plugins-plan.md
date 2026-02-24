# Plan: OpenPlanter Cross-Dataset Plugins Pack

**Generated**: 2026-02-23  
**Estimated Complexity**: High

## Overview
Implement a first-class cross-dataset plugin pack that mirrors the OpenPlanter-inspired spec in `docs/statistics_harness_openplanter_cross_dataset_plugins.md` while preserving this repo's non-negotiables: streaming-first execution, deterministic outputs, fail-closed behavior, no runtime network access, and evidence-traceable findings.

Approach:
1. Build shared core utilities first (entity resolution, evidence links, SQL dump import).
2. Add the 9 plugins in contract-first order (ingest -> transform -> analysis -> report).
3. Enforce deterministic evidence schemas and artifact registration.
4. Add targeted unit/integration/fail-closed tests and resource/determinism gates.
5. Ship with a small repeatable fixture pack and operator runbook.

## Global Hard Gates
- No plugin in this pack may return a silent non-result for missing prerequisites. Missing required config/fields/datasets must return `status="error"` with explicit `error.type` and actionable `error.message`.
- Pipeline fail-closed behavior must be preserved: if one plugin errors, pipeline still completes and writes both `report.json` and `report.md` with error summaries.
- Runtime network access remains prohibited and must be validated by tests (not only manifest declaration).
- Determinism requires seeded RNG utilities and end-to-end stable outputs for same input/settings/run_seed.

## Prerequisites
- Python 3.11 venv and existing repo dev dependencies installed.
- Existing plugin contracts and schemas:
  - `docs/plugin_manifest.schema.json`
  - `docs/run_manifest.schema.json`
  - `docs/report.schema.json`
- Existing streaming interfaces and helpers:
  - `src/statistic_harness/core/dataset_io.py`
  - `src/statistic_harness/core/report_v2_utils.py`
  - `src/statistic_harness/core/pipeline.py`
- Test harness availability:
  - `pytest`
  - synthetic fixture datasets under `tests/fixtures/`

## Skill Usage Strategy
- `plan-harder`: structure full phased implementation and committable work breakdown.
- `config-matrix-validator`: define and validate role/field mapping contracts across dataset versions.
- `python-testing-patterns` + `testing`: design deterministic unit + integration + fail-closed suites.
- `discover-observability` + `python-observability` + `observability-engineer`: add runtime counters/health telemetry for plugin quality and performance.
- `resource-budget-enforcer`: verify streaming/memory budgets on large synthetic inputs.
- `deterministic-tests-marshal`: prove deterministic seeds and stable IDs for repeated runs.
- `golden-answer-harness`: establish regression-safe expected outputs for core cross-dataset insights.

## Sprint 1: Foundation Contracts and Scaffolding
**Goal**: Establish stable contracts and skeletons before algorithm work.  
**Skill Usage**: `plan-harder` for decomposition, `config-matrix-validator` for mapping schemas.  
**Demo/Validation**:
- All 9 plugin folders exist with valid manifests/schemas.
- Schema validation passes for each plugin manifest and config schema.

### Task 1.1: Create Shared Contract Doc Addendum
- **Location**: `docs/statistics_harness_openplanter_cross_dataset_plugins.md`, `docs/plugin_dev_guide.md`
- **Description**: Add implementation notes clarifying dataset role mapping, evidence row refs, stable IDs, and deterministic seeds.
- **Complexity**: 3/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - Mapping model and fail-closed expectations are explicit and unambiguous.
  - Determinism and artifact registration requirements are explicitly listed.
- **Validation**:
  - Doc review checklist completed.

### Task 1.2: Scaffold Plugin Directories and Base Schemas
- **Location**: `plugins/ingest_sql_dump_v1/`, `plugins/transform_entity_resolution_map_v1/`, `plugins/transform_cross_dataset_link_graph_v1/`, `plugins/analysis_bundled_donations_v1/`, `plugins/analysis_contribution_limit_flags_v1/`, `plugins/analysis_vendor_influence_breadth_v1/`, `plugins/analysis_vendor_politician_timing_permutation_v1/`, `plugins/analysis_red_flags_refined_v1/`, `plugins/report_evidence_index_v1/`
- **Description**: Add `plugin.yaml`, `config.schema.json`, `output.schema.json`, `plugin.py`, `__init__.py` skeletons using repo conventions.
- **Complexity**: 4/10
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Each plugin declares `sandbox.no_network: true` and includes `run_dir` in `fs_allowlist`.
  - Each plugin has required output keys (`status/summary/metrics/findings/artifacts/budget/error/references/debug`).
- **Validation**:
  - Plugin registry load test passes.
  - Schema checks pass for every new plugin.

### Task 1.3: Add Cross-Dataset Config Matrix Template
- **Location**: `docs/schemas/cross_dataset_role_map.schema.json`, `docs/plugin_data_access_overrides.json`
- **Description**: Define a reusable dataset-role/field mapping schema and example config payloads.
- **Complexity**: 5/10
- **Dependencies**: Task 1.2
- **Acceptance Criteria**:
  - Supports explicit role to dataset_version_id mapping.
  - Supports field-level mappings and conditional filters (`when` blocks).
- **Validation**:
  - Config examples validate against schema.

### Task 1.4: Add Golden Truth Tables Up Front
- **Location**: `tests/fixtures/openplanter_pack/ground_truth.yaml`
- **Description**: Define expected entities, links, bundles, limit flags, timing pairs, and negative controls before plugin implementation.
- **Complexity**: 4/10
- **Dependencies**: Task 1.3
- **Acceptance Criteria**:
  - Truth table contains explicit expected IDs/counts for at least one positive and one negative case per plugin.
- **Validation**:
  - Fixture lint and schema checks pass.

## Sprint 2: Shared Core Libraries
**Goal**: Implement common deterministic logic once and reuse across plugins.  
**Skill Usage**: `plan-harder`, `resource-budget-enforcer`, `deterministic-tests-marshal`.  
**Demo/Validation**:
- Core modules compile and unit tests pass.
- Determinism tests show stable hashes/IDs/output ordering.

### Task 2.1: Implement Entity Resolution Core
- **Location**: `src/statistic_harness/core/entity_resolution.py`
- **Description**: Add normalization, aggressive token normalization, tokenization, inverted index builder, deterministic candidate ranking, optional rapidfuzz path with deterministic fallback.
- **Complexity**: 8/10
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Normalization behavior matches documented suffix/punctuation rules.
  - Tie-breaking is stable and deterministic.
- **Validation**:
  - `tests/core/test_entity_resolution.py`
  - Repeatability test over shuffled input order.

### Task 2.2: Implement Evidence Link Helpers
- **Location**: `src/statistic_harness/core/evidence_links.py`
- **Description**: Add row-ref formatter, evidence link constructor, artifact registration helper with sha256 and count metadata.
- **Complexity**: 6/10
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Row refs use `db://{dataset_version_id}#row_index={row_index}`.
  - Evidence links contain deterministic IDs and traceable metadata.
- **Validation**:
  - `tests/core/test_evidence_links.py`

### Task 2.3: Implement Safe SQL Dump Importer
- **Location**: `src/statistic_harness/core/sql_dump_import.py`
- **Description**: Add streaming SQL parser with allowlisted statements and parameterized insert execution; reject unsafe/dialect-incompatible forms.
- **Complexity**: 9/10
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Supports large statement streams with chunk flush.
  - Rejects `ATTACH`, `PRAGMA`, `DROP`, `ALTER`, `VACUUM`, triggers/views.
- **Validation**:
  - `tests/core/test_sql_dump_import.py`
  - fail-closed tests for unsafe statements.

### Task 2.4: Add Seeded RNG Utility and Enforcement
- **Location**: `src/statistic_harness/core/stat_controls.py`, `src/statistic_harness/core/utils.py`, `tests/core/test_seed_propagation.py`
- **Description**: Add a shared helper for deriving plugin-local RNG seeds from `run_seed` + stable plugin scope; prohibit unseeded/global RNG usage in new plugins.
- **Complexity**: 7/10
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Same run inputs produce byte-identical random draws for timing permutations.
  - Static checks fail when unseeded RNG calls are introduced in cross-dataset pack code.
- **Validation**:
  - Seed propagation tests and deterministic rerun checks pass.

## Sprint 3: Ingest Plugin (SQL Dump)
**Goal**: Deliver production-grade SQL dump ingest with fail-closed behavior.  
**Skill Usage**: `resource-budget-enforcer`, `python-testing-patterns`.  
**Demo/Validation**:
- `ingest_sql_dump_v1` imports synthetic 11M-row SQL dumps without OOM.
- Manifest artifact includes row counts and statement stats.
- Missing required ingest config returns `status="error"` (never skipped/no-op).

### Task 3.1: Implement ingest_sql_dump_v1 Runtime
- **Location**: `plugins/ingest_sql_dump_v1/plugin.py`
- **Description**: Wire `sql_dump_import.py` into plugin runtime; update dataset/table metadata and deterministic row ordering.
- **Complexity**: 8/10
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - Deterministic table population and metadata updates.
  - Actionable error when unsupported SQL dialect is detected.
  - Unsafe/missing inputs return `status="error"` with stable error code.
- **Validation**:
  - Plugin unit tests and one integration test with synthetic dump.

### Task 3.2: Add Ingest Artifacts and Contracts
- **Location**: `plugins/ingest_sql_dump_v1/output.schema.json`, `plugins/ingest_sql_dump_v1/config.schema.json`
- **Description**: Include import manifest artifact and optional canonical export controls.
- **Complexity**: 4/10
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Artifact metadata includes hash, table stats, and ingest counters.
- **Validation**:
  - Schema validation and artifact existence test.

## Sprint 4: Transform Plugins (Entity Map + Link Graph)
**Goal**: Build deterministic cross-dataset linkage substrate with evidence chain.  
**Skill Usage**: `config-matrix-validator`, `deterministic-tests-marshal`, `python-testing-patterns`.  
**Demo/Validation**:
- Entity map and cross-link graph generated from fixture datasets.
- Links include `match_type`, `confidence_tier`, features, and row refs.
- Missing mappings/datasets return `status="error"` (never skipped/no-op).

### Task 4.1: Implement transform_entity_resolution_map_v1
- **Location**: `plugins/transform_entity_resolution_map_v1/plugin.py`
- **Description**: Stream configured roles/fields across dataset_version_ids, normalize aliases, emit canonical entity map and flat alias table.
- **Complexity**: 8/10
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - Stable `entity_id` creation and alias provenance mapping.
  - Handles missing roles/fields with fail-closed actionable errors.
  - Match output counts are deterministic across reruns with shuffled source row order.
- **Validation**:
  - `tests/plugins/test_transform_entity_resolution_map_v1.py`

### Task 4.2: Implement transform_cross_dataset_link_graph_v1
- **Location**: `plugins/transform_cross_dataset_link_graph_v1/plugin.py`
- **Description**: Build left-side indexes, stream right-side rows, perform exact/fuzzy/token-overlap matching, emit edge list and summaries.
- **Complexity**: 9/10
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Stable `edge_id` and deterministic ranking/tie-breakers.
  - Supports JSONL/CSV parity and confidence breakdown summary.
  - Parity check enforces identical record count between `cross_links.csv` and `cross_links.jsonl`.
- **Validation**:
  - `tests/plugins/test_transform_cross_dataset_link_graph_v1.py`
  - deterministic rerun equality test.

## Sprint 5: Analysis Plugins (5-plugin Pack)
**Goal**: Deliver actionable cross-dataset signals with evidence-backed findings.  
**Skill Usage**: `python-testing-patterns`, `testing`, `deterministic-tests-marshal`.  
**Demo/Validation**:
- All five analysis plugins emit findings/artifacts on fixture data.
- Timing permutation is deterministic at fixed seed.
- Missing required inputs/configs return `status="error"` (never skipped/no-op).

### Task 5.1: Implement Bundling Analysis
- **Location**: `plugins/analysis_bundled_donations_v1/plugin.py`
- **Description**: Detect same-employer/same-day/same-candidate donation bundles using scratch-SQL aggregations.
- **Complexity**: 7/10
- **Dependencies**: Sprint 4
- **Acceptance Criteria**:
  - Emits top bundle findings with sample evidence refs.
  - Missing required columns returns `status="error"` with explicit missing-field list.
- **Validation**:
  - `tests/plugins/test_analysis_bundled_donations_v1.py`

### Task 5.2: Implement Contribution Limit Flags
- **Location**: `plugins/analysis_contribution_limit_flags_v1/plugin.py`
- **Description**: Group donor-year contributions and flag excess over configured limit.
- **Complexity**: 7/10
- **Dependencies**: Sprint 4
- **Acceptance Criteria**:
  - Correct donor key construction and annual limit checks.
  - Invalid date/amount parsing returns `status="error"` with row-count summary of rejects.
- **Validation**:
  - `tests/plugins/test_analysis_contribution_limit_flags_v1.py`

### Task 5.3: Implement Vendor Influence Breadth
- **Location**: `plugins/analysis_vendor_influence_breadth_v1/plugin.py`
- **Description**: Aggregate cross-links into vendor breadth metrics (candidate reach, donor counts, totals).
- **Complexity**: 6/10
- **Dependencies**: Sprint 4
- **Acceptance Criteria**:
  - Emits ranked breadth outputs and evidence-backed top findings.
  - Missing/invalid link artifact returns `status="error"` with actionable remediation text.
- **Validation**:
  - `tests/plugins/test_analysis_vendor_influence_breadth_v1.py`

### Task 5.4: Implement Timing Permutation Analysis
- **Location**: `plugins/analysis_vendor_politician_timing_permutation_v1/plugin.py`
- **Description**: Compute observed vs null temporal clustering with deterministic RNG and permutation p-values/effect sizes.
- **Complexity**: 9/10
- **Dependencies**: Sprint 4
- **Acceptance Criteria**:
  - Stable outputs under fixed seeds.
  - Actionable error when date fields are missing/invalid.
  - Repeat run with same seed yields identical findings IDs and identical artifact hashes.
- **Validation**:
  - `tests/plugins/test_analysis_vendor_politician_timing_permutation_v1.py`
  - repeat-run exact-output test.

### Task 5.5: Implement Refined Red Flags
- **Location**: `plugins/analysis_red_flags_refined_v1/plugin.py`
- **Description**: Merge inputs from links/bundling/limits/timing and optional procurement-method signals into transparent multi-factor rules.
- **Complexity**: 8/10
- **Dependencies**: Tasks 5.1-5.4
- **Acceptance Criteria**:
  - Findings reference contributing signal artifacts by stable IDs.
  - Thresholds configurable and schema validated.
  - If upstream signals are absent, returns `status="error"` with explicit missing-artifact codes.
- **Validation**:
  - `tests/plugins/test_analysis_red_flags_refined_v1.py`

## Sprint 6: Report Plugin and Evidence Traceability
**Goal**: Produce one authoritative evidence index for the pack.  
**Skill Usage**: `golden-answer-harness`, `python-testing-patterns`.  
**Demo/Validation**:
- `report_evidence_index_v1` emits machine and human summaries.
- Every artifact includes path/hash/record-count/plugin source.
- Report outputs must remain available for both success runs and fail-closed runs.

### Task 6.1: Implement report_evidence_index_v1
- **Location**: `plugins/report_evidence_index_v1/plugin.py`
- **Description**: Scan run artifacts for this plugin pack, compute hashes/counts, publish evidence index findings.
- **Complexity**: 6/10
- **Dependencies**: Sprints 3-5
- **Acceptance Criteria**:
  - Output contains deterministic sorted artifact list and hashes.
  - Missing required artifacts fail closed with actionable message.
  - Evidence index hash list is byte-identical across reruns with same inputs.
- **Validation**:
  - `tests/plugins/test_report_evidence_index_v1.py`

### Task 6.2: Wire Report Surface Hooks
- **Location**: `src/statistic_harness/core/report.py`, `src/statistic_harness/core/plain_report.py`
- **Description**: Ensure new findings/evidence index are surfaced in plain-language and machine-readable report sections without breaking existing report contracts.
- **Complexity**: 6/10
- **Dependencies**: Task 6.1
- **Acceptance Criteria**:
  - Report references evidence index artifact and top cross-dataset findings.
  - Pipeline writes both `report.json` and `report.md` even when one cross-dataset plugin fails.
- **Validation**:
  - Report schema validation in integration tests.

## Sprint 7: Observability, Resource Gates, and CLI Workflow
**Goal**: Make pack operations measurable, bounded, and operator-friendly.  
**Skill Usage**: `discover-observability`, `python-observability`, `observability-engineer`, `resource-budget-enforcer`.  
**Demo/Validation**:
- Plugin runtime stats are visible in outputs/logs.
- Large-dataset runs stay within configured memory guardrails.

### Task 7.1: Add Plugin Runtime Counters and Diagnostics
- **Location**: `src/statistic_harness/core/plugin_runner.py`, plugin `debug` payloads
- **Description**: Add per-plugin counters for rows scanned, match attempts, index sizes, cache hits, and fallback-path usage.
- **Complexity**: 6/10
- **Dependencies**: Sprints 3-6
- **Acceptance Criteria**:
  - Debug payload includes deterministic, non-sensitive runtime metrics.
- **Validation**:
  - Unit tests asserting required debug metrics exist.

### Task 7.2: Add Cross-Dataset Pack Runner Helper
- **Location**: `scripts/run_cross_dataset_openplanter_pack.py`, `docs/operator_playbook_autonomous_insights.md`
- **Description**: Add one deterministic script to run the pack with explicit dataset-role mappings and seed controls.
- **Complexity**: 5/10
- **Dependencies**: Sprints 1-6
- **Acceptance Criteria**:
  - Script validates required roles/mappings before execution.
- **Validation**:
  - CLI smoke test on synthetic fixture set.

### Task 7.3: Add Resource Budget Checks
- **Location**: `tests/perf/test_cross_dataset_pack_resource_budget.py`
- **Description**: Validate no full-DF load for large scans and bounded resource profile behavior.
- **Complexity**: 7/10
- **Dependencies**: Sprints 3-6
- **Acceptance Criteria**:
  - Fails if plugin tries unbounded `dataset_loader()` on large fixture.
  - 11M-row synthetic scan completes under configured respectful profile without full-table DataFrame load.
- **Validation**:
  - Resource-budget test suite with strict failure mode.

## Sprint 8: End-to-End Validation and Release Gate
**Goal**: Prove pack correctness, determinism, and regression safety.  
**Skill Usage**: `testing`, `deterministic-tests-marshal`, `golden-answer-harness`.  
**Demo/Validation**:
- Full pytest and fixture gauntlet green.
- Golden fixture run produces stable recommendation/evidence signatures.

### Task 8.1: Build Tiny Cross-Dataset Golden Fixtures
- **Location**: `tests/fixtures/openplanter_pack/`
- **Description**: Create compact contracts/contributions/candidates fixtures with known expected links, bundles, limit flags, and timing signals.
- **Complexity**: 7/10
- **Dependencies**: Task 1.4, Sprints 3-6
- **Acceptance Criteria**:
  - Fixtures explicitly encode expected true positives and known negatives.
- **Validation**:
  - Fixture assertions in integration tests.

### Task 8.2: Add Full Integration Tests for Pack
- **Location**: `tests/integration/test_openplanter_cross_dataset_pack.py`
- **Description**: Run full plugin sequence and validate report/output artifacts, evidence chains, and deterministic IDs.
- **Complexity**: 8/10
- **Dependencies**: Task 8.1
- **Acceptance Criteria**:
  - All expected artifacts and top-level findings present.
  - Deterministic rerun yields byte-identical `report.json` and identical artifact hash manifest.
- **Validation**:
  - `python -m pytest -q` includes this suite.

### Task 8.3: Add Fail-Closed Regression Tests
- **Location**: `tests/plugins/test_cross_dataset_fail_closed.py`
- **Description**: Validate explicit failures for missing mappings, unsafe SQL statements, missing required fields, and schema mismatches.
- **Complexity**: 6/10
- **Dependencies**: Sprints 1-6
- **Acceptance Criteria**:
  - Every unsafe/missing prerequisite path returns actionable error and `status="error"`.
- **Validation**:
  - Dedicated fail-closed suite must pass.

### Task 8.4: Add Pipeline Fail-Closed + Network-Isolation Tests
- **Location**: `tests/integration/test_cross_dataset_pipeline_fail_closed.py`, `tests/security/test_cross_dataset_no_network.py`
- **Description**: Verify one forced plugin failure still yields completed pipeline reports; monkeypatch network/socket calls and assert cross-dataset plugins never attempt runtime networking.
- **Complexity**: 8/10
- **Dependencies**: Sprints 3-7
- **Acceptance Criteria**:
  - Forced plugin failure run ends with both `report.json` and `report.md` present and schema-valid.
  - Any attempted network call triggers test failure.
- **Validation**:
  - Integration and security tests pass in CI and local runs.

### Task 8.5: Add Release Blocking Gate Artifact
- **Location**: `scripts/verify_openplanter_pack_release_gate.py`, `docs/release_evidence/openplanter_pack_release_gate.json`
- **Description**: Add a release gate script that runs `python -m pytest -q`, records result counts, and hard-fails on unapproved `skip`/`xfail`/failures.
- **Complexity**: 6/10
- **Dependencies**: Tasks 8.2-8.4
- **Acceptance Criteria**:
  - Release gate writes blocking evidence artifact with test totals and status.
  - Any regression status emits non-zero exit code and `DO_NOT_SHIP=true`.
- **Validation**:
  - Manual run and CI run both produce matching gate behavior.

## Testing Strategy
- Unit tests:
  - Core helpers (`entity_resolution`, `evidence_links`, `sql_dump_import`).
  - Each of 9 plugins.
- Integration tests:
  - End-to-end cross-dataset pack run with fixture mappings.
  - Report/evidence schema assertions.
- Determinism tests:
  - Fixed-seed snapshot checks for timing permutation outputs.
  - Stable entity/edge/finding IDs across reruns and shuffled source ordering.
  - Byte-identical `report.json` and stable artifact hashes for repeated full-pack runs.
- Performance tests:
  - Streaming-only scans and resource budget checks on large synthetic data.
- Fail-closed tests:
  - Unsafe SQL, missing roles/fields, invalid schemas, unsupported dialect.
  - Plugin failure still produces final report outputs with error summaries.
  - Network call attempts are blocked and tested.

## Potential Risks and Gotchas
- SQL dump parser edge cases (escaped quotes, multiline inserts, dialect variants) can create silent corruption if parser is permissive.
  - Mitigation: strict parser states, deny unsupported syntax, high-coverage parser tests.
- Cross-dataset role mapping ambiguity can produce false links.
  - Mitigation: explicit mapping schema + required role checks + confidence tiers.
- Fuzzy matching thresholds can over-link or under-link.
  - Mitigation: threshold configs with golden fixture calibration and explainable match features.
- Timing permutation can become expensive on high-cardinality pairs.
  - Mitigation: minimum donation thresholds, capped permutations, scratch indexing.
- Artifact growth on large runs.
  - Mitigation: JSONL streaming outputs, bounded excerpts, top-K finding emission.

## Rollback Plan
1. Keep all new plugins isolated by IDs and additive core modules.
2. If regressions occur, remove plugin IDs from invoked sets while retaining core infra.
3. Revert individual sprint commits in reverse order:
   - Sprint 8 -> 7 -> 6 -> 5 -> 4 -> 3 -> 2 -> 1.
4. Preserve generated fixture evidence in `docs/release_evidence/` for postmortem.

## Post-Plan Review Notes
- This plan intentionally sequences contracts and shared libraries before plugin logic to reduce duplicated matcher/parser implementations.
- The highest risk work is Sprint 2 (parser + matcher determinism) and Sprint 5.4 (timing permutation scaling).
- Do-not-ship gate remains: full `python -m pytest -q` must pass with new suites included.
