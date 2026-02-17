# Plan: 1-32-2026 Recommendations Implementation

**Generated**: January 31, 2026
**Estimated Complexity**: High

## Overview
Implement all recommendations R1–R20 to move the harness to a fully SQLite‑backed, deterministic, append‑only, auditable pipeline with project/dataset lineage, explicit evidence for findings, AutoPlanner selection, backfill for new plugins, and a hardened UI that exposes projects, traces, and method history. All analysis must operate on data stored in SQLite (not the uploaded file) and align with the four pillars: **performance**, **security**, **accuracy**, and **citeability** in balanced priority. The plan prioritizes schema/migration foundations, streaming ingest with safe column canonicalization, dataset/project dedupe, versioned results, lineage graph + parameter normalization, deterministic planning, sequence mining, statistical controls, delivery tracking with hashes, and robust UI/test coverage.

## Prerequisites
- Python 3.11+ and existing dependencies from `pyproject.toml`.
- SQLite with JSON1 and FTS5 enabled (standard in Python builds).
- Ability to run `python -m pytest -q` locally.

## Sprint 1: Storage & Migration Foundations
**Goal**: Establish robust, versioned schema with migrations and thread‑safe DB access (R14, R15 baseline).
**Demo/Validation**:
- `python -m pytest -q` (new migration and concurrency tests).
- Start UI and run two parallel runs without DB errors.

### Task 1.1: Introduce migration framework
- **Location**: `src/statistic_harness/core/storage.py`, new `src/statistic_harness/core/migrations.py`, `tests/test_migrations.py`
- **Description**: Add schema versioning via `PRAGMA user_version`, a `schema_migrations` table, and ordered migration functions. Migrate from existing schema without data loss.
- **Dependencies**: None
- **Acceptance Criteria**:
  - New DB initializes at latest version.
  - Existing `runs` and `plugin_results` data preserved after migration.
- **Validation**:
  - Unit tests covering empty DB and legacy DB migration path.

### Task 1.2: Thread‑safe DB access + WAL
- **Location**: `src/statistic_harness/core/storage.py`, `src/statistic_harness/ui/server.py`
- **Description**: Refactor `Storage` to open per‑operation or per‑request connections, enable WAL and foreign keys, remove global shared connection usage in FastAPI app.
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Multiple background runs complete without `sqlite3.ProgrammingError` or locking issues.
- **Validation**:
  - Concurrency test spawns parallel runs and verifies consistency.

### Task 1.3: Core schema scaffolding for projects/datasets/runs/uploads
- **Location**: `src/statistic_harness/core/migrations.py`, `src/statistic_harness/core/storage.py`
- **Description**: Add base tables for `projects`, `datasets`, `dataset_versions`, `uploads`, and update `runs` to reference dataset_version and project IDs.
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - New tables created and referenced with FK constraints.
- **Validation**:
  - Migration tests assert presence of new tables/columns.

## Sprint 2: Full‑Dataset Ingest & SQLite Row Storage
**Goal**: Stream ingest without truncation, store all rows in SQLite, and canonicalize columns safely (R1, R4, R20).
**Demo/Validation**:
- Integration test with >10k rows stored fully in SQLite.
- `report.json` uses DB‑derived row counts.

### Task 2.1: Streaming ingest for CSV/XLSX/JSON
- **Location**: `plugins/ingest_tabular/plugin.py`, `src/statistic_harness/core/dataset_io.py`
- **Description**: Replace `head(max_rows)` with streaming/batched read; compute row counts and persist to DB in chunks. Keep optional canonical CSV artifact for debugging only.
- **Dependencies**: Sprint 1 schema in place
- **Acceptance Criteria**:
  - Ingest handles >10k rows without truncation.
  - Ingest does not load entire dataset into memory when possible.
- **Validation**:
  - Unit test with large fixture asserts exact row count.

### Task 2.2: Column canonicalization + safe identifiers
- **Location**: `plugins/ingest_tabular/plugin.py`, new helper in `src/statistic_harness/core/utils.py`
- **Description**: Map original headers to safe internal IDs; store mapping in `dataset_columns` table; handle duplicates/reserved words; enforce parameterized SQL.
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Fuzzed headers ingest without SQL errors.
- **Validation**:
  - New fuzz test for hostile headers.

### Task 2.3: SQLite raw row storage design
- **Location**: `src/statistic_harness/core/storage.py`, `src/statistic_harness/core/dataset_io.py`
- **Description**: Implement per‑dataset wide table for typed columns plus a `row_json` column for round‑trip fidelity; store `row_index` and `dataset_version_id` for traceability.
- **Dependencies**: Task 2.1, 2.2
- **Acceptance Criteria**:
  - Round‑trip test CSV → SQLite → export preserves row count and column mapping.
- **Validation**:
  - Property test on synthetic dataset fixtures.

### Task 2.4: DB‑first dataset access
- **Location**: `src/statistic_harness/core/dataset_io.py`, `src/statistic_harness/core/pipeline.py`
- **Description**: Replace file‑based `DatasetAccessor` with SQLite‑backed access (chunked row iteration and column metadata); prohibit analysis plugins from reading raw upload paths.
- **Dependencies**: Task 2.3
- **Acceptance Criteria**:
  - All plugins can operate using DB access only.
  - Canonical CSV (if retained) is not used for analysis.
- **Validation**:
  - Integration test asserts DB row counts used in report.

## Sprint 3: Project/Dataset Dedup + Append‑Only Guarantees
**Goal**: Deduplicate uploads, reuse datasets across runs, and enforce append‑only raw data (R2, R3, R5).
**Demo/Validation**:
- Re‑uploading identical bytes reuses dataset_id.
- Attempts to UPDATE/DELETE raw rows fail.

### Task 3.1: Content fingerprinting + upload records
- **Location**: `src/statistic_harness/ui/server.py`, `src/statistic_harness/core/storage.py`
- **Description**: Compute SHA‑256 at upload; create `uploads` entry; map to project/dataset by fingerprint.
- **Dependencies**: Sprint 1 schema
- **Acceptance Criteria**:
  - Same bytes with different filenames map to same project.
- **Validation**:
  - Unit test for dedupe behavior.

### Task 3.2: Project/dataset resolution in pipeline
- **Location**: `src/statistic_harness/core/pipeline.py`, `src/statistic_harness/core/storage.py`
- **Description**: On run start, resolve `project_id` and `dataset_id` by fingerprint; create new dataset_version if needed; store IDs on runs.
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - New run reuses existing dataset when file unchanged.
- **Validation**:
  - Integration test: rerun with same file creates new run but same dataset_id.

### Task 3.3: Append‑only enforcement
- **Location**: `src/statistic_harness/core/migrations.py`
- **Description**: Add SQLite triggers to prevent DELETE/UPDATE on raw dataset tables; ensure storage layer avoids destructive SQL.
- **Dependencies**: Task 2.3
- **Acceptance Criteria**:
  - DELETE/UPDATE raises SQLite error.
- **Validation**:
  - DB tests attempt UPDATE/DELETE and assert failure.

## Sprint 4: Versioned Results + Evidence‑First Reports
**Goal**: Never overwrite results, attach evidence to every finding, and update report schema (R6, R11, R13) to maximize citeability.
**Demo/Validation**:
- Re‑running a plugin stores a new result record.
- Report schema validates with evidence fields.

### Task 4.1: Versioned plugin result storage + hashes
- **Location**: `src/statistic_harness/core/storage.py`, `src/statistic_harness/core/pipeline.py`
- **Description**: Replace PK `(run_id, plugin_id)` with `result_id` and store `plugin_version`, `executed_at`, `code_hash`, `settings_hash`, `dataset_hash`; add “latest” query helpers to detect stale results when data or plugin changes.
- **Dependencies**: Sprint 1 migrations
- **Acceptance Criteria**:
  - Two runs of same plugin create two records.
- **Validation**:
  - Unit test for historical results retention.

### Task 4.2: Evidence schema + report updates
- **Location**: `src/statistic_harness/core/types.py`, `src/statistic_harness/core/report.py`, `docs/report.schema.json`
- **Description**: Require `evidence` entries per finding (dataset_id, row_ids, column_ids, query snippet); update report builder and JSON schema.
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Report generation fails if findings lack evidence.
- **Validation**:
  - Schema validation tests for evidence fields.

### Task 4.3: Resource budget reporting (unlimited defaults)
- **Location**: `src/statistic_harness/core/types.py`, `src/statistic_harness/core/pipeline.py`
- **Description**: Add budget metadata to context and results; default to “unlimited” but report whether sampling/approximation was used.
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Report includes budget usage fields (even when unlimited).
- **Validation**:
  - Unit test ensures default budget metadata present.

## Sprint 5: Parameter Normalization + Lineage Graph + Indexes
**Goal**: Normalize parameter entities, build association graph, and enable trace queries (R7, R8, R17).
**Demo/Validation**:
- Trace endpoint returns deterministic graph for fixture.
- Parameter text variants normalize to same entity.

### Task 5.1: Parameter entity schema + normalization
- **Location**: `src/statistic_harness/core/storage.py`, `plugins/profile_basic/plugin.py`
- **Description**: Add tables `parameter_entities`, `parameter_kv`, `row_parameter_link`; use heuristic extraction to normalize parameter fields into key/value entities (no fixed schema).
- **Dependencies**: Sprint 2 row storage
- **Acceptance Criteria**:
  - Different textual forms map to same entity_id.
- **Validation**:
  - Fuzz tests on whitespace/order/case.

### Task 5.2: Column role inference (heuristics)
- **Location**: `plugins/profile_basic/plugin.py`, new helper in `src/statistic_harness/core/utils.py`
- **Description**: Infer probable roles (case/process id, activity, timestamp, parameter blob, numeric measures) using heuristics; persist roles in DB for AutoPlanner and analysis plugins.
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - Role inference is deterministic for the same dataset.
- **Validation**:
  - Unit tests with varied fixtures.

### Task 5.3: Lineage graph tables + APIs
- **Location**: `src/statistic_harness/core/storage.py`, new `src/statistic_harness/core/lineage.py`
- **Description**: Add `entities` and `edges` tables, helper APIs, and recursive CTE trace queries.
- **Dependencies**: Task 5.1, 5.2
- **Acceptance Criteria**:
  - Trace query returns reachable nodes for fixture graph.
- **Validation**:
  - Golden test for trace results.

### Task 5.4: Index strategy (FTS + key/value)
- **Location**: `src/statistic_harness/core/migrations.py`
- **Description**: Add indexes on entity key/type, edge src/dst, parameter_kv; optional FTS for raw parameter text.
- **Dependencies**: Task 5.1, 5.3
- **Acceptance Criteria**:
  - EXPLAIN shows index usage for trace/search.
- **Validation**:
  - Performance regression test with synthetic scale.

## Sprint 6: Deterministic AutoPlanner + Plugin Capability Tags
**Goal**: Automatically select plugins based on dataset profile, deterministically (R9).
**Demo/Validation**:
- Planner selects expected plugins for numeric vs event‑log fixtures.

### Task 6.1: Capability tags in plugin manifests
- **Location**: `plugins/*/plugin.yaml`, `src/statistic_harness/core/plugin_manager.py`
- **Description**: Extend plugin manifests with tags (needs_numeric, needs_timestamp, needs_eventlog, etc.) and load into specs.
- **Dependencies**: None
- **Acceptance Criteria**:
  - Plugin specs expose capability tags.
- **Validation**:
  - Unit test on plugin discovery.

### Task 6.2: AutoPlanner phase in pipeline
- **Location**: `src/statistic_harness/core/pipeline.py`, new `src/statistic_harness/core/planner.py`
- **Description**: Add deterministic planner after ingest/profile to select plugins; log plan and store in DB.
- **Dependencies**: Task 6.1, Sprint 2, Sprint 5
- **Acceptance Criteria**:
  - Same dataset always yields identical plan.
- **Validation**:
  - Snapshot tests for planner output.

## Sprint 7: Sequence Mining + Evidence‑Rich Findings
**Goal**: Add process/sequence mining plugin with evidence and parameter linkage (R10).
**Demo/Validation**:
- Synthetic event‑log fixture yields known variants and transitions.

### Task 7.1: Event‑log detection utilities
- **Location**: `src/statistic_harness/core/utils.py`, `plugins/analysis_process_sequence/plugin.py` (new)
- **Description**: Detect case/activity/timestamp columns from profile metadata and dataset schema.
- **Dependencies**: Sprint 2, Sprint 5
- **Acceptance Criteria**:
  - Event‑log columns detected deterministically for fixtures.
- **Validation**:
  - Unit tests for detection heuristics.

### Task 7.2: Sequence mining plugin implementation
- **Location**: `plugins/analysis_process_sequence/*`
- **Description**: Compute frequent variants (n‑grams + full sequences), transition matrices, rare path anomalies; store findings with evidence row pointers.
- **Dependencies**: Task 7.1
- **Acceptance Criteria**:
  - Findings include evidence with row_ids and parameter entity references.
- **Validation**:
  - Plugin unit tests with synthetic fixtures.

## Sprint 8: Statistical Controls Library
**Goal**: Reduce false discoveries across plugins (R12).
**Demo/Validation**:
- Null datasets produce low finding counts.

### Task 8.1: Add statistical controls module
- **Location**: new `src/statistic_harness/core/stat_controls.py`
- **Description**: Implement effect sizes, multiple‑testing corrections, confidence scoring utilities.
- **Dependencies**: None
- **Acceptance Criteria**:
  - Functions deterministic with fixed seeds.
- **Validation**:
  - Unit tests on known distributions.

### Task 8.2: Integrate controls into analysis plugins
- **Location**: `plugins/analysis_*/*`
- **Description**: Apply scoring and corrections to plugin outputs; include confidence in findings.
- **Dependencies**: Task 8.1
- **Acceptance Criteria**:
  - Findings include score/confidence fields.
- **Validation**:
  - Regression tests on synthetic null data.

## Sprint 9: Backfill + Job Queue + Delivery Tracking
**Goal**: Backrun new plugins over old datasets and track executed vs pending vs delivered (R18) using hashes for data/plugin changes.
**Demo/Validation**:
- Adding new plugin schedules jobs for existing datasets.

### Task 9.1: Analysis job queue in SQLite
- **Location**: `src/statistic_harness/core/migrations.py`, `src/statistic_harness/core/storage.py`
- **Description**: Add `analysis_jobs` table; create enqueue/dequeue helpers; idempotent scheduling per dataset/plugin/version.
- **Dependencies**: Sprint 4 (versioned results)
- **Acceptance Criteria**:
  - Jobs are not duplicated for same dataset/plugin/version.
- **Validation**:
  - Unit test for enqueue idempotency.

### Task 9.2: Delivery state + hash tracking
- **Location**: `src/statistic_harness/core/storage.py`, `src/statistic_harness/core/migrations.py`
- **Description**: Add tables for `deliveries` (project_id, dataset_version_id, plugin_id, plugin_hash, dataset_hash, delivered_at, notes). Determine “needs re-run” when dataset hash or plugin hash changes since delivery.
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Delivered plugins remain marked delivered unless data or plugin hash changes.
- **Validation**:
  - Unit tests covering hash changes.

### Task 9.3: Backfill CLI command
- **Location**: `src/statistic_harness/cli.py`
- **Description**: Implement `backfill --plugin <id> --all-projects` to schedule jobs; worker processes jobs sequentially.
- **Dependencies**: Task 9.1
- **Acceptance Criteria**:
  - Backfill schedules expected jobs and produces results.
- **Validation**:
  - Integration test with fixture DB.

## Sprint 10: UI/UX Hardening + Project/Trace Views
**Goal**: Bulletproof UI with project index, upload → run flow, trace views, and method history (R16).
**Demo/Validation**:
- Upload triggers run automatically and shows progress + method list.
- Project list and trace endpoints work with fixtures.

### Task 10.1: Upload flow + validation
- **Location**: `src/statistic_harness/ui/server.py`, `src/statistic_harness/ui/templates/*`
- **Description**: Validate file types/sizes; compute hash at upload; auto‑create run; return run_id without manual upload_id.
- **Dependencies**: Sprint 3 (upload records)
- **Acceptance Criteria**:
  - Upload returns run_id immediately.
- **Validation**:
  - FastAPI TestClient smoke tests.

### Task 10.2: Project list + dataset views
- **Location**: `src/statistic_harness/ui/server.py`, `src/statistic_harness/ui/templates/*`
- **Description**: Add `/projects` list, project detail, dataset metadata, and global search by parameter/process/hash.
- **Dependencies**: Sprint 5 indexes
- **Acceptance Criteria**:
  - UI renders project list with pagination.
- **Validation**:
  - UI tests with TestClient.

### Task 10.3: Trace endpoint and method history
- **Location**: `src/statistic_harness/ui/server.py`, templates
- **Description**: Add `/trace` endpoint returning lineage graph; show methods run vs pending vs delivered per dataset (based on job queue, results, and deliveries).
- **Dependencies**: Sprint 5, Sprint 9
- **Acceptance Criteria**:
  - Method history shows executed timestamps and pending plugin list.
- **Validation**:
  - Deterministic response tests.

### Task 10.4: Delivery controls in UI
- **Location**: `src/statistic_harness/ui/server.py`, templates
- **Description**: Add UI controls to mark project delivered (all selected plugins) and per‑plugin delivered; display stale/due when plugin or data hash changes.
- **Dependencies**: Task 9.2
- **Acceptance Criteria**:
  - User can mark delivery at project or plugin level.
- **Validation**:
  - UI tests for delivery state toggles.

## Sprint 11: PII Tagging + Governance
**Goal**: Identify PII columns and propagate tags without exporting raw data (R19).
**Demo/Validation**:
- PII fixture columns detected and tagged in report.

### Task 11.1: PII detection in profile plugin
- **Location**: `plugins/profile_basic/plugin.py`, `src/statistic_harness/core/storage.py`
- **Description**: Add regex/pattern heuristics for PII (email, phone, SSN, address); tag columns and store in DB.
- **Dependencies**: Sprint 2 schema
- **Acceptance Criteria**:
  - PII tags appear in report metadata.
- **Validation**:
  - PII fixture tests.

### Task 11.2: Redaction policy for reports
- **Location**: `src/statistic_harness/core/report.py`, `docs/report.schema.json`
- **Description**: Ensure reports do not include raw PII; only tags and aggregated insights.
- **Dependencies**: Task 11.1
- **Acceptance Criteria**:
  - Report omits raw PII values.
- **Validation**:
  - Report schema + regression tests.

## Sprint 12: Test Harness Expansion
**Goal**: Comprehensive unit/integration tests + evaluator harness updates (AGENTS.md).
**Demo/Validation**:
- `python -m pytest -q` green with new tests.

### Task 12.1: Update evaluator harness
- **Location**: `src/statistic_harness/core/evaluation.py`, `tests/test_evaluation.py`
- **Description**: Align evaluator with new report/evidence schema and new findings structure.
- **Dependencies**: Sprint 4
- **Acceptance Criteria**:
  - Ground truth YAML checks pass with updated schema.
- **Validation**:
  - Evaluator tests pass.

### Task 12.2: Integration tests for full pipeline
- **Location**: `tests/test_pipeline_integration.py`, new fixtures in `tests/fixtures/*`
- **Description**: Run full pipeline against dataset fixtures; assert report outputs exist and validate; assert backfill works.
- **Dependencies**: Sprints 1–11
- **Acceptance Criteria**:
  - Reports generated and validate against schema.
- **Validation**:
  - Integration tests pass.

## Testing Strategy
- Unit tests per plugin for deterministic outputs and evidence presence (accuracy + citeability).
- Integration tests that run full pipeline and validate `report.json` against updated schema.
- Migration tests for legacy → latest schema (data preserved).
- Performance tests for large CSV ingestion and trace query latency.
- Security tests for path traversal prevention and safe SQL identifier handling.
- Concurrency tests for multi‑run parallel execution.

## Potential Risks & Gotchas
- Schema migrations must preserve historical data; failure risks data loss.
- Wide per‑dataset tables can be large; ensure batching and index strategy to avoid slow ingest.
- Deterministic AutoPlanner must avoid non‑deterministic ordering from dict/SQL.
- Sequence mining on very large datasets may be heavy; consider staged computation with DB‑side aggregation.
- UI trace/query endpoints must avoid exposing raw PII values.
- Delivery state can drift if hashes are computed inconsistently; define canonical hashing inputs and version them.

## Rollback Plan
- Keep migration steps reversible where possible; provide a downgrade script for schema version N → N‑1.
- Gate new behavior behind config flags if needed (AutoPlanner on/off).
- Maintain read compatibility for legacy report schema during transition.
