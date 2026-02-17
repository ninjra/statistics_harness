# Plan: Cohesive Roadmap (3 Plans + Phase 2 TODOs)

**Generated**: February 2, 2026
**Estimated Complexity**: High

## Overview
This plan unifies the three existing plans (kernel/plugin stability, queue/capacity plugins + evaluation, and 1‑32‑2026 recommendations) into one sequenced roadmap and adds explicit Phase 2 TODO/Not‑Implemented closure. It prioritizes the four pillars (performance, security, accuracy, citeability) equally, enforces Phase 1 offline constraints, and gates Phase 2 on a fully passing test suite. It also encodes user decisions: shared DB tenancy via `tenant_id`, auth via both session tokens and API keys, sqlite-vec builtin for vectors, no dataset row truncation (streaming ingest for million‑row sheets), loose performance thresholds, and PII anonymization via entity hashes before any explicit cloud/API egress.

## Prerequisites
- Python 3.11+ (`python3` is available) and `python -m pytest -q` passes on baseline before changes.
- SQLite with JSON1 and builtin `sqlite-vec` extension.
- Real Quorum‑style fixture for queue/capacity parity validation.
- Phase 1 must remain local‑only (no runtime network calls).
- Default top‑K vector query target is 10 (configurable only if explicitly required).

## Sprint 1: Determinism + Guardrails Baseline
**Goal**: Close determinism gaps and security guardrails before feature expansion.
**Demo/Validation**:
- Run the same pipeline twice with the same seed → byte‑stable `report.json`.
- Oversized uploads and traversal attempts are rejected deterministically.

### Task 1.1: Deterministic seeding + canonical JSON
- **Location**: `src/statistic_harness/core/plugin_runner.py`, `src/statistic_harness/core/pipeline.py`, `src/statistic_harness/core/utils.py`, `src/statistic_harness/core/report.py`
- **Description**: Seed `random`/NumPy (if available), set stable env (`PYTHONHASHSEED`, `TZ=UTC`, `LC_ALL=C`), and add canonical JSON formatting for report output ordering and float precision.
- **Complexity**: 6
- **Dependencies**: None
- **Acceptance Criteria**:
  - Identical runs on the same dataset produce byte‑identical `report.json`.
- **Validation**:
  - New determinism test runs pipeline twice and diffs output.

### Task 1.2: Offline + path + upload guardrails
- **Location**: `src/statistic_harness/ui/server.py`, `src/statistic_harness/core/utils.py`, `src/statistic_harness/core/storage.py`, `tests/test_security_paths.py`, `tests/test_offline.py`
- **Description**: Enforce max upload size and path traversal protections; keep offline network denial in UI + plugin runner.
- **Complexity**: 5
- **Dependencies**: None
- **Acceptance Criteria**:
  - Oversized uploads fail with a deterministic error.
  - Path traversal attempts fail.
- **Validation**:
  - Security tests for size + traversal + offline access.

### Task 1.3: Performance smoke test (loose thresholds)
- **Location**: `tests/test_performance_smoke.py`
- **Description**: Add a small benchmark with loose thresholds to catch regressions without blocking normal runs.
- **Complexity**: 3
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Performance smoke test is stable on local dev.
- **Validation**:
  - `python -m pytest -q` passes including the smoke test.

### Task 1.4: Kernel boundary security guardrails
- **Location**: `src/statistic_harness/core/utils.py`, `src/statistic_harness/core/storage.py`, `src/statistic_harness/ui/server.py`, `tests/test_security_paths.py`
- **Description**: Enforce file type/size validation, block `pickle`/`eval`/shelling in analysis paths, and add tests for each guardrail.
- **Complexity**: 5
- **Dependencies**: Task 1.2
- **Acceptance Criteria**:
  - Disallowed file types, sizes, and dangerous operations are rejected deterministically.
- **Validation**:
  - Security tests cover each guardrail.

## Sprint 2: Plugin Contracts + Isolation
**Goal**: Harden plugin interfaces and enforce isolation via subprocess execution.
**Demo/Validation**:
- `stat-harness list-plugins` shows validated manifest metadata.
- Plugins run in subprocesses with audit entries for every execution.

### Task 2.1: Plugin manifest schema validation
- **Location**: `docs/plugin_manifest.schema.json`, `src/statistic_harness/core/plugin_manager.py`, `plugins/*/plugin.yaml`
- **Description**: Define required manifest fields (id, version, type, entrypoint, schemas, sandbox flags) and validate on load.
- **Complexity**: 6
- **Dependencies**: None
- **Acceptance Criteria**:
  - Invalid manifests fail fast with clear errors.
- **Validation**:
  - Unit tests for manifest validation.

### Task 2.2: Per‑plugin config/output schemas
- **Location**: `plugins/*/config.schema.json`, `plugins/*/output.schema.json`
- **Description**: Add minimal deterministic schemas for plugin inputs/outputs (including measurement labels and evidence fields).
- **Complexity**: 7
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Kernel validates config and output per plugin run.
- **Validation**:
  - Schema validation tests.

### Task 2.3: Subprocess runner + sandbox + audit log
- **Location**: `src/statistic_harness/core/plugin_runner.py`, `src/statistic_harness/core/pipeline.py`, `src/statistic_harness/core/migrations.py`, `src/statistic_harness/core/storage.py`
- **Description**: Execute plugins in subprocesses with deterministic I/O, block network, restrict FS allowlist, and record `plugin_executions` metrics.
- **Complexity**: 8
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Every plugin execution writes an audit row.
  - Network/file access outside allowlist fails.
- **Validation**:
  - Integration tests with a “malicious” plugin fixture.

### Task 2.4: Deterministic parallel scheduling
- **Location**: `src/statistic_harness/core/pipeline.py`, `src/statistic_harness/core/report.py`
- **Description**: Run independent plugins in parallel by DAG layer and preserve deterministic ordering in report assembly.
- **Complexity**: 6
- **Dependencies**: Task 2.3
- **Acceptance Criteria**:
  - Parallel execution yields identical output to sequential runs.
- **Validation**:
  - Integration test comparing sequential vs parallel outputs.

### Task 2.5: Kernel minimization + plugin‑only steps
- **Location**: `src/statistic_harness/core/pipeline.py`, `src/statistic_harness/core/plugin_manager.py`, `plugins/planner_basic/*`, `plugins/transform_template/*`
- **Description**: Remove kernel fallback logic so planner/transform/analysis/report/llm steps are plugin‑only; fail closed with error summaries when missing.
- **Complexity**: 6
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Auto‑runs require planner plugin output; transform runs only via plugin.
- **Validation**:
  - Unit tests for missing‑plugin error summaries.

## Sprint 3: Storage + Ingest Completeness
**Goal**: Store all rows in SQLite (no truncation), dedupe uploads, and enforce append‑only raw data.
**Demo/Validation**:
- >1M‑row sheet ingests without truncation.
- Re‑uploading identical data reuses dataset IDs.

### Task 3.1: Migration framework + thread‑safe DB access
- **Location**: `src/statistic_harness/core/migrations.py`, `src/statistic_harness/core/storage.py`, `src/statistic_harness/ui/server.py`, `tests/test_migrations.py`
- **Description**: Ensure versioned migrations, WAL, and per‑request connections; add missing migrations if needed.
- **Complexity**: 6
- **Dependencies**: None
- **Acceptance Criteria**:
  - Legacy DB upgrades to latest schema without data loss.
- **Validation**:
  - Migration tests pass for all fixtures.

### Task 3.2: Streaming ingest (no truncation)
- **Location**: `plugins/ingest_tabular/plugin.py`, `src/statistic_harness/core/dataset_io.py`
- **Description**: Stream CSV/XLSX/JSON in chunks, compute full row counts, and persist all rows to SQLite without caps.
- **Complexity**: 7
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Large fixtures ingest without truncation and without full‑file memory load.
- **Validation**:
  - Large‑fixture test asserting exact row count.

### Task 3.3: Column canonicalization + safe identifiers
- **Location**: `src/statistic_harness/core/utils.py`, `plugins/ingest_tabular/plugin.py`, `src/statistic_harness/core/storage.py`
- **Description**: Map raw headers to safe IDs, store mapping in `dataset_columns`, and guard against SQL injection in identifiers.
- **Complexity**: 5
- **Dependencies**: Task 3.2
- **Acceptance Criteria**:
  - Hostile headers ingest without SQL errors.
- **Validation**:
  - Fuzz tests for header normalization.

### Task 3.4: Append‑only enforcement + dedupe
- **Location**: `src/statistic_harness/core/migrations.py`, `src/statistic_harness/core/storage.py`, `src/statistic_harness/ui/server.py`
- **Description**: Add triggers to block UPDATE/DELETE on raw tables, hash uploads for dedupe, and reuse dataset IDs for identical content.
- **Complexity**: 6
- **Dependencies**: Task 3.2
- **Acceptance Criteria**:
  - UPDATE/DELETE fails for raw row tables.
  - Duplicate uploads map to the same dataset.
- **Validation**:
  - Tests for append‑only enforcement and dedupe behavior.

### Task 3.5: DB‑first dataset access
- **Location**: `src/statistic_harness/core/dataset_io.py`, `src/statistic_harness/core/pipeline.py`
- **Description**: Ensure analysis plugins read datasets exclusively from SQLite (no raw upload path access), with chunked iteration helpers.
- **Complexity**: 5
- **Dependencies**: Task 3.2
- **Acceptance Criteria**:
  - Analysis plugins operate without reading original upload files.
- **Validation**:
  - Integration test asserts DB row counts are used in report.

### Task 3.6: Golden DB fixtures + migration safety
- **Location**: `tests/fixtures/db/*`, `tests/test_migrations.py`
- **Description**: Generate a small SQLite fixture per schema version and verify upgrades preserve core data (runs/results/templates).
- **Complexity**: 5
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Migrations pass across all versions with preserved data.
- **Validation**:
  - `python -m pytest -q` passes `test_migrations.py`.

## Sprint 4: Evidence‑First Reporting + Evaluation
**Goal**: Guarantee citeable outputs with evidence links, evaluator parity, and measurement labeling.
**Demo/Validation**:
- `report.md` and `report.json` always generated and validate against schema.

### Task 4.1: Versioned results + evidence schema
- **Location**: `src/statistic_harness/core/storage.py`, `src/statistic_harness/core/report.py`, `docs/report.schema.json`
- **Description**: Store versioned plugin results with hashes, require evidence entries per finding (row/column references, query snippets), and always emit `report.md` + `report.json` even if plugins fail.
- **Complexity**: 6
- **Dependencies**: Sprint 3
- **Acceptance Criteria**:
  - Re‑running a plugin creates a new result record.
  - Reports fail validation if evidence is missing.
- **Validation**:
  - Schema validation tests.

### Task 4.2: Evaluator harness + expected metrics
- **Location**: `src/statistic_harness/core/evaluation.py`, `docs/evaluation.md`, `tests/test_evaluation.py`
- **Description**: Add numeric expected metrics with abs/rel tolerance; default strict evaluation (unexpected findings fail unless allowed).
- **Complexity**: 5
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Evaluation fails when metrics drift beyond tolerance.
- **Validation**:
  - Unit tests with known expected metrics.

### Task 4.3: Measurement labeling contract
- **Location**: `docs/report.schema.json`, `src/statistic_harness/core/report.py`, `plugins/*/output.schema.json`
- **Description**: Require `measurement_type` (measured/modeled/not_applicable/error) in findings and metrics.
- **Complexity**: 4
- **Dependencies**: Task 4.2
- **Acceptance Criteria**:
  - All plugin outputs validate with measurement labels.
- **Validation**:
  - Schema tests + plugin output tests.

### Task 4.4: AutoPlanner + capability tags
- **Location**: `plugins/*/plugin.yaml`, `src/statistic_harness/core/planner.py`, `src/statistic_harness/core/pipeline.py`
- **Description**: Add capability tags (needs_eventlog, needs_timestamp, etc.) and deterministic planner selection based on profile metadata.
- **Complexity**: 6
- **Dependencies**: Task 2.1, Sprint 3
- **Acceptance Criteria**:
  - Same dataset yields identical plan; event‑log datasets select queue plugins.
- **Validation**:
  - Snapshot tests for planner output.

### Task 4.5: Parameter entities + normalization
- **Location**: `src/statistic_harness/core/storage.py`, `plugins/profile_basic/plugin.py`
- **Description**: Add `parameter_entities`, `parameter_kv`, and `row_parameter_link` tables; normalize parameter text into deterministic entities.
- **Complexity**: 6
- **Dependencies**: Sprint 3
- **Acceptance Criteria**:
  - Textual variants map to the same entity_id.
- **Validation**:
  - Fuzz tests on whitespace/case/order variants.

### Task 4.6: Column role inference (general)
- **Location**: `plugins/profile_basic/plugin.py`, `src/statistic_harness/core/utils.py`
- **Description**: Infer probable roles (case/process id/activity/timestamp/measure) for non‑eventlog datasets to feed AutoPlanner.
- **Complexity**: 5
- **Dependencies**: Task 4.5
- **Acceptance Criteria**:
  - Role inference is deterministic for fixtures.
- **Validation**:
  - Unit tests for role assignment.

### Task 4.7: Lineage graph tables + trace queries
- **Location**: `src/statistic_harness/core/storage.py`, `src/statistic_harness/core/lineage.py`, `src/statistic_harness/core/migrations.py`
- **Description**: Add `entities`/`edges` tables, recursive CTE trace queries, and indexes for graph traversal.
- **Complexity**: 6
- **Dependencies**: Task 4.5
- **Acceptance Criteria**:
  - Trace query returns deterministic reachable nodes.
- **Validation**:
  - Golden test for trace results.

## Sprint 5: Queue/Capacity Plugin Suite + Parity
**Goal**: Implement queue/capacity plugins with deterministic parity against real fixtures.
**Demo/Validation**:
- Quorum fixture parity suite passes.

### Task 5.1: Field inference + ERP scoping
- **Location**: `plugins/profile_eventlog/`, `src/statistic_harness/core/storage.py`, `src/statistic_harness/core/migrations.py`, `src/statistic_harness/ui/server.py`
- **Description**: Infer event‑log roles with confidence scores; store candidates; scope known‑issues by ERP type (default “unknown”).
- **Complexity**: 7
- **Dependencies**: Sprint 3
- **Acceptance Criteria**:
  - Deterministic role map with confidence scores and warnings for low confidence.
- **Validation**:
  - Unit tests on synthetic event‑log fixtures.

### Task 5.2: Core queue plugins (decomposition → attribution)
- **Location**: `plugins/analysis_queue_delay_decomposition/`, `plugins/analysis_dependency_resolution_join/`, `plugins/analysis_sequence_classification/`, `plugins/analysis_tail_isolation/`, `plugins/analysis_percentile_analysis/`, `plugins/analysis_attribution/`, `plugins/analysis_determinism_discipline/`
- **Description**: Implement measured findings with evidence and measurement labels; skip gracefully when roles are missing.
- **Complexity**: 8
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - Deterministic metrics and labeled outputs for each plugin.
- **Validation**:
  - Unit tests per plugin with synthetic fixtures.

### Task 5.3: Modeled plugins (sequence + concurrency + scaling)
- **Location**: `plugins/analysis_chain_makespan/`, `plugins/analysis_concurrency_reconstruction/`, `plugins/analysis_capacity_scaling/`
- **Description**: Add modeled outputs with explicit assumptions and evidence, separating measured vs modeled results.
- **Complexity**: 7
- **Dependencies**: Task 5.2
- **Acceptance Criteria**:
  - Modeled outputs include assumptions and are labeled correctly.
- **Validation**:
  - Plugin tests with known fixtures.

### Task 5.4: Real‑fixture parity suite
- **Location**: `tests/fixtures/`, `tests/test_quorum_evaluation.py`, `tests/plugins/*`
- **Description**: Add the real Quorum fixture and expected metrics/findings with loose thresholds.
- **Complexity**: 6
- **Dependencies**: Tasks 5.1–5.3
- **Acceptance Criteria**:
  - Parity suite passes only when metrics match within tolerance.
- **Validation**:
  - `python -m pytest -q` passes with parity suite.

### Task 5.5: Process sequence mining plugin
- **Location**: `plugins/analysis_process_sequence/*`, `src/statistic_harness/core/utils.py`
- **Description**: Implement sequence mining (variants, transitions, anomalies) with evidence row references and parameter entity linkage.
- **Complexity**: 6
- **Dependencies**: Sprint 3, Task 4.1
- **Acceptance Criteria**:
  - Findings include evidence and are deterministic for fixtures.
- **Validation**:
  - Plugin unit tests with synthetic event‑log fixtures.

### Task 5.6: Statistical controls module
- **Location**: `src/statistic_harness/core/stat_controls.py`
- **Description**: Add effect sizes, multiple‑testing corrections (BH/FDR), and confidence scoring utilities with deterministic outputs.
- **Complexity**: 5
- **Dependencies**: None
- **Acceptance Criteria**:
  - Deterministic results for fixed inputs.
- **Validation**:
  - Unit tests on known distributions.

### Task 5.7: Integrate statistical controls into analysis plugins
- **Location**: `plugins/analysis_*/*`, `plugins/*/output.schema.json`
- **Description**: Apply corrections and add confidence/score fields to findings and metrics across analysis plugins.
- **Complexity**: 6
- **Dependencies**: Task 5.6
- **Acceptance Criteria**:
  - Findings include confidence fields; results remain deterministic.
- **Validation**:
  - Regression tests on synthetic null datasets.

### Task 5.8: Manual role overrides (optional)
- **Location**: `src/statistic_harness/core/storage.py`, `src/statistic_harness/core/migrations.py`, `src/statistic_harness/ui/server.py`, `src/statistic_harness/ui/templates/*`
- **Description**: Allow per‑project role overrides when inference is ambiguous; overrides are deterministic and never required to run.
- **Complexity**: 5
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - Overrides take precedence over inferred roles when present.
- **Validation**:
  - Unit test verifying override precedence.

## Sprint 6: UI + Ops Workflow (Phase 1)
**Goal**: Deliver end‑to‑end project/dataset visibility and operational controls.
**Demo/Validation**:
- Upload → run flow is deterministic; project list and trace view render.

### Task 6.1: Upload flow improvements
- **Location**: `src/statistic_harness/ui/server.py`, `src/statistic_harness/ui/templates/*`
- **Description**: Validate file types/sizes, compute hash at upload, auto‑create run, and return run_id immediately.
- **Complexity**: 5
- **Dependencies**: Sprint 3
- **Acceptance Criteria**:
  - Upload returns a run_id without manual steps.
- **Validation**:
  - FastAPI TestClient upload tests.

### Task 6.2: Project/dataset views + trace UI
- **Location**: `src/statistic_harness/ui/server.py`, `src/statistic_harness/ui/templates/*`, `src/statistic_harness/core/lineage.py`
- **Description**: Add project index, dataset metadata, and a trace endpoint/UI using lineage graph queries.
- **Complexity**: 6
- **Dependencies**: Task 4.7
- **Acceptance Criteria**:
  - Trace view returns deterministic graph results.
- **Validation**:
  - UI smoke tests for project list and trace endpoints.

### Task 6.3: Mapping presets + combined analysis filters
- **Location**: `src/statistic_harness/core/storage.py`, `src/statistic_harness/core/migrations.py`, `src/statistic_harness/ui/server.py`, `src/statistic_harness/ui/templates/*`
- **Description**: Persist mapping presets per raw format and add filtered combined runs (project/format/date range).
- **Complexity**: 6
- **Dependencies**: Sprint 3
- **Acceptance Criteria**:
  - Presets can be saved/selected; combined runs respect filters.
- **Validation**:
  - UI tests for presets + filter fields.

### Task 6.4: Backfill + delivery tracking
- **Location**: `src/statistic_harness/core/migrations.py`, `src/statistic_harness/core/storage.py`, `src/statistic_harness/cli.py`, `src/statistic_harness/ui/server.py`
- **Description**: Add job queue + delivery tables, backfill CLI, and UI controls to mark delivered/stale based on hashes.
- **Complexity**: 7
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Backfill schedules jobs deterministically; delivery state updates correctly.
- **Validation**:
  - Integration tests for backfill + delivery state.

## Phase Gate: Phase 1 Completion
**Requirement**: Do not start Phase 2 until Sprints 1–6 are complete and `python -m pytest -q` passes fully.

## Sprint 7: Phase 2 Tenancy + Auth (and TODO Closure)
**Goal**: Add tenant isolation and auth while preserving local‑only behavior by default.
**Demo/Validation**:
- Two tenants can run analyses without data leakage.
- Auth gates UI/API/CLI deterministically.

### Task 7.1: Tenant context + migrations
- **Location**: `src/statistic_harness/core/tenancy.py`, `src/statistic_harness/core/migrations.py`, `src/statistic_harness/core/storage.py`
- **Description**: Implement shared‑DB tenancy via `tenant_id` columns, tenant‑scoped appdata paths, and migration of legacy data into default tenant.
- **Complexity**: 7
- **Dependencies**: Phase Gate
- **Acceptance Criteria**:
  - Tenant A cannot access Tenant B data.
- **Validation**:
  - Tenant isolation tests (uploads, runs, vectors).

### Task 7.2: Auth tables + middleware + admin flows
- **Location**: `src/statistic_harness/core/migrations.py`, `src/statistic_harness/ui/server.py`, `src/statistic_harness/ui/templates/*`, `src/statistic_harness/cli.py`, `src/statistic_harness/core/auth.py`
- **Description**: Add user/tenant/membership tables, session tokens and API keys, login + bootstrap admin, and CLI user/key management.
- **Complexity**: 8
- **Dependencies**: Task 7.1
- **Acceptance Criteria**:
  - Protected routes require auth; CLI respects API keys.
- **Validation**:
  - UI/API auth tests with valid/invalid tokens.

### Task 7.3: Phase 2 flags + TODO closure
- **Location**: `src/statistic_harness/core/utils.py`, `src/statistic_harness/ui/server.py`, `src/statistic_harness/cli.py`, `codex/PROJECT_SPEC.md`, `README.md`
- **Description**: Add feature flags for tenancy/auth/vector store and remove “Phase 2 stub only” language; scan repo for TODO/NotImplemented markers and resolve or document them.
- **Complexity**: 4
- **Dependencies**: Task 7.2
- **Acceptance Criteria**:
  - Phase 1 behavior is unchanged when flags are off.
  - Phase 2 TODOs are resolved or explicitly tracked.
- **Validation**:
  - Flag‑on/off tests; doc update review.

## Sprint 8: Phase 2 Vector Store + Stateless Flow
**Goal**: Provide a local vector store with deterministic, stateless, repeatable query flow.
**Demo/Validation**:
- Add/query/delete vectors with deterministic ordering and tenant isolation.

### Task 8.1: sqlite‑vec backend + registry
- **Location**: `src/statistic_harness/core/vector_store.py`, `src/statistic_harness/core/migrations.py`
- **Description**: Implement add/query/delete on sqlite‑vec, store collection metadata in DB, and enforce tenant scoping.
- **Complexity**: 6
- **Dependencies**: Sprint 7
- **Acceptance Criteria**:
  - Vector queries return deterministic ordering.
- **Validation**:
  - Unit tests for add/query/delete and isolation.

### Task 8.2: Deterministic embedding/indexing plugin
- **Location**: `plugins/report_bundle/plugin.py` (or new `plugins/analysis_vector_index/`)
- **Description**: Create deterministic embeddings (hash‑based or TF‑IDF) and index findings/notes into the vector store.
- **Complexity**: 5
- **Dependencies**: Task 8.1
- **Acceptance Criteria**:
  - Index is created and queryable for a run.
- **Validation**:
  - Integration test that indexes and retrieves expected neighbors.

### Task 8.3: Two‑step stateless query flow (repeatable)
- **Location**: `src/statistic_harness/ui/server.py`, `src/statistic_harness/ui/templates/vectors.html`, `src/statistic_harness/cli.py`
- **Description**: Step 1 creates a query snapshot (as_of timestamp + signed cursor). Step 2 paginates via cursor only, ensuring repeatable results and deterministic ordering. Default top‑K is 10; allow override only when explicitly configured.
- **Complexity**: 6
- **Dependencies**: Task 8.1
- **Acceptance Criteria**:
  - Cursor‑only pagination works without server‑side state.
  - Results are repeatable using the same cursor.
- **Validation**:
  - API/UI tests for cursor pagination and repeatability.

### Task 8.4: Ultra‑flexible vector collections
- **Location**: `src/statistic_harness/core/vector_store.py`, `src/statistic_harness/core/storage.py`
- **Description**: Support multiple collections per run, metadata filters, and future embedding variants without schema changes.
- **Complexity**: 4
- **Dependencies**: Task 8.1
- **Acceptance Criteria**:
  - Metadata filters restrict queries deterministically.
- **Validation**:
  - Unit tests for filtered queries.

## Sprint 9: Phase 2 Governance + Docs + Regression
**Goal**: Add governance metadata and finalize documentation/tests.
**Demo/Validation**:
- `python -m pytest -q` passes with Phase 2 features enabled.

### Task 9.1: Resource budget metadata
- **Location**: `src/statistic_harness/core/types.py`, `src/statistic_harness/core/report.py`, `docs/report.schema.json`, `plugins/*/output.schema.json`
- **Description**: Add budget metadata with defaults of “unlimited” and include in report output.
- **Complexity**: 4
- **Dependencies**: Sprint 7
- **Acceptance Criteria**:
  - Budget fields appear in all plugin outputs.
- **Validation**:
  - Schema tests for budget defaults.

### Task 9.2: PII tagging + cloud‑egress anonymization
- **Location**: `plugins/profile_basic/plugin.py`, `src/statistic_harness/core/migrations.py`, `src/statistic_harness/core/storage.py`, `plugins/llm_prompt_builder/*`, `tests/test_llm_prompt_builder.py`
- **Description**: Detect PII columns, store entity hashes in DB, and anonymize raw PII only when sending to explicit cloud/API egress (never during local analysis).
- **Complexity**: 6
- **Dependencies**: Sprint 7
- **Acceptance Criteria**:
  - Outbound payloads contain no raw PII; local reports still store full data internally.
- **Validation**:
  - Tests scan outbound payloads for PII patterns.

### Task 9.3: Docs + migration fixtures + regression tests
- **Location**: `README.md`, `docs/*`, `tests/fixtures/db/*`, `tests/test_migrations.py`, `tests/*`
- **Description**: Update docs to reflect Phase 2 features and regenerate migration fixtures; expand regression coverage for tenancy, auth, vector store, and PII.
- **Complexity**: 5
- **Dependencies**: Sprint 7–8
- **Acceptance Criteria**:
  - Docs match actual behavior; migration fixtures pass.
- **Validation**:
  - `python -m pytest -q` passes with updated fixtures.

## Testing Strategy
- Determinism: run pipeline twice; diff outputs.
- Security: upload size limits, traversal protections, offline network denial.
- AutoPlanner + lineage: snapshot planner output and trace query results.
- Queue/capacity: synthetic + real fixture parity with loose thresholds.
- Statistical controls: null dataset regression for low false positives.
- Tenancy/auth: isolation tests across uploads, runs, vectors, and API.
- Vector store: add/query/delete with deterministic ordering, cursor pagination, and isolation.
- PII governance: ensure anonymization only for explicit cloud/API egress.

## Potential Risks & Gotchas
- Determinism drift from third‑party libs; enforce seeding and stable ordering.
- Large datasets may stress SQLite; rely on chunked inserts and indexes.
- Stateless cursors can be large; mitigate by compact encoding + HMAC signing.
- Role inference ambiguity can cause false results; emit warnings and not‑applicable findings.
- PII anonymization must never regress into outbound payloads; keep tests strict.

## Rollback Plan
- Keep feature flags for Phase 2 to preserve Phase 1 behavior.
- Migrations remain additive; avoid destructive schema changes.
- Subprocess runner can fall back to in‑process execution for debugging (dev‑only).
