# Plan: Cohesive Harness Roadmap (Plans + Phase 2)

**Generated**: February 2, 2026
**Estimated Complexity**: High

## Overview
This roadmap consolidates the three plan docs (recommendations, kernel/plugin stability, queue-delay plugins) into one cohesive sequence and adds Phase 2 deliverables (multi-tenant + auth + vector store). The repo already implements many items from those plans; this plan focuses on remaining deltas, verification hardening, and the Phase 2 buildout. Phase 2 work is gated on completing the queue/capacity suite and a fully passing test run. Emphasis stays balanced across performance, security, accuracy, and citeability.

## Prerequisites
- Python 3.11+, `python -m pytest -q` passes on the current baseline.
- Phase 2 tenancy model: shared DB with `tenant_id` columns (confirmed).
- Auth model: local users + session tokens AND API keys (confirmed).
- SQLite vector store: `sqlite-vec` available as a builtin extension.
- Queue/capacity plugin suite must complete and all tests must pass before Phase 2 work begins.
- Quorum Upstream fixture available (real fixture).

## Sprint 1: Determinism + Kernel Hardening Deltas
**Goal**: Close remaining determinism gaps and security guardrails not fully covered by existing code/tests.
**Demo/Validation**:
- Run pipeline twice with same seed; `report.json` is byte-stable.
- Uploads enforce file size limits; offline guardrails remain intact.

### Task 1.1: Deterministic runtime seeding in subprocess runner
- **Location**: `src/statistic_harness/core/plugin_runner.py`, `src/statistic_harness/core/pipeline.py`
- **Description**: Ensure deterministic runtime by seeding `random`, NumPy (if available), and setting `PYTHONHASHSEED`, `TZ=UTC`, `LC_ALL=C` for subprocess execution. Pass a sanitized env into `subprocess.run`.
- **Complexity**: 6
- **Dependencies**: None
- **Acceptance Criteria**:
  - Identical runs on the same dataset produce byte-identical `report.json`.
- **Validation**:
  - New determinism test that runs the same pipeline twice and diffs output.

### Task 1.2: Stable ordering + canonical JSON rounding policy
- **Location**: `src/statistic_harness/core/utils.py`, `src/statistic_harness/core/report.py`, `src/statistic_harness/core/pipeline.py`
- **Description**: Add a canonical JSON helper that stabilizes float formatting (fixed precision) and ensure findings/metrics are consistently ordered in report output.
- **Complexity**: 5
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Report ordering is stable across runs even with unordered inputs.
- **Validation**:
  - Snapshot test for report ordering and canonical JSON output.

### Task 1.3: Upload size limits + explicit guardrails
- **Location**: `src/statistic_harness/ui/server.py`, `src/statistic_harness/core/utils.py`, `tests/test_security_paths.py`
- **Description**: Enforce max upload size (configurable), and add explicit tests for size violations and traversal attempts. Use existing `file_size_limit` helper.
- **Complexity**: 4
- **Dependencies**: None
- **Acceptance Criteria**:
  - Oversized uploads fail with a deterministic error.
- **Validation**:
  - New/extended security tests for file size enforcement.

### Task 1.4: Performance smoke regression test
- **Location**: `tests/test_performance_smoke.py`
- **Description**: Add a lightweight benchmark that runs a small fixture through the subprocess path and checks elapsed time within a loose threshold (enforced in CI).
- **Complexity**: 3
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Performance smoke test is stable on local dev machines.
- **Validation**:
  - `python -m pytest -q` passes including the performance smoke test.

## Sprint 2: Resource Budgets + PII Governance
**Goal**: Add explicit resource budget metadata and PII tagging/anonymization to improve citeability and governance.
**Demo/Validation**:
- Reports include budget metadata (even when unlimited).
- PII tags appear; any cloud/API egress (LLM or external service calls) is anonymized via entity hash table.

### Task 2.1: Resource budget metadata
- **Location**: `src/statistic_harness/core/types.py`, `src/statistic_harness/core/report.py`, `docs/report.schema.json`, `plugins/*/output.schema.json`
- **Description**: Add budget metadata to `PluginContext` and `PluginResult` (row limits, sampling flags, time/CPU budget) with defaults of “unlimited”. Emit in report output and update plugin output schemas.
- **Complexity**: 5
- **Dependencies**: None
- **Acceptance Criteria**:
  - All plugin outputs include budget fields with defaults.
- **Validation**:
  - Schema + unit tests for budget defaults in report output.

### Task 2.2: PII detection + tagging
- **Location**: `plugins/profile_basic/plugin.py`, `src/statistic_harness/core/migrations.py`, `src/statistic_harness/core/storage.py`
- **Description**: Add PII heuristics (email/phone/SSN/address) and store tags plus entity hashes. Introduce a `pii_entities` table that records raw value, deterministic hash, type, and tenant_id. Use a per-tenant salt so hashes are stable within a tenant and non-linkable across tenants.
- **Complexity**: 6
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - PII columns are tagged deterministically for fixtures.
- **Validation**:
  - New tests with a PII fixture.

### Task 2.3: PII anonymization policy for cloud egress
- **Location**: `plugins/llm_prompt_builder/*`, `src/statistic_harness/core/report.py`, `docs/report.schema.json`
- **Description**: Define “offsystem egress” as explicit cloud/API calls (LLM or other external services). Before any such call, replace raw PII values with entity hashes and type labels using the `pii_entities` table. Preserve raw PII internally and in local reports.
- **Complexity**: 4
- **Dependencies**: Task 2.2
- **Acceptance Criteria**:
  - LLM prompt builder never emits raw PII in outbound payloads.
- **Validation**:
  - Tests that scan outbound payloads for raw PII patterns on fixtures.

### Task 2.4: LLM egress safety checks
- **Location**: `plugins/llm_prompt_builder/*`, `tests/test_llm_prompt_builder.py`
- **Description**: Add unit/integration tests ensuring LLM prompt payloads are anonymized via entity hashes and include PII type labels when applicable.
- **Complexity**: 4
- **Dependencies**: Task 2.3
- **Acceptance Criteria**:
  - Tests fail if any raw PII appears in outbound payloads.
- **Validation**:
  - `python -m pytest -q` passes with new LLM egress checks.

## Sprint 3: Statistical Controls Library
**Goal**: Reduce false discoveries across analysis plugins and improve confidence scoring.
**Demo/Validation**:
- Null datasets yield low false-positive findings.

### Task 3.1: Implement statistical controls helpers
- **Location**: `src/statistic_harness/core/stat_controls.py`
- **Description**: Add effect size calculations, multiple-testing corrections (BH/FDR), and confidence scoring utilities with deterministic behavior.
- **Complexity**: 6
- **Dependencies**: None
- **Acceptance Criteria**:
  - Deterministic outputs for fixed inputs and seeds.
- **Validation**:
  - Unit tests on known distributions.

### Task 3.2: Integrate controls into analysis plugins
- **Location**: `plugins/analysis_*/*`, `plugins/*/output.schema.json`
- **Description**: Apply corrections and add confidence/score fields to findings and metrics; update schemas accordingly.
- **Complexity**: 7
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Findings include confidence fields; tests remain deterministic.
- **Validation**:
  - Plugin regression tests with null datasets.

## Sprint 4: Queue/Delay Plugin Suite (Core Set)
**Goal**: Complete the missing queue/capacity plugins for the core queue-delay decomposition set.
**Demo/Validation**:
- New plugins run on a synthetic event-log fixture and emit expected findings with evidence.

### Task 4.1: Dependency-resolution join plugin
- **Location**: `plugins/analysis_dependency_resolution_join/`, `plugins/analysis_dependency_resolution_join/*.schema.json`
- **Description**: Join dependency end timestamps to compute START-DEP_END and emit lag distribution stats.
- **Complexity**: 6
- **Dependencies**: Sprint 2 (role inference already present)
- **Acceptance Criteria**:
  - Emits deterministic lag distribution metrics.
- **Validation**:
  - Unit test with synthetic dependency fixture.

### Task 4.2: Tail isolation plugin
- **Location**: `plugins/analysis_tail_isolation/`
- **Description**: Filter eligible-wait > threshold and attribute tail to process/module/user/sequence with evidence.
- **Complexity**: 5
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Tail findings include evidence row IDs.
- **Validation**:
  - Plugin test with threshold fixture.

### Task 4.3: Percentile analysis plugin
- **Location**: `plugins/analysis_percentile_analysis/`
- **Description**: Compute p50/p95/p99 for eligible wait and completion time per process/module.
- **Complexity**: 4
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Percentiles are deterministic and tolerance-checked.
- **Validation**:
  - Numeric tolerance tests in evaluation.

### Task 4.4: Attribution analysis plugin
- **Location**: `plugins/analysis_attribution/`
- **Description**: Aggregate eligible wait and tail counts by PROCESS_ID, MODULE_CD, USER_ID with deterministic ranking.
- **Complexity**: 4
- **Dependencies**: Task 4.1
- **Acceptance Criteria**:
  - Deterministic ranking with evidence slices.
- **Validation**:
  - Fixture test for top contributor stability.

### Task 4.5: Determinism discipline plugin
- **Location**: `plugins/analysis_determinism_discipline/`
- **Description**: Inspect plugin outputs for measurement_type usage and emit violations if missing or inconsistent.
- **Complexity**: 5
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - Emits a summary finding and violations for mis-labeled outputs.
- **Validation**:
  - Unit test with mocked plugin outputs.

### Task 4.6: Update planner capability tags for new plugins
- **Location**: `plugins/*/plugin.yaml`, `src/statistic_harness/core/planner.py`
- **Description**: Add capability tags for newly added queue plugins and ensure deterministic planner selection.
- **Complexity**: 3
- **Dependencies**: Tasks 4.1–4.5
- **Acceptance Criteria**:
  - Planner selects queue plugins only when event-log roles are present.
- **Validation**:
  - Planner selection test on event-log vs numeric fixtures.

## Sprint 5: Queue/Capacity Modeling + Parity Suite
**Goal**: Implement advanced modeled plugins and parity checks with Quorum-style fixtures.
**Demo/Validation**:
- Full queue plugin suite passes evaluation parity on the Quorum fixture.

### Task 5.1: Chain makespan plugin
- **Location**: `plugins/analysis_chain_makespan/`
- **Description**: Compute sequence makespan, idle gaps, and critical path effects with evidence.
- **Complexity**: 7
- **Dependencies**: Sprint 4
- **Acceptance Criteria**:
  - Outputs per-sequence makespan metrics.
- **Validation**:
  - Synthetic fixture with known makespan.

### Task 5.2: Concurrency reconstruction plugin
- **Location**: `plugins/analysis_concurrency_reconstruction/`
- **Description**: Reconstruct concurrency via START/END overlaps; output distribution and peak concurrency.
- **Complexity**: 7
- **Dependencies**: Sprint 4
- **Acceptance Criteria**:
  - Deterministic concurrency metrics with evidence.
- **Validation**:
  - Fixture with known overlaps.

### Task 5.3: Capacity scaling model plugin
- **Location**: `plugins/analysis_capacity_scaling/`
- **Description**: Apply modeled scaling to eligible-wait bucket; emit modeled reduction stats with explicit assumptions.
- **Complexity**: 6
- **Dependencies**: Tasks 5.1–5.2
- **Acceptance Criteria**:
  - Modeled outputs labeled with assumptions and measurement_type.
- **Validation**:
  - Evaluation checks modeled metrics within tolerance.

### Task 5.4: Quorum parity evaluation suite
- **Location**: `tests/fixtures/`, `tests/test_quorum_evaluation.py`, `tests/plugins/*`
- **Description**: Add/extend fixture and expected metrics/findings to enforce parity for queue plugins.
- **Complexity**: 6
- **Dependencies**: Sprint 4–5 plugins
- **Acceptance Criteria**:
  - Evaluation passes only when metrics are within tolerance.
- **Validation**:
- `python -m pytest -q` passes with parity suite.

## Phase Gate: Phase 1 Completion
**Requirement**: Do not begin Sprint 6 until Sprint 1–5 are complete and `python -m pytest -q` passes with the queue/capacity parity suite enabled.

## Sprint 6: Phase 2 Tenancy + Auth Foundation
**Goal**: Implement multi-tenant isolation and authentication while preserving Phase 1 local-only constraints.
**Demo/Validation**:
- Two tenants can run analyses without data leakage.
- Auth gates UI/API and CLI actions deterministically.

### Task 6.1: Tenancy model + migration plan
- **Location**: `src/statistic_harness/core/tenancy.py`, `src/statistic_harness/core/migrations.py`, `docs/references.md`
- **Description**: Implement shared-DB tenancy with `tenant_id` columns. Add a TenantContext (tenant_id, appdata_root, db_path) and migration path for existing data.
- **Complexity**: 8
- **Dependencies**: None
- **Acceptance Criteria**:
  - Default tenant created for existing data; tenant context resolves DB + appdata paths deterministically.
- **Validation**:
  - Migration test that upgrades an existing DB into tenant-aware state.

### Task 6.2: Tenant-aware storage and appdata isolation
- **Location**: `src/statistic_harness/core/storage.py`, `src/statistic_harness/core/utils.py`, `src/statistic_harness/core/pipeline.py`
- **Description**: Scope all storage queries and file paths by tenant (tenant_id column or per-tenant DB + appdata root). Update pipeline initialization accordingly.
- **Complexity**: 8
- **Dependencies**: Task 6.1
- **Acceptance Criteria**:
  - Tenant A cannot access Tenant B datasets or files.
- **Validation**:
  - New tests for tenant isolation and path scoping.

### Task 6.3: Auth tables + middleware
- **Location**: `src/statistic_harness/core/migrations.py`, `src/statistic_harness/ui/server.py`, `src/statistic_harness/ui/templates/*`, `src/statistic_harness/cli.py`
- **Description**: Add users/tenants/memberships and token/API key storage; implement login + session auth and API key auth for UI/API and CLI operations (with strong password hashing and session/token rotation).
- **Complexity**: 8
- **Dependencies**: Tasks 6.1, 6.2
- **Acceptance Criteria**:
  - Auth required for protected routes; unauthenticated requests fail.
- **Validation**:
  - UI/API auth tests with valid/invalid tokens.

### Task 6.4: Tenant-aware uploads + runs
- **Location**: `src/statistic_harness/ui/server.py`, `src/statistic_harness/core/storage.py`
- **Description**: Ensure uploads, runs, and artifacts are stored under tenant-specific namespaces and recorded with tenant_id.
- **Complexity**: 7
- **Dependencies**: Task 6.2
- **Acceptance Criteria**:
  - Upload and run records are isolated by tenant.
- **Validation**:
  - Integration tests with two tenants performing uploads and runs.

### Task 6.5: Admin bootstrap + tenant/user management flows
- **Location**: `src/statistic_harness/ui/server.py`, `src/statistic_harness/ui/templates/*`, `src/statistic_harness/cli.py`
- **Description**: Add bootstrap flow for initial admin creation, tenant creation, user invites, token/API key issuance, and revocation/reset workflows.
- **Complexity**: 7
- **Dependencies**: Task 6.3
- **Acceptance Criteria**:
  - New tenants/users can be created and revoked without DB edits.
- **Validation**:
  - UI/CLI tests for admin bootstrap and token revocation.

### Task 6.6: Phase 2 feature flags
- **Location**: `src/statistic_harness/core/utils.py`, `src/statistic_harness/ui/server.py`, `src/statistic_harness/cli.py`
- **Description**: Add config flags to gate tenancy/auth/vector store behavior and preserve Phase 1 defaults. Wire flags through UI/API/CLI and add tests for flag on/off behavior.
- **Complexity**: 5
- **Dependencies**: Task 6.1
- **Acceptance Criteria**:
  - Phase 1 behavior remains unchanged when flags are off.
- **Validation**:
  - Tests covering flag-disabled and flag-enabled modes.

## Sprint 7: Phase 2 Vector Store
**Goal**: Implement a local vector store for embeddings with tenant isolation and deterministic queries.
**Demo/Validation**:
- Add/query/delete vectors locally with deterministic nearest-neighbor results.

### Task 7.1: Vector store interface + sqlite-vec backend
- **Location**: `src/statistic_harness/core/vector_store.py`, `src/statistic_harness/core/migrations.py`
- **Description**: Replace the placeholder with a concrete API (`add`, `query`, `delete`) backed by builtin `sqlite-vec`. Store vectors per tenant and enforce deterministic ordering.
- **Complexity**: 7
- **Dependencies**: Sprint 6
- **Acceptance Criteria**:
  - Vector store supports add/query/delete with deterministic ordering.
- **Validation**:
  - Unit tests for vector operations and ordering.

### Task 7.2: Embedding pipeline integration
- **Location**: `plugins/llm_prompt_builder/*`, `plugins/*/plugin.yaml`, optional new `plugins/analysis_vector_index/`
- **Description**: Add a plugin (or extend existing) that generates local embeddings (offline; e.g., TF-IDF or deterministic hashing) and indexes them in the vector store for retrieval.
- **Complexity**: 6
- **Dependencies**: Task 7.1
- **Acceptance Criteria**:
  - Vector index is created and queryable for a run.
- **Validation**:
  - Integration test that indexes findings and retrieves expected neighbors.

### Task 7.3: Tenant isolation for vector store
- **Location**: `src/statistic_harness/core/vector_store.py`, `tests/test_vector_store.py`
- **Description**: Enforce tenant scoping in vector store queries and deletes.
- **Complexity**: 5
- **Dependencies**: Task 7.1
- **Acceptance Criteria**:
  - Tenant A cannot query or delete Tenant B vectors.
- **Validation**:
  - Isolation tests for vector operations.

## Sprint 8: Documentation + Regression Suite
**Goal**: Update documentation and ensure all new capabilities are covered by tests.
**Demo/Validation**:
- `python -m pytest -q` green; docs describe Phase 2 and queue plugins.

### Task 8.1: Documentation updates
- **Location**: `README.md`, `docs/plugin_dev_guide.md`, `docs/references.md`
- **Description**: Document determinism, budgets, PII policy, queue plugins, multi-tenant/auth model, and vector store usage.
- **Complexity**: 3
- **Dependencies**: Sprints 1–7
- **Acceptance Criteria**:
  - Docs match actual behavior and configs.
- **Validation**:
  - Manual doc review; spot-check with example commands.

### Task 8.2: Integration/regression tests expansion
- **Location**: `tests/*`
- **Description**: Add integration tests that cover queue plugin suite, tenancy isolation, vector store, and PII redaction.
- **Complexity**: 6
- **Dependencies**: Sprints 2–7
- **Acceptance Criteria**:
  - New tests pass and cover Phase 2 behavior.
- **Validation**:
  - `python -m pytest -q` passes.

### Task 8.3: Update migration fixtures for new schema versions
- **Location**: `tests/fixtures/db/*`, `tests/test_migrations.py`
- **Description**: Generate new golden DB fixtures and update expectations for added tables/columns (PII tags, tenancy, vector store).
- **Complexity**: 4
- **Dependencies**: Sprints 2, 6, 7
- **Acceptance Criteria**:
  - Migration tests pass for all versions including new schemas.
- **Validation**:
  - `python -m pytest -q` passes `test_migrations.py`.

## Testing Strategy
- Determinism tests: run pipeline twice with identical seeds and diff outputs.
- Security tests: upload size limits, path traversal, offline network guards.
- Queue/capacity plugin tests: synthetic fixtures + Quorum parity evaluation.
- Governance tests: PII tagging plus cloud/API egress anonymization via entity hash table; budget metadata presence.
- Tenancy/auth tests: tenant isolation across uploads, runs, vector queries.
- Vector store tests: add/query/delete with deterministic ordering and isolation using `sqlite-vec`.

## Potential Risks & Gotchas
- Determinism may be violated by third-party libs; enforce seeding and canonical ordering.
- Queue/capacity plugins rely on inferred roles; ambiguous inference should emit warnings and not fail.
- PII anonymization must avoid leaking raw values via cloud/API egress; enforce prompt sanitization and audit tests.
- `sqlite-vec` behavior/performance may vary by build; keep queries bounded and deterministic.
- Tenant-scoped hashing must be stable for each tenant and salted to avoid cross-tenant linkage.

## Rollback Plan
- Gate Phase 2 features behind config flags to keep Phase 1 behavior stable.
- Keep migrations additive and reversible where feasible (avoid destructive schema changes).
- Retain legacy code paths for report generation until new schema changes are validated.
