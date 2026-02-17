# Plan: Stat Redteam 2-6-26

**Generated**: 2026-02-06
**Estimated Complexity**: High

## Overview
Implement the recommendations in `docs/stat_redteam2-6-26.md` using the document’s own Phase 0-3 roadmap ordering, with an emphasis on deterministic, local-first behavior and test-gated changes.

This plan is written to be executed incrementally:
- Each sprint results in a runnable system where `.venv/bin/python -m pytest -q` passes.
- Each sprint adds explicit regression tests for the new behavior before moving on.

## Prerequisites
- Python 3.11+ with a working venv (repo uses `.venv/` in practice).
- Ability to run the full test suite:
  - Primary gate: `python -m pytest -q` (repo policy)
  - Practical gate in this environment: `.venv/bin/python -m pytest -q`

## Sprint 0: Phase 0 Safety Baselines
**Goal**: Ship the Phase 0 items listed in the doc (atomic artifact writes, config defaulting/validation, DAG validation, plugin validation CLI, UI artifact hardening headers, and test coverage).
**Demo/Validation**:
- Run: `.venv/bin/python -m pytest -q`
- Verify: reports/artifacts write atomically; schema defaults apply deterministically; DAG cycles are actionable; UI responses include security headers; CLI validation works.

### Task 0.1: Atomic Writes (FND-02)
- **Location**: `src/statistic_harness/core/utils.py`, `src/statistic_harness/core/report.py`
- **Description**: Add atomic write helpers and route JSON + primary markdown artifacts through them.
- **Complexity**: 3/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - JSON artifacts are never partially written (temp + replace).
  - Report markdown artifacts are written atomically.
- **Validation**:
  - Unit tests that simulate `os.replace` failure keep the original content.

### Task 0.2: Deterministic Config Resolution (EXEC-06)
- **Location**: `src/statistic_harness/core/plugin_manager.py`, `src/statistic_harness/core/pipeline.py`
- **Description**: Apply JSONSchema `default` values deterministically before validation and execution; ensure validated/resolved settings are what plugins receive.
- **Complexity**: 4/10
- **Dependencies**: Task 0.1
- **Acceptance Criteria**:
  - Missing optional fields are defaulted deterministically.
  - Resolved config validates against the schema.
- **Validation**:
  - Unit tests with a schema fixture that includes nested defaults.

### Task 0.3: DAG Cycle Errors + Dependency Preflight (EXEC-07)
- **Location**: `src/statistic_harness/core/pipeline.py`
- **Description**: Improve cycle detection errors with a cycle path + edge list; add dependency preflight that can fail closed on missing deps.
- **Complexity**: 5/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - Cycles fail with an actionable message including at least one cycle path and the relevant edges.
  - Missing deps are detected and recorded before proceeding.
- **Validation**:
  - Unit tests for cycle error formatting and missing-dep detection.

### Task 0.4: CLI `plugins validate` (EXT-02)
- **Location**: `src/statistic_harness/cli.py`, `src/statistic_harness/core/plugin_manager.py`
- **Description**: Add `stat-harness plugins validate` to validate manifests/schemas and import + instantiate plugin entrypoints (single plugin or all).
- **Complexity**: 4/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - Returns nonzero if any plugin fails validation/import/smoke.
  - Supports `--plugin-id` for fast validation.
- **Validation**:
  - Unit test validates a known-good plugin (`profile_basic`).

### Task 0.5: Security Headers For UI + Artifact Serving (SEC-05)
- **Location**: `src/statistic_harness/ui/server.py`
- **Description**: Add a single middleware to apply CSP + basic hardening headers to all responses.
- **Complexity**: 2/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - Responses include CSP and `nosniff`/frame/referrer hardening headers.
- **Validation**:
  - Unit test of header application helper (no external HTTP client dependency).

## Sprint 1: Phase 1 Provenance + Crash Recovery
**Goal**: Implement Phase 1 items from the doc: crash-safe run lifecycle, upload CAS hardening, startup integrity checks, backup/restore, canonical run manifests, execution fingerprints, event telemetry, and diagnostics bundles.
**Demo/Validation**:
- Run: `.venv/bin/python -m pytest -q`
- Verify: kill/restart behavior is deterministic; manifests exist; hashes match; corruption is detected; diag bundle is self-contained.

### Task 1.1: Crash-Safe Run Directories (FND-01)
- **Location**: `src/statistic_harness/core/pipeline.py`, `src/statistic_harness/core/storage.py`
- **Description**: Add an atomic run dir creation/journaling scheme so partial runs are marked `ABORTED` and never masquerade as complete.
- **Complexity**: 7/10
- **Dependencies**: Task 0.1
- **Acceptance Criteria**:
  - Aborted runs are discoverable and don’t corrupt later runs.
- **Validation**:
  - Integration test that simulates mid-run interruption.

### Task 1.2: Upload CAS Verify/Quarantine/Refcount (FND-03)
- **Location**: `src/statistic_harness/ui/server.py`, `src/statistic_harness/core/storage.py`, `src/statistic_harness/core/migrations.py`
- **Description**: Quarantine uploads before “promote”, verify SHA256-on-disk, and track references for safe GC.
- **Complexity**: 8/10
- **Dependencies**: Task 0.1
- **Acceptance Criteria**:
  - Corruption is detected; unreferenced blobs can be GC’d safely.
- **Validation**:
  - Corruption fixture test.

### Task 1.3: Startup Integrity Check + Backup/Restore (FND-04, FND-05)
- **Location**: `src/statistic_harness/core/storage.py`, `src/statistic_harness/cli.py`
- **Description**: Add integrity check modes and add CLI commands for SQLite online backup/restore with retention.
- **Complexity**: 7/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - Corruption fails closed; backup/restore roundtrips deterministically.
- **Validation**:
  - Corrupted DB fixture tests; backup/restore integration tests.

### Task 1.4: Canonical Run Manifest + Execution Fingerprints (META-01, META-03)
- **Location**: `docs/run_manifest.schema.json`, `src/statistic_harness/core/pipeline.py`, `src/statistic_harness/core/report.py`
- **Description**: Persist `run_manifest.json` per run containing input hash, resolved configs, plugin list + versions + code hashes, flags, seeds, and artifact registry hashes.
- **Complexity**: 8/10
- **Dependencies**: Task 0.1, Task 0.2
- **Acceptance Criteria**:
  - Every completed run includes a valid `run_manifest.json`.
- **Validation**:
  - Schema validation tests and fixture run assertions.

## Sprint 2: Phase 2 Replay + UX Surfacing
**Goal**: Deterministic replay and caching keyed by fingerprints, plus UI surfacing of the provenance, evidence registry, and run timeline.
**Demo/Validation**:
- Deterministic rerun matches artifact hashes.
- UI and report show resolved seeds + sampling metadata and evidence tags.

## Sprint 3: Phase 3 Extensions Lifecycle + Privacy Options
**Goal**: Offline extension lifecycle, capability model, sandbox write minimization, privacy toggles, and performance streaming.
**Demo/Validation**:
- Plugins cannot write outside run dir unless explicitly allowed.
- Streaming dataset access prevents OOM on large datasets.

## Testing Strategy
- Unit tests for: atomic writes, schema default application, DAG validation formatting, security headers helper.
- Integration tests for: crash recovery, backup/restore, manifest generation and hashing, replay verification.
- Keep the full suite as a hard gate each sprint.

## Potential Risks & Gotchas
- Enforcing plugin dependencies can explode compute cost and can change outputs; dependency declarations must reflect *true* hard requirements.
- Adding CSP can break templates if they rely on inline script/style; start permissive and tighten incrementally with tests.
- Defaulting via JSONSchema must avoid overreaching into complex schemas (`oneOf`/`anyOf`) without a deterministic selection rule.

## Rollback Plan
- Atomic write changes can be reverted by restoring direct writes in `write_json` and `write_report`.
- Config defaulting can be rolled back by switching pipeline back to `validate_config()` without applying defaults.
- Security headers can be rolled back by removing the middleware and keeping the path traversal guards.

