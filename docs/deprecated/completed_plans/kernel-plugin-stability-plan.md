# Plan: Kernel-Plugin Stability & Isolation

**Generated**: January 31, 2026
**Estimated Complexity**: High

## Overview
Implement the 10 stability recommendations with a plugin-first architecture: the kernel only orchestrates logging, plugin management, UI, and storage while all default logic/config lives in plugins (ingest, profile/validate, planner, transform, analysis, report, llm/offline prompt builder). Add OS-process isolation for plugins, strict schema validation, sandboxing, audit logging, deterministic parallel execution, template mapping presets, filtered combined analysis runs, and explicit determinism controls (run_seed propagation, canonical JSON, stable ordering). Enforce Phase 1 offline execution and citeable outputs (report.md + report.json schema validation, evaluator harness). Emphasize the four pillars equally: performance, security, accuracy, citeability.

## Prerequisites
- Python 3.11+ (existing project).
- SQLite with JSON1 (already in use).
- Ability to run `python -m pytest -q` locally.
- Local-only runtime: no network calls in Phase 1 (plugins and UI).

## Sprint 1: Plugin Contracts & Registry Hardening
**Goal**: Make plugin metadata/schema mandatory and validated; establish stable contracts for inputs/outputs.
**Demo/Validation**:
- `python -m pytest -q` with new validation tests.
- `stat-harness list-plugins` shows schema metadata.

### Task 1.1: Define plugin manifest schema + required fields
- **Location**: `docs/plugin_manifest.schema.json` (new), `src/statistic_harness/core/plugin_manager.py`
- **Description**: Add a JSON schema for plugin manifests (`plugin.yaml`) requiring fields: `id`, `name`, `version`, `type` (ingest/profile/analysis/report/llm/planner/transform), `entrypoint`, `capabilities`, `config_schema`, `output_schema`, `sandbox` (no-network, fs-allowlist). Update `PluginManager` to validate against schema.
- **Complexity**: 6
- **Dependencies**: None
- **Acceptance Criteria**:
  - Invalid manifests fail fast with clear errors.
  - All existing plugins include required fields.
- **Validation**:
  - Unit test for manifest validation.

### Task 1.2: Add per-plugin config/output schemas
- **Location**: `plugins/*/config.schema.json` (new), `plugins/*/output.schema.json` (new)
- **Description**: Add JSON schemas for plugin input config and output structure. Keep schemas minimal but explicit for determinism/citeability (e.g., finding kinds, evidence and citation fields, stable ordering expectations).
- **Complexity**: 7
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Kernel validates config and output for each plugin run.
- **Validation**:
  - Tests cover schema validation failure cases.

### Task 1.3: Update plugin manifests to reference schemas
- **Location**: `plugins/*/plugin.yaml`
- **Description**: Add `config_schema` and `output_schema` references to each plugin manifest, plus sandbox flags.
- **Complexity**: 5
- **Dependencies**: Task 1.1, 1.2
- **Acceptance Criteria**:
  - All manifests pass validation.
- **Validation**:
  - `test_plugin_discovery` + new manifest tests.

## Sprint 2: OS-Process Plugin Runner + Sandbox
**Goal**: Run each plugin in a separate OS process with deterministic I/O, strict sandbox, and audit logging.
**Demo/Validation**:
- Plugins run via subprocess and return identical results to current flow.
- Plugins cannot access network or files outside allowlist.

### Task 2.1: Implement plugin runner subprocess
- **Location**: `src/statistic_harness/core/plugin_runner.py` (new), `src/statistic_harness/core/pipeline.py`
- **Description**: Add a subprocess runner that reads a request JSON (run context + settings), executes plugin entrypoint in a separate Python process, and writes response JSON. Kernel orchestrates runs via subprocess, not in-process.
- **Complexity**: 8
- **Dependencies**: Sprint 1 schemas
- **Acceptance Criteria**:
  - All plugins run in isolated processes.
- **Validation**:
  - Integration test runs pipeline and verifies reports.

### Task 2.2: Sandboxing (no network + FS allowlist)
- **Location**: `src/statistic_harness/core/plugin_runner.py`
- **Description**: Block network by monkeypatching sockets/HTTP before plugin load; restrict file access to run dir + appdata + plugins dir by guarding `open()`/`Path` usage where feasible and validating file paths on kernel API boundaries.
- **Complexity**: 7
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Network calls fail.
  - Writes outside allowlist fail.
- **Validation**:
  - New tests with a "malicious" plugin fixture.

### Task 2.3: Execution audit table + runtime metrics
- **Location**: `src/statistic_harness/core/migrations.py`, `src/statistic_harness/core/storage.py`, `src/statistic_harness/core/plugin_runner.py`
- **Description**: Add `plugin_executions` table capturing: started/ended timestamps, duration, CPU user/system, max RSS, rows read/written, rows output, warnings count, exit status, stdout/stderr (truncated). Populate from runner and context wrappers.
- **Complexity**: 8
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Every plugin run writes an audit row.
- **Validation**:
  - Tests verify audit rows exist and are deterministic.

### Task 2.4: Deterministic execution context + run_seed propagation
- **Location**: `src/statistic_harness/core/plugin_runner.py`, `src/statistic_harness/core/pipeline.py`, `src/statistic_harness/core/types.py`
- **Description**: Propagate `run_seed` to all plugin runs; seed `random` and optional NumPy/Scipy if present; set `PYTHONHASHSEED`, `TZ=UTC`, `LC_ALL=C`, and stable locale; canonicalize JSON serialization (sorted keys, stable float rounding); ensure deterministic glob/file ordering in any kernel helpers.
- **Complexity**: 7
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Identical runs on the same dataset produce byte-stable `report.json`.
- **Validation**:
  - Determinism tests run pipeline twice and diff outputs.

### Task 2.5: Phase 1 offline enforcement (kernel + UI)
- **Location**: `src/statistic_harness/ui/server.py`, `src/statistic_harness/core/plugin_runner.py`, `tests/test_offline.py` (new)
- **Description**: Bind UI to localhost only, prevent external assets, and add kernel-level network deny checks so no runtime code can reach the network. Enforce hard-fail by default with a dev-only override flag (default off). Add tests that attempts to access external URLs fail.
- **Complexity**: 6
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Network calls fail in plugins and UI runtime paths.
- **Validation**:
  - Offline tests pass under `python -m pytest -q`.

## Sprint 3: Deterministic Parallel Execution
**Goal**: Run independent plugins in parallel while keeping deterministic ordering and results.
**Demo/Validation**:
- Multiple analysis plugins run concurrently without altering report ordering or contents.

### Task 3.1: DAG-aware parallel scheduling
- **Location**: `src/statistic_harness/core/pipeline.py`
- **Description**: Group plugins by dependency level (toposort layers) and run each layer in parallel via subprocess runner; store results sorted by plugin_id in report output.
- **Complexity**: 7
- **Dependencies**: Sprint 2 runner
- **Acceptance Criteria**:
  - Parallel execution matches sequential outputs.
- **Validation**:
  - Integration test compares outputs between sequential and parallel modes.

### Task 3.2: Deterministic output ordering
- **Location**: `src/statistic_harness/core/report.py`
- **Description**: Ensure report assembly orders plugins deterministically (sorted by id) and evidence rows sorted consistently.
- **Complexity**: 4
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - `report.json` stable across runs.
- **Validation**:
  - Snapshot test on report outputs.

### Task 3.3: Performance regression benchmarks (lightweight)
- **Location**: `tests/test_performance_smoke.py` (new), `tools/benchmarks/*` (optional)
- **Description**: Add a small performance smoke test to measure subprocess overhead and parallel speedups on a tiny fixture. Keep thresholds loose to catch regressions without blocking normal runs.
- **Complexity**: 3
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Performance smoke test is stable on local dev.
- **Validation**:
  - `python -m pytest -q` passes with the smoke test enabled.

## Sprint 4: Kernel Minimization & Plugin-First Defaults
**Goal**: Move default logic/config to plugins; kernel only orchestrates.
**Demo/Validation**:
- Auto runs depend on planner plugin; transform mappings always via transform plugin.

### Task 4.1: Planner plugin is the only auto-selection path
- **Location**: `src/statistic_harness/core/pipeline.py`, `plugins/planner_basic/*`
- **Description**: Remove kernel fallback planner logic; auto runs require planner plugin to produce selection. Manual selection still works.
- **Complexity**: 5
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Auto plan uses planner plugin output.
- **Validation**:
  - Unit test confirms planner output controls selection.

### Task 4.2: Transform template is a plugin-only step
- **Location**: `plugins/transform_template/*`, `src/statistic_harness/core/template.py`
- **Description**: Ensure template conversions are only triggered via transform plugin (kernel does not call apply_template directly).
- **Complexity**: 4
- **Dependencies**: Existing transform plugin
- **Acceptance Criteria**:
  - Mapping requests run through transform plugin.
- **Validation**:
  - Template conversion tests pass.

### Task 4.3: Enforce plugin-only pipeline steps
- **Location**: `src/statistic_harness/core/pipeline.py`, `src/statistic_harness/core/plugin_manager.py`, `plugins/*/plugin.yaml`
- **Description**: Ensure ingest, profile/validate, analysis, report, and llm steps are always resolved through plugins; kernel rejects missing step types and records errors in the report rather than crashing.
- **Complexity**: 6
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Pipeline fails closed with an error summary if a required step plugin is missing.
- **Validation**:
  - Unit test stubs missing plugin and asserts error reporting.

### Task 4.4: Security guardrails at kernel boundaries
- **Location**: `src/statistic_harness/core/storage.py`, `src/statistic_harness/ui/server.py`, `src/statistic_harness/core/utils.py`
- **Description**: Add path traversal protection for downloads/artifacts, file type/size validation for uploads, and explicit bans on `pickle`, `eval`, and shelling out in analysis paths. Add tests for each guardrail.
- **Complexity**: 7
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Invalid paths and file types are rejected deterministically.
- **Validation**:
  - Security tests cover traversal, size, and type checks.

## Sprint 5: Mapping Presets + Raw Format Learning
**Goal**: Persist mapping presets per raw format; allow user to choose among presets.
**Demo/Validation**:
- Raw format view shows mapping presets and notes.

### Task 5.1: Raw format mapping preset storage
- **Location**: `src/statistic_harness/core/migrations.py`, `src/statistic_harness/core/storage.py`
- **Description**: Add `raw_format_mappings` table and CRUD helpers; hash mapping JSON for determinism and dedupe.
- **Complexity**: 5
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Multiple presets saved per format.
- **Validation**:
  - Unit tests for preset save/list.

### Task 5.2: UI for mapping preset selection
- **Location**: `src/statistic_harness/ui/server.py`, `src/statistic_harness/ui/templates/raw_format.html`
- **Description**: Add form to store presets; list presets per format; allow copy/paste into template mapping.
- **Complexity**: 4
- **Dependencies**: Task 5.1
- **Acceptance Criteria**:
  - User can add/view presets in UI.
- **Validation**:
  - UI smoke tests (FastAPI TestClient).

## Sprint 6: Combined Analysis Filters (Project/Format/Date)
**Goal**: Allow filtered cross-project analysis without merging all data indiscriminately.
**Demo/Validation**:
- Combined analysis runs scoped by project or raw format.

### Task 6.1: Aggregate dataset filters
- **Location**: `src/statistic_harness/core/storage.py`, `src/statistic_harness/core/dataset_io.py`, `src/statistic_harness/core/template.py`
- **Description**: Extend aggregate datasets to carry filters (project_ids, dataset_ids, raw_format_ids, date range) in mapping_json; TemplateAccessor applies SQL WHERE filters deterministically.
- **Complexity**: 7
- **Dependencies**: Sprint 4
- **Acceptance Criteria**:
  - Aggregate runs only include filtered datasets.
- **Validation**:
  - New tests verifying filters.

### Task 6.2: UI for filtered combined runs
- **Location**: `src/statistic_harness/ui/server.py`, `src/statistic_harness/ui/templates/template.html`
- **Description**: Add filter fields (project_ids, raw_format_ids, date range) to combined run form.
- **Complexity**: 4
- **Dependencies**: Task 6.1
- **Acceptance Criteria**:
  - Combined run respects filters.
- **Validation**:
  - UI test + integration test.

## Sprint 7: Golden DB Fixtures + Migration Safety
**Goal**: Full migration coverage with golden DB fixtures.
**Demo/Validation**:
- Migrations from v1...vN pass and preserve data.

### Task 7.1: Golden DB generator and fixtures
- **Location**: `tests/fixtures/db/*`, `tests/test_migrations.py` (new)
- **Description**: Create small SQLite fixture DBs for each schema version; test upgrades to latest preserving runs/results/templates.
- **Complexity**: 6
- **Dependencies**: All migrations
- **Acceptance Criteria**:
  - Migrations pass across all versions.
- **Validation**:
  - `python -m pytest -q`.

## Sprint 8: Reporting + Evaluator Non-Negotiables
**Goal**: Always emit `report.md` and `report.json`, validate against schema, and add evaluator harness.
**Demo/Validation**:
- Integration test asserts both report files exist and `report.json` validates.
- Evaluator harness checks `ground_truth.yaml` against report output.

### Task 8.1: Report artifacts + schema validation
- **Location**: `src/statistic_harness/core/report.py`, `docs/report.schema.json`, `tests/test_report_outputs.py` (new)
- **Description**: Ensure every run produces `report.md` and `report.json` and validate JSON against schema. If a plugin fails, include error summaries and still write reports.
- **Complexity**: 6
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Reports are always generated, even on plugin errors.
- **Validation**:
  - Tests assert report files exist and schema validation passes.

### Task 8.2: Evaluator harness with `ground_truth.yaml`
- **Location**: `src/statistic_harness/core/evaluator.py` (new), `tests/test_evaluator.py` (new)
- **Description**: Implement evaluator that reads `ground_truth.yaml`, compares required attributes to `report.json` within tolerances (support absolute and relative), and fails deterministically.
- **Complexity**: 5
- **Dependencies**: Task 8.1
- **Acceptance Criteria**:
  - Evaluator test passes on synthetic fixture.
- **Validation**:
  - Integration test runs evaluator as part of pipeline test.

## Testing Strategy
- Unit tests for plugin manifest schema validation.
- Isolation tests for filesystem/network restrictions.
- Integration tests for parallel execution and deterministic output.
- Migration golden tests.
- Report artifact + schema validation tests.
- Evaluator harness tests with `ground_truth.yaml`.

## Potential Risks & Gotchas
- OS-level sandboxing may not fully prevent network if plugins import low-level modules; Python-level guards mitigate but are not foolproof.
- Parallel subprocess execution increases load; deterministic ordering and JSON serialization must be enforced.
- Schema adoption requires updating all existing plugins; rollout must be coordinated to avoid blocked runs.
- Offline enforcement must cover UI assets and any helper utilities.
- Determinism needs consistent run_seed propagation, RNG seeding, and locale/timezone control.

## Rollback Plan
- Keep runner feature flag to fall back to in-process execution during transition (dev-only).
- Preserve legacy plugin manifests in a migration branch if needed.
- Maintain previous schema migration steps; do not drop legacy tables.
