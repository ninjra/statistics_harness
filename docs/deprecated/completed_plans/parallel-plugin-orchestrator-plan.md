# Plan: Parallel Plugin Orchestrator + Hang Protection

**Generated**: 2026-02-05  
**Estimated Complexity**: High

## Overview
Introduce a deterministic, parallel plugin orchestrator that respects dependencies, limits heavy
plugins, and prevents hangs with watchdogs and retries. The orchestration must keep the 4 pillars:
performant (parallel scheduling), accurate (deterministic ordering + stable seeds), secure (no
network calls), and citable (no change to references generation). The gauntlet should complete even
if some plugins fail, with clear error logging for later fixes.

## Prerequisites
- Python 3.11+.
- No new runtime network dependencies required.
- Confirm `STAT_HARNESS_CLI_PROGRESS` stays local-only; no external egress.

## Sprint 1: Orchestrator Core (Parallel + Deterministic)
**Goal**: Add a parallel scheduler that respects dependency ordering, throttles heavy plugins, and
produces deterministic results ordering.
**Demo/Validation**:
- Run with `parallelism=cpu_count` and `heavy_parallelism=cpu_count/2`.
- Verify report ordering is stable across runs with same seed.

### Task 1.1: Orchestrator Module
- **Location**: `src/statistic_harness/core/orchestrator.py` (new), `src/statistic_harness/core/pipeline.py`
- **Description**: Implement a DAG-based scheduler:
  - Build dependency graph from `depends_on` + planner order.
  - Partition into `ready` sets; submit to worker pool.
  - Two worker pools: `normal` and `heavy`, with heavy cap at `ceil(cpu_count/2)`.
  - Maintain deterministic output by sorting plugin IDs when multiple are ready.
- **Complexity**: 8
- **Dependencies**: None
- **Acceptance Criteria**:
  - Respects dependencies and planner order.
  - Deterministic execution order when parallelism=1.
  - Result collection order deterministic (sorted by plugin_id).
- **Validation**:
  - New unit tests: DAG order + determinism.

### Task 1.2: Heavy Plugin Classification
- **Location**: `docs/plugin_manifest.schema.json`, `src/statistic_harness/core/plugin_manager.py`,
  `plugins/*/plugin.yaml`
- **Description**: Add optional manifest metadata for resource class:
  - `resource_class: light|normal|heavy` (default `normal`).
  - Use this to route to heavy pool.
- **Complexity**: 6
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Schema allows optional resource_class.
  - Existing plugins default to `normal`.
- **Validation**:
  - `tests/test_plugin_manifest_schema.py` updated and passing.

### Task 1.3: Pipeline Integration
- **Location**: `src/statistic_harness/core/pipeline.py`
- **Description**: Replace sequential loop with orchestrator when `settings["orchestrator"]["enabled"]=true`.
  Keep sequential mode for baseline/test reproducibility.
- **Complexity**: 7
- **Dependencies**: Tasks 1.1–1.2
- **Acceptance Criteria**:
  - Sequential and parallel modes both supported.
  - Output ordering stable across modes (sorted by plugin_id).
- **Validation**:
  - Integration tests compare report ordering.

## Sprint 2: Hang Protection + Intelligent Retries
**Goal**: Prevent indefinite runs without prematurely killing baseline execution.
**Demo/Validation**:
- Simulate a hung plugin; orchestrator kills it and continues.
- Retry policy applied and logged.

### Task 2.1: Watchdog for Hung Plugins
- **Location**: `src/statistic_harness/core/plugin_runner.py`, `src/statistic_harness/core/orchestrator.py`
- **Description**:
  - Use `subprocess.Popen` with polling.
  - Add heartbeat file per plugin (e.g., `run_dir/logs/<plugin_id>.heartbeat`).
  - Orchestrator monitors heartbeat and elapsed wall-time.
  - Default safety cap: configurable `orchestrator.hard_timeout_ms` (e.g., 6 hours) to avoid infinite loops.
- **Complexity**: 8
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Hung plugin is terminated and marked error.
  - Error recorded in `plugin_executions` with clear timeout reason.
- **Validation**:
  - Unit test with a fake long-running plugin.

### Task 2.2: Intelligent Retry Policy
- **Location**: `src/statistic_harness/core/orchestrator.py`, `src/statistic_harness/core/plugin_runner.py`
- **Description**:
  - Retry only for transient failures (e.g., timeout, resource errors).
  - Max retries configurable (`orchestrator.retry.max_attempts`, default 1).
  - Backoff delay configurable and deterministic.
- **Complexity**: 6
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Retries logged with reason.
  - Permanent failures do not block pipeline.
- **Validation**:
  - Tests for retry on timeout vs no retry on schema failure.

## Sprint 3: Progress + Observability
**Goal**: Transparent per-plugin progress and clearer logs for debugging.
**Demo/Validation**:
- CLI shows periodic status updates for running plugins.
- Logs written per plugin with timing and exit codes.

### Task 3.1: Status Emission
- **Location**: `src/statistic_harness/core/orchestrator.py`, `src/statistic_harness/core/pipeline.py`
- **Description**:
  - Periodic summary log: running/queued/complete/error counts.
  - Emit per-plugin start/stop messages with duration.
- **Complexity**: 5
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Log entries show clear progress without flooding.
- **Validation**:
  - Smoke test in `tests/test_pipeline_integration.py`.

## Sprint 4: Full Test + Gauntlet Baseline
**Goal**: Ensure everything passes and baseline run completes.
**Demo/Validation**:
- `python -m pytest -q` passes.
- Gauntlet completes with parallel mode and produces report artifacts.

### Task 4.1: Update Tests
- **Location**: `tests/`, `tests/plugins/`
- **Description**: Add tests for orchestrator behavior, timeout handling, and retry policy.
- **Complexity**: 6
- **Dependencies**: Sprints 1–3
- **Acceptance Criteria**:
  - Determinism test ensures same ordering of results.
- **Validation**:
  - Run full test suite.

### Task 4.2: Baseline Gauntlet
- **Location**: `scripts/run_gauntlet.ps1`, `scripts/run_gauntlet_latest.ps1`
- **Description**:
  - Add orchestrator settings to gauntlet (parallel enabled, heavy cap).
  - Execute baseline run and generate reports.
- **Complexity**: 4
- **Dependencies**: All prior sprints
- **Acceptance Criteria**:
  - `report.md`, `report.json`, slide kit generated.
- **Validation**:
  - Validate report schema and summary outputs.

## Testing Strategy
- Unit tests for scheduler (DAG order, heavy pool limit).
- Integration tests for pipeline parallel mode.
- Timeout/hang simulation tests.
- Deterministic output ordering tests.

## Potential Risks & Gotchas
- SQLite contention under parallel execution (need per-thread connections).
- Non-deterministic ordering if collection isn’t explicitly sorted.
- Heavy plugins could still starve smaller ones if queueing is naive.
- Hard timeouts may need tuning after baseline.

## Rollback Plan
- Disable orchestrator via config and revert pipeline to sequential mode.
- Remove resource_class metadata if needed.
