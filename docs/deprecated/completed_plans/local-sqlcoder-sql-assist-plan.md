# Plan: Local SQLCoder Plugin + SQL Assist Layer

**Generated**: 2026-02-10  
**Estimated Complexity**: High

## Overview
We want all plugins to benefit from custom, dataset-specific SQL while preserving the 4 pillars:
- **Performant**: avoid repeated full-table pandas loads; prefer SQLite-side projection/aggregation; enable streaming.
- **Accurate**: scan full data; allow plugins to filter/aggregate without forcing downsampling; keep multi-pass OK.
- **Secure**: no network; no shelling out; read-only SQL execution; strict allowlists.
- **Citable**: every LLM-produced (or human-produced) query is persisted with schema hash, params, rowcounts, and a result hash; reports can point to the exact query+evidence.

This plan intentionally builds on `two-million-rows-support-plan.md` (streaming, large-dataset policy, caching). The SQL Assist layer complements batching by letting plugins move expensive groupby/join logic into SQLite and by enabling a “generate once, replay forever” SQL pack workflow.

## Current Baseline (From `docs/plugin_data_access_matrix.*`)
- Plugins total: 139
- Plugins calling `ctx.dataset_loader()` (unbounded by default): 121
- Plugins using `ctx.dataset_iter_batches()`: 0
- Plugins using direct SQL today: 3 (`ingest_tabular`, `profile_eventlog`, `transform_normalize_mixed`)

## Decisions (Confirmed)
1. **Local model**: Use SQLCoder2 (`defog/sqlcoder2`) via vLLM. Network access is allowed for model/dependency download.
2. **Safety**: Plugins must treat the **normalized layer as read-only**. Controlled materialization is allowed, but only into plugin-owned scratch tables (not into normalized tables).
3. **Citeability**: SQL must be “generate once, replay forever”:
   - Persist generated SQL, schema snapshot/hash, decode config, params, rowcounts, and result hashes.
   - Re-running on the same normalized DB state must reproduce identical results.

## Implementation Matrix (Do This First, Avoid Duplicate Functions)
| Component | Location(s) | Purpose | Output Artifacts | Tests/Validation |
|---|---|---|---|---|
| SQL execution API (read-only state DB) | `src/statistic_harness/core/sql_assist.py` (new), `src/statistic_harness/core/storage.py` | Safe, deterministic SQL execution helpers; reads normalized DB; captures provenance | `artifacts/<plugin>/sql/*.sql`, `*.json` result manifests | Unit tests for validator + provenance; plugin smoke tests |
| Scratch DB for plugin-owned tables | `src/statistic_harness/core/plugin_runner.py`, `src/statistic_harness/core/storage.py` | Allow plugins to create their own tables/views without touching normalized tables | `run_dir/scratch.sqlite` + per-plugin manifests | Unit tests + integration “cannot write state DB” gate |
| SQL statement validator | `src/statistic_harness/core/sql_assist.py` | Fail-closed: only allow safe statements; forbid multi-statement, DDL/DML, PRAGMA/ATTACH | Included in manifests | Unit tests (denylist/allowlist cases) |
| Schema snapshot (normalized DB) | `src/statistic_harness/core/sql_schema_snapshot.py` (new) | Deterministic snapshot of tables/cols/indexes used for prompts and citations | `schema_snapshot.json` | Unit tests for stable ordering/hashes |
| SQL pack format | `docs/sql_pack.schema.json` (new) | Define “generated once” SQL pack contract (queries, params, expected outputs) | `sql_pack.json` | JSON schema validation tests |
| Prompt pack plugin (current) | `plugins/transform_sql_intents_pack_v1/*` | Writes schema snapshot + query intents used for text2sql generation | `schema_snapshot.json`, `sql_intents.json` | Integration: plugin runs, artifacts exist |
| vLLM generator script (operator step) | `scripts/generate_sql_pack_with_vllm.py` | Runs local model once to convert prompt->SQL pack; stores pack deterministically | `sql_pack.json` | Smoke test if model present; skipped in CI if absent |
| Optional in-pipeline SQL generation plugin | `plugins/llm_text2sql_local_generate_v1/*` | Runs vLLM locally during pipeline with strict gating | `artifacts/.../sql_pack.json` | Integration test behind env flag; deterministic decode config |
| SQL pack materialization transform | `plugins/transform_sqlpack_materialize_v1/*` (new) | Executes safe subset of pack to produce derived views/tables for downstream plugins | Derived tables/views + manifest | Integration test: derived objects created; idempotent |
| Plugin integration API | `src/statistic_harness/core/types.py`, `src/statistic_harness/core/plugin_runner.py` | Add `ctx.sql` helper and/or `ctx.sql_query(...)` for uniform access | N/A | Unit tests for ctx wiring |
| Adoption tracking matrix | extend `scripts/plugin_data_access_matrix.py`, `scripts/sql_assist_adoption_matrix.py` | Track which plugins use SQL Assist and how | `docs/sql_assist_adoption_matrix.md/json` | Verify scripts in `scripts/verify_docs_and_plugin_matrices.py` |
| Plain-English report (separate) | `plugins/report_plain_english_v1/*`, `src/statistic_harness/core/plain_report.py` | Render plugin outputs into non-technical English `plain_report.md` while keeping `report.json` technical | `plain_report.md` | Snapshot tests on fixture run |

## Sprint 1: SQL Assist Core + Scratch DB (No LLM, No Plugin Refactors Yet)
**Goal**: Provide safe, deterministic SQL access to the normalized dataset while enforcing “normalized DB is read-only” and enabling plugin-owned tables in a scratch DB.

**Demo/Validation**:
- Unit tests pass: `python -m pytest -q`
- Create a tiny “sql assist” fixture plugin that runs a query and emits a citeable finding.

### Task 1.1: Add `SqlAssist` + Read-Only Validator
- **Location**: `src/statistic_harness/core/sql_assist.py` (new)
- **Description**:
  - Implement `SqlAssist` with methods:
    - `query_rows(sql: str, params: dict|tuple|list|None) -> list[sqlite3.Row]`
    - `query_df(sql: str, params: ...) -> pandas.DataFrame` (optional)
    - `explain_query_plan(sql, params) -> list[dict]`
  - Enforce fail-closed validation:
    - Allow only a single statement.
    - Allow only `SELECT` / `WITH`.
    - Forbid `;`, `PRAGMA`, `ATTACH`, `DETACH`, `VACUUM`, `INSERT`, `UPDATE`, `DELETE`, `DROP`, `CREATE`, `ALTER`.
    - Forbid references to `sqlite_master` unless explicitly allowlisted (optional).
  - Always capture provenance:
    - normalized schema hash
    - query text hash
    - params JSON
    - rowcount
    - deterministic result hash (e.g., hash of `json.dumps(rows, sort_keys=True)` with stable ordering)
- **Complexity**: 7/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - Safe validator blocks dangerous SQL.
  - Helper stores a `sql_manifest.json` artifact for each executed query.
- **Validation**:
  - New unit tests for allow/deny cases and manifest determinism.

### Task 1.2: Wire `SqlAssist` Into `PluginContext`
- **Location**: `src/statistic_harness/core/types.py`, `src/statistic_harness/core/plugin_runner.py`
- **Description**:
  - Add a new `ctx.sql` attribute (or `ctx.sql_query(...)` function) that uses `ctx.storage.connection()` under the hood.
  - Keep backwards compatibility with existing plugins.
- **Complexity**: 4/10
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Plugins can call `ctx.sql.query_rows(...)` without reaching into `ctx.storage` directly.
- **Validation**:
  - Smoke test plugin + unit tests for ctx initialization.

### Task 1.3: Enforce “State DB Read-Only” in Plugin Subprocesses
- **Location**: `src/statistic_harness/core/storage.py`, `src/statistic_harness/core/plugin_runner.py`
- **Description**:
  - Add `Storage(read_only=True)` mode used only inside plugin subprocesses:
    - Skip migrations.
    - Open sqlite with `mode=ro` (URI form) and skip write-oriented PRAGMAs.
  - Add a run-scoped scratch database for plugin-owned tables:
    - Path: `appdata/runs/<run_id>/scratch.sqlite` (or `run_dir/scratch.sqlite`).
    - Provide it as `ctx.scratch_storage` and `ctx.scratch_sql`.
  - Update sqlite connect guard to allow only:
    - read-only state DB path
    - scratch DB path
- **Complexity**: 7/10
- **Dependencies**: Task 1.1, Task 1.2
- **Acceptance Criteria**:
  - Attempts to write to the state DB from a plugin subprocess fail (hard).
  - Plugins can create/read their own tables in scratch DB.
- **Validation**:
  - Unit test: plugin tries `INSERT` into normalized table -> fails.
  - Integration test: plugin creates scratch table -> succeeds.

### Task 1.4: Deterministic Schema Snapshot
- **Location**: `src/statistic_harness/core/sql_schema_snapshot.py` (new)
- **Description**:
  - Implement a stable snapshot of relevant DB objects:
    - normalized tables, columns, types
    - indexes
    - row counts where cheap
  - Return `schema_snapshot.json` and `schema_hash`.
- **Complexity**: 5/10
- **Dependencies**: None
- **Acceptance Criteria**:
  - Same DB state produces identical snapshot bytes/hash.
- **Validation**:
  - Unit test on a temporary sqlite DB.

## Sprint 2: SQLCoder Prompt Pack (Compliant “LLM Prompt Builder Only” Path)
**Goal**: Create a plugin that outputs a schema-aware prompt pack so SQL can be generated once, stored, and replayed deterministically.

**Demo/Validation**:
- Run full harness; confirm prompt pack artifacts exist under the run.

### Task 2.1: Define SQL Pack Schema
- **Location**: `docs/sql_pack.schema.json` (new)
- **Description**:
  - Define a strict format:
    - `schema_hash`
    - `dialect: "sqlite"`
    - `queries: [{id, purpose, sql, params_schema, expected_outputs}]`
    - `safety: {read_only: true}`
    - `generated_by: {tool, model, version, decode_config, created_at}`
  - Add validator utility used by materializer and tests.
- **Complexity**: 4/10
- **Dependencies**: Sprint 1
- **Acceptance Criteria**:
  - Schema rejects missing fields and multi-statement SQL.
- **Validation**:
  - JSON schema tests.

### Task 2.2: Add `transform_sql_intents_pack_v1`
- **Location**: `plugins/transform_sql_intents_pack_v1/plugin.py`, `plugins/transform_sql_intents_pack_v1/plugin.yaml`, `src/statistic_harness/core/sql_intents.py`
- **Description**:
  - Capture:
    - schema snapshot
    - deterministic query intents (SQLite dialect, deterministic ordering)
  - Write:
    - `schema_snapshot.json`
    - `sql_intents.json`
- **Complexity**: 6/10
- **Dependencies**: Sprint 1 (schema snapshot)
- **Acceptance Criteria**:
  - Plugin runs with no model present.
  - Intents are deterministic for the same schema hash.
- **Validation**:
  - Integration test in `tests/plugins/`.

### Task 2.3: Add Operator Script To Generate SQL Pack With vLLM
- **Location**: `scripts/generate_sql_pack_with_vllm.py` (new)
- **Description**:
  - Load model from `/mnt/d/autocapture/models/<model_name>/` (read-only).
  - Deterministic decode:
    - greedy (`temperature=0`, `top_p=1`, `top_k=-1` or equivalent)
    - fixed max tokens
  - Produce:
    - `appdata/sqlpacks/<schema_hash>/sql_pack.json`
    - `appdata/sqlpacks/<schema_hash>/generation_manifest.json` (model+config+prompt hash)
- **Complexity**: 7/10
- **Dependencies**: Task 2.1, Task 2.2
- **Acceptance Criteria**:
  - Re-running produces identical pack given same prompt/model/config.
- **Validation**:
  - Smoke test skipped unless model exists locally.

## Sprint 3: Optional “In-Pipeline” Local SQLCoder Generation (Only If Allowed)
**Goal**: Fully automate SQL pack generation as part of the harness while keeping security/determinism.

**Demo/Validation**:
- Full run produces `sql_pack.json` without operator scripts when env flag is enabled.

### Task 3.1: Add `llm_text2sql_local_generate_v1` Plugin (Gated)
- **Location**: `plugins/llm_text2sql_local_generate_v1/*`
- **Description**:
  - Hard gate behind `STAT_HARNESS_ENABLE_LOCAL_LLM_SQL=1`.
  - Requires model path configured (env var).
  - Writes pack + manifest into run artifacts and `appdata/sqlpacks/<schema_hash>/`.
- **Complexity**: 8/10
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - When disabled, plugin is `skipped` with a clear message.
  - When enabled, outputs are deterministic and schema-valid.
- **Validation**:
  - Integration tests gated by env var.

## Sprint 4: “All Plugins Benefit” Integration (Refactor/Extension)
**Goal**: Make SQL Assist usable everywhere, and progressively migrate plugin families so they use SQL where it helps (projection/aggregation/joins), while keeping correctness and citeability.

**Demo/Validation**:
- Full harness run on the 500k dataset finishes and produces additional actionable findings attributable to SQL-powered pre-aggregation.
- `analysis_actionable_ops_levers_v1` produces findings with evidence pointing to SQL manifests and derived tables where used.

### Task 4.1: Add Query Intents Registry (Central, Deterministic)
- **Location**: `src/statistic_harness/core/sql_intents.py` (new)
- **Description**:
  - Define a stable set of reusable intents, e.g.:
    - `eventlog_core_projection` (process/timestamps/durations/deps)
    - `per_process_wait_stats`
    - `per_process_hourly_medians`
    - `dependency_hotspots`
    - `top_variants_by_duration`
  - Each intent declares:
    - required normalized columns (semantic names)
    - acceptable fallbacks
    - output schema
  - SQLCoder prompt pack consumes these intents, not per-plugin ad-hoc prose.
- **Complexity**: 7/10
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - Intents cover at least the operational/actionable family (queue delay, conformance, ops levers, ideaspace).
- **Validation**:
  - Unit tests: intent resolution chooses correct normalized columns.

### Task 4.2: Add SQL Pack Materializer Transform Plugin
- **Location**: `plugins/transform_sqlpack_materialize_v1/*`
- **Description**:
  - Validate pack against schema hash.
  - Execute allowlisted queries (read-only by default).
  - Optional “materialize” mode (if permitted):
    - write derived tables as `derived_<intent_id>_<schema_hash_prefix>`
    - record DDL + hashes in a manifest
  - Always emit citations:
    - `sql/<intent_id>.sql`
    - `sql/<intent_id>_manifest.json`
    - `sql/<intent_id>_sample.json` (small, bounded)
- **Complexity**: 8/10
- **Dependencies**: Sprint 1, Sprint 2
- **Acceptance Criteria**:
  - Idempotent: re-run doesn’t duplicate or drift.
  - Fail-closed if any referenced columns can’t be mapped; emits a “critical stop” marker that prevents downstream analysis stage.
- **Validation**:
  - Integration test: missing-column pack fails closed.

### Task 4.3: Plugin Family Migrations (How “All Plugins” Benefit)
- **Location**: `src/statistic_harness/core/stat_plugins/registry.py`, `src/statistic_harness/core/stat_plugins/*`, selective plugin wrappers
- **Description**:
  - Add a standard pattern:
    - Prefer SQL core projections (narrow columns) before pandas-heavy work.
    - Use SQL aggregates when output is groupby/join based.
  - Targeted early wins (pilot set):
    - `analysis_actionable_ops_levers_v1`: compute `over_threshold` totals per process in SQL (faster + citeable).
    - `analysis_conformance_*`: compute alignments inputs (sequence extraction) in SQL.
    - `analysis_dependency_resolution_join`: do join logic in SQL and only load final pairs.
    - `analysis_ideaspace_*`: compute candidate “gap” measures via SQL aggregates.
  - Then expand to the rest:
    - Most “Family D” (classical stats): SQL to compute per-group samples/summary; pandas only for final tests.
    - ML plugins: SQL to select and type-coerce numeric feature matrix columns deterministically.
- **Complexity**: 9/10
- **Dependencies**: Task 4.1, Task 4.2
- **Acceptance Criteria**:
  - No plugin is forced to downsample.
  - For large datasets, full-DF loads become rare; most plugins operate on projected columns or aggregates.
- **Validation**:
  - `python -m pytest -q`
  - Add a “large dataset simulated” integration test ensuring unbounded loads are blocked unless explicitly approved.

### Task 4.4: Add/Update Adoption Matrices
- **Location**: `scripts/plugin_data_access_matrix.py` (extend), `scripts/sql_assist_adoption_matrix.py` (new)
- **Description**:
  - Track per plugin:
    - uses SQL Assist: `0/1`
    - which intents used
    - still loads full DF unbounded: `0/1`
  - Keep docs up to date under `docs/`.
- **Complexity**: 4/10
- **Dependencies**: Sprint 1+
- **Acceptance Criteria**:
  - Matrices can be verified in CI (`--verify` mode).
- **Validation**:
  - Update `scripts/verify_docs_and_plugin_matrices.py` to include new docs.

## Sprint 5: Plain-English Output (Separate Report)
**Goal**: Produce `plain_report.md` that is non-technical English while keeping `report.json` technical and citeable.

**Demo/Validation**:
- Full run produces `plain_report.md` alongside `report.md` and `answers_recommendations.md`.

### Task 5.1: Add `report_plain_english_v1`
- **Location**: `plugins/report_plain_english_v1/*`, `src/statistic_harness/core/plain_report.py`
- **Description**:
  - Render each plugin’s key findings into:
    - “What it means”
    - “What to do next”
    - “How to validate” (with pointers to artifacts / SQL manifests)
  - Strictly avoid jargon; include the minimum technical metadata only as citations.
- **Complexity**: 6/10
- **Dependencies**: Sprint 1+ (for citations)
- **Acceptance Criteria**:
  - Report is understandable without statistical jargon.
- **Validation**:
  - Snapshot tests from fixture run outputs.

## Plugin Benefit Analysis (High-Level)
Custom SQL helps almost every plugin in one of these ways:
- **Projection**: only load the columns the plugin needs (huge memory win).
- **Aggregation**: compute per-process/per-variant/per-hour/per-server stats in SQLite.
- **Join**: dependency resolution and “linkage” style plugins can do joins in SQL and only move final pairs to pandas.
- **Pre-materialization**: create derived tables/views once per run to avoid repeating heavy scans.

Where SQL does *not* replace work:
- Algorithms requiring full in-memory numeric matrices (PCA/IForest/LOF/etc) still need arrays, but SQL can build the matrix deterministically and reduce Python-side parsing.

## Which Plugin Families Benefit Most (Concrete)
Prioritize SQL adoption for these families first (because they are currently full-DF and naturally SQL-friendly):
- **Operational levers / queueing / conformance / sequence**:
  - `analysis_actionable_ops_levers_v1`, `analysis_queue_delay_decomposition`, `analysis_busy_period_segmentation_v2`,
    `analysis_conformance_*`, `analysis_process_*`, `analysis_dependency_*`.
  - Typical SQL intents: per-process delay stats, busy-period windows, dependency hotspots, variant frequency + medians.
- **“Classical stats by group”** (fast win: aggregate in SQL, small samples in pandas):
  - `analysis_*ttest*`, `analysis_*anova*`, `analysis_*chi*`, `analysis_effect_size_report`.
- **Linkage / joins**:
  - `analysis_upload_linkage`, `analysis_dependency_resolution_join`.
- **Ideaspace**:
  - `analysis_ideaspace_*` benefits from SQL pre-aggregation for candidate gaps and action candidates (still citeable).

## Testing Strategy
- Unit:
  - SQL validator deny/allow cases
  - schema snapshot determinism
  - sql pack schema validation
- Integration:
  - Full harness run on fixture dataset produces:
    - `report.md`, `report.json`, `plain_report.md`
    - SQL artifacts/manifests when enabled
- Gate:
  - `python -m pytest -q` must pass before shipping.

## Potential Risks & Gotchas
- **LLM determinism**: GPU inference can drift. Mitigate by:
  - greedy decode
  - pinning model + tokenizer versions
  - storing prompt + output hashes
  - “generate once, replay forever” default workflow
- **Hallucinated columns**: SQLCoder may reference non-existent columns. Must fail closed before analysis starts.
- **Security**: SQL must be read-only unless explicitly enabling materialization with strict guardrails.
- **Performance**: poorly generated SQL can be slow. Mitigate by:
  - query plan inspection + index suggestions
  - prefer intent templates + fill-in rather than free-form SQL.
 - **Filesystem location**: model storage target `/mnt/d/autocapture/models/` is outside the repo. Implement a dedicated downloader that:
   - writes to a versioned subfolder under `/mnt/d/autocapture/models/sql/defog/sqlcoder2/`
   - records `sha256` of config + key weight files (or `snapshot_download` commit hash)
   - supports “already present” fast-path without network.

## Rollback Plan
- Keep SQL Assist API isolated; default off via settings.
- If any regression:
  1. Disable SQL pack materialization plugin.
  2. Fall back to existing `ctx.dataset_loader(columns=...)` projections.
  3. Keep prompt pack plugin available for future iteration.
