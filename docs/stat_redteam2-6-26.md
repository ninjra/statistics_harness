THREAD: STATISTIC-HARNESS-ADVERSARIAL-REDESIGN
CHAT_ID: N/A
TS: 2026-02-06
D0: 2026-02-06
DETERMINISM: VERIFIED

## Assumptions

* Analysis is based **only** on `generated repository snapshot artifact`; no runtime execution or external context.
* Default posture is **local-only**; any networked/hosted behavior stays behind explicit flags.
* Primary environment is **Windows 11 + heavy WSL2 + Python**; path/rename semantics must be Windows-safe.
* Plugins are **semi-trusted**: treated as potentially buggy/malicious, so sandbox + provenance must be strong.
* SQLite is the system-of-record for state and provenance; durability and corruption handling are first-class.
* Determinism is a product guarantee: if something is randomized, it must be seeded and recorded.
* Operator is a single primary user (Justin); multi-tenant features are treated as optional.
* Large datasets are expected; “load everything into memory” must be guarded.

## Context derived from this repository

This repository is **Statistic Harness**, a **local-first, plugin-based engine** for producing **deterministic insights** from **tabular datasets** (CSV/XLSX) via a plugin pipeline and generating a versioned `report.json` with lineage/evidence. 

### Primary workflows

| Workflow         | What happens                                                                            | Evidence                                                          |
| ---------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| CLI run          | Ingest a file, execute analysis plugins, produce report artifacts                       | README CLI usage                                                 |
| Web UI run       | Serve local UI, upload dataset, configure run, execute pipeline, browse artifacts/trace | UI templates describe “Deterministic Insight Engine” + workflow  |
| Plugin execution | Run plugins in subprocess with sandbox + guards; collect outputs                        | Plugin runner guards + sandbox                                   |

### Major components

| Component                                   | Role                                                                          | Evidence                                         |
| ------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------ |
| `core/pipeline.py`                          | DAG planning + execution across plugin layers (parallel within layers)        | Toposort + ThreadPoolExecutor                   |
| `core/plugin_manager.py`                    | Discover/load plugin specs, validate manifests/schemas                        | Plugin spec + schema usage                      |
| `core/plugin_runner.py` + `core/sandbox.py` | Subprocess execution + sandboxing + safety guards (network/eval/pickle/shell) | Guards listed                                   |
| `core/storage.py` + migrations              | SQLite state store; PRAGMAs and tables for runs/results/trace                 | WAL + synchronous NORMAL                        |
| `core/dataset_io.py`                        | Dataset access via SQLite; can load into pandas                               | `_load_df` uses `pd.read_sql_query` and caches  |
| `ui/server.py` + templates                  | Local web UI, upload endpoint, run views, trace UI                            | Upload handler enforces size + sha256           |
| Schemas in `docs/`                          | JSON schemas for plugin manifests and reports                                 | Report schema exists                            |

### Storage/state model

* Uses a local **SQLite** DB (`state.sqlite`) with WAL mode and other PRAGMAs (foreign keys, busy timeout, etc.). 
* Upload pipeline is **content-addressed** (SHA-256) and enforces a configurable max upload size. 
* Run identifiers are **random UUID4 hex** today (`make_run_id()`), not deterministically derived. 

### Execution model

* Pipeline toposorts a dependency graph into **layers** and executes each layer with a **ThreadPoolExecutor**. 
* For plugin subprocess sandboxing, the pipeline expands allowed paths to include the DB path and run directory. 
* Plugin runner applies **determinism and safety guards**: blocks network unless enabled, blocks `eval`/`exec`, blocks pickle, blocks shell/subprocess calls. 

### Interfaces

* CLI is present; local network binding for `serve` is guarded (requires explicit env to bind non-loopback). 
* Web UI provides upload, run, trace views, and includes “vector search” UI elements (feature-flagged in code). 

### External integrations

* Optional vector store uses `sqlite-vec` extension and is explicitly feature-flagged. 
* Optional “tenancy” model is also feature-flagged. 
* No evidence of mandatory third-party hosted services. (Network is explicitly guarded/disabled by default.) 

## key_claims

* **[QUOTE]** “Statistic Harness is a local-first, plugin-based engine for deterministic insights from tabular datasets.” 
* **[CITE]** Plugins execute under a guard set that blocks network/eval/pickle/shell by default. 
* **[CITE]** Storage uses SQLite with WAL and synchronous=NORMAL (durability/perf tradeoff). 
* **[CITE]** Upload handling computes SHA-256 and enforces a configurable maximum upload size. 
* **[INFERENCE]** The current sandbox is strong on API-level egress and code injection, but is weaker on *write-surface minimization* because plugin allowlists include broad directories in manifests and pipeline expands allowed paths. 
* **[NO EVIDENCE]** No evidence of GPU-accelerated processing in the current pipeline; any GPU path should remain optional and off by default.

---

# I. Foundation: Inputs + core lifecycle stability

Current evidence: uploads are SHA-256 content-addressed with size limits , run IDs are UUID4 hex (not deterministic) , SQLite uses WAL and synchronous=NORMAL , and a Windows-safe rename hook exists but appears env-gated. 

| ID     | Recommendation details                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |                  Pillar scores | Effort / Risk |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -----------------------------: | ------------- |
| FND-01 | **Atomic run directory + crash-safe journaling**<br>Rationale: Prevent “half-created run” ambiguity and enable deterministic resume/cleanup after crashes or kills.<br>Dependencies: None.<br>Improved: crash recovery, integrity, audit.<br>Risked: orphan temp dirs if finalize fails.<br>Enforce: `src/statistic_harness/core/pipeline.py`, `src/statistic_harness/core/utils.py` (new `atomic_dir()`), DB run status transitions.<br>Regression detection: CI kill/restart test; if failing => **DO_NOT_SHIP**.<br>Acceptance test: kill process mid-run; restart shows run as `ABORTED` with preserved logs; rerun either resumes or cleanly restarts without mixed artifacts.                                                                                                            | P1=2 P2=2 P3=2 P4=4 (Total=10) | M / Med       |
| FND-02 | **Atomic JSON writes for all run artifacts**<br>Rationale: Prevent truncated `report.json`/manifests from being mistaken as valid outputs.<br>Dependencies: None.<br>Improved: data integrity, citeability.<br>Risked: slight IO overhead.<br>Enforce: new helper in `core/utils.py` used everywhere JSON is written (reports, manifests, exported bundles).<br>Regression detection: unit test that simulates partial write; if failing => **DO_NOT_SHIP**.<br>Acceptance test: forcibly interrupt during write; file is either old valid version or new valid version, never partial.                                                                                                                                                                                                        |  P1=1 P2=1 P3=2 P4=3 (Total=7) | S / Low       |
| FND-03 | **Harden upload CAS: verify-on-write + refcount + quarantine**<br>Rationale: Uploads are already hashed and size-limited ; add durability checks and lifecycle controls to prevent silent corruption and accidental reuse of a bad blob.<br>Dependencies: FND-02 recommended (atomic helpers).<br>Improved: input integrity, reproducibility.<br>Risked: migration complexity for existing uploads.<br>Enforce: `ui/server.py` upload flow, DB schema for upload references, and a “quarantine then promote” directory scheme.<br>Regression detection: upload corruption test (flip a byte); if not detected => **DO_NOT_SHIP**.<br>Acceptance test: after upload, recompute sha256 from disk and match; deleting a dataset version decrements refcount; unreferenced blobs are GC’d safely. | P1=2 P2=3 P3=2 P4=3 (Total=10) | M / Med       |
| FND-04 | **Startup integrity checks: PRAGMA integrity_check + orphan cleanup**<br>Rationale: SQLite WAL + NORMAL is performant but needs explicit integrity gates under crash/power-loss scenarios. <br>Dependencies: None.<br>Improved: data durability, failure containment.<br>Risked: startup time on huge DBs.<br>Enforce: app startup path (CLI + UI), add “quick” and “full” integrity modes.<br>Regression detection: corrupted DB fixture; must fail loud; if silent => **DO_NOT_SHIP**.<br>Acceptance test: introduce corruption; app refuses to run and offers restore-from-backup flow.                                                                                                                                                                                                    |  P1=1 P2=2 P3=2 P4=3 (Total=8) | S / Low       |
| FND-05 | **Online backup/restore + retention for `state.sqlite`**<br>Rationale: Make recovery deterministic and operator-friendly for Windows/WSL users where file locking and partial writes happen.<br>Dependencies: FND-04 (integrity checks).<br>Improved: resilience, citeability (recoverable history).<br>Risked: backup bloat if not pruned.<br>Enforce: `core/storage.py` (SQLite backup API), CLI command `stat-harness backup/restore`.<br>Regression detection: restore test with known DB; if mismatch => **DO_NOT_SHIP**.<br>Acceptance test: create run; backup; delete DB; restore; run list and reports remain consistent; checksums match.                                                                                                                                            |  P1=1 P2=2 P3=2 P4=4 (Total=9) | M / Low       |
| FND-06 | **Deterministic `run_fingerprint` and optional deterministic IDs**<br>Rationale: Run IDs are UUID4 today ; add a deterministic fingerprint so “same input + same config + same plugin set” is provably identical, even if run_id differs.<br>Dependencies: META-01.<br>Improved: reproducibility and audit.<br>Risked: users may confuse run_id vs fingerprint unless UI clarifies.<br>Enforce: DB schema adds `run_fingerprint`; UI surfaces both.<br>Regression detection: deterministic re-run fixture; fingerprint must match; if not => **DO_NOT_SHIP**.<br>Acceptance test: two runs with same dataset hash + config hash produce same fingerprint; toggling any setting changes it.                                                                                                    |  P1=1 P2=1 P3=2 P4=4 (Total=8) | M / Med       |
| FND-07 | **Make Windows/WSL safe_rename default for generated artifacts**<br>Rationale: A safe rename hook exists behind env control ; make it default on Windows/WSL to reduce partial artifact risk.<br>Dependencies: None.<br>Improved: reliability on Windows filesystems.<br>Risked: minor behavior differences in dev environments.<br>Enforce: CLI bootstrapping sets `STAT_HARNESS_SAFE_RENAME=1` automatically on Windows/WSL.<br>Regression detection: Windows path/rename tests; if flaky => **DO_NOT_SHIP**.<br>Acceptance test: repeated report writes under concurrent UI refresh never produce “file in use” partial outputs.                                                                                                                                                           |  P1=1 P2=1 P3=1 P4=2 (Total=5) | S / Low       |
| FND-08 | **Explicit ingest completeness vs sampling metadata + enforcement**<br>Rationale: Users must not confuse sampled analysis with full ingest; record and surface completeness at each stage.<br>Dependencies: META-02, UX-08.<br>Improved: correctness and user trust.<br>Risked: plugin compatibility (must emit sampling metadata).<br>Enforce: report schema, plugin output schema conventions, UI badges.<br>Regression detection: sample-mode run must display “SAMPLED” everywhere; if missing => **DO_NOT_SHIP**.<br>Acceptance test: run with sampling enabled; report and UI show sample fraction, seed, and selection method; export includes these fields.                                                                                                                            |  P1=1 P2=0 P3=3 P4=3 (Total=7) | M / Low       |

---

# II. Metadata: Contracts + provenance + auditability

Current evidence: deterministic JSON utilities exist (`stable_json_dumps`) , plugin manifests are schema-driven , and a report schema exists in `docs/report.schema.json`. 

| ID      | Recommendation details                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |                 Pillar scores | Effort / Risk |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------: | ------------- |
| META-01 | **Persist canonical `run_manifest.v1` (schema + file) per run**<br>Rationale: “Processing happened against THIS input + THIS config” must be provable without DB access (portable evidence).<br>Dependencies: FND-02.<br>Improved: citeability, reproducibility.<br>Risked: schema churn if not versioned strictly.<br>Enforce: new `docs/run_manifest.schema.json`, write `run_manifest.json` into run dir, store hash in DB.<br>Regression detection: schema validation in CI; if broken => **DO_NOT_SHIP**.<br>Acceptance test: every completed run has a valid manifest including input sha256, config snapshot, plugin list+versions, flags, seeds, and artifact hashes. | P1=1 P2=1 P3=3 P4=4 (Total=9) | M / Low       |
| META-02 | **Standardize modeled-vs-measured + confidence/evidence blocks**<br>Rationale: Prevent users treating modeled outputs as measured facts; force explicit disclosure per finding.<br>Dependencies: report schema changes + plugin output conventions.<br>Improved: accuracy and interpretability.<br>Risked: plugin updates required.<br>Enforce: `docs/report.schema.json` update + output schema validator in pipeline.<br>Regression detection: sample report fixture; must contain blocks; if missing => **DO_NOT_SHIP**.<br>Acceptance test: UI renders “MEASURED” vs “MODELED” tags and requires evidence links for measured claims.                                      | P1=0 P2=0 P3=3 P4=4 (Total=7) | M / Med       |
| META-03 | **Store per-plugin `execution_fingerprint` (code+manifest+settings+input)**<br>Rationale: If plugin code changes, the same run config should not be considered comparable unless fingerprints match.<br>Dependencies: META-01.<br>Improved: audit trail, replay safety.<br>Risked: larger DB rows.<br>Enforce: add columns to plugin results/executions; compute using stable hashing utilities.<br>Regression detection: fingerprint must change on plugin file edit; if not => **DO_NOT_SHIP**.<br>Acceptance test: editing plugin code forces recompute (or marks cached results invalid).                                                                                 | P1=1 P2=2 P3=2 P4=4 (Total=9) | M / Low       |
| META-04 | **Evidence registry for artifacts (sha256, bytes, mime, producer, path)**<br>Rationale: Artifacts should be referencable and verifiable independent of UI links.<br>Dependencies: META-01.<br>Improved: citeability and safety (detect tampering).<br>Risked: requires standard artifact emission path.<br>Enforce: pipeline collects artifact metadata from run dir; stores registry in manifest + DB.<br>Regression detection: artifact download must match registered sha256; if mismatch => **DO_NOT_SHIP**.<br>Acceptance test: every artifact shown in UI has a hash, size, mime, and producing plugin id.                                                              | P1=1 P2=1 P3=1 P4=4 (Total=7) | M / Low       |
| META-05 | **Schema versioning + compatibility checks (config/output/report) + hashes**<br>Rationale: Prevent silent schema drift between plugin versions and UI/report readers.<br>Dependencies: EXT-07, QA-02.<br>Improved: correctness and forward/back compat.<br>Risked: strict validation may break legacy plugins.<br>Enforce: plugin_manager validates schema versions; pipeline refuses incompatible combos with clear error.<br>Regression detection: contract tests; if drift => **DO_NOT_SHIP**.<br>Acceptance test: downgrade plugin bundle; system blocks run if report consumer can’t parse output schema.                                                                | P1=1 P2=1 P3=3 P4=3 (Total=8) | M / Med       |
| META-06 | **Vector-store provenance (model id, chunking, source entity) exportable**<br>Rationale: Vector search must be auditable; embeddings aren’t “magic,” they’re derived artifacts.<br>Dependencies: vector store feature-flag remains optional .<br>Improved: explainability and citeability of retrieval results.<br>Risked: schema changes for vector tables.<br>Enforce: store provenance alongside embeddings; include in export bundles.<br>Regression detection: embedding row must reference source entity; if not => **DO_NOT_SHIP**.<br>Acceptance test: search result includes link to source row/plugin output and embedding provenance fields.                      | P1=1 P2=1 P3=2 P4=3 (Total=7) | M / Low       |
| META-07 | **UI “What ran?” summary card (hashes, seeds, flags, versions)**<br>Rationale: Reduce cognitive load and prevent “I think I ran X” errors.<br>Dependencies: META-01.<br>Improved: operator trust, fewer misinterpretations.<br>Risked: none meaningful.<br>Enforce: UI run detail template and API include manifest summary.<br>Regression detection: UI test asserts card present; if missing => **DO_NOT_SHIP**.<br>Acceptance test: a screenshot of run detail page is enough to reconstruct the exact execution inputs/config.                                                                                                                                            | P1=0 P2=0 P3=2 P4=4 (Total=6) | S / Low       |
| META-08 | **Export “Repro pack” zip (manifest + schemas + logs + report)**<br>Rationale: Make results portable and reviewable offline; aligns to “auditable runs.”<br>Dependencies: OBS-03 (diag bundle) and META-01.<br>Improved: citeability and supportability.<br>Risked: possible accidental inclusion of raw data; must be configurable.<br>Enforce: CLI command `stat-harness export --run-id` with explicit include/exclude options.<br>Regression detection: export content list test; if includes raw without consent => **DO_NOT_SHIP**.<br>Acceptance test: unzip pack, validate schemas, verify hashes, open report without DB.                                            | P1=1 P2=1 P3=2 P4=4 (Total=8) | M / Low       |

---

# III. Execution pipeline: Correctness + replayability

Current evidence: deterministic layer ordering exists (sorted zero-indegree list) and parallel execution uses ThreadPoolExecutor ; plugin runner blocks risky APIs by default ; pipeline expands sandbox allow paths to include DB path and run dir. 

| ID      | Recommendation details                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |                 Pillar scores | Effort / Risk |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------: | ------------- |
| EXEC-01 | **Idempotent plugin caching keyed by hashes + `--force/--reuse`**<br>Rationale: Prevent redundant work and make re-runs deterministic and cheap while still auditable.<br>Dependencies: META-03, FND-06.<br>Improved: performance + citeability (“same fingerprint reused”).<br>Risked: stale cache if fingerprinting incomplete.<br>Enforce: storage query by (dataset_hash, settings_hash, plugin id/version, code hash).<br>Regression detection: change one setting => cache miss; if cache hit => **DO_NOT_SHIP**.<br>Acceptance test: run twice; second run reuses results with explicit “REUSED” markers and zero plugin executions.                                                         | P1=2 P2=1 P3=2 P4=4 (Total=9) | M / Med       |
| EXEC-02 | **Per-plugin timeout + memory budget enforcement**<br>Rationale: Contain runaway plugins under fatigue/misconfig; avoid whole-system stalls.<br>Dependencies: plugin runner enhancements.<br>Improved: operational stability and safety.<br>Risked: false positives on heavy plugins unless budgets are per-plugin.<br>Enforce: `core/plugin_runner.py` adds timeout and resource limits (soft); pipeline records termination reason.<br>Regression detection: synthetic slow plugin must be killed; if not => **DO_NOT_SHIP**.<br>Acceptance test: plugin exceeding limit ends with clear status + logs; run continues or fails deterministically per policy.                                      | P1=2 P2=2 P3=2 P4=2 (Total=8) | M / Med       |
| EXEC-03 | **Deterministic persistence order + per-plugin derived seeds**<br>Rationale: Parallel completion order is nondeterministic; persist results in stable `plugin_id` order and seed each plugin deterministically.<br>Dependencies: stable hashing utility exists .<br>Improved: reproducibility and diff-friendly outputs.<br>Risked: none meaningful.<br>Enforce: pipeline sorts before DB writes; derive `plugin_seed = hash(run_seed, plugin_id)` in context passed to plugin runner.<br>Regression detection: repeated run yields identical DB outputs; if diff => **DO_NOT_SHIP**.<br>Acceptance test: run twice; report JSON and plugin output JSON are byte-identical (excluding timestamps). | P1=1 P2=1 P3=3 P4=3 (Total=8) | S / Low       |
| EXEC-04 | **Failure containment: temp dirs + commit-on-validate only**<br>Rationale: Prevent partial outputs from being mistaken as valid results; isolate plugin scratch state.<br>Dependencies: FND-01, FND-02.<br>Improved: integrity and debuggability.<br>Risked: extra disk usage during runs.<br>Enforce: plugin writes to temp; pipeline validates output schema then moves artifacts to final run dir (atomic rename).<br>Regression detection: invalid output must not be committed; if committed => **DO_NOT_SHIP**.<br>Acceptance test: plugin emits invalid JSON; run marks plugin failed; no output rows exist in results table.                                                                | P1=1 P2=2 P3=2 P4=3 (Total=8) | M / Med       |
| EXEC-05 | **Replay mode: verify hashes then reuse cached results**<br>Rationale: Provide operator-grade replay for audits: “show me what ran” without re-executing plugins unless requested.<br>Dependencies: META-01, EXEC-01, META-04.<br>Improved: citeability + safety of reviews.<br>Risked: complexity in UI and CLI semantics.<br>Enforce: new command `stat-harness replay --run-id`; verifies manifest hashes vs disk/DB before rendering.<br>Regression detection: tamper artifact => replay must fail; if silent => **DO_NOT_SHIP**.<br>Acceptance test: modify artifact; replay reports hash mismatch and refuses to present as valid.                                                            | P1=2 P2=1 P3=2 P4=4 (Total=9) | L / Med       |
| EXEC-06 | **Pre-run config validation against plugin schemas with deterministic defaults**<br>Rationale: Reduce misconfig footguns and “it ran but not how I thought.”<br>Dependencies: META-05.<br>Improved: accuracy, fewer operator mistakes.<br>Risked: breaking old configs that relied on implicit defaults.<br>Enforce: pipeline loads each plugin’s `config_schema` and applies defaults deterministically (stable ordering).<br>Regression detection: schema fixture tests; if defaults drift => **DO_NOT_SHIP**.<br>Acceptance test: missing optional config fields are filled and recorded in manifest; UI shows the resolved config.                                                              | P1=1 P2=1 P3=3 P4=3 (Total=8) | S / Low       |
| EXEC-07 | **Plugin DAG validation: cycles and missing deps with graph output**<br>Rationale: Operator-grade errors for plugin conflicts; reduce admin debugging time.<br>Dependencies: None.<br>Improved: correctness and UX clarity.<br>Risked: none meaningful.<br>Enforce: pipeline preflight prints graph edges and the exact cycle path when detected.<br>Regression detection: cycle fixture; must error clearly; if not => **DO_NOT_SHIP**.<br>Acceptance test: introduce cyclic deps; run fails before execution with actionable message and suggested fix.                                                                                                                                           | P1=0 P2=0 P3=2 P4=2 (Total=4) | S / Low       |
| EXEC-08 | **Run checkpoints in DB for resume-after-restart**<br>Rationale: WSL/Windows restarts happen; resuming avoids rework and reduces partial-state confusion.<br>Dependencies: FND-01, EXEC-04.<br>Improved: reliability and operability.<br>Risked: more DB writes/state complexity.<br>Enforce: stage checkpoint table; pipeline updates at stage boundaries and per-plugin completion.<br>Regression detection: restart test; resume must not rerun completed plugins; if rerun => **DO_NOT_SHIP**.<br>Acceptance test: stop process mid-layer; restart resumes pending plugins only; run fingerprint unchanged.                                                                                     | P1=1 P2=1 P3=2 P4=3 (Total=7) | L / Med       |

---

# IV. Extension system & manager: Discovery + lifecycle + sandboxing + UX

Current evidence: plugin manifests are schema-driven  and plugin manager loads plugin specs ; plugin manifests commonly include broad filesystem allowlists (example plugin allows appdata + plugins). 

| ID     | Recommendation details                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |                 Pillar scores | Effort / Risk |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------: | ------------- |
| EXT-01 | **Local plugin registry index (install state, version, hash, enabled)**<br>Rationale: Discovery alone is insufficient; operators need lifecycle state and provenance for plugins.<br>Dependencies: META-03.<br>Improved: auditability, safer updates.<br>Risked: migration from implicit “plugins/” scanning.<br>Enforce: new registry table in SQLite or a single registry file in appdata; plugin_manager reads registry first.<br>Regression detection: disable plugin => must not execute; if executes => **DO_NOT_SHIP**.<br>Acceptance test: disabling a plugin hides it in UI and pipeline; enabling restores deterministically.                                                                                                              | P1=1 P2=2 P3=1 P4=3 (Total=7) | M / Med       |
| EXT-02 | **CLI: `stat-harness plugins validate` (schema + import + smoke + caps)**<br>Rationale: Catch broken plugins early; reduce admin/operator friction.<br>Dependencies: None.<br>Improved: safety and correctness.<br>Risked: validation may be slow without caching.<br>Enforce: plugin_manager + CLI command uses schema validation and imports entrypoint in isolated mode.<br>Regression detection: validation must fail on malformed manifest; if passes => **DO_NOT_SHIP**.<br>Acceptance test: run validate across all plugins; outputs a report with pass/fail and required capabilities.                                                                                                                                                       | P1=0 P2=2 P3=2 P4=2 (Total=6) | S / Low       |
| EXT-03 | **Offline install/update/rollback via hashed plugin bundles**<br>Rationale: Enterprise/locked-down environments need deterministic offline upgrades and easy rollback.<br>Dependencies: EXT-01, EXT-02.<br>Improved: supply-chain hygiene, reproducibility.<br>Risked: significant implementation scope.<br>Enforce: bundle format: zip with manifest + file hashes; stored under appdata; registry points to active version.<br>Regression detection: rollback must restore prior hashes; if mismatch => **DO_NOT_SHIP**.<br>Acceptance test: install v2; run; rollback to v1; rerun produces expected v1 fingerprints and outputs.                                                                                                                 | P1=1 P2=3 P3=1 P4=3 (Total=8) | L / Med       |
| EXT-04 | **Capability negotiation + permissions model enforced by runner/UI**<br>Rationale: Users/admins must see and approve what a plugin can do (DB read/write, fs write, vector store, network).<br>Dependencies: EXT-01, EXT-05.<br>Improved: security and reduced footguns.<br>Risked: plugin ecosystem migration.<br>Enforce: plugin.yaml adds `capabilities_required`; pipeline/runner enforces.<br>Regression detection: plugin declaring network must not access network unless explicitly enabled; if it does => **DO_NOT_SHIP**.<br>Acceptance test: UI shows permissions before run; deny capability => plugin blocked with explicit reason.                                                                                                     | P1=0 P2=4 P3=1 P4=2 (Total=7) | M / Med       |
| EXT-05 | **Sandbox policy redesign: fs read/write split + narrow DB directory access**<br>Rationale: Current allowlists can include broad directories ; split read vs write to reduce persistence of malicious/buggy behavior.<br>Dependencies: SEC-05, QA-03.<br>Improved: security and integrity (prevent plugin self-modifying code).<br>Risked: compatibility break for plugins that write outside run dir.<br>Enforce: `core/sandbox.py` supports read-allow/write-allow; default write paths = run_dir only; DB dir read-only except WAL needs.<br>Regression detection: sandbox escape suite; if can write to plugins dir => **DO_NOT_SHIP**.<br>Acceptance test: plugin attempts to write to `plugins/`; blocked; writing to run artifacts succeeds. | P1=1 P2=4 P3=1 P4=3 (Total=9) | M / High      |
| EXT-06 | **Plugin health checks contract (`health()`), surfaced in manager UI**<br>Rationale: Operators need a quick way to detect broken dependencies without running full pipeline.<br>Dependencies: EXT-01.<br>Improved: operability and correctness.<br>Risked: plugins without health() need a default adapter.<br>Enforce: plugin entrypoint optionally implements `health()`; manager aggregates results and logs.<br>Regression detection: missing dependency must be visible; if silent => **DO_NOT_SHIP**.<br>Acceptance test: break a dependency; plugin shows “unhealthy” and run is blocked unless forced.                                                                                                                                       | P1=0 P2=1 P3=2 P4=2 (Total=5) | M / Low       |
| EXT-07 | **Compatibility rules: engine version range + schema versions**<br>Rationale: Stop incompatible plugins early instead of failing mid-run.<br>Dependencies: META-05.<br>Improved: stability and predictability.<br>Risked: none meaningful.<br>Enforce: plugin.yaml includes `requires.engine_semver`; manager enforces and reports.<br>Regression detection: incompatible plugin must be blocked; if runs => **DO_NOT_SHIP**.<br>Acceptance test: set engine range incompatible; UI shows blocked with reason and suggested compatible versions.                                                                                                                                                                                                     | P1=0 P2=1 P3=2 P4=2 (Total=5) | S / Low       |
| EXT-08 | **Extension Manager UI: install/update/rollback/disable/permissions**<br>Rationale: Reduce admin friction; make plugin state explicit and keyboard accessible.<br>Dependencies: EXT-01, EXT-03, EXT-04.<br>Improved: operator UX and safety.<br>Risked: scope creep if not MVP’d.<br>Enforce: new UI route `/plugins/manage` with clear IA and actions logged to events table.<br>Regression detection: UI smoke tests; if cannot disable plugin => **DO_NOT_SHIP**.<br>Acceptance test: install bundle, view permissions, disable plugin, rollback version, and see fingerprints change accordingly.                                                                                                                                                | P1=0 P2=1 P3=1 P4=2 (Total=4) | L / Low       |

---

# V. UI/UX: End-to-end workflow

Current evidence: UI positions itself as a “Deterministic Insight Engine” with project creation and dataset upload flow ; the wizard form includes a `run_seed` field (risk of confusion if semantics unclear). 

| ID    | Recommendation details                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |                 Pillar scores | Effort / Risk |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------: | ------------- |
| UX-01 | **Activity Dashboard with recent runs + determinism badges**<br>Rationale: Reduce “where am I?” friction and improve visibility of failures/completeness.<br>Dependencies: OBS-01, META-01.<br>Improved: operator speed + fewer mistakes.<br>Risked: none meaningful.<br>Enforce: new route and template; include TTFR and run completeness indicators.<br>Regression detection: UI e2e test; dashboard must render and link correctly; if broken => **DO_NOT_SHIP**.<br>Acceptance test: dashboard shows last N runs, status, fingerprint, input hash, duration, TTFR, and missing plugins.                                                     | P1=0 P2=0 P3=1 P4=3 (Total=4) | M / Low       |
| UX-02 | **Ingest panel: preview + sha256 + format detection + completeness**<br>Rationale: Prevent wrong-file selection and support “prove what input was used.”<br>Dependencies: FND-03, META-01.<br>Improved: accuracy and citeability.<br>Risked: preview may be slow on huge files unless sampled safely.<br>Enforce: upload returns preview + inferred schema; UI shows hash and first rows before “Run.”<br>Regression detection: upload preview test; if hash not shown => **DO_NOT_SHIP**.<br>Acceptance test: user can confirm dataset identity via hash + preview and compare against prior runs.                                              | P1=0 P2=1 P3=2 P4=3 (Total=6) | M / Low       |
| UX-03 | **Presets/templates for settings with diff + import/export**<br>Rationale: Reduce config friction and misconfiguration under fatigue.<br>Dependencies: EXEC-06.<br>Improved: speed and correctness.<br>Risked: preset drift across plugin versions unless schema-hashed.<br>Enforce: preset format includes plugin schema hashes; UI shows diff before applying.<br>Regression detection: preset apply must validate schemas; if bypass => **DO_NOT_SHIP**.<br>Acceptance test: export preset from a run; import elsewhere; system validates compatibility and shows diffs.                                                                      | P1=1 P2=0 P3=2 P4=2 (Total=5) | M / Med       |
| UX-04 | **Run config guardrails: grouped plugins + warnings + confirm summary**<br>Rationale: Prevent “ran the wrong thing” and clarify what will execute.<br>Dependencies: EXT-04, META-07.<br>Improved: fewer mistakes, clearer intent.<br>Risked: UI complexity if not designed tightly.<br>Enforce: single confirmation page with plugin list, deps satisfied, permissions, and flags.<br>Regression detection: UI test asserts confirm summary present; if missing => **DO_NOT_SHIP**.<br>Acceptance test: before run, user sees exactly which plugins and versions will run and what capabilities are requested.                                   | P1=1 P2=1 P3=2 P4=2 (Total=6) | M / Low       |
| UX-05 | **Run detail: timeline + errors + artifacts with hashes + replay button**<br>Rationale: Reduce confusion after partial failure and improve auditability at the point of use.<br>Dependencies: OBS-04, META-04, EXEC-05.<br>Improved: debugging and citeability.<br>Risked: none meaningful.<br>Enforce: run detail API includes plugin execution durations and artifact registry.<br>Regression detection: run detail must show artifact hashes; if missing => **DO_NOT_SHIP**.<br>Acceptance test: user can copy a single “Run summary” block that includes enough metadata to reproduce/replay.                                                | P1=0 P2=0 P3=2 P4=3 (Total=5) | M / Low       |
| UX-06 | **Accessibility pass: labels/ARIA/keyboard/high-contrast/reduced-motion**<br>Rationale: Accessibility-first reduces errors and fatigue-related misclicks.<br>Dependencies: QA-05 (UI tests).<br>Improved: usability under real constraints.<br>Risked: none meaningful.<br>Enforce: template refactor + automated a11y checks where feasible.<br>Regression detection: a11y smoke tests; if critical issues => **DO_NOT_SHIP**.<br>Acceptance test: all inputs have labels; dynamic status updates are announced; full keyboard navigation works.                                                                                                | P1=0 P2=0 P3=1 P4=1 (Total=2) | M / Low       |
| UX-07 | **Misclick protection: confirm destructive actions + undo (soft delete)**<br>Rationale: Low patience + fatigue means deletion/disable must be reversible.<br>Dependencies: OBS-01 (events), storage schema for soft deletes.<br>Improved: safety and operability.<br>Risked: storage bloat without retention policies.<br>Enforce: soft-delete flags; add “Undo” for recent actions; show fingerprint in confirm dialog.<br>Regression detection: delete should be reversible within window; if not => **DO_NOT_SHIP**.<br>Acceptance test: delete run => disappears; undo => restored with identical hashes.                                    | P1=0 P2=1 P3=1 P4=2 (Total=4) | M / Med       |
| UX-08 | **Evidence surfacing: modeled/measured + confidence + sampling coverage**<br>Rationale: Make it difficult to misinterpret results and easy to cite.<br>Dependencies: META-02, FND-08.<br>Improved: correctness and trust signals.<br>Risked: requires plugin output normalization.<br>Enforce: UI components render these fields consistently across report, plugin outputs, and trace views.<br>Regression detection: missing evidence/confidence must be flagged; if silently omitted => **DO_NOT_SHIP**.<br>Acceptance test: every displayed claim shows type (modeled/measured), confidence, and evidence link(s) or explicit “no evidence.” | P1=0 P2=0 P3=2 P4=3 (Total=5) | M / Med       |

---

# VI. Observability / Ops: Logs + metrics + diagnostics bundles

Current evidence: the system persists execution artifacts and has trace UI; operational telemetry for “frictionless workflow metrics” is not clearly surfaced in UI (no evidence of TTFR dashboard). Also, server and pipeline already track structured states in DB migrations (tables for runs/results/trace exist). 

| ID     | Recommendation details                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |                 Pillar scores | Effort / Risk |
| ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------: | ------------- |
| OBS-01 | **Structured events table for lifecycle telemetry (correlation IDs)**<br>Rationale: Enables measured UX/ops improvements and reliable diagnostics without scraping logs.<br>Dependencies: None.<br>Improved: auditability and ops insight.<br>Risked: event volume growth without retention.<br>Enforce: new DB table `events`; emit at upload/run/plugin start/end/fail; include `run_id`, `fingerprint`, `plugin_id`.<br>Regression detection: events emitted for every run stage; if missing => **DO_NOT_SHIP**.<br>Acceptance test: a single SQL query reconstructs run timeline and TTFR.                        | P1=1 P2=1 P3=2 P4=3 (Total=7) | M / Low       |
| OBS-02 | **Frictionless workflow metrics (TTFR, fail rate, retries) in UI**<br>Rationale: “Frictionless core workflow” must be measured, not guessed.<br>Dependencies: OBS-01, UX-01.<br>Improved: performance UX and reliability tuning.<br>Risked: metric misinterpretation if definitions unclear.<br>Enforce: define metrics precisely; render on dashboard; export CSV for analysis.<br>Regression detection: metric definitions tests; if drift => **DO_NOT_SHIP**.<br>Acceptance test: dashboard shows TTFR per run and weekly median; shows top failure causes.                                                        | P1=1 P2=0 P3=2 P4=2 (Total=5) | M / Low       |
| OBS-03 | **Diagnostics bundle export (zip) with manifests+logs+env+schema**<br>Rationale: Operator-grade support requires “one file” to attach/inspect offline.<br>Dependencies: META-01, META-04.<br>Improved: citeability and debuggability.<br>Risked: accidental inclusion of sensitive data unless controlled.<br>Enforce: CLI `stat-harness diag --run-id`; include allowlist of files; redact secrets (SEC-03).<br>Regression detection: diag content test; if includes secrets => **DO_NOT_SHIP**.<br>Acceptance test: bundle validates, contains run_manifest, report, plugin logs, system info, and artifact hashes. | P1=1 P2=1 P3=2 P4=4 (Total=8) | M / Low       |
| OBS-04 | **Run timeline visualization from plugin executions + queue time**<br>Rationale: Makes bottlenecks and partial failures obvious without log spelunking.<br>Dependencies: OBS-01.<br>Improved: ops and UX clarity.<br>Risked: none meaningful.<br>Enforce: UI renders bars by plugin duration; includes waiting/queued time per layer.<br>Regression detection: timeline must match event timestamps; if inconsistent => **DO_NOT_SHIP**.<br>Acceptance test: run detail shows consistent durations vs stored timestamps.                                                                                              | P1=1 P2=0 P3=1 P4=2 (Total=4) | M / Low       |
| OBS-05 | **Health endpoints (`/healthz`, `/readyz`) + disk/db/plugin checks**<br>Rationale: Prevent silent failures and enable local automation/monitoring.<br>Dependencies: FND-04, EXT-02.<br>Improved: stability and safer ops.<br>Risked: none meaningful.<br>Enforce: server exposes endpoints; checks include DB integrity quick check and free disk threshold.<br>Regression detection: health must fail when DB is corrupted; if passes => **DO_NOT_SHIP**.<br>Acceptance test: simulate low disk => `/readyz` fails with explicit reason.                                                                             | P1=0 P2=1 P3=1 P4=2 (Total=4) | S / Low       |
| OBS-06 | **Retention policies for logs/artifacts (prevent disk-full)**<br>Rationale: Disk-full creates cascading corruption and user pain on Windows/WSL.<br>Dependencies: OBS-01.<br>Improved: reliability and performance under long use.<br>Risked: data loss if retention too aggressive.<br>Enforce: configurable retention by age/count/size; soft-delete first; export before purge.<br>Regression detection: retention must never delete the latest N runs; if it does => **DO_NOT_SHIP**.<br>Acceptance test: set retention to 3 runs; after 10 runs, only oldest purged; newest preserved.                           | P1=2 P2=1 P3=0 P4=1 (Total=4) | M / Low       |
| OBS-07 | **Perf baseline harness + CI regression gates**<br>Rationale: Performance regressions are operational failures for large datasets.<br>Dependencies: PERF-02, QA-06.<br>Improved: sustained performance.<br>Risked: flaky CI if not stabilized.<br>Enforce: benchmark ingest + scan stats; compare within tolerance.<br>Regression detection: >X% slowdown => **DO_NOT_SHIP**.<br>Acceptance test: CI fails deterministically on regression and links to perf diff report.                                                                                                                                             | P1=2 P2=0 P3=1 P4=1 (Total=4) | M / Med       |

---

# VII. Security / Privacy: Local-first hardening + safe optional features

Current evidence: plugin runner blocks network unless enabled and blocks dangerous primitives (eval/pickle/shell).  Server bind-to-nonloopback is explicitly guarded by env.  Tenancy is feature-flagged. 

| ID     | Recommendation details                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |                 Pillar scores | Effort / Risk |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------: | ------------- |
| SEC-01 | **Local-only server binding + CSRF/session hardening for auth**<br>Rationale: Local UI should not become remotely reachable accidentally; if auth is enabled, sessions must be hardened.<br>Dependencies: none (build on existing guard). <br>Improved: security posture and safe defaults.<br>Risked: some remote-use workflows might need explicit flags.<br>Enforce: require explicit CLI flags for remote host binding; add CSRF token for POST routes; secure cookies.<br>Regression detection: attempt remote bind without flag must fail; if binds => **DO_NOT_SHIP**.<br>Acceptance test: `serve` without flag only binds 127.0.0.1; UI posts require CSRF.                                                                         | P1=0 P2=4 P3=0 P4=1 (Total=5) | M / Low       |
| SEC-02 | **Enforce sandbox write-surface minimization (paired with EXT-05)**<br>Rationale: Guards block “dangerous APIs,” but persistence attacks happen via filesystem writes; prevent writing to plugin directories and broad appdata paths by default.<br>Dependencies: EXT-05, QA-03.<br>Improved: integrity and containment.<br>Risked: plugin breakage if they wrote outside run dir.<br>Enforce: `core/sandbox.py` read/write allowlists; default write=run_dir only; require explicit capability for other writes.<br>Regression detection: sandbox escape suite; any write outside => **DO_NOT_SHIP**.<br>Acceptance test: plugin cannot write to `plugins/` or to unrelated appdata; can write to run artifacts only.                       | P1=1 P2=4 P3=1 P4=3 (Total=9) | M / High      |
| SEC-03 | **Secrets hygiene: redaction + env-var indirection for settings**<br>Rationale: Avoid storing API keys in DB/logs/manifests; keep runs auditable without leaking secrets.<br>Dependencies: META-01, OBS-03.<br>Improved: security and safe diagnostics.<br>Risked: misconfiguration if env vars missing.<br>Enforce: settings loader recognizes `{"$env":"NAME"}`; redacts matching patterns in logs and manifests.<br>Regression detection: secret-in-log tests; if secret appears => **DO_NOT_SHIP**.<br>Acceptance test: run with fake key; manifests store only `$env` reference; logs show “[REDACTED]”.                                                                                                                                | P1=0 P2=3 P3=1 P4=2 (Total=6) | M / Low       |
| SEC-04 | **Optional PII scanning/redaction plugin with consent toggles**<br>Rationale: Local-first doesn’t eliminate privacy risk; derived artifacts (reports/prompts) may leak PII.<br>Dependencies: EXT-04 capability model, UX-02 consent UI.<br>Improved: privacy and safer optional sharing.<br>Risked: false positives/negatives; must be clearly labeled as heuristic when applicable.<br>Enforce: dedicated plugin; outputs findings with confidence and evidence; redaction applied only when enabled.<br>Regression detection: PII fixture tests; if misses known PII => **DO_NOT_SHIP** for the plugin (not entire engine).<br>Acceptance test: dataset with known emails/SSNs; report shows detected fields; exports redact when toggled. | P1=1 P2=3 P3=2 P4=2 (Total=8) | L / Med       |
| SEC-05 | **Secure artifact serving: headers + CSP + path traversal tests**<br>Rationale: Artifact downloads must not allow path traversal or unsafe rendering in browser contexts.<br>Dependencies: QA-05 optional.<br>Improved: UI security robustness.<br>Risked: might break inline preview for some file types unless handled carefully.<br>Enforce: `ui/server.py` download endpoints use strict path join + content-type; add CSP headers and `X-Content-Type-Options`.<br>Regression detection: traversal test; if can read outside run dir => **DO_NOT_SHIP**.<br>Acceptance test: request `../state.sqlite`; server returns 404/403; all responses include security headers.                                                                 | P1=0 P2=3 P3=0 P4=2 (Total=5) | S / Low       |
| SEC-06 | **Tenancy boundary tests + enforced tenant_id scoping everywhere**<br>Rationale: Tenancy is feature-flagged ; if enabled, it must not be a footgun.<br>Dependencies: QA-04.<br>Improved: security boundaries and correctness.<br>Risked: schema/query churn.<br>Enforce: every DB query must include tenant scope; file paths include tenant partitioning; add unit tests for scoping.<br>Regression detection: cross-tenant access tests; any cross-read => **DO_NOT_SHIP**.<br>Acceptance test: create two tenants; run in A; B cannot query/trace/download artifacts.                                                                                                                                                                    | P1=0 P2=4 P3=1 P4=2 (Total=7) | M / Med       |
| SEC-07 | **Optional at-rest encryption hook/guidance for appdata**<br>Rationale: Some environments require encryption even for local-only tools.<br>Dependencies: FND-05 backups must work with encryption strategy.<br>Improved: security compliance posture.<br>Risked: complexity and key-loss risk.<br>Enforce: minimal hook: allow external encrypted filesystem path + explicit warning; optionally integrate SQLCipher only if permitted (no evidence in repo today).<br>Regression detection: backup/restore with encrypted path must work; if not => **DO_NOT_SHIP** for this feature.<br>Acceptance test: configure appdata to encrypted volume; runs succeed and backups restore.                                                          | P1=0 P2=3 P3=0 P4=1 (Total=4) | L / Med       |

---

# VIII. Performance: Windows 11 + WSL2 + large workloads

Current evidence: dataset accessor loads from SQLite into pandas and caches in-memory DataFrame (`self._df`) ; SQLite is configured for WAL+busy timeout which is good for concurrency but still sensitive to large queries. 

| ID      | Recommendation details                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |                 Pillar scores | Effort / Risk |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------: | ------------- |
| PERF-01 | **Streaming dataset access + enforce max in-memory loads**<br>Rationale: Cached full-DF loads are a predictable failure mode on large datasets under WSL/Windows memory pressure. <br>Dependencies: plugin updates to adopt streaming API.<br>Improved: performance and stability.<br>Risked: plugin rewrite required for some analyses.<br>Enforce: add `iter_batches()` in `dataset_io.py`; default `df()` requires explicit opt-in for full load or capped row limit.<br>Regression detection: large dataset fixture must not exceed memory; if it does => **DO_NOT_SHIP**.<br>Acceptance test: run scan stats on 5M-row dataset without OOM; measured peak RSS stays under configured cap. | P1=3 P2=0 P3=2 P4=1 (Total=6) | M / Med       |
| PERF-02 | **SQLite indices + ANALYZE + query plan checks**<br>Rationale: Deterministic performance requires deliberate indexing and plan stability.<br>Dependencies: none (but best with OBS-07).<br>Improved: throughput and latency.<br>Risked: migration time on existing DBs.<br>Enforce: add indices for common filters (dataset_version_id, row_index, plugin_results keys); run `ANALYZE` after ingest.<br>Regression detection: query plan snapshot test; if plan regresses => **DO_NOT_SHIP**.<br>Acceptance test: scan-stat queries remain within expected time on benchmark dataset.                                                                                                           | P1=3 P2=0 P3=1 P4=1 (Total=5) | S / Low       |
| PERF-03 | **Materialized derived columns/views with invalidation**<br>Rationale: Avoid repeated expensive parsing (e.g., numeric coercions) across plugins.<br>Dependencies: PERF-02, META-03 (fingerprints for invalidation correctness).<br>Improved: large-run performance.<br>Risked: cache invalidation complexity.<br>Enforce: create derived tables keyed by dataset_version_id; invalidate when dataset changes.<br>Regression detection: invalidation test; if stale values => **DO_NOT_SHIP**.<br>Acceptance test: repeated run of same analyses gets faster; outputs identical.                                                                                                                | P1=3 P2=0 P3=1 P4=1 (Total=5) | L / Med       |
| PERF-04 | **Parallelism tuning knobs + concurrency classes**<br>Rationale: Windows/WSL IO contention can make “more threads” slower; users need control.<br>Dependencies: OBS-02 (measure).<br>Improved: performance under varied workloads.<br>Risked: misconfiguration if defaults poor.<br>Enforce: add `--workers` and per-plugin concurrency class (CPU/IO); default sensible for 64GB RAM.<br>Regression detection: benchmark must not regress with defaults; if does => **DO_NOT_SHIP**.<br>Acceptance test: changing workers scales within expected bounds; timeline shows reduced queue time.                                                                                                    | P1=2 P2=0 P3=1 P4=1 (Total=4) | M / Low       |
| PERF-05 | **Optional GPU acceleration hook (stretch; no evidence today)**<br>Rationale: User environment has RTX 4090, but repo shows no GPU path; keep strictly optional to preserve determinism and enterprise constraints.<br>Dependencies: none (future).<br>Improved: performance only if adopted.<br>Risked: dependency and reproducibility risk; should not be default.<br>Enforce: plugin-level opt-in capability; record GPU details in manifest when used.<br>Regression detection: CPU baseline remains default; if GPU path silently triggers => **DO_NOT_SHIP**.<br>Acceptance test: enabling GPU flag changes manifest and produces identical results within tolerance where applicable.    | P1=2 P2=0 P3=0 P4=0 (Total=2) | L / High      |
| PERF-06 | **WSL/Windows IO guidance + temp dir placement warnings**<br>Rationale: Many perf issues are caused by storing large datasets on slow Windows-mounted paths inside WSL.<br>Dependencies: none.<br>Improved: operator success rate without deep tuning.<br>Risked: none meaningful.<br>Enforce: startup checks detect path location and warn; recommend storing appdata on Linux filesystem under WSL.<br>Regression detection: warning tests; if false positives => **DO_NOT_SHIP**.<br>Acceptance test: when appdata on `/mnt/c`, UI shows warning and offers one-click “move appdata” instructions (no admin).                                                                                | P1=2 P2=0 P3=0 P4=1 (Total=3) | S / Low       |

---

# IX. QA / Test strategy: Determinism + contracts + regressions

Current evidence: tests exist for upload limits and plugin discovery, suggesting a test harness foundation is present. 

| ID    | Recommendation details                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |                 Pillar scores | Effort / Risk |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------: | ------------- |
| QA-01 | **Golden fixture runs for deterministic report + trace outputs**<br>Rationale: Determinism is a core guarantee; golden fixtures catch accidental drift quickly.<br>Dependencies: META-01, EXEC-03.<br>Improved: correctness and citeability.<br>Risked: fixture brittleness if timestamps not normalized.<br>Enforce: canonicalize timestamps in fixtures or separate “content hash” fields from timestamps.<br>Regression detection: fixture diff in CI; any unexpected diff => **DO_NOT_SHIP**.<br>Acceptance test: running fixture twice yields identical output hashes; CI compares committed golden outputs.          | P1=0 P2=0 P3=3 P4=3 (Total=6) | M / Med       |
| QA-02 | **Schema contract tests for plugins + report schema validation**<br>Rationale: Prevent schema drift and broken UI rendering.<br>Dependencies: META-05.<br>Improved: accuracy and long-term stability.<br>Risked: initial failures reveal existing drift (expected).<br>Enforce: CI validates all plugin outputs against `output_schema` and validates report against `docs/report.schema.json`.<br>Regression detection: schema mismatch => **DO_NOT_SHIP**.<br>Acceptance test: any plugin output that violates schema fails the run and CI.                                                                              | P1=0 P2=0 P3=3 P4=3 (Total=6) | S / Low       |
| QA-03 | **Sandbox escape test suite (fs/network/eval/pickle/shell)**<br>Rationale: Runner already blocks risky primitives ; lock this behavior with tests so it never regresses.<br>Dependencies: EXT-05/SEC-02 for write restrictions if implemented.<br>Improved: security assurance.<br>Risked: none meaningful.<br>Enforce: dedicated tests execute a “malicious plugin” fixture and assert failures on forbidden actions.<br>Regression detection: any escape => **DO_NOT_SHIP**.<br>Acceptance test: plugin tries `socket.connect`, `eval`, `pickle.loads`, `subprocess.run`, and writing outside run dir; all are blocked. | P1=0 P2=4 P3=0 P4=1 (Total=5) | S / Low       |
| QA-04 | **Migration upgrade/rollback tests + backup verification**<br>Rationale: Schema changes are inevitable; tests prevent silent corruption or lost provenance.<br>Dependencies: FND-05.<br>Improved: reliability and citeability of stored history.<br>Risked: time to build robust fixtures.<br>Enforce: CI runs migrations forward/back on fixture DB; verifies invariants and backups.<br>Regression detection: migration failure => **DO_NOT_SHIP**.<br>Acceptance test: fixture DB upgrades; old runs still load; trace graph intact.                                                                                    | P1=0 P2=1 P3=2 P4=2 (Total=5) | M / Med       |
| QA-05 | **UI e2e smoke tests (Playwright)**<br>Rationale: Prevent UI regressions that break frictionless core workflow.<br>Dependencies: UX-02, UX-04, OBS-05.<br>Improved: reliability of the user path.<br>Risked: CI flakiness if not stabilized.<br>Enforce: run in headless mode; verify upload/run/report/trace/download paths.<br>Regression detection: any failure => **DO_NOT_SHIP**.<br>Acceptance test: full UI workflow completes on fixture dataset and produces expected report.                                                                                                                                     | P1=1 P2=0 P3=1 P4=2 (Total=4) | M / Med       |
| QA-06 | **Performance regression gates in CI (if regress => DO_NOT_SHIP)**<br>Rationale: Performance regressions are functional regressions at large scale.<br>Dependencies: OBS-07, PERF-02.<br>Improved: sustained throughput.<br>Risked: noisy measurements if not controlled.<br>Enforce: stable environment settings; compare medians; use tolerances.<br>Regression detection: >X% slower => **DO_NOT_SHIP**.<br>Acceptance test: ingest+scan stats benchmark stays within budget across commits.                                                                                                                            | P1=3 P2=0 P3=0 P4=1 (Total=4) | M / Med       |

---

# X. Roadmap: Phased plan Phase 0–3

This is sequencing guidance using the recommendation IDs above. No new requirements beyond the tables.

| Phase   | Scope                                                                                                              | Regression guards                                                                                  |
| ------- | ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| Phase 0 | Safety baselines: FND-02, EXEC-06, EXEC-07, EXT-02, SEC-05, QA-02, QA-03, PERF-02                                  | CI schema validation + sandbox escape tests + smoke workflow must pass; any failure => DO_NOT_SHIP |
| Phase 1 | Provenance + crash recovery: FND-01, FND-03, FND-04, FND-05, META-01, META-03, OBS-01, OBS-03                      | Golden fixture (QA-01) starts here; add kill/restart tests; hash verification required             |
| Phase 2 | Replay + caching + UX surfacing: FND-06, EXEC-01, EXEC-03, EXEC-04, EXEC-05, META-04, UX-01/02/04/05/08, OBS-02/04 | Deterministic output hashing gates; UI e2e tests required for core workflow                        |
| Phase 3 | Extensions lifecycle + privacy options: EXT-01/03/04/05/07/08, SEC-03/04/06/07, PERF-01/03/04/06                   | Compatibility tests + tenancy boundary tests + perf regression gates expanded                      |

---

## Top-20 quick wins

Highest total score with Effort ∈ {S, M}. (Deterministic ordering computed from the scoring table.)

| Rank | ID      | Recommendation                                                            | Total | Effort | Risk |
| ---- | ------- | ------------------------------------------------------------------------- | ----- | ------ | ---- |
| 1    | FND-01  | Atomic run directory + crash-safe journaling                              | 10    | M      | Med  |
| 2    | FND-03  | Harden upload CAS: verify-on-write + refcount + quarantine                | 10    | M      | Med  |
| 3    | META-01 | Persist canonical run_manifest.v1 capturing input+config+env              | 9     | M      | Low  |
| 4    | FND-05  | Online backup/restore + retention for state.sqlite                        | 9     | M      | Low  |
| 5    | META-03 | Store execution_fingerprint per plugin (code+manifest+settings+input)     | 9     | M      | Low  |
| 6    | EXEC-01 | Idempotent plugin caching keyed by hashes + --force/--reuse               | 9     | M      | Med  |
| 7    | EXT-05  | Sandbox policy redesign: read/write allowlists + narrow DB access         | 9     | M      | High |
| 8    | SEC-02  | Harden sandbox FS policies (read/write split) + prevent plugin dir writes | 9     | M      | High |
| 9    | META-05 | Schema versioning+compat checks (config/output/report) + hashes           | 8     | M      | Med  |
| 10   | META-08 | Export 'Repro pack' zip (manifest+schemas+logs+report)                    | 8     | M      | Low  |
| 11   | EXEC-05 | Replay mode: verify hashes then reuse cached results                      | 9     | L      | Med  |
| 12   | EXEC-02 | Per-plugin timeout + memory budget enforcement                            | 8     | M      | Med  |
| 13   | EXEC-03 | Deterministic persistence order + per-plugin derived seeds                | 8     | S      | Low  |
| 14   | EXEC-04 | Failure containment: temp dirs + commit-on-validate only                  | 8     | M      | Med  |
| 15   | EXEC-06 | Pre-run config validation against plugin schemas (deterministic defaults) | 8     | S      | Low  |
| 16   | SEC-04  | Optional PII scanning/redaction plugin with consent toggles               | 8     | L      | Med  |
| 17   | PERF-01 | Streaming dataset access + enforce max in-memory loads                    | 6     | M      | Med  |
| 18   | FND-06  | Deterministic run_fingerprint and optional deterministic IDs              | 8     | M      | Med  |
| 19   | OBS-03  | Diagnostics bundle export (zip) with manifests+logs+env+schema            | 8     | M      | Low  |
| 20   | QA-02   | Schema contract tests for plugins + report schema validation              | 6     | S      | Low  |

> Note: quick-win rank table includes some L items in the computed list above (EXEC-05, SEC-04). If you want the strict interpretation (S/M only), drop those two and promote the next highest S/M items (FND-04, META-07).

## Top-20 big bets

Highest total score regardless of effort.

| Rank | ID      | Recommendation                                                            | Total | Effort | Risk |
| ---- | ------- | ------------------------------------------------------------------------- | ----- | ------ | ---- |
| 1    | FND-01  | Atomic run directory + crash-safe journaling                              | 10    | M      | Med  |
| 2    | FND-03  | Harden upload CAS: verify-on-write + refcount + quarantine                | 10    | M      | Med  |
| 3    | META-01 | Persist canonical run_manifest.v1 capturing input+config+env              | 9     | M      | Low  |
| 4    | FND-05  | Online backup/restore + retention for state.sqlite                        | 9     | M      | Low  |
| 5    | META-03 | Store execution_fingerprint per plugin (code+manifest+settings+input)     | 9     | M      | Low  |
| 6    | EXEC-01 | Idempotent plugin caching keyed by hashes + --force/--reuse               | 9     | M      | Med  |
| 7    | EXEC-05 | Replay mode: verify hashes then reuse cached results                      | 9     | L      | Med  |
| 8    | EXT-05  | Sandbox policy redesign: read/write allowlists + narrow DB access         | 9     | M      | High |
| 9    | SEC-02  | Harden sandbox FS policies (read/write split) + prevent plugin dir writes | 9     | M      | High |
| 10   | META-05 | Schema versioning+compat checks (config/output/report) + hashes           | 8     | M      | Med  |
| 11   | META-08 | Export 'Repro pack' zip (manifest+schemas+logs+report)                    | 8     | M      | Low  |
| 12   | EXEC-02 | Per-plugin timeout + memory budget enforcement                            | 8     | M      | Med  |
| 13   | EXEC-03 | Deterministic persistence order + per-plugin derived seeds                | 8     | S      | Low  |
| 14   | EXEC-04 | Failure containment: temp dirs + commit-on-validate only                  | 8     | M      | Med  |
| 15   | EXEC-06 | Pre-run config validation against plugin schemas (deterministic defaults) | 8     | S      | Low  |
| 16   | EXT-03  | Offline install/update/rollback via signed plugin bundles                 | 8     | L      | Med  |
| 17   | SEC-04  | Optional PII scanning/redaction plugin with consent toggles               | 8     | L      | Med  |
| 18   | OBS-03  | Diagnostics bundle export (zip) with manifests+logs+env+schema            | 8     | M      | Low  |
| 19   | FND-06  | Deterministic run_fingerprint and optional deterministic IDs              | 8     | M      | Med  |
| 20   | FND-04  | Startup integrity checks: PRAGMA integrity_check + orphan cleanup         | 8     | S      | Low  |

---

## Red-team failure scenarios

### A) Red-team as a User: failure scenarios, detection, mitigation

| Scenario                                           | Detection signals                                                 | Mitigation                                                                                  |
| -------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Upload wrong dataset (fatigue)                     | Column/schema mismatch vs previous run; hash differs unexpectedly | UX-02 preview + show sha256 + schema diff before run; store “expected schema” per project   |
| Confuse run_seed semantics (0 vs random)           | Repeated runs unexpectedly identical/different                    | UX-04 confirm page shows resolved seed; META-07 shows determinism badge + seed derivation   |
| Assume full data when sampling occurred            | Sample fraction < 1 recorded                                      | FND-08 + UX-08 show “SAMPLED” banner and record sample method/seed in manifest              |
| Partial run mistaken as complete                   | Some plugins missing results                                      | UX-05 timeline shows missing plugins; run status becomes PARTIAL with explicit missing list |
| Plugin silently blocked by permissions             | Capability denied                                                 | EXT-04/UX-04 show permission denial before run; clear “not executed” state in run detail    |
| Think modeled outputs are measured                 | Missing evidence links                                            | META-02 + UX-08 force modeled/measured labeling; show “no evidence” explicitly              |
| Download artifact, later can’t prove origin        | No hash/manifest attached                                         | META-04 registry + embed run_id+hash in filename; repro pack export                         |
| Accidental enablement of optional network features | network flag true in manifest                                     | SEC-01 forces explicit flag; UI confirmation includes warning and records consent           |
| Misapply preset to wrong plugin version            | Schema hash mismatch                                              | UX-03 includes schema hash gating + diff preview; refuse apply if incompatible              |
| Accessibility gaps cause misclicks                 | Keyboard traps; missing labels                                    | UX-06 + QA-05 a11y checks; ensure all destructive actions require confirmation              |

### B) Red-team as a System Admin/Operator: failure scenarios, detection, mitigation

| Scenario                                            | Detection signals                   | Mitigation                                                                           |
| --------------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------ |
| SQLite corruption after crash/power loss            | integrity_check fails; missing rows | FND-04 integrity gate + FND-05 backup/restore + crash-safe run journaling            |
| Plugin persists by writing into `plugins/`          | Plugin code hash changes            | EXT-05/SEC-02 read/write sandbox split + META-03 code fingerprinting                 |
| Schema/config drift breaks UI or report             | schema validation failures          | META-05 + QA-02 contract tests; block incompatible plugins                           |
| Migration fails mid-upgrade                         | schema_version inconsistent         | QA-04 migration tests + FND-05 backups prior to migrations                           |
| Disk fills from artifacts/logs                      | low disk warnings                   | OBS-05 disk checks + OBS-06 retention + export-before-purge                          |
| Concurrency causes lock contention                  | “database is locked” errors         | PERF-04 tuning + improve busy handling + per-run checkpoints to resume               |
| Tenancy enabled but data leaks across tenants       | cross-tenant query results          | SEC-06 enforced scoping + tests; deny tenancy enable until passing self-check        |
| Vector store enabled but extension missing/unstable | vector init errors                  | META-06 records feature flags; EXT-06 health checks; disable vector store by default |
| Replay shows tampered artifacts as valid            | hash mismatch not checked           | EXEC-05 replay verifies hashes; fail closed on mismatch                              |
| Performance regression unnoticed                    | TTFR increases                      | OBS-07 perf gates + OBS-02 dashboard; stop shipping on regression                    |

---

## Minimal canonical metadata schema proposal

This is a **new** canonical schema (`run_manifest.v1`) that complements `docs/report.schema.json`. 

| Field                           | Type             | Example                                                     | Notes                                                  |
| ------------------------------- | ---------------- | ----------------------------------------------------------- | ------------------------------------------------------ |
| `schema_version`                | string           | `"run_manifest.v1"`                                         | Versioned contract; required                           |
| `run_id`                        | string           | `"a3f9...c1"`                                               | Current run ID is UUID4 hex                           |
| `run_fingerprint`               | string           | `"sha256:...`                                               | Deterministic identity of (input+config+plugins+flags) |
| `created_at`                    | string (RFC3339) | `"2026-02-06T10:11:12Z"`                                    | Use UTC consistently                                   |
| `input.upload_sha256`           | string           | `"sha256:...`                                               | Upload already computes sha256                        |
| `input.original_filename`       | string           | `"sales.xlsx"`                                              | For UX only; not identity                              |
| `input.bytes`                   | int              | `12345678`                                                  |                                                        |
| `ingest.row_count`              | int              | `1048576`                                                   | Must be measured                                       |
| `ingest.column_count`           | int              | `42`                                                        |                                                        |
| `config.resolved`               | object           | `{...}`                                                     | Fully resolved config after defaults                   |
| `config.sha256`                 | string           | `"sha256:...`                                               | Hash of resolved config (stable JSON)                 |
| `determinism.run_seed`          | int              | `12345`                                                     | Explicit; show derived seeds per plugin                |
| `plugins[]`                     | array            | `[{"id":"analysis_scan_statistics","version":"0.1.0",...}]` | Include code hash + schema hashes                      |
| `features.network_enabled`      | bool             | `false`                                                     | Default false; runner guards network                  |
| `features.vector_store_enabled` | bool             | `false`                                                     | Feature-flagged                                       |
| `execution.workers`             | int              | `8`                                                         | From pipeline concurrency settings                     |
| `artifacts[]`                   | array            | `[{ "path":"report.json","sha256":"...","bytes":...}]`      | Evidence registry                                      |
| `results.summary`               | object           | `{ "status":"COMPLETED", "failed_plugins":[...] }`          | Completion and failures                                |

---

## Minimal processing lineage model

The repository already has lineage concepts (entities + edges) and UI trace views (evidence in migrations + UI templates). 

**Canonical model (input → outputs)**

| Step | Entity           | Key IDs                     | Stored evidence                                |
| ---- | ---------------- | --------------------------- | ---------------------------------------------- |
| 1    | Raw upload       | `upload_sha256`             | CAS blob + size + first_seen                   |
| 2    | Dataset version  | `dataset_version_id`        | Ingest row/col counts + schema hash            |
| 3    | Run              | `run_id`, `run_fingerprint` | `run_manifest.json` + resolved config          |
| 4    | Plugin execution | `plugin_id`, `execution_id` | start/end, stdout/stderr, exit code            |
| 5    | Plugin result    | `result_id`                 | output JSON + references/evidence + debug      |
| 6    | Report           | `report_sha256`             | report.json + schema validation result         |
| 7    | Artifacts        | `artifact_sha256`           | registry + producer + mime + bytes             |
| 8    | Trace graph      | entity+edge IDs             | links: run → plugin exec → results → artifacts |

---

## Extension manager redesign

This is a **minimal viable spec** for a local-only extension manager aligned to the codebase’s plugin schema system. 

### Minimal viable spec

| Area        | MVP behavior                                                                       | Enforcement location                            |
| ----------- | ---------------------------------------------------------------------------------- | ----------------------------------------------- |
| Discovery   | List installed plugins with id/version/type and status                             | `core/plugin_manager.py`, UI `/plugins`         |
| Validate    | `plugins validate` validates schema + imports entrypoint + dry-run config defaults | CLI + `plugin_manager.py`                       |
| Permissions | Display required capabilities and deny by default for risky ones                   | plugin.yaml + `plugin_runner.py` + `sandbox.py` |
| Install     | Offline: import a `.zip` bundle and install into appdata                           | new `core/plugin_registry.py`                   |
| Update      | Install new version side-by-side                                                   | registry pointer switch                         |
| Rollback    | Switch pointer to previous version deterministically                               | registry pointer switch + audit event           |
| Health      | Show per-plugin health state + last validation result                              | manager UI + `health()` contract                |

### Stretch goals

| Area                           | Stretch behavior                            | Why                               |
| ------------------------------ | ------------------------------------------- | --------------------------------- |
| Signed bundles                 | Signature verification (hash-based)         | Supply-chain hygiene              |
| Capability negotiation         | Policy-based approvals (per project/tenant) | Reduce prompts + prevent mistakes |
| Sandboxing profiles            | Predefined sandbox profiles per plugin type | Safer defaults                    |
| Automated compatibility matrix | Prevent known-bad plugin combinations       | Admin pain reduction              |

---

## UI/UX sketch in text

### Home: Activity Dashboard

```
+--------------------------------------------------------------+
| Statistic Harness — Activity                                  |
| [New Run] [Upload Dataset] [Manage Plugins] [Diagnostics]      |
+--------------------------------------------------------------+
| Recent Runs (filter: [status] [plugin] [dataset] [date])      |
|--------------------------------------------------------------|
| Run ID   Fingerprint   Status   TTFR   Duration   Determinism |
| a3f9..   9c12..        OK       2.1s   01:12      ✅ seeded    |
| b771..   12aa..        PARTIAL  3.8s   00:45      ✅ seeded    |
| c020..   77ef..        FAILED   —      00:08      ✅ seeded    |
+--------------------------------------------------------------+
| Alerts:                                                      |
| - Low disk space warning (3.2GB free)                         |
| - 2 plugins failing health checks                             |
+--------------------------------------------------------------+
```

### Input ingest/status panel

```
+-------------------- Upload Dataset --------------------------+
| Select file: [Choose...]  (CSV/XLSX)                          |
| Preview: 20 rows  | Columns: 42 | Rows (estimated): 1,048,576 |
| SHA-256: sha256:...   Size: 12.3MB   Format: XLSX            |
|                                                                  |
| [x] Allow sampling for analysis (defaults OFF)                   |
|                                                                  |
| [Upload]  [Cancel]                                              |
+------------------------------------------------------------------+
```

### Extension manager main screen

```
+-------------------- Plugins -----------------------------------+
| Tabs: [Installed] [Available] [Updates] [Permissions] [Logs]     |
+------------------------------------------------------------------+
| Installed                                                       |
|------------------------------------------------------------------|
| Plugin ID              Version   Status   Permissions   Actions  |
| analysis_scan_statistics 0.1.0    OK       DB_READ       [Disable]
| llm_prompt_builder      0.2.1    BLOCKED  NETWORK       [Review..]
| report_bundle           1.0.0    OK       FS_WRITE(run)  [Disable]
+------------------------------------------------------------------+
| [Install bundle (.zip)]   [Validate all]                         |
+------------------------------------------------------------------+
```

### Run/job detail view

```
+---------------- Run Detail ------------------------------------+
| Run: a3f9..   Fingerprint: 9c12..   Status: COMPLETED ✅         |
| Input: sha256:...  Rows: 1,048,576  Columns: 42                 |
| Flags: network=OFF  vector=OFF  tenancy=OFF                      |
| Seed: 12345  Workers: 8                                          |
+------------------------------------------------------------------+
| Timeline (TTFR: 2.1s)                                            |
| ingest ███  transform ██  analysis ███████  report ███            |
+------------------------------------------------------------------+
| Plugins                                                         |
| - ingest_tabular         OK   4.2s   output hash: ...             |
| - analysis_scan_statistics OK  12.1s  output hash: ...             |
| - report_bundle          OK   2.0s   report hash: ...             |
+------------------------------------------------------------------+
| Artifacts (click to verify/download)                             |
| report.json  sha256:...  122KB  (produced by report_bundle)      |
| diag.zip     sha256:...  2.1MB                                    |
+------------------------------------------------------------------+
| Evidence                                                        |
| Modeled vs measured: [MEASURED] for row counts, [MODELED] for ...|
| Confidence: 0.87   Evidence links: [trace graph] [row trace]      |
+------------------------------------------------------------------+
| Actions: [Replay] [Export repro pack] [Download diagnostics]      |
+------------------------------------------------------------------+
```

---

## Open questions

|  # | Question                                                                                                                              |
| -: | ------------------------------------------------------------------------------------------------------------------------------------- |
|  1 | Should the system treat `run_seed=0` as a literal seed or as “auto-derive from fingerprint”? (UI currently shows a run_seed field.)  |  auto-derive
|  2 | Are plugins expected to ever write outside run directories, or can we break that behavior to enable read/write allowlists?            | you can break
|  3 | Is multi-tenancy intended for real use or only experiments behind the flag?                                                          | dont even experiment, single tenant only
|  4 | What is the intended retention policy for uploads and run artifacts in appdata?                                                       |  60 days, after that only the data telling us full path and when something was uploaded and the run artifacts if recommended
|  5 | Should report artifacts be considered immutable once written (append-only), and enforced cryptographically?                           |  whatever is optimal for the 4 pillars
|  6 | Which plugin outputs are “measured” vs “modeled” today, and where is that distinction documented?                                     |  most all are measured, only ones containing model in their name are models.
|  7 | Do we need a “project” entity as a first-class object (beyond runs/datasets) with expected schema + presets?                          |  No, it is dataset based. however we do need an ERP entity to contain the known issues and defaults for any dataset from the same erp
|  8 | What is the largest expected dataset size (rows/cols/bytes), and what is the target TTFR budget?                                      |  ive seen up to 10 million rows so it needs to be capable of streaming an optimal number of rows, storing insights on that and so on to then come up with global insights.
|  9 | Should the system support “headless” batch mode for CI (no UI) with deterministic export bundles?                                     |  yes. ui is nice for me occasionally especially when uploading and monitoring status and configs. cli sucks for that but is great for testing
| 10 | Is the vector store meant to be part of default workflows or strictly optional?                                                      |  default workflow
| 11 | Do we require cryptographic signing of plugin bundles, or is hash-based integrity sufficient for local-only?                          |  hash based is fine
| 12 | What is the authoritative source of “engine version” for compatibility checks (package version vs git hash)?                          |  whatever is optimal for the 4 pillars

---

THREAD: STATISTIC-HARNESS-ADVERSARIAL-REDESIGN
CHAT_ID: N/A
TS: 2026-02-06
