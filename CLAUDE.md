# Statistics Harness

## Project Overview

Local-only, plugin-first statistical analysis harness for tabular data. Accepts CSV/TSV/XLSX/JSON files, runs a configurable pipeline of statistical analysis plugins in sandboxed subprocesses, and produces ranked actionable recommendations as both human-readable (`report.md`) and machine-readable (`report.json`) outputs.

The core value proposition: ingest an arbitrary dataset, run 275 statistical analysis plugins against it, and produce a prioritized list of business recommendations ranked by modeled impact (delta hours saved, touch reduction, contention reduction). All execution is deterministic (seeded RNG per run), all plugins run in sandboxed subprocesses with no network access by default.

The project is in active development with a "four pillars" quality framework: Performant, Accurate, Secure, Citable.

## Architecture

### Language & Framework
- **Python 3.11+** (running 3.12.3 in current venv)
- **FastAPI** for UI/API server (optional)
- **SQLite** for state storage (`appdata/state.sqlite`)
- **src/ layout** (`src/statistic_harness/`)

### Core Components

```
src/statistic_harness/
├── cli.py                    # Entry point: `stat-harness` command
├── core/
│   ├── pipeline.py           # Main execution pipeline orchestrator
│   ├── plugin_runner.py      # Subprocess sandbox executor (security guards, resource limits)
│   ├── plugin_manager.py     # Plugin discovery and lifecycle
│   ├── plugin_registry.py    # Plugin registry for thin-wrapper delegation
│   ├── report.py             # Report builder (filtering, ranking, enrichment) — largest file
│   ├── ranking.py            # Weighted scoring: delta_hours(0.40), hours(0.25), touches(0.20), contention(0.15)
│   ├── recommendation_filters.py  # Process targeting and action type classification
│   ├── normalized_reader.py  # Streaming batch reader for large datasets
│   ├── run_context.py        # Deterministic RNG and seed propagation
│   ├── windowing.py          # Accounting window resolution
│   ├── user_effort_model.py  # Effort scoring: (hours * 1.0) + (touches ** 0.5)
│   ├── storage.py            # SQLite state layer
│   ├── dataset_io.py         # Dataset I/O
│   ├── four_pillars.py       # Four-pillars scorecard
│   ├── evaluation.py         # Evaluator framework (ground_truth.yaml)
│   ├── known_issue_compiler.py    # Known issue template matching
│   ├── types.py              # Plugin protocol, PluginResult, PluginContext
│   ├── stat_plugins/         # Statistical plugin utilities (FDR, sampling, etc.)
│   ├── leftfield_top20/      # 20 specialized analysis implementations
│   └── ... (50+ more modules)
├── ui/                       # FastAPI UI server
```

### Plugin System (275 plugins)

Plugins live in `plugins/<plugin_id>/` with `plugin.py` + `plugin.yaml`. Every plugin implements:

```python
class Plugin:
    def run(self, ctx: PluginContext) -> PluginResult:
        ...
```

**Plugin categories:**
- **256 analysis** — statistical tests, anomaly detection, causal inference, clustering, time-series, process mining, graph analysis, survival models, etc.
- **6 transform** — entity resolution, link graphs, SQL materialization, normalization
- **6 report** — bundle, decision, evidence index, payout, plain English, slide kit
- **2 ingest** — tabular (CSV/JSON/Excel), SQL dump
- **2 profile** — basic column stats + PII detection, event log profiling
- **2 LLM** — prompt builder (PII-aware), local text2sql
- **1 planner** — plugin selection and planning

**Three implementation patterns:**
1. **Thin wrapper** (158 plugins): delegates to `stat_plugins.registry.run_plugin()`
2. **Full implementation** (111 plugins): direct logic in `plugin.py`
3. **Openplanter delegator** (6 plugins): delegates to `openplanter_pack`

**Sandbox security (plugin_runner.py):**
- Network guard (blocks sockets)
- Eval guard (blocks eval/exec from project code)
- Pickle guard (blocks untrusted deserialization)
- Shell guard (blocks subprocess/os.system)
- SQLite monitor
- File sandbox (read/write path whitelisting)
- Resource limits (memory, CPU, timeout)
- Deterministic environment (PYTHONHASHSEED, thread limits)

### Data Flow

```
Ingest (CSV/XLSX/JSON) → Profile → Plan → Analysis (275 plugins) → Report → Output
                                                                        ↓
                                                              Recommendations
                                                              (ranked by weighted score)
```

### Key Scripts (scripts/)
- `run_release_gate.py` — orchestrates multi-step release validation
- `run_loaded_dataset_full.py` — full pipeline execution
- `evaluator_harness.py` — evaluates report against ground truth
- `show_actionable_results.py` — formats/filters actionable results
- `build_final_validation_summary.py` — comprehensive validation
- `audit_plugin_actionability.py` — plugin actionability audit
- `run_chunk_invariance.py` — tests chunk-size invariance
- `verify_no_runtime_network.py` — network sandbox compliance
- `verify_plugin_test_coverage.py` — test coverage verification
- 80+ more audit, matrix, and validation scripts

### Configuration (config/)
- `actionability_thresholds.yaml` — minimum thresholds for actionable recommendations
- `recommendation_weights.yaml` — ranking weight distribution (sums to 1.0)
- `reporting.yaml` — top-N recommendation limit (default: 20)
- `chunk_invariance_tolerances.yaml` — strict tolerances for determinism testing
- `plugin_test_coverage_exemptions.json` — 61 plugins validated via integration rather than unit tests
- `app.yaml` — appdata dir, max upload size

## Environment

- **Python:** 3.12.3 (requires >=3.11)
- **Venv:** `.venv/` (use `.venv/bin/python`)
- **Platform:** WSL2 (Linux 6.6.87 on Windows) — multiple WSL-specific fixes in place
- **GPU:** NVIDIA RTX 4090 24GB, CUDA 12.8 available (PyTorch 2.9.1 + cupy-cuda12x installed)
- **Key packages:** pandas 3.0, numpy 2.4, scikit-learn 1.8, networkx 3.6, fastapi 0.128, pydantic 2.12, torch 2.9
- **Storage:** SQLite state database at `appdata/state.sqlite` (~11GB with cached run data)
- **Install:** `make dev` (WSL) or `pip install -e ".[dev]"` (editable install)

### Feature Flags (env vars)
- `STAT_HARNESS_NETWORK_MODE=off|localhost|on` (default: off — no network)
- `STAT_HARNESS_ENABLE_AUTH=1` — enable authentication
- `STAT_HARNESS_ENABLE_TENANCY=1` — enable multi-tenancy
- `STAT_HARNESS_ENABLE_VECTOR_STORE=1` — enable vector store
- `STAT_HARNESS_MAX_UPLOAD_BYTES` — upload size limit
- `STAT_HARNESS_MAX_FULL_DF_ROWS` — dataframe row limit (default: 1,000,000)
- `STAT_HARNESS_PRE_REPORT_FILTER_MODE=strict` — strict pre-report filtering (auto-set in tests)

### Running Tests
```bash
.venv/bin/python -m pytest -q          # 921 tests, ~37s collection
```

### CLI
```bash
stat-harness list-plugins              # list all plugins
python scripts/run_release_gate.py --run-id <id>   # release gate
```

## Current State

### What's Working
- Full plugin pipeline: ingest → profile → plan → analysis → report
- All 275 plugins are structurally valid and implement the `Plugin` protocol
- 921 tests collected; comprehensive security/sandbox/determinism coverage
- Release gate orchestration (`run_release_gate.py`) with multi-step validation
- Plugin subprocess sandboxing with 6 orthogonal security guards
- Deterministic execution via run_seed propagation
- Recommendation ranking with weighted scoring and process targeting
- Known issue rediscovery matching
- Four pillars scorecard generation
- Evaluator harness with ground truth validation
- Chunk invariance testing for streaming correctness
- CUDA/GPU available for torch-based plugins

### What's In Progress (current branch: feature/next-20260226-postmerge)
- Post-merge stabilization after merging `feature/next-20260226-stability`
- New core modules added but uncommitted: `normalized_reader.py`, `ranking.py`, `recommendation_filters.py`, `run_context.py`, `user_effort_model.py`, `windowing.py`
- New scripts added but uncommitted: `evaluator_harness.py`, `run_chunk_invariance.py`, `verify_no_runtime_network.py`, `verify_plugin_test_coverage.py`, `verify_report_artifacts_contract.py`, `audit_plugin_process_targeting.py`, `audit_plugin_streaming_contract.py`, `audit_plugin_targeting_windows.py`
- New config files uncommitted: `chunk_invariance_tolerances.yaml`, `plugin_test_coverage_exemptions.json`, `recommendation_weights.yaml`, `reporting.yaml`
- 15 new test files uncommitted
- Docs matrices modified (implementation_matrix, binding_matrix, redteam_ids, etc.)

### What's Not Yet Implemented
- `unimplemented/codex_decision_report_v2.md` — decision-oriented stakeholder reports (spec exists)
- `unimplemented/codex_stat_plugins_spec_pack.md` — additional plugin families (spec exists)
- 3 test files referenced in four-pillars plan but not yet created:
  - `tests/contracts/test_chunk_invariance.py`
  - `tests/perf/test_plugin_resource_budget.py`
  - `tests/integration/test_known_issue_rediscovery_baseline.py`

## Known Issues

1. **61 plugins lack direct unit tests** — validated only via integration/matrix contracts (tracked in `config/plugin_test_coverage_exemptions.json`)
2. **Large state database** — `appdata/state.sqlite` is ~11GB; no automatic pruning
3. **report.py is massive** — the main report builder has 100+ helper functions; complex multi-stage filtering pipeline with extensive env var feature flags
4. **Stale files in repo root** — single-letter scripts (`g`, `r`, `s`, `w`) are convenience wrappers that depend on external linter at `~/.codex/skills/shell-lint-ps-wsl/`; `get` is an empty file
5. **Missing four-pillars test files** — 3 tests referenced in the plan not yet created (see above)
6. **GEMINI.md is empty** — 0-byte file at repo root
7. **Heavy venv** — includes full PyTorch+CUDA stack, likely oversized for the core statistical harness functionality
8. **No pre-commit hooks configured** — repo-hygiene runs in CI only (`.github/workflows/repo-hygiene.yml`)

## Session Log

<!-- Updated at end of each session -->

## Decisions

1. **Local-only by default** — No network calls at runtime. Network mode is opt-in via env var. This is a Phase 1 hard constraint.
2. **Plugin subprocess isolation** — Each plugin runs in its own subprocess with 6 security guards (network, eval, pickle, shell, sqlite, filesystem). Defense-in-depth.
3. **Deterministic execution** — Every run uses a `run_seed` propagated through `RunContext.child_seed()`. PYTHONHASHSEED, thread counts, and RNG all controlled.
4. **src/ layout** — Package under `src/statistic_harness/` with `--import-mode=importlib` in pytest. Standard Python packaging.
5. **SQLite for state** — All run state, plugin results, and caching stored in SQLite. No external database required.
6. **Thin-wrapper plugin pattern** — 158/275 plugins delegate to a central registry, allowing implementation changes without touching plugin.py files.
7. **Weighted ranking** — Recommendations ranked by: modeled_delta_hours_close_dynamic (0.40) + modeled_delta_hours (0.25) + manual_touch_reduction_count (0.20) + close_contention_reduction_pct (0.15). Weights in config YAML.
8. **Contract-first testing** — Many plugins validated through schema contracts and integration matrices rather than individual unit tests. 61 explicit exemptions tracked.
9. **Four pillars framework** — All quality work organized under Performant/Accurate/Secure/Citable. Five-sprint implementation plan in `docs/four-pillars-implementation-plan.md`.
10. **Strict pre-report filtering in tests** — `STAT_HARNESS_PRE_REPORT_FILTER_MODE=strict` auto-set via conftest fixture for test consistency.

## Next Steps

1. **Commit uncommitted work** — 30+ new files and modifications on `feature/next-20260226-postmerge` need staging and commit
2. **Create missing four-pillars test files** — `test_chunk_invariance.py`, `test_plugin_resource_budget.py`, `test_known_issue_rediscovery_baseline.py`
3. **Close plugin test coverage gap** — Reduce the 61-plugin exemption list by adding targeted tests
4. **Implement decision report v2** — Spec exists at `unimplemented/codex_decision_report_v2.md`
5. **Address report.py complexity** — Consider breaking the 100+ function report builder into focused modules
6. **Database pruning** — Add automatic state.sqlite pruning/archival (currently 11GB)
7. **Clean repo root** — Remove or gitignore stale files (`g`, `r`, `s`, `w`, `get`, empty `GEMINI.md`)
8. **Merge to main** — Current branch diverges significantly from main; plan merge strategy
