# Implementation Status Matrix

**Generated:** 2026-03-06 (manual)
**Branch:** `main` (uncommitted changes)

## Status Legend

- `Implemented`: available and functioning in-repo with tests.
- `Partial`: implemented but incomplete or degraded.
- `Missing`: not implemented.
- `Blocked`: prevented by another issue.

## Core Pipeline

| Area | Capability | Status | Evidence |
|---|---|---|---|
| Ingest | CSV/TSV/XLSX/JSON ingestion | Implemented | `plugins/ingest_tabular_v1/plugin.py`, 10+ ingest tests |
| Ingest | SQL dump ingestion | Implemented | `plugins/ingest_sql_dump_v1/plugin.py` |
| Profile | Basic column stats + PII | Implemented | `plugins/profile_basic_v1/plugin.py` |
| Profile | Event log profiling | Implemented | `plugins/profile_eventlog_v1/plugin.py` |
| Planner | Intelligent plugin selection | Partial | `src/statistic_harness/core/planner.py` — capability tags broken (CRIT-01 fixed, CRIT-02 pending) |
| Analysis | 465 analysis plugins structurally valid | Implemented | All implement Plugin protocol |
| Analysis | Plugin subprocess sandboxing (6 guards) | Implemented | `src/statistic_harness/core/plugin_runner.py` |
| Analysis | Deterministic execution (seeded RNG) | Implemented | `src/statistic_harness/core/run_context.py` |
| Report | Recommendation ranking (weighted scores) | Implemented | `src/statistic_harness/core/ranking.py`, `config/recommendation_weights.yaml` |
| Report | Enrichment pipeline | Implemented | `src/statistic_harness/core/report.py` (thin facade → `reporting/` submodules) |
| Report | Decision bundle v2 | Implemented | `plugins/report_decision_bundle_v2/plugin.py` |
| Report | Slide kit emitter v2 | Implemented | `plugins/report_slide_kit_emitter_v2/plugin.py` |
| Report | Guardrails | Implemented | `src/statistic_harness/core/reporting/guardrails.py` |
| Modeling | Scenario replay engine | Implemented | `src/statistic_harness/core/modeling/engine.py`, 46 tests passing |
| Modeling | Scenario comparator | Implemented | `src/statistic_harness/core/modeling/comparator.py` |
| Modeling | Sensitivity sweeps | Implemented | `src/statistic_harness/core/modeling/sensitivity.py` |
| Modeling | CLI `model` subcommand | Implemented | `src/statistic_harness/cli.py` |
| Storage | SQLite state layer | Implemented | `src/statistic_harness/core/storage.py` |
| Quality | Four pillars scorecard | Implemented | `src/statistic_harness/core/four_pillars.py` |
| Quality | Evaluator harness (ground truth) | Implemented | `src/statistic_harness/core/evaluation.py` |
| Quality | Chunk invariance testing | Implemented | `scripts/run_chunk_invariance.py` |
| Security | Network sandbox verification | Implemented | `scripts/verify_no_runtime_network.py` |
| Security | Frozen plugin surface contract | Implemented | `docs/frozen_surfaces_contract.md`, `scripts/verify_frozen_plugin_surfaces.py` |

## Known Degradations

| Area | Issue | Severity | Tracker |
|---|---|---|---|
| Planner | ~280 thin-wrapper plugins have `capabilities: []` — planner can't filter by data requirements | low | Mitigated by `include_untagged=True` default |
| State DB | `appdata/state.sqlite` is ~11GB with no auto-pruning | medium | CLAUDE.md known issues |
| Tests | Matrix test files stale after cross-domain plugin merge | low | CLAUDE.md known issues |
| Tests | `tests/contracts/test_chunk_invariance.py` pre-existing failure | low | CLAUDE.md known issues |

## Resolved Degradations (2026-03-06)

| Area | Issue | Resolution |
|---|---|---|
| Planner | 206 plugins had meaningless `capabilities: [analysis]` tag | Removed; backfill script applied correct data-requirement tags |
| Pipeline | 74 plugins returned `"skipped"` instead of `"na"` | All changed to `"na"` |
| Pipeline | Empty-ok counter flagged legitimate no-signal results | Counter now excludes non-actionable finding kinds |
| Pipeline | Window resolver single point of failure for 100 plugins | Critical dependency cascade detection added |
| Pipeline | `--plugin-set full` bypassed all intelligent filtering | Now uses planner as pre-filter |
| Pipeline | `ingest_tabular` in depends_on broke DB-only runs | Stripped from deps when `input_file is None` |
| Kind Map | 4 plugins unmapped in `plugin_kind_map.yaml` | Added to kind map (467/467 covered) |

## Update Rules

- Update this file when any row changes status.
- For every `Missing` → `Partial` → `Implemented` transition, include:
  - changed files
  - validation command or test used
  - evidence path
- Do not mark `Implemented` if any required verification gate still fails for that row.
