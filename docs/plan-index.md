# Plan Index

Last updated: 2026-03-06

Implementation matrix source of truth:
- `docs/implementation-status-matrix.md` (hand-maintained rollup)
- `docs/redteam-status.md` (redteam item tracker)

## Active Plans

| File | Status | Summary |
|---|---|---|
| `statistics_harness_redteam_2026-03-05.md` | `active` | Red-team report: planner capability-tag mismatch, skipped→na, golden mode false positives, capability backfill |
| `modeling_harness_design.md` | `completed` | What-if modeling harness: scenario replay, comparator, sensitivity sweeps |
| `docs/four-pillars-implementation-plan.md` | `active` | Five-sprint 4-pillars quality framework (Performant/Accurate/Secure/Citable) |
| `docs/each-plugin-delivering-actionable-results-plan.md` | `active` | Per-plugin actionability improvements |
| `docs/kona-current-to-ideal-pathfinding-actionability-plan.md` | `active` | Kona route planning actionability |
| `docs/non-actionable-reasons-resolution-plan.md` | `active` | Resolve non-actionable recommendation patterns |
| `docs/plugin-actionability-recommended-next-steps-plan.md` | `active` | Plugin actionability next steps |

## Completed Plans

| File | Status | Summary |
|---|---|---|
| `modeling_harness_design.md` | `completed` | All 4 phases done, 46 tests passing |
| `docs/deprecated/completed_plans/codex-plugins-unimplemented-plan.md` | `completed` | 66 unimplemented plugins — all now implemented |
| `docs/deprecated/completed_plans/topo-tda-addon-pack-plan.md` | `completed` | 13 topo/TDA plugins |
| `docs/deprecated/completed_plans/top20-additional-methods-plugins-plan.md` | `completed` | 20 leftfield methods |
| `docs/deprecated/completed_plans/codex-stat-plugins-spec-pack-plan.md` | `completed` | Stat plugins spec pack |
| `docs/deprecated/completed_plans/two-million-rows-support-plan.md` | `completed` | Streaming/chunked processing for large datasets |
| `docs/deprecated/completed_plans/intelligent-plugin-orchestrator-two-lanes-plan.md` | `completed` | Two-lane orchestrator |
| `docs/deprecated/completed_plans/known-issue-gauntlet-recommendation-plan.md` | `completed` | Known issue rediscovery |

## Update Rules

- Add new plans here when created.
- Move plans to Completed when all items are done and verified.
- Do not delete completed plans — move them and note completion date.
- Every active plan must have a corresponding status tracker (either in `docs/redteam-status.md` or in the plan file itself).
