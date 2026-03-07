# Red-Team Status (2026-03-05)

Source spec: `statistics_harness_redteam_2026-03-05.md`

## Phase 1: Immediate Unblocking — COMPLETE

| ID | Status | Evidence | Notes |
|---|---|---|---|
| CRIT-01 | done | `src/statistic_harness/core/planner.py:240-245` | Flipped `include_untagged` default to True (opt-out instead of opt-in) |
| CRIT-03 | done | `scripts/run_full_test_dataset.sh:52`, `scripts/start_full_loaded_dataset_bg.sh:124`, `scripts/run_new_dataset_bounded.sh:35` | `STAT_HARNESS_ENABLE_TOPO_TDA` defaults to 1 in all run scripts |
| HIGH-03 | done | `scripts/run_full_test_dataset.sh:53`, `scripts/start_full_loaded_dataset_bg.sh:125`, `scripts/run_new_dataset_bounded.sh:36` | `STAT_HARNESS_PLANNER_INCLUDE_UNTAGGED` defaults to 1 in all run scripts |

## Phase 2: Status Normalization — COMPLETE

| ID | Status | Evidence | Notes |
|---|---|---|---|
| CRIT-04 | done | 74 plugin.py files changed, 181 replacements. 0 `PluginResult("skipped")` remain. | Changed all `"skipped"` to `"na"`. Tests updated (172 test files). |
| HIGH-04 | done | `src/statistic_harness/core/pipeline.py:2267-2289` | Empty-ok counter now excludes `plugin_not_applicable` and `analysis_no_action_diagnostic` finding kinds |

## Phase 3: Capability Tagging — COMPLETE

| ID | Status | Evidence | Notes |
|---|---|---|---|
| HIGH-01 | done | 0 plugin.yaml files contain `capabilities: [analysis]` | Removed from all 206 files via `scripts/backfill_plugin_capabilities.py` |
| CRIT-02 | done | `scripts/backfill_plugin_capabilities.py` created and run | 84 plugins tagged `needs_numeric`, 66 `needs_eventlog`, 34 `needs_timestamp`, 13 `needs_host`, 13 `topo_tda_addon`, 3 `diagnostic_only`. Remaining ~280 thin-wrapper plugins get `[]` and are included via `include_untagged=True`. |
| MED-01 | done | `config/plugin_kind_map.yaml` — 467/467 analysis plugins mapped | Added 4 missing: `causal_recommendations_v1`, `changepoint_detection_v1`, `process_mining_conformance_v1`, `process_mining_discovery_v1` |

## Phase 4: Resilience — COMPLETE

| ID | Status | Evidence | Notes |
|---|---|---|---|
| CRIT-05 | done | `tests/plugins/test_analysis_close_cycle_window_resolver.py` (pre-existing), `src/statistic_harness/core/pipeline.py` critical dependency check | Added `_CRITICAL_DEPS` cascade detection for window_resolver, profile_basic, profile_eventlog. Emits `critical_dependency_failed` policy violation on error. |
| MED-02 | done | Covered by CRIT-05 `_CRITICAL_DEPS` dict | `profile_basic` (37 dependents) and `profile_eventlog` (12 dependents) both monitored |
| MED-03 | done (already handled) | `src/statistic_harness/core/close_cycle.py:resolve_close_cycle_masks` | Function already has fallback logic via `fallback_reason` field when backtrack data is missing |
| MED-04 | done (already satisfied) | `src/statistic_harness/core/stat_plugins/runbook30_surrogates.py:58-69` | Only 10 plugins use `run_surrogate`, all 10 already have titles in `_PLUGIN_TITLES`. Redteam's "59 stubs" count was incorrect. |

## Phase 5: Cleanup — COMPLETE

| ID | Status | Evidence | Notes |
|---|---|---|---|
| HIGH-02 | done | `scripts/run_loaded_dataset_full.py:2122-2150` | `--plugin-set full` now runs planner `select_plugins` as pre-filter. Prints `PLANNER_PRE_FILTER=selected:N,skipped:M`. Falls back gracefully on error. |
| MED-05 | done | `src/statistic_harness/core/pipeline.py:1975-1979` | `ingest_tabular` stripped from expanded deps and missing_deps when `input_file is None` |

## Validation

```bash
# Verify planner selects all plugins
STAT_HARNESS_PLANNER_INCLUDE_UNTAGGED=1 STAT_HARNESS_ENABLE_TOPO_TDA=1 \
  .venv/bin/python -c "
from pathlib import Path
from statistic_harness.core.plugin_manager import PluginManager
mgr = PluginManager(Path('plugins'))
specs = mgr.discover()
analysis = [s for s in specs if s.type == 'analysis']
print(f'Discovered {len(specs)} plugins ({len(analysis)} analysis)')
"

# Verify no skipped status in plugins
grep -rn 'PluginResult("skipped"' plugins/*/plugin.py | wc -l  # expect 0

# Verify no capabilities: [analysis] in plugin.yaml
grep -c 'capabilities:.*analysis' plugins/*/plugin.yaml | grep -v ':0$' | wc -l  # expect 0

# Verify all plugin.yaml files parse
.venv/bin/python -c "
import yaml; from pathlib import Path
errors = [str(f) for f in Path('plugins').glob('*/plugin.yaml')
          if not isinstance(yaml.safe_load(f.read_text()), dict)]
print(f'Parse errors: {len(errors)}')
"

# Verify kind map covers all analysis plugins
.venv/bin/python -c "
import yaml; from pathlib import Path
km = set(yaml.safe_load(Path('config/plugin_kind_map.yaml').read_text()).get('mappings',{}).keys())
ap = {yaml.safe_load(f.read_text())['id'] for f in Path('plugins').glob('*/plugin.yaml')
      if yaml.safe_load(f.read_text()).get('type')=='analysis'}
print(f'Unmapped: {len(ap-km)}')
"
```

## Update Rules

- Update this file in the same commit as the code change.
- Every status transition must include updated Evidence column.
- Do not mark `done` without a test or verification command that confirms the fix.
