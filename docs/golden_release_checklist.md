# Golden Release Checklist

## Preflight
- required: `python -m pytest -q`
- required: `stat-harness list-plugins`
- required: `python scripts/verify_docs_and_plugin_matrices.py`

## Golden Runtime Policy
- required: `STAT_HARNESS_NETWORK_MODE` set for target run (`off` or `localhost`)
- required: `STAT_HARNESS_GOLDEN_MODE` set (`default` or `strict`)
- required: `STAT_HARNESS_MAX_FULL_DF_ROWS` reviewed for large dataset policy

## Full Gauntlet
- required: baseline real dataset full run completed
- required: synthetic 6-month dataset full run completed
- required: synthetic 8-month dataset full run completed
- required: in strict mode, zero `skipped` plugins
- required: in default mode, any `skipped` plugins have deterministic/citable reasons

## Evidence Pack
- required: `python scripts/golden_release_delta_map.py`
- required: `python scripts/rank_streaming_offenders.py`
- required: `python scripts/build_golden_release_evidence_pack.py --latest 3`
- required: `python scripts/compare_dataset_runs.py --dataset-before <id> --dataset-after <id>`

## Final Validation
- required: `report.json` + `report.md` present for all target runs
- required: 4-pillar scorecard present for all target runs
- required: determinism rerun confirms stable outcome with same seed/dataset DB state
