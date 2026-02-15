# Golden Release Baseline Inputs

## Purpose
Pin the canonical run inputs used for deterministic before/after golden-release comparisons.

## Dataset Baseline
- ERP: `quorum`
- baseline dataset version id: `TBD_AT_RUN_TIME`
- synthetic dataset version id (6mo): `TBD_AT_RUN_TIME`
- synthetic dataset version id (8mo): `TBD_AT_RUN_TIME`

## Run Baseline
- run_before (pre-change): `TBD_AT_RUN_TIME`
- run_after (post-change): `TBD_AT_RUN_TIME`
- plugin_set: `full`
- run_seed: `123`

## Reproduction Commands
- Baseline full run:
  - `scripts/run_loaded_dataset_full.py --dataset-version-id <id> --run-seed 123 --plugin-set full --force`
- Dataset comparison:
  - `scripts/compare_dataset_runs.py --dataset-before <id_before> --dataset-after <id_after> --plugin-set full --seed 123`

