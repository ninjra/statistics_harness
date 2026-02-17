# Golden Release Baseline Inputs

## Purpose
Pin the canonical run inputs used for deterministic before/after golden-release comparisons.

## Dataset Baseline
- ERP: `quorum`
- baseline dataset version id: `3246cc7cd7d57a317ddc05e80e6f6f5bfe7f50deb0ee7af8db50d04bae180e1a`
- synthetic dataset version id (6mo): `de7c1da5a4ea6e8c684872d7857bb608492f63a9c7e0b7ca014fa0f093a88e66`
- synthetic dataset version id (8mo): `d1cf34daa3ad6289468136707d204d486245dfc46b8f4a3212111f4308cdbf39`

## Run Baseline
- run_before (pre-change): `eb325f419aa64de392bc941d09d16f02`
- run_after (post-change, full set): `c17a88de24624053bf5f201d664d7f22`
- plugin_set: `full`
- run_seed: `123`

## Reproduction Commands
- Baseline full run:
  - `scripts/run_loaded_dataset_full.py --dataset-version-id <id> --run-seed 123 --plugin-set full --force`
- Dataset comparison:
  - `scripts/compare_dataset_runs.py --dataset-before <id_before> --dataset-after <id_after> --plugin-set full --seed 123`
