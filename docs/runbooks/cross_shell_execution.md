# Cross-Shell Execution Runbook

All commands below are one-line and valid in both shell lanes used by this repo.

## Validate
- Bash (WSL): `.venv/bin/python -m pytest -q`
- PowerShell: `.venv\\Scripts\\python.exe -m pytest -q`

## Plugin Validate
- Bash (WSL): `.venv/bin/python -m statistic_harness.cli plugins validate --json`
- PowerShell: `.venv\\Scripts\\python.exe -m statistic_harness.cli plugins validate --json`

## Full Loaded Dataset
- Bash (WSL): `bash scripts/start_full_loaded_dataset_bg.sh`
- PowerShell: `powershell.exe -File scripts\\run_gauntlet_latest.ps1`

## Cross-Dataset Openplanter Pack
- Bash (WSL): `.venv/bin/python scripts/run_cross_dataset_openplanter_pack.py --input tests/fixtures/synth_linear.csv`
- PowerShell: `.venv\\Scripts\\python.exe scripts\\run_cross_dataset_openplanter_pack.py --input tests\\fixtures\\synth_linear.csv`

