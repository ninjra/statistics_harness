@echo off
setlocal
set "RUN_ID=full_loaded_3246cc7c_20260218T221940Z"
if not "%~1"=="" set "RUN_ID=%~1"
wsl -d Ubuntu-24.04 -- /mnt/d/projects/statistics_harness/statistics_harness/.venv/bin/python /mnt/d/projects/statistics_harness/statistics_harness/scripts/show_actionable_results.py --run-id "%RUN_ID%" --top-n 10 --max-per-plugin 10 --theme cyberpunk
endlocal
