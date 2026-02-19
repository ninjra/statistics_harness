@echo off
setlocal
wsl -d Ubuntu-24.04 -- /mnt/d/projects/statistics_harness/statistics_harness/.venv/bin/python /mnt/d/projects/statistics_harness/statistics_harness/tools/cyberpunk_style_probe.py --force-color
endlocal
