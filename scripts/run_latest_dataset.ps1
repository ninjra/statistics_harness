$ErrorActionPreference = "Stop"

$RepoRootWsl = "/mnt/d/projects/statistics_harness/statistics_harness"
$PythonWsl = "$RepoRootWsl/.venv_wsl/bin/python"
$DataFileWsl = "$RepoRootWsl/appdata/uploads/e9c3e32292cf42f2a36624ce44c0d7c2/proc log 1-14-26.csv"

$Command = "cd $RepoRootWsl; STAT_HARNESS_CLI_PROGRESS=1 $PythonWsl -m statistic_harness.cli run --file '$DataFileWsl' --plugins auto --run-seed 123"

& wsl.exe -- bash -lc $Command
