$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$env:STAT_HARNESS_SAFE_RENAME = "1"
$env:PYTHONPATH = "$RootDir\\tools" + ($(if ($env:PYTHONPATH) { ";" + $env:PYTHONPATH } else { "" }))

pip install -e ".[dev]" --no-build-isolation
