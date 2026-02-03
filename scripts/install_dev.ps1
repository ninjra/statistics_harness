$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$env:STAT_HARNESS_SAFE_RENAME = "1"
$env:PYTHONPATH = "$RootDir\\tools" + ($(if ($env:PYTHONPATH) { ";" + $env:PYTHONPATH } else { "" }))

$VenvDir = Join-Path $RootDir ".venv"
if (-not (Test-Path $VenvDir)) {
    $PythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCmd) {
        $PythonCmd = Get-Command py -ErrorAction SilentlyContinue
    }
    if (-not $PythonCmd) {
        throw "Python not found. Install Python 3 to continue."
    }
    & $PythonCmd.Source -m venv $VenvDir
}

$PipTmp = Join-Path $RootDir ".pip-tmp"
if (-not (Test-Path $PipTmp)) {
    New-Item -ItemType Directory -Path $PipTmp | Out-Null
}
$env:TEMP = $PipTmp
$env:TMP = $PipTmp

& (Join-Path $VenvDir "Scripts\\Activate.ps1")
python -m pip install -e ".[dev]" --no-build-isolation
