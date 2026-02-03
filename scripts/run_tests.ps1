$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
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

& (Join-Path $VenvDir "Scripts\Activate.ps1")
$env:PYTHONPATH = "$RootDir" + ($(if ($env:PYTHONPATH) { ";" + $env:PYTHONPATH } else { "" }))

python -m pytest -q @args
