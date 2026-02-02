$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RootDir

function Resolve-Python {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return @("py", "-3")
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @("python")
    }
    throw "Python not found. Install Python 3.11+ and re-run."
}

$PythonCmd = Resolve-Python
$PythonExe = $PythonCmd[0]
$PythonArgs = @()
if ($PythonCmd.Length -gt 1) {
    $PythonArgs = $PythonCmd[1..($PythonCmd.Length - 1)]
}
$VenvActivate = Join-Path $RootDir ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $VenvActivate)) {
    & $PythonExe @PythonArgs -m venv .venv
}

. $VenvActivate

$env:STAT_HARNESS_SAFE_RENAME = "1"
$env:PYTHONPATH = "$RootDir\\tools" + ($(if ($env:PYTHONPATH) { ";" + $env:PYTHONPATH } else { "" }))

$VecScript = @'
import importlib.util
import pathlib

spec = importlib.util.find_spec("sqlite_vec")
path = ""
if spec and spec.origin:
    base = pathlib.Path(spec.origin).resolve().parent
    matches = [p for p in base.rglob("vec0.*") if p.suffix in (".so", ".dylib", ".dll")]
    if matches:
        path = str(matches[0])
print(path)
'@
$VecPath = & $PythonExe @PythonArgs -c $VecScript
if ($VecPath) {
    $env:STAT_HARNESS_ENABLE_VECTOR_STORE = "1"
    $env:STAT_HARNESS_SQLITE_VEC_PATH = $VecPath
}

python -m pip install -U pip setuptools wheel
python -m pip install -e ".[dev]" --no-build-isolation
python -m pip install sqlite-vec

python -m statistic_harness.cli serve
