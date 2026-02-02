.\.venv\Scripts\Activate.ps1

$PythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonCmd) {
    $PythonCmd = Get-Command py -ErrorAction SilentlyContinue
}
if ($PythonCmd) {
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
    $VecPath = & $PythonCmd -c $VecScript
    if ($VecPath) {
        $env:STAT_HARNESS_ENABLE_VECTOR_STORE = "1"
        $env:STAT_HARNESS_SQLITE_VEC_PATH = $VecPath
    }
}

stat-harness run --file tests/fixtures/synth_linear.csv --plugins ingest_tabular,profile_basic,report_bundle --run-seed 42
