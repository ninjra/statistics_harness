param(
  [string]$DatasetVersionId = "",
  [int]$RunSeed = 123,
  [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

function Test-BadPythonPath($exePath) {
  return ($exePath -match "WindowsApps" -or $exePath -match "wsl.exe" -or $exePath -match "^/usr/" -or $exePath -match "\\wsl$")
}

function Get-PythonCmd {
  param([string]$Preferred)

  if ($Preferred) {
    $Preferred = ($Preferred -replace "[\r\n\t]", "").Trim()
    $Preferred = $Preferred -replace ":\s+\\", ":\\"  # fix accidental newline/space after drive colon
    $Preferred = $Preferred -replace "\\\s+", "\\"    # fix accidental whitespace after backslash

    if ($Preferred -eq "py") {
      $cmd = Get-Command py -ErrorAction SilentlyContinue
      if ($cmd) { return @{ Exe = $cmd.Source; Args = @("-3") } }
      throw "Python launcher 'py' not found on PATH."
    }
    $cmd = Get-Command $Preferred -ErrorAction SilentlyContinue
    if ($cmd) {
      $exe = $cmd.Source
      if (Test-BadPythonPath $exe) {
        Write-Host "Preferred Python resolves to '$exe' (not usable for native Windows). Falling back to WSL." -ForegroundColor Yellow
      } else {
        return @{ Exe = $exe; Args = @() }
      }
    }
    if (Test-Path $Preferred) {
      $exe = (Resolve-Path $Preferred).Path
      if (Test-BadPythonPath $exe) {
        Write-Host "Preferred Python resolves to '$exe' (not usable for native Windows). Falling back to WSL." -ForegroundColor Yellow
      } else {
        return @{ Exe = $exe; Args = @() }
      }
    }
  }

  foreach ($name in @("py","python3","python")) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if (!$cmd) { continue }
    if ($name -eq "py") {
      return @{ Exe = $cmd.Source; Args = @("-3") }
    }
    $exe = $cmd.Source
    if (Test-BadPythonPath $exe) { continue }
    return @{ Exe = $exe; Args = @() }
  }
  return $null
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPath = Join-Path $RepoRoot ".venv"
$PythonCmd = Get-PythonCmd -Preferred $PythonExe

$env:STAT_HARNESS_APPDATA = (Join-Path $RepoRoot "appdata")

$tmp = Join-Path $env:TEMP "stat_harness_run_analysis.py"
$py = @"
import os
import sqlite3
from pathlib import Path
import yaml

from statistic_harness.core.pipeline import Pipeline
from statistic_harness.core.tenancy import get_tenant_context
from statistic_harness.core.report import build_report, write_report

def latest_dataset_version_id(db_path: Path) -> str | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT dataset_version_id FROM dataset_versions ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return row["dataset_version_id"] if row else None

ctx = get_tenant_context()

db_path = ctx.appdata_root / "state.sqlite"

dataset_version_id = os.environ.get("DATASET_VERSION_ID", "")
if not dataset_version_id:
    dataset_version_id = latest_dataset_version_id(db_path)
if not dataset_version_id:
    raise SystemExit("No dataset_version_id found. Upload data first.")

run_seed = int(os.environ.get("RUN_SEED", "123"))

analysis_ids = []
for manifest in Path("plugins").glob("*/plugin.yaml"):
    data = yaml.safe_load(manifest.read_text())
    if data.get("type") == "analysis":
        analysis_ids.append(data["id"])
analysis_ids = sorted(set(analysis_ids))

pipeline = Pipeline(ctx.appdata_root, Path("plugins"), tenant_id=ctx.tenant_id)
run_id = pipeline.run(
    input_file=None,
    plugin_ids=analysis_ids,
    settings={},
    run_seed=run_seed,
    dataset_version_id=dataset_version_id,
)
run_dir = ctx.tenant_root / "runs" / run_id
report = build_report(pipeline.storage, run_id, run_dir, Path("docs/report.schema.json"))
write_report(report, run_dir)
print(f"RUN_ID={run_id}")
print(str(run_dir / "report.md"))
"@

Set-Content -Path $tmp -Value $py -Encoding UTF8

function Invoke-WSLRun {
  $wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
  if (!$wsl) { throw "WSL not found. Install WSL or provide a native Windows Python." }
  $RepoRootWsl = (& wsl.exe wslpath -u $RepoRoot).Trim()
  $TmpWsl = (& wsl.exe wslpath -u $tmp).Trim()
  if (-not $RepoRootWsl) { throw "Failed to resolve WSL path for $RepoRoot" }
  if (-not $TmpWsl) { throw "Failed to resolve WSL path for $tmp" }
  $DatasetArg = $DatasetVersionId
  $SeedArg = $RunSeed
  $bash = @"
set -e
cd "$RepoRootWsl"
if [ ! -d ".venv_wsl" ]; then
  python3 -m venv .venv_wsl
fi
. .venv_wsl/bin/activate
python -m pip install -e .
STAT_HARNESS_APPDATA="$RepoRootWsl/appdata" DATASET_VERSION_ID="$DatasetArg" RUN_SEED="$SeedArg" python "$TmpWsl"
"@
  $output = & wsl.exe bash -lc $bash
  if (!$output) { throw "WSL run produced no output." }
  $runIdLine = ($output | Where-Object { $_ -match "^RUN_ID=" } | Select-Object -First 1)
  if (!$runIdLine) { throw "Could not parse RUN_ID from WSL output." }
  $runId = $runIdLine -replace "^RUN_ID=", ""
  $reportWsl = "$RepoRootWsl/appdata/runs/$runId/report.md"
  $reportWin = (& wsl.exe wslpath -w $reportWsl).Trim()
  Write-Host $runId
  Write-Host $reportWin
}

if ($PythonCmd -eq $null) {
  Invoke-WSLRun
  Remove-Item $tmp -ErrorAction SilentlyContinue
  exit 0
}

$VenvPython = Join-Path $VenvPath "Scripts\\python.exe"
if (!(Test-Path $VenvPython)) {
  & $PythonCmd.Exe @($PythonCmd.Args) -m venv $VenvPath
}
if (!(Test-Path $VenvPython)) {
  Write-Host "Native venv not available, falling back to WSL." -ForegroundColor Yellow
  Invoke-WSLRun
  Remove-Item $tmp -ErrorAction SilentlyContinue
  exit 0
}

& $VenvPython -m pip install -e $RepoRoot | Out-Host

$env:DATASET_VERSION_ID = $DatasetVersionId
$env:RUN_SEED = [string]$RunSeed
& $VenvPython $tmp
Remove-Item $tmp -ErrorAction SilentlyContinue
