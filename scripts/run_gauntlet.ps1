param(
  [string]$DatasetVersionId = "",
  [int]$RunSeed = 123,
  [string]$PythonExe = "",
  [ValidateSet("console","open","none")]
  [string]$ReportView = "console"
)

$ErrorActionPreference = "Stop"

$IsWsl = $false
if ($env:WSL_DISTRO_NAME -or $env:WSL_INTEROP) {
  $IsWsl = $true
} elseif (Test-Path "/proc/version") {
  try {
    $versionText = Get-Content -Path "/proc/version" -ErrorAction SilentlyContinue
    if ($versionText -match "microsoft") { $IsWsl = $true }
  } catch {
    $IsWsl = $false
  }
}

function Test-BadPythonPath($exePath) {
  return ($exePath -match "WindowsApps" -or $exePath -match "wsl.exe" -or $exePath -match "^/usr/" -or $exePath -match "\\wsl$")
}

function Test-PythonExe {
  param(
    [string]$Exe,
    [string[]]$Args,
    [int]$MinMajor = 3,
    [int]$MinMinor = 11
  )
  try {
    $out = & $Exe @Args -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
    if ($LASTEXITCODE -ne 0) { return $false }
    if ($out -notmatch "^\d+\.\d+$") { return $false }
    $parts = $out.Split(".")
    $maj = [int]$parts[0]
    $min = [int]$parts[1]
    if ($maj -gt $MinMajor) { return $true }
    if ($maj -lt $MinMajor) { return $false }
    return ($min -ge $MinMinor)
  } catch {
    return $false
  }
}

function Get-PythonCmd {
  param([string]$Preferred)

  if ($Preferred) {
    $Preferred = ($Preferred -replace "[\r\n\t]", "").Trim()
    $Preferred = $Preferred -replace ":\s+\\", ":\\"  # fix accidental newline/space after drive colon
    $Preferred = $Preferred -replace "\\\s+", "\\"    # fix accidental whitespace after backslash

    if ($Preferred -eq "py") {
      $cmd = Get-Command py -ErrorAction SilentlyContinue
      if ($cmd) {
        $candidate = @{ Exe = $cmd.Source; Args = @("-3") }
        if (Test-PythonExe $candidate.Exe $candidate.Args) { return $candidate }
      }
      throw "Python launcher 'py' not found on PATH."
    }
    $cmd = Get-Command $Preferred -ErrorAction SilentlyContinue
    if ($cmd) {
      $exe = $cmd.Source
      if (Test-BadPythonPath $exe) {
        Write-Host "Preferred Python resolves to '$exe' (not usable for native Windows). Falling back to WSL." -ForegroundColor Yellow
      } else {
        $candidate = @{ Exe = $exe; Args = @() }
        if (Test-PythonExe $candidate.Exe $candidate.Args) { return $candidate }
      }
    }
    if (Test-Path $Preferred) {
      $exe = (Resolve-Path $Preferred).Path
      if (Test-BadPythonPath $exe) {
        Write-Host "Preferred Python resolves to '$exe' (not usable for native Windows). Falling back to WSL." -ForegroundColor Yellow
      } else {
        $candidate = @{ Exe = $exe; Args = @() }
        if (Test-PythonExe $candidate.Exe $candidate.Args) { return $candidate }
      }
    }
  }

  foreach ($name in @("py","python3","python")) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if (!$cmd) { continue }
    if ($name -eq "py") {
      $candidate = @{ Exe = $cmd.Source; Args = @("-3") }
      if (Test-PythonExe $candidate.Exe $candidate.Args) { return $candidate }
      continue
    }
    $exe = $cmd.Source
    if (Test-BadPythonPath $exe) { continue }
    $candidate = @{ Exe = $exe; Args = @() }
    if (Test-PythonExe $candidate.Exe $candidate.Args) { return $candidate }
  }
  return $null
}

function Show-Result {
  param(
    [string[]]$OutputLines
  )
  if (!$OutputLines) { throw "Gauntlet run produced no output." }
  $runIdLine = ($OutputLines | Where-Object { $_ -match "^RUN_ID=" } | Select-Object -First 1)
  $reportLine = ($OutputLines | Where-Object { $_ -match "^REPORT=" } | Select-Object -First 1)
  $truthLine = ($OutputLines | Where-Object { $_ -match "^GROUND_TRUTH=" } | Select-Object -First 1)
  $okLine = ($OutputLines | Where-Object { $_ -match "^GAUNTLET_OK=" } | Select-Object -First 1)
  $failLines = ($OutputLines | Where-Object { $_ -match "^FAILURE=" })
  $reportPath = ""

  if ($runIdLine) { Write-Host ($runIdLine -replace "^RUN_ID=", "") }
  if ($reportLine) {
    $reportPath = ($reportLine -replace "^REPORT=", "")
    Write-Host $reportPath
  }
  if ($truthLine) { Write-Host ("Ground truth: " + ($truthLine -replace "^GROUND_TRUTH=", "")) }

  $okValue = ""
  if ($okLine) { $okValue = ($okLine -replace "^GAUNTLET_OK=", "").Trim().ToLowerInvariant() }
  $isOk = $okValue -in @("true", "1", "yes", "y")
  Write-Host "GAUNTLET RESULT:" -NoNewline
  if ($isOk) {
    Write-Host " YES" -ForegroundColor Green
  } else {
    Write-Host " NO" -ForegroundColor Red
    foreach ($line in $failLines) {
      Write-Host ("- " + ($line -replace "^FAILURE=", ""))
    }
  }

  return [pscustomobject]@{
    ReportPath = $reportPath
    IsOk = $isOk
  }
}

function Show-ReportView {
  param(
    [string]$ReportPath,
    [string]$Mode
  )
  if ([string]::IsNullOrWhiteSpace($ReportPath)) { return }
  if ($Mode -eq "open") {
    Start-Process $ReportPath | Out-Null
    return
  }
  if ($Mode -eq "console") {
    Write-Host ""
    Write-Host "===== REPORT ====="
    Get-Content -Path $ReportPath
  }
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPath = Join-Path $RepoRoot ".venv"
$PythonCmd = $null
if ($PythonExe) {
  $PythonCmd = Get-PythonCmd -Preferred $PythonExe
}

$env:STAT_HARNESS_APPDATA = (Join-Path $RepoRoot "appdata")
$LogPath = Join-Path $env:STAT_HARNESS_APPDATA "gauntlet_last.log"
if (!(Test-Path $env:STAT_HARNESS_APPDATA)) {
  New-Item -ItemType Directory -Force -Path $env:STAT_HARNESS_APPDATA | Out-Null
}

function Write-Log {
  param([string[]]$Lines)
  if ($null -eq $Lines) { $Lines = @() }
  Add-Content -Path $LogPath -Value ($Lines -join "`n") -Encoding UTF8
}

Set-Content -Path $LogPath -Value ("started_at=" + (Get-Date).ToString("s")) -Encoding UTF8
Write-Log -Lines @("repo_root=" + $RepoRoot)

$TempRoot = $env:TEMP
if ([string]::IsNullOrWhiteSpace($TempRoot)) { $TempRoot = $env:TMP }
if ([string]::IsNullOrWhiteSpace($TempRoot)) { $TempRoot = "/tmp" }
$tmp = Join-Path $TempRoot "stat_harness_run_gauntlet.py"
$py = @"
import os
import sqlite3
from pathlib import Path
import yaml

from statistic_harness.core.pipeline import Pipeline
from statistic_harness.core.tenancy import get_tenant_context
from statistic_harness.core.report import build_report, write_report
from statistic_harness.core.evaluation import evaluate_report
from statistic_harness.core.utils import now_iso, json_dumps

def latest_dataset_version_id(db_path: Path) -> str | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT dataset_version_id FROM dataset_versions ORDER BY row_count DESC, created_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return row["dataset_version_id"] if row else None

def ground_truth_template(report: dict) -> str:
    features = [
        f.get("feature")
        for f in report.get("plugins", {})
        .get("analysis_gaussian_knockoffs", {})
        .get("findings", [])
    ]
    template = {
        "strict": False,
        "features": [f for f in features if f],
        "changepoints": [],
        "dependence_shift_pairs": [],
        "anomalies": [],
        "min_anomaly_hits": 0,
        "changepoint_tolerance": 3,
    }
    return yaml.safe_dump(template)

ctx = get_tenant_context()
db_path = ctx.appdata_root / "state.sqlite"

dataset_version_id = os.environ.get("DATASET_VERSION_ID", "")
if not dataset_version_id:
    dataset_version_id = latest_dataset_version_id(db_path)
if not dataset_version_id:
    raise SystemExit("No dataset_version_id found. Upload data first.")

run_seed = int(os.environ.get("RUN_SEED", "123"))

pipeline = Pipeline(ctx.appdata_root, Path("plugins"), tenant_id=ctx.tenant_id)
ctx_row = pipeline.storage.get_dataset_version_context(dataset_version_id)
if not ctx_row:
    raise SystemExit(f"Dataset version not found: {dataset_version_id}")

project_id = ctx_row.get("project_id")
settings = {"__run_meta": {"plugins": ["all"]}}

columns_meta = pipeline.storage.fetch_dataset_columns(dataset_version_id)
columns = [row.get("original_name") for row in columns_meta if row.get("original_name")]
colset = set(columns)

def has(name: str) -> bool:
    return name in colset

base = {}
if has("PROCESS_ID"):
    base["process_column"] = "PROCESS_ID"
elif has("PROCESS_QUEUE_ID"):
    base["process_column"] = "PROCESS_QUEUE_ID"
if has("LOCAL_MACHINE_ID"):
    base["host_column"] = "LOCAL_MACHINE_ID"
    base["server_column"] = "LOCAL_MACHINE_ID"
if has("QUEUE_DT"):
    base["queue_column"] = "QUEUE_DT"
if has("START_DT"):
    base["start_column"] = "START_DT"
    base["timestamp_column"] = "START_DT"
if has("END_DT"):
    base["end_column"] = "END_DT"

if base:
    settings["analysis_queue_delay_decomposition"] = {
        k: base[k]
        for k in ("process_column", "queue_column", "start_column", "end_column")
        if k in base
    }
    settings["analysis_close_cycle_contention"] = {
        k: base[k]
        for k in ("process_column", "server_column", "timestamp_column", "start_column", "end_column")
        if k in base
    }
    settings["analysis_close_cycle_capacity_impact"] = {
        k: base[k]
        for k in ("process_column", "host_column", "queue_column", "eligible_column", "start_column", "end_column")
        if k in base
    }
    settings["analysis_close_cycle_capacity_model"] = {
        k: base[k]
        for k in ("process_column", "host_column", "queue_column", "eligible_column", "start_column", "end_column")
        if k in base
    }
    settings["analysis_capacity_scaling"] = {
        k: base[k]
        for k in ("process_column", "host_column", "queue_column", "eligible_column", "start_column")
        if k in base
    }

run_id = pipeline.run(
    input_file=None,
    plugin_ids=["all"],
    settings=settings,
    run_seed=run_seed,
    dataset_version_id=dataset_version_id,
    project_id=project_id,
)
run_dir = ctx.tenant_root / "runs" / run_id
report = build_report(pipeline.storage, run_id, run_dir, Path("docs/report.schema.json"))
write_report(report, run_dir)

ground_truth = None
source = "template"
if project_id:
    project_row = pipeline.storage.fetch_project(project_id)
    if project_row and project_row.get("erp_type"):
        erp_type = str(project_row.get("erp_type") or "unknown").strip() or "unknown"
        known = pipeline.storage.fetch_known_issues(erp_type, "erp_type")
        if known:
            payload = {
                "strict": bool(known.get("strict", False)),
                "notes": known.get("notes") or "",
                "expected_findings": known.get("expected_findings") or [],
            }
            ground_truth = yaml.safe_dump(payload, sort_keys=False)
            source = "known"

if ground_truth is None:
    ground_truth = ground_truth_template(report)
    source = "template"

gt_path = run_dir / "ground_truth.yaml"
gt_path.write_text(ground_truth, encoding="utf-8")
ok, messages = evaluate_report(run_dir / "report.json", gt_path)
eval_payload = {
    "evaluated_at": now_iso(),
    "result": "passed" if ok else "failed",
    "ok": bool(ok),
    "messages": messages,
}
(run_dir / "evaluation.json").write_text(json_dumps(eval_payload), encoding="utf-8")

print(f"RUN_ID={run_id}")
print(f"REPORT={run_dir / 'report.md'}")
print(f"GROUND_TRUTH={source}")
print(f"GAUNTLET_OK={ok}")
for msg in messages:
    print(f"FAILURE={msg}")
"@

Set-Content -Path $tmp -Value $py -Encoding UTF8

function Invoke-WSLRun {
  $wsl = $null
  if (-not $IsWsl) {
    $wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
    if (!$wsl) { throw "WSL not found. Install WSL or provide a native Windows Python." }
  }
  function Convert-ToWslPath {
    param([string]$Path)
    $full = (Resolve-Path $Path).Path
    if ($full -match "^([A-Za-z]):\\(.+)$") {
      $drive = $Matches[1].ToLowerInvariant()
      $rest = $Matches[2] -replace "\\", "/"
      return "/mnt/$drive/$rest"
    }
    return ($full -replace "\\", "/")
  }
  $RepoRootWsl = Convert-ToWslPath $RepoRoot
  $TmpWsl = Convert-ToWslPath $tmp
  $RepoRootWsl = $RepoRootWsl.Replace("`r", "").Replace("`n", "")
  $TmpWsl = $TmpWsl.Replace("`r", "").Replace("`n", "")
  if (-not $RepoRootWsl) { throw "Failed to resolve WSL path for $RepoRoot" }
  if (-not $TmpWsl) { throw "Failed to resolve WSL path for $tmp" }
  $DatasetArg = $DatasetVersionId
  $SeedArg = $RunSeed
  $bashTmp = Join-Path $TempRoot "stat_harness_run_gauntlet.sh"
  $bashLines = @(
    'set -e',
    "cd `"$RepoRootWsl`"",
    'PYTHON_BIN=""',
    'for candidate in python3.12 python3.11 python3; do',
    '  if command -v "$candidate" >/dev/null 2>&1; then',
    '    ver=$("$candidate" -c "import sys; print(str(sys.version_info[0]) + \".\" + str(sys.version_info[1]))")',
    '    major=${ver%%.*}',
    '    minor=${ver#*.}',
    '    if [ "$major" -gt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; }; then',
    '      PYTHON_BIN="$candidate"',
    '      break',
    '    fi',
    '  fi',
    'done',
    'if [ -z "$PYTHON_BIN" ]; then echo "WSL_PYTHON_MISSING=python3.11+"; exit 1; fi',
    'if [ ! -d ".venv_wsl" ]; then "$PYTHON_BIN" -m venv .venv_wsl; fi',
    '. .venv_wsl/bin/activate',
    "export PYTHONPATH=`"$RepoRootWsl`"",
    "STAT_HARNESS_APPDATA=`"$RepoRootWsl/appdata`" PYTHONPATH=`"$RepoRootWsl`" DATASET_VERSION_ID=`"$DatasetArg`" RUN_SEED=`"$SeedArg`" python `"$TmpWsl`""
  )
  $bashText = ($bashLines -join "`n")
  [System.IO.File]::WriteAllText($bashTmp, $bashText, (New-Object System.Text.UTF8Encoding($false)))
  $bashTmpWsl = Convert-ToWslPath $bashTmp
  $bash = "bash $bashTmpWsl"
  $bash = $bash.Trim()
  Write-Log -Lines @(
    "wsl_repo_root=" + $RepoRootWsl,
    "wsl_tmp=" + $TmpWsl,
    "wsl_script=" + $bashTmpWsl,
    "dataset_version_id=" + $DatasetArg,
    "run_seed=" + $SeedArg,
    "wsl_cmd=" + $bash
  )
  if (-not $bash) { throw "WSL command is empty." }
  try {
    if ($IsWsl) {
      $output = & bash -lc $bash 2>&1
    } else {
      $output = & wsl.exe bash -lc $bash 2>&1
    }
  } catch {
    $err = $_ | Out-String
    Write-Log -Lines @("WSL_INVOKE_ERROR", $err)
    throw
  }
  Write-Log -Lines $output
  if (!$output) { throw "WSL run produced no output." }
  $missing = $output | Where-Object { $_ -match "^WSL_PYTHON_MISSING=" } | Select-Object -First 1
  if ($missing) {
    throw "WSL Python 3.11+ is missing. Install python3.11 (or newer) in your WSL distro and rerun."
  }
  $hasRunId = $output | Where-Object { $_ -match "^RUN_ID=" } | Select-Object -First 1
  if (-not $hasRunId) {
    throw ("WSL run did not produce a RUN_ID. Output:`n" + ($output -join "`n"))
  }
  $reportLine = ($output | Where-Object { $_ -match "^REPORT=" } | Select-Object -First 1)
  if ($reportLine) {
    $reportWsl = $reportLine -replace "^REPORT=", ""
    if ($reportWsl -match "^/mnt/([a-z])/([^:]+)$") {
      $drive = $Matches[1].ToUpperInvariant()
      $rest = $Matches[2] -replace "/", "\\"
      $reportWin = "$drive`:\$rest"
    } else {
      $reportWin = $reportWsl
    }
    $output = $output | Where-Object { $_ -notmatch "^REPORT=" }
    $output += "REPORT=$reportWin"
  }
  $result = Show-Result -OutputLines $output
  Show-ReportView -ReportPath $result.ReportPath -Mode $ReportView
  return $result
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
if (-not (Test-PythonExe $VenvPython @())) {
  Write-Host "Native venv Python did not execute, falling back to WSL." -ForegroundColor Yellow
  Invoke-WSLRun
  Remove-Item $tmp -ErrorAction SilentlyContinue
  exit 0
}

& $VenvPython -m pip install -e $RepoRoot --no-build-isolation --no-deps | Out-Host

$env:DATASET_VERSION_ID = $DatasetVersionId
$env:RUN_SEED = [string]$RunSeed
$output = & $VenvPython $tmp 2>&1
Write-Log -Lines $output
if ($LASTEXITCODE -ne 0 -or -not ($output | Where-Object { $_ -match "^RUN_ID=" } | Select-Object -First 1)) {
  Write-Host "Native run failed, falling back to WSL." -ForegroundColor Yellow
  Invoke-WSLRun
  Remove-Item $tmp -ErrorAction SilentlyContinue
  exit 0
}
$result = Show-Result -OutputLines $output
Show-ReportView -ReportPath $result.ReportPath -Mode $ReportView
Remove-Item $tmp -ErrorAction SilentlyContinue
