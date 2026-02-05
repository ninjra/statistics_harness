$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Gauntlet = Join-Path $ScriptRoot "run_gauntlet.ps1"

& $Gauntlet
