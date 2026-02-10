$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BashScript = "/mnt/d/projects/statistics_harness/statistics_harness/scripts/watch_gauntlet.sh"

& wsl.exe -- bash -lc $BashScript
