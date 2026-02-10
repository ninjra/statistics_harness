$ErrorActionPreference = "Stop"

$winget = Get-Command winget -ErrorAction SilentlyContinue
if (-not $winget) {
  throw "winget not found. Install App Installer or run from Microsoft Store."
}

& $winget.Source install --id Microsoft.PowerShell -e --source winget
