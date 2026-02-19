param(
  [ValidateRange(25,95)][int]$MemoryPercent = 75,
  [ValidatePattern('^\d+GB$')][string]$Swap = '8GB'
)

$os = Get-CimInstance Win32_OperatingSystem
$totalGB = [math]::Floor($os.TotalVisibleMemorySize / 1MB)
$capGB = [math]::Max(4, [math]::Floor($totalGB * ($MemoryPercent / 100.0)))

$cfg = Join-Path $env:USERPROFILE '.wslconfig'
$bak = "$cfg.bak.$(Get-Date -Format yyyyMMdd_HHmmss)"
if (Test-Path $cfg) { Copy-Item $cfg $bak -Force }

@"
[wsl2]
memory=${capGB}GB
swap=$Swap
pageReporting=true
localhostForwarding=true

[experimental]
autoMemoryReclaim=gradual
sparseVhd=true
"@ | Set-Content -Path $cfg -Encoding ASCII

Write-Host "Wrote $cfg (memory=${capGB}GB, swap=$Swap)."
if (Test-Path $bak) { Write-Host "Backup: $bak" }
Write-Host "No shutdown performed. Settings apply on next WSL restart."
