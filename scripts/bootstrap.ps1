$python = Get-Command python3 -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command python -ErrorAction SilentlyContinue
}
if (-not $python) {
    throw "Python not found. Install Python 3.11+."
}
& $python.Path -m venv .venv
.\.venv\Scripts\Activate.ps1
.\scripts\install_dev.ps1
