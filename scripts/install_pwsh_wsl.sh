#!/usr/bin/env bash
set -euo pipefail

if command -v pwsh >/dev/null 2>&1; then
  echo "pwsh already installed"
  exit 0
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get not found; install PowerShell manually for this distro."
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y curl
fi

. /etc/os-release
ver="${VERSION_ID:-22.04}"
tmp="/tmp/pwsh-msft.deb"

curl -fsSL "https://packages.microsoft.com/config/ubuntu/${ver}/packages-microsoft-prod.deb" -o "$tmp"
sudo dpkg -i "$tmp"
sudo apt-get update
sudo apt-get install -y powershell
