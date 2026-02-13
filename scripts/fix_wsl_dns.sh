#!/usr/bin/env bash
set -euo pipefail

# Fix WSL DNS when /etc/resolv.conf points to systemd-resolved stub (127.0.0.53)
# but systemd-resolved isn't available/allowed.
#
# This script must be run as root (via sudo) in WSL.
#
# It:
# - sets /etc/wsl.conf to disable resolv.conf auto-generation
# - replaces /etc/resolv.conf with a static nameserver list

DNS1="${DNS1:-1.1.1.1}"
DNS2="${DNS2:-8.8.8.8}"

ts="$(date -u +%Y%m%dT%H%M%SZ)"

if [[ "$(id -u)" != "0" ]]; then
  echo "ERROR: must run as root (try: sudo bash $0)" >&2
  exit 2
fi

echo "writing /etc/wsl.conf (disable auto resolv.conf generation)"
cat >/etc/wsl.conf <<'EOF'
[network]
generateResolvConf = false
EOF

if [[ -e /etc/resolv.conf ]]; then
  echo "backing up /etc/resolv.conf -> /etc/resolv.conf.bak.${ts}"
  cp -a /etc/resolv.conf "/etc/resolv.conf.bak.${ts}" || true
fi

echo "replacing /etc/resolv.conf with static DNS: ${DNS1}, ${DNS2}"
rm -f /etc/resolv.conf
{
  printf 'nameserver %s\n' "${DNS1}"
  printf 'nameserver %s\n' "${DNS2}"
} >/etc/resolv.conf
chmod 644 /etc/wsl.conf /etc/resolv.conf

echo "done. current /etc/resolv.conf:"
cat /etc/resolv.conf
echo ""
echo "NOTE: you may need to restart WSL for /etc/wsl.conf to take effect."

