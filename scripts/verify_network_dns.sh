#!/usr/bin/env bash
set -euo pipefail

host="${1:-pypi.org}"

echo "resolv.conf:"
cat /etc/resolv.conf || true
echo ""
echo "getent hosts ${host}:"
getent hosts "${host}" || true
echo ""
echo "curl head https://${host}/:"
curl -I "https://${host}/" | head -n 5

