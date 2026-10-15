#!/usr/bin/env bash
# Stop Laika and remove its systemd unit. Run on the Pi: sudo deploy/uninstall-service.sh
# Keeps /etc/laika (secrets) and /var/lib/laika (data); delete them by hand if you want them gone.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Run as root: sudo $0" >&2
    exit 1
fi

UNIT=/etc/systemd/system/laika.service

# Stopping runs ExecStopPost, which leaves the robot stopped.
systemctl disable --now laika.service 2>/dev/null || true
rm -f "$UNIT"
systemctl daemon-reload
systemctl reset-failed laika.service 2>/dev/null || true

echo "Laika service removed. Kept /etc/laika and /var/lib/laika."
