#!/usr/bin/env bash
# Install and start Laika as a systemd service. Run on the Pi: sudo deploy/install-service.sh [REPO_PATH]
# Safe to run again: it refreshes the unit and keeps an existing /etc/laika/laika.env.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Run as root: sudo $0" >&2
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${1:-$HERE/..}" && pwd)"
UNIT_SRC="$HERE/laika.service"
UNIT_DST=/etc/systemd/system/laika.service
ENV_DIR=/etc/laika
ENV_FILE="$ENV_DIR/laika.env"

if [[ ! -f "$REPO/run-laika.py" ]]; then
    echo "No run-laika.py in $REPO: pass the path of the home-robot clone." >&2
    exit 1
fi

sed "s|@REPO@|$REPO|g" "$UNIT_SRC" > "$UNIT_DST"
chmod 0644 "$UNIT_DST"

install -d -m 0700 "$ENV_DIR"
if [[ ! -f "$ENV_FILE" ]]; then
    umask 077
    cat > "$ENV_FILE" <<'ENVEOF'
# Secrets for the Laika service, read by systemd. Keep this file root-only (0600).
# Uncomment and set the key, or put it in config/laika.toml instead.
#OPENAI_API_KEY=
ENVEOF
    echo "Created $ENV_FILE: add OPENAI_API_KEY there (or set [ai] api_key in config/laika.toml)."
fi
chmod 0600 "$ENV_FILE"

systemctl daemon-reload
systemctl enable --now laika.service

echo "Laika is installed and running from $REPO."
echo "Status: systemctl status laika"
echo "Logs:   journalctl -u laika -f"
