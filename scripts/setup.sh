#!/usr/bin/env bash
# One-time setup: Python venv + deps, a local AVATAR_HOME with config and a random access secret.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
HOME_DIR=${AVATAR_HOME:-$REPO/.avatar-home}
PY=${PYTHON:-python3}
[ -d "$REPO/.venv" ] || "$PY" -m venv "$REPO/.venv"
"$REPO/.venv/bin/pip" install -q --upgrade pip
"$REPO/.venv/bin/pip" install -q -r "$REPO/requirements.txt"
mkdir -p "$HOME_DIR/state" "$HOME_DIR/data" "$HOME_DIR/cache"
TEMPLATE=${1:-$REPO/deploy/config.local.example.json}
if [ ! -f "$HOME_DIR/state/config.json" ]; then
  sed -e "s#__REPO__#$REPO#g" -e "s#__HOME__#$HOME_DIR#g" "$TEMPLATE" > "$HOME_DIR/state/config.json"
  echo "wrote $HOME_DIR/state/config.json from $(basename "$TEMPLATE")"
fi
if [ ! -s "$HOME_DIR/state/secret" ]; then
  "$REPO/.venv/bin/python" -c 'import secrets; print(secrets.token_urlsafe(24))' > "$HOME_DIR/state/secret"
  chmod 600 "$HOME_DIR/state/secret"
fi
[ -f "$REPO/.env" ] || cp "$REPO/.env.example" "$REPO/.env"
echo "setup done. Fill in .env, then: scripts/run_local.sh"
