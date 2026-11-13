#!/usr/bin/env bash
# Run the avatar server in the foreground on 127.0.0.1:$PORT (default 10250) with the local renderer.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
# .env fills in variables that are not already set in your shell (empty values in .env are ignored)
load_env() {
  local k v
  while IFS='=' read -r k v || [ -n "$k" ]; do
    k=$(printf '%s' "$k" | tr -d '[:space:]')
    [[ $k =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    v=$(printf '%s' "$v" | sed -e 's/[[:space:]]#.*$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/")
    if [ -n "$v" ] && [ -z "${!k:-}" ]; then export "$k=$v"; fi
  done < "$1"
}
if [ -f "$REPO/.env" ]; then load_env "$REPO/.env"; fi
export AVATAR_HOME=${AVATAR_HOME:-$REPO/.avatar-home}
export PORT=${PORT:-10250}
[ -f "$AVATAR_HOME/state/config.json" ] || { echo "run scripts/setup.sh first" >&2; exit 1; }
echo "open: http://127.0.0.1:$PORT/?k=$(cat "$AVATAR_HOME/state/secret")   (Chrome; allow camera + microphone)"
cd "$REPO/app" && exec "$REPO/.venv/bin/python" app.py
