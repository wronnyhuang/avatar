#!/usr/bin/env bash
# devinst.sh: an isolated avatar app server + local renderer from a worktree, for evals and tests.
#   tools/devinst.sh <instname> <port> [worktree-root (default: this repo)] [backend=cpu|gpu|auto] [react-json]
# Env:
#   AVATAR_DEV_ROOT     instances root (default <repo>/runs/devinst); instance = $AVATAR_DEV_ROOT/inst/<name>,
#                       render cache = $AVATAR_DEV_ROOT/cache
#   AVATAR_BASE_CONFIG  base state/config.json to copy (default <repo>/.avatar-home/state/config.json from
#                       scripts/setup.sh, else <repo>/deploy/config.local.example.json)
#   PYTHON              interpreter for app/app.py and the render child (default <repo>/.venv/bin/python, else python3)
#   INTENT=1            keeps the server's fast intent matcher ON (default); INTENT=0 turns it off
#   DEVINST_CAP_USD     the instance's daily voice budget (default 1.0)
#   LP_NUM_THREADS      CPU render threads for backend=cpu (default 8)
# The instance key is "dev-<instname>-0123456789abcdef"; the script prints the local URL.
# Stop: kill -TERM -- -$(cat "$AVATAR_DEV_ROOT/inst/<name>/server.pid")   (only YOUR instance)
set -euo pipefail
NAME=${1:?instname}; PORT=${2:?port}; WT=${3:-}; BACKEND=${4:-cpu}; REACT=${5:-'{}'}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WT="$(cd "${WT:-$REPO}" && pwd)"
T=${AVATAR_DEV_ROOT:-$REPO/runs/devinst}
H=$T/inst/$NAME
if [ -z "${AVATAR_BASE_CONFIG:-}" ]; then
  if [ -f "$REPO/.avatar-home/state/config.json" ]; then AVATAR_BASE_CONFIG=$REPO/.avatar-home/state/config.json
  else AVATAR_BASE_CONFIG=$REPO/deploy/config.local.example.json; fi
fi
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$REPO/.venv/bin/python" ]; then PYTHON=$REPO/.venv/bin/python; else PYTHON=$(command -v python3); fi
fi
mkdir -p "$H/state" "$H/data" "$T/cache"
SECRET="dev-${NAME}-0123456789abcdef"
printf '%s\n' "$SECRET" > "$H/state/secret"; chmod 600 "$H/state/secret"
python3 - "$H" "$WT" "$BACKEND" "$REACT" "${INTENT:-1}" "$AVATAR_BASE_CONFIG" "$PYTHON" "$T/cache" <<'PY'
import json, sys, os
h, wt, backend, react, intent, base_cfg, python, cache = sys.argv[1:9]
base = json.load(open(base_cfg))
base['backend'] = backend
base['daily_cap_usd'] = float(os.environ.get('DEVINST_CAP_USD', '1.0'))
base.setdefault('local', {})
base['local']['render_root'] = os.path.join(wt, 'renderer')
base['local']['python'] = python
env = base['local'].setdefault('env', {})
env['TA_OUT'] = cache
env['AVATAR_NVENC'] = '0'
if backend == 'cpu':
    base['backend'] = 'rt'; env['AVATAR_RT_DEVICE'] = 'cpu'; env['LP_NUM_THREADS'] = os.environ.get('LP_NUM_THREADS', '8')
if intent == '0':
    base['intent'] = dict(base.get('intent') or {}, enabled=False)
r = json.loads(react)
if r: base['react'] = dict(base.get('react') or {}, **r)
json.dump(base, open(os.path.join(h, 'state', 'config.json'), 'w'), indent=1)
PY
if [ -f "$H/server.pid" ] && kill -0 "$(cat "$H/server.pid")" 2>/dev/null; then
  echo "instance $NAME already running pid $(cat "$H/server.pid")"; else
  ( cd "$WT/app" && AVATAR_HOME="$H" PORT="$PORT" OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
      setsid -f nice -n 10 bash -c 'echo $$ > "$0/server.pid"; exec "$2" "$1/app/app.py"' "$H" "$WT" "$PYTHON" > "$H/server.log" 2>&1 < /dev/null )
  sleep 1.5
fi
echo "pid $(cat "$H/server.pid")  url http://127.0.0.1:$PORT/?k=$SECRET"
