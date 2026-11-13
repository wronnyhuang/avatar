#!/usr/bin/env bash
# One eval run on a small local GPU (developed on a 2 GB Quadro K620): (re)start a GPU dev instance, run, stop the
# instance — all inside one GPU-lock hold so concurrent runs never share the card.
#   gpu_run.sh <outname> <mode> [vad] [extra run_eval args...]
# Env: WT (source tree to run the instance from; default this repo), AVATAR_DEV_ROOT (devinst instances; default
#      <repo>/runs/devinst), VOICE_EVAL_DIR (runs go to $VOICE_EVAL_DIR/runs/<outname>; default <repo>/runs/voice_eval),
#      AVATAR_GPU_LOCK (lock file; default <repo>/runs/gpu.lock), INST (instance name, default voicegpu), PORT (18911).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
WT=${WT:-$REPO}
DEV_ROOT=${AVATAR_DEV_ROOT:-$REPO/runs/devinst}
E=${VOICE_EVAL_DIR:-$REPO/runs/voice_eval}
LOCK=${AVATAR_GPU_LOCK:-$REPO/runs/gpu.lock}
INST=${INST:-voicegpu}; PORT=${PORT:-18911}
OUT=${1:?outname}; MODE=${2:?mode}; VAD=${3:-server}; shift 3 || shift $#
mkdir -p "$(dirname "$LOCK")" "$E/runs"
exec 9>"$LOCK"
flock -w 900 9 || { echo "no gpu lock"; exit 1; }
echo "gpu lock held $(date +%T)"
AVATAR_DEV_ROOT="$DEV_ROOT" bash "$WT/tools/devinst.sh" "$INST" "$PORT" "$WT" auto > /dev/null 2>&1 < /dev/null
D=$DEV_ROOT/inst/$INST
PID=$(cat "$D/server.pid")
KEY=$(cat "$D/state/secret" 2>/dev/null || echo "dev-$INST-0123456789abcdef")
URL="http://127.0.0.1:$PORT/?k=$KEY"
# prewarm the renderer before the clock starts (first GPU init compiles shaders)
curl -s -X POST -H 'x-requested-with: avatar-live' -H 'content-type: application/json' \
  -d '{"character":"fox","quality":"auto","codec":"h264"}' "http://127.0.0.1:$PORT/api/warm?k=$KEY" > /dev/null
for i in $(seq 1 60); do grep -q "prewarmed fox" "$D/server.log" && break; sleep 1; done
grep "prewarmed\|render child" "$D/server.log" | tail -2
cd "$HERE" && AVATAR_DEV_ROOT="$DEV_ROOT" timeout 400 node run_eval.js --url "$URL" --mode "$MODE" --vad "$VAD" \
  --out "$E/runs/$OUT" --inst "$INST" "$@" 2>&1 | grep -v "^t+" | tail -6
sleep 2
kill -TERM -- -"$PID" 2>/dev/null; sleep 3; kill -0 "$PID" 2>/dev/null && kill -KILL -- -"$PID"
tail -3 "$D/server.log"
echo "gpu lock released $(date +%T)"
