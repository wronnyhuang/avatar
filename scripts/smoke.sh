#!/usr/bin/env bash
# Smoke test: start the server on a scratch port with a scratch home, check /health, the page, /api/state and a
# renderer warm-up (real SDF renderer, CPU OK), then stop. No API keys needed. Exit 0 = pass.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PORT=${SMOKE_PORT:-18765}
H=$(mktemp -d); trap 'kill -TERM -- -$PID 2>/dev/null; sleep 1; rm -rf "$H"' EXIT
AVATAR_HOME=$H bash "$REPO/scripts/setup.sh" >/dev/null
K=$(cat "$H/state/secret")
( cd "$REPO/app" && AVATAR_HOME=$H PORT=$PORT AVATAR_REC_ROOT=$H/rec exec setsid "$REPO/.venv/bin/python" app.py > "$H/server.log" 2>&1 ) &
PID=$!
fail() { echo "SMOKE FAIL: $*"; tail -20 "$H/server.log"; exit 1; }
for i in $(seq 1 30); do curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q '"ok": true' && break; sleep 1; done
curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q '"ok": true' || fail "/health"
echo "ok  /health: $(curl -s -m 2 "http://127.0.0.1:$PORT/health")"
C=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://127.0.0.1:$PORT/?k=$K"); [ "$C" = 200 ] || fail "page $C"; echo "ok  page 200"
C=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://127.0.0.1:$PORT/?k=wrong"); [ "$C" != 200 ] || fail "bad key accepted"; echo "ok  wrong key refused ($C)"
API=(-H 'x-requested-with: avatar-live' -H 'content-type: application/json')
curl -s -m 5 "${API[@]}" "http://127.0.0.1:$PORT/api/state?k=$K" | grep -q '"characters"' || fail "/api/state"; echo "ok  /api/state"
curl -s -m 10 "${API[@]}" -X POST -d '{"character":"muse","quality":"270p","codec":"h264"}' "http://127.0.0.1:$PORT/api/warm?k=$K" >/dev/null
for i in $(seq 1 ${SMOKE_WARM_S:-180}); do grep -qE "prewarmed muse" "$H/server.log" && break; sleep 1; done
grep -qE "prewarmed muse" "$H/server.log" && echo "ok  renderer warm: $(grep -E "prewarmed|render child" "$H/server.log" | tail -1 | cut -c1-140)" || fail "renderer did not warm"
grep -q "stub" "$H/server.log" && fail "stub renderer was used, not the real one"
echo "SMOKE PASS"
