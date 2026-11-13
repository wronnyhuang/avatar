#!/usr/bin/env bash
# Re-validate the fake mic / fake camera plumbing against testpage/ (no app server, no vendor calls, ~1.5 min;
# FULL=1 adds the 157 s mic_mixed run). Serves the test page from the scratch dir on 127.0.0.1:$PORT and
# kills only that server (by pid) on exit.
# Env: EVAL_DIR (scratch dir holding mic_smoke.wav / mic_mixed.wav from make_clips.py; default <repo>/runs/voice_eval),
#      CAM (fake-camera .y4m, 320x240 expected by the checks below — required), PORT, PYTHON.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
E=${EVAL_DIR:-$REPO/runs/voice_eval}
PORT=${PORT:-18916}
CAM=${CAM:?set CAM to a 320x240 .y4m camera clip (e.g. ffmpeg -f lavfi -i testsrc=size=320x240:rate=15 -t 30 cam.y4m)}
PY=${PYTHON:-python3}
mkdir -p "$E/testpage" "$E/runs"
cp "$HERE/testpage/index.html" "$E/testpage/index.html"
( cd "$E/testpage" && exec "$PY" -m http.server "$PORT" --bind 127.0.0.1 > http.log 2>&1 ) &
SRV=$!
trap 'kill "$SRV" 2>/dev/null || true' EXIT
sleep 1
U="http://127.0.0.1:$PORT/index.html?k=test"
node "$HERE/run_eval.js" --url "$U" --mode mock --mic "$E/mic_smoke.wav" --cam "$CAM" --out "$E/runs/plumb_noloop" --seconds 40
node "$HERE/run_eval.js" --url "$U" --mode mock --mic "$E/mic_smoke.wav" --cam "$CAM" --out "$E/runs/plumb_loop" --seconds 45 --loop
RUNS=("$E/runs/plumb_noloop" "$E/runs/plumb_loop")
if [ "${FULL:-0}" = 1 ]; then
  node "$HERE/run_eval.js" --url "$U" --mode mock --mic "$E/mic_mixed.wav" --cam "$CAM" --out "$E/runs/plumb_mixed"
  RUNS+=("$E/runs/plumb_mixed")
fi
"$PY" -I "$HERE/analyze.py" "${RUNS[@]}"
"$PY" -I - "${RUNS[@]}" <<'PY'
import json, sys
ok = True
for d in sys.argv[1:]:
    r = json.load(open(d + '/results.json')); rep = json.load(open(d + '/report.json')); cam = rep['test']['cam']
    clean = [x for x in r['crosscheck'] if not x.get('flag')]
    extra = [x for x in r['crosscheck'] if (x.get('flag') or '').startswith('extra')]
    bad = [x for x in r['crosscheck'] if x.get('flag') and not x['flag'].startswith('extra')]
    lag = r['detect_lag_ms']; motion = [l[4] for l in cam.get('lumas', []) if l[4] is not None]
    want_extra = d.endswith('plumb_loop')
    good = (not bad and lag['n'] and max(abs(lag['p50']), abs(lag['max'])) < 80 and (len(extra) > 0) == want_extra
            and cam.get('w') == 320 and cam.get('h') == 240 and (rep['test'].get('cam_fps') or 0) > 12)
    ok &= bool(good)
    print(f"{'PASS' if good else 'FAIL'} {d.rsplit('/', 1)[-1]}: {len(clean)} clean turns, {len(extra)} loop-2, {len(bad)} bad; "
          f"lag p50 {lag['p50']} max {lag['max']} ms; cam {cam.get('w')}x{cam.get('h')} @ {rep['test'].get('cam_fps')} fps "
          f"'{cam.get('label', '').rsplit('/', 1)[-1]}' frame-diff mean {sum(motion) / len(motion) if motion else 0:.2f}")
sys.exit(0 if ok else 1)
PY
