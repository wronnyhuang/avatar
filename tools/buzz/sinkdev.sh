#!/bin/bash
# sinkdev.sh LABEL [devcheck args] : the real page on a running server instance, speaker output recorded from the
# PulseAudio null sink (create it first: pactl load-module module-null-sink sink_name=$BUZZ_SINK).
# Env: AVATAR_URL / AVATAR_KEY  the running server and its key (required)
#      BUZZ_SINK                null sink name (default avatarbuzz)
#      BUZZ_OUT                 capture dir (default <repo>/runs/buzz/cap)
L=${1:?label}; shift
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
SINK=${BUZZ_SINK:-avatarbuzz}
OUT=${BUZZ_OUT:-$REPO/runs/buzz/cap}; mkdir -p "$OUT"
U=${AVATAR_URL:?set AVATAR_URL}; K=${AVATAR_KEY:?set AVATAR_KEY}
if curl -s -m 10 "${U%/}/health" | grep -q '"session": true'; then echo "SKIP: the instance has a live session"; exit 3; fi
parec -d "$SINK.monitor" --rate 48000 --channels 1 --format s16le --raw > "$OUT/$L.sink.raw" &
PR=$!
cd "$HERE"
timeout 600 node devcheck.cjs --url "$U" --key "$K" --out "$OUT" --label "$L" --sink "$SINK" "$@"
rc=$?; sleep 0.5; kill $PR
ffmpeg -v error -y -f s16le -ar 48000 -ac 1 -i "$OUT/$L.sink.raw" "$OUT/$L.sink.wav" && rm "$OUT/$L.sink.raw"
exit $rc
