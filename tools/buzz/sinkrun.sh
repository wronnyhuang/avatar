#!/bin/bash
# sinkrun.sh LABEL APPDIR WAVS [run.cjs args...] : records Chrome's real output (null sink monitor) during the run.
#   WAVS = comma-separated reply clips for run.cjs --wavs.
# Env: BUZZ_SINK (default avatarbuzz), BUZZ_OUT capture dir (default <repo>/runs/buzz/cap)
L=${1:?label}; APP=${2:?appdir}; WAVS=${3:?wavs}; shift 3
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
SINK=${BUZZ_SINK:-avatarbuzz}
OUT=${BUZZ_OUT:-$REPO/runs/buzz/cap}; mkdir -p "$OUT"
parec -d "$SINK.monitor" --rate 48000 --channels 1 --format s16le --raw > "$OUT/$L.sink.raw" &
P=$!
cd "$HERE"
timeout 200 node run.cjs --app "$APP" --out "$OUT" --label "$L" --wavs "$WAVS" --sink "$SINK" "$@"
sleep 0.5; kill $P
ffmpeg -v error -y -f s16le -ar 48000 -ac 1 -i "$OUT/$L.sink.raw" "$OUT/$L.sink.wav" && rm "$OUT/$L.sink.raw"
