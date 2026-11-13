#!/bin/bash
# fps table on the K620: one flock acquisition per run (each < ~2 min), JSON lines appended to $OUT.
#   tools/rig_fps_table.sh OUT.jsonl "fox bear" "480x270 640x360" "rest move" [--root TREE] [--preset P]
OUT=$1; CHARS=$2; SIZES=$3; MOTIONS=$4; shift 4
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
LOCK=${AVATAR_GPU_LOCK:-${TMPDIR:-/tmp}/avatar-gpu.lock}
for c in $CHARS; do for s in $SIZES; do for m in $MOTIONS; do
  w=${s%x*}; h=${s#*x}
  flock -w 3000 $LOCK timeout 400 nice -n 5 $PY "$HERE/rig_bench.py" --animal $c --w $w --h $h --frames 600 --encode --motion $m "$@" 2>/dev/null | grep '^{' >> "$OUT"
done; done; done
