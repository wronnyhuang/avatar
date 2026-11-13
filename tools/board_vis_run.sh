#!/usr/bin/env bash
# Board visibility: the board-visibility metric over a set of lessons (sim -> real-renderer masks -> metric).
#   tools/board_vis_run.sh OUTDIR SRC_DIR "id1 id2 ..." [CFG_JSON] [RENDER_EVERY] [JOBS]
# Env: PYTHON = interpreter with the renderer deps (default python3); PNG=N also writes every Nth mask as PNG.
# SRC_DIR holds board-brain replay cases (<id>.json). CPU render (no GPU lock), nice 10, 12 threads in total.
set -u
OUT=$1; SRC=$2; IDS=$3; CFG=${4:-}; EVERY=${5:-4}; JOBS=${6:-2}
WT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python3}
TH=$((12 / JOBS))
mkdir -p "$OUT"
one() {
  id=$1; d="$OUT/$id"
  if [ -n "$CFG" ]; then node "$WT/tools/board_vis_sim.cjs" --lesson "$SRC/$id.json" --out "$d" --cfg "$CFG" > "$d.sim.log" 2>&1
  else node "$WT/tools/board_vis_sim.cjs" --lesson "$SRC/$id.json" --out "$d" > "$d.sim.log" 2>&1; fi
  (cd "$WT/renderer" && AVATAR_RT_DEVICE=cpu LP_NUM_THREADS=$TH nice -n 10 $PY tools/board_vis_render.py --rows "$d/rows.jsonl" \
     --out "$d/masks.npz" --backend cpu --threads $TH --every "$EVERY" ${PNG:+--png-dir "$d/png" --png-every $PNG} > "$d.render.log" 2>&1)
}
export -f one; export OUT SRC CFG EVERY WT PY TH PNG
mkdir -p "$OUT"; for id in $IDS; do mkdir -p "$OUT/$id"; done
printf '%s\n' $IDS | xargs -P "$JOBS" -I{} bash -c 'one {}'
$PY "$WT/tools/board_vis_metric.py" $(for id in $IDS; do echo "$OUT/$id"; done) --json "$OUT/metric.json"
