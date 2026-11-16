#!/usr/bin/env bash
# One command: render the fixed shot list for a build, then judge it.
#   run_round.sh NAME WORKTREE ASSET_DIR [CHAR=juno] [MESH_CFG_JSON]
#   e.g. run_round.sh v7 $T/wt-integ $T/asset/v7/junor.asset juno $T/scratch/judge/v6_meshcfg.json
# ASSET_DIR = the junor.asset directory (regroom it first with renderer/humans/junor/groom/regroom.py if the build has no hair groom).
# MESH_CFG_JSON = a json dict merged over the character's mesh config (JUNO_MESH_CFG), e.g. junor.json "mesh" with the SHADE look;
#   needed while `junor` is not wired into engine/render.py (use --char juno + this file). Wall time: render 10-15 min (PBR look, 4 shots at a time on CPU),
#   judge 4-6 min. Output: $T/judge/ours/NAME/{closeup,bust}/*.png and $T/judge/rounds/NAME/report.md
set -euo pipefail
T=${JT:-$(cd "$(dirname "$0")" && pwd)/work}; PY=${PY:-python3}
NAME=${1:?name}; WT=${2:?worktree}; ASSET=${3:?asset dir}; CHAR=${4:-juno}; CFG=${5:-}
[ -n "$CFG" ] && export JUNO_MESH_CFG="$(cat "$CFG")"
$PY "$(dirname "$0")/render_shots.py" --worktree "$WT" --char "$CHAR" --asset "$ASSET" --out "$T/judge/ours/$NAME" --jobs 4
$PY -I "$(dirname "$0")/aaa_judge.py" round --ours "$T/judge/ours/$NAME" --name "$NAME" --n 20 --sets closeup,bust
echo "report: $T/judge/rounds/$NAME/report.md"
