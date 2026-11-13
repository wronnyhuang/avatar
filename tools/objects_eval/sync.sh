#!/bin/bash
# objects-eval: copy a candidate worktree's app/ + renderer/ (working tree, not just HEAD) into
# $OBJECTS_EVAL_DIR/candidate/<name>/source and record its commit (+dirty) in candidate/<name>/COMMIT.
#   sync.sh <worktree> [name]     (name default: the worktree's basename)
# Env: OBJECTS_EVAL_DIR  eval workspace (default <repo>/runs/objects_eval)
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
E=${OBJECTS_EVAL_DIR:-$REPO/runs/objects_eval}; W=$(realpath "${1:?worktree}"); N=${2:-$(basename "$W")}
mkdir -p "$E/candidate/$N/source"
rsync -a --delete --exclude __pycache__ --exclude '*.pyc' "$W/app" "$W/renderer" "$E/candidate/$N/source/"
C=$(git -C "$W" rev-parse --short HEAD)
[ -n "$(git -C "$W" status --porcelain -- app renderer)" ] && C="$C+dirty"
printf '%s\n' "$C" > "$E/candidate/$N/COMMIT"
printf '%s %s\n' "$W" "$(date '+%F %T')" > "$E/candidate/$N/SOURCE"
echo "synced $N = $C from $W"
