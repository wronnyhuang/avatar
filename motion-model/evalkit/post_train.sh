#!/bin/bash
# phase 2 post-training hook (tools/train/p2_collect.py calls it after the bundle is exported):
#   evalkit/post_train.sh <bundle_dir> <tag> [extra scorecard.py args, e.g. --no-judge | --kw '{"guidance":1.5}']
# -> $SC/<tag>/scorecard.{json,md,log} and the table appended to $SC/INDEX.md
#    ($SC = $SCORECARD_ROOT, default $MM_WORK_DIR/p2/scorecards).
# Runtime code for the bundle: $SCORECARD_CODE, else $EVAL_MODEL_CODE, else this checkout (snapshotted once per tag).
# Interpreter: $PYTHON (default python3).
# Exit code = scorecard.py's (0 = scorecard written, whatever the verdict); the verdict line is printed last.
set -u
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MM="$(cd "$KIT/.." && pwd)"
PY="${PYTHON:-python3}"
WORK="${MM_WORK_DIR:-$MM/work}"
SC="${SCORECARD_ROOT:-$WORK/p2/scorecards}"
[ $# -ge 2 ] || { echo "usage: $0 <bundle_dir> <tag> [scorecard args]" >&2; exit 2; }
B="$1"; TAG="$2"; shift 2
CODE="${SCORECARD_CODE:-${EVAL_MODEL_CODE:-$MM}}"
mkdir -p "$SC/$TAG"
cd "$MM" || exit 2
PYTHONDONTWRITEBYTECODE=1 "$PY" -I "$KIT/scorecard.py" --bundle "$B" --tag "$TAG" --code "$CODE" --index "$@" \
  > "$SC/$TAG/post_train.out" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  grep -m1 "beats rules" "$SC/$TAG/scorecard.md"
else
  echo "scorecard $TAG FAILED (rc=$rc): see $SC/$TAG/post_train.out" >&2
  tail -n 15 "$SC/$TAG/post_train.out" >&2
fi
exit $rc
