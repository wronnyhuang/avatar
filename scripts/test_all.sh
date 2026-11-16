#!/usr/bin/env bash
# Offline unit suites: Python (tests/test_*.py) + Node (tests/*_test.cjs) + the render server's rig-debug suites
# (renderer/avatar/test_rigdbg.py, CPU) + the semantic interpreter smoke (tools/sem_smoke.py). No keys, no GPU.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); PY=${PY:-$REPO/.venv/bin/python}
pass=0; fail=0; failed=()
run() {   # run <name> <dir> <cmd...>
  local name=$1 dir=$2; shift 2
  if (cd "$dir" && timeout 600 "$@" >/tmp/avatar_test.log 2>&1); then pass=$((pass+1)); else fail=$((fail+1)); failed+=("$name"); fi
}
for f in "$REPO"/tests/test_*.py; do run "$(basename "$f")" "$REPO/tests" "$PY" "$f"; done
if command -v node >/dev/null; then
  for f in "$REPO"/tests/*_test.cjs; do run "$(basename "$f")" "$REPO/tests" node "$f"; done
fi
run "test_rigdbg rig" "$REPO/renderer" "$PY" -m avatar.test_rigdbg rig
run "test_rigdbg session stub" "$REPO/renderer" "$PY" -m avatar.test_rigdbg session --target stub
run "sem_smoke" "$REPO" "$PY" tools/sem_smoke.py
echo "suites passed: $pass  failed: $fail ${failed[*]:-}"
[ $fail = 0 ]
