#!/usr/bin/env bash
# Offline unit suites: Python (tests/test_*.py) + Node (tests/*_test.cjs). No keys, no GPU.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); PY=${PY:-$REPO/.venv/bin/python}
pass=0; fail=0; failed=()
for f in "$REPO"/tests/test_*.py; do
  if (cd "$REPO/tests" && timeout 300 "$PY" "$f" >/tmp/avatar_test.log 2>&1); then pass=$((pass+1)); else fail=$((fail+1)); failed+=("$(basename "$f")"); fi
done
if command -v node >/dev/null; then
  for f in "$REPO"/tests/*_test.cjs; do
    if (cd "$REPO/tests" && timeout 300 node "$f" >/tmp/avatar_test.log 2>&1); then pass=$((pass+1)); else fail=$((fail+1)); failed+=("$(basename "$f")"); fi
  done
fi
echo "suites passed: $pass  failed: $fail ${failed[*]:-}"
[ $fail = 0 ]
