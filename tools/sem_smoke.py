#!/usr/bin/env python3
"""No-key smoke test of the semantic motion API interpreter (app/semantic): every example program of the planner prompt
(app/semantic/prompt/prompt_examples.md) and the first --corpus programs of tests/fixtures/sem_programs.json are
replayed for each character, then streamed in small text deltas the way a
model writes it. Checks: poses come out, no interpreter errors, every channel is a renderer rig parameter, values are
finite, the stream equals the whole-program replay. Prints one line per character; exit 1 on a failure.

    .venv/bin/python tools/sem_smoke.py [--chars muse,fox] [--corpus 12]
"""
import argparse
import math
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "app"))
sys.path.insert(0, os.path.join(REPO, "renderer"))
import semantic as sem  # noqa: E402
from anim.schema import IDX  # noqa: E402


def examples():
    text = open(os.path.join(REPO, "app", "semantic", "prompt", "prompt_examples.md"), encoding="utf-8").read()
    return re.findall(r'Command: "([^"]+)"\s*```python\n(.*?)```', text, re.S)


def corpus(n):
    import json
    p = os.path.join(REPO, "tests", "fixtures", "sem_programs.json")
    if n <= 0 or not os.path.isfile(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [(x["id"], x["program"].rstrip("\n") + "\n") for x in json.load(f)["programs"][:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", default=",".join(sem.CHARACTERS))
    ap.add_argument("--corpus", type=int, default=12)
    a = ap.parse_args()
    progs = examples() + corpus(a.corpus)
    if not progs:
        print("SEM SMOKE FAIL: no example programs found")
        return 1
    fails = 0
    for ch in a.chars.split(","):
        targets = sem.resolve_targets(ch, None)
        n_poses, ms, bad = 0, [], []
        for cmd, prog in progs:
            t0 = time.perf_counter()
            poses, rep = sem.replay(ch, prog, targets)
            ms.append((time.perf_counter() - t0) * 1000)
            n_poses += len(poses)
            if not poses or rep.get("errors") or not rep.get("n_statements"):
                bad.append(f"{cmd!r}: {len(poses)} poses, errors {rep.get('errors')}")
                continue
            keys = {k for p in poses for k in p if k != "t"}
            unknown = sorted(k for k in keys if k not in IDX)
            if unknown:
                bad.append(f"{cmd!r}: channels the renderer does not know: {unknown}")
            if any(not math.isfinite(v) for p in poses for v in p.values()):
                bad.append(f"{cmd!r}: non-finite values")
            s = sem.Stream(ch, targets=targets)
            text = "```python\n" + prog + "```\n"
            got = []
            for i in range(0, len(text), 7):
                got += [q for b in s.feed(text[i:i + 7]) for q in b["poses"]]
            got += [q for b in s.finish() for q in b["poses"]]
            if len(got) != len(poses) or any(abs(got[i].get(k, 0) - v) > 1e-6 for i, p in enumerate(poses)
                                             for k, v in p.items()):
                bad.append(f"{cmd!r}: streamed poses differ from the replay")
        fails += len(bad)
        ms.sort()
        print(f"{'ok  ' if not bad else 'FAIL'} semantic {ch}: {len(progs)} programs, {n_poses} poses, "
              f"replay median {ms[len(ms) // 2]:.0f} ms" + ("".join("\n      " + b for b in bad)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
