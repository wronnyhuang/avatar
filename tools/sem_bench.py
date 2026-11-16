#!/usr/bin/env python3
"""Bench + regression corpus: every program of tests/fixtures/sem_programs.json (or --corpus: a JSON file of the same
shape, or a glob of *.json files with a "script" or "program" field)
streamed through app/semantic in token-sized deltas, per character.

  python tools/sem_bench.py [--chars muse,fox] [--corpus FILE|GLOB] [--chunk 6] [--limit N] [--json OUT]

Reports: feed() time per statement (median / p95 / max, only the calls that completed a statement), finish() time,
whole-program replay time (median / p95; and for programs of 5-7 s), exceptions (must be 0), stream-vs-replay max
difference (must be 0), statements skipped with errors.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "app"))
import semantic as sem  # noqa: E402

CORPUS = os.path.join(os.path.dirname(HERE), "tests", "fixtures", "sem_programs.json")


def pct(a, q):
    return round(float(np.percentile(a, q)), 2) if len(a) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", default="muse,fox")
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--chunk", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json")
    a = ap.parse_args()
    progs = []
    if os.path.isfile(a.corpus):
        with open(a.corpus, encoding="utf-8") as f:
            progs = [(p["id"], p["program"]) for p in json.load(f)["programs"]]
    else:
        for f in sorted(glob.glob(a.corpus)):
            try:
                d = json.load(open(f))
            except ValueError:
                continue
            if d.get("script") or d.get("program"):
                progs.append((os.path.basename(f)[:-5], d.get("script") or d.get("program")))
    if a.limit:
        progs = progs[:a.limit]
    out = {}
    for ch in a.chars.split(","):
        targets = sem.resolve_targets(ch, None)
        feed_ms, fin_ms, rep_ms, rep6_ms, exc, mism, errs, durs = [], [], [], [], [], [], 0, []
        for name, src in progs:
            text = "```python\n" + src.strip("\n") + "\n```\n"
            try:
                s = sem.Stream(ch, targets=targets)
                got = []
                for i in range(0, len(text), a.chunk):
                    t0 = time.perf_counter()
                    b = s.feed(text[i:i + a.chunk])
                    dt = (time.perf_counter() - t0) * 1000
                    if b:
                        feed_ms.append(dt / max(1, len(b)))
                    got += [p for x in b for p in x["poses"]]
                t0 = time.perf_counter()
                b = s.finish()
                fin_ms.append((time.perf_counter() - t0) * 1000)
                got += [p for x in b for p in x["poses"]]
                r = s.report()
                errs += len(r["errors"])
                t0 = time.perf_counter()
                poses, rep = sem.replay(ch, src, targets=targets)
                ms = (time.perf_counter() - t0) * 1000
                rep_ms.append(ms)
                durs.append(rep["duration_s"])
                if 5.0 <= rep["duration_s"] <= 7.0:
                    rep6_ms.append(ms)
                d = 0.0
                if len(poses) != len(got):
                    d = float("inf")
                else:
                    for p, q in zip(got, poses):
                        for k in set(p) | set(q):
                            d = max(d, abs(p.get(k, 0.0) - q.get(k, 0.0)) if k in p and k in q else
                                    (0.0 if k == "t" else float("inf")))
                if d > 1e-6:
                    mism.append((name, d))
            except Exception as e:  # noqa: BLE001
                exc.append((name, f"{type(e).__name__}: {e}"))
        out[ch] = {"programs": len(progs), "exceptions": len(exc), "exception_list": exc[:5],
                   "stream_vs_replay_mismatch": len(mism), "mismatch_list": mism[:5], "statement_errors": errs,
                   "feed_ms_per_stmt": {"median": pct(feed_ms, 50), "p95": pct(feed_ms, 95), "max": pct(feed_ms, 100),
                                        "n": len(feed_ms)},
                   "finish_ms": {"median": pct(fin_ms, 50), "p95": pct(fin_ms, 95)},
                   "replay_ms": {"median": pct(rep_ms, 50), "p95": pct(rep_ms, 95), "max": pct(rep_ms, 100)},
                   "replay_ms_5to7s": {"median": pct(rep6_ms, 50), "p95": pct(rep6_ms, 95), "n": len(rep6_ms)},
                   "duration_s": {"median": pct(durs, 50), "max": pct(durs, 100)}}
        print(ch, json.dumps(out[ch]), flush=True)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
