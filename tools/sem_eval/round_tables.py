#!/usr/bin/env python3
"""round tables (markdown) from runs + judge: did-it / naturalness / pairwise per mode x character, for ALL commands
and for PLANNED ones only (library hits are the same engine move in both modes and dilute the comparison), the judge's
noise floor (library pairs: same move judged twice), latency per mode x character (fresh, cached), error counts.
  python3 tools/sem_eval/round_tables.py --runs EVAL/runs/<tag> --judge EVAL/judge/<tag> [--load EVAL/<tag>_load.txt]
"""
import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo))


def fmt(v):
    return "–" if v is None else (f"{v / 1000:.2f} s" if v >= 1000 else f"{v} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--load", default="")
    a = ap.parse_args()
    runs = {}
    for p in glob.glob(os.path.join(a.runs, "*", "*", "*", "*.json")):
        if p.endswith("run.json"):
            continue
        R = json.load(open(p))
        if "timings" in R:
            runs[(R["char"], R["mode"], R["pass"], R["id"])] = R
    J = {(r["char"], r["mode"], r["id"]): r for r in map(json.loads, open(os.path.join(a.judge, "action.jsonl"))) if "verdict" in r}
    PW = [r for r in map(json.loads, open(os.path.join(a.judge, "pairwise.jsonl"))) if "better" in r] \
        if os.path.exists(os.path.join(a.judge, "pairwise.jsonl")) else []
    chars = ["muse", "fox", "juno"]
    modes = ["current", "semantic"]
    src = lambda ch, m, cid, ps="fresh": ((runs.get((ch, m, ps, cid)) or {}).get("timings") or {}).get("source")
    planned = lambda ch, cid: src(ch, "current", cid) != "library" and src(ch, "semantic", cid) != "library"
    out = []
    w = out.append

    def jrow(keys):
        v = [J[k] for k in keys if k in J]
        if not v:
            return None
        n = len(v)
        c = defaultdict(int)
        for x in v:
            c[x["verdict"]] += 1
        return n, c["yes"] / n, c["partly"] / n, c["no"] / n, (c["yes"] + .5 * c["partly"]) / n, st.mean(x.get("natural") or 0 for x in v)

    for title, filt in (("all 41 commands (library hits included)", lambda ch, cid: True), ("PLANNED commands only (neither mode a library hit)", planned)):
        w(f"\n### Did it (blind judge, fresh) — {title}\n")
        w("| planner | character | n | yes | partly | no | score | natural 1-5 |")
        w("|---|---|---|---|---|---|---|---|")
        for m in modes:
            for ch in chars + ["all"]:
                ks = [(c, m, cid) for (c, mm, cid) in J if mm == m and (ch == "all" or c == ch) and filt(c, cid)]
                r = jrow(ks)
                if r:
                    w(f"| {m} | {ch} | {r[0]} | {r[1]:.0%} | {r[2]:.0%} | {r[3]:.0%} | {r[4]:.0%} | {r[5]:.2f} |")
        if PW:
            w(f"\nBlind pairwise ({title}; ties = ½): share won by each planner")
            w("\n| character | n | better: current | better: semantic | natural: current | natural: semantic |")
            w("|---|---|---|---|---|---|")
            for ch in chars + ["all"]:
                sub = [p for p in PW if (ch == "all" or p["char"] == ch) and filt(p["char"], p["id"])]
                if not sub:
                    continue
                res = []
                for f in ("better", "more_natural"):
                    s = defaultdict(float)
                    for p in sub:
                        if p.get(f + "_mode"):
                            s[p[f + "_mode"]] += 1
                        else:
                            s["current"] += .5
                            s["semantic"] += .5
                    res += [s["current"] / len(sub), s["semantic"] / len(sub)]
                w(f"| {ch} | {len(sub)} | " + " | ".join(f"{x:.0%}" for x in res) + " |")
    # judge noise: library hits judged in both modes (same engine move)
    lib = [(ch, cid) for (ch, m, cid) in J if m == "current" and src(ch, "current", cid) == "library" and src(ch, "semantic", cid) == "library"
           and (ch, "semantic", cid) in J]
    if lib:
        agree = sum(J[(ch, "current", cid)]["verdict"] == J[(ch, "semantic", cid)]["verdict"] for ch, cid in lib)
        dn = [abs((J[(ch, "current", cid)].get("natural") or 0) - (J[(ch, "semantic", cid)].get("natural") or 0)) for ch, cid in lib]
        pl = [p for p in PW if (p["char"], p["id"]) in set(lib)]
        ties = sum(1 for p in pl if p.get("better") == "tie")
        w(f"\nJudge noise floor — library hits (the SAME engine move in both modes, n {len(lib)}): verdicts agree {agree}/{len(lib)}, "
          f"naturalness differs by {st.mean(dn):.2f} on average; pairwise calls them a tie {ties}/{len(pl)}.")
    # latency
    w("\n### Latency (ms after the command reaches the page; median / p90)\n")
    w("| planner | character | pass | n | planner calls (paid fallback) | hub first pose | move starts on avatar | move ends |")
    w("|---|---|---|---|---|---|---|---|")
    for m in modes:
        for ps in ("fresh", "cached"):
            for ch in chars + ["all"]:
                rs = [R for (c, mm, p, cid), R in runs.items() if mm == m and p == ps and (ch == "all" or c == ch) and R["timings"].get("source") != "library"]
                if not rs:
                    continue
                fb = sum(1 for R in rs if any(k.get("done") and float(k.get("usd") or 0) > 0 for k in R.get("kf") or []))
                T = lambda k: [R["timings"].get(k) for R in rs]
                w(f"| {m} | {ch} | {ps} | {len(rs)} | {fb} | {fmt(pct(T('hub_first_pose_ms'), .5))} / {fmt(pct(T('hub_first_pose_ms'), .9))} | "
                  f"{fmt(pct(T('first_motion_ms'), .5))} / {fmt(pct(T('first_motion_ms'), .9))} | {fmt(pct(T('end_ms'), .5))} / {fmt(pct(T('end_ms'), .9))} |")
    lib_ms = [R["timings"].get("first_motion_ms") for R in runs.values() if R["timings"].get("source") == "library"]
    w(f"\nLibrary hits (both modes, n {len(lib_ms)}): move starts {fmt(pct(lib_ms, .5))} / {fmt(pct(lib_ms, .9))}.")
    errs = defaultdict(int)
    for (c, m, p, cid), R in runs.items():
        for e in R.get("errors") or []:
            errs[(m, p, e[:70])] += 1
    w("\nErrors: " + ("; ".join(f"{m}/{p}: {e} x{n}" for (m, p, e), n in sorted(errs.items())) or "none"))
    if a.load and os.path.exists(a.load):
        L = sorted(float(l.split()[1]) for l in open(a.load) if len(l.split()) > 1)
        w(f"\nMachine load during the round (1-min load average, sampled every minute on a 70-thread box): min {L[0]:.0f}, "
          f"median {L[len(L) // 2]:.0f}, max {L[-1]:.0f}. Both planners ran interleaved per command on the same page and hub, "
          "so the comparison is fair; absolute numbers are inflated.")
    print("\n".join(out))


if __name__ == "__main__":
    main()
