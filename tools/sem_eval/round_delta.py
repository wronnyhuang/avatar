#!/usr/bin/env python3
"""round A -> round B delta per character x mode (planned commands only = neither mode a library hit in that round):
did-it score (yes + ½ partly), 'no' share, naturalness, pairwise share won by semantic, first-motion median / p90 (fresh).
  python3 tools/sem_eval/round_delta.py --a EVAL/runs/r1,EVAL/judge/r1 --b EVAL/runs/r2,EVAL/judge/r2 [--json out.json]
Only the (char, id) cells present in BOTH rounds' judge files are compared (juno: the commands rendered in both).
"""
import argparse
import glob
import json
import os
import statistics as st


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo))


def load(spec):
    rd, jd = spec.split(",")
    runs = {}
    for p in glob.glob(os.path.join(rd, "*", "*", "fresh", "*.json")):
        if p.endswith("run.json"):
            continue
        R = json.load(open(p))
        if "timings" in R:
            runs[(R["char"], R["mode"], R["id"])] = R
    J = {(r["char"], r["mode"], r["id"]): r for r in map(json.loads, open(os.path.join(jd, "action.jsonl"))) if "verdict" in r}
    pp = os.path.join(jd, "pairwise.jsonl")
    PW = {(r["char"], r["id"]): r for r in map(json.loads, open(pp)) if "better" in r} if os.path.exists(pp) else {}
    return runs, J, PW


def stats(runs, J, PW, ch, cells):
    out = {}
    for m in ("current", "semantic"):
        v = [J[(ch, m, i)] for i in cells if (ch, m, i) in J]
        n = len(v)
        out[m] = {"n": n, "score": round(sum(1 if x["verdict"] == "yes" else .5 if x["verdict"] == "partly" else 0 for x in v) / n, 3) if n else None,
                  "no": round(sum(x["verdict"] == "no" for x in v) / n, 3) if n else None,
                  "natural": round(st.mean(x.get("natural") or 0 for x in v), 2) if n else None,
                  "first_motion_med": pct([runs[(ch, m, i)]["timings"].get("first_motion_ms") for i in cells if (ch, m, i) in runs], .5),
                  "first_motion_p90": pct([runs[(ch, m, i)]["timings"].get("first_motion_ms") for i in cells if (ch, m, i) in runs], .9)}
    pw = [PW[(ch, i)] for i in cells if (ch, i) in PW]
    if pw:
        out["pairwise_semantic"] = {f: round(sum(1 if p.get(f + "_mode") == "semantic" else .5 if not p.get(f + "_mode") else 0 for p in pw) / len(pw), 3)
                                    for f in ("better", "more_natural")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    res = {}
    print("| character | metric (planned only) | current r1 | current r2 | semantic r1 | semantic r2 |")
    print("|---|---|---|---|---|---|")
    for ch in ("muse", "fox", "juno"):
        def planned(R, i):
            return all((R[0].get((ch, m, i)) or {}).get("timings", {}).get("source") != "library" for m in ("current", "semantic"))
        cells = sorted({i for (c, m, i) in A[1] if c == ch} & {i for (c, m, i) in B[1] if c == ch})
        cells = [i for i in cells if planned(A, i) and planned(B, i)]
        if not cells:
            continue
        sa, sb = stats(*A, ch, cells), stats(*B, ch, cells)
        res[ch] = {"n": len(cells), "r1": sa, "r2": sb}
        f = lambda x, pctg=True: "–" if x is None else (f"{x:.0%}" if pctg else f"{x}")
        for key, lab, pc in (("score", "did-it score", True), ("no", "'no' share", True), ("natural", "natural 1-5", False),
                             ("first_motion_med", "move starts median ms", False), ("first_motion_p90", "move starts p90 ms", False)):
            print(f"| {ch} (n {len(cells)}) | {lab} | {f(sa['current'][key], pc)} | {f(sb['current'][key], pc)} | {f(sa['semantic'][key], pc)} | {f(sb['semantic'][key], pc)} |")
        for k in ("better", "more_natural"):
            x, y = (sa.get("pairwise_semantic") or {}).get(k), (sb.get("pairwise_semantic") or {}).get(k)
            print(f"| {ch} (n {len(cells)}) | pairwise {k}: semantic share | | | {f(x)} | {f(y)} |")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
