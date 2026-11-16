#!/usr/bin/env python3
"""Runs + judge -> RESULTS.md / RESULTS.json.

  python3 tools/sem_eval/aggregate.py --runs EVAL/runs/<tag> --judge EVAL/judge/<tag> --out EVAL/RESULTS[-<tag>]
        [--cmds tests/sem_eval/commands.jsonl]

Per planner x character (and x category): did-it yes / partly / no (judge), naturalness mean (1-5), pairwise wins
(better / more natural; ties = 0.5), sources (library / cache / planner), latency median / p90 (ms after the call, page
clock) for: hub first pose (hub-side, command received -> first pose sent), first pose at the page, first motion of the
planned move on the rig (perform marks: after the cue hold), any motion (incl. the thinking lead-in) — fresh and cached
passes; failure list (errors, outcome != done, verdict no) with reasons.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict

LAT = [("hub_first_pose_ms", "hub first pose"), ("first_pose_ms", "page first pose"), ("first_motion_ms", "move starts on rig"),
       ("any_motion_ms", "any motion (incl. thinking)"), ("end_ms", "move ends")]


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo))


def med(xs):
    return pct(xs, 0.5)


def load_runs(root):
    rows = []
    for p in glob.glob(os.path.join(root, "*", "*", "*", "*.json")):
        if os.path.basename(p) == "run.json":
            continue
        try:
            R = json.load(open(p))
        except ValueError:
            continue
        if "timings" not in R and "errors" not in R:
            continue
        rows.append(R)
    return rows


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--judge", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cmds", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                                    "tests", "sem_eval", "commands.jsonl"))
    a = ap.parse_args()
    cmds = {c["id"]: c for c in (json.loads(l) for l in open(a.cmds) if l.strip())}
    runs = load_runs(a.runs)
    act = load_jsonl(os.path.join(a.judge, "action.jsonl")) if a.judge else []
    pair = load_jsonl(os.path.join(a.judge, "pairwise.jsonl")) if a.judge else []
    modes = sorted({r["mode"] for r in runs})
    chars = [c for c in ("muse", "fox") if any(r["char"] == c for r in runs)]
    cats = sorted({c["category"] for c in cmds.values()})
    J = {(r["char"], r["mode"], r["id"]): r for r in act if "verdict" in r}
    out = {"runs": a.runs, "judge": a.judge, "modes": modes, "chars": chars, "n_runs": len(runs), "n_judged": len(J), "groups": {},
           "latency": {}, "pairwise": {}, "failures": []}

    def judge_stats(keys):
        v = [J[k] for k in keys if k in J]
        n = len(v)
        if not n:
            return {"n": 0}
        c = defaultdict(int)
        for x in v:
            c[x["verdict"]] += 1
        return {"n": n, "yes": round(c["yes"] / n, 3), "partly": round(c["partly"] / n, 3), "no": round(c["no"] / n, 3),
                "score": round((c["yes"] + 0.5 * c["partly"]) / n, 3),
                "natural": round(st.mean(x.get("natural") or 0 for x in v), 2)}

    # judge: per mode x char, mode x category, mode overall
    for m in modes:
        fresh = [r for r in runs if r["mode"] == m and r["pass"] == "fresh"]
        out["groups"][m] = {"all": judge_stats([(r["char"], m, r["id"]) for r in fresh])}
        for ch in chars:
            out["groups"][m][ch] = judge_stats([(r["char"], m, r["id"]) for r in fresh if r["char"] == ch])
        for cat in cats:
            out["groups"][m]["cat:" + cat] = judge_stats([(r["char"], m, r["id"]) for r in fresh if r.get("category") == cat])
        for ch in chars:
            for cat in cats:
                out["groups"][m][f"{ch}:{cat}"] = judge_stats([(r["char"], m, r["id"]) for r in fresh if r["char"] == ch and r.get("category") == cat])
        # sources + latency
        for ps in ("fresh", "cached"):
            rs = [r for r in runs if r["mode"] == m and r["pass"] == ps]
            if not rs:
                continue
            L = {}
            for scope, sub in [("all", rs)] + [(ch, [r for r in rs if r["char"] == ch]) for ch in chars] + \
                    [("src:" + s, [r for r in rs if (r.get("timings") or {}).get("source") == s]) for s in ("library", "cache", "planner")]:
                if not sub:
                    continue
                d = {"n": len(sub), "sources": dict(sorted(defaultdict(int, {}).items()))}
                srcs = defaultdict(int)
                for r in sub:
                    s0 = (r.get("timings") or {}).get("source") or "none"
                    dn = [k for k in r.get("kf") or [] if k.get("done")]
                    if s0 == "planner" and dn and float(dn[-1].get("usd") or 0) > 0:
                        s0 = "planner via paid fallback"     # the current planner's Gemini/OpenRouter fallback answered
                    srcs[s0] += 1
                d["sources"] = dict(srcs)
                for k, _ in LAT:
                    xs = [(r.get("timings") or {}).get(k) for r in sub]
                    d[k] = {"med": med(xs), "p90": pct(xs, 0.9), "n": sum(x is not None for x in xs)}
                if m == "semantic":
                    for k in ("first_token", "first_stmt", "first_pose", "done"):
                        xs = [((r.get("timings") or {}).get("sem") or {}).get(k) for r in sub]
                        d["sem_" + k] = {"med": med(xs), "p90": pct(xs, 0.9), "n": sum(x is not None for x in xs)}
                L[scope] = d
            out["latency"][f"{m}/{ps}"] = L
    # pairwise
    if pair:
        for f in ("better", "more_natural"):
            for scope in ["all"] + chars + ["cat:" + c for c in cats]:
                sub = [p for p in pair if scope == "all" or p["char"] == scope or ("cat:" + (p.get("category") or "")) == scope]
                if not sub:
                    continue
                w = defaultdict(float)
                for p in sub:
                    if p.get(f + "_mode"):
                        w[p[f + "_mode"]] += 1
                    elif p.get(f) == "tie":
                        for m in modes:
                            w[m] += 0.5
                out["pairwise"][f"{f}/{scope}"] = {"n": len(sub), **{m: round(w[m] / len(sub), 3) for m in modes}}
    # failures
    for r in sorted(runs, key=lambda r: (r["mode"], r["char"], r["id"], r["pass"])):
        T = r.get("timings") or {}
        j = J.get((r["char"], r["mode"], r["id"])) if r["pass"] == "fresh" else None
        why = []
        if r.get("errors"):
            why += r["errors"]
        if T.get("outcome") not in ("done", None) or not T:
            why.append(f"outcome {T.get('outcome')}")
        if j and j["verdict"] == "no":
            why.append("judge: no — " + j.get("reason", ""))
        if why:
            out["failures"].append({"mode": r["mode"], "char": r["char"], "id": r["id"], "pass": r["pass"], "source": T.get("source"), "why": why})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    json.dump(out, open(a.out + ".json", "w"), indent=1)

    # ---- markdown
    L = []
    w = L.append
    w(f"# live-pipeline eval — {', '.join(modes)} x {', '.join(chars)}\n")
    w(f"runs `{a.runs}` ({len(runs)} rows)  judge `{a.judge}` ({len(J)} clips judged, blind Claude Sonnet on contact sheets)\n")
    w("## Did it do the command (judge, fresh pass): yes / partly / no, score = yes + ½ partly, naturalness 1-5\n")
    w("| planner | scope | n | yes | partly | no | score | natural |")
    w("|---|---|---|---|---|---|---|---|")
    for m in modes:
        for scope in ["all"] + chars + ["cat:" + c for c in cats]:
            g = out["groups"][m].get(scope) or {}
            if g.get("n"):
                w(f"| {m} | {scope} | {g['n']} | {g['yes']:.0%} | {g['partly']:.0%} | {g['no']:.0%} | {g['score']:.0%} | {g['natural']} |")
    if out["pairwise"]:
        w("\n## Blind pairwise (same character, same command; ties = ½)\n")
        w("| question | scope | n | " + " | ".join(modes) + " |")
        w("|---|---|---|" + "---|" * len(modes))
        for k, v in out["pairwise"].items():
            f, scope = k.split("/", 1)
            w(f"| {f} | {scope} | {v['n']} | " + " | ".join(f"{v[m]:.0%}" for m in modes) + " |")
    w("\n## Latency (ms after the call, median / p90)\n")
    w("| planner/pass | scope | n | sources | " + " | ".join(lbl for _, lbl in LAT) + " |")
    w("|---|---|---|---|" + "---|" * len(LAT))
    for key, Ls in out["latency"].items():
        for scope, d in Ls.items():
            srcs = ", ".join(f"{k} {v}" for k, v in d["sources"].items())
            w(f"| {key} | {scope} | {d['n']} | {srcs} | " + " | ".join(
                f"{d[k]['med']} / {d[k]['p90']}" if d[k]["med"] is not None else "–" for k, _ in LAT) + " |")
    sem_rows = [(k, s, d) for k, Ls in out["latency"].items() for s, d in Ls.items() if "sem_first_stmt" in d]
    if sem_rows:
        w("\n### Semantic planner stages (hub ms from the hub's receipt, median / p90)\n")
        w("| pass | scope | first token | first statement | first pose | done |")
        w("|---|---|---|---|---|---|")
        for k, s, d in sem_rows:
            w(f"| {k} | {s} | " + " | ".join(f"{d['sem_' + x]['med']} / {d['sem_' + x]['p90']}" for x in ("first_token", "first_stmt", "first_pose", "done")) + " |")
    w(f"\n## Failures ({len(out['failures'])})\n")
    for f in out["failures"]:
        w(f"- {f['mode']} / {f['char']} / {f['id']} ({f['pass']}, {f['source']}): " + " | ".join(str(x)[:200] for x in f["why"]))
    robotic = [(k, v.get("robotic")) for k, v in J.items() if v.get("robotic")]
    if robotic:
        w("\n## What the judge called robotic (fresh clips)\n")
        for (ch, m, cid), txt in sorted(robotic):
            w(f"- {m} / {ch} / {cid}: {txt}")
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print(open(a.out + ".md").read()[:3000])


if __name__ == "__main__":
    main()
