#!/usr/bin/env python3
"""what a fallback start threshold (planner.fallback.after_s) does, from measured first-pose times.

    python3 tools/plan_race_sim.py --claude probe.json[,probe2.json] [--arm q1nt] --gem <plan_eval run dir of the fallback alone>
        [--q-claude 3.50 --q-gem 3.10] [--thresholds 1.5,2,2.5,3,3.5,4.5]

Claude = the primary's first pose per call (probe rows, no fallback); gem = the fallback's first pose per item (plan.json
ms_first_pose of a run of the fallback alone). Every Claude call is paired with every gem time (all pairs): the fallback
starts at T when Claude has no pose yet; the first pose wins (planner.py semantics). -> served first pose p50 / p90 / max,
the share of plans the fallback serves, and (with --q-*) the expected mean judge score."""
import argparse, json, os


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))] if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude", required=True); ap.add_argument("--arm", default=None)
    ap.add_argument("--gem", required=True); ap.add_argument("--thresholds", default="1.5,2,2.5,3,3.5,4.5")
    ap.add_argument("--q-claude", type=float); ap.add_argument("--q-gem", type=float)
    a = ap.parse_args()
    C = []
    for p in a.claude.split(","):
        for r in json.load(open(p))["rows"]:
            if (a.arm is None or r["arm"] == a.arm) and r.get("primary_first_pose"):
                C.append(r["primary_first_pose"])
    G = []
    for d in sorted(os.listdir(a.gem)):
        pj = os.path.join(a.gem, d, "plan.json")
        if os.path.exists(pj):
            ev = json.load(open(pj))
            m = ev.get("ms_first_pose") if isinstance(ev, dict) else None
            if m and ev.get("source") == "planner":
                G.append(m)
    print(f"claude n={len(C)} p50={q(C, .5)} p90={q(C, .9)} max={max(C)} | gem n={len(G)} p50={q(G, .5)} p90={q(G, .9)} max={max(G)}")
    print("after_s | served p50  p90   max | fallback serves | fallback started" + (" | mean score" if a.q_claude else ""))
    for T in [float(x) for x in a.thresholds.split(",")]:
        served, gem_wins, started = [], 0, 0
        for c in C:
            for g in G:
                t = T * 1000
                if c <= t:
                    served.append(c)
                    continue
                started += 1
                if t + g < c:
                    served.append(t + g); gem_wins += 1
                else:
                    served.append(c)
        n = len(C) * len(G)
        share = gem_wins / n
        line = f"{T:7.1f} | {q(served, .5):6.0f} {q(served, .9):5.0f} {max(served):5.0f} | {share * 100:5.1f} %         | {started / n * 100:5.1f} %"
        if a.q_claude is not None and a.q_gem is not None:
            line += f" | {a.q_claude * (1 - share) + a.q_gem * share:.3f}"
        print(line)


if __name__ == "__main__":
    main()
