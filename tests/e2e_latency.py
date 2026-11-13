"""pipeline latency from an e2e run (tests/e2e_react.cjs dump.json [+ the worker's --trace]):
camera frame capture -> the avatar frame that USED it is drawn on the page, per rig tick.
  rules  : rig vector at timeline rt was computed from perception features captured at f.t  -> draw(rt) - f.t
  learned: the frame shown at tick j was planned at tick j-1 from the partner frame captured at pt[j-1] (worker trace)
           -> draw(j/30) - capture(pt[j-1]); add the planner's own reaction time (react_latency.py) for cue -> response.
Also: the page's playout delay (rig stamp -> its frame drawn), uplink bytes of the learned extras.
  python tests/e2e_latency.py RUN_DIR [--trace worker_trace.jsonl] [--json OUT]"""
import argparse
import bisect
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--trace", default="")
ap.add_argument("--json", default="")
A = ap.parse_args()
d = json.load(open(f"{A.run}/dump.json"))
mp = np.array(d["map"], np.float64)                       # [perf ms, timeline s]
if len(mp) < 2:
    raise SystemExit("no timeline map")
slope, icpt = np.polyfit(mp[:, 1], mp[:, 0], 1)           # perf = icpt + slope * timeline
fit_err = float(np.abs(icpt + slope * mp[:, 1] - mp[:, 0]).max())
perf_of = lambda tl: icpt + slope * tl                     # noqa: E731
draws = sorted((r["t"], r["perf"]) for r in d["draw"])
dt_ = [x[0] for x in draws]


def draw_perf(tl):
    """perf ms when the first frame depicting timeline >= tl was drawn (None if never)."""
    i = bisect.bisect_left(dt_, tl - 1e-4)
    return draws[i][1] if i < len(draws) and draws[i][0] - tl < 0.08 else None


def pq(xs):
    xs = np.sort(np.asarray([x for x in xs if x is not None], np.float64))
    if not len(xs):
        return None
    return {"n": int(len(xs)), "p50": round(float(np.percentile(xs, 50)), 1), "p95": round(float(np.percentile(xs, 95)), 1)}


out = {"run": A.run, "learned": d.get("learned"), "map_fit_err_ms": round(fit_err, 2), "slope_ms_per_s": round(slope, 3),
       "frames_drawn": len(draws)}
# playout: a rig vector stamped rt -> its frame on screen
play = [draw_perf(r["rt"]) - r["perf"] for r in d["rig"] if r.get("rt") is not None and draw_perf(r["rt"]) is not None]
out["rig_stamp_to_drawn_ms"] = pq(play)
# rules pipeline (the page's own vector)
rules = []
for r in d["rig"]:
    if r.get("rt") is None or r.get("ft") is None:
        continue
    p = draw_perf(r["rt"])
    if p is not None:
        rules.append(p - 1000 * r["ft"])
out["rules_capture_to_drawn_ms"] = pq(rules)
if A.trace:
    tr = [json.loads(x) for x in open(A.trace) if x.strip()]
    tr = [x for x in tr if "j" in x]
    by = {x["j"]: x for x in tr}
    lr = []
    for x in tr:
        prev = by.get(x["j"] - 1)
        if not prev or prev.get("pt") is None:
            continue
        p = draw_perf(x["j"] / 30)
        if p is not None:
            lr.append(p - perf_of(prev["pt"]))
    out["learned_capture_to_drawn_ms"] = pq(lr)
    out["learned_ticks_traced"] = len(tr)
    out["partner_face_frac"] = round(float(np.mean([x["pf"] for x in tr])), 3) if tr else None
rx = d.get("rx") or {}
print(json.dumps(out, indent=1))
if A.json:
    json.dump(out, open(A.json, "w"), indent=1)
