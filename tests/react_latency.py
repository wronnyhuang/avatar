"""the user's cue -> the avatar's response, learned mode, at the controller level (the same measure as the rules-mode
test in tests/motion_test.cjs: "his face -> tracker -> avatar latency: smile/surprise/frown 0.50 s, tilt 0.40 s, nod
back 0.37 s" = time from his change to 50 % of the avatar's response on the channel).
Runs the hub's own engine (app/react_worker.py: planner + retarget + blink events + merge) on synthetic partner frames.
  python3 -I tests/react_latency.py [--planner stub|bundle|delaycopy]
      [--bundle DIR] [--delays 0,0.25,0.5] [--root REACT_ROOT] [--character fox] [--json OUT]
The planner's step time (p50/p95) is reported from the same run."""
import argparse
import json
import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.environ.get("REACT_ROOT", ""), help="learned-reaction checkout (react/spec.py)")
ap.add_argument("--planner", default="stub")
ap.add_argument("--bundle", default="")
ap.add_argument("--infer", default="")
ap.add_argument("--delays", default="0,0.5")
ap.add_argument("--character", default="fox")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--json", default="")
ap.add_argument("--commit-ahead", type=int, default=0)
ap.add_argument("--plan-ms", type=float, default=33.0)
ap.add_argument("--xfade", type=int, default=3)
ap.add_argument("--trials", type=int, default=1, help="seeded runs averaged before t50 (stochastic planners)")
A = ap.parse_args()
if not A.root or not os.path.isfile(os.path.join(A.root, "react", "spec.py")):
    print("SKIP: needs a learned-reaction checkout (--root or REACT_ROOT, with react/spec.py); not shipped with this repo")
    sys.exit(0)
sys.path[:0] = [os.path.join(HERE, "..", "app"), A.root]
from react import spec  # noqa: E402
import react_worker as W  # noqa: E402
try:
    from react import retarget as RT  # noqa: E402
    rt_name = "react.retarget"
except ImportError:
    RT = W.mini_retarget(); rt_name = "mini"
B = spec.BS
CASES = {   # the rules test's cues (tests/motion_test.cjs), in his numbers
    "smile": ({"mouthSmileLeft": 0.6, "mouthSmileRight": 0.6, "eyeSquintLeft": 0.2, "eyeSquintRight": 0.2}, {}, "smile"),
    "surprise": ({"browInnerUp": 0.5, "browOuterUpLeft": 0.5, "browOuterUpRight": 0.5, "eyeWideLeft": 0.4, "eyeWideRight": 0.4, "jawOpen": 0.4}, {}, "eye_wide"),
    "frown": ({"browDownLeft": 0.5, "browDownRight": 0.5, "mouthPressLeft": 0.3, "mouthPressRight": 0.3}, {}, "brow_knit"),
    "tilt": ({}, {54: 0.2}, "head_roll"),
}
RIG = ["smile", "head_yaw", "head_pitch", "head_roll", "head_x", "head_y", "head_z", "breath", "blink_l", "blink_r",
       "gaze_x", "gaze_y", "brow_l", "brow_r", "brow_knit", "eye_wide", "squint", "ear_l", "ear_r"]


def neutral():
    f = np.zeros(86, np.float32)
    f[B["eyeBlinkLeft"]] = f[B["eyeBlinkRight"]] = 0.12
    f[60:78] = np.array([[0.5, 0, 0], [-0.5, 0, 0], [0.6, -1.0, 0], [-0.6, -1.0, 0], [0.6, -1.9, 0.1], [-0.6, -1.9, 0.1]], np.float32).ravel()
    return f


def run(delay, frames, ch_name, seconds=7.0):
    """mean response over --trials seeded runs -> rows [(t, value)], last engine"""
    acc = None
    for k in range(A.trials):
        rows, eng = run1(delay, frames, ch_name, seconds, seed=k)
        acc = np.array([v for _, v in rows]) if acc is None else acc + np.array([v for _, v in rows])
    return [(t, float(v)) for (t, _), v in zip(rows, acc / A.trials)], eng


def run1(delay, frames, ch_name, seconds=7.0, seed=0):
    args = types.SimpleNamespace(seed=seed, planner=A.planner, bundle=A.bundle, stub_delay=delay, threads=A.threads, plan_ms=A.plan_ms,
                                 commit_ahead=A.commit_ahead, xfade=A.xfade, ear_springs=1, root=A.root, trace="", infer=A.infer)
    eng = W.Engine(args, spec, RT); eng.character = A.character; eng.rig_params = RIG
    out = []
    for j in range(int(seconds * 30)):
        t = (j + 1) / 30
        eng.add_human(t - 1e-3, frames(t), [1, 1, 0, 0])
        f = eng.step([(t, np.zeros(len(RIG), np.float32), [], 0.0)])[0]
        out.append((t, float(f[RIG.index(ch_name)])))
    return out, eng


def t50(rows, t_cue=3.0, t_end=6.0, sign=None):
    base = np.mean([v for t, v in rows if 2.5 <= t < 2.95])
    win = [(t, v - base) for t, v in rows if t_cue < t < t_end]
    s = sign if sign is not None else (1 if max(d for _, d in win) >= -min(d for _, d in win) else -1)
    pk = max(s * d for _, d in win)
    if pk < 1e-3:
        return None, 0.0
    return next(t for t, d in win if s * d >= 0.5 * pk) - t_cue, pk


res = {"planner": A.planner, "retarget": rt_name, "character": A.character, "cases": {}}
for d in [float(x) for x in A.delays.split(",")]:
    lab = f"{A.planner}" + (f"(delay {d:.2f} s)" if A.planner != "bundle" else "") + f" commit_ahead {A.commit_ahead} plan_ms {A.plan_ms:g}"
    row = {}
    for name, (bs, cols, ch) in CASES.items():
        def frames(t, bs=bs, cols=cols):
            f = neutral()
            if 3 <= t < 6:
                for k, v in bs.items():
                    f[B[k]] = v
                for i, v in cols.items():
                    f[i] = v
            return f
        rows, eng = run(d, frames, ch)
        lat, pk = t50(rows)
        row[name] = {"t50_s": None if lat is None else round(lat, 3), "amp": round(pk, 3), "channel": ch}
    # a nod of his (pitch swing, 0.6 s) -> the avatar's head_pitch response
    def nodf(t):
        f = neutral(); ph = t - 3
        if 0 < ph < 0.6:
            f[53] = 0.09 * math.sin(math.pi * ph / 0.3) ** 2
        return f
    rows, eng = run(d, nodf, "head_pitch")
    lat, pk = t50(rows, 3.0, 4.5, sign=1)
    row["nod"] = {"t50_s": None if lat is None else round(lat, 3), "amp": round(pk, 3), "channel": "head_pitch"}
    rep = eng.report()
    row["planner_step_ms"] = {"plan": rep["plan_ms"], "push_commit": rep["push_commit_ms"], "tick": rep["planner_tick_ms"]}
    row["crossfades_per_s"] = round(rep.get("crossfades", 0) / 7.0, 2)
    row["trials"] = A.trials
    res["cases"][lab] = row
    print(lab, " · ".join(f"{k} {v['t50_s']} s (amp {v['amp']})" for k, v in row.items() if isinstance(v, dict) and "t50_s" in v),
          "| xf/s", row["crossfades_per_s"], "| tick ms", row["planner_step_ms"]["tick"])
if A.json:
    json.dump(res, open(A.json, "w"), indent=1)
