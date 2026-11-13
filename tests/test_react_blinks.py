"""learned mode's blinks come from the page's rule-based blinks by default (react.blinks = "rules").
motion.js (node, learned mode on, so the page also emits its affine map) runs 3 min of a face with alternating avatar
speech / the user's speech / silence and a few reflex events; its rig stream goes through the hub's engine
(app/react_worker.py) with the stub planner, whose partner blinks only every 10 s (6/min, like today's model).
  blinks=rules   -> the shown blink rate == the rules stream's rate (same events, same frames)
  blinks=learned -> the planner's rate (~6/min)
  python3 -I tests/test_react_blinks.py [--root REACT_ROOT]   (SKIPs without a learned-reaction checkout)"""
import argparse
import json
import os
import subprocess
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.environ.get("REACT_ROOT", ""), help="learned-reaction checkout (react/spec.py)")
ap.add_argument("--seconds", type=float, default=180)
A = ap.parse_args()
if not A.root or not os.path.isfile(os.path.join(A.root, "react", "spec.py")):
    print("SKIP: needs a learned-reaction checkout (--root or REACT_ROOT, with react/spec.py); not shipped with this repo")
    sys.exit(0)
sys.path[:0] = [os.path.join(HERE, "..", "app"), A.root]
from react import spec  # noqa: E402
from react import retarget as RT  # noqa: E402
import react_worker as W  # noqa: E402

NODE = r"""
const M = require(process.argv[1] + '/load.cjs')('app/motion.js');
const T = +process.argv[2], DT = 1 / 30, m = M.createMotion({seed: 11, preset: 'default', learned: true}), out = [];
const EV = {600: {type: 'face_appear'}, 1500: {type: 'approach_fast', rate: 1.1}, 2700: {type: 'look_back', away_s: 3}, 3900: {type: 'nod'}};
for (let k = 0; k < T * 30; k++) {
  const t = k * DT, ph = Math.floor(t / 15) % 3;            // 15 s phases: avatar speaks / he speaks / quiet
  const sp = ph === 0 && Math.sin(t * 3.1) > -0.6, his = ph === 1 && Math.sin(t * 2.7) > -0.6;
  const face = {present: true, x: 0.08 * Math.sin(t / 3), y: 0.04, z: 0.45, yaw: 0.06 * Math.sin(t / 2.3), pitch: 0, roll: 0, blend: {}};
  const ev = EV[k] ? [Object.assign({t: k}, EV[k])] : [];
  const o = m.step(DT, {features: {t, face, attending: true, attention: 0.9, motion: 0.05, hands: []}, events: ev,
    avatarVoice: {rms: sp ? 0.05 : 0, voiced: sp, pitch: sp ? 180 : 0}, userVoice: {rms: his ? 0.06 : 0.001, pitch: his ? 130 : 0},
    speaking: sp, listening: his});
  out.push([m.vector(o), m.learnedParts(M.RIG_PARAMS)]);
}
console.log(JSON.stringify({params: M.RIG_PARAMS, rows: out}));
"""
d = json.loads(subprocess.check_output(["node", "-e", NODE, HERE, str(A.seconds)], cwd=os.path.join(HERE, "..")))
P = d["params"]; rows = d["rows"]
iL, iR = P.index("blink_l"), P.index("blink_r")


def onsets(x, th=0.5):
    x = np.asarray(x)
    return int(np.sum((x[1:] >= th) & (x[:-1] < th)))


def partner(j):
    f = W.StubPlanner._neutral(None)
    f[spec.BS["eyeBlinkLeft"]] = f[spec.BS["eyeBlinkRight"]] = 0.12
    if j % 300 < 6:                                     # one blink every 10 s
        f[spec.BS["eyeBlinkLeft"]] = f[spec.BS["eyeBlinkRight"]] = 0.75
    return f


def run(mode):
    args = types.SimpleNamespace(planner="stub", stub_delay=0.5, threads=1, plan_ms=33.0, commit_ahead=0, xfade=3,
                                 ear_springs=1, root=A.root, bundle="", trace="", blinks=mode)
    eng = W.Engine(args, spec, RT); eng.character = "fox"; eng.rig_params = P
    out = []
    for j, (a, lp) in enumerate(rows):
        t = (j + 0.5) / 30
        eng.add_human(j / 30, partner(j), [1, 1, 0, 0])
        ab = [(int(i), float(al), float(be)) for i, al, be in lp["ab"]]
        out.append(eng.step([(t, np.asarray(a, np.float32), ab, float(lp["rb"]))])[0])
    return np.array(out), eng


mins = len(rows) / 30 / 60
rules_l = [r[0][iL] for r in rows]; rules_r = [r[0][iR] for r in rows]
r_rate = onsets(np.maximum(rules_l, rules_r)) / mins
fr, er = run("rules")
fl, el = run("learned")
lr_rate = onsets(np.maximum(fr[:, iL], fr[:, iR])) / mins
ll_rate = onsets(np.maximum(fl[:, iL], fl[:, iR])) / mins
ref = np.array([[r[0][iL], r[0][iR]] for r in rows], np.float32)
dif = np.abs(fr[:, [iL, iR]] - ref)
same = bool(dif.max() < 1e-6)
if not same:
    bad = np.where(dif.max(1) >= 1e-6)[0]
    print("mismatch frames", bad[:10], fr[bad[:5]][:, [iL, iR]], ref[bad[:5]])
print(f"blinks/min over {mins:.1f} min: rules stream {r_rate:.1f} | learned+blinks=rules {lr_rate:.1f} | learned+blinks=learned {ll_rate:.1f} (partner 6.0)")
fails = []


def check(name, ok, det=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  — {det}" if det else ""))
    if not ok:
        fails.append(name)


check("blinks=rules: learned mode blinks at the rules rate (within 5%)", abs(lr_rate - r_rate) <= 0.05 * r_rate, f"{lr_rate:.1f} vs {r_rate:.1f}/min")
check("blinks=rules: the shown lids are exactly the page's rules blinks (frame for frame)", same)
check("rules blink rate is in the real range (15-35/min)", 15 <= r_rate <= 35, f"{r_rate:.1f}")
check("blinks=learned: the planner's own rate shows instead (~6/min here)", abs(ll_rate - 6.0) <= 1.5, f"{ll_rate:.1f}")
check("default is rules", W.Engine(types.SimpleNamespace(planner="stub", stub_delay=0.5, threads=1, plan_ms=33.0, commit_ahead=0, xfade=3,
                                                         ear_springs=1, root=A.root, bundle="", trace=""), spec, RT).blinks == "rules")
# the committed own frame carries the blinks that were shown (the model's own-motion history is what it saw)
own_bl = np.array([er.shown[j][spec.BS["eyeBlinkRight"]] for j in sorted(er.shown)[-60:]])
check("blinks=rules: committed own frames carry the shown blinks", own_bl.max() > 0.5 or np.max(fr[-60:, [iL, iR]]) < 0.5,
      f"max eyeBlink in the last 60 committed frames {own_bl.max():.2f}")
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
