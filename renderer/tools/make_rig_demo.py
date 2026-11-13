#!/usr/bin/env python3
"""A rig-vector timeline that walks through every contract key pose and the face/ear/nose/tongue/eye/tail/body params,
with minimum-jerk transitions (350 ms), a wave, a clap, blinks and breath — for tools/render_timeline.py and QA.

  python tools/make_rig_demo.py --out demo.jsonl [--hz 30]
  python tools/render_timeline.py --character fox --rig demo.jsonl --backend gpu --out fox_demo.mp4
"""
import argparse, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from anim.rig import RIG_PARAMS  # noqa: E402
from anim.rig_poses import KEY_POSES, REST  # noqa: E402

# (label, hold seconds, targets on top of REST, oscillation spec)
SEQ = [
    ("REST", 1.2, {}, None),
    ("WAVE_UP", 2.6, KEY_POSES["WAVE_UP"], "wave_r"),
    ("REST", 0.8, {}, None),
    ("POINT_CAMERA", 1.6, KEY_POSES["POINT_CAMERA"], None),
    ("POINT_SIDE", 1.4, KEY_POSES["POINT_SIDE"], None),
    ("CLAP", 2.2, KEY_POSES["CLAP"], "clap"),
    ("THUMBS_UP", 1.6, KEY_POSES["THUMBS_UP"], None),
    ("CHIN", 1.8, dict(KEY_POSES["CHIN"], eye_wide=0.0, squint=0.3, head_roll=0.12, gaze_y=0.4, gaze_x=-0.4), None),
    ("SHRUG", 1.6, dict(KEY_POSES["SHRUG"], brow_l=0.8, brow_r=0.8), None),
    ("COVER_EYES", 1.8, KEY_POSES["COVER_EYES"], None),
    ("HAND_ON_HEART", 1.6, dict(KEY_POSES["HAND_ON_HEART"], smile=0.8), None),
    ("REST", 0.6, {}, None),
    ("ears up/out", 1.2, {"ear_l": 1.0, "ear_r": 1.0, "ear_l_turn": 1.0, "ear_r_turn": 1.0}, None),
    ("ears back/in", 1.2, {"ear_l": -1.0, "ear_r": -1.0, "ear_l_turn": -1.0, "ear_r_turn": -1.0}, None),
    ("ear flicks", 1.2, {}, "ear_flick"),
    ("nose wiggle", 1.2, {}, "nose"),
    ("tongue out", 1.4, {"tongue_out": 1.0, "squint": 0.4}, None),
    ("surprise", 1.2, {"eye_wide": 1.0, "mouth_open": 0.6, "brow_l": 1.0, "brow_r": 1.0}, None),
    ("wink", 1.0, {"blink_r": 1.0, "smile": 0.8, "squint": 0.3}, None),
    ("tail", 2.2, {"tail_raise": 1.0, "tail_curl": 0.6}, "tail"),
    ("body lean/turn", 2.4, {}, "body"),
    ("REST", 1.0, {}, None),
]


def minjerk(u):
    u = min(max(u, 0.0), 1.0)
    return u*u*u*(10 - 15*u + 6*u*u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--ease", type=float, default=0.35)
    a = ap.parse_args()
    # absolute targets per segment
    segs, t = [], 0.0
    for label, hold, tg, osc in SEQ:
        full = dict(REST); full.update(tg)
        segs.append((t, t + hold, label, full, osc))
        t += hold
    total = t
    names = list(RIG_PARAMS)
    out = [json.dumps({"rig_params": names})]
    prev = dict(REST)
    n = int(total*a.hz)
    for k in range(n + 1):
        tt = k/a.hz
        i = max(j for j, s in enumerate(segs) if s[0] <= tt + 1e-9)
        t0, t1, label, cur, osc = segs[i]
        before = segs[i - 1][3] if i > 0 else REST
        w = minjerk((tt - t0)/a.ease)
        keys = set(before) | set(cur)
        v = {kk: before.get(kk, 0.0) + (cur.get(kk, 0.0) - before.get(kk, 0.0))*w for kk in keys}
        ph = tt - t0
        if osc == "wave_r":
            s = math.sin(2*math.pi*2.5*ph)*w
            v["hand_r_x"] = v.get("hand_r_x", 0) + 0.12*s
            v["hand_r_turn"] = v.get("hand_r_turn", 0) - 0.3*(0.5 + 0.5*s)*w
        elif osc == "clap":
            apart = (0.5 + 0.5*math.cos(2*math.pi*2.0*ph))*w
            v["hand_l_x"] = v.get("hand_l_x", 0) + 0.22*apart
            v["hand_r_x"] = v.get("hand_r_x", 0) + 0.22*apart
        elif osc == "ear_flick":
            v["ear_flick_l"] = max(0.0, math.sin(2*math.pi*3.0*ph))**4
            v["ear_flick_r"] = max(0.0, math.sin(2*math.pi*3.0*ph + 1.8))**4
        elif osc == "nose":
            v["nose_wiggle"] = math.sin(2*math.pi*5.0*ph)*min(1.0, ph/0.2)
        elif osc == "tail":
            v["tail_wag"] = math.sin(2*math.pi*1.4*ph)
        elif osc == "body":
            v["body_lean"] = 0.8*math.sin(2*math.pi*0.42*ph)
            v["body_yaw"] = 0.7*math.sin(2*math.pi*0.42*ph + 1.2)
            v["body_pitch"] = 0.6*math.sin(2*math.pi*0.84*ph)
            v["body_bounce"] = 0.5*max(0.0, math.sin(2*math.pi*1.6*ph))
        # life the page would add: blinks, breath, a little head
        v["breath"] = math.sin(2*math.pi*0.25*tt)
        if (tt % 3.1) < 0.12 and label != "wink":
            v["blink_l"] = v["blink_r"] = 1.0
        v["head_yaw"] = v.get("head_yaw", 0.0) + 0.06*math.sin(tt*0.9)
        v["smile"] = max(v.get("smile", 0.0), 0.25)
        out.append(json.dumps({"type": "rig", "t": round(tt, 4), "a": [round(float(v.get(nm, 0.0)), 4) for nm in names],
                               "label": label}))
    open(a.out, "w").write("\n".join(out) + "\n")
    print(json.dumps({"out": a.out, "seconds": round(total, 2), "rows": n + 1}))


if __name__ == "__main__":
    main()
