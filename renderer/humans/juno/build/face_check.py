"""asset-face — numeric checks of face_shapes: lid penetration (eyeball + cornea profile) and inner-lip gap.

python face_check.py <mpfb2 data dir>
"""
import sys

import numpy as np

import juno_base
import face_shapes
from face_rig import LID_CLEAR
import eyes_geo


def pose(base, S, w):
    w = face_shapes.weights_with_correctives(w)
    P = base.VA.copy()
    for k, v in w.items():
        if v and k in S:
            P += v * S[k]
    return P


def lid_pen(base, rig, P):
    out = {}
    for s in ("l", "r"):
        E = rig.eyes[s]
        c, R = E["c"], E["R"]
        q0 = base.VA - c
        r0 = np.linalg.norm(q0, axis=1)
        reg = E["region"] & (r0 >= E["floor"](q0[:, 2] / np.maximum(r0, 1e-9)) - LID_CLEAR)
        q = P[reg] - c
        r = np.linalg.norm(q, axis=1)
        out[s] = (float((r - R).min() * 1000), float((r - (E["floor"](q[:, 2] / r) - LID_CLEAR)).min() * 1000))
    return out


def blink_gap(rig, P, s):
    """outer lid margins (r > 1.05 R): arc distance mm between the upper and the lower margin along the central 80%
    of the aperture width (+ = open gap, - = upper lid overlaps the lower). Returns (max, min, mean)."""
    E = rig.eyes[s]
    c, R = E["c"], E["R"]
    out = []
    for ids in (E["up"], E["lo"]):
        ids = ids[np.linalg.norm(rig.VA[ids] - c, axis=1) > 1.05 * R]
        d = P[ids] - c
        el = np.arctan2(d[:, 1], d[:, 2])
        az = np.arctan2(d[:, 0] * (1 if s == "l" else -1), d[:, 2])
        o = np.argsort(az)
        out.append((az[o], el[o], np.linalg.norm(d, axis=1)[o]))
    (au, eu, ru), (al, e_l, rl) = out
    a0, a1 = max(au[0], al[0]), min(au[-1], al[-1])
    g = np.linspace(a0 + 0.1 * (a1 - a0), a1 - 0.1 * (a1 - a0), 30)
    gap = (np.interp(g, au, eu) - np.interp(g, al, e_l)) * np.interp(g, au, ru) * 1000
    return float(gap.max()), float(gap.min()), float(gap.mean())


def lip_gap(rig, P):
    d = np.linalg.norm(P[rig.seam_u] - P[rig.seam_l], axis=1)
    return float(d.max() * 1000), float(d.mean() * 1000)


def main():
    base = juno_base.Base(sys.argv[1])
    S = face_shapes.build(base)
    rig = face_shapes.build.rig
    S = {k: v.copy() for k, v in S.items()}
    rep = eyes_geo.conform_lids(base, {k: v for k, v in S.items() if not k.startswith("corr_")})
    print("conform_lids touched:", rep)
    print(f"jaw_open_deg {face_shapes.jaw_open_deg(base, S):.2f}  shapes {len(S)}  corr {len(face_shapes.CORRECTIVES)}")
    print("lid penetration: min(r-R) mm, min(r-cornea) mm  [rest-outside skin only]")
    sets = {"rest": {}}
    for wb in (0.25, 0.5, 0.75, 1.0):
        sets[f"blink{wb}"] = {"eyeBlinkLeft": wb, "eyeBlinkRight": wb}
    sets["smileshot"] = {"mouthSmileLeft": .9, "mouthSmileRight": .9, "cheekSquintLeft": .5, "cheekSquintRight": .5,
                         "eyeHappyLeft": .3, "eyeHappyRight": .3, "jawOpen": .12}
    for X in ("eyeSquint", "cheekSquint", "eyeHappy", "mouthSmile", "eyeLookDown", "eyeLookUp", "eyeWide"):
        sets[X] = {X + "Left": 1, X + "Right": 1}
        for wb in (0.5, 1.0):
            sets[f"blink{wb}+{X}"] = {X + "Left": 1, X + "Right": 1, "eyeBlinkLeft": wb, "eyeBlinkRight": wb}
    for b in ("browDown", "browOuterUp", "browTilt"):
        sets[b] = {b + "Left": 1, b + "Right": 1}
    for n, w in sets.items():
        P = pose(base, S, w)
        lp = lid_pen(base, rig, P)
        extra = ""
        if n.startswith("blink"):
            extra = "  lid gap mm max/min/mean L %.2f/%.2f/%.2f" % blink_gap(rig, P, "l") + " R %.2f/%.2f/%.2f" % blink_gap(rig, P, "r")
        print(f"  {n:24s} L {lp['l'][0]:6.2f} {lp['l'][1]:6.2f}   R {lp['r'][0]:6.2f} {lp['r'][1]:6.2f}{extra}")
    print("inner-lip gap (seam pairs, %d): max / mean mm" % len(rig.seam_u))
    for n, w in {"rest": {}, "jaw0.3": {"jawOpen": .3}, "jaw0.3+close0.3": {"jawOpen": .3, "mouthClose": .3},
                 "jaw1+close1": {"jawOpen": 1, "mouthClose": 1}, "jaw0.3+lipsTogether": {"jawOpen": .3, "lipsTogether": 1},
                 "jaw0.3+pressLR": {"jawOpen": .3, "mouthPressLeft": 1, "mouthPressRight": 1},
                 "press LR": {"mouthPressLeft": 1, "mouthPressRight": 1}, "pucker": {"mouthPucker": 1},
                 "smile LR": {"mouthSmileLeft": 1, "mouthSmileRight": 1}, "cheekPuff": {"cheekPuff": 1}}.items():
        print(f"  {n:24s} %.2f / %.2f" % lip_gap(rig, pose(base, S, w)))


if __name__ == "__main__":
    main()
