#!/usr/bin/env python3
"""where the Muse-family paw really goes (a port of renderer/animals/muse.glsl charInit's arm block; the
mascots share the hand MAP: engine/mascot_rig.glsl) and the paw-control values that land it on a body part.

    python3 tools/kf_reach.py                 # the landmark table the planner prompt quotes (make_prompts.py REACH_*)
    python3 tools/kf_reach.py --at -0.9 0.45 0.45     # where (paw_x, paw_y, paw_z) puts the wrist / palm

Units are the shader's (the character is M_H = 2.30 tall); the frame is the upper body's, mirrored so +x is the arm's own
side. fk(hx, hy, hz) -> {"want": the mapped target, "wrist": after the midline / torso / head push-outs and the reach
clamp, "palm", "short": how far the wrist stops short of the pushed target, "on_head", "on_torso"}."""
import argparse, itertools, json, math

L = 0.57                       # M_UPPER + M_FORE
SHO = (0.405, 1.035, 0.0)
REST = (0.540, 0.555, 0.130)
TORSO_C, TORSO_R = (0.0, 0.88, 0.0), (0.56, 0.56, 0.52)
HEAD_C, HEAD_R = (0.0, 1.64, 0.03), (0.70 + 0.085, 0.64 + 0.085, 0.62 + 0.085)
H = 2.30


def _sub(a, b): return tuple(x - y for x, y in zip(a, b))
def _len(a): return math.sqrt(sum(x * x for x in a))
def _ss(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a))); return t * t * (3 - 2 * t)


def fk(hx, hy, hz):
    T = [REST[0] + (0.95 if hx > 0 else 1.05) * L * hx, REST[1] + 1.40 * L * hy, REST[2] + 0.80 * L * hz]
    want = tuple(T)
    T[0] = max(T[0], 0.07)
    e = [(T[i] - TORSO_C[i]) / TORSO_R[i] for i in range(3)]
    el = _len(e)
    on_torso = el < 1.0
    if on_torso:
        T = [TORSO_C[i] + e[i] / max(el, 1e-6) * TORSO_R[i] for i in range(3)]
    k = 1.0
    q = [T[0] - HEAD_C[0], T[1] + 0.09 - HEAD_C[1], T[2] - HEAD_C[2]]
    eh = [q[i] / HEAD_R[i] for i in range(3)]
    elh = _len(eh)
    on_head = elh < 1.0
    if on_head:
        rad = [eh[i] / max(elh, 1e-4) * HEAD_R[i] for i in range(3)]
        front = [q[0], q[1], HEAD_R[2] * math.sqrt(max(1.0 - eh[0] ** 2 - eh[1] ** 2, 0.0))]
        m = _ss(0.0, 0.35, q[2] / HEAD_R[2])
        qn = [rad[i] + (front[i] - rad[i]) * m for i in range(3)]
        T = [HEAD_C[0] + qn[0], HEAD_C[1] + qn[1] - 0.09, HEAD_C[2] + qn[2]]
        k = min(1.7, max(1.0, _len(_sub(T, SHO)) / (0.995 * L)))
    d = _sub(T, SHO)
    Ld = _len(d)
    lmax, lmin = L * k * 0.995, abs(0.30 - 0.27) * k + 0.02
    Lc = min(lmax, max(lmin, Ld))
    W = tuple(SHO[i] + d[i] / max(Ld, 1e-6) * Lc for i in range(3))
    up = 0.75 * _ss(0.85, 1.35, W[1])
    return {"want": want, "wrist": W, "palm": (W[0], W[1] + 0.095 * up, W[2]), "short": Ld - Lc, "on_head": on_head,
            "on_torso": on_torso}


def solve(target, step=0.05, tol_short=0.03):
    """the (hx, hy, hz) on the control grid whose wrist ends nearest `target` (ties: the least stretched request)"""
    best = None
    rng = [round(-1 + step * i, 2) for i in range(int(2.2 / step) + 1)]
    for hx, hy, hz in itertools.product(rng, rng, rng):
        r = fk(hx, hy, hz)
        d = _len(_sub(r["wrist"], target)) + 0.3 * max(0.0, r["short"] - tol_short)
        if best is None or d < best[0] - 1e-9:
            best = (d, hx, hy, hz, r)
    return best


# body spots in the same frame (x toward the arm's own side). Surface points of the torso / head ellipsoids + free space.
SPOTS = {
    "hand on its hip (elbow out)": (0.50, 0.66, 0.16),
    "on the belly, its own side of the navel": (0.20, 0.70, 0.50),
    "on the chest / heart": (0.14, 0.98, 0.50),
    "paws together in front of the chest (clap, pray, hold)": (0.07, 0.95, 0.52),
    "in front of the mouth": (0.10, 1.38, 0.70),
    "on the cheek, its own side": (0.52, 1.45, 0.50),
    "over its own-side eye": (0.25, 1.68, 0.70),
    "on the forehead / brow (salute)": (0.30, 1.95, 0.62),
    "beside the head, out in the open (wave, hello)": (0.95, 1.30, 0.15),
    "straight out to its side at shoulder height": (0.97, 1.04, 0.0),
    "as high as it gets (up / overhead)": (0.62, 1.56, 0.10),
    "straight forward at the camera": (0.40, 1.05, 0.56),
    "behind its back": (0.30, 0.62, -0.42),
    "down by its thigh": (0.50, 0.50, 0.10),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--at", nargs=3, type=float); ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.at:
        print(json.dumps({k: ([round(x, 3) for x in v] if isinstance(v, tuple) else v) for k, v in fk(*a.at).items()}))
    else:
        out = {}
        for name, tgt in SPOTS.items():
            d, hx, hy, hz, r = solve(tgt)
            out[name] = {"paw": [hx, hy, hz], "miss": round(d, 3), "wrist": [round(x, 2) for x in r["wrist"]]}
            if not a.json:
                print(f"{name:58s} x {hx:5.2f} y {hy:5.2f} z {hz:5.2f}   miss {d:.3f} wrist {out[name]['wrist']} head={r['on_head']}")
        if a.json:
            print(json.dumps(out))
