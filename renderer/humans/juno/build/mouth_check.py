"""asset-face r3 — numeric check of the procedural teeth/gums against the posed skin.

python mouth_check.py <mpfb2 data dir>
For closed-lip poses (rest, jawOpen w + mouthClose w, smile shot, press ...) every tooth / gum vertex must be hidden from
the camera side: a ray from the vertex toward the viewer (front, +-25 deg yaw, +-15 deg pitch) has to hit skin (lips,
cheeks, chin). Prints, per pose, the count of exposed vertices (0 = nothing pokes through). The lower row rides the
jaw bone: rotated about the hinge by jaw_open_deg * jawOpen (same as RIG).
"""
import sys

import numpy as np
from scipy.spatial import cKDTree

import juno_base
import face_shapes
import mouth_teeth
import eyes_mouth
from face_check import pose
from face_rig import rotate


def exposed(Pskin, tris, O, dirs):
    """per origin: True if some ray (any dir) from it misses every triangle."""
    vis = np.zeros(len(O), bool)
    for d in dirs:
        d = d / np.linalg.norm(d)
        # basis with d = local z
        a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(a, d); e1 /= np.linalg.norm(e1)
        e2 = np.cross(d, e1)
        M = np.stack([e1, e2, d])
        Q = Pskin @ M.T
        o = O @ M.T
        A, B, C = Q[tris[:, 0]], Q[tris[:, 1]], Q[tris[:, 2]]
        cen = (A + B + C)[:, :2] / 3
        rad = np.max(np.stack([np.linalg.norm(X[:, :2] - cen, axis=1) for X in (A, B, C)]), 0)
        kd = cKDTree(cen)
        R = rad.max()
        hit = np.zeros(len(O), bool)
        for i, cand in enumerate(kd.query_ball_point(o[:, :2], R)):
            if not cand:
                continue
            cand = np.array(cand)
            a2, b2, c2 = A[cand], B[cand], C[cand]
            p = o[i, :2]
            v0, v1, v2 = b2[:, :2] - a2[:, :2], c2[:, :2] - a2[:, :2], p - a2[:, :2]
            den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
            ok = np.abs(den) > 1e-16
            den = np.where(ok, den, 1)
            l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
            l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
            ins = ok & (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
            z = a2[:, 2] + l1 * (b2[:, 2] - a2[:, 2]) + l2 * (c2[:, 2] - a2[:, 2])
            hit[i] = np.any(ins & (z > o[i, 2] + 1e-5))
        vis |= ~hit
    return vis


def main():
    base = juno_base.Base(sys.argv[1])
    S = face_shapes.build(base)
    rig = face_shapes.build.rig
    cav = eyes_mouth._cavity_faces(base)
    cv = np.unique(np.concatenate([np.array([v for v, _ in base.obj.faces[f]]) for f in cav]))
    parts, rep = mouth_teeth.build(base, rig, cv)
    print("teeth report", rep)
    tris = base.tris[base.skin_mask[base.tris].all(1)]
    mo = rig.mouth_c
    near = np.linalg.norm(base.VA - mo, axis=1) < 0.09
    tris = tris[near[tris].all(1)]
    th = np.radians(face_shapes.jaw_open_deg(base, S))
    dirs = [np.array(v, float) for v in ([0, 0, 1], [0.42, 0, 0.9], [-0.42, 0, 0.9], [0, 0.26, 0.97], [0, -0.26, 0.97])]
    SM = {"mouthSmileLeft": .9, "mouthSmileRight": .9, "cheekSquintLeft": .5, "cheekSquintRight": .5, "eyeHappyLeft": .3,
          "eyeHappyRight": .3, "jawOpen": .12}
    poses = {"rest": {}, "jaw.12": {"jawOpen": .12}, "smileshot": SM}
    for j in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        poses[f"jaw{j}+close"] = {"jawOpen": j, "mouthClose": j}
    poses["pressLR"] = {"mouthPressLeft": 1, "mouthPressRight": 1}
    poses["smileLR"] = {"mouthSmileLeft": 1, "mouthSmileRight": 1}
    poses["pucker"] = {"mouthPucker": 1}
    poses["funnel"] = {"mouthFunnel": 1}
    for n, w in poses.items():
        P = pose(base, S, w)
        jw = w.get("jawOpen", 0.0)
        line = []
        for p in parts:
            Q = p.P
            if p.J[0, 0] == base.bidx["jaw"]:
                Q = rotate(Q, rig.hinge, rig.jaw_axis, np.full(len(Q), th * jw))
            v = exposed(P, tris, Q, dirs)
            line.append(f"{p.name} {int(v.sum())}/{len(Q)}")
        print(f"  {n:18s}", "  ".join(line))


if __name__ == "__main__":
    main()
