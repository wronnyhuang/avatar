"""rig: a contact sheet of Juno's skeleton + shadow proxies + hand/grasp frames for named poses (no mesh needed).
Each cell: front view (x, y) and side view (z, y) in char-local scene units, world after the root, floor line, the
chalkboard plane (side view) and the chair seat / desk top heights for scale.

  python renderer/humans/tools/rig_sheet.py --asset DIR --out sheet.png [--poses REST,WAVE,...] [--face]
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from anim.schema import IDX, NPARAMS  # noqa: E402
from humans.rig import JunoRig  # noqa: E402
from humans.juno_poses_test import POSES  # noqa: E402

CELL = 360
COL = {0: (90, 90, 230), 1: (90, 160, 230), 2: (80, 200, 120), 3: (80, 200, 200), 4: (230, 160, 60), 5: (230, 90, 200),
       6: (160, 120, 230), 7: (120, 120, 120), 8: (40, 80, 160)}


def vec(pose):
    P = np.zeros(NPARAMS)
    for k, v in pose.items():
        P[IDX[k]] = v
    return P


def draw(img, f, rig, ox, oy, scale, view):
    def pr(p):
        p = np.asarray(p, float)
        x = p[0] if view == 0 else p[2]
        return int(ox + x*scale), int(oy - p[1]*scale)
    cv2.line(img, (ox - CELL//4, oy), (ox + CELL//4, oy), (60, 60, 60), 1)
    for y, c in ((0.44, (40, 90, 40)), (0.92, (40, 60, 90)), (1.97, (90, 40, 40))):
        cv2.line(img, (ox - CELL//5, int(oy - y*scale)), (ox - CELL//5 + 8, int(oy - y*scale)), c, 1)
    for cap in f.proxies:
        a, b, r, k = cap[0:3], cap[3:6], float(cap[6]), int(cap[7])
        c = COL.get(k, (200, 200, 200))
        pa, pb = pr(a), pr(b)
        cv2.line(img, pa, pb, c, max(1, int(2*r*scale)), cv2.LINE_AA)
    W = f.bones[:, :3, 3]
    for i, p in enumerate(rig.parent):
        if p >= 0:
            cv2.line(img, pr(W[p]), pr(W[i]), (240, 240, 240), 1, cv2.LINE_AA)
    for s in range(2):
        G = f.grasp[s]
        o = G[:3, 3]
        for j, c in ((0, (0, 0, 255)), (1, (0, 255, 0)), (2, (255, 0, 0))):
            cv2.line(img, pr(o), pr(o + G[:3, j]*0.08), c, 1, cv2.LINE_AA)
    cv2.circle(img, pr(f.gaze if np.linalg.norm(f.gaze - f.eyes[0]) < 1.0 else f.eyes[0]), 2, (0, 255, 255), -1)
    for e in f.eyes:
        d = f.gaze - e
        d = d/np.linalg.norm(d)
        cv2.line(img, pr(e), pr(e + d*0.15), (0, 220, 255), 1, cv2.LINE_AA)


def sheet(rig, names, out, cols=4, t=0.0):
    rows = (len(names) + cols - 1)//cols
    img = np.full((rows*CELL, cols*CELL*2 // 1, 3), 24, np.uint8)
    for n, name in enumerate(names):
        f = rig.solve(vec(POSES[name]), t=t)
        r, c = divmod(n, cols)
        x0, y0 = c*CELL*2, r*CELL
        scale = CELL/3.0
        draw(img, f, rig, x0 + CELL//2, y0 + CELL - 30, scale*1.0, 0)
        draw(img, f, rig, x0 + CELL + CELL//2, y0 + CELL - 30, scale*1.0, 1)
        cv2.putText(img, name, (x0 + 6, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(img, "front          side (camera ->)", (x0 + 6, y0 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (150, 150, 150), 1, cv2.LINE_AA)
        cv2.rectangle(img, (x0, y0), (x0 + 2*CELL - 1, y0 + CELL - 1), (70, 70, 70), 1)
    cv2.imwrite(out, img)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--poses", default=",".join(POSES))
    ap.add_argument("--cols", type=int, default=4)
    a = ap.parse_args()
    rig = JunoRig(a.asset)
    print(sheet(rig, a.poses.split(","), a.out, a.cols))


if __name__ == "__main__":
    main()
