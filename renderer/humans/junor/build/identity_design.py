"""junor — design the face IN the ICT identity space (no sculpting): measurable targets on Multi-PIE landmarks
(all linear in the vertex positions, hence in the PCA coefficients) solved as a ridge regression around a picked
start identity, so the result stays a plausible scan-space face.

usage: python identity_design.py <ict_npz> <start.npy> <out.npy> [--lam 0.6]
Targets (v2, lead feedback 01:50): bigger palpebral opening, shorter philtrum, fuller upper lip, smaller nose tip /
narrower alae, fuller cheeks, softer narrower jaw, slightly shorter chin.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ict_head import ICTHead  # noqa: E402

LM = [1225, 1888, 1052, 367, 1719, 1722, 2199, 1447, 966, 3661, 4390, 3927, 3924, 2608, 3272, 4088, 3443, 268, 493, 1914,
      2044, 1401, 3615, 4240, 4114, 2734, 2509, 978, 4527, 4942, 4857, 1140, 2075, 1147, 4269, 3360, 1507, 1542, 1537, 1528,
      1518, 1511, 3742, 3751, 3756, 3721, 3725, 3732, 5708, 5695, 2081, 0, 4275, 6200, 6213, 6346, 6461, 5518, 5957, 5841,
      5702, 5711, 5533, 6216, 6207, 6470, 5517, 5966]

# (name, terms [(lm index or 'v<id>', axis, coef)], mode, value, weight): mode 'mul' = x start value, 'add' = + metres
TARGETS = [
    ("philtrum", [(33, 1, 1), (51, 1, -1)], "mul", 0.86, 3.0),
    ("upper_lip_h", [(51, 1, 1), (62, 1, -1)], "mul", 1.30, 3.0),
    ("lower_lip_h", [(66, 1, 1), (57, 1, -1)], "mul", 1.05, 1.0),
    ("alar_w", [(35, 0, 1), (31, 0, -1)], "mul", 0.92, 2.0),
    ("tip_proj", [(30, 2, 1), (33, 2, -1)], "mul", 0.92, 1.0),
    ("eye_open_r", [(37, 1, .5), (38, 1, .5), (41, 1, -.5), (40, 1, -.5)], "mul", 1.15, 4.0),
    ("eye_open_l", [(43, 1, .5), (44, 1, .5), (47, 1, -.5), (46, 1, -.5)], "mul", 1.15, 4.0),
    ("eye_w_r", [(39, 0, 1), (36, 0, -1)], "mul", 1.04, 1.0),
    ("eye_w_l", [(45, 0, 1), (42, 0, -1)], "mul", 1.04, 1.0),
    ("jaw_w", [(12, 0, 1), (4, 0, -1)], "mul", 0.95, 2.0),
    ("lowface_w", [(10, 0, 1), (6, 0, -1)], "mul", 0.94, 2.0),
    ("chin_h", [(57, 1, 1), (8, 1, -1)], "mul", 0.95, 1.0),
    ("face_h", [(27, 1, 1), (8, 1, -1)], "mul", 0.98, 1.0),
    ("mouth_w", [(54, 0, 1), (48, 0, -1)], "mul", 1.0, 1.0),
    ("cheek_l", [("cheek_l", 2, 1)], "add", 0.0025, 3.0),
    ("cheek_r", [("cheek_r", 2, 1)], "add", 0.0025, 3.0),
    ("cheek_wl", [("cheek_l", 0, 1)], "add", 0.0012, 1.0),
    ("cheek_wr", [("cheek_r", 0, 1)], "add", -0.0012, 1.0),
]


def _cheek_ids(H):
    P = H.Vcm
    face = H.region("face")
    out = {}
    for nm, sx in (("cheek_l", 1), ("cheek_r", -1)):
        t = np.array([sx * 3.6, P[LM[33], 1] + 1.2, 0.0])
        cand = face[(np.abs(P[face, 0] - t[0]) < 0.6) & (np.abs(P[face, 1] - t[1]) < 0.6)]
        out[nm] = int(cand[np.argmax(P[cand, 2])])
    return out


def design(npz, c0, lam=0.6, targets=TARGETS):
    H = ICTHead(npz, None)
    I = np.load(npz)["ident"].astype(np.float64) * 0.01           # (K,N,3) metres
    V0 = H.Vcm * 0.01
    ids = _cheek_ids(H)
    rows, tv, wv, names = [], [], [], []
    base = V0 + np.tensordot(c0, I, 1)
    for name, terms, mode, val, w in targets:
        a = np.zeros(I.shape[0])
        m0 = 0.0
        for li, ax, k in terms:
            vi = ids[li] if isinstance(li, str) else LM[li]
            a += k * I[:, vi, ax]
            m0 += k * V0[vi, ax]
        cur = m0 + a @ c0
        tgt = cur * val if mode == "mul" else cur + val
        rows.append(a)
        tv.append(tgt - m0)
        wv.append(w / max(abs(cur), 1e-3) if mode == "mul" else w / 0.003)
        names.append((name, cur))
    A = np.array(rows) * np.array(wv)[:, None]
    b = np.array(tv) * np.array(wv)
    K = I.shape[0]
    c = np.linalg.solve(A.T @ A + lam * np.eye(K), A.T @ b + lam * c0)
    rep = {}
    for (name, cur), row, t, w in zip(names, rows, tv, wv):
        m0 = t - (row @ c0) + (row @ c0)
        rep[name] = {"start_mm": round(cur * 1000, 2), "now_mm": round((cur + row @ (c - c0)) * 1000, 2)}
    rep["|c0|"], rep["|c|"] = round(float(np.linalg.norm(c0)), 2), round(float(np.linalg.norm(c)), 2)
    return c, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("start")
    ap.add_argument("out")
    ap.add_argument("--lam", type=float, default=0.6)
    a = ap.parse_args()
    c, rep = design(a.npz, np.load(a.start), a.lam)
    np.save(a.out, c)
    print(json.dumps(rep, indent=0))


if __name__ == "__main__":
    main()
