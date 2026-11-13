"""Population reference statistics for the retarget, from Seamless SMPL-H (native CANON v1 body/fingers).

    python -I -m retarget.tools.pop_stats --n 240 --out retarget/maps/_pop_stats.json

Reads train-split SMPL-H only (person-disjoint from test by construction of the dataset's own splits), builds the canon
body columns (root relative to the session median, body joints, fingers), and stores:
  neutral: per-joint chordal-median local rotation (aa) -> the retarget's NEUTRAL pose
  wrist_ref: median wrist position (chest frame, rel. rest shoulder, metres) per side, + percentiles of the offsets
  chest/head deltas (yaw/pitch/roll percentiles), finger curl stats (per finger, per hand), collar elevation stats
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from canon1 import spec
from retarget import skeleton as sk

from canon1 import paths as _paths  # noqa: E402

ROOT = str(_paths.SEAMLESS)          # raw Seamless Interaction download ($SEAMLESS_DIR)


def load_file(fid_path):
    d = os.path.dirname(os.path.dirname(fid_path))
    name = os.path.basename(fid_path)
    g = lambda k: np.load(os.path.join(d, k, name), allow_pickle=False)  # noqa: E731
    bp, go = g("body_pose").astype(np.float64), g("global_orient").astype(np.float64)
    lh, rh = g("left_hand_pose").astype(np.float64), g("right_hand_pose").astype(np.float64)
    v = g("is_valid").astype(bool).ravel()
    T = len(bp)
    m = np.zeros((T, spec.D_MOTION))
    m[:, 0:3] = spec.relative_root(go, v)
    for j in spec.JOINTS[1:]:
        m[:, spec.jcols(j)] = bp[:, spec.SMPLH_JOINT_ID[j] - 1]
    m[:, spec.SL_FINGERS_L] = lh.reshape(T, 45)
    m[:, spec.SL_FINGERS_R] = rh.reshape(T, 45)
    return m[v]


def chordal_median_aa(aa):
    R = spec.aa_to_mat(aa)
    M = np.median(R, 0)
    U, _, Vt = np.linalg.svd(M)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt
    return spec.mat_to_aa(Rm[None])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(f"{ROOT}/*/train/smplh/body_pose/*.npy"))
    rs = np.random.default_rng(0)
    files = [files[i] for i in rs.choice(len(files), min(a.n, len(files)), replace=False)]
    ms = []
    for f in files:
        try:
            m = load_file(f)
        except Exception as e:  # noqa: BLE001
            print("skip", f, e)
            continue
        if len(m):
            ms.append(m[:: a.stride])
    M = np.concatenate(ms)
    print("frames", len(M), "files", len(ms))
    out = {"source": f"seamless train smplh, {len(ms)} files, every {a.stride}th valid frame", "frames": int(len(M))}
    neutral = np.zeros(spec.D_MOTION)
    for j in spec.JOINTS[1:]:
        neutral[spec.jcols(j)] = chordal_median_aa(M[:, spec.jcols(j)])
    out["neutral_body"] = {j: np.round(neutral[spec.jcols(j)], 4).tolist() for j in spec.JOINTS[1:]}
    # fingers: median per column (small angles about the MANO mean)
    out["neutral_fingers"] = np.round(np.median(M[:, 45:135], 0), 4).tolist()
    # chest / head deltas against the neutral
    fk = sk.upper_fk(M)
    fk0 = sk.upper_fk(neutral[None])
    Rc = fk["spine3"][0] @ np.swapaxes(fk0["spine3"][0], -1, -2)
    y, p, r = sk.euler_yxz(Rc)
    pc = lambda x: np.round(np.percentile(x, [1, 5, 25, 50, 75, 95, 99]), 4).tolist()  # noqa: E731
    out["chest_delta_ypr_pct"] = {"yaw": pc(y), "pitch": pc(p), "roll": pc(r)}
    Rh = np.swapaxes(fk["spine3"][0], -1, -2) @ fk["head"][0]
    Rh0 = np.swapaxes(fk0["spine3"][0], -1, -2) @ fk0["head"][0]
    y, p, r = sk.euler_yxz(Rh @ np.swapaxes(Rh0, -1, -2))
    out["head_delta_ypr_pct"] = {"yaw": pc(y), "pitch": pc(p), "roll": pc(r)}
    arms = sk.chest_frame_arm(M)
    out["wrist"] = {}
    for s in "lr":
        w = arms[s]["wrist"]
        med = np.median(w, 0)
        d = (w - med) / sk.ARM
        out["wrist"][s] = {"median": np.round(med, 4).tolist(), "pct_x": pc(d[:, 0]), "pct_y": pc(d[:, 1]),
                           "pct_z": pc(d[:, 2]),
                           "elbow_median": np.round(np.median(arms[s]["elbow"], 0), 4).tolist()}
        # collar elevation: the shoulder joint's height change (m)
        sh = arms[s]["shoulder"][:, 1]
        out["wrist"][s]["shoulder_dy_pct"] = pc(sh - np.median(sh))
    # finger curl: flexion = -z for the left hand fingers, +z for the right (spec frame), 3 joints summed
    fc = {}
    for hand, sl, sg in (("l", 45, -1.0), ("r", 90, 1.0)):
        for k, fn in enumerate(["index", "middle", "pinky", "ring", "thumb"]):
            cols = [sl + 9 * k + 3 * q + 2 for q in range(3)]
            c = sg * M[:, cols].sum(1)
            fc[f"{hand}_{fn}"] = pc(c)
    out["finger_curl_pct"] = fc
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("neutral_fingers",)}, indent=1)[:6000])


if __name__ == "__main__":
    main()
