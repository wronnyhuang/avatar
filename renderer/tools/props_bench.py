#!/usr/bin/env python3
"""props: K620 frame-time bench of a props scene, through tools/rig_bench.py with extra motions:
  props_home   every prop at home (the moving head/mouth of rig_bench 'rest')
  props_face   the red mug (slot 4) held in the right paw by the face, the hand moving a little
  props_desk   seated at the desk (turned 0.1: seated_desk34): chair pulled up, the muse in it (root + feet), laptop open,
               paws typing on its keys, the camera panned to it
  props_pull   hands on the chair's backrest (props_poses pull_chair), the chair sliding a little
  props_worn   headphones worn (slot 0, hh = 1), head moving
  objects_worn the straw hat + sunglasses worn (slots 8, 9), head moving
  objects_pour the watering can held in the right paw, tipped and pouring (slot 11, s = 1), gift box open
  flock gpu.lock python tools/props_bench.py --animal muse --motion props_face --encode
(AVATAR_NO_PROPS=1 = the same character with no props file: the baseline.)"""
import os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rig_bench  # noqa: E402
from anim.schema import IDX  # noqa: E402

_base = rig_bench.pose_track


def _poses(name, tool="props_poses.py"):
    out = subprocess.run([sys.executable, os.path.join(HERE, tool), name], capture_output=True, text=True).stdout.strip()
    return {} if out in ("", "REST") else {k: float(v) for k, v in (t.split("=") for t in out.split())}


def pose_track(n, motion, fps=30):
    if not motion.startswith(("props_", "objects_")):
        return _base(n, motion, fps)
    P = _base(n, "rest", fps)
    t = np.arange(n)/fps
    def s(name, v):
        if name in IDX:
            P[:, IDX[name]] = v
    if motion == "props_face":
        s("hand_r_x", 0.05 + 0.05*np.sin(t*1.3)); s("hand_r_y", 0.95 + 0.05*np.sin(t*0.9)); s("hand_r_z", 0.55); s("hand_r_turn", 0.5)
        s("prop4_hr", 1.0)
    elif motion == "props_desk":
        for k, v in _poses("seated_desk34").items():
            s(k, v)
        s("cam_x", 0.7*P[0, IDX["root_x"]])           # the camera follows the root, as the controller does
        hl, hr = P[:, IDX["hand_l_y"]].copy(), P[:, IDX["hand_r_y"]].copy()
        s("hand_l_y", hl + 0.04*np.maximum(np.sin(t*11.0), 0)); s("hand_r_y", hr + 0.04*np.maximum(np.sin(t*11.0 + 2.0), 0))
        s("head_yaw", 0.06*np.sin(t*0.9))
    elif motion == "props_pull":
        for k, v in _poses("pull_chair").items():
            s(k, v)
        x0 = P[:, IDX["prop1_x"]].copy()
        s("prop1_x", x0 - 0.03 + 0.03*np.sin(t*0.8))
    elif motion == "props_worn":
        s("prop0_hh", 1.0)
    elif motion in ("objects_worn", "objects_pour"):
        names = ["both_worn"] if motion == "objects_worn" else ["can_pour", "box_open"]
        for nm in names:
            for k, v in _poses(nm, "objects_poses.py").items():
                s(k, v)
        if motion == "objects_pour":
            s("hand_r_y", P[:, IDX["hand_r_y"]] + 0.03*np.sin(t*1.1))
    return P


rig_bench.pose_track = pose_track
if __name__ == "__main__":
    rig_bench.main()
