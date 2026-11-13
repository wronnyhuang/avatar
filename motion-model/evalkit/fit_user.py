#!/usr/bin/env python3
"""Run the live Fitter (canon1/fit.py) on one clip's mp_live.npz.
Separate process, so the eval interpreter never has to import the fitter's dependencies.

  python -I evalkit/fit_user.py <inputs_dir>        -> <inputs_dir>/user_fit.npz {motion, mask, fit_ms, fitter}
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DATA_WT = Path(__file__).resolve().parents[1]      # the motion-model dir (canon1/fit.py)
sys.path.insert(0, str(DATA_WT))
from canon1.fit import Fitter  # noqa: E402


def frames(mp):
    n = len(mp["face"])
    wh = tuple(int(x) for x in mp["wh"])
    for i in range(n):
        hands = []
        for j in range(int(mp["hand_n"][i])):
            lab = int(mp["hand_label"][i][j])
            hands.append({"handedness": "Left" if lab == 0 else "Right", "world": mp["hands_world"][i][j],
                          "image": mp["hands"][i][j], "score": float(mp["hand_score"][i][j])})
        yield {"face_present": bool(mp["face"][i]), "blendshapes": mp["bs"][i], "face_matrix": mp["mat"][i],
               "pose_world": mp["pose_world"][i] if mp["pose_ok"][i] else None,
               "pose_image": mp["pose"][i] if mp["pose_ok"][i] else None,
               "hands": hands, "image_wh": wh}


def main():
    d = Path(sys.argv[1])
    z = np.load(d / "mp_live.npz")
    mp = {k: z[k] for k in z.files}
    f = Fitter()
    mo, ma, ms = [], [], []
    for fr in frames(mp):
        t0 = time.perf_counter()
        m189, m9 = f.step(fr)
        ms.append((time.perf_counter() - t0) * 1000)
        mo.append(np.asarray(m189, np.float32)); ma.append(np.asarray(m9, np.uint8))
    rev = subprocess.run(["git", "-C", str(DATA_WT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(DATA_WT), "status", "--porcelain", "canon1/fit.py"], capture_output=True,
                                text=True).stdout.strip())
    info = {"data_rev": rev, "fit_py_dirty": dirty,
            "fit_py_sha": hashlib.sha256((DATA_WT / "canon1/fit.py").read_bytes()).hexdigest()[:12],
            "params": str(getattr(f, "params_path", "")), "n": len(mo)}
    np.savez_compressed(d / "user_fit.npz", motion=np.stack(mo), mask=np.stack(ma), fit_ms=np.array(ms, np.float32),
                        fitter=np.array(json.dumps(info)))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
