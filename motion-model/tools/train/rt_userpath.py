"""does the scorecard harness feed the model the user the model was trained on?
For each Seamless TEST clip: the harness's user stream (evalkit user_fit.npz = the live Fitter on the clip's camera
MediaPipe detections, audio42_user.npy from mic.wav) vs the OFFLINE canon of the same person and frames (the
Seamless canon npz = what the packs are built from; face there = mapper mlp_v1, packs v3 = calibrated to MP).
Prints per-signal correlation, SD ratio (harness / offline), mean shift in offline SDs, mask presence.
    python -I tools/train/rt_userpath.py [--clips N]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from canon1 import spec  # noqa: E402
from canon1 import paths as _paths  # noqa: E402

E = os.environ.get("EVAL_OUT", str(_paths.WORK / "eval"))
D = str(_paths.DATA / "canon1" / "seamless")
C = spec.COL
BS = lambda n: 135 + spec.BS[n]      # noqa: E731
SIG = {"head_pitch": [C["neck_x"], C["head_x"]], "head_yaw": [C["neck_y"], C["head_y"]], "head_roll": [C["neck_z"], C["head_z"]],
       "torso_x": [C["spine1_x"], C["spine2_x"], C["spine3_x"]], "gaze_yaw": [C["gaze_yaw"]], "gaze_pitch": [C["gaze_pitch"]],
       "smile": [BS("mouthSmileLeft"), BS("mouthSmileRight")], "blink": [BS("eyeBlinkLeft"), BS("eyeBlinkRight")],
       "jaw": [BS("jawOpen")], "brow_up": [BS("browInnerUp")], "brow_down": [BS("browDownLeft"), BS("browDownRight")],
       "shoulder_r_z": [C["shoulder_r_z"]], "elbow_r_y": [C["elbow_r_y"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=29)
    a = ap.parse_args()
    man = [c for c in json.load(open(f"{E}/clips/manifest.json"))["clips"] if c["group"] == "seamless"][: a.clips]
    rows = {k: [] for k in SIG}
    mask_h, mask_o, aud = [], [], []
    for c in man:
        f = f"{D}/{c['session']}.npz"
        uf = f"{E}/clips/{c['id']}/user_fit.npz"
        if not (os.path.exists(f) and os.path.exists(uf)):
            continue
        d = spec.load_session(f)
        persons = list(d["meta"]["persons"])
        pi = persons.index(c["user_person"]) if c["user_person"] in persons else persons.index(c["user_person"].split(":")[-1])
        z = np.load(uf)
        hm, hk = z["motion"].astype(np.float64), z["mask"].astype(np.float64)
        N = len(hm)
        f0 = int(c["frame0"])
        om, ok_ = d["motion"][pi, f0:f0 + N].astype(np.float64), d["mask"][pi, f0:f0 + N].astype(np.float64)
        n = min(N, len(om))
        hm, hk, om, ok_ = hm[:n], hk[:n], om[:n], ok_[:n]
        mask_h.append(hk.mean(0)); mask_o.append(ok_.mean(0))
        both = (hk[:, spec.GROUP_NAMES.index("head")] > 0) & (ok_[:, spec.GROUP_NAMES.index("head")] > 0) & \
               (hk[:, spec.GROUP_NAMES.index("face_upper")] > 0) & (ok_[:, spec.GROUP_NAMES.index("face_upper")] > 0)
        if both.sum() < 100:
            continue
        for k, cols in SIG.items():
            x, y = hm[both][:, cols].sum(1), om[both][:, cols].sum(1)
            sd_o = y.std() + 1e-9
            r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-9 and y.std() > 1e-9 else np.nan
            # best lag in +-10 frames (the live path may lag)
            best = (r, 0)
            for lag in range(-10, 11):
                if lag == 0:
                    continue
                xa, ya = (x[lag:], y[:-lag]) if lag > 0 else (x[:lag], y[-lag:])
                if xa.std() > 1e-9 and ya.std() > 1e-9:
                    rr = float(np.corrcoef(xa, ya)[0, 1])
                    if np.isfinite(rr) and (not np.isfinite(best[0]) or rr > best[0]):
                        best = (rr, lag)
            rows[k].append((r, x.std() / sd_o, (x.mean() - y.mean()) / sd_o, best[0], best[1]))
        ua = np.load(f"{E}/clips/{c['id']}/audio42_user.npy")[:n]
        oa = d["audio"][pi, f0:f0 + n]
        aud.append((float(np.corrcoef(ua[:, 40], oa[:, 40])[0, 1]), float((ua[:, 41] > .5).mean()), float((oa[:, 41] > .5).mean()),
                    float(np.mean((ua[:, 41] > .5) == (oa[:, 41] > .5))), float(np.mean(ua[:, 40] - oa[:, 40]))))
    print(f"{len(aud)} clips; harness user (live Fitter on MediaPipe) vs offline canon of the same person/frames")
    print(f"{'signal':<14}{'corr':>7}{'best':>7}{'lag':>5}{'sd ratio':>10}{'mean shift (sd)':>17}")
    for k, v in rows.items():
        v = np.array(v, np.float64)
        print(f"{k:<14}{np.nanmedian(v[:, 0]):7.2f}{np.nanmedian(v[:, 3]):7.2f}{np.nanmedian(v[:, 4]):5.0f}"
              f"{np.nanmedian(v[:, 1]):10.2f}{np.nanmedian(v[:, 2]):17.2f}")
    mh, mo = np.mean(mask_h, 0), np.mean(mask_o, 0)
    print("mask presence harness / offline:", {g: f"{mh[i]:.2f}/{mo[i]:.2f}" for i, g in enumerate(spec.GROUP_NAMES)})
    A = np.array(aud)
    print(f"audio: loud_db corr {np.median(A[:, 0]):.2f}, vad frac harness {A[:, 1].mean():.2f} vs offline {A[:, 2].mean():.2f}, "
          f"vad agreement {A[:, 3].mean():.2f}, loud_db shift {np.median(A[:, 4]):+.1f} dB")


if __name__ == "__main__":
    main()
