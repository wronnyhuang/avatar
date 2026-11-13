"""objective checks for the judge's on-screen complaints on scorecard canon traces (eval/runs/<run>/<clip>/canon.npz)
vs the real partner (eval/runs/human): eyes half-closed / asymmetric (blink, squint), head roll, arm pose excursions.
    python -I tools/train/rt_artifacts.py <run> [<run> ...]      (run = e.g. unified-p2rt_ev1-h2m2, human)
"""
import json
import sys

import numpy as np
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from canon1 import spec  # noqa: E402

from canon1 import paths as _paths  # noqa: E402

E = os.environ.get("EVAL_OUT", str(_paths.WORK / "eval"))
C = spec.COL
B = lambda n: 135 + spec.BS[n]       # noqa: E731


def stats(m):
    bl = m[:, [B("eyeBlinkLeft"), B("eyeBlinkRight")]]
    sq = m[:, [B("eyeSquintLeft"), B("eyeSquintRight")]]
    roll = m[:, C["neck_z"]] + m[:, C["head_z"]]
    arm = m[:, spec.GROUPS["arm_l"] + spec.GROUPS["arm_r"]]
    dev = np.abs(arm - np.median(arm, 0)).max(1)
    blm = bl.mean(1)
    # half-closed = 0.3..0.7 held: fraction of frames inside runs >= 10 frames
    half = (blm > 0.3) & (blm < 0.8)
    run_frac = 0.0
    k = 0
    for i in range(len(half) + 1):
        if i < len(half) and half[i]:
            k += 1
        else:
            if k >= 10:
                run_frac += k
            k = 0
    return dict(blink_mean=float(blm.mean()), closed_frac=float((blm > 0.5).mean()), half_closed_held=float(run_frac / len(m)),
                blink_asym=float(np.abs(bl[:, 0] - bl[:, 1]).mean()), squint_mean=float(sq.mean()),
                roll_sd_deg=float(np.degrees(roll.std())), arm_dev_p95=float(np.percentile(dev, 95)),
                arm_dev_gt05=float((dev > 0.5).mean()))


def main():
    man = [c for c in json.load(open(f"{E}/clips/manifest.json"))["clips"] if c["group"] == "seamless"]
    for run in sys.argv[1:]:
        rows = []
        for c in man:
            f = f"{E}/runs/{run}/{c['id']}/canon.npz"
            if os.path.exists(f):
                rows.append(stats(np.load(f)["motion"].astype(np.float64)))
        if not rows:
            print(run, "no traces"); continue
        print(f"{run:<36}" + " ".join(f"{k} {np.mean([r[k] for r in rows]):.3f}" for k in rows[0]) + f" (n={len(rows)})")


if __name__ == "__main__":
    main()
