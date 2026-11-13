#!/usr/bin/env python3
"""Board visibility: score the stance probes (tools/board_stance_probe.cjs + board_vis_render.py masks).
left = visible share of the current line's text left of the pen (u in [pen-0.30, pen-0.03], v +-0.05 round the pen row)
near = visible share right next to the pen (u in [pen-0.08, pen-0.02])
board = visible share of the whole reach band"""
import json, sys, os
import numpy as np
d = sys.argv[1]
meta = json.load(open(os.path.join(d, "meta.json")))
M = np.load(os.path.join(d, "masks.npz")); m = M["masks"].astype(bool); gu, gv = M["grid"]
uu, vv = np.meshgrid((np.arange(gu) + .5)/gu, (np.arange(gv) + .5)/gv)
out = []
for i, q in enumerate(meta):
    if not q["ok"]:
        continue
    u, v = q["u"], q["v"] + 0.015
    row = (vv >= v - 0.05) & (vv <= v + 0.05)
    left = row & (uu >= u - 0.30) & (uu <= u - 0.03)
    near = row & (uu >= u - 0.08) & (uu <= u - 0.02)
    band = (vv >= 0.32) & (vv <= 0.95)
    f = q["forced"] or {}
    out.append(dict(u=q["u"], v=q["v"], yaw=q["yaw"], d=q["d"], lat=q["lat"], e=round(f.get("e", 9), 3), bend=round(f.get("bend", 0), 2),
                    left=round(float(m[i][left].mean()), 2), near=round(float(m[i][near].mean()), 2), board=round(float(m[i][band].mean()), 2)))
json.dump(out, open(os.path.join(d, "score.json"), "w"))
for r in out:
    print(r)
