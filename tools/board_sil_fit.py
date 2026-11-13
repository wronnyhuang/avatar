#!/usr/bin/env python3
"""Board visibility: fit the muse's silhouette model (spheres + arm capsules in its body frame) to the REAL rendered
occlusion masks, so app/teach.js can predict what of the board its body hides from any stance.

    python tools/board_sil_fit.py DIR [DIR ...] [--out PARAMS.json]

DIR = a lesson recorded by tools/board_vis_sim.cjs (rows.jsonl with the paws' world points in "x") and rendered by
renderer/tools/board_vis_render.py (masks.npz; --every must match: read from masks/rows times).
Model (H units, body frame: x = its left->right as seen from the front at yaw 0, y up, z forward): spheres head,
chest, hips, feet; per arm a capsule shoulder -> paw (the paw's world point from the engine) with a paw sphere. Each
sphere projects from the camera onto the board plane as a disc scaled by the distance ratio. Loss = pixels where
model and render disagree (hidden vs visible), over the band rows the chalk reaches.
"""
import json, os, sys
import numpy as np

PROPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "renderer", "animals", "muse.props.json")
CAM = np.array([0.0, 0.5652, 3.3478])

P0 = dict(hx=0.0, hy=0.76, hz=0.0, hr=0.25, cy=0.45, cr=0.21, py=0.22, pr=0.19, fy=0.07, fx=0.1, fr=0.08,
          sx=0.17, sy=0.47, ar=0.06, pw=0.06, earx=0.17, eary=0.98, earr=0.06)
KEYS = list(P0)


def board():
    pl = {p["id"]: p for p in json.load(open(PROPS))["places"]}["board"]["plane"]
    O, U, V, N = (np.array(pl[k], float) for k in ("origin", "u", "v", "n"))
    W, Hh = pl["size"]
    return O, U, V, N, W, Hh


def rot_y(yaw_turns):
    a = 2*np.pi*yaw_turns
    c, s = np.cos(a), np.sin(a)
    # + yaw: the front (+z) swings to screen-right (+x)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def proj_disc(Pw, r, B):
    """sphere (world point, radius) -> (u, v, ru, rv) on the board's uv"""
    O, U, V, N, W, Hh = B
    d = Pw - CAM
    s = np.dot(O - CAM, N)/np.dot(d, N)
    X = CAM + d*s
    k = s                                   # distance ratio (board / sphere) along the ray
    u = np.dot(X - O, U)/W
    v = np.dot(X - O, V)/Hh
    dn = d/np.linalg.norm(d)
    cu = abs(np.dot(dn, N))                 # incidence: stretched along u by 1/cos
    return u, v, r*k/W/max(cu, 0.5), r*k/Hh


def model_mask(p, root, paws, bend, B, gu, gv, uu, vv):
    """bool (gv, gu) hidden"""
    x, z, yaw = root
    R = rot_y(yaw)
    a = 0.28*bend
    cb, sb = np.cos(a), np.sin(a)
    def world(loc, upper=True):
        l = np.array(loc, float)
        if upper and l[1] > 0.3:            # the waist bend tips the upper body forward (about y 0.3)
            yy, zz = l[1] - 0.3, l[2]
            l = np.array([l[0], 0.3 + yy*cb - zz*sb, yy*sb + zz*cb])
        return np.array([x, 0, z]) + R @ l
    sph = [((p["hx"], p["hy"], p["hz"]), p["hr"]), ((0, p["cy"], 0), p["cr"]), ((0, p["py"], 0), p["pr"]),
           ((p["fx"], p["fy"], 0), p["fr"]), ((-p["fx"], p["fy"], 0), p["fr"]),
           ((p["earx"], p["eary"], 0), p["earr"]), ((-p["earx"], p["eary"], 0), p["earr"])]
    hid = np.zeros(uu.shape, bool)
    for loc, r in sph:
        u, v, ru, rv = proj_disc(world(loc), r, B)
        hid |= ((uu - u)/ru)**2 + ((vv - v)/rv)**2 <= 1
    for side, sxs in (("l", -1), ("r", 1)):
        S = world((sxs*p["sx"], p["sy"], 0))
        Pp = np.array(paws[side], float)
        for f in np.linspace(0, 1, 6):
            Q = S + (Pp - S)*f
            r = p["ar"] + (p["pw"] - p["ar"])*f
            u, v, ru, rv = proj_disc(Q, r, B)
            hid |= ((uu - u)/ru)**2 + ((vv - v)/rv)**2 <= 1
    return hid


def load(dirs):
    data = []
    for d in dirs:
        M = np.load(os.path.join(d, "masks.npz"))
        rows = [json.loads(l) for l in open(os.path.join(d, "rows.jsonl")) if l.strip()]
        byt = {round(r["t"], 3): r for r in rows}
        for m, t in zip(M["masks"], M["t"]):
            r = byt.get(round(float(t), 3))
            if r is None or not r.get("x"):
                continue
            pp = r["p"]
            data.append(dict(root=(pp.get("root_x", 0), pp.get("root_z", 0), pp.get("root_yaw", 0)),
                             paws={"l": r["x"]["pl"], "r": r["x"]["pr"]}, bend=r["x"]["bp"], hid=~m.astype(bool)))
    return data


def loss(p, data, B, uu, vv, band):
    e = 0
    for f in data:
        h = model_mask(p, f["root"], f["paws"], f["bend"], B, uu.shape[1], uu.shape[0], uu, vv)
        e += np.sum((h != f["hid"]) & band)
    return e/len(data)


def main():
    dirs = [a for a in sys.argv[1:] if not a.startswith("--") and os.path.isdir(a)]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    data = load(dirs)
    rng = np.random.default_rng(0)
    if len(data) > 260:
        data = [data[i] for i in sorted(rng.choice(len(data), 260, replace=False))]
    B = board()
    gu, gv = data[0]["hid"].shape[1], data[0]["hid"].shape[0]
    uu, vv = np.meshgrid((np.arange(gu) + 0.5)/gu, (np.arange(gv) + 0.5)/gv)
    band = (vv >= 0.3) & (vv <= 0.97)
    p = dict(P0)
    if "--init" in sys.argv:
        p.update(json.load(open(sys.argv[sys.argv.index("--init") + 1])))
    best = loss(p, data, B, uu, vv, band)
    print("init", round(best, 1), "px/frame of", int(band.sum()), "frames", len(data), flush=True)
    step = {k: 0.04 for k in KEYS}
    for it in range(int(os.environ.get("FIT_IT", 6))):
        for k in KEYS:
            for sg in (1, -1):
                while True:
                    q = dict(p); q[k] = p[k] + sg*step[k]
                    if k.endswith("r") and q[k] < 0.01:
                        break
                    e = loss(q, data, B, uu, vv, band)
                    if e < best - 1e-6:
                        p, best = q, e
                    else:
                        break
        step = {k: v*0.5 for k, v in step.items()}
        print("it", it, round(best, 1), json.dumps({k: round(v, 3) for k, v in p.items()}), flush=True)
    # accuracy summary: per-frame IoU and the fraction of truly hidden band pixels the model misses
    ious, miss, fa = [], 0, 0
    tot_h, tot_v = 0, 0
    for f in data:
        h = model_mask(p, f["root"], f["paws"], f["bend"], B, gu, gv, uu, vv) & band
        g = f["hid"] & band
        ious.append((h & g).sum()/max(1, (h | g).sum()))
        miss += (g & ~h).sum(); fa += (h & ~g).sum(); tot_h += g.sum(); tot_v += (~g & band).sum()
    res = {"params": {k: round(v, 4) for k, v in p.items()}, "iou_mean": round(float(np.mean(ious)), 3),
           "miss_of_hidden": round(float(miss/max(1, tot_h)), 3), "false_hidden_of_visible": round(float(fa/max(1, tot_v)), 3), "frames": len(data)}
    print(json.dumps(res))
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
