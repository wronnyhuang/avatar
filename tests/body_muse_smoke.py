#!/usr/bin/env python3
"""body: smoke test of the muse character and the generic root transform / camera (CPU backend, ~1-3 min).

  python tests/body_muse_smoke.py [--set studio] [--keep DIR]

Renders the muse through the real-time engine (Mesa llvmpipe) and checks, from the character's pixel mask (frame minus
a character-free frame): it renders; root_x > 0 moves it screen-right and root_x < 0 left; root_y lifts it; root_yaw
0.5 changes the picture (its back) and 1.0 equals 0 (periodic); squash > 0 makes it wider and shorter; cam_zoom 1
shrinks it (wide shot) and cam_x pans; foot_l_y lifts the screen-left foot only; fingers_r changes only the screen-right
hand.  Exit 0 = all pass.
"""
import argparse, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "renderer")
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine")]
os.environ.setdefault("TA_OUT", os.path.join(ROOT, "..", "out", "cache"))
from anim.schema import IDX, NPARAMS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--keep")
ap.add_argument("--threads", type=int, default=12)
a = ap.parse_args()
import rt_engine  # noqa: E402

eng = rt_engine.RTEngine("muse", 320, 180, "q50", backend="cpu", threads=a.threads)
fails = 0


def check(name, ok, info=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info else ""), flush=True)
    fails += 0 if ok else 1


def vec(**kw):
    v = np.zeros(NPARAMS, "f4")
    for k, x in kw.items():
        v[IDX[k]] = x
    return v


def render(v, settle=4, nochar=False):
    eng.no_char = nochar
    eng.reset_history()
    for fb in eng.f_hist:            # tiles outside this pose's mask keep stale history: clear it for the coverage test
        fb.clear(0, 0, 0, 0)
    for f in range(settle):
        img = eng.render(200 + f, v)
    eng.no_char = False
    return img.astype(np.int16)


bg = {}


def mask(v, **kw):
    cam = (float(v[IDX["cam_zoom"]]), float(v[IDX["cam_x"]]), float(v[IDX["cam_y"]]))
    if cam not in bg:
        bv = np.zeros(NPARAMS, "f4"); bv[IDX["cam_zoom"]], bv[IDX["cam_x"]], bv[IDX["cam_y"]] = cam
        bg[cam] = render(bv, nochar=True)
    img = render(v, **kw)
    # the character's own coverage: the alpha of the TAA history layer of the last frame (premultiplied rgba, f16)
    k = (eng.nsub - 1) & 1
    al = np.frombuffer(eng.t_hc[k].read(), dtype="f2").reshape(eng.h, eng.w, 4)[::-1, :, 3].astype("f4")
    m = al > 0.5
    if a.keep:
        from PIL import Image
        os.makedirs(a.keep, exist_ok=True)
        tag = "_".join(f"{k}{float(v[IDX[k]]):g}" for k in ("root_x", "root_y", "root_yaw", "squash", "cam_zoom", "cam_x",
                                                         "foot_l_y", "fingers_r") if v[IDX[k]] != 0) or "rest"
        Image.fromarray(img.astype(np.uint8)).save(os.path.join(a.keep, f"muse_{tag}.png"))
    return m, img


def stats(m):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return dict(n=0, cx=0, cy=0, w=0, h=0)
    return dict(n=len(xs), cx=xs.mean(), cy=ys.mean(), w=xs.max() - xs.min(), h=ys.max() - ys.min(), top=ys.min(), bot=ys.max())


m0, i0 = mask(vec())
s0 = stats(m0)
check("rest renders the character", s0["n"] > 500, f"{s0['n']} px")
s = stats(mask(vec(root_x=0.5))[0]); check("root_x +0.5 moves screen-right", s["cx"] > s0["cx"] + 10, f"{s0['cx']:.0f} -> {s['cx']:.0f}")
s = stats(mask(vec(root_x=-0.5))[0]); check("root_x -0.5 moves screen-left", s["cx"] < s0["cx"] - 10, f"{s0['cx']:.0f} -> {s['cx']:.0f}")
s = stats(mask(vec(root_y=0.2))[0]); check("root_y +0.2 lifts it", s["bot"] < s0["bot"] - 5, f"bottom {s0['bot']} -> {s['bot']}")
_, ib = mask(vec(root_yaw=0.5))
nd = int((np.abs(ib - i0).max(axis=2) > 40).sum())
check("root_yaw 0.5 shows another side (no face)", nd > 60, f"{nd} px differ by > 40")
_, i1 = mask(vec(root_yaw=1.0))
check("root_yaw 1.0 == 0 (periodic)", np.abs(i1 - i0).max() <= 2, f"max |diff| {np.abs(i1 - i0).max()}")
s = stats(mask(vec(squash=0.5))[0]); check("squash +0.5: wider and shorter", s["w"] > s0["w"] and s["h"] < s0["h"],
                                         f"w {s0['w']}->{s['w']} h {s0['h']}->{s['h']}")
s = stats(mask(vec(cam_zoom=1.0))[0]); check("cam_zoom 1 = wide shot (smaller)", s["n"] < 0.5*s0["n"], f"{s0['n']} -> {s['n']} px")
sz = s
s = stats(mask(vec(cam_zoom=1.0, cam_x=0.6))[0]); check("cam_x +0.6 pans right (character moves left)", s["cx"] < sz["cx"] - 3,
                                                       f"{sz['cx']:.0f} -> {s['cx']:.0f}")
mf, _ = mask(vec(foot_l_y=0.08))
d = mf ^ m0
ys, xs = np.nonzero(d)
check("foot_l_y lifts the screen-left foot", len(xs) > 20 and xs.mean() < s0["cx"] and ys.mean() > s0["cy"], f"{len(xs)} px changed")
mh, ih = mask(vec(hand_r_y=0.85, hand_r_x=0.6, hand_r_z=0.5, hand_r_turn=1.0, fingers_r=1.0))
mh5, ih5 = mask(vec(hand_r_y=0.85, hand_r_x=0.6, hand_r_z=0.5, hand_r_turn=1.0, fingers_r=5.0))
dd = np.abs(ih5 - ih).max(axis=2) > 18
ys, xs = np.nonzero(dd)
check("fingers_r 1 -> 5 changes only the screen-right hand", len(xs) > 5 and xs.min() > s0["cx"], f"{len(xs)} px, x >= {xs.min() if len(xs) else -1}")
eng.close()
print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
