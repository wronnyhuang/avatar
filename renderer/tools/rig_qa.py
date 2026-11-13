#!/usr/bin/env python3
"""Rig QA through the real-time engine: every contract key pose, and every rig param at its extremes, with a pixel-diff
table proving each one moves (and moves on the right SIDE for _l/_r params).  CPU (llvmpipe) by default, no GPU lock.

  python tools/rig_qa.py --character fox --out DIR [--threads 12] [--w 480 --h 270]

Writes DIR/<char>_keys.png (key poses), DIR/<char>_params.png (each case: render + heat map of |case - base|),
DIR/<char>_qa.json (per case: changed pixels > 12/255, bbox, centroid relative to the character's head centre, the side
verdict), and prints a one-line summary.  Same frame indices for every render: only the params differ.

Side convention (docs/CONTRACT.md): _l = SCREEN LEFT.  A _l case must move pixels left of the head centre (or the
centroid of its change must sit left of the _r case's), and vice versa.
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine")]
from anim.schema import IDX, NPARAMS  # noqa: E402
from anim.rig_poses import KEY_POSES, REST, hands  # noqa: E402

THR = 12

# (name, base pose dict, override dict, expected region, side)
def cases():
    up_l = hands(l=(0.15, 0.60, 0.35, 0.6, 0.0))
    up_r = hands(r=(0.15, 0.60, 0.35, 0.6, 0.0))
    C = []
    add = lambda name, over, region, side=None, base=None: C.append((name, base or REST, over, region, side))
    for s, nm in (("l", "L"), ("r", "R")):
        add(f"blink_{s}", {f"blink_{s}": 1.0}, "eye", s)
        add(f"brow_{s}+", {f"brow_{s}": 1.0}, "brow", s)
        add(f"brow_{s}-", {f"brow_{s}": -1.0}, "brow", s)
        add(f"ear_{s}+", {f"ear_{s}": 1.0}, "ear", s)
        add(f"ear_{s}-", {f"ear_{s}": -1.0}, "ear", s)
        add(f"ear_flick_{s}", {f"ear_flick_{s}": 1.0}, "ear", s)
        add(f"ear_{s}_turn+", {f"ear_{s}_turn": 1.0}, "ear", s)
        add(f"ear_{s}_turn-", {f"ear_{s}_turn": -1.0}, "ear", s)
        add(f"hand_{s}_x+", {f"hand_{s}_x": 0.6}, "hand", s)
        add(f"hand_{s}_x-", {f"hand_{s}_x": -0.4, f"hand_{s}_y": 0.4, f"hand_{s}_z": 0.3}, "hand", s)
        add(f"hand_{s}_y+", {f"hand_{s}_y": 1.0}, "hand", s)
        add(f"hand_{s}_z+", {f"hand_{s}_z": 0.8, f"hand_{s}_y": 0.3}, "hand", s)
        add(f"hand_{s}_open", {f"hand_{s}_open": 1.0}, "hand", s, base=up_l if s == "l" else up_r)
        add(f"hand_{s}_turn+", {f"hand_{s}_turn": 1.0}, "hand", s, base=up_l if s == "l" else up_r)
        add(f"hand_{s}_turn-", {f"hand_{s}_turn": -1.0}, "hand", s, base=up_l if s == "l" else up_r)
        add(f"thumb_{s}", {f"thumb_{s}": 1.0, f"hand_{s}_open": 0.0}, "hand", s, base=up_l if s == "l" else up_r)
        add(f"point_{s}", {f"point_{s}": 1.0}, "hand", s, base=up_l if s == "l" else up_r)
        add(f"shoulder_{s}", {f"shoulder_{s}": 1.0}, "shoulder", s)
    for n, v, reg in (("gaze_x", 1.0, "eye"), ("gaze_y", 1.0, "eye"), ("squint", 1.0, "eye"), ("eye_wide", 1.0, "eye"),
                      ("brow_knit", 1.0, "brow"), ("nose_wiggle", 1.0, "nose"), ("nose_wiggle", -1.0, "nose"),
                      ("nostril", 1.0, "nose"), ("smile", 1.0, "mouth"), ("mouth_open", 1.0, "mouth"),
                      ("tongue_out", 1.0, "mouth"), ("head_yaw", 0.4, "head"), ("head_pitch", 0.3, "head"),
                      ("head_roll", 0.3, "head"), ("breath", 1.0, "body"), ("body_yaw", 1.0, "body"),
                      ("body_yaw", -1.0, "body"), ("body_lean", 1.0, "body"), ("body_lean", -1.0, "body"),
                      ("body_pitch", 1.0, "body"), ("body_pitch", -1.0, "body"), ("body_bounce", 1.0, "body"),
                      ("body_bounce", -1.0, "body"), ("tail_wag", 1.0, "tail"), ("tail_wag", -1.0, "tail"),
                      ("tail_raise", 1.0, "tail"), ("tail_curl", 1.0, "tail")):
        add(f"{n}{'+' if v > 0 else '-'}", {n: v}, reg)
    return C


def vec(d):
    v = np.zeros(NPARAMS, "f4")
    for k, x in d.items():
        v[IDX[k]] = x
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="fox")
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=int, default=480)
    ap.add_argument("--h", type=int, default=270)
    ap.add_argument("--preset", default="auto")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--settle", type=int, default=4)
    ap.add_argument("--no-keys", action="store_true")
    a = ap.parse_args()
    import rt_engine
    from PIL import Image, ImageDraw
    preset = a.preset
    if preset == "auto":
        from live.backends.rt import default_preset
        preset = default_preset(a.character, a.w, a.h, 24)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    eng = rt_engine.RTEngine(a.character, a.w, a.h, preset, backend=a.backend, threads=a.threads)

    def render(d, no_char=False):
        eng.no_char = no_char
        eng.reset_history()
        v = vec(d)
        for f in range(a.settle):
            img = eng.render(200 + f, v)
        eng.no_char = False
        return img

    base = render(REST)
    bg = render(REST, no_char=True)
    m = np.abs(base.astype(int) - bg.astype(int)).max(2) > 20
    ys, xs = np.nonzero(m)
    top = ys.min() if ys.size else 0
    hh = (ys.max() - top) if ys.size else a.h
    head = m & (np.arange(a.h)[:, None] < top + 0.45*hh)
    hy, hx = np.nonzero(head)
    cx = float(np.median(hx)) if hx.size else a.w/2
    res = {"character": a.character, "preset": preset, "w": a.w, "h": a.h, "head_cx": round(cx, 1),
           "char_bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if xs.size else None, "cases": {}}
    tiles = []
    for name, b, over, region, side in cases():
        d = dict(b); d.update(over)
        img = render(d)
        ref = base if b is REST else render(b)
        diff = np.abs(img.astype(int) - ref.astype(int)).max(2)
        mk = diff > THR
        yy, xx = np.nonzero(mk)
        r = {"region": region, "changed_px": int(mk.sum()), "mean_abs": round(float(diff.mean()), 3)}
        if xx.size:
            w8 = diff[mk].astype(float)
            mx = float((xx*w8).sum()/w8.sum())
            r.update(bbox=[int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())], cx_rel=round(mx - cx, 1),
                     cy=round(float((yy*w8).sum()/w8.sum()), 1))
        if side:
            r["side"] = side
        res["cases"][name] = r
        tiles.append((name, img, diff))
    # side verdicts: _l vs _r pairs
    for name, r in res["cases"].items():
        s = r.get("side")
        if not s or r["changed_px"] == 0:
            r["verdict"] = "NO CHANGE" if r["changed_px"] < 30 else "ok"
            continue
        other = name.replace(f"_{s}", "_r" if s == "l" else "_l", 1) if f"_{s}" in name else None
        o = res["cases"].get(other) if other else None
        if r["changed_px"] < 30:
            r["verdict"] = "NO CHANGE"
        elif o and o.get("cx_rel") is not None and r.get("cx_rel") is not None:
            ok = (r["cx_rel"] < o["cx_rel"]) if s == "l" else (r["cx_rel"] > o["cx_rel"])
            r["verdict"] = "ok" if ok else "SIDE?"
        else:
            r["verdict"] = "ok" if ((r.get("cx_rel", 0) < 0) == (s == "l")) else "SIDE?"
    for r in res["cases"].values():
        r.setdefault("verdict", "ok" if r["changed_px"] >= 30 else "NO CHANGE")
    json.dump(res, open(os.path.join(a.out, f"{a.character}_qa.json"), "w"), indent=1)
    # params sheet: render | heat map
    tw, th = a.w//2, a.h//2
    cols = 4
    rows = (len(tiles) + cols - 1)//cols
    sheet = Image.new("RGB", (cols*tw*2, rows*(th + 14)), (16, 16, 16))
    dr = ImageDraw.Draw(sheet)
    for i, (name, img, diff) in enumerate(tiles):
        x, y = (i % cols)*tw*2, (i//cols)*(th + 14)
        sheet.paste(Image.fromarray(img).resize((tw, th)), (x, y + 14))
        hm = np.clip(diff*4, 0, 255).astype(np.uint8)
        heat = np.stack([hm, (hm*0.5).astype(np.uint8), 255 - hm], -1)
        sheet.paste(Image.fromarray(heat).resize((tw, th)), (x + tw, y + 14))
        r = res["cases"][name]
        dr.text((x + 3, y + 1), f"{name}  {r['changed_px']}px  {r['verdict']}", fill=(255, 255, 255) if r["verdict"] == "ok" else (255, 90, 90))
    sheet.save(os.path.join(a.out, f"{a.character}_params.png"))
    if not a.no_keys:
        keys = [(k, render(v)) for k, v in KEY_POSES.items()]
        cols = 4
        rows = (len(keys) + cols - 1)//cols
        ks = Image.new("RGB", (cols*a.w, rows*(a.h + 16)), (16, 16, 16))
        dk = ImageDraw.Draw(ks)
        for i, (k, img) in enumerate(keys):
            x, y = (i % cols)*a.w, (i//cols)*(a.h + 16)
            ks.paste(Image.fromarray(img), (x, y + 16))
            dk.text((x + 4, y + 2), f"{a.character}: {k}", fill=(255, 255, 255))
        ks.save(os.path.join(a.out, f"{a.character}_keys.png"))
    bad = {k: v["verdict"] for k, v in res["cases"].items() if v["verdict"] != "ok"}
    print(json.dumps({"character": a.character, "cases": len(res["cases"]), "not_ok": bad, "s": round(time.time() - t0)}))
    eng.close()


if __name__ == "__main__":
    main()
