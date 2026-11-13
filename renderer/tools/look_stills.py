#!/usr/bin/env python3
"""BEFORE / AFTER stills of the look upgrade — the same character, pose, camera and preset rendered twice
through the real-time engine (look off = the picture before the upgrade, look on), side by side in one PNG.

  python tools/look_stills.py --characters muse,fox,orb --w 1280 --h 720 --preset taa --out-dir DIR
  python tools/look_stills.py --characters muse --w 480 --h 270 --preset auto --zoom 2 --out-dir DIR

Poses: a named key pose (anim/rig_poses.py) or "name=value ..." on top of REST; --poses takes several, separated by
commas.  --crop x0,y0,x1,y1 (fractions of the frame) cuts both halves the same way (a face close-up).
"""
import argparse, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine"), HERE]
from rig_still import parse_pose  # noqa: E402

DEFAULT_POSES = ["REST", "WAVE_UP smile=0.8 mouth_open=0.7 lip_open=0.8 jaw=0.6 brow_l=0.6 brow_r=0.6 head_yaw=0.25 head_roll=0.06"]


def render_pose(eng, vec, settle):
    eng.reset_history()
    img = None
    for k in range(settle):
        img = eng.render(k, vec)
    return np.array(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--characters", default="muse,fox,orb")
    ap.add_argument("--poses", default=None)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--preset", default="taa")
    ap.add_argument("--settle", type=int, default=10)
    ap.add_argument("--zoom", type=int, default=1)
    ap.add_argument("--crop")
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", choices=["before", "after"], help="render one side only (a quick look while tuning)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    import rt_engine
    from PIL import Image, ImageDraw
    os.makedirs(a.out_dir, exist_ok=True)
    poses = [p for p in (a.poses.split(",") if a.poses else DEFAULT_POSES) if p.strip()]
    for ch in a.characters.split(","):
        preset = a.preset
        if preset == "auto":
            from live.backends.rt import default_preset
            preset = default_preset(ch, a.w, a.h, 24)
        base = rt_engine.resolve_preset(preset)
        imgs = {}
        for side, look in (("before", False), ("after", True)):
            if a.only and a.only != side:
                continue
            p = dict(base, look=look)
            eng = rt_engine.RTEngine(ch, a.w, a.h, p, backend="gpu")
            imgs[side] = [render_pose(eng, parse_pose(s)[1], a.settle) for s in poses]
            eng.close()
        for i, spec in enumerate(poses):
            halves = []
            for side in ("before", "after"):
                if side not in imgs:
                    continue
                im = Image.fromarray(imgs[side][i])
                if a.crop:
                    x0, y0, x1, y1 = (float(v) for v in a.crop.split(","))
                    im = im.crop((int(x0*a.w), int(y0*a.h), int(x1*a.w), int(y1*a.h)))
                if a.zoom > 1:
                    im = im.resize((im.width*a.zoom, im.height*a.zoom), Image.LANCZOS)
                d = ImageDraw.Draw(im)
                d.rectangle((0, 0, 58 if side == "before" else 48, 14), fill=(0, 0, 0))
                d.text((3, 2), side.upper(), fill=(255, 255, 255))
                halves.append(im)
            sheet = Image.new("RGB", (sum(h.width for h in halves) + 4*(len(halves) - 1), halves[0].height), (20, 20, 20))
            x = 0
            for hv in halves:
                sheet.paste(hv, (x, 0)); x += hv.width + 4
            name = f"{ch}_{a.w}x{a.h}_{i:02d}{('_' + a.tag) if a.tag else ''}.png"
            sheet.save(os.path.join(a.out_dir, name))
            print(os.path.join(a.out_dir, name), flush=True)


if __name__ == "__main__":
    main()
