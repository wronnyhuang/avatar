#!/usr/bin/env python3
"""Stills of named rig poses through the REAL-TIME engine (TAA converged over a few frames of the same pose).

  python tools/rig_still.py --character fox --poses REST,WAVE_UP,"ear_l=1 ear_r=-1" --out-dir DIR [--backend cpu] [--w 480 --h 270]
  python tools/rig_still.py --character fox --sheet DIR/fox_sheet.png --poses all

A pose is a key-pose name from anim/rig_poses.py (KEY_POSES), or "name=value name=value ..." on top of REST.
"""
import argparse, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine")]
from anim.schema import IDX, NPARAMS  # noqa: E402
from anim.rig_poses import KEY_POSES, pose_vector  # noqa: E402


def parse_pose(spec):
    spec = spec.strip()
    if spec in KEY_POSES:
        return spec, pose_vector(KEY_POSES[spec])
    d = {}
    base = "REST"
    for tok in spec.split():
        if "=" not in tok:
            base = tok
            continue
        k, v = tok.split("=", 1)
        if k not in IDX:
            raise KeyError(f"unknown param {k}")
        d[k] = float(v)
    full = dict(KEY_POSES.get(base, {}))
    full.update(d)
    return spec, pose_vector(full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="fox")
    ap.add_argument("--poses", default="REST")
    ap.add_argument("--w", type=int, default=480)
    ap.add_argument("--h", type=int, default=270)
    ap.add_argument("--preset", default="auto")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--settle", type=int, default=6)
    ap.add_argument("--out-dir")
    ap.add_argument("--sheet")
    ap.add_argument("--cols", type=int, default=4)
    a = ap.parse_args()
    import rt_engine
    from PIL import Image, ImageDraw
    preset = a.preset
    if preset == "auto":
        from live.backends.rt import default_preset
        preset = default_preset(a.character, a.w, a.h, 24)
    if a.poses == "all":
        specs = list(KEY_POSES)
    else:
        specs = [s for s in a.poses.split(",") if s.strip()]
    t0 = time.time()
    eng = rt_engine.RTEngine(a.character, a.w, a.h, preset, backend=a.backend, threads=a.threads)
    print(f"engine {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
    imgs = []
    for k, spec in enumerate(specs):
        name, vec = parse_pose(spec)
        eng.reset_history()
        for f in range(a.settle):
            img = eng.render(100 + f, vec)
        imgs.append((name, img))
        if a.out_dir:
            os.makedirs(a.out_dir, exist_ok=True)
            fn = name.replace(" ", "_").replace("=", "").replace("/", "_")[:60]
            Image.fromarray(img).save(os.path.join(a.out_dir, f"{a.character}_{k:02d}_{fn}.png"))
        print(f"{name}  {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
    if a.sheet:
        cols = min(a.cols, len(imgs))
        rows = (len(imgs) + cols - 1)//cols
        W, H = a.w, a.h + 16
        sheet = Image.new("RGB", (cols*W, rows*H), (20, 20, 20))
        dr = ImageDraw.Draw(sheet)
        for i, (name, img) in enumerate(imgs):
            x, y = (i % cols)*W, (i//cols)*H
            sheet.paste(Image.fromarray(img), (x, y + 16))
            dr.text((x + 4, y + 2), f"{a.character}: {name}"[:70], fill=(255, 255, 255))
        os.makedirs(os.path.dirname(os.path.abspath(a.sheet)), exist_ok=True)
        sheet.save(a.sheet)
        print(a.sheet)
    eng.close()


if __name__ == "__main__":
    main()
