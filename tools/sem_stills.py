#!/usr/bin/env python3
"""Contact sheet of semantic programs -> rig poses -> the REAL renderer on the CPU (look at them!).

  python tools/sem_stills.py --character muse|fox --sheet OUT.png [--set accuracy|expr|all] [--w 480 --h 270]
      [--programs FILE (one program per line, `;` = newline)] [--threads 12] [--at end|peak]

Each program is replayed (app/semantic.replay), the pose at the end of its last statement (before the automatic return)
is put on top of the rig REST vector and rendered (TAA settled).  The label = the program.  No live layers: every dial
the program does not drive stays at rest.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(REPO, "app"), os.path.join(REPO, "renderer"), os.path.join(REPO, "renderer", "engine"),
                os.path.join(REPO, "renderer", "mesh")]

SETS = {
    "accuracy": [
        "touch('right', 'nose')",
        "touch('left', 'nose')",
        "pat('right', 'top_of_head', times=1)",
        "reach('right', 'chest')",
        "reach('right', 'left_shoulder')",
        "reach('left', 'right_shoulder')",
        "cover('eyes')",
        "cover('ears')",
        "cover('mouth', hand='right')",
        "reach('both', 'cheeks')",
        "point_at('left', hand='left')",
        "point_at('right', hand='right')",
        "point_at('up', hand='right')",
        "point_at('camera', hand='right')",
        "raise_hand('left')",
        "hands_on_hips()",
    ],
    "look": [
        "look_at('left')", "look_at('right')", "look_at('up')", "look_at('down')",
        "look_at('user')", "tilt_head('left', 20)", "head_turn('right', 35)", "lean('left', 15)",
        "twist('left', 25)", "shrug()", "crouch()", "jump()",
    ],
    "expr": [
        "expression('smile')", "expression('big_smile')", "expression('sad')", "expression('angry')",
        "expression('surprise')", "expression('disgust')", "expression('confused')", "expression('sleepy')",
        "wink('left')", "close_eyes()", "raise_brow('left')", "open_mouth()", "stick_out_tongue()", "squint()",
        "sneer()", "expression('fear')",
    ],
}


PEAK = False


def pose_at_end(character, prog, targets):
    import semantic as sem
    src = prog.replace(";", "\n")
    s = sem.Stream(character, targets=targets)
    bs = s.feed("```python\n" + src + "\nhold(0.6)\n```\n")
    poses = [p for b in bs for p in b["poses"]]
    rep = s.report()
    s.finish()
    rep = s.report()
    if not poses:
        return {}, rep
    # the last pose before the tail = held end pose; dials listed at any time keep their last value
    cur = {}
    best, bd = {}, -1.0
    for p in poses:
        cur.update({k: v for k, v in p.items() if k != "t"})
        d = sum(abs(v - (0.2 if k.endswith("_open") else 0.0)) for k, v in cur.items())
        if d > bd:
            best, bd = dict(cur), d
    return (best if PEAK else cur), rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="muse")
    ap.add_argument("--set", default="accuracy")
    ap.add_argument("--programs")
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--w", type=int, default=480)
    ap.add_argument("--h", type=int, default=270)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--settle", type=int, default=6)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--json")
    ap.add_argument("--at", default="end", choices=("end", "peak"))
    ap.add_argument("--extra", default="", help="'k=v k=v' undriven live-layer values to add under the program's pose")
    ap.add_argument("--crop", help="x0,y0,x1,y1 fractions of the frame to keep (e.g. juno: .3,0,.7,.6)")
    a = ap.parse_args()
    global PEAK
    PEAK = a.at == "peak"
    if a.programs:
        progs = [l.strip() for l in open(a.programs) if l.strip() and not l.startswith("#")]
    else:
        progs = sum((SETS[k] for k in (SETS if a.set == "all" else a.set.split(","))), [])
    import semantic as sem
    targets = sem.resolve_targets(a.character, None)
    from anim.schema import IDX, NPARAMS
    from anim.rig_poses import KEY_POSES, pose_vector
    rest = dict(KEY_POSES["REST"])
    t0 = time.time()
    if a.character == "juno":
        import assetload
        from engine import MeshEngine
        asset = assetload.load(None, lod="lod0")
        eng = MeshEngine("juno", a.w, a.h, "full1", backend="cpu", threads=a.threads, log=False, asset_dir=asset.dir,
                         lod="lod0")
    else:
        import rt_engine
        from live.backends.rt import default_preset
        eng = rt_engine.RTEngine(a.character, a.w, a.h, default_preset(a.character, a.w, a.h, 24), backend="cpu",
                                 threads=a.threads)
    print(f"engine {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
    from PIL import Image, ImageDraw
    imgs, meta = [], []
    for k, prog in enumerate(progs):
        pose, rep = pose_at_end(a.character, prog, targets)
        d = dict(rest)
        for tok in a.extra.split():
            k_, v_ = tok.split("=")
            d[k_] = float(v_)
        d.update({p: v for p, v in pose.items() if p in IDX})
        vec = pose_vector(d)
        if hasattr(eng, "reset_history"):
            eng.reset_history()
        img = None
        for f in range(a.settle):
            if hasattr(eng, "_t_now"):
                eng._t_now = (200 + f) / 30.0
            img = eng.render(200 + f, vec)
        if a.crop:
            x0, y0, x1, y1 = (float(v) for v in a.crop.split(","))
            im = np.asarray(img)
            img = im[int(y0 * im.shape[0]):int(y1 * im.shape[0]), int(x0 * im.shape[1]):int(x1 * im.shape[1])]
        imgs.append((prog, img))
        meta.append({"program": prog, "pose": pose, "errors": rep["errors"], "warnings": rep["warnings"]})
        print(f"{prog}  {time.time() - t0:.1f}s  {json.dumps({p: round(v, 2) for p, v in pose.items()})}",
              file=sys.stderr, flush=True)
    cols = min(a.cols, len(imgs))
    rows = (len(imgs) + cols - 1) // cols
    W, H = imgs[0][1].shape[1], imgs[0][1].shape[0] + 16
    sheet = Image.new("RGB", (cols * W, rows * H), (20, 20, 20))
    dr = ImageDraw.Draw(sheet)
    for i, (name, img) in enumerate(imgs):
        x, y = (i % cols) * W, (i // cols) * H
        sheet.paste(Image.fromarray(np.asarray(img)), (x, y + 16))
        dr.text((x + 4, y + 2), f"{a.character}: {name}"[:78], fill=(255, 255, 255))
    os.makedirs(os.path.dirname(os.path.abspath(a.sheet)), exist_ok=True)
    sheet.save(a.sheet)
    if a.json:
        json.dump(meta, open(a.json, "w"), indent=1)
    print(a.sheet)
    try:
        eng.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
