#!/usr/bin/env python3
"""objects-eval: render chosen KEY FRAMES of a recorded rig (engine.cjs dump) with the candidate's own real-time
renderer — the same mixing path as renderer/tools/render_timeline.py (LiveCore idle life + anim.rig.RigMixer smoothing of
the page's rig vectors), but only the requested frames are drawn (each settled a few TAA frames), so a 30 s case costs
~12 renders instead of 720. Optional camera floor: cam_zoom = max(page value, --zoom[i]) so things away from the muse stay
in the frame (cam_x / cam_y stay the page's).

  python render_keys.py --renderer <cand>/source/renderer --rig rig.jsonl --keys 0.5,1.2,... [--zoom 0.3,0.3,...]
                        --out-dir DIR [--w 640 --h 360 --threads 12 --settle 3]
-> prints JSON {"frames": [png paths]}
"""
import argparse, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--renderer", required=True)
ap.add_argument("--rig", required=True)
ap.add_argument("--keys", required=True)
ap.add_argument("--zoom", default="")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--w", type=int, default=640)
ap.add_argument("--h", type=int, default=360)
ap.add_argument("--threads", type=int, default=12)
ap.add_argument("--settle", type=int, default=3)
ap.add_argument("--fps", type=int, default=24)
a = ap.parse_args()
ROOT = os.path.abspath(a.renderer)
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine"), os.path.join(ROOT, "tools")]
from anim.schema import IDX, SAMPLE_RATE as SR  # noqa: E402
from anim.rig import RigMixer, RIG_PARAMS  # noqa: E402
from render_timeline import read_rig  # noqa: E402
from avatar.render_server import Core  # noqa: E402
from PIL import Image  # noqa: E402

keys = [float(x) for x in a.keys.split(",") if x.strip()]
zooms = [float(x) for x in a.zoom.split(",") if x.strip()] or [0.0] * len(keys)
rig = read_rig(a.rig)
dur = max(keys) + 0.3
n_s = int(round(dur * SR))
core = Core("muse", 133.0, a.fps)
jobs = []
pcm = np.zeros(n_s + int(SR * 0.4), np.int16)
for k in range(0, pcm.size, 2400):
    jobs += core.push(pcm[k:k + 2400])
n_frames = int(np.floor(dur * a.fps))
jobs = [jb for jb in jobs if jb.j < n_frames]
want = {}
for i, t in enumerate(keys):
    want.setdefault(min(n_frames - 1, int(round(t * a.fps))), []).append(i)
mix = RigMixer(character="muse")
rows, k = {}, 0
for jb in jobs:
    t = jb.j / float(a.fps)
    while k < len(rig) and rig[k][0] <= t + 1e-9:
        tr, kind, payload, names = rig[k]
        if kind == "a":
            mix.push_rig(tr, payload, now=tr, names=names)
        k += 1
    p = mix.apply(jb.params, t, now=t)
    if jb.j in want:
        rows[jb.j] = (jb, np.array(p, copy=True))
import rt_engine  # noqa: E402
from live.backends.rt import default_preset  # noqa: E402
eng = rt_engine.RTEngine("muse", a.w, a.h, default_preset("muse", a.w, a.h, 24), backend="cpu", threads=a.threads)
os.makedirs(a.out_dir, exist_ok=True)
out = [None] * len(keys)
t0 = time.time()
for j, idxs in sorted(want.items()):
    if j not in rows:
        continue
    jb, p = rows[j]
    for i in idxs:
        q = p.copy()
        q[IDX["cam_zoom"]] = max(float(q[IDX["cam_zoom"]]), zooms[i] if i < len(zooms) else 0.0)
        eng.reset_history()
        for s in range(a.settle):
            img = eng.render(int(jb.index) + s, q)
        f = os.path.join(a.out_dir, f"k{i:02d}.png")
        Image.fromarray(img).save(f)
        out[i] = f
eng.close()
print(json.dumps({"frames": out, "render_s": round(time.time() - t0, 1)}))
