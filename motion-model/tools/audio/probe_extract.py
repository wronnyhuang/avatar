"""Probe-set extraction: ALL layers of one encoder for a small set of streams (layer pick / look-ahead study).
Writes <out>/<enc>_W<window>_L<lookahead>/<stream_id>.npz with arrays l00..lNN float16 (n_frames, dim)."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from unified import speech_enc as E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--enc", required=True)
ap.add_argument("--window", type=float, default=1.0)
ap.add_argument("--lookahead", type=int, nargs="+", default=[0])
ap.add_argument("--ids", required=True, help="file with one stream_id per line")
ap.add_argument("--out", required=True)
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--layers", default="all")
ap.add_argument("--mode", default="sliding", choices=["sliding", "kv"])
a = ap.parse_args()
torch.set_num_threads(a.threads)
rows = {r["stream_id"]: r for r in C.streams()}
ids = [l.strip() for l in open(a.ids) if l.strip()]
m = E.WindowEncoder(a.enc)
layers = list(range(m.n_layers + 1)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
t0 = time.time()
n_steps = 0
for L in a.lookahead:
    od = os.path.join(a.out, f"{a.enc}_W{a.window:g}_L{L}" if a.mode == "sliding" else f"{a.enc}_kv")
    os.makedirs(od, exist_ok=True)
    for i, sid in enumerate(ids):
        fn = os.path.join(od, sid + ".npz")
        if os.path.exists(fn):
            continue
        r = rows[sid]
        if a.mode == "kv":
            x16, base = C.stream_audio(r, pre_s=3.0, post_s=0.5)
            o = E.kv_extract_stream(m, x16, r["n_frames"], r["frame0"], layers, base)
        else:
            x16, base = C.stream_audio(r, pre_s=a.window + 0.5, post_s=1.5)
            o = E.extract_stream(m, x16, r["n_frames"], r["frame0"], a.window, L, layers, batch=a.batch, base=base)
        np.savez(fn + ".tmp.npz", **{f"l{k:02d}": v.astype(np.float16) for k, v in o.items()})
        os.replace(fn + ".tmp.npz", fn)
        n_steps += (r["n_frames"] + 2) // 3
        print(json.dumps(dict(enc=a.enc, L=L, i=i, sid=sid, steps=n_steps, s=round(time.time() - t0, 1),
                              ms_per_step=round((time.time() - t0) * 1e3 / max(n_steps, 1), 2))), flush=True)
print("DONE", a.enc, round(time.time() - t0, 1), flush=True)
