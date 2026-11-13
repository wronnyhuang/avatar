"""Split-frame rendering for the v2 engine: the character pass of ONE frame is shared by K processes, everything after
it (character DoF, living background, god rays, composite, bloom, grade) stays full-frame in one of them.

    --backend live.strips:StripBackend --opt strips=4 --opt backend=cpu --opt threads=16      (run_live: --strips 4)

Why only the character pass: it is the only pass whose pixels are independent (each 8x8 tile of the prepass is one
point, its fur march touches nothing else), and it is most of the cost (llvmpipe 640x360: orb 41 of 114 ms, fox 149 of
233).  The post chain is full-frame (bloom mipmaps, the character DoF reads neighbours), so it is never split.

How, with engine/render.py untouched: Engine.render(frame, params, profile) calls profile[name] = ... after every
pass when a profile dict is given.  A dict whose __setitem__ acts on "prepass" and "char" is the hook:
  * after the prepass every process reads the same tile map, splits the occupied tile rows into K bands of equal tile
    count (identical arithmetic everywhere, so identical bands) and puts its own band in f_char's scissor, so its char
    pass shades only those tiles;
  * after the char pass a helper reads its band of the two float targets (colour, depth) back into shared memory and
    aborts the render; the finisher (strip 0, this process) writes every helper's band into its own targets and lets
    Engine.render carry on with the full-frame passes.
Per-pixel sums are formed in the same order as in one process, so the frame is bit-identical (tests below).  The price:
each helper runs its own prepass, and the gather moves K-1 bands of 32 bytes/pixel (colour + depth, RGBA32F).
On a single GPU this buys nothing (one device does the work either way); it is for CPU (llvmpipe), where one frame stops
scaling at ~16 threads.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from multiprocessing import connection, shared_memory

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [p for p in (ROOT, os.path.join(ROOT, "engine")) if p not in sys.path]

from live.backend_api import Backend, fit_params    # noqa: E402

TILE = 8


class _Abort(Exception):
    pass


def bands(pre, K):
    """Tile-row boundaries [b0=0 .. bK=th] giving each band an equal share of the occupied tiles (3x3 dilated, as the
    engine's own tile list).  Rows are in GL order (row 0 = bottom)."""
    th, tw = pre.shape
    pad = np.pad(pre, 1, mode="constant", constant_values=1e6)
    k = np.min(np.stack([pad[1 + dy:1 + dy + th, 1 + dx:1 + dx + tw] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]), axis=0)
    cum = np.cumsum((k < 1e5).sum(axis=1))
    total = int(cum[-1]) if cum.size else 0
    b = [0]
    for s in range(1, K):
        if total == 0:
            r = (th * s) // K
        else:
            r = int(np.searchsorted(cum, total * s / K)) + 1
        b.append(min(max(r, b[-1]), th))
    b.append(th)
    return b


class _Hooks(dict):
    def __init__(self, on_prepass, on_char):
        super().__init__()
        self.on_prepass, self.on_char = on_prepass, on_char

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        if k == "prepass":
            self.on_prepass()
        elif k == "char":
            self.on_char()


def _rows(eng, b, s):
    y0, y1 = b[s] * TILE, min(b[s + 1] * TILE, eng.h)
    return y0, max(y1 - y0, 0)


def _split(eng, K, s):
    """Run after the prepass: put band s in f_char's scissor (moderngl applies a framebuffer's own scissor on use())."""
    pre = np.frombuffer(eng.t_pre.read(), dtype="f4").reshape(eng.th, eng.tw, 4)[:, :, 0]
    b = bands(pre, K)
    y, hh = _rows(eng, b, s)
    eng.f_char.scissor = (0, y, eng.w, max(hh, 0)) if hh > 0 else (0, 0, 0, 0)
    return b


def _helper(s, K, character, w, h, spp, backend, threads, job_r, res_w, shm_name):
    try:
        import render
        eng = render.Engine(character, w, h, spp, backend, threads)
        shm = shared_memory.SharedMemory(name=shm_name)
        buf = shm.buf
        res_w.send(("ready", s))
        state = {}

        def pre():
            state["b"] = _split(eng, K, s)

        def char():
            eng.f_char.scissor = None
            eng.ctx.scissor = None
            y, hh = _rows(eng, state["b"], s)
            n = w * hh * 16
            if hh > 0:
                buf[:n] = eng.f_char.read(viewport=(0, y, w, hh), components=4, attachment=0, dtype="f4")
                buf[n:2 * n] = eng.f_char.read(viewport=(0, y, w, hh), components=4, attachment=1, dtype="f4")
            state["y"], state["hh"] = y, hh
            raise _Abort

        while True:
            msg = job_r.recv()
            if msg is None:
                break
            index, params = msg
            t0 = time.monotonic()
            try:
                eng.render(index, params, _Hooks(pre, char))
            except _Abort:
                pass
            res_w.send(("done", s, state["y"], state["hh"], t0, time.monotonic()))
        try:
            eng.ctx.release()
        except Exception:
            pass
        del buf
        shm.close()
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
        pass
    except Exception:
        import traceback
        try:
            res_w.send(("error", s, traceback.format_exc()))
        except Exception:
            pass


class StripBackend(Backend):
    name = "v2-strips"

    def __init__(self, character, width, height, strips=2, spp=3, backend="cpu", threads=8, **opts):
        super().__init__(character, width, height, **opts)
        import render
        self.K = int(strips)
        self.eng = render.Engine(character, width, height, spp, backend, threads)
        ctx = mp.get_context("spawn")
        self.helpers, self.jobs, self.res, self.shm = [], [], [], []
        n = width * height * 16 * 2
        for s in range(1, self.K):
            shm = shared_memory.SharedMemory(create=True, size=n)
            job_r, job_w = ctx.Pipe(duplex=False)
            res_r, res_w = ctx.Pipe(duplex=False)
            p = ctx.Process(target=_helper, daemon=True, name=f"strip-{s}",
                            args=(s, self.K, character, width, height, spp, backend, threads, job_r, res_w, shm.name))
            p.start()
            job_r.close(); res_w.close()
            self.helpers.append(p); self.jobs.append(job_w); self.res.append(res_r); self.shm.append(shm)
        for r in self.res:
            m = r.recv()
            if m[0] == "error":
                raise RuntimeError(f"strip helper failed:\n{m[2]}")
        self.last = {}

    def render(self, index, params):
        p = fit_params(params)
        for j in self.jobs:
            j.send((int(index), p))
        eng, w = self.eng, self.w
        st = {}
        t0 = time.monotonic()

        def pre():
            st["b"] = _split(eng, self.K, 0)
            st["t_pre"] = time.monotonic()

        def char():
            st["t_char"] = time.monotonic()
            eng.f_char.scissor = None
            eng.ctx.scissor = None
            want = {s: _rows(eng, st["b"], s) for s in range(1, self.K)}
            left = set(range(1, self.K))
            while left:
                for c in connection.wait([self.res[s - 1] for s in left]):
                    m = c.recv()
                    if m[0] == "error":
                        raise RuntimeError(f"strip helper failed:\n{m[2]}")
                    _, s, y, hh, h0, h1 = m
                    st.setdefault("helpers", {})[s] = 1000 * (h1 - h0)
                    if (y, hh) != want[s]:
                        raise RuntimeError(f"strip {s}: band {(y, hh)} != {want[s]} (tile maps disagree)")
                    if hh > 0:
                        n = w * hh * 16
                        buf = self.shm[s - 1].buf
                        eng.t_c.write(buf[:n], viewport=(0, y, w, hh))
                        eng.t_d.write(buf[n:2 * n], viewport=(0, y, w, hh))
                        del buf
                    left.discard(s)
            st["t_gather"] = time.monotonic()

        img = eng.render(int(index), p, _Hooks(pre, char))
        t1 = time.monotonic()
        self.last = {"prepass_ms": 1000 * (st["t_pre"] - t0), "char_ms": 1000 * (st["t_char"] - st["t_pre"]),
                     "gather_ms": 1000 * (st["t_gather"] - st["t_char"]), "post_ms": 1000 * (t1 - st["t_gather"]),
                     "bands": st["b"], "helper_ms": st.get("helpers", {})}
        return img

    def close(self):
        for j in self.jobs:
            try:
                j.send(None)
            except Exception:
                pass
        for p in self.helpers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        for s in self.shm:
            try:
                s.close(); s.unlink()
            except Exception:
                pass
        try:
            self.eng.ctx.release()
        except Exception:
            pass


def selftest(argv=None):
    """Bit-identity against the unsplit engine, and per-frame timing.  python -m live.strips --char fox --K 1,2,4"""
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", default="fox")
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=360)
    ap.add_argument("--K", default="2,4")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--spp", type=int, default=1)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--params", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import render
    P = np.load(a.params or os.path.join(os.path.dirname(ROOT), "out", "params", f"{a.char}_clip.npy"))
    idx = list(range(300, 300 + a.frames))
    eng = render.Engine(a.char, a.w, a.h, a.spp, a.backend, a.threads)
    ref, t_ref = [], []
    for i in idx:
        t = time.monotonic(); ref.append(eng.render(i, P[i])); t_ref.append(1000 * (time.monotonic() - t))
    eng.ctx.release()
    res = {"char": a.char, "res": f"{a.w}x{a.h}", "backend": a.backend, "threads_per_proc": a.threads,
           "single": {"mean_ms": round(float(np.mean(t_ref[3:])), 1), "p95_ms": round(float(np.percentile(t_ref[3:], 95)), 1)}}
    for K in [int(k) for k in a.K.split(",")]:
        be = StripBackend(a.char, a.w, a.h, strips=K, spp=a.spp, backend=a.backend, threads=a.threads)
        ts, same, parts = [], 0, []
        for n, i in enumerate(idx):
            t = time.monotonic(); img = be.render(i, P[i]); ts.append(1000 * (time.monotonic() - t))
            same += int(img.tobytes() == ref[n].tobytes())
            parts.append(be.last)
        be.close()
        res[f"K{K}"] = {"mean_ms": round(float(np.mean(ts[3:])), 1), "p95_ms": round(float(np.percentile(ts[3:], 95)), 1),
                        "bit_identical": f"{same}/{len(idx)}",
                        "parts_mean_ms": {k: round(float(np.mean([p[k] for p in parts[3:]])), 1)
                                          for k in ("prepass_ms", "char_ms", "gather_ms", "post_ms")},
                        "bands_last": parts[-1]["bands"],
                        "helper_ms_mean": {str(k): round(float(np.mean([p["helper_ms"][k] for p in parts[3:]])), 1)
                                           for k in range(1, K)}}
        print(K, res[f"K{K}"], flush=True)
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    selftest()
