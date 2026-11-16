"""render6: JUNO_POISON -- fill every GL object the mesh pipeline allocates WITHOUT data with garbage at creation.

A cloud A40 showed sharp black SSS-shaped blobs on Juno's face in every process of one pod and none on another pod
with the same code + driver (PERF2 05:46 / 06:08): the signature of a read of uninitialised GPU memory (VRAM contents
differ per machine; llvmpipe and the K620 happened to hand out zeros).  This makes such reads reproducible anywhere:

    JUNO_POISON=1|nan    NaN in float objects, 0xFF bytes in 8-bit ones, depth 0.0 (nearest: hides everything behind)
    JUNO_POISON=big      +65504 (fp16 max) / 3e38 in float objects, depth 0.0
    JUNO_POISON=rand     random bytes everywhere (NaN / Inf / denormals / huge), depth random in [0, 1]
    JUNO_POISON=neg      -65504 / -3e38 (a negative radiance spreads through the SSS blur like NaN)

Scope: moderngl.Context.{buffer, texture, texture_array, texture3d, texture_cube, depth_texture, renderbuffer,
depth_renderbuffer} calls whose CALLER is a file under renderer/mesh/ (the engine's own SDF targets are not touched:
Muse stays bit-identical).  Objects created WITH data are left alone except buffers reserved larger than their data
(not a moderngl option: reserve= has no data).  Renderbuffers (MSAA) are poisoned through a scratch framebuffer clear.
Off (the default) = nothing is patched."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = os.environ.get("JUNO_POISON", "").strip().lower()
if MODE in ("1", "true", "yes", "on"):
    MODE = "nan"
_INSTALLED = False
STATS = {}


def _ours(depth=2):
    f = sys._getframe(depth)
    fn = os.path.abspath(f.f_code.co_filename)
    return fn.startswith(HERE + os.sep) and not fn.endswith("poison.py")


def _fill(nbytes, dtype):
    """garbage bytes for an object of `nbytes` whose elements are `dtype` ('f1'/'u1'/'f2'/'f4'/'i4'/'u4'/'depth')."""
    rng = np.random.default_rng(4430 + nbytes % 997)
    if MODE == "rand":
        if dtype == "depth":
            return rng.random(nbytes//4, dtype=np.float32).tobytes()
        return rng.integers(0, 256, nbytes, dtype=np.uint8).tobytes()
    if dtype == "depth":
        return np.zeros(nbytes//4, np.float32).tobytes()
    if dtype == "f2":
        v = {"nan": np.nan, "big": 65504.0, "neg": -65504.0}.get(MODE, np.nan)
        return np.full(nbytes//2, v, np.float16).tobytes()
    if dtype in ("f4",):
        v = {"nan": np.nan, "big": 3e38, "neg": -3e38}.get(MODE, np.nan)
        return np.full(nbytes//4, v, np.float32).tobytes()
    return np.full(nbytes, 0xFF, np.uint8).tobytes()


def _nbytes(size, comps, dtype, layers=1):
    w, h = size[0], size[1]
    per = {"f1": 1, "u1": 1, "i1": 1, "f2": 2, "u2": 2, "i2": 2, "f4": 4, "u4": 4, "i4": 4}.get(dtype, 4)
    return int(w*h*layers*comps*per)


def _count(kind):
    STATS[kind] = STATS.get(kind, 0) + 1


def install():
    global _INSTALLED
    if _INSTALLED or not MODE:
        return False
    import moderngl
    C = moderngl.Context
    o_buffer, o_tex, o_arr, o_3d, o_cube = C.buffer, C.texture, C.texture_array, C.texture3d, C.texture_cube
    o_dtex, o_rb, o_drb = C.depth_texture, C.renderbuffer, C.depth_renderbuffer

    def buffer(self, data=None, reserve=0, dynamic=False):
        b = o_buffer(self, data, reserve, dynamic)
        if data is None and reserve and _ours():
            b.write(_fill(b.size, "f4")); _count("buffer")
        return b

    def texture(self, size, components, data=None, *a, **kw):
        dt = kw.get("dtype", a[2] if len(a) > 2 else "f1")
        ms = kw.get("samples", a[0] if a else 0) or kw.get("renderbuffer", False)
        if data is None and not ms and _ours():
            data = _fill(_nbytes(size, components, dt), dt); _count("texture")
        return o_tex(self, size, components, data, *a, **kw)

    def texture_array(self, size, components, data=None, *a, **kw):
        dt = kw.get("dtype", a[1] if len(a) > 1 else "f1")
        if data is None and _ours():
            data = _fill(_nbytes(size, components, dt, size[2]), dt); _count("texture_array")
        return o_arr(self, size, components, data, *a, **kw)

    def texture3d(self, size, components, data=None, *a, **kw):
        dt = kw.get("dtype", a[1] if len(a) > 1 else "f1")
        if data is None and _ours():
            data = _fill(_nbytes(size, components, dt, size[2]), dt); _count("texture3d")
        return o_3d(self, size, components, data, *a, **kw)

    def texture_cube(self, size, components, data=None, *a, **kw):
        dt = kw.get("dtype", a[1] if len(a) > 1 else "f1")
        if data is None and _ours():
            data = _fill(_nbytes(size, components, dt, 6), dt); _count("texture_cube")
        return o_cube(self, size, components, data, *a, **kw)

    def depth_texture(self, size, data=None, *a, **kw):
        if data is None and _ours():
            data = _fill(_nbytes(size, 1, "f4"), "depth"); _count("depth_texture")
        return o_dtex(self, size, data, *a, **kw)

    def _clear_rb(ctx, rb, depth):
        try:
            fb = ctx.framebuffer(depth_attachment=rb) if depth else ctx.framebuffer([rb])
            if depth:
                fb.clear(depth=float(np.random.default_rng(1).random()) if MODE == "rand" else 0.0)
            else:
                v = {"nan": float("nan"), "big": 65504.0, "neg": -65504.0}.get(MODE)
                if v is None:
                    r = np.random.default_rng(2).normal(0, 3e4, 4)
                    fb.clear(*[float(x) for x in r])
                else:
                    fb.clear(v, v, v, v)
            fb.release()
        except Exception as e:                      # never break a render over the debug switch
            print(f"[poison] renderbuffer clear failed: {e}", flush=True)

    def renderbuffer(self, size, components=4, *a, **kw):
        rb = o_rb(self, size, components, *a, **kw)
        if _ours():
            _clear_rb(self, rb, False); _count("renderbuffer")
        return rb

    def depth_renderbuffer(self, size, *a, **kw):
        rb = o_drb(self, size, *a, **kw)
        if _ours():
            _clear_rb(self, rb, True); _count("depth_renderbuffer")
        return rb

    C.buffer, C.texture, C.texture_array, C.texture3d, C.texture_cube = buffer, texture, texture_array, texture3d, texture_cube
    C.depth_texture, C.renderbuffer, C.depth_renderbuffer = depth_texture, renderbuffer, depth_renderbuffer
    _INSTALLED = True
    import atexit
    atexit.register(lambda: print(f"[poison] poisoned {STATS}", flush=True))
    print(f"[poison] JUNO_POISON={MODE}: mesh-pipeline allocations without data are filled with garbage", flush=True)
    return True


install()
