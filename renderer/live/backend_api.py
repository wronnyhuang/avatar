"""The one contract between the live loop and any renderer.

    from live.backend_api import get_backend
    B = get_backend("rt")                      # "v2" | "rt" | "sprite" | "raster"
    be = B("fox", 1280, 720, **opts)           # opts are backend specific (spp, quality, ...)
    img = be.render(index, params)             # (h, w, 3) uint8 RGB, row 0 = top of the image
    be.close()

index  = the 30 fps frame index of the animator (anim.schema.FPS); time = index / 30 s.  A backend may use it for
         time-driven scenery (foliage, motes, dapples) and per-frame noise, never to look up future data.
params = one row of the animator's output, anim.schema.PARAMS order.  Shorter rows are zero padded, longer rows
         (round three appends squint, smirk and hand columns at the END) are cut to this tree's NPARAMS — so a round-3
         animator can drive these backends today and its new columns are simply ignored until a rig reads them.
Strictly causal: render(i, p) may keep state from earlier calls (temporal accumulation, caches) but never waits for or
reads a later frame.
"""
from __future__ import annotations

import importlib
import numpy as np

from anim.schema import NPARAMS

REGISTRY = {
    "v2": "live.backends.v2:V2Backend",            # the shipped engine, unchanged (engine/render.py)
    "rt": "live.backends.rt:RTBackend",            # GPU real-time raymarch (same shaders)
    "sprite": "live.backends.sprite:SpriteBackend",  # pre-rendered layers, CPU compositing
    "raster": "live.backends.raster:RasterBackend",  # meshed rigs, rasterized
}


def fit_params(params, n: int = NPARAMS) -> np.ndarray:
    p = np.asarray(params, dtype=np.float32).ravel()
    if p.size >= n:
        return p[:n].copy()
    return np.concatenate([p, np.zeros(n - p.size, np.float32)])


class Backend:
    name = "base"

    def __init__(self, character: str, width: int, height: int, **opts):
        self.character, self.w, self.h, self.opts = character, width, height, opts

    def render(self, index: int, params) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


def get_backend(name: str):
    mod, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(mod), cls)
