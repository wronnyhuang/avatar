"""The shipped v2 engine behind the live contract: engine/render.py, unchanged.  opts: spp (3), backend gpu|cpu, threads."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine")]
import render  # noqa: E402

from live.backend_api import Backend, fit_params  # noqa: E402


class V2Backend(Backend):
    name = "v2"

    def __init__(self, character, width, height, spp=3, backend="gpu", threads=8, **opts):
        super().__init__(character, width, height, **opts)
        self.eng = render.Engine(character, width, height, spp, backend, threads)

    def render(self, index, params):
        return self.eng.render(int(index), fit_params(params))

    def close(self):
        try:
            self.eng.ctx.release()
        except Exception:
            pass
