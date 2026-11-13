"""GPU real-time raymarch: the v2 shaders through engine/rt_engine.py (no CPU round trip, presets, temporal resolve).

    be = RTBackend("fox", 960, 540)                    # default preset for (character, size) from rt_presets.json
    be = RTBackend("orb", 1280, 720, preset="taa")     # or a named preset / a dict (engine/rt_engine.py PRESETS)
    img = be.render(index, params)                     # (h, w, 3) uint8 RGB, row 0 = top

opts: preset (name | dict | "auto" = "k620": the measured 24 fps choice on the dev machine's K620 for this character and size,
"k620_20": the 20 fps one, "quality": full-resolution temporal for a big GPU), target_fps (24 | 20, used by "auto"), backend ("gpu" | "cpu" llvmpipe),
threads (llvmpipe), pipelined (False: render() returns frame `index`; True: render() returns the PREVIOUS frame,
one frame of extra latency for overlap of readback and drawing — use submit()/flush() directly for that).
Causal: the temporal resolve only reads frames already rendered.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine")]
import rt_engine  # noqa: E402

from live.backend_api import Backend, fit_params  # noqa: E402

PRESET_TABLE = os.path.join(ROOT, "engine", "rt_presets.json")


def default_preset(character, width, height, target_fps=24):
    """The measured choice for this character and size (engine/rt_presets.json, written by tools/rt_table.py from
    tools/rt_bench.py runs on the K620): the best (whole + mouth SSIM) preset that held target_fps, else the fastest
    measured one; the nearest measured height (rounding up) when this height was not measured."""
    try:
        table = json.load(open(PRESET_TABLE))
    except (OSError, ValueError):
        return "q50"
    per = table.get(character, {})
    if not per:
        return table.get("_default", "q50")
    hs = sorted(int(k) for k in per if k.isdigit())
    best = min(hs, key=lambda x: (abs(x - height), -x))
    p = per[str(best)].get(str(int(target_fps))) or per[str(best)].get("24")
    return json.loads(p) if isinstance(p, str) and p.startswith("{") else p


class RTBackend(Backend):
    name = "rt"

    def __init__(self, character, width, height, preset="auto", target_fps=24, backend=None, threads=None,
                 tier="auto", budget_ms=None, **opts):
        # AVATAR_RT_DEVICE=cpu runs the same engine on Mesa llvmpipe (functional tests without the K620)
        backend = backend or os.environ.get("AVATAR_RT_DEVICE") or "gpu"
        threads = int(threads or os.environ.get("LP_NUM_THREADS", "8"))
        super().__init__(character, width, height, **opts)
        if preset in (None, "auto", "k620"):          # the K620's measured 24 fps choice for this character / size
            preset = default_preset(character, width, height, target_fps)
        elif preset == "k620_20":
            preset = default_preset(character, width, height, 20)
        self.preset = preset
        # quality tiers.  tier "auto" (default) = the adaptive controller: it starts on the preset's tier and
        # moves on the measured frame time (render + encode against budget_ms, default a 30 fps period: 9 ms of room
        # under the 24 fps frame); "fixed" (or AVATAR_ADAPTIVE=0) = the preset alone, as before; an int = that tier.
        P = rt_engine.resolve_preset(preset)
        if os.environ.get("AVATAR_ADAPTIVE", "1") == "0" and tier == "auto":
            tier = "fixed"
        adaptive = tier == "auto" and P["taa"] and backend == "gpu"
        if adaptive or isinstance(tier, int):
            P["tiers"] = True
        self.eng = rt_engine.RTEngine(character, width, height, P, backend=backend, threads=threads)
        if os.environ.get("AVATAR_PREWARM", "1") != "0":
            self.eng.prewarm()
        self.tier_names = ["q%d" % round(100*sc) for sc in self.eng.tiers]
        self.ctl = None
        if isinstance(tier, int):
            self.eng.set_tier(tier)
        elif adaptive and len(self.eng.tiers) > 1:
            from rt_adaptive import AdaptiveTier
            budget = float(budget_ms or os.environ.get("AVATAR_BUDGET_MS") or min(33.3, 800.0/max(target_fps, 1)))
            self.ctl = AdaptiveTier([sc*sc for sc in self.eng.tiers], start=self.eng.tier, budget_ms=budget,
                                    hard_ms=1000.0/max(target_fps, 1), fps=target_fps)
        self._last_ms = None
        self._noted = False
        self._enc_ms = 0.0

    @property
    def tier_now(self):
        return self.eng.tier if self.eng.tier_pending is None else self.eng.tier_pending

    def _apply(self, t):
        if t != self.tier_now:
            if os.environ.get("AVATAR_TIER_LOG", "1") != "0":
                why = self.ctl.log[-1][3] if self.ctl and self.ctl.log else ""
                print(f"[rt adaptive] {self.character}: tier {self.tier_names[self.tier_now]} -> {self.tier_names[t]} ({why})",
                      file=sys.stderr, flush=True)
            self.eng.set_tier(t)

    def note_frame(self, render_ms, encode_ms=0.0):
        """The caller's measured times for the frame it just finished (the render worker: render + encode).  Without
        it the backend feeds the controller its own render time."""
        self._enc_ms = 0.8*self._enc_ms + 0.2*float(encode_ms)
        self._noted = True
        if self.ctl is not None:
            self._apply(self.ctl.note(float(render_ms) + float(encode_ms)))

    def render(self, index, params):
        import time
        if self.ctl is not None and not self._noted and self._last_ms is not None:
            self._apply(self.ctl.note(self._last_ms + self._enc_ms))
        self._noted = False
        t = time.perf_counter()
        img = self.eng.render(int(index), fit_params(params))
        self._last_ms = 1000.0*(time.perf_counter() - t)
        return img

    # scene: the app server's live surfaces (board / tv): premultiplied RGBA8 sub-rect uploads, row 0 = the canvas top
    def surface_names(self):
        return self.eng.surface_names()

    def set_surface(self, name, x, y, w, h, data):
        return self.eng.surface_write(name, x, y, w, h, data)

    # pipelined use: submit(i, p) -> (i-1, image) or None; flush() -> remaining [(i, image)]
    def submit(self, index, params):
        return self.eng.submit(int(index), fit_params(params))

    def flush(self):
        return self.eng.flush()

    def close(self):
        self.eng.close()
