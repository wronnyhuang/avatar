"""A stand-in backend for loop tests: no GL, a fixed render cost, and a mouth you can measure in pixels.

    --backend live.testing:MouthBackend --opt ms=40 --opt jitter=5

Draws a grey card with a dark mouth whose opening is proportional to max(lip_open, jaw), and the 30 fps index as a
binary strip along the top so a recorded frame can be traced back to its row.  ms: render cost (sleep, like waiting on a
GPU), busy=true spins the CPU instead; slow_ms / slow_from / slow_to make rows [from, to) cost slow_ms (a GPU
neighbour arriving and leaving).
"""
from __future__ import annotations

import time

import numpy as np

from anim.schema import IDX
from live.backend_api import Backend, fit_params


class MouthBackend(Backend):
    name = "mouth"

    def __init__(self, character, width, height, ms=0.0, jitter=0.0, busy=False, seed=0, slow_ms=0.0, slow_from=-1,
                 slow_to=-1, **opts):
        super().__init__(character, width, height, **opts)
        self.ms, self.jitter, self.busy = float(ms), float(jitter), bool(busy)
        self.slow = (float(slow_ms), int(slow_from), int(slow_to))     # rows [from, to) cost slow_ms: a GPU neighbour
        self.rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:height, 0:width]
        self.cx, self.cy = width / 2, height * 0.62
        self.dx = (xx - self.cx) / (0.18 * width)
        self.dy = yy - self.cy
        self.bg = np.full((height, width, 3), 150, np.uint8)

    def mouth_box(self):
        w, h = self.w, self.h
        return int(0.3 * w), int(0.45 * h), int(0.4 * w), int(0.34 * h)      # x, y, w, h

    def render(self, index, params):
        t0 = time.monotonic()
        p = fit_params(params)
        o = max(float(p[IDX["lip_open"]]), float(p[IDX["jaw"]]))
        half = 2.0 + 0.15 * self.h * o
        img = self.bg.copy()
        m = self.dx ** 2 + (self.dy / half) ** 2 <= 1.0
        img[m] = (40, 10, 20)
        bits = np.array([(index >> b) & 1 for b in range(16)], np.uint8)
        bw = self.w // 16
        for b in range(16):
            img[:6, b * bw:(b + 1) * bw] = 255 * bits[b]
        cost = self.ms + (self.rng.uniform(-self.jitter, self.jitter) if self.jitter else 0.0)
        if self.slow[1] <= index < self.slow[2]:
            cost = self.slow[0]
        end = t0 + max(cost, 0.0) / 1000.0
        if self.busy:
            while time.monotonic() < end:
                pass
        else:
            d = end - time.monotonic()
            if d > 0:
                time.sleep(d)
        return img
