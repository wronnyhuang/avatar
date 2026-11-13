"""The causal heart of the live loop, with no clock and no renderer in it: audio chunks in, output-frame jobs out.

    core = LiveCore("orb", lookahead_ms=133, fps=24, resample="hold")
    for job in core.push(pcm):          # int16 PCM, 24 kHz mono, any chunk size
        ...                             # job.j, job.t, job.index, job.params, job.horizon, job.cursor
    for job in core.flush(): ...        # end of stream only

The animator (anim.lipstream.StreamingLipAnimator) releases 30 fps rows; row i depends on audio[: 800 i + H] with
H = max(400 A, 1608) samples (A = lookahead in 16.7 ms model steps; life keeps its own 67 ms) and is released the moment
that much audio has arrived.  The output frame rate is 30 (row j = frame j) or 24, resampled causally:

  hold    frame j (time u = j/24) shows row floor(30 u) = floor(5 j / 4): the newest row whose time is <= u.  It needs no
          audio later than row floor(30 u) does, so it adds NO latency and no lookahead; the instant shown trails the
          frame's own time by 0, 8.3, 16.7 or 25 ms (cycle of 4 frames, mean 12.5 ms).  Row 4k+4 of every 5 is unused.
  linear  frame j blends rows floor(30 u) and the next one with weight frac(30 u), so it shows exactly the instant u.
          The next row is released (t_next - u) later than u's own: 0, 25, 16.7, 8.3 ms over the 4-frame cycle (mean
          12.5 ms, max 25 ms), so the frame's audio horizon and its latency both grow by that much.

job.horizon is the number of samples the job's params may depend on; tests/test_live_causal.py proves no job depends on
audio past it and that it never exceeds t + lookahead (+ the linear resampler's documented extra).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from anim.lipstream import DEFAULT_LEAD, StreamingLipAnimator
from anim.schema import FPS as ROW_FPS, SAMPLE_RATE
from anim.stream import SPF

RESAMPLERS = ("hold", "linear")


@dataclass
class Job:
    j: int                  # output frame index at the output fps
    t: float                # stream time (s) this frame depicts: j / fps
    index: int              # 30 fps row index handed to the backend (time-driven scenery uses index / 30)
    params: np.ndarray      # float32 row (blended for linear)
    rows: tuple             # the 30 fps rows it was made from
    horizon: int            # samples of audio its params may depend on
    cursor: int             # samples received when it was released
    meta: dict = field(default_factory=dict)


class LiveCore:
    def __init__(self, character="orb", lookahead_ms=133.0, fps=30, resample="hold", lead_frames=DEFAULT_LEAD,
                 animator=None):
        if fps not in (24, 30):
            raise ValueError("fps must be 24 or 30")
        if resample not in RESAMPLERS:
            raise ValueError(f"resample must be one of {RESAMPLERS}")
        self.fps, self.resample = int(fps), resample
        self.anim = animator or StreamingLipAnimator(character, lookahead_ms=lookahead_ms, lead_frames=lead_frames)
        self.H = int(self.anim._look_samples)       # per-row horizon past the row's own time, samples
        self.cursor = 0
        self.rows = {}                  # 30 fps rows not yet consumed by every output frame that needs them
        self.next_j = 0
        self.done = False

    # ----------------------------------------------------------- the resampler's arithmetic (exact integers)
    def rows_for(self, j):
        """(i0, i1, frac) for output frame j: i1 is None when the frame lands exactly on row i0."""
        if self.fps == ROW_FPS:
            return j, None, 0.0
        num = j * ROW_FPS
        i0, r = divmod(num, self.fps)
        if r == 0 or self.resample == "hold":
            return i0, None, 0.0
        return i0, i0 + 1, r / self.fps

    def allowed_horizon(self, j):
        """The most audio (samples) frame j may depend on: its own time + the lookahead, plus for linear 24 fps
        the distance to the next row.  Used by the tests, and logged."""
        i0, i1, _ = self.rows_for(j)
        last = i0 if i1 is None else i1
        return last * SPF + self.H

    def lookahead_samples(self):
        return self.H

    # ----------------------------------------------------------- streaming
    def push(self, pcm):
        pcm = np.asarray(pcm)
        self.cursor += pcm.size
        return self._take(self.anim.push(pcm))

    def flush(self):
        out = self._take(self.anim.flush())
        self.done = True
        return out

    def _take(self, frames):
        for f in frames:
            self.rows[f.index] = f.params
        out = []
        while True:
            i0, i1, frac = self.rows_for(self.next_j)
            last = i0 if i1 is None else i1
            if last not in self.rows:
                break
            j = self.next_j
            if i1 is None:
                p = self.rows[i0].copy()
                used = (i0,)
            else:
                a, b = self.rows[i0], self.rows[i1]
                p = (a + np.float32(frac) * (b - a)).astype(np.float32)
                used = (i0, i1)
            out.append(Job(j, j / self.fps, i0, p, used, last * SPF + self.H, self.cursor))
            self.next_j += 1
            # rows older than the next frame's first row are never needed again
            n0 = self.rows_for(self.next_j)[0]
            for k in [k for k in self.rows if k < n0]:
                del self.rows[k]
        return out


def sample_index(t):
    return int(round(t * SAMPLE_RATE))
