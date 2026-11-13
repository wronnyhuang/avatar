"""Video sources (YouTube, ViCo) -> CANON v1 through MediaPipe + the Fitter.

  fit_stream(raw)            MediaPipe raw arrays of one person stream -> motion (T,189), mask (T,9) on a 30 fps grid
                             (the real canon1.fit.Fitter stepped causally, exactly as live)
  audio_16k(path)            any media file -> 16 kHz mono float32 (ffmpeg)
  talk_score(motion, mask)   mouth activity per frame (for attributing a shared mix to the person who talks)
  speaking_flags(...)        energy VAD of the mix AND mouth activity (the louder mover wins in a two-person call)
  gate(mix_feats, flag)      the mix's features where flag, the silence floor elsewhere (mel log 1e-6, loud -100, vad 0)
  segments(ok, cuts)         face-present runs (gaps < 1 s bridged) split at scene cuts, >= 3 s
"""
from __future__ import annotations

import subprocess

import numpy as np
from scipy.ndimage import uniform_filter1d

from . import spec as S
from .fit import Fitter, frames_from_raw

SIL_MEL = float(np.log(1e-6))
MOUTH_IDX = [135 + S.BS[n] for n in ("jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthLowerDownLeft",
                                     "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight")]


def fit_stream(raw, T=None, fitter=None, reset_at=()):
    f = fitter or Fitter()
    frs = frames_from_raw(raw, T)
    T = len(frs)
    M = np.zeros((T, S.D_MOTION), np.float32)
    K = np.zeros((T, S.D_MASK), np.uint8)
    rs = set(int(r) for r in reset_at)
    for k, fr in enumerate(frs):
        if k in rs:
            f.reset()
        M[k], K[k] = f.step(fr)
    return M, K


def audio_16k(path, start=None, dur=None):
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "1"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path)]
    if dur is not None:
        cmd += ["-t", str(dur)]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=False).stdout
    return np.frombuffer(out, np.float32).copy()


def talk_score(motion, mask):
    """(T,) mouth activity: 0.4 s moving std of the mouth blendshapes + mean |velocity|; 0 where the mouth is unseen."""
    m = motion[:, MOUTH_IDX].astype(np.float64)
    mu = uniform_filter1d(m, 12, axis=0, mode="nearest")
    var = uniform_filter1d(m ** 2, 12, axis=0, mode="nearest") - mu ** 2
    v = np.abs(np.diff(m, axis=0, prepend=m[:1])).sum(1)
    s = np.sqrt(np.maximum(var, 0)).sum(1) + uniform_filter1d(v, 12, mode="nearest")
    s[mask[:, S.GROUP_NAMES.index("mouth")] == 0] = 0
    return s


def speaking_flags(mix_vad, scores, rel=1.3, floor_q=40):
    """mix_vad (T,) 0/1; scores (P,T) talk_score -> (P,T) bool: speech in the mix AND this face's mouth moves above its
    own quiet level (the floor_q-th percentile of its score when the mix is silent x rel) AND (P==2) it moves more than
    the other face. Hysteresis-free; smoothed with a 0.3 s majority filter."""
    P, T = scores.shape
    v = mix_vad > 0.5
    out = np.zeros((P, T), bool)
    for p in range(P):
        q = scores[p][~v & (scores[p] > 0)]
        base = np.percentile(q, floor_q) if len(q) > 30 else np.percentile(scores[p][scores[p] > 0], 25) \
            if (scores[p] > 0).any() else 0.0
        out[p] = v & (scores[p] > rel * base) & (scores[p] > 0)
    if P == 2:
        a, b = scores
        out[0] &= a >= b
        out[1] &= b > a
    sm = uniform_filter1d(out.astype(np.float32), 9, axis=1, mode="nearest") > 0.5
    return sm


def gate(feats, flag):
    g = feats.copy()
    off = ~np.asarray(flag, bool)
    g[off, :40] = SIL_MEL
    g[off, 40] = -100.0
    g[:, 41] = np.asarray(flag, np.float32)
    return g


def segments(ok, cuts=None, gap=30, min_len=90):
    """ok (T,) bool -> [(a,b)] runs of ok with gaps < gap frames bridged, split at cuts (T,) bool, len >= min_len."""
    ok = np.asarray(ok, bool)
    T = len(ok)
    br = ok.copy()
    i = 0
    while i < T:
        if not ok[i]:
            j = i
            while j < T and not ok[j]:
                j += 1
            if 0 < i and j < T and j - i < gap:
                br[i:j] = True
            i = j
        else:
            i += 1
    if cuts is not None:
        br = br & ~np.asarray(cuts, bool)
    out = []
    i = 0
    while i < T:
        if br[i]:
            j = i
            while j < T and br[j]:
                j += 1
            # trim to first/last truly-ok frame
            a, b = i, j
            while a < b and not ok[a]:
                a += 1
            while b > a and not ok[b - 1]:
                b -= 1
            if b - a >= min_len:
                out.append((a, b))
            i = j
        else:
            i += 1
    return out


def energy_stats(motion, mask):
    from .seamless import energy_tracks
    return energy_tracks(motion, mask)
