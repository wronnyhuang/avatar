"""Forward-only speech feature extraction on a fixed sample grid.

Everything here is a pure function of one analysis window.  Temporal state lives in stream.py and
life.py.  The window is the WIN samples ending at a hop boundary, so a feature computed at hop h has
seen audio[: h * HOP] and nothing later.

Cost is one 1024-point FFT, one inverse FFT for voicing, one for the LPC autocorrelation, an order-10
Levinson recursion and a companion-matrix root solve per hop (120 hops/s).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SR = 24000
HOP = 200           # 8.33 ms; 4 hops per 30 fps video frame (800 samples)
WIN = 576           # 24 ms
NFFT = 1024
LPC_ORDER = 10
LPC_BINS = 171      # bins 0..170 span 0..3984 Hz => selective LPC on a virtual 7.97 kHz signal
EPS = 1e-14


def _bin(hz: float) -> int:
    return int(round(hz * NFFT / SR))


@dataclass(slots=True)
class HopFeat:
    db: float           # total level, dBFS
    db_vowel: float     # 300..3500 Hz level: the band syllable nuclei live in
    db_low: float       # 80..400 Hz: voice bar, nasal murmur
    db_mid: float       # 1000..3500 Hz: collapses during oral closure
    db_high: float      # 4000..11000 Hz: frication and alveolar bursts
    centroid: float     # Hz, 100..8000
    hf_ratio: float     # energy >= 3 kHz / total
    voiced: float       # 0..1 normalised autocorrelation peak
    f0: float           # Hz, 0 when unvoiced
    f1: float           # Hz
    f2: float           # Hz


SILENT = HopFeat(-140.0, -140.0, -140.0, -140.0, -140.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class Analyzer:
    def __init__(self) -> None:
        self.window = np.hanning(WIN + 2)[1:-1]
        self.win_energy = float(np.sum(self.window ** 2))
        freqs = np.arange(NFFT // 2 + 1) * SR / NFFT
        self.freqs = freqs
        self.k = {name: _bin(hz) for name, hz in dict(
            b80=80, b100=100, b300=300, b400=400, b1k=1000, b3k=3000, b3k5=3500, b4k=4000,
            b8k=8000, b11k=11000).items()}
        # voicing: autocorrelation of the sub-1 kHz band, corrected for the window's own taper
        lp = np.clip((1200.0 - freqs) / 400.0, 0.0, 1.0)
        self.voice_lp = lp * np.clip((freqs - 50.0) / 30.0, 0.0, 1.0)
        wac = np.fft.irfft(np.abs(np.fft.rfft(self.window, NFFT)) ** 2)
        self.win_ac = np.maximum(wac / wac[0], 1e-3)
        self.lag_min = int(SR / 420)
        self.lag_max = int(SR / 75)
        # LPC: +6 dB/oct pre-emphasis in the power spectrum, band-limited, slight lag window
        w = np.pi * freqs[:LPC_BINS] / (freqs[LPC_BINS - 1])
        self.pre = 1.0 + 0.94 ** 2 - 2.0 * 0.94 * np.cos(w)
        lags = np.arange(LPC_ORDER + 1)
        self.lag_win = np.exp(-0.5 * (2 * np.pi * 45.0 * lags / (2 * freqs[LPC_BINS - 1])) ** 2)
        self.fs_lpc = 2.0 * float(freqs[LPC_BINS - 1])
        self.centroid_f = freqs[self.k['b100']:self.k['b8k']]

    def analyze(self, x: np.ndarray) -> HopFeat:
        """x: WIN float64 samples in [-1, 1], oldest first."""
        if not x.any():
            return SILENT
        spec = np.fft.rfft(x * self.window, NFFT)
        p = spec.real * spec.real + spec.imag * spec.imag
        cum = np.concatenate(([0.0], np.cumsum(p)))
        k = self.k
        scale = 2.0 / (self.win_energy * NFFT)
        total = float(cum[-1]) * scale + EPS
        band = lambda lo, hi: float(cum[hi] - cum[lo]) * scale + EPS
        vowel = band(k['b300'], k['b3k5'])
        low = band(k['b80'], k['b400'])
        mid = band(k['b1k'], k['b3k5'])
        high = band(k['b4k'], k['b11k'])
        hf = band(k['b3k'], NFFT // 2 + 1)
        c_p = p[k['b100']:k['b8k']]
        centroid = float(np.dot(c_p, self.centroid_f)) / (float(c_p.sum()) + EPS)

        ac = np.fft.irfft(p * self.voice_lp * self.voice_lp)
        voiced = 0.0
        f0 = 0.0
        if ac[0] > 0.0:
            r = ac[self.lag_min:self.lag_max] / ac[0] / self.win_ac[self.lag_min:self.lag_max]
            j = int(np.argmax(r))
            voiced = max(0.0, min(1.0, float(r[j])))
            if voiced > 0.5:
                f0 = SR / (self.lag_min + j)

        f1, f2 = self._formants(p[:LPC_BINS])
        return HopFeat(
            10 * math.log10(total), 10 * math.log10(vowel), 10 * math.log10(low),
            10 * math.log10(mid), 10 * math.log10(high), centroid, hf / total, voiced, f0, f1, f2)

    def _formants(self, p: np.ndarray) -> tuple[float, float]:
        r = np.fft.irfft(p * self.pre)[:LPC_ORDER + 1] * self.lag_win
        if r[0] <= 0.0:
            return 0.0, 0.0
        a = levinson(r.tolist(), LPC_ORDER)
        z = np.roots(a)
        z = z[z.imag > 0.0]
        if z.size == 0:
            return 0.0, 0.0
        f = np.angle(z) * self.fs_lpc / (2 * np.pi)
        bw = -np.log(np.abs(z) + 1e-9) * self.fs_lpc / np.pi
        keep = (f > 180.0) & (f < 3800.0) & (bw < 550.0)
        f = np.sort(f[keep])
        if f.size == 0:
            return 0.0, 0.0
        f1 = float(f[0])
        rest = f[f > f1 + 180.0]
        return f1, float(rest[0]) if rest.size else 0.0


def levinson(r: list[float], order: int) -> list[float]:
    a = [1.0] + [0.0] * order
    err = r[0] * 1.0001
    for i in range(1, order + 1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k = -acc / err
        prev = a[:]
        for j in range(1, i):
            a[j] = prev[j] + k * prev[i - j]
        a[i] = k
        err *= 1.0 - k * k
        if err <= 0.0:
            break
    return a
