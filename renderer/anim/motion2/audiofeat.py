"""live: STREAMING canon1 audio features (40 causal log-mel + loud_db + vad @ 30 fps) from the renderer's 24 kHz PCM.

A streaming port of canon1/causal_audio.py (X/canon1/spec.py audio_features / energy_vad / resample_to_16k) so the
render image needs nothing outside this tree.  Numpy only.  Frame k of the PLAYBACK TIMELINE covers [k/30, (k+1)/30) s and
uses only 16 kHz samples before its end (exactly the offline definition, timeline sample 0 = 16 kHz sample 0).
The 24 -> 16 kHz resampler is a polyphase Kaiser FIR equivalent to scipy.signal.resample_poly(x, 2, 3) (same filter,
zero-phase: an output sample waits for 0.3 ms of input past it).

    af = StreamFeat()
    feats = af.push(s0_24k, pcm_int16)     # -> (n, 42) float32 for every frame completed by this chunk (maybe 0 rows)
    af.next_frame                          # global index of the next frame it will emit
"""
from __future__ import annotations

import bisect
import math

import numpy as np

FPS = 30
SR = 16000
N_MEL = 40
N_FFT = 512
HOP = int(round(SR / FPS))      # 533: the loudness window


def _mel_fb(n_mel=N_MEL, n_fft=N_FFT, sr=SR, fmin=60.0, fmax=7600.0):
    def hz2mel(f): return 2595.0 * np.log10(1 + f / 700.0)
    def mel2hz(m): return 700.0 * (10 ** (m / 2595.0) - 1)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mel + 2)
    hz = mel2hz(mels)
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mel, len(bins)))
    for i in range(n_mel):
        l, c, r = hz[i], hz[i + 1], hz[i + 2]
        fb[i] = np.clip(np.minimum((bins - l) / (c - l), (r - bins) / (r - c)), 0, None)
    return fb.astype(np.float32)


MEL_FB = _mel_fb()
_WIN = np.hanning(N_FFT).astype(np.float32)


def frame_end_sample(k):
    return int(round((k + 1) * SR / FPS))


def _resample_filter(up=2, down=3):
    """scipy.signal.resample_poly's default filter: firwin(2*10*max+1, 1/max, window=('kaiser', 5.0)) * up."""
    mx = max(up, down)
    half = 10 * mx
    n = np.arange(2 * half + 1) - half
    fc = 1.0 / mx
    h = fc * np.sinc(fc * n) * np.kaiser(2 * half + 1, 5.0)
    h = h / h.sum()
    return (h * up).astype(np.float64), half


class VAD:
    """energy_vad, one frame at a time (identical arithmetic)."""

    def __init__(self, on_margin=18.0, off_margin=12.0, hang=6):
        self.on_m, self.off_m, self.hang = on_margin, off_margin, hang
        self.hist = []
        self.srt = []
        self.on = False
        self.hold = 0

    def step(self, loud):
        h = self.hist
        x = float(loud)
        h.append(x)
        bisect.insort(self.srt, x)
        if len(h) > 300:
            del self.srt[bisect.bisect_left(self.srt, h.pop(0))]
        srt = self.srt                       # np.percentile(h, 10), linear interpolation, on the sorted copy
        r = 0.1 * (len(srt) - 1)
        lo = int(r)
        floor = srt[lo] + (srt[min(lo + 1, len(srt) - 1)] - srt[lo]) * (r - lo)
        if loud > floor + self.on_m:
            self.on = True
            self.hold = self.hang
        elif loud < floor + self.off_m:
            if self.hold > 0:
                self.hold -= 1
            else:
                self.on = False
        return 1.0 if self.on else 0.0


class StreamFeat:
    def __init__(self):
        self.h, self.half = _resample_filter()
        self.h0 = self.h[0::2]          # taps for 16 kHz samples m even
        self.h1 = self.h[1::2]          # m odd
        self.reset()

    def reset(self):
        self.x = np.zeros(0, np.float64)     # 24 kHz input, x[0] = global sample self.x0
        self.x0 = None
        self.y = np.zeros(0, np.float32)     # 16 kHz output, y[0] = global sample self.y0
        self.y0 = None
        self.m_next = None                   # next 16 kHz sample to produce
        self.next_frame = None               # next 30 fps frame to emit
        self.vad = VAD()

    def start(self, s0):
        """Begin a timeline at 24 kHz sample s0 (everything before it counts as silence)."""
        self.reset()
        self.x0 = int(s0)
        self.m_next = int(math.ceil(2 * self.x0 / 3))
        self.y0 = self.m_next
        # the frame containing the start (its window is partly zeros = silence before the timeline)
        self.next_frame = max(0, (self.m_next * FPS) // SR)

    def push(self, s0, pcm):
        """pcm int16 (or float in [-1, 1]) contiguous at 24 kHz global sample s0 (gaps must be pushed as zeros)."""
        if self.x0 is None:
            self.start(s0)
        p = np.asarray(pcm)
        x = p.astype(np.float64) / 32768.0 if p.dtype == np.int16 else p.astype(np.float64)
        end_have = self.x0 + self.x.size
        if s0 > end_have:                       # a hole: zeros
            x = np.concatenate([np.zeros(s0 - end_have), x])
        elif s0 < end_have:                     # overlap: drop what we have
            x = x[end_have - s0:]
        if x.size:
            self.x = np.concatenate([self.x, x])
        self._resample()
        return self._frames()

    def _resample(self):
        xe = self.x0 + self.x.size              # inputs known: [x0, xe)
        # zero-phase polyphase of upsample-2 / FIR / downsample-3: u = 3m + half (the filter centre on the 48 kHz grid),
        # y[m] = sum_q h[par + 2q] * x[floor(u/2) - q], par = u % 2  (needs 0.3 ms of input past the output instant)
        hf = self.half
        m_last = (2 * (xe - 1) - hf) // 3       # largest m with floor((3m + half)/2) <= xe - 1
        while (3 * (m_last + 1) + hf) // 2 <= xe - 1:
            m_last += 1
        while m_last >= 0 and (3 * m_last + hf) // 2 > xe - 1:
            m_last -= 1
        m0 = self.m_next
        if m_last < m0:
            return
        ms = np.arange(m0, m_last + 1)
        L = self.h.size
        i_lo, i_hi = 3 * m0 + hf, 3 * m_last + hf          # needed outputs on the 48 kHz grid
        a = i_lo - (L - 1)                                  # first upsampled input index the filter touches
        seg = np.zeros(i_hi - a + 1)
        e0 = a + (a & 1)                                    # first even index >= a  (xu[2k] = x[k])
        ks = np.arange(e0 // 2, i_hi // 2 + 1)
        kv = ks - self.x0
        ok = kv >= 0
        seg[(2 * ks - a)[ok]] = self.x[kv[ok]]
        z = np.convolve(seg, self.h, "valid")               # z[i - i_lo] for i in [i_lo, i_hi]
        out = z[3 * (ms - m0)]
        self.y = np.concatenate([self.y, out.astype(np.float32)])
        self.m_next = m_last + 1
        # keep the input tail the filter still needs
        keep_from = (3 * self.m_next + hf) // 2 - L - 2
        cut = keep_from - self.x0
        if cut > 4096:
            self.x = self.x[cut:]
            self.x0 += cut

    def _frames(self):
        rows = []
        ye = self.y0 + self.y.size
        while frame_end_sample(self.next_frame) <= ye:
            e = frame_end_sample(self.next_frame) - self.y0
            seg = self.y[max(0, e - N_FFT):e]
            if seg.size < N_FFT:
                seg = np.concatenate([np.zeros(N_FFT - seg.size, np.float32), seg])
            spec = np.abs(np.fft.rfft(seg * _WIN)) ** 2
            mel = np.log(np.maximum(spec @ MEL_FB.T, 1e-6))
            s2 = self.y[max(0, e - HOP):e]
            if s2.size < HOP:
                s2 = np.concatenate([np.zeros(HOP - s2.size, np.float32), s2])
            rms = math.sqrt(float(np.mean(s2.astype(np.float64) ** 2)) + 1e-12)
            loud = max(20 * math.log10(rms + 1e-12), -100.0)
            v = self.vad.step(loud)
            rows.append(np.concatenate([mel, [loud, v]]).astype(np.float32))
            self.next_frame += 1
        cut = (ye - N_FFT - HOP - 8) - self.y0
        if cut > 8192:
            self.y = self.y[cut:]
            self.y0 += cut
        return np.stack(rows) if rows else np.zeros((0, 42), np.float32)
