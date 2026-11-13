"""Streaming audio -> animation parameters.  Strictly causal, chunk-size independent.

    anim = StreamingAnimator("fox")
    for chunk in chunks:
        for frame in anim.push(chunk):      # int16 or float PCM, 24 kHz mono, any chunk size
            render(frame.index, frame.params)
    for frame in anim.flush():              # end of stream only
        render(frame.index, frame.params)

Timing grid (samples, independent of how the caller chunks the audio):
    hop h ends at sample h * HOP (200 = 8.33 ms) and is processed once the cursor has reached it.
    frame i is latched right after hop 4 * i + 8, i.e. from audio[: 800 * i + 1600], and released
    when the cursor reaches 800 * i + 1608 (t + LOOKAHEAD_MS).
Nothing reads the buffer anywhere except the WIN samples ending at the hop being processed, and all
state advances once per hop or once per frame, so identical audio gives identical bytes.

How the 67 ms lookahead is spent: the newest analysis window is centred about 55 ms after the frame's
time.  The jaw spring's own delay (2 / omega, 30-50 ms) eats most of that, which leaves the mouth
leading the voice by a few milliseconds to a frame.  Press and jaw therefore start moving before the
consonant is heard, as a real speaker's do.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .dsp import HOP, WIN, Analyzer, HopFeat
from .life import Life, Prosody, Style, load_style
from .schema import FPS, IDX, LOOKAHEAD_MS, NPARAMS, SAMPLE_RATE

SPF = SAMPLE_RATE // FPS                                # 800 samples per video frame
HOPS_PER_FRAME = SPF // HOP                             # 4
LOOK_SAMPLES = round(SAMPLE_RATE * LOOKAHEAD_MS / 1000)  # 1608
LOOK_HOPS = LOOK_SAMPLES // HOP                         # 8
DT = HOP / SAMPLE_RATE

# Level tracking, dB on the 300-3500 Hz band
REF_PRIOR_DB = -28.0        # low on purpose: attack is fast, so the first syllable is never undersized
REF_MAX_STEP_DB = 15.0
REF_ATTACK = 1 - math.exp(-DT / 0.040)
REF_RELEASE = 1 - math.exp(-DT / 3.0)
ABS_GATE_DB = -62.0
GATE_BELOW_REF_DB = 38.0
GATE_ABOVE_NOISE_DB = 9.0
NOISE_RISE_DB = 1.0 * DT    # noise floor follows minima, climbs 1 dB/s
RANGE_DB = 24.0             # the band of loudness the jaw maps over
PEAK_DECAY_DB = 42.0 * DT   # syllable peak follower

HOLD_HOPS = 13              # ~108 ms below the gate seals the mouth
SNAP = 0.004                # when sealed, a channel this close to 0 becomes exactly 0

# Mouth spring rates, rad/s.  Delay is 2 / omega.
W_JAW_OPEN = 62.0
W_JAW_CLOSE = 44.0
W_JAW_SNAP = 80.0           # closing into a bilabial
W_LIP = 38.0
W_PRESS_ON = 150.0
W_PRESS_OFF = 60.0
W_TONGUE = 75.0
W_DENTAL = 70.0
W_SEAL = 40.0

CLOSURE_REL_DB = -22.0      # this far below the syllable peak is an oral closure
MIN_CLOSURE_HOPS = 3
MAX_CLOSURE_HOPS = 36       # a longer gap is a pause, not a stop
DEEP_CLOSURE_DB = 34.0      # shallower dips are nasals and glides
PRESS_HOLD_HOPS = 4         # measured on the hop clock, so longer holds sit on the vowel itself
TONGUE_HOLD_HOPS = 4
PAUSE_HOPS = 22             # a closure this long is a pause unless the onset is a sharp labial


def sat(x: float) -> float:
    return 0.0 if x <= 0.0 else 1.0 if x >= 1.0 else x


def smooth(x: float) -> float:
    x = sat(x)
    return x * x * (3.0 - 2.0 * x)


class Spring:
    """Critically damped second-order follower, exact for a target held over one hop."""
    __slots__ = ("x", "v")

    def __init__(self) -> None:
        self.x = 0.0
        self.v = 0.0

    def step(self, target: float, omega: float) -> float:
        e = math.exp(-omega * DT)
        y = self.x - target
        tmp = (self.v + omega * y) * DT
        self.x = target + (y + tmp) * e
        self.v = (self.v - omega * tmp) * e
        return self.x

    def seal(self) -> None:
        if abs(self.x) < SNAP:
            self.x = 0.0
            self.v = 0.0


class Median:
    """Running median of the last n accepted values."""
    __slots__ = ("vals", "n")

    def __init__(self, n: int, init: float) -> None:
        self.vals = [init] * n
        self.n = n

    def push(self, v: float) -> float:
        self.vals = self.vals[1:] + [v]
        return sorted(self.vals)[self.n // 2]

    @property
    def value(self) -> float:
        return sorted(self.vals)[self.n // 2]


class MouthModel:
    """One step per hop.  Holds every piece of speech state; exposes per-frame Prosody for Life."""

    def __init__(self, style: Style) -> None:
        self.style = style
        self.ref = REF_PRIOR_DB
        self.ref_seen = 0
        self.noise = -120.0
        self.peak = -140.0
        self.silent = 10 ** 6
        self.jaw, self.round, self.spread = Spring(), Spring(), Spring()
        self.press, self.tongue, self.labdent = Spring(), Spring(), Spring()
        self.channels = (self.jaw, self.round, self.spread, self.press, self.tongue, self.labdent)
        self.f1 = Median(5, 500.0)
        self.f2 = Median(5, 1500.0)
        self.round_est = 0.0
        self.spread_est = 0.0
        self.closure_hops = 0
        self.closure_depth = 0.0
        self.prev_vowel = -140.0
        self.release_at = -1
        self.release_cen0 = 0.0
        self.release_cen = [0.0, 0.0]
        self.release_f2 = 0.0
        self.closure_len = 0
        self.seal_onset = False
        self.press_hold = 0
        self.tongue_hold = 0
        self.fast = -140.0
        self.slow = -140.0
        self.onset_wait = 0
        self.level = Spring()
        self.voice = Spring()
        # prosody
        self.phrase_hops = 0
        self.f0_log = 0.0
        self.f0_mean = 0.0
        self.f0_slow = 0.0
        self.onset = 0.0
        self.stress = 0.0
        self.pitch_rise = 0.0
        self.phrase_end_len = 0.0
        self.phrase_start = False
        self.was_sealed = True

    # ------------------------------------------------------------------ one hop
    def step(self, f: HopFeat) -> None:
        db_v = f.db_vowel
        self.noise = min(f.db, self.noise + NOISE_RISE_DB)
        gate = max(ABS_GATE_DB, self.noise + GATE_ABOVE_NOISE_DB, self.ref - GATE_BELOW_REF_DB)
        active = f.db > gate
        self.silent = 0 if active else self.silent + 1
        sealed = self.silent >= HOLD_HOPS

        if db_v > self.ref:
            self.ref += (db_v - self.ref) * REF_ATTACK
        elif active and f.voiced > 0.6:
            # Slides down like a running mean at first, so a quiet source is learned within the
            # first words, then settles to the slow release.  The clamp keeps it on the peaks.
            self.ref_seen += 1
            rate = max(REF_RELEASE, 1.0 / (self.ref_seen + 40.0))
            self.ref += max(db_v - self.ref, -REF_MAX_STEP_DB) * rate

        self.peak = db_v if sealed else max(db_v, self.peak - PEAK_DECAY_DB)
        rel = db_v - self.peak if active else -60.0

        loud = sat((db_v - self.ref + RANGE_DB + 2.0) / RANGE_DB)
        vocalic = active and f.voiced > 0.5 and rel > -11.0
        if vocalic:
            if 180.0 < f.f1 < 1150.0:
                self.f1.push(f.f1)
            if f.f2 > 500.0:
                self.f2.push(f.f2)
            f1, f2 = self.f1.value, self.f2.value
            back = sat((f.db_low - f.db_mid - 19.0) / 6.0) * sat((1850.0 - f2) / 300.0)
            stressed = sat((loud - 0.5) / 0.25)     # reduced vowels get no lip gesture
            lips = max(sat((1220.0 - f2) / 220.0), back)
            self.round_est = lips * sat((700.0 - f1) / 130.0) * stressed
            self.spread_est = 0.85 * sat((f2 - 1750.0) / 480.0) * sat((850.0 - f1) / 200.0 + 0.3)
        else:
            self.round_est *= 0.985
            self.spread_est *= 0.985

        # jaw: loudness across the phrase times depth below the syllable peak times vowel height
        beat = smooth((rel + 12.5) / 8.5)
        height = sat((self.f1.value - 270.0) / 500.0)
        jaw_t = loud * beat * (0.40 + 0.68 * height)

        # consonants ------------------------------------------------------------------------
        fricative = active and f.hf_ratio > 0.5 and f.voiced < 0.97
        strong_fric = fricative and f.db > self.ref - 15.0
        weak_fric = fricative and not strong_fric
        spread_t = self.spread_est
        if strong_fric:
            spread_t = max(spread_t, 0.6)
            jaw_t = max(jaw_t, 0.07)

        # Stops are named by their release.  A lip closure and a tongue closure sound alike going in;
        # coming out, a labial goes straight to low-band voicing while /t k tS/ burst high.  The
        # release reaches this code about 55 ms before the frame that must show the closure.
        closed = rel < CLOSURE_REL_DB
        if closed:
            self.closure_hops = min(self.closure_hops + 1, MAX_CLOSURE_HOPS + 1)
            self.closure_depth = max(self.closure_depth, -rel)
        elif self.closure_hops:
            if MIN_CLOSURE_HOPS <= self.closure_hops <= MAX_CLOSURE_HOPS \
                    and self.closure_depth >= DEEP_CLOSURE_DB:
                self.release_at = 0
                self.closure_len = self.closure_hops
                self.release_cen0 = f.centroid
            self.closure_hops = 0
            self.closure_depth = 0.0
        elif self.release_at >= 0:
            self.release_at += 1
            if self.release_at == 1:
                self.release_cen = sorted((self.release_cen0, f.centroid))
                self.release_f2 = f.f2
            else:
                self.release_at = -1
                self._classify_release(f)
        if self.seal_onset:                 # one hop later: a /b/ stays low, a /t/ bursts high
            self.seal_onset = False
            if f.centroid < 600.0:
                self.press_hold = PRESS_HOLD_HOPS
        elif self.was_sealed and active and db_v - self.prev_vowel >= 22.0 and f.centroid < 450.0:
            self.seal_onset = True
        self.prev_vowel = db_v

        press_t = 0.0
        if self.press_hold > 0:
            press_t = 1.0
            self.press_hold -= 1
        tongue_t = 0.0
        if self.tongue_hold > 0:
            tongue_t = 1.0
            self.tongue_hold -= 1
        nasal = active and f.voiced > 0.7 and -20.0 < rel < -6.0 and 245.0 < f.centroid < 650.0
        if nasal and self.f1.value < 520.0:
            tongue_t = max(tongue_t, 0.6)

        labdent_t = 0.0
        if weak_fric:
            labdent_t = 0.85
            spread_t = min(spread_t, 0.3)

        jaw_t *= 1.0 - 0.9 * press_t

        # dynamics --------------------------------------------------------------------------
        if sealed:
            self.round_est = self.spread_est = 0.0
            self.press_hold = self.tongue_hold = 0
            for ch in self.channels:
                ch.step(0.0, W_SEAL)
                ch.seal()
        else:
            if jaw_t > self.jaw.x:
                w = W_JAW_OPEN
            else:
                w = W_JAW_SNAP if press_t > 0.5 else W_JAW_CLOSE
            self.jaw.step(jaw_t, w)
            self.round.step(self.round_est, W_LIP)
            self.spread.step(spread_t, W_LIP)
            self.press.step(press_t, W_PRESS_ON if press_t > self.press.x else W_PRESS_OFF)
            self.tongue.step(tongue_t, W_TONGUE)
            self.labdent.step(labdent_t, W_DENTAL)

        self._prosody(f, active, sealed, loud)

    def _classify_release(self, f: HopFeat) -> None:
        """Two hops into a stop release.  Waiting longer would eat the lead the press needs."""
        low, high = self.release_cen
        f2s = [v for v in (self.release_f2, f.f2) if v > 0.0]
        f2 = min(f2s) if f2s else 9999.0
        tilt = f.db_low - f.db_mid
        if low > 2300.0:
            self.tongue_hold = TONGUE_HOLD_HOPS
            return
        if low >= 1700.0 or high >= 2050.0 or f.db_low < self.ref - 28.0:
            return
        labial_locus = 600.0 < f2 < 1830.0 and tilt >= 5.0  # lips closed: F2 low, low band leads
        aspirated = high >= 1100.0 and tilt <= 8.0          # /p/ burst with no low-band lead
        long_pause = self.closure_len > PAUSE_HOPS
        if long_pause and high >= 450.0:
            return
        if labial_locus or aspirated:
            self.press_hold = PRESS_HOLD_HOPS

    def _prosody(self, f: HopFeat, active: bool, sealed: bool, loud: float) -> None:
        self.level.step(loud * (1.0 if active else 0.0), 45.0)
        self.voice.step(loud * f.voiced if active else 0.0, 45.0)
        db_v = max(f.db_vowel, -90.0)
        self.fast += (db_v - self.fast) * 0.5
        self.slow += (db_v - self.slow) * 0.09
        if self.onset_wait > 0:
            self.onset_wait -= 1
        flux = self.fast - self.slow
        if active and flux > 7.0 and self.onset_wait == 0:
            self.onset_wait = 14
            self.onset = max(self.onset, smooth(flux / 22.0))
            self.stress = max(self.stress, sat((f.db_vowel - self.ref + 13.0) / 13.0))

        if self.was_sealed and not sealed and active:
            self.phrase_start = True
            self.phrase_hops = 0
        if not sealed:
            self.phrase_hops += 1
        elif not self.was_sealed:
            self.phrase_end_len = self.phrase_hops * DT
        self.was_sealed = sealed

        if f.f0 > 0.0 and active:
            lf = math.log2(f.f0)
            if self.f0_mean == 0.0:
                self.f0_mean = self.f0_slow = lf
            while lf - self.f0_mean > 0.5:      # fold octave slips of the tracker
                lf -= 1.0
            while lf - self.f0_mean < -0.5:
                lf += 1.0
            self.f0_mean += (lf - self.f0_mean) * 0.004
            self.f0_slow += (lf - self.f0_slow) * 0.03
            self.f0_log += (lf - self.f0_log) * 0.3
            self.pitch_rise = max(self.pitch_rise, 12.0 * (self.f0_log - self.f0_slow))

    def frame_prosody(self) -> Prosody:
        hops_silent = min(self.silent, 10 ** 6)
        p = Prosody(
            level=self.level.x, voice=self.voice.x, onset=self.onset, stress=self.stress,
            pitch_rise=self.pitch_rise, phrase_start=self.phrase_start,
            phrase_end=self.phrase_end_len, silence=hops_silent * DT,
            sealed=self.silent >= HOLD_HOPS, jaw=self.jaw.x, round=self.round.x)
        self.onset = self.stress = self.pitch_rise = 0.0
        self.phrase_start = False
        self.phrase_end_len = 0.0
        return p


@dataclass(slots=True)
class FrameOut:
    index: int
    time: float
    params: np.ndarray      # float32, length NPARAMS, schema order


class StreamingAnimator:
    """leak_frames and leak_peek exist only so tests can prove they catch a causality leak."""

    def __init__(self, style: str | Style = "fox", *, leak_frames: int = 0,
                 leak_peek: bool = False) -> None:
        self.style = load_style(style) if isinstance(style, str) else style
        self.analyzer = Analyzer()
        self.mouth = MouthModel(self.style)
        self.life = Life(self.style)
        self._tail = np.zeros(WIN, dtype=np.float64)    # the WIN samples before _tail_end
        self._tail_end = 0
        self._pending = np.zeros(0, dtype=np.float64)   # samples at or after _tail_end
        self._cursor = 0                                # total samples received
        self._hops = 0
        self._frames = 0                                # frames latched
        self._emitted = 0
        self._queue: list[tuple[int, np.ndarray]] = []
        self._look_hops = LOOK_HOPS + leak_frames * HOPS_PER_FRAME
        self._look_samples = LOOK_SAMPLES + leak_frames * SPF
        self._leak_peek = leak_peek
        self._flushed_to: int | None = None

    def push(self, pcm: np.ndarray) -> list[FrameOut]:
        x = _to_float(pcm)
        self._pending = np.concatenate((self._pending, x)) if self._pending.size else x
        self._cursor += x.size
        while self._pending.size >= HOP:
            self._hop()
        return self._release()

    def flush(self) -> list[FrameOut]:
        """End of stream: treat what follows as digital silence and release every frame whose time
        lies inside the audio.  The stream is finished afterwards."""
        end = self._cursor
        n_frames = (end + SPF - 1) // SPF
        out: list[FrameOut] = []
        need = (n_frames - 1) * SPF + self._look_samples - end if n_frames else 0
        if need > 0:
            out = self.push(np.zeros(need, dtype=np.float64))
        return [f for f in out if f.index < n_frames]

    # ------------------------------------------------------------------ internals
    def _hop(self) -> None:
        step = self._pending[:HOP]
        window = np.concatenate((self._tail[HOP:], step))
        if self._leak_peek:         # deliberately broken: reads samples the hop has not reached
            extra = min(SPF, self._pending.size - HOP)
            window = np.concatenate((window, self._pending[HOP:HOP + extra]))[-WIN:]
        self._pending = self._pending[HOP:]
        self._tail = window
        self._tail_end += HOP
        self._hops += 1
        self.mouth.step(self.analyzer.analyze(window))
        if self._hops >= self._look_hops and \
                (self._hops - self._look_hops) % HOPS_PER_FRAME == 0:
            self._queue.append((self._frames, self._frame_vector()))
            self._frames += 1

    def _frame_vector(self) -> np.ndarray:
        m = self.mouth
        s = self.style
        p = np.zeros(NPARAMS, dtype=np.float64)
        prosody = m.frame_prosody()
        self.life.step(prosody, p)
        p[IDX["jaw"]] = sat(m.jaw.x * s.jaw_gain)
        p[IDX["round"]] = sat(m.round.x * s.round_gain)
        p[IDX["spread"]] = sat(m.spread.x * s.spread_gain)
        p[IDX["press"]] = sat(m.press.x)
        p[IDX["labdent"]] = sat(m.labdent.x)
        p[IDX["tongue"]] = sat(m.tongue.x)
        return p.astype(np.float32)

    def _release(self) -> list[FrameOut]:
        out: list[FrameOut] = []
        while self._queue:
            i, vec = self._queue[0]
            if i * SPF + self._look_samples > self._cursor:
                break
            self._queue.pop(0)
            self._emitted += 1
            out.append(FrameOut(i, i / FPS, vec))
        return out


def _to_float(pcm: np.ndarray) -> np.ndarray:
    x = np.asarray(pcm)
    if x.ndim != 1:
        x = x.reshape(-1)
    if x.dtype.kind in "iu":
        return x.astype(np.float64) / 32768.0
    return x.astype(np.float64)


__all__ = ["StreamingAnimator", "FrameOut", "LOOK_SAMPLES", "SPF", "HOP"]
