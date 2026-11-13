"""Everything that makes the character alive besides the mouth.

Stepped once per video frame.  The only inputs are a seeded RNG stream, advanced frame by frame, and a
Prosody record built from audio the frame is already allowed to see.  Nothing here knows the clip's
length or what comes next.

Sign conventions: head_yaw > 0 turns toward screen right; head_pitch > 0 is nose down (a nod);
head_roll > 0 tilts the crown toward screen right; head_z > 0 leans toward camera.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .schema import FPS, IDX

DT = 1.0 / FPS
STYLES_PATH = Path(__file__).with_name("styles.json")


@dataclass(frozen=True)
class Style:
    name: str = "fox"
    seed: int = 1
    jaw_gain: float = 1.0
    round_gain: float = 1.0
    spread_gain: float = 1.0
    head_gain: float = 1.0      # drift and turn amplitude
    head_speed: float = 7.0     # head follow rate, rad/s
    nod_gain: float = 1.0
    nod_hz: float = 2.1         # how fast this head nods and settles
    nod_chance: float = 0.7     # how many stressed syllables earn a nod
    gaze_gain: float = 1.0
    fixation: float = 1.0       # fixation hold multiplier
    blink_rate: float = 16.0    # blinks per minute
    double_blink: float = 0.12
    ear_gain: float = 1.0
    ear_flick_rate: float = 6.0     # idle flicks per minute per ear
    ear_onset_flick: float = 0.1    # chance a syllable onset flicks an ear
    brow_gain: float = 1.0
    smile_base: float = 0.18
    cheek_gain: float = 1.0
    nostril_gain: float = 1.0
    nostril_twitch_rate: float = 1.0    # per minute
    breath_rate: float = 0.25   # Hz at rest
    breath_gain: float = 1.0
    listen_tilt: float = 0.05   # head roll while listening, radians


def load_style(name: str) -> Style:
    table = json.loads(STYLES_PATH.read_text())
    known = {f.name for f in fields(Style)}
    return Style(name=name, **{k: v for k, v in table[name].items() if k in known})


@dataclass(slots=True)
class Prosody:
    level: float            # 0..1 loudness
    voice: float            # 0..1 voiced loudness
    onset: float            # 0..1 strongest syllable onset this frame
    stress: float           # 0..1 how loud that onset's syllable is against the running level
    pitch_rise: float       # semitones above the recent pitch, this frame's maximum
    phrase_start: bool
    phrase_end: float       # length in seconds of the phrase that just ended, else 0
    silence: float          # seconds since sound
    sealed: bool
    jaw: float
    round: float


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class Drift:
    """Causal value noise: quintic blend between knots drawn from a seeded stream."""

    def __init__(self, rng: random.Random, period: float) -> None:
        self.rng, self.period = rng, period
        self.a, self.b = rng.gauss(0, 1), rng.gauss(0, 1)
        self.t = rng.random()
        self.span = period

    def step(self) -> float:
        self.t += DT / self.span
        if self.t >= 1.0:
            self.t -= 1.0
            self.a, self.b = self.b, self.rng.gauss(0, 1)
            self.span = self.period * (0.7 + 0.6 * self.rng.random())
        t = self.t
        return self.a + (self.b - self.a) * t * t * t * (t * (t * 6 - 15) + 10)


class Octaves:
    def __init__(self, rng: random.Random, period: float) -> None:
        self.parts = [(Drift(rng, period), 0.62), (Drift(rng, period * 0.47), 0.28),
                      (Drift(rng, period * 0.21), 0.10)]

    def step(self) -> float:
        return sum(d.step() * g for d, g in self.parts)


class Kick:
    """Damped oscillator hit by impulses.  Used for nods and brow raises."""
    __slots__ = ("x", "v", "w", "zeta")

    def __init__(self, hz: float, zeta: float) -> None:
        self.x = self.v = 0.0
        self.w = 2 * math.pi * hz
        self.zeta = zeta

    def hit(self, amount: float) -> None:
        self.v += amount * self.w

    def step(self, target: float = 0.0) -> float:
        for _ in range(2):      # two substeps keep semi-implicit Euler honest at 30 fps
            a = -self.w * self.w * (self.x - target) - 2 * self.zeta * self.w * self.v
            self.v += a * DT / 2
            self.x += self.v * DT / 2
        return self.x


class Follow:
    """Critically damped follower at frame rate."""
    __slots__ = ("x", "v")

    def __init__(self, x: float = 0.0) -> None:
        self.x, self.v = x, 0.0

    def step(self, target: float, w: float) -> float:
        e = math.exp(-w * DT)
        y = self.x - target
        tmp = (self.v + w * y) * DT
        self.x = target + (y + tmp) * e
        self.v = (self.v - w * tmp) * e
        return self.x


class Blinker:
    def __init__(self, style: Style, rng: random.Random) -> None:
        self.s, self.rng = style, rng
        self.since = 1.0 + rng.random() * 1.5
        self.curve: list[float] = []
        self.again = 0
        self.boost = 0.0
        self.asym = 0.0

    def nudge(self, seconds: float) -> None:
        self.boost = max(self.boost, seconds)

    def _start(self) -> None:
        rng = self.rng
        close = (0.55, 1.0)
        hold = (1.0,) * rng.choice((0, 0, 1))
        open_ = rng.choice(((0.62, 0.2), (0.72, 0.36, 0.1), (0.78, 0.45, 0.14)))
        self.curve = list(close + hold + open_)
        self.asym = rng.uniform(-0.1, 0.1)
        self.since = 0.0
        if rng.random() < self.s.double_blink:
            self.again = rng.randint(2, 4)

    def step(self, listening: float) -> tuple[float, float]:
        self.since += DT
        self.boost = max(0.0, self.boost - DT)
        if not self.curve:
            if self.again > 0:
                self.again -= 1
                if self.again == 0:
                    self._start()
            else:
                ready = clamp((self.since - 0.8) / 1.0, 0.0, 1.0)
                hazard = self.s.blink_rate / 60.0 * ready * 1.25
                hazard *= 1.0 + 0.25 * listening
                if self.boost > 0.0:
                    hazard = max(hazard, 2.4 * clamp((self.since - 0.5) / 0.4, 0.0, 1.0))
                if self.rng.random() < 1.0 - math.exp(-hazard * DT):
                    self._start()
        if not self.curve:
            return 0.0, 0.0
        x = self.curve.pop(0)
        skew = self.asym * x * (1.0 - x) * 4.0
        return clamp(x + skew, 0.0, 1.0), clamp(x - skew, 0.0, 1.0)


class Life:
    def __init__(self, style: Style) -> None:
        self.s = style
        seed = style.seed
        stream = lambda tag: random.Random(f"{seed}:{tag}")
        self.r_look, self.r_kick, self.r_ear, self.r_misc = (
            stream("look"), stream("kick"), stream("ear"), stream("misc"))
        d = stream("drift")
        self.yaw_d, self.pitch_d, self.roll_d = Octaves(d, 4.3), Octaves(d, 3.4), Octaves(d, 5.3)
        self.sway_d = Octaves(d, 6.1)
        self.ear_ld, self.ear_rd = Octaves(d, 3.1), Octaves(d, 3.7)
        self.brow_d, self.smile_d = Drift(d, 2.9), Drift(d, 5.5)
        self.lid_d = Drift(d, 2.3)
        self.blink = Blinker(style, stream("blink"))

        self.t = 0.0
        self.look = [0.0, 0.0]                  # where attention points, gaze units
        self.head_look = [0.0, 0.0]             # the head target, arrives late so eyes lead
        self.head_delay = 0
        self.head_pending: list[float] | None = None
        self.fix_left = 0.3 + self.r_look.random()
        self.micro_left = 0.4
        self.micro = [0.0, 0.0]
        self.gaze_x, self.gaze_y = Follow(), Follow()
        self.yaw, self.pitch, self.roll = Follow(), Follow(), Follow()
        self.nod = Kick(style.nod_hz, 0.5)
        self.sway_kick = Kick(1.4, 0.6)
        self.nod_wait = 0.0
        self.nod_due = -1
        self.first = True
        self.activity = Follow(0.75)
        self.tilt = Follow()
        self.tilt_delay = 1.0
        self.tilt_ramp = 1.0
        self.nod_size = 0.0
        self.listen_nod_wait = 2.0 + 2.0 * self.r_kick.random()
        self.listening = Follow()
        self.speaking = Follow()
        self.brow = Kick(2.4, 0.75)
        self.brow_skew = 0.0
        self.brow_wait = 0.0
        self.knit = Follow()
        self.think_d = Drift(d, 3.3)
        self.flick_l, self.flick_r = 0.0, 0.0
        self.flick_lv, self.flick_rv = 0.0, 0.0
        self.ear_wait_l, self.ear_wait_r = 0.0, 0.0
        self.ear_l, self.ear_r = Follow(), Follow()

        self.lung = Follow(0.0)
        self.breath_phase = self.r_misc.random() * math.tau
        self.inhale = 0.0
        self.inhale_size = 0.0
        self.inhale_to = 0.0
        self.exhale_to = 0.0
        self.exhaling = False
        self.sniff = 0.0
        self.sniff_v = 0.0
        self.jaw_slow = Follow()
        self.smile = Follow(style.smile_base)

    def step(self, p: Prosody, out: np.ndarray) -> None:
        s = self.s
        self.t += DT
        speaking = 0.0 if p.sealed else 1.0
        listening = clamp((p.silence - 0.7) / 0.8, 0.0, 1.0) if p.sealed else 0.0
        lis = self.listening.step(listening, 2.6)
        spk = self.speaking.step(speaking, 7.0)

        self._attention(p, lis, spk)
        self._head(p, lis, spk, out)
        self._breath(p, spk, out)
        if p.phrase_end > 0.0:
            self.blink.nudge(0.5)
        blink_l, blink_r = self.blink.step(lis)
        lid = 0.035 + 0.02 * self.lid_d.step()
        lid = clamp(lid - 0.03 * lis, 0.0, 0.1)
        out[IDX["blink_l"]] = max(blink_l, lid)
        out[IDX["blink_r"]] = max(blink_r, lid)
        self._brows(p, lis, spk, out)
        self._ears(p, lis, spk, out)
        self._face(p, lis, spk, out)

    # ------------------------------------------------------------------ gaze and head turns
    def _attention(self, p: Prosody, lis: float, spk: float) -> None:
        s, r = self.s, self.r_look
        self.fix_left -= DT
        if self.fix_left <= 0.0:
            hold = math.exp(r.gauss(math.log(1.1), 0.55)) * s.fixation
            self.fix_left = clamp(hold, 0.35, 4.5)
            away = 0.26 * (1.0 - lis) + 0.10
            if r.random() < 0.45 + 0.3 * lis:
                new = [r.gauss(0, 0.08), r.gauss(0, 0.05)]      # back toward the viewer
            else:
                new = [clamp(r.gauss(0, away), -0.6, 0.6), clamp(r.gauss(0.03, away * 0.5), -0.3, 0.35)]
            jump = math.hypot(new[0] - self.look[0], new[1] - self.look[1])
            self.look = new
            if jump > 0.2:
                self.head_pending = new
                self.head_delay = r.randint(2, 4)      # eyes go first, head a beat later
                if r.random() < 0.55:
                    self.blink.nudge(0.2)
        if self.head_pending is not None:
            self.head_delay -= 1
            if self.head_delay <= 0:
                self.head_look = self.head_pending
                self.head_pending = None
        self.micro_left -= DT
        if self.micro_left <= 0.0:
            self.micro_left = 0.3 + r.random() * 0.8
            self.micro = [r.gauss(0, 0.018), r.gauss(0, 0.014)]

    def _head(self, p: Prosody, lis: float, spk: float, out: np.ndarray) -> None:
        s, r = self.s, self.r_kick
        activity = 0.75 + 0.5 * spk * clamp(p.level * 1.4, 0.0, 1.0) - 0.15 * lis
        g = s.head_gain * self.activity.step(activity, 0.35 * s.head_speed)

        self.nod_wait -= DT
        if p.phrase_start:
            self.nod.hit(-0.05 * s.nod_gain * (0.6 + 0.8 * r.random()))
            self.sway_kick.hit(r.gauss(0, 0.03) * s.head_gain)
            self.nod_wait = 0.16
        elif p.onset > 0.0 and p.stress > 0.45 and self.nod_wait <= 0.0:
            if r.random() < s.nod_chance:
                strength = p.stress * p.stress * (0.55 + 0.45 * p.onset)
                self.nod_due = r.randint(0, 3)      # each head finds the beat in its own time
                self.nod_size = 0.10 * s.nod_gain * strength * (0.7 + 0.6 * r.random())
                self.sway_kick.hit(r.gauss(0, 0.03) * s.head_gain * strength)
                self.nod_wait = 0.3 + 0.5 * r.random()
            else:
                self.nod_wait = 0.1 + 0.2 * r.random()
        if self.nod_due >= 0:
            if self.nod_due == 0:
                self.nod.hit(self.nod_size)
            self.nod_due -= 1

        if lis > 0.6:
            self.listen_nod_wait -= DT
            if self.listen_nod_wait <= 0.0:
                self.listen_nod_wait = 1.5 + 2.6 * r.random()
                self.nod.hit(0.06 * s.nod_gain * (0.7 + 0.6 * r.random()))
                if r.random() < 0.3:        # mm-hm: a quick second dip
                    self.listen_nod_wait = 0.42
        nod = self.nod.step()
        sway = self.sway_kick.step()

        yaw_n, pitch_n, roll_n = self.yaw_d.step(), self.pitch_d.step(), self.roll_d.step()
        yaw_t = 0.07 * g * yaw_n + 0.10 * s.head_gain * self.head_look[0] + sway
        pitch_t = 0.02 * g * pitch_n + 0.06 * s.head_gain * self.head_look[1]
        if p.phrase_start:          # each head settles into its listening tilt in its own time
            self.tilt_delay = 0.7 + 2.5 * r.random()
            self.tilt_ramp = 0.6 + 1.2 * r.random()
        tilt = clamp((p.silence - self.tilt_delay) / self.tilt_ramp, 0.0, 1.0) if p.sealed else 0.0
        tilt = self.tilt.step(tilt, 0.25 * s.head_speed)
        roll_t = 0.022 * g * roll_n + 0.28 * yaw_t + tilt * s.listen_tilt
        if self.first:              # start already posed: no swing into place on the first frames
            self.first = False
            self.yaw.x, self.pitch.x, self.roll.x = yaw_t, pitch_t, roll_t
        yaw = self.yaw.step(yaw_t, s.head_speed)
        pitch = self.pitch.step(pitch_t, s.head_speed * 1.1)
        roll = self.roll.step(roll_t, s.head_speed * 0.8)
        out[IDX["head_yaw"]] = yaw
        out[IDX["head_pitch"]] = pitch + nod
        out[IDX["head_roll"]] = roll
        out[IDX["head_x"]] = 0.012 * s.head_gain * self.sway_d.step() + 0.16 * yaw
        out[IDX["head_y"]] = -0.10 * (pitch + nod) + 0.004 * s.breath_gain * self.lung.x
        out[IDX["head_z"]] = 0.018 * lis + 0.05 * spk * p.level * 0.2

        # eyes: fixation target, minus part of the head's turn so they stay on target
        gx = s.gaze_gain * self.look[0] + self.micro[0] - 2.2 * (yaw - self.head_look[0] * 0.10)
        gy = s.gaze_gain * self.look[1] + self.micro[1] + 4.0 * nod
        if self.t <= DT * 1.5:
            self.gaze_x.x, self.gaze_y.x = gx, gy
        out[IDX["gaze_x"]] = clamp(self.gaze_x.step(gx, 55.0), -1.0, 1.0)
        out[IDX["gaze_y"]] = clamp(self.gaze_y.step(gy, 55.0), -1.0, 1.0)

    # ------------------------------------------------------------------ breathing
    def _breath(self, p: Prosody, spk: float, out: np.ndarray) -> None:
        s = self.s
        if p.phrase_end > 0.0:
            self.inhale = 0.45
            self.inhale_size = clamp(0.35 + p.phrase_end / 4.0, 0.35, 1.0)
            self.inhale_to = clamp(self.lung.x + 1.5 * self.inhale_size, -0.2, 0.95)
            self.sniff_v += 0.14 * self.inhale_size
        if self.inhale > 0.0:
            self.inhale -= DT
            target = self.inhale_to
            w = 9.0
            if self.inhale <= 0.0:
                self.breath_phase = 0.5 * math.pi    # roll into the resting cycle from the top
                self.exhaling = False
        elif spk > 0.5:
            if not self.exhaling:
                self.exhaling = True
                self.exhale_to = self.lung.x
            self.exhale_to = max(-0.85, self.exhale_to - 0.32 * DT)     # air leaves as words do
            target = self.exhale_to
            w = 5.0
        else:
            self.exhaling = False
            self.breath_phase += math.tau * s.breath_rate * DT
            sn = math.sin(self.breath_phase)
            target = 0.62 * sn + 0.12 * sn * abs(sn)
            w = 5.0
        out[IDX["breath"]] = clamp(self.lung.step(target, w) * s.breath_gain, -1.0, 1.0)

    # ------------------------------------------------------------------ brows
    def _brows(self, p: Prosody, lis: float, spk: float, out: np.ndarray) -> None:
        s, r = self.s, self.r_misc
        self.brow_wait -= DT
        rise = clamp((p.pitch_rise - 1.2) / 3.0, 0.0, 1.0)
        emph = max(rise, 0.75 * p.stress * p.onset)
        if emph > 0.25 and self.brow_wait <= 0.0:
            self.brow.hit(1.3 * s.brow_gain * emph * (0.75 + 0.5 * r.random()))
            self.brow_skew = r.uniform(-0.3, 0.3)
            self.brow_wait = 0.3 + 0.35 * r.random()
        elif p.phrase_end > 0.0:
            self.brow.hit(0.3 * s.brow_gain)
        base = 0.05 * s.brow_gain * self.brow_d.step() + 0.14 * lis
        think = clamp((p.silence - 2.5) / 2.0, 0.0, 1.0) if p.sealed else 0.0
        wander = clamp(0.5 + 0.55 * self.think_d.step(), 0.0, 1.0)
        knit = self.knit.step(0.42 * think * wander, 4.0)
        b = self.brow.step(base - 0.18 * knit)
        out[IDX["brow_l"]] = clamp(b * (1.0 + self.brow_skew), -1.0, 1.0)
        out[IDX["brow_r"]] = clamp(b * (1.0 - self.brow_skew), -1.0, 1.0)
        out[IDX["brow_knit"]] = clamp(knit, 0.0, 1.0)

    # ------------------------------------------------------------------ ears
    def _ears(self, p: Prosody, lis: float, spk: float, out: np.ndarray) -> None:
        s, r = self.s, self.r_ear
        self.ear_wait_l -= DT
        self.ear_wait_r -= DT
        idle = s.ear_flick_rate / 60.0 * DT
        go_l = r.random() < idle
        go_r = r.random() < idle
        if p.onset > 0.3:
            go_l = go_l or r.random() < s.ear_onset_flick * p.onset
            go_r = go_r or r.random() < s.ear_onset_flick * p.onset
        if p.phrase_start:          # sound came back after quiet: both ears notice
            go_l = go_l or r.random() < 0.55
            go_r = go_r or r.random() < 0.55
        if go_l and self.ear_wait_l <= 0.0:
            self.flick_lv += (0.6 + 0.4 * r.random()) * 0.7
            self.ear_wait_l = 0.55 + r.random()
        if go_r and self.ear_wait_r <= 0.0:
            self.flick_rv += (0.6 + 0.4 * r.random()) * 0.7
            self.ear_wait_r = 0.55 + r.random()
        self.flick_l, self.flick_lv = self._flick(self.flick_l, self.flick_lv)
        self.flick_r, self.flick_rv = self._flick(self.flick_r, self.flick_rv)

        attn = 0.34 * lis - 0.08 * spk + 0.12 * p.level
        l_t = s.ear_gain * (0.20 * self.ear_ld.step() + attn)
        r_t = s.ear_gain * (0.20 * self.ear_rd.step() + attn)
        if self.t <= DT * 1.5:
            self.ear_l.x, self.ear_r.x = l_t, r_t
        out[IDX["ear_l"]] = clamp(self.ear_l.step(l_t, 5.0), -1.0, 1.0)
        out[IDX["ear_r"]] = clamp(self.ear_r.step(r_t, 5.0), -1.0, 1.0)
        out[IDX["ear_flick_l"]] = clamp(self.flick_l, 0.0, 1.0)
        out[IDX["ear_flick_r"]] = clamp(self.flick_r, 0.0, 1.0)

    @staticmethod
    def _flick(x: float, v: float) -> tuple[float, float]:
        x = x + v
        return x * 0.72, v * 0.25

    # ------------------------------------------------------------------ cheeks, nose, smile
    def _face(self, p: Prosody, lis: float, spk: float, out: np.ndarray) -> None:
        s, r = self.s, self.r_misc
        smile_t = s.smile_base + 0.06 * self.smile_d.step() + 0.13 * lis + 0.07 * spk * p.level
        smile = self.smile.step(clamp(smile_t, 0.0, 1.0), 2.2)
        jaw_slow = self.jaw_slow.step(p.jaw * p.voice, 14.0)
        out[IDX["smile"]] = clamp(smile, 0.0, 1.0)
        out[IDX["cheek"]] = clamp(s.cheek_gain * (0.5 * jaw_slow + 0.55 * smile), 0.0, 1.0)

        if r.random() < s.nostril_twitch_rate / 60.0 * DT:
            self.sniff_v += 0.22 + 0.2 * r.random()
        self.sniff += self.sniff_v
        self.sniff_v *= 0.35
        self.sniff *= 0.8
        inhale_flare = 0.45 * clamp(self.inhale * 3.0, 0.0, 1.0) * self.inhale_size
        drive = 0.15 * p.onset * p.stress + 0.06 * clamp(self.lung.v, 0.0, 2.0)
        out[IDX["nostril"]] = clamp(s.nostril_gain * (self.sniff + inhale_flare + drive), 0.0, 1.0)
        out[IDX["voice"]] = clamp(p.voice, 0.0, 1.0)
