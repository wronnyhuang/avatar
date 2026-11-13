"""Streaming audio -> full parameter vector with real lips, for every character, animal or shape.

    anim = StreamingLipAnimator("orb", lookahead_ms=100, lead_frames=2)
    for chunk in chunks:
        for frame in anim.push(chunk):      # int16 or float PCM, 24 kHz mono, any chunk size
            render(frame.index, frame.params)
    for frame in anim.flush():              # end of stream only
        render(frame.index, frame.params)

The audio path of the mouth, with no loudness anywhere in it:

    audio -> log-mel, relative to its own recent peak -> GRU -> viseme weights at a chosen time ->
    this character's pose table -> one follower per column at 60 Hz -> 11 lip columns -> legacy columns

Life (blinks, head, gaze, brows, ears, cheek, smile) is StreamingAnimator's, bit for bit.  Its speech model still runs,
for prosody; none of it reaches a lip column, and tests/test_not_volume.py replaces it with noise to prove that.

Timing, in steps of 16.7 ms.  A is the audio delay: a lip column of frame i depends only on audio up to sample
800 i + 400 A, and frame i is latched at step 2 i + A.  L is the lead in video frames: the frame shows the sound heard
L frames after it.  So the sound shown is 2 L steps ahead of the frame and A - 2 L steps behind the newest audio.  A
follower of delay f steps needs a head f steps later than that, so column group g reads head A - 2 L - f_g, and
a slower character reads a head further on instead of drifting late.  Heads exist every 2 steps; between two of them the
logits are interpolated.

Closures spend the lead.  Lips travel to the seal over about two frames, so press starts when the sound shown reaches the
closure and arrives when the sound heard does, and the same closure reading, 2 L steps old, holds it to the burst heard.

Life keeps its own 67 ms.  With A under 67 ms the lip columns still obey A, and the whole frame goes out at t + 67 ms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bridge import legacy_from_lips
from .dsp import HOP
from .life import Style
from .poses import FINE, LIP_PARAMS, gains as pose_gains, load_table, pose_matrix
from .schema import FPS, IDX
from .stream import HOPS_PER_FRAME, LOOK_SAMPLES, SPF, FrameOut, StreamingAnimator, sat
from .visemes import N_MELS, NET_PATH, REL_FLOOR, STEP_HOPS, VISEMES, WIN as MEL_WIN, LevelNorm, LogMel, VisemeNet

LIP_IDX = [IDX[n] for n in LIP_PARAMS]
L = {n: i for i, n in enumerate(LIP_PARAMS)}
V = {n: i for i, n in enumerate(FINE)}
DT = STEP_HOPS * HOP / 24000.0
STEP_MS = 1000.0 * DT
SNAP = 0.004
DEFAULT_LOOKAHEAD_MS = 133
DEFAULT_LEAD = 2
# three nets of one shape, each with its capacity on the delays it serves: 4 percent lower loss and better closures than one
# net with every head.  The middle one is the shipped pair's, the others keep the sweep honest.
NET_BEFORE = NET_PATH.with_name("viseme_net_before.bin")        # delays -100 to 0 ms
NET_AFTER = NET_PATH.with_name("viseme_net_after.bin")          # delays 67 to 133 ms


def net_for(total: int) -> Path:
    """total = A - 2 L in steps, the net's context past the sound shown."""
    return NET_BEFORE if total <= 0 else NET_PATH if total <= 4 else NET_AFTER
SEALED = 0.9        # at this press the rig needs lip_open exactly 0

FAST_COLS = [L[n] for n in ("lip_open", "teeth_gap", "tongue_tip", "tongue_dental", "tongue_back")]
SLOW_COLS = [L[n] for n in ("lip_width", "lip_protrude", "lip_upper", "lip_lower")]
CONS = [V[n] for n in ("PP", "FF", "TH", "DD", "NN", "LL", "kk", "CH", "SS", "RR", "WW")]
OPEN_V = [V[n] for n in ("AA", "AE", "AH", "AX", "EH", "IH", "IY")]     # vowels whose lips are not rounded
ROUND_V = [V[n] for n in ("AO", "OW", "UH", "UW")]
ROUND_ALL = ROUND_V + [V[n] for n in ("WW", "RR", "CH")]
WIDE = [V[n] for n in ("AE", "EH", "IH", "IY", "SS")]
OLD15 = {"nn": ("NN", "LL"), "aa": ("AA", "AE", "AH", "AX"), "E": ("EH",), "ih": ("IH", "IY"),
         "oh": ("AO", "OW"), "ou": ("UH", "UW", "WW")}
TO15 = np.zeros((len(FINE), len(VISEMES)))
for _o, _n in enumerate(VISEMES):
    for _f in OLD15.get(_n, (_n,)):
        TO15[V[_f], _o] = 1.0


@dataclass(frozen=True)
class LipTune:
    """Set on the development sessions by tools/v2/tune.py, never on held-out clips.  Rates are rad/s."""
    sharpen_fast: float = 1.5   # posterior exponent before blending: decisive shapes, not an averaged mush
    sharpen_slow: float = 1.0
    veto: float = 0.95               # how far a sure vowel in the fast head overrules the slow head's lip shape
    cons_open: float = 0.3      # consonants' share of lip_open: several poses open as wide as /ih/, and this is the beat
    w_open_up: float = 140.0
    w_open_down: float = 90.0
    w_shape_in: float = 80.0
    w_shape_out: float = 80.0
    w_tongue: float = 140.0
    w_press_on: float = 70.0    # lips travel to the seal over about two frames
    w_press_off: float = 120.0  # and part quickly at the burst
    w_tuck_on: float = 70.0
    w_tuck_off: float = 90.0
    w_seal: float = 130.0       # every column, on the way to rest
    tongue_gamma: float = 0.5
    max_press: float = 0.28    # most press may rise per step: the lips travel to the seal, they do not slam
    max_open: float = 0.15      # most lip_open and teeth_gap may rise per step at speed 1: a jaw has mass
    max_shut: float = 0.2       # most they may fall per step, closures aside: press does that
    max_tongue: float = 0.4         # the tongue is quick, but it does not teleport
    max_shape: float = 0.17     # most the lip shape columns may move per step: this is the slow set, it never jumps
    close_lo: float = 0.3       # closure evidence maps to press through a smoothstep over this range
    close_hi: float = 0.7
    tuck_lo: float = 0.3
    tuck_hi: float = 0.6
    commit: float = 0.7         # press or tuck target this high goes all the way, however short the evidence
    tuck_steps: int = 2
    hold_lo: float = 0.25        # closure at the frame's own time keeps a seal through this range
    hold_hi: float = 0.65
    seal_steps: int = 4         # sealed lips stay at least this long, so two video frames sit sealed whatever the phase
    sil_on: float = 0.5         # silence weight that rests the mouth
    sil_off: float = 0.4        # and that lets it go
    sil_steps: int = 3


def delay_steps(w: float) -> float:
    """Group delay of two one-pole lags, in steps."""
    p = math.exp(-w * DT)
    return 2.0 * p / (1.0 - p)


def smoothstep(x: float, lo: float, hi: float) -> float:
    u = sat((x - lo) / max(hi - lo, 1e-3))
    return u * u * (3.0 - 2.0 * u)


class Follower:
    """Two one-pole lags in series, critically damped.  Each update is a convex mix, so the output stays inside the
    range of its past targets whatever the rate does between steps: no overshoot is possible."""
    __slots__ = ("a", "x")

    def __init__(self) -> None:
        self.a = 0.0
        self.x = 0.0

    def step(self, target: float, k: float, up: float = 9.0, down: float = 9.0) -> float:
        """up, down: the most the first lag may move in one step.  The second lag rounds the corners of that ramp."""
        d = k * (target - self.a)
        self.a += up if d > up else -down if d < -down else d
        self.x += k * (self.a - self.x)
        return self.x


def sharpen(p: np.ndarray, g: float) -> np.ndarray:
    w = p ** g
    return w / w.sum()


class LipModel:
    """The mouth.  One step per model step, from weights that describe one instant.  Holds no audio and no level."""

    def __init__(self, style: Style, character: str | None = None, tune: LipTune = LipTune(),
                 poses: np.ndarray | None = None) -> None:
        self.s = style
        self.t = tune
        self.P = pose_matrix(character) if poses is None else poses
        table = load_table(character)
        self.speed = float(table["speed"])
        self.gains = self._gains(style) * pose_gains(character)
        self.ch = [Follower() for _ in LIP_PARAMS]
        self.out = np.zeros(len(LIP_PARAMS))
        self.fast = np.zeros(len(FINE))
        self.fast[0] = 1.0
        self.resting = True
        self.sil_run = 0
        self.hold = 0
        self.in_seal = False
        self.going = False
        self.tuck_left = 0
        s = self.speed
        k = lambda w: 1.0 - math.exp(-w * s * DT)
        self.k_open = (k(tune.w_open_up), k(tune.w_open_down))
        self.k_shape = (k(tune.w_shape_in), k(tune.w_shape_out))
        self.k_tongue = k(tune.w_tongue)
        self.k_press = (k(tune.w_press_on), k(tune.w_press_off))
        self.k_tuck = (k(tune.w_tuck_on), k(tune.w_tuck_off))
        self.k_seal = k(tune.w_seal)
        self.up, self.down = tune.max_open * s, tune.max_shut * s
        self.shape_max = tune.max_shape * s
        self.press_up = tune.max_press * min(s, 1.0)        # a quick rig still travels, a heavy one takes longer
        self.tongue_max = tune.max_tongue * s
        self.limit = [9.0] * len(LIP_PARAMS)                # most a column may move per step on its way to rest
        for n in ("lip_open", "teeth_gap"):
            self.limit[L[n]] = self.down
        for n in ("lip_width", "lip_protrude", "lip_upper", "lip_lower"):
            self.limit[L[n]] = self.shape_max
        for n in ("tongue_tip", "tongue_dental", "tongue_back"):
            self.limit[L[n]] = self.tongue_max
        self.limit[L["lip_tuck"]] = self.press_up
        P = self.P
        self.Pfast = P[:, FAST_COLS]
        self.Pslow = P[:, SLOW_COLS]
        self.Pcons_open = P[CONS, L["lip_open"]]
        self.press_amp = float(P[V["PP"], L["lip_press"]])
        self.tuck_amp = float(P[V["FF"], L["lip_tuck"]])

    @staticmethod
    def _gains(s: Style) -> np.ndarray:
        g = np.ones(len(LIP_PARAMS))
        g[L["lip_open"]] = g[L["teeth_gap"]] = g[L["lip_lower"]] = s.jaw_gain
        g[L["lip_protrude"]] = s.round_gain
        return g

    def delays(self) -> dict[str, float]:
        """Follower delay per head reader, in steps, at this character's speed."""
        t, s = self.t, self.speed
        return dict(fast=delay_steps(0.5 * (t.w_open_up + t.w_open_down) * s),
                    slow=delay_steps(0.5 * (t.w_shape_in + t.w_shape_out) * s),
                    press=delay_steps(t.w_press_on * s), tuck=delay_steps(t.w_tuck_on * s))

    def reads(self, total: float, lead: int) -> dict[str, float]:
        """The delay in steps at which each group reads the net.  total = A - 2 L.  The lead pays for lip travel."""
        d = self.delays()
        return dict(fast=total - d["fast"], slow=total - d["slow"], press=total - max(0.0, d["press"] - 2 * lead),
                    tuck=total - max(0.0, d["tuck"] - 2 * lead))

    def step(self, fast: np.ndarray, slow: np.ndarray, close: float, tuck: float, no_audio: bool = False,
             close_now: float = 0.0) -> None:
        """no_audio: the sound this step shows is digital silence.  That is no signal at all, not a low level.
        close_now: closure at the frame's own time, which holds a seal that has started until the burst is heard."""
        t = self.t
        ch = self.ch
        self.fast = fast
        quiet = 1.0 if no_audio else float(fast[0])
        self.sil_run = self.sil_run + 1 if quiet >= t.sil_on else 0
        if self.sil_run >= t.sil_steps or no_audio:
            self.resting = True
        elif quiet < t.sil_off:
            self.resting = False
        if self.resting:
            self.going = False
            self.tuck_left = 0
            press = ch[L["lip_press"]]
            if self.hold > 0:           # a seal that was under way finishes its two frames
                self.hold -= 1
                press.step(1.0, self.k_press[0], self.press_up)
            else:
                self.in_seal = False
            for i, c in enumerate(ch):
                if c is press and self.hold > 0:
                    continue
                c.step(0.0, self.k_seal, self.limit[i], self.limit[i])
                if abs(c.x) < SNAP and abs(c.a) < SNAP:
                    c.a = c.x = 0.0
            self._publish()
            return

        # the seal the lips are heading for: teeth and opening fall towards it through their followers, never by a scale
        # on their output, so a jaw takes its own time to come up behind lips that have already met
        shut = 1.0 - sat(ch[L["lip_press"]].x)
        w = sharpen(fast, t.sharpen_fast)
        tf = np.einsum("i,ij->j", w, self.Pfast)
        tf[0] -= (1.0 - t.cons_open) * float(np.einsum("i,i->", w[CONS], self.Pcons_open))
        tf[0] *= shut
        tf[1] *= 0.2 + 0.8 * shut
        tf[2:] = np.maximum(tf[2:], 0.0) ** t.tongue_gamma      # a tongue tap is short, and it must still land
        for i, c in enumerate(FAST_COLS):
            f = ch[c]
            if i < 2:
                f.step(tf[i], self.k_open[0] if tf[i] > f.a else self.k_open[1], self.up, self.down)
            else:
                f.step(tf[i], self.k_tongue, self.tongue_max, self.tongue_max)

        # the fast head knows the vowel of this moment: where it is sure, the slow head's lean towards the next one stands down
        if t.veto > 0.0:
            slow = slow.copy()
            slow[ROUND_ALL] *= 1.0 - t.veto * float(np.einsum("i->", fast[OPEN_V]))
            slow[WIDE] *= 1.0 - t.veto * float(np.einsum("i->", fast[ROUND_V]))
        ts = np.einsum("i,ij->j", sharpen(slow, t.sharpen_slow), self.Pslow)
        for i, c in enumerate(SLOW_COLS):
            f = ch[c]
            grow = abs(ts[i]) > abs(f.a) or ts[i] * f.a < 0.0
            f.step(ts[i], self.k_shape[0] if grow else self.k_shape[1], self.shape_max, self.shape_max)

        # closure: a continuous map of the evidence.  Lips travel to the seal, stay there at least seal_steps once they arrive,
        # so two video frames sit sealed whatever the phase, and let go when the closure ends in the sound heard, which is the
        # burst.  No threshold starts or ends anything by itself, and nothing can re-trigger it.
        f = ch[L["lip_press"]]
        p = smoothstep(close, t.close_lo, t.close_hi)
        if p >= t.commit:           # evidence this sure sends the lips all the way: a short closure still gets its travel
            self.going = True
        if f.x >= 0.3:
            p = max(p, smoothstep(close_now, t.hold_lo, t.hold_hi))
        if self.going:
            p = 1.0
        if f.x >= SEALED:
            self.going = False
            if not self.in_seal:
                self.in_seal = True
                self.hold = t.seal_steps
        elif f.x < 0.5:
            self.in_seal = False
        if self.hold > 0:
            self.hold -= 1
            p = 1.0
        f.step(p, self.k_press[0] if p > f.a else self.k_press[1], self.press_up)

        q = smoothstep(tuck, t.tuck_lo, t.tuck_hi)
        if q >= t.commit:
            self.tuck_left = t.tuck_steps
        if self.tuck_left > 0:
            self.tuck_left -= 1
            q = 1.0
        f = ch[L["lip_tuck"]]
        f.step(q, self.k_tuck[0] if q > f.a else self.k_tuck[1], self.press_up, self.press_up)
        if ch[L["lip_press"]].x >= SEALED:      # sealed lips hold no opening, so letting go starts from 0, not from a jump
            for n in ("lip_open", "lip_upper", "lip_lower"):
                ch[L[n]].a = ch[L[n]].x = 0.0
        self._publish()

    def _publish(self) -> None:
        o = self.out
        ch = self.ch
        press = sat(ch[L["lip_press"]].x)
        tuck = sat(ch[L["lip_tuck"]].x)
        keep = sat(1.0 - press / SEALED)
        o[L["lip_press"]] = press * self.press_amp
        o[L["lip_tuck"]] = tuck * self.tuck_amp
        o[L["lip_open"]] = sat(ch[L["lip_open"]].x) * keep
        o[L["lip_upper"]] = sat(ch[L["lip_upper"]].x) * keep
        o[L["lip_lower"]] = sat(ch[L["lip_lower"]].x) * keep * (1.0 - 0.7 * tuck)
        o[L["teeth_gap"]] = sat(ch[L["teeth_gap"]].x)
        o[L["lip_width"]] = max(-1.0, min(1.0, ch[L["lip_width"]].x))
        o[L["lip_protrude"]] = sat(ch[L["lip_protrude"]].x)
        for n in ("tongue_tip", "tongue_dental", "tongue_back"):
            o[L[n]] = sat(ch[L[n]].x)

    def params(self) -> np.ndarray:
        v = self.out * self.gains
        v[L["lip_width"]] = max(-1.0, min(1.0, v[L["lip_width"]]))
        for i in range(len(v)):
            if i != L["lip_width"]:
                v[i] = sat(v[i])
        return v


class HeadReader:
    """Reads the net at a delay in steps, between heads if it must.  Logits are interpolated."""

    def __init__(self, net: VisemeNet, delay: float) -> None:
        d = sorted(net.head["delays"].items(), key=lambda kv: kv[1])
        ds = [v for _, v in d]
        delay = float(min(max(delay, ds[0]), ds[-1]))
        hi = next(i for i, v in enumerate(ds) if v >= delay)
        lo = max(hi - 1, 0)
        self.u = 0.0 if hi == lo else (delay - ds[lo]) / (ds[hi] - ds[lo])
        self.w0, self.b0 = net.heads[d[lo][0]]
        self.w1, self.b1 = net.heads[d[hi][0]]

    def __call__(self, h: np.ndarray, lo: int, hi: int) -> np.ndarray:
        a = np.einsum("ij,j->i", self.w0[lo:hi], h) + self.b0[lo:hi]
        if self.u == 0.0:
            return a
        b = np.einsum("ij,j->i", self.w1[lo:hi], h) + self.b1[lo:hi]
        return a + self.u * (b - a)


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x > -40.0 else 0.0


@dataclass(slots=True)
class LipFrameOut(FrameOut):
    visemes: np.ndarray = None      # float32, the 15 old visemes, for tests/eval_visemes.py
    fine: np.ndarray = None         # float32, the 23 fine visemes, fast weights at the sound this frame shows


class StreamingLipAnimator(StreamingAnimator):
    """leak exists only so tests can prove they catch a causality leak: 'feature' advances the mel stream by a
    frame, 'model' reads the net one frame late and calls it on time, 'peek' reads ahead in the chunk buffer."""

    def __init__(self, character: str | Style = "orb", *, lookahead_ms: float = DEFAULT_LOOKAHEAD_MS,
                 lead_frames: int = DEFAULT_LEAD, tune: LipTune = LipTune(), leak: str | None = None,
                 net_path: Path | None = None) -> None:
        super().__init__(character)
        name = character if isinstance(character, str) else None
        self.character = name
        self.A = int(round(lookahead_ms / STEP_MS))
        self.lead = int(lead_frames)
        self.mel = LogMel()
        self.level = LevelNorm()
        self.net = VisemeNet(net_path or net_for(self.A - 2 * self.lead))
        assert self.net.head.get("version") == 2, "not a version 2 net"
        self.lips = LipModel(self.style, name, tune)
        b = self.net.head["blocks"]
        self.b_fast, self.b_slow, self.b_close, self.b_tuck = b["fast"], b["slow"], b["closure"], b["tuck"]
        # past the last head more lookahead buys nothing: the same mouth is latched earlier and waits for the frame
        top = max(self.net.head["delays"].values())
        self.A_used = min(self.A, top + 1 + 2 * self.lead)
        total = self.A_used - 2 * self.lead
        d = self.lips.reads(total, self.lead)
        self.read_fast = HeadReader(self.net, d["fast"])
        self.read_slow = HeadReader(self.net, d["slow"])
        self.read_press = HeadReader(self.net, d["press"])
        self.read_tuck = HeadReader(self.net, d["tuck"])
        self._close_was = [0.0] * max(2 * self.lead - 1, 0)     # closure readings: the oldest is the sound heard one step on
        self._prev = None
        self._zero_steps = 0            # steps of digital silence at the newest audio
        self._zero_need = max(total, 0) + 2     # so this many steps put the sound shown two steps into it
        self._mel_hops = 0
        self._floor_rel = np.full(N_MELS, REL_FLOOR)
        self._leak = leak
        self._lip_hops = 2 * self.A_used
        self._look_samples = max(LOOK_SAMPLES, 400 * self.A)
        if leak in ("feature", "model"):
            self._lip_hops += HOPS_PER_FRAME
            self._look_samples += SPF
        self._await: list[tuple[int, np.ndarray]] = []      # life, waiting for its lips
        self._ready: list[tuple[np.ndarray, np.ndarray]] = []       # lips, waiting for their life
        if self._lip_hops == 0:     # with no lookahead frame 0 is latched before any audio: the mouth at rest
            self._ready.append((self.lips.params().astype(np.float32), self._fine.astype(np.float32)))

    # ------------------------------------------------------------------ internals
    def _hop(self) -> None:
        step = self._pending[:HOP]
        window = np.concatenate((self._tail[HOP:], step))
        mel_win = window[-MEL_WIN:]
        if self._leak == "peek":    # deliberately broken: reads samples the hop has not reached
            extra = min(SPF, self._pending.size - HOP)
            mel_win = np.concatenate((window, self._pending[HOP:HOP + extra]))[-MEL_WIN:]
        self._pending = self._pending[HOP:]
        self._tail = window
        self._tail_end += HOP
        self._hops += 1
        self.mouth.step(self.analyzer.analyze(window))
        if self._leak != "feature" or self._hops > HOPS_PER_FRAME:
            self._mel_hop(mel_win)
        if self._hops >= self._look_hops and (self._hops - self._look_hops) % HOPS_PER_FRAME == 0:
            self._await.append((self._frames, self._frame_vector()))
            self._frames += 1
        if self._hops >= self._lip_hops and (self._hops - self._lip_hops) % HOPS_PER_FRAME == 0:
            self._ready.append((self.lips.params().astype(np.float32), self._fine.astype(np.float32)))
        while self._await and self._ready:
            i, vec = self._await.pop(0)
            lip, fine = self._ready.pop(0)
            vec[LIP_IDX] = lip
            legacy_from_lips(vec)
            self._queue.append((i, vec, fine))

    def _mel_hop(self, x: np.ndarray) -> None:
        if x.any():
            mel = self.level.step(self.mel.frame(x))
            self._zero_hops = 0
        else:
            mel = self._floor_rel
            self._zero_hops += 1
        self._mel_hops += 1
        if self._mel_hops % STEP_HOPS:
            self._prev = mel
            return
        net = self.net
        h = net.step(self._prev, mel)
        a, b = self.b_fast
        fast = softmax(self.read_fast(h, a, b))
        a, b = self.b_slow
        slow = softmax(self.read_slow(h, a, b))
        close = sigmoid(float(self.read_press(h, self.b_close, self.b_close + 1)[0]))
        was = self._close_was
        now = was[0] if was else close
        if was:
            self._close_was = was[1:] + [close]
        tuck = sigmoid(float(self.read_tuck(h, self.b_tuck, self.b_tuck + 1)[0]))
        self._zero_steps = self._zero_steps + 1 if self._zero_hops >= STEP_HOPS else 0
        self.lips.step(fast, slow, close, tuck, self._zero_steps >= self._zero_need, now)
        self._fine = fast

    _fine = np.eye(23)[0]
    _zero_hops = 0

    def _release(self) -> list[FrameOut]:
        out: list[FrameOut] = []
        while self._queue:
            i, vec, fine = self._queue[0]
            if i * SPF + self._look_samples > self._cursor:
                break
            self._queue.pop(0)
            self._emitted += 1
            out.append(LipFrameOut(i, i / FPS, vec, (fine @ TO15).astype(np.float32), fine))
        return out


__all__ = ["StreamingLipAnimator", "LipFrameOut", "LipModel", "LipTune", "LIP_PARAMS", "LIP_IDX"]
