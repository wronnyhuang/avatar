"""Learned-reaction worker: the server's planner process (stdin/stdout frames from app/react_link.py).

Runs under its OWN interpreter (config react.python: numpy, torch), never inside app.py's event loop,
so a planner step can never block the server. One request per uplink batch:

  server -> worker  {"op":"step", "character", "rig_params"?, "human":[[t,mask_bits]...], "mic":[[seq,n,rate]...],
                  "own":[[s0,n]...], "sync":[[seq,t]...], "ticks":[[t, n_ab, rb]...]} + blobs
                  (H motion float32 86 each | mic int16 | own int16 24 kHz | page rig float32 n each | ab float32 3 each)
  worker -> server  {"id", "finals": n_ticks, "stats"?} + blob float32 (n_ticks x len(rig_params))

Per 30 fps tick j of the playback timeline (the page stamps its rig vectors with it):
  1. the frame shown at j = the schedule's frame for j (planned at an earlier tick), retargeted to the character
     (react/retarget.py), blinks as EVENTS (onset -> a full template, never a blended half-blink), merged with the
     page's affine map (reflexes + commands: final = alpha * learned + beta; blinks alpha * max(learned, reflex) + beta);
     channels the retarget does not cover keep the page's rules value.
  2. Planner.push(partner frame at j, mask, partner audio feats at j, own audio feats at j); Planner.commit(what was
     shown at j, with page overrides mapped back by retarget.rig_to_human); every plan_ms of timeline: plan(committed)
     -> 15 frames; frames already committed are kept (in-filling, no averaging of overlapping chunks); a 2-3 frame
     crossfade (head rotation slerped as quaternions, blink columns never blended) only when the new plan's first free
     frame truly disagrees with the scheduled one.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import math
import os
import struct
import sys
import time
import traceback

import numpy as np
from scipy.signal import firwin, resample_poly

U32 = struct.Struct("<I")
_H23 = firwin(2 * 30 + 1, 1.0 / 3, window=("kaiser", 5.0))    # resample_poly(x, 2, 3)'s own default FIR, built once
FPS = 30
TICK = 1.0 / FPS

# anim/rig.py RANGES of the renderer (copied: the worker does not import the renderer)
_RANGES = {
    "smile": (0, 1), "brow_knit": (0, 1), "blink_l": (0, 1), "blink_r": (0, 1), "ear_flick_l": (0, 1),
    "ear_flick_r": (0, 1), "nostril": (0, 1), "cheek": (0, 1), "voice": (0, 1), "breath": (-1, 1),
    "head_yaw": (-0.8, 0.8), "head_pitch": (-0.6, 0.6), "head_roll": (-0.6, 0.6),
    "head_x": (-0.3, 0.3), "head_y": (-0.3, 0.3), "head_z": (-0.3, 0.3),
    "gaze_x": (-1, 1), "gaze_y": (-1, 1), "brow_l": (-1, 1), "brow_r": (-1, 1), "ear_l": (-1, 1), "ear_r": (-1, 1),
    "lip_width": (-1, 1), "body_lean": (-1, 1), "body_bounce": (-1, 1), "tail_wag": (-1, 1), "tail_raise": (0, 1),
    "body_yaw": (-1, 1), "hand_l_turn": (-1, 1), "hand_r_turn": (-1, 1), "ear_l_turn": (-1, 1), "ear_r_turn": (-1, 1),
    "nose_wiggle": (-1, 1), "body_pitch": (-1, 1),
}
for _s in "lr":
    for _a in "xyz":
        _RANGES[f"hand_{_s}_{_a}"] = (-1.2, 1.4)


def rng(p):
    return _RANGES.get(p, (0.0, 1.0))


# ------------------------------------------------------------------------------------------------ framing
def read_exact(f, n):
    b = b""
    while len(b) < n:
        c = f.read(n - len(b))
        if not c:
            raise EOFError
        b += c
    return b


def read_frame(f):
    total = U32.unpack(read_exact(f, 4))[0]
    data = read_exact(f, total)
    hl = U32.unpack_from(data, 0)[0]
    hdr = json.loads(data[4:4 + hl].decode())
    body = data[4 + hl:]
    blobs, off = [], 0
    for n in hdr.get("blobs", []):
        blobs.append(body[off:off + n]); off += n
    return hdr, blobs


def write_frame(f, hdr, blobs=()):
    blobs = list(blobs)
    hdr = dict(hdr); hdr["blobs"] = [len(b) for b in blobs]
    h = json.dumps(hdr, separators=(",", ":")).encode()
    body = b"".join(blobs)
    f.write(U32.pack(4 + len(h) + len(body)) + U32.pack(len(h)) + h + body)
    f.flush()


# ------------------------------------------------------------------------------------------------ head rotation
def _rot_from_euler(y, p, r):
    """Rotation matrix whose spec.euler_from_mat is exactly (yaw y, pitch p, roll r): R = Ry(y) Rx(p) Rz(rho) with rho
    solved so atan2(R01, R11) == r (the MediaPipe-style roll of the spec is not a plain Tait-Bryan angle)."""
    cy, sy, cp, sp = math.cos(y), math.sin(y), math.cos(p), math.sin(p)
    rho = math.atan2(sy * sp * math.cos(r) - cp * math.sin(r), cy * math.cos(r))
    cr, sr = math.cos(rho), math.sin(rho)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def _euler_from_rot(R):
    yaw = math.atan2(R[0, 2], R[2, 2])
    pitch = math.atan2(-R[1, 2], math.hypot(R[0, 2], R[2, 2]))
    roll = math.atan2(R[0, 1], R[1, 1])
    return yaw, pitch, roll


def _quat(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def _rot_from_quat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp_euler(a, b, w):
    """(yaw, pitch, roll) a -> b by w along the shortest rotation (quaternion slerp)."""
    qa, qb = _quat(_rot_from_euler(*a)), _quat(_rot_from_euler(*b))
    d = float(np.dot(qa, qb))
    if d < 0:
        qb, d = -qb, -d
    if d > 0.9995:
        q = qa + w * (qb - qa)
    else:
        th = math.acos(min(1.0, d))
        q = (math.sin((1 - w) * th) * qa + math.sin(w * th) * qb) / math.sin(th)
    q /= np.linalg.norm(q)
    return _euler_from_rot(_rot_from_quat(q))


# ------------------------------------------------------------------------------------------------ audio
class StreamingVAD:
    """spec.energy_vad, one value at a time (identical output on the same loudness sequence): the 10th percentile of
    the last 300 values from a sorted window (numpy's 'linear' percentile, its own lerp form), O(log n) per frame."""

    def __init__(self, on_margin=18.0, off_margin=12.0, hang=6):
        self.on_m, self.off_m, self.hang = on_margin, off_margin, hang
        self.hist = collections.deque()
        self.sorted: list = []
        self.on = False
        self.hold = 0

    def _p10(self):
        s = self.sorted
        n = len(s)
        pos = np.float64(0.1) * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        t = pos - lo
        a, b = np.float32(s[lo]), np.float32(s[hi])
        d = b - a
        return (b - d * np.float32(1 - t)) if t >= 0.5 else (a + d * np.float32(t))

    def __call__(self, loud):
        loud = np.float32(loud)
        self.hist.append(loud)
        bisect.insort(self.sorted, loud)
        if len(self.hist) > 300:
            old = self.hist.popleft()
            del self.sorted[bisect.bisect_left(self.sorted, old)]
        floor = self._p10()
        if loud > floor + self.on_m:
            self.on = True; self.hold = self.hang
        elif loud < floor + self.off_m:
            if self.hold > 0:
                self.hold -= 1
            else:
                self.on = False
        return 1.0 if self.on else 0.0


class AudioTrack:
    """Timeline-indexed mono ring buffer (int16 samples at `rate`); position 0 = playback-timeline 0 s.
    dither_db: a fixed low noise floor added on READ (indexed by absolute sample, so overlapping windows agree). The
    page's mic gate and the voice tap produce exact digital zeros, which recordings never contain: with a -100 dB floor
    spec.energy_vad calls -70 dB hiss speech. -70 dBFS keeps live silence looking like a quiet room."""

    def __init__(self, rate, seconds=6.0, dither_db=None):
        self.rate = rate
        self.dither = None
        if dither_db is not None:
            g = np.random.default_rng(12345 + rate)
            self.dither = (g.standard_normal(rate * 7) * 10 ** (dither_db / 20)).astype(np.float32)
        self.n = int(rate * seconds)
        self.buf = np.zeros(self.n, np.float32)
        self.head = 0                      # one past the newest written sample index (absolute)

    def write(self, start, x):
        x = np.asarray(x, np.float32)
        if len(x) == 0 or start + len(x) <= self.head - self.n:
            return
        if start < self.head - self.n:
            x = x[self.head - self.n - start:]; start = self.head - self.n
        if start > self.head:                                   # gap: zeros
            self._zero(self.head, start)
        idx = (start + np.arange(len(x))) % self.n
        self.buf[idx] = x
        self.head = max(self.head, start + len(x))

    def _zero(self, a, b):
        if b - a >= self.n:
            self.buf[:] = 0; return
        idx = np.arange(a, b) % self.n
        self.buf[idx] = 0

    def read(self, a, b):
        """samples [a, b) (absolute); anything not written (future / too old / before 0) reads as 0."""
        out = np.zeros(b - a, np.float32)
        lo, hi = max(a, self.head - self.n, 0), min(b, self.head)
        if hi > lo:
            out[lo - a:hi - a] = self.buf[np.arange(lo, hi) % self.n]
        if self.dither is not None:
            out += self.dither[np.arange(a, b) % len(self.dither)]
        return out


class FeatStream:
    """spec.audio_features semantics frame by frame: 40 log-mel of the 512-sample window ENDING at the frame end
    (16 kHz), loudness over the last hop, streaming energy VAD."""

    def __init__(self, spec):
        self.spec = spec
        self.hop = int(round(spec.SR / spec.FPS))
        self.vad = StreamingVAD()

    def frame(self, x16_tail):
        """x16_tail: the max(512, hop) samples ending at the frame end."""
        sp = self.spec
        seg = x16_tail[-sp.N_FFT:][None, :] * sp._WIN[None]
        pw = np.abs(np.fft.rfft(seg, axis=1)) ** 2
        mel = np.log(np.maximum(pw @ sp.MEL_FB.T, 1e-6))[0]
        seg2 = x16_tail[-self.hop:]
        rms = np.sqrt(np.mean(seg2 ** 2) + 1e-12)
        loud = max(20 * np.log10(rms + 1e-12), -100.0)
        v = self.vad(loud)
        return np.concatenate([mel, [loud, v]]).astype(np.float32)


# ------------------------------------------------------------------------------------------------ planners
class StubPlanner:
    """Same interface as react/infer.py Planner: copies the partner's face/head/gaze/body with a delay, plus a gentle
    idle (head sway, spontaneous blinks when the partner's face is missing). delay_s = 0 measures the pipeline."""

    H = 15

    def __init__(self, spec, delay_s=0.5, idle=True, seed=1, mirror=True):
        self.spec = spec
        self.D = max(0, int(round(delay_s * FPS)))
        self.idle = idle
        self.mirror = mirror
        if spec is not None:
            # face-to-face mirror flip (like the eval agent's DelayCopy): swap anatomical L/R (blendshape pairs, body
            # joints, hands), negate head yaw / roll / x-translation, gaze x and the body x coordinates
            perm = np.arange(86); sign = np.ones(86, np.float32); nm = spec.BS_NAMES
            for i, n in enumerate(nm):
                if n.endswith("Left"):
                    j = nm.index(n[:-4] + "Right"); perm[i], perm[j] = j, i
            for c in ("head_yaw", "head_roll", "head_tx", "gaze_x"):
                sign[spec.MOTION_COLS.index(c)] = -1
            for jn in ("shoulder", "elbow", "wrist"):
                for ax in "xyz":
                    a = spec.MOTION_COLS.index(f"{jn}_l_{ax}"); b = spec.MOTION_COLS.index(f"{jn}_r_{ax}")
                    perm[a], perm[b] = b, a
                    if ax == "x":
                        sign[a] = sign[b] = -1
            for k in ("open", "thumb", "point", "palm"):
                a = spec.MOTION_COLS.index(f"hand_l_{k}"); b = spec.MOTION_COLS.index(f"hand_r_{k}")
                perm[a], perm[b] = b, a
            self.perm, self.sign = perm, sign
        self.rs = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.hist = collections.deque(maxlen=self.D + 64)
        self.k = 0
        self.phase = self.rs.uniform(0, 6.28, 3)
        self.next_blink = 3.0

    def _neutral(self):
        f = np.zeros(86, np.float32)
        f[60:78] = np.array([[0.5, 0, 0], [-0.5, 0, 0], [0.6, -1.0, 0], [-0.6, -1.0, 0], [0.6, -1.9, 0.1],
                             [-0.6, -1.9, 0.1]], np.float32).ravel()
        return f

    def push(self, pm, pmask, pa, oa, cue=None):
        f = self._neutral()
        if pmask[0]:
            f[0:60] = pm[0:60]
        if pmask[1]:
            f[60:78] = pm[60:78]
        if pmask[2]:
            f[78:82] = pm[78:82]
        if pmask[3]:
            f[82:86] = pm[82:86]
        if self.mirror:
            f = (f[self.perm] * self.sign).astype(np.float32)
        self.hist.append((f, bool(pmask[0])))
        self.k += 1

    def commit(self, own):
        pass

    def plan(self, committed):
        c = 0 if committed is None else len(committed)
        out = np.zeros((self.H, 86), np.float32)
        n = len(self.hist)
        for i in range(self.H):
            back = self.D - 1 - i                               # how many frames before the newest
            src = self.hist[max(0, n - 1 - back)] if n and back >= 0 else (self.hist[-1] if n else (self._neutral(), False))
            f = src[0].copy()
            if self.idle:
                tk = (self.k + i + 1) * TICK
                f[52] += 0.025 * math.sin(0.7 * tk + self.phase[0]) + 0.01 * math.sin(1.9 * tk + self.phase[1])
                f[53] += 0.015 * math.sin(0.9 * tk + self.phase[2])
                if not src[1]:                                  # nobody to copy: blink on our own
                    if tk >= self.next_blink and tk < self.next_blink + 0.2:
                        f[9] = f[10] = 0.9
                    elif tk >= self.next_blink + 0.2 and i == 0:
                        self.next_blink = tk + float(self.rs.uniform(2.5, 5.5))
            out[i] = f
        if c:
            out[:c] = committed
        return out


def _load_infer(args):
    """the model agent's react/infer.py: --infer PATH, else <root>/react/infer.py (loaded by path: it imports its own
    spec/model modules from its directory)."""
    import importlib.util
    path = getattr(args, "infer", "") or os.path.join(args.root, "react", "infer.py")
    sp = importlib.util.spec_from_file_location("react_infer_live", path)
    mod = importlib.util.module_from_spec(sp)
    sys.modules["react_infer_live"] = mod
    sp.loader.exec_module(mod)
    return mod


def load_planner(args, spec):
    if args.planner == "bundle":
        P = _load_infer(args).Planner
        seed = getattr(args, "seed", None)
        try:
            return P(args.bundle, threads=args.threads, **({} if seed is None else {"seed": int(seed)}))
        except TypeError:
            return P(args.bundle, threads=args.threads)
    if args.planner == "delaycopy":
        sys.path.insert(0, args.root)
        from react.baselines import DelayCopyPlanner  # noqa: E402
        try:
            return DelayCopyPlanner(args.bundle or None, threads=args.threads, delay_s=args.stub_delay)
        except TypeError:
            return DelayCopyPlanner(delay_s=args.stub_delay)
    return StubPlanner(spec, delay_s=args.stub_delay, mirror=bool(getattr(args, "stub_mirror", 1)))


# ------------------------------------------------------------------------------------------------ rig-space events
class BlinkEvents:
    """Learned eye closure -> blink EVENTS: an onset (closure crosses `on` while rising) plays one full template on
    both eyes (or one, for a wink); partial closures that never cross never show; nothing is ever blended."""
    TEMPLATE = [0.45, 0.9, 1.0, 1.0, 0.8, 0.55, 0.3, 0.12]   # 267 ms, fast close, slower open (the motion prior's blink shape)

    def __init__(self, on=0.55, refractory=0.25):
        self.on, self.refr = on, refractory
        self.i = -1; self.side = (1.0, 1.0); self.prev = 0.0; self.last_end = -9.0; self.t = 0.0; self.n = 0

    def step(self, cl, cr):
        self.t += TICK
        m = max(cl, cr)
        if self.i < 0 and m >= self.on and self.prev < self.on and self.t - self.last_end >= self.refr:
            wink = min(cl, cr) < 0.25 and m >= self.on
            self.side = ((1.0, 0.0) if cl > cr else (0.0, 1.0)) if wink else (1.0, 1.0)
            self.i = 0; self.n += 1
        self.prev = m
        if self.i < 0:
            return 0.0, 0.0
        v = self.TEMPLATE[self.i]
        self.i += 1
        if self.i >= len(self.TEMPLATE):
            self.i = -1; self.last_end = self.t
        return v * self.side[0], v * self.side[1]


class EarSprings:
    """Animals' ears swing with the learned head's nods (inertia), like today's controller."""

    def __init__(self):
        self.x = np.zeros(2); self.v = np.zeros(2); self.prev = None; self.pv = 0.0

    def step(self, pitch):
        hv = 0.0 if self.prev is None else (pitch - self.prev) / TICK
        acc = float(np.clip((hv - self.pv) / TICK, -60, 60))
        self.prev, self.pv = pitch, hv
        w, z = 2 * math.pi * 3.1, 0.32
        self.v += (-1.0 * acc - w * w * self.x - 2 * z * w * self.v) * TICK
        self.x += self.v * TICK
        return self.x


# ------------------------------------------------------------------------------------------------ the session engine
GROUP_TOL = None


def disagreement(a, b, spec):
    """max over channel groups of |a-b| / tolerance (1 = a jump one would see)."""
    d = np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32))
    face = [i for i in spec.GROUPS["face"] if i not in spec.MOUTH_BS and i not in (9, 10)]
    parts = [(d[52:55].max(), 0.06), (d[55:58].max(), 0.08), (d[58:60].max(), 0.15), (d[face].max(), 0.2),
             (d[60:78].max(), 0.15), (d[78:86].max(), 0.3)]
    return max(x / t for x, t in parts)


class Engine:
    def __init__(self, args, spec, retarget):
        self.args, self.spec, self.rt = args, spec, retarget
        self.planner = load_planner(args, spec)
        self.rig_params: list[str] = []
        self.character = "fox"
        # blinks: "rules" = the page's rule-based blinks (life layer: log-normal clock, gaze-shift blinks, + reflex and
        # commanded blinks, winks) pass through untouched; "learned" = the planner's eye closure as events
        self.blinks = getattr(args, "blinks", "rules") if getattr(args, "blinks", "rules") in ("rules", "learned") else "rules"
        self.stats = collections.defaultdict(float)
        self.plan_ms = collections.deque(maxlen=2000)
        self.trace = open(args.trace, "a", buffering=1 << 16) if getattr(args, "trace", "") else None
        self.step_ms = collections.deque(maxlen=2000)
        self.enc_ms = collections.deque(maxlen=2000)        # push + commit (the model's encoder step) per tick
        self.reset()

    def save_recording(self):
        """canon session (spec.save_session) of what the worker saw (partner = the user) and showed (avatar), if >= 5 s."""
        rec = getattr(self, "rec", None)
        d = getattr(self.args, "record_dir", "")
        self.rec = []
        if not d or not rec or len(rec) < 5 * FPS:
            return
        try:
            sp = self.spec
            js = [r[0] for r in rec]
            T = js[-1] - js[0] + 1
            mot = np.zeros((2, T, sp.D_MOTION), np.float32); msk = np.zeros((2, T, 4), np.uint8)
            aud = np.zeros((2, T, sp.D_AUDIO), np.float32)
            aud[:, :, :40] = np.log(1e-6); aud[:, :, 40] = -100.0
            for j, own, pm, pmask, pa, oa in rec:
                k = j - js[0]
                mot[0, k], msk[0, k], aud[0, k] = own, 1, oa
                mot[1, k], msk[1, k], aud[1, k] = pm, pmask, pa
            os.makedirs(d, exist_ok=True)
            sid = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
            meta = {"dataset": "user", "session": sid, "persons": ["avatar", "user"], "speakers": ["avatar", "user"],
                    "split": "test", "source": "live-page", "character": self.character,
                    "notes": "person 0 = what the avatar showed (human space), person 1 = the user's camera numbers "
                             "as uplinked ('H'), audio features only (no raw audio, no video)"}
            sp.save_session(os.path.join(d, sid + ".npz"), mot, msk, aud, meta)
        except Exception:  # noqa: BLE001
            traceback.print_exc()

    def reset(self):
        sp = self.spec
        if getattr(self, "rec", None):
            self.save_recording()
        self.rec = []
        try:
            self.planner.reset()
        except Exception:  # noqa: BLE001
            pass
        self.partner = []                  # sorted [(t, motion86 canonical, mask4)]
        self.med = [[], [], []]            # running median of the partner's raw head translation (cm), per axis
        self.med_q = collections.deque()
        dz = None if getattr(self.args, "dither_db", -70.0) is None or getattr(self.args, "dither_db", -70.0) <= -200 else float(getattr(self.args, "dither_db", -70.0))
        self.mic = AudioTrack(sp.SR, dither_db=dz)
        self.own24 = AudioTrack(24000, dither_db=dz)
        self.mic_sync = None               # (seq, t_end)
        self.mic_pending = []
        self.fmic, self.fown = FeatStream(sp), FeatStream(sp)
        self.cur = None                    # next tick to step
        self.sched: dict[int, np.ndarray] = {}
        self.shown: dict[int, np.ndarray] = {}
        self.last_plan_tick = -10 ** 9
        self.xf = None
        self.last_own = None
        self.blink = BlinkEvents()
        self.ears = EarSprings()
        self.last_final: dict[int, np.ndarray] = {}
        self.neutral = StubPlanner._neutral(None)
        self.audio_at(0)                                   # warm the FFT / filter paths before the first real tick
        self.fmic, self.fown = FeatStream(self.spec), FeatStream(self.spec)

    # -------------------------------------------------------------- inputs
    def add_human(self, t, m, mask):
        m = np.array(m, np.float32)
        if mask[0]:
            tcm = m[55:58].astype(np.float64)
            for ax in range(3):
                bisect.insort(self.med[ax], float(tcm[ax]))
            self.med_q.append(tcm)
            if len(self.med_q) > 9000:                            # ~5 min of face frames
                old = self.med_q.popleft()
                for ax in range(3):
                    del self.med[ax][bisect.bisect_left(self.med[ax], float(old[ax]))]
            med = np.array([L[len(L) // 2] if len(L) % 2 else 0.5 * (L[len(L) // 2 - 1] + L[len(L) // 2])
                            for L in self.med])
            m[55:58] = ((tcm - med) / 10.0).astype(np.float32)
        else:
            m[0:60] = 0.0
        i = bisect.bisect([p[0] for p in self.partner], t)
        self.partner.insert(i, (t, m, np.asarray(mask, np.uint8)))
        if len(self.partner) > 300:
            del self.partner[:len(self.partner) - 300]

    def add_sync(self, seq, t):
        self.mic_sync = (int(seq), float(t))
        for s, x in self.mic_pending:
            self._place_mic(s, x)
        self.mic_pending = []

    def _place_mic(self, seq, x16):
        s0, t0 = self.mic_sync
        t_end = t0 + (seq - s0) * 0.02
        self.mic.write(int(round(t_end * self.spec.SR)) - len(x16), x16)

    def add_mic(self, seq, pcm, rate):
        x = np.frombuffer(pcm, "<i2").astype(np.float32) / 32768.0
        if rate != self.spec.SR:
            x = self.spec.resample_to_16k(x, rate)
        if self.mic_sync is None:
            self.mic_pending.append((seq, x))
            self.mic_pending = self.mic_pending[-200:]
        else:
            self._place_mic(seq, x)

    def add_own(self, s0, pcm):
        self.own24.write(int(s0), np.frombuffer(pcm, "<i2").astype(np.float32) / 32768.0)

    # -------------------------------------------------------------- per-tick observations
    def partner_at(self, j):
        te = (j + 1) * TICK
        ts = [p[0] for p in self.partner]
        i = bisect.bisect_right(ts, te) - 1
        if i < 0:
            return self.neutral.copy(), np.zeros(4, np.uint8), None
        t, m, mk = self.partner[i]
        if te - t > 0.6:                                  # perception stalled / camera off: nothing valid
            return self.neutral.copy(), np.zeros(4, np.uint8), t
        return m, mk, t

    def audio_at(self, j):
        sp = self.spec
        e = sp.frame_end_sample(j)
        L = max(sp.N_FFT, self.fmic.hop)
        pa = self.fmic.frame(self.mic.read(e - L, e))
        m0 = (e - L - 24) & ~1                            # even 16 kHz index -> integer 24 kHz index
        e1 = (e + 1) & ~1
        x24 = self.own24.read(m0 * 3 // 2, e1 * 3 // 2 + 30)
        y = resample_poly(x24, 2, 3, window=_H23.copy()).astype(np.float32)
        oa = self.fown.frame(y[e - L - m0:e - m0])
        return pa, oa

    # -------------------------------------------------------------- retarget + merge
    def covered(self):
        cov = self.rt.RIG_COVERED.get(self.character) if hasattr(self.rt, "RIG_COVERED") else None
        return set(cov or [])

    def learned_rig(self, h):
        r = dict(self.rt.human_to_rig(h, self.character))
        cl, cr = r.get("blink_l", 0.0), r.get("blink_r", 0.0)
        bl, br = self.blink.step(float(cl), float(cr))
        r["blink_l"], r["blink_r"] = bl, br
        if "ear_l" in r and self.args.ear_springs:
            ex = self.ears.step(float(r.get("head_pitch", 0.0)))
            r["ear_l"] = float(r["ear_l"]) + 0.12 * float(ex[0])
            r["ear_r"] = float(r["ear_r"]) + 0.12 * float(ex[0])
        g = getattr(self.args, "rig_gain", None) or {}
        if g:                                               # per-channel calibration of the learned output (rig units)
            for k, v in g.items():
                if k in r:
                    r[k] = float(r[k]) * float(v)
        return r

    def merge(self, L, a, ab, rb):
        cov = self.covered()
        out = np.array(a, np.float32)
        lv = np.full(len(out), np.nan, np.float32)
        amap = {int(i): (float(al), float(be)) for i, al, be in ab}
        over = {}
        for i, p in enumerate(self.rig_params):
            if i >= len(out):
                break
            if p not in cov or p not in L:
                continue
            if self.blinks == "rules" and p in ("blink_l", "blink_r"):
                over[p] = float(out[i])        # the page's own blink is what shows: commit it as the avatar's lids
                continue
            x = float(L[p])
            lv[i] = x
            if p in ("blink_l", "blink_r"):
                x = max(x, float(rb))
            al, be = amap.get(i, (1.0, 0.0))
            f = al * x + be
            lo, hi = rng(p)
            f = min(hi, max(lo, f))
            out[i] = f
            if abs(f - float(L[p])) > 2e-3 and p not in ("blink_l", "blink_r"):   # (learned blinks are events)
                over[p] = f
        self.last_lv = lv
        return out, over

    # -------------------------------------------------------------- planning
    def _plan(self, j):
        c = self.args.commit_ahead
        committed = np.stack([self.sched[j + 1 + i] for i in range(c)]) if c and all(
            (j + 1 + i) in self.sched for i in range(c)) else np.zeros((0, 86), np.float32)
        c = len(committed)
        t0 = time.perf_counter()
        new = np.asarray(self.planner.plan(committed), np.float32)
        self.plan_ms.append((time.perf_counter() - t0) * 1000)
        self.stats["plans"] += 1
        old = [self.sched.get(j + 1 + i) for i in range(len(new))]
        xf = self.args.xfade
        X = self.xf if self.xf and j + 1 < self.xf["end"] else None
        if X is None and c < len(new) and xf > 0:
            ref = old[c] if old[c] is not None else self.last_own
            if ref is not None and disagreement(new[c], ref, self.spec) > 1.0:
                # the new plan truly disagrees with what is about to be on screen: fade from the scheduled trajectory
                # over xf frames (kept across re-plans until it ends, so it is exactly xf frames long)
                self.stats["crossfades"] += 1
                frm, last_old = {}, ref
                for i in range(c, min(len(new), c + xf)):
                    last_old = old[i] if old[i] is not None else last_old
                    frm[j + 1 + i] = last_old
                X = self.xf = {"start": j + 1 + c, "end": j + 1 + c + xf, "from": frm}
        if X is not None:
            for i in range(c, len(new)):
                k = j + 1 + i
                if k >= X["end"]:
                    break
                o = X["from"].get(k)
                if o is None:
                    continue
                w = (k - X["start"] + 1) / (xf + 1)
                f = o + w * (new[i] - o)
                f[9:11] = new[i][9:11]                       # blinks are events: never averaged
                f[52:55] = slerp_euler(o[52:55].astype(float), new[i][52:55].astype(float), w)
                new[i] = f
        for i in range(len(new)):
            self.sched[j + 1 + i] = new[i]
        for k in [k for k in self.sched if k < j - 90]:
            del self.sched[k]
        self.last_plan_tick = j

    def tick(self, j, page):
        """show tick j (page = (a, ab, rb) or None for a gap tick) -> final rig vector (or None)."""
        h = self.sched.get(j)
        if h is None:
            self.stats["sched_miss"] += 1
            h = self.last_own if self.last_own is not None else self.neutral.copy()
        L = self.learned_rig(h)
        final, over = (None, {})
        if page is not None:
            final, over = self.merge(L, *page)
            self.last_final[j] = final
        own = h.copy()
        if over:
            part, cov = self.rt.rig_to_human(over, self.character)
            own[cov] = part[cov]
        pm, pmask, pt = self.partner_at(j)
        pa, oa = self.audio_at(j)
        if self.trace is not None and page is not None:
            # per tick: what it saw (newest partner capture used, the user's smile/yaw, voice levels) and what it showed
            self.trace.write(json.dumps({"j": j, "pt": None if pt is None else round(pt, 4), "pf": int(pmask[0]),
                "p_smile": round(float(0.5 * (pm[44] + pm[45])), 4), "p_yaw": round(float(pm[52]), 4),
                "p_pitch": round(float(pm[53]), 4), "p_roll": round(float(pm[54]), 4), "p_brow": round(float(pm[3]), 4),
                "p_loud": round(float(pa[40]), 2), "p_vad": int(pa[41]), "o_loud": round(float(oa[40]), 2), "o_vad": int(oa[41]),
                "L": {k: round(float(v), 4) for k, v in L.items()},
                "page": [round(float(x), 4) for x in page[0]], "final": [round(float(x), 4) for x in final]}) + "\n")
        t0 = time.perf_counter()
        self.planner.push(pm, pmask, pa, oa, None)
        self.planner.commit(own)
        self.enc_ms.append((time.perf_counter() - t0) * 1000)
        self.shown[j] = own
        self.last_own = own
        if getattr(self.args, "record_dir", "") and page is not None:
            self.rec.append((j, own.copy(), np.array(pm, np.float32), np.array(pmask, np.uint8),
                             np.array(pa, np.float32), np.array(oa, np.float32)))
            if len(self.rec) > 30 * 60 * FPS:            # 30-min cap per session file
                self.save_recording()
        for k in [k for k in self.shown if k < j - 90]:
            del self.shown[k]
        for k in [k for k in self.last_final if k < j - 90]:
            del self.last_final[k]
        if (j - self.last_plan_tick) * TICK >= self.args.plan_ms / 1000.0 - 1e-6:
            self._plan(j)
        return final

    def step(self, ticks):
        """ticks: [(t, a float32, ab [(i, alpha, beta)], rb)] -> finals (n, len(a))."""
        t0 = time.perf_counter()
        finals = []
        for t, a, ab, rb in sorted(ticks, key=lambda x: x[0]):
            j = int(math.floor(t * FPS + 0.5))
            if self.cur is None or j > self.cur + 90 or j < self.cur - 90:      # (re)start the clock
                if self.cur is not None:
                    self.stats["clock_resets"] += 1
                self.cur = j
                self.sched = {}
                self._plan(j - 1)
            if j < self.cur:                                    # late (another lane's older batch)
                self.stats["late_ticks"] += 1
                h = self.shown.get(j, self.last_own if self.last_own is not None else self.neutral)
                f, _ = self.merge(self.rt.human_to_rig(h, self.character), a, ab, rb)
                prev = self.last_final.get(j)
                if prev is not None and self.blinks == "learned":
                    for name in ("blink_l", "blink_r"):
                        if name in self.rig_params:
                            k = self.rig_params.index(name)
                            if k < len(f):
                                f[k] = prev[k]
                finals.append(f)
                continue
            while self.cur < j:                                 # page skipped ticks: step them anyway
                self.stats["gap_ticks"] += 1
                self.tick(self.cur, None)
                self.cur += 1
            finals.append(self.tick(j, (a, ab, rb)))
            self.cur = j + 1
            self.stats["ticks"] += 1
        self.step_ms.append((time.perf_counter() - t0) * 1000)
        return finals

    def report(self):
        def pq(xs):
            if not xs:
                return None
            a = np.sort(np.asarray(xs))
            return {"n": int(len(a)), "p50": round(float(a[len(a) // 2]), 3), "p95": round(float(a[int(0.95 * (len(a) - 1))]), 3),
                    "max": round(float(a[-1]), 3)}
        tick = [a + b for a, b in zip(list(self.enc_ms)[-len(self.plan_ms):], list(self.plan_ms)[-len(self.enc_ms):])]
        return {"plan_ms": pq(self.plan_ms), "push_commit_ms": pq(self.enc_ms), "planner_tick_ms": pq(tick),
                "step_ms": pq(self.step_ms), "blinks": self.blink.n, "blinks_mode": self.blinks,
                **{k: int(v) for k, v in self.stats.items()}, "planner": type(self.planner).__name__,
                "character": self.character}


# ------------------------------------------------------------------------------------------------ main
def mini_retarget():
    """Fallback when react/retarget.py is not importable: head / gaze / brows / blink / smile only."""
    import types
    m = types.SimpleNamespace()
    cov = ["head_yaw", "head_pitch", "head_roll", "gaze_x", "gaze_y", "brow_l", "brow_r", "blink_l", "blink_r", "smile"]
    m.RIG_COVERED = {c: list(cov) for c in ["fox", "bear", "rabbit", "hippo", "monkey", "orb", "cube", "drop", "star", "pyramid", "blob", "muse"]}

    def h2r(f, ch):
        f = np.asarray(f, np.float32)
        bl = lambda v: float(np.clip((v - 0.35) / 0.4, 0, 1))  # noqa: E731
        return {"head_yaw": 1.3 * f[52], "head_pitch": 1.3 * f[53], "head_roll": 1.3 * f[54],
                "gaze_x": float(np.clip(f[58] * 1.5, -1, 1)), "gaze_y": float(np.clip(f[59] * 1.5, -1, 1)),
                "brow_l": float(np.clip(0.5 * f[5] + 0.5 * f[3] - f[2], -1, 1)),
                "brow_r": float(np.clip(0.5 * f[4] + 0.5 * f[3] - f[1], -1, 1)),
                "blink_l": bl(f[10]), "blink_r": bl(f[9]), "smile": float(np.clip(0.5 * (f[44] + f[45]) * 1.4, 0, 1))}

    def r2h(rig, ch=None):
        out = np.zeros(86, np.float32); cv = np.zeros(86, bool)
        for k, i in (("head_yaw", 52), ("head_pitch", 53), ("head_roll", 54)):
            if k in rig:
                out[i] = rig[k] / 1.3; cv[i] = True
        return out, cv
    m.human_to_rig, m.rig_to_human = h2r, r2h
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="react repo checkout (react/spec.py, react/retarget.py, react/infer.py)")
    ap.add_argument("--planner", default="stub", choices=["stub", "delaycopy", "bundle"])
    ap.add_argument("--bundle", default="")
    ap.add_argument("--infer", default="", help="path of react/infer.py (planner bundle); default <root>/react/infer.py")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--stub-delay", type=float, default=0.5)
    ap.add_argument("--plan-ms", type=float, default=33.0)
    ap.add_argument("--commit-ahead", type=int, default=0)
    ap.add_argument("--xfade", type=int, default=3)
    ap.add_argument("--ear-springs", type=int, default=1)
    ap.add_argument("--stub-mirror", type=int, default=1, help="stub: copy the user face-to-face mirrored (1) or as is (0)")
    ap.add_argument("--dither-db", type=float, default=-70.0, help="noise floor added to the user's mic + own voice before the "
                    "spec audio features (exact digital zeros otherwise fool energy_vad); <= -200 = off")
    ap.add_argument("--trace", default="", help="JSON lines per shown tick (latency analysis)")
    ap.add_argument("--blinks", default="rules", choices=["rules", "learned"],
                    help="rules = the page's rule-based blinks (default); learned = the planner's, as events")
    ap.add_argument("--rig-gain", default="", help='JSON {"smile": 0.75, ...}: scale learned rig channels')
    ap.add_argument("--record-dir", default="", help="save each session's partner/avatar numbers (canon npz, no video, "
                    "no raw audio) for evaluation / fine-tuning; off when empty")
    args = ap.parse_args()
    args.rig_gain = json.loads(args.rig_gain) if args.rig_gain else {}
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    sys.path.insert(0, args.root)
    from react import spec  # noqa: E402
    try:
        from react import retarget  # noqa: E402
        rt_name = "react.retarget"
    except ImportError:
        retarget = mini_retarget(); rt_name = "mini"
    fin, fout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr                           # stray prints must not corrupt the frame stream
    eng = Engine(args, spec, retarget)
    write_frame(fout, {"op": "ready", "planner": type(eng.planner).__name__, "retarget": rt_name, "pid": os.getpid()})
    while True:
        try:
            hdr, blobs = read_frame(fin)
        except EOFError:
            eng.save_recording()
            return
        op = hdr.get("op")
        rid = hdr.get("id")
        try:
            if op == "reset":
                if hdr.get("blinks") in ("rules", "learned"):
                    eng.blinks = hdr["blinks"]
                eng.character = hdr.get("character", eng.character)
                eng.rig_params = list(hdr.get("rig_params") or eng.rig_params)
                eng.reset()
                if eng.trace is not None:
                    eng.trace.write(json.dumps({"reset": True, "character": eng.character, "rig_params": eng.rig_params,
                                                "wall": time.time()}) + "\n")
                write_frame(fout, {"id": rid, "ok": True})
            elif op == "step":
                if hdr.get("character"):
                    eng.character = hdr["character"]
                if hdr.get("blinks") in ("rules", "learned"):
                    eng.blinks = hdr["blinks"]
                if hdr.get("rig_params"):
                    eng.rig_params = list(hdr["rig_params"])
                bi = 0
                for t, bits in hdr.get("human", []):
                    m = np.frombuffer(blobs[bi], np.float32); bi += 1
                    eng.add_human(float(t), m, [(int(bits) >> k) & 1 for k in range(4)])
                for seq, t in hdr.get("sync", []):
                    eng.add_sync(seq, t)
                for seq, rate in hdr.get("mic", []):
                    eng.add_mic(int(seq), blobs[bi], int(rate)); bi += 1
                for s0 in hdr.get("own", []):
                    eng.add_own(int(s0), blobs[bi]); bi += 1
                ticks = []
                for t, nab, rb in hdr.get("ticks", []):
                    a = np.frombuffer(blobs[bi], np.float32); bi += 1
                    ab = np.frombuffer(blobs[bi], np.float32).reshape(-1, 3) if nab else np.zeros((0, 3), np.float32); bi += 1
                    ticks.append((float(t), a, [(int(r[0]), float(r[1]), float(r[2])) for r in ab], float(rb)))
                finals = eng.step(ticks) if ticks else []
                body = b"".join(np.asarray(f, np.float32).tobytes() for f in finals)
                lv = getattr(eng, "last_lv", None)
                out = {"id": rid, "n": len(finals)}
                blobs_out = [body]
                if lv is not None and finals:
                    blobs_out.append(np.asarray(lv, np.float32).tobytes())
                if hdr.get("want_stats"):
                    out["stats"] = eng.report()
                write_frame(fout, out, blobs_out)
            elif op == "stats":
                write_frame(fout, {"id": rid, "stats": eng.report()})
            else:
                write_frame(fout, {"id": rid, "error": f"unknown op {op}"})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            write_frame(fout, {"id": rid, "error": str(e)[:300]})


if __name__ == "__main__":
    main()
