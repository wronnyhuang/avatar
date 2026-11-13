"""The page's rig vector at the renderer (docs/CONTRACT.md, docs/RIG.md): which columns the page owns, their ranges,
light critically-damped smoothing over 30 Hz packet jitter, the combination with lipstream, and the legacy columns
mapped onto the positional hand targets.  Shared by avatar/render_server.py and tools/render_timeline.py.

    mix = RigMixer()
    mix.push_rig(t, values, now=mono())           # {"type":"rig","t":..,"a":[...]} in RIG_PARAMS order (or names=...)
    mix.push_pose({"smile": .6, ...}, now=mono())   # the old {"type":"pose"} dict path (life stays on)
    params = mix.apply(job_params, t_frame, now)    # -> the row to render (a copy)

Rules (contract): lip columns (LIP_OWNED) come from lipstream, everything else from the page's latest rig vector;
jaw/teeth_gap/lip_open = max(lipsync, mouth_open); smile = rig smile + a little of the stream's own smile; the server's
procedural life (blinks, breath, gaze, head, brows, ears) is replaced while rig vectors arrive and comes back, faded
over FADE_OUT_S, after RIG_TIMEOUT_S without one.
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import numpy as np

from .schema import IDX, NPARAMS, PARAMS

# columns lipstream owns (contract "Combination at the renderer")
LIP_OWNED = ["jaw", "round", "spread", "press", "labdent", "tongue", "cheek", "voice",
             "lip_open", "lip_upper", "lip_lower", "lip_width", "lip_protrude", "lip_press", "lip_tuck",
             "teeth_gap", "tongue_tip", "tongue_dental", "tongue_back"]
LEGACY = ["arm_l", "arm_r", "wave_l", "wave_r", "point_l", "point_r", "paw_open_l", "paw_open_r", "clap", "thumbs_up"]
# what the page sends, in this order (config.rig_params).  point_l/r stay: in the new semantics they are the finger.
RIG_PARAMS = [p for p in PARAMS if p not in LIP_OWNED and p not in LEGACY or p in ("point_l", "point_r")]

_R = {
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
        _R[f"hand_{_s}_{_a}"] = (-1.2, 1.4)
# body (docs/INTENT-RULES.md): root motion, camera, counting, props.  PERIODIC columns (turns, period 1) are
# unwrapped against the previous vector by the mixer and never clipped by range (the shader takes fract()).
_R.update({"root_x": (-3, 3), "root_y": (-0.6, 3), "root_z": (-3, 1.5), "squash": (-1, 1), "gaze_free": (0, 1),
           "cam_zoom": (-1, 1), "cam_x": (-3, 3), "cam_y": (-1, 3), "stand": (0, 1),   # scene: cam_zoom < 0 pushes in
           "fingers_l": (0, 5), "fingers_r": (0, 5)})
PERIODIC = ["root_yaw", "root_pitch", "root_roll"]
for _k in range(6):
    for _a in "xyz":
        _R[f"prop{_k}_{_a}"] = (-3, 3)
    for _a in ("rx", "ry", "rz"):
        PERIODIC.append(f"prop{_k}_{_a}")
for _k in range(6, 16):   # objects: slots 8-15 appended (docs/OBJECTS.md)
    for _a in "xyz":
        _R[f"prop{_k}_{_a}"] = (-3, 3)
    for _a in ("rx", "ry", "rz"):
        PERIODIC.append(f"prop{_k}_{_a}")
for _s in "lr":
    for _a in "xz":
        _R[f"foot_{_s}_{_a}"] = (-2, 2)
    _R[f"foot_{_s}_y"] = (-0.3, 1.5)
for _p in PERIODIC:
    _R[_p] = (-3.2767, 3.2767)
RANGES = {p: _R.get(p, (0, 1)) for p in PARAMS}
LO = np.array([RANGES[p][0] for p in PARAMS], "f4")
HI = np.array([RANGES[p][1] for p in PARAMS], "f4")
# body: the mixer never clips PERIODIC columns (RANGES keeps their wire limit); they are unwrapped instead
PER_IDX = np.array([IDX[p] for p in PERIODIC], int)
_CLIP_LO, _CLIP_HI = LO.copy(), HI.copy()
_CLIP_LO[PER_IDX], _CLIP_HI[PER_IDX] = -np.inf, np.inf
REBASE_TURNS = 8.0               # an unwrapped periodic column past this many turns is shifted back by whole turns

SMOOTH_S = 0.045                 # critically damped smoothing time of every page column (contract: ~40-60 ms)
FAST = {"blink_l": 0.022, "blink_r": 0.022, "ear_flick_l": 0.025, "ear_flick_r": 0.025, "nose_wiggle": 0.025}
RIG_TIMEOUT_S = 2.0              # no rig vector for this long: the server's own life comes back
FADE_IN_S, FADE_OUT_S = 0.20, 0.50
POSE_TIMEOUT_S = 1.5             # the old pose path's hold

RIG_IDX = np.array([IDX[p] for p in RIG_PARAMS], int)
_SMOOTH = np.array([FAST.get(p, SMOOTH_S) for p in RIG_PARAMS], "f4")
_PER_RIG = np.array([RIG_PARAMS.index(p) for p in PERIODIC], int)    # body: PERIODIC positions in RIG_IDX
_PERIODIC_SET = frozenset(PERIODIC)
# the old pose path sent body_bounce 0..1 and tail_wag as AMPLITUDES the shader oscillated; now they are positions,
# so the pose path oscillates them here
_POSE_LEGACY_OSC = ("body_bounce", "tail_wag")


def clip_row(row):
    # body: PERIODIC columns pass through unclipped (bounds ±inf)
    return np.clip(row, _CLIP_LO, _CLIP_HI)


def wrap_turns(d):
    """body: an angle in turns into [-0.5, 0.5) (the shortest arc of a difference)."""
    return d - np.floor(d + 0.5)


# ------------------------------------------------------------------------------------------------ combination
def combine(params, rig=None):
    """params: a full row whose lip columns are lipstream's (and the other columns its life, used when rig is None).
    rig: a full-length row (only RIG_PARAMS columns are read) or None.  Returns the row to render."""
    out = np.array(params, "f4", copy=True)
    if out.size < NPARAMS:
        out = np.concatenate([out, np.zeros(NPARAMS - out.size, "f4")])
    if rig is not None:
        life_smile = float(out[IDX["smile"]])
        out[RIG_IDX] = rig[RIG_IDX]
        out[IDX["smile"]] = min(1.0, max(0.0, float(rig[IDX["smile"]]) + 0.25*life_smile))
    mo = float(out[IDX["mouth_open"]])
    if mo > 0.0:
        for c, k in (("jaw", 1.0), ("teeth_gap", 1.0), ("lip_open", 1.0)):
            out[IDX[c]] = max(out[IDX[c]], k*mo)
        out[IDX["lip_press"]] *= 1.0 - mo
        out[IDX["press"]] *= 1.0 - mo
    return out


# ------------------------------------------------------------------------------------------------ legacy columns
_WAVE_UP = (0.20, 1.00, 0.20)
_POINT = (-0.10, 0.55, 0.85)
_CLAP = (-0.45, 0.45, 0.45)
_THUMB = (0.10, 0.45, 0.45)
_RAISE = (0.15, 0.80, 0.25)


def _blend_hand(out, side, xyz, w, open_=None, turn=None):
    for a, v in zip("xyz", xyz):
        i = IDX[f"hand_{side}_{a}"]
        out[i] = out[i] + (v - out[i])*w
    if open_ is not None:
        i = IDX[f"hand_{side}_open"]
        out[i] = out[i] + (open_ - out[i])*w
    if turn is not None:
        i = IDX[f"hand_{side}_turn"]
        out[i] = out[i] + (turn - out[i])*w


_ANIMALS = Path(__file__).resolve().parent.parent / "animals"


def point_moves_hand(character):
    """Does a point_l/point_r value also move that hand to the camera-point pose (_POINT)?  Yes for every character
    except one whose animals/<character>.json sets "point_finger_only": true (the muse): its controller places
    the hand itself (pointing to the side, up, at a spot) and the column is only the index finger (docs/CONTRACT.md
    "point_l/point_r = index finger extended only").  Read once per character switch."""
    if not character:
        return True
    try:
        cfg = json.loads((_ANIMALS / f"{character}.json").read_text())
    except (OSError, ValueError):
        return True
    return not bool(cfg.get("point_finger_only", False))


def map_legacy(row, t, point_hand=True):
    """The old arm/wave/point/paw/clap/thumbs columns -> positional hand targets (the shader reads only hand_*).
    Oscillations the old shader made with sin(uTime) are made here, on the frame's time t.  point_hand False: the
    point columns stay the finger only and never move the hand (see point_moves_hand)."""
    out = np.array(row, "f4", copy=True)
    g = lambda n: float(out[IDX[n]])
    if not any(g(n) > 1e-3 for n in LEGACY):
        return out
    for side in "lr":
        arm, wave, point, paw = g(f"arm_{side}"), g(f"wave_{side}"), g(f"point_{side}"), g(f"paw_open_{side}")
        if arm > 0:
            _blend_hand(out, side, _RAISE, min(arm, 1.0), open_=0.6, turn=0.6)
        if point > 0 and point_hand:
            _blend_hand(out, side, _POINT, min(point, 1.0), open_=0.0, turn=0.0)
        if wave > 0:
            ph = math.sin(2*math.pi*2.5*t)
            _blend_hand(out, side, (_WAVE_UP[0] + 0.12*ph, _WAVE_UP[1], _WAVE_UP[2]), min(wave, 1.0), open_=1.0,
                        turn=1.0 - 0.3*(0.5 + 0.5*ph))
        if paw > 0:
            i = IDX[f"hand_{side}_open"]
            out[i] = max(out[i], min(paw, 1.0))
    clap = g("clap")
    if clap > 0:
        apart = 0.5 + 0.5*math.sin(2*math.pi*2.0*t)       # 0 = palms together
        for side in "lr":
            _blend_hand(out, side, (_CLAP[0] + 0.25*apart, _CLAP[1], _CLAP[2]), min(clap, 1.0), open_=1.0, turn=0.0)
    th = g("thumbs_up")
    if th > 0:
        _blend_hand(out, "r", _THUMB, min(th, 1.0), open_=0.0, turn=0.0)
        out[IDX["thumb_r"]] = max(out[IDX["thumb_r"]], min(th, 1.0))
    for n in LEGACY:
        if n not in ("point_l", "point_r"):
            out[IDX[n]] = 0.0
    return out


# ------------------------------------------------------------------------------------------------ smoothing
def smooth_damp(x, v, target, smooth_time, dt):
    """Critically damped spring step (Game Programming Gems 4, 'SmoothDamp'); stable for any dt.  Arrays."""
    omega = 2.0/np.maximum(smooth_time, 1e-4)
    k = omega*dt
    e = 1.0/(1.0 + k + 0.48*k*k + 0.235*k*k*k)
    change = x - target
    temp = (v + omega*change)*dt
    v2 = (v - omega*temp)*e
    x2 = target + (change + temp)*e
    return x2, v2


class RigMixer:
    """Per session.  Rig vectors are kept with their playback-timeline stamps; a frame of time tj takes the vector
    interpolated at tj when the history covers it (frames are rendered for audio times; the page stamps rig vectors on
    the same timeline), else the newest.  Then one critically damped step per frame (frame-time dt)."""

    def __init__(self, names=None, character=None):
        self.set_names(names)
        self.set_character(character)
        self.hist = collections.deque()          # (t, full-length row)
        self.last_rig_at = None                  # monotonic arrival of the newest rig vector
        self.first_rig_at = None
        self.x = None; self.v = None; self.t_prev = None
        self.pose = {}; self.pose_at = 0.0
        self.n_rig = 0

    def set_character(self, character):
        self.point_hand = point_moves_hand(character)

    def set_names(self, names):
        names = list(names) if names else list(RIG_PARAMS)
        self.names = names
        self.map_idx = np.array([IDX[n] if n in IDX else -1 for n in names], int)

    # ---------------------------------------------------------------- in
    def push_rig(self, t, values, now, names=None):
        a = np.asarray(values, "f4").ravel()
        if names:
            if list(names) != self.names:
                self.set_names(names)
        elif a.size != len(self.names):
            # no names: RIG_PARAMS order when the length says so, else the FULL schema order (anim/schema.py PARAMS,
            # append-only, so any prefix is unambiguous) — the app server's config.rig_params is the full PARAMS list
            self.set_names(RIG_PARAMS if a.size == len(RIG_PARAMS) else PARAMS[:a.size])
        n = min(a.size, self.map_idx.size)
        if n == 0:
            return False
        row = np.zeros(NPARAMS, "f4")
        if self.hist:
            row[:] = self.hist[-1][1]
        sel = self.map_idx[:n]
        ok = (sel >= 0) & np.isfinite(a[:n])
        row[sel[ok]] = a[:n][ok]
        # body: PERIODIC columns are unwrapped against the previous vector (shortest arc): the sent value moves
        # by whole turns to within half a turn of the previous one (0.49 -> -0.49 becomes 0.51; a jump of exactly one
        # turn becomes none); with no history it is wrapped into [-0.5, 0.5).  Columns this vector doesn't carry
        # keep the previous value (shift 0).  Then clip_row leaves them alone.
        prev = self.hist[-1][1][PER_IDX] if self.hist else np.zeros(PER_IDX.size, "f4")
        row[PER_IDX] -= np.floor(row[PER_IDX] - prev + 0.5)
        row = clip_row(row)
        try:
            t = float(t)
        except (TypeError, ValueError):
            t = float("nan")
        if not math.isfinite(t):
            t = self.hist[-1][0] + 1/30 if self.hist else 0.0
        if self.hist and t < self.hist[-1][0] - 5.0:       # the page's timeline restarted
            self.hist.clear()
        while self.hist and self.hist[-1][0] >= t:          # out of order / duplicate stamp: newest wins
            self.hist.pop()
        # body: keep unwrapped values bounded.  A PERIODIC column past REBASE_TURNS is shifted by the same whole
        # number of turns in this row, the whole history and the smoother state (the spring is shift-invariant and
        # the renderer only takes the fractional turn, so nothing visible changes).
        shift = row[PER_IDX]
        shift = np.where(np.abs(shift) > REBASE_TURNS, np.floor(shift + 0.5), 0.0).astype("f4")
        if shift.any():
            row[PER_IDX] -= shift
            for _, r in self.hist:
                r[PER_IDX] -= shift
            if self.x is not None:
                self.x[_PER_RIG] -= shift
        self.hist.append((t, row))
        while len(self.hist) > 2 and self.hist[0][0] < t - 3.0:
            self.hist.popleft()
        if self.last_rig_at is None or now - self.last_rig_at > RIG_TIMEOUT_S:
            self.first_rig_at = now
        self.last_rig_at = now
        self.n_rig += 1
        return True

    def push_pose(self, raw, now):
        pose = {}
        for k, v in (raw or {}).items():
            if k in IDX and k not in ("voice",):
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    if k in _PERIODIC_SET:     # body: an angle in turns: wrapped into [-0.5, 0.5), never clipped
                        pose[k] = v - math.floor(v + 0.5)
                        continue
                    lo, hi = RANGES[k]
                    pose[k] = max(lo, min(hi, v))
        self.pose, self.pose_at = pose, now
        return pose

    def rig_active(self, now):
        return self.last_rig_at is not None and now - self.last_rig_at <= RIG_TIMEOUT_S + FADE_OUT_S

    # ---------------------------------------------------------------- out
    def _target(self, tj):
        h = self.hist
        if not h:
            return None
        # outside the stamped history (the frame is ahead of the page, or the page's clock disagrees): the newest
        if tj <= h[0][0] or tj >= h[-1][0]:
            return h[-1][1]
        lo, hi = 0, len(h) - 1
        while hi - lo > 1:
            mid = (lo + hi)//2
            if h[mid][0] <= tj:
                lo = mid
            else:
                hi = mid
        (t0, a), (t1, b) = h[lo], h[hi]
        u = (tj - t0)/max(t1 - t0, 1e-6)
        return a + (b - a)*u

    def apply(self, params, tj, now):
        params = np.asarray(params, "f4")
        base = params.copy() if params.size >= NPARAMS else np.concatenate([params, np.zeros(NPARAMS - params.size, "f4")])
        w = 0.0
        if self.last_rig_at is not None:
            age = now - self.last_rig_at
            if age <= RIG_TIMEOUT_S:
                w = min(1.0, (now - self.first_rig_at)/FADE_IN_S) if FADE_IN_S > 0 else 1.0
            elif age <= RIG_TIMEOUT_S + FADE_OUT_S:
                w = 1.0 - (age - RIG_TIMEOUT_S)/FADE_OUT_S
        out = base
        if w > 0.0:
            tgt = self._target(tj)
            if tgt is not None:
                tgt = map_legacy(tgt, tj, self.point_hand)
                cur = tgt[RIG_IDX]
                if self.x is None or self.t_prev is None or not (0.0 < tj - self.t_prev < 1.0):
                    self.x, self.v = cur.copy(), np.zeros_like(cur)
                else:
                    self.x, self.v = smooth_damp(self.x, self.v, cur, _SMOOTH, tj - self.t_prev)
                self.t_prev = tj
                rig = np.zeros(NPARAMS, "f4")
                rig[RIG_IDX] = self.x
                mixed = combine(base, rig)
                if w < 1.0:
                    # body: the fade of a PERIODIC column goes the shortest arc between base and mixed (an
                    # unwrapped 3.0 or 7.3 fades in/out over 0 / 0.3 turns, never spinning through whole turns)
                    d = mixed - base
                    d[PER_IDX] = wrap_turns(d[PER_IDX])
                    out = base + d*w
                else:
                    out = mixed
        else:
            self.x = None
        if self.pose and now - self.pose_at < POSE_TIMEOUT_S:
            out = out.copy()
            for name, value in self.pose.items():
                i = IDX[name]
                if name in LIP_OWNED:
                    # spoken mouth shapes win during voiced output; silent commands may hold the mouth open
                    if name in ("lip_open", "teeth_gap", "lip_width", "lip_protrude", "tongue_dental") and out[IDX["voice"]] < .12:
                        out[i] = value
                else:
                    out[i] = value
            for name in _POSE_LEGACY_OSC:
                if name in self.pose:
                    out[IDX[name]] = self.pose[name]*math.sin((7.0 if name == "body_bounce" else 11.0)*tj)
            out = map_legacy(out, tj, self.point_hand)
            out = combine(out)
        elif w <= 0.0:
            out = combine(map_legacy(out, tj, self.point_hand))
        return out
