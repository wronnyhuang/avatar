"""r3_semantic — a semantic goal/constraint motion API + a sandboxed interpreter that turns a short Python-subset
program into canon1 motion (T,189) @ 30 fps.  API reference for the code model: API.md (same folder).

    from semantic import run_program
    motion, report = run_program(src, seed=0, nat=NAT_DEFAULT)

Pipeline:
  1. parse (ast) + whitelist check (no imports / attributes / defs / exec; only API calls, literals, arithmetic,
     simple assignments, `for i in range(n)`, `with parallel()/sequence()/after(s)`)
  2. execute the program symbolically -> a timeline of channel SEGMENTS (start, end, target, easing) on channels
     root / torso / head / gaze / arm_l / arm_r / fing_l / fing_r / face, plus additive oscillation LAYERS
  3. render frame by frame: root -> torso -> head (look_at resolved against the CURRENT body) -> gaze (eyes lead the
     head) -> arms (IK goals resolved against the CURRENT body, joint-space min-jerk with successive breaking,
     body push-out) -> fingers -> face; then idle micro-motion + blinks; trim/pad to 2..10 s, auto return to neutral.
  Naturalness shaping is switchable (NAT dict) for ablations.
"""
from __future__ import annotations

import ast
import contextvars
import copy
import difflib
import math
import os
import re
import sys
import time

import numpy as np

from . import kin  # noqa: E402

FPS = 30
COL = kin.COL
BS_NAMES = [n for n in kin.BS_NAMES if n != "_neutral"]
NAT_DEFAULT = dict(min_jerk=True, anticipation=True, overshoot=True, successive=True, eye_lead=True, idle=True,
                   blinks=True, arcs=True, smooth=True)
NAT_OFF = {k: False for k in NAT_DEFAULT}
MAX_T = 10.0
MIN_T = 2.0


class ProgramError(Exception):
    pass


# ============================================================================================ timeline objects
class Seg:
    __slots__ = ("chan", "t0", "t1", "tgt", "ease", "kind", "src", "contact", "prio")

    def __init__(self, chan, t0, t1, tgt, ease="min_jerk", kind="", src="", contact=False, prio=1):
        self.chan, self.t0, self.t1, self.tgt, self.ease = chan, float(t0), float(max(t1, t0 + 1.0 / FPS)), tgt, ease
        self.kind, self.src, self.contact, self.prio = kind, src, contact, prio


class Osc:
    """additive oscillation on a scalar channel parameter: amp * sin(2 pi cycles u + phase) * envelope(u)."""
    __slots__ = ("chan", "param", "t0", "t1", "amp", "cycles", "phase", "src", "bias")

    def __init__(self, chan, param, t0, t1, amp, cycles, phase=0.0, src="", bias=0.0):
        self.chan, self.param, self.t0, self.t1 = chan, param, float(t0), float(t1)
        self.amp, self.cycles, self.phase, self.src, self.bias = float(amp), float(cycles), float(phase), src, bias


# ============================================================================================ expressions
def _face_preset(**kw):
    return kw


EXPRESSIONS = {
    "neutral": {},
    "smile": _face_preset(mouthSmileLeft=0.75, mouthSmileRight=0.75, cheekSquintLeft=0.35, cheekSquintRight=0.35,
                          eyeSquintLeft=0.3, eyeSquintRight=0.3),
    "big_smile": _face_preset(mouthSmileLeft=1.0, mouthSmileRight=1.0, jawOpen=0.25, mouthUpperUpLeft=0.3,
                              mouthUpperUpRight=0.3, cheekSquintLeft=0.6, cheekSquintRight=0.6, eyeSquintLeft=0.5,
                              eyeSquintRight=0.5, mouthDimpleLeft=0.3, mouthDimpleRight=0.3),
    "laugh": _face_preset(mouthSmileLeft=1.0, mouthSmileRight=1.0, jawOpen=0.5, mouthUpperUpLeft=0.4,
                          mouthUpperUpRight=0.4, cheekSquintLeft=0.7, cheekSquintRight=0.7, eyeSquintLeft=0.6,
                          eyeSquintRight=0.6, eyeBlinkLeft=0.35, eyeBlinkRight=0.35, browInnerUp=0.2),
    "frown": _face_preset(mouthFrownLeft=0.8, mouthFrownRight=0.8, browDownLeft=0.45, browDownRight=0.45,
                          mouthPressLeft=0.3, mouthPressRight=0.3, mouthShrugLower=0.3),
    "sad": _face_preset(mouthFrownLeft=0.7, mouthFrownRight=0.7, browInnerUp=0.85, mouthShrugLower=0.35,
                        eyeBlinkLeft=0.25, eyeBlinkRight=0.25, mouthPressLeft=0.2, mouthPressRight=0.2),
    "angry": _face_preset(browDownLeft=0.9, browDownRight=0.9, eyeSquintLeft=0.5, eyeSquintRight=0.5,
                          noseSneerLeft=0.4, noseSneerRight=0.4, mouthPressLeft=0.5, mouthPressRight=0.5,
                          mouthFrownLeft=0.35, mouthFrownRight=0.35, jawForward=0.2),
    "surprise": _face_preset(browInnerUp=0.9, browOuterUpLeft=0.9, browOuterUpRight=0.9, eyeWideLeft=0.9,
                             eyeWideRight=0.9, jawOpen=0.45, mouthFunnel=0.2, eyeBlinkLeft=0.0, eyeBlinkRight=0.0,
                             eyeSquintLeft=0.0, eyeSquintRight=0.0),
    "shock": _face_preset(browInnerUp=1.0, browOuterUpLeft=1.0, browOuterUpRight=1.0, eyeWideLeft=1.0,
                          eyeWideRight=1.0, jawOpen=0.75, mouthStretchLeft=0.3, mouthStretchRight=0.3,
                          eyeBlinkLeft=0.0, eyeBlinkRight=0.0, eyeSquintLeft=0.0, eyeSquintRight=0.0),
    "fear": _face_preset(browInnerUp=0.9, browOuterUpLeft=0.5, browOuterUpRight=0.5, eyeWideLeft=0.8,
                         eyeWideRight=0.8, mouthStretchLeft=0.6, mouthStretchRight=0.6, jawOpen=0.2),
    "disgust": _face_preset(noseSneerLeft=0.8, noseSneerRight=0.8, mouthUpperUpLeft=0.6, mouthUpperUpRight=0.6,
                            browDownLeft=0.5, browDownRight=0.5, eyeSquintLeft=0.45, eyeSquintRight=0.45,
                            mouthFrownLeft=0.3, mouthFrownRight=0.3),
    "confused": _face_preset(browOuterUpRight=0.7, browDownLeft=0.55, browInnerUp=0.3, mouthLeft=0.35,
                             mouthPressLeft=0.3, mouthPressRight=0.2, eyeSquintLeft=0.3),
    "thinking": _face_preset(browDownLeft=0.3, browDownRight=0.3, mouthRollLower=0.35, mouthPressLeft=0.3,
                             mouthPressRight=0.3, mouthRight=0.2, eyeSquintLeft=0.25, eyeSquintRight=0.25),
    "pout": _face_preset(mouthPucker=0.75, mouthShrugLower=0.55, browInnerUp=0.45, mouthFrownLeft=0.3,
                         mouthFrownRight=0.3),
    "kiss": _face_preset(mouthPucker=1.0, mouthFunnel=0.35, eyeBlinkLeft=0.35, eyeBlinkRight=0.35),
    "duck_face": _face_preset(mouthPucker=0.9, mouthFunnel=0.45, cheekSquintLeft=0.3, cheekSquintRight=0.3,
                              browOuterUpLeft=0.4, browOuterUpRight=0.4),
    "silly": _face_preset(jawOpen=0.45, jawLeft=0.45, mouthLeft=0.6, eyeWideLeft=0.9, eyeSquintRight=0.7,
                          eyeBlinkRight=0.45, browOuterUpLeft=1.0, browDownRight=0.5, cheekPuff=0.2,
                          mouthFunnel=0.3),
    "goofy": _face_preset(jawOpen=0.6, mouthFunnel=0.5, eyeWideLeft=0.8, eyeWideRight=0.8, browInnerUp=0.8,
                          browOuterUpLeft=0.8, browOuterUpRight=0.8, jawRight=0.3, mouthRight=0.4),
    "smirk": _face_preset(mouthSmileLeft=0.7, mouthDimpleLeft=0.4, cheekSquintLeft=0.3, browOuterUpLeft=0.3,
                          mouthPressRight=0.3),
    "sneer": _face_preset(noseSneerLeft=0.85, noseSneerRight=0.6, mouthUpperUpLeft=0.7, mouthUpperUpRight=0.3,
                          browDownLeft=0.4, browDownRight=0.4),
    "wince": _face_preset(eyeSquintLeft=0.85, eyeSquintRight=0.85, eyeBlinkLeft=0.6, eyeBlinkRight=0.6,
                          browDownLeft=0.6, browDownRight=0.6, mouthStretchLeft=0.6, mouthStretchRight=0.6,
                          noseSneerLeft=0.4, noseSneerRight=0.4, cheekSquintLeft=0.5, cheekSquintRight=0.5),
    "sleepy": _face_preset(eyeBlinkLeft=0.6, eyeBlinkRight=0.6, browInnerUp=0.2, jawOpen=0.08,
                           eyeSquintLeft=0.0, eyeSquintRight=0.0),
    "yawn": _face_preset(jawOpen=1.0, eyeBlinkLeft=0.8, eyeBlinkRight=0.8, browInnerUp=0.5, mouthFunnel=0.3,
                         mouthStretchLeft=0.3, mouthStretchRight=0.3),
    "scream": _face_preset(jawOpen=1.0, eyeWideLeft=0.8, eyeWideRight=0.8, browInnerUp=0.9, browOuterUpLeft=0.6,
                           browOuterUpRight=0.6, mouthStretchLeft=0.6, mouthStretchRight=0.6),
    "grimace": _face_preset(mouthStretchLeft=0.9, mouthStretchRight=0.9, mouthUpperUpLeft=0.5,
                            mouthUpperUpRight=0.5, mouthLowerDownLeft=0.5, mouthLowerDownRight=0.5,
                            browDownLeft=0.4, browDownRight=0.4, eyeSquintLeft=0.4, eyeSquintRight=0.4),
    "skeptical": _face_preset(browOuterUpLeft=0.8, browDownRight=0.6, eyeSquintRight=0.5, mouthPressLeft=0.3,
                              mouthRight=0.25),
    "determined": _face_preset(browDownLeft=0.55, browDownRight=0.55, mouthPressLeft=0.6, mouthPressRight=0.6,
                               jawForward=0.15, eyeSquintLeft=0.3, eyeSquintRight=0.3),
    "embarrassed": _face_preset(mouthSmileLeft=0.45, mouthSmileRight=0.45, browInnerUp=0.5, mouthPressLeft=0.3,
                                mouthPressRight=0.3, eyeSquintLeft=0.3, eyeSquintRight=0.3),
}
EXPR_ALIASES = {"happy": "smile", "grin": "big_smile", "joy": "laugh", "surprised": "surprise", "shocked": "shock",
                "scared": "fear", "afraid": "fear", "disgusted": "disgust", "mad": "angry", "anger": "angry",
                "unhappy": "sad", "crying": "sad", "puzzled": "confused", "think": "thinking", "pucker": "kiss",
                "funny": "silly", "silly_face": "silly", "tired": "sleepy", "smug": "smirk", "pain": "wince",
                "doubt": "skeptical", "doubtful": "skeptical", "shy": "embarrassed", "focused": "determined",
                "astonished": "shock", "amazed": "surprise", "stunned": "shock", "flabbergasted": "shock",
                "excited": "big_smile", "joyful": "big_smile", "ecstatic": "laugh", "delighted": "big_smile",
                "furious": "angry", "annoyed": "frown", "grumpy": "frown", "irritated": "frown", "nervous": "fear",
                "anxious": "fear", "worried": "sad", "terrified": "fear", "bored": "sleepy", "proud": "smirk",
                "cheeky": "smirk", "playful": "silly", "curious": "confused", "suspicious": "skeptical",
                "content": "smile", "calm": "neutral", "relaxed": "neutral", "serious": "determined", "upset": "sad",
                "disappointed": "sad", "laughing": "laugh", "smiling": "smile", "yawning": "yawn", "grossed_out": "disgust"}

# face channels an expression "owns" (reset to neutral by expression())
EXPR_KEYS = [n for n in BS_NAMES if not n.startswith("eyeLook")]


# ============================================================================================ program builder
class Builder:
    """symbolic executor: API calls append Segs/Oscs at the current cursor."""

    def __init__(self, nat):
        self.nat = nat
        self.segs, self.oscs = [], []
        self.warnings = []
        self.stack = [{"mode": "seq", "t": 0.0, "start": 0.0, "end": 0.0}]
        self.last_arm = {"l": None, "r": None}   # last IK spec per arm (for move_hand)
        self.ended_with_return = False
        self.keep_end_pose = False
        self.n_calls = 0
        self.expr_keys = set()
        self.explicit_keys = set()
        self.goals = []          # (t_check_start, t_check_end, kind, spec) for the validator

    # ---- time bookkeeping
    @property
    def ctx(self):
        return self.stack[-1]

    def start_time(self):
        c = self.ctx
        return c["start"] if c["mode"] == "par" else c["t"]

    def advance(self, t_end):
        c = self.ctx
        if c["mode"] == "par":
            c["end"] = max(c["end"], t_end)
        else:
            c["t"] = t_end
            c["end"] = max(c["end"], t_end)

    def push(self, mode, delay=0.0):
        t = self.start_time() + delay
        self.stack.append({"mode": mode, "t": t, "start": t, "end": t})

    def pop(self):
        c = self.stack.pop()
        self.advance(max(c["end"], c["t"]))

    def warn(self, msg):
        if msg not in self.warnings:
            self.warnings.append(msg)

    def ease(self, e, big=False, contact=False):
        if e:
            return e
        if not self.nat["min_jerk"]:
            return "linear"
        if big and self.nat["overshoot"] and not contact:
            return "settle"
        return "min_jerk"

    def seg(self, chan, t0, dur, tgt, ease=None, kind="", src="", big=False, contact=False, prio=1):
        """prio 0 = a macro's secondary side effect (loses same-time ties against explicit calls)."""
        s = Seg(chan, t0, t0 + max(dur, 1.0 / FPS), tgt, self.ease(ease, big, contact), kind, src, contact, prio)
        if self.nat["anticipation"] and big and not ease and chan in ("torso", "root", "head") and dur >= 0.3:
            s.kind = (kind + "+anticip").strip("+")
        self.segs.append(s)
        return s


# ============================================================================================ argument helpers
_SIDE = {"left": "l", "l": "l", "right": "r", "r": "r"}


def _sides(hand):
    h = str(hand).lower()
    if h in ("both", "two", "hands", "all"):
        return ["l", "r"]
    if h not in _SIDE:
        raise ProgramError(f"hand must be 'left', 'right' or 'both', got {hand!r}")
    return [_SIDE[h]]


def _num(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ProgramError(f"{name} must be a number, got {v!r}")
    return float(v)


def _dur(v, default):
    if v is None:
        return default
    d = _num(v, "dur")
    if d < 0 or d > 8:
        raise ProgramError(f"dur must be between 0 and 8 seconds, got {d}")
    return d


# live targets (user / scene things): {name: (x_cm, y_cm, z_cm) pelvis frame}, set per Stream around every exec
LIVE_TARGETS = contextvars.ContextVar("semantic_live_targets", default={})
LIVE_ALIASES = {"user_face": "user", "viewer_face": "user", "person": "user", "the_user": "user", "you": "user",
                "me": "user", "face_of_user": "user"}
WARN_SINK = contextvars.ContextVar("semantic_warn_sink", default=None)


def _twarn(msg):
    w = WARN_SINK.get()
    if w is not None and msg not in w:
        w.append(msg)


def _check_target(t):
    if isinstance(t, (list, tuple)):
        if len(t) != 3:
            raise ProgramError("a coordinate target must be (x_cm, y_cm, z_cm)")
        try:
            v = tuple(float(x) / 100.0 for x in t)
        except (TypeError, ValueError):
            raise ProgramError("a coordinate target must be (x_cm, y_cm, z_cm) numbers")
        if not all(math.isfinite(x) for x in v):
            raise ProgramError("a coordinate target must be finite")
        return ("xyz", tuple(float(np.clip(x, -5.0, 5.0)) for x in v))
    if not isinstance(t, str):
        raise ProgramError(f"a target is a name or (x_cm, y_cm, z_cm), got {t!r}")
    raw = t.strip().lower().replace(" ", "_").replace("-", "_")
    live = LIVE_TARGETS.get()
    ln = LIVE_ALIASES.get(raw, raw)
    if ln in live:          # live targets are WORLD-fixed (the person, the room): re-aimed after body turns
        return ("xyz", tuple(float(x) / 100.0 for x in live[ln]), "world")
    if ln == "user":                      # no perception of the person right now: the camera is where they are
        return "camera"
    n = kin.normalize_point_name(t)
    if n not in kin.POINT_NAMES:
        names = list(kin.POINT_NAMES) + list(live)
        close = difflib.get_close_matches(n, names, n=1, cutoff=0.75)
        if close:
            _twarn(f"unknown target {t!r}: used {close[0]!r}")
            return _check_target(close[0])
        _twarn(f"unknown target {t!r}: used 'camera'")
        return "camera"
    return n


def _offset(off):
    if off is None:
        return np.zeros(3)
    if isinstance(off, dict):
        v = np.zeros(3)
        for k, x in off.items():
            x = _num(x, "offset") / 100.0
            k = str(k).lower()
            if k in ("left",):
                v[0] += x
            elif k == "right":
                v[0] -= x
            elif k == "up":
                v[1] += x
            elif k == "down":
                v[1] -= x
            elif k in ("forward", "front", "out"):
                v[2] += x
            elif k in ("back", "backward"):
                v[2] -= x
            else:
                raise ProgramError(f"offset keys are left/right/up/down/forward/back, got {k!r}")
        return v
    if isinstance(off, (list, tuple)) and len(off) == 3:
        return np.array([float(x) for x in off]) / 100.0
    raise ProgramError("offset_cm must be (x, y, z) in cm (x = person's left) or dict(up=5, forward=3, ...)")


# ============================================================================================ the API
API = {}


def api(name=None, aliases=()):
    def deco(fn):
        API[name or fn.__name__] = fn
        for a in aliases:
            API[a] = fn
        return fn
    return deco


# ---- composition ---------------------------------------------------------------------------------------------------
@api()
def hold(B, seconds=0.5):
    d = _dur(seconds, 0.5)
    B.advance(B.start_time() + d)


API["wait"] = hold
API["pause"] = hold


@api()
def return_to_neutral(B, dur=0.8, parts="all"):
    d = _dur(dur, 0.8)
    t0 = B.start_time()
    chans = ["root", "torso", "head", "gaze", "arm_l", "arm_r", "fing_l", "fing_r", "face"] if parts == "all" else \
        _parts_to_chans(parts)
    for c in chans:
        B.seg(c, t0, d, {"neutral": True}, kind="neutral", src="return_to_neutral")
    for s in "lr":
        if "arm_" + s in chans:
            B.last_arm[s] = None
    if "face" in chans:
        B.explicit_keys = set(); B.expr_keys = set()
    B.ended_with_return = True
    B.advance(t0 + d)


API["reset"] = return_to_neutral
API["neutral"] = return_to_neutral


def _parts_to_chans(parts):
    if isinstance(parts, str):
        parts = [parts]
    out = []
    m = {"head": ["head", "gaze"], "eyes": ["gaze"], "gaze": ["gaze"], "torso": ["torso"], "body": ["root", "torso"],
         "root": ["root"], "arms": ["arm_l", "arm_r", "fing_l", "fing_r"], "left_arm": ["arm_l", "fing_l"],
         "right_arm": ["arm_r", "fing_r"], "left_hand": ["arm_l", "fing_l"], "right_hand": ["arm_r", "fing_r"],
         "hands": ["arm_l", "arm_r", "fing_l", "fing_r"], "face": ["face"], "fingers": ["fing_l", "fing_r"],
         "left": ["arm_l", "fing_l"], "right": ["arm_r", "fing_r"]}
    for p in parts:
        if p not in m:
            raise ProgramError(f"unknown part {p!r}; one of {sorted(m)}")
        out += m[p]
    return out


@api(aliases=("relax",))
def release(B, part="arms", dur=0.6):
    d = _dur(dur, 0.6)
    t0 = B.start_time()
    for c in _parts_to_chans(part):
        B.seg(c, t0, d, {"neutral": True}, kind="neutral", src="release")
        if c.startswith("arm_"):
            B.last_arm[c[-1]] = None
    B.advance(t0 + d)


@api()
def keep_pose(B):
    """end the clip in the current pose (no automatic return to neutral)."""
    B.keep_end_pose = True


# ---- arms / hands ----------------------------------------------------------------------------------------------------
def _arm_goal(B, s, target, part="palm", palm=None, offset_cm=None, shape=None, elbow=None, dur=0.7, ease=None,
              finger_dir=None, src="reach", contact=None, osc=None):
    if isinstance(target, str) and kin.normalize_point_name(target) in ("left_shoulder", "right_shoulder"):
        target = kin.normalize_point_name(target) + "_top"     # touching a shoulder = its top surface
    tgt = _check_target(target)
    if part not in ("palm", "wrist", "index_tip", "middle_tip", "thumb_tip", "fingertips", "knuckles",
                    "back_of_hand", "pinky_tip", "ring_tip"):
        raise ProgramError(f"part must be palm / index_tip / fingertips / knuckles / back_of_hand / thumb_tip / "
                           f"wrist, got {part!r}")
    on_body = isinstance(tgt, str) and kin.inward_normal(tgt, kin.neutral()) is not None
    if contact is None:
        contact = on_body and (offset_cm is None or np.linalg.norm(_offset(offset_cm)) < 0.04)
    if palm is None and on_body and part in ("palm", "fingertips", "index_tip", "middle_tip"):
        palm = "body"
    if finger_dir is None and contact and part in ("palm", "back_of_hand") and isinstance(tgt, str) and tgt in (
            "left_ear", "right_ear", "left_temple", "right_temple", "left_cheek", "right_cheek", "left_eye",
            "right_eye", "eyes", "forehead"):
        finger_dir = "up"                 # a flat hand on the side/front of the head: fingers point up, not across
    spec = dict(target=tgt, part=part, palm=palm, offset=_offset(offset_cm), elbow=elbow, finger_dir=finger_dir)
    t0 = B.start_time()
    d = _dur(dur, 0.7)
    if shape is not None:
        try:
            kin.hand_shape(shape, s)
        except ValueError as e:
            raise ProgramError(str(e))
        B.seg("fing_" + s, t0 + 0.15 * d if B.nat["successive"] else t0, d * 0.85 if B.nat["successive"] else d,
              {"shape": shape}, src=src)
    B.seg("arm_" + s, t0, d, {"ik": spec}, ease=ease, src=src, contact=contact, big=True)
    B.last_arm[s] = spec
    B.goals.append((t0 + d, s, spec, src))
    return t0 + d


@api(aliases=("place_hand", "put_hand"))
def reach(B, hand="right", target="in_front_of_chest", part="palm", palm=None, offset_cm=None, shape=None,
          elbow=None, dur=0.7, ease=None, finger_dir=None):
    pair = {"eyes": "eye", "ears": "ear", "cheeks": "cheek", "temples": "temple", "shoulders": "shoulder_top",
            "hips": "hip", "chests": "chest", "knees": "knee", "thighs": "thigh"}
    ends = [_arm_goal(B, s, target if not (len(_sides(hand)) == 2 and target in pair)
                      else f"{'left' if s == 'l' else 'right'}_{pair[target]}",
                      part, palm, offset_cm, shape, elbow, dur, ease, finger_dir, "reach") for s in _sides(hand)]
    B.advance(max(ends))


@api()
def touch(B, hand="right", target="nose", part="index_tip", dur=0.7, shape=None):
    if shape is None:
        shape = "point" if part == "index_tip" else ("flat" if part in ("palm", "fingertips") else None)
    ends = [_arm_goal(B, s, target, part, None, None, shape, None, dur, None, None, "touch") for s in _sides(hand)]
    B.advance(max(ends))


@api()
def cover(B, target="eyes", hand="both", dur=0.6):
    t = kin.normalize_point_name(target)
    sides = _sides(hand)
    for x in ("eye", "ear", "cheek", "temple"):        # both hands on ONE eye/ear makes no sense -> the pair
        if len(sides) == 2 and t in (f"left_{x}", f"right_{x}"):
            t = x + "s"
    if t == "temples":
        t = "temples_"
    ends = []
    for s in sides:
        sd = "left" if s == "l" else "right"
        if t == "temples_":
            tg = f"{sd}_temple"
        elif t in ("eyes", "ears", "cheeks") or t in ("eye", "ear"):
            base = t.rstrip("s")
            tg = f"{sd}_{base}" if len(sides) == 2 else ("eyes" if base == "eye" else f"{sd}_{base}")
        else:
            tg = t
        off = None
        if len(sides) == 2 and tg in ("mouth", "nose", "chin", "forehead", "face"):
            off = {"left": 2.5 if s == "l" else -2.5, "forward": 1.5 if s == "l" else 0.0}
        part = "palm"
        ends.append(_arm_goal(B, s, tg, part, "body", off, "flat", None, dur, None, None, "cover", contact=True))
    B.advance(max(ends))


@api()
def point_at(B, target="camera", hand="right", dur=0.6):
    tgt = _check_target(target)
    for s in _sides(hand):
        if isinstance(tgt, str) and kin.inward_normal(tgt, kin.neutral()) is not None:
            _arm_goal(B, s, tgt, "index_tip", None, {"forward": 0}, "point", None, dur, None, None, "point_at",
                      contact=True)
        else:
            spec = dict(target=tgt, part="index_tip", palm=None, offset=np.zeros(3), elbow=None, finger_dir=None,
                        aim=True)
            t0 = B.start_time()
            d = _dur(dur, 0.6)
            B.seg("fing_" + s, t0 + (0.1 * d if B.nat["successive"] else 0), d * 0.9, {"shape": "point"}, src="point_at")
            B.seg("arm_" + s, t0, d, {"ik": spec}, src="point_at", big=True)
            B.last_arm[s] = spec
            B.goals.append((t0 + d, s, spec, "point_at"))
    B.advance(B.start_time() + _dur(dur, 0.6))


_DIRS = {"up": (0.25, 1.0, 0.25), "forward": (0.15, 0.0, 1.0), "out": (1.0, 0.0, 0.25), "side": (1.0, 0.0, 0.2),
         "down": (0.25, -1.0, 0.2), "across": (-1.0, 0.0, 0.6), "up_out": (0.8, 0.8, 0.2), "forward_up": (0.1, 0.7, 1.0),
         "forward_down": (0.1, -0.5, 1.0), "back": (0.3, -0.6, -0.8), "overhead": (0.0, 1.0, 0.1)}
for _a, _b in (("sideways", "side"), ("sideway", "side"), ("outward", "out"), ("outwards", "out"), ("upward", "up"),
               ("upwards", "up"), ("downward", "down"), ("downwards", "down"), ("front", "forward"),
               ("ahead", "forward"), ("straight", "forward"), ("above", "overhead"), ("up_forward", "forward_up"),
               ("down_forward", "forward_down"), ("out_up", "up_out")):
    _DIRS[_a] = _DIRS[_b]


@api()
def arm(B, hand="right", direction="forward", extend=0.95, palm=None, shape=None, dur=0.7, ease=None, elbow=None,
        finger_dir=None):
    """straighten the arm toward a body-relative direction (or toward a named point) from the shoulder."""
    for s in _sides(hand):
        sg = 1.0 if s == "l" else -1.0
        dname = str(direction).lower()
        if dname in _DIRS:
            dv = np.array(_DIRS[dname], float) * np.array([sg, 1, 1])
            dv /= np.linalg.norm(dv)
            ext = float(np.clip(_num(extend, "extend"), 0.2, 1.0))
            spec = dict(target=("shoulder_dir", s, tuple(dv), ext), part="wrist", palm=palm, offset=np.zeros(3),
                        elbow=elbow, finger_dir=finger_dir)
            t0 = B.start_time(); d = _dur(dur, 0.7)
            if shape:
                B.seg("fing_" + s, t0 + 0.1 * d, 0.9 * d, {"shape": shape}, src="arm")
            B.seg("arm_" + s, t0, d, {"ik": spec}, ease=ease, src="arm", big=True)
            B.last_arm[s] = spec
            B.goals.append((t0 + d, s, spec, "arm"))
        else:
            point_at(B, direction, "left" if s == "l" else "right", dur)
    B.advance(B.start_time() + _dur(dur, 0.7))


@api(aliases=("hand_shape", "make_hand", "fingers"))
def hand(B, hand="right", shape="open", amount=1.0, dur=0.3):
    t0 = B.start_time(); d = _dur(dur, 0.3)
    for s in _sides(hand):
        try:
            kin.hand_shape(shape, s)
        except ValueError as e:
            raise ProgramError(str(e))
        B.seg("fing_" + s, t0, d, {"shape": shape, "amount": float(amount)}, src="hand")
    B.advance(t0 + d)


@api()
def move_hand(B, hand="right", left=0.0, up=0.0, forward=0.0, dur=0.4, ease=None):
    """shift the hand's current goal by cm in the body frame (+left = toward the person's left)."""
    t0 = B.start_time(); d = _dur(dur, 0.4)
    for s in _sides(hand):
        base = B.last_arm[s]
        if base is None:
            base = dict(target=f"{'left' if s == 'l' else 'right'}_wrist_rest", part="wrist", palm=None,
                        offset=np.zeros(3), elbow=None, finger_dir=None)
        spec = dict(base)
        spec["offset"] = np.asarray(base["offset"]) + np.array([_num(left, "left"), _num(up, "up"),
                                                                _num(forward, "forward")]) / 100.0
        B.seg("arm_" + s, t0, d, {"ik": spec}, ease=ease, src="move_hand")
        B.last_arm[s] = spec
        B.goals.append((t0 + d, s, spec, "move_hand"))
    B.advance(t0 + d)


@api()
def rotate_wrist(B, hand="right", flex=0.0, side=0.0, twist=0.0, dur=0.3):
    """additive wrist rotation in degrees: flex + = palm-ward, side + = toward the thumb, twist + = forearm roll."""
    t0 = B.start_time(); d = _dur(dur, 0.3)
    for s in _sides(hand):
        B.seg("wrist_" + s, t0, d, {"wr": (_num(flex, "flex"), _num(side, "side"), _num(twist, "twist"))},
              src="rotate_wrist")
    B.advance(t0 + d)


@api()
def raise_hand(B, hand="right", height="high", palm="camera", shape="open", dur=0.7):
    for s in _sides(hand):
        sg = 1.0 if s == "l" else -1.0
        h = {"high": (0.12, 0.45, 0.12), "head": (0.14, 0.27, 0.16), "shoulder": (0.2, 0.05, 0.22),
             "chest": (0.06, -0.08, 0.3)}.get(str(height))
        if h is None:
            raise ProgramError("height must be high / head / shoulder / chest")
        spec = dict(target=("shoulder_off", s, (sg * h[0], h[1], h[2])), part="wrist", palm=palm,
                    offset=np.zeros(3), elbow=(sg * 0.8, -0.75, -0.1), finger_dir="up")
        t0 = B.start_time(); d = _dur(dur, 0.7)
        B.seg("fing_" + s, t0 + 0.15 * d, 0.85 * d, {"shape": shape}, src="raise_hand")
        B.seg("arm_" + s, t0, d, {"ik": spec}, src="raise_hand", big=True)
        B.last_arm[s] = spec
        B.goals.append((t0 + d, s, spec, "raise_hand"))
    B.advance(B.start_time() + _dur(dur, 0.7))


@api()
def wave(B, hand="right", cycles=3, size_cm=12, speed=1.0, raise_dur=0.6):
    """raise the hand beside the head, palm to the camera, and wave side to side; the hand stays up afterwards."""
    sides = _sides(hand)
    t0 = B.start_time()
    rd = _dur(raise_dur, 0.6)
    B.push("par")
    for s in sides:
        raise_hand(B, "left" if s == "l" else "right", "head", "camera", "open", rd)
    B.pop()
    cyc = float(np.clip(_num(cycles, "cycles"), 1, 8))
    per = 0.42 / max(0.3, _num(speed, "speed"))
    t1 = t0 + rd - 0.1
    for s in sides:
        B.oscs.append(Osc("hand_" + s, "x", t1, t1 + cyc * per, _num(size_cm, "size_cm") / 200.0, cyc, src="wave"))
        B.oscs.append(Osc("wrist_" + s, "side", t1, t1 + cyc * per, 18.0, cyc, phase=0.4, src="wave"))
    B.advance(t1 + cyc * per)


@api()
def clap(B, times=3, speed=1.0):
    n = int(np.clip(_num(times, "times"), 1, 10))
    per = 0.38 / max(0.3, _num(speed, "speed"))
    t0 = B.start_time()
    for s in "lr":
        sg = 1.0 if s == "l" else -1.0
        spec = dict(target="in_front_of_chest", part="palm", palm="in", offset=np.array([sg * 0.12, -0.02, -0.02]),
                    elbow=None, finger_dir="forward_up")
        B.seg("fing_" + s, t0, 0.3, {"shape": "flat"}, src="clap")
        B.seg("arm_" + s, t0, 0.5, {"ik": spec}, src="clap", big=True)
        B.last_arm[s] = spec
        # the hands meet: osc pulls each palm toward the midline (bias + amp = full closure)
        B.oscs.append(Osc("hand_" + s, "x", t0 + 0.5, t0 + 0.5 + n * per, -sg * 0.0575, n, phase=-math.pi / 2,
                          src="clap", bias=-sg * 0.0575))
    B.advance(t0 + 0.5 + n * per)


def _surface_osc(B, hand, target, times, cm, part, shape, mode, speed, src):
    sides = _sides(hand)
    n = float(np.clip(_num(times, "times"), 1, 12))
    per = (0.38 if mode == "pat" else 0.32) / max(0.3, _num(speed, "speed"))
    a = float(np.clip(abs(_num(cm, "cm")), 1, 15)) / 100.0
    t0 = B.start_time()
    for s in sides:
        tg = target
        if len(sides) == 2 and target in ("eyes", "ears", "cheeks", "temples", "shoulders", "hips"):
            tg = f"{'left' if s == 'l' else 'right'}_{target[:-1] if target != 'shoulders' else 'shoulder_top'}"
        _arm_goal(B, s, tg, part, None, None, shape, None, 0.6, None, None, src, contact=True)
        t1 = t0 + 0.6
        if mode == "pat":           # lift off along the body-surface normal and come back
            B.oscs.append(Osc("hand_" + s, "normal", t1, t1 + n * per, -a / 2, n, phase=math.pi / 2, bias=a / 2,
                              src=src))
        else:                       # slide side to side / up and down along the surface
            B.oscs.append(Osc("hand_" + s, "x" if mode == "rub_x" else "y", t1, t1 + n * per, a / 2, n, src=src))
    B.advance(t0 + 0.6 + n * per)


@api(aliases=("tap",))
def pat(B, hand="right", target="top_of_head", times=3, cm=5, part="palm", speed=1.0):
    """repeatedly lift the hand off a body target and put it back (pat head, tap temple/chest, pat belly)."""
    _surface_osc(B, hand, target, times, cm, part, "flat" if part == "palm" else "point", "pat", speed, "pat")


@api(aliases=("scratch",))
def rub(B, hand="right", target="belly", times=4, cm=6, part="palm", direction="sideways", speed=1.0):
    """slide the hand back and forth on a body target (rub tummy, scratch head with part='fingertips', rub eyes)."""
    mode = "rub_x" if str(direction).startswith("side") else "rub_y"
    _surface_osc(B, hand, target, times, cm, part, "flat" if part == "palm" else "claw", mode, speed, "rub")


@api()
def hands_together(B, where="chest", fingers="up", dur=0.6):
    """both palms pressed together (praying, rubbing hands, holding hands) ~20 cm in front of a body target (default
    the chest; 'face', 'chin', 'belly' ...), fingers 'up' / 'forward'."""
    t0 = B.start_time()
    w = _check_target(where)
    fwd = 0.2 if (isinstance(w, str) and kin.inward_normal(w, kin.neutral()) is not None) else 0.0
    if w == "in_front_of_chest":
        w, fwd = "chest", 0.2
    for s in "lr":
        sg = 1.0 if s == "l" else -1.0
        spec = dict(target=w, part="palm", palm="in", offset=np.array([sg * 0.012, 0.0, fwd]),
                    elbow=None, finger_dir="up" if fingers == "up" else "forward")
        B.seg("fing_" + s, t0, dur * 0.8, {"shape": "flat"}, src="hands_together")
        B.seg("arm_" + s, t0, _dur(dur, 0.6), {"ik": spec}, src="hands_together", big=True, contact=True)
        B.last_arm[s] = spec
        B.goals.append((t0 + _dur(dur, 0.6), s, spec, "hands_together"))
    B.advance(t0 + _dur(dur, 0.6))


@api()
def hands_on_hips(B, dur=0.7):
    t0 = B.start_time()
    for s in "lr":
        _arm_goal(B, s, f"{'left' if s == 'l' else 'right'}_hip", "palm", "body", None, "flat", "out", dur, None,
                  None, "hands_on_hips", contact=True)
    B.advance(t0 + _dur(dur, 0.7))


@api()
def cross_arms(B, dur=0.8):
    t0 = B.start_time()
    _arm_goal(B, "l", "right_upper_arm", "palm", "body", {"forward": 7, "down": 4}, "relaxed", None, dur, None, None,
              "cross_arms", contact=True)
    _arm_goal(B, "r", "left_upper_arm", "palm", "body", {"forward": 12, "down": 6}, "relaxed", None, dur, None, None,
              "cross_arms", contact=True)
    B.advance(t0 + _dur(dur, 0.8))


@api()
def thumbs_up(B, hand="right", dur=0.6):
    for s in _sides(hand):
        sg = 1.0 if s == "l" else -1.0
        spec = dict(target=("shoulder_off", s, (sg * 0.05, -0.12, 0.36)), part="wrist", palm="in", offset=np.zeros(3),
                    elbow=None, finger_dir="forward")
        t0 = B.start_time(); d = _dur(dur, 0.6)
        B.seg("fing_" + s, t0 + 0.2 * d, 0.8 * d, {"shape": "thumbs_up"}, src="thumbs_up")
        B.seg("arm_" + s, t0, d, {"ik": spec}, src="thumbs_up", big=True)
        B.last_arm[s] = spec
    B.advance(B.start_time() + _dur(dur, 0.6))


@api()
def rest_arm(B, hand="both", dur=0.6):
    t0 = B.start_time(); d = _dur(dur, 0.6)
    for s in _sides(hand):
        B.seg("arm_" + s, t0, d, {"neutral": True}, src="rest_arm", kind="neutral")
        B.seg("fing_" + s, t0, d, {"neutral": True}, src="rest_arm", kind="neutral")
        B.last_arm[s] = None
    B.advance(t0 + d)


# ---- head / eyes ---------------------------------------------------------------------------------------------------
@api(aliases=("look",))
def look_at(B, target="camera", dur=0.5, eyes_only=False):
    tgt = _check_target(target)
    t0 = B.start_time(); d = _dur(dur, 0.5)
    if not eyes_only:
        B.seg("head", t0, d, {"look": tgt}, src="look_at", big=True)
    gd = d * 0.4 if B.nat["eye_lead"] else d
    B.seg("gaze", t0, gd, {"look": tgt, "eyes_only": bool(eyes_only)}, src="look_at")
    B.goals.append((t0 + d, "head", {"look": tgt}, "look_at"))
    B.advance(t0 + d)


@api()
def rotate(B, part="head", yaw=0.0, pitch=0.0, roll=0.0, dur=0.5, ease=None):
    """rotate head / torso / body (whole body, root) by degrees relative to neutral (absolute pose, not cumulative):
    yaw + = turn to the person's LEFT, pitch + = up (head) / lean BACK (torso, body), roll + = tilt to the person's
    LEFT."""
    t0 = B.start_time(); d = _dur(dur, 0.5)
    y, p, r = _num(yaw, "yaw"), _num(pitch, "pitch"), _num(roll, "roll")
    part = str(part).lower()
    big = max(abs(y), abs(p), abs(r)) > 10
    if part == "head":
        B.seg("head", t0, d, {"ypr": (y, p, r)}, ease=ease, src="rotate", big=big)
        if B.nat["eye_lead"] and abs(y) + abs(p) > 8:
            B.seg("gaze", t0, d * 0.35, {"ypr": (float(np.clip(y * 0.35, -20, 20)), float(np.clip(p * 0.35, -15, 15)))},
                  src="rotate", prio=0)
            B.seg("gaze", t0 + d * 0.5, d * 0.6, {"ypr": (0.0, 0.0)}, src="rotate", prio=0)
    elif part in ("torso", "chest", "spine", "upper_body"):
        B.seg("torso", t0, d, {"lean_fwd": -p, "lean_side": r, "twist": y}, ease=ease, src="rotate", big=big)
    elif part in ("body", "root", "whole_body", "hips"):
        B.seg("root", t0, d, {"yaw": y, "pitch": -p, "roll": r}, ease=ease, src="rotate", big=big)
    else:
        raise ProgramError("rotate part must be head / torso / body")
    B.advance(t0 + d)


@api()
def tilt_head(B, side="left", deg=15.0, dur=0.5):
    sg = 1.0 if _SIDE.get(str(side).lower()) == "l" else -1.0
    rotate(B, "head", 0, 0, sg * abs(_num(deg, "deg")), dur)


@api(aliases=("turn_head",))
def head_turn(B, side="left", deg=30.0, dur=0.5):
    sg = 1.0 if _SIDE.get(str(side).lower()) == "l" else -1.0
    rotate(B, "head", sg * abs(_num(deg, "deg")), 0, 0, dur)


@api()
def nod(B, times=2, deg=12.0, speed=1.0):
    n = float(np.clip(_num(times, "times"), 1, 8))
    per = 0.5 / max(0.3, _num(speed, "speed"))
    t0 = B.start_time()
    a = abs(_num(deg, "deg")) / 2
    B.oscs.append(Osc("head", "pitch", t0, t0 + n * per, a, n, phase=math.pi / 2, src="nod", bias=-a))
    B.advance(t0 + n * per)


@api(aliases=("shake",))
def shake_head(B, times=2, deg=18.0, speed=1.0):
    n = float(np.clip(_num(times, "times"), 1, 8))
    per = 0.55 / max(0.3, _num(speed, "speed"))
    t0 = B.start_time()
    B.oscs.append(Osc("head", "yaw", t0, t0 + n * per, abs(_num(deg, "deg")), n, src="shake_head"))
    B.advance(t0 + n * per)


@api()
def gaze(B, yaw=0.0, pitch=0.0, dur=0.2):
    t0 = B.start_time(); d = _dur(dur, 0.2)
    B.seg("gaze", t0, d, {"ypr": (_num(yaw, "yaw"), _num(pitch, "pitch"))}, src="gaze")
    B.advance(t0 + d)


@api()
def roll_eyes(B, dur=0.9):
    t0 = B.start_time(); d = _dur(dur, 0.9)
    pts = [(25, 0), (15, 22), (0, 25), (-15, 22), (-25, 5), (0, 0)]
    for k, (y, p) in enumerate(pts):
        B.seg("gaze", t0 + k * d / len(pts), d / len(pts), {"ypr": (y, p)}, ease="ease_in_out", src="roll_eyes")
    B.advance(t0 + d)


# ---- torso / root --------------------------------------------------------------------------------------------------
@api()
def lean(B, direction="forward", deg=15.0, dur=0.6):
    dn = str(direction).lower(); a = abs(_num(deg, "deg"))
    tg = {"forward": {"lean_fwd": a}, "back": {"lean_fwd": -a}, "backward": {"lean_fwd": -a},
          "left": {"lean_side": a}, "right": {"lean_side": -a}}.get(dn)
    if tg is None:
        raise ProgramError("lean direction: forward / back / left / right")
    t0 = B.start_time(); d = _dur(dur, 0.6)
    B.seg("torso", t0, d, tg, src="lean", big=a > 10)
    B.advance(t0 + d)


@api()
def twist(B, direction="left", deg=25.0, dur=0.6):
    sg = 1.0 if _SIDE.get(str(direction).lower()) == "l" else -1.0
    t0 = B.start_time(); d = _dur(dur, 0.6)
    B.seg("torso", t0, d, {"twist": sg * abs(_num(deg, "deg"))}, src="twist", big=True)
    B.advance(t0 + d)


@api()
def shoulders(B, side="both", amount=1.0, dur=0.35):
    """set shoulder elevation (collar): 1 = fully raised, 0 = normal, negative = dropped."""
    t0 = B.start_time(); d = _dur(dur, 0.35)
    for s in _sides(side):
        B.seg("shrug_" + s, t0, d, {"amount": _num(amount, "amount")}, src="shoulders")
    B.advance(t0 + d)


@api()
def shrug(B, side="both", amount=1.0, hold_s=0.35, palms_up=True):
    """gesture: shoulders up (+ palms up/out, small head tilt, brows up), hold, back down."""
    t0 = B.start_time()
    a = _num(amount, "amount")
    for s in _sides(side):
        B.seg("shrug_" + s, t0, 0.3, {"amount": a}, src="shrug")
        B.seg("shrug_" + s, t0 + 0.3 + hold_s, 0.35, {"amount": 0.0}, src="shrug")
        if palms_up:
            sg = 1.0 if s == "l" else -1.0
            spec = dict(target=("shoulder_off", s, (sg * 0.16, -0.30, 0.25)), part="wrist", palm="up",
                        offset=np.zeros(3), elbow=None, finger_dir="out")
            B.seg("arm_" + s, t0, 0.35, {"ik": spec}, src="shrug")
            B.seg("fing_" + s, t0, 0.3, {"shape": "open"}, src="shrug")
            B.seg("arm_" + s, t0 + 0.35 + hold_s, 0.45, {"neutral": True}, src="shrug", kind="neutral")
            B.seg("fing_" + s, t0 + 0.35 + hold_s, 0.4, {"neutral": True}, src="shrug", kind="neutral")
    B.seg("face", t0, 0.3, {"shapes": {"browInnerUp": 0.6, "browOuterUpLeft": 0.5, "browOuterUpRight": 0.5,
                                       "mouthShrugLower": 0.4}}, src="shrug")
    B.seg("face", t0 + 0.35 + hold_s, 0.4, {"shapes": {"browInnerUp": 0.04, "browOuterUpLeft": 0.08,
                                                       "browOuterUpRight": 0.07, "mouthShrugLower": 0.02}},
          src="shrug")
    B.advance(t0 + 0.75 + hold_s)


@api()
def bow(B, deg=30.0, hold_s=0.4, dur=0.7):
    a = abs(_num(deg, "deg")); t0 = B.start_time(); d = _dur(dur, 0.7)
    sp = min(a, 25.0)
    B.seg("torso", t0, d, {"lean_fwd": sp}, src="bow", big=True)
    B.seg("root", t0, d, {"pitch": a - sp}, src="bow", big=True)
    B.seg("head", t0, d, {"ypr": (0.0, -10.0, 0.0)}, src="bow", prio=0)
    B.seg("torso", t0 + d + hold_s, d, {"lean_fwd": 0.0}, src="bow", big=True)
    B.seg("root", t0 + d + hold_s, d, {"pitch": 0.0}, src="bow", big=True)
    B.seg("head", t0 + d + hold_s, d, {"ypr": (0.0, 0.0, 0.0)}, src="bow", prio=0)
    B.advance(t0 + 2 * d + hold_s)


@api()
def jump(B, height_cm=25.0, times=1):
    """anticipation dip -> takeoff -> airborne -> landing dip -> settle, on the root (y)."""
    h = float(np.clip(_num(height_cm, "height_cm"), 3, 80))
    n = int(np.clip(_num(times, "times"), 1, 6))
    t = B.start_time()
    air = 2 * math.sqrt(2 * h / 100.0 / 9.81)               # ballistic time in the air
    dip = -min(12.0, 4 + 0.25 * h) if B.nat["anticipation"] else 0.0
    for k in range(n):
        if dip:
            B.seg("root", t, 0.22, {"y": dip}, ease="ease_in_out", src="jump", kind="jump")
            B.seg("torso", t, 0.22, {"lean_fwd": 8.0}, src="jump", prio=0)
            t += 0.22
        B.seg("root", t, air / 2 + 0.08, {"y": h}, ease="ease_out", src="jump", kind="jump")
        B.seg("torso", t, 0.2, {"lean_fwd": 0.0}, src="jump", prio=0)
        for s in "lr":   # arms swing up a little in flight
            B.seg("shrug_" + s, t, 0.2, {"amount": 0.4}, src="jump", prio=0)
        t += air / 2 + 0.08
        land = -0.5 * abs(dip) if dip else 0.0
        B.seg("root", t, air / 2 + 0.04, {"y": land}, ease="ease_in", src="jump", kind="jump")
        for s in "lr":
            B.seg("shrug_" + s, t, 0.25, {"amount": 0.0}, src="jump", prio=0)
        t += air / 2 + 0.04
        if land:
            B.seg("torso", t - 0.04, 0.2, {"lean_fwd": 5.0}, src="jump", prio=0)
            B.seg("root", t, 0.3, {"y": 0.0}, ease=B.ease(None, True), src="jump", kind="jump")
            B.seg("torso", t + 0.15, 0.3, {"lean_fwd": 0.0}, src="jump", prio=0)
            t += 0.3
    B.goals.append((t, "root", {"jump": h}, "jump"))
    B.advance(t)


@api()
def spin(B, deg=360.0, direction="left", dur=None):
    a = abs(_num(deg, "deg")); sg = 1.0 if _SIDE.get(str(direction).lower(), "l") == "l" else -1.0
    d = _dur(dur, max(0.6, a / 360.0 * 1.3))
    t0 = B.start_time()
    B.seg("root", t0, d, {"yaw_add": sg * a}, ease=B.ease(None, True), src="spin", kind="spin")
    B.advance(t0 + d)


@api()
def turn(B, direction="left", deg=45.0, dur=0.7):
    sg = 1.0 if _SIDE.get(str(direction).lower(), "l") == "l" else -1.0
    t0 = B.start_time(); d = _dur(dur, 0.7)
    B.seg("root", t0, d, {"yaw": sg * abs(_num(deg, "deg"))}, src="turn", big=True)
    B.advance(t0 + d)


@api(aliases=("squat",))
def crouch(B, depth_cm=25.0, dur=0.6):
    dpt = float(np.clip(abs(_num(depth_cm, "depth_cm")), 2, 60))
    t0 = B.start_time(); d = _dur(dur, 0.6)
    B.seg("root", t0, d, {"y": -dpt, "pitch": min(15.0, dpt * 0.3)}, src="crouch", big=True)
    B.seg("torso", t0, d, {"lean_fwd": min(15.0, dpt * 0.35)}, src="crouch", prio=0)
    B.advance(t0 + d)


@api(aliases=("stand_up",))
def stand(B, dur=0.6):
    t0 = B.start_time(); d = _dur(dur, 0.6)
    B.seg("root", t0, d, {"y": 0.0, "pitch": 0.0, "x": 0.0, "z": 0.0}, src="stand", big=True)
    B.seg("torso", t0, d, {"lean_fwd": 0.0}, src="stand", prio=0)
    B.advance(t0 + d)


@api()
def step(B, direction="left", cm=25.0, dur=0.6):
    dn = str(direction).lower(); c = abs(_num(cm, "cm"))
    v = {"left": ("x", c), "right": ("x", -c), "forward": ("z", c), "back": ("z", -c), "backward": ("z", -c)}.get(dn)
    if v is None:
        raise ProgramError("step direction: left / right / forward / back")
    t0 = B.start_time(); d = _dur(dur, 0.6)
    B.seg("root", t0, d, {v[0]: ("add", v[1])}, src="step", big=True)
    B.oscs.append(Osc("root", "y", t0, t0 + d, 0.75, 1.0, phase=math.pi / 2, src="step", bias=-0.75))
    B.advance(t0 + d)


@api()
def sway(B, cycles=2, deg=8.0, speed=1.0):
    n = float(np.clip(_num(cycles, "cycles"), 0.5, 8)); per = 1.2 / max(0.3, _num(speed, "speed"))
    t0 = B.start_time()
    B.oscs.append(Osc("torso", "lean_side", t0, t0 + n * per, abs(_num(deg, "deg")), n, src="sway"))
    B.oscs.append(Osc("root", "x", t0, t0 + n * per, -abs(_num(deg, "deg")) * 0.4, n, src="sway"))
    B.advance(t0 + n * per)


@api()
def bounce(B, times=4, cm=4.0, speed=1.0):
    n = float(np.clip(_num(times, "times"), 1, 12)); per = 0.45 / max(0.3, _num(speed, "speed"))
    t0 = B.start_time(); a = abs(_num(cm, "cm"))
    B.oscs.append(Osc("root", "y", t0, t0 + n * per, a / 2, n, phase=-math.pi / 2, src="bounce", bias=a / 2))
    B.advance(t0 + n * per)


@api()
def flip(B, direction="back", height_cm=40.0, dur=None):
    """jump + a full 360 deg somersault of the whole body about the left-right axis (direction back / forward)."""
    h = float(np.clip(_num(height_cm, "height_cm"), 15, 80))
    sg = -1.0 if str(direction).lower().startswith("back") else 1.0
    t = B.start_time()
    air = max(0.6, 2 * math.sqrt(2 * h / 100.0 / 9.81) + 0.15) if dur is None else _dur(dur, 0.8)
    B.seg("root", t, 0.22, {"y": -10.0}, ease="ease_in_out", src="flip", kind="jump")
    t += 0.22
    B.seg("root", t, air / 2, {"y": h}, ease="ease_out", src="flip", kind="jump")
    B.seg("rootp", t, air, {"pitch_add": sg * 360.0}, ease="ease_in_out", src="flip", kind="flip")
    B.seg("root", t + air / 2, air / 2, {"y": -6.0}, ease="ease_in", src="flip", kind="jump")
    B.seg("root", t + air, 0.3, {"y": 0.0}, ease="min_jerk", src="flip", kind="jump")
    B.advance(t + air + 0.3)


@api()
def oscillate(B, part="head", param="yaw", amplitude=10.0, cycles=2, dur=None, phase_deg=0.0):
    """generic additive oscillation: part head (yaw/pitch/roll deg) | torso (lean_fwd/lean_side/twist deg) |
    root (x/y/z cm, yaw/pitch/roll deg) | left_hand/right_hand (x/y/z cm, body frame) | gaze (yaw/pitch deg) |
    left_shoulder/right_shoulder (amount)."""
    n = float(np.clip(_num(cycles, "cycles"), 0.5, 12))
    d = _dur(dur, 0.5 * n)
    chan = {"head": "head", "torso": "torso", "root": "root", "body": "root", "gaze": "gaze",
            "left_hand": "hand_l", "right_hand": "hand_r", "left_shoulder": "shrug_l",
            "right_shoulder": "shrug_r"}.get(str(part))
    if chan is None:
        raise ProgramError("oscillate part: head/torso/root/gaze/left_hand/right_hand/left_shoulder/right_shoulder")
    a = _num(amplitude, "amplitude")
    if chan.startswith("hand_"):
        a /= 100.0
        if param not in ("x", "y", "z"):
            raise ProgramError("hand oscillation param is x / y / z")
    t0 = B.start_time()
    B.oscs.append(Osc(chan, str(param), t0, t0 + d, a, n, phase=math.radians(_num(phase_deg, "phase_deg")),
                      src="oscillate"))
    B.advance(t0 + d)


# ---- face ----------------------------------------------------------------------------------------------------------
@api(aliases=("blendshapes", "shapes"))
def face(B, dur=0.3, **shapes):
    t0 = B.start_time(); d = _dur(dur, 0.3)
    clean = {}
    for k, v in shapes.items():
        if k not in BS_NAMES:
            close = difflib.get_close_matches(k, BS_NAMES, n=2)
            raise ProgramError(f"unknown blendshape {k!r}" + (f" (did you mean {', '.join(close)}?)" if close else ""))
        clean[k] = float(np.clip(_num(v, k), 0, 1))
    B.seg("face", t0, d, {"shapes": clean}, src="face")
    for k_, v_ in clean.items():             # remembered as explicit until set back near neutral
        if abs(v_ - float(kin.neutral()[COL["bs_" + k_]])) > 0.1:
            B.explicit_keys.add(k_)
        else:
            B.explicit_keys.discard(k_)
    B.advance(t0 + d)


@api(aliases=("emotion", "make_face"))
def expression(B, name="smile", intensity=1.0, dur=0.4):
    nm = str(name).lower().replace(" ", "_").replace("-", "_")
    nm = EXPR_ALIASES.get(nm, nm)
    if nm not in EXPRESSIONS:
        close = difflib.get_close_matches(nm, list(EXPRESSIONS), n=3)
        raise ProgramError(f"unknown expression {name!r}" + (f" (did you mean {', '.join(close)}?)" if close else ""))
    k = float(np.clip(_num(intensity, "intensity"), 0, 1.5))
    n = kin.neutral()
    # reset the shapes the previous expression set, then apply this one (other face calls are left alone)
    shapes = {key: float(n[COL["bs_" + key]]) for key in B.expr_keys}
    for key, v in EXPRESSIONS[nm].items():
        base = float(n[COL["bs_" + key]])
        shapes[key] = float(np.clip(base + (v - base) * k, 0, 1))
    if nm == "neutral":
        B.explicit_keys = set()
    for key in B.explicit_keys:              # shapes the program set explicitly (closed eyes, raised brow...) win
        shapes.pop(key, None)
    B.expr_keys = set(EXPRESSIONS[nm])
    t0 = B.start_time(); d = _dur(dur, 0.4)
    B.seg("face", t0, d, {"shapes": shapes}, src="expression")
    B.advance(t0 + d)


@api()
def neutral_face(B, dur=0.4):
    expression(B, "neutral", 1.0, dur)


@api()
def smile(B, intensity=0.8, dur=0.4):
    expression(B, "smile" if intensity < 0.9 else "big_smile", min(1.0, intensity / 0.8 if intensity < 0.9 else 1.0), dur)


@api()
def wink(B, side="left", hold_s=0.25):
    s = _SIDE.get(str(side).lower())
    if s is None:
        raise ProgramError("wink side: left / right")
    L = "Left" if s == "l" else "Right"
    t0 = B.start_time()
    B.seg("face", t0, 0.12, {"shapes": {f"eyeBlink{L}": 1.0, f"cheekSquint{L}": 0.5, f"mouthSmile{L}": 0.45,
                                        f"eyeSquint{L}": 0.5}}, ease="ease_out", src="wink")
    B.seg("face", t0 + 0.12 + hold_s, 0.18, {"shapes": {f"eyeBlink{L}": 0.08, f"cheekSquint{L}": 0.0,
                                                        f"mouthSmile{L}": 0.1, f"eyeSquint{L}": 0.2},
                                             "restore_from": t0}, ease="ease_in_out", src="wink")
    B.goals.append((t0 + 0.12, "face", {"wink": s}, "wink"))
    B.advance(t0 + 0.3 + hold_s)


@api()
def blink(B, times=1):
    n = int(np.clip(_num(times, "times"), 1, 6)); t0 = B.start_time()
    for k in range(n):
        t = t0 + k * 0.3
        B.seg("face", t, 0.08, {"shapes": {"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0}}, ease="ease_out", src="blink")
        B.seg("face", t + 0.1, 0.12, {"shapes": {"eyeBlinkLeft": 0.08, "eyeBlinkRight": 0.08}, "restore_from": t},
              src="blink")
    B.advance(t0 + n * 0.3)


@api()
def close_eyes(B, side="both", amount=1.0, dur=0.25):
    sh = {}
    for s in _sides(side):
        L = "Left" if s == "l" else "Right"
        sh[f"eyeBlink{L}"] = float(np.clip(_num(amount, "amount"), 0, 1)); sh[f"eyeWide{L}"] = 0.0
    face(B, dur, **sh)


@api()
def open_eyes(B, side="both", dur=0.25):
    sh = {}
    for s in _sides(side):
        L = "Left" if s == "l" else "Right"
        sh[f"eyeBlink{L}"] = 0.08
    face(B, dur, **sh)


@api(aliases=("raise_eyebrows", "raise_brows"))
def raise_brow(B, side="both", amount=1.0, dur=0.3):
    a = float(np.clip(_num(amount, "amount"), 0, 1)); sh = {}
    sides = _sides(side)
    for s in sides:
        L = "Left" if s == "l" else "Right"
        sh[f"browOuterUp{L}"] = a; sh[f"browDown{L}"] = 0.0
    if len(sides) == 2:
        sh["browInnerUp"] = a
    else:                                   # one brow only: the other one stays down (quizzical)
        O = "Right" if sides[0] == "l" else "Left"
        sh[f"browOuterUp{O}"] = 0.0; sh[f"browDown{O}"] = 0.35 * a; sh["browInnerUp"] = 0.1
    face(B, dur, **sh)


@api(aliases=("furrow_brows",))
def furrow(B, amount=0.8, dur=0.3):
    a = float(np.clip(_num(amount, "amount"), 0, 1))
    face(B, dur, browDownLeft=a, browDownRight=a, browInnerUp=0.0)


@api()
def puff_cheeks(B, amount=1.0, dur=0.35):
    a = float(np.clip(_num(amount, "amount"), 0, 1))
    face(B, dur, cheekPuff=a, mouthClose=0.4 * a, jawOpen=0.0, mouthPucker=0.2 * a)


@api()
def open_mouth(B, amount=0.7, dur=0.3):
    a = float(np.clip(_num(amount, "amount"), 0, 1))
    face(B, dur, jawOpen=a, mouthClose=0.0, mouthLowerDownLeft=0.3 * a, mouthLowerDownRight=0.3 * a)


@api()
def close_mouth(B, dur=0.3):
    if any(sg.chan == "tongue" for sg in B.segs):
        B.seg("tongue", B.start_time(), _dur(dur, 0.3), {"amount": 0.0}, src="tongue")
    face(B, dur, jawOpen=0.03, mouthClose=0.01, mouthLowerDownLeft=0.0, mouthLowerDownRight=0.0, mouthFunnel=0.0,
         mouthPucker=0.02, cheekPuff=0.0)


@api()
def stick_out_tongue(B, dur=0.35):
    """canon has no tongue channel: approximated by an open jaw + lowered lower lip + funnel (renderer may differ)."""
    d = _dur(dur, 0.35)
    B.seg("tongue", B.start_time(), d, {"amount": 1.0}, src="tongue")
    face(B, dur, jawOpen=0.3)


@api()
def squint(B, amount=0.8, dur=0.3):
    a = float(np.clip(_num(amount, "amount"), 0, 1))
    face(B, dur, eyeSquintLeft=a, eyeSquintRight=a, eyeBlinkLeft=0.3 * a, eyeBlinkRight=0.3 * a,
         cheekSquintLeft=0.4 * a, cheekSquintRight=0.4 * a)


@api()
def kiss(B, dur=0.4):
    expression(B, "kiss", 1.0, dur)


@api()
def sneer(B, side="left", dur=0.35):
    s = _SIDE.get(str(side).lower(), "l"); L = "Left" if s == "l" else "Right"
    face(B, dur, **{f"noseSneer{L}": 0.9, f"mouthUpperUp{L}": 0.7, f"browDown{L}": 0.4})


# ---- context managers (handled by the interpreter) -----------------------------------------------------------------
CTX = {"parallel", "together", "sequence", "after", "at_same_time"}
CONST = {"LEFT": "left", "RIGHT": "right", "BOTH": "both"}


# ============================================================================================ sandboxed interpreter
_ALLOWED_BIN = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
                ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
                ast.Pow: lambda a, b: a ** b if abs(b) <= 4 else (_ for _ in ()).throw(ProgramError("power too big"))}
MAX_STEPS = 600


def extract_code(text):
    """first ```python ...``` block (or ``` block), else the whole text."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1).strip("\n")
    m = re.search(r"```(?:python|py)?\s*\n(.*)$", text, re.S)     # unterminated fence
    if m:
        return m.group(1).strip("\n")
    return text.strip()


class Interp:
    def __init__(self, B, strict=False):
        self.B = B
        self.env = dict(CONST)
        self.steps = 0
        self.strict = strict
        self.errors = []        # (lineno, message)
        self.first_stmt_end_t = None

    def run(self, src):
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            raise ProgramError(f"syntax error line {e.lineno}: {e.msg}")
        self.static_check(tree)
        for st in tree.body:
            self.stmt(st, top=True)
            if self.first_stmt_end_t is None:
                self.first_stmt_end_t = self.B.ctx["end"]

    # static whitelist: rejects anything outside the subset before running
    def static_check(self, tree):
        ok = (ast.Module, ast.Expr, ast.Call, ast.Name, ast.Load, ast.Store, ast.Constant, ast.keyword, ast.With,
              ast.withitem, ast.For, ast.Assign, ast.BinOp, ast.UnaryOp, ast.USub, ast.UAdd, ast.Tuple, ast.List,
              ast.Dict, ast.Pass, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.IfExp,
              ast.If, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.BoolOp, ast.And, ast.Or,
              ast.Not)
        for node in ast.walk(tree):
            if not isinstance(node, ok):
                raise ProgramError(f"line {getattr(node, 'lineno', '?')}: {type(node).__name__} is not allowed "
                                   f"(only API calls, numbers/strings, arithmetic, simple assignments, "
                                   f"for-range loops and with parallel()/sequence()/after())")
            if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
                raise ProgramError(f"line {node.lineno}: only plain API function calls are allowed")
            if isinstance(node, ast.Name) and node.id.startswith("_"):
                raise ProgramError(f"line {node.lineno}: names starting with _ are not allowed")

    def tick(self):
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise ProgramError(f"program too long (> {MAX_STEPS} steps)")

    def stmt(self, st, top=False):
        self.tick()
        if isinstance(st, ast.Pass):
            return
        if isinstance(st, ast.Assign):
            if len(st.targets) != 1 or not isinstance(st.targets[0], ast.Name):
                raise ProgramError(f"line {st.lineno}: only `name = value` assignments")
            if isinstance(st.value, ast.Call) and st.value.func.id in API:
                self.call(st.value)                       # `x = hold(1)`: run the call, the name gets None
                self.env[st.targets[0].id] = None
                return
            self.env[st.targets[0].id] = self.expr(st.value)
            return
        if isinstance(st, ast.If):
            raise ProgramError(f"line {st.lineno}: `if` is not part of the motion API (write the move you want)")
        if isinstance(st, ast.Expr):
            if not isinstance(st.value, ast.Call):
                return                                   # bare strings / docstrings: ignore
            self.call(st.value)
            return
        if isinstance(st, ast.For):
            it = st.iter
            if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id in ("range", "repeat")):
                raise ProgramError(f"line {st.lineno}: loops must be `for i in range(n)`")
            args = [self.expr(a) for a in it.args]
            n = int(args[-1]) if len(args) == 1 else int(args[1]) - int(args[0])
            if n > 20:
                raise ProgramError(f"line {st.lineno}: at most 20 loop iterations")
            if not isinstance(st.target, ast.Name):
                raise ProgramError(f"line {st.lineno}: loop variable must be a plain name")
            start = int(args[0]) if len(args) > 1 else 0
            self.B.push("seq")                            # iterations run one after another, even inside parallel()
            for i in range(max(0, n)):
                self.env[st.target.id] = start + i
                for s2 in st.body:
                    self.stmt(s2)
            self.B.pop()
            return
        if isinstance(st, ast.With):
            if len(st.items) != 1 or not isinstance(st.items[0].context_expr, ast.Call):
                raise ProgramError(f"line {st.lineno}: use `with parallel():`, `with sequence():` or `with after(s):`")
            c = st.items[0].context_expr
            nm = c.func.id
            if nm in ("parallel", "together", "at_same_time"):
                self.B.push("par")
            elif nm == "sequence":
                self.B.push("seq")
            elif nm == "after":
                args = [self.expr(a) for a in c.args]
                kw = {k.arg: self.expr(k.value) for k in c.keywords}
                d = _num(args[0] if args else kw.get("seconds", 0.0), "after")
                self.B.push("seq", delay=max(0.0, d))
            else:
                raise ProgramError(f"line {st.lineno}: unknown block {nm!r}; use parallel / sequence / after")
            for s2 in st.body:
                self.stmt(s2)
            self.B.pop()
            return
        raise ProgramError(f"line {getattr(st, 'lineno', '?')}: statement not allowed")

    def expr(self, e):
        self.tick()
        if isinstance(e, ast.Constant):
            if isinstance(e.value, (int, float, str, bool)) or e.value is None:
                return e.value
            raise ProgramError("unsupported constant")
        if isinstance(e, ast.Name):
            if e.id in self.env:
                return self.env[e.id]
            if e.id in ("True", "False", "None"):
                return {"True": True, "False": False, "None": None}[e.id]
            raise ProgramError(f"line {e.lineno}: unknown name {e.id!r} (strings need quotes)")
        if isinstance(e, ast.BinOp):
            a, b = self.expr(e.left), self.expr(e.right)
            if not all(isinstance(x, (int, float)) for x in (a, b)):
                raise ProgramError(f"line {e.lineno}: arithmetic on non-numbers")
            return _ALLOWED_BIN[type(e.op)](a, b)
        if isinstance(e, ast.UnaryOp) and not isinstance(e.op, ast.Not):
            v = self.expr(e.operand)
            return -v if isinstance(e.op, ast.USub) else +v
        if isinstance(e, (ast.Tuple, ast.List)):
            return tuple(self.expr(x) for x in e.elts)
        if isinstance(e, ast.Dict):
            return {self.expr(k): self.expr(v) for k, v in zip(e.keys, e.values)}
        if isinstance(e, ast.Call) and e.func.id == "dict":
            return {k.arg: self.expr(k.value) for k in e.keywords}
        if isinstance(e, ast.IfExp):          # tolerated (evaluated), but flagged: not part of the API
            self.B.warn(f"line {e.lineno}: conditional expression evaluated; not part of the motion API")
            return self.expr(e.body) if self.expr(e.test) else self.expr(e.orelse)
        if isinstance(e, ast.BoolOp):
            vals = [self.expr(v) for v in e.values]
            return all(vals) if isinstance(e.op, ast.And) else any(vals)
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.Not):
            return not self.expr(e.operand)
        if isinstance(e, ast.Compare):
            a = self.expr(e.left)
            for op, c in zip(e.ops, e.comparators):
                b = self.expr(c)
                ok = {ast.Eq: a == b, ast.NotEq: a != b}.get(type(op))
                if ok is None:
                    if not all(isinstance(x, (int, float)) for x in (a, b)):
                        raise ProgramError(f"line {e.lineno}: comparison of non-numbers")
                    ok = {ast.Lt: a < b, ast.LtE: a <= b, ast.Gt: a > b, ast.GtE: a >= b}[type(op)]
                if not ok:
                    return False
                a = b
            return True
        raise ProgramError(f"line {getattr(e, 'lineno', '?')}: expression not allowed")

    def call(self, c):
        nm = c.func.id
        if nm in CTX:
            raise ProgramError(f"line {c.lineno}: {nm}() is a block: use `with {nm}():`")
        fn = API.get(nm)
        if fn is None:
            close = difflib.get_close_matches(nm, list(API), n=3)
            msg = f"line {c.lineno}: unknown function {nm!r}" + (f" (did you mean {', '.join(close)}?)" if close else "")
            if self.strict:
                raise ProgramError(msg)
            self.errors.append((c.lineno, msg))
            return
        args = [self.expr(a) for a in c.args]
        kw = {}
        for k in c.keywords:
            if k.arg is None:
                raise ProgramError(f"line {c.lineno}: **kwargs not allowed")
            kw[k.arg] = self.expr(k.value)
        snap = self._snap()
        try:
            fn(self.B, *args, **kw)
            return
        except ProgramError as e:
            msg = f"line {c.lineno} {nm}(): {e}"
        except TypeError as e:
            msg = f"line {c.lineno} {nm}(): bad arguments ({str(e).replace(fn.__name__ + '()', nm + '()')})"
        if self.strict:
            raise ProgramError(msg)
        # ONE bad argument must not cost the whole move: try the nearest valid value (or the default) per argument
        self._restore(snap)
        for desc, a2, k2 in _repairs(fn, args, kw):
            snap = self._snap()
            try:
                fn(self.B, *a2, **k2)
            except (ProgramError, TypeError):
                self._restore(snap)
                continue
            self.B.warn(f"line {c.lineno} {nm}(): {desc} ({msg.split(': ', 1)[-1][:120]})")
            return
        self.errors.append((c.lineno, msg))

    def _snap(self):
        B = self.B
        return (len(B.segs), len(B.oscs), copy.deepcopy(B.stack), dict(B.last_arm), B.ended_with_return,
                B.keep_end_pose, set(B.expr_keys), set(B.explicit_keys), len(B.goals), list(B.warnings))

    def _restore(self, sn):
        B = self.B
        (ns, no, stack, la, ewr, kep, ek, xk, ng, w) = sn
        del B.segs[ns:]; del B.oscs[no:]; B.stack = stack; B.last_arm = la; B.ended_with_return = ewr
        B.keep_end_pose = kep; B.expr_keys = ek; B.explicit_keys = xk; del B.goals[ng:]
        B.warnings[:] = w


VOCAB = {
    "part": ["palm", "fingertips", "index_tip", "thumb_tip", "knuckles", "back_of_hand", "wrist", "middle_tip"],
    "palm": ["body", "camera", "down", "up", "forward", "back", "in", "out"],
    "finger_dir": ["up", "down", "forward", "back", "in", "out"],
    "elbow": ["down", "out", "up", "forward", "back"],
    "height": ["high", "head", "shoulder", "chest"],
    "hand": ["left", "right", "both"], "side": ["left", "right", "both"],
    "direction": ["up", "forward", "out", "side", "down", "across", "up_out", "forward_up", "forward_down", "back",
                  "overhead", "left", "right", "backward", "sideways", "vertical"],
    "fingers": ["up", "forward"],
}


def _vocab(key, fn_name):
    if key in ("shape",):
        return list(kin.HAND_SHAPES)
    if key in ("name",) or (key == 0 and fn_name in ("expression",)):
        return list(EXPRESSIONS) + list(EXPR_ALIASES)
    if key in ("target", "where"):
        return list(kin.POINT_NAMES)
    if key == "part" and fn_name in ("rotate", "oscillate", "release"):
        return None
    return VOCAB.get(key)


def _repairs(fn, args, kw):
    """candidate (description, args, kwargs) with ONE argument fixed: unknown keyword dropped, a string value replaced by
    the nearest valid one, or dropped for its default."""
    import inspect
    try:
        params = list(inspect.signature(fn).parameters)[1:]       # skip B
    except (TypeError, ValueError):
        params = []
    out = []
    for k in list(kw):
        if params and k not in params and "shapes" not in params:
            k2 = dict(kw); k2.pop(k)
            out.append((f"ignored unknown argument {k}=", args, k2))
    for k, v in kw.items():
        if not isinstance(v, str):
            continue
        voc = _vocab(k, fn.__name__)
        if voc:
            near = difflib.get_close_matches(v.lower().replace(" ", "_"), voc, n=1, cutoff=0.4)
            if near and near[0] != v:
                k2 = dict(kw); k2[k] = near[0]
                out.append((f"{k}={v!r} -> {near[0]!r}", args, k2))
        k2 = dict(kw); k2.pop(k)
        out.append((f"{k}={v!r} dropped (default used)", args, k2))
    for i, v in enumerate(args):
        if not isinstance(v, str):
            continue
        key = params[i] if i < len(params) else None
        voc = _vocab(key if key else i, fn.__name__) or (_vocab(0, fn.__name__) if i == 0 else None)
        if voc:
            near = difflib.get_close_matches(v.lower().replace(" ", "_"), voc, n=1, cutoff=0.4)
            if near and near[0] != v:
                a2 = list(args); a2[i] = near[0]
                out.append((f"argument {v!r} -> {near[0]!r}", a2, kw))
    for k, v in kw.items():                                       # a bad number (dur out of range...): its default
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            k2 = dict(kw); k2.pop(k)
            out.append((f"{k}={v!r} dropped (default used)", args, k2))
    return out


# ============================================================================================ renderer
class Renderer:
    def __init__(self, B, seed=0):
        self.B, self.nat = B, B.nat
        self.rng = np.random.default_rng(seed)
        self.n = kin.neutral().astype(np.float64)
        self.ik_cache = {}
        self.arc_corr = {}
        self.goal_log = []
        self.apply_root = True      # False: root kept neutral in the canon frames (the rig gets root_* from self.R)
        self.R = None

    # ---- target resolution
    def _ik_target(self, spec, m, s):
        tg = spec["target"]
        Rb = kin.body_up_R(m)
        if isinstance(tg, tuple) and tg[0] == "xyz":
            p = np.array(tg[1])
        elif isinstance(tg, tuple) and tg[0] == "shoulder_dir":
            _, s_, dv, ext = tg
            Rg, P = kin._arm_chain(m)
            sh = P["shoulder_" + s_]
            p = sh + Rg["spine3"] @ (np.asarray(dv) * 0.52 * ext)
        elif isinstance(tg, tuple) and tg[0] == "shoulder_off":
            _, s_, off = tg
            Rg, P = kin._arm_chain(m)
            p = P["shoulder_" + s_] + Rg["spine3"] @ np.asarray(off)
        elif isinstance(tg, str) and tg.endswith("_wrist_rest"):
            s_ = "l" if tg.startswith("left") else "r"
            nn = m.copy(); nn[kin.spec.SL_BODY] = m[kin.spec.SL_BODY]
            for j in ("collar_", "shoulder_", "elbow_", "wrist_"):
                kin.jset(nn, j + s_, kin.jget(self.n, j + s_))
            p = kin.point_fast(f"{'left' if s_ == 'l' else 'right'}_wrist", nn)
        else:
            p = kin.point_fast(tg, m)
        if spec.get("aim"):
            Rg, P = kin._arm_chain(m)
            sh = P["shoulder_" + s]
            d = p - sh
            L = np.linalg.norm(d)
            reach = 0.66
            if L > reach:
                p = sh + d / L * reach
        return p + Rb @ np.asarray(spec["offset"])

    def _solve_arm(self, spec, m, s, osc_off=None, wr=None):
        p = self._ik_target(spec, m, s)
        if osc_off is not None:
            p = p + kin.body_up_R(m) @ osc_off
        key = (s, tuple(np.round(p, 4)), tuple(np.round(m[0:45], 4)), tuple(np.round(m[kin._fsl(s)], 3)),
               spec["part"], str(spec["palm"]), str(spec["elbow"]), str(spec["finger_dir"]))
        hit = self.ik_cache.get(key)
        if hit is None:
            palm = spec["palm"]
            fd = spec["finger_dir"]
            if fd == "forward_up":
                fd = kin.body_up_R(m) @ np.array([0, 0.6, 0.8])
            elif isinstance(fd, str) and fd in ("up", "down", "forward", "back", "out", "in"):
                sg = 1.0 if s == "l" else -1.0
                fd = kin.body_up_R(m) @ np.array({"up": (0, 1, 0.15), "down": (0, -1, 0.1), "forward": (0, 0, 1),
                                                  "back": (0, 0, -1), "out": (sg, 0, 0.3), "in": (-sg, 0, 0.3)}[fd])
            if spec.get("aim"):
                Rg, P = kin._arm_chain(m)
                fd = p - P["shoulder_" + s]
                palm = palm or "down"
            tname = spec["target"] if isinstance(spec["target"], str) and np.allclose(spec["offset"], 0) else None
            target = tname if (tname and osc_off is None and not spec.get("aim")) else p
            if palm == "body" and tname is None and isinstance(spec["target"], str):
                palm = kin.inward_normal(spec["target"], m)
            eh = spec["elbow"]
            if isinstance(eh, tuple):                     # chest-frame pole vector
                eh = kin._arm_chain(m)[0]["spine3"] @ np.asarray(eh, float)
            mm, info = kin.ik_arm(s, target, m, part=spec["part"], palm_facing=palm,
                                  elbow_hint=eh, finger_dir=fd)
            c = kin._fsl(s)
            hit = (np.concatenate([mm[kin.spec.jcols(j + "_" + s)] for j in ("collar", "shoulder", "elbow", "wrist")]),
                   info)
            if len(self.ik_cache) > 20000:
                self.ik_cache.clear()
            self.ik_cache[key] = hit
        return hit

    # ---- the main loop
    def render(self, T):
        B = self.B
        n = self.n
        F = np.arange(T) / FPS
        segs = sorted(B.segs, key=lambda s: (s.t0, s.t1))
        by = {}
        for s in segs:
            by.setdefault(s.chan, []).append(s)
        oscs = B.oscs
        out = np.repeat(n[None], T, 0)

        def osc_val(chan, param, t):
            v = 0.0
            for o in oscs:
                if o.chan == chan and o.param == param and o.t0 <= t <= o.t1:
                    u = (t - o.t0) / max(o.t1 - o.t0, 1e-6)
                    env = min(1.0, u / 0.12, (1 - u) / 0.12) if o.src not in ("clap",) else 1.0
                    env = max(0.0, env)
                    env = env * env * (3 - 2 * env)
                    v += (o.amp * math.sin(2 * math.pi * o.cycles * u + o.phase) + o.bias) * env
            return v

        # ---------- scalar-vector channels: evaluate value(t) with segment chaining
        def touches(sg, key, keys):
            tg = sg.tgt
            if "neutral" in tg or "look" in tg:
                return True
            if "ypr" in tg:
                return keys.index(key) < len(tg["ypr"])
            if key in tg:
                return True
            return (key == "yaw" and "yaw_add" in tg) or (key == "pitch" and "pitch_add" in tg)

        def eval_vec(chan, keys, neutral_vals, resolve=None):
            """each key is its own track: a segment only moves the keys it names (spin + jump can overlap); ties at the
            same start time go to explicit calls over macro side effects."""
            vals = np.tile(np.array(neutral_vals, float), (T, 1))
            sl = by.get(chan, [])
            for j, key in enumerate(keys):
                segs = sorted([sg for sg in sl if touches(sg, key, keys)], key=lambda sg: (round(sg.t0, 4), sg.prio))
                if not segs:
                    continue
                k = -1
                st = float(neutral_vals[j])
                for i, t in enumerate(F):
                    while k + 1 < len(segs) and segs[k + 1].t0 <= t + 1e-9:
                        k += 1
                        st = vals[i - 1, j] if i > 0 else float(neutral_vals[j])
                    if k < 0:
                        continue
                    sg = segs[k]
                    stv = np.full(len(keys), st)
                    tg = self._vec_target(sg, keys, stv, neutral_vals, i, resolve)[j]
                    u = (t - sg.t0) / max(sg.t1 - sg.t0, 1e-6)
                    vals[i, j] = st + (tg - st) * float(_ease_k(sg, u, self.nat))
            return vals

        # root: yaw pitch roll x y z
        root_keys = ["yaw", "pitch", "roll", "x", "y", "z"]
        R = eval_vec("root", root_keys, [0, 0, 0, 0, 0, 0])
        if "rootp" in by:
            R[:, 1] += eval_vec("rootp", ["pitch"], [0.0])[:, 0]
        for i, t in enumerate(F):
            for j, k in enumerate(root_keys):
                R[i, j] += osc_val("root", k, t)
        # canon stores the root as ONE axis-angle; near 360 deg a tilted axis makes that representation jump.  So the
        # root holds a pure yaw (written as a continuous angle) and any root pitch/roll goes into spine1 (same global
        # chest pose; only the pelvis itself stays level).  A flip (|pitch| > 150) with no yaw/roll is a pure pitch.
        self.R = R
        flip = np.abs(R[:, 1]).max() > 150 and np.abs(R[:, 0]).max() < 1 and np.abs(R[:, 2]).max() < 1
        for i in (range(T) if self.apply_root else ()):
            if flip:
                out[i, 0:3] = [math.radians(R[i, 1]), 0.0, 0.0]
                out[i, 3:6] = R[i, 3:6] / 10.0
                continue
            out[i] = kin.set_root(out[i], R[i, 0], 0.0, 0.0, (R[i, 3], R[i, 4], R[i, 5]))
            if abs(R[i, 1]) > 1e-6 or abs(R[i, 2]) > 1e-6:
                Rpr = kin.rot("x", R[i, 1]) @ kin.rot("z", -R[i, 2])
                kin.jset(out[i], "spine1", kin.mat_to_aa(Rpr @ kin.aa_to_mat(kin.jget(out[i], "spine1"))))
        # torso
        tk = ["lean_fwd", "lean_side", "twist"]
        Tq = eval_vec("torso", tk, [0, 0, 0])
        for i, t in enumerate(F):
            v = Tq[i] + np.array([osc_val("torso", k, t) for k in tk])
            if np.abs(v).max() > 1e-6:
                out[i] = kin.set_torso(out[i], *v)
        # head
        hk = ["yaw", "pitch", "roll"]
        self._look_cache = {}
        H = eval_vec("head", hk, [0, 0, 0], resolve=lambda sg, i: self._look_head(sg, out[i]))
        for i, t in enumerate(F):
            v = H[i] + np.array([osc_val("head", k, t) for k in hk])
            if np.abs(v).max() > 1e-6:
                out[i] = kin.set_head(out[i], *v)
        # gaze: eyes-lead with residual tracking
        G = eval_vec("gaze", ["yaw", "pitch"], [0, 0], resolve=lambda sg, i: self._look_gaze(sg, out[i]))
        for i, t in enumerate(F):
            gy = G[i, 0] + osc_val("gaze", "yaw", t); gp = G[i, 1] + osc_val("gaze", "pitch", t)
            out[i] = kin.set_gaze(out[i], gy, gp)
        # shoulders (collar elevation) as an additive amount
        shr = {}
        for s in "lr":
            shr[s] = eval_vec("shrug_" + s, ["amount"], [0.0])[:, 0] + np.array([osc_val("shrug_" + s, "amount", t) for t in F])
        # fingers
        for s in "lr":
            self._render_fingers(out, s, by.get("fing_" + s, []), F)
        # arms
        for s in "lr":
            self._render_arm(out, s, by.get("arm_" + s, []), F, osc_val, shr[s], by.get("wrist_" + s, []))
        # face
        self._render_face(out, by.get("face", []), F)
        return out

    def _vec_target(self, sg, keys, start, neutral_vals, i, resolve):
        tg = sg.tgt
        if "neutral" in tg:
            v = np.array(neutral_vals, float)
            if sg.chan == "root":       # after a spin, "neutral" is the nearest full turn (no spinning back)
                v[0] = round(start[0] / 360.0) * 360.0
            return v
        if resolve is not None and ("look" in tg):
            return np.asarray(resolve(sg, i), float)
        if "ypr" in tg:
            return np.asarray(tg["ypr"][:len(keys)], float)
        v = start.copy()
        for j, k in enumerate(keys):
            if k in tg:
                x = tg[k]
                if isinstance(x, tuple) and x[0] == "add":
                    v[j] = start[j] + x[1]
                else:
                    v[j] = float(x)
            if k == "yaw" and "yaw_add" in tg:
                v[j] = start[j] + tg["yaw_add"]
            if k == "pitch" and "pitch_add" in tg:
                v[j] = start[j] + tg["pitch_add"]
        if "amount" in tg and keys == ["amount"]:
            v[0] = tg["amount"]
        return v

    def _look_dirs(self, target, m):
        key = (str(target), tuple(np.round(m[0:15], 3)))
        r = self._look_cache.get(key)
        if r is None:
            tg = np.array(target[1]) if isinstance(target, tuple) else target
            _, info = kin.look_at(tg, m)
            r = (info["yaw_deg"], info["pitch_deg"], info["head_yaw"], info["head_pitch"])
            self._look_cache[key] = r
        return r

    def _look_head(self, sg, m):
        y, p, hy, hp = self._look_dirs(sg.tgt["look"], m)
        return (hy, hp, 0.0)

    def _look_gaze(self, sg, m):
        y, p, hy, hp = self._look_dirs(sg.tgt["look"], m)
        cy, cp, _ = kin.head_ypr(m)
        gy = float(np.clip(y - cy, *kin.JOINT_LIMITS["gaze_yaw"]))
        gp = float(np.clip(p - cp, *kin.JOINT_LIMITS["gaze_pitch"]))
        return (gy, gp)

    def _render_fingers(self, out, s, sl, F):
        c = kin._fsl(s)
        n45 = self.n[c].copy()
        cur = n45.copy()
        starts = {}
        k = -1
        for i, t in enumerate(F):
            while k + 1 < len(sl) and sl[k + 1].t0 <= t + 1e-9:
                k += 1
                starts[k] = out[i - 1, c].copy() if i > 0 else cur.copy()
            if k < 0:
                continue
            sg = sl[k]
            tg = n45 if "neutral" in sg.tgt else kin.hand_shape(sg.tgt["shape"], s, sg.tgt.get("amount", 1.0))
            u = (t - sg.t0) / max(sg.t1 - sg.t0, 1e-6)
            w = float(_ease_k(sg, u, self.nat))
            out[i, c] = kin.slerp_aa(starts[k].reshape(15, 3), tg.reshape(15, 3), w).reshape(45)

    def _render_arm(self, out, s, sl, F, osc_val, shrug, wsl):
        jn = [j + "_" + s for j in ("collar", "shoulder", "elbow", "wrist")]
        cols = [kin.spec.jcols(j) for j in jn]
        n12 = np.concatenate([self.n[c] for c in cols])
        T = len(F)
        arm = np.repeat(n12[None], T, 0)
        starts = {}
        k = -1
        has_osc = any(o.chan == "hand_" + s for o in self.B.oscs)
        lags = np.array([0.0, 0.0, 0.08, 0.16]) if self.nat["successive"] else np.zeros(4)
        for i, t in enumerate(F):
            while k + 1 < len(sl) and sl[k + 1].t0 <= t + 1e-9:
                k += 1
                starts[k] = arm[i - 1].copy() if i > 0 else n12.copy()
            if k < 0:
                continue
            sg = sl[k]
            m = out[i]
            if "neutral" in sg.tgt:
                tg = n12
                info = None
            else:
                spec = sg.tgt["ik"]
                off = None
                if has_osc:
                    ov = np.array([osc_val("hand_" + s, a, t) for a in "xyz"])
                    nv = osc_val("hand_" + s, "normal", t)
                    if abs(nv) > 1e-6 and isinstance(spec["target"], str):
                        nrm = kin.inward_normal(spec["target"], m)
                        if nrm is not None:
                            ov = ov - kin.body_up_R(m).T @ nrm * nv     # + = away from the body
                    if np.abs(ov).max() > 1e-6:
                        off = ov
                mm = m.copy()
                tg, info = self._solve_arm(spec, mm, s, off)
                if i == min(T - 1, int(round(sg.t1 * FPS))):
                    self.goal_log.append((sg.t1, s, sg.src, info))
            u = (t - sg.t0) / max(sg.t1 - sg.t0, 1e-6)
            st = starts[k]
            if u < 1.0:
                r = self._arc(st, tg, sg, s, out[i], u)
                if r is not None:
                    arm[i] = r
                    continue
            vals = []
            for j in range(4):
                if lags.any():
                    uj = np.clip((u - lags[j]) / (1.0 - lags.max()), 0, 1) if u < 1 else 1.0
                else:
                    uj = u
                w = float(_ease_k(sg, uj, self.nat))
                vals.append(kin.slerp_aa(st[3 * j:3 * j + 3], tg[3 * j:3 * j + 3], w))
            arm[i] = np.concatenate(vals)
        # apply + shrug + wrist offsets + body push-out
        wvals = self._wrist_offsets(wsl, F, s, osc_val)
        sg_ = 1.0 if s == "l" else -1.0
        for i in range(T):
            for j, c in enumerate(cols):
                out[i, c] = arm[i, 3 * j:3 * j + 3]
            if abs(shrug[i]) > 1e-6:
                Rc = kin.aa_to_mat(out[i, cols[0]])
                out[i, cols[0]] = kin.mat_to_aa(kin.rot("z", sg_ * 18.0 * shrug[i]) @ Rc)
            if wvals is not None and np.abs(wvals[i]).max() > 1e-6:
                fx, sd, tw = wvals[i]
                Rw = kin.aa_to_mat(out[i, cols[3]])
                # wrist-local axes (left hand): flex about local z (palm-ward = -z for left), side about y, twist about x
                Rd = kin.rot("z", -sg_ * fx) @ kin.rot("y", -sg_ * sd) @ kin.rot("x", tw)
                out[i, cols[3]] = kin.mat_to_aa(Rw @ Rd)
        self._push_out_arm(out, s, sl, F)

    def _wrist_offsets(self, wsl, F, s, osc_val):
        has_osc = any(o.chan == "wrist_" + s for o in self.B.oscs)
        if not wsl and not has_osc:
            return None
        T = len(F)
        v = np.zeros((T, 3)); cur = np.zeros(3)
        k = -1; starts = {}
        for i, t in enumerate(F):
            while k + 1 < len(wsl) and wsl[k + 1].t0 <= t + 1e-9:
                k += 1
                starts[k] = v[i - 1].copy() if i > 0 else cur
            if k >= 0:
                sg = wsl[k]
                tg = np.array(sg.tgt["wr"], float)
                u = (t - sg.t0) / max(sg.t1 - sg.t0, 1e-6)
                v[i] = starts[k] + (tg - starts[k]) * float(_ease_k(sg, u, self.nat))
            v[i] += np.array([osc_val("wrist_" + s, "flex", t), osc_val("wrist_" + s, "side", t),
                              osc_val("wrist_" + s, "twist", t)])
        return v

    def _arc(self, st, tg, sg, s, m, u):
        """Cartesian hand path for a transition: the wrist follows a quadratic Bezier from start to goal bulging
        away from the body (straight line when nat['arcs'] is off), pushed out of the body, min-jerk timed; the hand
        orientation is slerped with a lag (successive breaking), the elbow pole is blended; per-frame IK.  End-point
        corrections keep it exactly continuous with the start pose and the goal pose.  None -> small move, use
        joint-space slerp."""
        jn = [j + "_" + s for j in ("collar", "shoulder", "elbow", "wrist")]
        a = m.copy(); b = m.copy()
        for j, nm in enumerate(jn):
            kin.jset(a, nm, st[3 * j:3 * j + 3]); kin.jset(b, nm, tg[3 * j:3 * j + 3])
        Ra, Pa = kin._arm_chain(a); Rb_, Pb = kin._arm_chain(b)
        ws, wg = Pa["wrist_" + s], Pb["wrist_" + s]
        disp = float(np.linalg.norm(wg - ws))
        if disp < 0.08:
            return None
        lag = 0.12 if self.nat["successive"] else 0.0
        w = float(_ease_k(sg, u, self.nat))
        wh = float(_ease_k(sg, float(np.clip((u - lag) / (1 - lag), 0, 1)), self.nat))
        we = float(_ease_k(sg, float(np.clip(u / (1 - lag), 0, 1)), self.nat))
        mid = 0.5 * (ws + wg)
        Rc = Ra["spine3"]
        q = Rc.T @ (mid - Pa["spine3"])
        od = np.array([q[0], 0.0, max(q[2], 0.0) + 0.12])
        od = Rc @ (od / max(np.linalg.norm(od), 1e-9))
        bulge = 0.22 * disp if self.nat["arcs"] else 0.0
        tg_spec = sg.tgt.get("ik") if isinstance(sg.tgt, dict) else None
        if tg_spec is not None and isinstance(tg_spec.get("target"), str):
            nin = kin.inward_normal(tg_spec["target"], m)
            if nin is not None:            # approach a body target from outside its surface (ears from the side)
                od = 0.4 * od - nin
                od = od / max(np.linalg.norm(od), 1e-9)
                bulge = 0.3 * disp if self.nat["arcs"] else 0.0
        c = mid + od * bulge
        p = (1 - w) ** 2 * ws + 2 * (1 - w) * w * c + w ** 2 * wg
        if kin.inside_body(p, m, margin=0.04) is not None:
            p = kin.push_out(p, m, margin=0.05)
        qa = kin.mat_to_quat(Ra["wrist_" + s]); qb = kin.mat_to_quat(Rb_["wrist_" + s])
        Rh = kin.aa_to_mat(kin.quat_to_aa(kin._qslerp(qa, qb, wh)))
        pn = Rh @ np.array([0, -1.0, 0]); fd = Rh @ kin._mir([1.0, 0, 0], s)
        sh = Pa["shoulder_" + s]
        pole = (1 - we) * (Pa["elbow_" + s] - 0.5 * (sh + ws)) + we * (Pb["elbow_" + s] - 0.5 * (sh + wg))
        key = (id(sg), s)
        corr = self.arc_corr.get(key)
        if corr is None:
            ends = []
            for (ww, R_, P_, stt) in ((0.0, Ra, Pa, st), (1.0, Rb_, Pb, tg)):
                pp = ws if ww == 0 else wg
                pl = (Pa["elbow_" + s] - 0.5 * (sh + ws)) if ww == 0 else (Pb["elbow_" + s] - 0.5 * (sh + wg))
                mm, _ = kin.ik_arm(s, pp, m, part="wrist", palm_facing=R_["wrist_" + s] @ np.array([0, -1.0, 0]),
                                   finger_dir=R_["wrist_" + s] @ kin._mir([1.0, 0, 0], s), elbow_hint=pl)
                C = []
                for j, nm in enumerate(jn):
                    C.append(kin.aa_to_mat(stt[3 * j:3 * j + 3]) @ kin.aa_to_mat(kin.jget(mm, nm)).T)
                ends.append(C)
            corr = ends
            self.arc_corr[key] = corr
        mm, _ = kin.ik_arm(s, p, m, part="wrist", palm_facing=pn, finger_dir=fd, elbow_hint=pole)
        r = []
        for j, nm in enumerate(jn):
            C0, C1 = corr[0][j], corr[1][j]
            Cw = kin.aa_to_mat(kin.slerp_aa(kin.mat_to_aa(C0), kin.mat_to_aa(C1), w))
            r.append(kin.mat_to_aa(Cw @ kin.aa_to_mat(kin.jget(mm, nm))))
        return np.concatenate(r)

    def _cartesian(self, st, tg, sg, s, m, u):
        jn = [j + "_" + s for j in ("collar", "shoulder", "elbow", "wrist")]
        a = m.copy(); b = m.copy()
        for j, nm in enumerate(jn):
            kin.jset(a, nm, st[3 * j:3 * j + 3]); kin.jset(b, nm, tg[3 * j:3 * j + 3])
        part = sg.tgt["ik"]["part"]
        sd = "left" if s == "l" else "right"
        pa = kin.fk(a)[f"{sd}_{part if part != 'fingertips' else 'fingertips'}"]
        pb = kin.fk(b)[f"{sd}_{part}"]
        w = float(_ease_k(sg, u, self.nat))
        p = pa + (pb - pa) * w
        mm, _ = kin.ik_arm(s, p, m, part=part)
        r = np.concatenate([mm[kin.spec.jcols(nm)] for nm in jn])
        # orientation: slerp the wrist between endpoints
        r[9:12] = kin.slerp_aa(st[9:12], tg[9:12], w)
        return r

    def _push_out_arm(self, out, s, sl, F):
        """hands never pass through the head/torso: frames where a hand point is inside a body volume get re-solved
        with the wrist pushed out (contact goals are placed on the surface and are left alone)."""
        if not sl:
            return
        sd = "left" if s == "l" else "right"
        active = np.zeros(len(F), bool)
        for sg in sl:
            if "ik" in sg.tgt:
                active |= (F >= sg.t0) & (F <= sg.t1 + 0.6)
        for sg in sl:                                   # contact goals sit on the surface on purpose
            if sg.contact:
                active &= ~((F >= sg.t0 + 0.6 * (sg.t1 - sg.t0)) & (F <= sg.t1 + 0.6))
        idx = np.nonzero(active)[0]
        if not len(idx):
            return
        P, _, _ = kin.fk_full(out[idx])
        for q, i in enumerate(idx):
            m = out[i]
            pts = [P[f"{sd}_wrist"][q], P[f"{sd}_knuckles"][q], P[f"{sd}_back_of_hand"][q]]
            near = min(np.linalg.norm(pts[0] - P["head"][q]), np.linalg.norm(pts[0] - P["spine3"][q]),
                       np.linalg.norm(pts[0] - P["spine1"][q]))
            if near > 0.32:
                continue
            bad = [x for x in pts if kin.inside_body(x, m, margin=0.0) is not None]
            if not bad:
                continue
            wr = kin.push_out(P[f"{sd}_wrist"][q], m, margin=0.045)
            mm, _ = kin.ik_arm(s, wr, m, part="wrist")
            for j in ("collar", "shoulder", "elbow"):
                c = kin.spec.jcols(f"{j}_{s}")
                out[i, c] = mm[c]
            self.B.warn(f"{sd} hand path passed through the body near t={F[i]:.1f}s; pushed out") \
                if i % 10 == 0 else None

    def _render_face(self, out, sl, F):
        """each blendshape is its own channel: a segment only moves the shapes it names (so a wink and a smile can
        overlap)."""
        keys = BS_NAMES
        T = len(F)
        for kk in keys:
            if kk.startswith("eyeLook"):
                continue            # owned by gaze
            segs = [sg for sg in sl if "neutral" in sg.tgt or kk in sg.tgt["shapes"]]
            # same start time: the expression first, explicit shapes (face/close_eyes/wink/...) after -> they win
            segs.sort(key=lambda sg: (round(sg.t0, 4), 0 if sg.src == "expression" else 1))
            if not segs:
                continue
            c = COL["bs_" + kk]
            nv = float(self.n[c])
            vals = np.full(T, nv)
            k = -1
            st = nv
            for i, t in enumerate(F):
                while k + 1 < len(segs) and segs[k + 1].t0 <= t + 1e-9:
                    k += 1
                    st = vals[i - 1] if i > 0 else nv
                if k < 0:
                    continue
                sg = segs[k]
                tg = nv if "neutral" in sg.tgt else sg.tgt["shapes"][kk]
                if "restore_from" in sg.tgt:          # go back to what this shape was before the wink/blink
                    j0 = max(0, int(round(sg.tgt["restore_from"] * FPS)) - 1)
                    tg = vals[j0] if j0 < i else tg
                u = (t - sg.t0) / max(sg.t1 - sg.t0, 1e-6)
                vals[i] = st + (tg - st) * float(_ease_k(sg, u, self.nat))
            out[:, c] = vals


def _ease_k(sg, u, nat):
    u = float(np.clip(u, 0, 1))
    e = sg.ease
    if "anticip" in sg.kind and nat["anticipation"] and e in ("settle", "min_jerk"):
        # small counter-move (6 %) in the first 20 %, then the main move (with settle if requested)
        if u < 0.2:
            return -0.05 * math.sin(math.pi * u / 0.2) ** 2
        return float(kin.ease(e, (u - 0.2) / 0.8))
    if e == "settle" and not nat["overshoot"]:
        e = "min_jerk"
    if not nat["min_jerk"] and e in ("min_jerk", "settle"):
        e = "linear"
    return float(kin.ease(e, u))


# ============================================================================================ idle + post
def add_idle(motion, seed, nat, closed_mask=None):
    m = motion.copy()
    T = len(m)
    rng = np.random.default_rng(seed + 1000)
    t = np.arange(T) / FPS
    if nat["idle"]:
        def noise(amp, f0):
            ph = rng.uniform(0, 2 * np.pi, 4)
            fr = f0 * np.array([1.0, 2.3, 5.1, 9.7])
            return amp * (np.sin(2 * np.pi * fr[0] * t + ph[0]) + 0.5 * np.sin(2 * np.pi * fr[1] * t + ph[1]) +
                          0.25 * np.sin(2 * np.pi * fr[2] * t + ph[2]) + 0.1 * np.sin(2 * np.pi * fr[3] * t + ph[3])) / 1.85
        hy, hp, hr = noise(1.6, 0.23), noise(1.2, 0.31), noise(0.8, 0.19)
        arm_n = {s_: (noise(1.0, 0.2), noise(1.5, 0.26)) for s_ in "lr"}
        ty, tf = noise(0.5, 0.17), noise(0.4, 0.21)
        breath = 0.5 * np.sin(2 * np.pi * 0.24 * t + rng.uniform(0, 6.28))
        for i in range(T):
            mm = m[i]
            # small extra rotations applied on top (degrees -> tiny deltas on neck/head/spine2)
            Dh = kin.rot("y", hy[i]) @ kin.rot("x", -hp[i]) @ kin.rot("z", -hr[i])
            kin.jset(mm, "neck", kin.mat_to_aa(Dh @ kin.aa_to_mat(kin.jget(mm, "neck"))))
            Dt = kin.rot("y", ty[i]) @ kin.rot("x", tf[i] + 0.6 * breath[i])
            kin.jset(mm, "spine2", kin.mat_to_aa(Dt @ kin.aa_to_mat(kin.jget(mm, "spine2"))))
            for s, sg in (("l", 1.0), ("r", -1.0)):
                kin.jset(mm, "collar_" + s, kin.mat_to_aa(kin.rot("z", sg * 0.8 * breath[i]) @
                                                         kin.aa_to_mat(kin.jget(mm, "collar_" + s))))
                a1, a2 = arm_n[s]
                kin.jset(mm, "shoulder_" + s, kin.mat_to_aa(kin.aa_to_mat(kin.jget(mm, "shoulder_" + s)) @
                                                           kin.rot("y", a1[i])))
                kin.jset(mm, "elbow_" + s, kin.mat_to_aa(kin.aa_to_mat(kin.jget(mm, "elbow_" + s)) @
                                                        kin.rot("y", a2[i] * sg)))
        # face micro-motion: faces are never frozen (brows, lids, cheeks, mouth corners drift a little)
        for nm, amp, f0 in (("browInnerUp", 0.04, 0.21), ("browOuterUpLeft", 0.03, 0.17), ("browOuterUpRight", 0.03, 0.19),
                            ("eyeSquintLeft", 0.04, 0.23), ("eyeSquintRight", 0.04, 0.25), ("mouthSmileLeft", 0.03, 0.15),
                            ("mouthSmileRight", 0.03, 0.16), ("jawOpen", 0.015, 0.29), ("mouthPressLeft", 0.02, 0.2),
                            ("mouthPressRight", 0.02, 0.22), ("cheekSquintLeft", 0.02, 0.18), ("cheekSquintRight", 0.02, 0.2)):
            c = COL["bs_" + nm]
            m[:, c] = np.clip(m[:, c] + noise(amp, f0), 0, 1)
        # gaze micro-saccades: small steps every ~0.6-1.5 s
        gy = np.zeros(T); gp = np.zeros(T)
        k = 0
        while k < T:
            dur = int(rng.uniform(18, 45))
            gy[k:k + dur] = rng.normal(0, 1.2); gp[k:k + dur] = rng.normal(0, 0.8)
            k += dur
        from scipy.ndimage import gaussian_filter1d
        m[:, COL["gaze_yaw"]] += np.radians(gaussian_filter1d(gy, 1.0))
        m[:, COL["gaze_pitch"]] += np.radians(gaussian_filter1d(gp, 1.0))
    if nat["blinks"]:
        bl = COL["bs_eyeBlinkLeft"]; br = COL["bs_eyeBlinkRight"]
        tt = rng.uniform(0.6, 2.0)
        while tt < T / FPS - 0.3:
            i0 = int(tt * FPS)
            prof = np.array([0.35, 0.85, 1.0, 0.7, 0.35, 0.12])
            ok = True
            for q in range(len(prof)):
                if i0 + q < T and (m[i0 + q, bl] > 0.45 or m[i0 + q, br] > 0.45 or
                                   (closed_mask is not None and closed_mask[i0 + q])):
                    ok = False
            if ok:
                for q, v in enumerate(prof):
                    if i0 + q < T:
                        m[i0 + q, bl] = max(m[i0 + q, bl], v)
                        m[i0 + q, br] = max(m[i0 + q, br], v)
            tt += rng.uniform(2.2, 4.8)
    return m


def smooth(motion, sigma=1.0):
    from scipy.ndimage import gaussian_filter1d
    m = motion.copy()
    cols = [c for c in range(kin.D) if c not in (COL["bs_eyeBlinkLeft"], COL["bs_eyeBlinkRight"])]
    m[:, cols] = gaussian_filter1d(motion[:, cols], sigma, axis=0, mode="nearest")
    m[0] = motion[0]
    return m


# ============================================================================================ entry point
def build(src, nat=None, strict=False):
    nat = dict(NAT_DEFAULT if nat is None else nat)
    B = Builder(nat)
    I = Interp(B, strict=strict)
    I.run(src)
    return B, I


def run_program(src, seed=0, nat=None, strict=False):
    """-> (motion (T,189) float32, report dict)."""
    t_start = time.time()
    nat = dict(NAT_DEFAULT if nat is None else nat)
    B, I = build(src, nat, strict)
    end = max([B.ctx["end"], B.ctx["t"]] + [s.t1 for s in B.segs] + [o.t1 for o in B.oscs] + [0.0])
    tail = 0.0
    scale = 1.0
    if not B.ended_with_return and not B.keep_end_pose:
        # hold the final pose briefly, then return to neutral
        t0 = end + 0.35
        for c in ["root", "torso", "head", "gaze", "arm_l", "arm_r", "fing_l", "fing_r", "face", "shrug_l", "shrug_r",
                  "wrist_l", "wrist_r"]:
            B.segs.append(Seg(c, t0, t0 + 0.7, {"neutral": True} if not c.startswith(("shrug", "wrist"))
                              else ({"amount": 0.0} if c.startswith("shrug") else {"wr": (0.0, 0.0, 0.0)}),
                              B.ease(None), "neutral", "auto_return"))
        end = t0 + 0.7
    else:
        for c in ("shrug_l", "shrug_r"):
            if B.ended_with_return:
                B.segs.append(Seg(c, max(0.0, end - 0.6), end, {"amount": 0.0}, B.ease(None), "neutral", "auto"))
    total = end + 0.25
    if total > MAX_T - 1.0 / FPS:
        scale = (MAX_T - 0.1) / total
        B.warn(f"program lasts {total:.1f}s > {MAX_T:.0f}s: time-compressed x{scale:.2f}")
        for s in B.segs:
            s.t0 *= scale; s.t1 *= scale
        for o in B.oscs:
            o.t0 *= scale; o.t1 *= scale
        total = MAX_T - 0.1
    if total < MIN_T:
        tail = MIN_T - total
        total = MIN_T
    T = int(round(total * FPS))
    R = Renderer(B, seed)
    motion = R.render(T)
    motion = add_idle(motion, seed, nat)
    if nat["smooth"]:
        motion = smooth(motion)
    motion[0] = kin.neutral()
    motion = np.nan_to_num(motion).astype(np.float32)
    report = {"warnings": list(B.warnings), "errors": [e[1] for e in I.errors], "duration_s": T / FPS,
              "time_scale": scale, "n_segments": len(B.segs), "n_osc": len(B.oscs),
              "first_stmt_end_s": I.first_stmt_end_t, "goal_log": R.goal_log, "goals": B.goals,
              "exec_s": time.time() - t_start, "auto_return": not (B.ended_with_return or B.keep_end_pose)}
    return motion, report
