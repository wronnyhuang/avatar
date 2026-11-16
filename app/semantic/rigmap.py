"""canon frames (the interpreter's output) -> the live rig's dials, per character, ONLY for the channels a program drives.

RigMap(character).frames(M, R) -> {rig param: (T,) float}   M = (T,189) canon (root kept neutral), R = (T,6) root track
  (yaw, pitch, roll deg; x, y, z cm — the interpreter's root channel, before it is baked into canon).
driven_params(B, character) -> {rig param: [(t0, t1), ...]} the time spans in which the program's segments / oscillations
  touch each rig dial (a dial outside every span AND at its rest value is left out of the pose: the live layer keeps it).

Sides: canon / API "left" = the avatar's own left = rig `_r` (screen right). Rig `_l` = screen left = the avatar's right.
"""
from __future__ import annotations

import math

import numpy as np

from . import kin
from .retarget.retarget import Retargeter

# rig rest values (motion.js REST: everything 0, hand open 0.2)
REST = {"hand_l_open": 0.2, "hand_r_open": 0.2}
# motion.js RANGE (page clamps again; we emit range-safe values)
RANGE = {
    "smile": (0, 1), "head_yaw": (-0.6, 0.6), "head_pitch": (-0.45, 0.45), "head_roll": (-0.4, 0.4),
    "blink_l": (0, 1), "blink_r": (0, 1), "gaze_x": (-1, 1), "gaze_y": (-1, 1), "brow_l": (-1, 1), "brow_r": (-1, 1),
    "brow_knit": (0, 1), "ear_l": (-1, 1), "ear_r": (-1, 1), "nostril": (0, 1), "point_l": (0, 1), "point_r": (0, 1),
    "body_lean": (-1, 1), "body_bounce": (-1, 1), "body_yaw": (-1, 1), "body_pitch": (-1, 1),
    "hand_l_x": (-1, 1.2), "hand_l_y": (-1, 1.2), "hand_l_z": (-1, 1.2), "hand_r_x": (-1, 1.2), "hand_r_y": (-1, 1.2),
    "hand_r_z": (-1, 1.2), "hand_l_open": (0, 1), "hand_r_open": (0, 1), "hand_l_turn": (-1, 1), "hand_r_turn": (-1, 1),
    "thumb_l": (0, 1), "thumb_r": (0, 1), "shoulder_l": (0, 1), "shoulder_r": (0, 1), "nose_wiggle": (-1, 1),
    "tongue_out": (0, 1), "mouth_open": (0, 1), "eye_wide": (0, 1), "squint": (0, 1),
    "root_x": (-3, 3), "root_y": (-0.6, 3), "root_z": (-3, 1.5), "squash": (-1, 1),
    "root_yaw": (-50, 50), "root_pitch": (-50, 50), "root_roll": (-50, 50),
}
HAS_ROOT = {"muse": True, "juno": True, "fox": False}
HAS_EARS = {"muse": False, "juno": False, "fox": True}
H_CM = 170.0          # one rig body height H in canon centimetres (muse/juno H = 2.30 scene units)

FACE_TO_RIG = {
    "browInnerUp": ("brow_l", "brow_r"), "browOuterUpLeft": ("brow_r",), "browOuterUpRight": ("brow_l",),
    "browDownLeft": ("brow_r", "brow_knit"), "browDownRight": ("brow_l", "brow_knit"),
    "eyeBlinkLeft": ("blink_r",), "eyeBlinkRight": ("blink_l",), "eyeWideLeft": ("eye_wide",),
    "eyeWideRight": ("eye_wide",), "eyeSquintLeft": ("squint",), "eyeSquintRight": ("squint",),
    "cheekSquintLeft": ("squint",), "cheekSquintRight": ("squint",), "mouthSmileLeft": ("smile",),
    "mouthSmileRight": ("smile",), "jawOpen": ("mouth_open",), "noseSneerLeft": ("nostril",),
    "noseSneerRight": ("nostril",),
}
CANON_SIDE_TO_RIG = {"l": "r", "r": "l"}
_FACE_PIN = {"nose", "mouth", "chin", "jaw", "forehead", "top_of_head", "back_of_head", "eyes", "face", "lips"}


def _clip(p, v):
    lo, hi = RANGE.get(p, (-1e9, 1e9))
    return np.minimum(np.maximum(v, lo), hi)


class RigMap:
    def __init__(self, character):
        base = "muse" if character == "juno" else character
        self.character = character
        self.rt = Retargeter(base, overrides={"baseline": {"params": {}}})
        self.rt.base_ema = {}
        n = kin.neutral().astype(np.float64)
        self.n = n
        self.c0 = {k: float(v[0]) for k, v in self.rt.core(n[None]).items()}
        self.has_root = HAS_ROOT.get(character, False)

    def frames(self, M, R):
        """-> {param: (T,)} for every param this character can show (selection happens later)."""
        core = self.rt.core(M)
        out = {}
        for p, v in core.items():
            out[p] = REST.get(p, 0.0) + (v - self.c0[p])
        T = len(M)
        R = np.zeros((T, 6)) if R is None else np.asarray(R, np.float64)
        yaw, pitch, roll, x, y, z = (R[:, k] for k in range(6))
        if self.has_root:
            out["root_yaw"] = yaw / 360.0
            out["root_pitch"] = pitch / 360.0
            out["root_roll"] = roll / 360.0
            out["root_x"] = x / H_CM
            out["root_z"] = z / H_CM
            out["root_y"] = np.maximum(y, 0.0) / H_CM
            out["squash"] = np.minimum(np.maximum(-y, 0.0) / 70.0, 0.4)
        else:   # seated: whole-body moves degrade to the torso dials (no 360 deg turns, no flips: out and back)
            out["body_yaw"] = out.get("body_yaw", 0.0) + np.sin(np.radians(yaw) / 2.0)
            out["body_pitch"] = out.get("body_pitch", 0.0) + 0.9 * np.sin(np.radians(pitch) / 2.0)
            out["body_lean"] = out.get("body_lean", 0.0) + 0.9 * np.sin(np.radians(roll) / 2.0) + x / 30.0
            out["body_bounce"] = y / 20.0 + 0.0 * z
        if HAS_EARS.get(self.character):
            b = 0.5 * (out["brow_l"] + out["brow_r"])
            base = 0.45 * np.maximum(b, 0) + 0.3 * out["eye_wide"] + 0.25 * out["smile"] - 0.5 * out["brow_knit"] \
                - 0.45 * np.maximum(-b, 0)
            out["ear_l"] = base + 0.3 * (out["brow_l"] - b)
            out["ear_r"] = base + 0.3 * (out["brow_r"] - b)
        return {p: _clip(p, np.asarray(v, np.float64) * np.ones(T)) for p, v in out.items()}


def _seg_params(sg, character):
    """rig params a Seg touches."""
    ch, tg = sg.chan, sg.tgt
    root = HAS_ROOT.get(character, False)
    out = []
    if ch in ("root", "rootp"):
        keys = []
        if "neutral" in tg:
            keys = ["yaw", "pitch", "roll", "x", "y", "z"]
        else:
            keys = [k for k in ("yaw", "pitch", "roll", "x", "y", "z") if k in tg]
            if "yaw_add" in tg:
                keys.append("yaw")
            if "pitch_add" in tg:
                keys.append("pitch")
        for k in keys:
            out += _root_key_params(k, root)
    elif ch == "torso":
        m = {"lean_fwd": "body_pitch", "lean_side": "body_lean", "twist": "body_yaw"}
        out += list(m.values()) if "neutral" in tg else [m[k] for k in m if k in tg]
        if root:   # past the torso range the whole body tips (fastrender overflow)
            if "neutral" in tg or "lean_fwd" in tg:
                out.append("root_pitch")
            if "neutral" in tg or "lean_side" in tg:
                out.append("root_roll")
            if "neutral" in tg or "twist" in tg:
                out.append("root_yaw")
    elif ch == "head":
        if "neutral" in tg:
            out += ["head_yaw", "head_pitch", "head_roll"]
        elif "look" in tg:
            out += ["head_yaw", "head_pitch"]
            try:
                from .fastrender import _look_angles
                y, _ = _look_angles(tg["look"], 0.0, 0.0)
                if abs(y) > 25.0:          # past the head's range: the torso helps (fastrender look recruit)
                    out.append("body_yaw")
            except Exception:  # noqa: BLE001
                pass
        elif "ypr" in tg:
            out += ["head_yaw", "head_pitch", "head_roll"][:len(tg["ypr"])]
    elif ch == "gaze":
        out += ["gaze_x", "gaze_y"]
    elif ch.startswith("arm_"):
        s = CANON_SIDE_TO_RIG[ch[-1]]
        out += [f"hand_{s}_x", f"hand_{s}_y", f"hand_{s}_z", f"hand_{s}_turn", f"shoulder_{s}"]
    elif ch.startswith("wrist_"):
        out += [f"hand_{CANON_SIDE_TO_RIG[ch[-1]]}_turn"]
    elif ch.startswith("shrug_"):
        out += [f"shoulder_{CANON_SIDE_TO_RIG[ch[-1]]}"]
    elif ch.startswith("fing_"):
        s = CANON_SIDE_TO_RIG[ch[-1]]
        out += [f"hand_{s}_open", f"point_{s}", f"thumb_{s}"]
    elif ch == "tongue":
        out += ["tongue_out"]
    elif ch == "face":
        if "neutral" in tg:
            names = list(FACE_TO_RIG)
        else:
            names = list(tg.get("shapes", {}))
        for nm in names:
            out += list(FACE_TO_RIG.get(nm, ()))
        if HAS_EARS.get(character) and any(p.startswith(("brow", "eye_wide", "smile")) for p in out):
            out += ["ear_l", "ear_r"]
        if character == "fox" and "nostril" in out:
            out.append("nose_wiggle")
    return out


def _root_key_params(k, root):
    if root:
        return {"yaw": ["root_yaw"], "pitch": ["root_pitch"], "roll": ["root_roll"], "x": ["root_x"],
                "y": ["root_y", "squash"], "z": ["root_z"]}[k]
    # seated: the degraded whole-body move also carries the head
    return {"yaw": ["body_yaw", "head_yaw"], "pitch": ["body_pitch", "head_pitch"], "roll": ["body_lean"],
            "x": ["body_lean"], "y": ["body_bounce", "head_pitch"], "z": ["body_bounce"]}[k]


def _osc_params(o, character):
    ch, prm = o.chan, o.param
    root = HAS_ROOT.get(character, False)
    if ch == "root":
        return _root_key_params(prm, root) if prm in ("yaw", "pitch", "roll", "x", "y", "z") else []
    if ch == "torso":
        return [{"lean_fwd": "body_pitch", "lean_side": "body_lean", "twist": "body_yaw"}.get(prm, "body_yaw")]
    if ch == "head":
        return [{"yaw": "head_yaw", "pitch": "head_pitch", "roll": "head_roll"}.get(prm, "head_yaw")]
    if ch == "gaze":
        return ["gaze_x", "gaze_y"]
    if ch.startswith("hand_"):
        s = CANON_SIDE_TO_RIG[ch[-1]]
        return [f"hand_{s}_x", f"hand_{s}_y", f"hand_{s}_z"]
    if ch.startswith("wrist_"):
        return [f"hand_{CANON_SIDE_TO_RIG[ch[-1]]}_turn"]
    if ch.startswith("shrug_"):
        return [f"shoulder_{CANON_SIDE_TO_RIG[ch[-1]]}"]
    return []


def driven_params(B, character):
    spans = {}
    neut = {}
    for sg in B.segs:
        is_n = "neutral" in sg.tgt or sg.src in ("auto_return", "auto")
        for p in _seg_params(sg, character):
            if is_n:
                neut.setdefault(p, []).append((sg.t0, sg.t1))
            else:
                spans.setdefault(p, []).append((sg.t0, sg.t1))
    for o in B.oscs:
        for p in _osc_params(o, character):
            spans.setdefault(p, []).append((o.t0, o.t1))
    # a hand ON the body / head (reach, touch, cover, pat, rub ... onto a named body part) is placed relative to the torso
    # (and the head, for face targets): while the contact lasts, the torso dials the program does not drive are held at
    # rest (and the head at the program's value) so the live layers' lean / sway / head motion cannot move the hand off
    # its target.  Free-space gestures (wave, point, arm directions) pin nothing.  Causal: a contact lasts until the next
    # segment on that arm starts (open-ended until one exists).
    from . import kin as _kin
    for cs in "lr":
        ss = sorted([sg for sg in B.segs if sg.chan == "arm_" + cs], key=lambda sg: (round(sg.t0, 4), sg.prio))
        for k, sg in enumerate(ss):
            spec = sg.tgt.get("ik") if isinstance(sg.tgt, dict) else None
            if not spec or spec.get("aim"):
                continue
            tg = spec.get("target")
            if not isinstance(tg, str) or tg.endswith("_wrist_rest"):
                continue
            n = _kin.normalize_point_name(tg)
            try:
                on_body = _kin.inward_normal(n, _kin.neutral()) is not None
            except Exception:  # noqa: BLE001
                on_body = False
            if not on_body:
                continue
            end = ss[k + 1].t0 if k + 1 < len(ss) else 1e9
            pins = ["body_lean", "body_yaw", "body_pitch"]
            if n in _FACE_PIN or any(x in n for x in ("eye", "ear", "cheek", "temple")):
                pins += ["head_yaw", "head_pitch", "head_roll"]
            for p in pins:
                spans.setdefault(p, []).append((sg.t0, end))
    # a return-to-neutral only drives what the program moved before it (no listing of untouched dials at rest)
    for p, sp in neut.items():
        if p in spans:
            first = min(a for a, _ in spans[p])
            spans[p] += [x for x in sp if x[1] > first]
    return spans
