"""kin.py — shared kinematics of the semantic motion API (from the motion-planner bake-off).

STABLE API (other agents rely on these names/signatures; additions only, never re-purposed):

  Frames / units.  Canon1 (canon1/spec.py): x = the PERSON's LEFT, y = UP, z = FORWARD (out of the chest).
  World positions here are METRES, pelvis at the origin (+ root_shift, which canon stores in DECIMETRES).
  "left"/"right" always = the person's anatomical side (left_* points have x > 0 at neutral).

  neutral()                        -> (189,) float32 canon neutral: population-neutral body/fingers (_pop_stats.json),
                                      root 0, calm face (Seamless-derived), gaze 0 (looking at the camera).
  neutral_motion(T)                -> (T,189) neutral hold.
  fk(motion)                       -> dict name -> (3,) or (T,3) metres.  Names (see POINT_NAMES):
        joints:  pelvis spine1 spine2 spine3 neck head left_/right_ collar shoulder elbow wrist
        hand:    left_/right_ palm (palm-surface centre), knuckles, index_tip middle_tip ring_tip pinky_tip thumb_tip,
                 index_base ... ; palm_normal_l/_r and finger_dir_l/_r are UNIT VECTORS (not points), in fk(...)['vec']
        face:    nose mouth chin forehead top_of_head back_of_head left_/right_ eye ear cheek, eyes (midpoint), jaw
        body:    neck_front chest heart belly navel left_/right_ shoulder_top hip chest_side, lap, back
        space:   camera (world-fixed: eye height, 1.5 m in front), up down left right forward front back
                 (1.5 m from the chest in the BODY frame; up/down tilt 17/27 deg forward), in_front_of_chest in_front_of_face above_head
  fk_full(motion)                  -> (points dict, rotations dict name -> (T,3,3) global, vec dict)
  ik_arm(side, target, motion=None, part='wrist'|'palm'|'index_tip'|'middle_tip'|'thumb_tip'|'knuckles'|...,
         palm_facing=None|'down'|'up'|'forward'|'back'|'in'|'out'|'left'|'right'|'camera'|'face'|'target'|vec3,
         elbow_hint=None|'down'|'out'|'up'|'forward'|'back'|vec3, hand=None|shape name|(45,) finger vec,
         collar=True, push_out=True)
                                   -> (motion (189,), info{"err_cm","reached","limit_hits":[...],"target":xyz})
                                      side in {'left','right','l','r'}; target = point name or xyz (metres, world).
  look_at(target, motion=None, head_share=0.75, neck_share=0.4)
                                   -> (motion (189,) with neck/head/gaze/eyeLook* set, info{"yaw_deg","pitch_deg",...})
  set_head(m, yaw=0, pitch=0, roll=0)  degrees, relative to neutral, in the BODY's upright frame:
                                      yaw + = turn to the person's LEFT, pitch + = chin UP, roll + = tilt the crown
                                      toward the person's LEFT shoulder.  Neck takes 40 %, head 60 %.
  set_torso(m, lean_fwd=0, lean_side=0, twist=0)   degrees, spread over spine1..3 (+ = forward / to the person's
                                      LEFT / chest turns to the person's LEFT).
  set_root(m, yaw=0, pitch=0, roll=0, shift_cm=(x,y,z))  whole-body root (pelvis) rotation (deg) and shift (cm).
  set_shrug(m, side='both', amount=1.0)  collar elevation (1.0 = full shrug ~ 18 deg).
  set_gaze(m, yaw_deg, pitch_deg)  gaze + matching eyeLook* blendshapes.
  hand_shape(name, side, amount=1.0, base=None) -> (45,) finger axis-angles;  set_hand(m, side, name, amount)
        HAND_SHAPES = relaxed open flat spread fist point thumbs_up thumbs_down peace ok cup claw pinch rock call_me
  ease(name, u)  u in [0,1] -> [0,1]: linear ease_in ease_out ease_in_out smoothstep min_jerk overshoot back_in
                 anticipate settle spring;  min_jerk(u)
  blend(a, b, w)  slerp per joint (rotation triples), lerp for root shift / face / gaze; a,b (189,) or (T,189)
  slerp_aa(a, b, w), unwrap_aa(seq) (continuous axis-angle over time for spins), rot(axis, deg) -> 3x3,
  aa_to_mat, mat_to_aa, quat helpers
  JOINT_LIMITS (degrees), check_limits(m) -> list of str
  push_out(point, motion, margin=0.02) -> point moved outside the head / neck / torso volumes; inside_body(point, m)
  bs(m, name, value) set a blendshape; BS_NAMES, FPS = 30

Accuracy: the skeleton is the SMPL neutral template (canon1/skeleton.py, ~1 cm); hand/face/body surface points are
hand-placed MANO/SMPL-like offsets (approx. +-1-2 cm).  IK is self-consistent with fk() to < 1 mm when reachable.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

from .canon1 import spec  # noqa: E402
from .canon1.skeleton import REST  # noqa: E402

import json as _json  # noqa: E402

FPS = spec.FPS
BS_NAMES = spec.BS_NAMES
COL = spec.COL
D = spec.D_MOTION
_POP = _json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "retarget", "maps", "_pop_stats.json")))

JOINTS = spec.JOINTS                      # root_rot spine1 spine2 spine3 neck head collar_l ... wrist_r
FINGER_JOINTS = spec.FINGER_JOINTS        # index1..3 middle1..3 pinky1..3 ring1..3 thumb1..3
FINGERS = ["index", "middle", "pinky", "ring", "thumb"]   # canon order of the 5 fingers


def _side(s):
    s = str(s).lower()
    if s in ("l", "left"):
        return "l"
    if s in ("r", "right"):
        return "r"
    raise ValueError(f"side must be 'left' or 'right', got {s!r}")


def _long(s):
    return "left" if _side(s) == "l" else "right"


# ============================================================================================ rotations
def aa_to_mat(aa):
    a = np.asarray(aa)
    if a.shape == (3,):
        return _aa2m(a)
    return spec.aa_to_mat(aa)


def mat_to_aa(R):
    R = np.asarray(R, np.float64)
    if R.shape == (3, 3):
        return _m2aa(R)
    return spec.mat_to_aa(R)


def _m2aa(R):
    """fast single 3x3 -> axis-angle (same branch logic as spec.mat_to_aa)."""
    cos = (R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2
    cos = 1.0 if cos > 1 else (-1.0 if cos < -1 else cos)
    th = math.acos(cos)
    v0, v1, v2 = R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]
    sn = math.sin(th)
    if sn >= 1e-6:
        f = th / (2 * sn)
        return np.array([v0 * f, v1 * f, v2 * f])
    if th < 1:
        return np.array([v0 / 2, v1 / 2, v2 / 2])
    return spec.mat_to_aa(R[None])[0]


def rot(axis, deg):
    """3x3 rotation about a named axis ('x','y','z' or a 3-vector) by deg degrees (right-handed)."""
    if isinstance(axis, str):
        axis = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    a = np.asarray(axis, np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    return aa_to_mat(a * math.radians(deg))


def mat_to_quat(R):
    R = np.asarray(R, np.float64)
    return aa_to_quat(mat_to_aa(R))


def aa_to_quat(aa):
    aa = np.asarray(aa, np.float64)
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(th, 1e-12)
    return np.concatenate([np.cos(th / 2), k * np.sin(th / 2)], -1)


def quat_to_aa(q):
    q = np.asarray(q, np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w = np.clip(q[..., :1], -1, 1)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    th = 2 * np.arctan2(s, w)
    return np.where(s > 1e-9, v / np.maximum(s, 1e-12) * th, 2 * v)


def _qslerp(q0, q1, w):
    q0 = np.asarray(q0, np.float64); q1 = np.asarray(q1, np.float64)
    w = np.asarray(w, np.float64)
    d = np.sum(q0 * q1, -1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    d = np.abs(d)
    w = w[..., None] if w.ndim == q0.ndim - 1 and w.ndim > 0 else w
    th = np.arccos(np.clip(d, -1, 1))
    s = np.sin(th)
    small = s < 1e-6
    a = np.where(small, 1 - w, np.sin((1 - w) * th) / np.where(small, 1, s))
    b = np.where(small, w, np.sin(w * th) / np.where(small, 1, s))
    return a * q0 + b * q1


def slerp_aa(a, b, w):
    """slerp between axis-angle vectors (...,3); w scalar or broadcastable (...,)."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    w = np.asarray(w, np.float64)
    if w.ndim > 0:
        w = w[..., None]
    q = _qslerp(aa_to_quat(a), aa_to_quat(b), w if w.ndim == 0 else w)
    return quat_to_aa(q)


def unwrap_aa(seq):
    """(T,3) axis-angle sequence -> equivalent sequence without +-pi flips (continuous for spins > 180 deg)."""
    s = np.array(seq, np.float64)
    for t in range(1, len(s)):
        th = np.linalg.norm(s[t])
        if th < 1e-9:
            continue
        k = s[t] / th
        best, bd = s[t], np.linalg.norm(s[t] - s[t - 1])
        for cand_th, cand_k in ((th - 2 * math.pi, k), (th + 2 * math.pi, k), (2 * math.pi - th, -k),
                                (-(2 * math.pi - th), -k), (th - 4 * math.pi, k), (th + 4 * math.pi, k)):
            c = cand_k * cand_th
            dd = np.linalg.norm(c - s[t - 1])
            if dd < bd - 1e-9:
                best, bd = c, dd
        s[t] = best
    return s


def rot_between(a, b):
    """minimal rotation 3x3 taking direction a onto direction b."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    a = a / max(np.linalg.norm(a), 1e-12); b = b / max(np.linalg.norm(b), 1e-12)
    v = np.cross(a, b); c = float(a @ b); s = float(np.linalg.norm(v))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        p = np.cross(a, [0, 1, 0])
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(a, [0, 0, 1])
        return _aa2m(p / np.linalg.norm(p) * math.pi)
    return _aa2m(v / s * math.atan2(s, c))


def _frame_from(a0, b0, a1, b1):
    """rotation R with R a0 = a1 and R b0 ~ b1 (Gram-Schmidt on both pairs)."""
    def basis(a, b):
        a = a / np.linalg.norm(a)
        b = b - a * (a @ b)
        nb = np.linalg.norm(b)
        if nb < 1e-9:
            b = np.cross(a, [0, 0, 1]) if abs(a[2]) < 0.9 else np.cross(a, [1, 0, 0])
            nb = np.linalg.norm(b)
        b = b / nb
        return np.stack([a, b, np.cross(a, b)], 1)
    return basis(np.asarray(a1, float), np.asarray(b1, float)) @ basis(np.asarray(a0, float), np.asarray(b0, float)).T


def _swing_twist(R, axis):
    """decompose R = swing @ twist with twist about `axis` (unit). -> (swing 3x3, twist angle rad)."""
    q = mat_to_quat(R)
    p = np.dot(q[1:], axis) * axis
    tw = np.concatenate([[q[0]], p])
    n = np.linalg.norm(tw)
    if n < 1e-9:
        return R, 0.0
    tw = tw / n
    ang = 2 * math.atan2(np.dot(tw[1:], axis), tw[0])
    ang = (ang + math.pi) % (2 * math.pi) - math.pi
    T = aa_to_mat(axis * ang)
    return R @ T.T, ang


# ============================================================================================ easing
def min_jerk(u):
    u = np.clip(np.asarray(u, np.float64), 0, 1)
    return u ** 3 * (10 - 15 * u + 6 * u * u)


def ease(name, u):
    """named easing u in [0,1] -> progress (overshoot/settle/spring may exceed 1 temporarily)."""
    u = np.clip(np.asarray(u, np.float64), 0, 1)
    name = (name or "min_jerk").lower()
    if name == "linear":
        return u
    if name in ("ease_in", "in"):
        return u ** 3
    if name in ("ease_out", "out"):
        return 1 - (1 - u) ** 3
    if name in ("ease_in_out", "inout", "cubic"):
        return np.where(u < 0.5, 4 * u ** 3, 1 - (-2 * u + 2) ** 3 / 2)
    if name == "smoothstep":
        return u * u * (3 - 2 * u)
    if name in ("min_jerk", "minjerk", "natural", "smooth", "default"):
        return min_jerk(u)
    if name in ("overshoot", "back_out"):
        c = 1.4
        v = u - 1
        return 1 + (c + 1) * v ** 3 + c * v ** 2
    if name in ("back_in", "anticipate"):
        c = 1.4
        return (c + 1) * u ** 3 - c * u ** 2
    if name in ("settle", "overshoot_settle"):       # min-jerk to ~106 % then settle back to 100 %
        return min_jerk(np.clip(u / 0.7, 0, 1)) * 1.06 - 0.06 * min_jerk(np.clip((u - 0.7) / 0.3, 0, 1))
    if name == "spring":
        return 1 - np.exp(-6 * u) * np.cos(9 * u) * (1 - u) - 0 * u
    raise ValueError(f"unknown easing {name!r}")


EASINGS = ["linear", "ease_in", "ease_out", "ease_in_out", "smoothstep", "min_jerk", "overshoot", "anticipate",
           "settle", "spring"]


# ============================================================================================ neutral
NEUTRAL_FACE = {  # calm resting face (Seamless medians, de-biased: talking-smile and squint halved, blink low)
    "browDownLeft": 0.04, "browDownRight": 0.04, "browInnerUp": 0.04, "browOuterUpLeft": 0.08,
    "browOuterUpRight": 0.07, "eyeBlinkLeft": 0.08, "eyeBlinkRight": 0.08, "eyeSquintLeft": 0.2,
    "eyeSquintRight": 0.18, "jawOpen": 0.03, "mouthClose": 0.01, "mouthDimpleLeft": 0.02, "mouthDimpleRight": 0.02,
    "mouthPressLeft": 0.06, "mouthPressRight": 0.06, "mouthPucker": 0.02, "mouthRollUpper": 0.02,
    "mouthShrugLower": 0.02, "mouthShrugUpper": 0.015, "mouthSmileLeft": 0.1, "mouthSmileRight": 0.09,
    "mouthStretchLeft": 0.01, "mouthStretchRight": 0.01, "mouthUpperUpLeft": 0.02, "mouthUpperUpRight": 0.02,
}

_NEUTRAL = None


def neutral():
    """(189,) float32 canon neutral pose (copy)."""
    global _NEUTRAL
    if _NEUTRAL is None:
        m = np.zeros(D, np.float64)
        for j, v in _POP["neutral_body"].items():
            m[spec.jcols(j)] = v
        m[spec.SL_FINGERS_L.start:spec.SL_FINGERS_R.stop] = _POP["neutral_fingers"]
        for n, v in NEUTRAL_FACE.items():
            m[COL["bs_" + n]] = v
        _NEUTRAL = m.astype(np.float32)
    return _NEUTRAL.copy()


def neutral_motion(T):
    return np.repeat(neutral()[None], int(T), 0)


def bs(m, name, value):
    m[COL["bs_" + name]] = float(np.clip(value, 0, 1))
    return m


def jget(m, j):
    return np.asarray(m, np.float64)[spec.jcols(j)]


def jset(m, j, aa):
    m[spec.jcols(j)] = aa
    return m


# ============================================================================================ hand geometry
# Wrist-local rest offsets of the LEFT hand (x = along the fingers in the T-pose, -y = palm side, +z = thumb side).
# MANO-like, metres. Right hand = mirror (x -> -x).
_FINGER_GEOM_L = {
    "index": ((0.088, 0.000, 0.025), (0.037, 0.025, 0.022)),
    "middle": ((0.092, 0.000, 0.004), (0.040, 0.028, 0.024)),
    "pinky": ((0.074, -0.004, -0.032), (0.028, 0.020, 0.020)),
    "ring": ((0.085, -0.002, -0.015), (0.037, 0.026, 0.022)),
    "thumb": ((0.025, -0.012, 0.022), (0.035, 0.032, 0.028)),
}
_THUMB_DIR_L = np.array([0.6, -0.25, 0.75]) / np.linalg.norm([0.6, -0.25, 0.75])
_PALM_L = np.array([0.050, -0.016, 0.002])          # palm-surface centre
_KNUCKLES_L = np.array([0.088, 0.008, 0.0])          # back of the knuckles (middle MCP)
_BACKHAND_L = np.array([0.050, 0.014, 0.0])          # back of the hand (dorsum) centre
HAND_THICK = 0.025


def _mir(v, s):
    v = np.asarray(v, np.float64)
    return v if s == "l" else v * np.array([-1.0, 1.0, 1.0])


def _fsl(s):
    return spec.SL_FINGERS_L if s == "l" else spec.SL_FINGERS_R


def hand_points_local(fingers45, s):
    """finger axis-angles (45,) or (T,45) -> dict of wrist-local points (T,3) for that hand."""
    f = np.asarray(fingers45, np.float64)
    if f.ndim == 1:
        f = f[None]
    T = len(f)
    R = aa_to_mat(f.reshape(T, 15, 3))                    # (T,15,3,3)
    out = {}
    for k, fn in enumerate(FINGERS):
        base, segs = _FINGER_GEOM_L[fn]
        base = _mir(base, s)
        if fn == "thumb":
            d0 = _mir(_THUMB_DIR_L, s)
        else:
            d0 = _mir([1.0, 0, 0], s)
        Rg = np.broadcast_to(np.eye(3), (T, 3, 3))
        p = np.broadcast_to(base, (T, 3)).copy()
        out[fn + "_base"] = p.copy()
        for q in range(3):
            Rg = Rg @ R[:, 3 * k + q]
            p = p + Rg @ (d0 * segs[q])
            if q == 0:
                out[fn + "_mid"] = p.copy()
        out[fn + "_tip"] = p
    out["palm"] = np.broadcast_to(_mir(_PALM_L, s), (T, 3)).copy()
    out["knuckles"] = np.broadcast_to(_mir(_KNUCKLES_L, s), (T, 3)).copy()
    out["back_of_hand"] = np.broadcast_to(_mir(_BACKHAND_L, s), (T, 3)).copy()
    out["wrist"] = np.zeros((T, 3))
    out["fingertips"] = np.mean([out[f + "_tip"] for f in ("index", "middle", "ring")], 0)
    return out


# ============================================================================================ face / body offsets
# head-local (head joint frame at rest, metres)
FACE_POINTS = {
    "nose": (0.0, 0.035, 0.118), "mouth": (0.0, -0.012, 0.108), "lips": (0.0, -0.012, 0.108),
    "chin": (0.0, -0.05, 0.092), "jaw": (0.0, -0.045, 0.06), "forehead": (0.0, 0.125, 0.098),
    "top_of_head": (0.0, 0.195, 0.01), "back_of_head": (0.0, 0.09, -0.095),
    "left_eye": (0.032, 0.075, 0.095), "right_eye": (-0.032, 0.075, 0.095), "eyes": (0.0, 0.075, 0.098),
    "left_ear": (0.078, 0.05, -0.005), "right_ear": (-0.078, 0.05, -0.005),
    "left_cheek": (0.052, 0.02, 0.085), "right_cheek": (-0.052, 0.02, 0.085),
    "left_temple": (0.068, 0.10, 0.05), "right_temple": (-0.068, 0.10, 0.05),
}
HEAD_ELLIPSOID = ((0.0, 0.075, 0.0), (0.075, 0.12, 0.092))   # centre, radii (head frame)
# chest-local (spine3 frame) / belly (spine1 frame) / pelvis frame
CHEST_POINTS = {"chest": (0.0, 0.06, 0.115), "heart": (0.055, 0.05, 0.11), "sternum": (0.0, 0.06, 0.115),
                "left_chest": (0.07, 0.06, 0.11), "right_chest": (-0.07, 0.06, 0.11),
                "back": (0.0, 0.06, -0.11), "left_chest_side": (0.16, 0.0, 0.0), "right_chest_side": (-0.16, 0.0, 0.0)}
CHEST_VOL = ((0.0, 0.02, 0.0), (0.15, 0.20, 0.105), (-0.17, 0.22))   # centre(x,_,z), half-axes(x,_,z), y-range
BELLY_POINTS = {"belly": (0.0, 0.02, 0.13), "navel": (0.0, -0.02, 0.13), "stomach": (0.0, 0.05, 0.125)}
BELLY_VOL = ((0.0, 0.0, 0.01), (0.14, 0.0, 0.115), (-0.18, 0.12))
PELVIS_POINTS = {"left_hip": (0.16, 0.0, 0.02), "right_hip": (-0.16, 0.0, 0.02), "lap": (0.0, -0.08, 0.30),
                 "waist": (0.0, 0.05, 0.12), "left_knee": (0.10, -0.42, 0.10), "right_knee": (-0.10, -0.42, 0.10),
                 "knees": (0.0, -0.42, 0.12), "left_thigh": (0.10, -0.20, 0.12), "right_thigh": (-0.10, -0.20, 0.12)}
NECK_RADIUS = 0.055

SPACE_DIRS = {"up": (0, 1, 0.3), "down": (0, -1, 0.5), "left": (1, 0, 0), "right": (-1, 0, 0), "forward": (0, 0, 1),
              "front": (0, 0, 1), "back": (0, 0, -1), "behind": (0, 0, -1), "sky": (0, 1, 0.3), "floor": (0, -1, 0.6),
              "ceiling": (0, 1, 0.2), "upper_left": (0.7, 0.7, 0.3), "upper_right": (-0.7, 0.7, 0.3),
              "lower_left": (0.7, -0.6, 0.4), "lower_right": (-0.7, -0.6, 0.4)}

POINT_NAMES = (["pelvis", "spine1", "spine2", "spine3", "neck", "head"] +
               [f"{sd}_{j}" for sd in ("left", "right") for j in
                ("collar", "shoulder", "elbow", "wrist", "palm", "knuckles", "back_of_hand", "fingertips",
                 "index_tip", "middle_tip", "ring_tip", "pinky_tip", "thumb_tip", "index_base", "middle_base",
                 "thumb_base", "index_mid", "shoulder_top", "hip", "upper_arm", "forearm")] +
               list(FACE_POINTS) + list(CHEST_POINTS) + list(BELLY_POINTS) + list(PELVIS_POINTS) +
               ["neck_front", "throat", "camera", "viewer", "in_front_of_chest", "in_front_of_face", "above_head"] +
               list(SPACE_DIRS))

_BODY_J = ["root_rot", "spine1", "spine2", "spine3", "neck", "head", "collar_l", "collar_r", "shoulder_l",
           "shoulder_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r"]
_CAM = None


def fk_full(motion):
    """(T,189) or (189,) -> (points{name:(T,3)}, rots{joint:(T,3,3) global}, vec{name:(T,3)}).
    Single-frame input returns (3,)-shaped entries."""
    m = np.asarray(motion, np.float64)
    single = m.ndim == 1
    if single:
        m = m[None]
    T = len(m)
    Rg, P = {}, {}
    for j in _BODY_J:
        par, off = REST[j]
        R = aa_to_mat(m[:, spec.jcols(j)])
        if par is None:
            Rg[j] = R
            P[j] = m[:, 3:6] / 10.0
        else:
            Rg[j] = Rg[par] @ R
            P[j] = P[par] + np.einsum("tij,j->ti", Rg[par], np.asarray(off))
    pts = {"pelvis": P["root_rot"]}
    for j in ("spine1", "spine2", "spine3", "neck", "head"):
        pts[j] = P[j]
    vec = {}
    for s, sd in (("l", "left"), ("r", "right")):
        for j in ("collar", "shoulder", "elbow", "wrist"):
            pts[f"{sd}_{j}"] = P[f"{j}_{s}"]
        Rw = Rg["wrist_" + s]
        hp = hand_points_local(m[:, _fsl(s)], s)
        for k, v in hp.items():
            if k == "wrist":
                continue
            pts[f"{sd}_{k}"] = P["wrist_" + s] + np.einsum("tij,tj->ti", Rw, v)
        vec["palm_normal_" + s] = np.einsum("tij,j->ti", Rw, np.array([0.0, -1.0, 0.0]))
        vec["finger_dir_" + s] = np.einsum("tij,j->ti", Rw, _mir([1.0, 0, 0], s))
        vec["thumb_dir_" + s] = np.einsum("tij,j->ti", Rw, np.array([0.0, 0.0, 1.0]))
        pts[f"{sd}_shoulder_top"] = P[f"shoulder_{s}"] + np.einsum("tij,j->ti", Rg["spine3"], np.array([0, 0.055, 0]))
        pts[f"{sd}_upper_arm"] = 0.5 * (P[f"shoulder_{s}"] + P[f"elbow_{s}"])
        pts[f"{sd}_forearm"] = 0.5 * (P[f"wrist_{s}"] + P[f"elbow_{s}"])
    for n, off in FACE_POINTS.items():
        pts[n] = P["head"] + np.einsum("tij,j->ti", Rg["head"], np.asarray(off))
    for n, off in CHEST_POINTS.items():
        pts[n] = P["spine3"] + np.einsum("tij,j->ti", Rg["spine3"], np.asarray(off))
    for n, off in BELLY_POINTS.items():
        pts[n] = P["spine1"] + np.einsum("tij,j->ti", Rg["spine1"], np.asarray(off))
    for n, off in PELVIS_POINTS.items():
        pts[n] = P["root_rot"] + np.einsum("tij,j->ti", Rg["root_rot"], np.asarray(off))
    pts["neck_front"] = P["neck"] + np.einsum("tij,j->ti", Rg["neck"], np.array([0, 0.03, 0.06]))
    pts["throat"] = pts["neck_front"]
    pts["camera"] = np.broadcast_to(camera_point(), (T, 3)).copy()
    pts["viewer"] = pts["camera"]
    chest = pts["chest"]
    pts["in_front_of_chest"] = chest + np.einsum("tij,j->ti", Rg["spine3"], np.array([0, 0, 0.30]))
    pts["in_front_of_face"] = pts["nose"] + np.einsum("tij,j->ti", Rg["head"], np.array([0, 0, 0.30]))
    pts["above_head"] = pts["top_of_head"] + np.array([0, 0.25, 0.05])
    for n, d in SPACE_DIRS.items():
        dv = np.asarray(d, np.float64) / np.linalg.norm(d)
        pts[n] = chest + np.einsum("tij,j->ti", Rg["root_rot"], dv * 1.5)
    for k in ("palm_normal", "finger_dir", "thumb_dir"):
        vec[k + "_left"] = vec[k + "_l"]; vec[k + "_right"] = vec[k + "_r"]
    if single:
        pts = {k: v[0] for k, v in pts.items()}
        Rg = {k: v[0] for k, v in Rg.items()}
        vec = {k: v[0] for k, v in vec.items()}
    return pts, Rg, vec


def fk(motion):
    """(T,189)|(189,) -> {point name: (T,3)|(3,)} metres; fk(m)['vec'] holds palm_normal_l/_r etc."""
    p, _, v = fk_full(motion)
    p = dict(p)
    p["vec"] = v
    return p


def camera_point():
    """the viewer / camera: 1.5 m in front of the neutral eyes, at eye height (world-fixed)."""
    global _CAM
    if _CAM is None:
        Rg, P = _arm_chain(neutral(), "l")
        e = P["head"] + Rg["head"] @ np.asarray(FACE_POINTS["eyes"])
        _CAM = np.array([0.0, e[1], e[2] + 1.5])
    return _CAM.copy()


def point(name, motion=None):
    """resolve a target: point name (aliases ok) or xyz -> (3,) metres in world for one pose."""
    if not isinstance(name, str):
        return np.asarray(name, np.float64)
    m = neutral() if motion is None else motion
    return point_fast(name, m)


_ALIASES = {"eye_l": "left_eye", "eye_r": "right_eye", "l_eye": "left_eye", "r_eye": "right_eye",
            "both_eyes": "eyes", "face": "nose", "lips": "mouth", "head_top": "top_of_head", "crown": "top_of_head",
            "top_head": "top_of_head", "brow": "forehead", "brows": "forehead", "tummy": "belly",
            "stomach": "stomach", "heart": "heart", "user": "camera", "you": "camera", "screen": "camera",
            "audience": "camera", "sky": "sky", "ground": "floor", "mouth_center": "mouth", "neck": "neck_front"}


def normalize_point_name(n):
    n = str(n).strip().lower().replace(" ", "_").replace("-", "_")
    n = _ALIASES.get(n, n)
    for a, b in (("l_", "left_"), ("r_", "right_")):
        if n.startswith(a) and ("left_" + n[2:] in POINT_NAMES or "right_" + n[2:] in POINT_NAMES):
            n = b + n[2:]
    if n.endswith("_l") and "left_" + n[:-2] in POINT_NAMES:
        n = "left_" + n[:-2]
    if n.endswith("_r") and "right_" + n[:-2] in POINT_NAMES:
        n = "right_" + n[:-2]
    if n.startswith("own_"):
        n = n[4:]
    return n


# ============================================================================================ collision
def _to_local(p, R, o):
    return R.T @ (np.asarray(p) - o)


def inside_body(pt, motion, margin=0.0):
    """-> name of the volume (head / neck / chest / belly) the point is inside, else None (one pose)."""
    return _push(pt, motion, margin, dry=True)


def push_out(pt, motion, margin=0.02):
    """move a world point (3,) out of the head ellipsoid / neck / chest / belly volumes (+ margin, metres)."""
    return _push(pt, motion, margin, dry=False)


def _push(pt, motion, margin, dry):
    m = np.asarray(motion, np.float64)
    Rg, Pj = _arm_chain(m, "l")
    p = np.asarray(pt, np.float64).copy()
    # head ellipsoid
    c, r = HEAD_ELLIPSOID
    R, o = Rg["head"], Pj["head"]
    q = _to_local(p, R, o) - np.asarray(c)
    rr = np.asarray(r) + margin
    k = np.sqrt(np.sum((q / rr) ** 2))
    if k < 1.0:
        if dry:
            return "head"
        q = q / max(k, 1e-6)
        p = o + R @ (q + np.asarray(c))
    # neck cylinder (vertical in the neck frame, between neck joint and head joint)
    R, o = Rg["neck"], Pj["neck"]
    q = _to_local(p, R, o)
    if -0.02 < q[1] < 0.07:
        rad = math.hypot(q[0], q[2] - 0.01)
        if rad < NECK_RADIUS + margin:
            if dry:
                return "neck"
            ang = math.atan2(q[0], q[2] - 0.01) if rad > 1e-6 else 0.0
            q[0] = math.sin(ang) * (NECK_RADIUS + margin); q[2] = 0.01 + math.cos(ang) * (NECK_RADIUS + margin)
            p = o + R @ q
    for name, jn, vol in (("chest", "spine3", CHEST_VOL), ("belly", "spine1", BELLY_VOL)):
        (cx, _, cz), (hx, _, hz), (y0, y1) = vol
        R, o = Rg[jn], Pj[jn]
        q = _to_local(p, R, o)
        if y0 < q[1] < y1:
            ex, ez = (q[0] - cx) / (hx + margin), (q[2] - cz) / (hz + margin)
            k = math.hypot(ex, ez)
            if k < 1.0:
                if dry:
                    return name
                if k < 1e-6:
                    ex, ez, k = 0.0, 1.0, 1.0
                q[0] = cx + ex / k * (hx + margin); q[2] = cz + ez / k * (hz + margin)
                p = o + R @ q
    return None if dry else p


def _joint_pos(m):
    m = np.asarray(m, np.float64)
    Rg, P = {}, {}
    for j in _BODY_J:
        par, off = REST[j]
        R = aa_to_mat(m[spec.jcols(j)])
        if par is None:
            Rg[j], P[j] = R, m[3:6] / 10.0
        else:
            Rg[j] = Rg[par] @ R
            P[j] = P[par] + Rg[par] @ np.asarray(off)
    return P


# ============================================================================================ joint limits
JOINT_LIMITS = {   # degrees (generous, anatomical-ish)
    "elbow_flex": (0, 150), "elbow_twist": (-175, 175), "wrist_swing": 80, "wrist_twist": 45,
    "upper_arm_behind": 0.75,      # max backward component (z) of the upper-arm direction in the chest frame
    "head_yaw": (-80, 80), "head_pitch": (-55, 50), "head_roll": (-40, 40),
    "gaze_yaw": (-35, 35), "gaze_pitch": (-25, 25),
    "spine_lean_fwd": (-30, 60), "spine_lean_side": (-35, 35), "spine_twist": (-50, 50),
    "collar_lift": (-5, 25),
}


def check_limits(m, sides=("l", "r"), head=True):
    """-> list of human-readable limit violations of one pose."""
    m = np.asarray(m, np.float64)
    out = []
    Rg, _ = _arm_chain(m)
    for s, sd in (("l", "left"), ("r", "right")):
        if s not in sides:
            continue
        Rs = Rg["shoulder_" + s]; Re = Rg["elbow_" + s]
        u = Rs @ (np.asarray(REST["elbow_" + s][1]) / np.linalg.norm(REST["elbow_" + s][1]))
        f = Re @ (np.asarray(REST["wrist_" + s][1]) / np.linalg.norm(REST["wrist_" + s][1]))
        flex = math.degrees(math.acos(np.clip(u @ f, -1, 1)))
        if flex > JOINT_LIMITS["elbow_flex"][1] + 1:
            out.append(f"{sd} elbow flexed {flex:.0f} deg > {JOINT_LIMITS['elbow_flex'][1]}")
        uc = Rg["spine3"].T @ u
        if uc[2] < -JOINT_LIMITS["upper_arm_behind"]:
            out.append(f"{sd} upper arm points behind the back")
        w = m[spec.jcols("wrist_" + s)]
        if math.degrees(np.linalg.norm(w - jget(neutral(), "wrist_" + s))) > JOINT_LIMITS["wrist_swing"] + 25:
            out.append(f"{sd} wrist bent {math.degrees(np.linalg.norm(w)):.0f} deg")
    y, p, r = head_ypr(m) if head else (0.0, 0.0, 0.0)
    for nm, v in (("head_yaw", y), ("head_pitch", p), ("head_roll", r)):
        lo, hi = JOINT_LIMITS[nm]
        if not (lo - 1 <= v <= hi + 1):
            out.append(f"{nm} {v:.0f} deg outside [{lo},{hi}]")
    return out


# ============================================================================================ head / torso / root
def _chest_R(m):
    _, Rg, _ = fk_full(m)
    return Rg


def _apply_world_delta(m, joints_shares, Dw, parent_R):
    """rotate a chain of joints (whose common parent has global rotation parent_R) by world delta Dw, split by
    shares (sum 1).  Shares are powers of the same rotation, so they commute and compose exactly."""
    aa = mat_to_aa(parent_R.T @ Dw @ parent_R)          # delta in the parent's frame
    acc = np.eye(3)
    for j, sh in joints_shares:
        Rl = aa_to_mat(jget(m, j))
        Dk = aa_to_mat(aa * sh)
        # local' = acc^T Dk acc Rl  (acc = product of previous local rotations of this chain, applied in parent frame)
        new = acc.T @ Dk @ acc @ Rl
        acc = acc @ Rl
        jset(m, j, mat_to_aa(new))
    return m


def body_up_R(m):
    """global rotation of the 'upright body' frame = the root rotation (pelvis)."""
    return aa_to_mat(np.asarray(m, np.float64)[0:3])


def set_head(m, yaw=0.0, pitch=0.0, roll=0.0, neck_share=0.4, relative_to="neutral"):
    """head rotation in degrees (yaw + left, pitch + up, roll + crown toward the person's LEFT shoulder), measured in
    the body's upright frame, relative to the NEUTRAL head (relative_to='neutral') or the current head ('current')."""
    m = np.array(m, np.float64)
    if relative_to == "neutral":
        n = neutral()
        jset(m, "neck", jget(n, "neck")); jset(m, "head", jget(n, "head"))
    Rb = body_up_R(m)
    Dw = Rb @ rot("y", yaw) @ rot("x", -pitch) @ rot("z", -roll) @ Rb.T
    Rg, _ = _arm_chain(m)
    _apply_world_delta(m, [("neck", neck_share), ("head", 1 - neck_share)], Dw, Rg["spine3"])
    return m


def head_ypr(m):
    """head orientation relative to the neutral head, in the body's upright frame: (yaw+left, pitch+up, roll+left) deg."""
    m = np.asarray(m, np.float64)
    Rg, _ = _arm_chain(m)
    n = neutral().astype(np.float64)
    n[0:6] = m[0:6]
    n[6:15] = m[6:15]                                     # same spine -> pure head delta
    Rn, _ = _arm_chain(n)
    Rb = body_up_R(m)
    Dl = Rb.T @ Rg["head"] @ Rn["head"].T @ Rb
    return ypr_of(Dl)


def ypr_of(R):
    """R = Ry(yaw) Rx(-pitch) Rz(-roll) -> (yaw + left, pitch + up, roll + crown to the LEFT) degrees (YXZ Euler)."""
    yaw = math.degrees(math.atan2(R[0, 2], R[2, 2]))
    pitch = -math.degrees(math.asin(np.clip(-R[1, 2], -1, 1)))
    roll = -math.degrees(math.atan2(R[1, 0], R[1, 1]))
    return yaw, pitch, roll


def set_torso(m, lean_fwd=0.0, lean_side=0.0, twist=0.0, relative_to="neutral"):
    """torso pose in degrees spread over spine1/2/3 (0.4/0.3/0.3): lean_fwd + forward, lean_side + crown to the
    person's LEFT, twist + chest turns to the person's LEFT.  Head/arms ride along."""
    m = np.array(m, np.float64)
    if relative_to == "neutral":
        n = neutral()
        for j in ("spine1", "spine2", "spine3"):
            jset(m, j, jget(n, j))
    Rb = body_up_R(m)
    Dw = Rb @ rot("y", twist) @ rot("x", lean_fwd) @ rot("z", -lean_side) @ Rb.T
    _apply_world_delta(m, [("spine1", 0.4), ("spine2", 0.3), ("spine3", 0.3)], Dw, Rb)
    return m


def set_root(m, yaw=0.0, pitch=0.0, roll=0.0, shift_cm=(0.0, 0.0, 0.0)):
    """whole-body root: yaw + turns to the person's LEFT (spins), pitch + leans the whole body FORWARD, roll + to
    the LEFT; shift_cm (x left, y up, z forward) in cm -> root_shift (decimetres)."""
    m = np.array(m, np.float64)
    R = rot("y", yaw) @ rot("x", pitch) @ rot("z", -roll)
    if abs(pitch) < 1e-9 and abs(roll) < 1e-9:
        m[0:3] = [0.0, math.radians(yaw), 0.0]           # keep > 180 deg yaw as written (continuous spins)
    else:
        m[0:3] = mat_to_aa(R)
    m[3:6] = np.asarray(shift_cm, np.float64) / 10.0
    return m


def set_shrug(m, side="both", amount=1.0):
    """collar elevation: amount 1.0 = a full shrug (~18 deg lift), negative = shoulders dropped."""
    m = np.array(m, np.float64)
    sides = ["l", "r"] if side in ("both", None) else [_side(side)]
    n = neutral()
    for s in sides:
        sg = 1.0 if s == "l" else -1.0
        Rn = aa_to_mat(jget(n, "collar_" + s))
        jset(m, "collar_" + s, mat_to_aa(rot("z", sg * 18.0 * amount) @ Rn))
    return m


def gaze_shapes(yaw_deg, pitch_deg):
    """eyeLook* blendshapes consistent with a gaze direction (yaw + = toward the person's left, pitch + = up)."""
    y = float(np.clip(yaw_deg / 30.0, -1, 1)); p = float(np.clip(pitch_deg / 25.0, -1, 1))
    out = {k: 0.0 for k in ("eyeLookOutLeft", "eyeLookInLeft", "eyeLookOutRight", "eyeLookInRight",
                            "eyeLookUpLeft", "eyeLookUpRight", "eyeLookDownLeft", "eyeLookDownRight")}
    if y > 0:
        out["eyeLookOutLeft"] = y; out["eyeLookInRight"] = y
    else:
        out["eyeLookInLeft"] = -y; out["eyeLookOutRight"] = -y
    if p > 0:
        out["eyeLookUpLeft"] = out["eyeLookUpRight"] = p
    else:
        out["eyeLookDownLeft"] = out["eyeLookDownRight"] = -p
    return out


def set_gaze(m, yaw_deg, pitch_deg):
    m = np.array(m, np.float64)
    yl, pl = JOINT_LIMITS["gaze_yaw"], JOINT_LIMITS["gaze_pitch"]
    y = float(np.clip(yaw_deg, *yl)); p = float(np.clip(pitch_deg, *pl))
    m[COL["gaze_yaw"]] = math.radians(y); m[COL["gaze_pitch"]] = math.radians(p)
    for k, v in gaze_shapes(y, p).items():
        m[COL["bs_" + k]] = v
    return m


def look_at(target, motion=None, head_share=0.75, neck_share=0.4, eyes=True):
    """turn head (+neck) and eyes toward a target (point name or world xyz, metres).  The head takes `head_share`
    of the yaw/pitch (clamped to limits), the eyes the rest (clamped).  -> (motion, info)."""
    m = np.array(neutral() if motion is None else motion, np.float64)
    tgt = point(target, m)
    base = set_head(m, 0, 0, 0, neck_share)
    eyes_p = point_fast("eyes", base)
    Rb = body_up_R(base)
    # neutral head forward in the body upright frame (the neutral head looks at the camera ~ +z)
    d = Rb.T @ (tgt - eyes_p)
    yaw = math.degrees(math.atan2(d[0], d[2]))
    pitch = math.degrees(math.atan2(d[1], math.hypot(d[0], d[2])))
    hy = float(np.clip(yaw * head_share if abs(yaw) > 8 else yaw * 0.3, *JOINT_LIMITS["head_yaw"]))
    hp = float(np.clip(pitch * head_share if abs(pitch) > 6 else pitch * 0.3, *JOINT_LIMITS["head_pitch"]))
    out = set_head(base, hy, hp, 0.0, neck_share)
    gy, gp = yaw - hy, pitch - hp
    if eyes:
        out = set_gaze(out, gy, gp)
    info = {"yaw_deg": yaw, "pitch_deg": pitch, "head_yaw": hy, "head_pitch": hp, "gaze_yaw": gy, "gaze_pitch": gp,
            "reached": abs(gy) <= JOINT_LIMITS["gaze_yaw"][1] + 0.5 and abs(gp) <= JOINT_LIMITS["gaze_pitch"][1] + 0.5}
    return out, info


# ============================================================================================ hand shapes
_CURL_W = np.array([0.42, 0.35, 0.23])
#             index middle pinky ring thumb   (flexion, rad, sum over 3 joints; thumb = z sum like the retarget)
HAND_SHAPES = {
    "relaxed": None,
    "open": ([0.12, 0.12, 0.15, 0.12], -0.15, 0.6),
    "flat": ([0.05, 0.05, 0.05, 0.05], -0.05, 0.0),
    "spread": ([0.05, 0.05, 0.1, 0.05], -0.25, 1.6),
    "fist": ([3.2, 3.3, 3.1, 3.2], 1.0, 0.3),
    "point": ([0.12, 3.2, 3.0, 3.1], 0.9, 0.3),
    "thumbs_up": ([3.2, 3.3, 3.1, 3.2], -0.4, 0.3),
    "thumbs_down": ([3.2, 3.3, 3.1, 3.2], -0.4, 0.3),
    "peace": ([0.1, 0.12, 3.0, 3.1], 0.9, 1.4),
    "victory": ([0.1, 0.12, 3.0, 3.1], 0.9, 1.4),
    "ok": ([2.1, 0.4, 0.5, 0.45], 0.95, 0.8),
    "cup": ([0.8, 0.8, 0.85, 0.8], 0.25, 0.2),
    "claw": ([2.2, 2.2, 2.1, 2.2], 0.6, 1.0),
    "pinch": ([1.9, 2.6, 2.6, 2.6], 0.9, 0.3),
    "rock": ([0.12, 3.2, 0.15, 3.1], 0.9, 0.8),
    "call_me": ([3.2, 3.3, 0.15, 3.2], -0.4, 1.2),
}
_SPREAD = {"index": -0.18, "middle": 0.0, "ring": 0.10, "pinky": 0.22, "thumb": 0.0}   # left-hand y-rotation per 1.0


def hand_shape(name, side, amount=1.0, base=None):
    """-> (45,) finger axis-angles for one hand: a named shape blended `amount` (0..1) from base (default relaxed)."""
    s = _side(side)
    n = neutral()
    rel = np.asarray(n[_fsl(s)], np.float64).reshape(5, 3, 3)
    base = rel if base is None else np.asarray(base, np.float64).reshape(5, 3, 3)
    name = str(name).lower().replace(" ", "_").replace("-", "_")
    if name not in HAND_SHAPES:
        raise ValueError(f"unknown hand shape {name!r}; one of {sorted(HAND_SHAPES)}")
    spec_ = HAND_SHAPES[name]
    if spec_ is None:
        tgt = rel.copy()
    else:
        curls, thumb, spread = spec_
        sg = -1.0 if s == "l" else 1.0                  # flexion = -z left, +z right
        tgt = rel.copy()
        tgt[:, :, 0] *= 0.5; tgt[:, :, 1] *= 0.5          # tidy the relaxed splay
        for k, fn in enumerate(["index", "middle", "pinky", "ring"]):
            tgt[k, :, 2] = sg * curls[k] * _CURL_W
            tgt[k, 0, 1] = -sg * _SPREAD[fn] * spread    # abduction about the palm normal (y)
        tz = rel[4, :, 2]
        cur = sg * tz.sum()
        tgt[4, :, 2] = tz + sg * (thumb - cur) * np.array([0.3, 0.35, 0.35])
        if thumb < 0:                                     # thumb up / out: less opposition
            tgt[4, 0, 0] = rel[4, 0, 0] * 0.3
    a = float(np.clip(amount, 0, 1))
    out = slerp_aa(base.reshape(15, 3), tgt.reshape(15, 3), a)
    return out.reshape(45)


def set_hand(m, side, name, amount=1.0):
    m = np.array(m, np.float64)
    m[_fsl(_side(side))] = hand_shape(name, side, amount)
    return m


# ============================================================================================ blending
_ROT_TRIPLES = [spec.jcols(j) for j in JOINTS] + [list(range(45 + 3 * k, 48 + 3 * k)) for k in range(30)]
_ROT_COLS = np.array(_ROT_TRIPLES)               # (44,3)
_LIN_COLS = np.array([3, 4, 5] + list(range(135, 189)))


def blend(a, b, w):
    """pose blend: slerp every rotation triple, lerp root shift / face / gaze.  a, b (189,) or (T,189); w scalar or (T,)."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    single = a.ndim == 1 and b.ndim == 1 and np.ndim(w) == 0
    A, B = np.atleast_2d(a), np.atleast_2d(b)
    T = max(len(A), len(B), np.size(w) if np.ndim(w) else 1)
    A = np.broadcast_to(A, (T, D)); B = np.broadcast_to(B, (T, D))
    W = np.broadcast_to(np.asarray(w, np.float64).reshape(-1) if np.ndim(w) else np.full(1, w), (T,))
    out = np.empty((T, D))
    qa = aa_to_quat(A[:, _ROT_COLS]); qb = aa_to_quat(B[:, _ROT_COLS])
    q = _qslerp(qa, qb, W[:, None, None])
    out[:, _ROT_COLS] = quat_to_aa(q)
    out[:, _LIN_COLS] = A[:, _LIN_COLS] + (B[:, _LIN_COLS] - A[:, _LIN_COLS]) * W[:, None]
    # keep the root yaw continuous (spins): lerp root rotation when both are pure yaw
    pa, pb = A[:, 0:3], B[:, 0:3]
    pure = (np.abs(pa[:, [0, 2]]).max(1) < 1e-9) & (np.abs(pb[:, [0, 2]]).max(1) < 1e-9)
    if pure.any():
        out[pure, 0:3] = pa[pure] + (pb[pure] - pa[pure]) * W[pure, None]
    return out[0] if single else out


# ============================================================================================ arm IK
_PALM_DIRS = {"down": (0, -1, 0), "up": (0, 1, 0), "forward": (0, 0, 1), "front": (0, 0, 1), "camera": None,
              "back": (0, 0, -1), "toward_body": (0, 0, -1), "left": (1, 0, 0), "right": (-1, 0, 0)}
_ELBOW_HINTS = {"down": (0.4, -1.0, -0.15), "out": (1.0, -0.5, -0.2), "up": (0.5, 1.0, -0.2),
                "forward": (0.2, -0.6, 1.0), "back": (0.3, -0.7, -1.0), "in": (-0.5, -1.0, 0.2)}


def inward_normal(target, m):
    """unit vector pointing INTO the body at a (named) body-surface target: toward the head centre for face points,
    toward the torso axis for chest/belly/hip points; None for points not on the body."""
    if not isinstance(target, str):
        return None
    n = normalize_point_name(target)
    Rg, P = _arm_chain(m, "l")
    tp = point(n, m)
    if n in FACE_POINTS:
        q = np.asarray(FACE_POINTS[n], np.float64)
        c = np.array([0.0, float(np.clip(q[1], 0.0, 0.14)), 0.0]) if n != "top_of_head" else np.array([0, 0.05, 0.0])
        v = Rg["head"] @ (c - q)
    elif n in ("neck_front", "throat"):
        v = P["neck"] + Rg["neck"] @ np.array([0, 0.03, 0.0]) - tp
    elif n in CHEST_POINTS:
        R, o = Rg["spine3"], P["spine3"]
        q = R.T @ (tp - o); v = R @ (np.array([0.0, q[1], 0.0]) - q)
    elif n in BELLY_POINTS:
        R, o = Rg["spine1"], P["spine1"]
        q = R.T @ (tp - o); v = R @ (np.array([0.0, q[1], 0.0]) - q)
    elif n in ("left_hip", "right_hip", "waist"):
        R, o = Rg["root_rot"], P["root_rot"]
        q = R.T @ (tp - o); v = R @ (np.array([0.0, q[1], 0.0]) - q)
    elif n in ("left_shoulder_top", "right_shoulder_top"):
        v = Rg["spine3"] @ np.array([0, -1.0, 0])
    else:
        return None
    nv = np.linalg.norm(v)
    return v / nv if nv > 1e-9 else None


def _resolve_dir(d, s, Rb, wrist_guess, target, m, target_name=None):
    """palm-facing / finger-direction spec -> world unit vector (or None)."""
    if d is None:
        return None
    if isinstance(d, str):
        d = d.lower()
        sg = 1.0 if s == "l" else -1.0
        if d in ("in", "medial", "inward", "midline"):
            v = Rb @ np.array([-sg, 0, 0])
        elif d in ("out", "outward", "lateral"):
            v = Rb @ np.array([sg, 0, 0])
        elif d in ("camera", "viewer", "you", "away", "outward_front"):
            v = camera_point() - target
        elif d in ("body", "onto", "into", "surface", "target", "self", "face", "toward_face", "toward_body"):
            v = inward_normal(target_name, m) if target_name is not None else None
            if v is None:
                v = (point("nose", m) if d in ("face", "toward_face") else point("chest", m)) - target
        elif d in _PALM_DIRS and _PALM_DIRS[d] is not None:
            v = Rb @ np.asarray(_PALM_DIRS[d], np.float64)
        else:
            v = point(d, m) - target            # face toward a named point
    else:
        v = np.asarray(d, np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def _aa2m(v):
    """fast single axis-angle -> 3x3 (pure float math)."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    th = math.sqrt(x * x + y * y + z * z)
    if th < 1e-12:
        return np.eye(3)
    kx, ky, kz = x / th, y / th, z / th
    c, sn = math.cos(th), math.sin(th)
    C = 1 - c
    return np.array([[c + kx * kx * C, kx * ky * C - kz * sn, kx * kz * C + ky * sn],
                     [ky * kx * C + kz * sn, c + ky * ky * C, ky * kz * C - kx * sn],
                     [kz * kx * C - ky * sn, kz * ky * C + kx * sn, c + kz * kz * C]])


_JCOLS = {j: spec.jcols(j) for j in _BODY_J}
_RESTO = {j: (REST[j][0], np.asarray(REST[j][1], np.float64)) for j in _BODY_J}


def _arm_chain(m, s=None):
    """global rotations / positions of the 14 body joints for ONE pose (fast path)."""
    m = np.asarray(m, np.float64)
    Rg, P = {}, {}
    for j in _BODY_J:
        par, off = _RESTO[j]
        c = _JCOLS[j]
        R = _aa2m(m[c[0]:c[2] + 1])
        if par is None:
            Rg[j], P[j] = R, m[3:6] / 10.0
        else:
            Rg[j] = Rg[par] @ R
            P[j] = P[par] + Rg[par] @ off
    return Rg, P


def point_fast(name, m, chain=None):
    """(3,) world point for one pose without the full fk (joints, face/chest/belly/pelvis points, space targets,
    camera); hand points fall back to fk_full."""
    if not isinstance(name, str):
        return np.asarray(name, np.float64)
    n = normalize_point_name(name)
    Rg, P = chain if chain is not None else _arm_chain(m)
    if n in FACE_POINTS:
        return P["head"] + Rg["head"] @ np.asarray(FACE_POINTS[n])
    if n in CHEST_POINTS:
        return P["spine3"] + Rg["spine3"] @ np.asarray(CHEST_POINTS[n])
    if n in BELLY_POINTS:
        return P["spine1"] + Rg["spine1"] @ np.asarray(BELLY_POINTS[n])
    if n in PELVIS_POINTS:
        return P["root_rot"] + Rg["root_rot"] @ np.asarray(PELVIS_POINTS[n])
    if n in ("camera", "viewer"):
        return camera_point()
    if n in SPACE_DIRS:
        dv = np.asarray(SPACE_DIRS[n], np.float64); dv = dv / np.linalg.norm(dv)
        return P["spine3"] + Rg["spine3"] @ np.asarray(CHEST_POINTS["chest"]) + Rg["root_rot"] @ (dv * 1.5)
    if n in ("pelvis", "spine1", "spine2", "spine3", "neck", "head"):
        return P[n if n != "pelvis" else "root_rot"]
    for sd, s_ in (("left_", "l"), ("right_", "r")):
        if n.startswith(sd) and n[len(sd):] in ("collar", "shoulder", "elbow", "wrist"):
            return P[n[len(sd):] + "_" + s_]
    if n in ("neck_front", "throat"):
        return P["neck"] + Rg["neck"] @ np.array([0, 0.03, 0.06])
    p, _, _ = fk_full(m)
    if n not in p:
        raise KeyError(f"unknown target point {name!r}")
    return p[n]


def ik_arm(side, target, motion=None, part="wrist", palm_facing=None, elbow_hint=None, hand=None, collar=True,
           push_out_body=True, finger_dir=None, iters=8):
    """place `part` of the hand at `target` (point name or world xyz in metres).
    part: wrist | palm | back_of_hand | knuckles | fingertips | index_tip | middle_tip | ring_tip | pinky_tip |
          thumb_tip | index_base ...
    palm_facing: desired palm normal: 'body' (= onto the touched body surface; also 'face'/'onto'), 'camera', 'down',
      'up', 'forward', 'back', 'in' (toward the midline), 'out', 'left', 'right', a point name, or a world vector.
    finger_dir: optional desired finger direction (same vocabulary).  Default: fingers continue the forearm.
    elbow_hint: 'down' (default; 'out' automatically blended in for targets above the shoulder) | 'out' | 'up' |
      'forward' | 'back' | 'in' | world vector (pole).
    hand: shape name or (45,) finger vector set before solving.
    Returns (motion (189,), info {err_cm, reached, limit_hits, target, part}).  Writes collar/shoulder/elbow/wrist
    (+ fingers if `hand`)."""
    s = _side(side)
    sg = 1.0 if s == "l" else -1.0
    m = np.array(neutral() if motion is None else motion, np.float64)
    if hand is not None:
        m[_fsl(s)] = hand_shape(hand, s) if isinstance(hand, str) else np.asarray(hand, np.float64)
    tname = target if isinstance(target, str) else None
    tgt = point(target, m).astype(np.float64)
    hits = []
    hp = hand_points_local(m[_fsl(s)], s)
    pkey = part if part in hp else {"palm_center": "palm", "index": "index_tip", "finger": "index_tip",
                                    "fingertip": "index_tip", "hand": "palm", "middle": "middle_tip",
                                    "thumb": "thumb_tip", "back": "back_of_hand", "fist": "knuckles"}.get(part)
    if pkey is None:
        raise ValueError(f"unknown hand part {part!r}")
    if push_out_body and pkey in ("wrist", "knuckles", "back_of_hand"):
        t2 = push_out(tgt, m, margin=0.035)
        if np.linalg.norm(t2 - tgt) > 1e-4:
            hits.append(f"{pkey} target was inside the body; pushed out {np.linalg.norm(t2 - tgt) * 100:.1f} cm")
            tgt = t2
    off_local = hp[pkey][0]
    n = neutral()
    Rg, P = _arm_chain(m, s)
    Rb = body_up_R(m)
    if collar:
        rise = float((Rb.T @ (tgt - P["shoulder_" + s]))[1])
        lift = float(np.clip((rise + 0.05) / 0.35, 0, 1)) * 12.0
        Rcn = aa_to_mat(jget(n, "collar_" + s))
        jset(m, "collar_" + s, mat_to_aa(rot("z", sg * lift) @ Rcn))
        Rg, P = _arm_chain(m, s)
    Rc, sh = Rg["collar_" + s], P["shoulder_" + s]
    Rchest = Rg["spine3"]
    u0 = np.asarray(REST["elbow_" + s][1]); L1 = np.linalg.norm(u0); u0 = u0 / L1
    w0 = np.asarray(REST["wrist_" + s][1]); L2 = np.linalg.norm(w0); w0 = w0 / L2
    if elbow_hint is None:
        pole_c = np.array(_ELBOW_HINTS["down"]) * np.array([sg, 1, 1])
        rel = Rchest.T @ (tgt - sh)
        hd = Rchest.T @ (tgt - P["head"])
        near_face = np.linalg.norm(hd - np.array([0, 0.06, 0.05])) < 0.2
        if near_face:           # hand at the face: elbow down, a little out and forward (cover / touch / think)
            pole_c = np.array([0.45 * sg, -1.0, 0.25])
        elif rel[1] > -0.05:    # hand near/above the shoulder: the elbow swings out to the side
            pole_c = pole_c + np.array([sg * 1.0, 0.5, -0.1]) * min(1.0, (rel[1] + 0.05) / 0.2)
        pole = Rchest @ pole_c
    elif isinstance(elbow_hint, str):
        pole = Rchest @ (np.array(_ELBOW_HINTS[elbow_hint]) * np.array([sg, 1, 1]))
    else:
        pole = np.asarray(elbow_hint, np.float64)
    Rn_wr = aa_to_mat(jget(n, "wrist_" + s))
    pf_fixed = _resolve_dir(palm_facing, s, Rb, tgt, tgt, m, tname) if palm_facing is not None else None
    fd_fixed = _resolve_dir(finger_dir, s, Rb, tgt, tgt, m, tname) if finger_dir is not None else None
    lim_tw = [math.radians(v) for v in JOINT_LIMITS["elbow_twist"]]
    lim_w = math.radians(JOINT_LIMITS["wrist_swing"])

    def solve(wt):
        flags = []
        dvec = wt - sh
        Ld = np.linalg.norm(dvec)
        reach = float(np.clip(Ld, 0.06, (L1 + L2) * 0.9995))
        dd = dvec / max(Ld, 1e-9)
        cos_a = np.clip((L1 ** 2 + reach ** 2 - L2 ** 2) / (2 * L1 * reach), -1, 1)
        a = math.acos(cos_a)
        pn = pole - dd * (dd @ pole)
        if np.linalg.norm(pn) < 1e-6:
            alt = Rchest @ np.array([0, -1.0, 0])
            pn = alt - dd * (dd @ alt)
        pn = pn / np.linalg.norm(pn)
        u = dd * math.cos(a) + pn * math.sin(a)
        elbow = sh + u * L1
        wpos = sh + dd * reach
        f = wpos - elbow; f = f / np.linalg.norm(f)
        fperp = f - u * (u @ f)
        if np.linalg.norm(fperp) < 1e-6:
            fperp = pn - u * (u @ pn)
        fperp = fperp / np.linalg.norm(fperp)
        h = np.cross(fperp, u) if s == "l" else np.cross(u, fperp)
        Rs_w = _frame_from(u0, np.array([0, 1.0, 0]), u, h)
        Re_l = rot_between(w0, Rs_w.T @ f)
        Re_w0 = Rs_w @ Re_l
        cur = Re_w0 @ Rn_wr
        tw = 0.0
        if pf_fixed is None and fd_fixed is None:
            Rhand_w = cur
        else:
            pf = pf_fixed if pf_fixed is not None else cur @ np.array([0, -1.0, 0])
            if fd_fixed is not None:
                fd = fd_fixed
            else:
                fd = f - pf * (pf @ f)
                if np.linalg.norm(fd) < 1e-6:
                    fd = cur @ _mir([1.0, 0, 0], s)
            fd = fd / np.linalg.norm(fd)
            pf_o = pf - fd * (pf @ fd)
            if np.linalg.norm(pf_o) < 1e-6:
                pf_o = cur @ np.array([0, -1.0, 0])
            Rhand_w = _frame_from(_mir([1.0, 0, 0], s), np.array([0, -1.0, 0]), fd, pf_o)
            # the forearm twist goes into the elbow (pronation / supination), the rest into the wrist
            _, tw = _swing_twist(Re_w0.T @ Rhand_w @ Rn_wr.T, w0)
            twc = float(np.clip(tw, *lim_tw))
            if abs(twc - tw) > 1e-3:
                flags.append(f"{_long(s)} forearm twist {math.degrees(tw):.0f} deg clamped")
            tw = twc
        Re_l2 = Re_l @ _aa2m(w0 * tw)
        Re_w = Rs_w @ Re_l2
        Rw_l = Re_w.T @ Rhand_w
        aw = mat_to_aa(Rw_l @ Rn_wr.T)
        ang = float(np.linalg.norm(aw))
        if ang > lim_w:
            flags.append(f"{_long(s)} wrist bend {math.degrees(ang):.0f} deg clamped to {JOINT_LIMITS['wrist_swing']}")
            Rw_l = _aa2m(aw / ang * lim_w) @ Rn_wr
        got = wpos + Re_w @ Rw_l @ off_local
        return (Rs_w, Re_l2, Rw_l), got, flags

    wt = tgt.copy()
    sol, got, flags = solve(wt)
    for it in range(iters):                       # fixed point on the wrist target
        err = tgt - got
        if np.linalg.norm(err) < 3e-4:
            break
        wt = wt + err
        sol, got, flags = solve(wt)
    if np.linalg.norm(tgt - got) > 3e-4 and np.linalg.norm(tgt - sh) < L1 + L2 + 0.2:   # finite-difference Newton (orientation depends on the wrist target)
        for it in range(8):
            J = np.zeros((3, 3))
            for k in range(3):
                e = np.zeros(3); e[k] = 1e-3
                J[:, k] = (solve(wt + e)[1] - got) / 1e-3
            step = np.linalg.lstsq(J, tgt - got, rcond=1e-3)[0]
            nrm = np.linalg.norm(step)
            if nrm > 0.15:
                step *= 0.15 / nrm
            sol2, got2, fl2 = solve(wt + step)
            if np.linalg.norm(tgt - got2) < np.linalg.norm(tgt - got):
                wt, sol, got, flags = wt + step, sol2, got2, fl2
            else:
                break
            if np.linalg.norm(tgt - got) < 3e-4:
                break
    Rs_w, Re_l2, Rw_l = sol
    hits += flags
    jset(m, "shoulder_" + s, mat_to_aa(Rc.T @ Rs_w))
    jset(m, "elbow_" + s, mat_to_aa(Re_l2))
    jset(m, "wrist_" + s, mat_to_aa(Rw_l))
    Rg2, P2 = _arm_chain(m, s)
    got = P2["wrist_" + s] + Rg2["wrist_" + s] @ off_local
    err_cm = float(np.linalg.norm(got - tgt) * 100)
    reached = err_cm < 1.5
    if not reached:
        far = (np.linalg.norm(tgt - sh) - (L1 + L2)) * 100
        hits.append(f"{_long(s)} hand {pkey} misses target by {err_cm:.1f} cm" +
                    (f" (target {far:.0f} cm beyond arm reach)" if far > 0 else ""))
    hits += check_limits(m, sides=(s,), head=False)
    return m, {"err_cm": err_cm, "reached": reached, "limit_hits": hits, "target": tgt, "part": pkey}


_HARNESS_NAMES = {"eye_l": "left_eye", "eye_r": "right_eye", "ear_l": "left_ear", "ear_r": "right_ear",
                  "cheek_l": "left_cheek", "cheek_r": "right_cheek", "top_head": "top_of_head", "back_head": "back_of_head",
                  "chest_l": "left_chest", "chest_r": "right_chest", "hip_l": "left_hip", "hip_r": "right_hip",
                  "hips_front": "waist", "nose": "nose", "mouth": "mouth", "chin": "chin", "forehead": "forehead",
                  "chest": "chest", "throat": "throat", "belly": "belly"}


def fk_points(motion, world=True):
    """{name: (T,3)} points only (no 'vec'); world=False zeroes root_rot/root_shift (pelvis frame)."""
    m = np.array(motion, np.float64)
    if not world:
        m[..., 0:6] = 0
    p, _, _ = fk_full(m)
    return p


def harness_points(motion, world=True):
    """the harness's (h_kin.py) target-point names -> this module's points, so scoring and approaches agree."""
    m = np.array(motion, np.float64)
    if m.ndim == 1:
        m = m[None]
    if not world:
        m[:, 0:6] = 0
    p, Rg, _ = fk_full(m)
    out = {h: p[k] for h, k in _HARNESS_NAMES.items()}
    out["head_c"] = p["head"] + np.einsum("tij,j->ti", Rg["head"], np.asarray(HEAD_ELLIPSOID[0]))
    out["neck_back"] = p["neck"] + np.einsum("tij,j->ti", Rg["neck"], np.array([0.0, 0.04, -0.06]))
    return out


def describe_pose(m):
    """compact numbers for logs: head ypr, wrist positions (cm, body frame), blinks."""
    p = fk(m)
    y, pi, r = head_ypr(m)
    return {"head_ypr": (round(y, 1), round(pi, 1), round(r, 1)),
            "lw_cm": np.round(p["left_wrist"] * 100, 1).tolist(), "rw_cm": np.round(p["right_wrist"] * 100, 1).tolist()}
