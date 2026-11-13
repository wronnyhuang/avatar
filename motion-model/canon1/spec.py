"""CANON v1 — the unified human representation for the avatar's learned motion model.

ONE definition shared by every extractor (data), the model, the retarget layer, the eval harness and the live
planner. Append-only: never re-order or re-purpose a column; new helpers are fine.

Per person, per frame, at FPS = 30:
  motion[189] float32   joint angles on one canonical skeleton + 52 face blendshapes + 2 gaze angles (MOTION_COLS)
  mask[9]     uint8     validity per channel GROUP (GROUPS / GROUP_NAMES); 1 = measured, 0 = unknown (never trained on)
  audio[42]   float32   that person's OWN voice: 40 causal log-mel + loud_db + vad (identical to the react_v1 spec)

Skeleton = the SMPL-H kinematic tree (upper body only), so Seamless SMPL-H data is native. Rotations are LOCAL
axis-angle vectors (radians) of each joint relative to its parent, in SMPL's frame: x = the person's LEFT,
y = UP, z = FORWARD (out of the chest). The rest pose is SMPL's T-pose (arms straight out sideways), so arms
hanging down = shoulder rotations of roughly +-1.2 rad about z. That is just the convention; nothing assumes small angles.

  root_rot   pelvis orientation RELATIVE to the session's median pelvis orientation (lean / turn), axis-angle
  root_shift pelvis translation minus the session median, in DECIMETRES (x left, y up, z forward)
  spine1..3, neck, head, collar_l/r, shoulder_l/r, elbow_l/r, wrist_l/r   local axis-angle, SMPL-H joints
             3, 6, 9, 12, 15, 13/14, 16/17, 18/19, 20/21
  fingers_l/r  15 MANO joints x 3 (index1-3, middle1-3, pinky1-3, ring1-3, thumb1-3), local axis-angle, as
             SMPL-H left_hand_pose / right_hand_pose (use_pca=False, flat_hand_mean=False convention of Seamless)
  bs_*       52 MediaPipe Face Landmarker blendshape scores 0..1 (ARKit names, MediaPipe order incl. _neutral)
  gaze_yaw, gaze_pitch   eye direction relative to the HEAD, radians; yaw + = toward the person's left, pitch + = up

Live-side convention: the user's camera is UNMIRRORED (the raw image, as a partner would see it). Whatever
the source, "left" always means the PERSON's anatomical left.
"""
from __future__ import annotations

import json
import math

import numpy as np

from .causal_audio import (_mel_fb, frame_end_sample, audio_features, energy_vad,  # noqa: F401
                           resample_to_16k)

FPS = 30
SR = 16000
VERSION = "canon1"

# --------------------------------------------------------------------------------------------- skeleton
JOINTS = ["root_rot", "spine1", "spine2", "spine3", "neck", "head", "collar_l", "collar_r", "shoulder_l",
          "shoulder_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r"]
SMPLH_JOINT_ID = {"root_rot": 0, "spine1": 3, "spine2": 6, "spine3": 9, "neck": 12, "head": 15, "collar_l": 13,
                  "collar_r": 14, "shoulder_l": 16, "shoulder_r": 17, "elbow_l": 18, "elbow_r": 19, "wrist_l": 20,
                  "wrist_r": 21}
# body_pose[T,21,3] in Seamless/SMPL-H holds joints 1..21 -> body_pose index = joint id - 1
FINGER_JOINTS = ["index1", "index2", "index3", "middle1", "middle2", "middle3", "pinky1", "pinky2", "pinky3",
                 "ring1", "ring2", "ring3", "thumb1", "thumb2", "thumb3"]

ROOT_COLS = ["root_rot_x", "root_rot_y", "root_rot_z", "root_shift_x", "root_shift_y", "root_shift_z"]
BODY_COLS = [f"{j}_{a}" for j in JOINTS[1:] for a in "xyz"]                    # 13 joints x 3 = 39
FINGER_COLS = [f"f{s}_{j}_{a}" for s in "lr" for j in FINGER_JOINTS for a in "xyz"]   # 90
BS_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft",
    "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]
assert len(BS_NAMES) == 52
BS = {n: i for i, n in enumerate(BS_NAMES)}
FACE_COLS = [f"bs_{n}" for n in BS_NAMES]
GAZE_COLS = ["gaze_yaw", "gaze_pitch"]

MOTION_COLS = ROOT_COLS + BODY_COLS + FINGER_COLS + FACE_COLS + GAZE_COLS
D_MOTION = len(MOTION_COLS)
assert D_MOTION == 189, D_MOTION
COL = {c: i for i, c in enumerate(MOTION_COLS)}

SL_ROOT_ROT = slice(0, 3)
SL_ROOT_SHIFT = slice(3, 6)
SL_BODY = slice(6, 45)            # spine1 .. wrist_r
SL_FINGERS_L = slice(45, 90)
SL_FINGERS_R = slice(90, 135)
SL_FACE = slice(135, 187)
SL_GAZE = slice(187, 189)


def jcols(j):
    """column indices of joint j (a JOINTS name; root_rot = the 3 root rotation cols)."""
    if j == "root_rot":
        return [0, 1, 2]
    i = 6 + 3 * (JOINTS.index(j) - 1)
    return [i, i + 1, i + 2]


# --------------------------------------------------------------------------------------------- groups + masks
_face_mouth = [135 + i for i, n in enumerate(BS_NAMES) if n.startswith(("mouth", "jaw")) or n == "cheekPuff"]
_face_look = [135 + i for i, n in enumerate(BS_NAMES) if n.startswith("eyeLook")]
_face_upper = [135 + i for i in range(52) if (135 + i) not in _face_mouth and (135 + i) not in _face_look]
GROUPS = {
    "torso": list(range(0, 6)) + jcols("spine1") + jcols("spine2") + jcols("spine3"),
    "head": jcols("neck") + jcols("head"),
    "arm_l": jcols("collar_l") + jcols("shoulder_l") + jcols("elbow_l") + jcols("wrist_l"),
    "arm_r": jcols("collar_r") + jcols("shoulder_r") + jcols("elbow_r") + jcols("wrist_r"),
    "fingers_l": list(range(45, 90)),
    "fingers_r": list(range(90, 135)),
    "face_upper": _face_upper,          # brows, blinks, squint, wide, cheekSquint, noseSneer, _neutral
    "mouth": _face_mouth,               # mouth*, jaw*, cheekPuff  (lip sync owns these while the avatar talks)
    "gaze": _face_look + [187, 188],    # eyeLook* blendshapes + gaze_yaw/pitch
}
GROUP_NAMES = list(GROUPS)
D_MASK = len(GROUP_NAMES)
assert D_MASK == 9
_all = sorted(c for g in GROUPS.values() for c in g)
assert _all == list(range(D_MOTION)), "groups must partition the columns"
GROUP_OF_COL = np.zeros(D_MOTION, np.int64)
for gi, g in enumerate(GROUP_NAMES):
    GROUP_OF_COL[GROUPS[g]] = gi
BODY_GROUPS = ["torso", "head", "arm_l", "arm_r", "fingers_l", "fingers_r"]   # what a commanded action may own


def col_mask(mask9):
    """(...,9) group mask -> (...,189) per-column mask."""
    m = np.asarray(mask9)
    return m[..., GROUP_OF_COL]


# --------------------------------------------------------------------------------------------- audio
AUDIO_COLS = [f"mel{i:02d}" for i in range(40)] + ["loud_db", "vad"]
D_AUDIO = len(AUDIO_COLS)
assert D_AUDIO == 42


# --------------------------------------------------------------------------------------------- mirror (L<->R)
def _mirror_tables():
    perm = np.arange(D_MOTION)
    sign = np.ones(D_MOTION, np.float32)
    # rotations: reflection x -> -x maps axis-angle (x,y,z) -> (x,-y,-z); swap _l/_r joints
    for j in JOINTS:
        c = jcols(j)
        if j.endswith("_l") or j.endswith("_r"):
            o = j[:-2] + ("_r" if j.endswith("_l") else "_l")
            perm[c] = jcols(o)
        sign[c[1]] = -1; sign[c[2]] = -1
    sign[3] = -1                                   # root_shift_x
    for k in range(45):                            # fingers: swap hands, (x,-y,-z)
        perm[45 + k], perm[90 + k] = 90 + k, 45 + k
        if k % 3 in (1, 2):
            sign[45 + k] = -1; sign[90 + k] = -1
    for i, n in enumerate(BS_NAMES):
        for a, b in (("Left", "Right"), ("Right", "Left")):
            if n.endswith(a):
                perm[135 + i] = 135 + BS[n[: -len(a)] + b]
    # eyeLookIn/Out naming is per eye; swapping Left<->Right already mirrors gaze correctly
    sign[187] = -1                                 # gaze_yaw
    return perm, sign


MIRROR_PERM, MIRROR_SIGN = _mirror_tables()
GROUP_MIRROR = np.array([GROUP_NAMES.index({"arm_l": "arm_r", "arm_r": "arm_l", "fingers_l": "fingers_r",
                                            "fingers_r": "fingers_l"}.get(g, g)) for g in GROUP_NAMES])


def mirror_motion(m):
    """(...,189) -> mirrored person (left/right swapped). out[..., i] = m[..., perm[i]] * sign[i]."""
    m = np.asarray(m)
    return m[..., MIRROR_PERM] * MIRROR_SIGN


def mirror_mask(k):
    return np.asarray(k)[..., GROUP_MIRROR]


# --------------------------------------------------------------------------------------------- rotation helpers
def aa_to_mat(aa):
    aa = np.asarray(aa, np.float64)
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(th, 1e-12)
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = k[..., 2], -k[..., 1], k[..., 0]
    th = th[..., None]
    I = np.broadcast_to(np.eye(3), K.shape)
    return I + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def mat_to_aa(R):
    R = np.asarray(R, np.float64)
    cos = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    th = np.arccos(cos)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(th)
    small = s < 1e-6
    k = v / np.where(small, 1.0, 2 * s)[..., None]
    out = k * th[..., None]
    # near 0: aa ~ v/2 ; near pi: use the diagonal
    out = np.where(small[..., None] & (th[..., None] < 1), v / 2, out)
    near_pi = small & (th >= 1)
    if np.any(near_pi):
        Rp = R[near_pi]
        d = np.sqrt(np.maximum((np.diagonal(Rp, axis1=-2, axis2=-1) + 1) / 2, 0))
        sgn = np.sign(np.stack([np.ones(len(Rp)), Rp[:, 0, 1] + Rp[:, 1, 0], Rp[:, 0, 2] + Rp[:, 2, 0]], -1))
        sgn[sgn == 0] = 1
        out[near_pi] = d * sgn * math.pi
    return out


def relative_root(global_orient_aa, valid=None):
    """(T,3) SMPL global_orient -> (T,3) rotation relative to the session's chordal-median orientation."""
    R = aa_to_mat(global_orient_aa)
    v = np.ones(len(R), bool) if valid is None else np.asarray(valid, bool)
    M = np.median(R[v] if v.any() else R, 0)
    U, _, Vt = np.linalg.svd(M)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1; Rm = U @ Vt
    return mat_to_aa(Rm.T[None] @ R)


# --------------------------------------------------------------------------------------------- file format
# One .npz per session (conversation / clip):
#   fps int32 (=30), motion float16 (P,T,189), mask uint8 (P,T,9), audio float16 (P,T,42),
#   meta json string, words json string (per person [[t0,t1,word],...]), optional x_* extra arrays.
# meta REQUIRED keys: version ("canon1"), dataset, session, persons [ids, globally unique e.g. "seamless:P0059"],
#   split ("train"|"val"|"test"), source per person (e.g. "smplh+mapper", "mediapipe+fit", "flame-head",
#   "panoptic-3d"), quality per person (0..1 weight), speaking_track ("vad"|"transcript"|"energy").
# Partner convention: for P == 2 the partner of person p is 1-p. P == 1 clips may carry the partner's audio as
#   x_partner_audio (1,T,42) (e.g. an off-screen interviewer); a missing partner = all-zero mask + zero audio.
# P == 3 (Panoptic haggling) may carry x_partner_index (P,T) int8 = whom each person is attending to (-1 none).
REQUIRED_META = ("version", "dataset", "session", "persons", "split", "source", "quality")


def save_session(path, motion, mask, audio, meta, words=None, extra=None):
    motion = np.asarray(motion, np.float32); mask = np.asarray(mask, np.uint8); audio = np.asarray(audio, np.float32)
    P, T, D = motion.shape
    assert D == D_MOTION and mask.shape == (P, T, D_MASK) and audio.shape == (P, T, D_AUDIO), \
        (motion.shape, mask.shape, audio.shape)
    assert np.isfinite(motion).all() and np.isfinite(audio).all()
    meta = dict(meta); meta.setdefault("version", VERSION)
    for k in REQUIRED_META:
        assert k in meta, k
    kw = dict(fps=np.int32(FPS), motion=motion.astype(np.float16), mask=mask, audio=audio.astype(np.float16),
              meta=np.array(json.dumps(meta)), words=np.array(json.dumps(words or [[] for _ in range(P)])))
    for k, v in (extra or {}).items():
        kw["x_" + k] = v
    np.savez_compressed(path, **kw)


def load_session(path):
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files}
    out["meta"] = json.loads(str(out["meta"]))
    out["words"] = json.loads(str(out["words"]))
    out["motion"] = out["motion"].astype(np.float32)
    out["audio"] = out["audio"].astype(np.float32)
    return out
