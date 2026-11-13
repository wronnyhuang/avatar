"""The canonical upper-body skeleton of CANON v1 (SMPL-H kinematic tree) and vectorised forward kinematics.

No SMPL-H model files are shipped (licence), so the rest-pose joint OFFSETS are the SMPL neutral template's
joint positions (J_regressor on v_template, metres, rounded) — bone directions and proportions are what matter here:
the retarget only uses NORMALISED quantities (wrist offset / arm length, directions, rotations).

Frame (spec.py): x = the person's LEFT, y = UP, z = FORWARD. Rest = T-pose (arms straight out sideways).
Local rotations are axis-angle relative to the parent (canon columns, spec.jcols).

    fk = upper_fk(motion)          # (T,189) -> dict of global rotations (T,3,3) and positions (T,3), pelvis at 0
"""
from __future__ import annotations

import numpy as np

from canon1 import spec

# Rest offsets (parent -> joint, metres, T-pose): the SAME numbers as canon1/skeleton.py (SMPL neutral
# mean-shape joint offsets, rounded), so the retarget's FK matches the fitter's IK exactly.
PARENT = {"spine1": "pelvis", "spine2": "spine1", "spine3": "spine2", "neck": "spine3", "head": "neck",
          "collar_l": "spine3", "collar_r": "spine3", "shoulder_l": "collar_l", "shoulder_r": "collar_r",
          "elbow_l": "shoulder_l", "elbow_r": "shoulder_r", "wrist_l": "elbow_l", "wrist_r": "elbow_r"}
_OFF = {
    "spine1": (-0.0025, 0.1088, -0.0267), "spine2": (0.0055, 0.1352, 0.0011), "spine3": (0.0015, 0.0529, 0.0254),
    "neck": (-0.0028, 0.2139, -0.0430), "head": (0.0052, 0.0650, 0.0513),
    "collar_l": (0.0788, 0.1217, -0.0341), "collar_r": (-0.0818, 0.1188, -0.0386),
    "shoulder_l": (0.0910, 0.0305, -0.0089), "shoulder_r": (-0.0960, 0.0326, -0.0091),
    "elbow_l": (0.2596, -0.0128, -0.0275), "elbow_r": (-0.2537, -0.0133, -0.0214),
    "wrist_l": (0.2492, 0.0090, -0.0012), "wrist_r": (-0.2553, 0.0078, -0.0056),
}
ORDER = ["spine1", "spine2", "spine3", "neck", "head", "collar_l", "collar_r", "shoulder_l", "shoulder_r",
         "elbow_l", "elbow_r", "wrist_l", "wrist_r"]
OFFSET = {j: np.asarray(_OFF[j], np.float64) for j in ORDER}
UPPER = float(np.linalg.norm(OFFSET["elbow_l"]))          # shoulder -> elbow  (~0.262 m)
FORE = float(np.linalg.norm(OFFSET["wrist_l"]))           # elbow -> wrist     (~0.263 m)
ARM = UPPER + FORE
HAND_LEN = 0.09                                           # wrist -> middle of the palm/fingers (for palm frames)


def aa_to_mat(aa):
    return spec.aa_to_mat(aa)


def mat_to_aa(R):
    return spec.mat_to_aa(R)


def joint_rot(motion, j):
    """(T,189) -> (T,3,3) local rotation of joint j (spec.JOINTS name)."""
    return aa_to_mat(np.asarray(motion, np.float64)[..., spec.jcols(j)])


_ALLJ = ["root_rot"] + ORDER
_ALLC = np.array([spec.jcols(j) for j in _ALLJ])          # (14,3) column indices


def local_rots(motion):
    """(T,189) -> {joint (incl. root_rot): (T,3,3)} local rotations, ONE batched Rodrigues call."""
    m = np.asarray(motion, np.float64)
    if m.ndim == 1:
        m = m[None]
    R = aa_to_mat(m[:, _ALLC])                            # (T,14,3,3)
    return {j: R[:, i] for i, j in enumerate(_ALLJ)}


def upper_fk(motion, root=True, rots=None):
    """(T,189) canon -> {joint: (R_global (T,3,3), p (T,3))}, pelvis at the origin; root=False ignores root_rot
    (pelvis = identity, i.e. the chest relative to the pelvis)."""
    m = np.asarray(motion, np.float64)
    if m.ndim == 1:
        m = m[None]
    T = len(m)
    L = local_rots(m) if rots is None else rots
    R0 = L["root_rot"] if root else np.broadcast_to(np.eye(3), (T, 3, 3))
    out = {"pelvis": (R0, np.zeros((T, 3)))}
    for j in ORDER:
        Rp, pp = out[PARENT[j]]
        p = pp + Rp @ OFFSET[j]
        out[j] = (Rp @ L[j], p)
    return out


def chest_frame_arm(motion, rots=None):
    """Wrist / elbow positions RELATIVE TO THE REST SHOULDER POSITION, expressed in the CHEST (spine3) frame, plus the
    wrist's rotation in that frame.  Independent of root / spine (the rig turns the upper body separately).
    -> dict side ('l','r') -> {"wrist": (T,3), "elbow": (T,3), "R_wrist": (T,3,3), "R_elbow": (T,3,3),
                                "shoulder": (T,3) the (collar-moved) shoulder joint}"""
    m = np.asarray(motion, np.float64)
    if m.ndim == 1:
        m = m[None]
    L = local_rots(m) if rots is None else rots
    out = {}
    for s in "lr":
        Rc = L["collar_" + s]
        p_sh = Rc @ OFFSET["shoulder_" + s]                                  # collar joint at 0
        Rs = Rc @ L["shoulder_" + s]
        p_el = p_sh + Rs @ OFFSET["elbow_" + s]
        Re = Rs @ L["elbow_" + s]
        p_wr = p_el + Re @ OFFSET["wrist_" + s]
        Rw = Re @ L["wrist_" + s]
        rest_sh = OFFSET["shoulder_" + s]
        out[s] = {"wrist": p_wr - rest_sh, "elbow": p_el - rest_sh, "shoulder": p_sh - rest_sh,
                  "R_wrist": Rw, "R_elbow": Re, "R_shoulder": Rs, "R_collar": Rc}
    return out


def two_bone_ik(target, side, pole=None, upper=UPPER, fore=FORE):
    """Inverse of the arm chain in the chest frame (collar at identity): wrist target relative to the rest shoulder
    -> (shoulder aa (T,3), elbow aa (T,3)) that put the wrist there with the elbow toward `pole` (default: down and
    back).  Used for the compositor's inverse map (rig hand target -> canon arm angles)."""
    t = np.atleast_2d(np.asarray(target, np.float64))
    T = len(t)
    sgn = 1.0 if side == "l" else -1.0
    L = np.linalg.norm(t, axis=1, keepdims=True)
    reach = np.clip(L, 0.05, (upper + fore) * 0.999)
    d = t / np.maximum(L, 1e-9)
    pole = np.array([0.15 * sgn, -1.0, -0.35]) if pole is None else np.asarray(pole, np.float64)
    # elbow angle from the law of cosines
    cos_e = np.clip((upper ** 2 + fore ** 2 - reach[:, 0] ** 2) / (2 * upper * fore), -1, 1)
    bend = np.pi - np.arccos(cos_e)                                     # 0 = straight
    cos_a = np.clip((upper ** 2 + reach[:, 0] ** 2 - fore ** 2) / (2 * upper * reach[:, 0]), -1, 1)
    a = np.arccos(cos_a)                                                # angle at the shoulder between d and the upper arm
    pn = pole[None] - d * (d @ pole)[:, None]
    pn /= np.maximum(np.linalg.norm(pn, axis=1, keepdims=True), 1e-9)
    u = d * np.cos(a)[:, None] + pn * np.sin(a)[:, None]                 # upper-arm direction
    elbow = u * upper
    f = (t - elbow)
    f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
    # shoulder rotation: rest upper-arm direction (+-x) -> u, with the elbow bending plane carried along
    x0 = np.array([sgn, 0.0, 0.0])
    Rs = _rot_between(np.broadcast_to(x0, (T, 3)), u)
    # elbow: in the shoulder frame, rotate the rest forearm direction (x0) onto Rs^T f, about the local bend axis
    fl = np.einsum("tji,tj->ti", Rs, f)
    Re = _rot_between(np.broadcast_to(x0, (T, 3)), fl)
    return mat_to_aa(Rs), mat_to_aa(Re), bend


def _rot_between(a, b):
    """minimal rotation taking unit vectors a -> b, batched (T,3) -> (T,3,3)."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    v = np.cross(a, b)
    c = np.sum(a * b, -1)
    s = np.linalg.norm(v, axis=-1)
    k = v / np.maximum(s, 1e-12)[..., None]
    th = np.arctan2(s, c)
    aa = k * th[..., None]
    # antiparallel: any axis perpendicular to a
    anti = (s < 1e-9) & (c < 0)
    if np.any(anti):
        perp = np.cross(a[anti], np.array([0.0, 1.0, 0.0]))
        bad = np.linalg.norm(perp, axis=-1) < 1e-6
        perp[bad] = np.cross(a[anti][bad], np.array([0.0, 0.0, 1.0]))
        perp /= np.linalg.norm(perp, axis=-1, keepdims=True)
        aa[anti] = perp * np.pi
    return aa_to_mat(aa)


def euler_yxz(R):
    """R = Ry(yaw) Rx(pitch) Rz(roll) (the renderer's head order: rotY*rotX*rotZ) -> (yaw, pitch, roll), batched."""
    R = np.asarray(R, np.float64)
    yaw = np.arctan2(R[..., 0, 2], R[..., 2, 2])
    pitch = np.arcsin(np.clip(-R[..., 1, 2], -1, 1))
    roll = np.arctan2(R[..., 1, 0], R[..., 1, 1])
    return yaw, pitch, roll


def mat_yxz(yaw, pitch, roll):
    yaw, pitch, roll = (np.asarray(v, np.float64) for v in (yaw, pitch, roll))
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    sh = np.broadcast(cy, cp, cr).shape
    Ry = np.zeros(sh + (3, 3)); Rx = np.zeros(sh + (3, 3)); Rz = np.zeros(sh + (3, 3))
    Ry[..., 0, 0] = cy; Ry[..., 0, 2] = sy; Ry[..., 1, 1] = 1; Ry[..., 2, 0] = -sy; Ry[..., 2, 2] = cy
    Rx[..., 0, 0] = 1; Rx[..., 1, 1] = cp; Rx[..., 1, 2] = -sp; Rx[..., 2, 1] = sp; Rx[..., 2, 2] = cp
    Rz[..., 0, 0] = cr; Rz[..., 0, 1] = -sr; Rz[..., 1, 0] = sr; Rz[..., 1, 1] = cr; Rz[..., 2, 2] = 1
    return Ry @ Rx @ Rz
