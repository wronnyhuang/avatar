"""Rough SMPL(-H) neutral rest skeleton (upper body) + forward kinematics for CANON v1.

Offsets are the SMPL neutral mean-shape joint offsets (metres, rest T-pose, x = person's LEFT, y = UP, z = FORWARD),
rounded; good to ~1 cm, which is plenty for IK targets, visual checks and frame-convention tests. Not a body model.
"""
from __future__ import annotations

import numpy as np

from .spec import JOINTS, aa_to_mat, jcols

# joint -> (parent joint name, offset from parent in the parent's rest frame)
REST = {
    "root_rot": (None, (0.0, 0.0, 0.0)),               # pelvis
    "spine1": ("root_rot", (-0.0025, 0.1088, -0.0267)),
    "spine2": ("spine1", (0.0055, 0.1352, 0.0011)),
    "spine3": ("spine2", (0.0015, 0.0529, 0.0254)),
    "neck": ("spine3", (-0.0028, 0.2139, -0.0430)),
    "head": ("neck", (0.0052, 0.0650, 0.0513)),
    "collar_l": ("spine3", (0.0788, 0.1217, -0.0341)),
    "collar_r": ("spine3", (-0.0818, 0.1188, -0.0386)),
    "shoulder_l": ("collar_l", (0.0910, 0.0305, -0.0089)),
    "shoulder_r": ("collar_r", (-0.0960, 0.0326, -0.0091)),
    "elbow_l": ("shoulder_l", (0.2596, -0.0128, -0.0275)),
    "elbow_r": ("shoulder_r", (-0.2537, -0.0133, -0.0214)),
    "wrist_l": ("elbow_l", (0.2492, 0.0090, -0.0012)),
    "wrist_r": ("elbow_r", (-0.2553, 0.0078, -0.0056)),
}
# end effectors (child offsets in the joint's rest frame): head top-ish, middle-finger knuckle
ENDS = {"head_end": ("head", (0.0, 0.10, 0.02)), "hand_l": ("wrist_l", (0.085, -0.005, 0.0)),
        "hand_r": ("wrist_r", (-0.085, -0.005, 0.0)), "nose": ("head", (0.0, 0.03, 0.11))}
ORDER = JOINTS  # parents always precede children in JOINTS


def fk(motion, with_root=True, extra=True):
    """motion (T,189) -> dict name -> (T,3) joint positions in the pelvis frame (with_root=False) or in the
    session-median pelvis frame (with_root=True: root_rot + root_shift applied), and name -> (T,3,3) global
    rotations. Positions in metres."""
    m = np.asarray(motion, np.float64)
    T = len(m)
    Rg, P = {}, {}
    for j in ORDER:
        par, off = REST[j]
        R = aa_to_mat(m[:, jcols(j)]) if (j != "root_rot" or with_root) else np.broadcast_to(np.eye(3), (T, 3, 3))
        if par is None:
            Rg[j] = R
            P[j] = (m[:, 3:6] / 10.0) if with_root else np.zeros((T, 3))
        else:
            Rg[j] = Rg[par] @ R
            P[j] = P[par] + np.einsum("tij,j->ti", Rg[par], np.asarray(off))
    if extra:
        for e, (par, off) in ENDS.items():
            P[e] = P[par] + np.einsum("tij,j->ti", Rg[par], np.asarray(off))
    return P, Rg
