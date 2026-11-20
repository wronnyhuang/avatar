"""RIG: drive Juno's skeleton DIRECTLY from a canon1 pose (T/CONTRACT.md; X/canon1/spec.py).  Pure numpy.

canon1 = SMPL-H upper-body LOCAL axis-angles on SMPL's T-pose rest whose axes are world aligned (x = the person's LEFT,
y up, z forward).  Juno's character-local frame is the same frame (x screen-right = her anatomical left when she faces
the camera, y up, z toward the camera), so for a canon joint j with SMPL global rotation G_j (FK of the local
rotations) Juno's world rotation delta is
    torso / head bones : Dc_b = G_j @ G0_j^T          (canon NEUTRAL -> Juno's REST posture; deviations copy 1:1)
    arm / hand / finger: Rel_b = G_spine3^T @ G_j @ C_b  (C_b = Juno's rest bone -> SMPL T-pose bone: direction + twist
                                                       reference (elbow hinge / palm normal)); the arm is posed RELATIVE
                                                       TO THE CHEST: D_b = D_spine_03(as posed) @ Rel_b, so arm-to-chest
                                                       directions equal SMPL's exactly (Seamless people sit with the
                                                       trunk pitched ~20 deg in canon; Juno stands upright)
Head is re-anchored on spine_03 as posed (Rel_b = Dc_s3^T Dc_b), fingers on the hand,
so mixing dial-driven and canon-driven groups stays consistent.

    cd = CanonDrive(rig)                  # (JunoRig builds one lazily: rig._canon_drive())
    st = cd.prepare(canon, canon_w, modes) -> per-solve state used by JunoRig.solve (see rig.py "")
Groups (anatomical sides, canon naming): torso head arm_l arm_r fingers_l fingers_r.  modes: head/torso "add"|"abs".
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = ("torso", "head", "arm_l", "arm_r", "fingers_l", "fingers_r")

# canon1 body joints (spec.JOINTS) with their column start, SMPL parent and the SMPL neutral rest offset (canon1/skeleton.py)
BODY = [("root_rot", 0, None, (0.0, 0.0, 0.0)),
        ("spine1", 6, "root_rot", (-0.0025, 0.1088, -0.0267)),
        ("spine2", 9, "spine1", (0.0055, 0.1352, 0.0011)),
        ("spine3", 12, "spine2", (0.0015, 0.0529, 0.0254)),
        ("neck", 15, "spine3", (-0.0028, 0.2139, -0.0430)),
        ("head", 18, "neck", (0.0052, 0.0650, 0.0513)),
        ("collar_l", 21, "spine3", (0.0788, 0.1217, -0.0341)),
        ("collar_r", 24, "spine3", (-0.0818, 0.1188, -0.0386)),
        ("shoulder_l", 27, "collar_l", (0.0910, 0.0305, -0.0089)),
        ("shoulder_r", 30, "collar_r", (-0.0960, 0.0326, -0.0091)),
        ("elbow_l", 33, "shoulder_l", (0.2596, -0.0128, -0.0275)),
        ("elbow_r", 36, "shoulder_r", (-0.2537, -0.0133, -0.0214)),
        ("wrist_l", 39, "elbow_l", (0.2492, 0.0090, -0.0012)),
        ("wrist_r", 42, "elbow_r", (-0.2553, 0.0078, -0.0056))]
BJ = {n: k for k, (n, _, _, _) in enumerate(BODY)}
BPAR = np.array([BJ[p] if p else -1 for (_, _, p, _) in BODY])
BCOL = np.array([c for (_, c, _, _) in BODY])
CANON_FINGERS = ("index", "middle", "pinky", "ring", "thumb")          # canon finger order (MANO)
THUMB_DIR_L = np.array([0.6, -0.25, 0.75])/np.linalg.norm([0.6, -0.25, 0.75])   # = B/common/kin.py _THUMB_DIR_L

# hand / face / chest geometry of B/common/kin.py (head joint frame / spine3 frame / wrist frame, metres, LEFT side)
PALM_L = np.array([0.050, -0.016, 0.002])
FACE_PTS = {"nose": (0.0, 0.035, 0.118), "mouth": (0.0, -0.012, 0.108), "chin": (0.0, -0.05, 0.092),
            "forehead": (0.0, 0.125, 0.098), "top_of_head": (0.0, 0.195, 0.01), "left_eye": (0.032, 0.075, 0.095),
            "right_eye": (-0.032, 0.075, 0.095), "left_ear": (0.078, 0.05, -0.005), "right_ear": (-0.078, 0.05, -0.005),
            "left_cheek": (0.052, 0.02, 0.085), "right_cheek": (-0.052, 0.02, 0.085), "back_of_head": (0.0, 0.09, -0.095),
            "left_temple": (0.068, 0.10, 0.05), "right_temple": (-0.068, 0.10, 0.05)}
CHEST_PTS = {"chest": (0.0, 0.06, 0.115), "left_chest": (0.07, 0.06, 0.11), "right_chest": (-0.07, 0.06, 0.11),
             "left_chest_side": (0.16, 0.0, 0.0), "right_chest_side": (-0.16, 0.0, 0.0)}
JUNO_FACE_LM = {"nose": "nose_tip", "mouth": "mouth", "chin": "chin", "top_of_head": "head_top", "left_eye": "eye_l",
                "right_eye": "eye_r"}
CONTACT_NEAR, CONTACT_FAR = 0.03, 0.15        # metres: canon palm within FAR of a landmark starts the contact fit (wide ramp: less pop)
GAP_FACE, GAP_CHEST = 0.045, 0.03              # metres: a canon palm this far OFF the surface still counts as touching it
                                               # (SMPL hands are thick, mocap palms hover); larger gaps shrink by this much
CHEST_CONTACT = 0.6                            # the chest fit is gentler (the torso push already keeps hands out of her)
CONTACT_TAU = 0.015                            # metres: softmin width over landmarks (continuity when the nearest one changes)
_ZDAMP = np.array([1.0, 0.3, 1.0])
# polish: how a hand meets her face.  landmark -> (finger direction in the HEAD frame for the anatomical RIGHT hand
# (x mirrored for the left), strength): the palm faces INTO the head and the fingers point this way, blended in by the
# contact weight.  Mocap / kin hands at the mouth come in fingers-up, which put the fingers over her eyes (lead, 11-20).
FACE_ORIENT = {"mouth": ((0.92, 0.38, 0.0), 1.0), "nose": ((0.75, 0.66, 0.0), 0.8), "chin": ((0.0, 1.0, 0.0), 0.0),
               "left_eye": ((0.22, 0.97, 0.0), 0.9), "right_eye": ((0.22, 0.97, 0.0), 0.9),
               "forehead": ((0.55, 0.83, 0.0), 0.9), "left_cheek": ((0.0, 1.0, 0.0), 0.6),
               "right_cheek": ((0.0, 1.0, 0.0), 0.6), "left_temple": ((0.0, 1.0, 0.0), 0.5),
               "right_temple": ((0.0, 1.0, 0.0), 0.5)}
TORSO_FIELD_BONES = ("pelvis", "spine_01", "spine_02", "spine_03", "clavicle_l", "clavicle_r")
TF_STEP = 0.012                                # metres: torso height-field cell
PUSH_TOL = 0.008                               # metres
PUSH_TAU = 0.006                               # metres: softmin width over the torso exits (continuous direction)
_EXITS = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
SIDE_SLACK = 0.03                              # metres of sleeve allowed into her side before a sideways push
PUSH_GAIN = 1.15                               # one-pass push overshoots a little (the surface curves away)
_TU = np.array([0.65, 0.85, 1.0]); _TF = np.array([0.35, 0.7, 1.0]); _TW = np.r_[_TF, 1.0]


_BCOLS = (BCOL[:, None] + np.arange(3)).reshape(-1)
_BPAR1 = np.array([BJ[p] for (_, _, p, _) in BODY[1:]])
_BOFF = np.array([o for (_, _, _, o) in BODY], float)
_PALM_LR = (PALM_L.copy(), PALM_L*np.array([-1.0, 1.0, 1.0]))
_FACE_N = list(FACE_PTS); _FACE_A = np.array([FACE_PTS[n] for n in _FACE_N], float)
_CHEST_N = list(CHEST_PTS); _CHEST_A = np.array([CHEST_PTS[n] for n in _CHEST_N], float)
_FACE_C = _FACE_A.mean(0); _FACE_R = float(np.linalg.norm(_FACE_A - _FACE_C, axis=1).max())
_CHEST_C = _CHEST_A.mean(0); _CHEST_R = float(np.linalg.norm(_CHEST_A - _CHEST_C, axis=1).max())


def _ssv(e0, e1, x):
    t = np.clip((x - e0)/(e1 - e0), 0.0, 1.0); return t*t*(3.0 - 2.0*t)


def _snap(off, nrm, gap):
    """shrink the off-surface (normal) part of a contact offset by `gap` (never below 0); tangential part kept."""
    on = float(off @ nrm)
    if on <= 0.0:
        return off
    return off - nrm*(on - max(0.0, on - gap))


def _ss(e0, e1, x):
    t = min(1.0, max(0.0, (x - e0)/(e1 - e0))); return t*t*(3.0 - 2.0*t)


TORSO_BONES = (("spine_01", "spine1"), ("spine_02", "spine2"), ("spine_03", "spine3"))
HEAD_BONES = (("neck_01", "neck"), ("head", "head"))
ARM_BONES = (("clavicle", "collar"), ("upperarm", "shoulder"), ("lowerarm", "elbow"), ("hand", "wrist"))


def rodrigues_aa(aa):
    """(n,3) axis-angle -> (n,3,3)."""
    th = np.sqrt((aa*aa).sum(1))
    k = aa/np.maximum(th, 1e-12)[:, None]
    x, y, z = k[:, 0], k[:, 1], k[:, 2]
    c, s = np.cos(th), np.sin(th); t = 1.0 - c
    R = np.empty((len(aa), 3, 3))
    R[:, 0, 0] = t*x*x + c; R[:, 0, 1] = t*x*y - s*z; R[:, 0, 2] = t*x*z + s*y
    R[:, 1, 0] = t*x*y + s*z; R[:, 1, 1] = t*y*y + c; R[:, 1, 2] = t*y*z - s*x
    R[:, 2, 0] = t*x*z - s*y; R[:, 2, 1] = t*y*z + s*x; R[:, 2, 2] = t*z*z + c
    return R


def _nrm(v):
    v = np.asarray(v, float); L = float(np.linalg.norm(v))
    return v/L if L > 1e-12 else v*0.0


def _frame(u, n):
    u = _nrm(u); n = _nrm(n - u*np.dot(n, u))
    return np.array([u, n, np.cross(u, n)]).T


def _align(a, b):
    """the minimal rotation taking direction a to direction b (closed form, no axis-angle round trip)."""
    a = _nrm(a); b = _nrm(b)
    v = np.cross(a, b); c = float(a @ b)
    if c < -0.999999:
        return rodrigues_aa((_nrm(np.cross(a, [1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.cross(a, [0.0, 1.0, 0.0]))*math.pi)[None])[0]
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + (K @ K)*(1.0/(1.0 + c))


def _seg_closest(p1, q1, p2, q2):
    """closest points of segments p1q1 / p2q2 -> (s, t, distance)."""
    d1 = q1 - p1; d2 = q2 - p2; r = p1 - p2
    a = float(d1 @ d1); e = float(d2 @ d2); f = float(d2 @ r)
    if a < 1e-12 and e < 1e-12:
        return 0.0, 0.0, float(np.linalg.norm(r))
    c = float(d1 @ r); b = float(d1 @ d2); den = a*e - b*b
    s = min(1.0, max(0.0, (b*f - c*e)/den)) if den > 1e-12 else 0.0
    t = (b*s + f)/e if e > 1e-12 else 0.0
    if t < 0.0:
        t = 0.0; s = min(1.0, max(0.0, -c/a)) if a > 1e-12 else 0.0
    elif t > 1.0:
        t = 1.0; s = min(1.0, max(0.0, (b - c)/a)) if a > 1e-12 else 0.0
    return s, t, float(np.linalg.norm(p1 + d1*s - p2 - d2*t))


def _seg_dist_batch(p1, q1, p2, q2):
    """(T,3) x 4 -> (T,) distance between segments p1q1 / p2q2 (same clamping as _seg_closest)."""
    d1 = q1 - p1; d2 = q2 - p2; r = p1 - p2
    a = (d1*d1).sum(1); e = (d2*d2).sum(1); f = (d2*r).sum(1); c = (d1*r).sum(1); b = (d1*d2).sum(1)
    den = a*e - b*b
    s = np.where(den > 1e-12, np.clip((b*f - c*e)/np.where(den > 1e-12, den, 1.0), 0.0, 1.0), 0.0)
    t = np.where(e > 1e-12, (b*s + f)/np.where(e > 1e-12, e, 1.0), 0.0)
    sa = np.where(a > 1e-12, a, 1.0)
    s = np.where(t < 0.0, np.where(a > 1e-12, np.clip(-c/sa, 0.0, 1.0), 0.0), np.where(t > 1.0, np.where(a > 1e-12, np.clip((b - c)/sa, 0.0, 1.0), 0.0), s))
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(p1 + d1*s[:, None] - p2 - d2*t[:, None], axis=1)


def _rot(u, a):
    """rotation by angle a about unit axis u (scalar math)."""
    x, y, z = float(u[0]), float(u[1]), float(u[2]); c, s_ = math.cos(a), math.sin(a); t = 1.0 - c
    return np.array([[t*x*x + c, t*x*y - s_*z, t*x*z + s_*y], [t*x*y + s_*z, t*y*y + c, t*y*z - s_*x],
                     [t*x*z - s_*y, t*y*z + s_*x, t*z*z + c]])


def _twist_angle(R, u):
    """twist of rotation R about unit axis u (swing-twist, twist applied first), radians in (-pi, pi]."""
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2]))*0.5
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if w > 1e-4:
        v = v/(4.0*w)
    else:                                              # ~180 deg: the axis from the symmetric part
        ax = np.sqrt(np.maximum((np.diag(R) + 1.0)*0.5, 0.0))
        if R[0, 1] + R[1, 0] < 0: ax[1] = -ax[1]
        if R[0, 2] + R[2, 0] < 0: ax[2] = -ax[2]
        v = ax
    th = 2.0*math.atan2(float(v @ u), w)
    return th - 2*math.pi if th > math.pi else th


def rpow(R, f):
    """R^f about its own axis (numpy only)."""
    c = (R[0, 0] + R[1, 1] + R[2, 2] - 1.0)*0.5
    a = math.acos(max(-1.0, min(1.0, c)))
    if a < 1e-9:
        return np.eye(3)
    s = math.sin(a)
    if s < 1e-6:                                            # ~180 deg: axis from the diagonal
        ax = np.sqrt(np.maximum((np.diag(R) + 1.0)*0.5, 0.0))
        if R[0, 1] + R[1, 0] < 0:
            ax[1] = -ax[1]
        if R[0, 2] + R[2, 0] < 0:
            ax[2] = -ax[2]
    else:
        ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])/(2.0*s)
    return rodrigues_aa((ax*(a*f))[None])[0]


def slerpR(A, B, w):
    if w <= 0.0:
        return A
    if w >= 1.0:
        return B
    return A @ rpow(A.T @ B, w)


def canon_neutral():
    return np.load(os.path.join(HERE, "canon_neutral.npy")).astype(np.float64)


def canon_globals(canon):
    """(189,) or (T,189) -> body globals (T,14,3,3) and finger rotations (T,2,5,3,3,3) local (canon finger order)."""
    m = np.asarray(canon, np.float64)
    if m.ndim == 1:
        m = m[None]
    T = len(m)
    if T == 1:                                          # one Rodrigues call for the 14 body + 30 finger joints
        R = rodrigues_aa(np.concatenate([m[0, _BCOLS], m[0, 45:135]]).reshape(-1, 3))
        nb = len(BODY)
        G = np.empty((1, nb, 3, 3)); g = G[0]
        for k in range(nb):
            p = BPAR[k]
            g[k] = R[k] if p < 0 else g[p] @ R[k]
        return G, R[nb:].reshape(1, 2, 5, 3, 3, 3)
    Rl = rodrigues_aa(m[:, BCOL[:, None] + np.arange(3)].reshape(-1, 3)).reshape(T, len(BODY), 3, 3)
    G = np.empty_like(Rl)
    for k in range(len(BODY)):
        p = BPAR[k]
        G[:, k] = Rl[:, k] if p < 0 else G[:, p] @ Rl[:, k]
    Rf = rodrigues_aa(m[:, 45:135].reshape(-1, 3)).reshape(T, 2, 5, 3, 3, 3)
    return G, Rf


def _lean_fwd_deg(G):
    """(T,3,3) global rotations -> forward lean (deg) of the bone's up axis (about x; + = toward the camera)."""
    up = G[:, :, 1]
    return np.degrees(np.arctan2(up[:, 2], up[:, 1]))


def rebase_lean(motion, thresh_deg=8.0, keep_head="auto"):
    """OFFLINE helper (CLI / scoring), not used by solve(): canon1 neutral = a SEATED Seamless person (chest leaning
    ~21 deg forward); a STANDING mocap clip (HumanML3D / Motion-X: chest ~2-7 deg) therefore renders on Juno leaning
    BACK by ~15-19 deg in abs mode.  If the clip's median chest lean differs from the canon neutral's by more than
    thresh_deg, the body is turned once about x at the root (root_rot) so the medians match; keep_head gives the turn
    back at the neck so the head keeps its world orientation (else she looks ~15-20 deg down: standing mocap heads are
    level in the world).  That moves hand-to-face relations by ~3 cm (the face turns about the neck: 'mouth' became
    'chin'), so keep_head="auto" (default) gives back only HALF when the clip has a hand at the face (<= ~1.5 cm,
    absorbed by the contact fit) and all of it otherwise.  -> (motion copy, applied degrees)."""
    m = np.array(motion, np.float64, copy=True)
    if m.ndim != 2 or len(m) == 0:
        return m, 0.0
    G0, _ = canon_globals(canon_neutral()[:135])
    G, _ = canon_globals(m[:, :135])
    k = BJ["spine3"]
    d = float(_lean_fwd_deg(G0[:, k])[0] - np.median(_lean_fwd_deg(G[:, k])))
    if abs(d) <= thresh_deg:
        return m, 0.0
    a = math.radians(d)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])
    R = Rx[None] @ rodrigues_aa(m[:, 0:3])
    m[:, 0:3] = _mat_to_aa(R)
    if keep_head == "auto":                            # hands at the face in this clip: half (contacts move <= ~1.5 cm)
        h = 0.5 if _clip_touches_face(motion) else 1.0
    else:
        h = float(keep_head)
    if h > 0.0:                                        # the head keeps (h of) its WORLD orientation: the neck turns back
        b = -a*h
        Rb = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(b), -math.sin(b)], [0.0, math.sin(b), math.cos(b)]])
        Gs3 = G[:, k]
        Ln = rodrigues_aa(m[:, 15:18])
        m[:, 15:18] = _mat_to_aa(np.swapaxes(Gs3, 1, 2) @ Rb[None] @ Gs3 @ Ln)
    return m, d


def _clip_touches_face(motion, near=0.15):
    """any frame with a canon palm within `near` m of a face landmark."""
    m = np.asarray(motion, np.float64)
    G, _ = canon_globals(m[:, :135])
    P = np.zeros((len(m), len(BODY), 3))
    for k_ in range(1, len(BODY)):
        P[:, k_] = P[:, BPAR[k_]] + G[:, BPAR[k_]] @ _BOFF[k_]
    kh = BJ["head"]
    L = P[:, kh][:, None] + np.einsum("tij,nj->tni", G[:, kh], _FACE_A)
    for a, pl in (("l", _PALM_LR[0]), ("r", _PALM_LR[1])):
        w = BJ["wrist_" + a]
        palm = P[:, w] + G[:, w] @ pl
        if (np.linalg.norm(L - palm[:, None], axis=2).min(1) < near).any():
            return True
    return False


def _mat_to_aa(R):
    """(T,3,3) -> (T,3) axis-angle (robust near pi)."""
    out = np.zeros((len(R), 3))
    for t, r in enumerate(R):
        c = (r[0, 0] + r[1, 1] + r[2, 2] - 1.0)*0.5
        a = math.acos(max(-1.0, min(1.0, c)))
        if a < 1e-9:
            continue
        if math.sin(a) > 1e-6:
            ax = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])/(2.0*math.sin(a))
        else:
            ax = np.sqrt(np.maximum((np.diag(r) + 1.0)*0.5, 0.0))
            if r[0, 1] + r[1, 0] < 0: ax[1] = -ax[1]
            if r[0, 2] + r[2, 0] < 0: ax[2] = -ax[2]
        out[t] = _nrm(ax)*a
    return out


class CanonDrive:
    def __init__(self, rig):
        self.rig = rig
        r, bi = rig.r, rig.bi
        self.neutral = canon_neutral()
        G0, _ = canon_globals(self.neutral)
        self.G0 = G0[0]
        # SMPL rest joint positions (rest rotations = identity)
        ps = np.zeros((len(BODY), 3))
        for k, (_, _, p, off) in enumerate(BODY):
            ps[k] = (ps[BPAR[k]] if p else 0.0) + np.asarray(off)
        self.smpl_rest = ps
        fwd = np.array([0.0, 0.0, 1.0])
        self.C = {}                                   # Juno bone index -> C_b
        self.arm = {}                                 # anatomical side -> {part: (bone idx, canon joint k)}
        self.fing = {}                                # anatomical side -> list of (bone idx, digit k (canon), level j)
        for a in ("l", "r"):
            sg = 1.0 if a == "l" else -1.0            # anatomical left = +x
            ib = {p: bi.get(f"{p}_{a}") for p in ("clavicle", "upperarm", "lowerarm", "hand")}
            if any(v is None for v in ib.values()):
                continue
            kj = {p: BJ[f"{c}_{a}"] for p, c in ARM_BONES}
            self.arm[a] = {p: (ib[p], kj[p]) for p, _ in ARM_BONES}
            # clavicle: direction + forward as the twist reference
            uj = r[ib["upperarm"]] - r[ib["clavicle"]]; us = ps[kj["upperarm"]] - ps[kj["clavicle"]]
            self.C[ib["clavicle"]] = _frame(us, fwd) @ _frame(uj, fwd).T
            # upper / lower arm: direction + elbow hinge (cross(upper, forearm-bent-forward))
            uj = _nrm(r[ib["lowerarm"]] - r[ib["upperarm"]]); fj = _nrm(r[ib["hand"]] - r[ib["lowerarm"]])
            hj = np.cross(uj, fj)
            hj = _nrm(hj) if np.linalg.norm(hj) > 0.07 else _nrm(np.cross(uj, fwd))
            us = _nrm(ps[kj["lowerarm"]] - ps[kj["upperarm"]]); fs = _nrm(ps[kj["hand"]] - ps[kj["lowerarm"]])
            hs = _nrm(np.cross(us, fwd))
            self.C[ib["upperarm"]] = _frame(us, hs) @ _frame(uj, hj).T
            self.C[ib["lowerarm"]] = _frame(fs, hs) @ _frame(fj, hj).T
            # hand: [x, y = fingers, z = palm normal] (JunoRig.hand_rest_frame convention) -> SMPL/MANO flat hand
            ps_i = 1 if a == "l" else 0               # rig param side of this anatomical side
            Fj = rig.hand_rest_frame[ps_i]
            ys = np.array([sg, 0.0, 0.0]); zs = np.array([0.0, -1.0, 0.0])
            Fs = np.array([np.cross(ys, zs), ys, zs]).T
            self.C[ib["hand"]] = Fs @ Fj.T
            self.Chand = getattr(self, "Chand", {}); self.Chand[a] = self.C[ib["hand"]]
            zj = Fj[:, 2]
            fl = []
            for dk, d in enumerate(CANON_FINGERS):
                ch = [bi.get(f"{d}_0{j}_{a}") for j in (1, 2, 3)]
                ds = THUMB_DIR_L*np.array([sg, 1.0, 1.0]) if d == "thumb" else np.array([sg, 0.0, 0.0])
                for j, b in enumerate(ch):
                    if b is None:
                        continue
                    nxt = ch[j + 1] if j < 2 else None
                    if nxt is not None:
                        u = r[nxt] - r[b]
                    else:
                        u = r[b] - r[ch[j - 1]] if j > 0 else r[b] - r[ib["hand"]]
                    self.C[b] = _frame(ds, zs) @ _frame(u, zj).T
                    fl.append((b, dk, j))
            self.fing[a] = fl
        self.fore_axis = {a: _nrm(r[p["hand"][0]] - r[p["lowerarm"][0]]) for a, p in self.arm.items()}
        self.roll_share = 0.0 if rig.tun.get("forearm_roll_share") is not None else 0.5
        self.arm_bone = {b: (a, part) for a, parts in self.arm.items() for part, (b, _) in parts.items()}
        self.torso = [(bi[b], BJ[c]) for b, c in TORSO_BONES if b in bi]
        self.head = [(bi[b], BJ[c]) for b, c in HEAD_BONES if b in bi]
        self.s3 = bi["spine_03"]; self.k_s3 = BJ["spine3"]
        # finger levels (for the batched FK after the bone loop): per level, bones / parents / offsets
        lv = [[], [], []]
        for a, fl in self.fing.items():
            for (b, dk, j) in fl:
                lv[j].append((b, a, dk))
        self.flev = []
        for j in range(3):
            if not lv[j]:
                continue
            idx = np.array([x[0] for x in lv[j]]); par = np.array([rig.parent[b] for b in idx])
            self.flev.append((j, idx, par, [x[1] for x in lv[j]], np.array([x[2] for x in lv[j]]), rig.r[idx] - rig.r[par]))
        self.Cf = {b: self.C[b] for a in self.fing for (b, _, _) in self.fing[a]}
        self._torso_field()
        self._contact_init()

    def _contact_init(self):
        """Juno's version of the kin landmarks: offsets from Juno's head bone / spine_03 in REST space (scene units).
        Canon offsets are taken in the canon NEUTRAL world frame (G0 @ local) so they compare with Juno's rest."""
        rig = self.rig; k = rig.k; bi = rig.bi
        lm = dict(rig.meta.get("landmarks") or {})
        hb = rig.r[bi["head"]]
        G0h = self.G0[BJ["head"]]; G0c = self.G0[self.k_s3]
        self.face_n = {n: G0h @ np.asarray(o) for n, o in FACE_PTS.items()}         # canon, neutral world, metres
        src, dst = [], []
        for n, ln in JUNO_FACE_LM.items():
            if ln in lm and len(lm[ln]) >= 3:
                src.append(self.face_n[n]); dst.append(np.asarray(lm[ln][:3], float) - hb/k)
        self.face_j = {}
        if len(src) >= 3:                            # per-axis scale + offset (least squares) canon -> Juno head space
            src = np.array(src); dst = np.array(dst); A = np.zeros(3); B = np.zeros(3)
            for ax in range(3):
                X = np.c_[src[:, ax], np.ones(len(src))]
                sol, *_ = np.linalg.lstsq(X, dst[:, ax], rcond=None)
                A[ax], B[ax] = (sol[0], sol[1]) if abs(src[:, ax]).max() > 1e-6 and np.ptp(src[:, ax]) > 0.02 else (1.0, sol[1])
            A = np.clip(A, 0.6, 1.6)
            self.face_scale = float(np.mean(A))
            for n, o in self.face_n.items():
                self.face_j[n] = (A*o + B)*k
            for n, ln in JUNO_FACE_LM.items():   # the measured landmarks themselves
                if ln in lm and len(lm[ln]) >= 3:
                    self.face_j[n] = (np.asarray(lm[ln][:3], float) - hb/k)*k
            # polish: eyes = the lid centres (the eyeball centre sat 2 cm inside), forehead from the brows
            for n_, ln_ in (("left_eye", "l"), ("right_eye", "r")):
                pts = [p_ for key in (f"lid_upper_{ln_}", f"lid_lower_{ln_}") for p_ in (lm.get(key) or [])]
                if len(pts) >= 3:
                    self.face_j[n_] = (np.mean(np.asarray(pts, float), 0) - hb/k)*k
            br = [p_ for key in ("brow_l", "brow_r") for p_ in (lm.get(key) or [])]
            if len(br) >= 4:
                self.face_j["forehead"] = (np.mean(np.asarray(br, float), 0) + np.array([0.0, 0.024, -0.004]) - hb/k)*k
        else:
            self.face_scale = 1.0
        # chest: x by the shoulder width, y by the shoulder height, z so the chest front lands on Juno's torso front
        s3 = rig.r[self.s3]
        self.chest_n = {n: G0c @ np.asarray(o) for n, o in CHEST_PTS.items()}
        G0 = self.G0
        sh_c = G0c @ (np.asarray(BODY[BJ["collar_l"]][3]) + G0c.T @ G0[BJ["collar_l"]] @ np.asarray(BODY[BJ["shoulder_l"]][3]))
        sh_j = (rig.r[bi["upperarm_l"]] - s3)/k
        sx = float(np.clip(sh_j[0]/max(sh_c[0], 1e-3), 0.6, 1.6))
        front_j = (rig.torso_c[2] + rig.torso_r[2] - s3[2])/k
        self.chest_j = {}
        for n, o in self.chest_n.items():
            self.chest_j[n] = np.array([o[0]*sx, o[1] + (sh_j[1] - sh_c[1]), o[2] - self.chest_n["chest"][2] + front_j])*k
        self.chest_scale = sx
        tf = self.tf
        if tf is not None:                           # the chest front = HER cardigan front (mesh height field)
            for n, o in self.chest_n.items():
                if abs(o[0]) > 0.12:
                    continue
                q = self.chest_j[n]/k + s3/k
                ix = int(round((q[0] - tf["x0"])/TF_STEP)); iy = int(round((q[1] - tf["y0"])/TF_STEP))
                if 0 <= ix < tf["NX"] and 0 <= iy < tf["NY"] and tf["v"][iy, ix]:
                    self.chest_j[n][2] = (tf["F"][iy, ix] - s3[2]/k)*k
        self.contact_on = os.environ.get("RIG_CANON_CONTACT", "1") != "0" and bool(self.face_j)
        # outward surface normals of the canon landmarks (neutral frame): the face's from the head centre, the chest's forward
        hcn = self.G0[BJ["head"]] @ np.array([0.0, 0.07, 0.0])
        self.face_nrm = [_nrm((self.face_n[n] - hcn)*_ZDAMP) for n in _FACE_N]
        self.chest_nrm = [_nrm(self.G0[self.k_s3] @ (np.array([1.0, 0.0, 0.0])*np.sign(CHEST_PTS[n][0]) if abs(CHEST_PTS[n][0]) > 0.12
                                                     else np.array([0.0, 0.0, 1.0]))) for n in _CHEST_N]

    def canon_points(self, st):
        """canon FK (metres, canon world): joint positions + both palms; cached in st."""
        if "P" in st:
            return st["P"]
        G = st["G"]; c = st["c"]
        P = np.empty((len(BODY), 3)); P[0] = c[3:6]/10.0
        GO = np.einsum("kij,kj->ki", G[_BPAR1], _BOFF[1:])            # each joint's offset in its parent's frame
        for k_ in range(1, len(BODY)):
            P[k_] = P[BPAR[k_]] + GO[k_ - 1]
        wl, wr = BJ["wrist_l"], BJ["wrist_r"]
        palm = {"l": P[wl] + G[wl] @ _PALM_LR[0], "r": P[wr] + G[wr] @ _PALM_LR[1]}
        st["P"] = (P, palm)
        return st["P"]

    def contact_plan(self, st):
        """per anatomical side: (frame 'head'|'chest', landmark, offset in the neutral frame (m), weight) or None.
        Vectorised; a palm farther than CONTACT_FAR + the landmark spread from the head and the chest exits early."""
        plan = {}
        if not self.contact_on:
            return plan
        act = [a for a in ("l", "r") if st["w"]["arm_" + a] > 0.0 and a in self.arm]
        if not act:
            return plan
        P, palm = self.canon_points(st)
        G = st["G"]
        kh = BJ["head"]
        Gh, Gc = G[kh], G[self.k_s3]
        hp, cp = P[kh], P[self.k_s3]
        hc = hp + Gh @ _FACE_C; cc = cp + Gc @ _CHEST_C
        for a in act:
            pa = palm[a]
            dh = float(np.linalg.norm(pa - hc)); dc = float(np.linalg.norm(pa - cc))
            if dh > _FACE_R + CONTACT_FAR and dc > _CHEST_R + CONTACT_FAR:
                continue
            Lh = hp + _FACE_A @ Gh.T; Lc = cp + _CHEST_A @ Gc.T
            L = np.vstack([Lh, Lc]); d = np.linalg.norm(pa - L, axis=1)
            dm = float(d.min())
            w = 1.0 - _ss(CONTACT_NEAR, CONTACT_FAR, dm)
            if w <= 0.0:
                continue
            # SOFT landmark weights (no pop when the nearest landmark changes): softmin over the near ones
            sel = np.nonzero(d < dm + 3.0*CONTACT_TAU)[0]
            wn = np.exp(-(d[sel] - dm)/CONTACT_TAU); wn = wn/wn.sum()
            ent = []
            nf = len(_FACE_N)
            for m, wm in zip(sel, wn):
                if m < nf:
                    off = self.G0[kh] @ (Gh.T @ (pa - L[m])); nrm = self.face_nrm[m]; gap = GAP_FACE
                    ent.append(("head", _FACE_N[m], _snap(off, nrm, gap), float(wm)))
                else:
                    off = self.G0[self.k_s3] @ (Gc.T @ (pa - L[m])); nrm = self.chest_nrm[m - nf]
                    ent.append(("chest", _CHEST_N[m - nf], _snap(off, nrm, GAP_CHEST), float(wm)))
            hw = sum(e[3] for e in ent if e[0] == "head")
            plan[a] = (ent, w*(hw + (1.0 - hw)*CHEST_CONTACT))
        return plan

    def contact(self, st, a, ps, D, pos):
        """step 2: after the rotation copy, move Juno's palm to the same place RELATIVE to her head / chest as the
        canon palm (proximity-blended), then push it out of her head / torso volumes; two-bone re-solve keeping the
        elbow swivel; the hand keeps its world orientation.  Mutates D / pos of upperarm, lowerarm, hand."""
        rig = self.rig; bi = rig.bi
        iu, il, ih = (self.arm[a][p][0] for p in ("upperarm", "lowerarm", "hand"))
        S, E, W = pos[iu].copy(), pos[il].copy(), pos[ih].copy()
        pl = st.get("plan", {}).get(a)
        if pl is not None:                    # the palm turns INTO her face, fingers per landmark (soft-blended)
            ent, w = pl
            hb = bi["head"]; Dh = D[hb]
            zs = np.zeros(3); fs = np.zeros(3); ws = 0.0
            for fr, n, off, wn in ent:
                if fr != "head" or n not in FACE_ORIENT:
                    continue
                fdir, strength = FACE_ORIENT[n]
                zs += wn*((rig.head_c - rig.r[hb] - self.face_j[n])*_ZDAMP)
                fs += wn*strength*np.asarray(fdir); ws += wn*strength
            ws *= w
            if ws > 1e-4:
                z = _nrm(Dh @ zs)
                f = Dh @ (fs*np.array([1.0 if a == "r" else -1.0, 1.0, 1.0]))
                y = _nrm(f - z*float(f @ z))
                Ddes = np.array([np.cross(y, z), y, z]).T @ rig.hand_rest_frame[ps].T
                D[ih] = slerpR(D[ih], Ddes, _ss(0.0, 0.6, ws))     # saturating: a hand AT the mouth turns fully
        Fh = D[ih] @ rig.hand_rest_frame[ps]
        pv = Fh @ rig.palm_local[ps]
        palm = W + pv
        tgt = palm.copy()
        if pl is not None:
            ent, w = pl
            want = np.zeros(3)
            for fr, n, off, wn in ent:
                if fr == "head":
                    ib = bi["head"]; Lj = pos[ib] + D[ib] @ self.face_j[n]; sc = self.face_scale
                else:
                    ib = self.s3; Lj = pos[ib] + D[ib] @ self.chest_j[n]; sc = self.chest_scale
                want += wn*(Lj + D[ib] @ (off*sc*rig.k))
            tgt = palm + (want - palm)*w
        # push-out: head ellipsoid (inflated by the hand's half thickness) and the chest ellipsoid
        hb = bi["head"]
        hc = pos[hb] + D[hb] @ (rig.head_c - rig.r[hb])
        R = rig.head_radii + rig.tun["face_pad"]*rig.stature
        q = D[hb].T @ (tgt - hc); e = q/R; el = float(np.linalg.norm(e))
        if el < 1.0:
            tgt = hc + D[hb] @ (e/max(el, 1e-6)*R)
        if self.tf is not None:                       # her real torso (mesh height field) for the palm
            tgt = tgt + self.tf_push(D, pos, tgt[None], 0.006)[0]
        else:
            s3p = pos[self.s3]; s3r = rig.r[self.s3]
            Rt = rig.torso_r + 0.6*rig.tun["face_pad"]*rig.stature
            q = D[self.s3].T @ (tgt - s3p) + s3r - rig.torso_c; e = q/Rt; el = float(np.linalg.norm(e))
            if el < 1.0:
                tgt = s3p + D[self.s3] @ (rig.torso_c + e/max(el, 1e-6)*Rt - s3r)
        dv = tgt - palm
        if float(dv @ dv) < 1e-10:
            return
        Wt = W + dv
        l1, l2 = float(np.linalg.norm(E - S)), float(np.linalg.norm(W - E))
        d0 = _nrm(W - S)
        pole = (E - S) - d0*float(np.dot(E - S, d0))           # the current swivel
        if float(pole @ pole) < 1e-10:
            pole = D[iu] @ np.array([0.0, 0.0, -1.0])
        from math import sqrt
        dd = Wt - S; L = float(np.linalg.norm(dd)); dr = dd/max(L, 1e-9)
        L = min(max(L, abs(l1 - l2) + 1e-4), (l1 + l2)*0.999)
        Wt = S + dr*L
        ca = max(-1.0, min(1.0, (l1*l1 + L*L - l2*l2)/(2.0*l1*L)))
        pp = pole - dr*float(np.dot(pole, dr)); pp = _nrm(pp) if float(pp @ pp) > 1e-12 else _nrm(np.cross(dr, [0, 0, 1.0]))
        Et = S + l1*(ca*dr + sqrt(max(0.0, 1.0 - ca*ca))*pp)
        R1 = _align(E - S, Et - S)
        R2 = _align(R1 @ (W - E), Wt - Et)
        D[iu] = R1 @ D[iu]
        D[il] = R2 @ R1 @ D[il]
        pos[il] = Et; pos[ih] = Wt

    # ---------------------------------------------------------------- polish: her real body volume
    def _torso_field(self):
        """A height field of HER torso (cardigan included) from the rest mesh: for each (x, y) cell in rest space, the
        front-most / back-most z of the vertices skinned mainly to pelvis / spine / clavicle bones, plus each row's
        x extent.  Arms are pushed out of it (forward, backward or sideways, whichever is shortest).  Also measures the
        sleeve radii (upper arm / forearm) from the mesh.  Metres, rest space."""
        rig = self.rig; self.tf = None
        self.arm_rad = (0.045, 0.036)
        self.rads = np.r_[np.full(3, 0.04), np.full(3, 0.036), 0.006]
        try:
            import json
            meta = rig.meta; d = rig.asset_dir
            lods = {l.get("name"): l for l in meta.get("lods", [])}
            l = lods.get("lod1") or lods.get("lod0") or (meta.get("lods") or [None])[0]
            mj = json.load(open(os.path.join(d, l["mesh"])))
            buf = open(os.path.join(d, os.path.dirname(l["mesh"]), "mesh.bin"), "rb").read()
            n = int(mj["vertex_count"]); B = mj["buffers"]
            V = np.frombuffer(buf, np.float32, n*3, B["position"]["offset"]).reshape(n, 3).astype(np.float64)
            Jn = np.frombuffer(buf, np.uint16, n*4, B["joints"]["offset"]).reshape(n, 4)
            Wt = np.frombuffer(buf, np.float32, n*4, B["weights"]["offset"]).reshape(n, 4)
        except Exception as e:                       # no mesh: the old ellipsoids only
            print(f"[canon_drive] torso field unavailable: {e}", file=sys.stderr)
            return
        keep = np.zeros(n, bool)
        for sm in mj.get("submeshes", []):
            if sm["name"] == "body" or sm["name"].startswith("cloth_"):
                keep[sm["first_vertex"]:sm["first_vertex"] + sm["vertex_count"]] = True
        dom = Jn[np.arange(n), Wt.argmax(1)]
        bi = rig.bi; k = rig.k
        tb = np.array([bi[b] for b in TORSO_FIELD_BONES if b in bi])
        sel = keep & np.isin(dom, tb)
        y0 = rig.r[bi["pelvis"]][1]/k - 0.14; y1 = rig.r[bi["neck_01"]][1]/k + 0.01
        P = V[sel]; P = P[(P[:, 1] >= y0) & (P[:, 1] <= y1)]
        nx = int(round(0.30/TF_STEP)); xs0 = -nx*TF_STEP; NX = 2*nx + 1
        NY = int(math.ceil((y1 - y0)/TF_STEP)) + 1
        ix = np.clip(np.round((P[:, 0] - xs0)/TF_STEP).astype(int), 0, NX - 1)
        iy = np.clip(np.round((P[:, 1] - y0)/TF_STEP).astype(int), 0, NY - 1)
        F = np.full((NY, NX), -np.inf); Bk = np.full((NY, NX), np.inf)
        np.maximum.at(F, (iy, ix), P[:, 2]); np.minimum.at(Bk, (iy, ix), P[:, 2])
        valid = np.isfinite(F)
        xl = np.full(NY, np.inf); xr = np.full(NY, -np.inf)
        np.minimum.at(xl, iy, P[:, 0]); np.maximum.at(xr, iy, P[:, 0])
        # close holes / be a little conservative: 2-cell max (front) / min (back) dilation
        Fd = F.copy(); Bd = Bk.copy()
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                Fd = np.maximum(Fd, np.roll(np.roll(F, dy, 0), dx, 1)); Bd = np.minimum(Bd, np.roll(np.roll(Bk, dy, 0), dx, 1))
        vd = np.isfinite(Fd)
        Ff, Bf = Fd.copy(), Bd.copy()                 # invalid cells take their valid neighbours (bilinear lookups near edges)
        for _ in range(3):
            for A, fn, bad in ((Ff, np.maximum, -np.inf), (Bf, np.minimum, np.inf)):
                nb = A.copy()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        sh = np.roll(np.roll(A, dy, 0), dx, 1)
                        nb = np.where(np.isfinite(nb), np.where(np.isfinite(sh), fn(nb, sh), nb), sh)
                A[~np.isfinite(A)] = nb[~np.isfinite(A)]
        Ff = np.where(np.isfinite(Ff), Ff, 0.0); Bf = np.where(np.isfinite(Bf), Bf, 0.0)
        bl = lambda A: sum(np.roll(np.roll(A, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1))/9.0
        Ff, Bf = np.maximum(bl(Ff), Ff - 0.004), np.minimum(bl(Bf), Bf + 0.004)    # soften the max-filter steps
        xl = np.where(np.isfinite(xl), xl, 0.0); xr = np.where(np.isfinite(xr), xr, 0.0)
        self.tf = {"F": Ff, "B": Bf, "FB": np.stack([Ff, Bf], -1), "v": vd, "xl": xl,
                   "xr": xr, "x0": xs0, "y0": y0, "NX": NX, "NY": NY,
                   "ysplit": rig.r[bi["spine_02"]][1]/k, "s1": bi["spine_01"], "s3": bi["spine_03"]}
        # sleeve radii: distance of the arm's skinned vertices to the bone line (75th percentile)
        rad = []
        for a0, a1 in (("upperarm_l", "lowerarm_l"), ("lowerarm_l", "hand_l")):
            if a0 not in bi or a1 not in bi:
                rad.append(None); continue
            A = rig.r[bi[a0]]/k; Bp = rig.r[bi[a1]]/k; u = _nrm(Bp - A); L = float(np.linalg.norm(Bp - A))
            Q = V[keep & (dom == bi[a0])] - A; t = Q @ u
            Q = Q[(t > 0.25*L) & (t < 0.85*L)]
            rad.append(float(np.percentile(np.linalg.norm(Q - np.outer(Q @ u, u), axis=1), 75)) if len(Q) > 20 else None)
        self.arm_rad = (rad[0] or 0.045, rad[1] or 0.036)
        self.rads = np.r_[np.full(3, 0.7*self.arm_rad[0]), np.full(3, self.arm_rad[1]), 0.006]

    def tf_push(self, D, pos, p, rad, front=None):
        """(N,3) points (scene units, char-local as posed), radius (metres) -> (N,3) displacement out of her torso.
        front: optional (N,) bool = the canon's version of that point is IN FRONT of the canon chest -> push forward
        only (a forearm crossed in front of HER deeper chest must come out the front, never the back or the side)."""
        tf = self.tf
        out = np.zeros_like(p)
        if tf is None:
            return out
        k = self.rig.k; r = self.rig.r
        s1, s3 = tf["s1"], tf["s3"]
        q3 = (p - pos[s3]) @ D[s3] + r[s3]
        q1 = (p - pos[s1]) @ D[s1] + r[s1]
        up = q3[:, 1]/k >= tf["ysplit"]
        q = np.where(up[:, None], q3, q1)/k
        rad = np.broadcast_to(np.asarray(rad, float), (len(p),))
        ins, d4 = self._tf_depths(q, rad)
        if not ins.any():
            return out
        # continuous push: softmin over the four exits (front, back, left, right) -> direction + depth; a point whose
        # canon twin is in front of the canon chest (front weight f in 0..1) is pushed out of the FRONT instead
        e = np.exp(-(d4 - d4.min(1, keepdims=True))/PUSH_TAU)
        wsum = e.sum(1)
        smin = d4.min(1) - PUSH_TAU*np.log(wsum)
        dirv = (e/wsum[:, None]) @ _EXITS
        dn = np.linalg.norm(dirv, axis=1, keepdims=True); dirv = dirv/np.maximum(dn, 1e-9)
        vec = np.maximum(smin - PUSH_TOL, 0.0)[:, None]*dirv
        if front is not None:
            f = np.asarray(front, float)[:, None]
            g = _ssv(0.0, 0.02, np.minimum(np.minimum(d4[:, 1], d4[:, 2]), d4[:, 3]))[:, None]
            fv = np.maximum(d4[:, 0] - PUSH_TOL, 0.0)[:, None]*_EXITS[0]*g
            vec = f*fv + (1.0 - f)*vec
        vec = vec*(ins[:, None]*k)
        if not vec.any():
            return out
        return np.where(up[:, None], vec @ D[s3].T, vec @ D[s1].T)

    def _tf_depths(self, q, rad):
        """q (...,3) rest-space metres, rad (...) -> inside (...) bool, depths (...,4) [front, back, left, right]
        (bilinear field lookups; a point is inside when all four are > 0)."""
        tf = self.tf
        fx = (q[..., 0] - tf["x0"])/TF_STEP; fy = (q[..., 1] - tf["y0"])/TF_STEP
        ok = (fx >= 0) & (fx <= tf["NX"] - 1) & (fy >= 0) & (fy <= tf["NY"] - 1)
        fx = np.clip(fx, 0, tf["NX"] - 1.001); fy = np.clip(fy, 0, tf["NY"] - 1.001)
        i0 = fx.astype(int); j0 = fy.astype(int); u = fx - i0; v = fy - j0
        A = tf["FB"]; uu = u[..., None]; vv = v[..., None]          # F and B in one gather
        FB = (A[j0, i0]*(1 - uu) + A[j0, i0 + 1]*uu)*(1 - vv) + (A[j0 + 1, i0]*(1 - uu) + A[j0 + 1, i0 + 1]*uu)*vv
        Fz = FB[..., 0]; Bz = FB[..., 1]
        xl = tf["xl"][j0]*(1 - v) + tf["xl"][j0 + 1]*v; xr = tf["xr"][j0]*(1 - v) + tf["xr"][j0 + 1]*v
        valid = ok & tf["v"][np.rint(fy).astype(int), np.rint(fx).astype(int)]
        x, z = q[..., 0], q[..., 2]
        rs = np.maximum(rad - SIDE_SLACK, 0.0)        # an arm resting against her SIDE is natural (the skin takes it)
        d4 = np.stack([Fz + rad - z, z - (Bz - rad), x - (xl - rs), xr + rs - x], -1)
        return valid & (d4.min(-1) > 0.0), d4

    def post_may_act(self, D, pos):
        """BATCHED (juno_fk): D (T,nb,3,3), pos (T,nb,3) of the pure rotation copy -> (T,) bool: frames where post_arms'
        torso push or arm layering could act (conservative: the probe with zero tolerance, sphere overlap for arms)."""
        rig = self.rig; T = len(D); flag = np.zeros(T, bool)
        tf = self.tf
        if tf is None or not self.contact_on:
            return flag
        k = rig.k; r = rig.r; s1, s3 = tf["s1"], tf["s3"]
        segs = {}
        for ps, a in ((0, "r"), (1, "l")):
            if a not in self.arm:
                continue
            iu, il, ih = (self.arm[a][p_][0] for p_ in ("upperarm", "lowerarm", "hand"))
            S, E, W = pos[:, iu], pos[:, il], pos[:, ih]
            palm = W + D[:, ih] @ (rig.hand_rest_frame[ps] @ rig.palm_local[ps])
            pts = np.concatenate([S[:, None] + _TU[None, :, None]*(E - S)[:, None], E[:, None] + _TF[None, :, None]*(W - E)[:, None],
                                  palm[:, None]], 1)                                   # (T,7,3)
            q3 = np.einsum("tnj,tjk->tnk", pts - pos[:, s3][:, None], D[:, s3]) + r[s3]
            q1 = np.einsum("tnj,tjk->tnk", pts - pos[:, s1][:, None], D[:, s1]) + r[s1]
            q = np.where((q3[..., 1]/k >= tf["ysplit"])[..., None], q3, q1)/k
            ins, d4 = self._tf_depths(q, np.broadcast_to(self.rads[None], q.shape[:2]))
            ins = ins & (d4[..., 0] > PUSH_TOL - 1e-6)                    # conservative (softmin <= every depth)
            flag |= ins.any(1)
            segs[a] = (S, E, W)
        if len(segs) == 2:
            ru, rf = self.arm_rad
            L, R = segs["l"], segs["r"]
            flag |= _seg_dist_batch(L[1], L[2], R[1], R[2]) < 2*rf*k + 1e-6
            flag |= _seg_dist_batch(L[1], L[2], R[0], R[1]) < (rf + ru)*k + 1e-6
            flag |= _seg_dist_batch(R[1], R[2], L[0], L[1]) < (rf + ru)*k + 1e-6
        return flag

    def tf_any(self, D, pos, p, rad):
        """cheap: does any of the points (N,3) touch her torso (incl. radius)?"""
        return bool(np.any(self.tf_push(D, pos, p, rad)))

    def _aim(self, D, pos, iu, il, ih, dE, dW):
        """rotate the upper arm so the elbow moves by ~dE, then the forearm so the wrist moves by ~dW (hand rides along)."""
        S, E, W = pos[iu], pos[il], pos[ih]
        Wt = W + dW                                    # the wrist goal is fixed BEFORE the upper arm turns
        if float(dE @ dE) > 1e-12:
            R1 = _align(E - S, E + dE - S)
            D[iu] = R1 @ D[iu]; D[il] = R1 @ D[il]; D[ih] = R1 @ D[ih]
            E2 = S + R1 @ (E - S); W = E2 + R1 @ (W - E); E = E2
            pos[il] = E; pos[ih] = W
        if float(dW @ dW) > 1e-12 or float(dE @ dE) > 1e-12:
            R2 = _align(W - E, Wt - E)
            D[il] = R2 @ D[il]; D[ih] = R2 @ D[ih]
            pos[ih] = E + R2 @ (W - E)

    def front_flags(self, st, a):
        """(7,) bool: the canon version of the 7 arm probe points (upper arm .55/.8/1, forearm .35/.7/1, palm) lies in
        front of the canon chest -> those points may only be pushed FORWARD out of her torso."""
        Pc, palm_c = self.canon_points(st)
        Gc = st["G"][self.k_s3]; c3 = Pc[self.k_s3]
        cS, cE, cW = Pc[BJ["shoulder_" + a]], Pc[BJ["elbow_" + a]], Pc[BJ["wrist_" + a]]
        Q = np.vstack([cS + np.outer(_TU, cE - cS), cE + np.outer(_TF, cW - cE), palm_c[a][None]]) - c3
        Q = Q @ Gc
        return _ssv(0.0, 0.05, Q[:, 2])*(1.0 - _ssv(0.12, 0.16, np.abs(Q[:, 0])))     # soft: no pop at the boundary

    def probe_pts(self, a, ps, D, pos):
        rig = self.rig
        iu, il, ih = (self.arm[a][p_][0] for p_ in ("upperarm", "lowerarm", "hand"))
        S, E, W = pos[iu], pos[il], pos[ih]
        palm = W + D[ih] @ (rig.hand_rest_frame[ps] @ rig.palm_local[ps])
        return np.vstack([S + np.outer(_TU, E - S), E + np.outer(_TF, W - E), palm[None]])

    def collide(self, st, a, ps, D, pos, w, d7, ff):
        """push HER upper arm, forearm and palm out of HER torso (mesh height field), from the probe's displacements d7
        (7,3); one pass: the upper arm first (elbow), then the forearm (wrist) re-probed if the upper arm moved."""
        iu, il, ih = (self.arm[a][p_][0] for p_ in ("upperarm", "lowerarm", "hand"))
        v = d7[:3]/_TU[:, None]; n = (v*v).sum(1); m = int(np.argmax(n))
        if n[m] > 0.0:
            self._aim(D, pos, iu, il, ih, v[m]*(w*PUSH_GAIN), np.zeros(3))
            d4 = self.tf_push(D, pos, self.probe_pts(a, ps, D, pos)[3:], self.rads[3:], ff[3:])
        else:
            d4 = d7[3:]
        v = d4/_TW[:, None]; n = (v*v).sum(1); m = int(np.argmax(n))
        if n[m] > 0.0:
            self._aim(D, pos, iu, il, ih, np.zeros(3), v[m]*(w*PUSH_GAIN))

    def arms_layer(self, st, D, pos):
        """crossed / touching forearms: keep the canon's front-to-back order and push the FRONT arm out until the sleeves
        no longer overlap (forearm vs the other forearm and upper arm)."""
        if not all(a in self.arm for a in ("l", "r")):
            return
        wl, wr = st["w"]["arm_l"], st["w"]["arm_r"]
        if wl <= 0.0 and wr <= 0.0:
            return
        ru, rf = self.arm_rad
        k = self.rig.k
        bl = {a: [self.arm[a][p][0] for p in ("upperarm", "lowerarm", "hand")] for a in ("l", "r")}
        def sph(a, i0, i1, r):
            A, B = pos[bl[a][i0]], pos[bl[a][i1]]
            return 0.5*(A + B), 0.5*float(np.linalg.norm(B - A)) + r*k
        fl, fr_ = sph("l", 1, 2, rf), sph("r", 1, 2, rf)
        ul, ur = sph("l", 0, 1, ru), sph("r", 0, 1, ru)
        ov = lambda x, y: float(np.linalg.norm(x[0] - y[0])) < x[1] + y[1]
        if not (ov(fl, fr_) or ov(fl, ur) or ov(fr_, ul)):
            return
        Pc, _ = self.canon_points(st)
        Gc = st["G"][self.k_s3]
        fwd = D[self.s3] @ np.array([0.0, 0.0, 1.0])
        bone = lambda a, p: self.arm[a][p][0]
        for _ in range(2):
            moved = False
            for (a1, p1, r1), (a2, p2, r2) in ((("l", "fore", rf), ("r", "fore", rf)), (("l", "fore", rf), ("r", "upper", ru)),
                                                 (("r", "fore", rf), ("l", "upper", ru))):
                seg = {}
                for a, pp in ((a1, p1), (a2, p2)):
                    i0, i1 = (bone(a, "upperarm"), bone(a, "lowerarm")) if pp == "upper" else (bone(a, "lowerarm"), bone(a, "hand"))
                    k0, k1 = (BJ["shoulder_" + a], BJ["elbow_" + a]) if pp == "upper" else (BJ["elbow_" + a], BJ["wrist_" + a])
                    seg[a] = (pos[i0].copy(), pos[i1].copy(), Pc[k0], Pc[k1])
                A0, A1, cA0, cA1 = seg[a1]; B0, B1, cB0, cB1 = seg[a2]
                s, t, dist = _seg_closest(A0, A1, B0, B1)
                # the sleeves taper to the wrists: hands meeting (clap, hand on the other forearm) are not a crossing
                ra = r1*(1.0 - 0.6*_ss(0.6, 1.0, s)) if p1 == "fore" else r1
                rb = r2*(1.0 - 0.6*_ss(0.6, 1.0, t)) if p2 == "fore" else r2
                need = (ra + rb)*k - dist
                if need <= 0.0:
                    continue
                # canon order at the same parameters, in the canon chest frame: who is in front?
                za = float((Gc.T @ (cA0 + (cA1 - cA0)*s))[2]); zb = float((Gc.T @ (cB0 + (cB1 - cB0)*t))[2])
                front = a1 if za >= zb else a2
                if st["w"]["arm_" + front] <= 0.0:
                    front = a2 if front == a1 else a1
                ww = st["w"]["arm_" + front]
                if ww <= 0.0:
                    continue
                pa = A0 + (A1 - A0)*s; pb = B0 + (B1 - B0)*t
                n = _nrm(pa - pb) if front == a1 else _nrm(pb - pa)
                if float(n @ fwd) < 0.3:
                    n = _nrm(n + fwd*(0.3 - float(n @ fwd))*2.0)
                disp = n*need*ww
                part = p1 if front == a1 else p2
                iu, il, ih = bone(front, "upperarm"), bone(front, "lowerarm"), bone(front, "hand")
                if part == "fore":
                    self._aim(D, pos, iu, il, ih, disp, disp)
                else:
                    tt = s if front == a1 else t
                    self._aim(D, pos, iu, il, ih, disp/max(tt, 0.3), np.zeros(3))
                moved = True
            if not moved:
                break

    def _near_head(self, a, ps, D, pos):
        rig = self.rig; hb = rig.bi["head"]; ih = self.arm[a]["hand"][0]
        palm = pos[ih] + D[ih] @ (rig.hand_rest_frame[ps] @ rig.palm_local[ps])
        hc = pos[hb] + D[hb] @ (rig.head_c - rig.r[hb])
        return float(np.linalg.norm(palm - hc)) < float(rig.head_radii.max()) + (rig.tun["face_pad"] + 0.01)*rig.stature

    def _untwist(self, a, D):
        """forearm twist hygiene: SMPL elbows carry twist (trackers park pronation there, often +-170 deg against an
        opposite wrist twist): the lowerarm becomes a pure hinge relative to the upper arm, the whole roll goes to the
        hand (wrapped to +-180: the hand's world orientation is unchanged) and a share of it back into the forearm
        (juno; junor's own _forearm_roll does that afterwards).  Rotations about the forearm's own axis: no position moves."""
        iu, il, ih = (self.arm[a][p_][0] for p_ in ("upperarm", "lowerarm", "hand"))
        u0 = self.fore_axis[a]
        th = _twist_angle(D[iu].T @ D[il], u0)
        if abs(th) > 1e-6:
            D[il] = D[il] @ _rot(u0, -th)
        if self.roll_share > 0.0:
            th = _twist_angle(D[il].T @ D[ih], u0)
            if abs(th) > 1e-6:
                D[il] = D[il] @ _rot(u0, self.roll_share*th)

    def post_arms(self, st, D, pos):
        """polish, after the bone loop (before the fingers): per canon-driven arm the face / chest contact fit, then
        the torso collision; then the arm-over-arm layering.  Mutates D / pos of clavicle..hand."""
        for a in ("r", "l"):
            if st["w"]["arm_" + a] > 0.0 and a in self.arm:
                self._untwist(a, D)
        if not self.contact_on:
            return
        act = [(ps, a) for ps, a in ((0, "r"), (1, "l")) if st["w"]["arm_" + a] > 0.0 and a in self.arm]
        for ps, a in act:
            if a in st["plan"] or self._near_head(a, ps, D, pos):
                self.contact(st, a, ps, D, pos)
        if self.tf is not None and act:               # ONE vectorised probe for both arms; the push only on a hit
            pts = np.vstack([self.probe_pts(a, ps, D, pos) for ps, a in act])
            dall = self.tf_push(D, pos, pts, np.tile(self.rads, len(act)))
            for m, (ps, a) in enumerate(act):
                d7 = dall[7*m:7*m + 7]
                if not d7.any():
                    continue
                ff = self.front_flags(st, a)          # a hit: redo with the canon's front/back knowledge
                if ff.any():
                    d7 = self.tf_push(D, pos, pts[7*m:7*m + 7], self.rads, ff)
                self.collide(st, a, ps, D, pos, st["w"]["arm_" + a], d7, ff)
                if a in st["plan"]:                   # the push moved the elbow out: put the palm back on its target
                    self.contact(st, a, ps, D, pos)   # (2-bone re-solve keeps the new swivel; hand keeps its turn)
        if self.tf is not None:
            self.arms_layer(st, D, pos)

    # ---------------------------------------------------------------- per solve
    def prepare(self, canon, canon_w=None, modes=None):
        c = np.asarray(canon, np.float64).reshape(-1)
        if c.shape[0] < 135:
            raise ValueError(f"canon pose needs >= 135 values (189 canon1), got {c.shape[0]}")
        c = np.where(np.isfinite(c), c, self.neutral[:c.shape[0]])
        if canon_w is None:
            w = {g: 1.0 for g in GROUPS}
        else:
            w = {g: min(1.0, max(0.0, float(canon_w.get(g, 0.0) or 0.0))) for g in GROUPS}
        md = {"head": "add", "torso": "add"}
        if modes:
            md.update({k: v for k, v in modes.items() if k in md and v in ("add", "abs")})
        G, Rf = canon_globals(c[:135] if c.shape[0] >= 135 else c)
        G = G[0]; Rf = Rf[0]
        st = {"w": w, "modes": md, "G": G, "Rf": Rf, "c": c}
        Dc = {}
        for (b, k) in self.torso + self.head:
            Dc[b] = G[k] @ self.G0[k].T
        st["Dc"] = Dc
        sh = np.clip(c[3:6]/10.0, -0.25, 0.25)*self.rig.k        # decimetres -> metres (clamped) -> scene units
        st["shift"] = sh*w["torso"]
        st["plan"] = self.contact_plan(st)
        return st

    def torso_D(self, st, b, Ddial):
        wt = st["w"]["torso"]
        if wt <= 0.0:
            return Ddial
        Dc = st["Dc"][b]
        if st["modes"]["torso"] == "abs":
            return slerpR(Ddial, Dc, wt)
        return (Dc if wt >= 1.0 else rpow(Dc, wt)) @ Ddial

    def head_D(self, st, b, Ddial, Ds3, Ds3_dial):
        """Ddial: the dial delta (rides the dial torso); Ds3: spine_03 as posed; Ds3_dial: spine_03 from the dials."""
        base = Ds3 @ (Ds3_dial.T @ Ddial) if Ds3 is not Ds3_dial else Ddial
        wh = st["w"]["head"]
        if wh <= 0.0:
            return base
        Dc3 = st["Dc"][self.s3]
        Rel = Dc3.T @ st["Dc"][b]
        if st["modes"]["head"] == "abs":
            return slerpR(base, Ds3 @ Rel, wh)
        return Ds3 @ (Rel if wh >= 1.0 else rpow(Rel, wh)) @ (Ds3_dial.T @ Ddial)

    def arm_D(self, st, a, part, Ddial, Ds3):
        wa = st["w"]["arm_" + a]
        if wa <= 0.0:
            return Ddial
        b, k = self.arm[a][part]
        G = st["G"]
        Rel = G[self.k_s3].T @ G[k] @ self.C[b]       # the arm RELATIVE TO THE CHEST (canon chest ~ Juno's rest chest)
        return slerpR(Ddial, Ds3 @ Rel, wa)

    def fingers(self, st, D, pos):
        """Re-pose every finger bone of a canon-driven hand (relative to the hand as posed) + their FK positions."""
        Rf = st["Rf"]; w = st["w"]
        if not hasattr(self, "_fl2"):
            self._fl2 = []
            for (j, idx, par, sides, dks, off) in self.flev:
                si = np.array([1 if a == "r" else 0 for a in sides])
                ih = np.array([self.arm[a]["hand"][0] for a in sides])
                CT = np.stack([self.Chand[a].T for a in sides]); Cf = np.stack([self.Cf[b] for b in idx])
                self._fl2.append((j, idx, par, sides, si, dks, ih, CT, Cf, off))
            # level j's parents are exactly level j-1's bones in the same order -> reuse the previous level's rotations
            self._chain_ok = all(np.array_equal(self._fl2[q][2], self._fl2[q - 1][1]) for q in range(1, len(self._fl2)))
        cum = [Rf[:, :, 0]]
        cum.append(cum[0] @ Rf[:, :, 1])
        cum.append(cum[1] @ Rf[:, :, 2])
        wf = {"l": w["fingers_l"], "r": w["fingers_r"]}
        prev = None
        for (j, idx, par, sides, si, dks, ih, CT, Cf, off) in self._fl2:
            Dh = np.stack([D[q] for q in ih])
            Dp = Dh if j == 0 else (prev if prev is not None else np.stack([D[q] for q in par]))
            pos[idx] = pos[par] + np.einsum("bij,bj->bi", Dp, off)
            tgt = Dh @ (CT @ cum[j][si, dks] @ Cf)
            prev = tgt if (wf["l"] >= 1.0 and wf["r"] >= 1.0 and self._chain_ok) else None
            for m, b in enumerate(idx):
                ww = wf[sides[m]]
                if ww >= 1.0:
                    D[b] = tgt[m]
                elif ww > 0.0:
                    D[b] = slerpR(D[b], tgt[m], ww)
