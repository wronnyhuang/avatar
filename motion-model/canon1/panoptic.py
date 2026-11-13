"""CMU Panoptic Studio haggling games (SSP processed release, Joo et al. CVPR 2019) -> CANON v1 (auxiliary).

Input: ssp_processed/haggling/panopticDB_{speech,body,face,hand}_pkl_hagglingProcessed.zip (180 games of 3 standing
people: buyer + two sellers; 3D COCO19 body, 70 3D face landmarks, 2 x 21 3D hand points, human speaking labels;
HD frame clock 29.97 fps). Pickles are python2-era; read ONLY through the restricted unpickler below (numpy arrays +
containers), never pickle.load.

Conversion (per game = one session, P = 3: [buyer, left seller, right seller]):
  frames     dome world (cm, y DOWN) -> W = (x, -y, -z) / 100 m (a proper rotation of the dome frame: y up). Resampled
             29.97 -> 30 fps: target frame k sits at source index k*1000/1001 (positions linearly interpolated, a frame is
             valid only if both source neighbours are; speaking labels nearest).
  root       pelvis frame: x = lHip - rHip, y = world up (orthogonalised), z = x cross y (= forward; checked: the nose is in
             front in 100 % of frames and the frame agrees with the SSP bodyNormal, cos 0.998).
             root_rot = spec.relative_root (session median); root_shift = (BodyCenter - median) in the median pelvis
             frame, decimetres, clipped +-10.
  spine      thorax frame: x = lShoulder - rShoulder, up = Neck - BodyCenter, rest-aligned to the SMPL rest direction
             hip-centre -> shoulder-midpoint (UP_REST, 4.4 deg backward tilt). spine1 = spine2 = spine3 =
             log(R_pelvis^T R_thorax) / 3.
  collars    fixed at the Seamless population median (not separable from the spine with shoulder points); the shoulder
             carries the rest of the arm direction (canon1.fit.arm_ik in the collar frame).
  arms       canon1.fit.arm_ik: shoulder swing+twist from the elbow bend plane, elbow flexion about local y (left -theta,
             right +theta), wrist from the 3D hand (wrist -> middle MCP direction + palm normal). arm_* mask = shoulder,
             elbow, wrist joints confident AND the 3D hand valid (else the wrist is unknown -> mask 0).
  head       head frame from rigid face points (eye corners, nose bridge, nose base); head-in-thorax rotation relative to
             the person's session median -> light smoothing -> neck/head split + Seamless baseline exactly as L2L
             (canon1.l2l.head_neck conventions: NECK_SHARE, NECK0, HEAD0). Mask = face + torso valid.
  fingers    mask 0: MANO's hand mean / rest frames are not available here (flat_hand_mean=False convention cannot be
             reproduced analytically) and the fitter's learned hand map (data/fit/fitter_params.npz) did not exist yet.
             The fitter's own finger INPUT (canon1.fit.hand_frame_feats, 63-d per hand) is stored in x_hand_feats with
             x_hand_valid so the fingers can be filled later with one matmul once that map exists.
  face/gaze  mask 0 (70 landmarks do not map honestly to ARKit blendshapes; no eye tracking).
  audio      no per-person audio exists (only a room mix in trimmed montage videos with an undocumented offset):
             mel = 0, loud_db = -100, vad = the human speaking label (speechData indicator). meta audio = "vad-only".
  attention  x_partner_index (P,T) int8 = the other person the head points at (smallest angle between the head forward
             axis and the direction to their face, < ATTEND_DEG; -1 none / face unseen); x_partner_speaker (P,T) int8 =
             the other person speaking now (if both, the one with more speech in the last 2 s; -1 none).
"""
from __future__ import annotations

import io
import json
import math
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import fit as FT
from . import l2l as L2
from . import spec as S

from . import paths as _paths  # noqa: E402

SSP = _paths.PANOPTIC / "ssp_processed/haggling"     # $PANOPTIC_DIR (or $DATASETS_DIR/cmu-panoptic)
ZIPS = {k: SSP / f"panopticDB_{k}_pkl_hagglingProcessed.zip" for k in ("speech", "body", "face", "hand")}
CONVERTER_VERSION = "panoptic-canon1-v1"
SRC_FPS = 30000.0 / 1001.0
BODY_MIN_SCORE = 0.10
FACE_MIN_SCORE = 0.30
HAND_MIN_SCORE = 0.30
HAND_WRIST_TOL_M = 0.10
ATTEND_DEG = 40.0
HEAD_SMOOTH = 1.0
LICENCE = ("CMU Panoptic Studio: research only, no commercial use, no redistribution without the organizers' permission; "
           "SSP processed files CC BY-NC 4.0")
TEST_SEQS_CMU = ("170221_haggling_b1", "170221_haggling_b2", "170221_haggling_b3", "170228_haggling_b1",
                 "170228_haggling_b2", "170228_haggling_b3")   # CMU's own SSP test split (we still put all in train)

# COCO19
NECK, NOSE, BODYC, LSH, LEL, LWR, LHIP, LKNEE, LANK, RSH, REL, RWR, RHIP = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
LEYE, LEAR, REYE, REAR = 15, 16, 17, 18
FACE_RIGID_UP = [27, 28, 36, 39, 42, 45]        # nose bridge + eye corners
FACE_NOSE_BASE = [31, 32, 33, 34, 35]
# SMPL neutral rest: hip joints (from the pelvis) and the shoulder midpoint -> thorax "up" direction at rest
_HIP_L = np.array([0.0695, -0.0914, -0.0042]); _HIP_R = np.array([-0.0677, -0.0905, -0.0043])


def _rest_up():
    from .skeleton import REST
    sp3 = np.sum([REST[j][1] for j in ("spine1", "spine2", "spine3")], 0)
    shl = sp3 + np.add(REST["collar_l"][1], REST["shoulder_l"][1])
    shr = sp3 + np.add(REST["collar_r"][1], REST["shoulder_r"][1])
    v = 0.5 * (shl + shr) - 0.5 * (_HIP_L + _HIP_R)
    return v / np.linalg.norm(v)


UP_REST = _rest_up()


# ----------------------------------------------------------------------------------------------- safe pickle reading
_ALLOWED = {
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("_codecs", "encode"), ("collections", "OrderedDict"),
    ("builtins", "list"), ("builtins", "dict"), ("builtins", "tuple"), ("builtins", "set"),
    ("__builtin__", "list"), ("__builtin__", "dict"), ("__builtin__", "tuple"), ("__builtin__", "set"),
}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name} in untrusted pickle")


def safe_unpickle(data: bytes):
    return _SafeUnpickler(io.BytesIO(data), encoding="latin1").load()


def game_names():
    with zipfile.ZipFile(ZIPS["speech"]) as z:
        return sorted(n.split("/")[-1][:-4] for n in z.namelist() if n.endswith(".pkl"))


def load_game(name, zips=None):
    """-> dict speech/body/face/hand; face or hand = None when that member is missing or unreadable (e.g. one SSP
    hand pickle is truncated); speech and body are required."""
    out = {}
    for k, zp in ZIPS.items():
        z = zips[k] if zips else zipfile.ZipFile(zp)
        try:
            out[k] = safe_unpickle(z.read(f"panopticDB_{k}_pkl_hagglingProcessed/{name}.pkl"))
        except (pickle.UnpicklingError, KeyError, EOFError, ValueError) as e:
            if k in ("speech", "body"):
                raise
            out[k] = None
            out.setdefault("_missing", {})[k] = repr(e)[:120]
        if not zips:
            z.close()
    return out


def _empty_hand(T):
    return dict(hand21=np.zeros((63, T), np.float32), scores=np.zeros((21, T), np.float32))


def _empty_face(T):
    return dict(face70=np.zeros((210, T), np.float32), scores=np.zeros((70, T), np.float32))


# ----------------------------------------------------------------------------------------------- vector helpers
def _nrm(v, eps=1e-9):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def frames_from(a, b):
    """(T,3) first axis a, second b orthogonalised -> (T,3,3) rotation, columns (a, b', a x b')."""
    a = _nrm(a)
    b = _nrm(b - a * np.sum(a * b, -1, keepdims=True))
    return np.stack([a, b, np.cross(a, b)], -1)


def chordal_median(R, valid):
    M = np.median(R[valid] if valid.any() else R, 0)
    U, _, Vt = np.linalg.svd(M)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt
    return Rm


def to_W(x_cm):
    """dome world (cm, y down) -> metres, y up, proper rotation (x, -y, -z)."""
    return np.asarray(x_cm, np.float64) * np.array([1.0, -1.0, -1.0]) / 100.0


def ffill(x, valid):
    x = np.array(x, np.float64, copy=True)
    v = np.asarray(valid, bool)
    if v.all() or not v.any():
        return x
    idx = np.where(v, np.arange(len(v)), 0)
    np.maximum.accumulate(idx, out=idx)
    first = int(np.argmax(v))
    idx[:first] = first
    return x[idx]


# ----------------------------------------------------------------------------------------------- resampling
def resample_index(Ts):
    """target 30 fps frames -> (i0, i1, w) into the 29.97 fps source."""
    T = int(math.floor((Ts - 1) * 1001.0 / 1000.0)) + 1
    pos = np.arange(T) * 1000.0 / 1001.0
    i0 = np.floor(pos).astype(np.int64)
    w = pos - i0
    i1 = np.minimum(i0 + 1, Ts - 1)
    return i0, i1, w, np.clip(np.rint(pos).astype(np.int64), 0, Ts - 1)


def resample(X, valid, idx):
    i0, i1, w, _ = idx
    sh = (-1,) + (1,) * (X.ndim - 1)
    Y = X[i0] * (1 - w).reshape(sh) + X[i1] * w.reshape(sh)
    shv = (-1,) + (1,) * (valid.ndim - 1)
    v = valid[i0] & (valid[i1] | (w <= 1e-9).reshape(shv))
    return Y, v


# ----------------------------------------------------------------------------------------------- per person
def _pad(X, valid, Ts):
    if len(X) >= Ts:
        return X[:Ts], valid[:Ts]
    padX = np.zeros((Ts - len(X),) + X.shape[1:], X.dtype)
    return np.concatenate([X, padX]), np.concatenate([valid, np.zeros(Ts - len(valid), bool)])


def person_tracks(body, face, hl, hr, Ts):
    """SSP subject dicts -> source-rate tracks in W (metres) + validity."""
    J = body["joints19"].T.reshape(-1, 19, 3)[:Ts]
    js = body["scores"].T[:Ts]
    Jw = to_W(J)
    jv = (js > BODY_MIN_SCORE) & (np.abs(J).sum(-1) > 0)
    F = face["face70"].T.reshape(-1, 70, 3)[:Ts]
    fs = face["scores"].T[:Ts]
    rig = FACE_RIGID_UP + FACE_NOSE_BASE
    fv = (fs[:, rig].mean(1) > FACE_MIN_SCORE) & (np.abs(F[:, rig]).sum((1, 2)) > 0)
    Fw = to_W(F)
    hands = {}
    for side, h, wr in (("l", hl, LWR), ("r", hr, RWR)):
        H = h["hand21"].T.reshape(-1, 21, 3)
        hs = h["scores"].T
        hv = (hs.mean(1) > HAND_MIN_SCORE) & (np.abs(H).sum((1, 2)) > 0)
        H, hv = _pad(H, hv, Ts)
        Hw = to_W(H)
        hv = hv & (np.linalg.norm(Hw[:, 0] - Jw[:len(Hw), wr], axis=1) < HAND_WRIST_TOL_M) & jv[:, wr]
        hands[side] = (Hw, hv)
    J_, jv_ = _pad(Jw, jv, Ts)
    F_, fv_ = _pad(Fw, fv, Ts)
    return dict(J=J_, jv=jv_, F=F_, fv=fv_, hands=hands)


def person_canon(tr, neutral):
    """30 fps tracks -> motion (T,189), mask (T,9), aux dict."""
    J, jv, F, fv = tr["J"], tr["jv"], tr["F"], tr["fv"]
    T = len(J)
    G = {g: i for i, g in enumerate(S.GROUP_NAMES)}
    motion = np.tile(neutral.astype(np.float64), (T, 1))
    mask = np.zeros((T, S.D_MASK), np.uint8)
    torso_ok = jv[:, [NECK, BODYC, LSH, RSH, LHIP, RHIP]].all(1)
    up = np.tile([0.0, 1.0, 0.0], (T, 1))
    Rp = frames_from(J[:, LHIP] - J[:, RHIP], up)
    A_rest = frames_from(np.array([[1.0, 0, 0]]), UP_REST[None])[0]
    Rt = frames_from(J[:, LSH] - J[:, RSH], J[:, NECK] - J[:, BODYC]) @ A_rest.T
    # root
    motion[:, 0:3] = S.relative_root(S.mat_to_aa(Rp), torso_ok)
    Rm = chordal_median(Rp, torso_ok)
    pc = J[:, BODYC]
    med = np.median(pc[torso_ok] if torso_ok.any() else pc, 0)
    motion[:, 3:6] = np.clip(((pc - med) @ Rm) * 10.0, -10, 10)
    # spine
    sp = S.mat_to_aa(np.swapaxes(Rp, 1, 2) @ Rt) / 3.0
    for j in ("spine1", "spine2", "spine3"):
        motion[:, S.jcols(j)] = sp
    mask[:, G["torso"]] = torso_ok
    # arms
    hand_feats = np.zeros((T, 2, 63), np.float32)
    hand_valid = np.zeros((T, 2), np.uint8)
    elbow_theta = np.zeros((T, 2))
    for si, (side, i_sh, i_el, i_wr) in enumerate((("l", LSH, LEL, LWR), ("r", RSH, REL, RWR))):
        Rc = S.aa_to_mat(neutral[S.jcols(f"collar_{side}")])
        Hw, hv = tr["hands"][side]
        ok = torso_ok & jv[:, [i_sh, i_el, i_wr]].all(1) & hv
        prev = None
        out = np.zeros((T, 9))
        for k in range(T):
            Rpar = Rt[k] @ Rc
            if hv[k]:
                hp = dict(wrist=Hw[k, 0], mid=Hw[k, 9], index=Hw[k, 5], pinky=Hw[k, 17])
                hd, hn = FT.hand_dir_nrm(hp, side)
                hand_feats[k, si] = FT.hand_frame_feats(Hw[k], side)
                hand_valid[k, si] = 1
            else:
                hd = hn = None
            sh_aa, el_aa, wr_aa, th, bend = FT.arm_ik(J[k, i_sh], J[k, i_el], J[k, i_wr], hd, hn, Rpar, side, prev)
            prev = Rpar.T @ bend
            out[k] = np.concatenate([sh_aa, el_aa, wr_aa])
            elbow_theta[k, si] = th
        armv = torso_ok & jv[:, [i_sh, i_el, i_wr]].all(1)
        out[:, :6] = ffill(out[:, :6], armv)
        out[:, 6:9] = ffill(out[:, 6:9], ok)
        if not ok.any():
            out[:, 6:9] = neutral[S.jcols(f"wrist_{side}")]
        if not armv.any():
            out[:, 0:6] = neutral[S.jcols(f"shoulder_{side}") + S.jcols(f"elbow_{side}")]
        motion[:, S.jcols(f"shoulder_{side}")] = out[:, 0:3]
        motion[:, S.jcols(f"elbow_{side}")] = out[:, 3:6]
        motion[:, S.jcols(f"wrist_{side}")] = out[:, 6:9]
        mask[:, G[f"arm_{side}"]] = ok
    # head
    xh = 0.5 * (F[:, 42] + F[:, 45]) - 0.5 * (F[:, 36] + F[:, 39])
    uh = F[:, FACE_RIGID_UP].mean(1) - F[:, FACE_NOSE_BASE].mean(1)
    Rh = frames_from(xh, uh)
    hv = fv & torso_ok
    Rhit = np.swapaxes(Rt, 1, 2) @ Rh
    head_dev = np.zeros((T, 3))
    if hv.any():
        Rmh = chordal_median(Rhit, hv)
        d = S.mat_to_aa(Rmh.T[None] @ Rhit)
        d = ffill(d, hv)
        if HEAD_SMOOTH > 0 and T > 4:
            d = gaussian_filter1d(d, HEAD_SMOOTH, axis=0, mode="nearest")
        motion[:, S.jcols("neck")] = S.mat_to_aa(S.aa_to_mat(L2.NECK0)[None] @ S.aa_to_mat(d * L2.NECK_SHARE))
        motion[:, S.jcols("head")] = S.mat_to_aa(S.aa_to_mat(L2.HEAD0)[None] @ S.aa_to_mat(d * (1 - L2.NECK_SHARE)))
        head_dev = d
    mask[:, G["head"]] = hv
    # hold torso values over invalid frames
    for cols, v in ((list(range(0, 6)) + S.jcols("spine1") + S.jcols("spine2") + S.jcols("spine3"), torso_ok),):
        motion[:, cols] = ffill(motion[:, cols], v) if v.any() else neutral[cols]
    if not np.isfinite(motion).all():
        bad = ~np.isfinite(motion).all(1)
        motion[bad] = neutral
        mask[bad] = 0
    head_c = np.where(fv[:, None], F[:, FACE_RIGID_UP].mean(1), J[:, NOSE])
    aux = dict(Rh=Rh, head_c=head_c, face_ok=fv, hand_feats=hand_feats, hand_valid=hand_valid, Rp=Rp, Rt=Rt,
               elbow_theta=elbow_theta, head_dev=head_dev)
    return motion, mask, aux


def attention(auxs):
    """x_partner_index (P,T): the other person the head forward axis points at (within ATTEND_DEG), else -1."""
    P = len(auxs)
    T = len(auxs[0]["Rh"])
    out = -np.ones((P, T), np.int8)
    for p in range(P):
        fwd = auxs[p]["Rh"][:, :, 2]
        best = np.full(T, np.inf); arg = -np.ones(T, np.int64)
        for q in range(P):
            if q == p:
                continue
            v = _nrm(auxs[q]["head_c"] - auxs[p]["head_c"])
            ang = np.degrees(np.arccos(np.clip(np.sum(fwd * v, -1), -1, 1)))
            better = ang < best
            best = np.where(better, ang, best); arg = np.where(better, q, arg)
        ok = auxs[p]["face_ok"] & (best < ATTEND_DEG)
        out[p] = np.where(ok, arg, -1)
    return out


def partner_speaker(vad, win=60):
    P, T = vad.shape
    out = -np.ones((P, T), np.int8)
    c = np.cumsum(np.concatenate([np.zeros((P, 1)), vad], 1), 1)
    recent = c[:, 1:] - c[:, np.maximum(np.arange(T) + 1 - win, 0)]
    for p in range(P):
        others = [q for q in range(P) if q != p]
        sp = np.stack([vad[q] > 0.5 for q in others])
        rc = np.stack([recent[q] for q in others])
        pick = np.where(rc[0] >= rc[1], others[0], others[1])
        one = sp.sum(0) == 1
        out[p] = np.where(sp.sum(0) == 0, -1, np.where(one, np.where(sp[0], others[0], others[1]), pick))
    return out


def convert_game(name, neutral, out_dir, split="train", zips=None):
    g = load_game(name, zips)
    sp, bd, fc, hd = g["speech"], g["body"], g["face"], g["hand"]
    seq = name.rsplit("_group", 1)[0]
    buyer = int(sp["buyerId"])
    order = [buyer, int(sp["leftSellerId"]), int(sp["rightSellerId"])]
    assert len(set(order)) == 3, order
    by = lambda lst: {int(d["humanId"]): d for d in (lst or [])}
    B, SD = by(bd["subjects"]), by(sp["speechData"])
    Fc = by(fc["subjects"]) if fc else {}
    HL = by(hd["hand_left"]) if hd else {}
    HR = by(hd["hand_right"]) if hd else {}
    missing = dict(g.get("_missing", {}))
    Ts = min(min(len(SD[h]["indicator"]), B[h]["joints19"].shape[1]) for h in order)
    for h in order:
        if h in Fc:
            Ts = min(Ts, Fc[h]["face70"].shape[1])
    for h in order:
        for nm, dct, mk in (("face", Fc, _empty_face), ("hand_left", HL, _empty_hand), ("hand_right", HR, _empty_hand)):
            if h not in dct:
                dct[h] = mk(Ts)
                missing[f"{nm}:{h}"] = "humanId absent"
    idx = resample_index(Ts)
    T = len(idx[0])
    M, K, A, auxs, infos = [], [], [], [], []
    for h in order:
        tr = person_tracks(B[h], Fc[h], HL[h], HR[h], Ts)
        J, jv = resample(tr["J"], tr["jv"], idx)              # per-joint validity (T,19)
        Fw, fv = resample(tr["F"], tr["fv"], idx)
        hands = {s: resample(*tr["hands"][s], idx) for s in "lr"}
        m, k, aux = person_canon(dict(J=J, jv=jv, F=Fw, fv=fv, hands=hands), neutral)
        vad = (np.asarray(SD[h]["indicator"], np.float64)[:Ts][idx[3]] > 0.5).astype(np.float32)
        a = np.zeros((T, S.D_AUDIO), np.float32)
        a[:, 40] = -100.0
        a[:, 41] = vad
        M.append(m); K.append(k); A.append(a); auxs.append(aux)
        infos.append(dict(hand_valid=[round(float(aux["hand_valid"][:, i].mean()), 4) for i in range(2)]))
    motion, mask, audio = np.stack(M).astype(np.float32), np.stack(K), np.stack(A)
    pidx = attention(auxs)
    pspk = partner_speaker(audio[..., 41])
    hand_feats = np.stack([a["hand_feats"] for a in auxs]).astype(np.float16)
    hand_valid = np.stack([a["hand_valid"] for a in auxs])
    energy = np.stack([L2.energy_tracks(motion[p], mask[p]) for p in range(3)]).astype(np.float16)
    stats = L2.session_stats(motion, mask, audio)
    G = S.GROUP_NAMES
    quality = []
    for p in range(3):
        core = (mask[p][:, G.index("torso")] & mask[p][:, G.index("head")]).astype(bool)
        arms = (mask[p][:, G.index("arm_l")] | mask[p][:, G.index("arm_r")]).astype(bool)
        quality.append(round(float(0.5 * core.mean() + 0.5 * (core & arms).mean()), 4))
        stats[p]["attend_frac"] = round(float((pidx[p] >= 0).mean()), 4)
        stats[p]["elbow_theta_med"] = [round(float(np.median(auxs[p]["elbow_theta"][:, i])), 3) for i in range(2)]
    persons = [f"panoptic:{seq}_{h}" for h in order]
    roles = ["buyer", "seller-left", "seller-right"]
    meta = dict(version=S.VERSION, dataset="panoptic", session=name, persons=persons, split=split,
                source=["panoptic-3d"] * 3, quality=quality, speaking_track="vad",
                vad_source="SSP speechData human speaking labels", audio="vad-only",
                audio_note="no per-person audio exists; mel=0, loud_db=-100, vad = human speaking label",
                label="haggling", roles=roles, human_ids=order, sequence=seq, winner=int(sp.get("winnerId", -1)),
                cmu_ssp_split="test" if seq in TEST_SEQS_CMU else "train",
                game_start_hd_frame=int(sp["startFrame"]), source_fps=round(SRC_FPS, 4),
                fps_note="29.97 -> 30 fps: frame k = source index k*1000/1001, linear interpolation",
                converter=CONVERTER_VERSION, missing_modalities=missing,
                method=dict(root="pelvis frame (hips + world up) relative to session median; shift in median pelvis frame, dm",
                            spine="log(Rpelvis^T Rthorax)/3 each, thorax from shoulders + Neck-BodyCenter, rest-aligned",
                            collars="fixed at the Seamless population median",
                            arms="canon1.fit.arm_ik; wrist from the 3D hand; arm mask needs the hand",
                            head="rigid face points -> head-in-thorax, relative to the person's session median, "
                                 "neck/head split + Seamless baseline as canon1.l2l",
                            fingers="mask 0; fitter inputs kept in x_hand_feats/x_hand_valid"),
                neutral_fill="Seamless population median of each column (valid frames); root = 0",
                valid_frac=[{g: round(float(mask[p][:, i].mean()), 4) for i, g in enumerate(S.GROUP_NAMES)}
                            for p in range(3)],
                hand_valid=[i["hand_valid"] for i in infos], stats=stats,
                x_partner_index="(P,T) int8 the other person (session index) the head points at within "
                                f"{ATTEND_DEG:.0f} deg, -1 none/unseen",
                x_partner_speaker="(P,T) int8 the other person speaking now (both: more speech in last 2 s), -1 none",
                x_hand_feats="(P,T,2,63) float16 canon1.fit.hand_frame_feats of the 3D hand [left, right] (fitter input)",
                x_hand_valid="(P,T,2) uint8",
                x_energy="(P,T,3) head ang speed rad/s, arm joint speed rad/s, blendshape L1 speed /s; smoothed 0.5 s",
                licence=LICENCE)
    words = [[], [], []]                                     # SSP has no transcripts
    out_path = Path(out_dir) / f"{name}.npz"
    tmp = out_path.with_name("." + out_path.stem + ".tmp.npz")
    S.save_session(tmp, motion, mask, audio, meta, words=words,
                   extra=dict(partner_index=pidx, partner_speaker=pspk, hand_feats=hand_feats, hand_valid=hand_valid,
                              energy=energy))
    os.replace(tmp, out_path)
    row = dict(path=str(out_path), dataset="panoptic", session=name, persons=persons, split=split, frames=T,
               person_hours=round(3 * T / S.FPS / 3600, 5), source=meta["source"], quality=quality, label="haggling",
               roles=roles, sequence=seq, audio="vad-only", stats=stats, valid_frac=meta["valid_frac"],
               converter=CONVERTER_VERSION)
    return row, auxs


def _runs(b):
    b = np.asarray(b, bool)
    e = np.flatnonzero(np.diff(np.concatenate([[0], b.astype(np.int8), [0]])))
    return [(int(x), int(y)) for x, y in zip(e[::2], e[1::2])]
