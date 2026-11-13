"""Copied from the first learned-reaction model's data tools (react_v1); imports patched for the canon1 package.
"""
"""Seamless per-frame inputs for the mapper (+ the analytic fallback mapping).

feature_blocks(o, T) -> dict name -> (T, d) float32 blocks, plus validity tracks. Blocks are normalized so they
are comparable across people/sessions (keypoints relative to shoulders / eyes / wrists; translations relative
to the session median)."""

import numpy as np

from . import seamless_lib as L
spec = L.spec

UPPER = [2, 5, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]   # body_pose rows: spine1..3, neck, collars, head, arms


def rotmat(aa):
    """(N,3) axis-angle -> (N,9) rotation matrices (Rodrigues)."""
    aa = np.asarray(aa, np.float64)
    th = np.linalg.norm(aa, axis=1, keepdims=True)
    k = aa / np.maximum(th, 1e-9)
    K = np.zeros((len(aa), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, :, None], np.cos(th)[:, :, None]
    R = np.eye(3)[None] + s * K + (1 - c) * (K @ K)
    return R.reshape(-1, 9)


def _med(x, v):
    return np.median(x[v], 0) if v.any() else np.zeros(x.shape[1:])


def feature_blocks(o, T):
    mv = L.movement(o, T)
    B = {}
    val = mv["valid"]
    for k, d in (("FAUValue", 24), ("expression", 128), ("gaze_encodings", 2), ("head_encodings", 3),
                 ("alignment_head_rotation", 3), ("frame_latent", 64), ("emotion_scores", 8)):
        a = mv.get(k)
        B[k] = np.zeros((T, d), np.float32) if a is None else a.reshape(T, -1)[:, :d]
    at = mv.get("alignment_translation")
    at = np.zeros((T, 6), np.float32) if at is None else at.reshape(T, -1)[:, :6]
    B["alignment_translation"] = (at - _med(at, val)).astype(np.float32)
    B["ver"] = np.stack([mv["ver"] == 4, mv["ver"] == 3], 1).astype(np.float32)
    # the OTHER Imitator version (v3 where v4 was used, zeros elsewhere / when invalid) as a second view
    v3 = L.fix_len(o.get("movement/is_valid"), T)
    v3 = np.zeros(T, bool) if v3 is None else (v3.reshape(T, -1)[:, 0] > 0.5)
    alt = []
    for k, dd in (("FAUValue", 24), ("expression", 128), ("gaze_encodings", 2), ("head_encodings", 3)):
        a3 = L.fix_len(o.get(f"movement/{k}"), T)
        x = np.zeros((T, dd), np.float32)
        if a3 is not None:
            m = v3 & (mv["ver"] == 4)
            x[m] = a3.reshape(T, -1)[m, :dd]
        if k == "head_encodings":
            x /= 30.0
        alt.append(x)
    B["mv_alt"] = np.concatenate(alt + [(v3 & (mv["ver"] == 4))[:, None].astype(np.float32)], 1)
    B["head_encodings"] = B["head_encodings"] / 30.0
    # SMPL-H
    sv = L.fix_len(o.get("smplh/is_valid"), T)
    sv = np.zeros(T, bool) if sv is None else sv.reshape(T).astype(bool)
    bp = L.fix_len(o.get("smplh/body_pose"), T)
    go = L.fix_len(o.get("smplh/global_orient"), T)
    lh = L.fix_len(o.get("smplh/left_hand_pose"), T)
    rh = L.fix_len(o.get("smplh/right_hand_pose"), T)
    tr = L.fix_len(o.get("smplh/translation"), T)
    z = lambda d: np.zeros((T, d), np.float32)
    B["smpl_upper"] = z(39) if bp is None else bp[:, UPPER].reshape(T, 39).astype(np.float32)
    B["smpl_orient"] = z(9) if go is None else rotmat(go).astype(np.float32)
    B["smpl_lhand"] = z(45) if lh is None else lh.reshape(T, 45).astype(np.float32)
    B["smpl_rhand"] = z(45) if rh is None else rh.reshape(T, 45).astype(np.float32)
    if tr is None:
        B["smpl_trans"] = z(3)
    else:
        trc = tr.astype(np.float64) - _med(tr.astype(np.float64), sv)
        trc[:, 2] /= 10.0
        B["smpl_trans"] = np.clip(trc, -5, 5).astype(np.float32)
    for k in ("smpl_upper", "smpl_orient", "smpl_lhand", "smpl_rhand", "smpl_trans"):
        B[k][~sv] = 0
    # keypoints (COCO-WholeBody 133, pixels of the 1080x1920 video + score)
    kp = L.fix_len(o.get("kp/keypoints"), T)
    if kp is None:
        kp = np.zeros((T, 133, 3), np.float32)
    kp = kp.astype(np.float64)
    xy, sc = kp[..., :2], kp[..., 2]
    sh_c = 0.5 * (xy[:, 5] + xy[:, 6])
    sw = np.linalg.norm(xy[:, 5] - xy[:, 6], axis=1)
    body_ok = (sc[:, 5] > 0.3) & (sc[:, 6] > 0.3) & (sw > 5)
    swm = np.where(body_ok, sw, 1.0)
    bxy = (xy[:, :13] - sh_c[:, None]) / swm[:, None, None]
    bxy[..., 1] *= -1                                     # y up, like spec.body_from_pose
    B["kp_body"] = np.clip(np.concatenate([bxy.reshape(T, 26), sc[:, :13]], 1), -6, 6).astype(np.float32)
    B["kp_body"][~body_ok, :26] = 0
    # face 68: centred on the eye midpoint, unit = inter-ocular distance
    f = xy[:, 23:91]
    le, re_ = f[:, 42:48].mean(1), f[:, 36:42].mean(1)
    eye_c = 0.5 * (le + re_)
    iod = np.linalg.norm(le - re_, axis=1)
    face_ok = (sc[:, 23:91].mean(1) > 0.3) & (iod > 3)
    iodm = np.where(face_ok, iod, 1.0)
    fxy = (f - eye_c[:, None]) / iodm[:, None, None]
    B["kp_face"] = np.clip(fxy.reshape(T, 136), -6, 6).astype(np.float32)
    B["kp_face"][~face_ok] = 0
    # head position / size relative to the session (for head translation)
    med_c = _med(eye_c, face_ok); med_iod = float(np.median(iod[face_ok])) if face_ok.any() else 1.0
    hp = np.stack([(eye_c[:, 0] - med_c[0]) / med_iod, (eye_c[:, 1] - med_c[1]) / med_iod,
                   np.log(np.maximum(iod, 1e-3) / med_iod), (eye_c[:, 0] - sh_c[:, 0]) / swm,
                   (eye_c[:, 1] - sh_c[:, 1]) / swm], 1)
    hp[~face_ok] = 0
    B["kp_headpos"] = np.clip(hp, -6, 6).astype(np.float32)
    # absolute placement in the (1080x1920 portrait) frame: camera height / distance vs the person sets
    # MediaPipe's perspective head pitch/yaw offsets
    WV, HV = 1080.0, 1920.0
    ab = np.stack([eye_c[:, 0] / WV, eye_c[:, 1] / HV, iod / WV * 10, sh_c[:, 0] / WV, sh_c[:, 1] / HV,
                   sw / WV * 3], 1)
    ab[~(face_ok & body_ok)] = 0
    B["kp_abs"] = np.clip(ab, -6, 6).astype(np.float32)
    # hands: 21 points relative to the hand's wrist, unit = wrist->middle MCP; plus score stats
    hb = []
    for a in (91, 112):
        h = xy[:, a:a + 21]; s = sc[:, a:a + 21]
        size = np.linalg.norm(h[:, 9] - h[:, 0], axis=1)
        ok = (s.mean(1) > 0.2) & (size > 2)
        hx = (h - h[:, :1]) / np.where(ok, size, 1.0)[:, None, None]
        hx[..., 1] *= -1
        hx[~ok] = 0
        hb.append(np.concatenate([np.clip(hx[:, 1:].reshape(T, 40), -6, 6), s.mean(1, keepdims=True),
                                  s.min(1, keepdims=True), np.percentile(s, 25, axis=1)[:, None]], 1))
    B["kp_lhand"] = hb[0].astype(np.float32)
    B["kp_rhand"] = hb[1].astype(np.float32)
    B["valid"] = np.stack([val, sv, body_ok, face_ok], 1).astype(np.float32)
    aux = dict(mv_valid=val, smpl_valid=sv, body_ok=body_ok, face_ok=face_ok, kp=kp, mv=mv)
    return B, aux


# inputs of each mapper group (names of blocks)
GROUP_INPUTS = {
    "face": ["FAUValue", "expression", "gaze_encodings", "head_encodings", "alignment_head_rotation", "ver",
             "kp_face", "valid"],
    "head": ["head_encodings", "alignment_head_rotation", "alignment_translation", "smpl_orient", "smpl_upper",
             "smpl_trans", "kp_headpos", "kp_body", "kp_face", "ver", "valid"],
    "body": ["kp_body", "smpl_upper", "smpl_orient", "kp_headpos", "valid"],
    "hands": ["kp_lhand", "kp_rhand", "smpl_lhand", "smpl_rhand", "kp_body", "smpl_upper", "valid"],
}
# canonical targets of each group (motion column indices) and mask columns predicted by it
GROUP_TARGETS = {
    "face": list(range(0, 52)) + list(range(58, 60)),
    "head": list(range(52, 58)),
    "body": list(range(60, 78)),
    "hands": list(range(78, 86)),
}
GROUP_MASKS = {"face": [0], "head": [], "body": [1], "hands": [2, 3]}
# which measured mask gates the loss of each target column
COL_MASK = np.zeros(86, int)
for _mi, _cols in spec.MASK_COVER.items():
    COL_MASK[_cols] = _mi


def group_X(B, g, inputs=None):
    """inputs of group g (block names from `inputs` or GROUP_INPUTS); when B["A"] (the calibrated analytic map,
    (T,86)) is present its columns for this group are appended (the MLP then predicts a residual on top)."""
    xs = [B[k] for k in (inputs or GROUP_INPUTS)[g]]
    if "A" in B:
        xs.append(B["A"][:, GROUP_TARGETS[g]])
    return np.concatenate(xs, 1)


# --------------------------------------------------------------------------------------------- analytic fallback
# FAU -> MediaPipe blendshapes (name, FAU, gain). Gains are a first guess (FAU 0..~2, blendshapes 0..1).
FAU2BS = [("browInnerUp", "InnerBrowRaiser", .5), ("browOuterUpLeft", "OuterBrowRaiser", .5),
          ("browOuterUpRight", "OuterBrowRaiser", .5), ("browDownLeft", "BrowLowerer", .5),
          ("browDownRight", "BrowLowerer", .5), ("eyeWideLeft", "UpperLidRaiser", .5),
          ("eyeWideRight", "UpperLidRaiser", .5), ("cheekSquintLeft", "CheekRaiser", .5),
          ("cheekSquintRight", "CheekRaiser", .5), ("eyeSquintLeft", "LidTightener", .5),
          ("eyeSquintRight", "LidTightener", .5), ("noseSneerLeft", "NoseWrinkler", .5),
          ("noseSneerRight", "NoseWrinkler", .5), ("mouthUpperUpLeft", "UpperLipRaiser", .5),
          ("mouthUpperUpRight", "UpperLipRaiser", .5), ("mouthSmileLeft", "LipCornerPuller", .5),
          ("mouthSmileRight", "LipCornerPuller", .5), ("cheekPuff", "CheekPuffer", .5),
          ("mouthDimpleLeft", "Dimpler", .5), ("mouthDimpleRight", "Dimpler", .5),
          ("mouthFrownLeft", "LipCornerDepressor", .5), ("mouthFrownRight", "LipCornerDepressor", .5),
          ("mouthLowerDownLeft", "LowerLipDepressor", .5), ("mouthLowerDownRight", "LowerLipDepressor", .5),
          ("mouthShrugLower", "ChinRaiser", .5), ("mouthPucker", "LipPuckerer", .5),
          ("mouthStretchLeft", "LipStretcher", .5), ("mouthStretchRight", "LipStretcher", .5),
          ("mouthFunnel", "LipFunneler", .5), ("mouthPressLeft", "LipPressor", .5),
          ("mouthPressRight", "LipPressor", .5), ("mouthRollLower", "LipSuck", .5), ("mouthRollUpper", "LipSuck", .5),
          ("jawOpen", "JawDrop", .5), ("eyeBlinkLeft", "EyesClosed", .5), ("eyeBlinkRight", "EyesClosed", .5),
          ("mouthClose", "LipTightener", .3)]


def analytic(B, aux, T, calib=None):
    """Hand-written Seamless -> canonical mapping (used before enough dev videos exist).
    calib: optional {col: (a, b)} linear corrections y = a*x + b fitted on measured frames (still per-channel
    analytic, no learned cross-channel terms). Returns motion (T,86), mask (T,4)."""
    mot = np.zeros((T, 86), np.float32)
    fau = B["FAUValue"]
    for bs, fn, g in FAU2BS:
        mot[:, spec.BS[bs]] = g * fau[:, L.FAU[fn]]
    # gaze from gaze_encodings (2) -> canonical gaze and the eyeLook blendshapes
    ge = B["gaze_encodings"]
    mot[:, 58] = 0.5 * ge[:, 0]
    mot[:, 59] = 0.5 * ge[:, 1]
    gx, gy = mot[:, 58], mot[:, 59]
    for n, v in (("eyeLookOutLeft", gx), ("eyeLookInRight", gx), ("eyeLookInLeft", -gx), ("eyeLookOutRight", -gx),
                 ("eyeLookUpLeft", gy), ("eyeLookUpRight", gy), ("eyeLookDownLeft", -gy),
                 ("eyeLookDownRight", -gy)):
        mot[:, spec.BS[n]] = np.clip(v, 0, 1)
    # head: head_encodings (already /30) ~ degrees/30 -> radians
    he = B["head_encodings"] * 30.0 * np.pi / 180.0
    mot[:, 52:55] = he[:, [0, 1, 2]]
    hp = B["kp_headpos"]
    mot[:, 55] = -hp[:, 0] * 0.6
    mot[:, 56] = -hp[:, 1] * 0.6
    mot[:, 57] = hp[:, 2] * 3.0
    # body: 2-D keypoints (x right, y up, unit shoulder width), z = 0
    kb = B["kp_body"][:, :26].reshape(T, 13, 2)
    body = np.zeros((T, 6, 3), np.float32)
    for j, c in enumerate([5, 6, 7, 8, 9, 10]):
        body[:, j, :2] = kb[:, c]
    mot[:, 60:78] = body.reshape(T, 18)
    # hands from 2-D hand keypoints (z = 0)
    for a, sl, is_left in ((B["kp_lhand"], spec.SL_HAND_L, True), (B["kp_rhand"], spec.SL_HAND_R, False)):
        pts = np.concatenate([np.zeros((T, 1, 2)), a[:, :40].reshape(T, 20, 2)], 1)
        pts[..., 1] *= -1
        h3 = np.concatenate([pts, np.zeros((T, 21, 1))], 2)
        mot[:, sl] = np.array([spec.hand_descriptor_side(h3[i], is_left) for i in range(T)], np.float32)
    if calib:
        for c, (a, b) in calib.items():
            mot[:, int(c)] = a * mot[:, int(c)] + b
    mot[:, :52] = np.clip(mot[:, :52], 0, 1)
    msk = np.zeros((T, 4), np.uint8)
    msk[:, 0] = aux["mv_valid"] & aux["face_ok"]
    msk[:, 1] = aux["body_ok"]
    msk[:, 2] = B["kp_lhand"][:, 40] > 0.5
    msk[:, 3] = B["kp_rhand"][:, 40] > 0.5
    return mot, msk
