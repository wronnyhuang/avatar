"""Seamless Interaction -> CANON v1.

Per participant file:
  body (root, spine..wrists) + fingers  <- SMPL-H straight (the canonical skeleton IS SMPL-H): local axis-angle copied;
                                          root_rot = spec.relative_root(global_orient); root_shift = (translation -
                                          median) rotated into the median pelvis frame, decimetres (depth rescaled,
                                          see ROOT_DEPTH_M).
  face blendshapes (52)                 <- the Seamless(Imitator)->MediaPipe mapper (mlp_v1, copied in react_v1/).
  gaze yaw/pitch (radians)              <- the mapped eyeLook blendshapes through GAZE_CALIB (see gaze_from_bs).
  head (neck, head)                     <- HEAD_SOURCE (decided on dev videos vs MediaPipe's face matrix; see log).
  audio (42)                            <- spec.audio_features on the 16 kHz resample of the own-voice wav, vad column
                                          = metadata/vad at 30 fps.
Masks: torso = SMPL-H valid; head = SMPL-H valid & face keypoints seen; arm_* = SMPL-H valid & elbow+wrist keypoints
seen in frame; fingers_* = SMPL-H valid & hand keypoints seen; face_upper / mouth / gaze = mapper face mask
(Imitator valid & face keypoints & mapper validity head).
Invalid frames keep the nearest valid value (no jumps for the model's continuity terms) but are masked 0.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import spec as S
from .react_v1 import seamless_features as F
from .react_v1 import seamless_lib as L

ROOT = L.ROOT
CONVERTER_VERSION = "seamless-canon1-v2"   # v2: axis-angles canonicalized to norm <= pi; iris gaze calib
ROOT_DEPTH_M = 1.5        # SMPL-H translation is weak-perspective (z ~ 37 'm'); depth deltas are rescaled by
#                          ROOT_DEPTH_M / median z, i.e. to a person ~1.5 m from the camera (x/y are already metric)
ROOT_Z_SIGMA = 2.0        # frames; depth is the noisiest translation axis
HEAD_SOURCE = "smplh"     # "smplh" | "imitator" (decided in tools/data/head_gaze_study.py; see log)
# gaze: radians per unit of the eyeLook difference (0.5*((Out-In)L + (In-Out)R)), from the iris-landmark calibration
# (tools/data/gaze_calib.py). Provisional values until that study lands are flagged in meta['gaze_calib'].
# Measured 2026-11-08 on 8 train-side Seamless videos (data/fit/gaze_calib.json): yaw R^2 .82 (gain .4985, bias -.0364);
# pitch: down-gain .27 (R^2 .48); the up-gain is not measurable from iris-vs-eye-corner geometry (the upper lid follows
# the iris) -> set equal to the down-gain; pitch bias 0 by definition (the iris sits above the corner line anatomically).
GAZE_CALIB = {"yaw_gain": 0.4985, "yaw_bias": -0.0364, "pitch_gain_up": 0.27, "pitch_gain_down": 0.27, "pitch_bias": 0.0,
              "status": "iris-calibrated"}


def canon_aa(aa):
    """axis-angle vectors (...,3) -> the equivalent rotation with norm <= pi. Seamless SMPL-H params jump between
    theta and theta-2pi representations (elbows/wrists/fingers up to 3pi/2): canonicalizing removes those jumps
    (elbow_l jumps/1000 frames 6.8 -> 0.004, wrists 6-7 -> 0.5-1.2; the rest are genuine near-pi flips)."""
    aa = np.asarray(aa, np.float64)
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    return np.where(th > np.pi, aa * (1 - 2 * np.pi / np.maximum(th, 1e-9)), aa)

FID_RE = re.compile(r"^V(\d+)_S(\d+)_I(\d+)_P(\d+)(A?)$")


def person_id(fid):
    """global person id: 'seamless:P0059' (vendor V00 only here; the trailing 'A' is collapsed, as in react_v1)."""
    m = FID_RE.match(fid)
    v = int(m.group(1))
    return f"seamless:P{m.group(4)}" if v == 0 else f"seamless:V{m.group(1)}P{m.group(4)}"


def _ffill(x, valid):
    """hold the nearest valid value over invalid frames (forward, then backward for a leading gap)."""
    x = np.array(x, np.float64, copy=True)
    v = np.asarray(valid, bool)
    if v.all() or not v.any():
        return x
    idx = np.where(v, np.arange(len(v)), 0)
    np.maximum.accumulate(idx, out=idx)
    first = int(np.argmax(v))
    idx[:first] = first
    return x[idx]


def gaze_from_bs(bs, calib=None):
    """(T,52) MediaPipe blendshapes -> (T,2) gaze yaw (+ = person's left), pitch (+ = up), radians."""
    c = calib or GAZE_CALIB
    b = np.asarray(bs, np.float64)
    B = S.BS
    gx = 0.5 * ((b[:, B["eyeLookOutLeft"]] - b[:, B["eyeLookInLeft"]]) +
                (b[:, B["eyeLookInRight"]] - b[:, B["eyeLookOutRight"]]))
    gy = 0.5 * ((b[:, B["eyeLookUpLeft"]] - b[:, B["eyeLookDownLeft"]]) +
                (b[:, B["eyeLookUpRight"]] - b[:, B["eyeLookDownRight"]]))
    yaw = c["yaw_gain"] * gx + c["yaw_bias"]
    pitch = np.where(gy >= 0, c["pitch_gain_up"] * gy, c["pitch_gain_down"] * gy) + c["pitch_bias"]
    return np.stack([yaw, pitch], 1)


def kp_masks(kp, T):
    """COCO-WholeBody keypoints (T,133,3) pixels of the 1080x1920 video -> per-frame visibility tracks."""
    if kp is None:
        z = np.zeros(T, bool)
        return dict(arm_l=z, arm_r=z, hand_l=z, hand_r=z, face=z)
    kp = np.asarray(kp, np.float64)
    xy, sc = kp[..., :2], kp[..., 2]
    W, H, mg = 1080.0, 1920.0, 4.0
    inf = (xy[..., 0] > mg) & (xy[..., 0] < W - mg) & (xy[..., 1] > mg) & (xy[..., 1] < H - mg)
    seen = (sc >= 0.5) & inf
    out = {}
    out["arm_l"] = seen[:, 5] & seen[:, 7] & seen[:, 9]
    out["arm_r"] = seen[:, 6] & seen[:, 8] & seen[:, 10]
    for side, a in (("hand_l", 91), ("hand_r", 112)):
        h, s = xy[:, a:a + 21], sc[:, a:a + 21]
        size = np.linalg.norm(h[:, 9] - h[:, 0], axis=1)
        out[side] = (s.mean(1) >= 0.6) & (inf[:, a:a + 21].mean(1) >= 0.9) & (size > 15)
    out["face"] = (sc[:, 23:91].mean(1) >= 0.5) & (inf[:, 23:91].mean(1) >= 0.9)
    return out


def person_canon(o, T, mapper):
    """one participant file (seamless_lib.load_file dict) -> motion (T,189) f32, mask (T,9) u8, info dict."""
    fx = lambda k: L.fix_len(o.get(k), T)
    sv = fx("smplh/is_valid")
    sv = np.zeros(T, bool) if sv is None else sv.reshape(T).astype(bool)
    bp, go, lh, rh, tr = (fx(f"smplh/{k}") for k in ("body_pose", "global_orient", "left_hand_pose",
                                                      "right_hand_pose", "translation"))
    motion = np.zeros((T, S.D_MOTION), np.float64)
    mask = np.zeros((T, S.D_MASK), np.uint8)
    G = {g: i for i, g in enumerate(S.GROUP_NAMES)}
    info = {}
    have_smpl = bp is not None and go is not None and sv.any()
    if have_smpl:
        bp = np.nan_to_num(bp.astype(np.float64)); go = np.nan_to_num(go.astype(np.float64))
        motion[:, 0:3] = S.relative_root(go, sv)
        if tr is not None:
            tr = np.nan_to_num(tr.astype(np.float64))
            med = np.median(tr[sv], 0)
            d = tr - med
            zmed = float(med[2]) if abs(med[2]) > 1e-3 else 1.0
            d[:, 2] *= ROOT_DEPTH_M / zmed
            d[:, 2] = gaussian_filter1d(_ffill(d[:, 2], sv), ROOT_Z_SIGMA, mode="nearest")
            Rm = _median_rot(go, sv)
            motion[:, 3:6] = np.clip((d @ Rm) * 10.0, -10, 10)      # Rm^T d  (row vectors), metres -> dm
            info["root_depth_median_raw"] = round(zmed, 3)
        for j in S.JOINTS[1:]:
            motion[:, S.jcols(j)] = canon_aa(bp[:, S.SMPLH_JOINT_ID[j] - 1])
        if lh is not None:
            motion[:, S.SL_FINGERS_L] = canon_aa(np.nan_to_num(lh.astype(np.float64))).reshape(T, 45)
        if rh is not None:
            motion[:, S.SL_FINGERS_R] = canon_aa(np.nan_to_num(rh.astype(np.float64))).reshape(T, 45)
    km = kp_masks(fx("kp/keypoints"), T)
    mask[:, G["torso"]] = sv
    mask[:, G["head"]] = sv & km["face"]
    mask[:, G["arm_l"]] = sv & km["arm_l"]
    mask[:, G["arm_r"]] = sv & km["arm_r"]
    mask[:, G["fingers_l"]] = sv & km["hand_l"] & (lh is not None)
    mask[:, G["fingers_r"]] = sv & km["hand_r"] & (rh is not None)
    if not have_smpl:
        mask[:, :6] = 0
    # face + gaze from the react_v1 mapper
    B, aux = F.feature_blocks(o, T)
    mot86, msk4 = predict_face(mapper, B, aux, T)
    fv = msk4[:, 0].astype(bool)
    motion[:, S.SL_FACE] = mot86[:, :52]
    motion[:, S.SL_GAZE] = gaze_from_bs(mot86[:, :52])
    mask[:, G["face_upper"]] = fv
    mask[:, G["mouth"]] = fv
    mask[:, G["gaze"]] = fv
    info["imitator_head_enc"] = B["head_encodings"] * 30.0       # degrees (kept for the head study, not saved)
    # hold values over invalid frames
    for g, cols in S.GROUPS.items():
        v = mask[:, G[g]].astype(bool)
        if g in ("torso", "head", "arm_l", "arm_r", "fingers_l", "fingers_r"):
            v = sv  # SMPL-H values are defined wherever SMPL-H is valid, even where the keypoint mask is 0
        motion[:, cols] = _ffill(motion[:, cols], v)
    if not np.isfinite(motion).all():
        bad = ~np.isfinite(motion)
        motion[bad] = 0.0
        mask[bad.any(1)] = 0
    info["valid_frac"] = {g: round(float(mask[:, i].mean()), 4) for i, g in enumerate(S.GROUP_NAMES)}
    info["smplh_valid"] = round(float(sv.mean()), 4)
    info["face_ver"] = {"v4": round(float((aux["mv"]["ver"] == 4).mean()), 4),
                        "v3": round(float((aux["mv"]["ver"] == 3).mean()), 4)}
    return motion.astype(np.float32), mask, info


def _analytic_face(B, T, calib):
    """the face/gaze part of react_v1 seamless_features.analytic (cols 0..51, 58, 59 of its 86-d layout) - the rest
    of that function (hand descriptors) is a slow per-frame loop we do not need."""
    mot = np.zeros((T, 86), np.float32)
    fau = B["FAUValue"]
    for bs, fn, g in F.FAU2BS:
        mot[:, F.spec.BS[bs]] = g * fau[:, L.FAU[fn]]
    ge = B["gaze_encodings"]
    mot[:, 58] = 0.5 * ge[:, 0]
    mot[:, 59] = 0.5 * ge[:, 1]
    gx, gy = mot[:, 58], mot[:, 59]
    for n, v in (("eyeLookOutLeft", gx), ("eyeLookInRight", gx), ("eyeLookInLeft", -gx), ("eyeLookOutRight", -gx),
                 ("eyeLookUpLeft", gy), ("eyeLookUpRight", gy), ("eyeLookDownLeft", -gy),
                 ("eyeLookDownRight", -gy)):
        mot[:, F.spec.BS[n]] = np.clip(v, 0, 1)
    if calib:
        for c, (a, b) in calib.items():
            if int(c) < 52 or int(c) in (58, 59):
                mot[:, int(c)] = a * mot[:, int(c)] + b
    mot[:, :52] = np.clip(mot[:, :52], 0, 1)
    return mot


def predict_face(mapper, B, aux, T):
    """react_v1 Mapper.predict restricted to its 'face' group (52 blendshapes + its gaze pair) and the face mask."""
    if mapper.mlp is None:
        return mapper.predict(B, aux, T)
    mot = np.zeros((T, 86), np.float32)
    msk = np.zeros((T, 4), np.uint8)
    if mapper.cfg.get("residual"):
        B = dict(B, A=_analytic_face(B, T, mapper.cfg["calib"]))
    thr = mapper.cfg.get("mask_thr", {})
    g = "face"
    y, lg = mapper._forward(g, F.group_X(B, g, mapper.cfg.get("group_inputs")))
    mot[:, F.GROUP_TARGETS[g]] = y
    for j, mi in enumerate(F.GROUP_MASKS[g]):
        msk[:, mi] = lg[:, j] > float(thr.get(str(mi), 0.0))
    mot[:, :52] = np.clip(mot[:, :52], 0, 1)
    msk[:, 0] &= (aux["mv_valid"] & aux["face_ok"]).astype(np.uint8)
    return mot, msk


def _median_rot(go, valid):
    R = S.aa_to_mat(go)
    M = np.median(R[valid] if valid.any() else R, 0)
    U, _, Vt = np.linalg.svd(M)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt
    return Rm


def person_audio(o, T):
    vad30 = L.vad_frames(o["vad"], T) if o["vad"] else None
    if o["wav"].exists():
        x16, dur = L.load_audio16k(o["wav"])
        a = S.audio_features(x16, n_frames=T, vad=vad30)
        n_ok = min(T, int(dur * S.FPS))
    else:
        a = np.zeros((T, S.D_AUDIO), np.float32)
        a[:, 40] = -100.0
        n_ok = 0
    return a, n_ok, (vad30 is not None)


def energy_tracks(motion, mask):
    """(T,189),(T,9) -> (T,3) float: head angular speed (rad/s), arm joint speed (rad/s), blendshape L1 speed (/s),
    Gaussian-smoothed (sigma 0.5 s); 0 where the group is masked."""
    G = {g: i for i, g in enumerate(S.GROUP_NAMES)}
    m = motion.astype(np.float64)
    d = np.diff(m, axis=0, prepend=m[:1]) * S.FPS
    head = np.linalg.norm(d[:, S.GROUPS["head"]], axis=1) * (mask[:, G["head"]] > 0)
    arms = np.linalg.norm(d[:, S.GROUPS["arm_l"] + S.GROUPS["arm_r"]], axis=1) * \
        ((mask[:, G["arm_l"]] > 0) & (mask[:, G["arm_r"]] > 0))
    face = np.abs(d[:, S.SL_FACE]).sum(1) * (mask[:, G["face_upper"]] > 0)
    e = np.stack([head, arms, face], 1)
    e = np.minimum(e, np.percentile(e, 99.5, axis=0) + 1e-6)
    return gaussian_filter1d(e, 15, axis=0, mode="nearest")


def session_stats(motion, mask, audio):
    """per-person summary of motion energy for index.jsonl (the model's sampler weights segments with x_energy)."""
    out = []
    for p in range(len(motion)):
        e = energy_tracks(motion[p], mask[p])
        G = {g: i for i, g in enumerate(S.GROUP_NAMES)}
        hv = mask[p][:, G["head"]] > 0
        fv = mask[p][:, G["face_upper"]] > 0
        bs = motion[p][:, S.SL_FACE]
        st = {
            "head_speed_med": round(float(np.median(e[hv, 0])) if hv.any() else 0.0, 4),
            "head_speed_p90": round(float(np.percentile(e[hv, 0], 90)) if hv.any() else 0.0, 4),
            "arm_speed_med": round(float(np.median(e[:, 1])), 4),
            "bs_speed_med": round(float(np.median(e[fv, 2])) if fv.any() else 0.0, 4),
            "bs_var_mean": round(float(bs[fv].var(0).mean()) if fv.sum() > 10 else 0.0, 6),
            "still_frac": round(float((e[hv, 0] < 0.05).mean()) if hv.any() else 1.0, 4),
            "speaking_frac": round(float((audio[p][:, 41] > 0.5).mean()), 4),
            "smile_mean": round(float(bs[fv][:, [S.BS["mouthSmileLeft"], S.BS["mouthSmileRight"]]].mean())
                                if fv.any() else 0.0, 4),
        }
        out.append(st)
    return out


def convert_pair(rec, split, out_path, mapper, video_ids=()):
    """rec: a done.jsonl/queue row. Writes the session npz; returns the index row."""
    label, msplit = rec["label"], rec["split"]
    objs = [L.load_file(label, msplit, f) for f in rec["files"]]
    T = int(min(o["T"] for o in objs))
    if T < S.FPS * 5:
        raise ValueError(f"too short T={T}")
    M, K, A, words, infos, audio_ok = [], [], [], [], [], []
    for o in objs:
        m, k, info = person_canon(o, T, mapper)
        a, n_ok, has_vad = person_audio(o, T)
        info["audio_frames_ok"] = n_ok
        info["vad_source"] = "metadata/vad" if has_vad else "energy"
        M.append(m); K.append(k); A.append(a); infos.append(info)
        words.append(L.words_list(o["transcript"]))
    motion, mask, audio = np.stack(M), np.stack(K), np.stack(A)
    G = S.GROUP_NAMES
    quality = []
    for p in range(len(objs)):
        k = mask[p]
        core = (k[:, G.index("torso")] & k[:, G.index("head")] & k[:, G.index("face_upper")]).astype(bool)
        quality.append(round(float(core.mean()), 4))
    energy = np.stack([energy_tracks(motion[p], mask[p]) for p in range(len(objs))]).astype(np.float16)
    stats = session_stats(motion, mask, audio)
    persons = [person_id(f) for f in rec["files"]]
    has_video = [bool(o["video"] is not None) for o in objs]
    meta = dict(version=S.VERSION, dataset="seamless", session=rec["pair"], persons=persons, split=split,
                source=["smplh+mapper"] * len(objs), quality=quality, speaking_track="vad",
                label=rec["label"], meta_split=msplit, file_ids=list(rec["files"]), has_video=has_video,
                converter=CONVERTER_VERSION, mapper=mapper.version, head_source=HEAD_SOURCE,
                gaze_calib=GAZE_CALIB, root_depth_m=ROOT_DEPTH_M,
                valid_frac=[i["valid_frac"] for i in infos], smplh_valid=[i["smplh_valid"] for i in infos],
                face_ver=[i["face_ver"] for i in infos], vad_source=[i["vad_source"] for i in infos],
                audio_frames_ok=[i["audio_frames_ok"] for i in infos], stats=stats,
                x_energy="(P,T,3) head ang speed rad/s, arm joint speed rad/s, blendshape L1 speed /s; smoothed 0.5 s",
                licence="CC BY-NC 4.0 (Meta Seamless Interaction) - non-commercial research only")
    out_path = Path(out_path)
    tmp = out_path.with_name("." + out_path.stem + ".tmp.npz")
    S.save_session(tmp, motion, mask, audio, meta, words=words, extra={"energy": energy})
    import os
    os.replace(tmp, out_path)
    row = dict(path=str(out_path), dataset="seamless", session=rec["pair"], persons=persons, split=split,
               frames=T, person_hours=round(len(objs) * T / S.FPS / 3600, 5), source=meta["source"],
               quality=quality, label=rec["label"], has_video=has_video, meta_split=msplit, stats=stats,
               valid_frac=meta["valid_frac"], converter=CONVERTER_VERSION)
    return row, [i["imitator_head_enc"] for i in infos]
