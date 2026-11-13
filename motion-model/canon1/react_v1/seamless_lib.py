"""Copied from the first learned-reaction model's data tools (react_v1); imports patched for the canon1 package.
"""
"""Shared helpers for the Seamless Interaction -> canonical conversion (react_v1).

Reads ONLY pairs listed in the downloader's manifest/done.jsonl. All downloaded files are untrusted data:
numpy with allow_pickle=False, json, soundfile; nothing is executed from the data folder.
"""

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from . import spec_v1 as spec  # noqa: E402
from .. import paths as _paths  # noqa: E402

ROOT = _paths.SEAMLESS                                  # raw Seamless Interaction download ($SEAMLESS_DIR)
CANON = None  # (unused here)
MAPPER = Path(__import__("os").environ.get("MAPPER_DIR", str(_paths.DATA / "mapper")))   # face mapper files (read-only)
MODELS = MAPPER / "models"
MP_RAW = MAPPER / "mp_raw"
SL_BS, SL_HROT, SL_HT = spec.SL_BS, spec.SL_HEAD_ROT, spec.SL_HEAD_T

FAU_NAMES = ["InnerBrowRaiser", "OuterBrowRaiser", "BrowLowerer", "UpperLidRaiser", "CheekRaiser", "LidTightener",
             "NoseWrinkler", "UpperLipRaiser", "LipCornerPuller", "CheekPuffer", "Dimpler", "LipCornerDepressor",
             "LowerLipDepressor", "ChinRaiser", "LipPuckerer", "LipStretcher", "LipFunneler", "LipTightener",
             "LipPressor", "LipsParts", "JawDrop", "LipSuck", "JawSideways", "EyesClosed"]
FAU = {n: i for i, n in enumerate(FAU_NAMES)}
MOV = ["FAUValue", "expression", "gaze_encodings", "head_encodings", "alignment_head_rotation",
       "alignment_translation", "emotion_valence", "emotion_arousal", "emotion_scores", "frame_latent", "is_valid"]
SMPLH = ["body_pose", "global_orient", "left_hand_pose", "right_hand_pose", "translation", "is_valid"]
KP = ["keypoints", "box", "is_valid_box"]
FID_RE = re.compile(r"^V(\d+)_S(\d+)_I(\d+)_P(\d+)(A?)$")


# ------------------------------------------------------------------------------------------- manifest
def read_jsonl(p):
    out = []
    p = Path(p)
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line being written right now
    return out


def queue():
    return read_jsonl(ROOT / "manifest/queue.jsonl")


def done_pairs():
    seen = {}
    for d in read_jsonl(ROOT / "manifest/done.jsonl"):
        if d.get("skip"):
            continue
        seen[d["pair"]] = d
    return list(seen.values())


def person_key(fid):
    """Participant identity used for split disjointness: vendor + numeric id. The trailing 'A' (seen on ~half
    the ids; 49 ids exist both with and without it) is dropped, which is the conservative choice."""
    m = FID_RE.match(fid)
    return f"V{m.group(1)}_P{m.group(4)}"


# ------------------------------------------------------------------------------------------- splits
SPLIT_PATTERN = ["mtrain", "mval", "mtrain", "mtest"]   # ~50/25/25 over connected components in queue order


def split_plan(q=None):
    """Deterministic from the FULL queue (so a pair's split never changes as more data lands).
    Dev pairs (all have video) are grouped into participant-connected components; components in order of first
    appearance in the queue cycle through mapper-train / mapper-val / mapper-train / mapper-test.
    Global split per participant: Meta-test participants -> test; dev mtrain -> val, mval -> val, mtest -> test.
    Pair split = the split of its participants; a pair whose participants disagree (only Meta-train pairs that
    contain a val/test participant, by construction) is dropped.
    Returns (pair -> dict(split, mapper_role or None, drop reason or None), person -> split, notes)."""
    q = q or queue()
    par = {}

    def find(a):
        par.setdefault(a, a)
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    dev = [x for x in q if x["split"] == "dev"]
    for x in dev:
        a, b = [person_key(f) for f in x["files"]]
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
    first = {}
    for x in dev:
        r = find(person_key(x["files"][0]))
        first.setdefault(r, x["idx"])
    comps = sorted(first, key=lambda r: first[r])
    role = {r: SPLIT_PATTERN[i % len(SPLIT_PATTERN)] for i, r in enumerate(comps)}
    comp_id = {r: i for i, r in enumerate(comps)}
    person = {}
    notes = defaultdict(int)
    for x in q:
        if x["split"] == "test":
            for f in x["files"]:
                person[person_key(f)] = "test"
    for x in dev:
        for f in x["files"]:
            pk = person_key(f)
            # mapper-train dev participants go to VAL (not train): the Meta-dev pairs are the only source of val
            # sessions (Meta-test -> test), and without them val would get ~1 session per 100 queued pairs
            s = {"mtrain": "val", "mval": "val", "mtest": "test"}[role[find(pk)]]
            if person.get(pk, s) != s:
                notes["dev_person_also_in_meta_test"] += 1
            person.setdefault(pk, s)
    plan = {}
    for x in q:
        pks = [person_key(f) for f in x["files"]]
        if x["split"] == "train":
            ss = {person.get(p, "train") for p in pks}
        else:
            ss = {person[p] for p in pks}
        if len(ss) == 1:
            s = ss.pop()
            mr = role[find(pks[0])] if x["split"] == "dev" else None
            plan[x["pair"]] = dict(split=s, meta_split=x["split"], mapper_role=mr,
                                   comp=comp_id[find(pks[0])] if x["split"] == "dev" else None, drop=None)
            if x["split"] == "train" and s != "train":
                notes["meta_train_pair_moved"] += 1   # cannot happen with the rules above; logged if it does
        else:
            plan[x["pair"]] = dict(split=None, meta_split=x["split"], mapper_role=None, comp=None,
                                   drop=f"participants span splits {sorted(ss)}")
            notes[f"dropped_{x['split']}_pairs"] += 1
    return plan, person, dict(notes)


# ------------------------------------------------------------------------------------------- loading
def _npy(p):
    if not p.exists() or p.stat().st_size < 200:
        return None
    a = np.load(p, allow_pickle=False)
    return a


def load_file(label, split, fid):
    """All per-frame Seamless arrays of one participant file (None where absent)."""
    d = ROOT / label / split
    o = {"fid": fid, "dir": d}
    for grp in ("movement", "movement_v4"):
        for k in MOV:
            o[f"{grp}/{k}"] = _npy(d / grp / k / f"{fid}.npy")
    for k in SMPLH:
        o[f"smplh/{k}"] = _npy(d / "smplh" / k / f"{fid}.npy")
    for k in KP:
        o[f"kp/{k}"] = _npy(d / "boxes_and_keypoints" / k / f"{fid}.npy")
    o["wav"] = d / "audio" / f"{fid}.wav"
    o["vad"] = read_jsonl(d / "metadata/vad" / f"{fid}.jsonl")
    o["transcript"] = read_jsonl(d / "metadata/transcript" / f"{fid}.jsonl")
    v = d / "video" / f"{fid}.mp4"
    o["video"] = v if v.exists() else None
    lens = [len(a) for k, a in o.items() if isinstance(a, np.ndarray) and a.ndim >= 1 and len(a) > 0]
    o["T"] = int(np.median(lens)) if lens else 0
    return o


def fix_len(a, T, fill=0.0):
    if a is None:
        return None
    a = np.asarray(a)
    if len(a) >= T:
        return a[:T]
    pad = np.full((T - len(a),) + a.shape[1:], fill, dtype=a.dtype)
    return np.concatenate([a, pad], 0)


def movement(o, T):
    """Per frame best Imitator version: v4 where its is_valid, else v3. Returns dict of (T,...) float32 and
    valid (T,) bool, ver (T,) 4/3/0."""
    out = {}
    v4 = fix_len(o.get("movement_v4/is_valid"), T)
    v3 = fix_len(o.get("movement/is_valid"), T)
    v4 = np.zeros(T, bool) if v4 is None else (v4.reshape(T, -1)[:, 0] > 0.5)
    v3 = np.zeros(T, bool) if v3 is None else (v3.reshape(T, -1)[:, 0] > 0.5)
    for k in MOV[:-1]:
        a4 = fix_len(o.get(f"movement_v4/{k}"), T)
        a3 = fix_len(o.get(f"movement/{k}"), T)
        ref = a4 if a4 is not None else a3
        if ref is None:
            out[k] = None
            continue
        ref = ref.reshape(T, -1).astype(np.float32)
        x = np.zeros_like(ref)
        if a4 is not None:
            x[v4] = a4.reshape(T, -1)[v4]
        if a3 is not None:
            m = v3 & ~v4
            x[m] = a3.reshape(T, -1)[m]
        out[k] = x
    out["valid"] = v4 | v3
    out["ver"] = np.where(v4, 4, np.where(v3, 3, 0)).astype(np.int8)
    return out


# ------------------------------------------------------------------------------------------- audio / words
def load_audio16k(wav):
    import soundfile as sf
    x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    x = x.mean(1)
    return spec.resample_to_16k(x, sr), len(x) / sr


def vad_frames(vad, T):
    """[{start,end}] seconds -> (T,) 0/1 at 30 fps; frame k = [k/30,(k+1)/30) is voiced if >= half covered."""
    hi = np.zeros(T * 10 + 10, np.float32)   # 300 Hz grid
    for s in vad:
        a = max(0, int(round(float(s["start"]) * 300))); b = min(len(hi), int(round(float(s["end"]) * 300)))
        if b > a:
            hi[a:b] = 1
    return (hi[:T * 10].reshape(T, 10).mean(1) >= 0.5).astype(np.float32)


def words_list(transcript):
    out = []
    for seg in transcript:
        for w in seg.get("words", []) or []:
            try:
                t0, t1 = float(w["start"]), float(w["end"])
            except (KeyError, TypeError, ValueError):
                continue
            out.append([round(t0, 3), round(t1, 3), str(w.get("word", "")).strip()])
    out.sort(key=lambda r: r[0])
    return out


# ------------------------------------------------------------------------------------------- MediaPipe -> canonical
def mp_raw_to_30(r, T, offset_frames=0.0, joint_vis=False):
    """Raw MediaPipe arrays of one video -> canonical (T,86) motion + (T,4) mask on the Seamless 30 fps grid.
    Each stream (face every frame; pose / hands on the frames where they ran) is resampled on its own timestamps
    (linear for values between valid samples, nearest for masks). Hands are assigned to the anatomical side by
    proximity to the POSE wrists (interpolated to the hand frame's time; no pose within 0.2 s -> hand masked)."""
    w, h = [int(v) for v in r["wh"]]
    N = len(r["face"])
    ts = r["ts_ms"].astype(np.float64) / 1000.0
    allr = np.ones(N, bool)
    pr = r["pose_ran"].astype(bool) if "pose_ran" in r else allr
    hr = r["hand_ran"].astype(bool) if "hand_ran" in r else allr
    mot = np.zeros((N, 86), np.float32)
    msk = np.zeros((N, 4), np.uint8)
    # face stream
    face = r["face"].astype(bool)
    bs = r["bs"].astype(np.float64)
    mot[:, SL_BS] = bs
    rot, tcm = spec.euler_from_mat(np.where(face[:, None], r["mat"], np.eye(4).reshape(1, 16)))
    mot[:, SL_HROT] = np.where(face[:, None], rot, 0)
    mot[:, SL_HT] = np.where(face[:, None], spec.head_translation_feature(tcm, face), 0)
    mot[:, spec.SL_GAZE] = spec.gaze_from_bs(bs)
    msk[:, 0] = face
    out = np.zeros((T, 86), np.float32); m30 = np.zeros((T, 4), np.uint8)
    o, k = resample_native_to_30(mot, msk, ts, T, offset_frames)
    out[:, :60] = o[:, :60]; m30[:, 0] = k[:, 0]
    # pose stream
    pi = np.flatnonzero(pr)
    pose = r["pose"][pi]
    body, ok = spec.body_from_pose(pose, w, h)
    ok = ok & r["pose_ok"][pi].astype(bool)
    pm = np.zeros((len(pi), 86), np.float32); pk = np.zeros((len(pi), 4), np.uint8)
    pm[:, spec.SL_BODY] = body; pk[:, 1] = ok
    o, k = resample_native_to_30(pm, pk, ts[pi], T, offset_frames)
    out[:, 60:78] = o[:, 60:78]; m30[:, 1] = k[:, 1]
    # per-joint visibility as spec.body_from_pose judges it (vis >= 0.5 and inside the frame); hidden joints carry
    # the rest-pose fill, so the mapper must not learn them as positions
    a6 = pose[:, spec.POSE_IDX]
    jv = (a6[..., 3] >= 0.5) & (a6[..., 0] > -0.02) & (a6[..., 0] < 1.02) & (a6[..., 1] > -0.02) & (a6[..., 1] < 1.02)
    jv &= ok[:, None]
    jm = np.zeros((len(pi), 86), np.float32); jm[:, 60:66] = jv
    jk = np.zeros((len(pi), 4), np.uint8); jk[:, 1] = 1
    jv30 = resample_native_to_30(jm, jk, ts[pi], T, offset_frames)[0][:, 60:66] > 0.5
    # hand stream
    hi = np.flatnonzero(hr)
    hm = np.zeros((len(hi), 86), np.float32); hk = np.zeros((len(hi), 4), np.uint8)
    pv = r["pose_ok"][pi].astype(bool)
    if pv.sum() >= 2:
        tp = ts[pi][pv]
        wr = {s_: [np.interp(ts[hi], tp, r["pose"][pi][pv][:, j, c]) for c in (0, 1)] for s_, j in (("l", 15), ("r", 16))}
        near = np.abs(tp[np.clip(np.searchsorted(tp, ts[hi]), 0, len(tp) - 1)] - ts[hi])
        near = np.minimum(near, np.abs(tp[np.clip(np.searchsorted(tp, ts[hi]) - 1, 0, len(tp) - 1)] - ts[hi]))
        hands = r["hands"]
        for q, i in enumerate(hi):
            n = int(r["hand_n"][i])
            if n == 0 or near[q] > 0.2:
                continue
            wl = (wr["l"][0][q] * w, wr["l"][1][q] * h); wrr = (wr["r"][0][q] * w, wr["r"][1][q] * h)
            hw = [(hands[i, kk, 0, 0] * w, hands[i, kk, 0, 1] * h) for kk in range(n)]
            a = spec.assign_hands_to_wrists(hw, wl, wrr)
            for side, sl, mi in (("l", spec.SL_HAND_L, 2), ("r", spec.SL_HAND_R, 3)):
                kk = a[side]
                if kk is None:
                    continue
                H = hands[i, kk].astype(np.float64) * np.array([w, h, w])
                hm[q, sl] = spec.hand_descriptor_side(H, side == "l")
                hk[q, mi] = 1
    o, k = resample_native_to_30(hm, hk, ts[hi], T, offset_frames)
    out[:, 78:86] = o[:, 78:86]; m30[:, 2:4] = k[:, 2:4]
    if joint_vis:
        return out, m30, jv30
    return out, m30


def resample_native_to_30(mot, msk, ts_s, T, offset_frames=0.0):
    """Native-timestamp tracks -> 30 fps grid t_k = (k + offset)/30 (linear for values, nearest for masks).
    Values are interpolated only between frames whose relevant mask is valid (else nearest)."""
    tk = (np.arange(T) + offset_frames) / spec.FPS
    ts = np.asarray(ts_s, np.float64)
    near = np.clip(np.searchsorted(ts, tk), 0, len(ts) - 1)
    prev = np.clip(near - 1, 0, len(ts) - 1)
    pick = np.where(np.abs(ts[prev] - tk) < np.abs(ts[near] - tk), prev, near)
    m30 = msk[pick].copy()
    out = mot[pick].copy()
    j = np.clip(np.searchsorted(ts, tk) - 1, 0, len(ts) - 2)
    a = np.clip((tk - ts[j]) / np.maximum(ts[j + 1] - ts[j], 1e-6), 0, 1)[:, None]
    lin = (1 - a) * mot[j] + a * mot[j + 1]
    for mi, cols in spec.MASK_COVER.items():
        both = (msk[j, mi] > 0) & (msk[j + 1, mi] > 0)
        out[np.ix_(both, cols)] = lin[np.ix_(both, cols)]
    gap = np.median(np.diff(ts)) if len(ts) > 1 else 1.0 / spec.FPS
    outside = (tk < ts[0] - 0.5 * gap - 1e-6) | (tk > ts[-1] + 0.5 * gap + 1e-6)
    m30[outside] = 0
    return out.astype(np.float32), m30.astype(np.uint8)
