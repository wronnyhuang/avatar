#!/usr/bin/env python3
"""Objective measures for EVAL: one COMMON space for every system = the human-unit motion table of the rules
motion statistics (tools/prior/motionlib COLS: head yaw/pitch/roll deg, head x/y/z cm, eye-in-head gaze deg, blinks,
brows, smile, squint, eye-wide, jaw, arm speed).

  * canon outputs (unified, copy-delay, human) -> table by forward kinematics (the data pipeline's SMPL rest skeleton) + blendshapes,
    with motionlib's exact formulas for the face columns ("canon space")
  * rig outputs (rules, gru-prior; and canon systems after the retarget module's map when its rig log exists) -> table by
    tools/motion_stats.py's rig->human conversion (the prior's default gains) ("rig space")
  * the USER's table from the same live detections (face matrix -> head angles, blendshapes -> face columns)

Measures:
  stats       human-range statistics per state (speaking / listening / idle): mstats.person_stats pooled per system vs
              the held-out Seamless TEST people (per-person stats from the data pipeline's canon; median + IQR across people):
              % of cells inside the human IQR, mean |log ratio| (psd cells: |Δlog10|·ln10/2), clip-bootstrap CIs
  respond     event-locked reactions: user smile onset -> avatar smile rise, user nod -> avatar nod, user speech
              start / stop -> avatar gaze shift (head+eye turn), user laugh -> avatar smile; within 0.2-2.0 s:
              hit rate, lift over chance (same window at random times), median latency; + smile cross-correlation
              with the user (avatar follows, 0-2 s) for the real user vs a SHUFFLED user (other clip)
  ablate      input-ablation sensitivity: RMS change of the output table per channel group (head, gaze, face_upper,
              smile) when the user stream is zeroed / shuffled / time-shifted, / the output's own spread
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import E, T, DATA, AVATAR, FPS, atomic_json, canon_spec  # noqa: E402

sys.path.insert(0, str(AVATAR / "tools/prior"))
sys.path.insert(0, str(AVATAR / "tools"))
import mstats  # noqa: E402
import motion_stats as MS  # noqa: E402

COLS = mstats.COLS
C = mstats.C
KEYS = MS.KEYS
STATE_NAMES = ("speaking", "listening", "idle")
GROUPS_T = {"head": [0, 1, 2], "head_pos": [3, 4, 5], "gaze": [6, 7], "face_upper": [8, 9, 10, 11, 13, 14],
            "smile": [12]}


# ============================================================================= tables
def _bs(motion, name):
    spec = canon_spec()
    return motion[:, 135 + spec.BS[name]]


def table_from_canon(motion):
    """(N,189) canon -> (N,17) human-unit table (motionlib conventions: screen sides for an unmirrored camera)."""
    from skeleton_data import fk
    m = np.asarray(motion, np.float64)
    N = len(m)
    P, Rg = fk(m, with_root=True, extra=True)
    R = Rg["head"]
    f = R[:, :, 2]          # forward (z) in the session-median pelvis frame
    u = R[:, :, 1]          # up
    T_ = np.full((N, len(COLS)), np.nan, np.float32)
    T_[:, 0] = np.degrees(np.arctan2(f[:, 0], f[:, 2]))                         # + = person's left = screen right
    T_[:, 1] = np.degrees(np.arctan2(-f[:, 1], np.hypot(f[:, 0], f[:, 2])))     # + = nose down
    T_[:, 2] = np.degrees(np.arctan2(u[:, 0], u[:, 1]))                         # + = crown to screen right
    nose = P["nose"] * 100.0
    T_[:, 3], T_[:, 4], T_[:, 5] = nose[:, 0], nose[:, 1], nose[:, 2]
    T_[:, 6] = np.degrees(m[:, 187]); T_[:, 7] = np.degrees(m[:, 188])
    b = lambda n: _bs(m, n)  # noqa: E731
    # screen-left = the person's RIGHT (MediaPipe ...Right)
    T_[:, 8], T_[:, 9] = b("eyeBlinkRight"), b("eyeBlinkLeft")
    inner = b("browInnerUp")
    T_[:, 10] = np.clip(0.5 * b("browOuterUpRight") + 0.5 * inner - b("browDownRight"), -1, 1)
    T_[:, 11] = np.clip(0.5 * b("browOuterUpLeft") + 0.5 * inner - b("browDownLeft"), -1, 1)
    T_[:, 12] = (b("mouthSmileLeft") + b("mouthSmileRight")) / 2
    T_[:, 13] = (b("eyeSquintLeft") + b("eyeSquintRight")) / 2
    T_[:, 14] = (b("eyeWideLeft") + b("eyeWideRight")) / 2
    T_[:, 15] = b("jawOpen")
    # arm: image-plane wrist speed in shoulder widths / s, 1-s envelope, log1p (the reference's arm column)
    sw = np.linalg.norm(P["shoulder_l"] - P["shoulder_r"], axis=1).mean() + 1e-6
    sp = []
    for w in ("wrist_l", "wrist_r"):
        p = P[w][:, :2] / sw
        v = np.zeros(N); v[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) * FPS
        sp.append(v)
    arm = np.convolve(np.mean(sp, 0), np.ones(30) / 30, "same")
    T_[:, 16] = np.log1p(arm)
    for j in range(8):
        T_[:, j] -= np.nanmedian(T_[:, j])
    return T_


def table_from_rig(rig_path):
    """rig.jsonl -> (table (N,17), per-frame speaking, listening, f0) via tools/motion_stats.py's conversion."""
    rows = MS.load_log(str(rig_path))
    a = Namespace(gain_head=1.3, gain_gaze=1.0, gain_brow=1.3, gain_trans=0.5)
    seg, N = MS.build(rows, a)
    return seg["M"].astype(np.float32), seg


def user_table(clip_dir):
    """the USER's table from the live detections (mp_live.npz): head angles/translation from the face matrix, face
    columns from the blendshapes (gaze from the eyeLook blendshapes: no landmarks kept). NaN where no face."""
    sys.path.insert(0, str(AVATAR / "tools/prior"))
    import motionlib
    z = np.load(Path(clip_dir) / "mp_live.npz")
    bs = z["bs"].astype(np.float64); pres = z["face"].astype(bool)
    yaw, pitch, roll, t = motionlib.euler_from_mat(z["mat"].astype(np.float64))
    N = len(pres)
    T_ = np.full((N, len(COLS)), np.nan, np.float32)
    T_[:, 0], T_[:, 1], T_[:, 2] = yaw, pitch, roll
    T_[:, 3], T_[:, 4], T_[:, 5] = t[:, 0], t[:, 1], t[:, 2]
    spec = canon_spec()
    B = spec.BS
    g = lambda n: bs[:, B[n]]  # noqa: E731
    T_[:, 6] = 25.0 * ((g("eyeLookOutRight") + g("eyeLookInLeft")) / 2 - (g("eyeLookInRight") + g("eyeLookOutLeft")) / 2)
    T_[:, 7] = 25.0 * ((g("eyeLookUpLeft") + g("eyeLookUpRight")) / 2 - (g("eyeLookDownLeft") + g("eyeLookDownRight")) / 2)
    T_[:, 8], T_[:, 9] = g("eyeBlinkRight"), g("eyeBlinkLeft")
    inner = g("browInnerUp")
    T_[:, 10] = np.clip(0.5 * g("browOuterUpRight") + 0.5 * inner - g("browDownRight"), -1, 1)
    T_[:, 11] = np.clip(0.5 * g("browOuterUpLeft") + 0.5 * inner - g("browDownLeft"), -1, 1)
    T_[:, 12] = (g("mouthSmileLeft") + g("mouthSmileRight")) / 2
    T_[:, 13] = (g("eyeSquintLeft") + g("eyeSquintRight")) / 2
    T_[:, 14] = (g("eyeWideLeft") + g("eyeWideRight")) / 2
    T_[:, 15] = g("jawOpen")
    T_[~pres, :16] = np.nan
    for j in range(8):
        if np.isfinite(T_[:, j]).any():
            T_[:, j] -= np.nanmedian(T_[:, j])
    return T_, pres


def blink_onsets(Tb):
    bl = np.nan_to_num(np.maximum(Tb[:, 8], Tb[:, 9]))
    N = len(bl)
    on = np.zeros(N); dur = np.zeros(N)
    above = bl > 0.5
    i = 1
    while i < N:
        if above[i] and not above[i - 1]:
            j = i
            while j < N and above[j]:
                j += 1
            on[i] = 1; dur[i] = (j - i) / FPS + 1 / FPS; i = j
        else:
            i += 1
    return on, dur


# ============================================================================= states
def spans_to_mask(spans, N, key0="t0", key1="t1"):
    m = np.zeros(N, bool)
    for s in spans:
        a, b = (s[key0], s[key1]) if isinstance(s, dict) else (s[0], s[1])
        m[max(0, int(a * FPS)):min(N, int(b * FPS) + 1)] = True
    return m


def clip_states(clip, N):
    """-> st (0 idle, 1 speaking = the avatar's reply, 2 listening = the user speaking), user speaking mask."""
    from systems import utterances
    ann = json.loads(Path(clip["annotation"]).read_text())
    usp = spans_to_mask(ann.get("user_speech") or [], N)
    asp = spans_to_mask(utterances(clip), N)
    st = np.zeros(N, int)
    st[usp] = 2
    st[asp] = 1          # the avatar's own turn wins (it is speaking)
    return st, usp, asp


def make_seg(Tb, st, usp, asp, pM=None, f0=None):
    on, dur = blink_onsets(Tb)
    N = len(Tb)
    return dict(M=Tb, blink=on, blinkdur=dur, st=st, selfsp=asp, partsp=usp,
                f0=np.zeros(N) if f0 is None else f0, pM=pM)


# ============================================================================= system outputs -> tables
def system_table(run, cid, space="auto"):
    """-> (N,17) table or None. space: canon (canon.npz) | rig (rig.jsonl) | auto (canon if present)."""
    od = E / "runs" / run / cid
    if space in ("auto", "canon") and (od / "canon.npz").exists():
        return table_from_canon(np.load(od / "canon.npz")["motion"])
    if space in ("auto", "rig") and (od / "rig.jsonl").exists():
        return table_from_rig(od / "rig.jsonl")[0]
    return None


def fit_len(Tb, N):
    if len(Tb) >= N:
        return Tb[:N]
    pad = np.repeat(Tb[-1:], N - len(Tb), 0)
    return np.concatenate([Tb, pad])


# ============================================================================= reference humans (held-out Seamless)
def reference_people(max_people=None):
    """per-person stats of the Seamless TEST people from the data pipeline's canon (all their landed sessions, valid frames)."""
    spec = canon_spec()
    sp = DATA / "splits.json"
    if not sp.exists():
        return None, {}
    S = json.loads(sp.read_text())
    rows = []
    for k, v in sorted((S.get("sessions") or {}).items()):
        if v == "test" and k.startswith("seamless:"):
            p = DATA / "canon1/seamless" / (k.split(":", 1)[1] + ".npz")
            if p.exists():
                rows.append({"path": str(p), "session": k})
    by_person = {}
    for r in rows:
        try:
            d = spec.load_session(r["path"])
        except Exception:
            continue
        P = d["motion"].shape[0]
        vad = d["audio"][:, :, 41] > 0.5
        for p in range(P):
            pid = d["meta"]["persons"][p]
            mk = d["mask"][p]
            ok = (mk[:, spec.GROUP_NAMES.index("head")] > 0) & (mk[:, spec.GROUP_NAMES.index("face_upper")] > 0)
            own = _bridge(vad[p], int(1.2 * FPS)); oth = vad[1 - p] if P == 2 else np.zeros_like(own)
            st = np.where(own, 1, np.where(oth, 2, 0))
            for a, b in mstats._runs(ok, 300):
                Tb = table_from_canon(d["motion"][p, a:b])
                by_person.setdefault(pid, []).append(make_seg(Tb, st[a:b], oth[a:b], own[a:b]))
        if max_people and len(by_person) >= max_people:
            break
    people = {pid: mstats.person_stats(segs) for pid, segs in by_person.items()}
    ref = mstats.aggregate(list(people.values()))
    return ref, people


def _bridge(v, gap):
    v = v.copy()
    r = mstats._runs(v)
    for (a0, b0), (a1, b1) in zip(r[:-1], r[1:]):
        if a1 - b0 <= gap:
            v[b0:a1] = True
    return v


def compare_ref(stats, ref, keys=KEYS):
    errs, cells = [], []
    for s in STATE_NAMES:
        for k in keys:
            r = (ref.get(s) or {}).get(k); g = (stats.get(s) or {}).get(k)
            if r is None or g is None or not np.isfinite(g):
                continue
            e = abs(g - r["med"]) * math.log(10) / 2 if "psd" in k else abs(math.log((g + 1e-3) / (r["med"] + 1e-3)))
            errs.append(e)
            cells.append(dict(state=s, metric=k, value=g, ref_med=r["med"], ref_q25=r["q25"], ref_q75=r["q75"],
                              inside=bool(r["q25"] <= g <= r["q75"]), log_err=e))
    return (float(np.mean(errs)) if errs else None,
            float(np.mean([c["inside"] for c in cells])) if cells else None, cells)


# ============================================================================= events / responsiveness
def user_events(Ut, usp, N):
    """user events (frame indices) from the user's live table + speech mask."""
    seg = make_seg(np.nan_to_num(Ut), np.zeros(N, int), usp, np.zeros(N, bool))
    ev = mstats.events(seg)
    sm = np.nan_to_num(Ut[:, C["smile"]]); jaw = np.nan_to_num(Ut[:, C["jaw"]])
    laugh_m = (sm > 0.45) & (jaw > 0.2) & usp
    laugh = np.array([a for a, b in mstats._runs(laugh_m, 6)], int)
    on = np.array([a for a, b in mstats._runs(usp, 9) if a >= 15 and not usp[max(0, a - 15):a].any()], int)
    off = np.array([b for a, b in mstats._runs(usp, 9) if b + 15 <= N and not usp[b:b + 15].any()], int)
    return {"smile": ev["smile"], "nod": ev["nod"], "speech_start": on, "speech_stop": off, "laugh": laugh}


def avatar_events(At, st):
    seg = make_seg(np.nan_to_num(At), st, st == 2, st == 1)
    ev = mstats.events(seg)
    return {"smile": ev["smile"], "nod": ev["nod"], "gshift": ev["gshift"], "brow": ev["brow"]}


REACTIONS = [("smile", "smile"), ("nod", "nod"), ("speech_start", "gshift"), ("speech_stop", "gshift"),
             ("speech_stop", "nod"), ("laugh", "smile")]
WIN = (int(0.2 * FPS), int(2.0 * FPS))


def reaction_counts(uev, aev, N):
    """per reaction pair: (n user events, hits, latencies[s], chance hits sum, chance n)."""
    out = {}
    for ue, ae in REACTIONS:
        u = np.asarray(uev[ue], int); a = np.asarray(aev[ae], int)
        u = u[u + WIN[1] < N]
        lat, hits = [], 0
        for t in u:
            w = a[(a >= t + WIN[0]) & (a <= t + WIN[1])]
            if len(w):
                hits += 1; lat.append((w[0] - t) / FPS)
        rnd = np.arange(0, max(1, N - WIN[1]), 5)
        ch = sum(1 for t in rnd if ((a >= t + WIN[0]) & (a <= t + WIN[1])).any())
        out[f"{ue}->{ae}"] = dict(n=int(len(u)), hits=hits, lat=lat, chance_hits=ch, chance_n=int(len(rnd)))
    return out


def smile_xcorr(user_s, av_s, max_lag=60):
    """max correlation of avatar smile with the user's smile, avatar lagging 0..2 s (frames with a user face)."""
    ok = np.isfinite(user_s)
    if ok.sum() < 150:
        return np.nan
    u = np.where(ok, user_s, np.nanmean(user_s)); a = np.nan_to_num(av_s)
    best = -1.0
    for L in range(0, max_lag + 1, 3):
        x = u[:len(u) - L] if L else u; y = a[L:]
        if x.std() < 1e-4 or y.std() < 1e-4:
            continue
        best = max(best, float(np.corrcoef(x, y)[0, 1]))
    return best if best > -1 else np.nan


# ============================================================================= bootstrap
def boot_ci(vals, fn=np.mean, n=2000, seed=7):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)] if v.ndim == 1 else v
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = [fn(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_ratio(items, n=2000, seed=7):
    """items: list of (num, den) per clip -> CI of sum(num)/sum(den) under clip resampling."""
    a = np.asarray(items, float)
    if len(a) < 2 or a[:, 1].sum() == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    r = []
    for _ in range(n):
        s = a[rng.integers(0, len(a), len(a))]
        if s[:, 1].sum() > 0:
            r.append(s[:, 0].sum() / s[:, 1].sum())
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))
