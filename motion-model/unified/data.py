"""Training pack (build from CANON v1 sessions) + on-device batch sampler for the unified model.

Pack (.npz, then zstd): every PERSON is a stream; streams of one session are time-aligned.
  motion f16 (N,189) raw canon, mask u8 (N,9), audio f16 (N,42), wexp f16 (N,) per-frame expressive weight,
  streams i64 (S,7) [start, length, partner_stream|-1, split 0/1/2, dataset_id, ghost 0/1, session_id],
  quality f32 (S,), names json (datasets, sessions, persons), norm: mean/std (189), amean/astd (42) from TRAIN.
A "ghost" stream is a partner with audio only (P==1 clips with x_partner_audio): it is never "self".
phase 2 optional members (pack v3 / slim): events (N,3) u8 {nod, blink, smile} onset labels, a2f (N,52) f16 or a2f_u8
  (N,52) u8 (x/255) Audio2Face teacher (canon blendshape order = motion cols 135..186), a2f_mask (N,) frames the person
  speaks, enc_<name> (N,k) speech-encoder features of the stream's OWN audio (float, or int8 + enc_<name>_scale/_qmean;
  enc_<name>_pca_mean/_pca_V when slim_pack.py reduced them: proj(f) = (f - pca_mean) @ V.T).
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from canon1 import spec  # noqa: E402

SPLITS = {"train": 0, "val": 1, "test": 2}
GI = {g: i for i, g in enumerate(spec.GROUP_NAMES)}
SILENCE_AUDIO = np.concatenate([np.full(40, math.log(1e-6)), [-100.0, 0.0]]).astype(np.float32)

# loss weight per group (spread over the group's columns)
GROUP_W = {"torso": 1.0, "head": 2.0, "arm_l": 1.0, "arm_r": 1.0, "fingers_l": 0.4, "fingers_r": 0.4,
           "face_upper": 1.5, "mouth": 1.0, "gaze": 1.5}


# (c): lip/jaw SHAPE blendshapes - lip sync owns them while the avatar talks, so they are loss-masked (and zeroed
# in the self history / known prefix, normalized units) on frames where the TARGET is speaking. Smiles, frowns, press,
# dimples stay learned.
LIP_NAMES = ["jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthRollLower", "mouthRollUpper",
             "mouthShrugLower", "mouthShrugUpper", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthUpperUpLeft",
             "mouthUpperUpRight", "mouthStretchLeft", "mouthStretchRight"]
LIP_COLS = np.array([135 + spec.BS[n] for n in LIP_NAMES], np.int64)
SPEAK_DILATE = 2            # frames of hangover on each side of the target's VAD for the lip mask
STYLE_NAMES = ["energy", "mirroring", "expressiveness"]
D_STYLE_IN = 6              # 3 values (0 when null) + 3 null flags (learned null per component)


def speaking_mask(vad, dilate=SPEAK_DILATE):
    """numpy: (...,T) vad -> bool (...,T) speaking, dilated by `dilate` frames each side (same rule as the sampler)."""
    v = np.asarray(vad) > 0.5
    if dilate <= 0:
        return v
    out = v.copy()
    for k in range(1, dilate + 1):
        out[..., k:] |= v[..., :-k]
        out[..., :-k] |= v[..., k:]
    return out


def col_weights(gw=None):
    gw = dict(GROUP_W, **(gw or {}))
    w = np.zeros(spec.D_MOTION, np.float32)
    for g, cols in spec.GROUPS.items():
        w[cols] = gw[g] / len(cols)
    return w / w.mean()


# ------------------------------------------------------------------------------------------------ build
def _smooth(x, n):
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.convolve(x, k, "same")


def activity(motion, mask):
    """per-frame activity features of one person: head speed (rad/frame) and face-upper+smile speed."""
    hc = spec.GROUPS["head"]
    fc = spec.GROUPS["face_upper"] + [135 + spec.BS["mouthSmileLeft"], 135 + spec.BS["mouthSmileRight"]]
    d = np.diff(motion, axis=0, prepend=motion[:1])
    hv = np.linalg.norm(d[:, hc], axis=1) * (mask[:, GI["head"]] > 0)
    fv = np.linalg.norm(d[:, fc], axis=1) * (mask[:, GI["face_upper"]] > 0)
    return _smooth(hv, 30), _smooth(fv, 30)


def _load_one(path):
    try:
        return spec.load_session(path)
    except Exception as e:                                            # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def build_pack(sessions, out_path, log=print, max_sessions=None, frac=None, min_seg_s=30.0, seed=0, workers=16):
    """sessions: list of dicts with path + split (+ dataset). Writes out_path (.npz, uncompressed).
    frac: {split: fraction of each session's frames to keep (one random contiguous segment, >= min_seg_s)}."""
    from concurrent.futures import ThreadPoolExecutor
    motion, mask, audio, streams, quality, persons, sess_names, dsets, acts = [], [], [], [], [], [], [], [], []
    n = 0
    hours = {}
    rng = np.random.default_rng(seed)
    frac = frac or {}
    sessions = sessions[:max_sessions] if max_sessions else sessions
    with ThreadPoolExecutor(workers) as ex:
        loaded = ex.map(lambda r: _load_one(r["path"]), sessions)
        for si, (rec, d) in enumerate(zip(sessions, loaded)):
            if isinstance(d, str):
                log(f"skip {rec['path']}: {d}")
                continue
            n = _add_session(rec, d, rng, frac, min_seg_s, n, motion, mask, audio, streams, quality, persons,
                             sess_names, dsets, acts, hours, log)
    return _finish_pack(out_path, motion, mask, audio, streams, quality, persons, sess_names, dsets, acts, hours, log)


def _add_session(rec, d, rng, frac, min_seg_s, n, motion, mask, audio, streams, quality, persons, sess_names, dsets,
                 acts, hours, log):
    if True:
        meta = d["meta"]
        P, T, _ = d["motion"].shape
        if P > 2:
            log(f"skip P={P} session {rec['path']} (v1 uses dyads/monologues only)")
            return n
        if T < 60:
            return n
        split = rec.get("split") or meta.get("split")
        if split not in SPLITS:
            return n
        f = frac.get(split, 1.0)
        if f <= 0:
            return n
        if f < 1.0:
            seg = int(max(min_seg_s * spec.FPS, f * T))
            if seg < T:
                a0 = int(rng.integers(0, T - seg + 1))
                for k in ("motion", "mask", "audio"):
                    d[k] = d[k][:, a0:a0 + seg]
                if "x_partner_audio" in d:
                    d["x_partner_audio"] = np.asarray(d["x_partner_audio"])[:, a0:a0 + seg]
                T = seg
        ds = meta.get("dataset", rec.get("dataset", "?"))
        if ds not in dsets:
            dsets.append(ds)
        sid = len(sess_names)
        sess_names.append(f"{ds}:{meta.get('session')}")
        q = meta.get("quality", [1.0] * P)
        base = len(streams)
        ids = []
        for p in range(P):
            m = np.nan_to_num(d["motion"][p]).astype(np.float32)
            k = d["mask"][p].astype(np.uint8)
            a = np.nan_to_num(d["audio"][p]).astype(np.float32)
            motion.append(m.astype(np.float16)); mask.append(k); audio.append(a.astype(np.float16))
            streams.append([n, T, -1, SPLITS[split], dsets.index(ds), 0, sid])
            quality.append(float(q[p]) if p < len(q) else 1.0)
            persons.append(meta["persons"][p])
            acts.append(activity(m, k))
            ids.append(base + p)
            n += T
            h = k.any(1).sum() / spec.FPS / 3600
            key = f"{ds}/{split}"
            hours[key] = hours.get(key, 0) + h
        if P == 2:
            streams[ids[0]][2] = ids[1]; streams[ids[1]][2] = ids[0]
        elif "x_partner_audio" in d:
            a = np.nan_to_num(np.asarray(d["x_partner_audio"], np.float32)[0])
            motion.append(np.zeros((T, spec.D_MOTION), np.float16)); mask.append(np.zeros((T, 9), np.uint8))
            audio.append(a.astype(np.float16))
            streams.append([n, T, ids[0], SPLITS[split], dsets.index(ds), 1, sid])
            quality.append(0.0); persons.append(meta["persons"][0] + ":partner"); acts.append(None)
            streams[ids[0]][2] = len(streams) - 1
            n += T
    return n


def _finish_pack(out_path, motion, mask, audio, streams, quality, persons, sess_names, dsets, acts, hours, log):
    if True:
        pass
    motion = np.concatenate(motion); mask = np.concatenate(mask); audio = np.concatenate(audio)
    streams = np.asarray(streams, np.int64); quality = np.asarray(quality, np.float32)
    # ---- normalization from TRAIN, non-ghost streams, valid entries only
    tr = [i for i, s in enumerate(streams) if s[3] == 0 and s[5] == 0]
    sel = np.concatenate([np.arange(streams[i, 0], streams[i, 0] + streams[i, 1]) for i in tr])
    if len(sel) > 3_000_000:
        sel = np.random.default_rng(0).choice(sel, 3_000_000, replace=False)
    m = motion[sel].astype(np.float64)
    cm = spec.col_mask(mask[sel]).astype(bool)
    cnt = np.maximum(cm.sum(0), 1)
    mean = np.where(cm, m, 0).sum(0) / cnt
    var = np.where(cm, (m - mean) ** 2, 0).sum(0) / cnt
    std = np.sqrt(var)
    floor = np.full(spec.D_MOTION, 0.02)
    floor[spec.SL_ROOT_SHIFT] = 0.05
    std = np.maximum(std, floor)
    mean[cm.sum(0) == 0] = 0
    a = audio[sel].astype(np.float64)
    a = np.where(np.isfinite(a), a, 0)
    a[:, 40] = np.maximum(a[:, 40], -80)
    amean = a.mean(0); astd = np.maximum(a.std(0), 0.05)
    # ---- expressive weights (rank-based over TRAIN frames)
    wexp = np.ones(len(motion), np.float32)
    hv_all = np.concatenate([acts[i][0] for i in tr]); fv_all = np.concatenate([acts[i][1] for i in tr])
    hq = np.sort(hv_all); fq = np.sort(fv_all)
    h995 = np.quantile(hv_all, 0.995)
    for i, s in enumerate(streams):
        if s[5] == 1:
            wexp[s[0]:s[0] + s[1]] = 0
            continue
        hv, fv = acts[i]
        r = 0.5 * (np.searchsorted(hq, hv) / len(hq)) + 0.5 * (np.searchsorted(fq, fv) / len(fq))
        w = np.where(r < 0.15, 0.15, np.where(r > 0.6, 1.5, 1.0))
        w = np.where(hv > h995, 0.25, w)
        # reaction moments: 0.2..1.5 s after a partner speech on/offset or partner smile onset -> x1.5
        pj = s[2]
        if pj >= 0:
            pa = audio[streams[pj, 0]:streams[pj, 0] + s[1], 41].astype(np.float32)
            ev = np.abs(np.diff(pa, prepend=pa[:1])) > 0.5
            pm = motion[streams[pj, 0]:streams[pj, 0] + s[1]]
            smile = _smooth(pm[:, 135 + spec.BS["mouthSmileLeft"]].astype(np.float32), 5)
            rise = np.zeros_like(smile, bool)
            rise[15:] = smile[15:] - smile[:-15] > 0.15
            ev[1:] |= rise[1:] & ~rise[:-1]
            boost = np.convolve(ev.astype(np.float32), np.ones(40), "full")[: s[1]]
            boost = np.concatenate([np.zeros(6), boost[:-6]]) > 0
            w = np.where(boost, w * 1.5, w)
        valid = mask[s[0]:s[0] + s[1]].any(1)
        wexp[s[0]:s[0] + s[1]] = w * valid
    names = dict(datasets=dsets, sessions=sess_names, persons=persons, hours=hours)
    np.savez(out_path, motion=motion, mask=mask, audio=audio, wexp=wexp.astype(np.float16), streams=streams,
             quality=quality, names=np.array(json.dumps(names)), mean=mean.astype(np.float32),
             std=std.astype(np.float32), amean=amean.astype(np.float32), astd=astd.astype(np.float32))
    log(f"pack {out_path}: {len(streams)} streams, {len(motion)} frames, hours {json.dumps(hours)}")
    return names


def load_index(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            p = os.path.join(p, "index.jsonl")
        base = os.path.dirname(p)
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if not os.path.isabs(r["path"]):
                    r["path"] = os.path.join(base, r["path"])
                out.append(r)
    return out


# ------------------------------------------------------------------------------------------------ style labels
_STYLE_G = {}


def _style_worker(args):
    from canon1 import style as ST
    M, K, st = _STYLE_G["M"], _STYLE_G["K"], _STYLE_G["st"]
    out = []
    for i in args:
        s0, n, pj, sp, _, gh, _ = st[i]
        if gh or n < 60:
            continue
        tr = ST.person_tracks(M[s0:s0 + n].astype(np.float32), K[s0:s0 + n])
        ptr = None
        if pj >= 0 and not st[pj, 5]:
            q0 = st[pj, 0]
            ptr = ST.person_tracks(M[q0:q0 + n].astype(np.float32), K[q0:q0 + n])
        for a, b in ST.grid(int(n)):
            c = ST.segment_comps(tr, ptr, a, b)
            out.append(dict(stream=int(i), a=int(a), b=int(b), split=int(sp),
                            **{k: (None if not np.isfinite(v) else float(v)) for k, v in c.items()}))
    return out


def pack_style(M, K, streams, tables=None, procs=16, log=print):
    """Per-frame STYLE labels for a pack, canon1.style definitions, applied per stream on the
    style grid of that stream (the dyad's two streams share it). tables = canon1.style tables (e.g.
    $MM_DATA_DIR/style/tables.json, so labels mean the same thing across packs) or None = fitted on this pack's TRAIN
    rows. Returns (style u8 (N,3) = round(v*254), 255 = unknown, tables)."""
    from multiprocessing import get_context
    from canon1 import style as ST
    _STYLE_G.update(M=M, K=K, st=np.asarray(streams))
    ids = [i for i in range(len(streams)) if not streams[i][5]]
    chunks = [ids[k::max(1, procs * 4)] for k in range(max(1, procs * 4))]
    try:
        with get_context("fork").Pool(procs) as pool:
            rows = [r for part in pool.map(_style_worker, chunks) for r in part]
    finally:
        _STYLE_G.clear()
    if tables is None:
        tables = ST.fit_tables([r for r in rows if r["split"] == 0])
        tables["sources"] = ["pack"]
    L = ST.label_rows(tables, rows)
    out = np.full((len(M), 3), 255, np.uint8)
    for k, r in enumerate(rows):
        s0 = int(streams[r["stream"]][0])
        for j, lab in enumerate(STYLE_NAMES):
            out[s0 + r["a"]:s0 + r["b"], j] = ST.to_u8(L[lab][k])
    known = (out < 255).mean(0)
    log(f"style labels: {len(rows)} person-segments, known fraction {dict(zip(STYLE_NAMES, known.round(3)))}")
    return out, tables


def style_sidecar_path(pack_path):
    base = pack_path[:-4] if pack_path.endswith(".npz") else pack_path
    return base + ".style.npz"


# ------------------------------------------------------------------------------------------------ phase 2 members
EVENT_NAMES = ["nod", "blink", "smile"]
EVENT_GROUPS = ["head", "face_upper", "mouth"]          # the group an event drives (runtime: no firing while owned)
EVENT_CURVE_LEN = [24, 10, 60]                          # frames of the canonical event curve
EVENT_REFRACTORY = [15, 10, 60]                         # frames between two fires of one class (runtime + calibration; >= curve length for blink/smile, nods may double)
EVENT_RADIUS = [10, 4, 15]                              # rare-event loss weighting: frames around an onset
_NEUTRAL = 135 + spec.BS["_neutral"]
# Audio2Face teacher columns: A2F's VALID dials (audio pipeline's calibration: train-people corr >= .15 = jawOpen,
# mouthLowerDown*, mouthPucker, mouthUpperUp*, mouthSmile*) minus the lip/jaw shapes lip sync owns while the avatar
# speaks -> the smiles. Its upper face is near-static (corr ~0), so brows/blinks/squint are NOT taught.
A2F_DIALS = ["mouthSmileLeft", "mouthSmileRight"]
A2F_COLS = np.array(sorted(135 + spec.BS[n] for n in A2F_DIALS), np.int64)


def event_cols(c):
    """motion columns the canonical curve of event class c moves."""
    if c == 0:                                           # nod: head (+ the gaze counter-movement the data shows)
        cols = set(spec.GROUPS["head"]) | set(spec.GROUPS["gaze"])
    elif c == 1:                                         # blink: upper face
        cols = set(spec.GROUPS["face_upper"])
    else:                                                # smile: mouth + cheeks/eyes, never lip/jaw shapes
        cols = set(spec.GROUPS["mouth"]) | set(spec.GROUPS["face_upper"])
    return np.array(sorted(cols - set(LIP_COLS.tolist()) - {_NEUTRAL}), np.int64)


def onsets(ev, starts=None):
    """(N,C) labels -> u8 onsets (rising edges; a 1-frame label is its own onset; stream starts count as fresh)."""
    e = np.asarray(ev) > 0
    out = e.copy()
    out[1:] &= ~e[:-1]
    if starts is not None:
        st = np.asarray(starts)
        out[st] = e[st]
    return out.astype(np.uint8)


def enc_member(z, name, log=print):
    """pack member enc_<name> -> (float16 (N,k) features, {"epca_mean","epca_V"} or None)."""
    key = f"enc_{name}"
    q = z[key]
    out = np.empty(q.shape, np.float16)
    if q.dtype == np.int8:
        sc = np.asarray(z[key + "_scale"], np.float32)
        qm = np.asarray(z[key + "_qmean"], np.float32) if key + "_qmean" in z.files else np.float32(0)
        for r0 in range(0, len(q), 1_000_000):
            out[r0:r0 + 1_000_000] = (q[r0:r0 + 1_000_000].astype(np.float32) * sc + qm).astype(np.float16)
    else:
        for r0 in range(0, len(q), 1_000_000):
            out[r0:r0 + 1_000_000] = np.nan_to_num(np.asarray(q[r0:r0 + 1_000_000], np.float32), posinf=0, neginf=0)
    pca = None
    if key + "_pca_V" in z.files:
        pca = dict(epca_mean=np.asarray(z[key + "_pca_mean"], np.float32), epca_V=np.asarray(z[key + "_pca_V"], np.float32))
    log(f"enc_{name}: {out.shape} ({q.dtype}{', PCA from ' + str(pca['epca_V'].shape[1]) + ' dims' if pca else ''})")
    return out, pca


def event_curves(M, K, ON, streams, lens=None, max_n=20000, seed=0):
    """Canonical event curves = event-triggered average of the TRAIN motion around onsets: (C, Lmax, 189) raw-unit
    deltas vs the frame before the onset, only event_cols(c), tapered to 0 over the last 30% (an event comes back).
    Also onset rates per minute of valid frames (train / val)."""
    lens = list(lens or EVENT_CURVE_LEN)
    C = ON.shape[1]
    Lmax = max(lens)
    curves = np.zeros((C, Lmax, spec.D_MOTION), np.float32)
    rng = np.random.default_rng(seed)
    info = dict(n_onsets_used=[], rate_train_per_min=[], rate_val_per_min=[])
    for c in range(C):
        L = lens[c]
        g = GI[EVENT_GROUPS[c]]
        cols = event_cols(c)
        idx, rates = [], {}
        for sp in (0, 1):
            n_on = n_fr = 0
            for s0, n, _, spl, _, gh, _ in streams:
                if spl != sp or gh:
                    continue
                okf = K[s0:s0 + n, g] > 0
                o = np.nonzero(ON[s0:s0 + n, c] & okf)[0]
                n_on += len(o); n_fr += int(okf.sum())
                if sp == 0:
                    o = o[(o >= 1) & (o < n - L)]
                    idx.append(o + s0)
            rates[sp] = n_on / max(n_fr / spec.FPS / 60, 1e-9)
        idx = np.concatenate(idx) if idx else np.zeros(0, np.int64)
        if len(idx) > max_n:
            idx = rng.choice(idx, max_n, replace=False)
        acc = np.zeros((L, len(cols)), np.float64)
        nok = 0
        ds = []
        for k0 in range(0, len(idx), 2000):
            ii = idx[k0:k0 + 2000][:, None] + np.arange(-1, L)[None]
            ok = (K[ii, g] > 0).all(1)
            Wv = M[ii[ok]][..., cols].astype(np.float32)                          # (n, L+1, nc)
            d = Wv[:, 1:] - Wv[:, :1]
            acc += d.sum(0)
            ds.append(d)
            nok += int(ok.sum())
        scale = 1.0
        if nok:
            eta = acc / nok
            tap = np.ones(L, np.float32)
            k1 = int(0.7 * L)
            tap[k1:] = 0.5 * (1 + np.cos(np.pi * (np.arange(L - k1) + 1) / (L - k1)))
            # amplitude-preserving: the average smears peaks (event timing jitters), so scale the curve until its
            # main column's peak equals the MEDIAN per-event peak (same sign), capped at 2.5x
            j = int(np.argmax(np.abs(eta).max(0)))
            kp = int(np.argmax(np.abs(eta[:, j])))
            pk = eta[kp, j]
            r = max(1, L // 6)                              # per-event extreme near the average's peak frame only
            dj = np.concatenate([d_[:, max(0, kp - r):kp + r + 1, j] for d_ in ds])
            per = np.median(dj.min(1) if pk < 0 else dj.max(1))
            if abs(pk) > 1e-6 and np.sign(per) == np.sign(pk):
                scale = float(np.clip(per / pk, 1.0, 2.5))
            curves[c, :L][:, cols] = (eta * scale).astype(np.float32) * tap[:, None]
        info.setdefault("amp_scale", []).append(round(scale, 3))
        info["n_onsets_used"].append(nok)
        info["rate_train_per_min"].append(round(rates[0], 3))
        info["rate_val_per_min"].append(round(rates[1], 3))
    info["lens"] = lens
    return curves, info


def ar1_filter_matrix(rho, n):
    """(D,n,n) lower-triangular matrices F with (F @ e)[t] = AR(1) of unit-variance innovations e (n[0] = e[0],
    n[t] = rho n[t-1] + sqrt(1-rho^2) e[t]) - one batched matmul instead of a python loop over frames."""
    rho = np.asarray(rho, np.float64)
    q = np.sqrt(1 - rho * rho)
    t = np.arange(n)
    lag = t[:, None] - t[None, :]                                                  # (n,n)
    P = np.where(lag[None] >= 0, rho[:, None, None] ** np.maximum(lag, 0)[None], 0.0)
    coef = np.where(t[None, None, :] == 0, 1.0, q[:, None, None])                 # innovation 0 has unit weight
    return (P * coef).astype(np.float32)


# ------------------------------------------------------------------------------------------------ compact pack
QSCALE = 256.0          # motion: normalized value * 256, int16, time-delta per stream, byte planes
ASCALE = 16.0           # audio: normalized value * 16, int8
BODY_FACE = np.r_[0:45, 135:189]


def write_qpack(raw_path, out_path, finger_pca=12, log=print):
    """raw pack (.npz from build_pack) -> compact pack: body+face+gaze int16 deltas (sigma/256), fingers as PCA
    coefficients per hand (int16 deltas), audio int8, byte planes column-major. zstd it afterwards."""
    z = np.load(raw_path, allow_pickle=False)
    M, K, Au, st = z["motion"], z["mask"], z["audio"], z["streams"]
    mean, std, amean, astd = z["mean"], z["std"], z["amean"], z["astd"]
    N = len(M)
    starts = st[:, 0]
    tr = np.concatenate([np.arange(s0, s0 + n) for s0, n, _, sp, _, gh, _ in st if sp == 0 and gh == 0])
    rng = np.random.default_rng(0)
    smp_idx = rng.choice(tr, min(len(tr), 400_000), replace=False)
    cols, bases = [], {}
    # body / face / gaze
    Z = ((M[:, BODY_FACE].astype(np.float32) - mean[BODY_FACE]) / std[BODY_FACE])
    cols.append(np.round(np.clip(Z, -120, 120) * QSCALE).astype(np.int32))
    for h, sl in (("l", slice(45, 90)), ("r", slice(90, 135))):
        X = M[smp_idx, sl].astype(np.float64)
        ok = K[smp_idx, spec.GROUP_NAMES.index("fingers_" + h)] > 0
        fm = X[ok].mean(0)
        _, S, Vt = np.linalg.svd(X[ok] - fm, full_matrices=False)
        V = Vt[:finger_pca]
        c = (M[:, sl].astype(np.float64) - fm) @ V.T
        cs = np.maximum(c[smp_idx][ok].std(0), 1e-4)
        ev = float((S[:finger_pca] ** 2).sum() / (S ** 2).sum())
        log(f"fingers_{h}: PCA{finger_pca} explains {ev:.3f}")
        bases[h] = (fm.astype(np.float32), V.astype(np.float32), cs.astype(np.float32))
        cols.append(np.round(np.clip(c / cs, -120, 120) * QSCALE).astype(np.int32))
    Q = np.concatenate(cols, 1)                                     # (N, 99 + 2*P)
    D = np.diff(Q, axis=0, prepend=np.zeros((1, Q.shape[1]), np.int32))
    D[starts[starts > 0]] = Q[starts[starts > 0]]                  # each stream starts absolute
    D = np.clip(D, -32768, 32767).astype(np.int16)
    b = np.ascontiguousarray(D.T).view(np.uint8).reshape(D.shape[1], N, 2)
    a = Au.astype(np.float32).copy()
    a[:, 40] = np.maximum(a[:, 40], -80)
    qa = np.clip(np.round((a - amean) / astd * ASCALE), -127, 127).astype(np.int8)
    w = np.clip(np.round(z["wexp"].astype(np.float32) * 50), 0, 255).astype(np.uint8)
    np.savez(out_path, qformat=np.array("q1"), m_lo=np.ascontiguousarray(b[..., 0]), m_hi=np.ascontiguousarray(b[..., 1]),
             a_q=np.ascontiguousarray(qa.T), mask=np.ascontiguousarray(K.T), wexp=w, streams=st, quality=z["quality"],
             names=z["names"], mean=mean, std=std, amean=amean, astd=astd,
             fl_mean=bases["l"][0], fl_V=bases["l"][1], fl_cs=bases["l"][2],
             fr_mean=bases["r"][0], fr_V=bases["r"][1], fr_cs=bases["r"][2])


def read_qpack(z, colblock=16):
    """compact pack (np.load result) -> motion f16 (N,189), mask u8 (N,9), audio f16 (N,42), wexp f16 (N,).
    Decodes in column blocks with int32 cumsums (low peak memory)."""
    st = z["streams"]
    lo, hi = z["m_lo"], z["m_hi"]
    C, N = lo.shape
    starts = st[:, 0]
    order = np.argsort(starts)
    assert starts[order][0] == 0
    seg_id = np.empty(N, np.int64)
    for i in order:
        seg_id[starts[i]:starts[i] + st[i, 1]] = i
    sp = starts > 0
    mean, std = z["mean"], z["std"]
    nb = len(BODY_FACE)
    Pc = (C - nb) // 2
    M = np.zeros((N, spec.D_MOTION), np.float16)
    fing = {"l": np.zeros((N, Pc), np.float32), "r": np.zeros((N, Pc), np.float32)}
    for c0 in range(0, C, colblock):
        c1 = min(C, c0 + colblock)
        Dq = (lo[c0:c1].astype(np.uint16) | (hi[c0:c1].astype(np.uint16) << 8)).view(np.int16)
        Q = np.cumsum(Dq, axis=1, dtype=np.int32)
        del Dq
        off = np.zeros((c1 - c0, len(st)), np.int32)
        off[:, sp] = Q[:, starts[sp] - 1]
        Q -= off[:, seg_id]
        q = Q.T.astype(np.float32) / QSCALE                                            # (N, cb)
        del Q
        for j, c in enumerate(range(c0, c1)):
            if c < nb:
                col = BODY_FACE[c]
                M[:, col] = (q[:, j] * std[col] + mean[col]).astype(np.float16)
            else:
                h = "l" if c - nb < Pc else "r"
                fing[h][:, (c - nb) % Pc] = q[:, j]
        del q
    for h, sl in (("l", slice(45, 90)), ("r", slice(90, 135))):
        c = fing[h] * z[f"f{h}_cs"]
        for r0 in range(0, N, 2_000_000):
            M[r0:r0 + 2_000_000, sl] = (z[f"f{h}_mean"] + c[r0:r0 + 2_000_000] @ z[f"f{h}_V"]).astype(np.float16)
    del fing, lo, hi
    aq = z["a_q"]                                   # (42, N) int8; decoded in row blocks (low peak memory)
    a = np.empty((N, aq.shape[0]), np.float16)
    astd, amean = z["astd"], z["amean"]
    for r0 in range(0, N, 2_000_000):
        a[r0:r0 + 2_000_000] = ((aq[:, r0:r0 + 2_000_000].T.astype(np.float32) / ASCALE) * astd + amean)
    del aq
    return M, np.ascontiguousarray(z["mask"].T), a, (z["wexp"].astype(np.float32) / 50).astype(np.float16)


# ------------------------------------------------------------------------------------------------ sampler
class MergedNpz:
    """np.load of a pack + extra member files (e.g. enc_<name> features added later); the first file wins.
    An entry may also be an already-open pack object (np.load result, ConcatPack)."""

    def __init__(self, paths):
        self.zs = [np.load(p, allow_pickle=False) if isinstance(p, (str, os.PathLike)) else p for p in paths]
        self.files = []
        for z in self.zs:
            self.files += [f for f in z.files if f not in self.files]

    def __getitem__(self, k):
        for z in self.zs:
            if k in z.files:
                return z[k]
        raise KeyError(k)


# scaling sweep: several pack files trained as ONE pack (nested increment packs inc25 + inc50 + inc100, + CANDOR)
_STREAM_OK = ("events_ok",)                     # per-stream ok vectors besides enc_<name>_ok (0 = labels missing)


class ConcatPack:
    """Pack files concatenated by STREAMS (unlike MergedNpz, which merges members). Every qpack is decoded with its
    OWN norms into raw motion/mask/audio/wexp (raw units are pack-independent), then concatenated: per-frame members
    (style, events, a2f*, enc_<name>), per-stream members (quality, *_ok), streams (starts / partner indices offset,
    session ids shifted only when they collide, dataset ids remapped by dataset NAME). Norms (mean/std/amean/astd) and other
    globals come from the FIRST pack. enc_<name> int8 members stay int8 when every pack has the same _scale/_qmean,
    else they are dequantized and re-quantized to the first pack's scale. A per-frame member missing from some pack
    is filled (enc/a2f/events zeros with the stream's ok flag 0, style 255 = unknown); lean_no_mel = any pack's.
    Exposes .files / [key] like np.load (no "qformat": the Sampler takes the raw-member path)."""

    FRAME_FILL = {"style": 255}

    def __init__(self, paths, log=print):
        zs = [_open_entry(p) for p in paths]
        self.paths = list(paths)
        sts = [np.asarray(z["streams"]) for z in zs]
        Ns = [int((s[:, 0] + s[:, 1]).max()) if len(s) else 0 for s in sts]
        N, S = sum(Ns), sum(len(s) for s in sts)
        offN = np.cumsum([0] + Ns)[:-1]
        offS = np.cumsum([0] + [len(s) for s in sts])[:-1]
        names = [json.loads(str(z["names"])) for z in zs]
        dsets = []
        for nm in names:
            dsets += [d for d in nm.get("datasets", []) if d not in dsets]
        m = {}
        motion = np.zeros((N, spec.D_MOTION), np.float16)
        mask = np.zeros((N, 9), np.uint8)
        audio = np.zeros((N, 42), np.float16)
        wexp = np.zeros(N, np.float16)
        streams = np.zeros((S, 7), np.int64)
        quality = np.zeros(S, np.float32)
        sess_off, seen_sess = 0, set()                   # session ids stay unique (increments keep theirs)
        frame_keys, stream_keys = set(), set()
        for z in zs:                             # by NAME (np.load reads a member in full on access)
            for k in z.files:
                if k.endswith("_ok") or k in _STREAM_OK:
                    stream_keys.add(k)
                elif k in _FRAME_KEYS or (k.startswith("enc_") and not k.endswith(DERIVED_ENC) and k != "enc_for"):
                    frame_keys.add(k)
        for i, (z, s) in enumerate(zip(zs, sts)):
            a, n, b = int(offN[i]), Ns[i], int(offS[i])
            if "qformat" in z.files:
                M_, K_, A_, W_ = read_qpack(z)
            else:
                M_, K_, A_, W_ = z["motion"], z["mask"], z["audio"], z["wexp"]
            motion[a:a + n] = M_; mask[a:a + n] = K_; audio[a:a + n] = A_; wexp[a:a + n] = W_
            del M_, K_, A_, W_
            ss = s.copy()
            ss[:, 0] += a
            ss[:, 2] = np.where(s[:, 2] >= 0, s[:, 2] + b, -1)
            if len(s) and seen_sess and not seen_sess.isdisjoint(np.unique(s[:, 6]).tolist()):
                ss[:, 6] = s[:, 6] + sess_off                # ids collide (another source): shift them
            dl = names[i].get("datasets", [])
            ss[:, 4] = [dsets.index(dl[d]) if d < len(dl) else d for d in s[:, 4]]
            if len(ss):
                seen_sess.update(np.unique(ss[:, 6]).tolist())
                sess_off = max(sess_off, int(ss[:, 6].max()) + 1)
            streams[b:b + len(s)] = ss
            quality[b:b + len(s)] = np.asarray(z["quality"], np.float32)
        m.update(motion=motion, mask=mask, audio=audio, wexp=wexp, streams=streams, quality=quality)
        z0 = zs[0]
        for k in ("mean", "std", "amean", "astd"):
            m[k] = np.asarray(z0[k])
        # per-stream ok vectors: the pack's own, else 1 if the pack has the frame member, else 0 (zeros filled in)
        for k in sorted(stream_keys | {f + "_ok" for f in frame_keys if f.startswith("enc_")}):
            v = np.zeros(S, np.uint8)
            base = k[:-3]
            for i, z in enumerate(zs):
                b, ns = int(offS[i]), len(sts[i])
                if k in z.files:
                    v[b:b + ns] = np.asarray(z[k]).reshape(-1)[:ns]
                elif base in z.files:
                    v[b:b + ns] = 1
            m[k] = v
        for k in sorted(frame_keys):
            ref = next(z[k] for z in zs if k in z.files)
            if k.startswith("enc_") and np.asarray(ref).dtype == np.int8:
                m.update(self._enc_int8(zs, Ns, offN, k, N, log))
                continue
            shp = (N,) + tuple(ref.shape[1:])
            out = np.full(shp, self.FRAME_FILL.get(k, 0), ref.dtype)
            for i, z in enumerate(zs):
                if k in z.files:
                    a, n = int(offN[i]), Ns[i]
                    out[a:a + n] = np.asarray(z[k])[:n]
            m[k] = out
            if k.startswith("enc_"):                  # float features: their PCA basis travels with them
                zp = next(z for z in zs if k in z.files)
                for sfx in ("_pca_mean", "_pca_V", "_pca_ev"):
                    if k + sfx in zp.files:
                        m[k + sfx] = np.asarray(zp[k + sfx])
        for k in z0.files:                       # globals of the first pack (incl. enc_<name>_lookahead, tables)
            if k not in m and k not in ("m_lo", "m_hi", "a_q", "qformat") and k not in frame_keys and k not in stream_keys:
                if not k.endswith(DERIVED_ENC) or k.endswith("_lookahead"):
                    m[k] = z0[k]
        if any("lean_no_mel" in z.files and int(np.asarray(z["lean_no_mel"])) for z in zs):
            m["lean_no_mel"] = np.array(1)
        hours = {}
        for nm in names:
            for k, v in (nm.get("hours") or {}).items():
                hours[k] = hours.get(k, 0.0) + float(v)
        nm0 = dict(names[0], datasets=dsets, hours=hours, concat=[_entry_name(p) for p in paths])
        m["names"] = np.array(json.dumps(nm0))
        self._m = m
        self.files = list(m)
        log(f"concat pack: {len(paths)} packs -> {S} streams, {N} frames ({', '.join(_entry_name(p) for p in paths)})")

    @staticmethod
    def _enc_int8(zs, Ns, offN, k, N, log):
        z0 = next(z for z in zs if k in z.files)
        D = z0[k].shape[1]
        sc0 = np.asarray(z0[k + "_scale"], np.float32)
        qm0 = np.asarray(z0[k + "_qmean"], np.float32) if k + "_qmean" in z0.files else np.zeros(D, np.float32)
        out = np.zeros((N, D), np.int8)
        for i, z in enumerate(zs):
            if k not in z.files:
                continue
            a, n = int(offN[i]), Ns[i]
            q = z[k]
            if q.dtype == np.int8:
                sc = np.asarray(z[k + "_scale"], np.float32)
                qm = np.asarray(z[k + "_qmean"], np.float32) if k + "_qmean" in z.files else np.zeros(D, np.float32)
            else:                                              # float features (e.g. a CANDOR pack): quantize them
                sc, qm = np.ones(D, np.float32), np.zeros(D, np.float32)
            if q.dtype == np.int8 and np.array_equal(sc, sc0) and np.array_equal(qm, qm0):
                out[a:a + n] = q[:n]
            else:
                log(f"concat pack: {k} of pack {i} ({q.dtype}) quantized to the first pack's int8 scale")
                for r0 in range(0, n, 1_000_000):
                    x = np.nan_to_num(np.asarray(q[r0:min(n, r0 + 1_000_000)], np.float32)) * sc + qm
                    out[a + r0:a + r0 + len(x)] = np.clip(np.round((x - qm0) / sc0), -127, 127).astype(np.int8)
        r = {k: out, k + "_scale": sc0, k + "_qmean": qm0}
        for sfx in ("_pca_mean", "_pca_V"):
            if k + sfx in z0.files:
                r[k + sfx] = np.asarray(z0[k + sfx])
        return r

    def __getitem__(self, k):
        return self._m[k]


DERIVED_ENC = ("_scale", "_qmean", "_pca_mean", "_pca_V", "_pca_ev", "_lookahead")
_FRAME_KEYS = {"style", "events", "a2f", "a2f_u8", "a2f_mask"}       # per-frame members besides enc_<name>


def _open_entry(p):
    """one pack entry: a path (np.load) or a list of paths = ONE pack whose members are split over files (e.g. the
    base increment + its encoder-feature file; the first file wins on a shared member)."""
    if isinstance(p, (str, os.PathLike)):
        return np.load(p, allow_pickle=False)
    return MergedNpz(list(p)) if len(p) > 1 else np.load(p[0], allow_pickle=False)


def _entry_name(p):
    return os.path.basename(str(p)) if isinstance(p, (str, os.PathLike)) else "+".join(os.path.basename(str(x)) for x in p)


def open_pack(pack_path, pack_extra=None, log=print):
    """pack_path: a path, or a list of entries (a path, or a list of member files of one pack); several entries =
    ConcatPack (streams concatenated) -> pack object (+ pack_extra members merged)."""
    entries = [pack_path] if isinstance(pack_path, (str, os.PathLike)) else list(pack_path)
    base = ConcatPack(entries, log=log) if len(entries) > 1 else _open_entry(entries[0])
    return MergedNpz([base] + list(pack_extra)) if pack_extra else base


class Sampler:
    """Holds a pack on a device and produces training batches (all on-device, vectorized)."""

    def __init__(self, pack_path, device, noise_profile=None, H=15, L=6, mem=30, seed=0, style_tables=None,
                 style_procs=16, log=print, norm=None, audio_in="mel", enc_shift=0, use_events=False, use_a2f=False,
                 event_tol=1, event_curve_len=None, pack_on_cpu=False, pack_extra=None, a2f_dials=None, lean=False,
                 enc_missing="silence", dataset_weights=None):
        """norm: {mean,std,amean,astd(,emean,estd,esil)} to use INSTEAD of the pack's own (continuing a model on
        another pack). phase 2: audio_in = "mel" (audio42) | "enc_<name>" (pack member; token audio = normalized
        features + loud_db/vad of the same person, A = k + 2); enc_shift = frames the USER's features are delayed
        (the encoder's look-ahead: live, frame f's feature exists only at f + shift); use_events = event-head targets +
        canonical curves from the `events` member; use_a2f = the Audio2Face teacher (a2f / a2f_u8 + a2f_mask).
        pack_on_cpu: the per-frame pack arrays stay in host RAM (gathered per batch, then copied to `device`) - lets
        several concurrent runs share one GPU (the pack on the GPU is ~1.8 GB per process for 3 M frames).
        scaling sweep: pack_path may be a LIST of pack files (ConcatPack: streams concatenated, first pack's norms).
        enc_<name>_ok (S,) per stream (0 = no features for that person, e.g. L2L/Panoptic without whisper features):
        enc_missing = "silence" (those frames get the silence feature, as a muted partner does) | "exclude" (such
        streams are never sampled as SELF; as a partner they are silence). lean = keep only loud_db/vad of the
        42-d audio on the device (enc runs never read the 40 mel columns) and int8 encoder features dequantized per
        gather -> ~460 instead of ~610 bytes per frame; refused with audio_in "mel". dataset_weights {name: w}
        multiplies the stream sampling probability (length x quality) of that dataset (e.g. a small CANDOR set)."""
        import torch
        self.torch = torch
        z = open_pack(pack_path, pack_extra, log=log)
        pack0 = pack_path
        while not isinstance(pack0, (str, os.PathLike)):
            pack0 = list(pack0)[0]
        self.dev = device
        store = "cpu" if pack_on_cpu else device                 # where the per-frame arrays live
        self.store = store
        self._devt = torch.empty(0, device=device).device         # canonical device (cuda -> cuda:0) for _take
        T = torch
        if "qformat" in z.files:
            M, K, Au, W = read_qpack(z)
        else:
            M, K, Au, W = z["motion"], z["mask"], z["audio"], z["wexp"]
        self.streams = z["streams"]
        files = set(z.files)
        starts = self.streams[:, 0]
        # ---- phase 2: speech-encoder features (normalized over TRAIN frames; silence feature for missing audio)
        self.audio_in = "mel" if audio_in in (None, "", "mel") else ("enc_" + audio_in[4:] if audio_in.startswith("enc_")
                                                                     else "enc_" + audio_in)
        if enc_shift == "auto":                      # look-ahead stored with the features (frames) + the 100 ms grid
            k_ = f"{self.audio_in}_lookahead"
            enc_shift = (int(np.asarray(z[k_])) + 2) if k_ in files else 3
            log(f"enc_shift auto: {enc_shift} ({k_ if k_ in files else 'default'})")
        self.enc_shift = int(enc_shift or 0)
        self.E = None
        self.pca = None
        self.lean = bool(lean)
        self.EOK = None                                # per-stream "has encoder features" (None = all have)
        self.enc_missing = enc_missing or "silence"
        if self.enc_missing not in ("silence", "exclude"):
            raise ValueError(f"enc_missing must be silence|exclude, not {enc_missing!r}")
        if self.audio_in == "mel" and ("lean_no_mel" in files and int(np.asarray(z["lean_no_mel"]))):
            raise ValueError("this pack is lean (mel rows dropped): audio_in 'mel' cannot train on it")
        if self.lean and self.audio_in == "mel":
            raise ValueError("lean=True needs a speech-encoder audio_in (mel runs read the 40 mel columns)")
        enorm = {}
        if self.audio_in != "mel":
            if self.audio_in not in files:
                raise KeyError(f"pack has no {self.audio_in} (members: {sorted(f for f in files if f.startswith('enc_'))})")
            okk = self.audio_in + "_ok"
            eok = np.asarray(z[okk], np.uint8).reshape(-1).astype(bool) if okk in files else None
            if eok is not None and eok.all():
                eok = None
            if eok is not None:
                log(f"{self.audio_in}: features for {int(eok.sum())}/{len(eok)} streams "
                    f"({self.streams[eok, 1].sum() / max(self.streams[:, 1].sum(), 1):.1%} of frames); missing -> "
                    f"{self.enc_missing}")
            q8 = None
            arr = z[self.audio_in] if self.lean else None
            if arr is not None and arr.dtype == np.int8:
                key = self.audio_in
                q8 = np.ascontiguousarray(arr)
                self._esc = np.asarray(z[key + "_scale"], np.float32)
                self._eqm = (np.asarray(z[key + "_qmean"], np.float32) if key + "_qmean" in files
                             else np.zeros(q8.shape[1], np.float32))
                if key + "_pca_V" in files:
                    self.pca = dict(epca_mean=np.asarray(z[key + "_pca_mean"], np.float32),
                                    epca_V=np.asarray(z[key + "_pca_V"], np.float32))
                log(f"{key}: {q8.shape} int8 kept int8 on the device (lean)")
                E_ = None
            else:
                E_, self.pca = enc_member(z, self.audio_in[4:], log=log)
            if norm is not None and "emean" in norm:
                enorm = {k: np.asarray(norm[k], np.float32) for k in ("emean", "estd", "esil")}
            else:
                tr = np.concatenate([np.arange(s0, s0 + n) for i_, (s0, n, _, sp, _, gh, _) in enumerate(self.streams)
                                     if sp == 0 and gh == 0 and (eok is None or eok[i_])])
                sel = np.sort(np.random.default_rng(0).choice(tr, min(len(tr), 400_000), replace=False))
                Es = (E_[sel].astype(np.float32) if q8 is None else q8[sel].astype(np.float32) * self._esc + self._eqm)
                loud = Au[sel, 40].astype(np.float32); vad = Au[sel, 41].astype(np.float32)
                quiet = (vad < 0.5) & (loud <= np.quantile(loud, 0.1))
                enorm = dict(emean=Es.mean(0), estd=np.maximum(Es.std(0), 1e-3),
                             esil=(Es[quiet].mean(0) if quiet.sum() > 100 else Es[vad < 0.5].mean(0)))
            self.E = T.from_numpy(E_ if q8 is None else q8).to(store)
            self.E8 = q8 is not None
            if q8 is not None:
                self.esc_t = T.from_numpy(self._esc).to(device); self.eqm_t = T.from_numpy(self._eqm).to(device)
            if eok is not None:
                self.EOK = T.from_numpy(eok).to(device)
                self._eok_np = eok
            del E_, q8, arr
        self.k_enc = 0 if self.E is None else int(self.E.shape[1])
        self.EVOK = None                               # per-stream "event labels valid" (events_ok member)
        self.aud_dim = 42 if self.E is None else self.k_enc + 2
        # ---- phase 2: event onsets (+ canonical curves) and the Audio2Face teacher
        self.EV = None
        self.event_info = None
        self.event_tol = int(event_tol)
        if use_events:
            if "events" not in files:
                log("WARNING: use_events but the pack has no `events` member -> event head off")
            else:
                ON = onsets(z["events"], starts)
                if "events_ok" in files:
                    evk = np.asarray(z["events_ok"], np.uint8).reshape(-1).astype(bool)
                    if not evk.all():
                        log(f"events: labels missing for {int((~evk).sum())} streams (events_ok) -> masked")
                        self.EVOK = T.from_numpy(evk).to(device)
                        for i_ in np.nonzero(~evk)[0]:              # no onsets there: curves / rates stay clean
                            ON[self.streams[i_, 0]:self.streams[i_, 0] + self.streams[i_, 1]] = 0
                curves, info = event_curves(M, K, ON.astype(bool), self.streams, lens=event_curve_len)
                self.event_info = dict(info, curves=curves)
                log(f"events: onsets/min train {info['rate_train_per_min']} val {info['rate_val_per_min']}, curves from "
                    f"{info['n_onsets_used']} onsets")
                self.EV = T.from_numpy(ON).to(store)
                del ON
        self.A2 = self.A2M = None
        if use_a2f:
            if "a2f_u8" in files or "a2f" in files:
                a2 = z["a2f_u8"] if "a2f_u8" in files else np.asarray(z["a2f"], np.float16)
                self.A2 = T.from_numpy(np.ascontiguousarray(a2)).to(store)
                am = np.asarray(z["a2f_mask"], np.uint8).reshape(-1) if "a2f_mask" in files else \
                    (np.asarray(Au[:, 41], np.float32) > 0.5).astype(np.uint8)
                self.A2M = T.from_numpy(am).to(store)
                log(f"a2f teacher: {'a2f_u8' if 'a2f_u8' in files else 'a2f'}, speaking frames {float(am.mean()):.3f}"
                    f"{'' if 'a2f_mask' in files else ' (no a2f_mask: own vad)'}")
                del a2, am
            else:
                log("WARNING: use_a2f but the pack has no a2f member -> teacher off")
        # ---- (a): STYLE labels per frame (u8, 255 = unknown): in the pack, a sidecar, or computed now
        sc = style_sidecar_path(str(pack0))
        self.style_tables = None
        if "style" in z.files:
            S_ = np.asarray(z["style"], np.uint8)
            if "style_tables" in z.files:
                self.style_tables = json.loads(str(z["style_tables"]))
            log("style labels: from the pack")
        elif os.path.exists(sc):
            zs = np.load(sc, allow_pickle=False)
            S_ = np.asarray(zs["style"], np.uint8)
            self.style_tables = json.loads(str(zs["tables"]))
            log(f"style labels: sidecar {sc}")
        else:
            t0 = __import__("time").time()
            S_, self.style_tables = pack_style(M, K, self.streams, tables=style_tables, procs=style_procs, log=log)
            log(f"style labels computed on the fly in {__import__('time').time() - t0:.0f}s")
        assert S_.shape == (len(M), 3), S_.shape
        self.M = T.from_numpy(M).to(store)
        self.K = T.from_numpy(K).to(store)
        # lean: loud_db/vad only (the batch rebuilds a 42-wide block with constant mel = amean)
        self.A = T.from_numpy(np.ascontiguousarray(Au[:, 40:42]) if self.lean else Au).to(store)
        self.W = T.from_numpy(W).to(store)
        self.S = T.from_numpy(S_).to(store)
        del M, K, Au, W, S_
        self.st = T.from_numpy(self.streams).to(device)
        self.quality = T.from_numpy(z["quality"]).to(device)
        self.names = json.loads(str(z["names"]))
        self._dsw = None
        if dataset_weights:
            dl = self.names.get("datasets", [])
            unknown = sorted(set(dataset_weights) - set(dl))
            if unknown:
                log(f"WARNING: dataset_weights for datasets not in the pack: {unknown}")
            w = np.array([float(dataset_weights.get(dl[d], 1.0)) if d < len(dl) else 1.0 for d in self.streams[:, 4]])
            self._dsw = w
            log(f"dataset sampling weights: { {d: dataset_weights[d] for d in dataset_weights if d in dl} }")
        self.norm = {k: (np.asarray(norm[k], np.float32) if norm is not None else z[k])
                     for k in ("mean", "std", "amean", "astd")}
        self.norm.update(enorm)
        if self.pca is not None:
            self.norm.update(self.pca)
        self.mean = T.from_numpy(self.norm["mean"]).to(device); self.std = T.from_numpy(self.norm["std"]).to(device)
        self.amean = T.from_numpy(self.norm["amean"]).to(device); self.astd = T.from_numpy(self.norm["astd"]).to(device)
        self.H, self.L, self.mem = H, L, mem
        self.perm = T.from_numpy(spec.MIRROR_PERM).to(device)
        self.sign = T.from_numpy(spec.MIRROR_SIGN).to(device)
        self.gmir = T.from_numpy(spec.GROUP_MIRROR).to(device)
        self.goc = T.from_numpy(spec.GROUP_OF_COL).to(device)
        self.colw = T.from_numpy(col_weights()).to(device)
        self.sil = self.norm_audio(T.from_numpy(SILENCE_AUDIO).to(device))
        if self.lean:
            self.mel0 = T.from_numpy(np.asarray(self.norm["amean"][:40], np.float32)).to(device)
        if self.E is not None:
            self.emean = T.from_numpy(enorm["emean"]).to(device); self.estd = T.from_numpy(enorm["estd"]).to(device)
            self.esil_n = self.norm_enc(T.from_numpy(enorm["esil"]).to(device))
            self.sil_blk = T.cat([self.esil_n, self.sil[40:42]])                   # (k+2,) silence audio block
        else:
            self.sil_blk = self.sil
        self.ev_gidx = T.tensor([GI[g] for g in EVENT_GROUPS], device=device)
        egc = np.zeros((len(EVENT_NAMES), spec.D_MOTION), np.float32)
        for c, g in enumerate(EVENT_GROUPS):
            egc[c, spec.GROUPS[g]] = 1
            if c == 2:
                egc[c, spec.GROUPS["face_upper"]] = 1
        self.ev_gcol = T.from_numpy(egc).to(device)                                  # (3,189) rare-event weighting
        a2c = np.zeros(52, bool)
        a2c[[spec.BS[n] for n in (a2f_dials or A2F_DIALS)]] = True
        a2c[LIP_COLS - 135] = False                                                  # never the lip/jaw shapes
        self.a2fcol = T.from_numpy(a2c).to(device)
        self.bsperm = T.from_numpy((spec.MIRROR_PERM[135:187] - 135).astype(np.int64)).to(device)
        self._arf = {}
        lip = np.zeros(spec.D_MOTION, bool); lip[LIP_COLS] = True
        self.lipcol = T.from_numpy(lip).to(device)
        self.noff = self.nbias = None
        if noise_profile and "columns" in noise_profile:         # noise_profile-v1 (DATA): per-column dicts
            cols = noise_profile["columns"]
            g = lambda k, d: [float(cols.get(c, {}).get(k, d)) for c in spec.MOTION_COLS]   # noqa: E731
            self.nsd = T.tensor(g("within_std", 0.04), dtype=T.float32, device=device)
            self.nrho = T.tensor(g("within_lag1", 0.8), dtype=T.float32, device=device).clamp(0, 0.995)
            self.noff = T.tensor(g("offset_std", 0.0), dtype=T.float32, device=device)
            self.nbias = T.tensor(g("bias", 0.0), dtype=T.float32, device=device)
        elif noise_profile:
            self.nsd = T.tensor(noise_profile["std"], dtype=T.float32, device=device)
            self.nrho = T.tensor(noise_profile["rho"], dtype=T.float32, device=device).clamp(0, 0.995)
        else:
            sd = np.full(spec.D_MOTION, 0.04, np.float32); sd[spec.SL_FACE] = 0.04; sd[spec.SL_ROOT_SHIFT] = 0.1
            self.nsd = T.from_numpy(sd).to(device); self.nrho = T.full((spec.D_MOTION,), 0.8, device=device)
        # live input rates (noise_profile input_rates_hz: face 18, pose 6, hands 10) -> sample-and-hold factors
        hold = []
        for gs, k in ((["torso", "arm_l", "arm_r"], 5), (["fingers_l", "fingers_r"], 3),
                      (["head", "face_upper", "mouth", "gaze"], 2)):
            cols = sorted(c for g_ in gs for c in spec.GROUPS[g_])
            hold.append((T.tensor(cols, device=device), k))
        self.hold_classes = hold
        self.g = T.Generator(device=device); self.g.manual_seed(seed)
        self._split_cache = {}

    def norm_audio(self, a):
        a = a.clone()
        a[..., 40] = a[..., 40].clamp(min=-80)
        return ((a - self.amean) / self.astd).clamp(-6, 6)

    def _take(self, X, idx):
        """X[idx] on self.dev (X may live in host RAM: pack_on_cpu)."""
        r = X[idx if idx.device == X.device else idx.to(X.device)]
        return r if r.device == self._devt else r.to(self._devt, non_blocking=True)

    def norm_enc(self, e):
        return ((e - self.emean) / self.estd).clamp(-6, 6)

    def ar1(self, B, nf, D):
        """(B,nf,D) unit-variance AR(1) noise per column (rho = self.nrho), one batched matmul."""
        T = self.torch
        F = self._arf.get(nf)
        if F is None:
            F = self._arf[nf] = T.from_numpy(ar1_filter_matrix(self.nrho.cpu().numpy(), nf)).to(self.dev)
        e = self._randn(B, nf, D)
        return T.einsum("dts,bsd->btd", F, e)

    def split_streams(self, split):
        if split not in self._split_cache:
            s = self.streams
            keep = (s[:, 3] == SPLITS[split]) & (s[:, 5] == 0) & (s[:, 1] >= 90)
            if getattr(self, "EOK", None) is not None and self.enc_missing == "exclude":
                keep &= self._eok_np
            ids = np.nonzero(keep)[0]
            self._split_cache[split] = ids
        return self._split_cache[split]

    def hours(self, split):
        ids = self.split_streams(split)
        return float(self.streams[ids, 1].sum() / 30 / 3600)

    def _rand(self, *shape):
        return self.torch.rand(*shape, generator=self.g, device=self.dev)

    def _randn(self, *shape):
        return self.torch.randn(*shape, generator=self.g, device=self.dev)

    def _randint(self, lo, hi, shape):
        return self.torch.randint(lo, hi, shape, generator=self.g, device=self.dev)

    def batch(self, B, S, K, split="train", aug=None, fixed_seed=None, style_override=None, drop=None):
        """Returns dict: tok (B,S,DIN), chunk tensors (B,K,...) and mem index (B,K,M).
        style_override: None = the labels; else a (B,6) or (6,) style input [e,m,x, null_e,null_m,null_x] for every
        frame (evaluation of the style knobs). drop (evaluation of input contributions): a set of "user_audio"
        (user audio = silence), "user_face" (user face_upper/mouth/gaze masked), "user_motion" (all user groups
        masked), "self_speech" (own audio = silence)."""
        drop = set(drop or ())
        T = self.torch
        F_ = T.nn.functional
        aug = dict(AUG_DEFAULT, **(aug or {}))
        if fixed_seed is not None:
            self.g.manual_seed(fixed_seed)
        ids = self.split_streams(split)
        st = self.streams[ids]
        p = st[:, 1].astype(np.float64) * self.quality.cpu().numpy()[ids]
        if self._dsw is not None:
            p = p * self._dsw[ids]
        p = p / p.sum()
        u = self._rand(B).cpu().numpy()
        pick = ids[np.minimum(np.searchsorted(np.cumsum(p), u), len(ids) - 1)]
        pick_t = T.from_numpy(pick).to(self.dev)
        start = self.st[pick_t, 0]; length = self.st[pick_t, 1]; partner = self.st[pick_t, 2]
        H, L = self.H, self.L
        smax = T.clamp(length - 60, min=1)
        s0 = (self._rand(B) * smax).long()
        nf = S + H + L + 2
        rel = s0[:, None] + T.arange(nf, device=self.dev)[None]            # frame index within the stream
        inb = rel < length[:, None]
        relc = T.minimum(rel, length[:, None] - 1)
        gi = start[:, None] + relc                                          # self global frame index
        has_p = partner >= 0
        pstart = T.where(has_p, self.st[partner.clamp(min=0), 0], T.zeros_like(start))
        pgi = pstart[:, None] + relc
        gi_s = gi.to(self.store); pgi_s = pgi.to(self.store)
        m_self = self._take(self.M, gi_s).float(); k_self = self._take(self.K, gi_s).float() * inb[..., None]
        a_self = self._take(self.A, gi_s).float()
        m_user = self._take(self.M, pgi_s).float(); k_user = self._take(self.K, pgi_s).float() * (inb & has_p[:, None])[..., None]
        a_user = self._take(self.A, pgi_s).float()
        if self.lean:                                           # rebuild the 42-wide block (mel columns unused)
            a_self = T.cat([self.mel0.expand(B, nf, 40), a_self], -1)
            a_user = T.cat([self.mel0.expand(B, nf, 40), a_user], -1)
        w_self = self._take(self.W, gi_s).float() * inb
        if self.E is not None:                                  # phase 2: encoder features (user delayed by shift)
            e_self = self._take(self.E, gi_s).float()
            relu = rel - self.enc_shift
            e_user = self._take(self.E, pstart[:, None] + T.minimum(relu.clamp(min=0), length[:, None] - 1)).float()
            if self.E8:                                         # lean int8 features
                e_self = e_self * self.esc_t + self.eqm_t
                e_user = e_user * self.esc_t + self.eqm_t
            e_user_ok = (relu >= 0) & inb & has_p[:, None]
            if self.EOK is not None:                            # people without encoder features = silence
                e_self_ok = self.EOK[pick_t]
                e_user_ok = e_user_ok & self.EOK[partner.clamp(min=0)][:, None]
        # ---- (c): where the TARGET speaks (its own VAD, dilated) - lip/jaw shapes are lip sync's
        if aug.get("lip_mask", True):
            spk = ((a_self[..., 41] > 0.5) & inb).float()
            if SPEAK_DILATE > 0:
                spk = F_.max_pool1d(spk[:, None], 2 * SPEAK_DILATE + 1, stride=1, padding=SPEAK_DILATE)[:, 0]
            lipz = (spk > 0.5)[..., None] & self.lipcol                             # (B,nf,189)
        else:
            lipz = T.zeros(B, nf, 189, dtype=T.bool, device=self.dev)
        # ---- audio level augmentation (raw space): random per-stream gain; live browser audio sits ~+26 dB above
        # Seamless. log-mel (natural log of power) shifts by g*ln(10)/10, loud_db by g; vad unchanged; floors kept.
        glo, ghi = aug.get("gain_db", [0.0, 0.0])
        if ghi > glo:
            for a_ in (a_self, a_user):
                g = (glo + (ghi - glo) * self._rand(B))[:, None]
                on = (self._rand(B) < aug.get("p_gain", 0.85)).float()[:, None]
                g = g * on
                a_[..., :40] = T.clamp(a_[..., :40] + (g * (math.log(10.0) / 10.0))[..., None], min=math.log(1e-6))
                a_[..., 40] = T.clamp(a_[..., 40] + g, min=-100.0)
        # ---- mirror (raw space, both people)
        mir = self._rand(B) < aug["p_mirror"]
        if mir.any():
            m_self[mir] = m_self[mir][..., self.perm] * self.sign
            m_user[mir] = m_user[mir][..., self.perm] * self.sign
            k_self[mir] = k_self[mir][..., self.gmir]
            k_user[mir] = k_user[mir][..., self.gmir]
            lipz[mir] = lipz[mir][..., self.perm]
        # ---- user noise (fitter profile: per-session offset + bias + AR(1) per column) in raw units
        ucols = 189
        if aug["user_noise"] > 0:
            scale = (aug["user_noise"] * (0.5 + self._rand(B)))[:, None]
            n = self.ar1(B, nf, ucols)
            nz = n * self.nsd
            if self.noff is not None:
                nz = nz + (self._randn(B, 1, ucols) * self.noff
                           + self.nbias * (self._rand(B, 1, 1) < aug.get("p_bias", 0.7)).float())
            m_user = m_user + nz * scale[:, None]
        # live camera rates: sample-and-hold per group class (pose ~6 Hz, hands ~10 Hz, face/head ~15-18 Hz)
        if aug.get("p_user_hold", 0) > 0:
            on = self._rand(B) < aug["p_user_hold"]
            tt = T.arange(nf, device=self.dev)[None]
            for cols, k in self.hold_classes:
                ph = self._randint(0, k, (B,))[:, None]
                idx = (((tt + ph) // k) * k - ph).clamp(0, nf - 1)
                idx = T.where(on[:, None], idx, tt.expand(B, nf))
                m_user[:, :, cols] = T.gather(m_user[:, :, cols], 1, idx[..., None].expand(B, nf, len(cols)))
        # user group dropout (per sample, whole window) + whole-modality dropout (camera off / mic off)
        gdrop = self._rand(B, 9) < T.tensor(aug["p_user_gdrop"], device=self.dev)
        k_user = k_user * (~gdrop)[:, None, :]
        mdrop = self._rand(B) < aug.get("p_user_motion_drop", 0.0)
        k_user = k_user * (~mdrop)[:, None, None]
        adrop = self._rand(B) < aug.get("p_user_audio_drop", 0.0)
        if "user_audio" in drop:
            adrop = T.ones_like(adrop)
        if "user_motion" in drop:
            k_user = T.zeros_like(k_user)
        if "user_face" in drop:
            k_user = k_user.clone()
            k_user[..., [GI["face_upper"], GI["mouth"], GI["gaze"]]] = 0
        # ---- normalize
        x_self = (m_self - self.mean) / self.std
        x_user = (m_user - self.mean) / self.std
        cm_self = k_self[..., self.goc]; cm_user = k_user[..., self.goc]
        x_self = x_self * cm_self; x_user = x_user * cm_user
        x_self = x_self.masked_fill(lipz, 0.0)                     # lips while speaking: neutral (= runtime rule)
        uok = inb & has_p[:, None] & ~adrop[:, None]
        if self.E is None:
            a_self_n = T.where(inb[..., None], self.norm_audio(a_self), self.sil)
            a_user_n = T.where(uok[..., None], self.norm_audio(a_user), self.sil)
        else:                                                   # [features | loud_db, vad] of the same person
            es = self.norm_enc(e_self)
            if self.EOK is not None:
                es = T.where(e_self_ok[:, None, None], es, self.esil_n)
            eu = T.where(e_user_ok[..., None], self.norm_enc(e_user), self.esil_n)
            a_self_n = T.where(inb[..., None], T.cat([es, self.norm_audio(a_self)[..., 40:42]], -1), self.sil_blk)
            a_user_n = T.where(uok[..., None], T.cat([eu, self.norm_audio(a_user)[..., 40:42]], -1), self.sil_blk)
        if "self_speech" in drop:
            a_self_n = self.sil_blk.expand_as(a_self_n).clone()
        x_self = x_self.clamp(-8, 8); x_user = x_user.clamp(-8, 8)
        # ---- (a): style input per frame: [e,m,x (0 if null), null flags]; 30% whole-null, 10% per component
        if style_override is not None:
            sv = T.as_tensor(style_override, dtype=T.float32, device=self.dev).reshape(-1, D_STYLE_IN)
            sty = sv.expand(B, D_STYLE_IN)[:, None, :].expand(B, nf, D_STYLE_IN)
        else:
            su = self._take(self.S, gi_s)                                                          # (B,nf,3) u8
            unk = (su == 255) | ~inb[..., None]
            sval = su.float() / 254.0
            if aug.get("style_jitter", 0) > 0:
                sval = (sval + aug["style_jitter"] * self._randn(B, 1, 3)).clamp(0, 1)
            drop = (self._rand(B) < aug.get("p_style_null", 0.3))[:, None, None] | \
                (self._rand(B, 1, 3) < aug.get("p_style_comp_null", 0.1))
            null = unk | drop
            sty = T.cat([sval * (~null), null.float()], -1)
        # ---- tokens: token tau uses self[tau], user[tau+1], uaudio[tau+1], saudio[tau+1+L], style[tau]
        sig = aug["self_noise"] * self._rand(B)[:, None, None]
        xs_hist = x_self[:, :S] + sig * self._randn(B, S, 189) * cm_self[:, :S] * (~lipz[:, :S])
        sdrop = (self._rand(B) < aug["p_self_drop"])[:, None, None]
        ks_hist = k_self[:, :S] * (~sdrop)
        xs_hist = xs_hist * (~sdrop)
        null = (self._rand(B) < aug["p_null"])[:, None, None]
        xu = x_user[:, 1:S + 1] * (~null); ku = k_user[:, 1:S + 1] * (~null)
        au = a_user_n[:, 1:S + 1] * (~null)
        asf = a_self_n[:, 1 + L:S + 1 + L]
        tok = T.cat([xs_hist, ks_hist, xu, ku, au, asf, null.float().expand(B, S, 1), sty[:, :S]], -1)
        # ---- chunks
        early = self._rand(B, K) < aug["p_early"]
        tau = T.where(early, (self._rand(B, K) * 8).long(), 8 + (self._rand(B, K) * (S - 8)).long())
        C = (self._rand(B, K) * (aug["max_prefix"] + 1)).long()
        C = T.minimum(C, tau + 1)
        hidx = tau[..., None] + 1 - C[..., None] + T.arange(H, device=self.dev)      # (B,K,H) frame idx rel s0
        bidx = T.arange(B, device=self.dev)[:, None, None]
        x1 = x_self[bidx, hidx]                                                     # clean (no history noise)
        k1 = k_self[bidx, hidx]                                                     # (B,K,H,9)
        lip1 = lipz[bidx, hidx]                                                     # (B,K,H,189)
        prev_i = (tau + 1 - C - 1).clamp(min=0)
        prev = x_self[bidx[..., 0], prev_i]
        prev_ok = cm_self[bidx[..., 0], prev_i] * (tau + 1 - C - 1 >= 0)[..., None]
        hh = T.arange(H, device=self.dev)
        known = (hh[None, None] < C[..., None]).float()[..., None].expand(B, K, H, 9).clone()
        # ---- (d): override simulation over EVERY group (body, hands, head, gaze, face): random groups over a
        # random span = whole chunk / until m (RELEASE: the model takes the group back) / from m (ONSET); 25% of
        # them HOLD a constant pose (set_override(values=None) at runtime)
        ov = self._rand(B, K) < aug["p_override"]
        gp = T.tensor(aug["ov_group_p"], device=self.dev)
        gsel = self._rand(B, K, 9) < gp
        none = ~gsel.any(-1)
        rb = self._randint(0, len(spec.BODY_GROUPS), (B, K))
        gsel = gsel | (none[..., None] & (T.arange(9, device=self.dev) == rb[..., None]))
        gm = gsel.float()
        kind = self._rand(B, K)
        m_ = (self._rand(B, K) * H).long()
        span = T.where((kind < 0.4)[..., None], T.ones(B, K, H, device=self.dev, dtype=T.bool),
                       T.where((kind < 0.7)[..., None], hh[None, None] < m_[..., None],
                               hh[None, None] >= m_[..., None]))
        ovg = (ov[..., None, None] & span[..., None]).float() * gm[:, :, None, :]          # (B,K,H,9)
        known = T.maximum(known, ovg)
        known = known * (k1 > 0)                                                       # known only if valid
        kcol = known[..., self.goc]
        vcol = k1[..., self.goc]
        lmask = vcol * (1 - kcol) * (~lip1)
        hold = ov & (self._rand(B, K) < aug.get("p_hold", 0.25))
        hf = T.where(kind >= 0.7, m_, T.zeros_like(m_)).clamp(max=H - 1)
        held = T.gather(x1, 2, hf[..., None, None].expand(B, K, 1, 189))                  # (B,K,1,189)
        kv_src = T.where(hold[..., None, None] & (ovg[..., self.goc] > 0), held.expand(B, K, H, 189), x1)
        kvals = kv_src * kcol
        wsrc = w_self if aug.get("expressive", True) else (k_self.sum(-1) > 0).float()
        w = (wsrc[bidx[..., 0], (tau + 1).clamp(max=nf - 1)] * self.quality[pick_t][:, None])
        mi = tau[..., None] - self.mem + 1 + T.arange(self.mem, device=self.dev)
        mok = mi >= 0
        style_c = sty[bidx[..., 0], tau]                                               # (B,K,6) at the last token
        out = dict(tok=tok, x1=x1, lmask=lmask, kvals=kvals, kmask=known, prev=prev, prev_ok=prev_ok, w=w,
                   mem_idx=mi.clamp(min=0), mem_ok=mok, tau=tau, C=C, pick=pick_t, s0=s0, style_c=style_c)
        # ---- phase 2 (a): event onsets. Token tau's head predicts an onset at frame tau+1 (+-tol frames)
        if self.EV is not None:
            ev = self._take(self.EV, gi_s).float() * inb[..., None]                                   # (B,nf,3) onsets
            evok = (k_self[..., self.ev_gidx] > 0) & inb[..., None]
            if self.EVOK is not None:
                evok = evok & self.EVOK[pick_t][:, None, None]
            tol = self.event_tol
            evd = F_.max_pool1d(ev.transpose(1, 2), 2 * tol + 1, stride=1, padding=tol).transpose(1, 2) if tol > 0 else ev
            out.update(ev_tgt=evd[:, 1:S + 1], ev_raw=ev[:, 1:S + 1], ev_ok=evok[:, 1:S + 1].float())
            ew = float(aug.get("event_weight", 1.0))
            if ew != 1.0:                                       # rare-event weighting: loss x ew near an onset
                rad = aug.get("event_radius", EVENT_RADIUS)
                near = T.stack([F_.max_pool1d(ev[:, None, :, c], 2 * int(rad[c]) + 1, stride=1, padding=int(rad[c]))[:, 0]
                                for c in range(ev.shape[-1])], -1)                     # (B,nf,3)
                emul = 1.0 + (ew - 1.0) * (near[..., :, None] * self.ev_gcol).amax(-2)  # (B,nf,189)
                out["emul"] = emul[bidx, hidx]
        # ---- phase 2 (e): Audio2Face teacher on the target's speaking frames (face dials minus lip/jaw shapes)
        if self.A2 is not None and aug.get("a2f", True):
            a2 = self._take(self.A2, gi_s).float()
            if self.A2.dtype == T.uint8:
                a2 = a2 / 255.0
            if mir.any():
                a2[mir] = a2[mir][..., self.bsperm]
            a2n = ((a2 - self.mean[135:187]) / self.std[135:187]).clamp(-8, 8)
            a2ok = (self._take(self.A2M, gi_s) > 0) & inb & (k_self[..., GI["face_upper"]] > 0) & (k_self[..., GI["mouth"]] > 0)
            out["a2f1"] = a2n[bidx, hidx]                                              # (B,K,H,52)
            out["a2m"] = (a2ok[bidx, hidx][..., None] & self.a2fcol).float() * (1 - kcol[..., 135:187])
        return out


AUG_DEFAULT = dict(gain_db=[-20.0, 35.0], p_gain=0.85, expressive=True, p_mirror=0.5, user_noise=1.0,
                   p_user_gdrop=[0.05, 0.08, 0.1, 0.1, 0.4, 0.4, 0.08, 0.08, 0.1],
                   p_user_motion_drop=0.08, p_user_audio_drop=0.08, p_user_hold=0.5, p_bias=0.7,
                   self_noise=0.3, p_self_drop=0.1, p_null=0.15, p_early=0.1, max_prefix=8, p_override=0.3,
                   ov_group_p=[0.3, 0.3, 0.35, 0.35, 0.3, 0.3, 0.1, 0.12, 0.3], p_hold=0.25,
                   lip_mask=True, p_style_null=0.3, p_style_comp_null=0.1, style_jitter=0.03)
AUG_NONE = dict(gain_db=[0.0, 0.0], p_mirror=0.0, user_noise=0.0, p_user_gdrop=[0.0] * 9, p_user_motion_drop=0.0,
                p_user_audio_drop=0.0, p_user_hold=0.0, self_noise=0.0, p_self_drop=0.0, p_null=0.0,
                p_early=0.1, max_prefix=8, p_override=0.3, p_hold=0.0, p_style_null=0.0, p_style_comp_null=0.0,
                style_jitter=0.0)
