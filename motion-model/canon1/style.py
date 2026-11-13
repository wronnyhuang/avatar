"""STYLE labels per person-segment: {energy, mirroring, expressiveness} in [0,1] (or None = unknown).

Segments: a fixed grid per session, SEG frames (20 s); a tail shorter than MIN_TAIL is merged into the previous
segment; a session shorter than SEG is one segment. All persons of a session share the grid (so a pack span is a run
of whole segments for every person).

Components (raw, per person-segment; NaN when the groups they need are not measured on >= MIN_VALID of the frames):
  head_speed   mean angular speed of (neck+head) rotation, rad/s, after a sigma-2-frame Gaussian (removes fitter jitter)
  arm_speed    mean joint speed of shoulder+elbow+wrist (per visible arm, averaged), rad/s, same smoothing
  face_speed   mean L1 speed of the EXPRESSION blendshapes (brows, eye squint/wide, cheek squint, nose sneer,
               smile, frown: speech shapes jaw/lips and blinks excluded), 1/s
  face_var     sum over EXPRESSION blendshapes of their variance within the segment
  arm_var      sum over shoulder+elbow+wrist angles of their variance within the segment (gesture amplitude)
  mir_*        lagged correlation self(t) vs partner(t - lag), averaged over lags 0..1.0 s (partner leads), for
               pitch velocity (nods), head speed, smile, brow raise; only frames where both are measured
Labels:
  energy         = rank of mean(pct(head_speed), pct(arm_speed), pct(face_speed))    [available components]
  expressiveness = rank of mean(pct(face_var), pct(arm_var))
  mirroring      = rank of mean(pct(mir_pitch), pct(mir_hspeed), pct(mir_smile), pct(mir_brow))
  pct(.) is the empirical CDF of the component over the TRAIN segments of every source pooled; rank(.) is the CDF of
  the mean over train segments with the SAME set of measured components (signature; pooled if < 200); a label is None
  when none of its components is available (e.g. mirroring without a measured partner, expressiveness for head-only
  L2L).
"""
from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import spec as S

SEG = 600
MIN_TAIL = 300
MIN_VALID = 0.5
SIGMA = 2.0
LAGS = list(range(0, 31, 3))
G = {g: i for i, g in enumerate(S.GROUP_NAMES)}
EXPR = ["browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "eyeSquintLeft",
        "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "cheekSquintLeft", "cheekSquintRight", "noseSneerLeft",
        "noseSneerRight", "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight"]
EXPR_IDX = [135 + S.BS[n] for n in EXPR]
SMILE_IDX = [135 + S.BS["mouthSmileLeft"], 135 + S.BS["mouthSmileRight"]]
BROW_IDX = [135 + S.BS[n] for n in ("browInnerUp", "browOuterUpLeft", "browOuterUpRight")]
ARM_COLS = {s: S.jcols(f"shoulder_{s}") + S.jcols(f"elbow_{s}") + S.jcols(f"wrist_{s}") for s in "lr"}
COMPS = ["head_speed", "arm_speed", "face_speed", "face_var", "arm_var", "mir_pitch", "mir_hspeed", "mir_smile",
         "mir_brow"]
LABELS = {"energy": ["head_speed", "arm_speed", "face_speed"], "expressiveness": ["face_var", "arm_var"],
          "mirroring": ["mir_pitch", "mir_hspeed", "mir_smile", "mir_brow"]}
LABEL_NAMES = ["energy", "mirroring", "expressiveness"]


def grid(T, seg=SEG, min_tail=MIN_TAIL):
    if T <= seg:
        return [(0, T)]
    out = [(a, min(a + seg, T)) for a in range(0, T, seg)]
    if len(out) > 1 and out[-1][1] - out[-1][0] < min_tail:
        a, _ = out[-2]
        out = out[:-2] + [(a, T)]
    return out


def person_tracks(motion, mask):
    """(T,189),(T,9) -> dict of per-frame tracks (float64) + validity (bool)."""
    m = gaussian_filter1d(motion.astype(np.float64), SIGMA, axis=0, mode="nearest")
    hv = mask[:, G["head"]] > 0
    h = m[:, S.jcols("neck")] + m[:, S.jcols("head")]
    dh = np.gradient(h, axis=0) * S.FPS
    tr = {"hv": hv, "head_speed": np.linalg.norm(dh, axis=1), "pitch_vel": dh[:, 0]}
    arms, avs = [], []
    for s in "lr":
        v = mask[:, G[f"arm_{s}"]] > 0
        d = np.gradient(m[:, ARM_COLS[s]], axis=0) * S.FPS
        arms.append(np.where(v, np.linalg.norm(d, axis=1), np.nan)); avs.append(v)
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        tr["arm_speed"] = np.nanmean(np.stack(arms), 0)
    tr["av"] = avs[0] | avs[1]
    tr["av_lr"] = avs
    fv = (mask[:, G["face_upper"]] > 0) & (mask[:, G["mouth"]] > 0)
    tr["fv"] = fv
    tr["face_speed"] = np.abs(np.gradient(m[:, EXPR_IDX], axis=0)).sum(1) * S.FPS
    tr["expr"] = motion[:, EXPR_IDX].astype(np.float64)
    tr["arm_raw"] = {s: motion[:, ARM_COLS[s]].astype(np.float64) for s in "lr"}
    tr["smile"] = m[:, SMILE_IDX].mean(1)
    tr["brow"] = m[:, BROW_IDX].mean(1)
    return tr


def _lagcorr(x, y, ok, lags=LAGS):
    """mean over lags of corr(x[t], y[t-lag]) on frames where ok[t] & ok_y[t-lag] (ok is (T,) for x, y has its own
    validity folded in as NaN)."""
    T = len(x)
    rs = []
    for l in lags:
        if T - l < 45:
            continue
        a = x[l:]; b = y[:T - l]
        k = ok[l:] & np.isfinite(b) & np.isfinite(a)
        if k.sum() < 45:
            continue
        a = a[k]; b = b[k]
        sa, sb = a.std(), b.std()
        if sa < 1e-9 or sb < 1e-9:
            continue
        rs.append(float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)))
    return float(np.mean(rs)) if rs else np.nan


def segment_comps(tracks, partner_tracks, a, b):
    """raw components of one person on [a,b) (partner_tracks None = no measured partner)."""
    n = b - a
    out = dict.fromkeys(COMPS, np.nan)
    t = tracks
    hv = t["hv"][a:b]
    if hv.mean() >= MIN_VALID:
        out["head_speed"] = float(t["head_speed"][a:b][hv].mean())
    av = t["av"][a:b]
    if av.mean() >= MIN_VALID:
        x = t["arm_speed"][a:b]
        out["arm_speed"] = float(np.nanmean(x[av]))
        vs = []
        for s, v in zip("lr", t["av_lr"]):
            vv = v[a:b]
            if vv.mean() >= MIN_VALID:
                vs.append(float(t["arm_raw"][s][a:b][vv].var(0).sum()))
        if vs:
            out["arm_var"] = float(np.mean(vs))
    fv = t["fv"][a:b]
    if fv.mean() >= MIN_VALID:
        out["face_speed"] = float(t["face_speed"][a:b][fv].mean())
        out["face_var"] = float(t["expr"][a:b][fv].var(0).sum())
    if partner_tracks is not None and n >= 60:
        p = partner_tracks
        ph = p["hv"][a:b]
        both_h = hv & ph
        if both_h.mean() >= MIN_VALID:
            out["mir_pitch"] = _lagcorr(t["pitch_vel"][a:b], np.where(ph, p["pitch_vel"][a:b], np.nan), hv)
            out["mir_hspeed"] = _lagcorr(t["head_speed"][a:b], np.where(ph, p["head_speed"][a:b], np.nan), hv)
        pf = p["fv"][a:b]
        if (fv & pf).mean() >= MIN_VALID:
            out["mir_smile"] = _lagcorr(t["smile"][a:b], np.where(pf, p["smile"][a:b], np.nan), fv)
            out["mir_brow"] = _lagcorr(t["brow"][a:b], np.where(pf, p["brow"][a:b], np.nan), fv)
    return out


def session_rows(motion, mask, partners, seg=SEG):
    """motion (P,T,189), mask (P,T,9), partners: list per person of the partner index (or None; or a (T,) int array
    of per-frame partner indices, -1 = none -> the segment's majority is used). -> list of rows."""
    P, T, _ = motion.shape
    tr = [person_tracks(motion[p], mask[p]) for p in range(P)]
    rows = []
    for si, (a, b) in enumerate(grid(T, seg)):
        for p in range(P):
            q = partners[p]
            if q is not None and not np.isscalar(q):
                v = np.asarray(q[a:b]); v = v[v >= 0]
                q = int(np.bincount(v).argmax()) if len(v) > (b - a) * 0.3 else None
            c = segment_comps(tr[p], tr[q] if q is not None else None, a, b)
            rows.append(dict(p=p, seg=si, a=int(a), b=int(b), partner=q, **{k: (None if not np.isfinite(v) else
                                                                                 round(float(v), 6))
                                                                            for k, v in c.items()}))
    return rows


# ------------------------------------------------------------------------------------------------ tables + labels
def _cdf(sorted_vals, x):
    if len(sorted_vals) == 0:
        return np.nan
    lo = np.searchsorted(sorted_vals, x, "left"); hi = np.searchsorted(sorted_vals, x, "right")
    return float((lo + hi) / 2 / len(sorted_vals))


def _sig(r, comps):
    return "+".join(c for c in comps if r.get(c) is not None)


def fit_tables(train_rows, nq=1001, min_sig=200):
    """train_rows: rows with raw components -> quantile tables for every component (pooled over the train population)
    and, per label, for the mean component-percentile: pooled ("q") and per availability SIGNATURE ("sig": the set of
    components measured, e.g. head-only L2L vs head+arms+face Seamless), so every label is uniform on [0,1] within each
    kind of measurement (averaging 3 percentiles is narrower than 1; a pooled rank would push 1-component sources to
    the extremes). Signatures with < min_sig train rows use the pooled table."""
    tab = {"version": "style-v1", "nq": nq, "comps": {}, "labels": {}, "seg_frames": SEG, "lags": LAGS,
           "sigma": SIGMA, "min_sig": min_sig}
    q = np.linspace(0, 1, nq)
    sv = {}
    for c in COMPS:
        v = np.array([r[c] for r in train_rows if r.get(c) is not None], np.float64)
        sv[c] = np.sort(v)
        tab["comps"][c] = dict(n=int(len(v)), q=np.quantile(v, q).round(6).tolist() if len(v) else [])
    for lab, comps in LABELS.items():
        means, sigs = [], []
        for r in train_rows:
            ps = [_cdf(sv[c], r[c]) for c in comps if r.get(c) is not None]
            if ps:
                means.append(np.mean(ps)); sigs.append(_sig(r, comps))
        means = np.array(means); sigs = np.array(sigs)
        ent = dict(n=int(len(means)), q=np.quantile(means, q).round(6).tolist() if len(means) else [], sig={})
        for sg in sorted(set(sigs.tolist())):
            v = means[sigs == sg]
            if len(v) >= min_sig:
                ent["sig"][sg] = dict(n=int(len(v)), q=np.quantile(v, q).round(6).tolist())
        tab["labels"][lab] = ent
    return tab


def _qcdf(qtab, x):
    """empirical CDF from a quantile table (piecewise-linear inverse)."""
    qv = np.asarray(qtab["q"], np.float64)
    if len(qv) == 0:
        return np.nan
    p = np.linspace(0, 1, len(qv))
    # ties (flat runs) -> centre of the run
    lo = np.searchsorted(qv, x, "left"); hi = np.searchsorted(qv, x, "right")
    if hi > lo:
        return float(p[lo:hi].mean())
    return float(np.interp(x, qv, p))


def _label_table(tab, lab, sig):
    ent = tab["labels"][lab]
    return ent.get("sig", {}).get(sig, ent)


def label_row(tab, r):
    out = {}
    for lab, comps in LABELS.items():
        ps = [_qcdf(tab["comps"][c], r[c]) for c in comps if r.get(c) is not None]
        out[lab] = None if not ps else round(_qcdf(_label_table(tab, lab, _sig(r, comps)), float(np.mean(ps))), 4)
    return out


def label_rows(tab, rows):
    """vectorized label_row over many rows."""
    comp_p = {}
    for c in COMPS:
        qv = np.asarray(tab["comps"][c]["q"], np.float64)
        x = np.array([np.nan if r.get(c) is None else r[c] for r in rows], np.float64)
        if len(qv) == 0:
            comp_p[c] = np.full(len(rows), np.nan); continue
        comp_p[c] = np.where(np.isfinite(x), np.interp(np.nan_to_num(x), qv, np.linspace(0, 1, len(qv))), np.nan)
    out = {}
    for lab, comps in LABELS.items():
        M = np.stack([comp_p[c] for c in comps], 1)
        with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            mu = np.nanmean(M, 1)
        sigs = np.array(["+".join(c for c, ok in zip(comps, row) if ok) for row in np.isfinite(M)])
        res = np.full(len(rows), np.nan)
        for sg in set(sigs.tolist()):
            if not sg:
                continue
            k = sigs == sg
            qv = np.asarray(_label_table(tab, lab, sg)["q"], np.float64)
            res[k] = np.interp(mu[k], qv, np.linspace(0, 1, len(qv)))
        out[lab] = res
    return out


def to_u8(v):
    """label array (NaN = unknown) -> uint8: round(v*254), 255 = unknown."""
    v = np.asarray(v, np.float64)
    return np.where(np.isfinite(v), np.clip(np.round(v * 254), 0, 254), 255).astype(np.uint8)


def dumps(tab):
    return json.dumps(tab)
