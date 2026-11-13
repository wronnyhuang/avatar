"""Motion statistics shared by the real-people reference (MOTION_STATS.json) and the avatar judge
(tools/motion_stats.py). Input = a list of 'people', each a list of continuous segments:

  seg = dict(M=(N,17) motion table in HUMAN units (motionlib.COLS: deg, cm, 0..1),
             blink=(N,) onset flags, blinkdur=(N,) seconds at onsets (0 elsewhere),
             st=(N,) 0 idle 1 speaking 2 listening, selfsp=(N,) bool own speech, partsp=(N,) bool partner speech,
             f0=(N,) Hz own voice (0 unvoiced), pM=(N,17) partner table or None)

Per-person values are computed per state; the reference reports the median / IQR ACROSS PEOPLE (robust to
any one video), plus the pooled value.
"""
import numpy as np
from scipy.signal import welch, find_peaks, butter, filtfilt

FPS = 30
COLS = ["yaw", "pitch", "roll", "tx", "ty", "tz", "gx", "gy", "blink_l", "blink_r", "brow_l", "brow_r", "smile",
        "squint", "wide", "jaw", "arm"]
C = {n: i for i, n in enumerate(COLS)}
STATES = {"idle": 0, "speaking": 1, "listening": 2}
NOD_PROM = 2.5      # deg, nose-down excursion of the 0.6-5 Hz band-passed pitch
BROW_RISE = 0.15    # brow units within 0.4 s
BANDS = [(0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0)]


def _runs(mask, min_len=1):
    m = np.concatenate([[0], mask.astype(np.int8), [0]]); d = np.diff(m)
    return [(a, b) for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]) if b - a >= min_len]


def _bp(x, lo, hi):
    if len(x) < 40:
        return x - np.mean(x)
    b, a = butter(2, [lo / (FPS / 2), hi / (FPS / 2)], "band")
    return filtfilt(b, a, x)


def events(seg):
    """discrete events (frame indices) for one segment"""
    M = seg["M"]; N = len(M)
    ev = {}
    # nods: nose-down peaks of band-passed pitch, prominence >= 1.5 deg, width 0.1-1.0 s
    p = _bp(np.nan_to_num(M[:, C["pitch"]]), 0.6, 5.0)
    pk, pr = find_peaks(p, prominence=NOD_PROM, width=(3, 30), distance=6)
    ev["nod"] = pk
    ev["bignod"] = pk[pr["prominences"] >= 5.0] if len(pk) else pk
    # head shakes not needed; gaze shifts: total gaze (head yaw + eye) change > 5 deg within 100 ms
    gz = np.nan_to_num(M[:, C["gx"]] + M[:, C["yaw"]]); gzy = np.nan_to_num(M[:, C["gy"]] - M[:, C["pitch"]])
    dg = np.hypot(gz[3:] - gz[:-3], gzy[3:] - gzy[:-3])
    sh = np.zeros(N, bool); sh[3:] = dg > 5.0
    ev["gshift"] = np.array([a for a, b in _runs(sh)], int)
    # smiles: rises >= 0.15 within 1 s
    sm = np.nan_to_num(M[:, C["smile"]])
    ev["smile"] = _rises(sm, 0.15, 30)
    br = np.nan_to_num((M[:, C["brow_l"]] + M[:, C["brow_r"]]) / 2)
    ev["brow"] = _rises(br, BROW_RISE, 12)
    ev["blink"] = np.where(seg["blink"] > 0)[0]
    # own speech onsets (after >= 0.5 s silence), partner pause onsets (partner speech ends, silent >= 0.25 s)
    ss = seg["selfsp"].astype(bool); ps = seg["partsp"].astype(bool)
    ev["sp_on"] = np.array([a for a, b in _runs(ss, 6) if a >= 15 and not ss[a - 15:a].any()], int)
    ev["p_pause"] = np.array([b for a, b in _runs(ps, 9) if b + 8 <= N and not ps[b:b + 8].any()], int)
    # pitch accents: local f0 maxima (semitones) with prominence >= 1.5 st while voiced
    f0 = seg["f0"]
    if (f0 > 0).sum() > 30:
        st_ = np.where(f0 > 0, 12 * np.log2(np.maximum(f0, 1) / np.median(f0[f0 > 0])), np.nan)
        x = np.nan_to_num(st_, nan=np.nanmin(st_) if np.isfinite(st_).any() else 0)
        pk, _ = find_peaks(x, prominence=2.0, distance=6)
        ev["accent"] = pk[(f0[pk] > 0)]
    else:
        ev["accent"] = np.array([], int)
    # gaze aversion episodes (>= 10 deg from the person's median direction, >= 200 ms)
    tot = np.hypot(gz - np.median(gz), gzy - np.median(gzy))
    ev["avert_mask"] = tot > 10.0
    ev["avert"] = [(a, b) for a, b in _runs(tot > 10.0, 6)]
    return ev


def _rises(x, amp, win):
    """onsets where x rises by >= amp within `win` frames (debounced)"""
    out = []; i = win; N = len(x)
    while i < N:
        lo = x[i - win:i].min()
        if x[i] - lo >= amp:
            out.append(i - win + int(np.argmin(x[i - win:i]))); i += win
        else:
            i += 1
    return np.array(out, int)


def _lift(ev_t, ref_t, N, lo, hi, mask=None):
    """P(event within [ref+lo, ref+hi]) / P(event within a random window of that size) (frames)"""
    if len(ref_t) == 0 or len(ev_t) == 0 or N < 60:
        return np.nan, 0
    e = np.zeros(N + 1, int); e[np.clip(ev_t, 0, N - 1)] = 1; cs = np.concatenate([[0], np.cumsum(e)])
    def hit(t):
        a = int(np.clip(t + lo, 0, N)); b = int(np.clip(t + hi + 1, 0, N)); return cs[b] - cs[a] > 0
    obs = np.mean([hit(t) for t in ref_t])
    rnd = np.arange(0, N, 5)
    base = np.mean([hit(t) for t in rnd])
    return (obs / base if base > 0 else np.nan), len(ref_t)


def person_stats(segs):
    """one person's statistics per state; returns dict state -> metric -> value, plus person-level couplings"""
    res = {}
    for sname, sv in list(STATES.items()) + [("all", -1)]:
        r = {}; dur = 0.0
        acc = {k: [] for k in ("ang", "speed", "psd", "win_sd", "tr_sd")}
        cnt = {k: 0 for k in ("nod", "bignod", "blink", "gshift", "smile", "brow", "avert")}
        bdur = []; adur = []; avert_frames = 0; still = 0; armv = []; arm_vis = 0; smv = []; brv = []
        for seg in segs:
            M = seg["M"]; N = len(M)
            m = np.ones(N, bool) if sv < 0 else (seg["st"] == sv)
            if m.sum() < 15:
                continue
            ev = seg.setdefault("_ev", events(seg))
            dur += m.sum() / FPS
            ang = np.nan_to_num(M[:, :3])
            vel = np.gradient(ang, axis=0) * FPS
            sp = np.linalg.norm(vel, axis=1)
            acc["speed"].append(np.abs(vel[m]))
            sp05 = np.convolve(sp, np.ones(15) / 15, "same")
            still += (sp05[m] < 5.0).sum()
            # state runs >= 4 s for windowed amplitude + PSD
            for a, b in _runs(m, 120):
                for w0 in range(a, b - 119, 150):
                    w1 = min(b, w0 + 300)
                    acc["win_sd"].append(ang[w0:w1].std(0))
                    acc["tr_sd"].append(np.nan_to_num(M[w0:w1, 3:6]).std(0))
                    f, P = welch(ang[w0:w1] - ang[w0:w1].mean(0), fs=FPS, nperseg=min(120, w1 - w0), axis=0)
                    acc["psd"].append(np.array([P[(f >= lo) & (f < hi)].mean(0) for lo, hi in BANDS]))
            for k in ("nod", "bignod", "blink", "gshift", "smile", "brow"):
                t = ev[k]
                cnt[k] += int(m[t].sum()) if len(t) else 0
            bd = seg["blinkdur"][ev["blink"]] if len(ev["blink"]) else np.array([])
            bdur += list(bd[m[ev["blink"]]] * 1000) if len(bd) else []
            for a, b in ev["avert"]:
                if m[a]:
                    cnt["avert"] += 1; adur.append((b - a) / FPS)
            avert_frames += ev["avert_mask"][m].sum()
            arm = M[:, C["arm"]]
            ok = m & np.isfinite(arm)
            armv.append(arm[ok]); arm_vis += ok.sum()
            smv.append(np.nan_to_num(M[m, C["smile"]])); brv.append(np.nan_to_num((M[m, C["brow_l"]] + M[m, C["brow_r"]]) / 2))
        if dur < 20:
            continue
        mins = dur / 60
        r["minutes"] = mins
        spd = np.concatenate(acc["speed"])
        for j, ax in enumerate(("yaw", "pitch", "roll")):
            r[f"head_{ax}_speed_med"] = float(np.median(spd[:, j]))
            r[f"head_{ax}_speed_p90"] = float(np.percentile(spd[:, j], 90))
            if acc["win_sd"]:
                r[f"head_{ax}_sd10s"] = float(np.median([w[j] for w in acc["win_sd"]]))
                ps = np.median(np.array(acc["psd"])[:, :, j], 0)
                for (lo, hi), v in zip(BANDS, ps):
                    r[f"head_{ax}_psd_{lo}_{hi}Hz"] = float(np.log10(v + 1e-9))
        if acc["tr_sd"]:
            for j, ax in enumerate(("tx", "ty", "tz")):
                r[f"head_{ax}_sd10s_cm"] = float(np.median([w[j] for w in acc["tr_sd"]]))
        r["stillness_frac"] = float(still / (dur * FPS))
        for k in ("nod", "bignod", "blink", "gshift", "smile", "brow", "avert"):
            r[f"{k}_per_min"] = cnt[k] / mins
        r["blink_dur_ms_med"] = float(np.median(bdur)) if bdur else np.nan
        r["avert_frac"] = float(avert_frames / (dur * FPS))
        r["avert_dur_s_med"] = float(np.median(adur)) if adur else np.nan
        sm = np.concatenate(smv); br = np.concatenate(brv)
        r["smile_mean"] = float(sm.mean()); r["smile_sd"] = float(sm.std())
        r["brow_sd"] = float(br.std())
        if arm_vis > 15 * 30:
            a = np.expm1(np.concatenate(armv))      # arm column = log1p(1-s wrist-speed envelope)
            r["arm_speed_med"] = float(np.median(a)); r["arm_speed_p90"] = float(np.percentile(a, 90))
            r["gesture_frac"] = float(np.mean(a > 0.6))
        res[sname] = r
    # couplings (person level, across states)
    cp = {}
    def pooled_lift(evk, refk, lo, hi, state=None):
        num = 0; den = 0; n = 0
        for seg in segs:
            ev = seg.setdefault("_ev", events(seg)); N = len(seg["M"])
            ref = ev[refk]
            if state is not None and len(ref):
                ref = ref[seg["st"][np.clip(ref, 0, N - 1)] == state]
            l, k = _lift(ev[evk], ref, N, lo, hi)
            if np.isfinite(l) and k:
                num += l * k; den += k
        return (num / den if den else np.nan), den
    cp["blink_after_speech_onset_lift"], cp["n_speech_onsets"] = pooled_lift("blink", "sp_on", 0, 15)
    cp["blink_at_gaze_shift_lift"], cp["n_gaze_shifts"] = pooled_lift("blink", "gshift", -6, 6)
    cp["brow_at_accent_lift"], cp["n_accents"] = pooled_lift("brow", "accent", -9, 9, state=1)
    cp["nod_at_accent_lift"], _ = pooled_lift("nod", "accent", -6, 9, state=1)
    cp["nod_after_partner_pause_lift"], cp["n_partner_pauses"] = pooled_lift("nod", "p_pause", -15, 30, state=2)
    cp["bignod_after_partner_pause_lift"], _ = pooled_lift("bignod", "p_pause", -15, 30, state=2)
    cp["bignod_at_accent_lift"], _ = pooled_lift("bignod", "accent", -6, 9, state=1)
    # nod timing relative to partner pause onsets (listening): histogram of offsets in s
    for nk, key in (("nod", "nod_offsets_from_partner_pause_s"), ("bignod", "bignod_offsets_from_partner_pause_s")):
        offs = []
        for seg in segs:
            ev = seg["_ev"]; st = seg["st"]
            for t in ev["p_pause"]:
                if st[min(t, len(st) - 1)] != 2 and st[max(t - 1, 0)] != 2:
                    continue
                for n in ev[nk]:
                    if -30 <= n - t <= 45:
                        offs.append((n - t) / FPS)
        cp[key] = offs
    # mirroring: smile cross-correlation with partner
    xs = []
    for seg in segs:
        pM = seg.get("pM")
        if pM is None or not np.isfinite(pM[:, C["smile"]]).any():
            continue
        a = np.nan_to_num(seg["M"][:, C["smile"]]); b = pM[:, C["smile"]]
        ok = np.isfinite(b)
        for r0, r1 in _runs(ok, 600):
            x = a[r0:r1] - a[r0:r1].mean(); y = b[r0:r1] - b[r0:r1].mean()
            if x.std() < 1e-3 or y.std() < 1e-3:
                continue
            L = 90
            cc = [np.corrcoef(x[max(0, l):len(x) + min(0, l)], y[max(0, -l):len(y) - max(0, l)])[0, 1] for l in range(-L, L + 1)]
            xs.append(np.array(cc))
    if xs:
        cc = np.mean(xs, 0); lags = np.arange(-90, 91) / FPS
        cp["smile_xcorr_peak"] = float(cc.max()); cp["smile_xcorr_lag_s"] = float(lags[np.argmax(cc)])
        cp["smile_xcorr_zero"] = float(cc[90])
        cp["smile_xcorr_curve"] = [float(v) for v in cc[::3]]
    res["coupling"] = cp
    return res


def aggregate(people):
    """list of person_stats dicts (each with 'weight' minutes) -> {state: {metric: {med, q25, q75, n}}}"""
    out = {}
    for sname in list(STATES) + ["all"]:
        keys = sorted({k for p in people if sname in p for k in p[sname]})
        out[sname] = {}
        for k in keys:
            v = np.array([p[sname][k] for p in people if sname in p and k in p[sname]], float)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            out[sname][k] = dict(med=float(np.median(v)), q25=float(np.percentile(v, 25)),
                                 q75=float(np.percentile(v, 75)), n=int(len(v)))
    cp = {}
    keys = sorted({k for p in people for k in p.get("coupling", {}) if not k.endswith(("_s", "curve"))})
    for k in keys:
        v = np.array([p["coupling"][k] for p in people if k in p.get("coupling", {})], float)
        v = v[np.isfinite(v)]
        if len(v):
            cp[k] = dict(med=float(np.median(v)), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)), n=int(len(v)))
    for key, name in (("nod_offsets_from_partner_pause_s", "nod_offset_hist"), ("bignod_offsets_from_partner_pause_s", "bignod_offset_hist")):
        offs = np.concatenate([np.array(p["coupling"].get(key, []), float) for p in people] or [np.array([])])
        if len(offs):
            h, e = np.histogram(offs, bins=np.arange(-1.0, 1.51, 0.25))
            cp[name] = dict(edges_s=[float(x) for x in e], counts=[int(x) for x in h], n=int(len(offs)),
                            median_s=float(np.median(offs)))
    curves = [p["coupling"]["smile_xcorr_curve"] for p in people if "smile_xcorr_curve" in p.get("coupling", {})]
    if curves:
        cc = np.mean(curves, 0); lags = np.arange(-90, 91, 3) / FPS
        cp["smile_xcorr_mean_curve"] = dict(lags_s=[float(x) for x in lags], r=[float(x) for x in cc],
                                            peak_lag_s=float(lags[np.argmax(cc)]), peak_r=float(cc.max()), n=len(curves))
    out["coupling"] = cp
    return out
