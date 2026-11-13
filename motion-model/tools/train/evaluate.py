"""Closed-loop validation of a bundle on held-out dyads (both people as "self", the partner as the user).

    python -I tools/train/evaluate.py --bundle <dir> --index <data dir/index.jsonl> [...] --split val
           [--seconds 60] [--max-sessions 12] [--workers 12] [--out eval.json] [--long-minutes 10]
    python -I tools/train/evaluate.py --bundle <dir> --pack <v3 qpack .npz> --split val [--seconds 60]
           [--max-sessions 12] [--workers 12] [--drop user_audio,user_face,self_speech] [--out eval.json]

--pack mode: the FULL pack (qpack q1 or raw, plus the optional v3 members events / enc_<encoder>) is decoded once in
the parent; dyads = non-ghost streams of the --split whose partner is a non-ghost stream of the same session; one
seeded --seconds window per dyad; the workers get numpy slices (they never read the pack). Reals = the pack's motion,
real event rates = the pack's `events` labels (label_<name>_per_min; onsets = rising edges of nonzero). A bundle whose
config.json has "audio": {"in": "enc_<name>", "user_shift": s, ...} is fed the RAW pack features as
step(..., user_feat=feats_user[t - s] (None while t < s, t = the step index), self_feat_ahead=feats_self[t + L]);
other bundles get the old 4-argument step(). --drop adds one condition per rollout and name: user_audio (user audio
and features None), user_face (the user's face_upper / mouth / gaze groups masked, cols 0), self_speech (own audio
and features None); each gets gen stats, the delta vs normal and a sensitivity. The output's flat "scorecard" holds
the ablation-table numbers (gen vs real, couplings, sensitivities, plan_ms).

For each (session, self person, condition) it runs UnifiedPlanner frame by frame with the partner's real motion
(+ the fitter noise profile if given), the partner's audio and the person's own audio (ahead by L), and compares
the generated motion with the real person's: motion energy, nods/min, look-away %, blink rate, stillness, smile
level, user->self smile coupling. Conditions: normal, user zeroed (None), user shuffled (another session's partner).
Sensitivity = mean |out_normal - out_ablated| / mean |out_normal - its mean| on head/face/gaze columns.
Seam measures (spectral(), head pitch "hp" and yaw "hy", gen and real): log10 band power 1-3 Hz (nods) / 4-8 Hz / 8-15
Hz at 0.125 Hz Welch bins, line10 / line5 (10 Hz = re-plan or patch every 3 frames, 5 Hz = every 6) = peak bin over
the flanking bins' median (~1 = no line), linemax(_hz) = the strongest line anywhere in 2-14 Hz.
--rt '{json}' passes extra UnifiedPlanner kwargs (e.g. the seam knobs box / ensemble / prefix).
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from canon1 import spec  # noqa: E402

FPS = spec.FPS
C = spec.COL
HEAD_COLS = spec.GROUPS["head"]
FACE_COLS = spec.GROUPS["face_upper"] + spec.GROUPS["mouth"]
GAZE_COLS = spec.GROUPS["gaze"]
SM = [135 + spec.BS["mouthSmileLeft"], 135 + spec.BS["mouthSmileRight"]]
BL = [135 + spec.BS["eyeBlinkLeft"], 135 + spec.BS["eyeBlinkRight"]]
EVENT_NAMES = ("nod", "blink", "smile")              # pack v3 `events` columns / planner event names
DROPS = ("user_audio", "user_face", "self_speech")
FACE_DROP_GROUPS = [spec.GROUP_NAMES.index(g) for g in ("face_upper", "mouth", "gaze")]
FACE_DROP_COLS = spec.GROUPS["face_upper"] + spec.GROUPS["mouth"] + spec.GROUPS["gaze"]
SELF_PAD = 64                                        # self audio / feature rows cut past the window (>= lookahead)
SC_KEYS = ("head_speed", "body_speed", "still_pct", "nods_per_min", "blinks_per_min", "smiles_per_min",
           "smile_mean", "look_away_listen_pct", "smile_coupling",
           "hp13_log10", "hp48_log10", "hp815_log10", "hp_line10", "hp_line5", "hy13_log10", "hy815_log10", "hy_line10")


def _smooth(x, n):
    return np.convolve(x, np.ones(n) / n, "same") if n > 1 else x


def onset_count(x):
    """per-frame labels (T,) -> number of onsets = rising edges of nonzero (single-frame onsets or spans)."""
    b = np.asarray(x) != 0
    return int(b[:1].sum() + (b[1:] & ~b[:-1]).sum()) if len(b) else 0


def smile_onsets(sm, thr=0.4, gap=FPS):
    """5-frame smoothed (edge-padded) smile crossing up through thr, >= gap frames apart."""
    s5 = np.convolve(np.pad(np.asarray(sm, np.float64), 2, mode="edge"), np.ones(5) / 5, "valid")
    n, last = 0, -10 ** 9
    for p in np.nonzero((s5[1:] >= thr) & (s5[:-1] < thr))[0] + 1:
        if p - last >= gap:
            n += 1; last = p
    return n


SPEC_NPERSEG = 240                                   # 0.125 Hz bins: 5 Hz and 10 Hz fall exactly on a bin
SPEC_BANDS = (("13", 1.0, 3.0), ("48", 4.0, 8.0), ("815", 8.0, 15.0))


def spectral(x, pre):
    """re-plan seam measures of one angle series x (T,) (rad): Welch (Hann, nperseg 240 = 0.125 Hz bins, linear
    detrend) -> {pre}{band}_log10 = log10 band power (rad^2, trapezoid) for 1-3 Hz (nods), 4-8 Hz, 8-15 Hz; {pre}_line10
    / {pre}_line5 = the max PSD bin within +-0.2 Hz of 10 Hz (re-plan every 3 frames) / 5 Hz (every 6) over the median
    PSD of the flanking bins (7-9 + 11-13 Hz / 3.5-4.5 + 5.5-6.5 Hz); ~1 = no line."""
    from scipy.signal import welch
    x = np.asarray(x, np.float64)
    if len(x) < 32 or not np.isfinite(x).all():
        return {}
    f, P = welch(x, fs=FPS, nperseg=min(SPEC_NPERSEG, len(x)), detrend="linear")
    out = {}
    for name, lo, hi in SPEC_BANDS:
        b = (f >= lo) & (f <= hi)
        out[f"{pre}{name}_log10"] = float(np.log10(np.trapezoid(P[b], f[b]) + 1e-14))
    for name, c, flank in (("line10", 10.0, ((7, 9), (11, 13))), ("line5", 5.0, ((3.5, 4.5), (5.5, 6.5)))):
        pk = P[np.abs(f - c) <= 0.2]
        nb = P[np.any([(f >= lo) & (f <= hi) for lo, hi in flank], axis=0)]
        if len(pk) and len(nb):
            out[f"{pre}_{name}"] = float(pk.max() / max(np.median(nb), 1e-20))
    # the strongest line anywhere in 2-14 Hz (other cadences: 7.5 Hz = every 4 frames, 3.33 Hz = a 9-frame cycle):
    # each bin over the median of the bins 0.4-1.5 Hz away
    best, bf = 0.0, float("nan")
    for i in np.nonzero((f >= 2) & (f <= 14))[0]:
        d = np.abs(f - f[i])
        nb = P[(d >= 0.4) & (d <= 1.5)]
        r = float(P[i] / max(np.median(nb), 1e-20))
        if r > best:
            best, bf = r, float(f[i])
    out[f"{pre}_linemax"], out[f"{pre}_linemax_hz"] = best, bf
    return out


def stats(m, user=None, listen=None, fired=None, labels=None):
    """m (T,189) one person's motion -> dict of human-range statistics. listen: bool (T,) frames where this person
    is NOT speaking (look-away is split into listening / speaking). fired: {event name: count} the planner fired
    (-> fired_<name>_per_min); labels: (T, 3) pack event labels (-> label_<name>_per_min)."""
    T = len(m)
    mins = T / FPS / 60
    pitch = m[:, C["neck_x"]] + m[:, C["head_x"]]
    yaw = m[:, C["root_rot_y"]] + m[:, C["spine1_y"]] + m[:, C["spine2_y"]] + m[:, C["spine3_y"]] + \
        m[:, C["neck_y"]] + m[:, C["head_y"]]
    hv = np.linalg.norm(np.diff(m[:, HEAD_COLS], axis=0), axis=1) * FPS             # rad/s
    # nods: band-passed pitch (0.3 s smoothing minus 1.5 s), peaks above 0.035 rad, >= 0.4 s apart
    bp = _smooth(pitch, 5) - _smooth(pitch, 45)
    pk = np.nonzero((bp[1:-1] > bp[:-2]) & (bp[1:-1] >= bp[2:]) & (bp[1:-1] > 0.035))[0] + 1
    nods, last = 0, -99
    for p in pk:
        if p - last >= 12:
            nods += 1; last = p
    gy = yaw - np.median(yaw) + m[:, C["gaze_yaw"]]
    gp = pitch - np.median(pitch) - m[:, C["gaze_pitch"]]
    away = (np.abs(gy) > 0.3) | (np.abs(gp) > 0.3)
    bl = m[:, BL].mean(1)
    blinks = int(((bl[1:] > 0.5) & (bl[:-1] <= 0.5)).sum())
    sm = m[:, SM].mean(1)
    extra = {"smiles_per_min": smile_onsets(sm) / mins}
    if fired is not None:
        for name, c in fired.items():
            extra[f"fired_{name}_per_min"] = c / mins
    if labels is not None and len(labels) == T:
        lab = np.asarray(labels).reshape(T, -1)
        for j, name in enumerate(EVENT_NAMES[: lab.shape[1]]):
            extra[f"label_{name}_per_min"] = onset_count(lab[:, j]) / mins
    if listen is not None and len(listen) == T:
        if listen.sum() > 30:
            extra["look_away_listen_pct"] = float(away[listen].mean() * 100)
        if (~listen).sum() > 30:
            extra["look_away_speak_pct"] = float(away[~listen].mean() * 100)
    # 4-8 Hz head-pitch power (log10 rad^2, Welch) - the re-plan jitter signature - and head jerk (rad/s^3)
    from scipy.signal import welch
    f_, P_ = welch(pitch - pitch.mean(), fs=FPS, nperseg=min(128, T))
    extra["hp48_log10"] = float(np.log10(np.trapezoid(P_[(f_ >= 4) & (f_ <= 8)], f_[(f_ >= 4) & (f_ <= 8)]) + 1e-12))
    extra["head_jerk"] = float(np.linalg.norm(np.diff(m[:, HEAD_COLS], n=3, axis=0), axis=1).mean() * FPS ** 3)
    extra.update(spectral(pitch, "hp"))
    extra.update(spectral(yaw, "hy"))
    out = dict(**extra, head_speed=float(hv.mean()), head_speed_p90=float(np.percentile(hv, 90)),
               head_yaw_sd=float(np.std(yaw)), head_pitch_sd=float(np.std(pitch)), nods_per_min=nods / mins,
               look_away_pct=float(away.mean() * 100), blinks_per_min=blinks / mins,
               still_pct=float((hv < 0.1).mean() * 100), smile_mean=float(sm.mean()),
               brow_sd=float(np.std(m[:, 135 + spec.BS["browInnerUp"]])),
               body_speed=float(np.linalg.norm(np.diff(m[:, 6:45], axis=0), axis=1).mean() * FPS))
    if user is not None:
        us = user[:, SM].mean(1)
        best = 0.0
        for lag in range(0, 46, 3):
            a, b = us[: T - lag], sm[lag:]
            if np.std(a) > 1e-4 and np.std(b) > 1e-4:
                best = max(best, float(np.corrcoef(a, b)[0, 1]))
        out["smile_coupling"] = best
    return out


def load_noise(path):
    if path and os.path.exists(path):
        with open(path) as f:
            prof = json.load(f)
        if "columns" in prof and "std" not in prof:      # noise_profile-v1 (DATA): per-column dicts -> std/rho (+ offset, bias)
            from canon1 import spec as _spec
            cols = prof["columns"]
            g = lambda k, d: [float(cols.get(c, {}).get(k, d)) for c in _spec.MOTION_COLS]   # noqa: E731
            prof = dict(prof, std=g("within_std", 0.04), rho=g("within_lag1", 0.8),
                        offset_std=g("offset_std", 0.0), bias=g("bias", 0.0))
        return prof
    return None


def ar1_noise(T, prof, rng):
    sd = np.asarray(prof["std"], np.float32); rho = np.clip(np.asarray(prof["rho"], np.float32), 0, 0.995)
    e = rng.standard_normal((T, len(sd))).astype(np.float32)
    n = np.empty_like(e); n[0] = e[0]
    q = np.sqrt(1 - rho * rho)
    for t in range(1, T):
        n[t] = rho * n[t - 1] + q * e[t]
    return n * sd


def _planner(job):
    from unified.runtime import UnifiedPlanner
    bundle, kw = job["bundle"], job["kw"]
    p = UnifiedPlanner(bundle, threads=kw.get("threads", 2), seed=job["seed"], replan_every=kw.get("replan"),
                       steps=kw.get("steps"), guidance=kw.get("guidance"), temperature=kw.get("temperature"),
                       blend=kw.get("blend"), consistent_noise=kw.get("consistent_noise"), smooth=kw.get("smooth"),
                       noise_rho=kw.get("noise_rho"), temperature_groups=kw.get("temperature_groups"),
                       pcm=False, **(kw.get("rt") or {}))  # features precomputed from the pack: no live encoders
    if job.get("style") is not None and hasattr(p, "set_style"):
        p.set_style(job["style"])
    return p


def _row(F, i):
    """feature row i of F as float32, or None when out of range / not finite."""
    if F is None or i < 0 or i >= len(F):
        return None
    r = np.asarray(F[i], np.float32)
    return r if np.isfinite(r).all() else None


def simulate(p, job, um, uk, ua, sa_rows, fu=None, fs=None):
    """Step the planner over the window. um/uk/ua (T, .) = the user's motion / mask / audio for step k; sa_rows = the
    self person's audio rows from the window start (step k reads row k + L, None past the end); fu (T, D) = the
    user's RAW encoder features from the window start, fs = the self person's from the window start (enc bundles).
    job: zero_user (user None), drop [user_audio | user_face | self_speech], enc, user_shift.
    -> (gen (T,189), plan_ms list, fired {name: count} or None when the planner reports no events)."""
    L = p.self_audio_lookahead
    drops = set(job.get("drop") or ())
    zero = bool(job.get("zero_user"))
    enc = bool(job.get("enc"))
    if enc:
        ain = getattr(p, "audio_in", None)
        if ain is not None and job.get("audio_in") and ain != job["audio_in"]:
            raise RuntimeError(f"planner audio_in {ain!r} != the bundle config's {job['audio_in']!r}")
        sh = getattr(p, "user_feat_shift", None)
        shift = int(job.get("user_shift") or 0) if sh is None else int(sh)
    if "user_face" in drops and not zero:
        um = np.array(um, np.float32); uk = np.array(uk, np.float32)
        um[:, FACE_DROP_COLS] = 0
        uk[:, FACE_DROP_GROUPS] = 0
    no_ua, no_sa = "user_audio" in drops, "self_speech" in drops
    T = len(um)
    out = np.zeros((T, spec.D_MOTION), np.float32)
    ms, fired = [], None
    for k in range(T):
        j = k + L
        sa = None if no_sa or j >= len(sa_rows) else sa_rows[j]
        fk = {}
        if enc:
            fk = dict(user_feat=None if zero or no_ua or k < shift else _row(fu, k - shift),
                      self_feat_ahead=None if no_sa else _row(fs, j))
        if zero:
            o = p.step(None, None, None, sa, **fk)
        else:
            o = p.step(um[k], uk[k], None if no_ua else ua[k], sa, **fk)
        out[k] = o["motion"]
        if o.get("plan_ms") is not None:
            ms.append(o["plan_ms"])
        ev = o.get("events")
        if ev is not None:
            if fired is None:
                fired = {n: 0 for n in EVENT_NAMES}
            for n in ev:
                fired[n] = fired.get(n, 0) + 1
    return out, ms, fired


def rollout(job):
    p = _planner(job)
    L = p.self_audio_lookahead
    d = spec.load_session(job["path"])
    Ts = d["motion"].shape[1]
    if job["t0"] + job["T"] + L > Ts:            # long rollouts: loop the session (both people, audio) end to end
        reps = int(np.ceil((job["t0"] + job["T"] + L + 1) / Ts))
        for k_ in ("motion", "mask", "audio"):
            d[k_] = np.concatenate([d[k_]] * reps, axis=1)
    a, b = job["self"], job["user"]
    t0, T = job["t0"], job["T"]
    um = d["motion"][b, t0:t0 + T].copy(); uk = d["mask"][b, t0:t0 + T]; ua = d["audio"][b, t0:t0 + T]
    if job.get("shuffle"):
        e = spec.load_session(job["shuffle"])
        n = min(T, e["motion"].shape[1])
        um = np.zeros_like(um); uk = np.zeros_like(uk); ua = np.tile(ua[:1] * 0 + ua.min(0), (T, 1))
        um[:n] = e["motion"][1, :n]; uk[:n] = e["mask"][1, :n]; ua[:n] = e["audio"][1, :n]
    noise = job.get("noise")
    if noise is not None:
        um = um + ar1_noise(T, noise, np.random.default_rng(job["seed"]))
    tic = time.time()
    out, ms, fired = simulate(p, job, um, uk, ua, d["audio"][a][t0:])
    real = d["motion"][a, t0:t0 + T]
    listen = d["audio"][a, t0:t0 + T, 41] < 0.5
    return dict(key=job["key"], cond=job["cond"], gen=out, real=real, user=d["motion"][b, t0:t0 + T], listen=listen,
                real_mask=d["mask"][a, t0:t0 + T], plan_ms=ms, secs=time.time() - tic, fired=fired)


def rollout_pack(job):
    """--pack mode: every array comes with the job (cut in the parent)."""
    p = _planner(job)
    f32 = lambda x: None if x is None else np.asarray(x, np.float32)      # noqa: E731
    um, uk, ua = f32(job["um"]), job["uk"], f32(job["ua"])
    if job.get("noise") is not None:
        um = um + ar1_noise(len(um), job["noise"], np.random.default_rng(job["seed"]))
    tic = time.time()
    out, ms, fired = simulate(p, job, um, uk, ua, f32(job["sa_rows"]), job.get("fu"), job.get("fs"))
    return dict(key=job["key"], cond=job["cond"], gen=out, real=f32(job.get("real")), user=f32(job.get("user")),
                listen=job.get("listen"), real_mask=job.get("real_mask"), labels=job.get("labels"), plan_ms=ms,
                secs=time.time() - tic, fired=fired)


def run_job(job):
    return rollout_pack(job) if job.get("mode") == "pack" else rollout(job)


def summarize(results, long=None):
    by = {}
    for r in results:
        by.setdefault(r["key"], {})[r["cond"]] = r
    rows, real_rows, sens = [], [], {"zero": [], "shuffle": []}
    drop_pairs, drop_sens = {}, {}
    for key, conds in by.items():
        n = conds.get("normal")
        if n is None:
            continue
        rows.append(stats(n["gen"], n["user"], n.get("listen"), fired=n.get("fired")))
        real_rows.append(stats(n["real"], n["user"], n.get("listen"), labels=n.get("labels")))
        cols = HEAD_COLS + FACE_COLS + GAZE_COLS
        g = n["gen"][:, cols]
        scale = np.abs(g - g.mean(0)).mean()
        for c in ("zero", "shuffle"):
            if c in conds:
                sens[c].append(float(np.abs(g - conds[c]["gen"][:, cols]).mean() / max(scale, 1e-6)))
        for c, r in conds.items():
            if str(c).startswith("drop:"):
                dn = c[5:]
                drop_pairs.setdefault(dn, []).append(
                    (stats(r["gen"], n["user"], n.get("listen"), fired=r.get("fired")), rows[-1]))
                drop_sens.setdefault(dn, []).append(float(np.abs(g - r["gen"][:, cols]).mean() / max(scale, 1e-6)))
        for c in ("zero", "shuffle"):
            if c in conds:
                s = stats(conds[c]["gen"], n["user"])
                rows[-1][f"smile_coupling_{c}"] = s.get("smile_coupling")
    keys = sorted({k for r in rows for k in r})
    rkeys = sorted({k for r in real_rows for k in r})
    gen = {k: float(np.nanmean([r[k] for r in rows if r.get(k) is not None])) for k in keys}
    real = {k: float(np.nanmean([r[k] for r in real_rows if r.get(k) is not None])) for k in rkeys}
    # % of generated sessions inside the real people's IQR, and mean |log ratio| vs the real person
    inside, logr = {}, {}
    for k in real:
        rv = np.array([np.nan if r.get(k) is None else r[k] for r in real_rows], np.float64)
        gv = np.array([np.nan if r.get(k) is None else r[k] for r in rows], np.float64)
        ok = np.isfinite(rv) & np.isfinite(gv)
        if ok.sum() < 2:
            continue
        q1, q3 = np.percentile(rv[ok], [25, 75])
        inside[k] = float(np.mean((gv[ok] >= q1) & (gv[ok] <= q3)) * 100)
        logr[k] = float(np.mean(np.abs(np.log((np.abs(gv[ok]) + 1e-3) / (np.abs(rv[ok]) + 1e-3)))))
    plan = np.concatenate([np.asarray(r["plan_ms"]) for r in results if r["plan_ms"]] or [np.zeros(1)])
    out = dict(n=len(rows), gen=gen, real=real, pct_inside_real_iqr=inside, mean_abs_log_ratio=logr,
               mean_abs_log_ratio_avg=float(np.mean([logr[k] for k in logr if k != "smile_coupling"])),
               sensitivity_user_zero=float(np.mean(sens["zero"])) if sens["zero"] else None,
               sensitivity_user_shuffle=float(np.mean(sens["shuffle"])) if sens["shuffle"] else None,
               plan_ms_p50=float(np.percentile(plan, 50)), plan_ms_p99=float(np.percentile(plan, 99)))
    if drop_pairs:
        out["drops"] = {}
        for dn, prs in drop_pairs.items():
            dk = sorted({k for d_, _ in prs for k in d_})
            mean_of = lambda f: {k: float(np.nanmean([v for v in (f(d_, n_, k) for d_, n_ in prs) if v is not None]))   # noqa: E731
                                 for k in dk if any(f(d_, n_, k) is not None for d_, n_ in prs)}
            out["drops"][dn] = dict(
                n=len(prs), gen=mean_of(lambda d_, n_, k: d_.get(k)),
                delta_vs_normal=mean_of(lambda d_, n_, k: None if d_.get(k) is None or n_.get(k) is None
                                        else d_[k] - n_[k]),
                sensitivity=float(np.mean(drop_sens[dn])))
    out["scorecard"] = scorecard(out)
    sty = {}
    for r in results:
        if str(r["cond"]).startswith("style:"):
            _, comp, lv = r["cond"].split(":")
            sty.setdefault(comp, {}).setdefault(lv, []).append(stats(r["gen"], r["user"], r.get("listen")))
    if sty:
        out["style_sweep"] = {}
        for comp, d in sty.items():
            o = {}
            for lv, rs in d.items():
                o[lv] = {k: round(float(np.nanmean([x[k] for x in rs if x.get(k) is not None])), 4)
                         for k in ("head_speed", "body_speed", "brow_sd", "smile_mean", "smile_coupling",
                                   "look_away_pct", "nods_per_min", "still_pct")}
                fv = [r["gen"][:, FACE_COLS].std(0).mean() for r in results if r["cond"] == f"style:{comp}:{lv}"]
                o[lv]["face_sd"] = round(float(np.mean(fv)), 4)
            out["style_sweep"][comp] = o
    if long is not None:
        g = long["gen"]
        mins = len(g) // (60 * FPS)
        per = [stats(g[i * 60 * FPS:(i + 1) * 60 * FPS]) for i in range(mins)]
        out["long_rollout"] = dict(minutes=mins, finite=bool(np.isfinite(g).all()),
                                   max_abs=float(np.abs(g).max()),
                                   head_speed_per_min=[round(x["head_speed"], 4) for x in per],
                                   look_away_per_min=[round(x["look_away_pct"], 1) for x in per],
                                   smile_per_min=[round(x["smile_mean"], 3) for x in per])
    return out


def scorecard(out):
    """flat dict for the ablation table."""
    gen, real = out["gen"], out["real"]
    sc = {}
    for k in SC_KEYS:
        sc[f"{k}_gen"] = gen.get(k)
        sc[f"{k}_real"] = real.get(k)
    if gen.get("smile_coupling") is not None and gen.get("smile_coupling_zero") is not None:
        sc["smile_coupling_gain_zero"] = gen["smile_coupling"] - gen["smile_coupling_zero"]
    sc["sens_zero"] = out.get("sensitivity_user_zero")
    sc["sens_shuffle"] = out.get("sensitivity_user_shuffle")
    sc["mean_abs_log_ratio_avg"] = out.get("mean_abs_log_ratio_avg")
    sc["plan_ms_p50"] = out.get("plan_ms_p50")
    sc["plan_ms_p99"] = out.get("plan_ms_p99")
    for name in EVENT_NAMES:
        if f"fired_{name}_per_min" in gen:
            sc[f"fired_{name}_per_min_gen"] = gen[f"fired_{name}_per_min"]
        if f"label_{name}_per_min" in real:
            sc[f"label_{name}_per_min_real"] = real[f"label_{name}_per_min"]
    for dn, d in (out.get("drops") or {}).items():
        sc[f"sens_drop_{dn}"] = d["sensitivity"]
        for k in ("smile_coupling", "look_away_listen_pct", "head_speed"):
            if k in d["delta_vs_normal"]:
                sc[f"delta_drop_{dn}_{k}"] = d["delta_vs_normal"][k]
    return sc


# ------------------------------------------------------------------------------------------- --pack mode
def bundle_audio(bundle):
    """-> (audio_in, enc mode?, audio cfg) from the bundle's config.json "audio" block (absent = old mel bundle)."""
    with open(os.path.join(bundle, "config.json")) as f:
        cfg = json.load(f)
    au = cfg.get("audio") if isinstance(cfg.get("audio"), dict) else {}
    ain = au.get("in")
    return ain, bool(ain) and str(ain).startswith("enc_"), au


def _take(X, idx, n, fill=0, dtype=None):
    """rows X[idx] padded to n rows with `fill`."""
    out = np.empty((n,) + X.shape[1:], dtype or X.dtype)
    out[...] = fill
    if len(idx):
        out[: len(idx)] = X[idx]
    return out


def pack_jobs(a, kw, noise):
    """decode the pack once; cut every job's arrays here (the workers never read the pack)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import slim_pack
    from unified.data import SILENCE_AUDIO, SPLITS, read_qpack
    ain, enc, au = bundle_audio(a.bundle)
    P = slim_pack.Pack(a.pack)
    files = set(P.files)
    t_dec = time.time()
    if "qformat" in files:
        M, K, A, _ = read_qpack(P)
    else:
        M, K, A = P["motion"], P["mask"], P["audio"]
    zs = np.load(a.pack, allow_pickle=False)
    st = np.asarray(zs["streams"])
    names = json.loads(str(zs["names"])) if "names" in files else {}
    EV = P["events"] if "events" in files else None
    F = None
    if enc:
        if ain not in files:
            raise SystemExit(f"bundle wants audio.in={ain!r}: not in the pack ({sorted(f for f in files if f.startswith('enc_'))})")
        F = P[ain]
        if F.dtype.kind != "f" or f"{ain}_pca_V" in files:
            raise SystemExit(f"{ain} is quantized (a slim pack?): evaluate needs the FULL v3 pack (raw features)")
        if au.get("feat_dim") and int(au["feat_dim"]) != F.shape[1]:
            raise SystemExit(f"{ain}: pack width {F.shape[1]} != bundle feat_dim {au['feat_dim']}")
    print(f"pack {a.pack}: {len(M)} frames, {len(st)} streams decoded in {time.time() - t_dec:.0f}s; "
          f"events={'yes' if EV is not None else 'no'} audio_in={ain or 'mel42'}", file=sys.stderr, flush=True)
    sp = SPLITS[a.split]
    dsets = names.get("datasets", [])
    keep_ds = set(a.datasets.split(",")) if a.datasets else None
    T = int(a.seconds * FPS)
    pairs = set()
    for i in range(len(st)):
        j = int(st[i, 2])
        if st[i, 3] != sp or st[i, 5] or j < 0 or st[j, 5] or st[j, 6] != st[i, 6]:
            continue
        if keep_ds is not None and (dsets[st[i, 4]] if st[i, 4] < len(dsets) else str(st[i, 4])) not in keep_ds:
            continue
        pairs.add((min(i, j), max(i, j)))
    pairs = sorted(pairs)
    rng = np.random.default_rng(0)
    rng.shuffle(pairs)
    n_all = len(pairs)
    all_pairs = pairs
    pairs = [pr for pr in pairs if min(st[pr[0], 1], st[pr[1], 1]) >= T + 30][: a.max_sessions]
    if not pairs:
        raise SystemExit(f"no {a.split} dyads of >= {a.seconds + 1:.0f} s in the pack ({n_all} dyads in the split)")
    sess_names = names.get("sessions", [])
    sname = lambda i: sess_names[st[i, 6]] if st[i, 6] < len(sess_names) else f"s{st[i, 6]}"   # noqa: E731
    span = lambda s, t0, n, wrap=False: (st[s, 0] + ((t0 + np.arange(n)) % st[s, 1]) if wrap   # noqa: E731
                                         else st[s, 0] + np.arange(t0, min(t0 + n, st[s, 1])))
    sil = SILENCE_AUDIO.astype(np.float32)

    def user_part(s, t0, n, wrap=False):
        ix = span(s, t0, n, wrap)
        d = dict(um=_take(M, ix, n), uk=_take(K, ix, n), ua=_take(A, ix, n, dtype=np.float32), user_stream=int(s),
                 user_t0=int(t0))
        if len(ix) < n:
            d["ua"][len(ix):] = sil
        if F is not None:
            d["fu"] = _take(F, ix, n, fill=np.nan)
        return d

    def self_part(s, t0, n, wrap=False):
        ix = span(s, t0, n + SELF_PAD, wrap)
        d = dict(sa_rows=np.asarray(A[ix]), self_stream=int(s), t0=int(t0))
        if F is not None:
            d["fs"] = np.asarray(F[ix])
        ir = ix[:n]
        d.update(real=np.asarray(M[ir]), real_mask=np.asarray(K[ir]), listen=np.asarray(A[ir, 41], np.float32) < 0.5,
                 labels=None if EV is None else np.asarray(EV[ir]))
        return d

    base0 = dict(mode="pack", bundle=a.bundle, kw=kw, noise=noise, enc=enc, audio_in=ain,
                 user_shift=int(au.get("user_shift") or 0))
    if a.style:
        base0["style"] = json.loads(a.style)
    drops = [d for d in (a.drop or "").split(",") if d]
    jobs = []
    for i, (s1, s2) in enumerate(pairs):
        ln = int(min(st[s1, 1], st[s2, 1]))
        t0 = int(rng.integers(0, ln - T - 10))
        others = [pr for pr in all_pairs if st[pr[0], 6] != st[s1, 6]]     # shuffle: a stranger from another session
        other = others[i % len(others)][1] if others else None
        for s, (me, you) in enumerate(((s1, s2), (s2, s1))):
            base = dict(base0, seed=i * 2 + s, key=f"{sname(me)}:{s}")
            u, sp_ = user_part(you, t0, T), self_part(me, t0, T)
            ins = dict(sa_rows=sp_["sa_rows"], fs=sp_.get("fs"), self_stream=int(me), t0=t0)
            jobs.append(dict(base, cond="normal", **u, **sp_, user=u["um"]))
            if i < a.style_sweep:
                for comp in ("energy", "mirroring", "expressiveness"):
                    for lv, v in (("lo", 0.1), ("hi", 0.9)):
                        jobs.append(dict(base, **u, **ins, user=u["um"], listen=sp_["listen"],
                                         cond=f"style:{comp}:{lv}", style={comp: v}, key=base["key"] + f":{comp}{lv}"))
            if not a.no_ablate:
                jobs.append(dict(base, **u, **ins, cond="zero", zero_user=True))
                if other is not None:
                    lo = int(st[other, 1])
                    jobs.append(dict(base, **user_part(other, max(0, min(t0, lo - T)), T), **ins, cond="shuffle"))
            for d in drops:
                jobs.append(dict(base, **u, **ins, cond=f"drop:{d}", drop=[d]))
    if a.long_minutes > 0:
        s1, s2 = max(pairs, key=lambda pr: st[pr[0], 1])
        Tl = int(a.long_minutes * 60 * FPS)
        u, sp_ = user_part(s2, 0, Tl, wrap=True), self_part(s1, 0, Tl, wrap=True)
        jobs.append(dict(base0, seed=99, key="long", cond="long", **u, sa_rows=sp_["sa_rows"], fs=sp_.get("fs"),
                         self_stream=int(s1), t0=0))
    meta = dict(mode="pack", pack=a.pack, audio_in=ain or "mel42", enc=enc, dyads=len(pairs), dyads_in_split=n_all,
                events_in_pack=EV is not None, drop_conditions=drops)
    del M, K, A, EV, F
    return jobs, len(pairs), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--index", action="append")
    src.add_argument("--pack", help="the FULL pack (v3 qpack .npz): decoded once in the parent")
    ap.add_argument("--drop", default="", help="comma list of extra conditions: " + ",".join(DROPS))
    ap.add_argument("--split", default="val")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--max-sessions", type=int, default=12)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--guidance", type=float, default=None, help="default: the bundle's runtime default")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--blend", type=int, default=None)
    ap.add_argument("--consistent-noise", type=int, default=None)
    ap.add_argument("--smooth", type=float, default=None)
    ap.add_argument("--noise-rho", type=float, default=None)
    ap.add_argument("--temperature-groups", default=None, help='json {"face_upper": 0.8, ...}')
    ap.add_argument("--rt", default=None, help='json dict of extra UnifiedPlanner kwargs, e.g. {"ensemble": {...}, '
                    '"prefix": 4} (default: the bundle\'s config["runtime"])')
    ap.add_argument("--noise-profile", default=None)
    ap.add_argument("--no-ablate", action="store_true")
    ap.add_argument("--long-minutes", type=float, default=0)
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--style", default=None, help='json dict for set_style on every rollout')
    ap.add_argument("--style-sweep", type=int, default=0, help="N sessions: each style knob at 0.1 and 0.9")
    a = ap.parse_args()
    bad = [d for d in a.drop.split(",") if d and d not in DROPS]
    if bad:
        ap.error(f"--drop: unknown {bad} (choose from {', '.join(DROPS)})")
    kw = dict(threads=a.threads, guidance=a.guidance, steps=a.steps, temperature=a.temperature, blend=a.blend,
              consistent_noise=None if a.consistent_noise is None else bool(a.consistent_noise), smooth=a.smooth,
              noise_rho=a.noise_rho,
              temperature_groups=json.loads(a.temperature_groups) if a.temperature_groups else None)
    if a.rt:
        kw["rt"] = json.loads(a.rt)
    noise = load_noise(a.noise_profile)
    meta = {}
    if a.pack:
        jobs, n_sess, meta = pack_jobs(a, kw, noise)
        return finish(a, jobs, n_sess, kw, meta)
    if bundle_audio(a.bundle)[1]:
        ap.error("this bundle reads encoder features (audio.in = enc_*): evaluate it with --pack <v3 pack>")
    from unified.data import load_index
    recs = [r for r in load_index(a.index) if r.get("split") == a.split]
    if a.datasets:
        recs = [r for r in recs if r.get("dataset") in a.datasets.split(",")]
    recs = [r for r in recs if len(r.get("persons", [])) == 2 or r.get("persons") is None]
    rng = np.random.default_rng(0)
    rng.shuffle(recs)
    recs = recs[: a.max_sessions]
    T = int(a.seconds * FPS)
    jobs = []
    for i, r in enumerate(recs):
        frames = int(r.get("frames") or spec.load_session(r["path"])["motion"].shape[1])
        if frames < T + 30:
            continue
        t0 = int(rng.integers(0, frames - T - 10))
        for s in (0, 1):
            base = dict(bundle=a.bundle, kw=kw, path=r["path"], self=s, user=1 - s, t0=t0, T=T, seed=i * 2 + s,
                        key=f"{r['path']}:{s}", noise=noise)
            if a.style:
                base["style"] = json.loads(a.style)
            jobs.append(dict(base, cond="normal"))
            if i < a.style_sweep:
                for comp in ("energy", "mirroring", "expressiveness"):
                    for lv, v in (("lo", 0.1), ("hi", 0.9)):
                        jobs.append(dict(base, cond=f"style:{comp}:{lv}", style={comp: v}, key=base["key"] + f":{comp}{lv}"))
            if not a.no_ablate:
                jobs.append(dict(base, cond="zero", zero_user=True))
                other = recs[(i + 1) % len(recs)]["path"]
                jobs.append(dict(base, cond="shuffle", shuffle=other))
            for d in [d for d in a.drop.split(",") if d]:
                jobs.append(dict(base, cond=f"drop:{d}", drop=[d]))
    long_job = None
    if a.long_minutes > 0:
        # a long closed loop: chain the longest val session's partner stream (looped) for drift checking
        r = max(recs, key=lambda r: r.get("frames") or 0)
        frames = int(r.get("frames"))
        long_job = dict(bundle=a.bundle, kw=kw, path=r["path"], self=0, user=1, t0=0, T=min(frames, int(a.long_minutes * 60 * FPS)),
                        seed=99, key="long", cond="long", noise=noise)
        jobs.append(long_job)
    return finish(a, jobs, len(recs), kw, dict(mode="index", drop_conditions=[d for d in a.drop.split(",") if d]))


def finish(a, jobs, n_sess, kw, meta):
    tic = time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        res = pool.map(run_job, jobs, chunksize=1)
    long_res = [r for r in res if r["cond"] == "long"]
    res = [r for r in res if r["cond"] != "long"]
    out = summarize(res, long_res[0] if long_res else None)
    out.update(bundle=a.bundle, split=a.split, seconds=a.seconds, sessions=n_sess, wall_s=round(time.time() - tic, 1),
               settings=kw, **meta)
    s = json.dumps(out, indent=1)
    print(s)
    if a.out:
        with open(a.out, "w") as f:
            f.write(s)


if __name__ == "__main__":
    main()
