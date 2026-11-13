#!/usr/bin/env python3
"""SPEAKING-FACE section of the scorecard: what each system's face does ON SCREEN while the avatar talks, vs the
real person in the Seamless video, on the 29 Seamless TEST clips.

  python -I evalkit/speakface.py [--runs rules,unified-abl_t0,a2f,rules+a2f-mouth] [--out f.json]

Speaking frames = the avatar's reply spans (the scorecard's speaking state, = what rules is told it says) where MediaPipe
sees the real face; clips with >= 2 s of them. TRUTH = MediaPipe blendshapes of the avatar side's real video (the face
extractor's cache $MM_WORK_DIR/p2/face/cache, the same frames). SYSTEMS = the rows the renderer draws (evalkit/onscreen.py: lipstream's
lip columns + the page rig through RigMixer, + a renderer lip override where the system has one): jaw (= max(lipsync,
mouth_open)), round, lip_lower, lip_upper, smile (= page smile + 0.25 lipstream life smile) — the Muse's mouth columns
standing for jawOpen, mouthPucker, mouthLowerDown, mouthUpperUp, mouthSmile.
Measures (per clip on speaking frames; correlations averaged over clips, rates pooled over minutes):
  corr        Pearson r with the real person's dial (scale-free); jaw also at its best lag within +-200 ms
  smile       onsets/min (rise >= 0.15 within 1 s, mstats' definition), mean / SD / p90 of the on-screen smile in rig
              units; real = the real person's canon smile retargeted to the Muse (the `human` run on screen) and, in
              MediaPipe units, MediaPipe's own
  mouth       openings/min (jaw peaks, prominence >= 0.5 x the clip's speaking SD, >= 100 ms apart), normalised speed
              median |d jaw/dt| / SD (1/s), share of 0.5-12 Hz jaw power in 2-8 Hz and the power centroid of 2.5-8 Hz (syllable band;
              Welch on speaking runs >= 2.1 s), jaw SD and log10 2-8 Hz power in the system's own units (rig units vs MediaPipe units:
              compare those only within a column)
  lag         peak of the cross-correlation of the z-scored jaw with the z-scored log energy of the avatar's own voice
              (lags +-333 ms; + = the jaw FOLLOWS the sound), and its r
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import E, FPS, canon_spec, read_wav_mono  # noqa: E402
import onscreen as OS  # noqa: E402

FACE_CACHE = OS.T / "p2/face/cache"
DIALS = {"jaw": (["jawOpen"], "jaw"), "pucker": (["mouthPucker"], "round"),
         "lowerDown": (["mouthLowerDownLeft", "mouthLowerDownRight"], "lip_lower"),
         "upperUp": (["mouthUpperUpLeft", "mouthUpperUpRight"], "lip_upper"),
         "smile": (["mouthSmileLeft", "mouthSmileRight"], "smile")}
MIN_FRAMES = 60
LAGS = np.arange(-10, 11)
VERSION = "sf-v1"


def _bs(A, names):
    B = canon_spec().BS
    return np.mean([A[:, B[n]] for n in names], 0).astype(np.float64)


def voice_energy(cid, N):
    x, sr = read_wav_mono(E / "clips" / cid / "voice.wav")
    hop = sr / FPS
    e = np.array([np.sqrt(np.mean(x[int(k * hop):int((k + 1) * hop)] ** 2) + 1e-10) if int(k * hop) < len(x) else 1e-5
                  for k in range(N)])
    return np.log(e + 1e-5)


def truth(clip, N):
    z = np.load(FACE_CACHE / (clip["avatar_file"] + ".npz"), allow_pickle=False)
    k0 = int(clip["frame0"])
    mp = z["mp_bs"][k0:k0 + N].astype(np.float32)
    ok = z["mp_face"][k0:k0 + N].astype(bool)
    if len(mp) < N:
        return None, None
    return mp, ok


def system_dials(run, cid):
    """on-screen rows -> {dial: (N,)} (rig units)."""
    R = OS.run_onscreen(run, cid)
    if R is None:
        return None
    IDX = OS.schema()[0]
    return {d: R[:, IDX[col]].astype(np.float64) for d, (_, col) in DIALS.items()}


def canon_dials(run, cid):
    p = E / "runs" / run / cid / "canon.npz"
    if not p.exists():
        return None
    m = np.load(p)["motion"].astype(np.float32)
    A = m[:, 135:187]
    return {d: _bs(A, names) for d, (names, _) in DIALS.items()}


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-6 and b.std() > 1e-6 else np.nan


def lag_curve(x, y, m):
    """r(L) = corr(x[t+L], y[t]) over t with m[t] & m[t+L] (x z-scored on m)."""
    out = []
    N = len(x)
    for L in LAGS:
        t = np.arange(max(0, -L), min(N, N - L))
        k = m[t] & m[t + L]
        out.append(_corr(x[t + L][k], y[t][k]) if k.sum() > 30 else np.nan)
    return np.array(out)


def runs_of(m, min_len=64):
    r, k, N = [], 0, len(m)
    while k < N:
        if m[k]:
            j = k
            while j < N and m[j]:
                j += 1
            if j - k >= min_len:
                r.append((k, j))
            k = j
        else:
            k += 1
    return r


def jaw_dyn(j, m):
    from scipy.signal import find_peaks, welch
    sd = float(j[m].std())
    pk, _ = find_peaks(j, prominence=max(0.5 * sd, 1e-4), distance=3)
    n_open = int(m[pk].sum()) if len(pk) else 0
    v = np.abs(np.diff(j)) * FPS
    mv = m[1:] & m[:-1]
    spd = float(np.median(v[mv]) / sd) if sd > 1e-6 and mv.any() else np.nan
    Ps, f = [], None
    for a, b in runs_of(m):
        f, Pw = welch(j[a:b] - j[a:b].mean(), fs=FPS, nperseg=64, detrend="linear")
        Ps.append(Pw * (b - a))
    spec = (f, np.sum(Ps, 0) / max(1, sum(b - a for a, b in runs_of(m)))) if Ps else None
    return {"sd": sd, "n_open": n_open, "speed_norm": spd, "spec": spec}


def smile_onsets(s, m):
    """mstats' smile rise (>= 0.15 within 1 s) onsets counted on speaking frames."""
    sys.path.insert(0, str(OS.AVATAR / "tools/prior"))
    import mstats
    ev = mstats._rises(np.nan_to_num(s), 0.15, 30)
    ev = np.asarray(ev, int)
    ev = ev[(ev >= 0) & (ev < len(m))]
    return int(m[ev].sum())


def clip_measures(clip, runs, M):
    N = int(round(float(clip["duration_s"]) * FPS))
    mp, face = truth(clip, N)
    if mp is None:
        return None
    st, _, _ = M.clip_states(clip, N)
    m = (st == 1) & face
    if m.sum() < MIN_FRAMES:
        return None
    en = voice_energy(clip["id"], N)
    real = {d: _bs(mp, names) for d, (names, _) in DIALS.items()}
    srcs = {"real": real}
    for label, run in runs.items():
        d = system_dials(run, clip["id"])
        if d is not None:
            srcs[label] = d
    hc = canon_dials("human", clip["id"])
    if hc is not None:
        srcs["real canon (video mapper)"] = hc
    mc = {lab: canon_dials(run, clip["id"]) for lab, run in runs.items() if run.startswith("unified")}
    for lab, d in mc.items():
        if d is not None:
            srcs[lab + " (model canon, before the renderer)"] = d
    hon = system_dials("human", clip["id"])
    out = {"frames": int(m.sum()), "minutes": float(m.sum() / FPS / 60), "sys": {}}
    ez = (en - en[m].mean()) / (en[m].std() + 1e-9)
    for lab, d in srcs.items():
        r = {}
        for dial in DIALS:
            r["corr_" + dial] = 1.0 if lab == "real" else _corr(d[dial][m], real[dial][m])
        j = d["jaw"]
        jz = (j - j[m].mean()) / (j[m].std() + 1e-9)
        if lab != "real":
            lc = lag_curve(jz, (real["jaw"] - real["jaw"][m].mean()) / (real["jaw"][m].std() + 1e-9), m)
            r["jaw_lagcorr_curve"] = lc.tolist()
        r["jaw_energy_curve"] = lag_curve(jz, ez, m).tolist()
        dy = jaw_dyn(j, m)
        r.update({"jaw_sd": dy["sd"], "n_open": dy["n_open"], "jaw_speed_norm": dy["speed_norm"]})
        if dy["spec"] is not None:
            r["jaw_spec_f"], r["jaw_spec_p"] = dy["spec"][0].tolist(), dy["spec"][1].tolist()
        s = d["smile"]
        if lab == "real" and hon is not None:
            r["smile_mp"] = {"n_on": smile_onsets(s, m), "mean": float(s[m].mean()), "sd": float(s[m].std()),
                             "p90": float(np.percentile(s[m], 90))}
            s = hon["smile"]                       # the real person's smile as the Muse shows it (canon -> retarget)
        r.update({"smile_n_on": smile_onsets(s, m), "smile_mean": float(s[m].mean()), "smile_sd": float(s[m].std()),
                  "smile_p90": float(np.percentile(s[m], 90))})
        out["sys"][lab] = r
    return out


def section(runs, ids, man, cache_dir=None):
    """runs: {label: run name} (on-screen systems). -> dict (per-system aggregates) for the scorecard."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import measures as M
    per = {}
    for cid in ids:
        r = clip_measures(man[cid], runs, M)
        if r is not None:
            per[cid] = r
    labels = ["real"] + list(runs) + [k for k in next(iter(per.values()))["sys"] if k not in runs and k != "real"] if per else []
    mins = sum(p["minutes"] for p in per.values())
    agg = {}
    for lab in labels:
        rows = [p["sys"][lab] for p in per.values() if lab in p["sys"]]
        cm = [p["minutes"] for p in per.values() if lab in p["sys"]]
        if not rows:
            continue
        a = {"n_clips": len(rows)}
        for dial in DIALS:
            v = np.array([r["corr_" + dial] for r in rows], float)
            a["corr_" + dial] = float(np.nanmean(v)) if np.isfinite(v).any() else None
        if "jaw_lagcorr_curve" in rows[0]:
            cur = np.nanmean([r["jaw_lagcorr_curve"] for r in rows], 0)
            k = int(np.nanargmax(cur))
            a["jaw_corr_bestlag"] = float(cur[k]); a["jaw_bestlag_ms"] = float(LAGS[k] * 1000 / FPS)
        cur = np.nanmean([r["jaw_energy_curve"] for r in rows], 0)
        k = int(np.nanargmax(cur))
        a["jaw_energy_lag_ms"] = float(LAGS[k] * 1000 / FPS); a["jaw_energy_r"] = float(cur[k])
        a["jaw_energy_r0"] = float(cur[list(LAGS).index(0)])
        a["jaw_energy_curve"] = {int(L * 1000 / FPS): round(float(c), 3) for L, c in zip(LAGS, cur)}
        tm = float(np.sum(cm))
        a["openings_per_min"] = float(np.sum([r["n_open"] for r in rows]) / tm)
        a["jaw_speed_norm"] = float(np.nanmedian([r["jaw_speed_norm"] for r in rows]))
        a["jaw_sd"] = float(np.mean([r["jaw_sd"] for r in rows]))
        sp = [r for r in rows if "jaw_spec_p" in r]
        if sp:
            f = np.array(sp[0]["jaw_spec_f"]); Pm = np.mean([r["jaw_spec_p"] for r in sp], 0)
            band = (f >= 2) & (f <= 8); tot = (f >= 0.5) & (f <= 12); pkb = (f >= 2.5) & (f <= 8)
            a["jaw_2_8Hz_share"] = float(Pm[band].sum() / max(Pm[tot].sum(), 1e-12))
            a["jaw_2_8Hz_log10"] = float(np.log10(Pm[band].mean() + 1e-12))
            a["jaw_centroid_hz"] = float((f[pkb] * Pm[pkb]).sum() / max(Pm[pkb].sum(), 1e-12))
        a["smiles_per_min"] = float(np.sum([r["smile_n_on"] for r in rows]) / tm)
        for k in ("smile_mean", "smile_sd", "smile_p90"):
            a[k] = float(np.mean([r[k] for r in rows]))
        if lab == "real" and "smile_mp" in rows[0]:
            a["smile_mp"] = {"smiles_per_min": float(np.sum([r["smile_mp"]["n_on"] for r in rows]) / tm),
                             **{k: float(np.mean([r["smile_mp"][k] for r in rows])) for k in ("mean", "sd", "p90")}}
        agg[lab] = a
    # paired clip bootstrap of the per-clip corr difference vs rules (same clips, 2000 resamples)
    if "rules" in agg:
        rng = np.random.default_rng(5)
        for lab in agg:
            if lab in ("rules", "real"):
                continue
            dd = {}
            for dial in DIALS:
                v = np.array([p["sys"][lab]["corr_" + dial] - p["sys"]["rules"]["corr_" + dial] for p in per.values()
                              if lab in p["sys"] and "rules" in p["sys"]], float)
                v = v[np.isfinite(v)]
                if len(v) >= 5:
                    bs = rng.choice(v, (2000, len(v))).mean(1)
                    dd[dial] = [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(len(v))]
            agg[lab]["corr_delta_vs_rules"] = dd
    return {"version": VERSION, "n_clips": len(per), "speaking_min": round(mins, 2), "runs": runs, "systems": agg,
            "clips": sorted(per), "def": __doc__.split("Measures")[0].strip()}


def md(sec, order=None):
    S = sec["systems"]
    cols = order or [k for k in S if not k.endswith("before the renderer)") and k != "real canon (video mapper)"]
    cols = [c for c in cols if c in S]
    cols = [c for c in cols if c != "real"] + ["real"]
    f = lambda v, nd=2: "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"  # noqa: E731
    L = [f"#### speaking face ON SCREEN while the avatar talks ({sec['n_clips']} clips, {sec['speaking_min']:.1f} speaking-min; "
         f"truth = MediaPipe of the real video)", "",
         "| measure | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]

    def row(label, fn):
        L.append(f"| {label} | " + " | ".join(fn(S[c], c) for c in cols) + " |")
    for d in DIALS:
        row(f"corr {d} with the real face", lambda a, c, d=d: "1 (truth)" if c == "real" else f(a.get("corr_" + d)))
    row("Δ corr vs rules, jaw / smile [95% CI, paired clips]", lambda a, c: "–" if not a.get("corr_delta_vs_rules") else
        " / ".join(f"{a['corr_delta_vs_rules'][d][0]:+.2f} [{a['corr_delta_vs_rules'][d][1]:+.2f}, {a['corr_delta_vs_rules'][d][2]:+.2f}]"
                   for d in ("jaw", "smile") if d in a["corr_delta_vs_rules"]))
    row("jaw corr at best lag (lag ms)", lambda a, c: "–" if c == "real" else
        f"{f(a.get('jaw_corr_bestlag'))} ({a.get('jaw_bestlag_ms', float('nan')):+.0f})")
    row("smile onsets/min while speaking (on screen, rig units)", lambda a, c: f(a["smiles_per_min"], 1))
    row("smile mean / SD / p90 (rig units)", lambda a, c: f"{f(a['smile_mean'])} / {f(a['smile_sd'])} / {f(a['smile_p90'])}")
    row("mouth openings/min", lambda a, c: f(a["openings_per_min"], 0))
    row("jaw speed / SD (1/s, median)", lambda a, c: f(a["jaw_speed_norm"], 1))
    row("jaw power share 2–8 Hz (of 0.5–12) / centroid of 2.5–8 Hz", lambda a, c: f"{f(a.get('jaw_2_8Hz_share'))} / {f(a.get('jaw_centroid_hz'), 2)}")
    row("jaw SD (own units: rig | MediaPipe for real)", lambda a, c: f(a["jaw_sd"], 3))
    row("jaw vs own voice energy: peak lag ms (+ = jaw follows) / r", lambda a, c:
        f"{a['jaw_energy_lag_ms']:+.0f} / {f(a['jaw_energy_r'])}")
    L.append("")
    re = S.get("real", {}).get("smile_mp")
    extra = []
    if re:
        extra.append(f"real smile in MediaPipe units: {re['smiles_per_min']:.1f} onsets/min, mean {re['mean']:.2f}, SD {re['sd']:.2f}")
    for k, a in S.items():
        if k.endswith("before the renderer)") or k == "real canon (video mapper)":
            extra.append(f"{k}: corr jaw {f(a['corr_jaw'])} pucker {f(a['corr_pucker'])} smile {f(a['corr_smile'])}")
    if extra:
        L.append("; ".join(extra))
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="rules,unified-abl_t0,a2f,rules+a2f-mouth")
    ap.add_argument("--labels", default="")
    ap.add_argument("--out")
    a = ap.parse_args()
    man = {c["id"]: c for c in json.loads((E / "clips/manifest.json").read_text())["clips"]}
    ids = sorted(c for c, v in man.items() if v["group"] == "seamless")
    rl = a.runs.split(",")
    lb = a.labels.split(",") if a.labels else [r.replace("unified-", "") for r in rl]
    sec = section(dict(zip(lb, rl)), ids, man)
    print(md(sec))
    if a.out:
        Path(a.out).write_text(json.dumps(sec, indent=1, default=float))
