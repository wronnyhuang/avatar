#!/usr/bin/env python3
"""Real-people motion statistics computed from an AVATAR rig-vector log, compared against the reference
docs/MOTION_STATS.json (the same code computes both: tools/prior/mstats.py).

Log format (JSONL, one line per frame, any rate; resampled to 30 Hz):
  {"t": 12.033, "v": {"head_yaw": 0.02, ...}}                         named values
  {"t": 12.033, "a": [...]}  + a header line {"params": [...]} or {"rig_params": [...]} or --params a,b,c
Optional per-line context (recommended; otherwise inferred, see below):
  "state": "speaking"|"listening"|"idle"   or  "speaking": bool, "listening": bool
  "f0": avatar voice pitch Hz (0 = unvoiced)  -> enables the brow/nod-at-pitch-accent couplings
  "partner": {"speaking": bool}  or "user_speaking": bool -> partner pauses (backchannel timing)
Inference when absent: speaking = rig `voice` > 0.08 or `jaw` activity (bridged 1.2 s); listening = the line's
user/partner speaking flag; else idle.

Rig -> human units (the inverse of app/prior.js defaults; override with flags if the controller scales differently):
  head_yaw/pitch/roll [rad] / (gain_head * pi/180) -> deg;  head_x/y/z [scene] / (gain_trans * 0.085) -> cm
  gaze_x/y [-1..1] * 30 / gain_gaze -> deg;  brow_l/r / gain_brow;  smile, squint, eye_wide as is
  blink = max(blink_l, blink_r): onset = upward crossing of 0.5, duration = time above 0.5
  arm = wrist speed from hand_{l,r}_{x,y,z} (arm lengths/s * 1.55 = shoulder widths/s), 1-s envelope

Usage: python3 tools/motion_stats.py avatar_log.jsonl [--ref docs/MOTION_STATS.json] [--out result.json]
       [--gain-head 1.3] [--gain-gaze 1.0] [--gain-brow 1.3] [--gain-trans 0.5]
"""
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "prior"))
import mstats  # noqa: E402

FPS = 30
COLS = mstats.COLS
KEYS = ["head_yaw_sd10s", "head_pitch_sd10s", "head_roll_sd10s", "head_yaw_speed_med", "head_pitch_speed_med",
        "head_roll_speed_med", "head_yaw_speed_p90", "head_pitch_speed_p90", "head_pitch_psd_0.1_0.5Hz",
        "head_pitch_psd_1.0_2.0Hz", "head_pitch_psd_4.0_8.0Hz", "head_yaw_psd_1.0_2.0Hz", "head_tx_sd10s_cm",
        "head_ty_sd10s_cm", "head_tz_sd10s_cm", "stillness_frac", "nod_per_min", "bignod_per_min", "blink_per_min",
        "blink_dur_ms_med", "gshift_per_min", "avert_frac", "avert_per_min", "avert_dur_s_med", "smile_sd",
        "smile_per_min", "brow_sd", "brow_per_min", "arm_speed_med", "gesture_frac"]


def load_log(path, params=None):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "params" in r and "t" not in r:
            params = r["params"]; continue
        if "rig_params" in r and "t" not in r:
            params = r["rig_params"]; continue
        if "t" not in r:
            continue
        if "v" in r:
            vals = r["v"]
        elif "a" in r and params:
            vals = dict(zip(params, r["a"]))
        else:
            continue
        rows.append((float(r["t"]), vals, r))
    rows.sort(key=lambda x: x[0])
    return rows


def resample(rows, name, t30, default=0.0):
    t = np.array([r[0] for r in rows]); v = np.array([float(r[1].get(name, np.nan)) for r in rows])
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return np.full(len(t30), default)
    return np.interp(t30, t[ok], v[ok])


def ctx(rows, t30, fn):
    t = np.array([r[0] for r in rows]); v = np.array([fn(r[2]) for r in rows], float)
    idx = np.clip(np.searchsorted(t, t30, side="right") - 1, 0, len(t) - 1)
    return v[idx]


def build(rows, a):
    t0, t1 = rows[0][0], rows[-1][0]
    t30 = np.arange(t0, t1, 1 / FPS)
    N = len(t30)
    g = lambda n, d=0.0: resample(rows, n, t30, d)
    M = np.full((N, len(COLS)), np.nan, np.float32)
    hd = a.gain_head * math.pi / 180
    M[:, 0], M[:, 1], M[:, 2] = g("head_yaw") / hd, g("head_pitch") / hd, g("head_roll") / hd
    ts = a.gain_trans * 0.085
    M[:, 3], M[:, 4], M[:, 5] = g("head_x") / ts, g("head_y") / ts, g("head_z") / ts
    M[:, 6], M[:, 7] = g("gaze_x") * 30 / a.gain_gaze, g("gaze_y") * 30 / a.gain_gaze
    bl = np.maximum(g("blink_l"), g("blink_r"))
    M[:, 8], M[:, 9] = g("blink_l"), g("blink_r")
    M[:, 10], M[:, 11] = g("brow_l") / a.gain_brow, g("brow_r") / a.gain_brow
    M[:, 12], M[:, 13], M[:, 14], M[:, 15] = g("smile"), g("squint"), g("eye_wide"), g("jaw")
    # arms: wrist targets (arm lengths) -> shoulder widths / s, 1-s envelope, log1p (the reference's arm column)
    sp = []
    for s in ("l", "r"):
        p = np.stack([g(f"hand_{s}_x"), g(f"hand_{s}_y"), g(f"hand_{s}_z")], 1)
        v = np.zeros(N); v[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) * FPS * 1.55
        sp.append(v)
    arm = np.convolve(np.mean(sp, 0), np.ones(30) / 30, "same")
    has_hands = any(f"hand_{s}_{c}" in r[1] for r in rows[:50] for s in ("l", "r") for c in ("x", "y", "z"))
    M[:, 16] = np.log1p(arm) if has_hands else np.nan      # no hand targets in the log -> arm metrics skipped
    # head angles relative to the log's median (like each real person's neutral)
    for j in range(8):
        M[:, j] -= np.nanmedian(M[:, j])
    # blinks
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
    # states
    st_given = ctx(rows, t30, lambda r: {"idle": 0, "speaking": 1, "listening": 2}.get(r.get("state"), -1))
    spk_flag = ctx(rows, t30, lambda r: 1.0 if r.get("speaking") else (0.0 if "speaking" in r else -1.0))
    lis_flag = ctx(rows, t30, lambda r: 1.0 if (r.get("listening") or r.get("user_speaking") or
                                                 (isinstance(r.get("partner"), dict) and r["partner"].get("speaking")))
                   else 0.0)
    voice = g("voice"); jaw = g("jaw")
    jstd = np.sqrt(np.maximum(np.convolve(jaw ** 2, np.ones(30) / 30, "same") - np.convolve(jaw, np.ones(30) / 30, "same") ** 2, 0))
    selfsp = np.where(spk_flag >= 0, spk_flag > 0, (np.convolve(voice, np.ones(9) / 9, "same") > 0.08) | (jstd > 0.035))
    st = np.zeros(N, int)
    if (st_given >= 0).any():
        st = np.where(st_given >= 0, st_given, 0).astype(int)
    else:
        sf = selfsp.copy()
        k = int(1.2 * FPS)                       # bridge pauses inside a turn
        last = -10 ** 9
        for i in range(N):
            if selfsp[i]:
                if 0 < i - last <= k:
                    sf[last:i] = True
                last = i
        st[sf] = 1; st[(lis_flag > 0) & ~sf] = 2
    f0 = ctx(rows, t30, lambda r: float(r.get("f0") or 0.0))
    seg = dict(M=M, blink=on, blinkdur=dur, st=st, selfsp=st == 1, partsp=lis_flag > 0, f0=f0 * (st == 1), pM=None)
    return seg, N


def compare(stats, ref):
    out = []; errs = []
    for s in ("speaking", "listening", "idle"):
        for k in KEYS:
            r = ref.get(s, {}).get(k); g = stats.get(s, {}).get(k)
            if r is None or g is None or not np.isfinite(g):
                continue
            e = abs(g - r["med"]) * math.log(10) / 2 if "psd" in k else abs(math.log((g + 1e-3) / (r["med"] + 1e-3)))
            errs.append(e)
            out.append(dict(state=s, metric=k, avatar=g, real_med=r["med"], real_q25=r["q25"], real_q75=r["q75"],
                            inside_iqr=bool(r["q25"] <= g <= r["q75"]), log_err=e))
    return out, (float(np.mean(errs)) if errs else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--ref", default=str(HERE.parent / "docs" / "MOTION_STATS.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--params", default=None, help="comma list of rig params for 'a' arrays without a header line")
    ap.add_argument("--gain-head", type=float, default=1.3); ap.add_argument("--gain-gaze", type=float, default=1.0)
    ap.add_argument("--gain-brow", type=float, default=1.3); ap.add_argument("--gain-trans", type=float, default=0.5)
    a = ap.parse_args()
    rows = load_log(a.log, a.params.split(",") if a.params else None)
    if len(rows) < 60:
        sys.exit("log too short")
    seg, N = build(rows, a)
    stats = mstats.person_stats([seg])
    ref = json.loads(Path(a.ref).read_text())
    cmp_, err = compare(stats, ref)
    res = dict(log=a.log, seconds=N / FPS, minutes_by_state={s: stats.get(s, {}).get("minutes", 0) for s in ("speaking", "listening", "idle")},
               mean_abs_log_err=err, frac_inside_real_iqr=float(np.mean([c["inside_iqr"] for c in cmp_])) if cmp_ else None,
               comparison=cmp_, couplings={k: v for k, v in stats.get("coupling", {}).items() if not isinstance(v, list)},
               avatar_stats={k: v for k, v in stats.items() if k != "coupling"})
    print(f"{a.log}: {N / FPS:.0f} s; minutes by state {res['minutes_by_state']}")
    print(f"mean |log(avatar/real)| = {err if err is None else round(err, 3)}; inside real IQR: {res['frac_inside_real_iqr']}")
    for c in cmp_:
        print(f"  {c['state']:9s} {c['metric']:26s} avatar {c['avatar']:8.3g}   real {c['real_med']:8.3g} "
              f"[{c['real_q25']:.3g}, {c['real_q75']:.3g}]  {'ok' if c['inside_iqr'] else '--'}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
