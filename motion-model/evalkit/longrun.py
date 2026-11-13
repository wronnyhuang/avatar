#!/usr/bin/env python3
"""Latency and the 10-minute closed-loop stability rollout of the unified planner (and latency of the rules controller).

  python -I evalkit/longrun.py latency [--bundle B] [--threads 2] [--tag t]
  python -I evalkit/longrun.py stability [--bundle B] [--minutes 10] [--tag t]

latency: per-step wall time of UnifiedPlanner.step at `threads` torch threads on the host CPU over a real 36 s clip
  (plan_ms = the frames where it re-planned; step_ms = every frame, i.e. amortized), p50/p99/max; the Fitter's ms per
  frame (same pipeline, from the clip cache); the rules controller's node step ms from its replay stats (all clips).
  On a shared, busy host these are upper bounds; an `isolated` pass repeats the planner on one clip 3x.
stability: the Seamless test clips' USER streams (fitter canon + mic audio42) and AVATAR voices concatenated to
  `minutes` of conversation, fed causally to one planner instance (its own committed output feeds back = closed loop).
  Checks: finite; per column inside the held-out humans' [p0.5, p99.5] widened by 25 % of that range (fraction of
  frames out, worst columns); drift = per-minute SD of the human-unit table channels (head yaw/pitch/roll, gaze,
  brows, smile, blink) relative to minute 1 (min/max ratio); longest frozen stretch (all table channels move < 1e-3
  per frame); high-frequency head energy (4-8 Hz pitch PSD) vs the human reference.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import E, T, DATA, FPS, atomic_json, canon_spec  # noqa: E402
import systems as SY  # noqa: E402
import measures as M  # noqa: E402

OUT = E / "measures"


def pct(a, q):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if len(a) else float("nan")


def planner(bundle, threads, seed=0, **kw):
    sys.path.insert(0, str(SY.MODEL_DIR))
    from unified.runtime import UnifiedPlanner  # noqa
    return UnifiedPlanner(str(bundle) if bundle else None, threads=threads, seed=seed, **kw)


def cmd_latency(bundle, threads, tag, clip_id=None):
    man = SY.manifest()
    sl = sorted(c for c, v in man.items() if v["group"] == "seamless" and (E / "clips" / c / "user_fit.npz").exists())
    cid = clip_id or (sl[0] if sl else None)
    res = {"threads": threads, "bundle": str(bundle), "clip": cid, "host_load": open("/proc/loadavg").read().split()[:3]}
    if cid:
        clip = man[cid]
        um, uk = SY.user_fit(clip)
        ua = np.load(E / "clips" / cid / "audio42_user.npy"); sa = np.load(E / "clips" / cid / "audio42_self.npy")
        runs = []
        for rep in range(3):
            p = planner(bundle, threads)
            L = int(getattr(p, "self_audio_lookahead", 6))
            plan, step = [], []
            for s in range(len(um)):
                k = min(len(um) - 1, s + L)
                t0 = time.perf_counter()
                r = p.step(um[s] if uk[s].any() else None, uk[s] if uk[s].any() else None, ua[s], sa[k])
                step.append((time.perf_counter() - t0) * 1000)
                if r.get("plan_ms") is not None:
                    plan.append(float(r["plan_ms"]))
            runs.append({"plan_ms": plan, "step_ms": step})
        plan = np.concatenate([np.asarray(r["plan_ms"], float) for r in runs]) if runs else np.array([])
        step = np.concatenate([np.asarray(r["step_ms"], float) for r in runs])
        res["unified"] = {"plan_ms_p50": pct(plan, 50), "plan_ms_p99": pct(plan, 99), "plan_ms_max": pct(plan, 100),
                          "n_plans": int(len(plan)), "step_ms_p50": pct(step, 50), "step_ms_p99": pct(step, 99),
                          "step_ms_mean": float(np.mean(step)), "step_ms_max": float(np.max(step)), "n_steps": int(len(step))}
        z = np.load(E / "clips" / cid / "user_fit.npz")
        if "fit_ms" in z.files:
            res["fitter"] = {"ms_p50": pct(z["fit_ms"], 50), "ms_p99": pct(z["fit_ms"], 99), "ms_mean": float(np.mean(z["fit_ms"]))}
    # rules controller (node, per-clip replay stats)
    rs = []
    for p in (E / "runs/rules").glob("*/rig.jsonl.stats.json"):
        j = json.loads(p.read_text())
        rs.append(j["step_ms"])
    if rs:
        res["rules"] = {"step_ms_p50_med": float(np.median([r["p50"] for r in rs])), "step_ms_p99_max": float(np.max([r["p99"] for r in rs])),
                        "step_ms_p99_med": float(np.median([r["p99"] for r in rs])), "n_clips": len(rs)}
    atomic_json(OUT / (f"latency_{tag}.json" if tag else "latency.json"), res)
    print(json.dumps(res, indent=1))
    return res


def human_bounds():
    """per-column [p0.5, p99.5] of the held-out Seamless TEST humans (valid frames), cached."""
    cp = OUT / "human_col_bounds.npz"
    if cp.exists():
        z = np.load(cp)
        return z["lo"], z["hi"]
    spec = canon_spec()
    S = json.loads((DATA / "splits.json").read_text())
    xs = []
    for k, v in sorted(S["sessions"].items()):
        if v != "test" or not k.startswith("seamless:"):
            continue
        p = DATA / "canon1/seamless" / (k.split(":", 1)[1] + ".npz")
        if not p.exists():
            continue
        d = spec.load_session(str(p))
        for q in range(d["motion"].shape[0]):
            cm = spec.col_mask(d["mask"][q]).astype(bool)
            x = d["motion"][q][::5].copy(); x[~cm[::5]] = np.nan
            xs.append(x)
    X = np.concatenate(xs)
    lo, hi = np.nanpercentile(X, 0.5, axis=0), np.nanpercentile(X, 99.5, axis=0)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(cp, lo=lo, hi=hi)
    return lo, hi


def cmd_stability(bundle, minutes, tag, threads=2):
    spec = canon_spec()
    man = SY.manifest()
    sl = sorted(c for c, v in man.items() if v["group"] == "seamless" and (E / "clips" / c / "user_fit.npz").exists())
    need = int(minutes * 60 * FPS)
    U, K, UA, SA, src = [], [], [], [], []
    while sum(len(u) for u in U) < need and sl:
        for cid in sl:
            um, uk = SY.user_fit(man[cid])
            U.append(um); K.append(uk); src.append(cid)
            UA.append(np.load(E / "clips" / cid / "audio42_user.npy")); SA.append(np.load(E / "clips" / cid / "audio42_self.npy"))
            if sum(len(u) for u in U) >= need:
                break
    um, uk, ua, sa = (np.concatenate(x)[:need] for x in (U, K, UA, SA))
    p = planner(bundle, threads)
    L = int(getattr(p, "self_audio_lookahead", 6))
    out, step = [], []
    for s in range(need):
        k = min(need - 1, s + L)
        t0 = time.perf_counter()
        r = p.step(um[s] if uk[s].any() else None, uk[s] if uk[s].any() else None, ua[s], sa[k])
        step.append((time.perf_counter() - t0) * 1000)
        out.append(np.asarray(r["motion"], np.float32))
    X = np.stack(out)
    np.save(OUT / (f"stability_{tag}.npy" if tag else "stability.npy"), X.astype(np.float16))
    res = {"minutes": minutes, "frames": need, "bundle": str(bundle), "source_clips": src, "finite": bool(np.isfinite(X).all())}
    lo, hi = human_bounds()
    rng_ = hi - lo
    lo2, hi2 = lo - 0.25 * rng_, hi + 0.25 * rng_
    out_mask = (X < lo2) | (X > hi2)
    res["frac_frames_any_col_out"] = float(out_mask.any(1).mean())
    res["frac_values_out"] = float(out_mask.mean())
    by_col = out_mask.mean(0)
    worst = np.argsort(-by_col)[:8]
    res["worst_cols"] = [{"col": spec.MOTION_COLS[i], "frac_out": float(by_col[i])} for i in worst if by_col[i] > 0]
    res["by_group_frac_out"] = {g: float(out_mask[:, spec.GROUPS[g]].any(1).mean()) for g in spec.GROUP_NAMES}
    Tb = M.table_from_canon(X.astype(np.float64))
    chans = {"yaw": 0, "pitch": 1, "roll": 2, "gx": 6, "gy": 7, "brow_l": 10, "smile": 12, "blink_l": 8}
    per_min = {}
    for name, j in chans.items():
        sds = [float(np.nanstd(Tb[a:a + 1800, j])) for a in range(0, need - 1799, 1800)]
        per_min[name] = sds
    res["per_minute_sd"] = per_min
    res["drift_ratio_min_max"] = {n: [float(min(v) / max(v[0], 1e-6)), float(max(v) / max(v[0], 1e-6))] for n, v in per_min.items() if v}
    mv = np.abs(np.diff(np.nan_to_num(Tb[:, [0, 1, 2, 6, 7, 10, 11, 12]]), axis=0)).max(1)
    frozen = mv < 1e-3
    longest = max((b - a for a, b in M.mstats._runs(frozen)), default=0)
    res["longest_frozen_s"] = longest / FPS
    from scipy.signal import welch
    f, P = welch(Tb[:, 1] - np.nanmean(Tb[:, 1]), fs=FPS, nperseg=120)
    res["head_pitch_psd_4_8Hz_log10"] = float(np.log10(P[(f >= 4) & (f < 8)].mean() + 1e-9))
    res["step_ms_p50"] = pct(step, 50); res["step_ms_p99"] = pct(step, 99)
    # verdict: stays in range (<1 % frames with any column out of the widened human range), finite, no drift beyond
    # 0.2x-5x of minute 1 on the table channels, no frozen stretch > 2 s
    dr = [v for v in res["drift_ratio_min_max"].values()]
    res["stable"] = bool(res["finite"] and res["frac_frames_any_col_out"] < 0.01 and res["longest_frozen_s"] <= 2.0
                         and all(a >= 0.2 and b <= 5.0 for a, b in dr))
    atomic_json(OUT / (f"stability_{tag}.json" if tag else "stability.json"), res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_minute_sd", "source_clips")}, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["latency", "stability"])
    ap.add_argument("--bundle", default=str(T / "runs/BEST")); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--minutes", type=float, default=10.0); ap.add_argument("--tag", default="")
    ap.add_argument("--clip")
    a = ap.parse_args()
    bundle = None if a.bundle in ("", "none", "stub") else Path(a.bundle).resolve()
    if a.cmd == "latency":
        cmd_latency(bundle, a.threads, a.tag, a.clip)
    else:
        cmd_stability(bundle, a.minutes, a.tag, a.threads)


if __name__ == "__main__":
    main()
