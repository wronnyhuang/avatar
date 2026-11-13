"""runtime probe: closed-loop rollouts of a bundle on the val pack with runtime variants, fast enough to sweep.

    python -I tools/train/rt_probe.py prep --bundle <enc bundle> --pack <v3 pack> --out jobs.pkl [--seconds 60]
           [--max-sessions 12] [--noise-profile noise_profile.json]
    python -I tools/train/rt_probe.py run --jobs jobs.pkl --bundle <dir> --tag <name> --rt '<json UnifiedPlanner kw>'
           [--workers 8] [--threads 1] [--n 24] [--outdir dir] [--save]
    python -I tools/train/rt_probe.py table <outdir>/*.json

prep cuts evaluate.py --pack's NORMAL-condition jobs once (the pack decode is the slow part) and pickles them.
run steps every job with UnifiedPlanner(bundle, pcm=False, **rt) (any runtime kwarg: temperature, guidance, steps,
noise_rho, temperature_groups, box, blend, events={mode, gain, feed_history, thresholds?, refractory?}, ...), records
the motion, the per-frame event probabilities and the fired events, and summarises:
  * evaluate.stats (gen vs real; listening / speaking split for the rates) incl. seam spectra;
  * event firing per class: fired / label rate per min (listening / speaking), p quantiles, and the threshold that
    would match the label rate on THESE closed-loop p traces (rising-crossing + refractory, as the runtime fires);
  * REALISM proxy = the scorecard's window-feature Fréchet distance (evalkit/scorecard.py seg_windows + Realism vs the
    held-out Seamless TEST reference), gen minus the real val people on the same windows (real = 0), per class + top
    feature gaps. Not the scorecard (other clips, other user path) - a fast ranking signal.
events.thresholds / events.refractory in --rt override the bundle's config["events"] values (runtime patch).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import get_context
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "train"))
from canon1 import spec  # noqa: E402
from canon1 import paths as _paths  # noqa: E402

FPS = spec.FPS
EV_NAMES = ("nod", "blink", "smile")


# ------------------------------------------------------------------------------------------------ prep
def prep(a):
    import evaluate as EV
    ns = SimpleNamespace(bundle=a.bundle, pack=a.pack, split=a.split, datasets=None, seconds=a.seconds,
                         max_sessions=a.max_sessions, style=None, style_sweep=0, no_ablate=True, drop="", long_minutes=0)
    noise = EV.load_noise(a.noise_profile)
    jobs, n, meta = EV.pack_jobs(ns, {}, noise)
    jobs = [j for j in jobs if j["cond"] == "normal"]
    for j in jobs:
        j.pop("bundle", None); j.pop("kw", None)
    with open(a.out, "wb") as f:
        pickle.dump(dict(jobs=jobs, meta=meta, noise=noise), f, protocol=4)
    print(f"{len(jobs)} jobs ({n} dyads) -> {a.out}")


# ------------------------------------------------------------------------------------------------ run
def _make_planner(bundle, rt, seed, threads):
    from unified.runtime import UnifiedPlanner
    return UnifiedPlanner(bundle, threads=threads, seed=seed, pcm=False, **json.loads(json.dumps(rt)))


def one(args):
    job, bundle, rt, threads = args
    import evaluate as EV
    p = _make_planner(bundle, rt, job["seed"], threads)
    f32 = lambda x: None if x is None else np.asarray(x, np.float32)      # noqa: E731
    um, uk, ua = f32(job["um"]), np.asarray(job["uk"], np.float32), f32(job["ua"])
    if job.get("noise") is not None:
        um = um + EV.ar1_noise(len(um), job["noise"], np.random.default_rng(job["seed"]))
    sa_rows, fu, fs = f32(job["sa_rows"]), job.get("fu"), job.get("fs")
    L = p.self_audio_lookahead
    shift = int(getattr(p, "user_feat_shift", 0) or 0)
    enc = p.feat_dim > 0
    T = len(um)
    gen = np.zeros((T, spec.D_MOTION), np.float32)
    evp = np.full((T, max(1, p.n_events)), np.nan, np.float32)
    fired = np.zeros((T, max(1, p.n_events)), np.int8)
    step_ms, plan_ms = np.zeros(T, np.float32), []
    for k in range(T):
        j = k + L
        sa = None if j >= len(sa_rows) else sa_rows[j]
        fk = {}
        if enc:
            fk = dict(user_feat=None if k < shift else EV._row(fu, k - shift), self_feat_ahead=EV._row(fs, j))
        t1 = time.perf_counter()
        o = p.step(um[k], uk[k], ua[k], sa, **fk)
        step_ms[k] = (time.perf_counter() - t1) * 1000
        gen[k] = o["motion"]
        if o.get("plan_ms") is not None:
            plan_ms.append(o["plan_ms"])
        if o.get("event_p") is not None:
            evp[k] = o["event_p"]
        for nme in o.get("events") or ():
            fired[k, p.event_names.index(nme)] = 1
    return dict(key=job["key"], gen=gen, real=f32(job["real"]), user=um, listen=np.asarray(job["listen"], bool),
                user_vad=np.asarray(job["ua"])[:, 41] > 0.5, labels=job.get("labels"), evp=evp, fired=fired,
                step_ms=step_ms, plan_ms=np.asarray(plan_ms, np.float32), runtime=p.runtime,
                thr=None if not p.n_events else p.ev_thr.tolist(), refr=None if not p.n_events else p.ev_refr.tolist())


def fire_count(p, th, refr):
    """rising crossings of th in one trace p (T,) with refractory (the runtime's threshold rule)."""
    q = np.where(np.isfinite(p), p, 0)
    up = np.nonzero((q[1:] >= th) & (q[:-1] < th))[0] + 1
    n, last = 0, -10 ** 9
    for u in up:
        if u - last >= refr:
            n += 1; last = u
    return n


def rate_match_threshold(ps, n_target, refr):
    """the HIGHEST threshold whose rising-crossing + refractory fire count over the traces ps (list of (T,)) reaches
    n_target (the count is not monotonic: a very low threshold is crossed rarely because p stays above it)."""
    grid = np.arange(0.99, 0.004, -0.005)
    for th in grid:
        if sum(fire_count(p, th, refr) for p in ps) >= n_target:
            return float(th)
    return float("nan")


_REAL = None


def realism_tools():
    global _REAL
    if _REAL is None:
        sys.path.insert(0, os.path.join(ROOT, "evalkit"))
        import scorecard as SC
        SC._kit()
        ref = SC.reference_windows(None)
        _REAL = (SC, SC.Realism(ref))
    return _REAL


def windows_of(m, listen, user_vad):
    SC, _ = realism_tools()
    M = SC.M
    Tb = M.table_from_canon(m)
    own = M._bridge(~np.asarray(listen, bool), int(1.2 * FPS))
    oth = np.asarray(user_vad, bool)
    st = np.where(own, 1, np.where(oth, 2, 0))
    seg = M.make_seg(Tb, st, oth, own)
    return SC.seg_windows(seg)


def summarize(res, tag, rt, bundle):
    import evaluate as EV
    rows_g, rows_r = [], []
    for r in res:
        rows_g.append(EV.stats(r["gen"], r["user"], r["listen"]))
        rows_r.append(EV.stats(r["real"], r["user"], r["listen"]))
    keys = ["head_speed", "head_speed_p90", "body_speed", "still_pct", "nods_per_min", "blinks_per_min",
            "smiles_per_min", "smile_mean", "look_away_listen_pct", "look_away_speak_pct", "smile_coupling", "brow_sd",
            "head_pitch_sd", "head_yaw_sd", "hp13_log10", "hp48_log10", "hp815_log10", "hp_line10", "head_jerk"]
    mean = lambda rows, k: float(np.nanmean([x.get(k, np.nan) for x in rows]))      # noqa: E731
    out = {"tag": tag, "bundle": bundle, "rt": rt, "n": len(res), "runtime": res[0]["runtime"],
           "gen": {k: mean(rows_g, k) for k in keys}, "real": {k: mean(rows_r, k) for k in keys}}
    # split listening / speaking rates (gen + real), gaze-away and eye movement SD
    for part, sel in (("L", lambda r: r["listen"]), ("S", lambda r: ~r["listen"])):
        for which in ("gen", "real"):
            n_nod = n_bl = n_sm = 0
            mins = 0.0
            for r in res:
                m = r[which]
                s = sel(r)
                if s.sum() < 30:
                    continue
                pitch = m[:, spec.COL["neck_x"]] + m[:, spec.COL["head_x"]]
                bp = EV._smooth(pitch, 5) - EV._smooth(pitch, 45)
                pk = np.nonzero((bp[1:-1] > bp[:-2]) & (bp[1:-1] >= bp[2:]) & (bp[1:-1] > 0.035))[0] + 1
                last, nods = -99, []
                for q in pk:
                    if q - last >= 12:
                        nods.append(q); last = q
                bl = m[:, EV.BL].mean(1)
                blinks = np.nonzero((bl[1:] > 0.5) & (bl[:-1] <= 0.5))[0] + 1
                sm = m[:, EV.SM].mean(1)
                s5 = np.convolve(np.pad(sm.astype(np.float64), 2, mode="edge"), np.ones(5) / 5, "valid")
                ups = np.nonzero((s5[1:] >= 0.4) & (s5[:-1] < 0.4))[0] + 1
                last, smiles = -10 ** 9, []
                for q in ups:
                    if q - last >= FPS:
                        smiles.append(q); last = q
                n_nod += int(s[np.asarray(nods, int)].sum()) if len(nods) else 0
                n_bl += int(s[blinks].sum()) if len(blinks) else 0
                n_sm += int(s[np.asarray(smiles, int)].sum()) if len(smiles) else 0
                mins += s.sum() / FPS / 60
            out[which][f"nods_{part}"] = n_nod / max(mins, 1e-6)
            out[which][f"blinks_{part}"] = n_bl / max(mins, 1e-6)
            out[which][f"smiles_{part}"] = n_sm / max(mins, 1e-6)
    # the judge's on-screen complaints: eyes held half-closed, asymmetric blinks, head roll, arm excursions
    def art(m):
        bl = m[:, EV.BL]
        blm = bl.mean(1)
        roll = m[:, spec.COL["neck_z"]] + m[:, spec.COL["head_z"]]
        arm = m[:, spec.GROUPS["arm_l"] + spec.GROUPS["arm_r"]]
        return dict(eye_closed_frac=float((blm > 0.5).mean()), blink_asym=float(np.abs(bl[:, 0] - bl[:, 1]).mean()),
                    roll_sd_deg=float(np.degrees(roll.std())),
                    arm_dev_gt05=float((np.abs(arm - np.median(arm, 0)).max(1) > 0.5).mean()))
    for which in ("gen", "real"):
        a_ = [art(r[which]) for r in res]
        for k in a_[0]:
            out[which][k] = float(np.mean([x[k] for x in a_]))
    # eye-in-head SD (gaze cols) - the scorecard's eye_sd gap
    gc = spec.GROUPS["gaze"]
    out["gen"]["gaze_sd"] = float(np.mean([r["gen"][:, gc].std(0).mean() for r in res]))
    out["real"]["gaze_sd"] = float(np.mean([r["real"][:, gc].std(0).mean() for r in res]))
    # events
    if res[0]["thr"] is not None:
        ev = {}
        for c, nm in enumerate(EV_NAMES):
            d = {}
            for part, sel in (("L", lambda r: r["listen"]), ("S", lambda r: ~r["listen"]), ("all", lambda r: np.ones_like(r["listen"]))):
                nf = sum(int(r["fired"][sel(r), c].sum()) for r in res)
                nl = 0
                for r in res:
                    if r["labels"] is None:
                        continue
                    lab = np.asarray(r["labels"])[:, c] != 0
                    on = np.zeros_like(lab); on[0] = lab[0]; on[1:] = lab[1:] & ~lab[:-1]
                    nl += int(on[sel(r)].sum())
                mins = sum(sel(r).sum() for r in res) / FPS / 60
                d[f"fired_{part}"] = nf / mins
                d[f"label_{part}"] = nl / mins
            ps = [r["evp"][:, c] for r in res]
            allp = np.concatenate(ps)
            allp = allp[np.isfinite(allp)]
            d["p_q"] = [float(np.percentile(allp, q)) for q in (50, 90, 99, 99.9)]
            d["thr"] = res[0]["thr"][c]
            d["refr"] = int(res[0]["refr"][c])
            d["scale_match"] = float(d["label_all"] * sum(len(r["listen"]) for r in res) / FPS / 60 / max(float(allp.sum()), 1e-9))
            mins_all = sum(len(r["listen"]) for r in res) / FPS / 60
            d["thr_match"] = rate_match_threshold(ps, d["label_all"] * mins_all, res[0]["refr"][c])
            # per-state thresholds
            for part, selm in (("L", True), ("S", False)):
                pp = [np.where(r["listen"] == selm, r["evp"][:, c], 0) for r in res]
                m_ = sum((r["listen"] == selm).sum() for r in res) / FPS / 60
                d[f"thr_match_{part}"] = rate_match_threshold(pp, d[f"label_{part}"] * m_, res[0]["refr"][c])
            ev[nm] = d
        out["events"] = ev
    # realism proxy
    try:
        SC, R = realism_tools()
        Wg, Kg, Wr, Kr = [], [], [], []
        for r in res:
            w, k = windows_of(r["gen"], r["listen"], r["user_vad"])
            Wg.append(w); Kg.append(k)
            w, k = windows_of(r["real"], r["listen"], r["user_vad"])
            Wr.append(w); Kr.append(k)
        Wg, Kg, Wr, Kr = map(np.concatenate, (Wg, Kg, Wr, Kr))
        fg, fr = R.fd(Wg, Kg), R.fd(Wr, Kr)
        per = {c: fg[c] - fr[c] for c in (0, 1)}
        out["realism"] = float(np.nanmean(list(per.values())))
        out["realism_per_class"] = [per[0], per[1]]
        out["realism_fd_real"] = [fr[0], fr[1]]
        out["realism_gaps"] = R.contrib(Wg, Kg, top=6)
    except Exception as e:  # noqa: BLE001
        out["realism_error"] = repr(e)[:300]
    sm = np.concatenate([r["step_ms"] for r in res])
    pm = np.concatenate([r["plan_ms"] for r in res])
    out["step_ms"] = [float(np.percentile(sm, q)) for q in (50, 90, 99)] + [float(sm.mean())]
    out["plan_ms"] = [float(np.percentile(pm, q)) for q in (50, 99)] if len(pm) else None
    return out


def run(a):
    with open(a.jobs, "rb") as f:
        J = pickle.load(f)
    jobs = J["jobs"][: a.n]
    rt = json.loads(a.rt) if a.rt else {}
    tic = time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        res = pool.map(one, [(j, a.bundle, rt, a.threads) for j in jobs], chunksize=1)
    wall = time.time() - tic
    out = summarize(res, a.tag, rt, a.bundle)
    out["wall_s"] = round(wall, 1)
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, f"{a.tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    if a.save:
        np.savez_compressed(os.path.join(a.outdir, f"{a.tag}.npz"),
                            gen=np.stack([r["gen"] for r in res]).astype(np.float16),
                            evp=np.stack([r["evp"] for r in res]), fired=np.stack([r["fired"] for r in res]),
                            listen=np.stack([r["listen"] for r in res]))
    print(line(out))


def line(o):
    g, r = o["gen"], o["real"]
    s = (f"{o['tag']:<28} R {o.get('realism', float('nan')):6.2f} [{o.get('realism_per_class', [0, 0])[0]:5.1f}/"
         f"{o.get('realism_per_class', [0, 0])[1]:5.1f}] hs {g['head_speed']:.3f}/{r['head_speed']:.3f} still "
         f"{g['still_pct']:4.0f}/{r['still_pct']:3.0f} nodL {g['nods_L']:4.1f}/{r['nods_L']:4.1f} blk "
         f"{g['blinks_per_min']:4.1f}/{r['blinks_per_min']:4.1f} smL {g['smiles_L']:4.1f}/{r['smiles_L']:4.1f} smm "
         f"{g['smile_mean']:.2f}/{r['smile_mean']:.2f} awL {g['look_away_listen_pct']:4.1f}/{r['look_away_listen_pct']:4.1f}"
         f" eyesC {g.get('eye_closed_frac', float('nan')):.2f}/{r.get('eye_closed_frac', float('nan')):.2f}"
         f" gsd {g['gaze_sd']:.3f}/{r['gaze_sd']:.3f} cpl {g['smile_coupling']:.2f}/{r['smile_coupling']:.2f} 815 "
         f"{g['hp815_log10']:.2f}/{r['hp815_log10']:.2f} ms {o['step_ms'][3]:.1f}")
    if o.get("events"):
        s += " | ev " + " ".join(f"{k[0]} {v['fired_all']:.0f}/{v['label_all']:.0f}" for k, v in o["events"].items())
    if o.get("realism_gaps"):
        s += " | gaps " + ",".join(f"{k}:{v:.1f}" for k, v in o["realism_gaps"][:4])
    return s


def calib(a):
    """closed-loop event calibration from a run json (best with events feed_history false or mode off: then p does not
    depend on firing and the rate match is exact): prints the runtime events dict to merge."""
    with open(a.run) as f:
        o = json.load(f)
    ev = o["events"]
    names = list(ev)
    out = {"thresholds": [round(ev[n]["thr_match_L" if a.split else "thr_match"], 4) for n in names]}
    if a.split:
        out["thresholds_speak"] = [round(ev[n]["thr_match_S"], 4) for n in names]
    out["scale"] = [round(ev[n]["scale_match"], 4) for n in names]
    print(json.dumps(out))


def table(a):
    for p in a.files:
        with open(p) as f:
            print(line(json.load(f)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prep")
    p1.add_argument("--bundle", required=True)
    p1.add_argument("--pack", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--split", default="val")
    p1.add_argument("--seconds", type=float, default=60)
    p1.add_argument("--max-sessions", type=int, default=12)
    p1.add_argument("--noise-profile", default=str(_paths.NOISE_PROFILE))
    p2 = sub.add_parser("run")
    p2.add_argument("--jobs", required=True)
    p2.add_argument("--bundle", required=True)
    p2.add_argument("--tag", required=True)
    p2.add_argument("--rt", default="{}")
    p2.add_argument("--workers", type=int, default=8)
    p2.add_argument("--threads", type=int, default=1)
    p2.add_argument("--n", type=int, default=24)
    p2.add_argument("--outdir", default=str(_paths.WORK / "runs/p2rt/probe"))
    p2.add_argument("--save", action="store_true")
    p4 = sub.add_parser("calib")
    p4.add_argument("run")
    p4.add_argument("--split", action="store_true", help="separate thresholds while the avatar speaks")
    p3 = sub.add_parser("table")
    p3.add_argument("files", nargs="+")
    a = ap.parse_args()
    {"prep": prep, "run": run, "table": table, "calib": calib}[a.cmd](a)


if __name__ == "__main__":
    main()
