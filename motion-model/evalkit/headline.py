#!/usr/bin/env python3
"""EVAL headline measures: one table, every system, the SAME pipeline (rig space = what is on screen).

  python -I evalkit/headline.py --runs rules,gru-prior,copy-delay,unified-<tag>,human [--base rules] [--out f.json]

Per system (clip groups: seamless = the 29 Seamless test sessions where the real partner `human` exists; youtube = the
21 held-out suite clips; all):
  human-range   mean |log err| vs the held-out Seamless TEST people (ref_seamless_test.json) + fraction of cells inside
                their IQR (measures.compare_ref); head speed (deg/s, median |angular velocity|, listening / speaking);
                gaze-away fraction while listening (mstats avert_frac: >10 deg from the person's median direction);
                nods/min while listening; blinks/min; smiles/min; head jerk (deg/s^3, median |3rd difference|) and
                4-8 Hz head-pitch power (log10) = smoothness
  reactions     event-locked (measures.REACTIONS, 0.2-2.0 s): hit rate, chance, excess = rate - chance, median lag;
                smile x-corr with the real user minus with a shuffled user
  vs base       paired clip-bootstrap (400x) of (system - base) on |log err|, gaze-away, and each reaction's excess
Space: `rig` converts every system's rig.jsonl with tools/motion_stats.py (rules' own units; canon systems after the
retarget = exactly what renders); `canon` (canon systems only) is the data pipeline's FK table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import E, FPS, atomic_json  # noqa: E402
import measures as M  # noqa: E402
import analyze as AN  # noqa: E402
import mstats  # noqa: E402

GROUPS = ("seamless", "youtube", "all")
PAIRS = [f"{a}->{b}" for a, b in M.REACTIONS]


def segs_for(run, man, ids, space):
    out = {}
    for cid in ids:
        try:
            s = AN.seg_for(run, man, cid, space)
        except Exception as e:  # noqa: BLE001
            print("seg", run, cid, repr(e)[:200], file=sys.stderr)
            s = None
        if s is not None:
            out[cid] = s
    return out


def head_speed(segs, state):
    v = []
    for s in segs:
        ang = np.nan_to_num(s["M"][:, :3])
        sp = np.linalg.norm(np.gradient(ang, axis=0) * FPS, axis=1)
        m = s["st"] == state if state is not None else np.ones(len(sp), bool)
        v.append(sp[m])
    v = np.concatenate(v) if v else np.array([])
    return float(np.median(v)) if len(v) > 30 else float("nan")


def head_jerk(segs):
    v = []
    for s in segs:
        ang = np.nan_to_num(s["M"][:, :3])
        j = np.diff(ang, n=3, axis=0) * FPS ** 3
        v.append(np.linalg.norm(j, axis=1))
    v = np.concatenate(v) if v else np.array([])
    return float(np.median(v)) if len(v) > 30 else float("nan")


def stat_block(segs, ref):
    st = mstats.person_stats(list(segs))         # events are cached per seg (seg["_ev"]): segs never change
    err, inside, _ = M.compare_ref(st, ref)
    g = lambda state, k: float((st.get(state) or {}).get(k, float("nan")))  # noqa: E731
    return {"abs_log_err": err, "inside_iqr": inside,
            "head_speed_listen": head_speed(segs, 2), "head_speed_speak": head_speed(segs, 1),
            "gaze_away_listen": g("listening", "avert_frac"), "gaze_away_speak": g("speaking", "avert_frac"),
            "nods_per_min_listen": g("listening", "nod_per_min"), "nods_per_min_speak": g("speaking", "nod_per_min"),
            "blinks_per_min": g("all", "blink_per_min"), "smiles_per_min": g("all", "smile_per_min"),
            "stillness_listen": g("listening", "stillness_frac"),
            "head_jerk_med": head_jerk(segs), "head_pitch_psd_4_8": g("all", "head_pitch_psd_4.0_8.0Hz"),
            "minutes": {k: (st.get(k) or {}).get("minutes") for k in M.STATE_NAMES}}


def reactions(run, man, ids, segs, space):
    per = {k: [] for k in PAIRS}
    xr, xs = [], []
    for i, cid in enumerate(ids):
        s = segs.get(cid)
        if s is None:
            continue
        N = len(s["M"])
        Ut = s["pM"]
        rc = M.reaction_counts(M.user_events(Ut, s["partsp"], N), M.avatar_events(s["M"], s["st"]), N)
        for k, v in rc.items():
            per[k].append((cid, v))
        xr.append(M.smile_xcorr(Ut[:, 12], s["M"][:, 12]))
        other = ids[(i + 1) % len(ids)]
        Uo = M.fit_len(AN.user_tab(other), N)
        xs.append(M.smile_xcorr(Uo[:, 12], s["M"][:, 12]))
    out = {}
    for k, rows in per.items():
        n = sum(r["n"] for _, r in rows); h = sum(r["hits"] for _, r in rows)
        ch = sum(r["chance_hits"] for _, r in rows) / max(1, sum(r["chance_n"] for _, r in rows))
        lat = [x for _, r in rows for x in r["lat"]]
        rate = h / n if n else float("nan")
        out[k] = {"n": n, "hits": h, "rate": rate, "chance": ch, "excess": rate - ch if n else float("nan"),
                  "lag_med_s": float(np.median(lat)) if lat else float("nan"), "n_lat": len(lat)}
    xr, xs = np.array(xr, float), np.array(xs, float)
    ok = np.isfinite(xr) & np.isfinite(xs)
    out["smile_xcorr_gain"] = float(np.mean(xr[ok] - xs[ok])) if ok.any() else float("nan")
    out["smile_xcorr_real"] = float(np.nanmean(xr)) if np.isfinite(xr).any() else float("nan")
    out["_per_clip"] = {k: {cid: r for cid, r in rows} for k, rows in per.items()}
    return out


def _excess(rows):
    n = sum(r["n"] for r in rows)
    if not n:
        return float("nan")
    ch = sum(r["chance_hits"] for r in rows) / max(1, sum(r["chance_n"] for r in rows))
    return sum(r["hits"] for r in rows) / n - ch


def _avert_counts(seg):
    ev = seg.setdefault("_ev", mstats.events(seg))
    m = seg["st"] == 2
    return float(ev["avert_mask"][m].sum()), float(m.sum())


def paired_vs(base, sysr, segs_b, segs_s, ref, ids, nboot=2000, nboot_err=100, seed=3, with_err=True):
    """paired clip bootstrap of (system - base): each reaction's excess and gaze-away while listening (cheap, per-clip
    counts, `nboot`), human-range |log err| / inside-IQR (pooled person_stats per resample, `nboot_err`)."""
    have = [c for c in ids if c in segs_b and c in segs_s]
    if len(have) < 4:
        return None
    rng = np.random.default_rng(seed)
    pcb, pcs = base["react"]["_per_clip"], sysr["react"]["_per_clip"]
    res = {"n_clips": len(have)}
    av_b = np.array([_avert_counts(segs_b[c]) for c in have]); av_s = np.array([_avert_counts(segs_s[c]) for c in have])

    def cheap(idx):
        pick = [have[i] for i in idx]
        d = {"gaze_away_listen": (av_s[idx, 0].sum() / max(1, av_s[idx, 1].sum())) - (av_b[idx, 0].sum() / max(1, av_b[idx, 1].sum()))}
        for k in PAIRS:
            d["excess " + k] = (_excess([pcs[k][c] for c in pick if c in pcs[k]])
                                - _excess([pcb[k][c] for c in pick if c in pcb[k]]))
        return d

    def costly(idx):
        pick = [have[i] for i in idx]
        sb = stat_block([segs_b[c] for c in pick], ref); sss = stat_block([segs_s[c] for c in pick], ref)
        return {"abs_log_err": sss["abs_log_err"] - sb["abs_log_err"], "inside_iqr": sss["inside_iqr"] - sb["inside_iqr"]}
    full = np.arange(len(have))
    point = cheap(full)
    boots = [cheap(rng.integers(0, len(have), len(have))) for _ in range(nboot)]
    if with_err:
        point.update(costly(full))
        eb = [costly(rng.integers(0, len(have), len(have))) for _ in range(nboot_err)]
        for b_, e_ in zip(boots, eb):
            b_.update(e_)
    for k, v in point.items():
        b = np.array([x[k] for x in boots if k in x], float)
        b = b[np.isfinite(b)]
        res[k] = {"delta": v, "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) > 10
                  else [float("nan")] * 2}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--base", default="rules")
    ap.add_argument("--space", default="rig")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--out")
    a = ap.parse_args()
    man = AN.manifest()
    ref = AN.get_ref()["ref"]
    runs = a.runs.split(",")
    res = {"space": a.space, "groups": {}, "pairs": PAIRS,
           "reference": {"n_people": AN.get_ref()["n_people"],
                         "listening": {k: ref["listening"][k] for k in ("avert_frac", "nod_per_min", "head_yaw_speed_med",
                                                                        "head_pitch_speed_med") if k in ref["listening"]},
                         "all": {k: ref["all"][k] for k in ("blink_per_min", "smile_per_min") if k in ref["all"]}}}
    for grp in GROUPS:
        ids = AN.clip_ids(man, None if grp == "all" else grp)
        G = {}
        segs_all = {}
        for run in runs:
            sp = a.space if not run.startswith("human") or a.space != "canon" else "canon"
            segs = segs_for(run, man, ids, sp)
            if len(segs) < 3:
                continue
            segs_all[run] = segs
            sb = stat_block(list(segs.values()), ref)
            sb["n_clips"] = len(segs)
            sb["react"] = reactions(run, man, [c for c in ids if c in segs], segs, sp)
            G[run] = sb
        if a.base in G:
            for run in G:
                if run == a.base:
                    continue
                pv = paired_vs(G[a.base], G[run], segs_all[a.base], segs_all[run], ref, ids, a.nboot,
                               nboot_err=100 if (run.startswith("unified") and grp == "all") else 0,
                               with_err=run.startswith("unified") and grp == "all")
                if pv:
                    G[run]["vs_" + a.base] = pv
        for run in G:
            G[run]["react"] = {k: v for k, v in G[run]["react"].items() if k != "_per_clip"}
        res["groups"][grp] = G
        for run, v in G.items():
            print(f"{grp:8s} {run:28s} n={v['n_clips']:2d} |log|={v['abs_log_err']:.3f} in={v['inside_iqr']:.2f} "
                  f"spd L/S={v['head_speed_listen']:.1f}/{v['head_speed_speak']:.1f} away={v['gaze_away_listen']:.2f} "
                  f"nod={v['nods_per_min_listen']:.1f} blink={v['blinks_per_min']:.1f} jerk={v['head_jerk_med']:.0f} "
                  + " ".join(f"{k}:{v['react'][k]['excess']:+.2f}" for k in PAIRS), flush=True)
    out = Path(a.out) if a.out else E / "measures" / f"headline_{a.space}.json"
    atomic_json(out, res)
    print(out)


if __name__ == "__main__":
    main()
