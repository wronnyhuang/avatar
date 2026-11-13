#!/usr/bin/env python3
"""Aggregate the objective measures over the clip set -> $T/eval/measures/<what>.json

  python -I evalkit/analyze.py ref                        # held-out Seamless TEST people -> reference stats (cached)
  python -I evalkit/analyze.py stats   --systems rules,gru-prior,copy-delay,unified,human [--space auto|rig] [--tag ..]
  python -I evalkit/analyze.py respond --systems ...
  python -I evalkit/analyze.py ablate  --systems rules,gru-prior,copy-delay,unified
See measures.py for the definitions. CIs: clip bootstrap (2000x; stats: 300x, paired with the same resamples).
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
import mstats  # noqa: E402

OUT = E / "measures"


def manifest():
    return {c["id"]: c for c in json.loads((E / "clips/manifest.json").read_text())["clips"]}


def clip_ids(man, group=None):
    return sorted(c for c, v in man.items() if (E / "clips" / c / "receipt.json").exists()
                  and (group is None or v["group"] == group))


# ----------------------------------------------------------------------------- reference
def get_ref(force=False):
    p = OUT / "ref_seamless_test.json"
    if p.exists() and not force:
        return json.loads(p.read_text())
    ref, people = M.reference_people()
    if ref is None:
        raise SystemExit("no DATA canon index yet")
    out = {"ref": ref, "n_people": len(people), "people": sorted(people),
           "note": "Seamless TEST-split people, all landed sessions, frames with valid head+face, canon -> human-unit table "
                   "(measures.table_from_canon); states from each person's own VAD (bridged 1.2 s) / partner VAD."}
    atomic_json(p, out)
    return out


# ----------------------------------------------------------------------------- per-clip segments
_UT = {}


def user_tab(cid):
    if cid not in _UT:
        _UT[cid] = M.user_table(E / "clips" / cid)[0]
    return _UT[cid]


def seg_for(run, man, cid, space="auto"):
    Tb = M.system_table(run, cid, space)
    if Tb is None:
        return None
    N = int(round(float(man[cid]["duration_s"]) * FPS))
    Tb = M.fit_len(Tb, N)
    st, usp, asp = M.clip_states(man[cid], N)
    return M.make_seg(Tb, st, usp, asp, pM=M.fit_len(user_tab(cid), N))


def stats_for(segs):
    for s in segs:
        s.pop("_ev", None)
    return mstats.person_stats(segs)


def cmd_stats(systems, tag, space, nboot=300):
    man = manifest()
    ref = get_ref()["ref"]
    res = {"space": space, "reference": "held-out Seamless TEST people (n=%d)" % get_ref()["n_people"], "systems": {}}
    groups = {"all": clip_ids(man), "seamless": clip_ids(man, "seamless"), "youtube": clip_ids(man, "youtube")}
    segs_by = {}
    for s in systems:
        run = s + (("-" + tag) if tag and s == "unified" else "")
        segs_by[s] = {cid: seg_for(run, man, cid, space) for cid in groups["all"]}
    rng = np.random.default_rng(11)
    for grp, ids in groups.items():
        # common clips across all requested systems (paired); human only exists for seamless -> computed separately
        for s in systems:
            have = [c for c in ids if segs_by[s].get(c) is not None]
            if len(have) < 3:
                continue
            st = stats_for([segs_by[s][c] for c in have])
            err, inside, cells = M.compare_ref(st, ref)
            boots = []
            for _ in range(nboot):
                pick = [have[i] for i in rng.integers(0, len(have), len(have))]
                e, ins, _ = M.compare_ref(stats_for([dict(segs_by[s][c]) for c in pick]), ref)
                boots.append((e, ins))
            b = np.array(boots, float)
            res["systems"].setdefault(s, {})[grp] = {
                "n_clips": len(have), "mean_abs_log_err": err, "inside_iqr": inside,
                "mean_abs_log_err_ci95": [float(np.nanpercentile(b[:, 0], 2.5)), float(np.nanpercentile(b[:, 0], 97.5))],
                "inside_iqr_ci95": [float(np.nanpercentile(b[:, 1], 2.5)), float(np.nanpercentile(b[:, 1], 97.5))],
                "cells": cells, "minutes": {k: (st.get(k) or {}).get("minutes") for k in M.STATE_NAMES},
                "couplings": {k: v for k, v in (st.get("coupling") or {}).items() if not isinstance(v, list)}}
    # paired differences vs rules (same clip resamples)
    if "rules" in systems:
        for grp, ids in groups.items():
            for s in systems:
                if s == "rules":
                    continue
                have = [c for c in ids if segs_by[s].get(c) is not None and segs_by["rules"].get(c) is not None]
                if len(have) < 3:
                    continue
                d = []
                for _ in range(nboot):
                    pick = [have[i] for i in rng.integers(0, len(have), len(have))]
                    e1 = M.compare_ref(stats_for([dict(segs_by[s][c]) for c in pick]), ref)
                    e0 = M.compare_ref(stats_for([dict(segs_by["rules"][c]) for c in pick]), ref)
                    d.append((e1[0] - e0[0], e1[1] - e0[1]))
                d = np.array(d, float)
                e1 = M.compare_ref(stats_for([segs_by[s][c] for c in have]), ref)
                e0 = M.compare_ref(stats_for([segs_by["rules"][c] for c in have]), ref)
                res["systems"][s].setdefault("vs_rules", {})[grp] = {
                    "n_clips": len(have), "d_mean_abs_log_err": e1[0] - e0[0], "d_inside_iqr": e1[1] - e0[1],
                    "d_err_ci95": [float(np.nanpercentile(d[:, 0], 2.5)), float(np.nanpercentile(d[:, 0], 97.5))],
                    "d_inside_ci95": [float(np.nanpercentile(d[:, 1], 2.5)), float(np.nanpercentile(d[:, 1], 97.5))]}
    name = f"stats_{space}" + (f"_{tag}" if tag else "")
    atomic_json(OUT / f"{name}.json", res)
    for s, v in res["systems"].items():
        for grp, g in v.items():
            if grp == "vs_rules":
                continue
            print(f"{s:11s} {grp:9s} n={g['n_clips']:2d} |log err| {g['mean_abs_log_err']:.3f} {np.round(g['mean_abs_log_err_ci95'], 3)} "
                  f"inside IQR {g['inside_iqr']:.2f} {np.round(g['inside_iqr_ci95'], 2)}")
    return res


# ----------------------------------------------------------------------------- responsiveness
def cmd_respond(systems, tag):
    man = manifest()
    res = {"window_s": [M.WIN[0] / FPS, M.WIN[1] / FPS], "pairs": [f"{a}->{b}" for a, b in M.REACTIONS], "systems": {}}
    for grp in ("all", "seamless", "youtube"):
        ids = clip_ids(man, None if grp == "all" else grp)
        for s in systems:
            run = s + (("-" + tag) if tag and s == "unified" else "")
            per = {}
            xc_real, xc_shuf = [], []
            for i, cid in enumerate(ids):
                Tb = M.system_table(run, cid)
                if Tb is None:
                    continue
                N = int(round(float(man[cid]["duration_s"]) * FPS))
                Tb = M.fit_len(Tb, N)
                st, usp, asp = M.clip_states(man[cid], N)
                Ut = M.fit_len(user_tab(cid), N)
                rc = M.reaction_counts(M.user_events(Ut, usp, N), M.avatar_events(Tb, st), N)
                for k, v in rc.items():
                    per.setdefault(k, []).append(v)
                xc_real.append(M.smile_xcorr(Ut[:, 12], Tb[:, 12]))
                other = ids[(i + 1) % len(ids)]
                Uo = M.fit_len(user_tab(other), N)
                xc_shuf.append(M.smile_xcorr(Uo[:, 12], Tb[:, 12]))
            if not per:
                continue
            out = {"n_clips": len(xc_real)}
            for k, rows in per.items():
                n = sum(r["n"] for r in rows); h = sum(r["hits"] for r in rows)
                ch = sum(r["chance_hits"] for r in rows) / max(1, sum(r["chance_n"] for r in rows))
                lat = [x for r in rows for x in r["lat"]]
                rate = h / n if n else float("nan")
                out[k] = {"n_events": n, "hits": h, "rate": rate, "rate_ci95": M.boot_ratio([(r["hits"], r["n"]) for r in rows]),
                          "chance": ch, "lift": rate / ch if ch > 0 and n else float("nan"),
                          "excess": rate - ch if n else float("nan"),
                          "latency_med_s": float(np.median(lat)) if lat else float("nan"), "n_lat": len(lat)}
            xr, xs = np.array(xc_real, float), np.array(xc_shuf, float)
            ok = np.isfinite(xr) & np.isfinite(xs)
            out["smile_xcorr_real"] = float(np.nanmean(xr)) if np.isfinite(xr).any() else float("nan")
            out["smile_xcorr_shuffled"] = float(np.nanmean(xs)) if np.isfinite(xs).any() else float("nan")
            out["smile_xcorr_gain"] = float(np.mean(xr[ok] - xs[ok])) if ok.any() else float("nan")
            out["smile_xcorr_gain_ci95"] = M.boot_ci(xr[ok] - xs[ok]) if ok.sum() > 1 else [float("nan")] * 2
            res["systems"].setdefault(s, {})[grp] = out
    name = "respond" + (f"_{tag}" if tag else "")
    atomic_json(OUT / f"{name}.json", res)
    for s, v in res["systems"].items():
        for grp, g in v.items():
            print(s, grp, "n", g["n_clips"], " ".join(f"{k}: {g[k]['rate']:.2f}/{g[k]['chance']:.2f} (lift {g[k]['lift']:.2f}, n={g[k]['n_events']})"
                                                     for k in res["pairs"] if k in g),
                  f"xcorr {g['smile_xcorr_real']:.3f} vs shuffled {g['smile_xcorr_shuffled']:.3f}")
    return res


# ----------------------------------------------------------------------------- ablation sensitivity
def cmd_ablate(systems, tag, kinds=("zero", "shuffle", "shift")):
    man = manifest()
    res = {"kinds": list(kinds), "groups": list(M.GROUPS_T), "systems": {}}
    for s in systems:
        base_run = s + (("-" + tag) if tag and s == "unified" else "")
        for grp in ("all", "seamless", "youtube"):
            ids = clip_ids(man, None if grp == "all" else grp)
            for kind in kinds:
                rows = []
                for cid in ids:
                    A = M.system_table(base_run, cid); B = M.system_table(base_run + "~" + kind, cid)
                    if A is None or B is None:
                        continue
                    n = min(len(A), len(B))
                    A, B = np.nan_to_num(A[:n]), np.nan_to_num(B[:n])
                    r = {}
                    for g, cols in M.GROUPS_T.items():
                        sd = A[:, cols].std(0)
                        diff = np.sqrt(((A[:, cols] - B[:, cols]) ** 2).mean(0))
                        okc = sd > 1e-6
                        r[g] = float(np.mean(diff[okc] / sd[okc])) if okc.any() else float("nan")
                    rows.append(r)
                if not rows:
                    continue
                o = {"n_clips": len(rows)}
                for g in M.GROUPS_T:
                    v = np.array([r[g] for r in rows], float)
                    o[g] = {"mean": float(np.nanmean(v)), "ci95": M.boot_ci(v)}
                allg = np.array([np.nanmean([r[g] for g in ("head", "gaze", "face_upper", "smile")]) for r in rows])
                o["overall"] = {"mean": float(np.nanmean(allg)), "ci95": M.boot_ci(allg)}
                res["systems"].setdefault(s, {}).setdefault(grp, {})[kind] = o
    name = "ablate" + (f"_{tag}" if tag else "")
    atomic_json(OUT / f"{name}.json", res)
    for s, v in res["systems"].items():
        for grp, g in v.items():
            print(s, grp, " ".join(f"{k}: {g[k]['overall']['mean']:.2f}" for k in kinds if k in g))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["ref", "stats", "respond", "ablate"])
    ap.add_argument("--systems", default="rules,gru-prior,copy-delay,unified,human")
    ap.add_argument("--tag", default=""); ap.add_argument("--space", default="auto")
    ap.add_argument("--nboot", type=int, default=300); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    S = a.systems.split(",")
    if a.cmd == "ref":
        r = get_ref(a.force)
        print(r["n_people"], "people")
        for st in M.STATE_NAMES:
            print(st, {k: round(v["med"], 3) for k, v in (r["ref"].get(st) or {}).items() if k in M.KEYS})
    elif a.cmd == "stats":
        cmd_stats(S, a.tag, a.space, a.nboot)
    elif a.cmd == "respond":
        cmd_respond(S, a.tag)
    else:
        cmd_ablate(S, a.tag)


if __name__ == "__main__":
    main()
