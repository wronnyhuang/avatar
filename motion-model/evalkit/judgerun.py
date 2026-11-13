#!/usr/bin/env python3
"""Judge the rendered runs (evalkit/judge/judge.py, gflash = google/gemini-3.7-flash via OpenRouter) and pair them.

  python -I evalkit/judgerun.py judge --runs rules,unified [--ids ..] [--jobs 6]       (paid: needs JUDGE_ALLOW_PAID=1)
  python -I evalkit/judgerun.py paired --base rules --runs unified,gru-prior,copy-delay,human [--out paired.json]

Paired: per axis, mean delta (system - base) over the clips both runs have, 95 % bootstrap CI (clips resampled,
5000x), win rate; per clip group (seamless / youtube / all) and a person-cluster bootstrap (clusters = the user person:
Seamless sessions share people) as a robustness check. Cost: every call is logged by judge.py to $E/spend.jsonl
(who=eval); judge.py refuses paid calls once the total passes JUDGE_SPEND_CAP.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import E, VENV_PY, KIT, atomic_json, eval_spend  # noqa: E402

AXES = ["weighted", "weighted_judge_only", "naturalness", "reaction_timing", "emotional_fit", "gaze", "motion_variety",
        "talk_amount", "lip_sync"]


def manifest():
    return {c["id"]: c for c in json.loads((E / "clips/manifest.json").read_text())["clips"]}


def judge(runs, ids=None, jobs=6, judge_id="gflash"):
    env = {**os.environ, "JUDGE_ALLOW_PAID": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    for run in runs:
        rd = E / "runs" / run
        cmd = [str(VENV_PY), "-I", str(KIT / "judge/judge.py"), "--run-dir", str(rd), "--judge", judge_id, "--jobs", str(jobs),
               "--manifest", str(E / "clips/manifest.json"), "--results-copy", str(E / "judge/results")]
        if ids:
            cmd += ["--clips", ",".join(ids)]
        print("+", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, env=env, text=True, capture_output=True)
        print(r.stdout[-3000:], r.stderr[-2000:], flush=True)
        print(f"eval judge spend so far ${eval_spend():.4f}", flush=True)


def load_scores(run, judge_id="gflash"):
    """per clip: weighted, weighted_judge_only, final axes, judge-only axes (from the per-clip judge.<id>.json files)."""
    out = {}
    rd = E / "runs" / run
    if not rd.exists():
        return out
    for cd in rd.iterdir():
        p = cd / f"judge.{judge_id}.json"
        if not p.exists():
            continue
        j = json.loads(p.read_text())
        row = {"weighted": j.get("weighted"), "weighted_judge_only": j.get("weighted_judge_only")}
        row.update(j.get("axes") or {})
        row.update({"judge_" + k: v for k, v in (j.get("axes_judge") or {}).items()})
        row["usd"] = j.get("usd")
        out[cd.name] = row
    return out


def boot(d, n=5000, seed=7, clusters=None):
    d = np.asarray(d, float)
    if len(d) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if clusters is None:
        m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    else:
        cl = np.asarray(clusters)
        keys = sorted(set(cl))
        idx = {k: np.where(cl == k)[0] for k in keys}
        m = []
        for _ in range(n):
            pick = rng.integers(0, len(keys), len(keys))
            ii = np.concatenate([idx[keys[p]] for p in pick])
            m.append(d[ii].mean())
        m = np.array(m)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(base, runs, judge_id="gflash"):
    man = manifest()
    B = load_scores(base, judge_id)
    res = {"base": base, "judge": judge_id, "systems": {}}
    for run in runs:
        S = load_scores(run, judge_id)
        common = sorted(set(B) & set(S) & set(man))
        sysres = {"n_common": len(common)}
        for group in ("all", "seamless", "youtube"):
            ids = [c for c in common if group == "all" or man[c]["group"] == group]
            g = {"n": len(ids)}
            for ax in AXES + ["judge_naturalness", "judge_gaze", "judge_reaction_timing", "judge_motion_variety",
                              "judge_emotional_fit"]:
                pr = [(B[c].get(ax), S[c].get(ax), man[c].get("user_person", c)) for c in ids
                      if isinstance(B[c].get(ax), (int, float)) and isinstance(S[c].get(ax), (int, float))]
                if not pr:
                    continue
                a = np.array([p[0] for p in pr]); b = np.array([p[1] for p in pr]); cl = [p[2] for p in pr]
                d = b - a
                lo, hi = boot(d)
                clo, chi = boot(d, clusters=cl)
                g[ax] = {"n": len(d), "base": round(float(a.mean()), 3), "sys": round(float(b.mean()), 3),
                         "delta": round(float(d.mean()), 3), "ci95": [round(lo, 3), round(hi, 3)],
                         "ci95_person_cluster": [round(clo, 3), round(chi, 3)], "n_clusters": len(set(cl)),
                         "win_rate": round(float(((d > 0) + 0.5 * (d == 0)).mean()), 3)}
            sysres[group] = g
        sysres["per_clip"] = {c: {"base": B[c].get("weighted"), "sys": S[c].get("weighted")} for c in common}
        res["systems"][run] = sysres
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["judge", "paired"])
    ap.add_argument("--runs", default="rules"); ap.add_argument("--base", default="rules")
    ap.add_argument("--ids"); ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--judge", default="gflash"); ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "judge":
        judge(a.runs.split(","), a.ids.split(",") if a.ids else None, a.jobs, a.judge)
    else:
        r = paired(a.base, a.runs.split(","), a.judge)
        out = Path(a.out) if a.out else E / "judge/paired.json"
        atomic_json(out, r)
        for s, v in r["systems"].items():
            for grp in ("all", "seamless", "youtube"):
                g = v.get(grp) or {}
                w = g.get("weighted")
                if w:
                    print(f"{s:12s} {grp:9s} n={w['n']:2d} weighted {w['base']:.2f} -> {w['sys']:.2f}  d={w['delta']:+.2f} "
                          f"CI{w['ci95']} cluster{w['ci95_person_cluster']} win {w['win_rate']:.2f}")
        print(out)


if __name__ == "__main__":
    main()
