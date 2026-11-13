#!/usr/bin/env python3
"""EVAL — the whole evaluation in one command.

  python -I evalkit/run_all.py --systems rules,prior,copy_delay,unified:<bundle> --out <dir>
         [--style "energy=0.1,expressiveness=0.1;energy=0.9,expressiveness=0.9"] [--stages all|a,b,..]
         [--judge-ids N|id,id] [--sbs-ids N|id,id] [--quick] [--no-paid]

<bundle> = BEST ($T/runs/BEST), a bundle dir, a run dir/name under $T/runs, or stub. Several
unified systems may be given; the FIRST is the primary candidate. `--style` adds the primary bundle at each style
setting (";"-separated) for the style-direction check. `human` (the real partner, Seamless clips) is always included.
Stages (default all, in order; each one caches, so a rerun only does new work):
  systems   per-clip rig traces for every system on the 50-clip set ($E/runs/<run>/<clip>/rig.jsonl [+ canon.npz])
  ablate    the same with the USER stream zeroed / shuffled (responsiveness)
  measures  headline table (human-range stats vs the held-out humans, reactions + lags, jerk) + ablation sensitivity
  style     style settings: direction of head speed / expressiveness / mirroring vs the neutral run
  drift     5-minute closed-loop rollout of every system on one concatenated conversation
  latency   per re-plan ms of the planner on this host (3 passes, 2 threads) + e2e through app/unified_worker.py
  render    Muse renders (640x360) of the primary unified + human on the judge/sbs clips (baselines cached)
  sbs       side-by-side videos (user | rules | unified | human) -> <out>/sbs/
  judge     blind pairwise judge (rules vs unified; once: rules vs human as the judge's sanity check), capped by EVAL_SPEND_CAP_USD
  report    <out>/REPORT.md + <out>/index.html (+ results.json)
Outputs: <out>/ (report, measures, judge summaries, sbs); traces in $E/runs (shared cache across invocations).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import E, T, KIT, VENV_PY, atomic_json, eval_spend  # noqa: E402
import bundles  # noqa: E402

PY = str(VENV_PY)
ALIAS = {"prior": "gru-prior", "gru_prior": "gru-prior", "copy_delay": "copy-delay", "copy": "copy-delay"}
STAGES = ["systems", "ablate", "measures", "style", "drift", "latency", "render", "sbs", "judge", "report"]
# judge / sbs clip choice (fixed so every bundle is judged on the same clips): Seamless = distinct user people first
SBS_DEFAULT = ["sl_S0152_0371_P0216", "sl_S0274_0130_P0370", "sl_S0515_0503_P0371", "sl_S0644_0578_P0299A",
               "sl_S0948_0372_P1040", "sl_S2025_1079_P1281A", "sl_S2090_1010_P1346A", "sl_S2091_1095_P1282A",
               "wave_vcd_bald_man", "silence_waiting_woman_turns_smiles"]
# the held-out suite clips the judge sees: one per behaviour category (presence, approach, boredom, emotions, listening,
# movement, questions, smile, startle, talk, wave)
YT_JUDGE = ["appear_bedroom_woman_goes_to_bed", "approach_leave_vcd_lean_in_man", "boredom_yawn_office_man",
            "emotion_surprise", "emotion_sad", "listen_musuweu_monologue", "move_room_walk_phone_woman",
            "question_gereon_editathons", "silence_waiting_woman_turns_smiles", "startle_vcd_hands_at_lens_woman",
            "talk_john_intro", "wave_vcd_bald_man"]


def log(msg, out):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(out / "run_all.log", "a") as f:
        f.write(line + "\n")


def sh(cmd, out, env=None, check=True):
    log("+ " + " ".join(map(str, cmd))[:400], out)
    r = subprocess.run(list(map(str, cmd)), env={**os.environ, **(env or {}), "PYTHONDONTWRITEBYTECODE": "1"},
                       capture_output=True, text=True)
    with open(out / "run_all.log", "a") as f:
        f.write(r.stdout[-6000:] + ("\nSTDERR:\n" + r.stderr[-3000:] if r.returncode else "") + "\n")
    if r.returncode and check:
        raise RuntimeError(f"failed ({r.returncode}): {' '.join(map(str, cmd))[:300]}\n{r.stderr[-1500:]}")
    return r


def manifest():
    return {c["id"]: c for c in json.loads((E / "clips/manifest.json").read_text())["clips"]}


def judge_ids(spec, man):
    if spec and not spec.isdigit():
        return spec.split(",")
    n = int(spec or 24)
    sl = sorted(c for c, v in man.items() if v["group"] == "seamless")
    yt = [c for c in YT_JUDGE if c in man] + sorted(c for c, v in man.items() if v["group"] == "youtube" and c not in YT_JUDGE)
    # Seamless: one clip per user person first, then the rest; YouTube: the whole suite in a fixed order
    by_p, seen = [], set()
    for c in sl:
        p = man[c].get("user_person")
        if p not in seen:
            seen.add(p); by_p.append(c)
    sl_order = by_p + [c for c in sl if c not in by_p]
    half = n // 2
    return sl_order[:half] + yt[:n - half]


def parse_systems(s):
    base, uni = [], []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        if x.startswith("unified"):
            uni.append(bundles.resolve(x))
        else:
            base.append(ALIAS.get(x, x))
    return base, uni


# ----------------------------------------------------------------------------- stages
def st_systems(base, uni, out, ablate=""):
    flag = ["--ablate", ablate] if ablate else []
    b = [s for s in base if s != "human" or not ablate]
    if b:
        sh([PY, "-I", KIT / "systems.py", "--systems", ",".join(b), "--jobs", "12"] + flag, out)
    for u in uni:
        kw = json.dumps({"style": u["style"]}) if u["style"] else "{}"
        sh([PY, "-I", KIT / "systems.py", "--systems", "unified", "--tag", u["tag"], "--bundle", u["bundle"] or "stub",
            "--kw", kw, "--jobs", "12"] + flag, out, env={"EVAL_MODEL_CODE": u["code"]})
    # verify every system produced every clip
    man = manifest()
    missing = {}
    for run in b + [u["run"] for u in uni]:
        name = run + (("~" + ablate) if ablate else "")
        need = [c for c, v in man.items() if not (run == "human" and v["group"] != "seamless")]
        fn = "canon.npz" if (ablate and run not in ("rules", "gru-prior")) else "rig.jsonl"
        miss = [c for c in need if not (E / "runs" / name / c / fn).exists()]
        if miss:
            missing[name] = miss
    if missing:
        log(f"MISSING traces: { {k: len(v) for k, v in missing.items()} }", out)
    return missing


def ablation_summary(runs, out):
    import measures as M
    import analyze as AN
    man = manifest()
    res = {}
    for run in runs:
        for kind in ("zero", "shuffle"):
            rows = []
            for cid in AN.clip_ids(man):
                A = M.system_table(run, cid); B = M.system_table(run + "~" + kind, cid)
                if A is None or B is None:
                    continue
                n = min(len(A), len(B)); A, B = np.nan_to_num(A[:n]), np.nan_to_num(B[:n])
                r = {}
                for g, cols in M.GROUPS_T.items():
                    sd = A[:, cols].std(0); diff = np.sqrt(((A[:, cols] - B[:, cols]) ** 2).mean(0)); ok = sd > 1e-6
                    r[g] = float(np.mean(diff[ok] / sd[ok])) if ok.any() else float("nan")
                rows.append(r)
            if not rows:
                continue
            allg = np.array([np.nanmean([r[g] for g in ("head", "gaze", "face_upper", "smile")]) for r in rows])
            res.setdefault(run, {})[kind] = {"n": len(rows), "overall": float(np.nanmean(allg)), "ci95": M.boot_ci(allg),
                                             **{g: float(np.nanmean([r[g] for r in rows])) for g in M.GROUPS_T}}
    atomic_json(out / "ablate.json", res)
    for run, v in res.items():
        log(f"ablate {run}: " + " ".join(f"{k}={v[k]['overall']:.2f}" for k in v), out)
    return res


def style_table(prim, styled, out):
    """direction check: each style run vs the neutral primary on the Seamless clips (rig space)."""
    import measures as M
    import analyze as AN
    import headline as HL
    man = manifest()
    ids = AN.clip_ids(man, "seamless")

    def summ(run):
        segs = HL.segs_for(run, man, ids, "rig")
        if len(segs) < 3:
            return None
        S = list(segs.values())
        motion_e = np.mean([np.nanstd(np.nan_to_num(s["M"][:, [0, 1, 2]]), 0).mean() for s in S])
        face_e = np.mean([np.nanstd(np.nan_to_num(s["M"][:, [10, 11, 12, 13, 14]]), 0).mean() for s in S])
        xg = HL.reactions(run, man, [c for c in ids if c in segs], segs, "rig")
        head_x = []
        for s in S:     # head-yaw/pitch following: max x-corr of avatar with user (0-1.5 s lag)
            u, a = s["pM"][:, :2], np.nan_to_num(s["M"][:, :2])
            ok = np.isfinite(u).all(1)
            best = np.nan
            if ok.sum() > 300:
                uu = np.where(ok[:, None], u, np.nanmean(u, 0))
                vals = []
                for L in range(0, 46, 3):
                    for j in range(2):
                        x = uu[:len(uu) - L, j] if L else uu[:, j]; y = a[L:, j]
                        if x.std() > 1e-3 and y.std() > 1e-3:
                            vals.append(np.corrcoef(x, y)[0, 1])
                best = max(vals) if vals else np.nan
            head_x.append(best)
        return {"head_speed_listen": HL.head_speed(S, 2), "head_speed_speak": HL.head_speed(S, 1),
                "head_motion_sd_deg": float(motion_e), "face_sd": float(face_e),
                "smile_xcorr_gain": xg["smile_xcorr_gain"], "smile_xcorr_real": xg["smile_xcorr_real"],
                "head_follow_xcorr": float(np.nanmean(head_x))}
    res = {"neutral": {"run": prim["run"], **(summ(prim["run"]) or {})}, "settings": []}
    for u in styled:
        s = summ(u["run"])
        res["settings"].append({"run": u["run"], "style": u["style"], **(s or {"error": "no traces"})})
    atomic_json(out / "style.json", res)
    for s in res["settings"]:
        log(f"style {s['style']}: " + " ".join(f"{k}={v:.3f}" for k, v in s.items() if isinstance(v, float)), out)
    return res


def st_latency(uni, out):
    res = {}
    for u in uni:
        r = sh([PY, "-I", KIT / "longrun.py", "latency", "--bundle", u["bundle"] or "stub", "--threads", "2", "--tag", u["tag"]],
               out, env={"EVAL_MODEL_CODE": u["code"]}, check=False)
        p = E / "measures" / f"latency_{u['tag']}.json"
        if p.exists():
            res[u["run"]] = json.loads(p.read_text())
    atomic_json(out / "latency.json", res)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default="rules,prior,copy_delay,unified:BEST")
    ap.add_argument("--out", required=True)
    ap.add_argument("--style", default="")
    ap.add_argument("--stages", default="all")
    ap.add_argument("--judge-ids", default="24")
    ap.add_argument("--sbs-ids", default=",".join(SBS_DEFAULT))
    ap.add_argument("--swap-ids", default="4", help="N clips also judged in the reversed order (position check)")
    ap.add_argument("--no-paid", action="store_true")
    ap.add_argument("--e2e-clips", default="")
    ap.add_argument("--publish", action="store_true", help="also copy REPORT.md + index.html to $E ($EVAL_OUT)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stages = STAGES if a.stages == "all" else a.stages.split(",")
    base, uni = parse_systems(a.systems)
    if "human" not in base:
        base.append("human")
    if not uni:
        raise SystemExit("no unified system given")
    prim = uni[0]
    styled = [bundles.resolve(f"unified:{prim['bundle'] or 'stub'}@{s}#{prim['base_tag']}") for s in a.style.split(";") if s.strip()]
    man = manifest()
    jids = judge_ids(a.judge_ids, man)
    sids = [c for c in (a.sbs_ids.split(",") if a.sbs_ids else []) if c in man]
    cfg = {"t": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "systems": base, "unified": uni, "styled": styled, "judge_ids": jids,
           "sbs_ids": sids, "stages": stages, "eval_root": str(E)}
    atomic_json(out / "config.json", cfg)
    log(f"run_all: base={base} unified={[u['run'] for u in uni]} styled={[u['run'] for u in styled]} out={out}", out)
    runs_all = base + [u["run"] for u in uni]
    t0 = time.time()
    if "systems" in stages:
        st_systems(base, uni + styled, out)
    if "ablate" in stages:
        for kind in ("zero", "shuffle"):
            st_systems([b for b in base if b != "human"], uni, out, ablate=kind)
    if "measures" in stages:
        # space auto = each system in its native space: rules/gru-prior rig -> human units (tools/motion_stats.py, the
        # rules-prior gains), canon systems (unified, copy-delay, human) by the data pipeline's FK = the held-out reference's own space
        # (rig space puts even the real human at |log err| 0.95 vs rules 0.64: the retarget's rig units are not the
        # reference's; validated on the stub)
        sh([PY, "-I", KIT / "headline.py", "--runs", ",".join(runs_all), "--base", "rules", "--space", "auto",
            "--out", out / "headline_rig.json"], out)
        ablation_summary([b for b in base if b != "human"] + [u["run"] for u in uni], out)
    if "style" in stages and styled:
        style_table(prim, styled, out)
    if "drift" in stages:
        for s in [b for b in base] + [f"unified:{u['bundle'] or 'stub'}#{u['base_tag']}" + ("" if not u["style"] else "")
                                       for u in uni if not u["style"]]:
            sh([PY, "-I", KIT / "drift.py", "run", "--system", s], out, check=False)
        sh([PY, "-I", KIT / "drift.py", "measure", "--runs", ",".join(runs_all), "--out", out / "drift.json"], out)
    if "latency" in stages:
        st_latency([prim], out)
        e2e = [PY, "-I", KIT / "e2e.py", "--unified", f"unified:{prim['bundle'] or 'stub'}#{prim['base_tag']}", "--rules",
               "--out", out / "e2e.json"]
        if a.e2e_clips:
            e2e += ["--clips", a.e2e_clips]
        sh(e2e, out, check=False)
    render_ids = sorted(set(jids) | set(sids))
    if "render" in stages:
        sh([PY, "-I", KIT / "render.py", "--runs", prim["run"], "--ids", ",".join(render_ids), "--jobs", "4", "--threads", "8"], out)
        hum = [c for c in render_ids if man[c]["group"] == "seamless"]
        if hum:
            sh([PY, "-I", KIT / "render.py", "--runs", "human", "--ids", ",".join(hum), "--jobs", "4", "--threads", "8"], out)
        sh([PY, "-I", KIT / "render.py", "--runs", "rules", "--ids", ",".join(render_ids), "--jobs", "4", "--threads", "8"], out)
    if "sbs" in stages and sids:
        sh([PY, "-I", KIT / "sbs.py", "--ids", ",".join(sids), "--runs", f"rules,{prim['run']},human", "--out-dir", out / "sbs",
            "--label-unified", "NEW unified model"], out)
    if "judge" in stages and not a.no_paid:
        env = {"JUDGE_ALLOW_PAID": "1"}
        nswap = int(a.swap_ids) if a.swap_ids.isdigit() else 0
        swap = ",".join(jids[:nswap // 2] + [c for c in jids if man[c]["group"] == "youtube"][:nswap - nswap // 2]) if nswap else ""
        sh([PY, "-I", KIT / "pairjudge.py", "judge", "--a", "rules", "--b", prim["run"], "--ids", ",".join(jids),
            "--swap-ids", swap, "--jobs", "4", "--out", out / "judge_rules_vs_unified.json"], out, env=env, check=False)
        cal = [c for c in jids if man[c]["group"] == "seamless"]
        sh([PY, "-I", KIT / "pairjudge.py", "judge", "--a", "rules", "--b", "human", "--ids", ",".join(cal), "--jobs", "4",
            "--out", out / "judge_rules_vs_human.json"], out, env=env, check=False)
        log(f"eval spend total ${eval_spend():.4f}", out)
    if "report" in stages:
        sh([PY, "-I", KIT / "report.py", "--out", out, "--cmd", " ".join(sys.argv)], out)
        if a.publish and out != E.resolve():
            import shutil
            for fn in ("REPORT.md", "index.html"):
                shutil.copy2(out / fn, E / fn)
            log(f"published REPORT.md + index.html to {E}", out)
    log(f"run_all done in {(time.time() - t0) / 60:.1f} min; eval spend ${eval_spend():.4f}", out)


if __name__ == "__main__":
    main()
