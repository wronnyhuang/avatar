"""Collect for the phase-2 ablations: every finished run of a multi.py job -> exported, tested, evaluated bundle
+ one ablation table.

    python -I tools/train/p2_collect.py --job <run dir name under $MM_WORK_DIR/runs, e.g. p2abl> [--names base,mel,...] [--eval-pack <v3+enc npz>]
        [--runtime '<json runtime overrides>'] [--no-score] [--judge] [--workers 12] [--table-only]

For each <name> with a DONE in runs/<job>/<name>/ (multi.py's run dir; a runs/<job>/latest/ snapshot layout is
also accepted):
  1. copy the bundle to runs/p2abl/<name>/bundle (never edits the snapshot); enc bundles get the encoder META
     (+ its PCA file, copied into the bundle) from $P/feats/<enc>/META.json unless already present;
     --runtime merges into config["runtime"] (e.g. the seam fix).
  2. unified/export.py (ORT graphs; max |ort - torch|), unified/test_runtime.py on it.
  3. tools/train/evaluate.py --pack (closed loop on the val streams; drops user_audio,user_face,self_speech).
  4. the scorecard hook evalkit/post_train.sh <bundle> p2abl_<name> [--no-judge] (SCORECARD_CODE = this tree).
Each step writes its output next to the bundle and is skipped when that output exists (re-run = resume).
Then prints the markdown table (train-side val numbers + closed-loop scorecard + scorecard hook summary).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T = os.environ.get("MM_WORK_DIR", os.path.join(ROOT, "work"))
P = T + "/p2"
PY = os.environ.get("PYTHON", sys.executable)
NOISE = os.environ.get("NOISE_PROFILE", os.path.join(os.environ.get("MM_DATA_DIR", os.path.join(ROOT, "data")),
                                                     "fit", "noise_profile.json"))
HOOK = os.path.join(ROOT, "evalkit", "post_train.sh")     # scorecard (+ PCM feeding for speech-encoder bundles)


def job_base(job):
    """runs/<job>/latest (snapshot layout) when it exists, else runs/<job> (multi.py's own run dir)."""
    lat = os.path.join(T, "runs", job, "latest")
    return lat if os.path.isdir(lat) else os.path.join(T, "runs", job)


def sh(cmd, log, env=None, timeout=7200):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env, timeout=timeout)
    return r.returncode


def prep_bundle(src, dst, runtime=None):
    if not os.path.exists(os.path.join(dst, "weights.pt")):
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
    cp = os.path.join(dst, "config.json")
    cfg = json.load(open(cp))
    au = cfg.get("audio") or {}
    if au.get("in", "mel") != "mel" and not au.get("encoder"):
        enc = au["in"][4:]
        mp = os.path.join(P, "feats", enc, "META.json")
        if os.path.exists(mp):
            au["encoder"] = json.load(open(mp))
    enc_meta = au.get("encoder") or {}
    for key in ("pca", "pca_path", "pca_file"):
        pp = enc_meta.get(key)
        if isinstance(pp, dict):
            for fk in ("file", "path"):
                f = pp.get(fk)
                if isinstance(f, str) and os.path.isabs(f) and os.path.exists(f):
                    shutil.copy(f, os.path.join(dst, "enc_pca.npz"))
                    pp[fk] = "enc_pca.npz"
        elif isinstance(pp, str) and os.path.isabs(pp) and os.path.exists(pp):
            shutil.copy(pp, os.path.join(dst, "enc_pca.npz"))
            enc_meta[key] = "enc_pca.npz"
    if au:
        cfg["audio"] = au
    if runtime:
        cfg["runtime"] = dict(cfg.get("runtime") or {}, **runtime)
    with open(cp, "w") as f:
        json.dump(cfg, f, indent=1)
    return cfg


def val_at(metrics_path, key="val_core"):
    rows = [json.loads(x) for x in open(metrics_path) if x.strip()]
    return {r["step"]: r[key] for r in rows if key in r and not r.get("final")}


VARIANT = ""


def table(names, job):
    out = []
    hdr = ["run", "steps", "val_core best", "val_core @common", "val_mae head/face_u/mouth", "val_resp",
           "event AP nod/blink/smile", "fire/label per min nod/blink/smile", "drop d_val_core uA/uF/self",
           "CL nods/blinks/smiles per min (real)", "CL head speed (real)", "CL smile coupl. gain", "sens zero/shuffle",
           "CL mean|log ratio|", "SCORE hook"]
    vals = {}
    for n in names:
        rd = os.path.join(job_base(job), n)
        mp = os.path.join(rd, "metrics.jsonl")
        if os.path.exists(mp):
            vals[n] = val_at(mp)
    common = None
    for v in vals.values():
        common = set(v) if common is None else common & set(v)
    cstep = max(common) if common else None
    out.append("| " + " | ".join(hdr) + " |")
    out.append("|" + "---|" * len(hdr))
    for n in names:
        rd = os.path.join(job_base(job), n)
        dp = os.path.join(rd, "DONE")
        if not os.path.exists(dp):
            out.append(f"| {n} | (no DONE) |" + " |" * (len(hdr) - 2))
            continue
        d = json.load(open(dp))
        fv = d.get("final_val") or {}
        mae = fv.get("val_mae") or {}
        ap = fv.get("val_event_ap") or {}
        cal = fv.get("events_cal") or {}
        drop = fv.get("val_drop") or {}
        dd = "/".join(f"{(drop.get(k) or {}).get('d_val_core', float('nan')):+.4f}" for k in ("user_audio", "user_face", "self_speech"))
        ev = {}
        ep = os.path.join(T, "runs", "p2abl", n, f"eval_val{VARIANT}.json")
        if os.path.exists(ep):
            ev = json.load(open(ep)).get("scorecard") or {}
        sc = ""
        sj = os.path.join(P, "scorecards", f"p2abl_{n}{VARIANT}", "scorecard.md")
        if os.path.exists(sj):
            lines = [x for x in open(sj) if "beats rules" in x or "REALISM" in x]
            sc = " ; ".join(x.strip().strip("|").replace("|", "/")[:80] for x in lines[:2])

        def g(k, nd=2):
            v = ev.get(k)
            return "-" if v is None else f"{v:.{nd}f}"
        fl = cal.get("fire_rate_per_min") or []
        lb = cal.get("label_rate_per_min") or []
        row = [n, str(d.get("step")), f"{fv.get('best_val_loss_cmp', float('nan')):.4f}",
               f"{vals.get(n, {}).get(cstep, float('nan')):.4f}" if cstep else "-",
               "/".join(f"{mae.get(k, float('nan')):.4f}" for k in ("head", "face_upper", "mouth")),
               str(fv.get("val_resp")),
               "/".join(str(ap.get(k, "-")) for k in ("nod", "blink", "smile")) if ap else "-",
               " ".join(f"{a:.0f}/{b:.0f}" for a, b in zip(fl, lb)) if fl else "-",
               dd if drop else "-",
               (f"{g('nods_per_min_gen', 1)}/{g('blinks_per_min_gen', 1)}/{g('smiles_per_min_gen', 1)} "
                f"({g('nods_per_min_real', 1)}/{g('blinks_per_min_real', 1)}/{g('smiles_per_min_real', 1)})") if ev else "-",
               f"{g('head_speed_gen', 3)} ({g('head_speed_real', 3)})" if ev else "-",
               g("smile_coupling_gain_zero", 3), f"{g('sens_zero')}/{g('sens_shuffle')}" if ev else "-",
               g("mean_abs_log_ratio_avg"), sc or "-"]
        out.append("| " + " | ".join(row) + " |")
    if cstep:
        out.append(f"\n(val_core @common = at step {cstep}, the largest step every run evaluated)")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--names", default=None)
    ap.add_argument("--eval-pack", default=None)
    ap.add_argument("--runtime", default=None)
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--max-sessions", type=int, default=12)
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--variant", default="", help="tag suffix for an extra eval/scorecard with planner kwargs --kw")
    ap.add_argument("--kw", default=None, help="json planner kwargs for --variant (e.g. {\"temperature\": 0.3})")
    a = ap.parse_args()
    base = job_base(a.job)
    names = a.names.split(",") if a.names else sorted(os.path.basename(os.path.dirname(p))
                                                       for p in glob.glob(os.path.join(base, "*", "DONE")))
    rt = json.loads(a.runtime) if a.runtime else None
    global VARIANT
    VARIANT = a.variant
    if not a.table_only:
        for n in names:
            src = os.path.join(base, n, "bundle")
            if not os.path.exists(os.path.join(base, n, "DONE")) or not os.path.exists(src):
                print(f"{n}: not done yet", flush=True)
                continue
            od = os.path.join(T, "runs", "p2abl", n)
            os.makedirs(od, exist_ok=True)
            log = os.path.join(od, "collect.log")
            dst = os.path.join(od, "bundle")
            t0 = time.time()
            prep_bundle(src, dst, rt)
            if not os.path.exists(os.path.join(dst, "head_step.onnx")):
                sh([PY, "-I", "unified/export.py", dst], log)
            if not os.path.exists(os.path.join(od, "test_runtime.ok")):
                env = dict(os.environ, UNIFIED_BUNDLE=dst)
                if sh([PY, "-I", "-m", "pytest", "-q", "unified/test_runtime.py"], log, env=env) == 0:
                    open(os.path.join(od, "test_runtime.ok"), "w").write("ok")
            ep = os.path.join(od, f"eval_val{a.variant}.json")
            kw = json.loads(a.kw) if a.kw else {}
            penv = dict(os.environ, OMP_WAIT_POLICY="PASSIVE")
            if a.eval_pack and not os.path.exists(ep):
                sh([PY, "-I", "tools/train/evaluate.py", "--bundle", dst, "--pack", a.eval_pack, "--split", "val",
                    "--seconds", str(a.seconds), "--max-sessions", str(a.max_sessions), "--workers", str(a.workers),
                    "--drop", "user_audio,user_face,self_speech", "--noise-profile", NOISE, "--out", ep]
                   + (["--rt", json.dumps(kw)] if kw else []), log, env=penv)
            tag = f"p2abl_{n}{a.variant}"
            if not a.no_score and os.path.exists(HOOK) and not os.path.exists(os.path.join(P, "scorecards", tag, "scorecard.md")):
                env = dict(penv, SCORECARD_CODE=ROOT)
                sh(["bash", HOOK, dst, tag] + ([] if a.judge else ["--no-judge"]) + (["--kw", json.dumps(kw)] if kw else []),
                   log, env=env)
            print(f"{n}: collected in {time.time() - t0:.0f}s ({log})", flush=True)
    print(table(names, a.job))


if __name__ == "__main__":
    main()
