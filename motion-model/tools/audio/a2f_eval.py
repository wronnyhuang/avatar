"""Audio2Face-3D as a SPEAKING-face baseline + its calibration to our canon scale (phase-2 audio, plan item 3).

  jobs     build the job list: CALIB = one speech-rich 40 s segment per TRAIN person of the MediaPipe dev videos
           ($P/face/cache, split train); EVAL = the 29 Seamless test clips of the eval harness (eval/clips sl_*,
           avatar voice, 36 s) whose avatar has MediaPipe video -> held-out people.
  run      A2F (identity --identity) over each job's own-voice audio with 1 s of real pre-roll -> raw ARKit weights
           in canon order at 30 fps: audio_work/a2f_eval/<job>.npz
  analyze  fit a per-dial monotone quantile map A2F -> MediaPipe on CALIB speaking frames (train people), apply it
           to EVAL, and compare on the avatar's speaking frames against MediaPipe (truth): A2F (raw, calibrated),
           the canon face the model trains on (mapper), the rules replay and the unified model (eval runs).
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import numpy as np  # noqa: E402

S = C.spec
P = C.P
WORK = os.path.join(P, "audio_work", "a2f_eval")
CACHE = os.path.join(P, "face", "cache")
EVAL = os.environ.get("EVAL_OUT", os.path.join(os.path.dirname(P), "eval"))
PRE = 30                       # 1 s of real pre-roll frames (GRU warm-up), dropped from the output
DIALS = ["jawOpen", "mouthClose", "mouthFunnel", "mouthPucker", "mouthLowerDownLeft", "mouthUpperUpLeft",
         "mouthStretchLeft", "mouthPressLeft", "mouthRollLower", "mouthShrugLower", "mouthSmileLeft",
         "mouthSmileRight", "mouthFrownLeft", "mouthDimpleLeft", "cheekSquintLeft", "browInnerUp", "browOuterUpLeft",
         "browDownLeft", "eyeBlinkLeft", "eyeSquintLeft", "eyeWideLeft", "noseSneerLeft"]


def wav_of(fid):
    g = glob.glob(os.path.join(C.SEAMLESS, "*", "*", "audio", f"{fid}.wav"))
    return g[0] if g else None


def make_jobs(a):
    os.makedirs(WORK, exist_ok=True)
    jobs = []
    per = {}
    for f in sorted(glob.glob(os.path.join(CACHE, "V*.npz"))):
        z = np.load(f, allow_pickle=False)
        if str(z["split"]) != "train":
            continue
        fid = os.path.basename(f)[:-4]
        vad = z["vad"].astype(np.float32) > 0.5
        ok = vad & z["mp_face"].astype(bool)
        n = a.calib_s * 30
        if len(ok) < n + PRE + 30:
            continue
        cs = np.convolve(ok.astype(np.float32), np.ones(n), "valid")
        cs[: PRE] = -1
        k0 = int(np.argmax(cs))
        sc = float(cs[k0] / n)
        p = str(z["person"])
        if p not in per or sc > per[p]["score"]:
            per[p] = dict(name="calib_" + fid, kind="calib", fid=fid, wav=wav_of(fid), k0=k0, n30=n, score=sc,
                          person=p)
    jobs += sorted(per.values(), key=lambda j: j["name"])
    man = json.load(open(os.path.join(EVAL, "clips", "manifest.json")))
    for c in man["clips"]:
        if not c["id"].startswith("sl_"):
            continue
        n = int(round(c["duration_s"] * 30))
        jobs.append(dict(name="eval_" + c["id"], kind="eval", fid=c["avatar_file"], wav=c["voice_wav"],
                         k0=int(c["frame0"]), n30=n, clip=c["id"], person=c["avatar_person"]))
    json.dump(jobs, open(os.path.join(WORK, "jobs.json"), "w"), indent=1)
    print(len([j for j in jobs if j["kind"] == "calib"]), "calib jobs,", len([j for j in jobs if j["kind"] == "eval"]),
          "eval jobs")


def run(a):
    import a2f as A
    jobs = json.load(open(os.path.join(WORK, "jobs.json")))
    k, n = [int(x) for x in a.shard.split("/")]
    jobs = [j for i, j in enumerate(jobs) if i % n == k]
    m = A.A2F(a.identity, threads=a.threads)
    for j in jobs:
        fn = os.path.join(WORK, f"{j['name']}_{a.identity}.npz")
        if os.path.exists(fn):
            continue
        t = time.time()
        k0 = j["k0"] - PRE
        x16 = C.load_segment(j["wav"], S.frame_end_sample(k0 - 1), S.frame_end_sample(k0 + j["n30"] + PRE - 1))
        o = m.run_track(x16, n30=j["n30"] + PRE)
        can = A.to_canon(o["w60"], o["pose_names"], j["n30"] + PRE)[PRE:]
        np.savez(fn, a2f=can.astype(np.float32), jaw=o["jaw60"][2 * PRE::2], eyes=o["eyes60"][2 * PRE::2])
        print(json.dumps(dict(job=j["name"], s=round(time.time() - t, 1), audio_s=j["n30"] / 30 + 1)), flush=True)
    print("DONE", a.shard, flush=True)


# ------------------------------------------------------------------------------------------------ analysis
def qmap_fit(x, y, nq=101):
    q = np.linspace(0, 1, nq)
    return np.quantile(x, q), np.quantile(y, q)


def qmap_apply(x, qx, qy):
    qx = np.maximum.accumulate(qx + np.arange(len(qx)) * 1e-9)
    return np.interp(x, qx, qy)


def events(bs, mask):
    # event tracks (nod / blink / smile onsets from blendshapes): the face-event module of the face pipeline,
    # importable as `events` from $FACE_TOOLS_DIR (not shipped)
    if os.environ.get("FACE_TOOLS_DIR"):
        sys.path.insert(0, os.environ["FACE_TOOLS_DIR"])
    import events as EV
    tr = EV.tracks(bs)
    out = {}
    m = np.asarray(mask, bool)
    mins = m.sum() / 30 / 60
    on = EV.blink_onsets(tr["blink"])
    out["blinks"] = float(m[on].sum() / max(mins, 1e-9)) if len(on) else 0.0
    sm = EV.rises(tr["smile"], EV.SMILE_RISE, EV.SMILE_WIN)
    out["smiles"] = float(m[sm].sum() / max(mins, 1e-9)) if len(sm) else 0.0
    br = EV.rises(tr["brow"], EV.BROW_RISE, EV.BROW_WIN)
    out["brows"] = float(m[br].sum() / max(mins, 1e-9)) if len(br) else 0.0
    from scipy.signal import find_peaks
    pk, _ = find_peaks(tr["jaw"], prominence=0.05, distance=3)
    out["mouth_opens"] = float(m[pk].sum() / max(mins, 1e-9)) if len(pk) else 0.0
    jaw = tr["jaw"][m]
    out["jaw_sd"] = float(jaw.std()) if m.any() else 0.0
    out["minutes"] = float(mins)
    return out


RIG_MAP = {"jaw": ["jawOpen"], "round": ["mouthPucker", "mouthFunnel"], "spread": ["mouthStretchLeft", "mouthStretchRight"],
           "press": ["mouthPressLeft", "mouthPressRight"], "smile": ["mouthSmileLeft", "mouthSmileRight"],
           "blink_l": ["eyeBlinkLeft"], "blink_r": ["eyeBlinkRight"], "squint": ["eyeSquintLeft", "eyeSquintRight"],
           "eye_wide": ["eyeWideLeft", "eyeWideRight"], "brow_l": ["browOuterUpLeft"], "brow_r": ["browOuterUpRight"],
           "lip_lower": ["mouthLowerDownLeft", "mouthLowerDownRight"], "lip_upper": ["mouthUpperUpLeft", "mouthUpperUpRight"],
           "cheek": ["cheekSquintLeft", "cheekSquintRight"]}


def rules_face(rig_path, n):
    """the rules controller's rig stream (rig.jsonl, one row per 30 fps frame) -> (n, 52) pseudo-blendshapes: each rig
    channel copied into the ARKit dial(s) it drives (RIG_MAP); browInnerUp = mean(brow_l, brow_r), browDown = brow_knit.
    Per-dial correlations are scale-free; event counts use the rig's own units (approximate)."""
    rows = [json.loads(l) for l in open(rig_path)]
    names = rows[0]["rig_params"]
    A = np.array([r["a"] for r in rows if r.get("type") == "rig"], np.float32)[:n]
    out = np.zeros((n, 52), np.float32)
    ix = {k: i for i, k in enumerate(names)}
    for ch, dials in RIG_MAP.items():
        if ch in ix:
            for d in dials:
                out[: len(A), S.BS[d]] = A[:, ix[ch]]
    out[: len(A), S.BS["browInnerUp"]] = 0.5 * (A[:, ix["brow_l"]] + A[:, ix["brow_r"]])
    if "brow_knit" in ix:
        out[: len(A), S.BS["browDownLeft"]] = A[:, ix["brow_knit"]]
        out[: len(A), S.BS["browDownRight"]] = A[:, ix["brow_knit"]]
    return out


def analyze(a):
    jobs = json.load(open(os.path.join(WORK, "jobs.json")))
    idn = a.identity
    cal_x, cal_y = [], []
    for j in jobs:
        if j["kind"] != "calib":
            continue
        fn = os.path.join(WORK, f"{j['name']}_{idn}.npz")
        if not os.path.exists(fn):
            continue
        z = np.load(os.path.join(CACHE, j["fid"] + ".npz"), allow_pickle=False)
        sl = slice(j["k0"], j["k0"] + j["n30"])
        ok = (z["vad"][sl].astype(np.float32) > 0.5) & z["mp_face"][sl].astype(bool)
        cal_x.append(np.load(fn)["a2f"][ok]); cal_y.append(z["mp_bs"][sl].astype(np.float32)[ok])
    X, Y = np.concatenate(cal_x), np.concatenate(cal_y)
    corr_cal = np.array([np.corrcoef(X[:, d], Y[:, d])[0, 1] if X[:, d].std() > 1e-4 and Y[:, d].std() > 1e-6
                         else 0.0 for d in range(52)])
    valid = corr_cal >= a.min_corr                        # chosen on TRAIN people only
    qx = np.zeros((52, 101)); qy = np.zeros((52, 101))
    for d in range(52):
        qx[d], qy[d] = qmap_fit(X[:, d], Y[:, d])
    gain = np.where(valid, Y.std(0) / np.maximum(X.std(0), 1e-6), 0)
    calib = dict(identity=idn, n_calib_frames=int(len(X)), n_calib_people=len(cal_x), valid=valid, qx=qx, qy=qy,
                 a2f_mean=X.mean(0), a2f_std=X.std(0), mp_mean=Y.mean(0), mp_std=Y.std(0), gain=gain,
                 corr_calib=corr_cal, method=np.array(a.cal))
    np.savez(os.path.join(WORK, f"calib_{idn}.npz"), **calib)
    mu_x, mu_y = X.mean(0), Y.mean(0)

    def cal(A_):
        out = np.tile(mu_y, (len(A_), 1))
        for d in np.flatnonzero(valid):
            out[:, d] = qmap_apply(A_[:, d], qx[d], qy[d]) if a.cal == "quantile" else \
                np.clip(mu_y[d] + (A_[:, d] - mu_x[d]) * gain[d], 0, 1)
        return out
    systems = ["a2f_raw", "a2f_cal", "canon_mapper", "rules", "model_abl_t0"]
    per_dial = {s: {n: [] for n in DIALS} for s in systems}
    ev = {s: [] for s in systems + ["mediapipe"]}
    pooled = {s: ([], []) for s in systems}
    used = 0
    for j in jobs:
        if j["kind"] != "eval":
            continue
        fn = os.path.join(WORK, f"{j['name']}_{idn}.npz")
        if not os.path.exists(fn):
            continue
        z = np.load(os.path.join(CACHE, j["fid"] + ".npz"), allow_pickle=False)
        sl = slice(j["k0"], j["k0"] + j["n30"])
        mp = z["mp_bs"][sl].astype(np.float32)
        ok = (z["vad"][sl].astype(np.float32) > 0.5) & z["mp_face"][sl].astype(bool)
        if ok.sum() < 60:
            continue
        raw = np.load(fn)["a2f"]
        sysd = {"a2f_raw": raw, "a2f_cal": cal(raw), "canon_mapper": z["canon"][sl].astype(np.float32)}
        p = os.path.join(EVAL, "runs", "unified-abl_t0", j["clip"], "canon.npz")
        if os.path.exists(p):
            sysd["model_abl_t0"] = np.load(p, allow_pickle=False)["motion"][: j["n30"], S.SL_FACE].astype(np.float32)
        p = os.path.join(EVAL, "runs", "rules", j["clip"], "rig.jsonl")
        if os.path.exists(p):
            sysd["rules"] = rules_face(p, j["n30"])
        used += 1
        ev["mediapipe"].append(events(mp, ok))
        for s, v in sysd.items():
            for n in DIALS:
                a_, b_ = v[ok, S.BS[n]], mp[ok, S.BS[n]]
                per_dial[s][n].append(float(np.corrcoef(a_, b_)[0, 1]) if a_.std() > 1e-6 and b_.std() > 1e-6
                                      else np.nan)
            ev[s].append(events(v, ok))
            pooled[s][0].append(v[ok]); pooled[s][1].append(mp[ok])
    out = {"identity": idn, "eval_clips": used, "calib": {"people": len(cal_x), "frames": int(len(X)), "method": a.cal,
           "min_corr": a.min_corr, "valid_dials": [n for n, v in zip(S.BS_NAMES, valid) if v],
           "corr_calib": {n: round(float(c), 3) for n, c in zip(S.BS_NAMES, corr_cal) if abs(c) > 0}}}
    out["per_dial_corr_mean_over_clips"] = {s: {n: (round(float(np.nanmean(v)), 3) if np.isfinite(v).any() else None)
                                               for n, v in d.items()} for s, d in per_dial.items()}
    out["pooled_corr"] = {}
    for s, (A_, B_) in pooled.items():
        if not A_:
            continue
        A_, B_ = np.concatenate(A_), np.concatenate(B_)
        out["pooled_corr"][s] = {n: (round(float(np.corrcoef(A_[:, S.BS[n]], B_[:, S.BS[n]])[0, 1]), 3)
                                     if A_[:, S.BS[n]].std() > 1e-6 else None) for n in DIALS}
    def agg(lst):
        if not lst:
            return {}
        mins = np.array([e["minutes"] for e in lst])
        return {k: round(float(np.sum([e[k] * e["minutes"] for e in lst]) / mins.sum()), 2) if k != "jaw_sd"
                else round(float(np.mean([e[k] for e in lst])), 3) for k in lst[0] if k != "minutes"} | \
               {"speaking_min": round(float(mins.sum()), 2)}
    out["events_per_speaking_min"] = {s: agg(v) for s, v in ev.items()}
    json.dump(out, open(os.path.join(WORK, f"report_{idn}.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["jobs", "run", "analyze"])
    ap.add_argument("--identity", default="James")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--calib-s", type=int, default=40)
    ap.add_argument("--cal", default="affine", choices=["affine", "quantile"])
    ap.add_argument("--min-corr", type=float, default=0.15)
    a = ap.parse_args()
    {"jobs": make_jobs, "run": run, "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
