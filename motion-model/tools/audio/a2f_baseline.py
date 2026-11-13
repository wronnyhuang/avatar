"""Audio2Face-3D v3.0 as a SPEAKING-FACE BASELINE system on the scorecard (successor of phase 2).

  calib    on the TRAIN calibration segments of a2f_eval (one 40 s speech-rich piece per train person, A2F already run):
           (i)  A2F smile -> the canon face's smile scale (affine std-match, speaking frames) so that the retarget's
                canon->Muse smile map (retarget/maps/_base.json: smile = 1.25 (s - 0.10)) applies unchanged;
           (ii) A2F mouth dials -> the Muse's lip columns at the renderer (the columns lipstream owns), affine std-match
                to lipstream's own per-column mean/SD on the same speech (units only; one semantic source per column).
           -> $P/a2fbase_work/a2f_onscreen_calib.npz + .json
  bench    A2F latency on the host CPU: ms per network run (1 s window -> 0.5 s of output) at 2 and 4 threads, p50/p99
           with the load average, + the blendshape solve per frame; look-ahead arithmetic.
  lookahead  quality vs look-ahead: same runs as the SDK (stride 0.5 s) but keep rows nearer the window END (look-ahead
           o..o+0.5 s instead of 0.25..0.75 s); lookahead_report scores jaw/smile/pucker corr vs MediaPipe (eval clips).
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import numpy as np  # noqa: E402

S = C.spec
P = C.P
WORK = os.path.join(P, "a2fbase_work")
A2F_EVAL = os.path.join(P, "audio_work", "a2f_eval")
CACHE = os.path.join(P, "face", "cache")
CALIB = os.path.join(WORK, "a2f_onscreen_calib.npz")
PRE = 30
IDENT = "James"

sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "evalkit"))
import a2f_face as AF  # noqa: E402  (self-contained: no evalkit `common`)

AVATAR_RENDERER = str(AF.RENDERER)
EVAL_CLIPS = os.path.join(os.environ.get("EVAL_OUT", os.path.join(os.path.dirname(P), "eval")), "clips")
LIP_SRC, SMILE, src_of = AF.LIP_SRC, AF.SMILE, AF.src_of


def load24(wav, t0_s, t1_s):
    """int16 24 kHz mono of [t0, t1) s of a wav (zeros outside)."""
    import soundfile as sf
    from scipy.signal import resample_poly
    info = sf.info(wav)
    sr = info.samplerate
    a = max(0, int(math.floor(t0_s * sr))); b = min(info.frames, int(math.ceil(t1_s * sr)))
    n_out = int(round((t1_s - t0_s) * 24000))
    out = np.zeros(n_out, np.float32)
    if b > a:
        x, _ = sf.read(wav, start=a, stop=b, dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != 24000:
            g = math.gcd(sr, 24000)
            x = resample_poly(x.astype(np.float64), 24000 // g, sr // g).astype(np.float32)
        o = int(round((a / sr - t0_s) * 24000))
        lo = max(0, o); hi = min(n_out, o + len(x))
        if hi > lo:
            out[lo:hi] = x[lo - o: hi - o]
    return np.clip(out * 32768.0, -32768, 32767).astype(np.int16)


def _lip_job(j):
    import onscreen as OS  # noqa: N812  (evalkit, self-contained)
    k0 = j["k0"] - PRE
    pcm = load24(j["wav"], k0 / 30.0, (k0 + j["n30"] + PRE) / 30.0)
    return j["name"], OS.lip_rows_pcm(pcm, j["n30"] + PRE)[PRE:]


def calib(a):
    import concurrent.futures as cf
    import multiprocessing as mp
    sys.path.insert(0, AVATAR_RENDERER)
    from anim.schema import IDX
    os.makedirs(WORK, exist_ok=True)
    jobs = [j for j in json.load(open(os.path.join(A2F_EVAL, "jobs.json"))) if j["kind"] == "calib"
            and os.path.exists(os.path.join(A2F_EVAL, f"{j['name']}_{IDENT}.npz"))]
    lipd = os.path.join(WORK, "calib_lip")
    os.makedirs(lipd, exist_ok=True)
    todo = [j for j in jobs if not os.path.exists(os.path.join(lipd, j["name"] + ".npy"))]
    if todo:
        with cf.ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as pool:
            for name, rows in pool.map(_lip_job, todo):
                np.save(os.path.join(lipd, name + ".npy"), rows)
                print("lip", name, rows.shape, flush=True)
    X, CAN, MP, LIP = [], [], [], []
    for j in jobs:
        z = np.load(os.path.join(CACHE, j["fid"] + ".npz"), allow_pickle=False)
        sl = slice(j["k0"], j["k0"] + j["n30"])
        ok = (z["vad"][sl].astype(np.float32) > 0.5) & z["mp_face"][sl].astype(bool) & z["canon_mask"][sl].astype(bool)
        X.append(np.load(os.path.join(A2F_EVAL, f"{j['name']}_{IDENT}.npz"))["a2f"][ok])
        CAN.append(z["canon"][sl].astype(np.float32)[ok]); MP.append(z["mp_bs"][sl].astype(np.float32)[ok])
        LIP.append(np.load(os.path.join(lipd, j["name"] + ".npy"))[ok])
    X, CAN, MP, LIP = (np.concatenate(v) for v in (X, CAN, MP, LIP))
    out = {"identity": IDENT, "people": len(jobs), "frames": int(len(X)),
           "doc": "fitted on speaking frames (own vad & MediaPipe face & canon valid) of the a2f_eval TRAIN calib segments"}
    # (i) smile -> canon scale (and MediaPipe, for reference)
    xs, cs, ms = src_of(X, SMILE), src_of(CAN, SMILE), src_of(MP, SMILE)
    out["smile_canon"] = {"a_mean": float(xs.mean()), "a_sd": float(xs.std()), "y_mean": float(cs.mean()),
                          "y_sd": float(cs.std()), "corr_train": float(np.corrcoef(xs, cs)[0, 1]),
                          "corr_train_mp": float(np.corrcoef(xs, ms)[0, 1])}
    xj, cj = X[:, S.BS["jawOpen"]], CAN[:, S.BS["jawOpen"]]
    out["jaw_canon"] = {"a_mean": float(xj.mean()), "a_sd": float(xj.std()), "y_mean": float(cj.mean()),
                        "y_sd": float(cj.std()), "corr_train": float(np.corrcoef(xj, cj)[0, 1])}
    # (ii) A2F -> Muse lip columns, std-matched to lipstream on the same speech; corr = how alike the two lip syncs are
    lip = {}
    for col, src in LIP_SRC.items():
        x = src_of(X, src); y = LIP[:, IDX[col]]
        lip[col] = {"src": src, "a_mean": float(x.mean()), "a_sd": float(x.std()), "y_mean": float(y.mean()),
                    "y_sd": float(y.std()), "corr_a2f_vs_lipstream": float(np.corrcoef(x, y)[0, 1]) if y.std() > 1e-6 else None}
    out["lip"] = lip
    # lipstream vs MediaPipe / canon on the same train speech (diagnostic)
    out["lipstream_vs_mp_train"] = {c: float(np.corrcoef(LIP[:, IDX[c]], src_of(MP, s))[0, 1])
                                    for c, s in (("jaw", ["jawOpen"]), ("round", ["mouthPucker"]),
                                                 ("lip_lower", LIP_SRC["lip_lower"]), ("lip_upper", LIP_SRC["lip_upper"]))}
    out["a2f_vs_mp_train"] = {c: float(np.corrcoef(src_of(X, s), src_of(MP, s))[0, 1])
                              for c, s in (("jaw", ["jawOpen"]), ("round", ["mouthPucker"]),
                                           ("lip_lower", LIP_SRC["lip_lower"]), ("lip_upper", LIP_SRC["lip_upper"]),
                                           ("smile", SMILE))}
    json.dump(out, open(CALIB[:-4] + ".json", "w"), indent=1)
    np.savez(CALIB, json=np.array(json.dumps(out)))
    print(json.dumps(out, indent=1))


def a2f_raw_for_clip(cid):
    p = os.path.join(A2F_EVAL, f"eval_{cid}_{IDENT}.npz")
    return np.load(p)["a2f"] if os.path.exists(p) else None


# ------------------------------------------------------------------------------------------------ latency
def bench(a):
    import a2f as A
    from scipy.optimize import lsq_linear  # noqa: F401  (import cost outside the timing)
    res = {"host_cpus": os.cpu_count(), "t": time.strftime("%Y-%m-%d %H:%M:%S")}
    wav = os.path.join(EVAL_CLIPS, "sl_S0644_0578_P0299A", "voice.wav")
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(wav, dtype="float32", always_2d=True)
    x = resample_poly(x.mean(1).astype(np.float64), 2, 3).astype(np.float32) if sr == 24000 else x.mean(1)
    for th in [int(t) for t in a.threads.split(",")]:
        m = A.A2F(IDENT, threads=th)
        m.reset()
        ms, la = [], []
        for i in range(a.warm + a.n):
            s0 = (i % 60) * 8000
            win = x[s0: s0 + 16000]
            if len(win) < 16000:
                win = np.concatenate([win, np.zeros(16000 - len(win), np.float32)])
            t0 = time.perf_counter()
            pred = m.run_window(win)
            dt = (time.perf_counter() - t0) * 1000
            if i >= a.warm:
                ms.append(dt); la.append(os.getloadavg()[0])
        # per-frame blendshape solve (the SDK's per-frame BVLS) and the batch FISTA solve over 30 frames
        sk = pred[15:45, :A.SKIN]
        t0 = time.perf_counter()
        for r in sk[:10]:
            m.solve_frame(r)
        solve_ms = (time.perf_counter() - t0) * 1000 / 10
        t0 = time.perf_counter()
        m.solve_batch(sk)
        batch_ms = (time.perf_counter() - t0) * 1000
        ms = np.array(ms)
        res[f"threads_{th}"] = {"n": len(ms), "run_ms_p50": round(float(np.percentile(ms, 50)), 1),
                                "run_ms_p90": round(float(np.percentile(ms, 90)), 1),
                                "run_ms_p99": round(float(np.percentile(ms, 99)), 1),
                                "run_ms_min": round(float(ms.min()), 1),
                                "load_avg_1m": [round(float(min(la)), 1), round(float(np.median(la)), 1), round(float(max(la)), 1)],
                                "solve_ms_per_frame_bvls": round(solve_ms, 2), "solve_ms_30_frames_fista": round(batch_ms, 1),
                                "rtf_stride_0.5s": round(float(np.percentile(ms, 50)) / 500.0, 2)}
        print(json.dumps({th: res[f"threads_{th}"]}), flush=True)
        del m
    res["lookahead"] = ("window 1.0 s, stride 0.5 s, the SDK keeps rows 15..44 of 60 (60 fps): a run on audio [t-1, t] "
                        "outputs frames [t-0.75, t-0.25) -> each frame waits 0.25-0.75 s for its audio + the run time")
    json.dump(res, open(os.path.join(WORK, "bench.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------------ look-ahead probe
def lookahead(a):
    """quality vs look-ahead at the SDK's own stride (0.5 s, GRU state semantics unchanged): each run on audio [t-1, t]
    outputs 60 rows (60 fps) over that window; the SDK keeps rows 15..44 (frames [t-0.75, t-0.25): look-ahead 0.25-0.75 s).
    Keeping rows [30 - 60 o', 60 - 60 o') instead (o' = offset of the kept block's END before the window end) gives
    look-ahead o'..o'+0.5 s. Same runs for every offset; jaw/smile corr vs MediaPipe in lookahead_report."""
    import a2f as A
    man = json.load(open(os.path.join(EVAL_CLIPS, "manifest.json")))["clips"]
    clips = [c for c in man if c["id"].startswith("sl_")]
    k_, n_ = [int(v) for v in a.shard.split("/")]
    if a.ids:
        clips = [c for c in clips if c["id"] in a.ids.split(",")]
    clips = [c for i, c in enumerate(clips) if i % n_ == k_][: a.max_clips]
    offs = [float(v) for v in a.offsets.split(",")]
    m = A.A2F(IDENT, threads=a.threads_la)
    od = os.path.join(WORK, "lookahead"); os.makedirs(od, exist_ok=True)
    for c in clips:
        fn = os.path.join(od, c["id"] + ".npz")
        if os.path.exists(fn):
            continue
        n30 = int(round(c["duration_s"] * 30))
        k0 = int(c["frame0"]) - PRE
        x16 = C.load_segment(c["voice_wav"], S.frame_end_sample(k0 - 1), S.frame_end_sample(k0 + n30 + PRE - 1))
        xp = np.concatenate([np.zeros(16000, np.float32), x16, np.zeros(32000, np.float32)])
        n60 = 2 * (n30 + PRE)
        sk = {o: np.zeros((n60, len(m.pos)), np.float32) for o in offs}
        m.reset()
        t0 = time.time()
        i = 0
        while 30 * i - 60 < n60:
            pred = m.run_window(xp[8000 * i: 8000 * i + 16000])
            for o in offs:
                r1 = 60 - int(round(60 * o))
                for r in range(r1 - 30, r1):
                    n = 30 * i - 60 + r
                    if 0 <= n < n60:
                        sk[o][n] = pred[r, :A.SKIN][m.pos]
            i += 1
        out = {}
        for o in offs:
            W = m.solve_batch_masked(sk[o])
            out[f"o{int(round(o * 1000))}"] = A.to_canon(W, m.pose_names, n30 + PRE)[PRE:]
        np.savez(fn, **out)
        print(json.dumps({"clip": c["id"], "runs": i, "s": round(time.time() - t0, 1)}), flush=True)
    print("DONE", a.shard, flush=True)


def lookahead_report(a):
    man = {c["id"]: c for c in json.load(open(os.path.join(EVAL_CLIPS, "manifest.json")))["clips"]}
    od = os.path.join(WORK, "lookahead")
    rows = {}
    for f in sorted(os.listdir(od)):
        if not f.endswith(".npz"):
            continue
        cid = f[:-4]; c = man[cid]
        z = np.load(os.path.join(od, f))
        fz = np.load(os.path.join(CACHE, c["avatar_file"] + ".npz"), allow_pickle=False)
        n30 = int(round(c["duration_s"] * 30)); sl = slice(int(c["frame0"]), int(c["frame0"]) + n30)
        mp = fz["mp_bs"][sl].astype(np.float32)
        ok = (fz["vad"][sl].astype(np.float32) > 0.5) & fz["mp_face"][sl].astype(bool)
        if ok.sum() < 60:
            continue
        ref = a2f_raw_for_clip(cid)
        srcs = {"sdk_centre(0.25-0.75s)": ref} | {k: z[k] for k in z.files}
        for k, v in srcs.items():
            for d, names in (("jaw", ["jawOpen"]), ("smile", SMILE), ("pucker", ["mouthPucker"])):
                x = src_of(v, names)[ok]; y = src_of(mp, names)[ok]
                rows.setdefault(k, {}).setdefault(d, []).append(float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-6 else np.nan)
    rep = {k: {d: round(float(np.nanmean(v)), 3) for d, v in dd.items()} | {"n_clips": len(dd["jaw"])} for k, dd in rows.items()}
    json.dump(rep, open(os.path.join(WORK, "lookahead_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calib", "bench", "lookahead", "lookahead_report"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", default="2,4")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max-clips", type=int, default=99)
    ap.add_argument("--ids", default="")
    ap.add_argument("--threads-la", type=int, default=4)
    ap.add_argument("--offsets", default="0.25,0.1,0.0")  # o' in s; 0.25 = the SDK
    a = ap.parse_args()
    {"calib": calib, "bench": bench, "lookahead": lookahead, "lookahead_report": lookahead_report}[a.cmd](a)


if __name__ == "__main__":
    main()
