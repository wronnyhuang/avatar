"""phase-2 plan item 3: Audio2Face-3D teacher targets for the streams of a stream list (phase-2 audio).

  run       raw A2F (identity James) on each stream's OWN-voice speaking segments (VAD runs merged over < 0.5 s gaps,
            +-10 frames, 1 s of real pre-roll for the GRU) -> audio_work/a2f_raw/<stream_id>.npz {a2f (n,52) canon order,
            raw A2F scale; computed (n,) bool}. Order: val streams first, then train streams in a fixed random order,
            so any prefix is a uniform subset. CPU: ~1.4x real time per 5-thread worker under load ~120.
  finalize  apply the calibration (audio_work/a2f_eval/calib_<identity>.npz: per-dial affine map A2F -> MediaPipe
            fitted on train people's speaking frames) and write, for every stream with a raw A2F run:
              $P/a2f/<stream_id>.npy       float16 (n_frames, 52) canon order/scale (0 outside the speaking segments)
              $P/a2f/<stream_id>.mask.npy  uint8 (n_frames,) 1 = the person speaks (VAD) AND A2F was computed
            Streams without a run get NO file (pack_v3 add-audio: a2f_ok = 0 -> no teacher there).
            + $P/a2f/META.json (licence, version, calibration, coverage, valid dials).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import numpy as np  # noqa: E402

S = C.spec
RAW = os.path.join(C.P, "audio_work", "a2f_raw")
OUT = os.path.join(C.P, "a2f")
PRE, PAD, GAP = 30, 10, 15


def order(rows):
    val = [r for r in rows if r["split"] == "val"]
    tr = [r for r in rows if r["split"] != "val"]
    rng = np.random.default_rng(4330)
    tr = [tr[i] for i in rng.permutation(len(tr))]
    return val + tr


def segments(vad):
    v = np.asarray(vad) > 0.5
    n = len(v)
    idx = np.flatnonzero(np.diff(np.concatenate([[0], v.astype(np.int8), [0]])))
    runs = list(zip(idx[::2], idx[1::2]))
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] < GAP:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(max(0, a - PAD), min(n, b + PAD)) for a, b in merged]


def run(a):
    import a2f as A
    os.makedirs(RAW, exist_ok=True)
    rows = order(C.streams(a.streams))
    k, n = [int(x) for x in a.shard.split("/")]
    rows = [r for i, r in enumerate(rows) if i % n == k]
    if a.limit:
        rows = rows[: a.limit]
    m = A.A2F(a.identity, threads=a.threads)
    t0 = time.time()
    done = 0
    audio_s = 0.0
    for r in rows:
        fn = os.path.join(RAW, r["stream_id"] + ".npz")
        if os.path.exists(fn):
            continue
        _, _, au = C.canon_person(r)
        vad = au[:, 41]
        N = r["n_frames"]
        out = np.zeros((N, 52), np.float32)
        comp = np.zeros(N, bool)
        for s0, s1 in segments(vad):
            k0 = r["frame0"] + s0 - PRE
            n30 = s1 - s0 + PRE
            x16 = C.load_segment(r["audio_self"], S.frame_end_sample(k0 - 1), S.frame_end_sample(k0 + n30 - 1))
            o = m.run_track(x16, n30=n30)
            out[s0:s1] = A.to_canon(o["w60"], o["pose_names"], n30)[PRE:]
            comp[s0:s1] = True
            audio_s += n30 / 30
        tmp = fn[:-4] + f".tmp{os.getpid()}.npz"
        np.savez(tmp, a2f=out, computed=comp, identity=np.array(a.identity), vad=vad.astype(np.float16))
        os.replace(tmp, fn)
        done += 1
        el = time.time() - t0
        print(json.dumps(dict(shard=a.shard, done=done, of=len(rows), sid=r["stream_id"], split=r["split"],
                              s=round(el, 1), rt_factor=round(el / max(audio_s, 1e-9), 2),
                              load=round(os.getloadavg()[0], 1))), flush=True)
        if a.deadline and time.time() > a.deadline:
            print("DEADLINE", flush=True)
            break
    print("DONE", a.shard, done, flush=True)


def finalize(a):
    import a2f_eval as AE
    cal = np.load(os.path.join(C.P, "audio_work", "a2f_eval", f"calib_{a.identity}.npz"))
    valid, qx, qy, mp_mean = cal["valid"], cal["qx"], cal["qy"], cal["mp_mean"]
    method = str(cal["method"]) if "method" in cal.files else "quantile"
    a_mean, gain = cal["a2f_mean"], cal["gain"]
    os.makedirs(OUT, exist_ok=True)
    rows = C.streams(a.streams)
    tot_f = os.path.join(C.P, "audio_work", "speak_total.json")
    if os.path.exists(tot_f):
        n_speak = json.load(open(tot_f))["speaking_frames"]
    else:
        n_speak = int(sum((C.canon_person(r)[2][:, 41] > 0.5).sum() for r in rows))
        json.dump({"speaking_frames": n_speak}, open(tot_f, "w"))
    n_comp = n_speak_comp = 0
    streams_comp = {"train": 0, "val": 0}
    for r in rows:
        fn = os.path.join(RAW, r["stream_id"] + ".npz")
        if not os.path.exists(fn):
            continue                          # no file = no teacher for this stream (pack_v3 add-audio: a2f_ok = 0)
        N = r["n_frames"]
        z = np.load(fn)
        raw, comp = z["a2f"], z["computed"]
        vad = z["vad"].astype(np.float32) if "vad" in z.files else C.canon_person(r)[2][:, 41]
        speak = vad > 0.5
        y = np.zeros((N, 52), np.float32)
        for d in range(52):
            if not valid[d]:
                y[:, d] = mp_mean[d]
            elif method == "quantile":
                y[:, d] = AE.qmap_apply(raw[:, d], qx[d], qy[d])
            else:
                y[:, d] = np.clip(mp_mean[d] + (raw[:, d] - a_mean[d]) * gain[d], 0, 1)
        y[~comp] = 0
        mk = (speak & comp).astype(np.uint8)
        n_comp += int(comp.sum()); n_speak_comp += int(mk.sum())
        streams_comp[r["split"]] = streams_comp.get(r["split"], 0) + 1
        for path, arr in ((os.path.join(OUT, r["stream_id"] + ".npy"), y.astype(np.float16)),
                          (os.path.join(OUT, r["stream_id"] + ".mask.npy"), mk)):
            tmp = path[:-4] + ".tmp.npy"
            np.save(tmp, arr)
            os.replace(tmp, path)
    rep = json.load(open(os.path.join(C.P, "audio_work", "a2f_eval", f"report_{a.identity}.json")))
    meta = dict(
        model="nvidia/Audio2Face-3D-v3.0 (diffusion, HuBERT audio encoder, 180M params), network.onnx on CPU "
              "(onnxruntime 1.31), identity " + a.identity + ", emotion = zeros (neutral), constant diffusion noise "
              "(seed 0), GRU state carried within each speaking segment (1 s real pre-roll)",
        licence="NVIDIA Open Model License (nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license, "
                "Oct 24 2025): use + derivative works + outputs permitted (outputs are not Derivative Models; NVIDIA "
                "claims no ownership of outputs); obligations only on redistribution of the model (Notice: 'Licensed by "
                "NVIDIA Corporation under the NVIDIA Open Model License') - we do not redistribute; Trustworthy AI terms. "
                "Cite: NVIDIA, Audio2Face-3D, arXiv 2508.16401.",
        solve="SDK blendshape solve re-implemented from Audio2Face-3D-SDK source (frontalMask vertices, active poses, "
              "L1/L2/temporal/symmetry regularisation, bounds 0..1; FISTA batch solver, max |diff| vs BVLS 5e-4)",
        timing="canon frame k (time k/30) = A2F 60 fps frame 2k; A2F look-ahead 0.25-0.75 s (offline teacher only)",
        files={"<stream_id>.npy": "float16 (n_frames, 52) canon order (canon1/spec.py BS_NAMES), MediaPipe scale",
               "<stream_id>.mask.npy": "uint8 (n_frames,) 1 = person speaks (canon audio42 vad) and A2F computed"},
        calibration=dict(method="per-dial affine std-match y = mp_mean + (x - a2f_mean) * sd_mp / sd_a2f, clipped "
                                "0..1, fitted on the speaking frames (vad & face) of one 40 s speech-rich segment per "
                                "TRAIN person of the Seamless MediaPipe dev videos ($P/face/cache); valid dials = "
                                "train-people corr >= 0.15" if method == "affine" else "per-dial quantile map",
                         params={n: dict(a2f_mean=round(float(a_mean[i]), 5), gain=round(float(gain[i]), 4),
                                         mp_mean=round(float(mp_mean[i]), 5)) for i, n in enumerate(S.BS_NAMES)
                                 if valid[i]},
                         people=int(rep["calib"]["people"]), frames=int(rep["calib"]["frames"]),
                         file=os.path.join(C.P, "audio_work", "a2f_eval", f"calib_{a.identity}.npz")),
        valid_dials=[n for n, v in zip(S.BS_NAMES, valid) if v],
        invalid_dials_note="every dial NOT in valid_dials (brows, blinks, squint, eyeLook*, ... - A2F's upper face does "
                           "not track real faces, corr ~0) holds the MediaPipe mean: apply the teacher loss on "
                           "valid_dials only",
        coverage=dict(streams_total=len(rows), streams_computed=streams_comp, frames_computed=n_comp,
                      speaking_frames_total=n_speak, speaking_frames_with_teacher=n_speak_comp),
        heldout_report=os.path.join(C.P, "audio_work", "a2f_eval", f"report_{a.identity}.json"),
        heldout_summary=dict(
            note="27 eval-harness Seamless test clips (9 held-out people), avatar speaking frames, MediaPipe = truth",
            corr_a2f_calibrated={n: rep["per_dial_corr_mean_over_clips"]["a2f_cal"].get(n)
                                 for n in rep["per_dial_corr_mean_over_clips"]["a2f_cal"]
                                 if rep["per_dial_corr_mean_over_clips"]["a2f_cal"].get(n) is not None},
            corr_model_abl_t0={n: rep["per_dial_corr_mean_over_clips"]["model_abl_t0"].get(n)
                               for n in ("jawOpen", "mouthPucker", "mouthSmileLeft", "eyeBlinkLeft", "browInnerUp")},
            events_per_speaking_min={k: rep["events_per_speaking_min"].get(k) for k in
                                     ("mediapipe", "a2f_cal", "canon_mapper", "rules", "model_abl_t0")}),
        version="p2-audio a2f v1",
    )
    json.dump(meta, open(os.path.join(OUT, "META.json"), "w"), indent=1)
    print(json.dumps(meta["coverage"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "finalize"])
    ap.add_argument("--identity", default="James")
    ap.add_argument("--streams", default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--deadline", type=float, default=0, help="unix time; stop after the stream in progress")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else finalize(a)


if __name__ == "__main__":
    main()
