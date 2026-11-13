"""Write $P/feats/<enc>/META.json (phase-2 plan item 2) from the PCA file, the bench, the probe and the extraction logs."""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import numpy as np  # noqa: E402

CFG = {
    "whisper_tiny": dict(mode="sliding", layer=2, window_s=1.0, lookahead_frames=1, hf="openai/whisper-tiny",
                         probe="probe_eval_wt_sliding.json", probe_key="whisper_tiny_W1_L1"),
    "whisper_base": dict(mode="sliding", layer=3, window_s=1.0, lookahead_frames=1, hf="openai/whisper-base",
                         probe="probe_eval_whisper_base_W1_L1.json", probe_key="whisper_base_W1_L1"),
    "wav2vec2_base": dict(mode="kv", layer=4, window_s=1.0, lookahead_frames=0, hf="facebook/wav2vec2-base",
                          probe="probe_eval_wav2vec2_base_kv.json", probe_key="wav2vec2_base_kv"),
    "hubert_base": dict(mode="kv", layer=6, window_s=1.0, lookahead_frames=0, hf="facebook/hubert-base-ls960",
                        probe="probe_eval_hubert_base_kv.json", probe_key="hubert_base_kv"),
}
W = os.path.join(C.P, "audio_work")


def main(enc):
    c = CFG[enc]
    d = os.path.join(C.P, "feats", enc)
    z = np.load(os.path.join(d, "pca.npz"))
    rows = C.streams()
    have = sum(os.path.exists(os.path.join(d, r["stream_id"] + ".npy")) for r in rows)
    bench = [b for b in json.load(open(os.path.join(W, "bench_final.json"))) if b["enc"] == enc] \
        if os.path.exists(os.path.join(W, "bench_final.json")) else []
    rt = {f"threads_{b['threads']}": dict(sliding=b["sliding"], kv_step=b["kv_step"], load_avg=b["load"]) for b in bench}
    sps = []
    for f in glob.glob(os.path.join(W, "logs", f"ext_{enc}_*.log")):
        for line in open(f):
            m = re.search(r'"s_per_stream": ([0-9.]+).*"load": ([0-9.]+)', line)
            if m:
                last = (float(m.group(1)), float(m.group(2)))
        sps.append(last)
    pr = {}
    pf = os.path.join(W, c["probe"])
    if os.path.exists(pf):
        P = json.load(open(pf))
        lay = P[c["probe_key"]]["layers"].get(f"l{c['layer']:02d}", {})
        pr = {k: lay[k] for k in ("self_mean", "cross_mean", "self:smile", "self:jawOpen", "self:head_speed",
                                  "cross:vad") if k in lay}
        pr["audio42_same_probe"] = {k: P["audio42"]["layers"]["a42"][k] for k in ("self_mean", "cross_mean",
                                                                                    "self:smile")}
    la = c["lookahead_frames"]
    meta = dict(
        encoder=enc, hf_weights=c["hf"], frozen=True, layer=c["layer"],
        layer_convention="hidden state after transformer layer `layer` (0 = transformer input)",
        mode=c["mode"],
        mode_desc=("SLIDING: every 100 ms step the encoder re-runs on the last window_s of audio ending lookahead "
                   "frames after the step's newest frame" if c["mode"] == "sliding" else
                   "KV (block-causal): per 100 ms step only the 5 new encoder frames are computed, attending to "
                   "cached keys/values of the last 1 s; streamed conv front end with arrival-time statistics"),
        window_s=c["window_s"], fps=30, step_frames=3,
        step_grid="absolute: step m ends at 16 kHz sample 1600*m (session start = 0) and outputs frames 3m-3-L..3m-1-L",
        framing="frame k = mean of the 50 Hz encoder frames whose receptive-field end is in (frame_end(k-1), "
                "frame_end(k)], frame_end(k) = round((k+1)*16000/30) (the audio42 framing)",
        lookahead_frames=la,
        lookahead_ms={"min": round(la * 1000 / 30, 1), "max": round((la + 2) * 1000 / 30, 1),
                      "mean": round((la + 1) * 1000 / 30, 1)},
        causal_use="live, after step m completes (audio up to the end of frame 3m-1) rows up to 3m-1-lookahead_frames "
                   "exist. A planner that re-plans on the step grid sees row t-lookahead_frames at frame t: feed row "
                   "k-lookahead_frames at position k in training (row k itself already contains up to 67 ms of "
                   "in-step look-ahead for its first frames, identical live and offline)",
        dims=int(z["comps"].shape[0]), input_dim=int(z["comps"].shape[1]),
        pca=dict(file=os.path.join(d, "pca.npz"), whitened=True, explained_variance=round(float(z["explained"]), 4),
                 fit_on=f"{int(z['n_streams'])} TRAIN streams of the probe set ({int(z['n_frames'])} frames)",
                 formula="z = ((h - mean) @ comps.T) / scale"),
        files="<stream_id>.npy float16 (n_frames, dims), the stream's OWN audio (audio_self); partner = partner "
              "stream's file",
        streams_written=have, streams_total=len(rows),
        runtime_ms_per_100ms_step=rt,
        extraction=dict(seconds_per_30s_stream_per_worker=[s for s, _ in sps], load_avg=[l for _, l in sps],
                        threads_per_worker=4, workers=len(sps)),
        probe=pr,
        code="unified/speech_enc.py, tools/audio/extract_feats.py",
    )
    json.dump(meta, open(os.path.join(d, "META.json"), "w"), indent=1)
    print(enc, "META written;", have, "/", len(rows), "streams")


if __name__ == "__main__":
    for e in sys.argv[1:]:
        main(e)
