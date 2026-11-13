"""Write the phase-2 ablation multi config (tools/train/multi.py format, batches run one after another on ONE GPU).

    python -I tools/train/p2_abl_config.py --out tools/train/configs/p2_abl.json --pack abl_v3s.npz \
        --base-enc whisper_tiny --la 1 [--a2f] [--late wav2vec2_base:enc_wav2vec2_base.npz,hubert_base:enc_hubert_base.npz]
        [--minutes 30] [--batches "base,mel,noev,regr,ctx45|base2,ctx180,a2f,enc_wav2vec2_base,enc_hubert_base"]

Baseline = corrected faces + event heads + the base encoder (+ the a2f teacher when --a2f: then the a2f knob is
"a2f weight 0", else "a2f weight 0.5" with the a2f member added later). One knob per run. "base2" repeats the
baseline in the second batch (a per-batch control: runs of a batch share the GPU equally). enc_shift = la + 2
(the encoder's look-ahead + the 100 ms step grid: the newest user feature a live runtime is sure to have).
"""
import argparse
import json
import os

RUNTIME = {"replan_every": 3, "steps": 6, "guidance": 2.0, "temperature": 0.0, "blend": 6, "consistent_noise": True,
           "smooth": 0.0, "noise_rho": 0.0, "temperature_groups": None,
           "events": {"mode": "threshold", "gain": [1.0, 1.0, 1.0], "feed_history": True}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--base-enc", required=True)
    ap.add_argument("--la", type=int, required=True, help="base encoder look-ahead (frames)")
    ap.add_argument("--a2f", action="store_true", help="the a2f member is in the main pack (base uses the teacher)")
    ap.add_argument("--a2f-late", default="", help="extension file with a2f members pushed mid-job (a2f knob run)")
    ap.add_argument("--late", default="", help="enc:file,... encoders pushed mid-job (enc:file:la)")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--batches", default="base,mel,noev,regr,ctx45|base2,ctx180,a2f,ENC")
    ap.add_argument("--runtime", default=None, help="json runtime dict override (e.g. the seam fix)")
    ap.add_argument("--audio-meta", default=None, help="META.json of the base encoder (copied into every bundle)")
    a = ap.parse_args()
    rt = json.loads(a.runtime) if a.runtime else RUNTIME
    common = {"pack": a.pack, "pack_on_cpu": True, "noise_profile": "noise_profile.json", "model": {"qknorm": True},
              "amp": "none", "spike_factor": 8, "blowup": 1.3, "batch": 64, "seq": 150, "chunks": 8, "lr": 4e-4,
              "max_minutes": a.minutes, "steps": 1000000, "eval_every": 1000, "log_every": 100, "val_batches": 4,
              "ckpt_seconds": 110, "warmup": 500, "end_margin_s": 150, "style_procs": 4,
              "aug": {"self_noise": 0.6, "p_self_drop": 0.25, "p_null": 0.25}, "guidance": 2.0,
              "audio_in": "enc_" + a.base_enc, "enc_shift": a.la + 2, "events": True, "event_tol": 1,
              "loss": {"a2f": 0.5 if a.a2f else 0.0, "event": 0.3}, "runtime": rt}
    if a.audio_meta and os.path.exists(a.audio_meta):
        common["audio_meta"] = json.load(open(a.audio_meta))
    runs = {
        "base": {},
        "base2": {},
        "mel": {"audio_in": "mel", "enc_shift": 0, "audio_meta": None},
        "noev": {"events": False, "aug": {"event_weight": 3.0}},
        "regr": {"head": "regression"},
        "ctx45": {"model": {"window": 45}},
        "ctx180": {"model": {"window": 180}, "seq": 240},
    }
    if a.a2f:
        runs["a2f"] = {"loss": {"a2f": 0.0}}                         # knob = teacher off
    elif a.a2f_late:
        runs["a2f"] = {"loss": {"a2f": 0.5}, "pack_extra": [a.a2f_late]}   # knob = teacher on (data pushed later)
    late = []
    for item in [x for x in a.late.split(",") if x]:
        enc, fn, la = (item.split(":") + [None])[:3]
        name = "enc_" + enc
        runs[name] = {"audio_in": name, "enc_shift": (int(la) + 2) if la not in (None, "auto") else "auto",
                      "pack_extra": [fn], "audio_meta": None}       # collect adds this encoder's META
        late.append(name)
    batches = []
    for b in a.batches.split("|"):
        names = []
        for n in b.split(","):
            if n == "ENC":
                names += late
            elif n in runs:
                names.append(n)
        if names:
            batches.append(names)
    used = {n for b in batches for n in b}
    runs = {k: v for k, v in runs.items() if k in used}
    mc = {"common": common, "runs": runs, "batches": batches, "min_batch_minutes": 12, "max_wait_minutes": 4}
    with open(a.out, "w") as f:
        json.dump(mc, f, indent=1)
    print(json.dumps({"batches": batches, "base": {k: common[k] for k in ("audio_in", "enc_shift", "loss")}}))


if __name__ == "__main__":
    main()
