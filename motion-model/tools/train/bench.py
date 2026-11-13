"""Latency bench of the runtime on the host CPU.

    python -I tools/train/bench.py --bundle <dir> [--threads 2] [--frames 900] [--steps 6] [--replan 3]
           [--rt '{"ensemble": {"plans": 3}, "prefix": [2, 3, 4], "box": 3}']    (extra UnifiedPlanner kwargs)
Prints plan_ms p50/p90/p99/max, amortized ms per 30 Hz frame (all step() time), params; for bundles that feed the
backbone every frame (event head) also the per-frame on_frame cost (one token step + the event head) p50/p99/max.
Speech-encoder (enc_*) bundles get random raw features of the bundle's feat_dim.

PCM mode (phase 2: the planner runs the speech encoders itself, two streams):
    python -I tools/train/bench.py --pcm --bundle <enc bundle with config["audio"]["encoder"]> [--backend ort]
    python -I tools/train/bench.py --pcm --like <bundle> --encoder '<encoder spec json>' [--events 3] [--backend ort]
The second form first writes a random-init SYNTHETIC enc bundle (--synth-dir, default a temp dir) with --like's model
size / norm / runtime, k = min(De, 64) PCA dims, --events event classes and the given encoder spec (feat_dim = the
encoder's output dims), exporting the ORT graphs for --backend ort. Both streams get pseudo-speech PCM. Adds:
enc_user / enc_self ms per encoder step (one 100 ms step of one stream) p50/p99/max, enc_ms_per_frame (both streams,
amortized per 30 Hz frame), the frame total (encoders + planner) mean/p99/max, and the load average.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from unified.runtime import REST_POSE, UnifiedPlanner  # noqa: E402


def speechy(n, seed=0):
    """n samples of 16 kHz pseudo speech (syllable-rate bursts of a gliding harmonic voice + noise)."""
    r = np.random.default_rng(seed)
    t = np.arange(n) / 16000.0
    f0 = 110 + 40 * np.sin(2 * np.pi * 0.7 * t + seed)
    ph = 2 * np.pi * np.cumsum(f0) / 16000.0
    v = sum(np.sin(h * ph) / h for h in range(1, 9))
    env = np.clip(np.sin(2 * np.pi * 2.5 * t + seed), 0, None) ** 0.5
    return (0.15 * env * v + 0.003 * r.standard_normal(n)).astype(np.float32)


def make_synth_enc_bundle(out, like, encoder, k_max=64, n_events=3, seed=0, backend="ort", user_shift=None):
    """random-init enc bundle with `like`'s model size / norm / runtime and the encoder spec `encoder` (dict)."""
    import torch
    from unified.audio_feats import _out_dim, build_encoder, encoder_spec
    from unified.model import UnifiedModel
    with open(os.path.join(like, "config.json")) as f:
        lc = json.load(f)
    sp = encoder_spec(encoder, like)
    enc = dict(encoder)
    if sp["pca"]:
        for key in ("pca_path", "pca_file"):
            enc.pop(key, None)
        enc["pca"] = sp["pca"]                       # absolute
    De = _out_dim(build_encoder(sp, 1))
    k = min(De, k_max)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    mc = dict(lc["model"], aud_dim=k + 2, n_events=n_events, head_kind="flow")
    m = UnifiedModel(mc)
    os.makedirs(out, exist_ok=True)
    torch.save(m.state_dict(), os.path.join(out, "weights.pt"))
    z = np.load(os.path.join(like, "norm.npz"))
    norm = {n: z[n] for n in ("mean", "std", "amean", "astd")}
    norm.update(emean=np.zeros(k, np.float32), estd=np.ones(k, np.float32), esil=np.zeros(k, np.float32))
    if k < De:
        norm.update(epca_mean=np.zeros(De, np.float32),
                    epca_V=np.linalg.qr(rng.standard_normal((De, k)))[0].T.astype(np.float32))
    np.savez(os.path.join(out, "norm.npz"), **norm)
    la = sp["lookahead"]
    cfg = {kk: lc[kk] for kk in ("self_audio_lookahead", "horizon", "prefix", "guidance", "steps", "temperature",
                                 "replan_every", "lip", "runtime") if kk in lc}
    cfg.update(kind="flow", version="bench-synth-enc", model=m.cfg,
               audio={"in": "enc_" + sp["name"], "feat_dim": De, "proj_dim": k,
                      "user_shift": la + 2 if user_shift is None else int(user_shift), "keep_vad": True,
                      "encoder": enc})
    if n_events > 0:
        cfg["events"] = dict(names=[f"ev{i}" for i in range(n_events)], thresholds=[0.5] * n_events,
                             refractory=[10] * n_events, groups=[None] * n_events)
        np.savez(os.path.join(out, "events.npz"), curves=np.zeros((n_events, 24, 189), np.float32),
                 lens=np.full(n_events, 8, np.int64))
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f)
    if backend == "ort":
        from unified.export import export
        export(out)
    return out


def _pct(x, q):
    return round(float(np.percentile(x, q)), 3) if len(x) else None


def run(bundle, threads=2, frames=900, steps=6, replan=3, guidance=1.5, seed=0, backend="torch", pcm=False,
        warm=30, rt=None):
    p = UnifiedPlanner(bundle, threads=threads, seed=seed, replan_every=replan, steps=steps, guidance=guidance,
                       backend=backend, pcm=bool(pcm), **(rt or {}))
    if pcm and not p.pcm_mode:
        raise SystemExit("--pcm needs an enc bundle with config['audio']['encoder']")
    rng = np.random.default_rng(seed)
    De = p.feat_dim
    of_ms = []
    if getattr(p.core, "feed_every_frame", False):
        of = p.core.on_frame

        def timed(hist, t):
            t0 = time.perf_counter()
            r = of(hist, t)
            of_ms.append((time.perf_counter() - t0) * 1000)
            return r
        p.core.on_frame = timed
    if pcm:
        from unified.audio_feats import pcm_frame
        P = p.self_pcm_lookahead
        from canon1.spec import frame_end_sample
        xu, xs = speechy(frame_end_sample(frames), seed + 1), speechy(frame_end_sample(frames + P), seed + 2)
    load0 = os.getloadavg()
    plan, tot, enc_u, enc_s, enc_f = [], [], [], [], []
    for k in range(frames):
        um = REST_POSE + 0.05 * rng.standard_normal(189).astype(np.float32)
        ua = rng.standard_normal(42).astype(np.float32) - 5
        sa = rng.standard_normal(42).astype(np.float32) - 5
        fk = {}
        if pcm:
            fk = dict(user_pcm=pcm_frame(xu, k), self_pcm_ahead=pcm_frame(xs, k + P))
        elif De > 0:
            fk = dict(user_feat=rng.standard_normal(De).astype(np.float32),
                      self_feat_ahead=rng.standard_normal(De).astype(np.float32))
        t0 = time.perf_counter()
        o = p.step(um, np.ones(9), ua, sa, **fk)
        tot.append((time.perf_counter() - t0) * 1000)
        if o["plan_ms"] is not None:
            plan.append(o["plan_ms"])
        em = o.get("enc_ms")
        if k >= warm:
            enc_f.append(0.0 if em is None else (em["user"] or 0.0) + (em["self"] or 0.0))
            if em is not None and em["user"] is not None:
                enc_u.append(em["user"])
            if em is not None and em["self"] is not None:
                enc_s.append(em["self"])
    load1 = os.getloadavg()
    plan = np.array(plan[10:]); tot = np.array(tot[warm:])
    n_params = sum(x.numel() for x in p.core.model.parameters()) if hasattr(p.core, "model") else 0
    r2 = lambda x: round(float(x), 3)  # noqa: E731
    res = dict(threads=threads, steps=p.runtime["steps"], replan_every=replan, frames=frames, params=n_params,
               backend=getattr(p.core, "backend", None), audio_in=p.audio_in, n_events=p.n_events,
               plan_p50=r2(np.percentile(plan, 50)), plan_p90=r2(np.percentile(plan, 90)),
               plan_p99=r2(np.percentile(plan, 99)), plan_max=r2(plan.max()),
               frame_mean=r2(tot.mean()), frame_p99=r2(np.percentile(tot, 99)), frame_max=r2(tot.max()),
               load_avg=[round(load0[0], 1), round(load1[0], 1)], omp_wait_policy=os.environ.get("OMP_WAIT_POLICY"))
    if of_ms:
        of_ms = np.array(of_ms[warm:])
        res.update(on_frame_p50=r2(np.percentile(of_ms, 50)), on_frame_p99=r2(np.percentile(of_ms, 99)),
                   on_frame_max=r2(of_ms.max()))
    if pcm:
        sp = p.runtime["pcm"]
        res.update(encoder=sp["name"], layer=sp["layer"], enc_mode=sp["mode"], window_s=sp["window_s"],
                   enc_lookahead=sp["lookahead"], self_pcm_lookahead=p.self_pcm_lookahead, feat_dim=De,
                   enc_user_p50=_pct(enc_u, 50), enc_user_p99=_pct(enc_u, 99),
                   enc_user_max=r2(max(enc_u)) if enc_u else None,
                   enc_self_p50=_pct(enc_s, 50), enc_self_p99=_pct(enc_s, 99),
                   enc_self_max=r2(max(enc_s)) if enc_s else None,
                   enc_ms_per_frame=r2(np.mean(enc_f)), planner_ms_per_frame=r2(tot.mean() - np.mean(enc_f)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--replan", type=int, default=3)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--backend", default="torch")
    ap.add_argument("--pcm", action="store_true", help="feed raw PCM (the planner runs both speech encoders)")
    ap.add_argument("--like", default=None, help="--pcm: build a synthetic enc bundle with this bundle's model size")
    ap.add_argument("--encoder", default=None, help="--pcm --like: encoder spec (json) for the synthetic bundle")
    ap.add_argument("--events", type=int, default=3, help="--like: event classes of the synthetic bundle")
    ap.add_argument("--synth-dir", default=None, help="--like: where to write the synthetic bundle (kept)")
    ap.add_argument("--rt", default=None, help="json dict of extra UnifiedPlanner kwargs")
    ap.add_argument("--speech-enc", default=None, help="path of a speech_enc.py to use as unified.speech_enc when "
                                                       "testing another copy")
    a = ap.parse_args()
    if a.speech_enc:
        try:
            import unified.speech_enc  # noqa: F401
        except ImportError:
            import importlib.util
            import unified
            spec = importlib.util.spec_from_file_location("unified.speech_enc", a.speech_enc)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["unified.speech_enc"] = unified.speech_enc = mod
            spec.loader.exec_module(mod)
    tmp = None
    bundle = a.bundle
    if a.like:
        if not (a.pcm and a.encoder):
            ap.error("--like needs --pcm and --encoder")
        out = a.synth_dir or (tmp := tempfile.mkdtemp(prefix="bench_enc_"))
        bundle = make_synth_enc_bundle(out, a.like, json.loads(a.encoder), n_events=a.events, backend=a.backend)
    if bundle is None:
        ap.error("--bundle or --like")
    try:
        print(json.dumps(run(bundle, a.threads, a.frames, a.steps, a.replan, a.guidance, backend=a.backend,
                             pcm=a.pcm, rt=json.loads(a.rt) if a.rt else None)))
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
