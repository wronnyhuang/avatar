"""Tests for PCM mode (raw 16 kHz audio -> the planner runs the speech encoder itself; unified/audio_feats.py).

    python -I -m pytest -q unified/test_audio_feats.py            # PARITY_FRAMES=<n> for a longer parity run

Tests that need unified.speech_enc's dependencies or the encoder weights ($SPEECH_ENC_MODELS) skip when they are
missing; the parity test also needs the extracted features under $MM_WORK_DIR/p2.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from canon1.spec import frame_end_sample  # noqa: E402
from unified.audio_feats import PcmFeatures, encoder_spec, frame_samples, pcm_frame  # noqa: E402
from unified.runtime import REST_POSE, UnifiedPlanner  # noqa: E402
from unified.test_runtime import make_p2_bundle  # noqa: E402

P2 = os.path.join(os.environ.get("MM_WORK_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                              "work")), "p2")
STREAMS = os.path.join(P2, "streams_abl.jsonl")
PARITY_STREAM = "V00_S1462_I00000297_p0_f000025"          # val stream, frame0 = 25 (cheap to replay from sample 0)
TINY = dict(encoder="whisper_tiny", layer=2, mode="sliding", window_s=1.0, lookahead=1, pca="enc_pca.npz", dims=64)


def speech_enc_or_skip():
    """unified.speech_enc, or skip when its dependencies are missing."""
    try:
        from unified import speech_enc
    except ImportError:
        pytest.skip("unified.speech_enc not available")
    if not os.path.isdir(os.path.join(speech_enc.MODELS_DIR, "whisper_tiny")):
        pytest.skip("whisper_tiny weights missing")
    return speech_enc


def write_pca(path, D=384, k=64, seed=0):
    """a random whitened 'PCA' in speech_enc.Projection's format (mean (D,), comps (k,D), scale (k,))."""
    r = np.random.default_rng(seed)
    comps = np.linalg.qr(r.standard_normal((D, k)))[0].T
    np.savez(path, mean=(0.1 * r.standard_normal(D)).astype(np.float32), comps=comps.astype(np.float32),
             scale=np.full(k, 2.0, np.float32))
    return path


def speechy(n, seed=0):
    """n samples of 16 kHz pseudo speech: syllable-rate bursts of a gliding harmonic voice + a little noise."""
    r = np.random.default_rng(seed)
    t = np.arange(n) / 16000.0
    f0 = 110 + 40 * np.sin(2 * np.pi * 0.7 * t + seed)
    ph = 2 * np.pi * np.cumsum(f0) / 16000.0
    v = sum(np.sin(h * ph) / h for h in range(1, 9))
    env = np.clip(np.sin(2 * np.pi * 2.5 * t + seed), 0, None) ** 0.5
    return (0.15 * env * v + 0.003 * r.standard_normal(n)).astype(np.float32)


def load16(wav, n):
    """the first n samples of a wav at 16 kHz (mono mean; resample_poly like the feature extractor's loader)."""
    import math

    import soundfile as sf
    from scipy.signal import resample_poly
    info = sf.info(wav)
    sr = info.samplerate
    stop = min(info.frames, int(math.ceil((n + 2 * 16000) * sr / 16000)))
    x, _ = sf.read(wav, start=0, stop=stop, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 16000:
        g = math.gcd(sr, 16000)
        x = resample_poly(x.astype(np.float64), 16000 // g, sr // g).astype(np.float32)
    out = np.zeros(n, np.float32)
    out[:min(n, len(x))] = x[:n]
    return out


# ------------------------------------------------------------------------------------------------ spec / framing
def test_encoder_spec_variants(tmp_path):
    b = str(tmp_path)
    write_pca(os.path.join(b, "enc_pca.npz"))
    s = encoder_spec(TINY, b)
    assert s["name"] == "whisper_tiny" and s["layer"] == 2 and s["mode"] == "sliding" and s["lookahead"] == 1
    assert s["window_s"] == 1.0 and s["pca"] == os.path.join(b, "enc_pca.npz") and s["dims"] == 64
    s = encoder_spec(dict(enc="whisper_base", layer=3, streaming="window", window_ms=2000, lookahead_ms=33.333,
                          pca={"file": os.path.join(b, "enc_pca.npz"), "dims": 64}), None)
    assert s["mode"] == "sliding" and s["window_s"] == 2.0 and s["lookahead"] == 1 and s["dims"] == 64
    assert os.path.isabs(s["pca"])
    s = encoder_spec(dict(name="hubert_base", layer=6, mode="kv", pca=None))
    assert s["mode"] == "kv" and s["lookahead"] == 0 and s["pca"] is None and s["window_s"] is None
    assert encoder_spec(dict(TINY, mode="KV-cache"), b)["mode"] == "kv"
    assert encoder_spec(dict(TINY, mode="sliding_window", lookahead=2.0), b)["lookahead"] == 2
    s = encoder_spec(dict(encoder="whisper_tiny", layer=2, mode="sliding", window_s=1, lookahead_frames=2,
                          pca_path="enc_pca.npz", dir=b), "/nonexistent")       # falls back to the META's dir
    assert s["lookahead"] == 2 and s["pca"] == os.path.join(b, "enc_pca.npz")
    for bad in (dict(TINY, pca="missing.npz"), dict(TINY, mode="magic"), dict(TINY, lookahead=None),
                dict(TINY, lookahead_ms=50.0, lookahead=None), dict(TINY, lookahead=1.5)):
        with pytest.raises((ValueError, FileNotFoundError)):
            encoder_spec(bad, b)
    # frame k = [frame_end_sample(k-1), frame_end_sample(k)); frame 0 starts at sample 0
    assert frame_samples(0) == (0, 533) and frame_samples(2) == (1067, 1600) and frame_samples(3) == (1600, 2133)
    x = np.arange(5000, dtype=np.float32)
    np.testing.assert_array_equal(pcm_frame(x, 1), x[533:1067])
    np.testing.assert_array_equal(pcm_frame(x, 1, base=600)[:67], 0)            # before the signal = zeros


# ------------------------------------------------------------------------------------------------ synthetic bundle
def _pcm_bundle(tmp_path, user_shift=3, encoder=TINY, name="b"):
    b = str(tmp_path / name)
    os.makedirs(b, exist_ok=True)
    write_pca(os.path.join(b, "enc_pca.npz"))
    return make_p2_bundle(b, De=64, k=8, user_shift=user_shift, encoder=encoder)


def test_pcm_synthetic_bundle(tmp_path):
    speech_enc_or_skip()
    b = _pcm_bundle(tmp_path)
    p = UnifiedPlanner(b, backend="torch", events="off", temperature=0.0, threads=2)
    L, P = p.self_audio_lookahead, p.self_pcm_lookahead
    assert p.pcm_mode and L == 6 and P == L + 1 + 2 and p.runtime["pcm"]["name"] == "whisper_tiny"
    N = 60
    xu, xs = speechy(frame_end_sample(N), 1), speechy(frame_end_sample(N + P), 2)

    def run(speech):
        p.reset()
        out = []
        for t in range(N):
            o = p.step(REST_POSE, np.ones(9), np.zeros(42), None,
                       user_pcm=pcm_frame(xu, t) if speech else None,
                       self_pcm_ahead=pcm_frame(xs, t + P) if speech else None)
            out.append((o, p.last_feats))
        return out

    a, sil, a2 = run(True), run(False), run(True)
    mot = lambda r: np.stack([o["motion"] for o, _ in r])      # noqa: E731
    assert np.abs(mot(a) - mot(sil)).max() > 1e-4, "speech PCM must change the output vs silence PCM"
    np.testing.assert_array_equal(mot(a), mot(a2))              # reset() resets the encoders
    # cadence: the user's encoder step m runs at t = 3m-1; the avatar's at t = 0 (its silent prefix) and t+P = 3m-1
    ran = lambda r, who: [t for t, (o, _) in enumerate(r) if o["enc_ms"] and o["enc_ms"][who] is not None]  # noqa
    assert ran(a, "user") == list(range(2, N, 3)), ran(a, "user")
    assert ran(a, "self") == [0] + [t for t in range(1, N) if (t + P) % 3 == 2], ran(a, "self")
    assert all(o["enc_ms"] is None for t, (o, _) in enumerate(a) if t not in ran(a, "user") + ran(a, "self"))
    assert all(f["user"] is None for _, f in a[:3]) and all(f["user"].shape == (64,) for _, f in a[3:])
    assert all(f["self"].shape == (64,) for _, f in a)
    assert np.abs(a[30][1]["user"] - sil[30][1]["user"]).max() > 1e-3
    # explicit features win over the derived ones; the encoders keep streaming
    p.reset()
    f = np.ones(64, np.float32)
    p.step(None, None, None, None, user_feat=f, user_pcm=pcm_frame(xu, 0))
    assert p.last_feats["user"] is f and p.last_feats["self"] is not None
    with pytest.raises(ValueError):                              # not the frame's sample count: nothing changes
        p.step(None, None, None, None, user_pcm=np.zeros(533, np.float32), self_pcm_ahead=np.zeros(500, np.float32))
    assert p.t == 1 and p._pcm.t == 1 and p._pcm.user.n == 533 and p.hist.user_motion.n == 1
    # int16 PCM == float PCM / 32768
    p.reset()
    q = np.round(xu[:frame_end_sample(5)] * 32767).astype(np.int16)
    for t in range(5):
        p.step(None, None, None, None, user_pcm=pcm_frame(q, t).astype(np.int16))
    u16 = p._pcm.user.get(1)
    p.reset()
    for t in range(5):
        p.step(None, None, None, None, user_pcm=pcm_frame(q.astype(np.float32) / 32768.0, t))
    np.testing.assert_allclose(u16, p._pcm.user.get(1), atol=1e-6)


def test_pcm_off_mel_and_bad_shift(tmp_path):
    speech_enc_or_skip()
    b = _pcm_bundle(tmp_path)
    p = UnifiedPlanner(b, backend="torch", events="off", pcm=False)        # precomputed features only
    assert not p.pcm_mode and p.self_pcm_lookahead is None and p.runtime["pcm"] is None
    o = p.step(None, None, None, None, user_feat=np.zeros(64, np.float32))
    assert o["enc_ms"] is None
    with pytest.raises(ValueError):
        p.step(None, None, None, None, user_pcm=np.zeros(533, np.float32))
    nb = make_p2_bundle(str(tmp_path / "noenc"), De=64, k=8, user_shift=3)  # enc bundle without an encoder spec
    p = UnifiedPlanner(nb, backend="torch", events="off")
    assert not p.pcm_mode
    with pytest.raises(ValueError):
        UnifiedPlanner(nb, backend="torch", events="off", pcm=True)
    mb = make_p2_bundle(str(tmp_path / "mel"), audio="mel", n_events=0)    # mel bundles ignore PCM
    p = UnifiedPlanner(mb, backend="torch", pcm=True)
    assert not p.pcm_mode and p.feat_dim == 0
    o = p.step(None, None, None, None, user_pcm=np.zeros(533, np.float32), self_pcm_ahead=np.zeros(7))
    assert o["enc_ms"] is None and np.isfinite(o["motion"]).all()
    with pytest.raises(ValueError, match="user_shift"):                     # la 1 needs user_shift >= 3
        UnifiedPlanner(_pcm_bundle(tmp_path, user_shift=2, name="s2"), backend="torch", events="off")
    with pytest.raises(ValueError, match="dims"):                           # PCA dims != the bundle's feat_dim
        UnifiedPlanner(_pcm_bundle(tmp_path, encoder=dict(TINY, pca=None, dims=None), name="nopca"),
                       backend="torch", events="off")


# ------------------------------------------------------------------------------------------------ parity (real audio)
def _parity_meta():
    """the encoder META the pack was extracted with (feats/whisper_tiny/META.json once the extractor writes it;
    until then their running extraction's settings: sliding, layer 2, 1 s window, lookahead 1, pca.npz)."""
    fd = os.path.join(P2, "feats", "whisper_tiny")
    mp = os.path.join(fd, "META.json")
    if os.path.exists(mp):
        with open(mp) as f:
            return json.load(f), fd, True
    meta = dict(encoder="whisper_tiny", layer=2, mode="sliding", window_s=1.0, lookahead=1)
    if os.path.exists(os.path.join(fd, "pca.npz")):
        meta["pca"] = os.path.join(fd, "pca.npz")
    return meta, fd, False


def test_pcm_parity_real_stream(tmp_path):
    """the features the planner derives from raw PCM == the offline extraction of the same audio
    (frame t - user_shift for the user, frame t + L for the avatar)."""
    E = speech_enc_or_skip()
    if not os.path.exists(STREAMS):
        pytest.skip("streams_abl.jsonl missing")
    row = next(json.loads(l) for l in open(STREAMS) if PARITY_STREAM in l)
    if not os.path.exists(row["audio_self"]):
        pytest.skip("Seamless audio missing")
    meta, fd, has_meta = _parity_meta()
    sp = encoder_spec(meta, fd)
    if sp["mode"] != "sliding" and not hasattr(E, "KVStreamer"):
        pytest.skip("META asks for the KV streamer, not in speech_enc yet")
    b = str(tmp_path / "b")
    os.makedirs(b)
    De = (np.load(sp["pca"])["comps"].shape[0] if sp["pca"] else E.WindowEncoder(sp["name"], 0).dim)
    la = sp["lookahead"]
    s = la + 3                                               # one frame more than the minimum
    make_p2_bundle(b, De=De, k=8, user_shift=s, encoder=dict(meta, pca_path=sp["pca"], pca_file=None, pca=None))
    p = UnifiedPlanner(b, backend="torch", events="off", threads=2)
    L, P = p.self_audio_lookahead, p.self_pcm_lookahead
    f0, n = int(row["frame0"]), int(os.environ.get("PARITY_FRAMES", 240))
    T = f0 + n                                               # steps (absolute frames 0..T-1)
    x16 = load16(row["audio_self"], frame_end_sample(T + P) + 16000)
    uf, sf = {}, {}
    for t in range(T):
        p.step(None, None, None, None, user_pcm=pcm_frame(x16, t), self_pcm_ahead=pcm_frame(x16, t + P))
        uf[t], sf[t] = p.last_feats["user"], p.last_feats["self"]
    # offline: the extraction over absolute frames 0 .. T+L of the same signal; the avatar's stream
    # = the same signal with its never-passed frames 0..P-1 silent
    m = E.WindowEncoder(sp["name"], max_layer=sp["layer"])
    proj = E.Projection(sp["pca"]) if sp["pca"] else None

    def offline(x):
        if sp["mode"] == "sliding":
            o = E.extract_stream(m, x, T + L + 1, 0, sp["window_s"], la, [sp["layer"]], base=0)[sp["layer"]]
        else:
            o = E.kv_extract_stream(m, x, T + L + 1, 0, [sp["layer"]], 0)[sp["layer"]]
        return proj(o) if proj is not None else o
    off = offline(x16)
    xs = x16.copy()
    xs[:frame_end_sample(P - 1)] = 0
    off_s = offline(xs)
    assert all(uf[t] is None for t in range(s))
    du = max(float(np.abs(uf[t] - off[t - s]).max()) for t in range(s, T))
    ds = max(float(np.abs(sf[t] - off_s[t + L]).max()) for t in range(T))
    scale = float(np.abs(off).max())
    print(f"\nPARITY {PARITY_STREAM} frames 0..{T - 1} ({sp['name']} L{sp['layer']} {sp['mode']} la={la}, De={De}, "
          f"META={'yes' if has_meta else 'no'}): max|diff| user {du:.3g} self {ds:.3g} (max|feat| {scale:.3g})")
    assert du < 1e-4 * max(1.0, scale) and ds < 1e-4 * max(1.0, scale), (du, ds, scale)
    # the training pack's rows of this stream (float16; absolute frames f0..f0+899) - informational tolerance
    pk = os.path.join(fd, PARITY_STREAM + ".npy")
    if os.path.exists(pk) and sp["pca"]:
        pack = np.load(pk).astype(np.float32)
        got = np.stack([uf[t + s] for t in range(f0, T - s)])               # user feature of frame t = uf[t + s]
        dp = float(np.abs(got - pack[: T - s - f0]).max())
        rel = dp / float(np.abs(pack).max())
        print(f"PACK {pk}: max|runtime - pack fp16| {dp:.3g} (rel {rel:.2e})")
        assert rel < 5e-3, (dp, rel)
