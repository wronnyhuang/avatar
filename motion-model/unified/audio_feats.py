"""unified/audio_feats.py - live speech-encoder features from RAW 16 kHz PCM for UnifiedPlanner (phase 2).

Speech-encoder bundles (config["audio"]["in"] = "enc_<name>") whose config["audio"]["encoder"] holds that encoder's
META (the audio pipeline's p2/feats/<encoder>/META.json, copied verbatim into the bundle at collect time) can be fed
raw audio instead of precomputed features: UnifiedPlanner.step(..., user_pcm=, self_pcm_ahead=). PcmFeatures runs
two streaming encoders from unified.speech_enc (the user's audio, the avatar's own audio) and turns their rows into
the per-step user_feat / self_feat_ahead the planner already consumes - the same numbers the offline extraction
wrote into the training pack (the streaming class and the offline extractor share unified/speech_enc.py; parity is
checked in unified/test_audio_feats.py).

Framing (identical to canon1 audio42, unified/speech_enc.py and the training pack):
  * absolute 30 fps frame k = 16 kHz samples [frame_end_sample(k-1), frame_end_sample(k)), frame_end_sample(k) =
    round((k+1)*16000/30) (533 or 534 samples; frame_samples(k) gives the slice). Frame 0 = absolute sample 0 = the
    first sample after reset(): the planner's step t is absolute frame t of both audio streams.
  * encoder step m (every 100 ms on the absolute grid) runs as soon as audio up to sample 1600*m =
    frame_end_sample(3m-1) has been pushed and yields the rows of frames newest-2..newest, newest = 3m-1-la
    (la = the encoder's lookahead in frames). So frame k's feature exists la .. la+2 frames after k ends.
  * user stream: step t pushes the user's frame t; the newest frame whose feature is guaranteed to exist is t-la-2,
    so the bundle's user_shift s must be >= la+2; user_feat(t) = feature of frame t-s (None while t < s).
  * self stream: step t pushes the AVATAR's frame t+P, P = self_pcm_lookahead = L + la + 2 (L = the bundle's
    self-audio lookahead; +2 = the worst case of the 3-frame step grid); the avatar's frames 0..P-1 are silence (never
    passed); self_feat_ahead(t) = feature of frame t+L.
  * None PCM = digital silence (zeros) for that frame. Input: 1-D float (in [-1, 1]) or int16 (scaled by 1/32768),
    exactly the frame's sample count (strict: the absolute sample position is what aligns the features).
  * Cost: with la = 1 and L = 6 (P = 9) both encoders run their 100 ms step on frames t = 3m-1 (the self stream also
    at t = 0, its silent prefix), the planner re-plans on t = 3m. Live processes should start with
    OMP_WAIT_POLICY=PASSIVE in the environment (before torch loads): torch's spinning OpenMP workers and the ORT
    planner's threads otherwise fight on a busy CPU (A/B on a shared host, load ~170/88 cores, 2 threads: encoder step
    95-117 -> 27-36 ms p50, whole frame 75-90 -> 39-50 ms mean).

Encoder spec (config["audio"]["encoder"], the META; extra keys are ignored) - encoder_spec() normalizes:
  encoder | enc | name         speech_enc.ENCODERS key (whisper_tiny / whisper_base / wav2vec2_base / hubert_base)
  layer                        hidden-state layer (0 = transformer input)
  mode | streaming | stream_mode   "sliding*" / "window*" (StreamingSpeechEncoder) | "kv*" / "block_causal" (the
                               KV-cache streamer, speech_enc.KVStreamer = KV_CLASS); default sliding
  lookahead_frames | lookahead (whole frames) | lookahead_ms (-> ms*30/1000, must be whole)   la; default 0 in kv mode
  window_s | window (s) | window_ms                     sliding mode's window
  pca_path | pca_file | pca (str, or {"path"|"file": str, "dims"})   whitened train PCA (speech_enc.Projection npz);
                               relative = against the bundle dir (then the META's own "dir"/"feats_dir"); absent/null
                               = no PCA
  feat_dim | out_dim | pca_dims | pca.dims | dims   output dims (informational; the encoder's actual output dims are
                               checked against the bundle's feat_dim)
  models_dir                   encoder weights root (default speech_enc.MODELS_DIR)
"""
from __future__ import annotations

import copy
import inspect
import os
import time

import numpy as np

from canon1.spec import frame_end_sample

SR = 16000
FPS = 30
STEP_FRAMES = 3
SPM = SR * STEP_FRAMES // FPS            # 1600 samples per encoder step
KV_CLASS = "KVStreamer"                  # speech_enc's KV-cache streaming class (mode "kv")
MODE_PREFIX = (("sliding", "sliding"), ("window", "sliding"), ("win", "sliding"), ("kv", "kv"),
               ("block_causal", "kv"))        # META mode value (lower-case, '-' -> '_') prefix -> mode


def frame_samples(k):
    """(start, stop) absolute 16 kHz sample range of 30 fps frame k."""
    return frame_end_sample(k - 1), frame_end_sample(k)


def pcm_frame(x16, k, base=0):
    """frame k's samples from a 16 kHz signal x16 whose x16[i] is absolute sample base + i (zeros outside)."""
    a, b = frame_samples(k)
    out = np.zeros(b - a, np.float32)
    lo, hi = max(a, base), min(b, base + len(x16))
    if hi > lo:
        out[lo - a: hi - a] = x16[lo - base: hi - base]
    return out


def _first(d, *keys):
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def _resolve(path, bases):
    if os.path.isabs(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"encoder PCA {path} not found")
        return path
    for b in bases:
        if b and os.path.exists(os.path.join(b, path)):
            return os.path.join(b, path)
    raise FileNotFoundError(f"encoder PCA {path!r} not found relative to {[b for b in bases if b]}")


def encoder_spec(meta, bundle_dir=None):
    """the bundle's config["audio"]["encoder"] (an encoder META) -> {name, layer, mode, lookahead (frames), window_s,
    pca (abs path | None), dims (int | None), models_dir (str | None)}."""
    m = dict(meta or {})
    name = _first(m, "encoder", "enc", "name")
    if not name:
        raise ValueError(f"encoder spec has no encoder name: {sorted(m)}")
    mode_raw = str(_first(m, "mode", "streaming", "stream_mode") or "sliding").lower().replace("-", "_")
    mode = next((v for k, v in MODE_PREFIX if mode_raw.startswith(k)), None)
    if mode is None:
        raise ValueError(f"unknown encoder streaming mode {mode_raw!r} (want sliding/window or kv)")
    la = _first(m, "lookahead_frames", "lookahead")
    if la is not None and float(la) != int(float(la)):
        raise ValueError(f"lookahead {la} must be whole frames (use lookahead_ms for milliseconds)")
    if la is None and m.get("lookahead_ms") is not None:
        f = float(m["lookahead_ms"]) * FPS / 1000.0
        if abs(f - round(f)) > 0.05:
            raise ValueError(f"lookahead_ms {m['lookahead_ms']} is not a whole number of 30 fps frames")
        la = int(round(f))
    if la is None:
        if mode != "kv":
            raise ValueError("sliding-window encoder spec needs lookahead (frames) or lookahead_ms")
        la = 0
    win = _first(m, "window_s", "window")
    if win is None and m.get("window_ms") is not None:
        win = float(m["window_ms"]) / 1000.0
    if mode == "sliding" and win is None:
        raise ValueError("sliding-window encoder spec needs window_s")
    pca = None
    for key in ("pca_path", "pca_file", "pca"):
        v = m.get(key)
        if isinstance(v, dict):
            v = v.get("path") or v.get("file")
        if isinstance(v, str) and v:
            pca = v
            break
    if pca is not None:
        pca = _resolve(pca, [bundle_dir, m.get("dir"), m.get("feats_dir")])
    dims = _first(m, "feat_dim", "out_dim", "pca_dims")
    if dims is None and isinstance(m.get("pca"), dict):
        dims = m["pca"].get("dims")
    if dims is None:
        dims = m.get("dims")
    if m.get("layer") is None:
        raise ValueError("encoder spec needs layer")
    return dict(name=str(name), layer=int(m["layer"]), mode=mode, lookahead=int(float(la)),
                window_s=None if win is None else float(win), pca=pca,
                dims=None if dims is None else int(dims), models_dir=m.get("models_dir"))


def _speech_enc():
    from unified import speech_enc
    return speech_enc


def build_encoder(sp, threads=None):
    """one streaming encoder object for a normalized spec: speech_enc.StreamingSpeechEncoder (sliding) or the KV
    class (mode "kv"; KV_CLASS - the one-line switch), constructed with the spec keys its signature accepts."""
    E = _speech_enc()
    cls = getattr(E, KV_CLASS) if sp["mode"] == "kv" else E.StreamingSpeechEncoder
    kw = dict(name=sp["name"], layer=sp["layer"], window_s=sp["window_s"], lookahead=sp["lookahead"],
              pca_path=sp["pca"], threads=threads)
    if sp.get("models_dir"):
        kw["models_dir"] = sp["models_dir"]
    params = inspect.signature(cls).parameters
    if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
        kw = {k: v for k, v in kw.items() if k in params and v is not None}
    return cls(**kw)


def _sliding(enc):
    return type(enc).__name__ == "StreamingSpeechEncoder"


def _clone_encoder(enc, sp, threads):
    """a second stream on the same frozen weights. A never-pushed StreamingSpeechEncoder's state (sample buffer, step
    counter) is only ever reassigned, so a shallow copy shares just the model; other classes: clone() if they have
    one, else a new instance."""
    if hasattr(enc, "clone"):
        return enc.clone()
    if _sliding(enc):
        return copy.copy(enc)
    return build_encoder(sp, threads)


def _out_dim(enc):
    pr = getattr(enc, "proj", None)
    if pr is not None and getattr(pr, "comps", None) is not None:
        return int(pr.comps.shape[0])
    md = getattr(enc, "model", None)
    return int(md.dim) if md is not None and hasattr(md, "dim") else None


class FeatStream:
    """one live audio stream through one streaming encoder: push samples (absolute positions from sample 0), run every
    encoder step that became due, keep the produced rows by absolute frame.
    Encoder API (unified.speech_enc): push(x16); step() -> (frames, (n, De) feats) for the next 100 ms step once its
    audio is complete, else None. The older step(newest) -> (3, De) form is driven on the absolute grid here (sliding:
    steps whose frames are all < min_frame are skipped; KV: every step runs, its cache needs them)."""

    def __init__(self, enc, sp, threads=None, min_frame=0, keep=256):
        self.enc, self.sp, self.threads = enc, sp, threads
        self.la, self.kv = int(sp["lookahead"]), sp["mode"] == "kv"
        self.min_frame, self.keep = int(min_frame), int(keep)
        self.pull = len(inspect.signature(enc.step).parameters) == 0      # step() schedules itself
        self._pristine = dict(enc.__dict__) if (_sliding(enc) and not hasattr(enc, "reset")) else None
        self.n = 0
        self.reset()

    def reset(self):
        if self.n > 0:                 # the encoder has state
            if hasattr(self.enc, "reset"):
                self.enc.reset()
            elif self._pristine is not None:
                self.enc.__dict__.clear()
                self.enc.__dict__.update(self._pristine)
            else:
                self.enc = build_encoder(self.sp, self.threads)
        self.n = 0                     # samples pushed = absolute index of the next sample
        self.m = 1                     # (step(newest) form) next encoder step, ends at sample SPM * m
        self.rows = {}
        self.newest = -1               # newest frame with a row
        self.steps = 0

    def push(self, x):
        x = np.asarray(x, np.float32)
        if len(x):
            self.enc.push(x)
            self.n += len(x)

    def _put(self, frames, feats):
        feats = np.asarray(feats, np.float32)
        for k, f in zip(frames, feats):
            if k >= 0:
                self.rows[int(k)] = f
                self.newest = max(self.newest, int(k))
        self.steps += 1

    def run_due(self):
        """run every due encoder step; returns the encoder wall ms (None when no step ran)."""
        ms = None
        if self.pull:
            while True:
                t0 = time.perf_counter()
                o = self.enc.step()
                if o is None:
                    break
                ms = (ms or 0.0) + (time.perf_counter() - t0) * 1000.0
                self._put(*o)
        else:
            while SPM * self.m <= self.n:
                newest = STEP_FRAMES * self.m - 1 - self.la
                if self.kv or newest >= self.min_frame:
                    t0 = time.perf_counter()
                    out = self.enc.step(newest)
                    ms = (ms or 0.0) + (time.perf_counter() - t0) * 1000.0
                    self._put(range(newest - STEP_FRAMES + 1, newest + 1), out)
                self.m += 1
        if len(self.rows) > self.keep + 8:
            lo = self.newest - self.keep
            self.rows = {k: v for k, v in self.rows.items() if k > lo}
        return ms

    def get(self, k):
        return self.rows.get(int(k))


class PcmFeatures:
    """The planner's PCM front end: two streaming encoders (user, self) -> per-step user_feat / self_feat_ahead."""

    def __init__(self, meta, bundle_dir=None, user_shift=0, self_lookahead=6, feat_dim=None, threads=2):
        self.spec = sp = encoder_spec(meta, bundle_dir)
        self.la = la = sp["lookahead"]
        self.user_shift, self.L = int(user_shift), int(self_lookahead)
        if self.user_shift < la + 2:
            raise ValueError(f"PCM mode: the bundle's user_shift {self.user_shift} < encoder lookahead {la} + 2 "
                             f"(frame t - user_shift has no feature yet at step t)")
        self.self_pcm_lookahead = self.L + la + 2
        enc_u = build_encoder(sp, threads)
        enc_s = _clone_encoder(enc_u, sp, threads)
        De = _out_dim(enc_u)
        if feat_dim is not None and De is not None and int(feat_dim) != De:
            raise ValueError(f"PCM mode: encoder output dims {De} != the bundle's feat_dim {feat_dim} "
                             f"(spec dims {sp['dims']}, pca {sp['pca']})")
        self.feat_dim = De
        self.user = FeatStream(enc_u, sp, threads, min_frame=0)
        self.self_ = FeatStream(enc_s, sp, threads, min_frame=self.L)
        self.reset()

    def reset(self):
        self.user.reset()
        self.self_.reset()
        self.t = 0

    @staticmethod
    def _pcm(x, k):
        a, b = frame_samples(k)
        if x is None:
            return np.zeros(b - a, np.float32)
        x = np.asarray(x)
        if x.dtype.kind in "iu":
            x = x.astype(np.float32) / 32768.0
        x = np.ascontiguousarray(x, np.float32).reshape(-1)
        if len(x) != b - a:
            raise ValueError(f"PCM frame {k} must have {b - a} samples (16 kHz [{a}, {b})), got {len(x)}")
        return x

    def step(self, t, user_pcm=None, self_pcm_ahead=None):
        """step t: push the user's frame t and the avatar's frame t+P, run the due encoder steps.
        -> (user_feat (De,) | None, self_feat_ahead (De,), enc_ms {"user": ms|None, "self": ms|None} | None)."""
        if t != self.t:
            raise RuntimeError(f"PcmFeatures stepped at {t}, expected {self.t} (reset() both together)")
        P = self.self_pcm_lookahead
        xu, xs = self._pcm(user_pcm, t), self._pcm(self_pcm_ahead, t + P)       # validate both before pushing
        if t == 0:                                       # the avatar's frames 0..P-1 = silence
            self.self_.push(np.zeros(frame_end_sample(P - 1), np.float32))
        self.user.push(xu)
        self.self_.push(xs)
        mu, ms = self.user.run_due(), self.self_.run_due()
        s = self.user_shift
        uf = self.user.get(t - s) if t >= s else None
        if t >= s and uf is None:
            raise RuntimeError(f"PCM user feature of frame {t - s} missing at step {t}")
        sf = self.self_.get(t + self.L)
        if sf is None:
            raise RuntimeError(f"PCM self feature of frame {t + self.L} missing at step {t}")
        self.t = t + 1
        return uf, sf, (None if mu is None and ms is None else {"user": mu, "self": ms})
