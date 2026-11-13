"""Frozen pretrained speech encoders as a STREAMING 30 fps audio feature (phase 2, phase-2 plan item 2).

One runtime for training-time extraction and the live worker, so the features are bit-for-bit the same thing:
every STEP (3 frames = 100 ms) the encoder runs on a sliding WINDOW of the last `window_s` seconds of 16 kHz audio
that ENDS `lookahead` frames after the newest frame of the step; each of the step's 3 frames takes the mean of the
encoder frames (50 Hz) whose receptive-field END falls inside that frame's 33.3 ms bin (audio42 framing: frame k
covers [k/30,(k+1)/30) and ends at sample frame_end_sample(k)). A frame therefore sees audio up to its own end +
lookahead (+33 ms for the middle frame of a step, +67 ms for the first) and nothing later.

Encoders (HF weights loaded as weights only, no remote code): whisper_tiny, whisper_base (encoder only; the window is
fed at Whisper positions 0..n-1 without the 30 s padding), wav2vec2_base, hubert_base. Per-window input normalisation
as each model's feature extractor does it (Whisper: log-mel, max-8 clamp, (x+4)/4; wav2vec2/HuBERT: zero mean / unit
variance, std floored at STD_FLOOR so silence stays quiet). Output = hidden state after transformer layer `layer`
(0 = transformer input), optionally reduced by a train-set PCA (whitened) to D dims.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

FPS = 30
SR = 16000
STEP_FRAMES = 3
STD_FLOOR = 3e-3          # wav2vec2/HuBERT waveform std floor (~ -50 dBFS rms)
NORM_STEPS = 10           # wav2vec2/HuBERT: waveform + GroupNorm statistics over the trailing 10 steps (1 s)

ENCODERS = {
    "whisper_tiny": ("whisper", "whisper_tiny"),
    "whisper_base": ("whisper", "whisper_base"),
    "wav2vec2_base": ("w2v", "wav2vec2_base"),
    "hubert_base": ("hubert", "hubert_base"),
}
# encoder weights root (<name>/ per ENCODERS entry): $SPEECH_ENC_MODELS, default $MM_WORK_DIR/p2/models
MODELS_DIR = os.environ.get("SPEECH_ENC_MODELS", os.path.join(
    os.environ.get("MM_WORK_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work")),
    "p2", "models"))


def frame_end_sample(k):
    """same as canon1.spec.frame_end_sample: sample index (exclusive) at which frame k ends."""
    return int(round((k + 1) * SR / FPS))


def _load_state(model, d):
    """weights only: safetensors if present, else torch.load(weights_only=True)."""
    st = os.path.join(d, "model.safetensors")
    if os.path.exists(st):
        from safetensors.torch import load_file
        sd = load_file(st)
    else:
        sd = torch.load(os.path.join(d, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    return sd


class WindowEncoder(torch.nn.Module):
    """Runs one encoder on a batch of equal-length windows; returns hidden states of layers 0..max_layer."""

    def __init__(self, name, max_layer=None, models_dir=MODELS_DIR):
        super().__init__()
        kind, sub = ENCODERS[name]
        d = os.path.join(models_dir, sub)
        self.name, self.kind = name, kind
        cfg_json = json.load(open(os.path.join(d, "config.json")))
        sd = _load_state(None, d)
        if kind == "whisper":
            from transformers import WhisperConfig
            from transformers.models.whisper.modeling_whisper import WhisperEncoder
            cfg = WhisperConfig(**{k: v for k, v in cfg_json.items() if not k.startswith("_")})
            enc = WhisperEncoder(cfg)
            esd = {k[len("model.encoder."):]: v for k, v in sd.items() if k.startswith("model.encoder.")}
            missing, unexpected = enc.load_state_dict(esd, strict=False)
            assert not [m for m in missing if "embed_positions" not in m], missing
            pp = json.load(open(os.path.join(d, "preprocessor_config.json")))
            fb = np.asarray(pp["mel_filters"], np.float32)
            self.register_buffer("mel_fb", torch.tensor(fb if fb.shape[0] == 201 else fb.T))   # (201, 80)
            self.register_buffer("hann", torch.hann_window(400))
            self.n_layers = cfg.encoder_layers
            self.dim = cfg.d_model
        else:
            if kind == "w2v":
                from transformers import Wav2Vec2Config, Wav2Vec2Model
                cfg = Wav2Vec2Config(**{k: v for k, v in cfg_json.items() if not k.startswith("_")})
                enc = Wav2Vec2Model(cfg)
                pre = "wav2vec2."
            else:
                from transformers import HubertConfig, HubertModel
                cfg = HubertConfig(**{k: v for k, v in cfg_json.items() if not k.startswith("_")})
                enc = HubertModel(cfg)
                pre = "hubert." if any(k.startswith("hubert.") for k in sd) else ""
            esd = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)} if pre else dict(sd)
            # old checkpoints store the weight-normed pos conv as weight_g/weight_v
            ren = {}
            for k, v in esd.items():
                k2 = k.replace("pos_conv_embed.conv.weight_g", "pos_conv_embed.conv.parametrizations.weight.original0")
                k2 = k2.replace("pos_conv_embed.conv.weight_v", "pos_conv_embed.conv.parametrizations.weight.original1")
                ren[k2] = v
            missing, unexpected = enc.load_state_dict(ren, strict=False)
            assert not [m for m in missing if "masked_spec_embed" not in m], missing
            self.n_layers = cfg.num_hidden_layers
            self.dim = cfg.hidden_size
        self.enc = enc.eval()
        self.max_layer = self.n_layers if max_layer is None else int(max_layer)
        for p in self.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------ geometry
    def n_out(self, n):
        if self.kind == "whisper":
            return n // 320
        L = n
        for k, s in zip(self.enc.config.conv_kernel, self.enc.config.conv_stride):
            L = (L - k) // s + 1
        return L

    def recept_end(self, n):
        """(T_enc,) receptive-field end (exclusive, in samples from the window start) of each encoder frame."""
        i = np.arange(self.n_out(n))
        return 320 * i + (520 if self.kind == "whisper" else 400)

    # ------------------------------------------------------------------ forward
    @torch.inference_mode()
    def forward(self, x, layers=None):
        """x: (B, N) float32 tensor at 16 kHz. Returns {layer: (B, T_enc, D)} for the requested layers."""
        layers = sorted(set(layers if layers is not None else [self.max_layer]))
        top = max(layers)
        out = {}
        if self.kind == "whisper":
            e = self.enc
            spec = torch.stft(x, 400, 160, window=self.hann, return_complex=True, center=True, pad_mode="reflect")
            mag = spec[..., :-1].abs() ** 2                                   # (B, 201, n//160)
            mel = torch.einsum("bft,fm->bmt", mag, self.mel_fb)
            lg = torch.clamp(mel, min=1e-10).log10()
            lg = torch.maximum(lg, lg.amax(dim=(1, 2), keepdim=True) - 8.0)
            lg = (lg + 4.0) / 4.0
            h = torch.nn.functional.gelu(e.conv1(lg))
            h = torch.nn.functional.gelu(e.conv2(h)).permute(0, 2, 1)
            h = h + e.embed_positions.weight[: h.shape[1]][None]
            if 0 in layers:
                out[0] = h
            for li in range(top):
                h = e.layers[li](h, None)
                if isinstance(h, tuple):
                    h = h[0]
                if li + 1 in layers:
                    out[li + 1] = e.layer_norm(h) if li + 1 == self.n_layers else h
            return out
        e = self.enc
        x = x - x.mean(1, keepdim=True)
        x = x / torch.clamp(x.std(1, keepdim=True), min=STD_FLOOR)
        f = e.feature_extractor(x).transpose(1, 2)
        fp = e.feature_projection(f)
        h = fp[0] if isinstance(fp, tuple) else fp
        en = e.encoder
        h = h + en.pos_conv_embed(h)
        h = en.layer_norm(h)
        if 0 in layers:
            out[0] = h
        for li in range(top):
            h = en.layers[li](h, attention_mask=None)
            if isinstance(h, tuple):
                h = h[0]
            if li + 1 in layers:
                out[li + 1] = h
        return out


    # ------------------------------------------------------------------ wav2vec2 / HuBERT: streaming conv front end
    # The conv feature encoder is LOCAL (receptive field 400 samples, stride 320) except for two whole-input
    # statistics: the waveform zero-mean/unit-variance normalisation and the per-channel GroupNorm after conv 0.
    # Streaming version: both use the statistics of the trailing NORM_STEPS steps (1 s) known when the samples
    # ARRIVE (per 100 ms step); every conv output is then computed once and cached. extract (whole stream at once)
    # and StreamingSpeechEncoder (incremental) compute exactly the same numbers.
    @torch.inference_mode()
    def w2v_frontend(self, x, base, m0):
        """x: (N,) float32 samples, x[i] = absolute sample base+i, base == 1600*m0 (a step boundary), N a multiple
        of 1600. Returns (F (T, D) projected conv features, renc_abs (T,) absolute receptive ends)."""
        spm = SR * STEP_FRAMES // FPS
        x = np.asarray(x, np.float32)
        N = len(x)
        assert base == spm * m0 and N % spm == 0
        G = N // spm
        xs = x.reshape(G, spm).astype(np.float64)
        ps, ps2 = np.concatenate([np.zeros(NORM_STEPS - 1), xs.sum(1)]), np.concatenate([np.zeros(NORM_STEPS - 1), (xs ** 2).sum(1)])
        k = np.ones(NORM_STEPS)
        n = NORM_STEPS * spm
        mean = np.convolve(ps, k, "valid") / n
        var = np.maximum(np.convolve(ps2, k, "valid") / n - mean ** 2, 0)
        std = np.maximum(np.sqrt(var), STD_FLOOR)
        xn = ((xs - mean[:, None]) / std[:, None]).astype(np.float32).reshape(-1)
        fe = self.enc.feature_extractor
        c0l = fe.conv_layers[0]
        c0 = c0l.conv(torch.from_numpy(xn)[None, None])[0]                  # (512, U), u covers [5u, 5u+10)
        U = c0.shape[1]
        g_of_u = np.minimum((5 * np.arange(U) + 10 - 1) // spm, G - 1)     # arrival step (end 5u+10 <= 1600(g+1))
        gi = torch.from_numpy(g_of_u)
        S1 = torch.zeros(c0.shape[0], G + NORM_STEPS - 1, dtype=torch.float64)
        S2 = torch.zeros_like(S1)
        S1.index_add_(1, gi + NORM_STEPS - 1, c0.double())
        S2.index_add_(1, gi + NORM_STEPS - 1, c0.double() ** 2)
        cnt = torch.zeros(G + NORM_STEPS - 1, dtype=torch.float64)
        cnt.index_add_(0, gi + NORM_STEPS - 1, torch.ones(U, dtype=torch.float64))
        cs = lambda a: torch.nn.functional.conv1d(a.reshape(-1, 1, a.shape[-1]), torch.ones(1, 1, NORM_STEPS, dtype=torch.float64)).reshape(*a.shape[:-1], -1)
        n1 = torch.clamp(cs(cnt[None])[0], min=1.0)
        mu = cs(S1) / n1
        va = torch.clamp(cs(S2) / n1 - mu ** 2, min=0)
        gn = c0l.layer_norm
        h = ((c0.double() - mu[:, gi]) / torch.sqrt(va[:, gi] + gn.eps)).float()
        h = h * gn.weight[:, None] + gn.bias[:, None]
        h = c0l.activation(h)[None]
        for layer in fe.conv_layers[1:]:
            h = layer(h)
        f = h.transpose(1, 2)
        fp = self.enc.feature_projection(f)
        F = (fp[0] if isinstance(fp, tuple) else fp)[0]
        renc = base + 320 * np.arange(F.shape[0]) + 400
        return F, renc

    @torch.inference_mode()
    def w2v_transformer(self, h, layers):
        """h: (B, NF, D) projected conv features of a window -> {layer: (B, NF, D)}."""
        en = self.enc.encoder
        out = {}
        h = h + en.pos_conv_embed(h)
        h = en.layer_norm(h)
        if 0 in layers:
            out[0] = h
        for li in range(max(layers)):
            h = en.layers[li](h, attention_mask=None)
            if isinstance(h, tuple):
                h = h[0]
            if li + 1 in layers:
                out[li + 1] = h
        return out


# ---------------------------------------------------------------------------------------- frame selection
def step_plan(n_frames, frame0, window_s, lookahead, step=STEP_FRAMES):
    """Steps on the ABSOLUTE 100 ms grid (step m ends at sample 1600*m = frame_end_sample(3m-1)); step m outputs
    frames newest-2..newest with newest = 3m-1-lookahead. For a stream whose frame j is absolute frame frame0+j:
    list of (window_start_sample, window_end_sample, [stream frames of this step]) covering every stream frame."""
    Wn = int(round(window_s * SR))
    spm = SR * step // FPS                                  # 1600 samples per step
    plan = []
    m = (frame0 + lookahead + 1) // step                    # first step whose newest frame >= frame0 - 2
    while True:
        newest = step * m - 1 - lookahead
        fr = [k - frame0 for k in range(newest - step + 1, newest + 1) if 0 <= k - frame0 < n_frames]
        if newest - step + 1 - frame0 >= n_frames:
            break
        if fr:
            t_end = spm * m
            assert t_end == frame_end_sample(newest + lookahead)
            plan.append((t_end - Wn, t_end, fr))
        m += 1
    return plan


def bin_index(renc_abs, frame_abs):
    """indices of encoder frames whose receptive end falls in frame `frame_abs`'s bin (e_{k-1}, e_k]."""
    lo, hi = frame_end_sample(frame_abs - 1), frame_end_sample(frame_abs)
    idx = np.nonzero((renc_abs > lo) & (renc_abs <= hi))[0]
    return idx


def take_windows(x16, plan, base=0):
    """(n_steps, Wn) float32 windows (zero-padded outside the recording); x16[i] = absolute sample base + i."""
    Wn = plan[0][1] - plan[0][0]
    out = np.zeros((len(plan), Wn), np.float32)
    N = len(x16)
    for i, (a, b, _) in enumerate(plan):
        a0, b0 = max(a - base, 0), min(b - base, N)
        if b0 > a0:
            out[i, a0 - (a - base): b0 - (a - base)] = x16[a0:b0]
    return out


def extract_stream(model, x16, n_frames, frame0, window_s, lookahead, layers, batch=32, proj=None, base=0):
    """Emulates the live runtime over a whole stream (x16[i] = absolute 16 kHz sample base + i; frame j of the stream
    is absolute frame frame0 + j). Returns {layer: (n_frames, D)} float32 (proj(layer, array) applied if given).
    x16 must cover the first window (and, for wav2vec2/HuBERT, NORM_STEPS more steps of statistics) before frame0."""
    plan = step_plan(n_frames, frame0, window_s, lookahead)
    outs = {l: None for l in layers}

    def put(sub, H, ra_of):
        for l in layers:
            Hl = H[l].float().numpy()
            if outs[l] is None:
                outs[l] = np.zeros((n_frames, Hl.shape[-1]), np.float32)
            for si, (a, b, fr) in enumerate(sub):
                ra = ra_of(si)
                for j in fr:
                    idx = bin_index(ra, frame0 + j)
                    outs[l][j] = Hl[si, idx].mean(0)

    if model.kind == "whisper":
        Wn = plan[0][1] - plan[0][0]
        renc = model.recept_end(Wn)
        for b0 in range(0, len(plan), batch):
            sub = plan[b0:b0 + batch]
            wins = torch.from_numpy(take_windows(x16, sub, base))
            H = model(wins, layers=layers)
            put(sub, H, lambda si: sub[si][0] + renc)
    else:
        spm = SR * STEP_FRAMES // FPS
        NF = int(round(window_s * SR)) // 320
        m0 = base // spm + (1 if base % spm else 0)                  # first full step inside x16
        m1 = plan[-1][1] // spm
        xa = np.zeros((m1 - m0) * spm, np.float32)
        lo = m0 * spm - base
        avail = x16[lo:lo + len(xa)]
        xa[:len(avail)] = avail
        F, renc = model.w2v_frontend(xa, m0 * spm, m0)
        ends = np.searchsorted(renc, [b for _, b, _ in plan], "right")
        assert ends.min() >= NF, "not enough audio before the stream for the first window"
        for b0 in range(0, len(plan), batch):
            sub = plan[b0:b0 + batch]
            e = ends[b0:b0 + batch]
            h = torch.stack([F[k - NF:k] for k in e])
            H = model.w2v_transformer(h, layers)
            put(sub, H, lambda si: renc[e[si] - NF:e[si]])
    if proj is not None:
        outs = {l: proj(l, v) for l, v in outs.items()}
    return outs


# ---------------------------------------------------------------------------------------- PCA + live runtime
class Projection:
    """train-set PCA, whitened: z = ((h - mean) @ comps.T) / scale."""

    def __init__(self, path):
        z = np.load(path)
        self.mean, self.comps, self.scale = z["mean"], z["comps"], z["scale"]

    def __call__(self, h):
        return ((h - self.mean) @ self.comps.T) / self.scale


class StreamingSpeechEncoder:
    """Live use (sliding mode, Whisper): push 16 kHz audio (absolute sample 0 = session start) as it arrives; every
    time a full 100 ms step m is buffered, step() returns (frames, feats): the 3 frames 3m-3-L..3m-1-L and their
    (3, D) features (PCA applied if given) - bit-identical to extract_stream() on the same audio."""

    def __init__(self, name, layer, window_s=1.0, lookahead=1, pca_path=None, threads=None, models_dir=MODELS_DIR):
        if threads:
            torch.set_num_threads(int(threads))
        self.model = WindowEncoder(name, max_layer=layer, models_dir=models_dir)
        assert self.model.kind == "whisper", "sliding runtime implemented for Whisper (wav2vec2/HuBERT: KV mode)"
        self.layer, self.window_s, self.lookahead = int(layer), float(window_s), int(lookahead)
        self.Wn = int(round(window_s * SR))
        self.renc = self.model.recept_end(self.Wn)
        self.proj = Projection(pca_path) if pca_path else None
        self.buf = np.zeros(0, np.float32)
        self.buf0 = 0                       # absolute sample index of buf[0]
        self.m = 0                          # last processed step

    def push(self, x16):
        self.buf = np.concatenate([self.buf, np.asarray(x16, np.float32)])

    def step(self):
        """process the next step if its audio is complete; returns (frames, feats) or None."""
        m = self.m + 1
        t_end = SR * STEP_FRAMES // FPS * m
        if self.buf0 + len(self.buf) < t_end:
            return None
        self.m = m
        a = t_end - self.Wn
        win = np.zeros(self.Wn, np.float32)
        lo, hi = max(a, self.buf0), t_end
        if hi > lo:
            win[lo - a: hi - a] = self.buf[lo - self.buf0: hi - self.buf0]
        keep = self.Wn + SR
        if len(self.buf) > keep:
            cut = len(self.buf) - keep
            self.buf = self.buf[cut:]
            self.buf0 += cut
        H = self.model(torch.from_numpy(win)[None], layers=[self.layer])[self.layer][0].float().numpy()
        ra = a + self.renc
        newest = STEP_FRAMES * m - 1 - self.lookahead
        frames = [k for k in range(newest - STEP_FRAMES + 1, newest + 1) if k >= 0]
        out = np.stack([H[bin_index(ra, k)].mean(0) for k in frames]) if frames else np.zeros((0, H.shape[1]))
        return frames, (self.proj(out) if self.proj is not None and len(out) else out)


# ======================================================================================== KV (block-causal) mode
# The DEFAULT streaming mode (cheap enough for two live streams at 2 threads and for training-set extraction):
# every 100 ms step only the ~5 NEW encoder frames are computed; their queries attend (per layer) to the cached
# keys/values of the last KV_STEPS steps (1 s) and to each other; a frame's states are never recomputed later.
# Front ends are streamed with arrival-time statistics (see w2v_frontend / whisper_frontend). Offline extraction
# runs the same thing over a whole stream with a block-causal attention mask (identical numbers up to float error;
# tools/audio/test_speech_enc.py checks KVStreamer against kv_extract_stream).
KV_STEPS = 10
SPM = SR * STEP_FRAMES // FPS        # 1600 samples per step
EPOCH = 1500                          # Whisper: positions restart every 1500 frames (30 s); two staggered streams


def _attn(mod, hq, hkv, mask):
    B, Tq, D = hq.shape
    Tk = hkv.shape[1]
    H, hd = mod.num_heads, mod.head_dim
    q = mod.q_proj(hq).view(B, Tq, H, hd).transpose(1, 2)
    k = mod.k_proj(hkv).view(B, Tk, H, hd).transpose(1, 2)
    v = mod.v_proj(hkv).view(B, Tk, H, hd).transpose(1, 2)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=mod.scaling)
    return mod.out_proj(o.transpose(1, 2).reshape(B, Tq, D))


def _layer(model, li, hq, hkv, mask):
    """one encoder layer for query rows hq attending to rows hkv (both = layer inputs)."""
    if model.kind == "whisper":
        L = model.enc.layers[li]
        a = _attn(L.self_attn, L.self_attn_layer_norm(hq), L.self_attn_layer_norm(hkv), mask)
        h = hq + a
        return h + L.fc2(L.activation_fn(L.fc1(L.final_layer_norm(h))))
    L = model.enc.encoder.layers[li]
    h = L.layer_norm(hq + _attn(L.attention, hq, hkv, mask))
    return L.final_layer_norm(h + L.feed_forward(h))


def kv_mask(arr_q, arr_k, ep_q=None, ep_k=None):
    """bool (Tq, Tk): key allowed iff it arrived in the same or one of the KV_STEPS-1 previous steps (and, for
    Whisper, lies in the same position epoch)."""
    d = arr_q[:, None] - arr_k[None, :]
    m = (d >= 0) & (d < KV_STEPS)
    if ep_q is not None:
        m &= ep_q[:, None] == ep_k[None, :]
    return torch.from_numpy(m)


@torch.inference_mode()
def whisper_frontend(model, x, base, mel_state=None):
    """Streaming Whisper front end over samples x (x[i] = absolute base+i, base a multiple of 1600, len multiple of
    1600). Hann-400/hop-160 power spectrum with ZERO padding before x (mel frame j centred on sample 160j covers
    [160j-200, 160j+200)), 80 mel bins, log10, then the Whisper clamp/scale with the max over the trailing KV_STEPS
    steps AT ARRIVAL; conv1+GELU, conv2+GELU. Returns (h (T, d) conv features, renc_abs (T,)): encoder frame i
    (absolute index base/320 + i) has receptive end 320*i_abs + 520.
    mel_state (runtime): dict with the previous 4 normalized mel frames + their (log-max per step) history."""
    e = model.enc
    x = np.asarray(x, np.float32)
    N = len(x)
    assert base % SPM == 0 and N % SPM == 0
    G = N // SPM
    j0 = base // 160
    nm = N // 160                                             # mel frames with centre in [base, base+N)
    xp = np.concatenate([np.zeros(200, np.float32), x, np.zeros(200, np.float32)])
    if mel_state is not None and mel_state.get("tail") is not None:
        xp[:200] = mel_state["tail"]
    idx = np.arange(nm)[:, None] * 160 + np.arange(400)[None]
    seg = torch.from_numpy(xp[idx]) * model.hann[None]
    mag = torch.fft.rfft(seg, dim=1).abs() ** 2                # (nm, 201)
    lg = torch.clamp(mag @ model.mel_fb, min=1e-10).log10()   # (nm, 80)
    arr = (160 * (j0 + np.arange(nm)) + 200 + SPM - 1) // SPM  # absolute arrival step of each mel frame
    fmax = lg.amax(1).double().numpy()
    steps = np.arange(arr.min(), arr.max() + 1)
    prev = mel_state.get("smax", {}) if mel_state is not None else {}
    smax = {int(s): -np.inf for s in steps}
    for s, v in prev.items():
        smax[int(s)] = max(smax.get(int(s), -np.inf), v)
    for a_, v in zip(arr, fmax):
        smax[int(a_)] = max(smax[int(a_)], v)
    run = np.array([max(smax.get(int(a_) - k, -np.inf) for k in range(KV_STEPS)) for a_ in arr])
    lg = torch.maximum(lg, torch.from_numpy(run).float()[:, None] - 8.0)
    lg = ((lg + 4.0) / 4.0).T[None]                           # (1, 80, nm)
    if mel_state is not None:
        # runtime: prepend the cached normalized mel frames so conv1/conv2 see the same neighbours as offline
        pre = mel_state.get("mel")
        if pre is not None:
            lg = torch.cat([pre, lg], 2)
    h1 = torch.nn.functional.gelu(e.conv1(lg))
    h2 = torch.nn.functional.gelu(e.conv2(h1))[0].T          # (T, d)
    return h2, lg, smax


def _w2v_pos(model, F, last_of):
    """pos-conv embedding where frame i only sees frames <= last_of[i] (its step's last frame): 5 masked passes."""
    pc = model.enc.encoder.pos_conv_embed
    w, b = pc.conv.weight, pc.conv.bias
    K = w.shape[-1]
    half = K // 2
    xin = F.T[None]                                            # (1, D, T)
    out = torch.zeros_like(F)
    off = last_of - np.arange(len(F))
    for o in np.unique(off):
        wm = w.clone()
        wm[..., half + int(o) + 1:] = 0                        # tap k reads frame i + k - half
        y = torch.nn.functional.conv1d(xin, wm, b, padding=half, groups=pc.conv.groups)[..., :-1]
        sel = np.nonzero(off == o)[0]
        out[sel] = pc.activation(y[0].T[sel])
    return out


@torch.inference_mode()
def kv_states(model, x, base, layers):
    """Offline KV mode over a whole buffer (x[i] = absolute sample base+i). Returns ({layer: (T, D)}, renc_abs)."""
    if model.kind == "whisper":
        h, _, _ = whisper_frontend(model, x, base)
        T = h.shape[0]
        i_abs = base // 320 + np.arange(T)
        renc = 320 * i_abs + 520
        arr = (renc + SPM - 1) // SPM
        res = {l: torch.zeros(T, model.dim) for l in layers}
        useA = ((i_abs % EPOCH) >= EPOCH // 4) & ((i_abs % EPOCH) < 3 * EPOCH // 4)
        for shift, use in ((0, useA), (EPOCH // 2, ~useA)):
            pos = (i_abs + shift) % EPOCH
            ep = (i_abs + shift) // EPOCH
            g = h + model.enc.embed_positions.weight[torch.from_numpy(pos)]
            M = kv_mask(arr, arr, ep, ep)
            sel = torch.from_numpy(use)
            if 0 in layers:
                res[0][sel] = g[sel]
            for li in range(max(layers)):
                g = _layer(model, li, g[None], g[None], M)[0]
                if li + 1 in layers:
                    v = model.enc.layer_norm(g) if li + 1 == model.n_layers else g
                    res[li + 1][sel] = v[sel]
        return res, renc
    m0 = base // SPM
    F, renc = model.w2v_frontend(x, base, m0)
    arr = (renc + SPM - 1) // SPM
    last_of = np.array([np.nonzero(arr == a_)[0].max() for a_ in arr])
    en = model.enc.encoder
    g = en.layer_norm(F + _w2v_pos(model, F, last_of))
    M = kv_mask(arr, arr)
    res = {}
    if 0 in layers:
        res[0] = g
    for li in range(max(layers)):
        g = _layer(model, li, g[None], g[None], M)[0]
        if li + 1 in layers:
            res[li + 1] = g
    return res, renc


def kv_extract_stream(model, x16, n_frames, frame0, layers, base, proj=None):
    """KV-mode features for stream frames 0..n_frames-1 (absolute frame frame0+j). x16[i] = absolute sample base+i
    must start >= 2 s before the stream (statistics + left context warm-up) and end >= 0.1 s after it.
    Frame k = mean of the encoder frames whose receptive end lies in (frame_end(k-1), frame_end(k)]."""
    m0 = -(-base // SPM)
    m1 = (frame_end_sample(frame0 + n_frames - 1) + SPM - 1) // SPM
    xa = np.zeros((m1 - m0) * SPM, np.float32)
    lo = m0 * SPM - base
    av = x16[lo:lo + len(xa)]
    xa[:len(av)] = av
    res, renc = kv_states(model, xa, m0 * SPM, layers)
    ends = np.array([frame_end_sample(frame0 + j) for j in range(-1, n_frames)])
    b = np.searchsorted(ends, renc, "left")                    # renc in (ends[b-1], ends[b]] -> stream frame b-1
    out = {}
    for l in layers:
        H = res[l].float().numpy()
        o = np.zeros((n_frames, H.shape[1]), np.float32)
        c = np.zeros(n_frames)
        ok = (b >= 1) & (b <= n_frames)
        np.add.at(o, b[ok] - 1, H[ok])
        np.add.at(c, b[ok] - 1, 1)
        assert c.min() >= 1
        out[l] = o / c[:, None]
    if proj is not None:
        out = {l: proj(l, v) for l, v in out.items()}
    return out
