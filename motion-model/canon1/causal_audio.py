"""Causal audio features, copied verbatim from the first learned-reaction model's spec (react_v1; identical to the
live planner process)."""
import math

import numpy as np

FPS = 30
SR = 16000
N_MEL = 40
N_FFT = 512


def _mel_fb(n_mel=N_MEL, n_fft=N_FFT, sr=SR, fmin=60.0, fmax=7600.0):
    def hz2mel(f): return 2595.0 * np.log10(1 + f / 700.0)
    def mel2hz(m): return 700.0 * (10 ** (m / 2595.0) - 1)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mel + 2)
    hz = mel2hz(mels)
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mel, len(bins)))
    for i in range(n_mel):
        l, c, r = hz[i], hz[i + 1], hz[i + 2]
        fb[i] = np.clip(np.minimum((bins - l) / (c - l), (r - bins) / (r - c)), 0, None)
    return fb.astype(np.float32)


MEL_FB = _mel_fb()
_WIN = np.hanning(N_FFT).astype(np.float32)


def frame_end_sample(k):
    """sample index (exclusive) at which frame k ends: frame k covers [k/30, (k+1)/30)."""
    return int(round((k + 1) * SR / FPS))


def audio_features(x16k, n_frames=None, vad=None):
    """x16k: float mono 16 kHz in [-1,1]. Returns (T,42): 40 log-mel (natural log of power, floor 1e-6),
    loud_db (20*log10 rms over the frame window, floor -100), vad (0/1). Frame k uses ONLY samples before the
    end of frame k (window = the 512 samples ending there). `vad`: optional precomputed (T,) 0/1 track
    (e.g. a dataset's own VAD resampled to 30 fps); otherwise energy_vad() is used."""
    x = np.asarray(x16k, np.float32)
    T = n_frames if n_frames is not None else int(len(x) * FPS / SR)
    ends = np.array([frame_end_sample(k) for k in range(T)])
    pad = np.concatenate([np.zeros(N_FFT, np.float32), x, np.zeros(N_FFT, np.float32)])
    idx = ends[:, None] + np.arange(-N_FFT, 0)[None, :] + N_FFT
    idx = np.clip(idx, 0, len(pad) - 1)
    seg = pad[idx] * _WIN[None]
    spec = np.abs(np.fft.rfft(seg, axis=1)) ** 2
    mel = np.log(np.maximum(spec @ MEL_FB.T, 1e-6))
    hop = int(round(SR / FPS))
    seg2 = pad[np.clip(ends[:, None] + np.arange(-hop, 0)[None, :] + N_FFT, 0, len(pad) - 1)]
    rms = np.sqrt(np.mean(seg2 ** 2, 1) + 1e-12)
    loud = np.maximum(20 * np.log10(rms + 1e-12), -100.0)
    v = energy_vad(loud) if vad is None else np.asarray(vad, np.float32)[:T]
    return np.concatenate([mel, loud[:, None], v[:, None]], 1).astype(np.float32)


def energy_vad(loud_db, on_margin=18.0, off_margin=12.0, hang=6):
    """Causal energy VAD with hysteresis relative to a running noise floor (10th percentile of the
    last ~10 s). Used for every dataset without its own VAD AND by the live planner, so it is identical."""
    T = len(loud_db)
    out = np.zeros(T, np.float32)
    hist = []
    on = False; hold = 0
    for k in range(T):
        hist.append(loud_db[k])
        if len(hist) > 300:
            hist.pop(0)
        floor = np.percentile(hist, 10)
        if loud_db[k] > floor + on_margin:
            on = True; hold = hang
        elif loud_db[k] < floor + off_margin:
            if hold > 0:
                hold -= 1
            else:
                on = False
        out[k] = 1.0 if on else 0.0
    return out


def resample_to_16k(x, sr):
    if sr == SR:
        return np.asarray(x, np.float32)
    from scipy.signal import resample_poly
    g = math.gcd(int(sr), SR)
    return resample_poly(np.asarray(x, np.float64), SR // g, int(sr) // g).astype(np.float32)

