"""mm/audio_feats.py — streaming canon1 audio features (40 log-mel + loud_db + vad @ 30 fps) from 16 kHz PCM. numpy only.
Math copied from canon1/causal_audio.py (audio_features / energy_vad), made incremental:

    fx = PcmFeaturizer()
    feats = fx.push(pcm16k_float32)      # -> (n,42) new complete frames (may be 0 rows)
    gen.push_audio(feats)
    fx.reset()                           # e.g. between utterances is NOT needed; reset only on a new session

Frame k covers [k/30, (k+1)/30) s and uses only samples before its end (causal), exactly like the offline extractor,
so frame k of a stream == row k of audio_features(whole_signal).  Silence must be pushed as zeros (time keeps running).
"""
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
HOP = int(round(SR / FPS))


def frame_end_sample(k):
    return int(round((k + 1) * SR / FPS))


class PcmFeaturizer:
    def __init__(self, on_margin=18.0, off_margin=12.0, hang=6):
        self.on_margin, self.off_margin, self.hang = on_margin, off_margin, hang
        self.reset()

    def reset(self):
        self.buf = np.zeros(N_FFT + HOP, np.float32)   # left zero padding, like the offline extractor
        self.base = -(N_FFT + HOP)                        # absolute sample index of buf[0]
        self.n_in = 0
        self.k = 0
        self.hist = []
        self.on = False
        self.hold = 0

    def push(self, x):
        x = np.asarray(x, np.float32).reshape(-1)
        self.buf = np.concatenate([self.buf, x])
        self.n_in += len(x)
        rows = []
        while frame_end_sample(self.k) <= self.n_in:
            e = frame_end_sample(self.k) - self.base
            seg = self.buf[e - N_FFT:e] * _WIN
            spec = np.abs(np.fft.rfft(seg)) ** 2
            mel = np.log(np.maximum(spec @ MEL_FB.T, 1e-6))
            s2 = self.buf[e - HOP:e]
            rms = np.sqrt(np.mean(s2 ** 2) + 1e-12)
            loud = max(20 * np.log10(rms + 1e-12), -100.0)
            self.hist.append(loud)
            if len(self.hist) > 300:
                self.hist.pop(0)
            floor = np.percentile(self.hist, 10)
            if loud > floor + self.on_margin:
                self.on = True; self.hold = self.hang
            elif loud < floor + self.off_margin:
                if self.hold > 0:
                    self.hold -= 1
                else:
                    self.on = False
            rows.append(np.concatenate([mel, [loud, 1.0 if self.on else 0.0]]))
            self.k += 1
        # keep only what the next frame needs
        keep_from = frame_end_sample(self.k) - max(N_FFT, HOP + 2) - self.base
        if keep_from > 0:
            self.buf = self.buf[keep_from:]
            self.base += keep_from
        return np.array(rows, np.float32).reshape(-1, 42)
