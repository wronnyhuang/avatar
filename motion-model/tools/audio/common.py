"""Shared helpers for the phase-2 audio tools (stream list, segment loading, canon targets)."""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from canon1 import spec  # noqa: E402

from canon1 import paths as _paths  # noqa: E402

P = str(_paths.P2)            # $MM_WORK_DIR/p2: stream lists, features, audio work, encoder models
SEAMLESS = str(_paths.SEAMLESS)
SR = 16000
FPS = 30


def streams(path=None):
    return [json.loads(l) for l in open(path or os.path.join(P, "streams_abl.jsonl"))]


def load_segment(wav, s0, s1, margin_s=2.0):
    """16 kHz float32 samples covering absolute 16 kHz sample range [s0, s1) (zeros outside the file).
    Reads only [s0 - margin, s1 + margin] at the native rate and resamples that piece."""
    import soundfile as sf
    from scipy.signal import resample_poly
    info = sf.info(wav)
    sr = info.samplerate
    m = int(margin_s * SR)
    a16, b16 = s0 - m, s1 + m
    a = max(0, int(math.floor(a16 * sr / SR)))
    b = min(info.frames, int(math.ceil(b16 * sr / SR)))
    out = np.zeros(s1 - s0, np.float32)
    if b <= a:
        return out
    x, _ = sf.read(wav, start=a, stop=b, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        g = math.gcd(sr, SR)
        x = resample_poly(x.astype(np.float64), SR // g, sr // g).astype(np.float32)
    first16 = int(round(a * SR / sr))            # 16 kHz index of x[0]
    lo, hi = max(s0, first16), min(s1, first16 + len(x))
    if hi > lo:
        out[lo - s0: hi - s0] = x[lo - first16: hi - first16]
    return out


def stream_audio(row, key="audio_self", pre_s=2.5, post_s=0.5):
    """(x16, base) where x16[i] is absolute 16 kHz sample base + i; covers the stream plus pre/post context."""
    f0 = int(row["frame0"])
    s0 = spec.frame_end_sample(f0 - 1) - int(pre_s * SR)
    s1 = spec.frame_end_sample(f0 + row["n_frames"] - 1) + int(post_s * SR)
    return load_segment(row[key], s0, s1), s0


def canon_person(row):
    """(motion (n,189) f32, mask (n,9) u8, audio42 (n,42) f32) for this stream's frames."""
    z = np.load(row["canon"], allow_pickle=False)
    p, f0, n = row["person_index"], row["frame0"], row["n_frames"]
    m = z["motion"][p, f0:f0 + n].astype(np.float32)
    k = z["mask"][p, f0:f0 + n]
    a = z["audio"][p, f0:f0 + n].astype(np.float32)
    return m, k, a
