"""Causal viseme estimator: log-mel features into a small GRU, plain numpy, no deep-learning runtime.

Grid, in samples, on the same clock as anim/dsp.py:
    mel hop h ends at sample 200 h and reads only the WIN samples before that.
    model step k runs after mel hop 2 k (sample 400 k, 16.7 ms) on mel hops 2 k - 1 and 2 k.
Output at step k is trained to describe the speech at step k - DELAY_STEPS, so a frame latched at
sample 800 i + 1600 (step 2 i + 4) reads a head with DELAY_STEPS <= 4 and stays inside the 67 ms budget.

The weights live in anim/viseme_net.bin: a JSON header line, then little-endian float32 arrays.  Inference
uses einsum, which runs its own single-threaded loops, so no BLAS thread pool can change the bytes or the cost.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

VISEMES = ["sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "ih", "oh", "ou"]
VIDX = {v: i for i, v in enumerate(VISEMES)}
NET_PATH = Path(__file__).with_name("viseme_net.bin")

SR = 24000
HOP = 200               # one mel hop
STEP_HOPS = 2           # one model step = 2 mel hops = 400 samples = 16.7 ms
WIN = 480               # 20 ms
NFFT = 512
N_MELS = 40
F_LO, F_HI = 60.0, 11000.0
FLOOR = 1e-11           # power floor, so digital silence maps to a fixed vector


def mel_matrix() -> np.ndarray:
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    inv = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    edges = inv(np.linspace(mel(F_LO), mel(F_HI), N_MELS + 2))
    f = np.arange(NFFT // 2 + 1) * SR / NFFT
    M = np.zeros((N_MELS, f.size))
    for i in range(N_MELS):
        a, c, b = edges[i], edges[i + 1], edges[i + 2]
        M[i] = np.clip(np.minimum((f - a) / (c - a), (b - f) / (b - c)), 0.0, None)
    return M / np.maximum(M.sum(axis=1, keepdims=True), 1e-9)


class LogMel:
    """Pure function of one window; the stream and batch paths share this arithmetic."""

    def __init__(self) -> None:
        self.window = np.hanning(WIN + 2)[1:-1]
        self.scale = 2.0 / float(np.sum(self.window ** 2))
        self.M = mel_matrix()

    def frame(self, x: np.ndarray) -> np.ndarray:
        """x: WIN float64 samples, oldest first.  Returns N_MELS dB values."""
        s = np.fft.rfft(x * self.window, NFFT)
        p = (s.real * s.real + s.imag * s.imag) * self.scale
        return 10.0 * np.log10(np.einsum("ij,j->i", self.M, p) + FLOOR)

    def batch(self, x: np.ndarray) -> np.ndarray:
        """Row h - 1 is the features of mel hop h, for h = 1 .. len(x) // HOP.  Only for training and tests."""
        x = np.concatenate((np.zeros(WIN), np.asarray(x, dtype=np.float64)))
        n = (len(x) - WIN) // HOP
        out = np.empty((n, N_MELS), np.float32)
        for lo in range(0, n, 4096):
            hi = min(n, lo + 4096)
            idx = HOP * (np.arange(lo, hi) + 1)[:, None] + np.arange(WIN)
            seg = x[idx] * self.window
            s = np.fft.rfft(seg, NFFT, axis=1)
            p = (s.real * s.real + s.imag * s.imag) * self.scale
            out[lo:hi] = 10.0 * np.log10(np.einsum("ij,bj->bi", self.M, p) + FLOOR)
        return out


# Level.  The net hears speech relative to its own recent peak, so a constant gain on the input changes nothing once the
# first syllable has passed.  Everything here is linear in dB, which is what makes the tracker move exactly with the gain.
LEVEL_FLOOR = -40.0     # dB: the reference never goes below this, so faint noise never reads as speech
LEVEL_ATTACK = 0.30     # share of the distance per mel hop, about 20 ms
LEVEL_DECAY = 4.0 * HOP / SR    # dB per hop while speech is on, 4 dB/s
LEVEL_GATE = 30.0       # further below the reference than this holds it, so a pause forgets nothing
REL_FLOOR = -60.0       # lowest level the net sees, and digital silence is exactly this


class LevelNorm:
    """One state per stream, stepped once per mel hop."""

    def __init__(self) -> None:
        self.ref = LEVEL_FLOOR

    def step(self, mel: np.ndarray) -> np.ndarray:
        top = float(mel.max())
        if top > self.ref:
            self.ref += LEVEL_ATTACK * (top - self.ref)
        elif top > self.ref - LEVEL_GATE:
            self.ref = max(LEVEL_FLOOR, self.ref - LEVEL_DECAY)
        return np.maximum(mel - self.ref, REL_FLOOR)


def level_track(top: np.ndarray, ref0: float = LEVEL_FLOOR) -> np.ndarray:
    """Batch twin of LevelNorm for training: top (hops, ...), the peak band of each hop.  Reference after each hop."""
    ref = np.empty_like(top, dtype=np.float32)
    r = np.full(top.shape[1:], ref0, np.float64)
    for i, t in enumerate(top):
        up = t > r
        down = ~up & (t > r - LEVEL_GATE)
        r = np.where(up, r + LEVEL_ATTACK * (t - r), np.where(down, np.maximum(LEVEL_FLOOR, r - LEVEL_DECAY), r))
        ref[i] = r
    return ref


def load_net(path: Path = NET_PATH) -> tuple[dict, dict[str, np.ndarray]]:
    raw = path.read_bytes()
    end = raw.index(b"\n")
    head = json.loads(raw[:end])
    weights = {}
    off = end + 1
    for name, shape in head["arrays"].items():
        n = int(np.prod(shape))
        weights[name] = np.frombuffer(raw, "<f4", n, off).reshape(shape).astype(np.float64)
        off += 4 * n
    return head, weights


def save_net(path: Path, head: dict, weights: dict[str, np.ndarray]) -> None:
    head = dict(head, arrays={k: list(v.shape) for k, v in weights.items()})
    with open(path, "wb") as f:
        f.write(json.dumps(head).encode() + b"\n")
        for v in weights.values():
            f.write(np.asarray(v, dtype="<f4").tobytes())


class VisemeNet:
    """Input layer, stacked GRU, one linear read-out per head.  step() is one model step."""

    def __init__(self, path: Path = NET_PATH) -> None:
        self.head, w = load_net(path)
        self.mean, self.std = w["mean"], w["std"]
        self.W_in, self.b_in = w["in.w"], w["in.b"]
        self.layers = []
        for i in range(self.head["layers"]):
            self.layers.append(tuple(w[f"gru{i}.{k}"] for k in ("wih", "wih_b", "whh", "whh_b")))
        self.heads = {name: (w[f"head.{name}.w"], w[f"head.{name}.b"]) for name in self.head["heads"]}
        self.hidden = [np.zeros(self.layers[0][2].shape[1]) for _ in self.layers]
        self.window = list(self.head.get("window", []))         # older steps read straight into the input layer
        self.past: list[np.ndarray] = []

    def _gru(self, i: int, x: np.ndarray) -> np.ndarray:
        wih, bi, whh, bh = self.layers[i]
        h = self.hidden[i]
        a = np.einsum("ij,j->i", wih, x) + bi
        b = np.einsum("ij,j->i", whh, h) + bh
        n = a.size // 3
        r = 1.0 / (1.0 + np.exp(-(a[:n] + b[:n])))
        z = 1.0 / (1.0 + np.exp(-(a[n:2 * n] + b[n:2 * n])))
        c = np.tanh(a[2 * n:] + r * b[2 * n:])
        h = (h - c) * z + c
        self.hidden[i] = h
        return h

    def step(self, mel_a: np.ndarray, mel_b: np.ndarray) -> np.ndarray:
        """Two mel rows, oldest first.  Returns the top GRU state; read heads with read()."""
        x = (np.concatenate((mel_a, mel_b)) - self.mean) / self.std
        if self.window:
            new = x[N_MELS:]
            if not self.past:           # before the start, the first step's frame stands in, as in training
                self.past = [new] * (max(self.window) + 1)
            self.past = self.past[1:] + [new]
            x = np.concatenate([x] + [self.past[-1 - w] for w in self.window])
        y = np.einsum("ij,j->i", self.W_in, x) + self.b_in
        y = np.maximum(y, 0.0)
        for i in range(len(self.layers)):
            y = self._gru(i, y)
        return y

    def read(self, name: str, h: np.ndarray) -> np.ndarray:
        w, b = self.heads[name]
        return np.einsum("ij,j->i", w, h) + b


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
