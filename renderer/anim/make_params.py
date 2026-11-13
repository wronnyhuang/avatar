#!/usr/bin/env python3
"""Run a clip through the streaming animator in fixed chunks; write one params file per animal.

    python -m anim.make_params [--wav audio_in/clip.wav] [--out DIR] [--chunk-ms 100] [--animals fox,bear]

Writes <out>/<animal>.npy (frames x NPARAMS, float32) and <out>/<animal>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from anim.schema import CHUNK_MS, FPS, LOOKAHEAD_MS, NPARAMS, PARAMS, SAMPLE_RATE
    from anim.stream import StreamingAnimator
else:
    from .schema import CHUNK_MS, FPS, LOOKAHEAD_MS, NPARAMS, PARAMS, SAMPLE_RATE
    from .stream import StreamingAnimator

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "out" / "params"
ANIMALS = ["fox", "bear", "rabbit", "hippo", "monkey"]


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: need {SAMPLE_RATE} Hz mono 16-bit")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def animate(x: np.ndarray, animal: str, chunk: int) -> tuple[np.ndarray, float]:
    anim = StreamingAnimator(animal)
    t0 = time.process_time()
    frames = []
    for i in range(0, len(x), chunk):
        frames += anim.push(x[i:i + chunk])
    frames += anim.flush()
    cpu = time.process_time() - t0
    assert [f.index for f in frames] == list(range(len(frames)))
    return np.stack([f.params for f in frames]).astype(np.float32), cpu


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav", type=Path, default=ROOT / "audio_in/clip.wav")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--chunk-ms", type=int, default=CHUNK_MS)
    ap.add_argument("--animals", default=",".join(ANIMALS))
    a = ap.parse_args()
    x = read_wav(a.wav)
    a.out.mkdir(parents=True, exist_ok=True)
    chunk = SAMPLE_RATE * a.chunk_ms // 1000
    seconds = len(x) / SAMPLE_RATE
    sha = hashlib.sha256(a.wav.read_bytes()).hexdigest()
    for animal in a.animals.split(","):
        P, cpu = animate(x, animal, chunk)
        assert P.shape[1] == NPARAMS
        np.save(a.out / f"{animal}.npy", P)
        side = dict(
            animal=animal, frames=len(P), fps=FPS, nparams=NPARAMS, params=PARAMS,
            wav=str(a.wav), wav_sha256=sha, audio_seconds=seconds, sample_rate=SAMPLE_RATE,
            chunk_ms=a.chunk_ms, lookahead_ms=LOOKAHEAD_MS, cpu_seconds=round(cpu, 3),
            cpu_per_audio_second=round(cpu / seconds, 4),
            ranges={n: [round(float(P[:, i].min()), 4), round(float(P[:, i].max()), 4)]
                    for i, n in enumerate(PARAMS)})
        (a.out / f"{animal}.json").write_text(json.dumps(side, indent=1))
        print(f"{animal}: {len(P)} frames, {cpu:.2f} s CPU, {cpu / seconds:.4f} s per audio second")


if __name__ == "__main__":
    main()
