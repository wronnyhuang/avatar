#!/usr/bin/env python3
"""Run clips through the lip animator in 100 ms chunks, for all eleven characters.

    python -m anim.make_lip_params --out DIR [--clips clip,clip2] [--lookahead A] [--lead L] [--characters orb,fox]

Writes <out>/<clip>/<character>.npy (frames x NPARAMS, float32), <character>.visemes.npy (frames x 23, the fine visemes)
and <character>.json (names, ranges, CPU cost).  A is milliseconds, L is video frames: docs/VISEMES.md.
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
    from anim.lipstream import DEFAULT_LEAD, DEFAULT_LOOKAHEAD_MS, StreamingLipAnimator
    from anim.poses import FINE
    from anim.schema import CHUNK_MS, FPS, NPARAMS, PARAMS, SAMPLE_RATE
else:
    from .lipstream import DEFAULT_LEAD, DEFAULT_LOOKAHEAD_MS, StreamingLipAnimator
    from .poses import FINE
    from .schema import CHUNK_MS, FPS, NPARAMS, PARAMS, SAMPLE_RATE

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT.parent / "out" / "params_lips"
CHARACTERS = ["fox", "bear", "rabbit", "hippo", "monkey", "orb", "cube", "drop", "star", "pyramid", "blob", "muse"]   # body: + muse


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: need {SAMPLE_RATE} Hz mono 16-bit")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def animate(x: np.ndarray, character: str, chunk: int, **kw) -> tuple[np.ndarray, np.ndarray, float]:
    anim = StreamingLipAnimator(character, **kw)
    t0 = time.process_time()
    frames = []
    for i in range(0, len(x), chunk):
        frames += anim.push(x[i:i + chunk])
    frames += anim.flush()
    cpu = time.process_time() - t0
    assert [f.index for f in frames] == list(range(len(frames)))
    return (np.stack([f.params for f in frames]).astype(np.float32),
            np.stack([f.fine for f in frames]).astype(np.float32), cpu)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--clips", default="clip,clip2")
    ap.add_argument("--chunk-ms", type=int, default=CHUNK_MS)
    ap.add_argument("--characters", default=",".join(CHARACTERS))
    ap.add_argument("--lookahead", type=float, default=DEFAULT_LOOKAHEAD_MS, help="A, ms of audio delay")
    ap.add_argument("--lead", type=int, default=DEFAULT_LEAD, help="L, video frames the mouth leads the voice")
    ap.add_argument("--net", type=Path, default=None, help="another net file, for comparing nets")
    a = ap.parse_args()
    chunk = SAMPLE_RATE * a.chunk_ms // 1000
    kw = dict(lookahead_ms=a.lookahead, lead_frames=a.lead)
    if a.net:
        kw["net_path"] = a.net
    for clip in a.clips.split(","):
        wav = ROOT / "audio_in" / f"{clip}.wav"
        x = read_wav(wav)
        seconds = len(x) / SAMPLE_RATE
        sha = hashlib.sha256(wav.read_bytes()).hexdigest()
        out = a.out / clip
        out.mkdir(parents=True, exist_ok=True)
        for name in a.characters.split(","):
            P, V, cpu = animate(x, name, chunk, **kw)
            assert P.shape[1] == NPARAMS
            np.save(out / f"{name}.npy", P)
            np.save(out / f"{name}.visemes.npy", V)
            side = dict(character=name, clip=clip, frames=len(P), fps=FPS, nparams=NPARAMS, params=PARAMS,
                        visemes=FINE, wav=str(wav), wav_sha256=sha, audio_seconds=seconds,
                        sample_rate=SAMPLE_RATE, chunk_ms=a.chunk_ms, lookahead_ms=a.lookahead, lead_frames=a.lead,
                        cpu_seconds=round(cpu, 3), cpu_per_audio_second=round(cpu / seconds, 4),
                        ranges={n: [round(float(P[:, i].min()), 4), round(float(P[:, i].max()), 4)]
                                for i, n in enumerate(PARAMS)})
            (out / f"{name}.json").write_text(json.dumps(side, indent=1))
            print(f"{clip}/{name}: {len(P)} frames, {cpu:.2f} s CPU, {cpu / seconds:.4f} s per audio second", flush=True)


if __name__ == "__main__":
    main()
