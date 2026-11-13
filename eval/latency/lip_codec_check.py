#!/usr/bin/env python3
"""Does mu-law on the voice tap hurt the lips? Run the renderer's causal lip model (anim.lipstream) on the same voice
twice -- int16 as before, and int16 -> G.711 mu-law -> int16 (exactly what the page/server now do) -- in 20 ms chunks
like the tap, and compare every mouth parameter frame by frame.

    python lip_codec_check.py --renderer <wt>/renderer --wav voice.wav [--wav ...] [--character fox] [--lookahead-ms 100]
"""
import argparse
import json
import subprocess
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--renderer", required=True)
ap.add_argument("--wav", action="append", required=True)
ap.add_argument("--character", default="fox")
ap.add_argument("--lookahead-ms", type=float, default=100)
a = ap.parse_args()
sys.path.insert(0, a.renderer)
sys.path.insert(0, a.renderer.rstrip("/").rsplit("/", 1)[0] + "/app")
from anim.lipstream import StreamingLipAnimator  # noqa: E402
from anim.schema import PARAMS  # noqa: E402
import wire  # noqa: E402

LIP = [p for p in ("jaw", "round", "spread", "press", "labdent", "tongue", "lip_open", "lip_width", "lip_protrude",
                   "lip_upper", "lip_lower", "teeth_gap", "tongue_tip", "tongue_dental", "tongue_back") if p in PARAMS]


def run(pcm):
    an = StreamingLipAnimator(a.character, lookahead_ms=a.lookahead_ms)
    out = []
    for i in range(0, len(pcm), 480):
        out += an.push(pcm[i:i + 480])
    out += an.flush()
    return np.stack([f.params for f in out])


res = {}
for w in a.wav:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", w, "-ac", "1", "-ar", "24000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, "<i2")
    y = np.frombuffer(wire.ulaw_decode(wire.ulaw_encode(x.tobytes())), "<i2")
    snr = 10 * np.log10(np.mean(x.astype(float) ** 2) / max(1e-9, np.mean((x.astype(float) - y) ** 2)))
    A, B = run(x), run(y)
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    per = {}
    for p in LIP:
        i = PARAMS.index(p)
        a_, b_ = A[:, i], B[:, i]
        rng = float(a_.max() - a_.min())
        if rng < 1e-4:
            continue
        per[p] = {"max_abs": round(float(np.abs(a_ - b_).max()), 4), "mean_abs": round(float(np.abs(a_ - b_).mean()), 4),
                  "range": round(rng, 3), "corr": round(float(np.corrcoef(a_, b_)[0, 1]), 4)}
    jaw = per.get("jaw") or per.get("lip_open") or {}
    res[w.rsplit("/", 1)[-1]] = {"frames": n, "ulaw_snr_db": round(float(snr), 1), "jaw": jaw,
                                 "worst_corr": min((v["corr"] for v in per.values()), default=None),
                                 "worst_mean_abs_over_range": max((v["mean_abs"] / v["range"] for v in per.values()), default=None),
                                 "params": per}
print(json.dumps(res, indent=1))
