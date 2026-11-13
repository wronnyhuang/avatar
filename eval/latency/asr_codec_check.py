#!/usr/bin/env python3
"""Does the 16 kHz mu-law mic change the server's intent ASR? Feed the same mic WAV to app/asr_worker.py twice, in 20 ms
chunks as the server does: (a) int16 24 kHz as before, (b) the page's path -- 24 kHz -> 16 kHz windowed-sinc (same taps as
page.html down16) -> mu-law -> server decode -> wire.Up16to24 -> 24 kHz. Compares texts and the audio time ('a') at which
each partial/final first shows its text.

    python asr_codec_check.py --app <wt>/app --wav mic.wav --python <interpreter with vosk> --model <vosk dir>
"""
import argparse
import json
import math
import struct
import subprocess
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--app", required=True)
ap.add_argument("--wav", required=True)
ap.add_argument("--python", required=True)
ap.add_argument("--model", required=True)
a = ap.parse_args()
sys.path.insert(0, a.app)
import wire  # noqa: E402


def down16(x):          # page.html DS/down16, ported 1:1
    T, H, fc = 16, 8, 7200 / 24000

    def mk(frac):
        h = np.zeros(T)
        for k in range(T):
            xx = k - (H - 1) - frac
            w = 0.5 + 0.5 * math.cos(math.pi * xx / (H + 0.5))
            h[k] = (2 * fc if xx == 0 else math.sin(2 * math.pi * fc * xx) / (math.pi * xx)) * w
        return h / h.sum()
    h0, h1 = mk(0), mk(0.5)
    tail, nxt, out = np.zeros(T), float(T), []
    for i in range(0, len(x), 480):
        buf = np.concatenate([tail, x[i:i + 480].astype(float)])
        p, res = nxt, []
        while math.floor(p) + H < len(buf):
            i0 = math.floor(p)
            h = h1 if p - i0 else h0
            res.append(float(h @ buf[i0 - (H - 1):i0 - (H - 1) + T]))
            p += 1.5
        tail, nxt = buf[-T:], p - (len(buf) - T)
        out.append(np.clip(np.rint(res), -32768, 32767).astype("<i2"))
    return out


def run(chunks):
    p = subprocess.Popen([a.python, "-I", f"{a.app}/asr_worker.py", "--model", a.model], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=False)
    for c in chunks:
        p.stdin.write(b"A" + struct.pack("<I", len(c)) + c)
    p.stdin.write(b"Q" + struct.pack("<I", 0))
    p.stdin.close()
    lines = [json.loads(l) for l in p.stdout.read().decode().splitlines() if l.strip()]
    p.wait()
    finals = [(l["a"], l["text"]) for l in lines if l.get("t") == "f" and l.get("text")]
    first_seen = {}
    for l in lines:
        if l.get("t") in ("p", "f") and l.get("text"):
            for w in l["text"].split():
                first_seen.setdefault(w, l["a"])
    return finals, first_seen


raw = subprocess.run(["ffmpeg", "-v", "error", "-i", a.wav, "-ac", "1", "-ar", "24000", "-f", "s16le", "-"],
                     capture_output=True, check=True).stdout
x = np.frombuffer(raw, "<i2")
old = [x[i:i + 480].tobytes() for i in range(0, len(x), 480)]
up = wire.Up16to24()
new = [up(wire.ulaw_decode(wire.ulaw_encode(c.tobytes()))) for c in down16(x)]
fo, so = run(old)
fn, sn = run(new)
lag = {w: round((sn[w] - so[w]) * 1000) for w in so if w in sn}
print(json.dumps({"finals_old": fo, "finals_new": fn,
                  "word_first_seen_delta_ms": lag, "median_delta_ms": float(np.median(list(lag.values()))) if lag else None}, indent=1))
