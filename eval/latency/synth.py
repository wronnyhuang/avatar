#!/usr/bin/env python3
"""Synthetic recordings with KNOWN latencies, to validate avlat.py.

    python synth.py --out DIR [--lip-ms 120] [--container mp4|webm]   -> DIR/synth.<ext>, DIR/ann.json, DIR/truth.json
    python synth.py --selftest DIR                                    -> builds several cases, runs avlat, asserts

The picture: a cartoon head (ellipse) with two eyes, a mouth whose opening follows the VOICE envelope delayed by
lip_ms, a right paw that rises at a known time after a command, eyes that jump sideways at a known time after a
camera cue, idle breathing + blinks, and a 500 ms display stall. Audio: stereo, L = user mic (syllabic noise bursts),
R = avatar voice (syllabic harmonic tone) starting a known time after each user utterance ends.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

W, H, FPS, SR = 480, 270, 24, 48000


def syllables(n, sr, rate_hz, rng, kind, loud_at="start"):
    """Syllabic signal whose envelope is LOUD at its start (a reply's onset) or at its end (an utterance's end), so the
    true onset/end is sharp and the truth is not blurred by a random syllable phase."""
    t = np.arange(n) / sr
    tref = 0.0 if loud_at == "start" else t[-1]
    env = np.clip(np.cos(2 * np.pi * rate_hz * (t - tref)) * 0.6 + 0.5, 0, 1) ** 1.5
    env *= np.clip(t / 0.03, 0, 1) * np.clip((t[-1] - t) / 0.03, 0, 1)
    if kind == "noise":
        x = rng.standard_normal(n) * 0.25
        x = np.convolve(x, np.ones(6) / 6, mode="same")
    else:
        f0 = 170 + 25 * np.sin(2 * np.pi * 0.7 * t)
        ph = 2 * np.pi * np.cumsum(f0) / sr
        x = 0.3 * (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph))
    return (x * env).astype(np.float32), env


def build(out: Path, lip_ms=120.0, container="mp4", seed=1, fa_ms=(900, 1500, 2200), cmd_ms=1800, reflex_ms=450,
          stall=(15.0, 15.5), dur=24.0):
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    n = int(dur * SR)
    L = np.zeros(n, np.float32)
    R = np.zeros(n, np.float32)
    venv = np.zeros(n, np.float32)
    utts = [(2.0, 3.5), (10.0, 11.2), (18.0, 19.4)]
    replies = []
    for (a, b), d in zip(utts, fa_ms):
        i0, i1 = int(a * SR), int(b * SR)
        x, _ = syllables(i1 - i0, SR, 3.5, rng, "noise", loud_at="end")
        L[i0:i1] += x
        s0 = b + d / 1000
        s1 = s0 + 2.6
        j0, j1 = int(s0 * SR), int(min(dur, s1) * SR)
        y, e = syllables(j1 - j0, SR, 4.2, rng, "tone")
        R[j0:j1] += y
        venv[j0:j1] = np.maximum(venv[j0:j1], e)
        replies.append((s0, s1))
    reflex_cue, cmd_cue = 6.0, utts[1][1]
    reflex_at, cmd_at = reflex_cue + reflex_ms / 1000, cmd_cue + cmd_ms / 1000
    # picture
    yy, xx = np.mgrid[0:H, 0:W]
    bg = (60 + 40 * (yy / H) + 8 * rng.standard_normal((H, W))).clip(0, 255)
    frames, times = [], []
    k = 0
    shown_t = 0.0
    while True:
        t = k / FPS
        if t >= dur:
            break
        k += 1
        if stall[0] <= t < stall[1]:
            frames.append(frames[-1])          # the display froze: same picture
            continue
        img = bg.copy()
        breath = 1.5 * np.sin(2 * np.pi * 0.25 * t)
        cy, cx = H * 0.36 + breath, W * 0.5
        head = ((xx - cx) / (W * 0.16)) ** 2 + ((yy - cy) / (H * 0.30)) ** 2 < 1
        img[head] = 190
        ex = 0.0 if t < reflex_at else W * 0.03 * min(1.0, (t - reflex_at) / 0.15)
        blink = (t % 3.1) < 0.12
        for sx in (-1, 1):
            ecx, ecy = cx + sx * W * 0.055 + ex, cy - H * 0.08
            if blink:
                eye = (np.abs(yy - ecy) < 1.5) & (np.abs(xx - ecx) < 8)
            else:
                eye = ((xx - ecx) ** 2 + (yy - ecy) ** 2) < 7 ** 2
            img[eye] = 30
        lt = t - lip_ms / 1000
        o = float(np.interp(lt, np.arange(n) / SR, venv)) if lt >= 0 else 0.0
        mh = 2 + 16 * o
        mouth = ((xx - cx) / 22) ** 2 + ((yy - (cy + H * 0.15)) / mh) ** 2 < 1
        img[mouth] = 40
        up = 0.0 if t < cmd_at else min(1.0, (t - cmd_at) / 0.3)
        py = H * 0.85 - up * H * 0.45
        paw = ((xx - W * 0.84) / 22) ** 2 + ((yy - py) / 28) ** 2 < 1
        img[paw] = 220
        frames.append(img.astype(np.uint8))
    vid = np.stack(frames)
    wavp = out / "audio.wav"
    with wave.open(str(wavp), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.stack([L, R], 1).clip(-1, 1) * 32767).astype(np.int16).tobytes())
    ext = "webm" if container == "webm" else "mp4"
    vpath = out / f"synth.{ext}"
    codec = ["-c:v", "libvpx", "-b:v", "2M", "-c:a", "libopus", "-b:a", "128k"] if ext == "webm" else \
        ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k"]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", str(FPS),
                          "-i", "-", "-i", str(wavp), *codec, "-shortest", str(vpath)], stdin=subprocess.PIPE)
    p.stdin.write(vid.tobytes()); p.stdin.close(); p.wait()
    ann = {"id": "synth", "duration_s": dur,
           "user_speech": [{"t0": a, "t1": b, "text": "synthetic"} for a, b in utts],
           "cues": [{"t": reflex_cue, "cue": "face_jump", "region": "eyes", "kind": "reflex", "window_s": [0.1, 1.5]},
                    {"t": cmd_cue, "cue": "raise your right paw", "region": "arm_r", "kind": "command", "window_s": [0.2, 4.0]}]}
    truth = {"first_audio_ms": list(fa_ms), "reflex_ms": reflex_ms, "command_ms": cmd_ms, "lip_ms": lip_ms,
             "stall_ms": round((stall[1] - stall[0]) * 1000), "fps": FPS}
    (out / "ann.json").write_text(json.dumps(ann, indent=1))
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    return vpath, out / "ann.json", truth


def selftest(root: Path):
    sys.path.insert(0, str(Path(__file__).parent))
    import avlat
    cases = [dict(lip_ms=120.0, container="mp4"), dict(lip_ms=-60.0, container="webm", seed=3, fa_ms=(650, 1200, 3100),
                                                        cmd_ms=900, reflex_ms=300),
             dict(lip_ms=0.0, container="mp4", seed=5, fa_ms=(400, 2500, 800), cmd_ms=3200, reflex_ms=700)]
    ok = True
    report = []
    for i, c in enumerate(cases):
        d = root / f"case{i}"
        v, ann, truth = build(d, **c)
        # mic from the stereo L channel AND from the annotation: both must agree
        r_ann = avlat.analyze(str(v), voice_channel=1, mic_channel=0, ann=str(ann))
        r_vad = avlat.analyze(str(v), voice_channel=1, mic_channel=0)
        fa_ann = [x["first_audio_ms"] for x in r_ann["first_audio"]]
        fa_vad = [x["first_audio_ms"] for x in r_vad["first_audio"]]
        mot = {m["kind"]: m["latency_ms"] for m in r_ann["motion"]}
        reg = {m["kind"]: m["strongest_region"] for m in r_ann["motion"]}
        lag = r_ann["lip_offset"].get("lag_ms")
        errs = {
            "first_audio_ann": max(abs(a - b) for a, b in zip(fa_ann, truth["first_audio_ms"])) if None not in fa_ann else 1e9,
            "first_audio_vad": max(abs(a - b) for a, b in zip(fa_vad, truth["first_audio_ms"])) if None not in fa_vad else 1e9,
            "reflex": abs((mot.get("reflex") or 1e9) - truth["reflex_ms"]),
            "command": abs((mot.get("command") or 1e9) - truth["command_ms"]),
            "lip": abs((lag if lag is not None else 1e9) - truth["lip_ms"]),
        }
        tol = {"first_audio_ann": 30, "first_audio_vad": 40, "reflex": 2000 / 24 + 10, "command": 2000 / 24 + 10, "lip": 20}
        passed = all(errs[k] <= tol[k] for k in tol)
        disp = r_ann["display"]
        stall_ok = disp["interval_ms"]["max"] >= truth["stall_ms"] - 50
        ok &= passed and stall_ok
        report.append({"case": i, **c, "truth": truth, "got": {"first_audio_ann": fa_ann, "first_audio_vad": fa_vad,
                       "motion": mot, "strongest": reg, "lip_ms": lag, "lip_corr": r_ann["lip_offset"].get("corr"),
                       "fps": disp.get("fps"), "max_interval": disp["interval_ms"]["max"]},
                       "errors": errs, "pass": passed and stall_ok})
        print(json.dumps(report[-1]), flush=True)
    (root / "selftest.json").write_text(json.dumps(report, indent=1))
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--lip-ms", type=float, default=120)
    ap.add_argument("--container", default="mp4")
    ap.add_argument("--selftest")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(Path(a.selftest)) else 1)
    build(Path(a.out), lip_ms=a.lip_ms, container=a.container)
