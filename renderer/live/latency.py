#!/usr/bin/env python3
"""Latency proof for the live loop: isolated syllables from silence -> wall clock from the chunk that carried each onset to
the presented frame whose mouth opens, read two ways: from lip_open in the released params, and from pixels in the mouth
crop of the frames actually presented (decoded back from the --record file, one frame per tick).

    python -m live.latency make-wav --out onset.wav            # 12 x pa/ma syllables from silence (generated, or
                                                               # your own --snippets DIR with tts_pa.pcm / tts_ma.pcm)
    python -m live.run_live --character orb ... --wav onset.wav --record run.mp4 --log run.json
    python -m live.latency analyze --log run.json --video run.mp4 --onsets onset.json --crop auto --character orb

Onset = the first sample whose 1 ms RMS reaches -40 dBFS (the snippets rise from -62 to -25 dB within 10 ms).
"Mouth opens" = the first tick in [onset - 300 ms, onset + 600 ms] whose shown frame has lip_open above 30 % of that
syllable's peak (params), or whose mouth crop differs from the closed-mouth crop (the tick 300 ms before the onset) by
more than 30 % of that syllable's peak difference (pixels).  The lips are built to lead the voice (the animator's lead
of 2 frames), so the mouth may open before the onset itself is played.  Aggregates count only "clean" syllables, whose
whole window was presented without a drop.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from anim.schema import IDX, NPARAMS, SAMPLE_RATE as SR     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def onset_of(x, db=-40.0):
    env = np.sqrt(np.convolve(x.astype(float) ** 2, np.ones(24) / 24, "same"))
    return int(np.argmax(20 * np.log10(env / 32768 + 1e-9) > db))


def synth_syllable(kind: str) -> np.ndarray:
    """A generated 'pa' / 'ma'-like syllable (int16, SR): 60 ms of silence, then a plosive click ('pa') or a short nasal
    hum ('ma') leading into a 300 ms voiced vowel (a 140 Hz glottal pulse train through two formant-ish resonators),
    with a sharp rise (well under 10 ms from -62 to -25 dBFS) so the onset detector has a clean edge."""
    rng = np.random.default_rng(7 if kind == "pa" else 11)
    pre = np.zeros(int(0.06 * SR))
    t = np.arange(int(0.30 * SR)) / SR
    f0 = 140.0 * (1.0 - 0.15 * t / t[-1])
    phase = 2 * np.pi * np.cumsum(f0) / SR
    src = sum(np.sin(k * phase) / k for k in range(1, 25))
    vowel = src * (0.6 + 0.4 * np.sin(2 * np.pi * 750 * t)) * (0.7 + 0.3 * np.sin(2 * np.pi * 1200 * t))
    env = np.minimum(1.0, t / 0.004) * np.exp(-np.maximum(0.0, t - 0.18) / 0.05)
    vowel = vowel * env
    if kind == "pa":
        burst = rng.standard_normal(int(0.012 * SR)) * np.exp(-np.arange(int(0.012 * SR)) / (0.003 * SR))
        head = np.concatenate([burst * 0.8, np.zeros(int(0.02 * SR))])
    else:
        tm = np.arange(int(0.05 * SR)) / SR
        head = np.sin(2 * np.pi * 140 * tm) * np.minimum(1.0, tm / 0.004) * 0.35
    y = np.concatenate([pre, head, vowel, np.zeros(int(0.05 * SR))])
    y = y / (np.abs(y).max() + 1e-9) * 0.5
    return (y * 32767).astype(np.int16)


def make_wav(a):
    """Snippets: <snippets>/tts_pa.pcm and tts_ma.pcm (raw int16 mono at 24 kHz) when present, else generated."""
    snips = {}
    for k in ("pa", "ma"):
        p = os.path.join(a.snippets, f"tts_{k}.pcm") if a.snippets else ""
        snips[k] = np.fromfile(p, np.int16) if p and os.path.isfile(p) else synth_syllable(k)
    n_total = int((a.lead + a.n * a.period + 1.0) * SR)
    y = np.zeros(n_total, np.int16)
    onsets, labels = [], []
    for i in range(a.n):
        k = ("pa", "ma")[i % 2]
        s = snips[k]
        at = int((a.lead + i * a.period) * SR)
        start = at - onset_of(s)
        y[start:start + s.size] = s
        onsets.append(at); labels.append(k)
    with wave.open(a.out, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(y.tobytes())
    meta = {"wav": a.out, "onset_samples": onsets, "labels": labels, "period_s": a.period, "lead_s": a.lead,
            "source": "pa/ma syllables (generated unless --snippets holds tts_pa.pcm / tts_ma.pcm), pcm 24 kHz, placed "
                      "so each -40 dBFS onset lands on its sample"}
    json.dump(meta, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "onset_samples"}))


def auto_crop(character, w, h, backend="v2", opts=None):
    """Mouth box: render rest vs wide open with the same index; bbox of the blurred difference above half its peak,
    padded by 1/7."""
    from live.backend_api import get_backend, REGISTRY
    from live.pool import _load
    cls = get_backend(backend) if backend in REGISTRY else _load(backend)
    be = cls(character, w, h, **(opts or {}))
    if hasattr(be, "mouth_box"):
        return be.mouth_box()
    rest = np.zeros(NPARAMS, np.float32)
    op = rest.copy()
    for k in ("lip_open", "teeth_gap", "lip_lower", "lip_upper", "jaw"):
        op[IDX[k]] = 1.0
    a = be.render(0, rest).astype(np.float32)
    b = be.render(0, op).astype(np.float32)
    be.close()
    import cv2
    d = np.abs(a - b).max(axis=2)
    k = max(9, (w // 40) | 1)           # blurred: the opening itself, not fur/shading noise around it
    db = cv2.blur(d, (k, k))
    ys, xs = np.nonzero(db > 0.5 * db.max())
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = max(4, (x1 - x0) // 7), max(4, (y1 - y0) // 7)
    x0, y0 = max(0, x0 - px), max(0, y0 - py)
    x1, y1 = min(w - 1, x1 + px), min(h - 1, y1 + py)
    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)


def read_crops(video, box):
    import cv2
    x, y, w, h = box
    cap = cv2.VideoCapture(video)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f[y:y + h, x:x + w, ::-1].astype(np.float32))
    return out


def analyze(a):
    log = json.load(open(a.log))
    on = json.load(open(a.onsets))
    cfg, s = log["config"], log["summary"]
    fps = cfg["fps"]
    D = s["delay_ms"]
    ends = np.array(log["chunks"]["end_sample"])
    arr = np.array(log["chunks"]["arrival_ms"])
    frames = {f["j"]: f for f in log["frames"]}
    ticks = sorted(frames)
    lip = {j: frames[j].get("lip_open") for j in ticks}
    shown_lip = np.array([lip.get(frames[j].get("shown"), 0.0) or 0.0 if frames[j].get("shown") is not None else 0.0
                          for j in ticks])
    present = np.array([frames[j].get("present_ms", np.nan) for j in ticks])
    crops = None
    box = None
    if a.video:
        if a.crop == "auto":
            opts = dict(kv.split("=", 1) for kv in a.opt)
            opts = {k: (int(v) if v.isdigit() else v) for k, v in opts.items()}
            box = auto_crop(cfg["character"], cfg["width"], cfg["height"], a.backend or cfg["backend"], opts)
        else:
            box = tuple(int(v) for v in a.crop.split(","))
        crops = read_crops(a.video, box)
    rows = []
    for k, s_on in enumerate(on["onset_samples"]):
        t_on = 1000 * s_on / SR
        c = int(np.searchsorted(ends, s_on, side="right"))
        arr_on = float(arr[min(c, len(arr) - 1)])
        play_on = t_on + D
        lo = int(np.ceil((t_on - 300) * fps / 1000)); hi = int((t_on + 600) * fps / 1000)
        win = [j for j in range(lo, hi + 1) if 0 <= j < len(ticks)]
        clean = all(frames[j]["status"] == "presented" for j in win)
        r = {"k": k, "label": on["labels"][k], "onset_ms": round(t_on, 2), "chunk_arrival_ms": round(arr_on, 2),
             "onset_played_ms": round(play_on, 2), "onset_play_latency_ms": round(play_on - arr_on, 2), "clean": clean}
        if win:
            L = shown_lip[win]
            th = 0.3 * L.max()
            hit = next((j for j in win if shown_lip[j] > th), None)
            if hit is not None and L.max() > 0.05:
                r.update(params_open_tick=hit, params_open_depicts_ms=round(1000 * frames[frames[hit]["shown"]]["j"] / fps, 2)
                         if frames[hit].get("shown") is not None else None,
                         params_open_present_ms=round(float(present[hit]), 2),
                         params_latency_ms=round(float(present[hit]) - arr_on, 2),
                         params_vs_sound_ms=round(float(present[hit]) - play_on, 2), lip_open_peak=round(float(L.max()), 3))
        if crops is not None and win:
            ref_j = max(0, int(round((t_on - 300) * fps / 1000)))
            ref = crops[ref_j]
            dd = np.array([np.abs(crops[j] - ref).mean() if j < len(crops) else 0.0 for j in win])
            th = 0.3 * dd.max()
            idx = next((i for i, v in enumerate(dd) if v > th), None)
            if idx is not None:
                hit = win[idx]
                r.update(pixel_open_tick=hit, pixel_open_present_ms=round(float(present[hit]), 2),
                         pixel_latency_ms=round(float(present[hit]) - arr_on, 2),
                         pixel_vs_sound_ms=round(float(present[hit]) - play_on, 2), pixel_peak_diff=round(float(dd.max()), 2))
        rows.append(r)

    def agg(key, label=None):
        # only syllables whose whole window [onset - 300, onset + 600 ms] was presented: a dropped frame repeats an
        # older picture, and the measure would time the drop instead of the loop
        v = [r[key] for r in rows if key in r and r["clean"] and (label is None or r["label"] == label)]
        return None if not v else {"mean": round(float(np.mean(v)), 1), "min": round(float(np.min(v)), 1),
                                    "max": round(float(np.max(v)), 1), "n": len(v)}
    res = {"log": a.log, "video": a.video, "crop": box, "delay_ms": D, "fps": fps,
           "clean_onsets": f"{sum(r['clean'] for r in rows)}/{len(rows)}",
           "onset_chunk_to_mouth_open_params_ms": agg("params_latency_ms"),
           "onset_chunk_to_mouth_open_pixels_ms": agg("pixel_latency_ms"),
           "onset_chunk_to_onset_played_ms": agg("onset_play_latency_ms"),
           "mouth_open_minus_sound_params_ms": agg("params_vs_sound_ms"),
           "mouth_open_minus_sound_pixels_ms": agg("pixel_vs_sound_ms"),
           "by_label": {lab: {k: agg(k, lab) for k in ("params_latency_ms", "pixel_latency_ms", "onset_play_latency_ms",
                                                       "params_vs_sound_ms", "pixel_vs_sound_ms")}
                        for lab in sorted(set(on["labels"]))},
           "per_onset": rows}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_onset"}, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-wav")
    m.add_argument("--out", required=True)
    m.add_argument("--snippets", default=os.path.join(HERE, "assets"), help="dir with tts_pa.pcm / tts_ma.pcm (optional)")
    m.add_argument("--n", type=int, default=12)
    m.add_argument("--period", type=float, default=1.25)
    m.add_argument("--lead", type=float, default=1.0)
    z = sub.add_parser("analyze")
    z.add_argument("--log", required=True)
    z.add_argument("--onsets", required=True)
    z.add_argument("--video")
    z.add_argument("--crop", default="auto", help="auto | x,y,w,h")
    z.add_argument("--backend", default=None, help="backend for --crop auto (default: the run's)")
    z.add_argument("--opt", action="append", default=[], help="backend opts for --crop auto, e.g. backend=cpu")
    z.add_argument("--out")
    a = ap.parse_args(argv)
    make_wav(a) if a.cmd == "make-wav" else analyze(a)


if __name__ == "__main__":
    main()
