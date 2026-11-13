#!/usr/bin/env python3
"""Latency analysis of ONE run: a recording of the rendered avatar (video) + the audio the page played, plus the
clip's annotations / the runner's events, plus (optional) the server's session JSON.

    python avlat.py --video rec.webm [--voice-channel 1 --mic-channel 0 | --mic mic.wav] \
        [--ann clip_annotation.json] [--events events.json] [--session session.json] [--regions regions.json] \
        [--out metrics.json]

Every time below is on the RECORDING's own timeline (seconds from the first sample/frame of the file), which for the
eval runner's avatar.mp4 is clip time (t = 0 = the clip started feeding the fake camera/mic).

What it reports (each with the method, so a table row can cite it):
  first_audio      end of each user utterance (annotation user_speech t1, or the mic channel's VAD) -> first voiced
                   onset of the avatar's voice after it (voice channel VAD, 10 ms hop).  p50/p95/each.
  motion           each cue/command (t, region) -> first sustained motion onset in that region of the rendered video
                   (frame-difference energy above baseline median + k*MAD for >= sustain frames), plus which region
                   responded most (to check it was the RIGHT region).
  lip_offset_ms    cross-correlation of the mouth region's luminance (auto-located: the blocks whose brightness best
                   follows the voice envelope) with the voice envelope; positive = picture LATER than sound.
  display          distinct-frame timestamps -> fps, frame-interval p50/p95/max, stalls > 150 ms.
  session          (if given) server/page numbers: page fps, sync, poll rtt, renderer frame time, kbps.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

# Regions on the 16:9 rendered frame, normalized (x0, y0, x1, y1). Medium close-up per CONTRACT.md: head upper middle,
# paws to the sides / lower third. Override per character with --regions. Head-type regions exclude the mouth box so
# speech lip motion is never read as a head/eye/ear response.
DEFAULT_REGIONS = {
    "any":   [[0.00, 0.00, 1.00, 1.00]],
    "head":  [[0.28, 0.00, 0.72, 0.68]],
    "face":  [[0.32, 0.12, 0.68, 0.66]],
    "eyes":  [[0.34, 0.18, 0.66, 0.44]],
    "mouth": [[0.38, 0.42, 0.62, 0.66]],
    "ears":  [[0.18, 0.00, 0.82, 0.24]],
    "arms":  [[0.00, 0.30, 0.30, 1.00], [0.70, 0.30, 1.00, 1.00]],
    "arm_l": [[0.00, 0.30, 0.30, 1.00]],
    "arm_r": [[0.70, 0.30, 1.00, 1.00]],
    "body":  [[0.22, 0.62, 0.78, 1.00]],
}
EXCLUDE_MOUTH = {"head", "face", "eyes", "ears", "any"}
ANALYSIS_W, ANALYSIS_H = 192, 108


# ----------------------------------------------------------------------------------------------------------- media IO
def probe(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def read_video(path: str, crop: str | None = None, w: int = ANALYSIS_W, h: int = ANALYSIS_H):
    """-> (pts seconds array, frames uint8 [N,h,w]) with every DECODED frame and its own timestamp (VFR kept)."""
    pts_txt = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "frame=best_effort_timestamp_time,pts_time", "-of", "csv=p=0", path],
                             capture_output=True, text=True, check=True).stdout
    pts = []
    for line in pts_txt.splitlines():
        vals = [v for v in line.split(",") if v and v != "N/A"]
        if vals:
            pts.append(float(vals[0]))
    vf = (f"crop={crop}," if crop else "") + f"scale={w}:{h}:flags=area,format=gray"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0", "-fps_mode", "passthrough", "-vf", vf,
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    n = len(raw) // (w * h)
    frames = np.frombuffer(raw[: n * w * h], np.uint8).reshape(n, h, w)
    if len(pts) < n:               # raw elementary streams carry no timestamps: CFR guess (callers pass frame_times)
        pts = list(np.arange(n) / 24.0)
    pts = np.array(pts[:n], float)
    return pts, frames


def read_audio(path: str, sr: int = 16000):
    """-> (start_time_s, sr, samples float32 [N, ch]) of the first audio stream, every channel."""
    info = probe(path)
    a = next((s for s in info["streams"] if s.get("codec_type") == "audio"), None)
    if a is None:
        return None
    ch = int(a.get("channels") or 1)
    st = float(a.get("start_time") or 0.0)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0", "-ar", str(sr), "-ac", str(ch),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    return st, sr, x.reshape(-1, ch)


# ---------------------------------------------------------------------------------------------------------------- VAD
def rms_db(x: np.ndarray, sr: int, hop_s: float = 0.01, win_s: float = 0.02):
    hop, win = int(sr * hop_s), int(sr * win_s)
    if len(x) < win:
        return np.zeros(0), np.zeros(0)
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    r = np.sqrt(np.mean(x[idx] ** 2, axis=1) + 1e-12)
    t = (np.arange(n) * hop + win / 2) / sr
    return t, 20 * np.log10(r)


def vad_segments(x: np.ndarray, sr: int, t0: float = 0.0, floor_db: float = -50.0, above_noise_db: float = 18.0,
                 min_len_s: float = 0.08, merge_gap_s: float = 0.25, abs_min_db: float = -48.0):
    """Voiced segments [(start, end)] (seconds, recording timeline). Threshold = max(abs_min_db, noise + above_noise_db)
    where noise = 10th percentile of frame dB (digital silence clamps to floor_db)."""
    t, db = rms_db(x, sr)
    if not len(t):
        return [], None
    db = np.maximum(db, floor_db - 40)
    noise = float(np.percentile(db, 10))
    thr = max(abs_min_db, min(noise, floor_db) + above_noise_db) if noise < floor_db else max(abs_min_db, noise + above_noise_db)
    on = db > thr
    segs, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            segs.append([t[i] - 0.01, t[j - 1] + 0.01])
            i = j
        else:
            i += 1
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = s[1]
        else:
            merged.append(list(s))
    merged = [(a + t0, b + t0) for a, b in merged if b - a >= min_len_s]
    return merged, thr


def envelope(x: np.ndarray, sr: int, rate: float = 100.0):
    hop = int(sr / rate)
    win = hop * 2
    n = max(0, 1 + (len(x) - win) // hop)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    e = np.sqrt(np.mean(x[idx] ** 2, axis=1))
    t = (np.arange(n) * hop + win / 2) / sr
    return t, e


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summary(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "p50": round(pct(xs, .5), 1), "p95": round(pct(xs, .95), 1), "min": round(min(xs), 1),
            "max": round(max(xs), 1), "mean": round(float(np.mean(xs)), 1)}


# ----------------------------------------------------------------------------------------------------- first audio
def utterances(segs, gap_s: float = 0.7):
    """User speech segments -> utterances (a pause shorter than gap_s is inside one utterance)."""
    out = []
    for a, b in segs:
        if out and a - out[-1][1] < gap_s:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [tuple(u) for u in out]


def first_audio(user_utts, voice_segs, max_s: float = 15.0):
    rows = []
    for i, (u0, u1) in enumerate(user_utts):
        nxt = user_utts[i + 1][0] if i + 1 < len(user_utts) else u1 + max_s
        active = any(a < u1 - 0.05 and b > u1 + 0.05 for a, b in voice_segs)
        onset = next((a for a, b in voice_segs if a >= u1 - 0.05 and a <= min(nxt, u1 + max_s)), None)
        rows.append({"user_start": round(u0, 3), "user_end": round(u1, 3),
                     "voice_onset": round(onset, 3) if onset is not None else None,
                     "first_audio_ms": round((onset - u1) * 1000) if onset is not None else None,
                     "avatar_was_talking": active})
    return rows


# ----------------------------------------------------------------------------------------------------------- motion
def region_masks(h: int, w: int, regions: dict, mouth_box=None):
    masks = {}
    mouth = np.zeros((h, w), bool)
    if mouth_box is not None:
        x0, y0, x1, y1 = mouth_box
        mouth[int(y0 * h):int(math.ceil(y1 * h)), int(x0 * w):int(math.ceil(x1 * w))] = True
    for name, boxes in regions.items():
        m = np.zeros((h, w), bool)
        for x0, y0, x1, y1 in boxes:
            m[int(y0 * h):int(math.ceil(y1 * h)), int(x0 * w):int(math.ceil(x1 * w))] = True
        if name in EXCLUDE_MOUTH and mouth.any():
            m &= ~mouth
        if name == "mouth" and mouth.any():
            m = mouth.copy()
        masks[name] = m
    return masks


def frame_diffs(frames: np.ndarray):
    f = frames.astype(np.int16)
    d = np.zeros(frames.shape, np.float32)
    d[1:] = np.abs(f[1:] - f[:-1])
    return d


def distinct(pts, frames, px_thr: int = 8, min_px: int = 3):
    """Indices of frames whose content differs from the previous kept frame (>= min_px pixels changed by more than
    px_thr gray levels): a CFR recording of a 24 fps canvas repeats frames; only changes count as displayed frames."""
    keep = [0]
    last = frames[0].astype(np.int16)
    for i in range(1, len(frames)):
        cur = frames[i].astype(np.int16)
        if int((np.abs(cur - last) > px_thr).sum()) >= min_px:
            keep.append(i)
            last = cur
    return np.array(keep, int)


def motion_onsets(pts, frames, masks, cues, k: float = 6.0, baseline_s: float = 0.5, max_s: float = 6.0,
                  v_floor: float = 0.4, p_floor: float = 1.0, rel: float = 0.3, hold_s: float = 0.3, hold_frac: float = 0.7):
    """For each cue {t, region, kind, window_s}:
      1. ESTABLISHED change: the first frame after t whose region differs from the pre-cue look (median image of the
         baseline_s before the cue) by more than max(median + p_floor, median + k*MAD, median + rel*(peak-median)) and
         stays above for >= hold_frac of the next hold_s (a blink / speech flap returns within ~150 ms; slow breathing
         drift stays under rel*peak);
      2. ONSET = walk back from there through the contiguous run of MOVING frames (frame-to-frame change above the
         pre-cue median + k*MAD): the first frame of that run is when the motion visibly started.
    Search = [t, t + window_s[1] or max_s]."""
    F = frames.astype(np.float32)
    rows = []
    for c in cues:
        t = float(c["t"])
        reg = c.get("region") or "any"
        if reg not in masks:
            reg = {"arm": "arms", "paw": "arms", "paws": "arms", "hands": "arms", "tail": "body", "nose": "face",
                   "brow": "eyes", "brows": "eyes", "gaze": "eyes"}.get(reg, "any")
        hi = t + (float(c["window_s"][1]) if c.get("window_s") else max_s)
        res = {"t": round(t, 3), "kind": c.get("kind"), "cue": c.get("cue") or c.get("command"), "region": reg}
        bidx = np.where((pts >= t - baseline_s) & (pts < t))[0]
        bidx = bidx[bidx >= 1]
        sidx = np.where((pts >= t) & (pts <= hi + hold_s))[0]
        if len(bidx) < 3 or len(sidx) < 3:
            res.update({"onset": None, "latency_ms": None, "why": "no baseline frames"})
            rows.append(res)
            continue
        ref = np.median(F[bidx], axis=0)
        allidx = np.concatenate([bidx, sidx])
        per = {}
        for name, m in masks.items():
            if not m.any() or (name == "any" and reg != "any"):
                continue
            Fm = F[allidx][:, m]
            dev = np.abs(Fm - ref[m][None]).mean(axis=1)
            vel = np.r_[np.abs(F[allidx[0]][m] - F[allidx[0] - 1][m]).mean(), np.abs(np.diff(Fm, axis=0)).mean(axis=1)]
            nb = len(bidx)
            bd, bv = dev[:nb], vel[:nb]
            medd, madd = float(np.median(bd)), float(np.median(np.abs(bd - np.median(bd)))) * 1.4826
            medv, madv = float(np.median(bv)), float(np.median(np.abs(bv - np.median(bv)))) * 1.4826
            st = pts[allidx]
            inwin = (st >= t) & (st <= hi)
            peak = float(dev[inwin].max()) if inwin.any() else medd
            thr_d = max(medd + p_floor, medd + k * madd, medd + rel * (peak - medd))
            thr_v = max(medv + v_floor, medv + k * madv)
            est = None
            for i in np.where(inwin & (dev > thr_d))[0]:
                w = (st >= st[i]) & (st <= st[i] + hold_s)
                if (dev[w] > thr_d).mean() >= hold_frac:
                    est = i
                    break
            onset, resp = None, None
            if est is not None:
                j = est
                while j - 1 >= nb and vel[j] > thr_v and vel[j - 1] > thr_v:
                    j -= 1
                if vel[j] <= thr_v and j + 1 <= est:
                    j += 1
                onset = float(st[j])
                resp = (float(np.median(dev[(st >= st[est]) & (st <= st[est] + hold_s)])) - medd) / max(madd, 0.3)
            per[name] = {"onset": onset, "thr_d": round(thr_d, 3), "thr_v": round(thr_v, 3),
                         "resp": round(resp, 2) if resp else None}
        r = per.get(reg) or {}
        res["onset"] = round(r["onset"], 3) if r.get("onset") is not None else None
        res["latency_ms"] = round((r["onset"] - t) * 1000) if r.get("onset") is not None else None
        ranked = sorted(((v["resp"] or 0, nm) for nm, v in per.items() if nm not in ("mouth", "any", "head", "arms")), reverse=True)
        res["strongest_region"] = ranked[0][1] if ranked and ranked[0][0] > 0 else None
        res["regions"] = {nm: {"onset_ms": round((v["onset"] - t) * 1000) if v["onset"] is not None else None,
                               "resp": v["resp"]} for nm, v in per.items()}
        exp = c.get("window_s")
        if exp and res["latency_ms"] is not None:
            res["in_window"] = exp[0] * 1000 <= res["latency_ms"] <= exp[1] * 1000
        rows.append(res)
    return rows


# -------------------------------------------------------------------------------------------------------- lip offset
def lip_offset(pts, frames, env_t, env, voice_segs, max_lag_s: float = 0.4, rate: float = 100.0, grid=(32, 18),
               mouth_hint=None):
    """Auto-locate the mouth (blocks whose luminance best tracks the voice envelope at some lag) and return the lag
    maximizing the cross-correlation. Positive lag = the picture changes AFTER the sound (video late)."""
    if not voice_segs or len(pts) < 20 or len(env) < 50:
        return {"lag_ms": None, "why": "no speech or too short"}
    t0 = max(pts[0], env_t[0]) + max_lag_s
    t1 = min(pts[-1], env_t[-1]) - max_lag_s
    grid_t = np.arange(t0, t1, 1.0 / rate)
    if len(grid_t) < 100:
        return {"lag_ms": None, "why": "overlap too short"}
    # speech mask: voiced segments +- 0.5 s (silence between turns is idle motion, not lips)
    mask = np.zeros(len(grid_t), bool)
    for a, b in voice_segs:
        mask |= (grid_t >= a - 0.5) & (grid_t <= b + 0.5)
    if mask.sum() < 100:
        return {"lag_ms": None, "why": "too little speech"}
    n, h, w = frames.shape
    gx, gy = grid
    bh, bw = h // gy, w // gx
    blocks = frames[:, : gy * bh, : gx * bw].reshape(n, gy, bh, gx, bw).mean(axis=(2, 4)).reshape(n, gy * gx)
    # interpolate each block series onto the grid (hold = what is on screen at that instant)
    # linear between frame instants: a frame's content depicts its own pts (holding it would bias +half a frame)
    B = np.stack([np.interp(grid_t, pts, blocks[:, b]) for b in range(blocks.shape[1])]).astype(np.float64)  # [blocks, T]
    E = np.interp(grid_t, env_t, env)
    lags = np.arange(-int(max_lag_s * rate), int(max_lag_s * rate) + 1)

    def z(v, m):
        v = v - v[..., m].mean(axis=-1, keepdims=True)
        s = v[..., m].std(axis=-1, keepdims=True) + 1e-9
        return v / s

    # correlate block(t) with env(t - lag): the picture at t shows audio from t - lag
    def xcorr(Bz, Ez):
        out = np.zeros((Bz.shape[0], len(lags)))
        for j, L in enumerate(lags):
            if L >= 0:
                b, e, m = Bz[:, L:], Ez[: len(Ez) - L], mask[L:] & mask[: len(mask) - L]
            else:
                b, e, m = Bz[:, : L], Ez[-L:], mask[: L] & mask[-L:]
            if m.sum() < 50:
                continue
            out[:, j] = (b[:, m] @ e[m]) / m.sum()
        return out
    Bz, Ez = z(B, mask), z(E[None, :], mask)[0]
    C = xcorr(Bz, Ez)
    if mouth_hint is not None:
        x0, y0, x1, y1 = mouth_hint
        sel = np.zeros((gy, gx), bool)
        sel[int(y0 * gy):int(math.ceil(y1 * gy)), int(x0 * gx):int(math.ceil(x1 * gx))] = True
        C = np.where(sel.reshape(-1)[:, None], C, 0)
    peak = np.abs(C).max(axis=1)
    best = int(np.argmax(peak))
    cand = np.where(peak >= 0.75 * peak[best])[0]
    by, bx = divmod(best, gx)
    cand = [c for c in cand if abs(c // gx - by) <= 3 and abs(c % gx - bx) <= 4][:16]
    signs = np.sign(C[cand, np.argmax(np.abs(C[cand]), axis=1)])
    S = (Bz[cand] * signs[:, None]).mean(axis=0, keepdims=True)
    cs = xcorr(z(S, mask), Ez)[0]
    j = int(np.argmax(cs))
    frac = 0.0
    if 0 < j < len(cs) - 1:
        a, b_, c = cs[j - 1], cs[j], cs[j + 1]
        den = a - 2 * b_ + c
        frac = 0.5 * (a - c) / den if den else 0.0
    lag = (lags[j] + frac) / rate
    ys = [c // gx for c in cand]
    xs = [c % gx for c in cand]
    box = [min(xs) / gx, min(ys) / gy, (max(xs) + 1) / gx, (max(ys) + 1) / gy]
    edge = j in (0, len(cs) - 1)
    return {"lag_ms": round(lag * 1000, 1), "corr": round(float(cs[j]), 3), "mouth_box": [round(v, 3) for v in box],
            "blocks": len(cand), "at_search_edge": edge, "speech_s": round(mask.sum() / rate, 1)}


# ----------------------------------------------------------------------------------------------------------- display
def display_stats(pts, frames):
    keep = distinct(pts, frames)
    tt = pts[keep]
    if len(tt) < 3:
        return {"frames": int(len(tt))}
    dt = np.diff(tt) * 1000
    dur = tt[-1] - tt[0]
    return {"distinct_frames": int(len(tt)), "decoded_frames": int(len(pts)), "span_s": round(float(dur), 2),
            "fps": round((len(tt) - 1) / dur, 2) if dur > 0 else None,
            "interval_ms": {"p50": round(float(np.percentile(dt, 50)), 1), "p95": round(float(np.percentile(dt, 95)), 1),
                            "p99": round(float(np.percentile(dt, 99)), 1), "max": round(float(dt.max()), 1)},
            "stalls_over_150ms": int((dt > 150).sum()), "stall_ms_total": round(float(dt[dt > 150].sum()), 0)}


# ---------------------------------------------------------------------------------------------------------- session
def session_numbers(s: dict) -> dict:
    if not s:
        return {}
    out = {}
    for k in ("first_audio_ms", "hub_user_end_to_vendor_ms", "fps_shown", "sync_offset_ms_p50", "sync_offset_abs_p95_ms",
              "poll_rtt_ms_p50", "poll_rtt_ms_p95", "late_frames", "dropped_frames", "audio_underruns", "delay_ms", "usd"):
        if k in s:
            out[k] = s[k]
    r = (s.get("render") or {})
    if isinstance(r.get("stats"), dict):
        out["render_stats_last"] = {k: r["stats"].get(k) for k in ("fps", "render_ms", "render_p95", "encode_ms", "encode_p95",
                                                                     "lag_ms", "lag_p95", "kbps", "late", "dropped")}
    return out


# ------------------------------------------------------------------------------------------------------------- main
def analyze(video, voice_channel=None, mic_channel=None, mic=None, mic_offset_s=0.0, ann=None, events=None,
            session=None, regions=None, crop=None, cue_k=6.0, audio=None, frame_times=None, mouth_hint=None):
    """audio: a separate file with the avatar voice (t = 0 at its first sample) instead of the video's own track;
    frame_times: list of seconds, one per decoded frame, overriding the container timestamps (e.g. a raw .h264
    capture whose frames carry their playback-timeline t)."""
    res = {"video": video, "method": {}}
    pts, frames = read_video(video, crop)
    if frame_times is not None:
        n = min(len(frames), len(frame_times))
        pts, frames = np.asarray(frame_times[:n], float), frames[:n]
        order = np.argsort(pts, kind="stable")
        pts, frames = pts[order], frames[order]
    res["display"] = display_stats(pts, frames)
    aud = read_audio(audio or video)
    voice_segs, user_segs, env_t, env = [], [], np.zeros(0), np.zeros(0)
    if aud:
        st, sr, X = aud
        ch = X.shape[1]
        vc = voice_channel if voice_channel is not None else (1 if ch >= 2 and mic_channel == 0 else 0)
        v = X[:, min(vc, ch - 1)]
        voice_segs, vthr = vad_segments(v, sr, st)
        env_t, env = envelope(v, sr)
        env_t = env_t + st
        res["voice"] = {"channel": vc, "threshold_db": round(vthr, 1) if vthr is not None else None,
                        "segments": [[round(a, 3), round(b, 3)] for a, b in voice_segs]}
        if mic_channel is not None and mic_channel < ch:
            user_segs, uthr = vad_segments(X[:, mic_channel], sr, st)
            res["mic"] = {"channel": mic_channel, "threshold_db": round(uthr, 1) if uthr is not None else None}
    if mic:
        m = read_audio(mic)
        if m:
            st, sr, X = m
            user_segs, _ = vad_segments(X[:, 0], sr, st + mic_offset_s)
            res["mic"] = {"file": mic, "offset_s": mic_offset_s}
    a = json.load(open(ann)) if ann else {}
    ev = json.load(open(events)) if events else {}
    spans = a.get("user_speech") or ev.get("user_speech")
    if spans:
        utts = utterances([(float(s["t0"]), float(s["t1"])) for s in spans])
        res["method"]["user_end"] = "annotation/runner user_speech t1"
    else:
        utts = utterances(user_segs)
        res["method"]["user_end"] = "mic VAD (10 ms hop, noise+18 dB)"
    res["first_audio"] = first_audio(utts, voice_segs)
    res["first_audio_ms"] = summary([r["first_audio_ms"] for r in res["first_audio"] if not r["avatar_was_talking"]])
    lip = lip_offset(pts, frames, env_t, env, voice_segs, mouth_hint=mouth_hint)
    res["lip_offset"] = lip
    mouth_box = lip.get("mouth_box") if lip.get("corr") and abs(lip["corr"]) > 0.25 else None
    regs = dict(DEFAULT_REGIONS)
    if regions:
        regs.update(json.load(open(regions)))
    masks = region_masks(frames.shape[1], frames.shape[2], regs, mouth_box or regs["mouth"][0])
    cues = list(a.get("cues") or []) + list(ev.get("cues") or [])
    res["motion"] = motion_onsets(pts, frames, masks, cues, k=cue_k)
    res["command_to_motion_ms"] = summary([r["latency_ms"] for r in res["motion"] if r.get("kind") == "command"])
    res["reflex_to_motion_ms"] = summary([r["latency_ms"] for r in res["motion"] if r.get("kind") != "command"])
    res["session"] = session_numbers(json.load(open(session))) if session else {}
    res["method"].update({
        "first_audio": "voice-channel VAD onset after the user utterance end (utterances = mic segments merged over <0.7 s pauses)",
        "motion": f"per-region ({ANALYSIS_W}x{ANALYSIS_H} gray) first frame after the cue whose frame-to-frame change > pre-cue median+{cue_k}*MAD AND whose change persists 150-450 ms later (blinks/speech flaps/breathing excluded); mouth blocks excluded from head/eye/ear regions",
        "lip_offset": "x-corr of auto-located mouth-block luminance vs voice RMS envelope at 100 Hz over speech +-0.5 s, +-400 ms",
        "display": "distinct frames (>= 3 px changed by > 8 gray) -> intervals"})
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--voice-channel", type=int)
    ap.add_argument("--mic-channel", type=int)
    ap.add_argument("--mic")
    ap.add_argument("--mic-offset-s", type=float, default=0.0)
    ap.add_argument("--ann")
    ap.add_argument("--events")
    ap.add_argument("--session")
    ap.add_argument("--regions")
    ap.add_argument("--crop", help="ffmpeg crop w:h:x:y of the avatar inside the recording")
    ap.add_argument("--cue-k", type=float, default=6.0)
    ap.add_argument("--mouth-box", help="x0,y0,x1,y1 normalized: search the mouth only here")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = analyze(a.video, a.voice_channel, a.mic_channel, a.mic, a.mic_offset_s, a.ann, a.events, a.session,
                  a.regions, a.crop, a.cue_k,
                  mouth_hint=[float(v) for v in a.mouth_box.split(",")] if a.mouth_box else None)
    js = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(js)
    short = {k: res.get(k) for k in ("first_audio_ms", "command_to_motion_ms", "reflex_to_motion_ms", "display")}
    short["lip_offset_ms"] = res["lip_offset"].get("lag_ms")
    short["lip_corr"] = res["lip_offset"].get("corr")
    print(json.dumps(short, indent=1))


if __name__ == "__main__":
    main()
