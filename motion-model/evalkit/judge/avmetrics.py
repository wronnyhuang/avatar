#!/usr/bin/env python3
"""avmetrics.py - objective audio/visual metrics for a talking-avatar recording.

    compute(video_path, annotation=None, mic_wav=None, plot_path=None) -> dict
    CLI: python avmetrics.py <avatar.mp4> [--annotation a.json] [--mic mic.wav]
                             [--out m.json] [--plot timeline.png]

Units
-----
* Times are seconds from the first video frame (t=0 = start of the user test clip).
* Flow / motion energy are pixels per frame at the ANALYSIS resolution
  (frame scaled so its longest side is ANALYSIS_MAX px, see "analysis_wh").
* Region boxes are normalized [x0, y0, x1, y1] in 0..1 of the full frame.

Sign convention
---------------
lipsync.offset_ms > 0  => mouth motion LAGS the audio (sound first, mouth later).
lipsync.offset_ms < 0  => mouth motion LEADS the audio (e.g. audio track delayed).

Method (short)
--------------
Frames are decoded by ffmpeg at the native frame rate (VFR normalised to CFR), gray,
scaled to ANALYSIS_MAX px. Farneback dense flow between consecutive frames; flow
magnitude / vectors and a "changed pixel" fraction are kept at a coarser storage grid.
The character bbox = union of the motion mask (temporal std of the frame) and an
appearance mask (pixels differing from the border/background colour). Regions are fixed
fractions of the bbox; the mouth box is located where per-pixel motion best correlates
with the voice envelope (any lag in -800..+800 ms), fallback = lower-central head.
Voice activity = RMS envelope above max(4 x noise floor, 0.02 x peak, -60 dBFS),
0.15 s hangover, spans >= 0.2 s.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import argparse
import json
import math
import subprocess
import sys

import numpy as np
import cv2

try:
    cv2.setNumThreads(8)
except Exception:
    pass

ANALYSIS_MAX = 288      # longest side of the frames the flow runs on
STORE_DIV = 2.5         # storage grid = analysis / STORE_DIV
SR = 16000              # audio sample rate
ENV_HZ = 100            # high-rate envelope for lip lag
GRID = 4                # GRID x GRID cells over the bbox
LAG_MAX_MS = 800
LIP_BIAS_MS = -150  # natural lead of mouth motion over voice energy (see lip_sync_obj)
FLOW_THREADS = int(os.environ.get("AVMETRICS_THREADS", "12"))


# --------------------------------------------------------------------------- utils
def _r(x, nd=3):
    """Round to nd decimals, map numpy/NaN/inf to plain python / None."""
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    try:
        f = float(x)
    except Exception:
        return x
    if not math.isfinite(f):
        return None
    return round(f, nd)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    if isinstance(o, (float, int, np.floating, np.integer, np.bool_, bool)) or o is None:
        return _r(o)
    return o


def _clip01(x):
    return float(min(1.0, max(0.0, x)))


def _box_smooth(x, n):
    n = int(max(1, n))
    if n <= 1 or len(x) < 2:
        return np.asarray(x, dtype=np.float64)
    k = np.ones(n) / n
    xp = np.pad(np.asarray(x, dtype=np.float64), (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(xp, k, mode="valid")


def _runs(mask):
    """[(start, end_exclusive)] of True runs."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0]
    return list(zip(s.tolist(), e.tolist()))


def _ffprobe(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
            capture_output=True, text=True, timeout=60)
        return json.loads(out.stdout or "{}")
    except Exception:
        return {}


def _rate(s):
    try:
        a, b = s.split("/")
        a, b = float(a), float(b)
        return a / b if b else 0.0
    except Exception:
        try:
            return float(s)
        except Exception:
            return 0.0


# --------------------------------------------------------------------------- decode
def _probe_video(path):
    info = _ffprobe(path)
    vs = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    as_ = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not vs:
        raise RuntimeError("no video stream in %s" % path)
    v = vs[0]
    w, h = int(v.get("width", 0)), int(v.get("height", 0))
    fps = _rate(v.get("avg_frame_rate", "0/0"))
    rfps = _rate(v.get("r_frame_rate", "0/0"))
    if not (5 <= fps <= 120):
        fps = rfps
    if not (5 <= fps <= 120):
        fps = 30.0
    if fps > 61:
        fps = 30.0
    # snap near-integer / NTSC rates
    for cand in (24.0, 25.0, 30.0, 50.0, 60.0, 30000 / 1001, 24000 / 1001, 60000 / 1001, 15.0):
        if abs(fps - cand) < 0.15:
            fps = cand
            break
    vstart = float(v.get("start_time", 0) or 0)
    astart = float(as_[0].get("start_time", 0) or 0) if as_ else 0.0
    dur = None
    for src in (v.get("duration"), info.get("format", {}).get("duration")):
        try:
            if src is not None:
                dur = float(src)
                break
        except Exception:
            pass
    return dict(w=w, h=h, fps=fps, vstart=vstart, astart=astart, has_audio_stream=bool(as_), dur=dur)


def _analysis_size(w, h):
    s = min(1.0, ANALYSIS_MAX / float(max(w, h)))
    aw = max(16, int(round(w * s / 2)) * 2)
    ah = max(16, int(round(h * s / 2)) * 2)
    return aw, ah


def _decode_and_flow(path, meta):
    """Stream frames through ffmpeg, compute flow; return per-frame storage arrays."""
    aw, ah = _analysis_size(meta["w"], meta["h"])
    sw = max(8, int(round(aw / STORE_DIV)))
    sh = max(8, int(round(ah / STORE_DIV)))
    fps = meta["fps"]
    cmd = ["ffmpeg", "-v", "error", "-threads", "4", "-i", path, "-an", "-sn",
           "-vf", "fps=%.6f,scale=%d:%d:flags=area,format=gray" % (fps, aw, ah),
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fsz = aw * ah
    frames = []
    while True:
        buf = proc.stdout.read(fsz)
        if buf is None or len(buf) < fsz:
            break
        frames.append(np.frombuffer(buf, dtype=np.uint8).reshape(ah, aw))
    try:
        proc.stdout.close()
        proc.wait(timeout=30)
    except Exception:
        pass
    first = frames[0].copy() if frames else None
    grays = [cv2.resize(g, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float16) for g in frames]
    z = np.zeros((sh, sw), np.float16)

    def pair(i):
        prev, g = frames[i - 1], frames[i]
        flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 13, 2, 5, 1.1, 0)
        fx = flow[..., 0]
        fy = flow[..., 1]
        mag = np.sqrt(fx * fx + fy * fy)
        chg = (cv2.absdiff(prev, g) > 6).astype(np.float32)
        r = lambda a: cv2.resize(a, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float16)
        return r(mag), r(fx), r(fy), r(chg)

    res = [(z, z, z, z)] if frames else []
    if len(frames) > 1:
        from concurrent.futures import ThreadPoolExecutor
        try:
            old_nt = cv2.getNumThreads()
            cv2.setNumThreads(1)
        except Exception:
            old_nt = None
        try:
            with ThreadPoolExecutor(max_workers=FLOW_THREADS) as ex:
                res += list(ex.map(pair, range(1, len(frames))))
        finally:
            if old_nt is not None:
                cv2.setNumThreads(old_nt)
    mags = [r[0] for r in res]; dxs = [r[1] for r in res]; dys = [r[2] for r in res]; chgs = [r[3] for r in res]
    del frames
    n = len(grays)
    if n == 0:
        raise RuntimeError("could not decode any frame from %s" % path)
    if n > 1:
        mags[0] = mags[1]; dxs[0] = dxs[1]; dys[0] = dys[1]; chgs[0] = chgs[1]
    return dict(aw=aw, ah=ah, sw=sw, sh=sh, n=n,
                mag=np.stack(mags), dx=np.stack(dxs), dy=np.stack(dys),
                chg=np.stack(chgs), gray=np.stack(grays), first=first)


# --------------------------------------------------------------------------- audio
def _load_audio_file(path, offset_s=0.0):
    """Mono float32 @ SR from any ffmpeg-readable file; None if no audio."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR),
             "-f", "f32le", "-"], capture_output=True, timeout=120)
        a = np.frombuffer(out.stdout, dtype=np.float32)
    except Exception:
        return None
    if a.size < SR // 10:
        return None
    a = a.astype(np.float32)
    if offset_s > 0.001:      # audio starts later than video -> pad front
        a = np.concatenate([np.zeros(int(round(offset_s * SR)), np.float32), a])
    elif offset_s < -0.001:   # audio starts earlier -> drop
        a = a[int(round(-offset_s * SR)):]
    return a


def _rms_env(a, hz, n_out, win_s):
    """RMS envelope sampled at k/hz for k in range(n_out), window win_s centered."""
    if a is None:
        return np.zeros(n_out)
    half = max(1, int(round(win_s * SR / 2)))
    c = np.concatenate([[0.0], np.cumsum(a.astype(np.float64) ** 2)])
    centers = np.round(np.arange(n_out) * SR / hz).astype(np.int64)
    lo = np.clip(centers - half, 0, len(a))
    hi = np.clip(centers + half, 0, len(a))
    cnt = np.maximum(hi - lo, 1)
    e = (c[hi] - c[lo]) / cnt
    e[hi <= lo] = 0.0
    return np.sqrt(np.maximum(e, 0))


def _vad(env100, hz=ENV_HZ):
    """Voice spans [[t0,t1],...] from a 100 Hz RMS envelope, plus threshold used."""
    if env100 is None or env100.size == 0:
        return [], None
    peak = float(np.percentile(env100, 99.5)) if env100.size else 0.0
    if peak < 1e-3:                       # below -60 dBFS => silent
        return [], None
    nz = env100[env100 > 1e-6]
    floor = float(np.percentile(nz, 10)) if nz.size else 0.0
    thr = max(floor * 4.0, 0.02 * peak, 1e-3)
    act = env100 > thr
    hang = int(round(0.15 * hz))
    spans = []
    for s, e in _runs(act):
        spans.append([s, e + hang])
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out = []
    n = env100.size
    for s, e in merged:
        e = min(e, n)
        if (e - s) / hz >= 0.2:
            out.append([s / hz, e / hz])
    return out, thr


def _span_mask(spans, n, hz):
    m = np.zeros(n, dtype=bool)
    for t0, t1 in spans:
        a = max(0, int(math.floor(t0 * hz)))
        b = min(n, int(math.ceil(t1 * hz)))
        if b > a:
            m[a:b] = True
    return m


def _overlap(spans_a, spans_b):
    tot = 0.0
    for a0, a1 in spans_a:
        for b0, b1 in spans_b:
            tot += max(0.0, min(a1, b1) - max(a0, b0))
    return tot


# --------------------------------------------------------------------------- spatial
def _bbox_from_mask(mask, pad=0.04):
    ys, xs = np.nonzero(mask)
    if xs.size < 4:
        return None
    h, w = mask.shape
    x0, x1 = np.percentile(xs, [0.5, 99.5])
    y0, y1 = np.percentile(ys, [0.5, 99.5])
    px, py = pad * w, pad * h
    x0 = max(0, int(math.floor(x0 - px))); x1 = min(w, int(math.ceil(x1 + 1 + px)))
    y0 = max(0, int(math.floor(y0 - py))); y1 = min(h, int(math.ceil(y1 + 1 + py)))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    return [x0, y0, x1, y1]


def _largest_components(mask, keep_frac=0.05):
    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = areas.max()
    keep = np.zeros(n, dtype=bool)
    keep[1:] = areas >= max(2, keep_frac * big)
    return keep[lab]


def _area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _sub(bb, u0, v0, u1, v1):
    """Sub-box of bbox bb given in bbox-relative fractions, integer storage px."""
    x0, y0, x1, y1 = bb
    W, H = x1 - x0, y1 - y0
    a = [x0 + u0 * W, y0 + v0 * H, x0 + u1 * W, y0 + v1 * H]
    b = [int(math.floor(a[0])), int(math.floor(a[1])), int(math.ceil(a[2])), int(math.ceil(a[3]))]
    if b[2] <= b[0]:
        b[2] = b[0] + 1
    if b[3] <= b[1]:
        b[3] = b[1] + 1
    return b


def _box_series(arr, b):
    x0, y0, x1, y1 = b
    return arr[:, y0:y1, x0:x1].astype(np.float32).reshape(arr.shape[0], -1).mean(1)


# --------------------------------------------------------------------------- spectra
def _welch(x, fs, nper):
    """Mean periodogram of 50%-overlapping Hann segments (numpy-only Welch)."""
    x = np.asarray(x, dtype=np.float64)
    nper = int(min(nper, x.size))
    step = max(1, nper // 2)
    win = np.hanning(nper)
    scale = 1.0 / (fs * (win * win).sum())
    segs = [x[i:i + nper] for i in range(0, x.size - nper + 1, step)]
    ps = []
    for sgm in segs:
        sgm = sgm - sgm.mean()
        ps.append(np.abs(np.fft.rfft(sgm * win)) ** 2 * scale)
    p = np.mean(ps, 0)
    p[1:-1] *= 2
    f = np.fft.rfftfreq(nper, 1.0 / fs)
    return f, p


def _band_ratio(sig, fps, eps=1e-3):
    """Power 6-15 Hz / power 0.5-4 Hz of a (N, k) velocity signal (summed over k)."""
    sig = np.asarray(sig, dtype=np.float64)
    if sig.ndim == 1:
        sig = sig[:, None]
    n = sig.shape[0]
    if n < 24:
        return None
    nper = int(min(n, max(32, round(fps * 2))))
    tot_lo = 0.0
    tot_hi = 0.0
    nyq = fps / 2.0
    hi_top = min(15.0, 0.92 * nyq)
    tt = np.arange(n)
    for k in range(sig.shape[1]):
        x = sig[:, k]
        x = x - np.polyval(np.polyfit(tt, x, 1), tt)
        f, p = _welch(x, fps, nper)
        df = f[1] - f[0] if f.size > 1 else 1.0
        tot_lo += float(p[(f >= 0.5) & (f <= 4.0)].sum() * df)
        tot_hi += float(p[(f >= 6.0) & (f <= hi_top)].sum() * df)
    return tot_hi / (tot_lo + eps)


YUNET = os.environ.get("AVMETRICS_YUNET") or next(
    (p for p in (os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_detection_yunet_2023mar.onnx"),)
     if os.path.exists(p)), "")      # OpenCV Zoo YuNet face detector (download it there or set AVMETRICS_YUNET)


def _face_mouth_box(path, meta, sw, sh, n_probe=8):
    """Mouth box (storage coords) from YuNet mouth-corner landmarks on n_probe frames; None if no stable face."""
    if not os.path.exists(YUNET) or not hasattr(cv2, "FaceDetectorYN"):
        return None
    try:
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:
            pass
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        W, H = int(meta["w"]), int(meta["h"])
        det = cv2.FaceDetectorYN.create(YUNET, "", (W, H), 0.7, 0.3, 50)
        pts = []
        for k in range(n_probe):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * max(n, 1) / n_probe))
            ok, fr = cap.read()
            if not ok:
                continue
            if fr.shape[1] != W or fr.shape[0] != H:
                fr = cv2.resize(fr, (W, H))
            _, faces = det.detect(fr)
            if faces is None or len(faces) == 0:
                continue
            f = max(faces, key=lambda r: r[2] * r[3])
            fx, fy, fw, fh = f[0:4]
            mx, my = (f[10] + f[12]) / 2, (f[11] + f[13]) / 2
            cw = abs(f[12] - f[10])
            pts.append((mx, my, max(1.7 * cw, 0.45 * fw), 0.36 * fh))
        cap.release()
        if len(pts) < max(3, n_probe // 2):
            return None
        mx, my, bw, bh = [float(np.median([p[i] for p in pts])) for i in range(4)]
        my += 0.06 * bh / 0.36  # include the jaw opening below the lip line
        sx, sy = sw / float(W), sh / float(H)
        x0, x1 = max(0, int((mx - bw / 2) * sx)), min(sw, int(np.ceil((mx + bw / 2) * sx)))
        y0, y1 = max(0, int((my - bh / 2) * sy)), min(sh, int(np.ceil((my + bh / 2) * sy)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        return [x0, y0, x1, y1]
    except Exception:
        return None


# --------------------------------------------------------------------------- main
def compute(video_path, annotation=None, mic_wav=None, plot_path=None):
    if isinstance(annotation, str):
        with open(annotation) as fh:
            annotation = json.load(fh)
    meta = _probe_video(video_path)
    fps = meta["fps"]
    D = _decode_and_flow(video_path, meta)
    N = D["n"]
    sw, sh = D["sw"], D["sh"]
    mag, dx, dy, chg, gray = D["mag"], D["dx"], D["dy"], D["chg"], D["gray"]
    duration = N / fps
    t_frame = np.arange(N) / fps                 # time of frame i
    t_mot = (np.arange(N) - 0.5) / fps           # time of motion sample i (frame i-1 -> i)
    t_mot[0] = 0.0
    flags = []

    # ---------------- foreground / bbox
    gf = gray.astype(np.float32)
    stdmap = gf.std(0)
    meanmag = mag.astype(np.float32).mean(0)
    thr_std = max(2.5, 0.2 * float(np.percentile(stdmap, 99)))
    mmask = (stdmap > thr_std) | (meanmag > max(0.15, 0.2 * float(np.percentile(meanmag, 99))))
    mmask = cv2.morphologyEx(mmask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    if mmask.sum() > 0:
        mmask = _largest_components(mmask, 0.03)
    mbox = _bbox_from_mask(mmask) if mmask.sum() >= max(4, 0.003 * sw * sh) else None

    # appearance: differs from border (background) colour
    sub = np.linspace(0, N - 1, min(N, 40)).astype(int)
    med = np.median(gf[sub], axis=0)
    ring = np.concatenate([med[:2].ravel(), med[-2:].ravel(), med[:, :2].ravel(), med[:, -2:].ravel()])
    bgc = float(np.median(ring))
    ring_mad = float(np.median(np.abs(ring - bgc)))
    abox = None
    amask = None
    if ring_mad < 6.0:
        amask = np.abs(med - bgc) > max(10.0, 4 * ring_mad)
        amask = cv2.morphologyEx(amask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
        if amask.sum() > 0:
            amask = _largest_components(amask, 0.05)
        frac = amask.mean()
        if 0.01 <= frac <= 0.85:
            abox = _bbox_from_mask(amask, pad=0.02)
        else:
            amask = None
    if mbox and abox:
        if _area(abox) <= 4 * _area(mbox):
            bbox = [min(mbox[0], abox[0]), min(mbox[1], abox[1]), max(mbox[2], abox[2]), max(mbox[3], abox[3])]
        else:
            bbox = abox if _area(mbox) < 0.02 * sw * sh else mbox
    else:
        bbox = mbox or abox or [0, 0, sw, sh]
    bb = bbox
    bx0, by0, bx1, by1 = bb

    # noise floor of flow magnitude (outside the motion mask, if any)
    outside = ~mmask
    outside[by0:by1, bx0:bx1] = False
    noise = float(np.median(meanmag[outside])) if outside.sum() > 20 else 0.02
    noise = float(min(max(noise, 0.005), 0.1))

    # ---------------- audio
    audio = None
    if meta["has_audio_stream"]:
        audio = _load_audio_file(video_path, meta["astart"] - meta["vstart"])
    n100 = int(math.ceil(duration * ENV_HZ)) + 1
    env100 = _rms_env(audio, ENV_HZ, n100, 0.025) if audio is not None else None
    env_f = _rms_env(audio, fps, N, 1.0 / fps) if audio is not None else None
    voice_spans, vthr = _vad(env100) if env100 is not None else ([], None)
    has_voice = len(voice_spans) > 0
    voiced_f = _span_mask(voice_spans, N, fps)

    # ---------------- region boxes (storage px)
    mag32 = mag.astype(np.float32)
    regions = {
        "head": _sub(bb, 0.2, 0.0, 0.8, 0.5),
        "ears": _sub(bb, 0.1, 0.0, 0.9, 0.2),
        "arms_l": _sub(bb, 0.0, 0.3, 0.3, 1.0),
        "arms_r": _sub(bb, 0.7, 0.3, 1.0, 1.0),
        "body": _sub(bb, 0.3, 0.5, 0.7, 1.0),
    }
    mw, mh = 0.24, 0.14
    mouth_center = (0.5, 0.36)
    mouth_found = False
    mouth_corr_map_max = None
    if has_voice and env_f is not None and N > 3 * fps:
        # per-pixel motion vs voice envelope, max over lags -800..+800 ms
        sx0, sy0, sx1, sy1 = _sub(bb, 0.125, 0.0, 0.875, 0.75)
        Mm = mag32[:, sy0:sy1, sx0:sx1]
        hh, ww = Mm.shape[1], Mm.shape[2]
        M = Mm.reshape(N, -1)
        M = cv2.blur(M, (1, 3)) if N >= 3 else M
        M = M - M.mean(0, keepdims=True)
        sd = M.std(0, keepdims=True)
        M = M / np.maximum(sd, 1e-6)
        a = np.log1p(_box_smooth(env_f, max(1, int(round(0.1 * fps)))) / (vthr or 1e-3))
        a = (a - a.mean()) / max(a.std(), 1e-9)
        maxlag = int(round(LAG_MAX_MS / 1000 * fps))
        step = max(1, int(round(0.1 * fps)))
        lags = list(range(-maxlag, maxlag + 1, step))
        A = np.zeros((N, len(lags)), np.float32)
        for j, L in enumerate(lags):     # mouth[t+L] ~ a[t]  => A[t+L] = a[t]
            if L >= 0:
                A[L:, j] = a[:N - L]
            else:
                A[:N + L, j] = a[-L:]
        C = (M.T @ A) / float(N)        # (P, nlags)
        cmap = C.max(1).reshape(hh, ww)
        cmap[sd.reshape(hh, ww) < 0.02] = 0.0   # pixels that never move
        kx = max(1, int(round(mw * (bx1 - bx0))))
        ky = max(1, int(round(mh * (by1 - by0))))
        cs = cv2.blur(cmap.astype(np.float32), (kx, ky))
        iy, ix = np.unravel_index(int(np.argmax(cs)), cs.shape)
        mouth_corr_map_max = float(cs[iy, ix])
        if mouth_corr_map_max >= 0.2:
            cx = (sx0 + ix + 0.5 - bx0) / max(1, bx1 - bx0)
            cy = (sy0 + iy + 0.5 - by0) / max(1, by1 - by0)
            mouth_center = (cx, cy)
            mouth_found = True
    mc = mouth_center
    regions["mouth"] = _sub(bb, max(0, mc[0] - mw / 2), max(0, mc[1] - mh / 2),
                            min(1, mc[0] + mw / 2), min(1, mc[1] + mh / 2))
    # a detected face (YuNet; photoreal and some stylised faces) beats the correlation search, which can latch onto
    # background motion that happens to correlate with the voice
    fm_box = _face_mouth_box(video_path, meta, sw, sh)
    if fm_box is not None:
        regions["mouth"] = fm_box
        mouth_found = "face"
    reg_series = {k: _box_series(mag32, b) for k, b in regions.items()}
    reg_series["arms"] = np.maximum(reg_series["arms_l"], reg_series["arms_r"])
    reg_series["any"] = _box_series(mag32, bb)

    # ---------------- motion stats
    fgm = mmask.copy()
    if fgm.sum() < 4:
        fgm = np.zeros((sh, sw), bool); fgm[by0:by1, bx0:bx1] = True
    fg_flow = mag32[:, fgm].mean(1)
    bbmag = mag32[:, by0:by1, bx0:bx1].reshape(N, -1)
    p99 = np.percentile(bbmag, 99, axis=1)
    act_thr = max(0.15, 3 * noise)
    active_frac = float((p99 > act_thr).mean())
    mean_flow = float(fg_flow.mean())
    p95_flow = float(np.percentile(fg_flow, 95))

    # static frames: consecutive change AND 0.25 s lagged change both ~0
    k = max(1, int(round(0.25 * fps)))
    gbb = gf[:, by0:by1, bx0:bx1]
    lagchg = np.zeros(N)
    if N > k:
        dd = np.abs(gbb[k:] - gbb[:-k]).reshape(N - k, -1)
        lagchg[k:] = (dd > 6).mean(1)
        lagchg[:k] = lagchg[k]
    conschg = chg[:, by0:by1, bx0:bx1].astype(np.float32).reshape(N, -1).mean(1)
    static = (lagchg < 0.002) & (conschg < 0.002) & (p99 < max(0.25, 3 * noise))
    frozen_runs = []
    longest_static = 0.0
    for s, e in _runs(static):
        d = (e - s) / fps
        longest_static = max(longest_static, d)
        if d > 1.0:
            frozen_runs.append([s / fps, e / fps])
    frozen_frac = sum(b - a for a, b in frozen_runs) / max(duration, 1e-9)
    for a_, b_ in frozen_runs[:4]:
        flags.append("frozen>1s at %.1f-%.1f" % (a_, b_))

    # jitter: global mean flow vector over fg + appearance centroid velocity
    gvx = dx.astype(np.float32)[:, fgm].mean(1)
    gvy = dy.astype(np.float32)[:, fgm].mean(1)
    # frame-rate conversion (a 24 fps stream shown/recorded at 25/30 fps) leaves isolated duplicate frames: zero
    # velocity on one frame, double on the next = a periodic hf component that is not shake. Re-time the
    # integrated position over the non-duplicate frames before measuring jitter.
    fgmag = mag32[:, fgm].mean(1) if fgm.any() else mag32.reshape(N, -1).mean(1)
    dup_iso = np.zeros(N, bool)
    if N > 3:
        mov = fgmag[fgmag > 1e-3]
        lowm = (conschg < 1e-4) | (fgmag < 0.15 * (float(np.median(mov)) if mov.size else 0.0))
        dup_iso[1:-1] = lowm[1:-1] & ~lowm[:-2] & ~lowm[2:]
    dup_frac = float(dup_iso.mean())
    if dup_frac > 0.03:
        tt_ = np.arange(N)
        keep_ = ~dup_iso

        def _retime(v):
            pos = np.cumsum(v)
            posi = np.interp(tt_, tt_[keep_], pos[keep_])
            return np.diff(posi, prepend=posi[0])
        gvx, gvy = _retime(gvx), _retime(gvy)
    jit_flow = _band_ratio(np.stack([gvx, gvy], 1), fps)
    jit_cent = None
    if amask is not None:
        fm = np.abs(gf - bgc) > max(10.0, 4 * ring_mad)
        cnt = fm.reshape(N, -1).sum(1)
        if cnt.min() > 10:
            ys = np.arange(sh)[None, :, None]
            xs = np.arange(sw)[None, None, :]
            cxs = (fm * xs).reshape(N, -1).sum(1) / cnt
            cys = (fm * ys).reshape(N, -1).sum(1) / cnt
            scale = D["aw"] / float(sw)
            vel = np.stack([np.diff(cxs, prepend=cxs[0]), np.diff(cys, prepend=cys[0])], 1) * scale
            jit_cent = _band_ratio(vel[1:], fps)
    jitter_ratio = jit_flow

    # grid cells
    cells = []
    cell_series = []
    cell_vec = []
    for gy in range(GRID):
        for gx in range(GRID):
            b = _sub(bb, gx / GRID, gy / GRID, (gx + 1) / GRID, (gy + 1) / GRID)
            cells.append(b)
            cell_series.append(_box_series(mag32, b))
            cell_vec.append((_box_series(dx, b), _box_series(dy, b)))
    CS = np.stack(cell_series, 1)                        # (N, 16)
    cell_e = np.maximum(CS.mean(0) - noise, 0)
    tot = cell_e.sum()
    if tot > 1e-4 and mean_flow > noise * 1.5:
        share = cell_e / tot
        regions_active = int((share >= 0.5 / len(cells)).sum())
        nzs = share[share > 0]
        region_entropy = float(-(nzs * np.log(nzs)).sum() / math.log(len(cells)))
    else:
        share = np.zeros(len(cells))
        regions_active = 0
        region_entropy = 0.0
    jr = []
    for i, (vx, vy) in enumerate(cell_vec):
        if share[i] >= 0.5 / len(cells):
            r = _band_ratio(np.stack([vx, vy], 1), fps)
            if r is not None:
                jr.append(r)
    jitter_ratio_regions = float(np.median(jr)) if jr else None

    # temporal entropy: entropy of the distribution of per-second motion LEVELS
    # (8 linear levels of the max); 0 for a flat or single-burst clip, 1 = varied.
    nb = int(duration)
    temporal_entropy = None
    temporal_share_entropy = None
    if nb >= 2:
        fgsub = np.maximum(fg_flow - noise, 0)
        bins = np.array([fgsub[int(i * fps):int((i + 1) * fps)].mean() for i in range(nb)])
        if bins.max() > 0.02:
            lv = np.minimum((bins / bins.max() * 8).astype(int), 7)
            p = np.bincount(lv, minlength=8) / float(nb)
            p = p[p > 0]
            temporal_entropy = float(-(p * np.log(p)).sum() / math.log(min(8, nb)))
            q = bins / bins.sum()
            q = q[q > 0]
            temporal_share_entropy = float(-(q * np.log(q)).sum() / math.log(nb))
        else:
            temporal_entropy = 0.0
            temporal_share_entropy = 0.0

    # loop: exact-repeat detector on bbox thumbnails. D(L) = mean |G_t - G_{t+L}|;
    # score(L) = (mean D at L+-2..4 frames - D(L)) / median_L D, max over lags 0.4-5 s.
    loop_score, loop_period = 0.0, None
    th = gbb
    tw = max(4, min(48, th.shape[2])); thh = max(4, int(round(tw * th.shape[1] / max(1, th.shape[2]))))
    G = np.stack([cv2.GaussianBlur(cv2.resize(fr, (tw, thh), interpolation=cv2.INTER_AREA), (3, 3), 0)
                  for fr in th]).reshape(N, -1)
    Lmin = max(2, int(round(0.4 * fps)))
    Lmax = min(int(round(5 * fps)), N // 2)
    if Lmax > Lmin + 8:
        Ls = np.arange(Lmin - 4, Lmax + 5)
        Ls = Ls[(Ls >= 1) & (Ls < N - 2)]
        Dl = np.array([np.abs(G[L:] - G[:-L]).mean() for L in Ls])
        medD = float(np.median(Dl))
        if medD > 0.4:
            best, bestL = 0.0, None
            sc = {}
            for i, L in enumerate(Ls):
                if L < Lmin or L > Lmax:
                    continue
                nbr = [Dl[j] for j in range(len(Ls)) if 2 <= abs(Ls[j] - L) <= 4]
                if not nbr:
                    continue
                s = (float(np.mean(nbr)) - float(Dl[i])) / medD
                sc[int(L)] = s
                if s > best:
                    best, bestL = s, int(L)
            if bestL is not None:
                # fundamental: smallest lag scoring >= 85% of best
                for L in sorted(sc):
                    if sc[L] >= 0.85 * best:
                        bestL = L
                        break
            loop_score = _clip01(best / 0.6) if best > 0 else 0.0
            loop_period = bestL / fps if (bestL and loop_score >= 0.3) else None
    # spec variant: max normalized autocorr of per-cell energy vectors at lags 0.4-5 s
    loop_ac = None
    if Lmax > Lmin:
        Z = CS - CS.mean(0, keepdims=True)
        if np.abs(Z).sum() > 1e-3 and mean_flow > noise * 1.5:
            best = 0.0
            for L in range(Lmin, Lmax + 1):
                a_, b_ = Z[:-L].ravel(), Z[L:].ravel()
                den = math.sqrt(float((a_ * a_).sum() * (b_ * b_).sum()))
                if den > 0:
                    best = max(best, float((a_ * b_).sum()) / den)
            loop_ac = best
    if loop_score >= 0.5 and loop_period:
        flags.append("loop %.1fs" % loop_period)

    hv = (_box_series(dx, regions["head"]), _box_series(dy, regions["head"]))
    head_motion_std = float(math.sqrt(np.var(hv[0]) + np.var(hv[1])))
    head_motion_mean = float(reg_series["head"].mean())

    if jitter_ratio is not None and jitter_ratio > 1.0:
        flags.append("jitter (hf/lf %.1f)" % jitter_ratio)

    # ---------------- lipsync
    lips = {"offset_ms": None, "peak_corr": None, "corr_at_zero": None, "mouth_voice_ratio": None,
            "mouth_found_by_corr": mouth_found}
    if has_voice and env100 is not None and N > 2 * fps:
        msig = reg_series["mouth"]
        m100 = np.interp(np.arange(n100) / ENV_HZ, t_mot, msig)
        m100 = _box_smooth(m100, 5)
        a100 = np.log1p(_box_smooth(env100, 6) / (vthr or 1e-3))
        corr = {}
        for L in range(-LAG_MAX_MS // 10, LAG_MAX_MS // 10 + 1):
            if L >= 0:
                x, y = m100[L:], a100[:n100 - L]
            else:
                x, y = m100[:n100 + L], a100[-L:]
            if x.size < 50 or x.std() < 1e-9 or y.std() < 1e-9:
                continue
            corr[L] = float(np.corrcoef(x, y)[0, 1])
        if corr:
            Lb = max(corr, key=lambda q: corr[q])
            # parabolic refinement
            off = float(Lb)
            if (Lb - 1) in corr and (Lb + 1) in corr:
                y0_, y1_, y2_ = corr[Lb - 1], corr[Lb], corr[Lb + 1]
                den = (y0_ - 2 * y1_ + y2_)
                if abs(den) > 1e-12:
                    off = Lb + 0.5 * (y0_ - y2_) / den
            lips["offset_ms"] = off * 10.0
            lips["peak_corr"] = corr[Lb]
            lips["corr_at_zero"] = corr.get(0)
        vf = voiced_f
        if vf.sum() >= 0.5 * fps and (~vf).sum() >= 0.5 * fps:
            mv = float(msig[vf].mean()); mu = float(msig[~vf].mean())
            lips["mouth_voice_ratio"] = (mv + 1e-3) / (mu + 1e-3)
        if lips["offset_ms"] is not None and lips["peak_corr"] is not None and lips["peak_corr"] >= 0.2 \
                and abs(lips["offset_ms"]) >= 150:
            flags.append("lip %s %dms" % ("lag" if lips["offset_ms"] > 0 else "lead", int(round(abs(lips["offset_ms"])))))
        mvr = lips["mouth_voice_ratio"]
        if (lips["peak_corr"] is not None and lips["peak_corr"] < 0.2) and (mvr is None or mvr < 1.3):
            flags.append("mouth not following voice")

    # ---------------- mic / user speech
    user_speech = []
    mic_spans = None
    if annotation and annotation.get("user_speech"):
        user_speech = [[float(u["t0"]), float(u["t1"])] for u in annotation["user_speech"]]
    if mic_wav:
        ma = _load_audio_file(mic_wav)
        if ma is not None:
            me = _rms_env(ma, ENV_HZ, int(math.ceil(len(ma) / SR * ENV_HZ)), 0.025)
            mic_spans, _ = _vad(me)
            if not user_speech:
                user_speech = [list(s) for s in mic_spans]

    # ---------------- talk
    talk = None
    avatar_talk_s = float(sum(b - a for a, b in voice_spans))
    if annotation is not None or user_speech:
        tmode = (annotation or {}).get("talk", "either")
        ov = _overlap(user_speech, voice_spans)
        us_tot = sum(b - a for a, b in user_speech)
        lat = []
        for u0, u1 in user_speech:
            nxt = [s for s, e in voice_spans if s >= u1 - 1e-6]
            lat.append((min(nxt) - u1) if nxt else None)
        talk = {"mode": tmode, "overlap_user_s": ov, "overlap_frac": (ov / us_tot) if us_tot > 0 else None,
                "reply_latency_s": lat,
                "talk_when_quiet_s": avatar_talk_s if tmode == "quiet" else 0.0,
                "avatar_talk_s": avatar_talk_s}
        if ov >= 0.3:
            flags.append("talked over user %.1fs" % ov)
        if tmode == "quiet" and avatar_talk_s >= 0.3:
            flags.append("talked %.1fs when quiet expected" % avatar_talk_s)
        if tmode == "reply" and user_speech and all(x is None for x in lat):
            flags.append("no reply after user speech")
    else:
        talk = {"mode": None, "overlap_user_s": None, "overlap_frac": None, "reply_latency_s": [],
                "talk_when_quiet_s": None, "avatar_talk_s": avatar_talk_s}

    # ---------------- cues
    cue_out = []
    rmap = {"head": "head", "face": "head", "eyes": "head", "mouth": "mouth", "ears": "ears",
            "arms": "arms", "body": "body", "any": "any"}
    for c in (annotation or {}).get("cues", []) or []:
        t = float(c.get("t", 0))
        reg = rmap.get(str(c.get("region", "any")).lower(), "any")
        w = c.get("window_s") or [0.2, 1.5]
        w0, w1 = float(w[0]), float(w[1])
        s = _box_smooth(reg_series[reg], max(1, int(round(0.1 * fps))))
        i_t = int(round(t * fps))
        b0 = max(0, int(round((t - 2.0) * fps)))
        base_seg = s[b0:i_t] if i_t > b0 + 2 else s[max(0, i_t - 3):i_t + 1]
        if base_seg.size == 0:
            base_seg = s[:max(1, int(fps))]
        base = float(np.median(base_seg))
        mad = float(np.median(np.abs(base_seg - base)))
        thr = base + max(4.0 * mad * 1.4826, 0.05 + 0.3 * base)
        end = min(N, int(round((t + w1 + 1.0) * fps)) + 1)
        need = max(1, int(round(0.15 * fps)))
        onset = None
        above = s > thr
        for i in range(max(0, i_t), end - need + 1):
            if above[i:i + need].all():
                onset = (i - 0.5) / fps
                break
        lat = (onset - t) if onset is not None else None
        if lat is not None and lat < 0:
            lat = 0.0
        ws, we = max(0, int(round((t + w0) * fps))), min(N, int(round((t + w1) * fps)) + 1)
        peak = float(s[ws:we].max()) if we > ws else None
        pk_t = (ws + int(np.argmax(s[ws:we]))) / fps if we > ws else None
        inw = bool(lat is not None and (w0 - 0.1) <= lat <= w1)
        cue_out.append({"t": t, "cue": c.get("cue"), "region": c.get("region"), "kind": c.get("kind"),
                        "window_s": [w0, w1], "onset_latency_s": lat, "in_window": inw,
                        "peak_ratio": (peak / max(base, noise, 0.02)) if peak is not None else None,
                        "peak_t": pk_t, "baseline": base})
        if not inw:
            flags.append("no %s reaction to '%s' at %.1f" % (reg, c.get("cue"), t) if lat is None
                         else "%s reaction to '%s' at %.1f outside window (%.2fs)" % (reg, c.get("cue"), t, lat))

    # ---------------- objective sub-scores (0..10)
    obj = {}
    # naturalness_obj = clip(10 - 8*frozen_frac - 4*J - 4*Lp - 2.5*A - 1*H - 1*R, 0, 10)
    #   J  = clip((jitter_ratio - 0.6)/1.4)   (hf/lf flow power; 0.6 ok, >=2 = full shake penalty)
    #   Lp = clip((loop_score - 0.4)/0.5)     (exact-repeat loop)
    #   A  = clip((0.6 - active_frac)/0.6)    (too little motion)
    #   H  = 1 - clip(head_motion_std/0.15)   (no head motion)
    #   R  = 1 - region_entropy               (motion confined to one spot)
    J = _clip01(((jitter_ratio or 0) - 0.6) / 1.4)
    Lp = _clip01((loop_score - 0.4) / 0.5)
    A = _clip01((0.6 - active_frac) / 0.6)
    H = 1 - _clip01(head_motion_std / 0.15)
    R = 1 - region_entropy
    obj["naturalness_obj"] = max(0.0, min(10.0, 10 - 8 * frozen_frac - 7 * J - 7 * Lp - 2.5 * A - 1 * H - 1 * R))
    # hard caps mirroring the rubric: frozen > 1 s -> <= 4; shake -> <= 3; identical loop -> <= 3
    if longest_static is not None and longest_static > 1.0 and frozen_frac > 0.08:
        obj["naturalness_obj"] = min(obj["naturalness_obj"], 4.0)
    if (jitter_ratio or 0) >= 3.0:  # clear per-frame shake (synthetic shake measures ~20; good demos 0.3-0.6)
        obj["naturalness_obj"] = min(obj["naturalness_obj"], 3.0)
    if Lp >= 0.6:
        obj["naturalness_obj"] = min(obj["naturalness_obj"], 3.0)
    # lip_sync_obj (null without voice) = 0.4*Soff + 0.3*Scorr + 0.3*Sratio
    #   = 0.5*Soff + 0.35*Scorr + 0.15*Sratio (calibrated 2026-11-06 on HeyGen/Tavus/Simli vs offset/still-mouth refs)
    #   Soff  = 10*clip(1 - (|offset_ms - LIP_BIAS_MS| - 120)/320) scaled by clip(peak_corr/0.15)
    #           (mouth MOTION naturally leads voice ENERGY by ~150 ms in real speech: every good demo measured
    #           -20..-390 ms, median ~-200, so the tolerance is centred on LIP_BIAS_MS)
    #   Scorr = 10*clip(peak_corr/0.5)
    #   Sratio= 10*clip(mouth_voice_ratio - 1)  (ratio 2 = full; unreliable when the voice never pauses -> low weight)
    if has_voice and lips["peak_corr"] is not None:
        pc = max(0.0, lips["peak_corr"])
        dev = abs(lips["offset_ms"] - LIP_BIAS_MS)
        so = 10 * _clip01(1 - (dev - 120) / 320.0) * _clip01(pc / 0.15)
        scr = 10 * _clip01(pc / 0.5)
        if lips["mouth_voice_ratio"] is not None:
            sr_ = 10 * _clip01(lips["mouth_voice_ratio"] - 1.0)
            obj["lip_sync_obj"] = 0.5 * so + 0.35 * scr + 0.15 * sr_
        else:
            obj["lip_sync_obj"] = 0.58 * so + 0.42 * scr
        # caps: best lag pinned at the search edge = no real alignment (mouth ignores the voice) -> <= 2;
        # a visible lag/lead >= 300 ms -> <= 4
        if abs(lips["offset_ms"]) >= 0.95 * LAG_MAX_MS:
            obj["lip_sync_obj"] = min(obj["lip_sync_obj"], 2.0)
        elif abs(lips["offset_ms"] - LIP_BIAS_MS) >= 350:
            obj["lip_sync_obj"] = min(obj["lip_sync_obj"], 4.0)
    else:
        obj["lip_sync_obj"] = None
    # reaction_obj (null without cues) = mean over cues of:
    #   in window: 6 + 4*clip((peak_ratio-1)/3);  onset found but outside window: 3;  none: 0
    if cue_out:
        sc = []
        for c in cue_out:
            if c["in_window"]:
                sc.append(6 + 4 * _clip01(((c["peak_ratio"] or 1) - 1) / 3.0))
            elif c["onset_latency_s"] is not None:
                sc.append(3.0)
            else:
                sc.append(0.0)
        obj["reaction_obj"] = float(np.mean(sc))
    else:
        obj["reaction_obj"] = None
    # talk_obj (null without annotation):
    #   reply : mean over user spans of Slat, minus 10*overlap_frac; Slat = 10 for 0.5-1.5 s,
    #           8 for <0.5 s, 10*(1-(lat-1.5)/3) for >1.5 s (0 at 4.5 s), 0 for no reply
    #   quiet : 10 - 4*talk_when_quiet_s - 10*overlap_frac
    #   either: 10 - 10*overlap_frac
    if annotation is not None:
        of = talk["overlap_frac"] or 0.0
        mode = talk["mode"]
        if mode == "reply":
            ls = []
            for x in talk["reply_latency_s"]:
                if x is None:
                    ls.append(0.0)
                elif x < 0.5:
                    ls.append(8.0)
                elif x <= 1.5:
                    ls.append(10.0)
                else:
                    ls.append(10 * _clip01(1 - (x - 1.5) / 3.0))
            base_s = float(np.mean(ls)) if ls else (10.0 if avatar_talk_s > 0.3 else 0.0)
            obj["talk_obj"] = max(0.0, min(10.0, base_s - 10 * of))
        elif mode == "quiet":
            obj["talk_obj"] = max(0.0, min(10.0, 10 - 4 * avatar_talk_s - 10 * of))
        else:
            obj["talk_obj"] = max(0.0, min(10.0, 10 - 10 * of))
    else:
        obj["talk_obj"] = None
    # variety_obj = 10*(1-frozen_frac)*(0.35*clip(regions_active/8) + 0.25*region_entropy
    #               + 0.25*temporal_entropy + 0.15*(1-loop_score))
    te = temporal_entropy or 0.0
    obj["variety_obj"] = 10 * (1 - frozen_frac) * (0.35 * _clip01(regions_active / 8.0) + 0.25 * region_entropy
                                                   + 0.25 * te + 0.15 * (1 - loop_score))

    def nb(b):
        return [b[0] / sw, b[1] / sh, b[2] / sw, b[3] / sh]

    out = {
        "duration_s": duration, "fps": fps, "n_frames": N, "frame_wh": [meta["w"], meta["h"]],
        "analysis_wh": [D["aw"], D["ah"]],
        "motion": {
            "mean_flow": mean_flow, "p95_flow": p95_flow, "active_frac": active_frac,
            "frozen_frac": frozen_frac, "frozen_runs": frozen_runs, "longest_static_s": longest_static,
            "jitter_ratio": jitter_ratio, "jitter_ratio_centroid": jit_cent, "dup_frame_frac": dup_frac,
            "jitter_ratio_regions": jitter_ratio_regions,
            "regions_active": regions_active, "region_entropy": region_entropy,
            "temporal_entropy": temporal_entropy, "temporal_share_entropy": temporal_share_entropy,
            "loop_score": loop_score, "loop_period_s": loop_period, "loop_ac": loop_ac,
            "head_motion_std": head_motion_std, "head_motion_mean": head_motion_mean,
            "noise_floor": noise,
            "region_mean": {k: float(reg_series[k].mean()) for k in ("head", "mouth", "ears", "arms", "body")},
        },
        "regions": dict({"bbox": nb(bb)}, **{k: nb(v) for k, v in regions.items()}),
        "audio": {"has_audio": audio is not None, "voice_active_frac": float(voiced_f.mean()) if N else 0.0,
                  "voice_spans": voice_spans, "first_voice_t": voice_spans[0][0] if voice_spans else None,
                  "peak_dbfs": (20 * math.log10(max(float(np.percentile(env100, 99.5)), 1e-9))
                                if env100 is not None else None)},
        "lipsync": lips,
        "talk": talk,
        "cues": cue_out,
        "objective": obj,
        "flags": flags,
    }
    if mic_spans is not None:
        out["mic"] = {"voice_spans": mic_spans}
    if plot_path:
        try:
            _plot(plot_path, duration, fps, t_mot, reg_series, env_f, voice_spans, user_speech,
                  cue_out, frozen_runs, annotation, out)
        except Exception as e:  # plotting must never break metrics
            out["plot_error"] = str(e)
    return _clean(out)


# --------------------------------------------------------------------------- plot
def _font(sz, bold=False):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
              "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    try:
        return ImageFont.load_default(size=sz)
    except Exception:
        return ImageFont.load_default()


def _plot(path, duration, fps, t_mot, reg_series, env_f, voice_spans, user_speech, cues,
          frozen_runs, annotation, out):
    from PIL import Image, ImageDraw
    W, Hh = 1400, 380
    L, Rm = 70, 20
    im = Image.new("RGB", (W, Hh), "white")
    d = ImageDraw.Draw(im)
    f_s, f_m, f_b = _font(12), _font(14), _font(14, True)
    top, mot_b = 44, 230           # motion lane
    voi_t, voi_b = 244, 312        # voice lane
    fr_t, fr_b = 320, 334          # frozen / spans lane
    X = lambda t: L + (W - L - Rm) * (t / max(duration, 1e-6))
    # user speech bands
    for u0, u1 in user_speech:
        d.rectangle([X(u0), top, X(u1), voi_b], fill=(255, 236, 179))
        d.text((X(u0) + 3, top + 2), "USER SPEAKS", font=f_s, fill=(150, 100, 0))
    # axes
    d.rectangle([L, top, W - Rm, mot_b], outline=(160, 160, 160))
    d.rectangle([L, voi_t, W - Rm, voi_b], outline=(160, 160, 160))
    step = 1.0 if duration <= 30 else (5.0 if duration <= 150 else 10.0)
    t = 0.0
    while t <= duration + 1e-6:
        x = X(t)
        d.line([x, mot_b, x, mot_b + 3], fill=(120, 120, 120))
        d.line([x, voi_b, x, voi_b + 3], fill=(120, 120, 120))
        if (t / step) % (2 if duration > 16 and step == 1.0 else 1) == 0:
            d.text((x - 6, fr_b + 4), "%g" % t, font=f_s, fill=(60, 60, 60))
        t += step
    d.text((W - Rm - 70, fr_b + 18), "time (s)", font=f_s, fill=(60, 60, 60))
    # motion lines
    names = [("head", (31, 119, 180)), ("mouth", (214, 39, 40)), ("arms", (44, 160, 44)), ("body", (148, 103, 189))]
    ymax = max(0.3, max(float(np.percentile(_box_smooth(reg_series[k], 3), 99.5)) for k, _ in names) * 1.1)
    for k, col in names:
        s = _box_smooth(reg_series[k], max(1, int(round(0.1 * fps))))
        idx = np.linspace(0, len(s) - 1, min(len(s), 1400)).astype(int)
        pts = [(X(t_mot[i]), mot_b - (mot_b - top) * min(1.0, s[i] / ymax)) for i in idx]
        if len(pts) > 1:
            d.line(pts, fill=col, width=2)
    d.text((6, top - 2), "%.2f" % ymax, font=f_s, fill=(60, 60, 60))
    d.text((6, mot_b - 14), "0", font=f_s, fill=(60, 60, 60))
    d.text((4, (top + mot_b) // 2 - 16), "motion", font=f_s, fill=(60, 60, 60))
    d.text((4, (top + mot_b) // 2 - 2), "px/frame", font=f_s, fill=(60, 60, 60))
    # voice envelope (filled)
    if env_f is not None and env_f.max() > 0:
        e = env_f / max(float(np.percentile(env_f, 99.5)), 1e-6)
        idx = np.linspace(0, len(e) - 1, min(len(e), 1400)).astype(int)
        poly = [(X(0), voi_b)] + [(X(i / fps), voi_b - (voi_b - voi_t - 4) * min(1.0, e[i])) for i in idx] + [(X(duration), voi_b)]
        d.polygon(poly, fill=(255, 160, 60))
    for s0, s1 in voice_spans:
        d.rectangle([X(s0), voi_t + 1, X(s1), voi_t + 5], fill=(200, 80, 0))
    d.text((4, voi_t + 10), "avatar", font=f_s, fill=(60, 60, 60))
    d.text((4, voi_t + 24), "voice", font=f_s, fill=(60, 60, 60))
    # frozen runs
    d.text((4, fr_t), "frozen", font=f_s, fill=(200, 0, 0))
    for a, b in frozen_runs:
        d.rectangle([X(a), fr_t, X(b), fr_b], fill=(220, 0, 0))
    # cues
    for i, c in enumerate(cues):
        x = X(c["t"])
        col = (0, 120, 0) if c["in_window"] else (200, 0, 0)
        for yy in range(top, voi_b, 6):
            d.line([x, yy, x, min(yy + 3, voi_b)], fill=col, width=2)
        w0, w1 = c["window_s"]
        d.line([X(c["t"] + w0), top + 18 + 14 * (i % 3), X(c["t"] + w1), top + 18 + 14 * (i % 3)], fill=col, width=3)
        lab = "%s [%s]" % (c["cue"], c["region"])
        if c["onset_latency_s"] is not None:
            lab += " +%.2fs" % c["onset_latency_s"]
            xo = X(c["t"] + c["onset_latency_s"])
            d.polygon([(xo - 5, top), (xo + 5, top), (xo, top + 8)], fill=col)
        d.text((x + 3, top - 16 - 0 * (i % 2)), lab, font=f_s, fill=col)
    # legend / title
    title = "avmetrics  %s" % (annotation.get("id", "") if annotation else "")
    d.text((L, 4), title, font=f_b, fill=(0, 0, 0))
    lx = 380
    for k, col in names:
        d.line([lx, 12, lx + 22, 12], fill=col, width=3)
        d.text((lx + 26, 4), k, font=f_m, fill=(0, 0, 0))
        lx += 90
    d.rectangle([lx, 6, lx + 20, 18], fill=(255, 160, 60)); d.text((lx + 26, 4), "voice", font=f_m, fill=(0, 0, 0)); lx += 90
    d.rectangle([lx, 6, lx + 20, 18], fill=(255, 236, 179)); d.text((lx + 26, 4), "user speaks", font=f_m, fill=(0, 0, 0)); lx += 125
    d.rectangle([lx, 6, lx + 20, 18], fill=(220, 0, 0)); d.text((lx + 26, 4), "frozen", font=f_m, fill=(0, 0, 0)); lx += 85
    d.line([lx, 6, lx, 18], fill=(0, 120, 0), width=2); d.text((lx + 6, 4), "cue (green=in window)", font=f_m, fill=(0, 0, 0))
    # (no objective scores printed on the plot: the LLM judge sees this image and must stay independent of them)
    im.save(path)


# --------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--annotation")
    ap.add_argument("--mic")
    ap.add_argument("--out")
    ap.add_argument("--plot")
    a = ap.parse_args(argv)
    ann = None
    if a.annotation:
        with open(a.annotation) as fh:
            ann = json.load(fh)
    res = compute(a.video, ann, a.mic, a.plot)
    txt = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(txt + "\n")
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
