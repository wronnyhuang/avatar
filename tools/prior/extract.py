"""Per-clip extraction: MediaPipe face (52 blendshapes + transformation matrix + iris/eyelid landmarks), pose
(shoulders/elbows/wrists), shot cuts, and 100 Hz audio prosody. One .npz per clip.

Layout probe decides 'single' (one person, full frame) or a split screen 'lr' / 'tb' (two people, one crop each).
Usage: python -I extract.py <models_dir> <out_dir> <clip.mp4> [<clip.mp4> ...]
"""
import sys, os, json, subprocess, time
from pathlib import Path
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

FPS = 30
AR = 16000
LMK = [1, 33, 133, 362, 263, 159, 145, 386, 374, 468, 473, 10, 152, 234, 454]   # nose, eye corners, lids, irises, face extents
POSE = [0, 11, 12, 13, 14, 15, 16]


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "json", "--", path], capture_output=True, text=True)
    s = json.loads(out.stdout)["streams"][0]
    return int(s["width"]), int(s["height"])


def frames(path, w, h, fps):
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
    p.wait()


def audio(path):
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-ac", "1", "-ar", str(AR), "-f", "f32le", "-"]
    b = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(b, np.float32).copy()


def prosody(x, sr=AR, hop=160, win=640, fmin=70.0, fmax=400.0):
    """100 Hz RMS (dBFS), YIN f0 (Hz, 0 unvoiced), aperiodicity."""
    n = max(0, 1 + (len(x) - win) // hop)
    if n == 0:
        return np.zeros((0, 3), np.float32)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    fr = x[idx].astype(np.float64)
    fr = fr - fr.mean(1, keepdims=True)
    rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    tmax = int(sr / fmin); tmin = int(sr / fmax)
    W = win - tmax
    # YIN difference via FFT autocorrelation of each frame
    nfft = 1 << int(np.ceil(np.log2(win * 2)))
    F = np.fft.rfft(fr, nfft)
    ac = np.fft.irfft(F * np.conj(F), nfft)[:, : tmax + 1]
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(fr ** 2, 1)], 1)
    e0 = cs[:, W]                                  # energy of the first W samples
    taus = np.arange(tmax + 1)
    eT = cs[:, taus + W] - cs[:, taus]             # energy of the shifted window
    # approximate: use full-frame autocorrelation as cross term (standard fast YIN approximation)
    d = e0[:, None] + eT - 2 * ac * (W / win)
    d[:, 0] = 0
    cm = np.cumsum(d[:, 1:], 1) / np.arange(1, tmax + 1)[None, :]
    dn = np.ones_like(d); dn[:, 1:] = d[:, 1:] / (cm + 1e-12)
    f0 = np.zeros(n); ap = np.ones(n)
    seg = dn[:, tmin: tmax]
    below = seg < 0.2
    first = np.where(below.any(1), below.argmax(1), seg.argmin(1))
    # walk to the local minimum after the first threshold crossing
    for k in range(3):
        nxt = np.minimum(first + 1, seg.shape[1] - 1)
        better = seg[np.arange(n), nxt] < seg[np.arange(n), first]
        first = np.where(better, nxt, first)
    val = seg[np.arange(n), first]
    tau = first + tmin
    # parabolic interpolation
    tl = np.clip(tau - 1, 1, tmax); tr = np.clip(tau + 1, 1, tmax)
    a, b, c = dn[np.arange(n), tl], dn[np.arange(n), tau], dn[np.arange(n), tr]
    den = (a - 2 * b + c); off = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / den, 0.0)
    tauf = tau + np.clip(off, -1, 1)
    f0 = sr / tauf
    ap = val
    voiced = (ap < 0.3) & (db > db.max() - 45) & (f0 > fmin) & (f0 < fmax)
    f0 = np.where(voiced, f0, 0.0)
    return np.stack([db, f0, ap], 1).astype(np.float32)


def face_opts(models, mode, n=1):
    return vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(Path(models) / "face_landmarker.task")),
        running_mode=mode, num_faces=n, output_face_blendshapes=True, output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.5, min_face_presence_confidence=0.5, min_tracking_confidence=0.5)


def pose_opts(models):
    return vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(Path(models) / "pose_landmarker_lite.task")),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)


CROPS = {"full": (0, 0, 1, 1), "L": (0, 0, .5, 1), "R": (.5, 0, 1, 1), "T": (0, 0, 1, .5), "B": (0, .5, 1, 1)}


def crop(img, c):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = CROPS[c]
    return np.ascontiguousarray(img[int(y0 * h): int(y1 * h), int(x0 * w): int(x1 * w)])


def detect_layout(path, models, w, h):
    hits = {k: 0 for k in CROPS}; n = 0; nfaces2 = 0
    with vision.FaceLandmarker.create_from_options(face_opts(models, vision.RunningMode.IMAGE, 2)) as fl:
        for i, img in enumerate(frames(path, w, h, 0.25)):
            n += 1
            for c in CROPS:
                r = fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop(img, c)))
                k = len(r.face_landmarks)
                if k >= 1:
                    # require a plausibly whole face inside the crop (not a half face at the edge)
                    xs = np.array([p.x for p in r.face_landmarks[0]]); ys = np.array([p.y for p in r.face_landmarks[0]])
                    if xs.min() > 0.0 and xs.max() < 1.0 and ys.min() > 0.0 and ys.max() < 1.0:
                        hits[c] += 1
                if c == "full" and k >= 2:
                    nfaces2 += 1
    n = max(n, 1)
    fr = {k: v / n for k, v in hits.items()}
    if fr["L"] >= .5 and fr["R"] >= .5 and fr["full"] >= .3:
        lay = "lr"
    elif fr["T"] >= .5 and fr["B"] >= .5:
        lay = "tb"
    else:
        lay = "single"
    return lay, fr, nfaces2 / n


def run(models, out_dir, path):
    t0 = time.time()
    path = str(path)
    name = Path(path).stem
    out = Path(out_dir) / f"{name}.npz"
    if out.exists():
        return
    W0, H0 = probe(path)
    w = min(640, W0 - W0 % 2); h = int(round(H0 * w / W0 / 2)) * 2
    lay, fr, f2 = detect_layout(path, models, w, h)
    streams = {"single": ["full"], "lr": ["L", "R"], "tb": ["T", "B"]}[lay]
    fls = [vision.FaceLandmarker.create_from_options(face_opts(models, vision.RunningMode.VIDEO)) for _ in streams]
    pls = [vision.PoseLandmarker.create_from_options(pose_opts(models)) for _ in streams]
    S = len(streams)
    rec = {k: [[] for _ in streams] for k in ("present", "blend", "mat", "lmk", "pose")}
    cut = []; prev = None
    for i, img in enumerate(frames(path, w, h, FPS)):
        ts = int(i * 1000 / FPS)
        g = img[::8, ::8].mean(2)
        cut.append(0.0 if prev is None else float(np.abs(g - prev).mean())); prev = g
        for s, c in enumerate(streams):
            im = crop(img, c)
            r = fls[s].detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=im), ts)
            if r.face_landmarks:
                L = r.face_landmarks[0]
                rec["present"][s].append(1)
                rec["blend"][s].append([b.score for b in r.face_blendshapes[0]])
                rec["mat"][s].append(np.asarray(r.facial_transformation_matrixes[0], np.float32).reshape(16))
                rec["lmk"][s].append([[L[j].x, L[j].y, L[j].z] for j in LMK])
            else:
                rec["present"][s].append(0)
                rec["blend"][s].append([0.0] * 52)
                rec["mat"][s].append(np.zeros(16, np.float32))
                rec["lmk"][s].append([[0, 0, 0]] * len(LMK))
            if i % 2 == 0:
                rp = pls[s].detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=im), ts)
                if rp.pose_landmarks:
                    P = rp.pose_landmarks[0]
                    rec["pose"][s].append([[P[j].x, P[j].y, P[j].z, P[j].visibility] for j in POSE])
                else:
                    rec["pose"][s].append([[0, 0, 0, 0]] * len(POSE))
            else:
                rec["pose"][s].append(rec["pose"][s][-1])
    for o in fls + pls:
        o.close()
    x = audio(path)
    pr = prosody(x)
    names = None
    np.savez_compressed(out, layout=lay, streams=np.array(streams), probe=json.dumps(fr), fps=FPS,
                        present=np.array(rec["present"], np.uint8), blend=np.array(rec["blend"], np.float32),
                        mat=np.array(rec["mat"], np.float32), lmk=np.array(rec["lmk"], np.float32),
                        pose=np.array(rec["pose"], np.float32), cut=np.array(cut, np.float32),
                        prosody=pr, wh=np.array([w, h]))
    n = len(cut)
    print(f"[ok] {name} layout={lay} frames={n} present={[float(np.mean(p)) for p in rec['present']]} "
          f"{time.time()-t0:.0f}s", flush=True)


def blend_names(models):
    with vision.FaceLandmarker.create_from_options(face_opts(models, vision.RunningMode.IMAGE)) as fl:
        pass


if __name__ == "__main__":
    models, out_dir = sys.argv[1], sys.argv[2]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for p in sys.argv[3:]:
        try:
            run(models, out_dir, p)
        except Exception as e:
            print(f"[err] {p}: {e!r}", flush=True)
