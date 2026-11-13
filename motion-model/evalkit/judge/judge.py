#!/usr/bin/env python3
"""Multimodal avatar judge — scores ONLY the rendered character in avatar.mp4.

Usage:
  judge.py --run-dir <S>/runs/<run> [--clips a,b] [--judge gflash|astra|gflite] [--repeat N] [--jobs 6] [--manifest M]
  judge.py --video avatar.mp4 --annotation ann.json [--camera cam.mp4] [--mic mic.wav] [--out-dir D] [--judge ..]
Primary judge gflash = google/gemini-3.7-flash via OpenRouter (video+audio + strips, ~$0.0085/clip);
second opinion astra = gpt-6-astra via the codex CLI (images only; off unless JUDGE_ALLOW_CODEX=1). Scoring: see eval/judge_rubric.md.

Inputs the judge sees (prompt v2):
  * the avatar video itself WITH its audio (OpenRouter/Gemini route only; Gemini samples ~1 fps, so it also gets ...)
  * dense frame strips: the character crop at STRIP_FPS with burned timestamps + per-frame marks
    (U = user speaking, A = avatar voice active, cue labels) — this is how the judge SEES motion;
  * a timeline plot from eval/avmetrics.py (motion energy per region, voice envelope, user speech, cues);
  * the clip's event timeline as text (user transcript, cues + expectations, talk expectation);
  * a low-res CONTEXT strip of the user's camera, clearly labelled as never scored (when the clip has one).
Outputs: judge.<id>.json per clip, scores.json per run (schema: eval/SCHEMA.md §3). Cached by
(video sha, annotation sha, judge, prompt version, repeat index) under CACHE_DIR so re-runs are free.
Every paid call is appended to a local cost ledger ($EVAL_OUT/spend.jsonl), capped by JUDGE_SPEND_CAP (USD).
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# every write (media cache, call cache) under $EVAL_OUT/judge; cost ledger $EVAL_OUT/spend.jsonl
# {"t","who":"eval","vendor","usd","what"}. $EVAL_OUT defaults to $MM_WORK_DIR/eval (canon1/paths.py).
sys.path.insert(0, str(HERE.parents[1]))
from canon1 import paths as _paths  # noqa: E402
EVAL_DIR = Path(os.environ.get("EVAL_OUT", str(Path(os.environ.get("EVAL_T", str(_paths.WORK))) / "eval")))
SCRATCH = EVAL_DIR                                # clips/<id>/{camera.mp4,mic.wav}
CACHE_DIR = EVAL_DIR / "judge/cache"
SPEND = EVAL_DIR / "spend.jsonl"
MANIFEST_CANDIDATES = [EVAL_DIR / "clips/manifest.json"]
ANNOTATION_DIRS = [EVAL_DIR / "clips/annotations"]


def default_manifest() -> Path:
    for p in MANIFEST_CANDIDATES:
        if p.exists():
            return p
    return MANIFEST_CANDIDATES[0]
CODEX = os.environ.get("CODEX_BIN", "codex")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

PROMPT_VERSION = "v4"
MEDIA_VERSION = 3  # bump when strips/mouth/plot rendering changes
STRIP_FPS = 5.0
STRIP_COLS, STRIP_ROWS = 6, 5          # 30 tiles = 6 s per sheet at 5 fps
TILE_W = 200

JUDGES = {
    # id: route, model, $/Mtok in, out
    "gflash": {"route": "openrouter", "model": "google/gemini-3.7-flash", "price": (0.75, 3.75), "video": True},
    "gflite": {"route": "openrouter", "model": "google/gemini-3.5-flash-lite", "price": (0.30, 2.5), "video": True},
    "astra": {"route": "codex", "model": "gpt-6-astra", "price": (0.0, 0.0), "video": False},
    # the same model, prompt and media as gflash, through Google's API directly (GEMINI_API_KEY; a fallback
    # when OpenRouter is unavailable). Re-judge BOTH runs of a pair with the same judge id: never mix gflash and gdirect.
    "gdirect": {"route": "gemini", "model": "gemini-3.7-flash", "price": (0.75, 3.75), "video": True,
                "key_env": "GEMINI_API_KEY"},
    # gemini-3.7/3.8-flash returned 503 "high demand" on the direct API; 3.5-flash answered. Calibrated separately
    # (CALIBRATION.md, "g35direct"); pairs judged with it are compared only with each other.
    "g35direct": {"route": "gemini", "model": "gemini-3.5-flash", "price": (0.50, 3.0), "video": True,
                  "key_env": "GEMINI_API_KEY"},
}
DEFAULT_JUDGE = "gflash"

AXES = ["naturalness", "reaction_timing", "emotional_fit", "gaze", "motion_variety", "commands", "talk_amount", "lip_sync"]
WEIGHTS = {"naturalness": 0.30, "reaction_timing": 0.15, "emotional_fit": 0.12, "gaze": 0.08,
           "motion_variety": 0.10, "commands": 0.10, "talk_amount": 0.08, "lip_sync": 0.07}
OBJ_MAP = {"naturalness": "naturalness_obj", "reaction_timing": "reaction_obj", "motion_variety": "variety_obj",
           "talk_amount": "talk_obj", "lip_sync": "lip_sync_obj"}
ALPHA = 0.35
ALPHA_AX = {"lip_sync": 0.6}  # frame-accurate mouth/voice cross-correlation beats a 4 s strip read by the judge
# objective veto: when avmetrics' naturalness hits a hard cap (frozen > 1 s, clear shake, identical loop -> <= 3),
# the final naturalness may not exceed objective + VETO_MARGIN (these detectors are reliable; judges miss shake)
VETO_AXES = {"naturalness": 2.0}
# weakest link: overall = (1 - MIN_BLEND) * weighted mean + MIN_BLEND * worst axis (one broken axis drags it down)
MIN_BLEND = 0.35


# ----------------------------------------------------------------------------- utils
def sha_file(p: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def sha_obj(o, n: int = 12) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()[:n]


SPEND_AGENT = "eval"
SPEND_CAP = float(os.environ.get("JUDGE_SPEND_CAP", "3.0"))   # USD cap on this work dir's paid judge calls
import threading  # noqa: E402
_SPEND_LOCK = threading.Lock()


def log_spend(vendor: str, usd: float, what: str):
    SPEND.parent.mkdir(parents=True, exist_ok=True)
    with _SPEND_LOCK, open(SPEND, "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "who": SPEND_AGENT, "vendor": vendor,
                            "usd": round(float(usd), 6), "what": what}) + "\n")


def my_spend() -> float:
    tot = 0.0
    if SPEND.exists():
        for line in SPEND.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("who") == SPEND_AGENT:
                tot += float(r.get("usd") or 0)
    return tot


def probe_duration(p: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def atomic_write(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp-", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, p)


# ----------------------------------------------------------------------------- objective metrics
def objective_metrics(video: Path, ann: dict | None, mic: Path | None, media_dir: Path) -> dict:
    out = media_dir / "avmetrics.json"
    plot = media_dir / "timeline.png"
    try:
        import avmetrics  # noqa
        am_sha = sha_file(Path(avmetrics.__file__), 8)
    except Exception as e:  # pragma: no cover
        return {"error": f"avmetrics unavailable: {e}"}
    if out.exists():
        try:
            d = json.loads(out.read_text())
            if d.get("_am_sha") == am_sha:
                return d
        except Exception:
            pass
    try:
        d = avmetrics.compute(str(video), annotation=ann, mic_wav=str(mic) if mic and mic.exists() else None,
                              plot_path=str(plot))
    except Exception as e:
        return {"error": f"avmetrics failed: {type(e).__name__}: {e}"}
    d["_am_sha"] = am_sha
    atomic_write(out, d)
    return d


# ----------------------------------------------------------------------------- media for the judge
def _font(size, bold=False):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(FONT_B if bold else FONT, size)
    except Exception:
        return ImageFont.load_default()


def _spans_contain(spans, t):
    return any(a <= t <= b for a, b in spans)


def read_frames_at(video: Path, times: list[float]):
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    want = sorted((int(round(t * fps)), i) for i, t in enumerate(times))
    res = [None] * len(times)
    idx, j, last = 0, 0, None
    while j < len(want):
        ok, fr = cap.read()
        if not ok:
            break
        last = fr
        while j < len(want) and want[j][0] <= idx:
            res[want[j][1]] = fr
            j += 1
        idx += 1
    while j < len(want):  # past the end
        res[want[j][1]] = last
        j += 1
    cap.release()
    return res, fps


def make_strips(video: Path, ann: dict | None, objm: dict, media_dir: Path, fps: float = STRIP_FPS) -> list[Path]:
    """Contact sheets of the character crop, STRIP_FPS, timestamps + U/A/cue marks under each tile."""
    import cv2
    from PIL import Image, ImageDraw
    dur = probe_duration(video)
    n = max(1, int(dur * fps))
    times = [k / fps for k in range(n)]
    frames, _ = read_frames_at(video, times)
    H, W = next(f for f in frames if f is not None).shape[:2]
    bbox = (objm.get("regions") or {}).get("bbox") if isinstance(objm, dict) else None
    if bbox and len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
        x0, y0, x1, y1 = bbox
        if max(bbox) <= 1.01:
            x0, x1, y0, y1 = x0 * W, x1 * W, y0 * H, y1 * H
        mx, my = 0.12 * (x1 - x0), 0.12 * (y1 - y0)
        x0, y0, x1, y1 = int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(W, x1 + mx)), int(min(H, y1 + my))
        if (x1 - x0) < 0.25 * W or (y1 - y0) < 0.25 * H:
            x0, y0, x1, y1 = 0, 0, W, H
    else:
        x0, y0, x1, y1 = 0, 0, W, H
    cw, ch = x1 - x0, y1 - y0
    tw = TILE_W
    th = int(round(tw * ch / cw))
    if th > int(tw * 1.4):
        th = int(tw * 1.4)
        tw = int(round(th * cw / ch))
    band = 22
    user_spans = [(s["t0"], s["t1"]) for s in (ann or {}).get("user_speech", [])]
    voice_spans = [tuple(s) for s in ((objm.get("audio") or {}).get("voice_spans") or [])] if isinstance(objm, dict) else []
    cues = (ann or {}).get("cues", [])
    per = STRIP_COLS * STRIP_ROWS
    sheets = []
    f_small, f_b = _font(13), _font(13, True)
    for si in range(0, n, per):
        idxs = list(range(si, min(n, si + per)))
        rows = math.ceil(len(idxs) / STRIP_COLS)
        head = 26
        sheet = Image.new("RGB", (STRIP_COLS * tw + (STRIP_COLS - 1) * 4, head + rows * (th + band + 4)), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        t_a, t_b = times[idxs[0]], times[idxs[-1]]
        d.text((4, 4), f"AVATAR frames {t_a:.1f}-{t_b:.1f} s @ {fps:g} fps (left->right, top->bottom). "
                       f"Marks: U=user speaking  A=avatar voice  red=cue", font=f_b, fill=(0, 0, 0))
        for k, i in enumerate(idxs):
            fr = frames[i]
            if fr is None:
                continue
            crop = cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            im = Image.fromarray(crop).resize((tw, th), Image.BILINEAR)
            r, c = divmod(k, STRIP_COLS)
            px, py = c * (tw + 4), head + r * (th + band + 4)
            sheet.paste(im, (px, py))
            t = times[i]
            by = py + th
            d.rectangle([px, by, px + tw - 1, by + band - 1], fill=(235, 235, 235))
            d.text((px + 3, by + 3), f"{t:5.1f}s", font=f_b, fill=(0, 0, 0))
            xx = px + 62
            if _spans_contain(user_spans, t):
                d.rectangle([xx, by + 3, xx + 16, by + band - 4], fill=(40, 90, 220))
                d.text((xx + 3, by + 3), "U", font=f_b, fill=(255, 255, 255))
            xx += 20
            if _spans_contain(voice_spans, t):
                d.rectangle([xx, by + 3, xx + 16, by + band - 4], fill=(230, 130, 20))
                d.text((xx + 3, by + 3), "A", font=f_b, fill=(255, 255, 255))
            xx += 20
            for cu in cues:
                if abs(t - float(cu["t"])) < 0.5 / fps + 1e-6:
                    d.rectangle([px, py, px + tw - 1, py + th - 1], outline=(220, 0, 0), width=3)
                    d.text((xx, by + 3), f"CUE {cu.get('cue', '')}"[:14], font=f_b, fill=(200, 0, 0))
        p = media_dir / f"strip_{len(sheets):02d}.jpg"
        sheet.save(p, quality=85)
        sheets.append(p)
    return sheets


def audio_envelope(video: Path, rate: int = 100) -> np.ndarray | None:
    """RMS envelope of the avatar audio at `rate` Hz (None when there is no audio track)."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                       capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return None
    x = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    hop = 16000 // rate
    n = len(x) // hop
    if n == 0:
        return None
    return np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(axis=1))


def make_mouth_strip(video: Path, objm: dict, media_dir: Path, fps: float = 10.0, secs: float = 4.0) -> Path | None:
    """Lip-sync close-up: lower-face crop at 10 fps over the most voiced 4 s, with the measured voice level under
    each tile (orange bar). Only when the avatar speaks."""
    from PIL import Image, ImageDraw
    import cv2
    if not isinstance(objm, dict):
        return None
    spans = (objm.get("audio") or {}).get("voice_spans") or []
    env = audio_envelope(video)
    if not spans or env is None or env.max() <= 1e-4:
        return None
    dur = probe_duration(video)
    best, best_t = -1.0, 0.0
    for t0 in np.arange(0, max(0.01, dur - secs), 0.5):
        cov = sum(max(0.0, min(b, t0 + secs) - max(a, t0)) for a, b in spans)
        if cov > best:
            best, best_t = cov, float(t0)
    times = [best_t + k / fps for k in range(int(secs * fps))]
    frames, _ = read_frames_at(video, times)
    fr0 = next((f for f in frames if f is not None), None)
    if fr0 is None:
        return None
    H, W = fr0.shape[:2]
    reg = objm.get("regions") or {}
    box = reg.get("mouth") or reg.get("head") or reg.get("bbox")
    if not box:
        return None
    x0, y0, x1, y1 = box
    if max(box) <= 1.01:
        x0, x1, y0, y1 = x0 * W, x1 * W, y0 * H, y1 * H
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(max(x1 - x0, y1 - y0) * 0.75, 12)  # tight: the mouth must fill the tile
    x0, x1 = int(max(0, cx - half)), int(min(W, cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(H, cy + half))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    cols, tw = 8, 160
    th = int(tw * (y1 - y0) / (x1 - x0))
    rows = math.ceil(len(times) / cols)
    band, head = 26, 26
    sheet = Image.new("RGB", (cols * (tw + 3), head + rows * (th + band + 3)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((4, 4), f"LIP-SYNC CLOSE-UP of the avatar mouth area, {times[0]:.1f}-{times[-1]:.1f} s @ {fps:g} fps; "
                   "orange bar = measured avatar voice loudness at that frame", font=_font(13, True), fill=(0, 0, 0))
    emax = float(np.percentile(env, 99)) or 1.0
    levels = []
    for k, (fr, t) in enumerate(zip(frames, times)):
        if fr is None:
            continue
        im = Image.fromarray(cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)).resize((tw, th))
        r, c = divmod(k, cols)
        px, py = c * (tw + 3), head + r * (th + band + 3)
        sheet.paste(im, (px, py))
        e = float(env[min(len(env) - 1, int(t * 100))]) / emax
        levels.append(round(min(1.0, e), 3))
        d.rectangle([px, py + th, px + tw - 1, py + th + band - 1], fill=(235, 235, 235))
        d.rectangle([px + 48, py + th + 5, px + 48 + int(min(1.0, e) * (tw - 52)), py + th + band - 6], fill=(230, 130, 20))
        d.text((px + 2, py + th + 5), f"{t:.1f}s", font=_font(12, True), fill=(0, 0, 0))
    p = media_dir / "mouth_strip.jpg"
    sheet.save(p, quality=85)
    (media_dir / "mouth_strip.json").write_text(json.dumps({"times": times, "levels": levels}))
    return p


def make_context_strip(camera: Path, media_dir: Path, dur: float) -> Path | None:
    """Low-res user camera at 1 fps — CONTEXT ONLY."""
    if not camera or not camera.exists():
        return None
    from PIL import Image, ImageDraw
    import cv2
    times = [float(k) for k in range(int(max(1, math.ceil(dur))))]
    frames, _ = read_frames_at(camera, times)
    frames = [f for f in frames if f is not None]
    if not frames:
        return None
    tw = 150
    H, W = frames[0].shape[:2]
    th = int(tw * H / W)
    cols = min(10, len(frames))
    rows = math.ceil(len(frames) / cols)
    head = 24
    sheet = Image.new("RGB", (cols * (tw + 3), head + rows * (th + 18)), (40, 40, 40))
    d = ImageDraw.Draw(sheet)
    d.text((4, 4), "CONTEXT ONLY - the human USER's camera at 1 fps. NEVER score this person.", font=_font(13, True),
           fill=(255, 255, 0))
    for k, (fr, t) in enumerate(zip(frames, times)):
        im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize((tw, th))
        r, c = divmod(k, cols)
        sheet.paste(im, (c * (tw + 3), head + r * (th + 18)))
        d.text((c * (tw + 3) + 2, head + r * (th + 18) + th + 2), f"user {t:.0f}s", font=_font(11), fill=(255, 255, 255))
    p = media_dir / "context_user.jpg"
    sheet.save(p, quality=70)
    return p


def make_judge_video(video: Path, media_dir: Path) -> Path:
    """Small copy for upload: ≤480 px wide, 15 fps, timestamp in a band BELOW the picture (not over the avatar)."""
    out = media_dir / "judge_video.mp4"
    if out.exists() and out.stat().st_mtime >= video.stat().st_mtime:
        return out
    vf = ("scale='min(480,iw)':-2,fps=15,pad=iw:ih+28:0:0:black,"
          f"drawtext=fontfile={FONT}:text='t=%{{pts\\:flt\\:%.1f}}s':x=6:y=h-24:fontsize=18:fontcolor=white")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "30", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac", "1", "-b:a", "48k", "-movflags", "+faststart",
           str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:  # e.g. no audio
        cmd = [c for c in cmd if c not in ("-c:a", "aac", "-ac", "1", "-b:a", "48k")]
        subprocess.run(cmd, check=True, capture_output=True)
    return out


# ----------------------------------------------------------------------------- prompt
def event_timeline(ann: dict | None, dur: float) -> str:
    if not ann:
        return "(no annotation)"
    L = [f"Clip length {dur:.1f} s (avatar.mp4 t=0 = the moment the user's clip started).",
         f"What the user does: {ann.get('description', '')}"]
    us = ann.get("user_speech") or []
    if us:
        L.append("User speech (the avatar hears this; it is NOT in the avatar audio):")
        for s in us:
            L.append(f"  {s['t0']:.1f}-{s['t1']:.1f} s: \"{s.get('text', '')}\"")
    else:
        L.append("User speech: none (the user is silent).")
    cues = ann.get("cues") or []
    if cues:
        L.append("Cues the avatar should react to:")
        for c in cues:
            w = c.get("window_s") or [0.2, 1.5]
            line = (f"  t={float(c['t']):.1f} s [{c.get('kind')}] {c.get('cue')} (region {c.get('region')}): expect "
                    f"\"{c.get('expect', '')}\" between {float(c['t']) + w[0]:.1f} and {float(c['t']) + w[1]:.1f} s")
            if c.get("command"):
                line += f"; spoken command: \"{c['command']}\""
            L.append(line)
    else:
        L.append("Cues: none.")
    L.append(f"Talk expectation: {ann.get('talk', 'either')} — {ann.get('talk_note', '')}")
    if ann.get("overall"):
        L.append(f"Overall expected demeanour: {ann['overall']}")
    return "\n".join(L)


def has_commands(ann: dict | None) -> bool:
    return any(c.get("kind") == "command" for c in (ann or {}).get("cues") or [])


def build_prompt(ann: dict | None, dur: float, n_strips: int, has_video: bool, has_plot: bool, has_ctx: bool,
                 has_mouth: bool = False) -> str:
    rubric = (HERE / "judge_rubric.md").read_text()
    rubric = rubric.split("## Final score")[0]
    media = []
    if has_video:
        media.append("- the AVATAR VIDEO with the avatar's own voice as audio (timestamp band below the picture). "
                     "Watch it and LISTEN to it; your video sampling is coarse, so use the frame strips for motion.")
    media.append(f"- {n_strips} AVATAR FRAME STRIP image(s): consecutive frames at {STRIP_FPS:g} fps (every "
                 f"{1 / STRIP_FPS:.1f} s) of the character crop, timestamp under each tile; U = user speaking at that "
                 "moment, A = avatar voice active, red border = cue moment. Compare NEIGHBOURING tiles to see motion: "
                 "identical neighbours = frozen; random position/scale changes between neighbours = jitter/shake; "
                 "the same sequence of poses repeating = a loop; mouth shape changing only while A is set = lip sync.")
    if has_mouth:
        media.append("- a LIP-SYNC CLOSE-UP strip (10 fps, 40 tiles) of the mouth area over the most voiced 4 s, with the "
                     "measured voice loudness under each tile: the mouth should open on loud tiles and close in pauses. "
                     "Read it tile by tile into `mouth_tiles` BEFORE scoring lip_sync (a mouth that never changes shape "
                     "while the bars move = lip_sync 1-2; openings consistently several tiles before/after the loud bars "
                     "= lag, lip_sync <= 4).")
    if has_plot:
        media.append("- a TIMELINE PLOT (measured): motion energy per body region, the avatar voice envelope, user "
                     "speech spans, cue lines, frozen runs. Use it to check timing and whether motion exists.")
    if has_ctx:
        media.append("- a CONTEXT strip of the human user's camera (1 fps): only to understand what the avatar is "
                     "reacting to. NEVER score the user.")
    mouth_field = ('"mouth_tiles": "<exactly 40 characters, one per LIP-SYNC tile in reading order: O = mouth visibly '
                   'open, C = closed or nearly closed>",\n ' if has_mouth else "")
    cmd_rule = ("This clip HAS command cues: score `commands`." if has_commands(ann)
                else "This clip has NO command cues: `commands` MUST be null.")
    return f"""You are a strict, calibrated judge of a REAL-TIME TALKING AVATAR's behaviour. You score ONLY the rendered
avatar character. The avatar may be a cartoon animal/shape or a photoreal person: never reward or penalize rendering
style or realism — judge BEHAVIOUR: is it alive, smooth, varied, timely, appropriate, in sync.

MEDIA YOU GET:
{chr(10).join(media)}

EVENT TIMELINE (ground truth about the user's side; the avatar should react to it):
{event_timeline(ann, dur)}

RUBRIC:
{rubric}
Use the FULL 1-10 range. A smooth, alive, varied, well-synced performance earns 7-9; an avatar that is frozen, shaking,
looping, or whose mouth ignores the voice earns 1-3 on the affected axes. Do not give 6-8 by default: decide from
evidence. Before scoring, inspect the strips tile by tile for: frozen runs, jitter, repeats, mouth vs A marks.
{cmd_rule} `reaction_timing` is null only if the timeline has no cues and no user speech. `lip_sync` is null only if
the avatar never speaks.

Return ONLY one JSON object, no markdown:
{{{mouth_field}"observations": "<3-6 sentences of what you saw, with timestamps>",
 "axes": {{"naturalness": {{"evidence": "<timestamped, specific>", "score": <1-10>}},
          "reaction_timing": {{"evidence": "...", "score": <1-10 or null>}},
          "emotional_fit": {{"evidence": "...", "score": <1-10>}},
          "gaze": {{"evidence": "...", "score": <1-10>}},
          "motion_variety": {{"evidence": "...", "score": <1-10>}},
          "commands": {{"evidence": "...", "score": <1-10 or null>}},
          "talk_amount": {{"evidence": "...", "score": <1-10>}},
          "lip_sync": {{"evidence": "...", "score": <1-10 or null>}}}},
 "worst_moment": {{"t": <seconds>, "what": "..."}},
 "verdict": "<one sentence>"}}"""


# ----------------------------------------------------------------------------- model calls
def _parse_json(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):]
    a, b = t.find("{"), t.rfind("}")
    obj = json.loads(t[a:b + 1])
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        obj = obj[0]
    return obj


def _validate(obj: dict, ann: dict | None) -> dict:
    axes = obj.get("axes") or {}
    clean = {}
    for ax in AXES:
        v = axes.get(ax)
        if isinstance(v, dict):
            s, ev = v.get("score"), v.get("evidence", "")
        else:
            s, ev = v, ""
        if isinstance(s, str):
            try:
                s = float(s)
            except ValueError:
                s = None
        if s is not None:
            s = float(min(10, max(1, s)))
        clean[ax] = {"score": s, "evidence": ev}
    if not has_commands(ann):
        clean["commands"]["score"] = None
    if ann is not None and not (ann.get("cues") or ann.get("user_speech")):
        clean["reaction_timing"]["score"] = None
    obj["axes"] = clean
    return obj


def call_openrouter(model: str, price, prompt: str, video: Path | None, images: list[Path], what: str,
                    temperature: float = 0.0, seed: int = 7) -> tuple[dict, float, dict]:
    import urllib.request
    content = [{"type": "text", "text": prompt}]
    if video is not None:
        content.append({"type": "video_url", "video_url": {
            "url": "data:video/mp4;base64," + base64.b64encode(video.read_bytes()).decode()}})
    for im in images:
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(im.read_bytes()).decode()}})
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "response_format": {"type": "json_object"}, "temperature": temperature, "seed": seed, "max_tokens": 4000,
               "reasoning": {"effort": "low"}, "usage": {"include": True},
               "provider": {"data_collection": "deny"}}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
                                          "Content-Type": "application/json"})
    last_err = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                resp = json.loads(r.read())
            break
        except Exception as e:  # network / 5xx
            last_err = e
            body = getattr(e, "read", lambda: b"")()
            last_err = RuntimeError(f"{e} {body[:300]!r}")
            time.sleep(3)
    else:
        raise last_err
    usage = resp.get("usage") or {}
    usd = usage.get("cost")
    if usd is None:
        usd = (usage.get("prompt_tokens", 0) * price[0] + usage.get("completion_tokens", 0) * price[1]) / 1e6
    log_spend("openrouter", usd, f"{what} {model} in={usage.get('prompt_tokens')} out={usage.get('completion_tokens')}")
    msg = resp["choices"][0]["message"]["content"]
    if isinstance(msg, list):
        msg = "".join(x.get("text", "") for x in msg if isinstance(x, dict))
    return _parse_json(msg), float(usd), usage


def call_gemini(model: str, price, prompt: str, video: Path | None, images: list[Path], what: str, key_env: str,
                temperature: float = 0.0, seed: int = 7) -> tuple[dict, float, dict]:
    import urllib.request
    parts = [{"text": prompt}]
    if video is not None:
        parts.append({"inline_data": {"mime_type": "video/mp4", "data": base64.b64encode(video.read_bytes()).decode()}})
    for im in images:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(im.read_bytes()).decode()}})
    payload = {"contents": [{"role": "user", "parts": parts}],
               "generationConfig": {"temperature": temperature, "seed": seed, "maxOutputTokens": 6000,
                                    "responseMimeType": "application/json", "thinkingConfig": {"thinkingBudget": 1024}}}
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                                 data=json.dumps(payload).encode(),
                                 headers={"x-goog-api-key": os.environ[key_env], "Content-Type": "application/json"})
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.loads(r.read())
            break
        except Exception as e:
            body = getattr(e, "read", lambda: b"")()
            last_err = RuntimeError(f"{e} {body[:300]!r}")
            time.sleep(4 + 6 * attempt)
    else:
        raise last_err
    um = resp.get("usageMetadata") or {}
    tin = um.get("promptTokenCount", 0)
    tout = um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0)
    usd = (tin * price[0] + tout * price[1]) / 1e6
    log_spend("google", usd, f"{what} {model} direct in={tin} out={tout}")
    cand = (resp.get("candidates") or [{}])[0]
    txt = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []) if not p.get("thought"))
    usage = {"prompt_tokens": tin, "completion_tokens": tout, "cost": usd}
    return _parse_json(txt), float(usd), usage


_QUOTA_STOP = False

def call_codex(model: str, prompt: str, images: list[Path], label: str = 'clip') -> tuple[dict, float, dict]:
    global _QUOTA_STOP
    if _QUOTA_STOP: raise SystemExit('QUOTA_STOP: batch already stopped')
    guard = os.environ.get("CODEX_QUOTA_CHECK")      # optional command that exits non-zero when the allowance is used up
    if guard:
        qr = subprocess.run(guard.split() + ['--label', label], capture_output=True, text=True)
        print(qr.stdout.strip(), flush=True)
        if qr.returncode:
            _QUOTA_STOP = True
            raise SystemExit('QUOTA_STOP: allowance guard refused; no retry')
    with tempfile.TemporaryDirectory(prefix="judge-codex-") as td:
        last = Path(td) / "last.txt"
        cmd = [CODEX, "exec", "-m", model, "--sandbox", "read-only", "--skip-git-repo-check", "--ignore-user-config",
               "--ephemeral", "-C", td, "-o", str(last)]
        for im in images:
            cmd += ["-i", str(im)]
        cmd += ["--", prompt + "\n\nDo not run any commands or tools; answer from the attached images only."]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
        if not last.exists() or not last.read_text().strip():
            raise RuntimeError(f"codex failed rc={r.returncode}: {(r.stderr or r.stdout)[-400:]}")
        return _parse_json(last.read_text()), 0.0, {"tokens": None}


# ----------------------------------------------------------------------------- scoring
def combine(axes_scores: dict, objective: dict) -> tuple[dict, float | None]:
    """axes_scores: axis -> judge score (float or None). Returns per-axis final + weighted."""
    obj = (objective or {}).get("objective") or {}
    final = {}
    for ax in AXES:
        s = axes_scores.get(ax)
        if s is None:
            final[ax] = None
            continue
        o = obj.get(OBJ_MAP.get(ax, ""), None)
        al = ALPHA_AX.get(ax, ALPHA)
        v = (1 - al) * s + al * float(o) if isinstance(o, (int, float)) else s
        if ax in VETO_AXES and isinstance(o, (int, float)) and o <= 3.0:
            v = min(v, float(o) + VETO_AXES[ax])
        final[ax] = round(min(10.0, max(1.0, v)), 3)
    return final, overall(final)


def overall(final: dict) -> float | None:
    vals = {a: v for a, v in final.items() if v is not None}
    if not vals:
        return None
    mean = sum(WEIGHTS[a] * v for a, v in vals.items()) / sum(WEIGHTS[a] for a in vals)
    return round((1 - MIN_BLEND) * mean + MIN_BLEND * min(vals.values()), 3)


def judge_one(video: Path, ann: dict | None, out_dir: Path, judge_id: str, repeat: int = 1,
              mic: Path | None = None, camera: Path | None = None, label: str = "", temperature: float = 0.0,
              out_name: str | None = None) -> dict:
    J = JUDGES[judge_id]
    vsha = sha_file(video)
    media_dir = CACHE_DIR / "media" / f"{vsha}-{sha_obj(ann)}"
    media_dir.mkdir(parents=True, exist_ok=True)
    dur = probe_duration(video)
    objm = objective_metrics(video, ann, mic, media_dir)
    stamp_p = media_dir / "media_stamp.json"
    # media depend only on these measured fields (not on the objective formulas), so formula tweaks never re-pay calls
    _rg = objm.get("regions") or {}
    stamp = {"media": MEDIA_VERSION, "bbox": [round(x, 3) for x in _rg.get("bbox") or []],
             "mouth": [round(x, 3) for x in _rg.get("mouth") or []],
             "voice": [[round(a, 2), round(b, 2)] for a, b in (objm.get("audio") or {}).get("voice_spans") or []]}
    if not stamp_p.exists() or json.loads(stamp_p.read_text()) != stamp:
        for old in list(media_dir.glob("strip_*.jpg")) + [media_dir / "mouth_strip.jpg"]:
            old.unlink(missing_ok=True)
    strips = sorted(media_dir.glob("strip_*.jpg"))
    if not strips:
        strips = make_strips(video, ann, objm, media_dir)
    plot = media_dir / "timeline.png"
    has_plot = plot.exists()
    mouth = media_dir / "mouth_strip.jpg"
    if not mouth.exists():
        mouth = make_mouth_strip(video, objm, media_dir) or mouth
    has_mouth = mouth.exists()
    stamp_p.write_text(json.dumps(stamp))
    ctx = media_dir / "context_user.jpg"
    if not ctx.exists() and camera is not None:
        ctx = make_context_strip(camera, media_dir, dur) or ctx
    has_ctx = ctx.exists()
    images = list(strips) + ([mouth] if has_mouth else []) + ([plot] if has_plot else []) + ([ctx] if has_ctx else [])
    jvideo = make_judge_video(video, media_dir) if J["video"] else None
    prompt = build_prompt(ann, dur, len(strips), J["video"], has_plot, has_ctx, has_mouth)
    msha = hashlib.sha256(b"".join(sha_file(p).encode() for p in images)).hexdigest()[:8]
    psha = sha_obj(prompt) + "-" + msha + (f"-t{temperature:g}" if temperature else "")
    runs, usd_total = [], 0.0
    for k in range(repeat):
        ck = CACHE_DIR / "calls" / f"{vsha}-{judge_id}-{PROMPT_VERSION}-{psha}-r{k}.json"
        if ck.exists():
            runs.append(json.loads(ck.read_text()))
            continue
        what = f"judge {label or video.parent.name} r{k}"
        if J["route"] == "codex":
            if os.environ.get("JUDGE_ALLOW_CODEX") != "1":
                raise SystemExit("judge astra uses the codex CLI: set JUDGE_ALLOW_CODEX=1")
        if os.environ.get("JUDGE_ALLOW_PAID") != "1":
            raise SystemExit(f"judge {judge_id} is a paid route: set JUDGE_ALLOW_PAID=1")
        if J["route"] == "openrouter":
            if my_spend() > SPEND_CAP:
                raise SystemExit(f"{SPEND_AGENT} judge allowance spent (${SPEND_CAP:.2f}, JUDGE_SPEND_CAP); refusing paid call")
            raw, usd, usage = call_openrouter(J["model"], J["price"], prompt, jvideo, images, what,
                                              temperature=temperature, seed=7 + k)
        elif J["route"] == "gemini":
            if my_spend() > SPEND_CAP:
                raise SystemExit(f"{SPEND_AGENT} judge allowance spent (${SPEND_CAP:.2f}, JUDGE_SPEND_CAP); refusing paid call")
            raw, usd, usage = call_gemini(J["model"], J["price"], prompt, jvideo, images, what, J["key_env"],
                                          temperature=temperature, seed=7 + k)
        else:
            raw, usd, usage = call_codex(J["model"], prompt, images, what)
        raw = _validate(raw, ann)
        rec = {"raw": raw, "usd": usd, "usage": usage, "model": J["model"], "ts": time.time()}
        atomic_write(ck, rec)
        usd_total += usd
        runs.append(rec)
    # average across repeats
    axes_mean, axes_std, notes = {}, {}, {}
    for ax in AXES:
        vals = [r["raw"]["axes"][ax]["score"] for r in runs if r["raw"]["axes"][ax]["score"] is not None]
        axes_mean[ax] = round(float(np.mean(vals)), 3) if vals else None
        axes_std[ax] = round(float(np.std(vals)), 3) if len(vals) > 1 else None
        notes[ax] = runs[0]["raw"]["axes"][ax].get("evidence", "")
    final, weighted = combine(axes_mean, objm)
    tile_corr = None
    try:
        mt = str(runs[0]["raw"].get("mouth_tiles") or "").upper().replace(" ", "")
        lv = json.loads((media_dir / "mouth_strip.json").read_text())["levels"] if has_mouth else []
        if mt and lv:
            n_ = min(len(mt), len(lv))
            o_ = np.array([1.0 if c == "O" else 0.0 for c in mt[:n_]])
            l_ = np.array(lv[:n_])
            if n_ >= 10 and o_.std() > 0 and l_.std() > 0:
                tile_corr = round(float(np.corrcoef(o_, l_)[0, 1]), 3)
            elif n_ >= 10:
                tile_corr = 0.0
    except Exception:
        pass
    judge_only = combine(axes_mean, {})[1]
    res = {"clip": (ann or {}).get("id") or video.parent.name, "judge": judge_id, "model": J["model"],
           "prompt_version": PROMPT_VERSION, "video_sha": vsha, "repeat": repeat,
           "axes": final, "axes_judge": axes_mean, "axes_std": axes_std if repeat > 1 else None,
           "notes": notes, "observations": runs[0]["raw"].get("observations"),
           "worst_moment": runs[0]["raw"].get("worst_moment"), "verdict": runs[0]["raw"].get("verdict"),
           "weighted": weighted, "weighted_judge_only": judge_only,
           "mouth_tiles": runs[0]["raw"].get("mouth_tiles"), "judge_tile_voice_corr": tile_corr,
           "objective": {k: objm.get(k) for k in ("objective", "flags", "motion", "lipsync", "talk", "cues", "audio")
                         if isinstance(objm, dict) and k in objm} if "error" not in objm else objm,
           "usd": round(sum(r.get("usd", 0) for r in runs), 6), "usd_new": round(usd_total, 6),
           "media_dir": str(media_dir)}
    if repeat > 1:
        res["weighted_runs"] = [combine({a: r["raw"]["axes"][a]["score"] for a in AXES}, objm)[1] for r in runs]
    res["temperature"] = temperature
    atomic_write(out_dir / (out_name or f"judge.{judge_id}.json"), res)
    return res


# ----------------------------------------------------------------------------- run-dir mode
def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    m = json.loads(path.read_text())
    root = path.parent.parent.parent  # <repo>/eval/clips/manifest.json
    out = {}
    for c in m.get("clips", []):
        ap = Path(c["annotation"])
        if not ap.is_absolute():
            ap = root / ap
        out[c["id"]] = {**c, "annotation_path": ap}
    return out


def find_annotation(clip: str, manifest: dict, clip_dir: Path) -> tuple[dict | None, Path | None, Path | None]:
    ann, mic, cam = None, None, None
    if clip in manifest:
        c = manifest[clip]
        if c["annotation_path"].exists():
            ann = json.loads(c["annotation_path"].read_text())
        mic = Path(c["mic_wav"]) if c.get("mic_wav") else None
        cam = Path(c["camera_mp4"]) if c.get("camera_mp4") else None
    # fallbacks: annotation.json in the clip dir, then the annotation dirs by clip id; media from <S>/clips/<id>/
    for cand in [clip_dir / "annotation.json"] + [d / f"{clip}.json" for d in ANNOTATION_DIRS]:
        if ann is None and cand.exists():
            ann = json.loads(cand.read_text())
    cdir = SCRATCH / "clips" / clip
    if mic is None and (cdir / "mic.wav").exists():
        mic = cdir / "mic.wav"
    if cam is None and (cdir / "camera.mp4").exists():
        cam = cdir / "camera.mp4"
    return ann, mic, cam


def spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3:
        return None
    r = spearmanr(a, b).correlation
    return None if r is None or (isinstance(r, float) and math.isnan(r)) else round(float(r), 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--clips", default="")
    ap.add_argument("--judge", default=DEFAULT_JUDGE, choices=list(JUDGES))
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--manifest", type=Path, default=None, help="default: $EVAL_OUT/clips/manifest.json")
    ap.add_argument("--video", type=Path)
    ap.add_argument("--annotation", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--camera", type=Path, help="single-file mode: the user's camera.mp4 (context strip only)")
    ap.add_argument("--mic", type=Path, help="single-file mode: the user's mic.wav (talk-overlap metrics)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--results-copy", type=Path, default=HERE / "results",
                    help="committed copy dir for <run>.scores.json")
    a = ap.parse_args()
    if a.video:
        ann = json.loads(a.annotation.read_text()) if a.annotation else None
        out = a.out_dir or a.video.parent
        r = judge_one(a.video, ann, out, a.judge, a.repeat, mic=a.mic, camera=a.camera, label=a.video.parent.name)
        print(json.dumps({k: r[k] for k in ("clip", "judge", "axes", "axes_std", "weighted", "usd", "verdict")}, indent=1))
        return
    if not a.run_dir:
        ap.error("--run-dir or --video required")
    manifest = load_manifest(a.manifest or default_manifest())
    want = [c for c in a.clips.split(",") if c]
    clip_dirs = sorted(d for d in a.run_dir.iterdir() if d.is_dir() and (d / "avatar.mp4").exists()
                       and (not want or d.name in want))
    results = {}

    def work(d: Path):
        ann, mic, cam = find_annotation(d.name, manifest, d)
        return d.name, judge_one(d / "avatar.mp4", ann, d, a.judge, a.repeat, mic=mic, camera=cam,
                                 label=f"{a.run_dir.name}/{d.name}")

    jobs = a.jobs if JUDGES[a.judge]["route"] == "openrouter" else min(a.jobs, 3)
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        for fut in cf.as_completed([ex.submit(work, d) for d in clip_dirs]):
            try:
                name, r = fut.result()
                results[name] = r
                print(f"{name}: weighted={r['weighted']} usd={r['usd_new']}", flush=True)
            except SystemExit:
                raise
            except Exception as e:
                print(f"ERROR {e}", file=sys.stderr, flush=True)
    sp = a.run_dir / "scores.json"
    prev = json.loads(sp.read_text()) if sp.exists() else {}
    judges = prev.get("judges", {})
    judges[a.judge] = {n: {"axes": r["axes"], "axes_judge": r["axes_judge"], "weighted": r["weighted"],
                           "weighted_judge_only": r["weighted_judge_only"],
                           "objective": (r.get("objective") or {}).get("objective"),
                           "flags": (r.get("objective") or {}).get("flags"), "usd": r["usd"]}
                       for n, r in sorted(results.items())}
    # merge with clips judged in earlier invocations
    for n, v in (prev.get("judges", {}).get(a.judge) or {}).items():
        judges[a.judge].setdefault(n, v)
    w = [v["weighted"] for v in judges[a.judge].values() if v["weighted"] is not None]
    summary = {"run": a.run_dir.name, "primary": DEFAULT_JUDGE, "prompt_version": PROMPT_VERSION,
               "weights": WEIGHTS, "alpha_objective": ALPHA, "judges": judges,
               "mean_weighted": {j: round(float(np.mean([v["weighted"] for v in jj.values() if v["weighted"] is not None])), 3)
                                 for j, jj in judges.items() if any(v["weighted"] is not None for v in jj.values())}}
    # agreement between judges
    if len(judges) > 1:
        ids = list(judges)
        agree = {}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                A, B = judges[ids[i]], judges[ids[j]]
                common = sorted(set(A) & set(B))
                pa = [A[c]["weighted_judge_only"] for c in common]
                pb = [B[c]["weighted_judge_only"] for c in common]
                mad = {}
                for ax in AXES:
                    dd = [abs(A[c]["axes_judge"][ax] - B[c]["axes_judge"][ax]) for c in common
                          if A[c]["axes_judge"].get(ax) is not None and B[c]["axes_judge"].get(ax) is not None]
                    mad[ax] = round(float(np.mean(dd)), 3) if dd else None
                agree[f"{ids[i]}~{ids[j]}"] = {"n": len(common), "spearman_weighted": spearman(pa, pb),
                                                "mean_abs_diff": mad}
        summary["agreement"] = agree
    atomic_write(sp, summary)
    try:
        a.results_copy.mkdir(parents=True, exist_ok=True)
        atomic_write(a.results_copy / f"{a.run_dir.name}.scores.json", summary)
    except Exception:
        pass
    print(json.dumps({"scores": str(sp), "mean_weighted": summary["mean_weighted"], "n": len(results),
                      "usd_this_round": round(sum(r["usd_new"] for r in results.values()), 4)}, indent=1))


if __name__ == "__main__":
    main()
