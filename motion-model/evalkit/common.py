"""EVAL — shared paths and small helpers. Every write goes under $E = $EVAL_OUT (default $MM_WORK_DIR/eval).
Data (Seamless canon, splits, fitter params) are read-only inputs under $MM_DATA_DIR (see canon1/paths.py)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[1]   # the motion-model directory (canon1 + fit, retarget, evalkit, unified)
KIT = WT / "evalkit"
sys.path.insert(0, str(WT))
from canon1 import paths as _paths  # noqa: E402

DATA = _paths.DATA                                             # processed data (canon, splits, fitter params)
T = Path(os.environ.get("EVAL_T", str(_paths.WORK)))          # work root: runs/, eval/, p2/
E = Path(os.environ.get("EVAL_OUT", str(T / "eval")))         # outputs
# the avatar repo checkout under test (app/, renderer/); default = the repo this motion-model/ lives in (read-only)
AVATAR = Path(os.environ.get("EVAL_AVATAR", str(WT.parent)))
VENV_PY = Path(os.environ.get("EVAL_PY", sys.executable))     # interpreter for eval subprocesses (numpy, torch)
# the unified runtime code matching the bundles under test (default: this checkout; read-only)
MODEL_CODE = Path(os.environ.get("EVAL_MODEL_CODE", str(WT)))
RENDER_PY = Path(os.environ.get("RENDER_PY", sys.executable))  # interpreter that runs the avatar renderer
MP_MODELS = Path(os.environ.get("MP_MODELS", str(AVATAR / "app/vision")))   # the live page's MediaPipe model files
SEAMLESS = _paths.SEAMLESS
# held-out YouTube/stock clip suite (manifest.json + annotations/) and the recorded mock-avatar voices for it
YT_CLIPS = Path(os.environ.get("EVAL_YT_CLIPS", str(T / "yt_clips")))
AVATAR_VOICES = Path(os.environ.get("EVAL_AVATAR_VOICES", str(T / "avatar_voices")))
SPEND = E / "spend.jsonl"                 # paid-judge ledger {t, who, vendor, usd, what}
SPEND_CAP = float(os.environ.get("EVAL_SPEND_CAP_USD", "3.0"))   # eval's paid judge budget per work dir
FPS = 30

sys.path.insert(0, str(KIT))


def canon_spec():
    """canon1/spec.py of this checkout."""
    import importlib.util
    for p in (WT / "canon1/spec.py",):
        if p.exists():
            sys.path.insert(0, str(p.parent.parent))
            import canon1.spec as spec  # noqa
            return spec
    raise ImportError("canon1/spec.py not found")


def sha_file(p, n=16):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def atomic_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp-", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1, default=_np_default)
    os.replace(tmp, p)


def _np_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def read_wav_mono(path):
    """-> float32 mono in [-1,1], sample rate (wav via soundfile)."""
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x.mean(1), sr


def write_wav(path, x, sr):
    x = np.clip(np.asarray(x, np.float32), -1, 1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(sr))
        w.writeframes((x * 32767).astype("<i2").tobytes())


def frame_rms(x, sr, n):
    """per-30Hz-frame rms of a mono float signal (frame k = [k/30,(k+1)/30))."""
    out = np.zeros(n, np.float32)
    for k in range(n):
        a, b = int(k * sr / FPS), int((k + 1) * sr / FPS)
        if a < len(x):
            seg = x[a:min(b, len(x))]
            out[k] = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
    return out


def log_spend(usd, what, vendor="openrouter"):
    with open(SPEND, "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "who": "eval", "vendor": vendor,
                            "usd": round(float(usd), 6), "what": what}) + "\n")


def eval_spend():
    tot = 0.0
    if SPEND.exists():
        for line in SPEND.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("who") == "eval":
                tot += float(r.get("usd") or 0)
    return tot


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), check=True, **kw)


def ffprobe_duration(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0
