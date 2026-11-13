"""Audio2Face-3D v3.0 speaking face for the scorecard systems `a2f` and `rules+a2f-mouth`. Self-contained (no
evalkit `common` import: tools/audio has its own `common`), used by evalkit/systems.py and tools/audio/a2f_baseline.py.

A2F per clip = the audio pipeline's run on the avatar side's own voice (p2/audio_work/a2f_eval/eval_<clip>_James.npz: identity
James, neutral emotion, constant noise, 1 s real pre-roll, SDK centre rows = look-ahead 0.25-0.75 s, canon order, raw
weights); computed here with the same settings if missing. Licence: NVIDIA Open Model License.

Mapping onto what the renderer draws (a2fbase_work/a2f_onscreen_calib.json, fitted on TRAIN people's speech only):
  smile  A2F (mean L/R) -> canon smile scale (affine std-match) -> the Muse rig smile by the retarget's own map
         (retarget/maps/_base.json face: smile = clip01(1.25 (s - 0.10))) = how a canon smile reaches the screen.
  lips   A2F mouth dials -> the Muse lip columns lipstream owns (jaw/teeth_gap/lip_open <- jawOpen, round/lip_protrude <-
         pucker, lip_lower <- lowerDown, lip_upper <- upperUp, press/lip_press <- press, spread <- stretch, lip_width <-
         stretch - pucker, lip_tuck/labdent <- roll), affine std-matched to lipstream's own per-column stats (units only).
         tongue*, cheek, voice stay lipstream's (A2F's tongue/cheek are not solved here).
Speaking weight: the avatar's reply spans (systems.utterances = what rules is told it is speaking = the scorecard's
speaking state), min-jerk ramps of RAMP_S in and out.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_MM = Path(__file__).resolve().parents[1]
P = Path(os.environ.get("MM_WORK_DIR", str(_MM / "work"))) / "p2"
WORK = P / "a2fbase_work"
A2F_EVAL = P / "audio_work" / "a2f_eval"
CALIB_JSON = WORK / "a2f_onscreen_calib.json"
IDENT = "James"
PRE = 30
RAMP_S = 0.15
FPS = 30
WT = _MM
RENDERER = Path(os.environ.get("EVAL_AVATAR", str(_MM.parent))) / "renderer"

LIP_SRC = {
    "jaw": ["jawOpen"], "teeth_gap": ["jawOpen"], "lip_open": ["jawOpen"],
    "round": ["mouthPucker"], "lip_protrude": ["mouthPucker"],
    "lip_lower": ["mouthLowerDownLeft", "mouthLowerDownRight"],
    "lip_upper": ["mouthUpperUpLeft", "mouthUpperUpRight"],
    "press": ["mouthPressLeft", "mouthPressRight"], "lip_press": ["mouthPressLeft", "mouthPressRight"],
    "spread": ["mouthStretchLeft", "mouthStretchRight"],
    "lip_width": ["mouthStretchLeft", "mouthStretchRight", "-mouthPucker"],
    "lip_tuck": ["mouthRollLower", "mouthRollUpper"], "labdent": ["mouthRollLower", "mouthRollUpper"],
}
SMILE = ["mouthSmileLeft", "mouthSmileRight"]


def spec():
    if str(WT) not in sys.path:
        sys.path.insert(0, str(WT))
    from canon1 import spec as S
    return S


def src_of(A, names):
    """(n,52) canon-order weights -> (n,) mean of the named dials ('-name' subtracts)."""
    B = spec().BS
    v = np.mean([A[:, B[n]] for n in names if not n.startswith("-")], 0)
    for n in names:
        if n.startswith("-"):
            v = v - A[:, B[n[1:]]]
    return v


def load_calib():
    return json.loads(CALIB_JSON.read_text())


def ranges():
    if str(RENDERER) not in sys.path:
        sys.path.insert(0, str(RENDERER))
    from anim.rig import RANGES
    return RANGES


def smile_canon(raw, cal):
    c = cal["smile_canon"]
    return np.clip(c["y_mean"] + (src_of(raw, SMILE) - c["a_mean"]) * c["y_sd"] / max(c["a_sd"], 1e-6), 0, 1)


def smile_rig(raw, cal):
    return np.clip(1.25 * (smile_canon(raw, cal) - 0.10), 0, 1)


def lip_cols(raw, cal):
    R = ranges()
    out = {}
    for col, d in cal["lip"].items():
        lo, hi = R.get(col, (0.0, 1.0))
        x = src_of(raw, d["src"])
        out[col] = np.clip(d["y_mean"] + (x - d["a_mean"]) * d["y_sd"] / max(d["a_sd"], 1e-6), lo, hi).astype(np.float32)
    return out


def a2f_raw(clip, threads=4):
    """(N,52) raw A2F weights in canon order for the clip's avatar voice (frame k = clip time k/30)."""
    p = A2F_EVAL / f"eval_{clip['id']}_{IDENT}.npz"
    if p.exists():
        return np.load(p)["a2f"].astype(np.float32)
    q = WORK / "a2f_clips" / f"{clip['id']}_{IDENT}.npz"
    if q.exists():
        return np.load(q)["a2f"].astype(np.float32)
    # same settings as tools/audio/a2f_eval.py run (1 s real pre-roll of the avatar's own wav)
    sys.path.insert(0, str(WT / "tools/audio"))
    import a2f as A
    S = spec()
    import soundfile as sf
    from scipy.signal import resample_poly
    import math
    n30 = int(round(float(clip["duration_s"]) * FPS))
    k0 = int(clip["frame0"]) - PRE
    s0, s1 = S.frame_end_sample(k0 - 1), S.frame_end_sample(k0 + n30 + PRE - 1)
    info = sf.info(clip["voice_wav"])
    sr = info.samplerate
    a, b = max(0, int(s0 * sr / 16000)), min(info.frames, int(math.ceil(s1 * sr / 16000)))
    x, _ = sf.read(clip["voice_wav"], start=a, stop=b, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 16000:
        g = math.gcd(sr, 16000)
        x = resample_poly(x.astype(np.float64), 16000 // g, sr // g).astype(np.float32)
    x16 = np.zeros(s1 - s0, np.float32)
    o = int(round(a * 16000 / sr)) - s0
    lo, hi = max(0, o), min(len(x16), o + len(x))
    x16[lo:hi] = x[lo - o:hi - o]
    m = A.A2F(IDENT, threads=threads)
    r = m.run_track(x16, n30=n30 + PRE)
    can = A.to_canon(r["w60"], r["pose_names"], n30 + PRE)[PRE:]
    q.parent.mkdir(parents=True, exist_ok=True)
    np.savez(q, a2f=can.astype(np.float32))
    return can.astype(np.float32)


def speak_weight(spans, N):
    """reply spans [[t0, t1, text]] -> (N,) 0..1: min-jerk ramp in over RAMP_S from t0, out over RAMP_S after t1."""
    on = np.zeros(N, bool)
    for a, b, *_ in spans:
        on[max(0, int(a * FPS)):min(N, int(b * FPS) + 1)] = True
    w = np.zeros(N, np.float32)
    x = 0.0
    step = 1.0 / (RAMP_S * FPS)
    for k in range(N):
        x = min(1.0, x + step) if on[k] else max(0.0, x - step)
        w[k] = x
    return w * w * w * (10 + w * (6 * w - 15))


VARIANTS = {   # system -> (A2F smile into the page rig?, A2F lips replace lipstream at the renderer?)
    "a2f": (True, True),                 # A2F owns the whole speaking face it covers (mouth + smile)
    "rules+a2f-mouth": (False, True),    # rules' smile, A2F's mouth (renderer-side lip source)
    "rules+a2f-page": (True, False),     # A2F through today's page: SPEECH_GROUP yields -> lipstream mouth, A2F smile
}


def build(clip, spans, base_rig, out_dir, variant):
    """rules' rig.jsonl -> <out_dir>/rig.jsonl (A2F's smile while speaking if the variant takes it) + lip_override.npz
    (A2F's lips while speaking, if the variant takes them)."""
    use_smile, use_lips = VARIANTS[variant]
    cal = load_calib()
    raw = a2f_raw(clip)
    N = int(round(float(clip["duration_s"]) * FPS))
    raw = raw[:N] if len(raw) >= N else np.concatenate([raw, np.repeat(raw[-1:], N - len(raw), 0)])
    w = speak_weight(spans, N)
    sm = smile_rig(raw, cal)
    lines = Path(base_rig).read_text().splitlines()
    head = json.loads(lines[0])
    names = head["rig_params"]
    i_sm = names.index("smile")
    head.update({"source": f"{variant}: rules rig + Audio2Face-3D v3.0 speaking face", "base": str(base_rig),
                 "lip_owner_while_speaking": "a2f (lip_override.npz)" if use_lips else "lipstream",
                 "smile_while_speaking": "a2f" if use_smile else "rules",
                 "a2f": f"identity {IDENT}, calib {CALIB_JSON.name}", "ramp_s": RAMP_S})
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rig.jsonl", "w") as f:
        f.write(json.dumps(head) + "\n")
        for ln in lines[1:]:
            if not ln.strip():
                continue
            r = json.loads(ln)
            if use_smile and "a" in r and "t" in r:
                k = min(N - 1, max(0, int(round(float(r["t"]) * FPS))))
                a = r["a"]
                a[i_sm] = round(float(a[i_sm] + (sm[k] - a[i_sm]) * w[k]), 4)
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    ovp = out_dir / "lip_override.npz"
    if use_lips:
        lc = lip_cols(raw, cal)
        cols = sorted(lc)
        np.savez(ovp, cols=np.array(cols), values=np.stack([lc[c] for c in cols], 1).astype(np.float32), w=w.astype(np.float32),
                 meta=np.array(json.dumps({"clip": clip["id"], "variant": variant, "calib": str(CALIB_JSON)})))
    elif ovp.exists():
        ovp.unlink()
    return out_dir / "rig.jsonl"
