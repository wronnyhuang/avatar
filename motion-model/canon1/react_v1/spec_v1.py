"""Canonical per-frame "human numbers" shared by every dataset, the model and the live server
(first learned-reaction model, react_v1).

ONE definition, imported by every extractor and by the live reaction planner. Do not fork it; changes are
append-only (new helpers, never re-ordered columns).

Coordinates are those of the person's OWN camera, UNMIRRORED (the raw camera image, as the conversation
partner would see them). The live page must feed raw (unmirrored) MediaPipe results, not the selfie preview.

Per person, per frame at 30 fps:
  motion[86]  float   (see MOTION_COLS)
  mask[4]     uint8   validity: face, body, hand_l, hand_r   (1 = measured/valid)
  audio[42]   float   40 causal log-mel + loudness + vad      (see AUDIO_COLS)
"""
from __future__ import annotations

import json
import math

import numpy as np

FPS = 30
SR = 16000                      # audio features are always computed from 16 kHz mono

# --------------------------------------------------------------------------------------------- motion
BS_NAMES = [  # MediaPipe Face Landmarker blendshape order (52, incl. _neutral)
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft",
    "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]
assert len(BS_NAMES) == 52
BS = {n: i for i, n in enumerate(BS_NAMES)}

HEAD_COLS = ["head_yaw", "head_pitch", "head_roll", "head_tx", "head_ty", "head_tz"]
GAZE_COLS = ["gaze_x", "gaze_y"]
BODY_JOINTS = ["shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r"]  # anatomical L/R
POSE_IDX = [11, 12, 13, 14, 15, 16]                                                  # MediaPipe Pose ids
BODY_COLS = [f"{j}_{a}" for j in BODY_JOINTS for a in "xyz"]
HAND_COLS = [f"hand_{s}_{k}" for s in "lr" for k in ("open", "thumb", "point", "palm")]
MOTION_COLS = [f"bs_{n}" for n in BS_NAMES] + HEAD_COLS + GAZE_COLS + BODY_COLS + HAND_COLS
D_MOTION = len(MOTION_COLS)
assert D_MOTION == 86

SL_BS = slice(0, 52)
SL_HEAD_ROT = slice(52, 55)
SL_HEAD_T = slice(55, 58)
SL_GAZE = slice(58, 60)
SL_BODY = slice(60, 78)
SL_HAND_L = slice(78, 82)
SL_HAND_R = slice(82, 86)

# Channel groups for reporting (face / head / gaze / hands / body)
GROUPS = {
    "face": [i for i in range(52) if not BS_NAMES[i].startswith("eyeLook")],
    "gaze": [i for i in range(52) if BS_NAMES[i].startswith("eyeLook")] + list(range(58, 60)),
    "head": list(range(52, 58)),
    "body": list(range(60, 78)),
    "hands": list(range(78, 86)),
}
# mouth columns: excluded from the avatar's loss weighting by default (lip sync owns the mouth on the avatar)
MOUTH_BS = [i for i, n in enumerate(BS_NAMES) if n.startswith(("mouth", "jaw")) or n in ("cheekPuff",)]

MASK_COLS = ["face", "body", "hand_l", "hand_r"]
# which motion columns each mask flag covers
MASK_COVER = {0: list(range(0, 60)), 1: list(range(60, 78)), 2: list(range(78, 82)), 3: list(range(82, 86))}

# --------------------------------------------------------------------------------------------- audio
N_MEL = 40
AUDIO_COLS = [f"mel{i:02d}" for i in range(N_MEL)] + ["loud_db", "vad"]
D_AUDIO = len(AUDIO_COLS)
assert D_AUDIO == 42
N_FFT = 512                      # 32 ms window ENDING at the frame's end time (causal)


def euler_from_mat(m):
    """m: (N,16) row-major 4x4 MediaPipe facial transformation matrices (unmirrored image) ->
    (N,3) yaw, pitch, roll in RADIANS and (N,3) translation in cm. Same formula as the rules-mode motion library."""
    M = np.asarray(m, np.float64).reshape(-1, 4, 4)
    R = M[:, :3, :3]
    yaw = np.arctan2(R[:, 0, 2], R[:, 2, 2])
    pitch = np.arctan2(-R[:, 1, 2], np.hypot(R[:, 0, 2], R[:, 2, 2]))
    roll = np.arctan2(R[:, 0, 1], R[:, 1, 1])
    return np.stack([yaw, pitch, roll], 1), M[:, :3, 3]


def head_translation_feature(t_cm, valid=None):
    """(N,3) cm -> canonical head_t: (t - median over valid frames) / 10.  Training uses the per-session
    median; the live planner uses the running median of the session so far (see RunningMedian)."""
    t = np.asarray(t_cm, np.float64)
    v = np.ones(len(t), bool) if valid is None else np.asarray(valid, bool)
    med = np.median(t[v], 0) if v.any() else np.zeros(3)
    return (t - med) / 10.0


def gaze_from_bs(bs):
    """(N,52) -> (N,2): gaze_x + = toward the person's LEFT eye side naming of MediaPipe
    (0.5*((OutL-InL)+(InR-OutR))), gaze_y + = up. Signs are fixed here; the rig retarget decides screen sides."""
    b = np.asarray(bs, np.float64)
    gx = 0.5 * ((b[:, BS["eyeLookOutLeft"]] - b[:, BS["eyeLookInLeft"]]) +
                (b[:, BS["eyeLookInRight"]] - b[:, BS["eyeLookOutRight"]]))
    gy = 0.5 * ((b[:, BS["eyeLookUpLeft"]] - b[:, BS["eyeLookDownLeft"]]) +
                (b[:, BS["eyeLookUpRight"]] - b[:, BS["eyeLookDownRight"]]))
    return np.stack([gx, gy], 1)


def body_from_pose(lm, width, height, vis_min=0.5):
    """lm: (N,33,>=3[,vis]) MediaPipe Pose NORMALIZED image landmarks (x,y in 0..1, z ~ x scale) or
    (N,6,>=3) already restricted to POSE_IDX. Returns (N,18) body features and (N,) valid mask.
    Frame: origin = shoulder midpoint, unit = 2-D shoulder width in pixels; x = image-right +,
    y = UP +, z = TOWARD the camera +. Valid = both shoulders visible (vis >= vis_min)."""
    a = np.asarray(lm, np.float64)
    if a.shape[1] == 33:
        a = a[:, POSE_IDX]
    px = a[..., 0] * width
    py = a[..., 1] * height
    pz = a[..., 2] * width
    vis = a[..., 3] if a.shape[-1] > 3 else np.ones(px.shape)
    cx = 0.5 * (px[:, 0] + px[:, 1]); cy = 0.5 * (py[:, 0] + py[:, 1]); cz = 0.5 * (pz[:, 0] + pz[:, 1])
    s = np.hypot(px[:, 0] - px[:, 1], py[:, 0] - py[:, 1])
    ok = (vis[:, 0] >= vis_min) & (vis[:, 1] >= vis_min) & (s > 1e-3)
    s = np.where(ok, s, 1.0)
    out = np.stack([(px - cx[:, None]) / s[:, None], -(py - cy[:, None]) / s[:, None],
                    -(pz - cz[:, None]) / s[:, None]], -1)  # (N,6,3)
    # joints that are not visible collapse to a neutral "arm hanging" guess so the vector stays finite
    rest = np.array([[0.5, 0, 0], [-0.5, 0, 0], [0.6, -1.0, 0], [-0.6, -1.0, 0], [0.6, -1.9, 0.1],
                     [-0.6, -1.9, 0.1]])
    # x sign of rest depends on which image side the person's left shoulder is on
    side = np.sign(out[:, 0, 0:1]).reshape(-1, 1, 1)
    side = np.where(side == 0, 1, side)
    rest_s = rest[None] * np.concatenate([side, np.ones_like(side), np.ones_like(side)], -1)
    # in-frame check: pose-lite invents wrists/elbows below the image -> hidden
    xin = (a[..., 0] > -0.02) & (a[..., 0] < 1.02) & (a[..., 1] > -0.02) & (a[..., 1] < 1.02)
    hid = ((vis < vis_min) | ~xin)[..., None]
    out = np.where(hid, rest_s, out)
    out = np.where(ok[:, None, None], out, rest_s)
    return out.reshape(len(a), 18).astype(np.float32), ok


def hand_descriptor(h):
    """h: (21,3) MediaPipe hand landmarks (any consistent metric units, x right, y down, z away).
    Returns [open, thumb, point, palm] where open/thumb/point in 0..1 and palm in -1..1.
    NOTE (integration check 17:20): with MediaPipe's axes the palm sign comes out OPPOSITE to "+1 = palm toward
    camera"; data already uses this convention and react/retarget.py compensates. Do not flip it here."""
    h = np.asarray(h, np.float64)

    def ext(tip, mcp, lo=1.1, hi=1.9):
        r = np.linalg.norm(h[tip] - h[0]) / max(np.linalg.norm(h[mcp] - h[0]), 1e-6)
        return float(np.clip((r - lo) / (hi - lo), 0, 1))

    fi, fm, fr, fp = ext(8, 5), ext(12, 9), ext(16, 13), ext(20, 17)
    op = (fi + fm + fr + fp) / 4
    th = float(np.clip((np.linalg.norm(h[4] - h[17]) / max(np.linalg.norm(h[5] - h[17]), 1e-6) - 0.9) / 0.7,
                       0, 1))
    point = float(np.clip(fi - (fm + fr + fp) / 3, 0, 1))
    n = np.cross(h[5] - h[0], h[17] - h[0])
    nz = -n[2] / max(np.linalg.norm(n), 1e-6)   # z away from camera in MediaPipe -> flip
    return np.array([op, th, point, nz])


def hand_descriptor_side(h, is_left):
    """Anatomical-side aware: palm sign flipped for the right hand so +1 = palm toward camera for both."""
    d = hand_descriptor(h)
    if not is_left:
        d[3] = -d[3]
    return d


def assign_hands_to_wrists(hand_wrists_xy, pose_wrist_l_xy, pose_wrist_r_xy):
    """Assign detected hands to the person's anatomical L/R by proximity to the POSE wrists (15/16).
    MediaPipe hand 'handedness' assumes a mirrored selfie image, so never trust it on raw frames.
    hand_wrists_xy: list of (x,y). Returns dict {'l': idx or None, 'r': idx or None}."""
    out = {"l": None, "r": None}
    if not hand_wrists_xy:
        return out
    P = np.asarray(hand_wrists_xy, np.float64)
    dl = np.linalg.norm(P - np.asarray(pose_wrist_l_xy), axis=1)
    dr = np.linalg.norm(P - np.asarray(pose_wrist_r_xy), axis=1)
    if len(P) == 1:
        out["l" if dl[0] <= dr[0] else "r"] = 0
    else:
        a = dl[0] + dr[1]; b = dl[1] + dr[0]
        out["l"], out["r"] = (0, 1) if a <= b else (1, 0)
    return out


# --------------------------------------------------------------------------------------------- audio
def _mel_fb(n_mel=N_MEL, n_fft=N_FFT, sr=SR, fmin=60.0, fmax=7600.0):
    def hz2mel(f): return 2595.0 * np.log10(1 + f / 700.0)
    def mel2hz(m): return 700.0 * (10 ** (m / 2595.0) - 1)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mel + 2)
    hz = mel2hz(mels)
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mel, len(bins)))
    for i in range(n_mel):
        l, c, r = hz[i], hz[i + 1], hz[i + 2]
        fb[i] = np.clip(np.minimum((bins - l) / (c - l), (r - bins) / (r - c)), 0, None)
    return fb.astype(np.float32)


MEL_FB = _mel_fb()
_WIN = np.hanning(N_FFT).astype(np.float32)


def frame_end_sample(k):
    """sample index (exclusive) at which frame k ends: frame k covers [k/30, (k+1)/30)."""
    return int(round((k + 1) * SR / FPS))


def audio_features(x16k, n_frames=None, vad=None):
    """x16k: float mono 16 kHz in [-1,1]. Returns (T,42): 40 log-mel (natural log of power, floor 1e-6),
    loud_db (20*log10 rms over the frame window, floor -100), vad (0/1). Frame k uses ONLY samples before the
    end of frame k (window = the 512 samples ending there). `vad`: optional precomputed (T,) 0/1 track
    (e.g. a dataset's own VAD resampled to 30 fps); otherwise energy_vad() is used."""
    x = np.asarray(x16k, np.float32)
    T = n_frames if n_frames is not None else int(len(x) * FPS / SR)
    ends = np.array([frame_end_sample(k) for k in range(T)])
    pad = np.concatenate([np.zeros(N_FFT, np.float32), x, np.zeros(N_FFT, np.float32)])
    idx = ends[:, None] + np.arange(-N_FFT, 0)[None, :] + N_FFT
    idx = np.clip(idx, 0, len(pad) - 1)
    seg = pad[idx] * _WIN[None]
    spec = np.abs(np.fft.rfft(seg, axis=1)) ** 2
    mel = np.log(np.maximum(spec @ MEL_FB.T, 1e-6))
    hop = int(round(SR / FPS))
    seg2 = pad[np.clip(ends[:, None] + np.arange(-hop, 0)[None, :] + N_FFT, 0, len(pad) - 1)]
    rms = np.sqrt(np.mean(seg2 ** 2, 1) + 1e-12)
    loud = np.maximum(20 * np.log10(rms + 1e-12), -100.0)
    v = energy_vad(loud) if vad is None else np.asarray(vad, np.float32)[:T]
    return np.concatenate([mel, loud[:, None], v[:, None]], 1).astype(np.float32)


def energy_vad(loud_db, on_margin=18.0, off_margin=12.0, hang=6):
    """Causal energy VAD with hysteresis relative to a running noise floor (10th percentile of the
    last ~10 s). Used for every dataset without its own VAD AND by the live planner, so it is identical."""
    T = len(loud_db)
    out = np.zeros(T, np.float32)
    hist = []
    on = False; hold = 0
    for k in range(T):
        hist.append(loud_db[k])
        if len(hist) > 300:
            hist.pop(0)
        floor = np.percentile(hist, 10)
        if loud_db[k] > floor + on_margin:
            on = True; hold = hang
        elif loud_db[k] < floor + off_margin:
            if hold > 0:
                hold -= 1
            else:
                on = False
        out[k] = 1.0 if on else 0.0
    return out


def resample_to_16k(x, sr):
    if sr == SR:
        return np.asarray(x, np.float32)
    from scipy.signal import resample_poly
    g = math.gcd(int(sr), SR)
    return resample_poly(np.asarray(x, np.float64), SR // g, int(sr) // g).astype(np.float32)


# --------------------------------------------------------------------------------------------- file format
def save_session(path, motion, mask, audio, meta, words=None, extra=None):
    """motion (P,T,86) float, mask (P,T,4) uint8, audio (P,T,42) float, meta dict (see CONTRACT.md),
    words: list (per person) of [[start_s, end_s, word], ...]. extra: dict of extra arrays (prefixed x_)."""
    motion = np.asarray(motion, np.float32); mask = np.asarray(mask, np.uint8); audio = np.asarray(audio, np.float32)
    P, T, D = motion.shape
    assert D == D_MOTION and mask.shape == (P, T, 4) and audio.shape == (P, T, D_AUDIO), (motion.shape, mask.shape, audio.shape)
    assert np.isfinite(motion).all() and np.isfinite(audio).all()
    for k in ("dataset", "session", "persons", "speakers", "split", "source"):
        assert k in meta, k
    kw = dict(fps=np.int32(FPS), motion=motion.astype(np.float16), mask=mask, audio=audio.astype(np.float16),
              meta=np.array(json.dumps(meta)), words=np.array(json.dumps(words or [[] for _ in range(P)])))
    for k, v in (extra or {}).items():
        kw["x_" + k] = v
    np.savez_compressed(path, **kw)


def load_session(path):
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files}
    out["meta"] = json.loads(str(out["meta"]))
    out["words"] = json.loads(str(out["words"]))
    return out


# --------------------------------------------------------------------------------------------- meaning cues
STYLE_VOCAB = ["neutral", "warm", "cheerful", "excited", "playful", "calm", "thoughtful", "serious",
               "concerned", "sad", "surprised", "annoyed"]
GESTURE_TAGS = ["nod", "head_shake", "head_tilt", "brow_raise", "smile", "laugh", "shrug", "beat", "point",
                "open_palms", "wave", "thinking"]
D_CUE = 3 + len(STYLE_VOCAB) + len(GESTURE_TAGS) + 1   # affect(3) + style one-hot + gesture pulses + present flag


def cue_frames(cue_person, T):
    """cue_person: {"affect": [[t0,t1,val,aro,eng,style], ...], "gestures": [[t0,t1,tag], ...]} (seconds)
    -> (T, D_CUE) float32 per-frame cue channels; last column = 1 where any cue info covers the frame."""
    out = np.zeros((T, D_CUE), np.float32)
    for t0, t1, va, ar, en, st in cue_person.get("affect", []):
        a, b = max(0, int(t0 * FPS)), min(T, int(math.ceil(t1 * FPS)))
        if b <= a:
            continue
        out[a:b, 0:3] = (va, ar, en)
        if st in STYLE_VOCAB:
            out[a:b, 3 + STYLE_VOCAB.index(st)] = 1
        out[a:b, -1] = 1
    base = 3 + len(STYLE_VOCAB)
    for t0, t1, tag in cue_person.get("gestures", []):
        if tag not in GESTURE_TAGS:
            continue
        a, b = max(0, int(t0 * FPS)), min(T, max(int(t0 * FPS) + 1, int(math.ceil(t1 * FPS))))
        out[a:b, base + GESTURE_TAGS.index(tag)] = 1
        out[a:b, -1] = 1
    return out
