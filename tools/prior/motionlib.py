"""Shared motion representation: raw MediaPipe extraction (.npz) -> per-person 30 Hz motion table + states.

Units of the motion table (all relative to the person's clip median = their neutral):
  yaw, pitch, roll   degrees, rig sign convention (yaw + = toward screen right, pitch + = nose down,
                     roll + = crown toward screen right)
  tx, ty, tz         cm (screen right, up, toward camera)
  gx, gy             eye-in-head gaze, degrees (screen right, up)
  blink_l, blink_r   0..1 eye closure (SCREEN-left / SCREEN-right eye; contract side convention)
  brow_l, brow_r     -1..1 brow (screen sides); smile 0..1; squint, wide 0..1; jaw 0..1
  arm                wrist speed, shoulder-widths / s (mean of the visible wrists)
"""
import numpy as np
from scipy.signal import butter, filtfilt, medfilt

FPS = 30
BN = ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff',
      'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight',
      'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight',
      'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight',
      'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel',
      'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker',
      'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft',
      'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight',
      'noseSneerLeft', 'noseSneerRight']
B = {n: i for i, n in enumerate(BN)}
# landmark subset order in extract.py
LM = {n: i for i, n in enumerate(["nose", "eR_out", "eR_in", "eL_in", "eL_out", "eR_top", "eR_bot", "eL_top", "eL_bot",
                                  "irisR", "irisL", "top", "chin", "sideR", "sideL"])}
COLS = ["yaw", "pitch", "roll", "tx", "ty", "tz", "gx", "gy", "blink_l", "blink_r", "brow_l", "brow_r", "smile",
        "squint", "wide", "jaw", "arm"]
C = {n: i for i, n in enumerate(COLS)}


def euler_from_mat(m):
    """m: (N,16) row-major 4x4 MediaPipe facial transformation matrices -> yaw,pitch,roll (deg), t (cm)."""
    M = m.reshape(-1, 4, 4)
    R = M[:, :3, :3]
    yaw = np.degrees(np.arctan2(R[:, 0, 2], R[:, 2, 2]))
    pitch = np.degrees(np.arctan2(-R[:, 1, 2], np.hypot(R[:, 0, 2], R[:, 2, 2])))
    roll = np.degrees(np.arctan2(R[:, 0, 1], R[:, 1, 1]))
    return yaw, pitch, roll, M[:, :3, 3]


def blend_side(bl, name):
    """MediaPipe '...Left' is the subject's left = SCREEN-RIGHT in an unmirrored video (verified in
    check_sides). Returns (screen_left, screen_right)."""
    return bl[:, B[name + "Right"]], bl[:, B[name + "Left"]]


def segments(mask, min_len):
    """runs of True -> list of (start, end) with end exclusive, length >= min_len"""
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    s = np.where(d == 1)[0]; e = np.where(d == -1)[0]
    return [(a, b) for a, b in zip(s, e) if b - a >= min_len]


def lowpass(x, hz, fs=FPS, order=2):
    if len(x) < 15:
        return x
    b, a = butter(order, hz / (fs / 2))
    return filtfilt(b, a, x, axis=0, padlen=min(len(x) - 1, 3 * max(len(a), len(b))))


def person_table(d, s):
    """raw arrays for stream s -> (table (N, len(COLS)) with NaN where face absent, valid mask, extras)"""
    pres = d["present"][s].astype(bool)
    bl = d["blend"][s]; lm = d["lmk"][s]; po = d["pose"][s]
    N = len(pres)
    yaw, pitch, roll, t = euler_from_mat(d["mat"][s])
    T = np.full((N, len(COLS)), np.nan, np.float32)
    T[:, 0], T[:, 1], T[:, 2] = yaw, pitch, roll
    T[:, 3], T[:, 4], T[:, 5] = t[:, 0], t[:, 1], t[:, 2]
    # eye-in-head gaze from iris vs eye corners (image coords; x right, y down)
    def eye(io, oo, ii, tt, bb):
        cx = (lm[:, oo, 0] + lm[:, ii, 0]) / 2; cy = (lm[:, tt, 1] + lm[:, bb, 1]) / 2
        w = np.hypot(lm[:, oo, 0] - lm[:, ii, 0], lm[:, oo, 1] - lm[:, ii, 1]) + 1e-6
        return (lm[:, io, 0] - cx) / w, (lm[:, io, 1] - cy) / w
    ax, ay = eye(LM["irisR"], LM["eR_out"], LM["eR_in"], LM["eR_top"], LM["eR_bot"])
    bx, by = eye(LM["irisL"], LM["eL_out"], LM["eL_in"], LM["eL_top"], LM["eL_bot"])
    ox, oy = (ax + bx) / 2, (ay + by) / 2
    # eye width ~30 mm, eyeball radius ~12 mm -> rotation = asin(offset * 30/12)
    T[:, 6] = np.degrees(np.arcsin(np.clip(ox * 2.5, -1, 1)))
    # vertical: iris vs lid midpoint is lid-contaminated; blend with eyeLookUp/Down blendshapes
    up = (bl[:, B["eyeLookUpLeft"]] + bl[:, B["eyeLookUpRight"]]) / 2
    dn = (bl[:, B["eyeLookDownLeft"]] + bl[:, B["eyeLookDownRight"]]) / 2
    T[:, 7] = 0.5 * np.degrees(np.arcsin(np.clip(-oy * 2.5, -1, 1))) + 0.5 * 25.0 * (up - dn)
    bLs, bRs = blend_side(bl, "eyeBlink")
    T[:, 8], T[:, 9] = bLs, bRs
    oLs, oRs = blend_side(bl, "browOuterUp"); dLs, dRs = blend_side(bl, "browDown")
    inner = bl[:, B["browInnerUp"]]
    T[:, 10] = np.clip(0.5 * oLs + 0.5 * inner - dLs, -1, 1)
    T[:, 11] = np.clip(0.5 * oRs + 0.5 * inner - dRs, -1, 1)
    T[:, 12] = (bl[:, B["mouthSmileLeft"]] + bl[:, B["mouthSmileRight"]]) / 2
    T[:, 13] = (bl[:, B["eyeSquintLeft"]] + bl[:, B["eyeSquintRight"]]) / 2
    T[:, 14] = (bl[:, B["eyeWideLeft"]] + bl[:, B["eyeWideRight"]]) / 2
    T[:, 15] = bl[:, B["jawOpen"]]
    # arms: wrist speed in shoulder widths / s (pose sampled every 2nd frame -> diff over 2 frames)
    sh = np.hypot(po[:, 1, 0] - po[:, 2, 0], po[:, 1, 1] - po[:, 2, 1]) + 1e-6
    vis_sh = (po[:, 1, 3] > .5) & (po[:, 2, 3] > .5)
    sp = []
    shs = np.convolve(sh, np.ones(15) / 15, "same")
    for j in (5, 6):
        p = po[:, j, :2] / shs[:, None]
        p = np.stack([np.convolve(p[:, 0], np.ones(9) / 9, "same"), np.convolve(p[:, 1], np.ones(9) / 9, "same")], 1)
        v = np.zeros(N); v[2:] = np.hypot(*(p[2:] - p[:-2]).T) * FPS / 2
        v[~((po[:, j, 3] > .5) & vis_sh)] = np.nan
        sp.append(v)
    sp = np.array(sp)
    with np.errstate(all="ignore"):
        T[:, 16] = np.nanmean(sp, 0)
    hands_vis = np.isfinite(sp).any(0)
    face_w = np.hypot(lm[:, LM["sideR"], 0] - lm[:, LM["sideL"], 0], lm[:, LM["sideR"], 1] - lm[:, LM["sideL"], 1])
    T[~pres, :16] = np.nan
    return T, pres, dict(face_w=face_w, hands_vis=hands_vis)


def cut_mask(cut):
    """frame indices that start a new shot"""
    c = cut.copy()
    thr = max(18.0, np.percentile(c, 99.5) if len(c) else 18.0)
    return c > min(thr, 30.0)


def clean_segments(T, pres, cuts, face_w, min_sec=3.0):
    """valid continuous runs: face present, no shot cut, no pose jumps, face big enough"""
    ok = pres.copy()
    ok &= face_w * 1.0 > 0.0
    jump = np.zeros(len(ok), bool)
    dy = np.abs(np.diff(T[:, :3], axis=0))
    jump[1:] = np.nan_to_num(dy, nan=0).max(1) > 12.0          # >12 deg in one frame = tracking glitch / cut
    ok &= ~cuts & ~jump
    return segments(ok, int(min_sec * FPS))


def smooth_table(T, segs):
    """zero-phase low-pass of the continuous channels inside each segment (tracker noise), blinks untouched"""
    S = T.copy()
    for a, b in segs:
        for j, hz in ((0, 6), (1, 6), (2, 6), (3, 4), (4, 4), (5, 3), (6, 8), (7, 8), (10, 8), (11, 8), (12, 6),
                      (13, 8), (14, 8)):
            S[a:b, j] = lowpass(T[a:b, j], hz)
    return S


# ------------------------------------------------------------------ audio -> 30 Hz voice features + VAD
def voice_30hz(pr, N):
    """pr: (M,3) 100 Hz [dB, f0, aperiodicity] -> per 30 Hz frame: rms_db, f0 (Hz, 0 unvoiced), voiced, speech"""
    M = len(pr)
    db, f0, ap = pr[:, 0], pr[:, 1], pr[:, 2]
    floor = np.percentile(db, 10) if M else -80
    peak = np.percentile(db, 98) if M else -20
    voiced = f0 > 0
    # speech: loud enough + voicing nearby (300 ms), hangover; gaps < 250 ms filled
    loud = db > max(floor + 10, peak - 35)
    vk = np.convolve(voiced.astype(float), np.ones(31) / 31, "same") > 0.12
    sp = loud & vk
    sp = fill_gaps(sp, 25)
    sp = remove_short(sp, 10)
    idx = np.clip((np.arange(N) / FPS * 100).astype(int), 0, max(M - 1, 0))
    out = np.zeros((N, 4), np.float32)
    if M:
        out[:, 0] = db[idx]; out[:, 1] = f0[idx]; out[:, 2] = voiced[idx]; out[:, 3] = sp[idx]
    return out, floor, peak


def fill_gaps(m, n):
    m = m.copy()
    for a, b in segments(~m, 1):
        if b - a < n and a > 0 and b < len(m):
            m[a:b] = True
    return m


def remove_short(m, n):
    m = m.copy()
    for a, b in segments(m, 1):
        if b - a < n:
            m[a:b] = False
    return m


def mouth_activity(jaw):
    """rolling mean |d jaw/dt| over 0.5 s (speech makes the jaw flutter)"""
    j = np.nan_to_num(jaw, nan=0.0)
    v = np.abs(np.diff(j, prepend=j[:1])) * FPS
    return np.convolve(v, np.ones(15) / 15, "same")


def sync_corr(jaw, lev, present, win=90):
    """sliding-window (3 s) correlation between this person's jawOpen and the audio level (max over
    0..100 ms audio lead) - a SyncNet-style 'is this face the voice' score per frame"""
    N = len(jaw)
    j = np.nan_to_num(jaw, nan=0.0).astype(np.float64)
    best = np.full(N, -1.0)
    k = np.ones(win) / win
    for lag in (0, 1, 2, 3):
        a = lev[: N - lag] if lag else lev
        b = j[lag:]
        ma = np.convolve(a, k, "same"); mb = np.convolve(b, k, "same")
        cov = np.convolve(a * b, k, "same") - ma * mb
        va = np.convolve(a * a, k, "same") - ma ** 2; vb = np.convolve(b * b, k, "same") - mb ** 2
        r = cov / np.sqrt(np.maximum(va, 1e-9) * np.maximum(vb, 1e-9))
        r[(vb < 1e-5)] = 0
        rr = np.full(N, -1.0); rr[lag:] = r if lag else r
        best = np.maximum(best, rr)
    pm = np.convolve(present.astype(float), k, "same") > 0.7
    best[~pm] = -1
    return best


def assign_speech_sync(speech, rs, present, thr=0.22):
    """speech (N,) bool; rs list per person of sync_corr -> per person self_talk"""
    P = len(rs)
    out = []
    for i in range(P):
        st = speech & (rs[i] > thr) & present[i]
        if P == 2:
            st &= rs[i] > rs[1 - i] + 0.05
        out.append(remove_short(fill_gaps(st, 15), 15) & speech)
    return out


def jaw_std(jaw, w=30):
    """rolling 1-s std of jawOpen: ~0.05-0.13 while this face talks, < 0.02 while it is silent"""
    x = np.nan_to_num(jaw, nan=0.0).astype(np.float64)
    k = np.ones(w) / w
    m = np.convolve(x, k, "same")
    return np.sqrt(np.maximum(np.convolve(x * x, k, "same") - m * m, 0))


def assign_speech_jaw(speech, jstds, present, hi=0.035, lo=0.02):
    """per person self_talk: speech (VAD, bridged 0.4 s) & this face's jaw is moving (hysteresis hi/lo);
    with two faces the talker must also move more than the other."""
    P = len(jstds)
    spb = fill_gaps(speech, 12)
    out = []
    for i in range(P):
        j = jstds[i]
        talk = np.zeros(len(j), bool); on = False
        for t in range(len(j)):
            if not present[i][t]:
                on = False
            elif on and j[t] < lo:
                on = False
            elif not on and j[t] > hi:
                on = True
            talk[t] = on
        st = spb & talk
        if P == 2:
            st &= j > 1.2 * jstds[1 - i]
        out.append(remove_short(fill_gaps(st, 9), 9))
    return out


def assign_speech(speech, acts, present):
    """speech (N,) bool; acts list per person of activity; -> per person self_talk (N,) bool"""
    P = len(acts)
    out = []
    for i in range(P):
        a = acts[i]
        sil = ~speech & present[i]
        base = np.percentile(a[sil], 75) if sil.sum() > 30 else 0.3
        spk = speech & present[i]
        p90 = np.percentile(a[spk], 90) if spk.sum() > 30 else 1.0
        thr = max(base * 1.3, 0.3 * p90, 0.25)
        st = speech & (a > thr)
        if P == 2:
            o = acts[1 - i]
            st &= a > 1.25 * o
        out.append(st)
    # smoothing: fill < 0.6 s gaps, drop < 0.3 s islands
    out = [remove_short(fill_gaps(s, 18), 9) & speech | (s & speech) for s in out]
    if P == 2:
        both = out[0] & out[1]
        out[0] = out[0] & ~(both & (acts[0] < acts[1])); out[1] = out[1] & ~(both & (acts[1] <= acts[0]))
    return out


def states(self_talk, speech, other_talk=None):
    """0 idle, 1 speaking (own turn incl. pauses < 1.2 s), 2 listening (someone else speaking)"""
    selfturn = fill_gaps(self_talk, int(1.2 * FPS))
    if other_talk is None:
        other = speech & ~self_talk
    else:
        other = other_talk | (speech & ~self_talk)
    other = fill_gaps(remove_short(other, 6), int(1.2 * FPS)) & ~selfturn
    st = np.zeros(len(speech), np.int8)
    st[selfturn] = 1
    st[other] = 2
    return st
