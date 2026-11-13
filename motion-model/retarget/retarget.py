"""CANON v1 (189 numbers / frame, canon1/spec.py) -> the avatar renderer's rig params, per character.

    from retarget.retarget import Retargeter, CHARACTERS
    rt = Retargeter("muse")                       # maps/muse.json; stateful (causal extras: ears, tail, breath)
    rig = rt.step(motion189, mask9=None)          # one 30 Hz frame -> {rig param: value}, clipped to anim/rig.py RANGES
    rigs = rt.batch(motion (T,189), mask (T,9))   # -> {param: (T,) float32}   (same numbers as T x step)
    part, groups = rt.inverse(rig_dict)           # rig values -> approx. canon (189,) + the canon groups they define

SIDES AND FRAMES.  Canon is anatomical: x = the person's LEFT, y up, z forward.  The avatar is a person facing the
viewer, so its anatomical left is SCREEN-RIGHT: canon x = screen right, y = up, z = toward the camera, and every rig
`_r` param (screen right) comes from a canon `_l` / "...Left" source; `_l` from `_r` / "...Right".  Rotations map 1:1
(the renderer composes rotY(yaw) rotX(pitch) rotZ(-roll), standard right-handed matrices): head_yaw = yaw about y
(+ toward screen right), head_pitch = about x (+ nose down), head_roll = -(about z) (+ crown toward screen right).

WHAT DRIVES WHAT (numbers live in maps/<character>.json, population references in maps/_pop_stats.json):
  * body_yaw / body_pitch / body_lean: the CHEST (root_rot o spine1..3) orientation relative to the population's
    neutral chest, as yaw/pitch/roll, divided by the character's own radians per unit (muse 0.40/0.28/0.20, animals
    and shapes 0.35/0.15/0.12) x gain; body_bounce from root_shift_y.
  * head_*: the head (neck o head) relative to the chest and to the neutral, x gain, soft knee inside the range.
  * gaze_x/y: the renderer's eyes are WORLD-referenced (gaze 0 = at the camera whatever the head does), so rig gaze =
    (head world yaw/pitch + canon eye-in-head gaze) in degrees / deg-per-unit.
  * hands: FK of the canonical arm (collar, shoulder, elbow) -> wrist in the chest frame, in arm lengths, relative to
    the population's resting wrist (hands in the lap / on the desk) -> the contract's hand target space by a
    piecewise map anchored on the contract key poses (REST, CHIN, WAVE_UP, HAND_ON_HEART, POINT_SIDE, POINT_CAMERA)
    -> the character's own IK (its shader keeps the paw out of the body/head, on its side, in reach).  hand_turn =
    the palm normal's roll about the forearm vs "palm to the midline" (the shader's own frame).  open / point / thumb
    from finger curls (MANO flexion, relative to the population).  shoulder (shrug) from the collar's lift.
  * face: 52 MediaPipe blendshapes -> brows, knit, blinks, eye_wide, squint, smile, mouth_open (only while silent:
    the compositor zeroes it when lip sync owns the mouth), nostril / nose_wiggle (animals).
  * rule-driven extras (causal, deterministic): ears from brows / eye-wide / smile / knit + a spring on head pitch
    velocity + rare flicks; tail (fox, monkey) wag / raise / curl from arousal (smile + motion energy); breath.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from canon1 import spec
from . import skeleton as sk

HERE = os.path.dirname(os.path.abspath(__file__))
MAPS = os.path.join(HERE, "maps")
CHARACTERS = ["muse", "fox", "bear", "rabbit", "hippo", "monkey", "orb", "cube", "drop", "star", "pyramid", "blob"]
FPS = spec.FPS
DT = 1.0 / FPS

# anim/rig.py RANGES (renderer clip) for every param the retarget may drive
RANGES = {
    "smile": (0, 1), "brow_knit": (0, 1), "blink_l": (0, 1), "blink_r": (0, 1), "ear_flick_l": (0, 1),
    "ear_flick_r": (0, 1), "nostril": (0, 1), "breath": (-1, 1),
    "head_yaw": (-0.8, 0.8), "head_pitch": (-0.6, 0.6), "head_roll": (-0.6, 0.6),
    "head_x": (-0.3, 0.3), "head_y": (-0.3, 0.3), "head_z": (-0.3, 0.3),
    "gaze_x": (-1, 1), "gaze_y": (-1, 1), "brow_l": (-1, 1), "brow_r": (-1, 1), "ear_l": (-1, 1), "ear_r": (-1, 1),
    "body_lean": (-1, 1), "body_bounce": (-1, 1), "tail_wag": (-1, 1), "tail_raise": (0, 1), "body_yaw": (-1, 1),
    "hand_l_turn": (-1, 1), "hand_r_turn": (-1, 1), "ear_l_turn": (-1, 1), "ear_r_turn": (-1, 1),
    "nose_wiggle": (-1, 1), "body_pitch": (-1, 1), "hand_l_open": (0, 1), "hand_r_open": (0, 1),
    "thumb_l": (0, 1), "thumb_r": (0, 1), "point_l": (0, 1), "point_r": (0, 1), "shoulder_l": (0, 1),
    "shoulder_r": (0, 1), "mouth_open": (0, 1), "eye_wide": (0, 1), "squint": (0, 1), "tail_curl": (0, 1),
    "tongue_out": (0, 1), "fingers_l": (0, 5), "fingers_r": (0, 5), "gaze_free": (0, 1),
}
for _s in "lr":
    for _a in "xyz":
        RANGES[f"hand_{_s}_{_a}"] = (-1.2, 1.4)

# rig param -> the canon group it expresses (screen side r = canon anatomical l).  Used by the compositor.
RIG_GROUP = {
    "body_yaw": "torso", "body_pitch": "torso", "body_lean": "torso", "body_bounce": "torso",
    "head_yaw": "head", "head_pitch": "head", "head_roll": "head", "head_x": "head", "head_y": "head", "head_z": "head",
    "hand_r_x": "arm_l", "hand_r_y": "arm_l", "hand_r_z": "arm_l", "hand_r_turn": "arm_l", "shoulder_r": "arm_l",
    "hand_l_x": "arm_r", "hand_l_y": "arm_r", "hand_l_z": "arm_r", "hand_l_turn": "arm_r", "shoulder_l": "arm_r",
    "hand_r_open": "fingers_l", "thumb_r": "fingers_l", "point_r": "fingers_l", "fingers_r": "fingers_l",
    "hand_l_open": "fingers_r", "thumb_l": "fingers_r", "point_l": "fingers_r", "fingers_l": "fingers_r",
    "brow_l": "face_upper", "brow_r": "face_upper", "brow_knit": "face_upper", "blink_l": "face_upper",
    "blink_r": "face_upper", "eye_wide": "face_upper", "squint": "face_upper", "nostril": "face_upper",
    "nose_wiggle": "face_upper",
    "smile": "mouth", "mouth_open": "mouth", "tongue_out": "mouth",
    "gaze_x": "gaze", "gaze_y": "gaze", "gaze_free": "gaze",
}
# whole-body / root columns (intent rules): a commanded move that drives them owns the torso (and everything rides on it)
for _p in ("root_x", "root_y", "root_z", "root_yaw", "root_pitch", "root_roll", "squash", "stand",
           "foot_l_x", "foot_l_y", "foot_l_z", "foot_r_x", "foot_r_y", "foot_r_z"):
    RIG_GROUP[_p] = "torso"
EXTRAS = ["ear_l", "ear_r", "ear_l_turn", "ear_r_turn", "ear_flick_l", "ear_flick_r", "tail_wag", "tail_raise",
          "tail_curl", "breath"]

B = spec.BS
# population rest of the canon blendshapes = the median of Seamless train canon (the "smplh+mapper" face, the
# distribution the model is trained on and outputs; data/canon1/seamless, 160k frames).  NOTE these are far
# from raw MediaPipe's (smile 0.19 vs 0.006, brows ~3x): the retarget is calibrated on what the model outputs.
REST_BS = {
    "eyeBlinkLeft": 0.193, "eyeBlinkRight": 0.161, "browOuterUpLeft": 0.099, "browOuterUpRight": 0.075,
    "browInnerUp": 0.026, "browDownLeft": 0.089, "browDownRight": 0.083, "mouthSmileLeft": 0.207,
    "mouthSmileRight": 0.178, "eyeWideLeft": 0.004, "eyeWideRight": 0.004, "eyeSquintLeft": 0.407,
    "eyeSquintRight": 0.372, "noseSneerLeft": 0.0, "noseSneerRight": 0.0, "jawOpen": 0.033,
    "eyeLookDownLeft": 0.274, "eyeLookDownRight": 0.271, "eyeLookInLeft": 0.047, "eyeLookInRight": 0.165,
    "eyeLookOutLeft": 0.162, "eyeLookOutRight": 0.046, "eyeLookUpLeft": 0.06, "eyeLookUpRight": 0.055,
}


# ------------------------------------------------------------------------------------------- small math
def _c01(v):
    return np.minimum(np.maximum(v, 0.0), 1.0)


def _soft(v, L, k=0.6):
    """identity up to k*L, then a tanh shoulder that approaches L."""
    v = np.asarray(v, np.float64)
    a = np.abs(v); kn = k * L; r = (1 - k) * L
    return np.sign(v) * np.where(a <= kn, a, kn + r * np.tanh((a - kn) / r))


def _soft_inv(y, L, k=0.6):
    kn = k * L; r = (1 - k) * L
    a = np.minimum(np.abs(y), L * (1 - 1e-6))
    return np.sign(y) * np.where(a <= kn, a, kn + r * np.arctanh(np.clip((a - kn) / r, 0, 1 - 1e-9)))


_CURL_IDX = np.array([[9 * kk + 3 * q + 2 for q in range(3)] for kk in range(5)])   # (5,3) z cols per finger


def _cross(a, b):
    return np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1], a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], -1)


def _pwl(x, xs, ys):
    return np.interp(np.asarray(x, np.float64), xs, ys)


def _smoothstep(x):
    x = _c01(x)
    return x * x * (3 - 2 * x)


def _deep_update(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_update(a[k], v)
        else:
            a[k] = v
    return a


_POP = None


def pop_stats():
    global _POP
    if _POP is None:
        _POP = json.load(open(os.path.join(MAPS, "_pop_stats.json")))
    return _POP


def neutral_motion():
    """(189,) the population neutral human: median joint rotations, MANO-median fingers, rest blendshapes, gaze 0."""
    P = pop_stats()
    m = np.zeros(spec.D_MOTION)
    for j, v in P["neutral_body"].items():
        m[spec.jcols(j)] = v
    m[45:135] = P["neutral_fingers"]
    for n, v in REST_BS.items():
        m[135 + B[n]] = v
    return m


def load_map(character):
    base = json.load(open(os.path.join(MAPS, "_base.json")))
    path = os.path.join(MAPS, f"{character}.json")
    if not os.path.exists(path):
        raise KeyError(f"no retarget map for {character!r} (one of {CHARACTERS})")
    return _deep_update(base, json.load(open(path)))


# ------------------------------------------------------------------------------------------- the retargeter
class Retargeter:
    def __init__(self, character="muse", seed=0, overrides=None):
        self.character = character
        self.cfg = load_map(character)
        if overrides:
            _deep_update(self.cfg, overrides)
        c = self.cfg
        self.covered = list(c["covered"])
        self.extras = [p for p in EXTRAS if p in self.covered]
        self.neutral = neutral_motion()
        n0 = self.neutral[None]
        fk0 = sk.upper_fk(n0, root=False)
        self.R_chest0 = fk0["spine3"][0][0]
        self.R_head0_rel = (fk0["spine3"][0][0].T @ fk0["head"][0][0])
        self.R_head0_world = fk0["head"][0][0]
        P = pop_stats()
        self.wrist_ref = {s: np.asarray(P["wrist"][s]["median"]) for s in "lr"}
        self.seed = seed
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self):
        rs = np.random.default_rng(self.seed + 7)
        self.t = 0
        self.ear_x = np.zeros(2); self.ear_v = np.zeros(2); self.prev_pitch = None; self.prev_pv = 0.0
        self.prev_head = None
        self.energy = 0.0; self.arousal = 0.3
        self.tail_ph = float(rs.uniform(0, 2 * math.pi))
        self.breath_ph = float(rs.uniform(0, 2 * math.pi))
        self.breath_hz = 0.24
        self.rs = rs
        self.next_flick = float(rs.uniform(4, 10)); self.flick_side = 0; self.flick_t = -9.0
        self.turn_lp = {"l": None, "r": None}
        self.base_ema = {p: 0.0 for p in self.cfg.get("baseline", {}).get("params", {}) if p in self.covered}

    # ---------------------------------------------------------------- stateless core (vectorised)
    def core(self, M, K=None):
        """(T,189), (T,9) -> {param: (T,) float64} for every covered non-extra param (no clipping yet)."""
        c = self.cfg
        M = np.asarray(M, np.float64)
        if M.ndim == 1:
            M = M[None]
        T = len(M)
        M = np.where(np.isfinite(M), M, self.neutral[None])
        if K is not None:
            K = np.asarray(K).reshape(T, spec.D_MASK).astype(bool)
            # invalid groups -> the neutral pose for those columns (rest, eyes open, looking at the camera)
            cm = spec.col_mask(K).astype(bool)
            M = np.where(cm, M, self.neutral[None])
        o = {}
        soft = {}                                             # param -> (L, k): soft-clipped together at the end
        cov = set(self.covered)
        # ---------------- torso + head + gaze (FK with the root)
        rots = sk.local_rots(M)
        fk = sk.upper_fk(M, root=True, rots=rots)
        Rc = fk["spine3"][0]
        Rd = Rc @ self.R_chest0.T
        y, p, r = sk.euler_yxz(Rd)
        b = c["body"]
        rpu = b["rad_per_unit"]
        o["body_yaw"] = b["gain"][0] * y / rpu[0]
        o["body_pitch"] = b["gain"][1] * (p - b.get("pitch0", 0.0)) / rpu[1]
        o["body_lean"] = b["gain"][2] * (-r) / rpu[2]
        o["body_bounce"] = b["bounce_gain"] * M[:, 4]
        soft.update(body_yaw=(1.0, 0.7), body_pitch=(1.0, 0.7), body_lean=(1.0, 0.7), body_bounce=(1.0, 0.6))
        Rh_rel = np.swapaxes(Rc, -1, -2) @ fk["head"][0]
        y, p, r = sk.euler_yxz(Rh_rel @ self.R_head0_rel.T)
        h = c["head"]
        o["head_yaw"] = h["gain"][0] * y
        o["head_pitch"] = h["gain"][1] * (p - h.get("pitch0", 0.0))
        o["head_roll"] = h["gain"][2] * (-r)
        soft.update(head_yaw=(h["limit"][0], 0.6), head_pitch=(h["limit"][1], 0.6), head_roll=(h["limit"][2], 0.6))
        # gaze: world = head world + eye-in-head (canon gaze yaw + = toward the person's left = screen right)
        yw, pw, _ = sk.euler_yxz(fk["head"][0] @ self.R_head0_world.T)
        g = c["gaze"]
        gx = np.degrees(g["head_comp"] * yw + M[:, 187] - g.get("yaw0", 0.0))
        gy = np.degrees(-g["head_comp"] * pw + M[:, 188] - g.get("pitch0", 0.0))
        o["gaze_x"] = gx / g["deg_per_unit"][0]
        o["gaze_y"] = gy / g["deg_per_unit"][1]
        soft.update(gaze_x=(1.0, 0.75), gaze_y=(1.0, 0.75))
        # ---------------- face (rig _r = canon "...Left")
        f = c["face"]
        F = M[:, 135:187]
        col = lambda n: F[:, B[n]]  # noqa: E731
        inner = col("browInnerUp") - REST_BS["browInnerUp"]
        for rs_, hs in (("r", "Left"), ("l", "Right")):
            raw = (0.5 * (col("browOuterUp" + hs) - REST_BS["browOuterUp" + hs]) + 0.5 * inner
                   - (col("browDown" + hs) - REST_BS["browDown" + hs]))
            o[f"brow_{rs_}"] = np.where(raw >= 0, f["brow_up_gain"] * raw, f["brow_down_gain"] * raw)
        dmean = 0.5 * (col("browDownLeft") + col("browDownRight"))
        o["brow_knit"] = _c01(f["knit_gain"] * (dmean - f["knit0"]))
        for rs_, hs in (("r", "Left"), ("l", "Right")):
            o[f"blink_{rs_}"] = _smoothstep((col("eyeBlink" + hs) - f["blink_lo"]) / (f["blink_hi"] - f["blink_lo"]))
        wide = 0.5 * (col("eyeWideLeft") + col("eyeWideRight"))
        o["eye_wide"] = _c01(f["wide_gain"] * (wide - f["wide0"]) + f["wide_from_brow"] * np.maximum(inner - 0.05, 0))
        sq = 0.5 * (col("eyeSquintLeft") + col("eyeSquintRight"))
        chs = 0.5 * (col("cheekSquintLeft") + col("cheekSquintRight"))
        o["squint"] = _c01(f["squint_gain"] * (sq - f["squint0"]) + f["cheek_squint_gain"] * chs)
        sm = 0.5 * (col("mouthSmileLeft") + col("mouthSmileRight"))
        o["smile"] = _c01(f["smile_gain"] * (sm - f["smile0"]))
        o["mouth_open"] = _c01(f["jaw_gain"] * (col("jawOpen") - f["jaw0"]))
        sn = 0.5 * (col("noseSneerLeft") + col("noseSneerRight"))
        o["nostril"] = _c01(sn / f["sneer_rng"])
        o["nose_wiggle"] = f["nose_wiggle_per_nostril"] * o["nostril"]
        # ---------------- arms / hands
        arms = sk.chest_frame_arm(M, rots=rots)
        hm = c["hands"]
        for cs, rs_ in (("l", "r"), ("r", "l")):           # canon anatomical side -> rig screen side
            A = arms[cs]
            out_sign = 1.0 if cs == "l" else -1.0
            d = (A["wrist"] - self.wrist_ref[cs]) / sk.ARM
            dx = out_sign * d[:, 0]
            dy, dz = d[:, 1], d[:, 2]
            k = hm["live"]
            hx = hm["gx_out"] * np.maximum(dx, 0) + hm["gx_in"] * np.minimum(dx, 0) + hm["cxy"] * np.maximum(dy, 0)
            hy = hm["gy_up"] * np.maximum(dy, 0) + hm["gy_dn"] * np.minimum(dy, 0)
            hz = hm["gz"] * dz + hm["czy"] * np.maximum(dy, 0)
            lim = hm["limit"]
            ro = hm.get("rest_offset", (0.0, 0.0, 0.0))
            hx, hy, hz = hx + ro[0] / k, hy + ro[1] / k, hz + ro[2] / k
            for ai, hv in enumerate((hx, hy, hz)):
                nm = f"hand_{rs_}_{'xyz'[ai]}"
                o[nm] = k * hv
                soft[nm] = (max(abs(lim[ai][0]), lim[ai][1]), 0.75, lim[ai][0], lim[ai][1])
            # palm roll about the forearm, the shader's convention: 0 = palm to the midline, +1 (x1.45 rad) to camera
            fdir = A["wrist"] - A["elbow"]
            fdir /= np.maximum(np.linalg.norm(fdir, axis=1, keepdims=True), 1e-9)
            n = A["R_wrist"] @ np.array([0.0, -1.0, 0.0])
            mid = np.array([-out_sign, 0.0, 0.0])
            n0 = mid[None] - fdir * (fdir @ mid)[:, None]
            nn = n - fdir * np.sum(n * fdir, 1)[:, None]
            ang = np.arctan2(np.sum(_cross(n0, nn) * fdir, 1), np.sum(n0 * nn, 1))
            s_scr = 1.0 if rs_ == "r" else -1.0
            o[f"hand_{rs_}_turn"] = np.minimum(np.maximum(hm["turn_gain"] * ang / (1.45 * s_scr), -1.0), 1.0)
            # shrug from the collar lifting the shoulder joint
            o[f"shoulder_{rs_}"] = _c01((A["shoulder"][:, 1] - hm["shrug0"]) / hm["shrug_m"])
            # fingers: MANO flexion (z) per finger, relative to the population
            sl = 45 if cs == "l" else 90
            sg = -1.0 if cs == "l" else 1.0
            curl = sg * M[:, _CURL_IDX + sl].sum(-1)                        # (T,5) flexion per finger
            fg = c["fingers"]
            others = curl[:, 1:4].mean(1)
            avg4 = curl[:, 0:4].mean(1)
            o[f"hand_{rs_}_open"] = _c01(_pwl(avg4, fg["open_xs"], fg["open_ys"]))
            o[f"point_{rs_}"] = _c01((others - curl[:, 0] - fg["point0"]) / fg["point_rng"]) * _c01((others - 1.6) / 0.6)
            o[f"thumb_{rs_}"] = _c01((avg4 - fg["thumb_fist0"]) / 0.6) * _c01((fg["thumb_straight0"] - curl[:, 4]) / 0.4)
        # one vectorised soft knee (+ hard limits) for every soft-clipped param
        names = [p for p in soft if p in cov]
        if names:
            X = np.stack([o[p] for p in names], 0)
            Lk = np.array([soft[p][:2] for p in names])
            L_, kn = Lk[:, 0:1], Lk[:, 0:1] * Lk[:, 1:2]
            r_ = L_ - kn
            a = np.abs(X)
            Y = np.sign(X) * (np.minimum(a, kn) + r_ * np.tanh(np.maximum(a - kn, 0.0) / r_))
            for i, p in enumerate(names):
                lim = soft[p]
                o[p] = Y[i] if len(lim) == 2 else np.minimum(np.maximum(Y[i], lim[2]), lim[3])
        return {k: v for k, v in o.items() if k in cov}

    # ---------------------------------------------------------------- causal extras (one frame)
    def _extras(self, r):
        c = self.cfg
        out = {}
        dt = DT
        t = self.t * dt
        ex = c.get("extras", {})
        # arousal: smile + head / hand motion energy, slow
        hv = 0.0
        hd = np.array([r.get("head_yaw", 0.0), r.get("head_pitch", 0.0), r.get("head_roll", 0.0)])
        if self.prev_head is not None:
            hv = float(np.linalg.norm(hd - self.prev_head)) / dt
        self.prev_head = hd
        self.energy += (min(hv, 3.0) - self.energy) * (1 - math.exp(-dt / 0.6))
        target = 0.25 + 0.5 * r.get("smile", 0.0) + 0.35 * min(self.energy, 1.0) + 0.2 * r.get("eye_wide", 0.0)
        self.arousal += (min(target, 1.0) - self.arousal) * (1 - math.exp(-dt / 1.2))
        if "breath" in self.covered:
            self.breath_ph += 2 * math.pi * self.breath_hz * dt
            if self.breath_ph > 2 * math.pi:
                self.breath_ph -= 2 * math.pi
                self.breath_hz = float(np.clip(0.24 + 0.05 * self.rs.standard_normal(), 0.16, 0.33))
            out["breath"] = ex.get("breath_amp", 0.6) * math.sin(self.breath_ph)
        if "ear_l" in self.covered:
            bl_, br_ = r.get("brow_l", 0.0), r.get("brow_r", 0.0)
            bavg = 0.5 * (bl_ + br_)
            up, dn = max(bavg, 0.0), max(-bavg, 0.0)
            base = (ex.get("ear_base", 0.10) + 0.45 * up + 0.3 * r.get("eye_wide", 0.0) + 0.3 * r.get("smile", 0.0)
                    - 0.45 * r.get("brow_knit", 0.0) - 0.3 * dn + 0.15 * (self.arousal - 0.4))
            pitch = r.get("head_pitch", 0.0)
            pv = 0.0 if self.prev_pitch is None else (pitch - self.prev_pitch) / dt
            acc = float(np.clip((pv - self.prev_pv) / dt, -60, 60))
            self.prev_pitch, self.prev_pv = pitch, pv
            w, z = 2 * math.pi * 3.1, 0.32
            self.ear_v += (-1.0 * acc - w * w * self.ear_x - 2 * z * w * self.ear_v) * dt
            self.ear_x += self.ear_v * dt
            sg = ex.get("ear_spring", 0.12)
            out["ear_l"] = base + 0.3 * (bl_ - bavg) + sg * self.ear_x[0]
            out["ear_r"] = base + 0.3 * (br_ - bavg) + sg * self.ear_x[0]
            if "ear_l_turn" in self.covered:
                out["ear_l_turn"] = out["ear_r_turn"] = -0.25 * up + 0.2 * r.get("brow_knit", 0.0)
            if "ear_flick_l" in self.covered:
                fl = [0.0, 0.0]
                if t >= self.next_flick:
                    self.flick_t, self.flick_side = t, int(self.rs.integers(0, 2))
                    self.next_flick = t + float(self.rs.uniform(5, 14))
                u = (t - self.flick_t) / 0.18
                if 0 <= u <= 1:
                    fl[self.flick_side] = math.sin(math.pi * u)
                out["ear_flick_l"], out["ear_flick_r"] = fl
        if "tail_wag" in self.covered:
            a = self.arousal
            hz = 0.5 + 0.9 * a
            self.tail_ph += 2 * math.pi * hz * dt
            out["tail_wag"] = (0.15 + 0.55 * a) * math.sin(self.tail_ph) + ex.get("tail_wag0", 0.0)
            out["tail_raise"] = 0.15 + 0.5 * a
            out["tail_curl"] = 0.15 + 0.3 * r.get("smile", 0.0)
        self.t += 1
        return out

    def _clip(self, o):
        return {k: float(min(max(v, RANGES[k][0]), RANGES[k][1])) if k in RANGES else float(v) for k, v in o.items()}

    def _baseline_step(self, o):
        bl = self.cfg.get("baseline") or {}
        if not self.base_ema:
            return o
        a = 1.0 - math.exp(-DT / max(bl.get("tau_s", 6.0), 1e-3))
        for p, w in bl["params"].items():
            if p in o and p in self.base_ema:
                v = o[p]
                self.base_ema[p] += (v - self.base_ema[p]) * a
                o[p] = v - w * self.base_ema[p]
        return o

    # ---------------------------------------------------------------- public
    def step(self, motion, mask=None):
        """one causal frame -> {param: float} (covered params, clipped)."""
        o = {k: float(v[0]) for k, v in self.core(np.asarray(motion)[None] if np.ndim(motion) == 1 else motion,
                                                 None if mask is None else np.asarray(mask)[None]).items()}
        o = self._baseline_step(o)
        o.update(self._extras(o))
        return self._clip(o)

    def batch(self, M, K=None):
        M = np.asarray(M, np.float64)
        core = self.core(M, K)
        T = len(M)
        bl = self.cfg.get("baseline") or {}
        if self.base_ema:
            from scipy.signal import lfilter
            a = 1.0 - math.exp(-DT / max(bl.get("tau_s", 6.0), 1e-3))
            for p, w in bl["params"].items():
                if p in core and p in self.base_ema:
                    v = core[p]
                    ema, zf = lfilter([a], [1.0, -(1.0 - a)], v, zi=[(1.0 - a) * self.base_ema[p]])
                    self.base_ema[p] = float(ema[-1])
                    core[p] = v - w * ema
        ext = {p: np.zeros(T) for p in self.extras}
        for i in range(T):
            e = self._extras({k: float(v[i]) for k, v in core.items()})
            for k, v in e.items():
                ext[k][i] = v
        out = {**core, **ext}
        return {k: np.clip(v, *RANGES[k]).astype(np.float32) if k in RANGES else v.astype(np.float32)
                for k, v in out.items()}

    # ---------------------------------------------------------------- inverse (rig -> canon), approximate
    def inverse(self, rig, base=None):
        """{rig param: value} -> ((189,) canon frame (base / neutral where not defined), [canon groups defined]).
        Cheap approximations, for the planner's set_override(values=...): head, torso, arms (two-bone IK on the
        canonical skeleton), fingers, gaze, face_upper/mouth (blink, brows, smile)."""
        c = self.cfg
        m = np.array(self.neutral if base is None else base, np.float64, copy=True)
        groups = set()
        g = {k: float(v) for k, v in rig.items() if v is not None and np.isfinite(v)}
        # torso
        if any(k in g for k in ("body_yaw", "body_pitch", "body_lean")):
            b = c["body"]; rpu = b["rad_per_unit"]
            y = _soft_inv(np.clip(g.get("body_yaw", 0.0), -0.999, 0.999), 1.0, 0.7) * rpu[0] / b["gain"][0]
            p = _soft_inv(np.clip(g.get("body_pitch", 0.0), -0.999, 0.999), 1.0, 0.7) * rpu[1] / b["gain"][1] + b.get("pitch0", 0.0)
            r = -_soft_inv(np.clip(g.get("body_lean", 0.0), -0.999, 0.999), 1.0, 0.7) * rpu[2] / b["gain"][2]
            Rd = sk.mat_yxz(y, p, r)
            # Rchest = Rroot S = Rd Rchest0 with the spine at neutral -> Rroot = Rd
            m[0:3] = sk.mat_to_aa(Rd[None])[0]
            for j in ("spine1", "spine2", "spine3"):
                m[spec.jcols(j)] = self.neutral[spec.jcols(j)]
            if "body_bounce" in g:
                m[4] = g["body_bounce"] / max(b["bounce_gain"], 1e-6)
            groups.add("torso")
        # head
        if any(k in g for k in ("head_yaw", "head_pitch", "head_roll")):
            h = c["head"]
            y = _soft_inv(np.clip(g.get("head_yaw", 0.0), -h["limit"][0], h["limit"][0]), h["limit"][0]) / h["gain"][0]
            p = _soft_inv(np.clip(g.get("head_pitch", 0.0), -h["limit"][1], h["limit"][1]), h["limit"][1]) / h["gain"][1] + h.get("pitch0", 0.0)
            r = -_soft_inv(np.clip(g.get("head_roll", 0.0), -h["limit"][2], h["limit"][2]), h["limit"][2]) / h["gain"][2]
            Rrel = sk.mat_yxz(y, p, r) @ self.R_head0_rel
            N0 = sk.aa_to_mat(self.neutral[spec.jcols("neck")][None])[0]
            m[spec.jcols("neck")] = self.neutral[spec.jcols("neck")]
            m[spec.jcols("head")] = sk.mat_to_aa((N0.T @ Rrel)[None])[0]
            groups.add("head")
        # arms
        hm = c["hands"]
        for cs, rs_ in (("l", "r"), ("r", "l")):
            ks = [f"hand_{rs_}_{a}" for a in "xyz"]
            if not any(k in g for k in ks):
                continue
            k = hm["live"]
            ro = hm.get("rest_offset", (0.0, 0.0, 0.0))
            g = dict(g)
            for ai, kk_ in enumerate(ks):
                g[kk_] = g.get(kk_, 0.0) - ro[ai]
            hy = g.get(ks[1], 0.0) / k
            dy = hy / hm["gy_up"] if hy >= 0 else hy / hm["gy_dn"]
            hx = g.get(ks[0], 0.0) / k - hm["cxy"] * max(dy, 0)
            dx = hx / hm["gx_out"] if hx >= 0 else hx / hm["gx_in"]
            dz = (g.get(ks[2], 0.0) / k - hm["czy"] * max(dy, 0)) / hm["gz"]
            out_sign = 1.0 if cs == "l" else -1.0
            target = self.wrist_ref[cs] + sk.ARM * np.array([out_sign * dx, dy, dz])
            # wrist target is relative to the REST shoulder in the chest frame; the IK starts at the shoulder joint
            # (collar neutral moves it a little: correct for that)
            Rc = sk.aa_to_mat(self.neutral[spec.jcols("collar_" + cs)][None])[0]
            sh = Rc @ sk.OFFSET["shoulder_" + cs] - sk.OFFSET["shoulder_" + cs]
            sa, ea, _ = sk.two_bone_ik((target - sh)[None], cs)
            # the IK's shoulder rotation is in the chest frame; canon's shoulder is local to the collar
            Rs = Rc.T @ sk.aa_to_mat(sa)[0]
            m[spec.jcols("collar_" + cs)] = self.neutral[spec.jcols("collar_" + cs)]
            m[spec.jcols("shoulder_" + cs)] = sk.mat_to_aa(Rs[None])[0]
            m[spec.jcols("elbow_" + cs)] = ea[0]
            m[spec.jcols("wrist_" + cs)] = self.neutral[spec.jcols("wrist_" + cs)]
            groups.add("arm_" + cs)
        # fingers
        fg = c["fingers"]
        for cs, rs_ in (("l", "r"), ("r", "l")):
            if f"hand_{rs_}_open" not in g and f"point_{rs_}" not in g:
                continue
            op = float(np.clip(g.get(f"hand_{rs_}_open", 0.2), 0, 1))
            ys, xs = np.asarray(fg["open_ys"]), np.asarray(fg["open_xs"])
            order = np.argsort(ys)
            avg = float(np.interp(op, ys[order], xs[order]))
            pt = float(np.clip(g.get(f"point_{rs_}", 0.0), 0, 1))
            sl = 45 if cs == "l" else 90
            sg = -1.0 if cs == "l" else 1.0
            med = np.asarray(self.neutral[sl:sl + 45]).reshape(5, 3, 3)
            P = pop_stats()["finger_curl_pct"]
            for kk, fn in enumerate(["index", "middle", "pinky", "ring"]):
                want = avg + (-(1.2 * pt) if kk == 0 else 0.8 * pt)
                cur = sg * med[kk, :, 2].sum()
                med[kk, :, 2] += sg * (want - cur) / 3.0
            m[sl:sl + 45] = med.ravel()
            groups.add("fingers_" + cs)
        # face (approximate; rarely owned): blinks, smile, brows up
        return m, sorted(groups)


def rig_names_for(character):
    return list(load_map(character)["covered"])
