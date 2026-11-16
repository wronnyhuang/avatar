"""Rig-native renderer: the interpreter's timeline (Builder segs + oscs) -> the character's rig dials, vectorised, at
the output rate.  Replaces interp.Renderer + canon FK + retarget for the live path (same segment semantics: each
dial is its own track, a segment eases from the value the dial had when it started to its target; additive
oscillations on top), but never builds canon frames: heads/torso/root/gaze from degrees, arms through arms.py (each
character's own hand map), face from the blendshape targets with a direct map, fingers from a shape table.

CAUSAL: a frame at time t depends only on segments with t0 <= t, so re-rendering a longer timeline reproduces every
earlier frame exactly (the streaming time-commitment rule relies on this; tests check it).

render(B, character, n_frames, hz) -> {rig param: (n_frames,) float64} for EVERY dial the program can touch.
"""
from __future__ import annotations

import math

import numpy as np

from . import arms as _arms
from . import interp, kin

R2D = 180.0 / math.pi
D2R = math.pi / 180.0
CHAR = {
    # head gain (rig radians per real radian), torso rad per rig unit (yaw, pitch, lean), gaze tangent per unit (x, y)
    # head_gain yaw/roll, pitch_gain head pitch, torso_gain body pitch/lean (beyond the rig's range the rest tips the
    # whole body on characters with a root), osc_* = oscillation gains (cm / deg the model writes are for a human; the
    # muse's short arms and heavy head need more to read), hand_osc_min = smallest readable hand wiggle (hand units)
    "muse": dict(head_gain=1.3, pitch_gain=1.7, torso_gain=1.5, rpu=(0.40, 0.28, 0.20), gaze=(0.30, 0.24), root=True,
                 ears=False, osc_head=1.7, osc_hand=1.6, hand_osc_min=0.09, overflow=True),
    "juno": dict(head_gain=1.0, pitch_gain=1.0, torso_gain=1.0, rpu=(0.40, 0.28, 0.20), gaze=(0.30, 0.24), root=True,
                 ears=False, osc_head=1.0, osc_hand=1.0, hand_osc_min=0.05, overflow=True),
    "fox": dict(head_gain=1.0, pitch_gain=1.2, torso_gain=1.0, rpu=(0.35, 0.15, 0.12), gaze=(0.30, 0.24), root=False,
                ears=True, osc_head=1.3, osc_hand=1.2, hand_osc_min=0.07, overflow=False),
}
H_CM = 170.0
EYES = None
SHAPES = {   # hand shape -> (open, point, thumb, turn add)
    "relaxed": (0.2, 0, 0, 0), "open": (1, 0, 0, 0), "flat": (1, 0, 0, 0), "spread": (1, 0, 0, 0),
    "fist": (0, 0, 0, 0), "point": (0, 1, 0, 0), "thumbs_up": (0, 0, 1, 0), "thumbs_down": (0, 0, 1, 0),
    "peace": (0.3, 0.7, 0, 0), "ok": (0.5, 0, 0, 0), "cup": (0.6, 0, 0, 0), "claw": (0.45, 0, 0, 0),
    "pinch": (0.3, 0, 0, 0), "rock": (0.2, 0.6, 0.3, 0), "call_me": (0, 0, 1, 0),
}
PALM_TURN = {"camera": 1.0, "up": 1.0, "forward": 0.7, "front": 0.7, "down": -0.8, "back": -0.8, "in": 0.0,
             "out": -0.5, "body": 0.0, "face": 0.0, "left": 0.0, "right": 0.0}


def ease_vec(sg, u, nat):
    u = np.clip(u, 0.0, 1.0)
    e = sg.ease
    if "anticip" in sg.kind and nat["anticipation"] and e in ("settle", "min_jerk"):
        a = -0.05 * np.sin(np.pi * np.minimum(u, 0.2) / 0.2) ** 2
        b = kin.ease(e, np.clip((u - 0.2) / 0.8, 0, 1))
        return np.where(u < 0.2, a, b)
    if e == "settle" and not nat["overshoot"]:
        e = "min_jerk"
    if not nat["min_jerk"] and e in ("min_jerk", "settle"):
        e = "linear"
    return kin.ease(e, u)


def _starts(segs, F):
    return [int(np.searchsorted(F, sg.t0 - 1e-9, side="left")) for sg in segs]


def chain(segs, F, n0, target, nat, record=None, bump=None):
    """one scalar (or vector) track. target(sg, a, b, st) -> value(s) for frames a..b."""
    T = len(F)
    n0 = np.asarray(n0, np.float64)
    vals = np.tile(n0, (T,) + (1,) * n0.ndim) if n0.ndim else np.full(T, float(n0))
    if not segs:
        return vals
    segs = sorted(segs, key=lambda sg: (round(sg.t0, 4), sg.prio))
    st_idx = _starts(segs, F)
    for k, sg in enumerate(segs):
        a = st_idx[k]
        b = st_idx[k + 1] if k + 1 < len(segs) else T
        if a >= b:
            continue
        st = vals[a - 1].copy() if a > 0 else n0.copy()
        u = (F[a:b] - sg.t0) / max(sg.t1 - sg.t0, 1e-6)
        w = ease_vec(sg, u, nat)
        tg = target(sg, a, b, st)
        if record is not None:
            g = np.asarray(tg, np.float64)
            record[id(sg)] = (st.copy(), g[0].copy() if g.ndim > n0.ndim else g.copy())
        if n0.ndim:
            w = w[:, None]
        vals[a:b] = st + (tg - st) * w
        if bump is not None:
            vals[a:b] += bump(sg, u, st, tg)
    return vals


def osc_sum(oscs, chan, param, F):
    v = np.zeros(len(F))
    for o in oscs:
        if o.chan != chan or o.param != param:
            continue
        m = (F >= o.t0) & (F <= o.t1)
        if not m.any():
            continue
        u = (F[m] - o.t0) / max(o.t1 - o.t0, 1e-6)
        if o.src == "clap":
            env = np.ones_like(u)
        else:
            env = np.clip(np.minimum(np.minimum(1.0, u / 0.12), (1 - u) / 0.12), 0, 1)
            env = env * env * (3 - 2 * env)
        v[m] += (o.amp * np.sin(2 * np.pi * o.cycles * u + o.phase) + o.bias) * env
    return v


def _keyed(B, chan, key):
    out = []
    for sg in B.segs:
        if sg.chan != chan:
            continue
        tg = sg.tgt
        if "neutral" in tg or key in tg or (key == "yaw" and "yaw_add" in tg) or (key == "pitch" and "pitch_add" in tg):
            out.append(sg)
    return out


def _scalar_target(key, neutral):
    def f(sg, a, b, st):
        tg = sg.tgt
        if "neutral" in tg:
            if key == "yaw" and sg.chan == "root":
                return round(float(st) / 360.0) * 360.0
            return neutral
        if key == "yaw" and "yaw_add" in tg:
            return float(st) + tg["yaw_add"]
        if key == "pitch" and "pitch_add" in tg:
            return float(st) + tg["pitch_add"]
        x = tg[key]
        if isinstance(x, tuple) and x[0] == "add":
            return float(st) + x[1]
        return float(x)
    return f


def _eyes():
    global EYES
    if EYES is None:
        EYES = _arms.canon_point("eyes")
    return EYES


def _look_angles(tgt, body_yaw_deg, t_live):
    """-> (yaw_deg, pitch_deg) of the target from the eyes, in the body's upright frame (body_yaw in degrees)."""
    if isinstance(tgt, tuple) and tgt[0] == "xyz":
        p = np.array(tgt[1], np.float64)
    else:
        p = _arms.canon_point(str(tgt))
    d = p - _eyes()
    yaw = math.degrees(math.atan2(d[0], d[2])) - body_yaw_deg
    yaw = (yaw + 180.0) % 360.0 - 180.0
    pitch = math.degrees(math.atan2(d[1], math.hypot(d[0], d[2])))
    return yaw, pitch


def render(B, character, T, hz):
    c = CHAR.get(character, CHAR["muse"])
    nat = B.nat
    F = np.arange(T) / float(hz)
    oscs = B.oscs
    out = {}
    # ---------------------------------------------------------------- root (deg, cm)
    R = {}
    for k in ("yaw", "pitch", "roll", "x", "y", "z"):
        R[k] = chain(_keyed(B, "root", k), F, 0.0, _scalar_target(k, 0.0), nat) + osc_sum(oscs, "root", k, F)
    rp = [sg for sg in B.segs if sg.chan == "rootp"]
    if rp:
        R["pitch"] = R["pitch"] + chain(rp, F, 0.0, _scalar_target("pitch", 0.0), nat)
    # ---------------------------------------------------------------- torso (deg)
    Tq = {}
    for k in ("lean_fwd", "lean_side", "twist"):
        Tq[k] = chain(_keyed(B, "torso", k), F, 0.0, _scalar_target(k, 0.0), nat) + osc_sum(oscs, "torso", k, F)
    Tq["twist"] = np.clip(Tq["twist"], -60.0, 60.0)       # a torso twist stops at 60 deg (turn() is for more)
    body_yaw = R["yaw"] + Tq["twist"]                     # degrees, the chest's heading (before any recruit)
    hg, pg = c["head_gain"], c["pitch_gain"]
    rpu = c["rpu"]
    head_lim = 0.95 * 0.6 / hg * R2D                     # what the rig head can turn, in real degrees
    pitch_lim = 0.95 * 0.45 / pg * R2D
    body_max = 0.9 * rpu[0] * R2D                         # the torso twist the rig can show
    # ---------------------------------------------------------------- head (deg): ypr / look / neutral
    hk = ("yaw", "pitch", "roll")
    look_cache = {}

    def look(sg, a, b):
        key = (id(sg), a)
        r = look_cache.get(key)
        if r is None:
            # resolved against the body heading at the segment's END (after earlier root / torso turns)
            i1 = min(b - 1, max(a, int(math.ceil(sg.t1 * hz - 1e-9)) - 1))
            y, p = _look_angles(sg.tgt["look"], float(body_yaw[i1]) if len(body_yaw) else 0.0, sg.t1)
            want = y * 0.75 if abs(y) > 8 else y * 0.3
            hy = float(np.clip(want, -head_lim, head_lim))
            # beyond the head's range the torso turns too (a share of the rest), the eyes take what is left
            by = float(np.clip(0.5 * (y - hy), -body_max, body_max)) if abs(want) > head_lim and \
                not sg.tgt.get("eyes_only") else 0.0
            # never look past ~profile: head + torso recruit + the body's own turn stay within 105 deg of the camera
            # (a target on the back wall is seen over the shoulder with the eyes, not with the back of the head)
            h0 = float(body_yaw[i1]) if len(body_yaw) else 0.0
            world = h0 + hy + by
            if abs(world) > 105.0 and abs(hy + by) > 1e-6:
                k_ = max(0.0, (105.0 * np.sign(world) - h0) / (hy + by))
                k_ = min(1.0, k_)
                hy, by = hy * k_, by * k_
            hp = float(np.clip(p * 0.75 if abs(p) > 6 else p * 0.3, -pitch_lim, pitch_lim))
            r = look_cache[key] = (y, p, hy, hp, by)
        return r

    H = {}
    for j, k in enumerate(hk):
        segs = [sg for sg in B.segs if sg.chan == "head" and ("neutral" in sg.tgt or ("look" in sg.tgt and j < 2) or
                                                              ("ypr" in sg.tgt and j < len(sg.tgt["ypr"])))]

        def tgt(sg, a, b, st, j=j):
            if "neutral" in sg.tgt:
                return 0.0
            if "look" in sg.tgt:
                return look(sg, a, b)[2 + j]
            return float(sg.tgt["ypr"][j])
        H[k] = chain(segs, F, 0.0, tgt, nat) + c["osc_head"] * osc_sum(oscs, "head", k, F)
    # the torso recruit of looks past the head's range (its own track: eases in / out with the look)
    hsegs = [sg for sg in B.segs if sg.chan == "head"]
    look_rec = chain(hsegs, F, 0.0, lambda sg, a, b, st: look(sg, a, b)[4] if "look" in sg.tgt else 0.0, nat) \
        if any("look" in sg.tgt for sg in hsegs) else np.zeros(T)
    # ---------------------------------------------------------------- gaze (eye in head, deg)
    G = {}
    for j, k in enumerate(("yaw", "pitch")):
        segs = [sg for sg in B.segs if sg.chan == "gaze"]

        def gt(sg, a, b, st, j=j, k=k):
            if "neutral" in sg.tgt:
                return 0.0
            if "look" in sg.tgt:
                y, p, hy, hp, by = look(sg, a, b)
                if sg.tgt.get("eyes_only"):
                    i1 = min(b - 1, max(a, int(math.ceil(sg.t1 * hz - 1e-9)) - 1))
                    hy, hp, by = float(H["yaw"][i1]), float(H["pitch"][i1]), 0.0
                lim = kin.JOINT_LIMITS["gaze_yaw" if j == 0 else "gaze_pitch"]
                return float(np.clip((y - hy - by) if j == 0 else (p - hp), *lim))
            return float(sg.tgt["ypr"][j]) if j < len(sg.tgt["ypr"]) else 0.0
        G[k] = chain(segs, F, 0.0, gt, nat) + osc_sum(oscs, "gaze", k, F)
    # ---------------------------------------------------------------- to rig: head / torso / root / gaze
    out["head_yaw"] = hg * H["yaw"] * D2R
    out["head_pitch"] = -pg * H["pitch"] * D2R
    out["head_roll"] = hg * H["roll"] * D2R
    tg_ = c["torso_gain"]
    out["body_yaw"] = (Tq["twist"] + look_rec) * D2R / rpu[0]
    bp = tg_ * Tq["lean_fwd"] * D2R / rpu[1]
    bl = tg_ * Tq["lean_side"] * D2R / rpu[2]
    over_p = over_l = np.zeros(T)
    if c.get("overflow"):
        # past the rig's torso range the whole body tips (about its centre of mass) by the remainder
        over_p = np.sign(bp) * np.maximum(np.abs(bp) - 0.85, 0.0) * rpu[1] * R2D
        over_l = np.sign(bl) * np.maximum(np.abs(bl) - 0.85, 0.0) * rpu[2] * R2D
        bp, bl = np.clip(bp, -0.85, 0.85), np.clip(bl, -0.85, 0.85)
    out["body_pitch"], out["body_lean"] = bp, bl
    over_y = np.zeros(T)
    if c.get("overflow"):          # a twist past the torso's range turns the whole body for the rest
        by_ = out["body_yaw"]
        over_y = np.sign(by_) * np.maximum(np.abs(by_) - 0.85, 0.0) * rpu[0] * R2D
        out["body_yaw"] = np.clip(by_, -0.85, 0.85)
    if c["root"]:
        out["root_yaw"] = (R["yaw"] + over_y) / 360.0
        out["root_pitch"] = (R["pitch"] + 0.6 * over_p) / 360.0
        out["root_roll"] = (R["roll"] + 0.6 * over_l) / 360.0
        out["root_x"] = R["x"] / H_CM
        out["root_z"] = R["z"] / H_CM
        out["root_y"] = np.maximum(R["y"], 0.0) / H_CM
        out["squash"] = np.minimum(np.maximum(-R["y"], 0.0) / 70.0, 0.4)
    else:   # seated: whole-body moves degrade to the torso dials, using their full range, out and back
        sy = np.sin(R["yaw"] * D2R / 2.0)
        sp = np.sin(R["pitch"] * D2R / 2.0)
        out["body_yaw"] = out["body_yaw"] + 1.15 * sy
        out["body_pitch"] = out["body_pitch"] + 1.15 * sp
        out["body_lean"] = out["body_lean"] + 1.0 * np.sin(R["roll"] * D2R / 2.0) + R["x"] / 20.0
        out["body_bounce"] = R["y"] / 14.0
        # the head and ears follow the degraded whole-body move so it reads as one
        out["head_yaw"] = out["head_yaw"] + 0.45 * sy
        out["head_pitch"] = out["head_pitch"] - 0.35 * sp - 0.012 * np.minimum(R["y"], 0.0)
    # world eye direction (deg) = body heading + head + eye in head; rig gaze is world-referenced (0 = camera)
    wy = body_yaw + look_rec + H["yaw"] + G["yaw"]
    wp = H["pitch"] + G["pitch"] - Tq["lean_fwd"] * 0.3
    out["gaze_x"] = wy / 28.0          # the renderer's eyes: ~17 deg off the lens at |gaze| 1 (+ the head carries)
    out["gaze_y"] = wp / 22.0
    # for the arms: world targets re-aimed in the body as the RIG turns it (root turn + the torso twist it can show)
    heading = ((R["yaw"] + over_y) if c["root"] else 0.0) + np.clip(out["body_yaw"], -1.0, 1.0) * rpu[0] * R2D
    # ---------------------------------------------------------------- shoulders
    for cs in "lr":
        rs = "r" if cs == "l" else "l"
        segs = [sg for sg in B.segs if sg.chan == "shrug_" + cs]
        v = chain(segs, F, 0.0, lambda sg, a, b, st: 0.0 if "neutral" in sg.tgt else float(sg.tgt.get("amount", 0.0)),
                  nat) + osc_sum(oscs, "shrug_" + cs, "amount", F)
        out["shoulder_" + rs] = v
    # ---------------------------------------------------------------- fingers
    for cs in "lr":
        rs = "r" if cs == "l" else "l"
        segs = [sg for sg in B.segs if sg.chan == "fing_" + cs]

        def ft(sg, a, b, st):
            if "neutral" in sg.tgt:
                return np.array([0.2, 0.0, 0.0])
            sh = SHAPES.get(str(sg.tgt.get("shape")), SHAPES["relaxed"])
            amt = float(np.clip(sg.tgt.get("amount", 1.0), 0, 1))
            base = np.array([0.2, 0.0, 0.0])
            return base + (np.array(sh[:3], float) - base) * amt
        V = chain(segs, F, np.array([0.2, 0.0, 0.0]), ft, nat)
        out[f"hand_{rs}_open"], out[f"point_{rs}"], out[f"thumb_{rs}"] = V[:, 0], V[:, 1], V[:, 2]
    # ---------------------------------------------------------------- arms (rig space)
    A = _arms.for_character(character)
    cache = B.__dict__.setdefault("_goal_cache", {}) if hasattr(B, "__dict__") else {}
    if len(cache) > 4000:
        cache.clear()
    head = (out["head_yaw"], out["head_pitch"], out["head_roll"])
    hpm = A.h_per_m()
    for cs in "lr":
        rs = "r" if cs == "l" else "l"
        sgn_out = 1.0 if cs == "l" else -1.0          # canon +x is outward for the avatar's left hand
        segs = [sg for sg in B.segs if sg.chan == "arm_" + cs]
        normal_osc = any(o.chan == "hand_" + cs and o.param == "normal" for o in oscs)
        nrm_track = np.zeros((T, 3))

        def at(sg, a, b, st):
            n = b - a
            if "neutral" in sg.tgt:
                return np.zeros(4)
            spec = sg.tgt["ik"]
            hd = tuple(x[a:b] for x in head)
            ck = (id(sg), rs, a, b, hash(np.stack(hd).tobytes()), float(heading[min(b - 1, T - 1)]))
            hit = cache.get(ck)
            if hit is None:
                i1 = min(b - 1, max(a, int(math.ceil(sg.t1 * hz - 1e-9)) - 1))
                h = A.goal_h(spec, rs, hd, n, heading=float(heading[i1]))
                turn = _turn_for(spec, h)
                hit = cache[ck] = np.concatenate([h, turn[:, None]], 1)
            if normal_osc:
                nrm_track[a:b] = _normal_dir(A, spec, rs, hd, n)
            return hit

        def arc(sg, u, st, tg):
            """a long transition bulges toward the camera (hands travel in front of the body, not through it); part
            of the track itself, so an interrupting move starts from where the hand really is (no pop)"""
            g = np.asarray(tg, np.float64)
            g0 = g[0] if g.ndim == 2 else g
            disp = float(np.linalg.norm(g0[:3] - st[:3]))
            out_ = np.zeros((len(u), 4))
            if sg.src != "move_hand" and disp >= 0.3 and nat.get("arcs", True):
                out_[:, 2] = 0.18 * disp * np.sin(np.pi * np.clip(u, 0, 1))
            return out_
        V = chain(segs, F, np.zeros(4), at, nat, bump=arc)
        # oscillations (canon metres, body frame) -> hand units
        # (character gain; a wiggle smaller than the character's readable minimum is scaled up to it)
        def hosc(param, axis):
            v = np.zeros(T)
            for o in oscs:
                if o.chan != "hand_" + cs or o.param != param:
                    continue
                k = hpm[axis] if axis is not None else float(np.mean(hpm))
                amp_h = abs(o.amp) * k * c["osc_hand"]
                g = c["osc_hand"]
                if 0 < amp_h < c["hand_osc_min"]:
                    g *= c["hand_osc_min"] / amp_h
                v += g * osc_sum([o], "hand_" + cs, param, F)
            return v
        ox = hosc("x", 0)
        oy = hosc("y", 1)
        oz = hosc("z", 2)
        if normal_osc:
            on = hosc("normal", None)
            ox, oy, oz = ox + on * nrm_track[:, 0], oy + on * nrm_track[:, 1], oz + on * nrm_track[:, 2]
        V[:, 0] += sgn_out * ox * hpm[0] * (1.0 if cs == "l" else 1.0)
        V[:, 1] += oy * hpm[1]
        V[:, 2] += oz * hpm[2]
        # wrist rotations -> forearm roll
        wsegs = [sg for sg in B.segs if sg.chan == "wrist_" + cs]
        if wsegs:
            Wv = chain(wsegs, F, np.zeros(3), lambda sg, a, b, st: np.array(sg.tgt.get("wr", (0, 0, 0)), float), nat)
            V[:, 3] += Wv[:, 2] / 90.0
        V[:, 3] += osc_sum(oscs, "wrist_" + cs, "side", F) / 120.0 + osc_sum(oscs, "wrist_" + cs, "twist", F) / 90.0
        out[f"hand_{rs}_x"], out[f"hand_{rs}_y"], out[f"hand_{rs}_z"], out[f"hand_{rs}_turn"] = V.T
        # thumbs down: the fist turns over
        if any(sg.chan == "fing_" + cs and sg.tgt.get("shape") == "thumbs_down" for sg in B.segs):
            out[f"hand_{rs}_turn"] = out[f"hand_{rs}_turn"] - 1.5 * out[f"thumb_{rs}"]
    # ---------------------------------------------------------------- face
    _face(B, F, out, character, nat)
    tsegs = [sg for sg in B.segs if sg.chan == "tongue"]
    out["tongue_out"] = chain(tsegs, F, 0.0, lambda sg, a, b, st: 0.0 if "neutral" in sg.tgt else
                              float(sg.tgt.get("amount", 0.0)), nat)
    # ---------------------------------------------------------------- clip
    # ballistic take-offs (jump / flip ease out at full speed) are a velocity step: a short CAUSAL critically damped
    # filter on the vertical dials rounds them off (earlier frames never depend on later ones: streaming stays exact)
    a_ = math.exp(-1.0 / (hz * 0.045))
    for p in ("root_y", "squash", "body_bounce"):
        if p in out:
            v = np.asarray(out[p], np.float64) * np.ones(T)
            for _ in range(2):
                y = np.empty_like(v)
                acc = v[0]
                for i in range(T):
                    acc = a_ * acc + (1 - a_) * v[i]
                    y[i] = acc
                v = y
            out[p] = v
    # soft limits: a dial pushed past its range by a big command eases into the end stop (tanh knee) instead of
    # sitting clipped and then jumping across the range in one frame when the command reverses
    from .rigmap import RANGE as _RG
    for p in ("head_yaw", "head_pitch", "head_roll", "body_yaw", "body_pitch", "body_lean", "body_bounce", "gaze_x",
              "gaze_y"):
        if p in out:
            lo, hi = _RG[p]
            L = min(-lo, hi)
            v = np.asarray(out[p], np.float64)
            kn = 0.6 * L
            a = np.abs(v)
            out[p] = np.sign(v) * np.where(a <= kn, a, kn + (L - kn) * np.tanh((a - kn) / (L - kn)))
    rr = {"hand_l_y": (-1.0, A.y_hi), "hand_r_y": (-1.0, A.y_hi)}
    from .rigmap import RANGE
    for p, v in out.items():
        lo, hi = rr.get(p, RANGE.get(p, (-1e9, 1e9)))
        out[p] = np.clip(np.nan_to_num(np.asarray(v, np.float64) * np.ones(T)), lo, hi)
    return out


def _turn_for(spec, h):
    palm = spec.get("palm")
    if spec.get("aim"):
        t = 0.0
    elif isinstance(palm, str):
        t = PALM_TURN.get(palm, 0.0)
        if palm == "camera":
            t = 1.0
    else:
        t = 0.0
    return np.full(len(h), t)


def _normal_dir(A, spec, rs, hd, n):
    """(n,3) outward surface normal of the spec's target, in CANON body axes (x = the avatar's left)."""
    tg = spec["target"]
    nm = kin.normalize_point_name(tg) if isinstance(tg, str) else ""
    if nm in ("top_of_head", "head", "crown"):
        v = np.array([0.0, 1.0, 0.0])
    elif "ear" in nm or "temple" in nm or "cheek" in nm:
        sg = 1.0 if nm.startswith("left") else -1.0
        v = np.array([sg, 0.0, 0.3])
    elif "shoulder" in nm:
        v = np.array([0.0, 1.0, 0.2])
    elif "hip" in nm:
        sg = 1.0 if nm.startswith("left") else -1.0
        v = np.array([sg, 0.0, 0.5])
    else:
        v = np.array([0.0, 0.0, 1.0])
    v = v / np.linalg.norm(v)
    return np.repeat(v[None], n, 0)


FACE_KEYS = [n for n in kin.BS_NAMES if not n.startswith("eyeLook") and n != "_neutral"]


def _face(B, F, out, character, nat):
    segs_all = [sg for sg in B.segs if sg.chan == "face"]
    n = kin.neutral()
    vals = {}
    T = len(F)
    if segs_all:
        for kk in FACE_KEYS:
            segs = [sg for sg in segs_all if "neutral" in sg.tgt or kk in sg.tgt.get("shapes", {})]
            nv = float(n[kin.COL["bs_" + kk]])
            if not segs:
                continue
            segs.sort(key=lambda sg: (round(sg.t0, 4), 0 if sg.src == "expression" else 1))
            st_idx = _starts(segs, F)
            v = np.full(T, nv)
            for k, sg in enumerate(segs):
                a = st_idx[k]
                b = st_idx[k + 1] if k + 1 < len(segs) else T
                if a >= b:
                    continue
                st = v[a - 1] if a > 0 else nv
                tg = nv if "neutral" in sg.tgt else sg.tgt["shapes"][kk]
                if "restore_from" in sg.tgt:
                    hz_ = 1.0 / (F[1] - F[0]) if T > 1 else 20.0
                    j0 = max(0, int(round(sg.tgt["restore_from"] * hz_)) - 1)
                    tg = v[j0] if j0 < a else tg
                u = (F[a:b] - sg.t0) / max(sg.t1 - sg.t0, 1e-6)
                v[a:b] = st + (tg - st) * ease_vec(sg, u, nat)
            vals[kk] = v

    def g(k):
        return vals[k] if k in vals else np.full(T, float(n[kin.COL["bs_" + k]]))

    def d(k):
        return g(k) - float(n[kin.COL["bs_" + k]])
    inner = d("browInnerUp")
    for rs, hs in (("r", "Left"), ("l", "Right")):
        up = 0.5 * d("browOuterUp" + hs) + 0.5 * inner
        dn = d("browDown" + hs)
        out["brow_" + rs] = np.where(up - dn >= 0, 1.3 * (up - dn), 1.2 * (up - dn))
        bl = g("eyeBlink" + hs)
        out["blink_" + rs] = np.clip((bl - 0.1) / 0.8, 0, 1)
    out["brow_knit"] = np.clip(1.4 * (0.5 * (d("browDownLeft") + d("browDownRight"))) - 0.3 * np.maximum(inner, 0), 0, 1)
    out["eye_wide"] = np.clip(0.5 * (g("eyeWideLeft") + g("eyeWideRight")) * 1.0, 0, 1)
    out["squint"] = np.clip(0.9 * 0.5 * (d("eyeSquintLeft") + d("eyeSquintRight")) +
                            0.6 * 0.5 * (g("cheekSquintLeft") + g("cheekSquintRight")), 0, 1)
    out["smile"] = np.clip(1.1 * 0.5 * (d("mouthSmileLeft") + d("mouthSmileRight")) / 0.9, 0, 1)
    out["mouth_open"] = np.clip(1.15 * d("jawOpen"), 0, 1)
    out["nostril"] = np.clip(0.5 * (g("noseSneerLeft") + g("noseSneerRight")) / 0.6, 0, 1)
    if character == "fox":
        out["nose_wiggle"] = 0.4 * out["nostril"]
        b = 0.5 * (out["brow_l"] + out["brow_r"])
        base = 0.45 * np.maximum(b, 0) + 0.3 * out["eye_wide"] + 0.25 * out["smile"] - 0.5 * out["brow_knit"] \
            - 0.45 * np.maximum(-b, 0)
        out["ear_l"] = base + 0.3 * (out["brow_l"] - b)
        out["ear_r"] = base + 0.3 * (out["brow_r"] - b)
