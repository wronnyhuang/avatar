"""motion: the LIVING-PERSON layer under the shared control vector (junor; `<char>.face.json` "life").

The page (app/motion.js) already decides WHAT she does: where she looks, when she blinks, how she breathes, what the
mouth says.  This layer only adds what a real body does underneath those inputs, so the realistic character does not
read as a game NPC.  Every section is switched on by its key inside "life"; a character whose face json has no "life"
key (juno, every other character) never builds this object, so its rig output is bit-identical.

  eyes    fixation scanning while she looks at someone (the viewer's two eyes and mouth, 1-3 saccades a second,
          main-sequence duration D = d0 + k*A), slow fixation drift; conjugate (both eyes the same offset in head space,
          so the vergence on the target distance is kept).  Deterministic on the timeline time (seeded events per chunk).
  lids    lids that follow the gaze beyond the asset's eyeLook* shapes: upper lid rides the iris down (eyeBlink share)
          and up (eyeWide share), lower lid lifts a touch on down-gaze, brows lift a touch on up-gaze.
  blink   the BLINK part of the lid columns (the rig's fast-rise split): one eye a few ms behind the other (power
          curve), lower lid participation, brow and cheek micro-compression — only on real blinks, never on held lids.
  speech  instantaneous reshaping of the lipstream columns (no filter, no look-ahead: the mouth timing is the
          stream's, frame for frame): bilabial seal curve (m/b/p close fully), labiodental (f/v) + tongue (l/th/n)
          curves; head-jaw coupling (the head extends a little as the jaw drops) and voice-scaled head micro-motion.
  body    head turns shared down the spine (chest ~15 %, neck ~25 %, head the rest; the chest a beat late), hand
          targets kept where the page put them; clavicles elevate as an arm rises overhead (scapulohumeral rhythm);
          breathing in the chest/belly volume (bone scale on spine_02/03 + spine_01) on top of the breath column.
  tone    resting muscle tone added to every frame (shape weights; default: the lower lids at the limbus).
  expr    expression columns pass a slew limiter (nothing snaps; smooth input passes untouched), the Duchenne part of
          a smile (cheeks + lower lids) follows the mouth a beat late, and the brows carry a slight L/R asymmetry.

State: the chest lag and the expression followers are causal first-order followers on the timeline time (like the
rig's blink split): the same frame sequence gives the same output; dt <= 0 holds; a jump > 0.5 s resets.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULTS = {
    "eyes": {"seed": 4557, "chunk_s": 8.0,
             "eye_deg": 2.8,           # half-separation of the viewer's eyes (deg) at the fixation triangle
             "eye_up_deg": 0.25,       # the viewer's eyes sit a touch above the aim point
             "mouth_deg": 3.6,         # the viewer's mouth below the aim point (deg)
             "p_mouth": 0.18,          # share of fixations on the mouth
             "jitter_deg": 0.35,       # landing scatter round each fixation point
             "dwell_s": [0.45, 0.5],   # lognormal median, sigma of the fixation dwell ...
             "dwell_clip": [0.18, 1.6],    # ... clipped (1-3 saccades a second)
             "main_seq": [0.021, 0.0022],  # saccade duration = a + b * amplitude(deg)  (s)
             "skew": 0.8,              # velocity profile: time warp u**skew (peak velocity early, like real saccades)
             "drift_deg": 0.12, "drift_hz": [0.31, 0.53, 0.83],
             "away_from": 0.25, "away_full": 0.6, "away_keep": 0.35},   # |gaze| beyond away_from: smaller scan
    "lids": {"down": {"eyeBlink": 0.30, "eyeSquint": 0.06}, "up": {"eyeWide": 0.25, "browInnerUp": 0.06, "browOuterUp": 0.05}},
    "blink": {"lag_pow": 1.35, "lag_eye": "Left", "lower": 0.16, "brow": 0.10, "cheek": 0.07,
              "retract_mm": 1.0, "retract_from": 0.05, "retract_full": 0.5},   # the globe retracts as the lid sweeps
    "speech": {"press": [0.12, 0.45], "tuck": [0.05, 0.35], "tongue": [0.05, 0.4], "tongue_gain": 1.15,
               "tuck_upper_up": 0.30,  # f/v: the upper lip lifts off the incisors that rest on the lower lip
               "jaw_head_deg": 1.8, "voice_head_deg": [0.55, 0.35, 0.25], "voice_hz": [0.9, 1.37, 0.71]},
    "body": {"chest_share": 0.15, "neck_share": 0.25, "chest_lag_s": 0.12,
             "clav_raise_deg": 12.0, "clav_from_deg": 70.0, "clav_full_deg": 160.0,
             "breath_chest": [0.010, 0.004, 0.018], "breath_belly": [0.006, 0.0, 0.014],
             "sway_deg": [0.55, 0.35, 0.3], "sway_hz": [0.13, 0.21, 0.34]},   # postural head drift (the eyes stay locked)
    "tone": {"eyeSquint": 0.10},   # resting muscle tone: the lower lid sits at the limbus (no scleral show below the iris)
    "expr": {"slew": {"brow_l": 7.0, "brow_r": 7.0, "brow_knit": 6.0, "brow_tilt": 6.0, "smile": 5.0, "mouth_curve": 5.0,
                      "squint": 6.0, "eye_wide": 8.0, "lid_happy": 5.0, "cheek": 5.0, "mouth_wide": 5.0,
                      "mouth_pout": 5.0, "mouth_skew": 4.0},
             "duchenne": {"cheekSquint": 0.42, "eyeSquint": 0.14, "eyeHappy": 0.22}, "duchenne_lag_s": 0.12,
             "duchenne_side": [1.0, 0.92],
             "brow_asym": [1.0, 0.9]},
}


def _merge(base, over):
    out = {}
    for k, v in base.items():
        out[k] = dict(v) if isinstance(v, dict) else v
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _ss(e0, e1, x):
    t = min(max((x - e0)/(e1 - e0), 0.0), 1.0) if e1 != e0 else float(x >= e0)
    return t*t*(3.0 - 2.0*t)


def _mj(u):
    u = min(max(u, 0.0), 1.0)
    return u*u*u*(10.0 + u*(-15.0 + 6.0*u))


class _Follow:
    """first-order follower on the timeline time (tau s); dt <= 0 holds, a jump > 0.5 s resets."""
    def __init__(self):
        self.v = None; self.t = None

    def step(self, x, t, tau):
        x = np.asarray(x, float)
        if self.v is None or t is None or self.t is None or t - self.t > 0.5 or t < self.t - 1e-9:
            self.v = x.copy(); self.t = t; return self.v.copy()
        dt = t - self.t
        if dt > 0.0:
            a = 1.0 - math.exp(-dt/max(tau, 1e-4))
            self.v = self.v + (x - self.v)*a
            self.t = t
        return self.v.copy()


class _Slew:
    """per-column slew-rate limiter (units/s): smooth input passes unchanged, a step becomes a short ramp."""
    def __init__(self, n):
        self.v = None; self.t = None

    def step(self, x, t, rate):
        x = np.asarray(x, float)
        if self.v is None or t is None or self.t is None or t - self.t > 0.5 or t < self.t - 1e-9:
            self.v = x.copy(); self.t = t; return self.v.copy()
        dt = t - self.t
        if dt > 0.0:
            d = np.clip(x - self.v, -rate*dt, rate*dt)
            self.v = self.v + d
            self.t = t
        return self.v.copy()


class HumanLife:
    def __init__(self, cfg, IDX):
        cfg = cfg or {}
        self.on = {k: (k in cfg and cfg[k] is not False) for k in DEFAULTS}
        self.cfg = {k: _merge(DEFAULTS[k], cfg.get(k) if isinstance(cfg.get(k), dict) else {}) for k in DEFAULTS}
        self.IDX = IDX
        self._ev = {}
        self._chest = _Follow()
        self._duch = _Follow()
        e = self.cfg["expr"]
        self._slew_ix = np.array([IDX[n] for n in e["slew"] if n in IDX], int)
        self._slew_rate = np.array([float(e["slew"][n]) for n in e["slew"] if n in IDX], float)
        self._slew = _Slew(len(self._slew_ix))
        self._t_off = None

    # ------------------------------------------------------------------ eyes: fixation scanning + drift
    def _events(self, c):
        """the fixation events of chunk c: arrays (start time, yaw deg, pitch deg); deterministic on (seed, c)."""
        if c in self._ev:
            return self._ev[c]
        E = self.cfg["eyes"]
        rng = np.random.default_rng([int(E["seed"]), int(c) + 1000003])
        T0 = c*float(E["chunk_s"]); T1 = T0 + float(E["chunk_s"])
        ts, ys, ps = [], [], []
        t = T0 + float(rng.uniform(0.0, 0.4))
        side = 1.0 if rng.uniform() < 0.5 else -1.0
        while t < T1:
            r = rng.uniform()
            if r < float(E["p_mouth"]):
                y0, p0 = 0.0, -float(E["mouth_deg"])
            else:
                side = -side if rng.uniform() < 0.7 else side      # mostly alternate between the viewer's eyes
                y0, p0 = side*float(E["eye_deg"]), float(E["eye_up_deg"])
            j = float(E["jitter_deg"])
            ts.append(t); ys.append(y0 + rng.normal()*j); ps.append(p0 + rng.normal()*j*0.7)
            m, s = E["dwell_s"]
            lo, hi = E["dwell_clip"]
            t += float(np.clip(m*math.exp(s*rng.normal()), lo, hi))
        out = (np.array(ts), np.array(ys), np.array(ps))
        if len(self._ev) > 6:
            self._ev.pop(min(self._ev))
        self._ev[c] = out
        return out

    def _fix_at(self, t):
        """(yaw, pitch) deg of the fixation at time t incl. the saccade in flight."""
        E = self.cfg["eyes"]
        c = int(math.floor(t/float(E["chunk_s"])))
        ts, ys, ps = self._events(c)
        k = int(np.searchsorted(ts, t, side="right")) - 1
        if k < 0:                                    # before this chunk's first event: the previous chunk's last
            pt, py, pp = self._events(c - 1)
            return float(py[-1]), float(pp[-1])
        if k >= 1:
            fy, fp = float(ys[k - 1]), float(ps[k - 1])
        else:
            pt, py, pp = self._events(c - 1)
            fy, fp = float(py[-1]), float(pp[-1])
        ty, tp = float(ys[k]), float(ps[k])
        amp = math.hypot(ty - fy, tp - fp)
        a, b = E["main_seq"]
        dur = float(a) + float(b)*amp
        u = (t - float(ts[k]))/max(dur, 1e-4)
        if u >= 1.0:
            return ty, tp
        s = _mj(max(u, 0.0)**float(E["skew"]))
        return fy + (ty - fy)*s, fp + (tp - fp)*s

    def eye_offset(self, t, gx, gy, gaze_free):
        """conjugate eye offset (yaw, pitch) in radians, head space; +yaw = toward head +x, +pitch = UP."""
        if not self.on["eyes"] or t is None:
            return 0.0, 0.0
        E = self.cfg["eyes"]
        fy, fp = self._fix_at(float(t))
        dr = float(E["drift_deg"])
        hz = E["drift_hz"]
        dy = dr*(0.6*math.sin(2*math.pi*hz[0]*t + 1.3) + 0.4*math.sin(2*math.pi*hz[2]*t + 0.2))
        dp = dr*(0.6*math.sin(2*math.pi*hz[1]*t + 2.1) + 0.4*math.sin(2*math.pi*hz[2]*1.13*t + 4.0))
        away = max(abs(gx), abs(gy), gaze_free)
        k = 1.0 - (1.0 - float(E["away_keep"]))*_ss(float(E["away_from"]), float(E["away_full"]), away)
        return math.radians(fy*k + dy), math.radians(fp*k + dp)

    # ------------------------------------------------------------------ the vector, before the solve
    def pre(self, P, t):
        """speech curves + head-jaw / voice coupling + expression slew (on a copy of the vector)."""
        I = self.IDX
        Q = np.array(P, dtype=np.float64, copy=True)
        if self.on["expr"] and len(self._slew_ix):
            Q[self._slew_ix] = self._slew.step(Q[self._slew_ix], t, self._slew_rate)
        if self.on["speech"]:
            S = self.cfg["speech"]
            for col, key in (("lip_press", "press"), ("lip_tuck", "tuck")):
                v = float(Q[I[col]])
                if v > 0.0:
                    a, b = S[key]
                    Q[I[col]] = max(v, _ss(a, b, v))       # partial closures reach the seal / the teeth (never less)
            for col in ("tongue_tip", "tongue_dental"):
                v = float(Q[I[col]])
                if v > 0.0:
                    a, b = S["tongue"]
                    Q[I[col]] = min(1.0, max(v*float(S["tongue_gain"]), _ss(a, b, v)))
        return Q

    def head_extra(self, P, t, jaw):
        """additive head (yaw, pitch, roll) radians: the head extends as the jaw drops + voice-scaled micro-motion."""
        if t is None:
            return 0.0, 0.0, 0.0
        sy = sp = sr = 0.0
        if self.on["body"]:                       # postural sway: slow, smooth, never still (eyes stay on target)
            B = self.cfg["body"]; a = B["sway_deg"]; hz = B["sway_hz"]
            sy = math.radians(a[0])*(0.6*math.sin(2*math.pi*hz[0]*t + 0.7) + 0.4*math.sin(2*math.pi*hz[2]*t + 2.9))
            sp = math.radians(a[1])*(0.6*math.sin(2*math.pi*hz[1]*t + 4.1) + 0.4*math.sin(2*math.pi*hz[0]*1.7*t + 1.2))
            sr = math.radians(a[2])*math.sin(2*math.pi*hz[1]*0.8*t + 5.3)
        if not self.on["speech"]:
            return sy, sp, sr
        S = self.cfg["speech"]
        v = min(max(float(P[self.IDX["voice"]]), 0.0), 1.0)
        a = S["voice_head_deg"]; hz = S["voice_hz"]
        y = math.radians(a[0])*v*(0.6*math.sin(2*math.pi*hz[0]*t + 0.4) + 0.4*math.sin(2*math.pi*hz[1]*t + 2.2))
        p = math.radians(a[1])*v*(0.6*math.sin(2*math.pi*hz[1]*t + 1.1) + 0.4*math.sin(2*math.pi*hz[2]*t + 5.0))
        r = math.radians(a[2])*v*math.sin(2*math.pi*hz[2]*t + 3.3)
        p -= math.radians(float(S["jaw_head_deg"]))*jaw            # chin up a touch (negative pitch = chin up)
        return y + sy, p + sp, r + sr

    def speech_weights(self, w, si, tuck):
        """f/v: the upper teeth show over the tucked lower lip (labiodental contact reads only with the incisors seen)."""
        if not self.on["speech"] or tuck <= 0.0:
            return
        g = float(self.cfg["speech"]["tuck_upper_up"])*tuck
        for nm in ("mouthUpperUpLeft", "mouthUpperUpRight"):
            j = si.get(nm)
            if j is not None:
                w[j] += g

    # ------------------------------------------------------------------ body
    def chest_angles(self, hy, hp, hr, t):
        """the lagged head angles the chest share is taken of."""
        return self._chest.step([hy, hp, hr], t, float(self.cfg["body"]["chest_lag_s"]))

    # ------------------------------------------------------------------ shape weights (after the table + mouth logic)
    def tone_weights(self, w, si):
        if not self.on["tone"]:
            return
        for nm, g in self.cfg["tone"].items():
            for j in ((si.get(nm),) if nm in si else (si.get(nm + "Left"), si.get(nm + "Right"))):
                if j is not None:
                    w[j] += float(g)

    def lid_weights(self, w, si, ang, ym, pm):
        """ang: {'Left': (yaw, pitch_down_positive)} eye angles in radians."""
        if not self.on["lids"]:
            return
        L = self.cfg["lids"]
        for suf, (yaw, pit) in ang.items():
            dn = max(pit, 0.0)/pm; up = max(-pit, 0.0)/pm
            for amt, table in ((dn, L["down"]), (up, L["up"])):
                if amt <= 0.0:
                    continue
                for nm, g in table.items():
                    j = si.get(nm + suf, si.get(nm))
                    if j is not None:
                        w[j] += float(g)*amt*(0.5 if nm == "browInnerUp" else 1.0)

    def eye_retract_m(self, b):
        """globe retraction (metres, along the head's -z) for a lid closure b: the lid clears the cornea bulge."""
        if not self.on["blink"] or b <= 0.0:
            return 0.0
        B = self.cfg["blink"]
        return 1e-3*float(B["retract_mm"])*_ss(float(B["retract_from"]), float(B["retract_full"]), b)

    def blink_transient(self, btr):
        """one eye a few ms behind the other: the lagging eye's blink share = share ** lag_pow."""
        if not self.on["blink"] or not btr:
            return btr
        B = self.cfg["blink"]
        out = dict(btr)
        e = B["lag_eye"]
        if e in out and out[e] > 0.0:
            out[e] = float(out[e])**float(B["lag_pow"])
        return out

    def blink_weights(self, w, si, btr):
        """lower lid / brow / cheek micro-compression with the blink (after the lid governor)."""
        if not self.on["blink"] or not btr:
            return
        B = self.cfg["blink"]
        for suf, tr in btr.items():
            if tr <= 0.0:
                continue
            for nm, g in (("eyeSquint", B["lower"]), ("browDown", B["brow"]), ("cheekSquint", B["cheek"])):
                j = si.get(nm + suf)
                if j is not None:
                    w[j] += float(g)*tr

    def expr_weights(self, w, si, smile, t, keep_open=1.0):
        """the Duchenne part of a smile (cheeks up, lower lids), a beat behind the mouth."""
        if not self.on["expr"]:
            return
        E = self.cfg["expr"]
        s = float(self._duch.step([max(smile, 0.0)], t, float(E["duchenne_lag_s"]))[0])
        if s <= 0.0:
            return
        sd = E["duchenne_side"]
        for nm, g in E["duchenne"].items():
            for suf, k in (("Left", sd[0]), ("Right", sd[1])):
                j = si.get(nm + suf)
                if j is not None:
                    w[j] += float(g)*k*s

    def brow_asym(self, w, si):
        if not self.on["expr"]:
            return
        a = self.cfg["expr"]["brow_asym"]
        for nm in ("browOuterUp", "browDown", "browTilt"):
            for suf, k in (("Left", a[0]), ("Right", a[1])):
                j = si.get(nm + suf)
                if j is not None:
                    w[j] *= float(k)
