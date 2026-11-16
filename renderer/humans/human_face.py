"""rig4: the HUMAN expression governor for Juno (table-driven; juno.face.json "human_face" over DEFAULTS).

The shared control vector was built for the plush (cartoon head throws, squeezed ^^ eyes, smirks, floating marks).
RIG3's `human_params` rescales columns; this layer adds the constraints a Pixar HUMAN face needs, applied to the
rig's own outputs so every feeling (and every blend nobody designed) obeys them — no per-feeling or per-phrase rules:

  head   (on the vector, before the solve)  the head's chin-up pitch and its roll, counted together with the
         upper-body pitch / lean that carries it, pass a soft knee to a cap (a human beams with a small lift, not a
         thrown-back head); chin-down and yaw are untouched (looking at objects / the board / the TV).
  smile  (shape weights)  a closed smile is capped where the closed-lip shapes stay clean (a broad closed smile reads
         as a wavy smear); beyond that the smile OPENS: the jaw drops a little and the upper lip lifts so the upper
         teeth row shows (the judged smile_teeth recipe: smile .7, jaw .2, upper-up .3), and jaw x smile stays < .5.
         The jaw part is added to the jaw value itself (bone == shape).
  sym    (shape weights)  L/R pairs of the mouth, cheeks and lids differ by at most `asym_max` of the larger one;
         the sideways mouth / jaw shift is capped at `skew_max` (no one-sided grimace).  Brows and blinks keep their
         asymmetry (one raised brow, a wink-free blink is the page's).
  brows  (shape weights)  the inner-brow lift that a tilt adds fades under a smile (inner-up over a smile reads
         pained); sadness / worry keep it.
  lids   (shape weights)  the lid APERTURE is measured from the asset (upper/lower lid margin vertices per eye: a
         linear coefficient per shape); the HELD closure from squint / happy lids / cheeks / sustained lid columns
         is scaled so the eye stays >= `open_min` (`open_min_smile` while smiling) of its rest opening.  Real blinks
         are told apart in time (a fast-rise follower): a blink still closes fully.
"""
from __future__ import annotations

import math
import os

import numpy as np

DEFAULTS = {
    "head": {"pitch_up_knee_deg": 3.0, "pitch_up_cap_deg": 6.5, "roll_knee_deg": 2.0, "roll_cap_deg": 5.5,
             "body_pitch_rad": 0.28, "body_lean_rad": 0.20},
    "smile": {"closed_max": 0.42, "open_from": 0.62, "open_full": 1.0, "open_extra": 0.28, "open_jaw": 0.2,
              "open_upper": 0.34, "jaw_x_smile_max": 0.45, "cheek_max": 0.25,
              "unpout": ["mouthPucker", "mouthShrugLower", "mouthShrugUpper", "mouthFunnel"], "unpout_from": 0.25, "unpout_full": 0.5},
    "sym": {"asym_max": 0.10, "skew_max": 0.15,
            "pairs": ["mouthSmile", "mouthFrown", "mouthStretch", "mouthUpperUp", "mouthLowerDown", "mouthPress",
                      "mouthDimple", "cheekSquint", "eyeSquint", "eyeHappy", "eyeWide", "noseSneer"]},
    "brows": {"inner_under_smile": 0.5, "smile_from": 0.25, "smile_full": 0.6, "opposed_keep": 0.2, "raise_pair": 0.55},
    "lids": {"open_min": 0.6, "open_min_smile": 0.85, "smile_from": 0.25, "smile_full": 0.55,
             "blink_rise_s": 0.35, "governed": ["eyeBlink", "eyeSquint", "eyeHappy", "cheekSquint", "eyeLookDown"]},
}


def _ss(e0, e1, x):
    t = min(max((x - e0)/(e1 - e0), 0.0), 1.0) if e1 != e0 else float(x >= e0)
    return t*t*(3.0 - 2.0*t)


def _soft_cap(x, knee, cap):
    """identity below knee, then a smooth approach to cap (x >= 0)."""
    if x <= knee:
        return x
    r = cap - knee
    return knee + r*math.tanh((x - knee)/r) if r > 1e-9 else knee


def _merge(base, over):
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def lid_coefficients(asset_dir, shape_names, lod="lod0"):
    """per eye ('Left'/'Right' = anatomical) a float32[nshapes]: the change of the lid aperture (fraction of the rest
    opening) per unit weight, measured at the upper lid margin vertex that the blink moves most and the lower lid margin
    vertex it lands on.  None if the asset has no usable head / blink shapes."""
    try:
        from . import asset_io
    except ImportError:                                  # loaded as a top-level module (tools)
        import asset_io  # type: ignore
    try:
        A = asset_io.load_asset(asset_dir, lod, hair=False, validate=False)
    except Exception:
        return None
    sn = A.shape_names
    si = {n: i for i, n in enumerate(sn)}
    P = A.mesh["position"]
    head = [s for s in A.submeshes if s.get("material") == "skin_head" or s.get("name") == "head"]
    if not head:
        return None
    vs = np.arange(head[0]["first_vertex"], head[0]["first_vertex"] + head[0]["vertex_count"])
    bones = {b["name"]: np.asarray(b["rest"], float).reshape(4, 4) for b in A.bones}
    out = {}
    for eye, suf in (("eye_l", "Left"), ("eye_r", "Right")):
        if eye not in bones or ("eyeBlink" + suf) not in si:
            return None
        R = bones[eye]
        c = R[:3, 3] if np.linalg.norm(R[:3, 3]) > 0.1 else R[3, :3]
        dmap = {}
        for (nm, idx, dp, _dn) in A.shapes:
            dmap[nm] = (idx, dp)

        def delta(nm, v):
            idx, dp = dmap[nm]
            j = np.searchsorted(idx, v)
            ok = (j < len(idx)) & (idx[np.minimum(j, len(idx) - 1)] == v)
            d = np.zeros((len(v), 3), np.float32)
            d[ok] = dp[j[ok]]
            return d
        near = vs[np.linalg.norm(P[vs] - c, axis=1) < 0.03]
        bl = delta("eyeBlink" + suf, near)
        mv = np.linalg.norm(bl, axis=1)
        if mv.max() < 1e-4:
            return None
        k = int(np.argmax(mv)); up = near[k]
        tgt = P[up] + bl[k]
        cand = near[(mv < 0.15*mv.max()) & (P[near][:, 1] < c[1])]
        if not len(cand):
            return None
        lo = cand[np.argmin(np.linalg.norm(P[cand] - tgt, axis=1))]
        ap0 = float(P[up][1] - P[lo][1])
        if ap0 <= 1e-5:
            return None
        coef = np.zeros(len(sn), np.float32)
        pair = np.array([up, lo])
        for nm in sn:
            d = delta(nm, pair)
            coef[si[nm]] = (d[0, 1] - d[1, 1])/ap0
        out[suf] = coef
    return out


class HumanFace:
    def __init__(self, cfg, shape_names, asset_dir=None, lod="lod0"):
        self.cfg = _merge(DEFAULTS, cfg or {})
        self.sn = list(shape_names)
        self.si = {n: i for i, n in enumerate(self.sn)}
        si = self.si
        self.pairs = [(si[p + "Left"], si[p + "Right"]) for p in self.cfg["sym"]["pairs"]
                      if p + "Left" in si and p + "Right" in si]
        self.smile = [si[n] for n in ("mouthSmileLeft", "mouthSmileRight") if n in si]
        self.upper = [si[n] for n in ("mouthUpperUpLeft", "mouthUpperUpRight") if n in si]
        self.cheek = [si[n] for n in ("cheekSquintLeft", "cheekSquintRight") if n in si]
        self.jaw = si.get("jawOpen")
        self.close = si.get("mouthClose")
        self.inner = si.get("browInnerUp")
        self.skew = [(si.get("mouthLeft"), si.get("mouthRight"))]     # (jawLeft/Right: capped on the jaw value, bone == shape)
        self.lid = None
        if asset_dir and os.environ.get("JUNO_LIDGOV", "1") != "0":
            self.lid = lid_coefficients(asset_dir, self.sn, lod)
        gov = self.cfg["lids"]["governed"]
        self.lid_gov = {}
        for suf in ("Left", "Right"):
            m = np.zeros(len(self.sn), bool)
            for g in gov:
                j = si.get(g + suf)
                if j is not None:
                    m[j] = True
            self.lid_gov[suf] = m
        self._blk = None                                   # blink follower state {param: (value, t)}

    # ---- head (vector space, before the solve)
    def head(self, Q, IDX):
        h = self.cfg["head"]
        bp = float(Q[IDX["body_pitch"]])*h["body_pitch_rad"]
        hp = float(Q[IDX["head_pitch"]])
        tp = hp + bp                                       # negative = chin up
        if tp < 0.0:
            tp2 = -math.radians(_soft_cap(math.degrees(-tp), h["pitch_up_knee_deg"], h["pitch_up_cap_deg"]))
            Q[IDX["head_pitch"]] = hp + (tp2 - tp)
        bl = float(Q[IDX["body_lean"]])*h["body_lean_rad"]
        hr = float(Q[IDX["head_roll"]])
        tr = hr + bl
        a = abs(tr)
        if a > 1e-9:
            tr2 = math.copysign(math.radians(_soft_cap(math.degrees(a), h["roll_knee_deg"], h["roll_cap_deg"])), tr)
            Q[IDX["head_roll"]] = hr + (tr2 - tr)
        return Q

    # ---- smile -> jaw (the jaw bone and jawOpen share it: called from the rig's jaw value)
    def smile_open(self, sm):
        s = self.cfg["smile"]
        return _ss(s["open_from"], s["open_full"], sm)

    def smile_jaw(self, sm):
        return self.cfg["smile"]["open_jaw"]*self.smile_open(sm)

    # ---- blink split: the held part of a lid column vs a fast blink
    def blink_split(self, b, key, t):
        """-> (held, transient) for a blink value b at time t: the held part follows b down at once and up slowly."""
        st = self._blk if self._blk is not None else {}
        if t is None or key not in st or b <= st[key][0]:
            f = b
        else:
            f0, t0 = st[key]
            dt = min(max(0.0, t - t0), 1.0)
            f = f0 + (b - f0)*(1.0 - math.exp(-dt/max(self.cfg["lids"]["blink_rise_s"], 1e-3)))
        st[key] = (f, t)
        self._blk = st
        r = (b - f)/(1.0 - f) if f < 0.999 else 0.0       # the blink's share of the remaining opening
        return f, max(0.0, min(1.0, r))

    # ---- the weight pass (after the table + mouth logic, before the brow amp + correctives)
    def weights(self, w, sm_raw, jaw, blink_tr=None, blink_ix=None, speaking=False):
        """w: shape weights (modified in place).  sm_raw: the mean smile weight the mouth logic produced; jaw: the jaw
        value already in w (incl. smile_jaw).  blink_tr: {suffix: transient blink weight} already REMOVED from w (added back
        after the lid governor so a blink closes fully)."""
        c = self.cfg
        s = c["smile"]
        if self.smile:
            o = self.smile_open(sm_raw)
            cap = s["closed_max"] + s["open_extra"]*o
            if jaw > 1e-6:
                cap = min(cap, max(s["closed_max"]*0.6, s["jaw_x_smile_max"]/jaw))
            for j in self.smile:
                w[j] = min(w[j], cap) if sm_raw > 1e-6 else w[j]
            if o > 0.0:
                for j in self.upper:
                    w[j] = max(w[j], s["open_upper"]*o)
                if self.close is not None:
                    w[self.close] *= 1.0 - o
            for j in self.cheek:
                w[j] = min(w[j], s["cheek_max"])
            ku = _ss(s["unpout_from"], s["unpout_full"], sm_raw)     # a smile with a pout / purse reads smug
            if ku > 0.0 and not speaking:
                for nm in s["unpout"]:
                    j = self.si.get(nm)
                    if j is not None:
                        w[j] *= 1.0 - ku
        # brows: the inner lift under a smile
        b = c["brows"]
        if self.inner is not None:
            k = _ss(b["smile_from"], b["smile_full"], sm_raw)
            if k > 0.0:
                outer = 0.0
                for nm in ("browOuterUpLeft", "browOuterUpRight"):
                    j = self.si.get(nm)
                    if j is not None:
                        outer = max(outer, float(w[j]))
                keep = 1.0 - k*(1.0 - b["inner_under_smile"])
                w[self.inner] = max(min(w[self.inner], outer), w[self.inner]*keep) if outer > 0 else w[self.inner]*keep
        # opposed brows (one up, the other pulled down) read sceptical on a human: the lowered one mostly relaxes
        for up_nm, dn_nm in (("browOuterUpLeft", "browDownRight"), ("browOuterUpRight", "browDownLeft")):
            ju, jd = self.si.get(up_nm), self.si.get(dn_nm)
            if ju is not None and jd is not None:
                k = _ss(0.15, 0.45, float(w[ju]) - float(w[self.si.get(up_nm.replace("Left", "X").replace("Right", "Left").replace("X", "Right"), ju)]))
                if k > 0.0:
                    w[jd] *= 1.0 - k*(1.0 - b["opposed_keep"])
        # one brow raised alone reads sceptical: the other rises with it (>= raise_pair of it; curiosity = both up, one higher)
        jl, jr = self.si.get("browOuterUpLeft"), self.si.get("browOuterUpRight")
        if jl is not None and jr is not None:
            hi = max(float(w[jl]), float(w[jr]))
            w[jl] = max(float(w[jl]), b["raise_pair"]*hi); w[jr] = max(float(w[jr]), b["raise_pair"]*hi)
        # symmetry
        am = c["sym"]["asym_max"]
        for a, bb in self.pairs:
            x, y = float(w[a]), float(w[bb])
            hi = max(x, y)
            if hi > 1e-6 and abs(x - y) > am*hi:
                m = 0.5*(x + y); d = am*m/(2.0 - am)          # |a - b| = asym_max x the larger one
                w[a], w[bb] = (m + d, m - d) if x > y else (m - d, m + d)
        sk = c["sym"]["skew_max"]
        for a, bb in self.skew:
            if a is not None and bb is not None:
                net = float(w[a] - w[bb])
                if abs(net) > sk:
                    f = sk/abs(net)
                    w[a] *= f; w[bb] *= f
        # lids
        self.last_open = None
        if self.lid is not None:
            lc = c["lids"]
            mn = lc["open_min"] + (lc["open_min_smile"] - lc["open_min"])*_ss(lc["smile_from"], lc["smile_full"], sm_raw)
            opens = {}
            for suf, coef in self.lid.items():
                cw = coef*w
                gm = self.lid_gov[suf] & (cw < 0.0)
                closing = float(-cw[gm].sum())
                ap = 1.0 + float(cw.sum())
                if ap < mn and closing > 1e-6:
                    f = max(0.0, 1.0 - (mn - ap)/closing)
                    idx = np.nonzero(gm)[0]
                    w[idx] *= f
                    ap = 1.0 + float((coef*w).sum())
                opens[suf] = ap
            self.last_open = opens
        if blink_tr and blink_ix:
            for suf, tr in blink_tr.items():
                j = blink_ix.get(suf)
                if j is not None and tr > 0.0 and w[j] < 1.0:
                    w[j] = w[j] + tr*(1.0 - w[j])
        return w
