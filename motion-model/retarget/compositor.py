"""The override compositor (docs/CONTRACT.md §0, §5): one rig vector per 30 Hz frame from
   (1) the unified planner's canon frame (retargeted to the character),
   (2) lip sync: owns the mouth while the avatar talks (the renderer's lipstream makes the lips; here mouth_open /
       tongue_out from the model fade out with a short crossfade and come back after the turn),
   (3) commanded actions (the action engine's rig values + which rig channels it owns, with weights 0..1).
While an action owns rig channels, the canon GROUPS behind them (retarget.RIG_GROUP, contract BODY_GROUPS) are handed
to the planner with set_override(values = the action's pose mapped back to canon) so the model keeps the rest of the
body coherent with it; when the action lets go, release_override(fade_out_s) makes the model blend back FROM the
action's last pose; the planner's own canon blend is kept SHORT (planner_fade_out_s 0.12 s: a linear axis-angle blend
between distant poses is not a natural path) and hidden under the OUTPUT crossfade (fade_out_s 0.5 s, min-jerk, from
the action's last rig values to the model's), so nothing pops.  On the way in the output blends from the pose that
was on screen when the action took the channel (the planner may jump to the action pose at once: inpainting).  Face and gaze stay the model's unless an action touches those exact channels.

    comp = Compositor(Retargeter("muse"), planner=p)            # planner: unified.runtime.UnifiedPlanner (or None)
    rig = comp.step(motion189, talking=bool, action={"values": {p: v}, "weights": {p: w}, "add": {p: dv}})
    comp.stats()  # per-frame ms, overrides set/released

Action weights are the OWNERSHIP of each rig channel (1 = the action's value, 0 = the model's).  An action engine that
already eases (the page's affine map: final = alpha*learned + beta -> weight 1-alpha, value beta/(1-alpha)) is
followed as is on the way in; a weight that drops faster than fade_out_s allows is decayed at that rate from the
action's last value (an abrupt stop never pops).  "add" = additive offsets on top (reflexes), never ownership.
"""
from __future__ import annotations

import time

import numpy as np

from canon1 import spec
from .retarget import RIG_GROUP, RANGES

MOUTH_CH = ("mouth_open", "tongue_out")
OVERRIDABLE = set(spec.BODY_GROUPS)


def _mj(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 + s * (6 * s - 15))


class _Ramp:
    """0..1 envelope with a min-jerk ramp of `dur_up` / `dur_dn` seconds toward its target (retargets mid-ramp from
    the current value, so it is C0 always and smooth at the ends)."""

    def __init__(self, dur_up, dur_dn):
        self.up, self.dn = dur_up, dur_dn
        self.v = 0.0; self.a = 0.0; self.b = 0.0; self.s = 1.0; self.dur = 1.0

    def step(self, target, dt):
        if target != self.b:
            self.a, self.b, self.s = self.v, target, 0.0
            self.dur = max(1e-3, (self.up if target > self.v else self.dn) * abs(target - self.v))
        self.s = min(1.0, self.s + dt / self.dur)
        self.v = self.a + (self.b - self.a) * _mj(self.s)
        return self.v


class Compositor:
    def __init__(self, retargeter, planner=None, lip_fade_s=0.15, fade_in_s=0.15, fade_out_s=0.5,
                 planner_fade_in_s=0.15, planner_fade_out_s=0.12,
                 talk_hang_s=0.25, on_th=0.02, inverse=True, drop_th=0.25, release_frac=0.97,
                 freeze_s=0.5):
        self.rt = retargeter
        self.planner = planner
        self.fade_in_s, self.fade_out_s = fade_in_s, fade_out_s
        self.p_fade_in_s, self.p_fade_out_s = planner_fade_in_s, planner_fade_out_s
        self.talk = _Ramp(lip_fade_s, lip_fade_s)
        self.talk_hang = talk_hang_s
        self.on_th = on_th
        self.inverse = inverse
        self.drop_th, self.release_frac = drop_th, release_frac
        self.freeze_s = freeze_s
        self.dt = 1.0 / spec.FPS
        self.reset()

    def reset(self):
        self.w = {}                 # rig channel -> {"w": effective ownership weight, "ramp": release ramp}
        self.last_val = {}          # rig channel -> the action's last value (held while fading out)
        self.groups_on = set()      # canon groups currently overridden in the planner
        self.peak = {}
        self.ov_canon = None        # the action pose in canon (what set_override's callable returns)
        self.talk_off_at = None
        self.n = 0
        self.ms = []
        self.events = []
        self.talk.v = 0.0; self.talk.b = 0.0
        if hasattr(self.rt, "reset"):
            self.rt.reset()

    # the planner reads this every frame while an override is on (values=callable)
    def _ov_values(self, k=None):
        return self.ov_canon

    def step(self, motion, talking=False, action=None, mask=None):
        t0 = time.perf_counter()
        dt = self.dt
        L = self.rt.step(motion, mask)
        out = dict(L)
        # ---------------- lip sync owns the mouth while the avatar talks (with a hangover so pauses don't flicker)
        if talking:
            self.talk_off_at = None
            tgt = 1.0
        else:
            if self.talk_off_at is None:
                self.talk_off_at = self.n
            tgt = 1.0 if (self.n - self.talk_off_at) * dt < self.talk_hang and self.talk.v > 0 else 0.0
        e = self.talk.step(tgt, dt)
        if e > 0:
            for p in MOUTH_CH:
                if p in out:
                    out[p] = out[p] * (1.0 - e)
        # ---------------- commanded actions
        vals = (action or {}).get("values") or {}
        wts = (action or {}).get("weights") or {}
        adds = (action or {}).get("add") or {}
        for p in set(self.w) | set(wts):
            raw = float(wts.get(p, 0.0))
            raw = 0.0 if not np.isfinite(raw) else min(max(raw, 0.0), 1.0)
            st = self.w.get(p)
            if st is None:
                st = self.w[p] = {"w": 0.0, "ramp": None}
            if p in vals and np.isfinite(vals[p]) and raw > 0:
                v_new = float(vals[p])
                st["vel"] = v_new - self.last_val[p] if p in self.last_val else 0.0
                st["held"] = 0
                self.last_val[p] = v_new
            elif p in self.last_val and st.get("vel"):
                # the action let go while moving: its value glides on with a decaying velocity (tau 80 ms) instead
                # of stopping dead, so the release has no velocity step
                st["held"] = st.get("held", 0) + 1
                st["vel"] *= np.exp(-dt / 0.08)
                self.last_val[p] += st["vel"]
            w = st["w"]
            r = st["ramp"]
            if r is None and abs(raw - w) <= self.drop_th:
                st["w"] = raw                            # the action engine's own ease (in or out), as is
            else:
                # an abrupt change (an action that starts / stops without easing): min-jerk ramp from where the
                # weight is to the new target, fade_in_s up / fade_out_s down (scaled by the distance)
                if r is None or abs(raw - r[1]) > self.drop_th:
                    dur = (self.fade_in_s if raw > w else self.fade_out_s) * abs(raw - w)
                    # a release of a channel whose canon group the planner holds waits until the planner's own
                    # (short, linear canon) release blend is over, so it never shows through
                    hold = 0
                    if raw < w and self.planner is not None and RIG_GROUP.get(p) in self.groups_on:
                        hold = int(np.ceil(self.p_fade_out_s / dt)) + 1
                    r = st["ramp"] = [w, raw, 0.0, max(1e-3, dur), hold]
                else:
                    r[1] = raw                           # small drift of the target: keep the ramp's progress
                if r[4] > 0:
                    r[4] -= 1
                else:
                    r[2] = min(1.0, r[2] + dt / r[3])
                st["w"] = r[0] + (r[1] - r[0]) * _mj(r[2])
                if r[2] >= 1.0:
                    st["ramp"] = None
            if st["w"] <= 1e-4 and raw <= 0:
                self.w.pop(p, None); self.last_val.pop(p, None)
                continue
            v = self.last_val.get(p, out.get(p, 0.0))
            base = out.get(p, 0.0)
            # blend IN from the pose that was on screen when the action took the channel (held for up to
            # freeze_s while the weight rises): the planner's own canon output for an overridden group may jump to
            # the action's pose at once (inpainting) — that jump must not show through (1 - w)
            if w <= 1e-4 and st["w"] > w:
                st["from"], st["from_n"] = base, self.n
            if "from" in st:
                if st["w"] >= 1.0 - 1e-6 or st["w"] < w - 1e-9 or (self.n - st["from_n"]) * dt > self.freeze_s:
                    del st["from"]
                else:
                    base = st["from"]
            out[p] = base + (v - base) * st["w"]
        for p, dv in adds.items():
            if np.isfinite(dv):
                out[p] = out.get(p, 0.0) + float(dv)
        # ---------------- planner overrides for the body groups an action owns
        if self.planner is not None:
            graw = {}
            for p, w in wts.items():
                g = RIG_GROUP.get(p)
                if g in OVERRIDABLE and np.isfinite(w):
                    graw[g] = max(graw.get(g, 0.0), float(w))
            if self.inverse and (graw or self.groups_on):
                owned = {p: self.last_val[p] for p in self.w if p in self.last_val}
                full = dict(L); full.update(owned)
                self.ov_canon, _ = self.rt.inverse(full, base=motion)
            new, gone = [], []
            for g in OVERRIDABLE:
                r = graw.get(g, 0.0)
                if g not in self.groups_on:
                    if r > self.on_th:
                        new.append(g); self.peak[g] = r
                else:
                    self.peak[g] = max(self.peak.get(g, 0.0), r)
                    if r <= self.on_th or r < self.release_frac * self.peak[g]:
                        gone.append(g)
            if new:
                new.sort()
                try:
                    self.planner.set_override(groups=new, values=self._ov_values if self.inverse else None,
                                              fade_in_s=self.p_fade_in_s)
                except Exception as ex:  # noqa: BLE001
                    self.events.append(("set_override_error", self.n, str(ex)[:120]))
                self.groups_on |= set(new)
                self.events.append(("set", self.n, new))
            if gone:
                gone.sort()
                try:
                    self.planner.release_override(groups=gone, fade_out_s=self.p_fade_out_s)
                except Exception as ex:  # noqa: BLE001
                    self.events.append(("release_error", self.n, str(ex)[:120]))
                self.groups_on -= set(gone)
                for g in gone:
                    self.peak.pop(g, None)
                self.events.append(("release", self.n, gone))
        for p, v in out.items():
            r = RANGES.get(p)
            if r is not None:
                out[p] = float(min(max(v, r[0]), r[1]))
        self.n += 1
        self.ms.append((time.perf_counter() - t0) * 1000)
        if len(self.ms) > 4000:
            del self.ms[:2000]
        return out

    @staticmethod
    def from_affine(names, ab, page=None):
        """the page's learned-mode map (motion.js learnedParts: [(index, alpha, beta)], final = alpha*learned + beta)
        -> an action dict: weight = 1 - alpha, value = beta / (1 - alpha) (alpha ~1: beta is an additive offset)."""
        vals, wts, adds = {}, {}, {}
        for i, al, be in ab or []:
            i = int(i)
            if i < 0 or i >= len(names):
                continue
            p = names[i]
            w = 1.0 - float(al)
            if w > 1e-3:
                wts[p] = w
                vals[p] = float(be) / w
            elif abs(be) > 1e-6:
                adds[p] = float(be)
        return {"values": vals, "weights": wts, "add": adds}

    def stats(self):
        a = np.asarray(self.ms[-2000:]) if self.ms else np.zeros(1)
        return {"n": self.n, "ms_p50": round(float(np.median(a)), 3), "ms_p95": round(float(np.percentile(a, 95)), 3),
                "overrides": [e for e in self.events[-20:]], "groups_on": sorted(self.groups_on)}
