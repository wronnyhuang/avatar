"""ACT clip player  — numpy only.  Plays curated real-mocap action clips on top of a conversational base pose.

    from player import ClipPlayer
    P = ClipPlayer(pkg_dir, seed=0)
    P.names()                         -> [slug, ...]
    P.match("wave hello")             -> (slug, score 0..1)
    P.plan("nod and then give me a thumbs up") -> [{"slug","score","mirror"}, ...]   (compound / left-right aware)
    P.play(slug, variant=None, mirror=False, speed=1.0, hold_s=None)   # interrupt-safe; queue=True appends
    P.release()                       # end a sustained hold ("cover your eyes" holds until released or hold_s)
    P.stop()                          # blend out now
    pose, weights = P.step(base_pose189)   # once per 30 fps frame

Composition per frame (canon1 cols 0:135; face/gaze cols 135:189 always pass through from the base):
  * the arm(s) / fingers the clip OWNS are taken from the clip (absolute local rotations: contacts like hand-on-face
    need them);
  * torso (root rot + shift, spine) and head/neck are ADDITIVE: base (+) (clip - clip's first frame), gain 1 when the
    clip owns that group (nod, bow, look_*), else 0.8 torso / 0.5 head (live head keeps following the user);
  * everything else = base.
  Every switch (start, interrupt, end, release, stop) is INERTIALIZED: the offset between what was on screen and the
  new target (rotation-vector space per joint + its velocity) decays with a critically damped spring (~0.35 s in,
  ~0.45 s out) — never a cross-fade to a frozen pose.  Per play: speed +-8 %, amplitude +-10 %, random variant.
  weights{group: 0..1} tell the mixer which groups the clip currently drives (decays during the blend-out).
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

NB = 135                                   # body cols
_TRI = np.array([[0, 1, 2]] + [[6 + 3 * k, 7 + 3 * k, 8 + 3 * k] for k in range(43)])   # 44 rotation triples
_JN = ["root_rot", "spine1", "spine2", "spine3", "neck", "head", "collar_l", "collar_r", "shoulder_l", "shoulder_r",
       "elbow_l", "elbow_r", "wrist_l", "wrist_r"]
_TRI_GROUP = np.array([0, 0, 0, 0, 1, 1, 2, 3, 2, 3, 2, 3, 2, 3] + [4] * 15 + [5] * 15)
GROUPS = ["torso", "head", "arm_l", "arm_r", "fingers_l", "fingers_r"]
_ABS = {"arm_l", "arm_r", "fingers_l", "fingers_r"}


# ----------------------------------------------------------------------------------------------- quaternions (w,x,y,z)
def _aa2q(v):
    th = np.linalg.norm(v, axis=-1, keepdims=True)
    s = np.where(th > 1e-9, np.sin(th / 2) / np.maximum(th, 1e-12), 0.5)
    return np.concatenate([np.cos(th / 2), v * s], -1)


def _q2aa(q):
    q = q * np.where(q[..., :1] < 0, -1.0, 1.0)
    n = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    th = 2 * np.arctan2(n, q[..., :1])
    return q[..., 1:] * np.where(n > 1e-9, th / np.maximum(n, 1e-12), 2.0)


def _qmul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def _qinv(q):
    return q * np.array([1.0, -1, -1, -1])


def _ngrams(s, n=3):
    s = f"  {s} "
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _norm(t):
    return re.sub(r"[^a-z0-9 ]+", " ", str(t).lower().replace("'", "")).split()


_STOP = set("a an the to me my your you please can could would will just like at with of and on in up some for it "
            "do give make show i im lets let us be so".split())


_SOFT = set("a an the your my her his their our".split())


def _soft(toks):
    """drop articles/possessives and plural s: 'hands on your hips' ~ 'hand on hip'."""
    return " ".join(t[:-1] if len(t) > 3 and t.endswith("s") else t for t in toks if t not in _SOFT)


class ClipPlayer:
    def __init__(self, pkg_dir, seed=0, tau_in=0.35, tau_out=0.45):
        self.dir = pkg_dir
        idx = json.load(open(os.path.join(pkg_dir, "index.json")))
        self.meta = {a["slug"]: a for a in idx["actions"]}
        z = np.load(os.path.join(pkg_dir, "library.npz"))
        self.clips = {k: z[k].astype(np.float64) for k in z.files}
        mp = np.asarray(idx["mirror_perm"], np.int64)
        ms = np.asarray(idx["mirror_sign"], np.float64)
        self._mperm, self._msign = mp, ms
        self.neutral = np.asarray(idx.get("neutral_body", np.zeros(NB)), np.float64)
        self._mcache = {}
        self.rng = np.random.default_rng(seed)
        self.tau_in, self.tau_out = tau_in, tau_out
        self.fps = 30.0
        self._alias = []
        for s, a in self.meta.items():
            for al in [s.replace("_", " ")] + list(a.get("aliases", [])):
                self._alias.append((s, " ".join(_norm(al)), _ngrams(" ".join(_norm(al))), _soft(_norm(al))))
        self.reset()

    # ------------------------------------------------------------------------------------------- public
    def names(self):
        return list(self.meta)

    def reset(self):
        self.cur = None            # dict for the clip being played
        self.queue = []
        self.prev_out = None
        self.prev_out2 = None
        self.prev_base = None
        self.off = None            # (44,3) rotation-vector offset, (3,) shift offset
        self.offv = None
        self.off_t = 0.0
        self.off_tau = self.tau_in
        self.out_env = 0.0         # weight envelope during a blend-out
        self.out_groups = {}
        self._pending_switch = False

    def match(self, text):
        r = self._match(text)
        return r["slug"], r["score"]

    def plan(self, text, min_score=0.6):
        """compound commands: split on 'then' / 'and then' / 'and' / ',' -> one match per part.
        Returns [] when nothing matches with score >= min_score (fuzzy n-gram hits below ~0.6 are often a different action,
        e.g. 'jump' -> startle 0.41): the caller then falls back to the planner.  Exact alias hits score >= 0.75."""
        parts = [p for p in re.split(r"\bthen\b|,|;|\band\b", str(text).lower()) if p.strip()]
        out = []
        for p in parts:
            r = self._match(p)
            if r["slug"] and r["score"] >= 0.35 and (not out or out[-1]["slug"] != r["slug"]):
                out.append(r)
        if not out:
            r = self._match(text)
            out = [r] if r["slug"] else []
        return [r for r in out if r["score"] >= min_score]

    def play(self, slug, variant=None, mirror=False, speed=1.0, hold_s=None, queue=False, amp=None):
        if slug not in self.meta:
            raise KeyError(slug)
        m = self.meta[slug]
        vs = m["variants"]
        if variant is None:
            w = np.array([v.get("weight", 1.0) for v in vs], np.float64)
            vi = int(self.rng.choice(len(vs), p=w / w.sum()))
        elif isinstance(variant, int):
            vi = variant % len(vs)
        else:
            vi = next(i for i, v in enumerate(vs) if v["key"] == variant)
        v = vs[vi]
        clip = self._clip(v["key"], mirror)
        sp = float(speed) * float(self.rng.uniform(0.92, 1.08))
        am = float(amp) if amp is not None else float(self.rng.uniform(0.9, 1.1))
        if m["kind"] in ("head",):
            am = float(np.clip(am, 0.95, 1.1))
        owns = set(m["owns"])
        if mirror:
            owns = {{"arm_l": "arm_r", "arm_r": "arm_l", "fingers_l": "fingers_r", "fingers_r": "fingers_l"}.get(g, g) for g in owns}
        hold = None
        if m.get("sustain") and v.get("hold"):
            h0, h1 = v["hold"]
            hold = (h0, h1, float(hold_s if hold_s is not None else 3.0))
        if m.get("head_lock"):
            owns = owns | {"head"}
        c = dict(lock=bool(m.get("head_lock")), slug=slug, key=v["key"], clip=clip, t=0.0, speed=sp, amp=am, owns=owns, hold=hold, held=0.0,
                 dirn=1.0, released=False, mirror=bool(mirror), variant=vi, ref=clip[0].copy(),
                 mean=clip.mean(0))
        if queue and (self.cur is not None or self.queue):
            self.queue.append(c)
            return c["key"]
        self.queue = []
        self._start(c)
        return c["key"]

    def release(self):
        if self.cur is not None:
            self.cur["released"] = True

    def stop(self):
        self.queue = []
        if self.cur is not None:
            self.cur = None
            self._pending_switch = True

    @property
    def active(self):
        return self.cur is not None or self.out_env > 0.01

    def step(self, base):
        base = np.asarray(base, np.float64)
        out = base.copy()
        if self.prev_out is None:
            self.prev_out = base[:NB].copy(); self.prev_out2 = base[:NB].copy(); self.prev_base = base[:NB].copy()
        c = self.cur
        if c is not None:
            self._advance(c)
            if c.get("done"):
                self.cur = None
                if self.queue:
                    self._start(self.queue.pop(0))
                    c = self.cur
                    self._advance_first = True
                else:
                    self._pending_switch = True
                    c = None
        tgt, groups = self._target(base, c)
        if self._pending_switch:
            self._switch(tgt, base, c)
        res = self._apply_offset(tgt)
        out[:NB] = res
        self.prev_out2 = self.prev_out
        self.prev_out = res.copy()
        self.prev_base = base[:NB].copy()
        if c is not None:
            w = {g: (1.0 if g in groups["abs"] else (groups["add"].get(g, 0.0))) for g in GROUPS}
        else:
            w = {g: self.out_env * self.out_groups.get(g, 0.0) for g in GROUPS}
        return out.astype(np.float32), w

    # ------------------------------------------------------------------------------------------- internals
    def _match(self, text):
        toks = [t for t in _norm(text) if t not in _STOP]
        q = " ".join(_norm(text))
        best, bs = None, 0.0
        qs = _soft(_norm(text))
        for slug, al, ng, als in self._alias:
            if not al:
                continue
            if re.search(r"\b" + re.escape(al) + r"\b", q):
                s = 0.75 + 0.25 * min(1.0, len(al) / max(8, len(q)))
            elif als and re.search(r"\b" + re.escape(als) + r"\b", qs):
                s = 0.72 + 0.25 * min(1.0, len(als) / max(8, len(qs)))
            else:
                qn = _ngrams(" ".join(toks))
                s = 0.9 * len(qn & ng) / max(1, len(qn | ng)) ** 0.5 / max(1, len(ng)) ** 0.5
            if s > bs:
                best, bs = slug, s
        mirror = False
        if best is not None:
            h = self.meta[best].get("hand")
            words = set(_norm(text))
            if h == "right" and "left" in words and not best.endswith("_right"):
                mirror = True
            if h == "left" and "right" in words and not best.endswith("_left"):
                mirror = True
        return {"slug": best, "score": round(float(bs), 3), "mirror": mirror}

    def _clip(self, key, mirror):
        c = self.clips[key]
        if not mirror:
            return c
        if key not in self._mcache:
            full = np.zeros((len(c), 189)); full[:, :NB] = c
            self._mcache[key] = (full[:, self._mperm] * self._msign)[:, :NB]
        return self._mcache[key]

    def _start(self, c):
        self.cur = c
        self._pending_switch = True

    def _advance(self, c):
        if c.get("_first", True):
            c["_first"] = False
            return
        n = len(c["clip"])
        h = c["hold"]
        if h is not None and not c["released"] and c["held"] < h[2] and h[0] <= c["t"] <= h[1] + 1e-6:
            c["held"] += 1.0 / self.fps
            nt = c["t"] + c["dirn"] * c["speed"] * 0.6        # ping-pong slowly inside the hold segment
            if nt > h[1] or nt < h[0]:
                c["dirn"] *= -1.0
                nt = min(max(nt, h[0]), h[1])
            c["t"] = nt
            return
        c["t"] += c["speed"]
        if c["t"] >= n - 1:
            c["t"] = n - 1
            c["done"] = True

    def _frame(self, c, t=None):
        t = c["t"] if t is None else t
        clip = c["clip"]
        i = int(np.floor(t)); f = t - i
        i = min(max(i, 0), len(clip) - 1); j = min(i + 1, len(clip) - 1)
        fr = clip[i] * (1 - f) + clip[j] * f
        return c["mean"] + c["amp"] * (fr - c["mean"])

    def _prep(self, c):
        """per-play precompute: additive deltas of the 6 torso/head triples vs the clip's first frame (log space)."""
        clip = c["mean"] + c["amp"] * (c["clip"] - c["mean"])
        c["clipa"] = clip
        tri6 = _TRI[:6]
        qr = _aa2q(clip[0][tri6]); qc = _aa2q(clip[:, tri6])
        c["dl"] = _q2aa(_qmul(np.broadcast_to(_qinv(qr), qc.shape), qc))      # (T,6,3)
        c["dsh"] = clip[:, 3:6] - clip[0, 3:6]
        owns = c["owns"]
        c["gain"] = {"torso": 1.0 if "torso" in owns else 0.8, "head": 1.0 if "head" in owns else 0.5}
        c["g6"] = np.array([c["gain"]["torso"]] * 4 + [c["gain"]["head"]] * 2)[:, None]
        cols = []
        for k, g in enumerate(_TRI_GROUP):
            if GROUPS[g] in owns and GROUPS[g] in _ABS:
                cols.extend(_TRI[k].tolist())
        if c.get("lock"):
            cols.extend(_TRI[4].tolist() + _TRI[5].tolist())
        c["abs_cols"] = np.array(cols, np.int64)

    def _target(self, base, c, t=None):
        b = base[:NB]
        if c is None:
            return b.copy(), {"abs": set(), "add": {}}
        if "dl" not in c:
            self._prep(c)
        t = c["t"] if t is None else t
        n = len(c["clipa"])
        i = int(np.floor(t)); f = t - i
        i = min(max(i, 0), n - 1); j = min(i + 1, n - 1)
        tgt = b.copy()
        d = (c["dl"][i] * (1 - f) + c["dl"][j] * f) * c["g6"]
        tri6 = _TRI[:6]
        tgt[tri6] = _q2aa(_qmul(_aa2q(b[tri6]), _aa2q(d)))
        tgt[3:6] = b[3:6] + c["gain"]["torso"] * (c["dsh"][i] * (1 - f) + c["dsh"][j] * f)
        ac = c["abs_cols"]           # owned arms/fingers (+ neck/head for hand-to-face contact: base yaw would miss the face)
        if len(ac):
            tgt[ac] = c["clipa"][i, ac] * (1 - f) + c["clipa"][j, ac] * f
        return tgt, {"abs": {g for g in c["owns"] if g in _ABS}, "add": dict(c["gain"])}

    def _switch(self, tgt, base, c):
        """new inertialization offset = what is on screen minus the new target (+ velocity difference)."""
        self._pending_switch = False
        if c is not None and c["t"] > 0:
            tprev, _ = self._target(self.prev_base, c, max(0.0, c["t"] - c["speed"]))
        elif c is not None:
            tprev, _ = self._target(self.prev_base, c, 0.0)
            f1, _ = self._target(self.prev_base, c, min(len(c["clip"]) - 1, c["speed"]))
            tprev = tgt - (f1 - tgt)          # clip's own velocity at its first frame
        else:
            tprev = self.prev_base.copy()
        qo = _aa2q(self.prev_out[_TRI]); qt = _aa2q(tgt[_TRI])
        off = _q2aa(_qmul(qo, _qinv(qt)))
        qo2 = _aa2q(self.prev_out2[_TRI]); qt2 = _aa2q(tprev[_TRI])
        off2 = _q2aa(_qmul(qo2, _qinv(qt2)))
        self.off = np.concatenate([off.reshape(-1), self.prev_out[3:6] - tgt[3:6]])
        prev_lin = self.prev_out2[3:6] - tprev[3:6]
        self.offv = np.concatenate([(off - off2).reshape(-1), (self.prev_out[3:6] - tgt[3:6]) - prev_lin]) * self.fps
        self.offv = np.clip(self.offv, -12.0, 12.0)
        self.off_t = 0.0
        self.off_tau = self.tau_in if c is not None else self.tau_out
        if c is None:
            self.out_env = 1.0
            last = getattr(self, "_last_groups", {})
            self.out_groups = last
        else:
            self._last_groups = {g: 1.0 for g in c["owns"]}
            self._last_groups.update({"torso": max(self._last_groups.get("torso", 0), 0.8),
                                      "head": max(self._last_groups.get("head", 0), 0.5)})
            self.out_env = 0.0

    def _apply_offset(self, tgt):
        if self.off is None:
            return tgt
        y = 3.5 / self.off_tau
        t = self.off_t
        e = np.exp(-y * t)
        x = (self.off + (self.offv + self.off * y) * t) * e
        self.off_t += 1.0 / self.fps
        if self.cur is None:
            self.out_env = float((1 + y * t) * e)
        if e * (1 + y * t) < 1e-3:
            self.off = None
            if self.cur is None:
                self.out_env = 0.0
            return tgt
        res = tgt.copy()
        rot = x[:132].reshape(44, 3)
        res[_TRI] = _q2aa(_qmul(_aa2q(rot), _aa2q(tgt[_TRI])))
        res[3:6] = tgt[3:6] + x[132:]
        return res
