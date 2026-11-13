"""Compositor tests: an action owning arm_r for 2 s then releasing -> no pop (bounded jerk) and the
face / gaze unaffected; lip sync takes the mouth with a crossfade; the page's affine map round-trips.

    python -I -m pytest -q retarget/tests/test_compositor.py      (from motion-model/)
Uses unified.runtime's UnifiedPlanner (copy-delay stub when no bundle) if importable, else a local stub with the API.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
MODEL_WT = os.environ.get("MODEL_CODE", REPO)     # checkout providing unified.runtime

from canon1 import spec  # noqa: E402
from retarget.retarget import Retargeter, RIG_GROUP  # noqa: E402
from retarget.compositor import Compositor  # noqa: E402

FPS = 30
CANON = os.environ.get("TEST_CANON_NPZ", os.path.join(os.environ.get("MM_DATA_DIR", os.path.join(REPO, "data")),
                                                      "canon1", "seamless", "V00_S2090_I00001011.npz"))


def planner():
    try:
        sys.path.insert(0, MODEL_WT)
        from unified.runtime import UnifiedPlanner
        return UnifiedPlanner(None, threads=1)
    except Exception:  # noqa: BLE001
        return _Stub()


class _Stub:
    """the runtime API with a copy-the-user core and linear override weights (same semantics as the MODEL stub)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.own = np.zeros(9); self.tgt = np.zeros(9); self.rate = np.zeros(9); self.ov = [None] * 9; self.t = 0
        self.last = None

    def set_override(self, groups, values=None, fade_in_s=0.15):
        for g in groups:
            i = spec.GROUP_NAMES.index(g)
            self.ov[i] = values; self.tgt[i] = 1; self.rate[i] = 1 / max(1, fade_in_s * FPS)

    def release_override(self, groups=None, fade_out_s=0.4):
        for g in groups or spec.GROUP_NAMES:
            i = spec.GROUP_NAMES.index(g)
            if self.ov[i] is not None and callable(self.ov[i]):
                v = self.ov[i](0); self.ov[i] = (lambda k, v=v: v)
            self.tgt[i] = 0; self.rate[i] = 1 / max(1, fade_out_s * FPS)

    def step(self, um, uk, ua, sa):
        self.own = np.where(self.tgt > self.own, np.minimum(self.own + self.rate, 1), np.maximum(self.own - self.rate, 0))
        m = spec.mirror_motion(um).astype(np.float32)
        for i, g in enumerate(spec.GROUP_NAMES):
            if self.own[i] > 0 and self.ov[i] is not None:
                c = spec.GROUPS[g]
                v = self.ov[i](0) if callable(self.ov[i]) else self.ov[i]
                m[c] = (1 - self.own[i]) * m[c] + self.own[i] * np.asarray(v)[c]
        self.t += 1
        return {"motion": m, "owned": self.own.copy(), "plan_ms": None}


def user_clip(n):
    if not os.path.exists(CANON):          # no Seamless canon data: a smooth synthetic user clip
        t = np.arange(n, dtype=np.float32)[:, None] / FPS
        m = (0.05 * np.sin(2 * np.pi * 0.3 * t + np.arange(spec.D_MOTION)[None] * 0.1)).astype(np.float32)
        return m, np.ones((n, spec.D_MASK), np.uint8)
    d = np.load(CANON, allow_pickle=False)
    return d["motion"][1, 1800:1800 + n].astype(np.float32), d["mask"][1, 1800:1800 + n]


ARM_CH = ["hand_l_x", "hand_l_y", "hand_l_z", "hand_l_turn", "hand_l_open"]    # screen-left paw = canon arm_r
FACE_CH = ["brow_l", "brow_r", "brow_knit", "blink_l", "blink_r", "eye_wide", "squint", "smile", "gaze_x", "gaze_y"]


def run(action_fn, n=240, talk=None):
    um, uk = user_clip(n)
    p = planner()
    comp = Compositor(Retargeter("muse"), planner=p)
    rows, owned = [], []
    for k in range(n):
        out = p.step(um[k], uk[k], None, None)
        r = comp.step(out["motion"], talking=bool(talk[k]) if talk is not None else False, action=action_fn(k))
        rows.append(r); owned.append(out["owned"])
    names = list(rows[0])
    return {nm: np.array([r[nm] for r in rows]) for nm in names}, np.array(owned), comp


def wave(k):
    """an action that OWNS the screen-left arm (canon arm_r) from 1.5 s to 3.5 s, starting and stopping ABRUPTLY
    (no easing of its own: the compositor must fade it in and out)."""
    if 45 <= k < 105:
        ph = np.sin(2 * np.pi * 2.5 * k / FPS)
        v = {"hand_l_x": 0.2 + 0.12 * ph, "hand_l_y": 1.0, "hand_l_z": 0.2, "hand_l_turn": 1.0, "hand_l_open": 1.0}
        return {"values": v, "weights": {c: 1.0 for c in v}}
    return None


def jerk(x):
    return np.abs(np.diff(x, 3)) * FPS ** 3


def test_action_release_no_pop_and_face_untouched():
    base, own0, _ = run(lambda k: None)
    act, own1, comp = run(wave)
    # ownership reached the planner and was released
    gi = spec.GROUP_NAMES.index("arm_r")
    assert own1[:, gi].max() > 0.99, "planner never owned arm_r"
    assert own1[-1, gi] == 0.0, "arm_r not released"
    ev = [e[0] for e in comp.events]
    assert "set" in ev and "release" in ev, comp.events
    # face / gaze untouched by the action (the model keeps them)
    for c in FACE_CH:
        assert np.allclose(base[c], act[c], atol=1e-6), c
    # the action pose is reached while it holds
    assert abs(act["hand_l_y"][80] - 1.0) < 1e-3
    # no pop: during the fade in (45..) and the release (105..) the per-frame step stays a fraction of the travel,
    # and the jerk is bounded by a min-jerk ramp's own (+ the underlying motion's)
    for c in ARM_CH:
        win = slice(100, 150)
        step = np.abs(np.diff(act[c]))[win].max()
        bstep = np.abs(np.diff(base[c]))[40:].max()           # (the copy-delay stub pops once at its 1 s delay)
        travel = np.ptp(act[c][95:160]) + 1e-6
        jb = jerk(base[c])[40:].max()
        ja_in = jerk(act[c])[38:62].max()
        ja = jerk(act[c])[100:150].max()
        tin = np.ptp(act[c][40:60]) + 1e-6
        mj_in = 60 * tin / 0.15 ** 3               # peak jerk of the 0.15 s min-jerk fade-in over its travel
        mj_bound = 60 * travel / 0.5 ** 3          # ... of the 0.5 s release crossfade
        assert ja_in < 2.5 * (mj_in + jb) + 50, (c, "in", ja_in, mj_in, jb)
        print(f"{c}: release max step {step:.3f} (base {bstep:.3f}) travel {travel:.3f} jerk {ja:.0f} (base {jb:.0f}, min-jerk bound {mj_bound:.0f}); fade-in jerk {ja_in:.0f} (bound {mj_in:.0f})")
        assert step < max(0.35 * travel, 1.5 * bstep) + 0.02, (c, step, travel, bstep)
        assert ja < 2.5 * (mj_bound + jb) + 50, (c, ja, mj_bound, jb)
    # after the release the arm is the model's again
    for c in ("hand_l_y",):
        assert np.allclose(act[c][200:], base[c][200:], atol=0.05), c


def test_lipsync_owns_mouth_with_crossfade():
    n = 150
    talk = np.zeros(n, bool); talk[30:90] = True
    um, uk = user_clip(n)
    rt = Retargeter("muse")
    comp = Compositor(rt, planner=None)
    m = np.tile(rt.neutral, (n, 1)); m[:, 135 + spec.BS["jawOpen"]] = 0.5           # the model opens its jaw
    mo = np.array([comp.step(m[k], talking=bool(talk[k]))["mouth_open"] for k in range(n)])
    assert mo[:30].min() > 0.5                         # silent: the model's mouth
    assert mo[45:90].max() < 1e-6                      # talking: lip sync owns it (mouth_open 0)
    assert np.abs(np.diff(mo)).max() < 0.5 * mo[0]     # crossfades over several frames, no one-frame snap
    assert mo[-1] > 0.5                                # comes back after the turn


def test_affine_roundtrip():
    names = ["head_yaw", "hand_l_y", "smile"]
    a = Compositor.from_affine(names, [(0, 1.0, 0.1), (1, 0.25, 0.6), (2, 0.0, 0.7)])
    assert a["add"] == {"head_yaw": 0.1}
    assert abs(a["weights"]["hand_l_y"] - 0.75) < 1e-9 and abs(a["values"]["hand_l_y"] - 0.8) < 1e-9
    assert a["weights"]["smile"] == 1.0 and abs(a["values"]["smile"] - 0.7) < 1e-9
    assert RIG_GROUP["hand_l_y"] == "arm_r" and RIG_GROUP["thumb_r"] == "fingers_l"


if __name__ == "__main__":
    test_action_release_no_pop_and_face_untouched(); test_lipsync_owns_mouth_with_crossfade(); test_affine_roundtrip()
    print("ok")
