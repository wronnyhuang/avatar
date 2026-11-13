"""unit tests of app/react_worker.py (learned-reaction mode; SKIPs without a learned-reaction checkout):
   python3 -I tests/test_react_worker.py [--root REACT_ROOT]
Streaming audio features == spec.audio_features (same 16 kHz samples), StreamingVAD == spec.energy_vad, euler <->
rotation round trip + slerp, blink events (no half blinks), crossfade only on disagreement, in-filling keeps the
committed frames, the affine merge, a full engine run with the stub planner (delay copy shows up delay later)."""
import argparse
import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.environ.get("REACT_ROOT", ""), help="learned-reaction checkout (react/spec.py)")
A = ap.parse_args()
if not A.root or not os.path.isfile(os.path.join(A.root, "react", "spec.py")):
    print("SKIP: needs a learned-reaction checkout (--root or REACT_ROOT, with react/spec.py); not shipped with this repo")
    sys.exit(0)
sys.path[:0] = [os.path.join(HERE, "..", "app"), A.root]
from react import spec  # noqa: E402
import react_worker as W  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


rs = np.random.default_rng(0)
# --- audio features: speech-like bursts + noise, 16 kHz
T = 200
x = (0.01 * rs.standard_normal(int(T * spec.SR / spec.FPS) + 1000)).astype(np.float32)
for k in range(5):
    a = int((0.6 + k * 1.2) * spec.SR); x[a:a + 8000] += 0.3 * np.sin(2 * np.pi * 220 * np.arange(8000) / spec.SR) * np.hanning(8000)
ref = spec.audio_features(x, n_frames=T)
fs = W.FeatStream(spec)
L = max(spec.N_FFT, fs.hop)
pad = np.concatenate([np.zeros(L, np.float32), x])
got = np.stack([fs.frame(pad[spec.frame_end_sample(k):spec.frame_end_sample(k) + L]) for k in range(T)])
d = np.abs(got - ref)
check("streaming audio features == spec.audio_features (mel, loud, vad)", d[:, :40].max() < 1e-3 and d[:, 40].max() < 1e-3 and (d[:, 41] == 0).all(),
      f"max mel {d[:, :40].max():.2e} loud {d[:, 40].max():.2e} vad mismatches {(d[:, 41] > 0).sum()}")
loud = rs.uniform(-80, -10, 700).astype(np.float32)
v = W.StreamingVAD(); sv = np.array([v(l) for l in loud])
check("StreamingVAD == spec.energy_vad", np.array_equal(sv, spec.energy_vad(loud)))

# --- AudioTrack placement via the timeline
tr = W.AudioTrack(16000, 2.0); tr.write(1000, np.ones(320)); tr.write(1320, 2 * np.ones(320))
r = tr.read(990, 1650)
check("AudioTrack writes/reads by absolute timeline sample", r[:10].sum() == 0 and r[10:330].mean() == 1 and r[330:650].mean() == 2 and r[650:].sum() == 0)

# --- rotations
M = W._rot_from_euler(0.3, -0.2, 0.15)
e = spec.euler_from_mat(np.concatenate([np.concatenate([M, np.zeros((3, 1))], 1), [[0, 0, 0, 1]]], 0).reshape(1, 16))[0][0]
check("euler -> rotation -> spec.euler_from_mat round trip", np.allclose(e, [0.3, -0.2, 0.15], atol=1e-9), str(np.round(e, 6)))
s0, s1 = W.slerp_euler((0.1, 0.0, 0.0), (0.3, 0.1, -0.1), 0.0), W.slerp_euler((0.1, 0.0, 0.0), (0.3, 0.1, -0.1), 1.0)
mid = W.slerp_euler((0.0, 0.0, 0.0), (0.4, 0.0, 0.0), 0.5)
check("slerp endpoints + midpoint", np.allclose(s0, (0.1, 0, 0), atol=1e-9) and np.allclose(s1, (0.3, 0.1, -0.1), atol=1e-9) and abs(mid[0] - 0.2) < 1e-9)

# --- blink events: a partial closure never shows; a crossing plays the whole template once
b = W.BlinkEvents()
outs = [b.step(c, c) for c in [0.1, 0.3, 0.45, 0.5, 0.4, 0.2]]
check("blink events: partial closure (< on) shows nothing", all(o == (0.0, 0.0) for o in outs))
outs = [b.step(c, c)[0] for c in [0.2, 0.7, 0.9, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]
check("blink events: crossing plays the full template", max(outs) == 1.0 and outs[1:9] == W.BlinkEvents.TEMPLATE, str(outs))
outs = [b.step(c, 0.05) for c in [0.1] * 10 + [0.8, 0.9, 0.2] + [0] * 8]
check("blink events: one-eyed closure -> wink on that side only", max(o[0] for o in outs) == 1.0 and max(o[1] for o in outs) == 0.0)


# --- engine with a scripted planner (crossfade / in-filling) and a fake retarget
class Script:
    H = 15

    def __init__(self):
        self.next = None; self.got = []

    def reset(self): pass
    def push(self, *a): pass
    def commit(self, own): pass

    def plan(self, committed):
        self.got.append(len(committed))
        out = np.repeat(self.next[None], 15, 0).copy()
        if len(committed):
            out[:len(committed)] = committed
        return out


rt = types.SimpleNamespace(RIG_COVERED={"fox": ["head_yaw", "smile", "blink_l", "blink_r"]},
                           human_to_rig=lambda f, c: {"head_yaw": float(f[52]), "smile": float(f[44]), "blink_l": float(f[10]), "blink_r": float(f[9])},
                           rig_to_human=lambda rig, c=None: (np.array([0] * 52 + [rig.get("head_yaw", 0)] + [0] * 33, np.float32), np.array([False] * 52 + ["head_yaw" in rig] + [False] * 33)))
args = types.SimpleNamespace(planner="stub", stub_delay=0.0, threads=1, plan_ms=33.0, commit_ahead=0, xfade=3, ear_springs=0, root=A.root, bundle="")
eng = W.Engine(args, spec, rt)
sc = Script(); eng.planner = sc
eng.rig_params = ["head_yaw", "smile", "blink_l", "blink_r", "breath"]
f0 = np.zeros(86, np.float32); sc.next = f0.copy()
page = lambda: (np.array([0.5, 0.5, 0.5, 0.5, 0.33], np.float32), [], 0.0)  # noqa: E731
fin = []
for j in range(10):
    fin += eng.step([((j + 0.02) / 30, *page())])
f1 = f0.copy(); f1[52] = 0.02; sc.next = f1                       # small change: no crossfade
for j in range(10, 14):
    fin += eng.step([((j + 0.02) / 30, *page())])
n_xf_small = eng.stats["crossfades"]
f2 = f0.copy(); f2[52] = 0.4; sc.next = f2                        # a jump: crossfade over 3 frames
ys = []
for j in range(14, 22):
    ys.append(float(eng.step([((j + 0.02) / 30, *page())])[0][0]))
check("no crossfade for a small change", n_xf_small == 0)
check("crossfade (3 frames, slerp) on a real jump", eng.stats["crossfades"] >= 1 and 0.02 < ys[1] < ys[2] < ys[3] < 0.4 and abs(ys[4] - 0.4) < 1e-6, str(np.round(ys, 3)))
check("uncovered channels keep the page's rules value", abs(fin[-1][4] - 0.33) < 1e-6)
# affine merge: a held pose (alpha 0, beta 0.25) on head_yaw, reflex blink floor
a = np.array([0.5, 0.5, 0.5, 0.5, 0.33], np.float32)
out = eng.step([((22 + 0.02) / 30, a, [(0, 0.0, 0.25), (1, 0.5, 0.1)], 0.8)])[0]
check("blinks=rules (default): the lids are the page's own blink value, untouched", eng.blinks == "rules" and abs(out[2] - 0.5) < 1e-6 and abs(out[3] - 0.5) < 1e-6, str(out))
eng.blinks = "learned"
out = eng.step([((23 + 0.02) / 30, a, [(0, 0.0, 0.25), (1, 0.5, 0.1)], 0.8)])[0]
check("affine merge: hold wins, alpha/beta mix, reflex blink floors the lids (blinks=learned)", abs(out[0] - 0.25) < 1e-6 and abs(out[1] - 0.1) < 1e-6 and abs(out[2] - 0.8) < 1e-6, str(out))
eng.blinks = "rules"
# in-filling: commit_ahead frames passed back as committed
eng.args.commit_ahead = 2
eng.step([((24 + 0.02) / 30, *page())])
check("in-filling: committed frames are handed to plan()", sc.got[-1] == 2)

# --- full engine with the stub planner (delay copy): a partner smile shows up `delay` later
args2 = types.SimpleNamespace(planner="stub", stub_delay=0.5, threads=1, plan_ms=33.0, commit_ahead=0, xfade=3, ear_springs=0, root=A.root, bundle="")
eng = W.Engine(args2, spec, rt); eng.rig_params = ["head_yaw", "smile", "blink_l", "blink_r", "breath"]
sm = []
for j in range(150):
    m = np.zeros(86, np.float32); m[44] = m[45] = 0.7 if j >= 60 else 0.05
    eng.add_human((j + 0.0) / 30, m, [1, 0, 0, 0])
    sm.append(float(eng.step([((j + 0.5) / 30, *page())])[0][1]))
on = next(i for i, v in enumerate(sm) if v > 0.4)
check("stub delay-copy: partner smile at tick 60 shows ~15 ticks later", 74 <= on <= 77, f"first > 0.4 at tick {on}")
# --- dither: a gated mic (exact zeros) with -68 dB hiss must not read as speech; real speech still does
def vad_frac(dither):
    tr = W.AudioTrack(16000, 12.0, dither_db=dither); fs = W.FeatStream(spec); g = np.random.default_rng(3)
    for k in range(0, 10 * 16000, 320):
        blk = np.zeros(320, np.float32) if (k // 16000) % 2 == 0 else (g.standard_normal(320) * 10 ** (-68 / 20)).astype(np.float32)
        if 6 * 16000 <= k < 7 * 16000:
            blk = (0.2 * np.sin(2 * np.pi * 200 * (k + np.arange(320)) / 16000)).astype(np.float32)
        tr.write(k, blk)
    L = max(spec.N_FFT, fs.hop); v = []
    for j in range(290):
        e = spec.frame_end_sample(j); v.append(fs.frame(tr.read(e - L, e))[41])
    v = np.array(v); return float(v[:180].mean()), float(v[182:208].mean())
nd, d70 = vad_frac(None), vad_frac(-70.0)
check("dither -70 dB: gated-mic zeros + -68 dB hiss stop reading as speech; a real voice still does",
      nd[0] > 0.3 and d70[0] < 0.05 and d70[1] > 0.9, f"no dither: hiss VAD {nd[0]:.2f}; dither: hiss {d70[0]:.2f}, voice {d70[1]:.2f}")
# --- session recorder: canon npz of what it saw/showed, loadable by spec.load_session
import tempfile, glob as _glob
_rd = tempfile.mkdtemp(prefix="react_rec_")
args3 = types.SimpleNamespace(planner="stub", stub_delay=0.5, threads=1, plan_ms=33.0, commit_ahead=0, xfade=3, ear_springs=0,
                              root=A.root, bundle="", record_dir=_rd)
eng = W.Engine(args3, spec, rt); eng.rig_params = ["head_yaw", "smile", "blink_l", "blink_r", "breath"]
for j in range(200):
    m = np.zeros(86, np.float32); m[44] = m[45] = 0.7 if j >= 60 else 0.05
    eng.add_human((j + 0.0) / 30, m, [1, 0, 0, 0])
    eng.step([((j + 0.5) / 30, *page())])
eng.reset()
_f = _glob.glob(_rd + "/*.npz")
_ok = len(_f) == 1
if _ok:
    _d = spec.load_session(_f[0]); _ok = _d["motion"].shape[:2] == (2, 200) and _d["meta"]["dataset"] == "user" and float(_d["motion"][1, 100, 44]) > 0.6
check("recorder: reset() writes one canon session (avatar + user numbers)", _ok, str(_f))
print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
