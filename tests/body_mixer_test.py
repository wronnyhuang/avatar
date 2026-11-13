"""body: RigMixer PERIODIC columns (docs/INTENT-RULES.md section 1) — unwrap on push, rebase, shortest-arc fade.

A page sends rig vectors at 30 Hz stamped on the playback timeline; the renderer calls mixer.apply(zeros, tj, now) at
24 fps.  Plain script: prints PASS/FAIL lines, exit 0 = all pass.

    python3 tests/body_mixer_test.py
"""
import math
import os
import subprocess
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "renderer"))

from anim import rig as R  # noqa: E402
from anim.schema import IDX, NPARAMS  # noqa: E402

ORIG_COMMIT = "d61971e"          # anim/rig.py before the periodic mixer (main after the 11-07 schema change)
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def wrap(x):
    x = np.asarray(x, "f8")
    return x - np.floor(x + 0.5)


def check_tracking(name, y, tjs, rate, extra_lag_s, spread_s=0.004):
    """The output follows the true angle rate*tj mod 1 with a steady lag: the spring's ramp lag (~27 ms for the
    discrete SMOOTH_S step at 24 fps) + extra_lag_s (frames ahead of the newest stamp; then the target is the newest
    30 Hz vector, so the lag also wobbles by up to one packet interval: spread_s)."""
    settle = tjs > 0.6
    lag = wrap(rate*tjs[settle] - y[settle])/rate - extra_lag_s          # seconds, per frame
    ok = lag.max() - lag.min() < spread_s and 0.0 <= lag.mean() < 0.08
    check(name, ok, f"lag {lag.mean()*1000:.1f} ms (spread {(lag.max() - lag.min())*1000:.2f} ms) + "
                    f"{extra_lag_s*1000:.0f} ms ahead")


def simulate(send, T, cols, mixer=None, rig_hz=30.0, fps=24.0, latency=0.03, frame_off=-0.10, stop_at=None,
             pose_at=None, mod=R):
    """send(t) -> list of values for `cols` (or None = nothing sent at t).  Rig vector stamped t arrives at now = t +
    latency; a frame at now renders tj = now + frame_off (negative: inside the stamped history -> interpolation).
    Returns (frame times tj, outputs [frames x NPARAMS])."""
    mix = mixer or mod.RigMixer()
    ev = []
    k = 0
    while k/rig_hz <= T:
        t = k/rig_hz
        if stop_at is None or t < stop_at:
            ev.append((t + latency, 0, t))
        k += 1
    f = 0
    while f/fps <= T:
        ev.append((f/fps, 1, None))
        f += 1
    for p_now, p in (pose_at or []):
        ev.append((p_now, 2, p))
    ev.sort(key=lambda e: (e[0], e[1]))
    zeros = np.zeros(NPARAMS, "f4")
    tjs, outs = [], []
    for now, kind, arg in ev:
        if kind == 0:
            v = send(arg)
            if v is not None:
                mix.push_rig(arg, v, now, names=cols)
        elif kind == 2:
            mix.push_pose(arg, now)
        else:
            tj = now + frame_off
            tjs.append(tj)
            outs.append(np.array(mix.apply(zeros, tj, now), "f8"))
    return np.array(tjs), np.array(outs)


# ------------------------------------------------------------------------------------------------ (a) / (f) spin
def spin_case(tag, col, rate=1.5, T=4.0):
    for mode, off in (("interp", -0.10), ("ahead", +0.15)):
        tjs, out = simulate(lambda t: [float(wrap(rate*t))], T, [col], frame_off=off)
        y = out[:, IDX[col]]
        raw = np.abs(np.diff(y))
        wd = np.abs(wrap(np.diff(y)))
        lim = rate/24*1.3 + 0.01
        check(f"({tag}) {col} spin {rate}/s sent wrapped, {mode}: raw output continuous", raw.max() < lim,
              f"max |d| {raw.max():.4f} < {lim:.4f}")
        check(f"({tag}) {col} spin, {mode}: cos/sin continuous", wd.max() < lim, f"max wrapped |d| {wd.max():.4f}")
        # interp: the target is the true angle at tj.  ahead: the newest stamp, latency + offset behind the frame
        check_tracking(f"({tag}) {col} spin, {mode}: tracks true angle mod 1 with a small steady lag", y, tjs, rate,
                       0.0 if off < 0 else off + 0.03, 0.004 if off < 0 else 1/30 + 0.004)


# ------------------------------------------------------------------------------------------------ (b) whole-turn jump
def jump_case():
    for J in (+1.0, -1.0, +3.0):
        tjs, out = simulate(lambda t: [0.2 + (J if t >= 1.0 else 0.0)], 2.5, ["root_yaw"])
        y = out[:, IDX["root_yaw"]]
        late = tjs > 0.4                      # past the fade-in
        d = np.abs(np.diff(y[late]))
        c = np.abs(np.diff(np.cos(2*np.pi*y[late]))).max() + np.abs(np.diff(np.sin(2*np.pi*y[late]))).max()
        check(f"(b) jump of {J:+.0f} turn(s) in the sent value: no visible change", d.max() < 1e-5 and c < 1e-5,
              f"max |d| {d.max():.2e}, cos/sin {c:.2e}, out {y[-1]:.4f}")


# ------------------------------------------------------------------------------------------------ (c) short way
def short_way_case():
    tjs, out = simulate(lambda t: [0.49 if t < 1.0 else -0.49], 2.0, ["root_yaw"])
    y = out[:, IDX["root_yaw"]]
    late = tjs > 0.4
    travel = np.abs(np.diff(y[late])).sum()
    check("(c) 0.49 -> -0.49 moves the short way (0.02 turns)", abs(travel - 0.02) < 1e-3 and
          abs(wrap(y[-1] - (-0.49))) < 1e-4, f"travel {travel:.4f}, final {y[-1]:.4f} = {float(wrap(y[-1])):.4f} mod 1")


# ------------------------------------------------------------------------------------------------ (d) long spin
def long_spin_case():
    rate, T = 3.0, 7.0                   # ~20 turns
    for col in ("root_yaw", "prop3_rz"):
        mix = R.RigMixer()
        tjs, out = simulate(lambda t: [float(wrap(rate*t))], T, [col], mixer=mix)
        y = out[:, IDX[col]]
        wd = np.abs(wrap(np.diff(y)))
        lim = rate/24*1.3 + 0.01
        rebases = int((np.abs(np.diff(y)) > 0.5).sum())
        hist_max = max(abs(float(r[IDX[col]])) for _, r in mix.hist)
        check(f"(d) {col} {rate*T:.0f}-turn spin stays bounded", np.abs(y).max() < 10 and hist_max < 10,
              f"max |out| {np.abs(y).max():.3f}, max |hist| {hist_max:.3f}, rebases seen {rebases}")
        check(f"(d) {col} long spin rebased at least once", rebases >= 1, f"{rebases}")
        check(f"(d) {col} cos/sin continuous across the rebase", wd.max() < lim, f"max wrapped |d| {wd.max():.4f}")
        check_tracking(f"(d) {col} long spin tracks true angle mod 1 across rebases", y, tjs, rate, 0.0)


# ------------------------------------------------------------------------------------------------ (e) fades
def fade_case():
    # first vector root_yaw = 0.5 for 1 s, then silence: fade in (0.2 s), hold, 2 s timeout, 0.5 s fade out
    for v0 in (0.5, 3.0, -0.3):
        tjs, out = simulate(lambda t: [v0], 4.5, ["root_yaw"], stop_at=1.0)
        y = out[:, IDX["root_yaw"]]
        wd = np.abs(wrap(np.diff(y)))
        travel = wd.sum()
        expect = 2*abs(float(wrap(v0)))      # in and back out, shortest arc
        step = max(abs(float(wrap(v0)))/(R.FADE_IN_S*24), 1e-9)*1.15 + 0.01
        check(f"(e) fade-in/out with root_yaw {v0}: shortest arc, no extra turns",
              abs(travel - expect) < 0.01 and wd.max() < step and abs(y[-1]) < 1e-6,
              f"travel {travel:.4f} (expect {expect:.2f}), max step {wd.max():.4f} < {step:.4f}, end {y[-1]:.4f}")
    # an unwrapped 2.3 (after a spin) fading out to the server's 0, then fading in again from the stale history
    rate = 1.0

    def send(t):
        if t < 2.3:
            return [float(wrap(rate*t))]
        if t < 3.0:
            return [0.3]
        if 6.0 <= t < 7.0:
            return [0.35]
        return None
    mix = R.RigMixer()
    tjs, out = simulate(send, 8.0, ["root_yaw"], mixer=mix)
    y = out[:, IDX["root_yaw"]]
    wd = np.abs(wrap(np.diff(y)))
    fo = (tjs > 4.9) & (tjs < 5.7)            # fade-out window (last vector ~3.0 s + 2 s timeout)
    fi = (tjs > 5.9) & (tjs < 6.6)
    t_fo = wd[fo[1:]].sum()
    t_fi = wd[fi[1:]].sum()
    check("(e) fade-out from an unwrapped 2.3 goes 0.3 turns, not 2.3", abs(t_fo - 0.3) < 0.01, f"travel {t_fo:.4f}")
    check("(e) fade-in again from stale history goes 0.35 turns", abs(t_fi - 0.35) < 0.01 and
          abs(wrap(y[-1] - 0.35)) < 1e-4, f"travel {t_fi:.4f}, final {y[-1]:.4f}")
    lim = max(rate/24, 0.35/(R.FADE_IN_S*24))*1.15 + 0.01        # spin step / the 0.35-turn fade-in step
    check("(e) fades: per-frame change small", wd.max() < lim, f"max wrapped |d| {wd.max():.4f} < {lim:.4f}")


# ------------------------------------------------------------------------------------------------ pose path
def pose_case():
    m = R.RigMixer()
    p = m.push_pose({"root_yaw": 3.25, "prop0_ry": -0.75, "root_x": 9.0, "head_yaw": -4.0}, now=0.0)
    ok = (abs(p["root_yaw"] - 0.25) < 1e-9 and abs(p["prop0_ry"] - 0.25) < 1e-9 and p["root_x"] == 3.0
          and p["head_yaw"] == -0.8)
    check("pose path: periodic wrapped into [-0.5, 0.5), others clipped to RANGES", ok, str(p))
    out = m.apply(np.zeros(NPARAMS, "f4"), 0.1, 0.1)
    check("pose path: apply shows the wrapped periodic value", abs(out[IDX["root_yaw"]] - 0.25) < 1e-6,
          f"{out[IDX['root_yaw']]:.4f}")


# ------------------------------------------------------------------------------------------------ (g) non-periodic
def load_original():
    try:
        src = subprocess.run(["git", "-C", ROOT, "show", f"{ORIG_COMMIT}:renderer/anim/rig.py"], capture_output=True,
                             text=True, check=True).stdout
    except Exception as e:  # noqa: BLE001
        return None, str(e)
    m = types.ModuleType("anim._rig_orig")
    m.__package__ = "anim"
    m.__file__ = os.path.join(ROOT, "renderer", "anim", "_rig_orig.py")
    sys.modules[m.__name__] = m
    exec(compile(src, m.__file__, "exec"), m.__dict__)
    return m, ""


def nonperiodic_case():
    names = list(R.RIG_PARAMS)
    per = np.array([n in R.PERIODIC for n in names])
    rng = np.random.default_rng(7)
    lo = np.array([R.RANGES[n][0] for n in names])
    hi = np.array([R.RANGES[n][1] for n in names])
    ph = rng.uniform(0, 2*np.pi, len(names))
    fr = rng.uniform(0.2, 1.5, len(names))

    def vec(t):
        if 2.5 <= t < 5.3:                  # silence: timeout + fade-out, then a fade-in
            return None
        mid, amp = (lo + hi)/2, (hi - lo)/2
        v = mid + 1.3*amp*np.sin(fr*t + ph)          # overshoots the ranges: clip must apply
        v[per] = 0.2*np.sin(fr[per]*t + ph[per])     # small: every step (gaps included) < 0.5 turn -> no unwrap
        return v.tolist()

    poses = [(1.0, {"smile": 0.7, "root_x": 7.0, "head_yaw": 2.0, "root_yaw": 0.2}), (6.0, {"brow_l": -3.0})]
    orig, err = load_original()
    if orig is None:
        print(f"SKIP (g) original-algorithm comparison: cannot load {ORIG_COMMIT}: {err}")
    else:
        # the same packet sequence + out-of-order / duplicate stamps + a timeline restart, through both mixers
        outs = []
        for mod in (R, orig):
            mix = mod.RigMixer()
            _, o = simulate(vec, 7.0, names, mixer=mix, pose_at=poses, mod=mod)
            extra = []
            for (t, now) in ((7.02, 7.05), (6.99, 7.06), (7.02, 7.07), (0.5, 7.08), (0.53, 7.12)):
                mix.push_rig(t, vec(min(t, 2.0)), now, names=names)
                extra.append(np.array(mix.apply(np.zeros(NPARAMS, "f4"), t + 0.01, now + 0.01), "f8"))
            outs.append(np.concatenate([o, np.array(extra)]))
        new, old = outs
        # integ: compare only columns whose range + periodicity the original already had (widened cam_zoom
        # to -1..1 and appended prop8-15; those are new behaviour by design, not a mixer change)
        from anim.schema import PARAMS as _P
        same = [i for i, n in enumerate(_P) if R.RANGES.get(n) == orig.RANGES.get(n)
                and (n in R.PERIODIC) == (n in orig.PERIODIC)]
        print(f"  (g) {NPARAMS - len(same)} columns changed by design excluded")
        npi = np.array([i for i in same if i not in set(R.PER_IDX.tolist())])
        pci = np.array([i for i in same if i in set(R.PER_IDX.tolist())])
        check("(g) non-periodic columns byte-identical to the original mixer", np.array_equal(new[:, npi], old[:, npi]),
              f"{len(new)} frames, max |diff| {np.abs(new[:, npi] - old[:, npi]).max():.2e}")
        check("(g) small continuous periodic values also identical", np.array_equal(new[:, pci], old[:, pci]))
    # clip to RANGES still applies to root_x / head_yaw, never to root_yaw
    tjs, out = simulate(lambda t: [5.0, 2.0, 2.75], 1.0, ["root_x", "head_yaw", "root_yaw"])
    rx, hy, ry = out[-1, IDX["root_x"]], out[-1, IDX["head_yaw"]], out[-1, IDX["root_yaw"]]
    check("(g) root_x / head_yaw still clipped to RANGES; root_yaw wrapped, not clipped",
          abs(rx - 3.0) < 1e-6 and abs(hy - 0.8) < 1e-6 and abs(ry - (-0.25)) < 1e-6,
          f"root_x {rx:.4f}, head_yaw {hy:.4f}, root_yaw {ry:.4f}")


if __name__ == "__main__":
    spin_case("a", "root_yaw")
    spin_case("f", "prop0_ry")
    jump_case()
    short_way_case()
    long_spin_case()
    fade_case()
    pose_case()
    nonperiodic_case()
    print(f"{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED'}")
    sys.exit(1 if FAILS else 0)
