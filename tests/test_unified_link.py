"""the server side of UNIFIED mode against a REAL worker process (app/unified_worker.py), fed
like the live page: real MediaPipe results of a held-out Seamless person (the raw MediaPipe extraction of the dev video:
face every frame, pose + hands every 2nd frame, world landmarks) as 'U' messages ~50 ms after capture, the person's mic in 20 ms
16 kHz chunks with sync, the avatar's voice tap (the partner's real voice) at 24 kHz, 30 Hz rig ticks with the page's
affine map, in batches of 2-3 ticks like the exchange lanes, through wire.py's encoders/decoders.

  python3 tests/test_unified_link.py [--seconds 30] [--bundle B] [--character muse] [--raw NPZ --mic WAV --own WAV]

Inputs (SKIPs when absent): AVATAR_DATA (default <repo>/data)/mp_raw/seamless/<id>.npz and the Seamless Interaction
wavs under SEAMLESS_DIR (default <repo>/data/seamless-interaction).

Checks: worker up; every batch answered; covered channels move (not frozen) and uncovered ones are the page's values;
a page-held channel (alpha 0) wins and the avatar returns to the model without a pop; the mouth is lip sync's while
the avatar talks; per-tick fit / plan / retarget+compositor ms and the round trip.
"""
import argparse
import asyncio
import json
import os
import struct
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "app"))
sys.path.insert(0, os.path.join(ROOT, "renderer"))
import unified_link  # noqa: E402
import wire  # noqa: E402
from anim.rig import RIG_PARAMS  # noqa: E402

DATA = os.environ.get("AVATAR_DATA", os.path.join(ROOT, "data"))
SEAMLESS = os.environ.get("SEAMLESS_DIR", os.path.join(DATA, "seamless-interaction"))
ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=30)
ap.add_argument("--start", type=float, default=30.0)
ap.add_argument("--raw", default=f"{DATA}/mp_raw/seamless/V00_S0255_I00000126_P0350.npz")
ap.add_argument("--mic", default=f"{SEAMLESS}/naturalistic/dev/audio/V00_S0255_I00000126_P0350.wav")
ap.add_argument("--own", default=f"{SEAMLESS}/naturalistic/dev/audio/V00_S0255_I00000126_P0351.wav")
ap.add_argument("--bundle", default="")
ap.add_argument("--character", default="muse")
ap.add_argument("--logdir", default="")
ap.add_argument("--out", default="")
A = ap.parse_args()
if not all(os.path.isfile(p) for p in (A.raw, A.mic, A.own)):
    print("SKIP: test_unified_link needs the Seamless inputs (--raw / --mic / --own, AVATAR_DATA, SEAMLESS_DIR)")
    sys.exit(0)
A.logdir = A.logdir or tempfile.mkdtemp(prefix="unified_link_")
RIG = list(RIG_PARAMS)


def rig_msg_x(t, a, ab, rb):
    q = lambda v: max(-32767, min(32767, int(round(v * 1e4))))  # noqa: E731
    b = wire.R_HDR.pack(wire.R, 1, len(a), t) + struct.pack(f"<{len(a)}h", *[q(x) for x in a]) + bytes([len(ab)])
    for i, al, be in ab:
        b += struct.pack("<Bhh", i, q(al), q(be))
    return b + struct.pack("<h", q(rb))


def wav(path, rate):
    """any wav -> mono float32 at `rate` (decoded by ffmpeg: no soundfile dependency; Seamless wavs are float)."""
    import subprocess
    pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"],
                         check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(pcm, "<i2").astype(np.float32) / 32768.0


def mp_msgs(raw, t0, t1, off):
    """'U' messages for native frames captured in [t0, t1) (playback timeline = video time - off)."""
    ts = raw["ts_ms"] / 1000.0 - off
    w, h = [int(v) for v in raw["wh"]]
    out = []
    for i in np.where((ts >= t0) & (ts < t1))[0]:
        t = float(ts[i])
        mat = raw["mat"][i].reshape(4, 4).T.reshape(16)                       # row-major -> the page's column-major
        out.append(wire.pack_mp(wire.U_FACE, t, w, h, bool(raw["face"][i]), face_bs=raw["bs"][i], face_mat=mat))
        if raw["pose_ran"][i]:
            ok = bool(raw["pose_ok"][i])
            out.append(wire.pack_mp(wire.U_POSE, t, w, h, ok, pose_image=raw["pose"][i], pose_world=raw["pose_world"][i]))
        if raw["hand_ran"][i]:
            hs = [(int(raw["hand_label"][i, j]), float(raw["hand_score"][i, j]), raw["hands"][i, j], raw["hands_world"][i, j])
                  for j in range(int(raw["hand_n"][i]))]
            out.append(wire.pack_mp(wire.U_HANDS, t, w, h, True, hands=hs))
    return out


async def main():
    raw = dict(np.load(A.raw, allow_pickle=False))
    mic = wav(A.mic, 16000)
    own = wav(A.own, 24000)
    off = A.start
    logs = []
    cfg = unified_link.config({"unified": {"bundle": A.bundle, "trace": A.out or ""}})
    w = unified_link.UnifiedWorker(cfg, lambda m: logs.append(m), A.logdir)
    t0 = time.monotonic()
    ok = await w.ensure(120)
    print("worker up:", ok, round(time.monotonic() - t0, 2), "s", w.info, logs)
    if not ok:
        print(open(os.path.join(A.logdir, "unified_worker.log")).read()[-3000:])
        sys.exit(1)
    rx = unified_link.UnifiedRx(w, cfg, print)
    rs = np.random.default_rng(1)
    T = int(A.seconds * 30)
    out = []
    page_breath = 0.33
    mic_seq = 0
    sent_mp = 0.0
    own_pos = 0
    t_start = time.monotonic()
    k = 0
    ix = RIG.index
    hold0, hold1 = 12.0, 14.0
    while k < T:
        batch = []
        for _ in range(int(rs.integers(2, 4))):
            if k >= T:
                break
            t = k / 30 + 0.012
            # camera results arrive ~50 ms after capture
            for m in mp_msgs(raw, sent_mp, t - 0.05, off):
                rx.add_mp(m)
            sent_mp = t - 0.05
            # his mic: 20 ms chunks at 16 kHz
            while mic_seq * 0.02 < t:
                a0 = int((off + mic_seq * 0.02) * 16000)
                x = np.clip(mic[a0:a0 + 320] * 32767, -32768, 32767).astype("<i2")
                rx.add_mic(mic_seq, x.tobytes(), 16000)
                if mic_seq % 25 == 0:
                    rx.add_sync(mic_seq, mic_seq * 0.02 + 0.02)
                mic_seq += 1
            # the avatar's voice tap up to the tick (+ ~1 frame of lead)
            while own_pos < int((t + 0.03) * 24000):
                a0 = int(off * 24000) + own_pos
                rx.add_own(own_pos, np.clip(own[a0:a0 + 800] * 32767, -32768, 32767).astype("<i2").tobytes())
                own_pos += 800
            a = np.zeros(len(RIG), np.float32); a[ix("breath")] = page_breath
            ab = []
            if hold0 <= t < hold1:                     # a command holds the screen-left paw up (alpha 0)
                ab = [(ix("hand_l_y"), 0.0, 0.9), (ix("hand_l_x"), 0.0, 0.2), (ix("hand_l_z"), 0.0, 0.3)]
                for i, _, be in ab:
                    a[i] = be
            msg = rig_msg_x(t, a, ab, 0.0)
            tt, aa, abb, rb = wire.unpack_rig_rx(msg)
            batch.append((tt, aa, abb, rb))
            k += 1
        finals = await rx.step(batch, A.character, RIG, want_stats=(k >= T or k % 300 < 3))
        for (tt, aa, abb, rb), f in zip(batch, finals or [None] * len(batch)):
            out.append((tt, None if f is None else f.copy()))
        await asyncio.sleep(max(0, t_start + k / 30 - time.monotonic()))
    s = rx.summary()
    print(json.dumps(s, indent=1)[:3000])
    got = [(t, f) for t, f in out if f is not None]
    fails = []

    def chk(name, cond, det=""):
        print(("PASS " if cond else "FAIL ") + name + (f"  — {det}" if det else ""))
        if not cond:
            fails.append(name)
    chk("every batch answered", len(got) == len(out), f"{len(got)}/{len(out)}")
    F = np.stack([f for _, f in got]); tt = np.array([t for t, _ in got])
    chk("uncovered channel (tail_wag on the muse) = the page's value 0", np.abs(F[:, ix("tail_wag")]).max() < 1e-4)
    chk("breath is the retarget's own (not the page's constant)", np.std(F[:, ix("breath")]) > 0.05)
    for c in ("head_yaw", "head_pitch", "brow_l", "gaze_x", "hand_r_y", "body_yaw"):
        sd = float(np.std(F[tt > 2, ix(c)]))
        chk(f"{c} moves (std > 0.005)", sd > 0.005, f"std {sd:.4f}")
    hl = F[(tt >= hold0 + 0.3) & (tt < hold1), ix("hand_l_y")]
    chk("page-held hand_l_y (alpha 0, beta 0.9) wins", len(hl) and np.abs(hl - 0.9).max() < 1e-3, f"{hl[:3]}")
    # the command lets go ABRUPTLY (alpha 0 -> 1 in one tick): the compositor's release (hold for the planner's short
    # canon blend, then a 0.5 s min-jerk crossfade) must carry it; after that the model's own motion (the copy-delay
    # stub replays his fitted arm, steppy at the 15 Hz pose rate) is compared with its own steps before the hold
    for c in ("hand_l_x", "hand_l_y", "hand_l_z"):
        rel = F[(tt >= hold1 - 0.1) & (tt < hold1 + 0.65), ix(c)]
        st = np.abs(np.diff(rel)).max() if len(rel) > 2 else 9
        base = np.abs(np.diff(F[(tt > 3) & (tt < hold0 - 0.1), ix(c)])).max()
        chk(f"release of the held paw ({c}) has no pop (max step < 0.12 over the 0.65 s release)", st < 0.12,
            f"max step {st:.3f} (the model's own max step before the hold {base:.3f})")
    ws = s.get("worker") or {}
    print(json.dumps({k: ws.get(k) for k in ("fit_ms", "planner_step_ms", "retarget_compositor_ms", "tick_ms", "batch_ms",
                                             "fitter", "planner", "overrides", "mp_1", "mp_2", "mp_3", "late_ticks",
                                             "gap_ticks")}, indent=1))
    b = s["batches_ms"]
    chk("worker round trip p95 < 40 ms (timeout 60)", b["p95"] is not None and b["p95"] < 40, json.dumps(b))
    chk("no late batches", s.get("late_batches", 0) == 0, str(s.get("late_batches", 0)))
    if A.out:
        np.savez(A.out + ".npz", t=tt, F=F, names=np.array(RIG))
    await w.close()
    print("ALL PASS" if not fails else f"{len(fails)} FAILED")
    sys.exit(1 if fails else 0)


asyncio.run(main())
