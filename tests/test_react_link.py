"""the server side of learned mode against a REAL worker process (the worker runs with react.python, importing a
learned-reaction checkout: --root / REACT_ROOT; SKIPs without one). Simulates the page's uplink at the real cadence (30 Hz rig ticks, 18 Hz 'H' frames, 20 ms mic chunks, the
avatar's voice tap) in batches of 2-3 ticks like the 3-lane exchange, through wire.py's encoders/decoders.
  python3 tests/test_react_link.py [--root REACT_ROOT] [--seconds 20] [--delay 0.5] [--out FILE]
Checks: worker comes up, every batch answered within the timeout, uncovered channels = the page's values, the partner's
smile / head turn reach the avatar after the stub delay, round-trip p50/p95."""
import argparse
import asyncio
import json
import math
import os
import struct
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
import react_link  # noqa: E402
import wire  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=20)
ap.add_argument("--delay", type=float, default=0.5)
ap.add_argument("--planner", default="stub")
ap.add_argument("--bundle", default="")
ap.add_argument("--out", default="")
ap.add_argument("--root", default=os.environ.get("REACT_ROOT", ""), help="learned-reaction checkout (react/spec.py)")
A = ap.parse_args()
if not A.root or not os.path.isfile(os.path.join(A.root, "react", "spec.py")):
    print("SKIP: needs a learned-reaction checkout (--root or REACT_ROOT, with react/spec.py); not shipped with this repo")
    sys.exit(0)

RIG = ["smile", "head_yaw", "head_pitch", "head_roll", "head_x", "head_y", "head_z", "breath", "blink_l", "blink_r",
       "gaze_x", "gaze_y", "brow_l", "brow_r", "brow_knit", "ear_l", "ear_r", "ear_flick_l", "ear_flick_r", "nostril",
       "point_l", "point_r", "body_lean", "body_bounce", "tail_wag", "tail_raise", "body_yaw", "hand_l_x", "hand_l_y",
       "hand_l_z", "hand_r_x", "hand_r_y", "hand_r_z", "hand_l_open", "hand_r_open", "hand_l_turn", "hand_r_turn",
       "thumb_l", "thumb_r", "shoulder_l", "shoulder_r", "ear_l_turn", "ear_r_turn", "nose_wiggle", "tongue_out",
       "mouth_open", "eye_wide", "squint", "body_pitch", "tail_curl"]


def rig_msg_x(t, a, ab, rb):
    q = lambda v: max(-32767, min(32767, int(round(v * 1e4))))  # noqa: E731
    b = wire.R_HDR.pack(wire.R, 1, len(a), t) + struct.pack(f"<{len(a)}h", *[q(x) for x in a]) + bytes([len(ab)])
    for i, al, be in ab:
        b += struct.pack("<Bhh", i, q(al), q(be))
    return b + struct.pack("<h", q(rb))


async def main():
    logs = []
    cfg = dict(react_link.DEFAULTS, root=A.root, planner=A.planner, bundle=A.bundle, stub_delay_s=A.delay, trace=A.out or "", stub_mirror=0)   # sign check: as-is copy
    w = react_link.Worker(cfg, lambda m: logs.append(m), tempfile.mkdtemp(prefix="react_link_"))
    t0 = time.monotonic()
    ok = await w.ensure(60)
    print("worker up:", ok, round(time.monotonic() - t0, 2), "s", w.info, logs)
    rx = react_link.SessionRx(w, cfg, print)
    rs = np.random.default_rng(1)
    T = int(A.seconds * 30)
    out = []
    page_breath = 0.33
    mic_seq = 0
    t_start = time.monotonic()
    k = 0
    while k < T:
        batch = []
        for _ in range(int(rs.integers(2, 4))):
            if k >= T:
                break
            t = k / 30 + 0.012
            # partner: smile on 4-8 s, head turned to image-right 10-13 s
            if k % 2 == 0 or k % 5 == 0:
                m = np.zeros(86, np.float32); m[9] = m[10] = 0.12
                m[44] = m[45] = 0.7 if 4 <= t < 8 else 0.05
                m[52] = 0.3 if 10 <= t < 13 else 0.0
                m[55:58] = [1.0, 2.0, -45.0]
                msg = wire.pack_human(t - 0.06, m, [1, 1, 0, 0])
                tt, v, mk = wire.unpack_human(msg)
                rx.add_human(tt, v, msg[1], len(msg))
            # mic: 20 ms chunks at 16 kHz (speech 2-4 s)
            for _ in range(2 if k % 3 else 1):
                x = (0.2 * np.sin(2 * np.pi * 180 * np.arange(320) / 16000) * 32767).astype("<i2") if 2 <= t < 4 else np.zeros(320, "<i2")
                rx.add_mic(mic_seq, x.tobytes(), 16000)
                if mic_seq % 25 == 0:
                    rx.add_sync(mic_seq, mic_seq * 0.02 + 0.02)
                mic_seq += 1
            # the avatar's own voice tap (24 kHz) 6-9 s
            if 6 <= t < 9:
                rx.add_own(int(t * 24000), (0.1 * np.sin(2 * np.pi * 150 * np.arange(800) / 24000) * 32767).astype("<i2").tobytes())
            a = np.zeros(len(RIG), np.float32); a[RIG.index("breath")] = page_breath
            ab = [(RIG.index("hand_l_y"), 0.0, 0.6)] if 15 <= t < 16 else []
            if ab:
                a[RIG.index("hand_l_y")] = 0.6          # the page's rules vector carries its own hold too
            msg = rig_msg_x(t, a, ab, 0.0)
            tt, aa, abb, rb = wire.unpack_rig_rx(msg)
            batch.append((tt, aa, abb, rb))
            k += 1
        finals = await rx.step(batch, "fox", RIG, want_stats=(k >= T))
        for (tt, aa, abb, rb), f in zip(batch, finals or [None] * len(batch)):
            out.append((tt, None if f is None else f.copy()))
        await asyncio.sleep(max(0, t_start + k / 30 - time.monotonic()))
    s = rx.summary()
    print(json.dumps(s, indent=1)[:1800])
    got = [(t, f) for t, f in out if f is not None]
    ix = RIG.index
    fails = []

    def chk(name, cond, det=""):
        print(("PASS " if cond else "FAIL ") + name + (f"  — {det}" if det else ""))
        if not cond:
            fails.append(name)
    chk("every batch answered", len(got) == len(out), f"{len(got)}/{len(out)}")
    chk("uncovered channel (breath) = the page's value", all(abs(f[ix("breath")] - page_breath) < 1e-4 for _, f in got))
    sm = [(t, f[ix("smile")]) for t, f in got]
    on = next((t for t, v in sm if t > 4 and v > 0.5 * max(v for _, v in sm)), None)
    chk("partner smile at 4.0 s (captured 3.94) reaches the avatar ~delay later", on is not None and abs(on - (3.94 + A.delay)) < 0.2, f"50% at {on}")
    yw = [(t, f[ix("head_yaw")]) for t, f in got]
    on = next((t for t, v in yw if t > 10 and v > 0.15), None)
    chk("partner head turn to image-right -> head_yaw > 0 after the delay", on is not None and abs(on - (9.94 + A.delay)) < 0.25, f"at {on}")
    hl = [f[ix("hand_l_y")] for t, f in got if 15.1 <= t < 15.9]
    chk("page-held hand_l_y (alpha 0, beta 0.6) wins", hl and all(abs(v - 0.6) < 1e-3 for v in hl))
    b = s["batches_ms"]
    chk("worker round trip p95 < 30 ms (timeout 60)", b["p95"] is not None and b["p95"] < 30, json.dumps(b))
    await w.close()
    print("ALL PASS" if not fails else f"{len(fails)} FAILED")
    sys.exit(1 if fails else 0)


asyncio.run(main())
