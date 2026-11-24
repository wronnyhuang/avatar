#!/usr/bin/env python3
"""Tests of the libx264 preset choice (avatar/encoder.py x264_preset) and the runtime preset guard (PresetGuard + its
wiring in avatar/render_worker.py).  CPU only, no GPU, no network (~10 s).

    python -m avatar.test_x264_guard            # all
    python -m avatar.test_x264_guard unit       # pure logic only (no encoder)

unit:   preset precedence (env > explicit > size default), the guard on synthetic encode times (warm-up frames ignored,
        p90 over the window, trips once, never flaps, reset keeps a trip, disabled by env).
encoder: H264Encoder opens with the size-aware preset (the x264 SEI option string shows deblocking on at superfast,
        off at ultrafast, lookahead 0 and no B-frames at both), next_is_key() follows the GOP.
worker: avatar/render_worker.main in a child process with the CPU mock backend (live.testing:MouthBackend); the
        child's superfast encode is slowed by a synthetic 25 ms (> 45 % of the 24 fps period) -> the worker reopens at
        ultrafast at the next keyframe, the first frame of the new encoder is an IDR, its "encoder" message carries
        preset/preset_guard, no frame is lost, and there is no further switch (no flapping).  A fast encoder never trips.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        FAILS.append(what)


class _Env:
    def __init__(self, **kv):
        self.kv, self.old = kv, {}

    def __enter__(self):
        for k, v in self.kv.items():
            self.old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ------------------------------------------------------------------ unit
def test_unit():
    from avatar.encoder import x264_preset, PresetGuard
    with _Env(AVATAR_X264_PRESET=None):
        check(x264_preset(480, 270) == ("superfast", "size"), "270p default superfast")
        check(x264_preset(1280, 720) == ("superfast", "size"), "720p default superfast (inclusive)")
        check(x264_preset(1920, 1080) == ("ultrafast", "size"), "1080p default ultrafast")
        check(x264_preset(1280, 720, "ultrafast") == ("ultrafast", "arg"), "explicit preset beats the size default")
    with _Env(AVATAR_X264_PRESET="veryfast"):
        check(x264_preset(480, 270, "ultrafast") == ("veryfast", "env"), "env beats an explicit preset")
        check(x264_preset(1920, 1080) == ("veryfast", "env"), "env beats the size default")
    with _Env(AVATAR_X264_PRESET="  "):
        check(x264_preset(480, 270)[1] == "size", "blank env = unset")

    with _Env(AVATAR_X264_GUARD=None, AVATAR_X264_GUARD_FRAC=None):
        g = PresetGuard(fps=24)
        check(abs(g.limit_ms - 18.75) < 1e-6 and g.n == 48, "limit 45% of 41.7 ms, window 48 frames")
        fired = [g.note(100.0) for _ in range(12)]         # warm-up frames: never counted
        check(not any(fired) and not g.times, "warm-up frames ignored")
        fired = [g.note(5.0) for _ in range(200)]
        check(not any(fired), "fast encoder never trips")
        from avatar.encoder import RetryGate
        r = RetryGate(18.75, now=100.0, fps=24, retry_s=60)
        check(not any(r.note(4.0, 100.0 + i / 24) for i in range(24 * 59)), "retry gate: nothing before retry_s")
        check(any(r.note(4.0, 160.0 + i / 24) for i in range(8)) and r.done, "retry gate: fires once after retry_s with room")
        check(not r.note(4.0, 200.0), "retry gate: only once")
        r = RetryGate(18.75, now=0.0, fps=24, retry_s=60)
        check(not any(r.note(12.0, 61.0 + i / 24) for i in range(200)), "retry gate: no retry while the fast preset has no room")
        g = PresetGuard(fps=24, warm=0)
        # 10 % of frames slow (one in ten): p90 is still fast -> no trip (IDR spikes, a hiccup)
        fired = [g.note(30.0 if i % 10 == 0 else 6.0) for i in range(300)]
        check(not any(fired), "isolated spikes (<= 10 %) do not trip")
        g = PresetGuard(fps=24, warm=0)
        seq = [10.0] * 60 + [22.0] * 60
        fired = [g.note(x) for x in seq]
        check(sum(fired) == 1, "sustained slow encodes trip exactly once")
        k = fired.index(True)
        # p90 over 48 frames passes the limit once more than ~10 % of the window (5 frames) is slow
        check(60 + 3 <= k <= 60 + 7, f"trips once ~10% of the window is slow (frame {k})")
        check(g.tripped and "p90" in (g.reason or ""), "reason names the statistic")
        fired = [g.note(1.0) for _ in range(200)] + [g.note(50.0) for _ in range(200)]
        check(not any(fired) and g.tripped, "no flapping: stays tripped, never fires again")
        g.reset()
        check(g.tripped and not g.times, "reset keeps the trip")
        g = PresetGuard(fps=30, warm=0)
        fired = [g.note(16.0) for _ in range(100)]       # 16 ms > 45 % of 33.3 ms
        check(any(fired), "the limit follows the session fps (30 fps: 15 ms)")
    with _Env(AVATAR_X264_GUARD="0"):
        g = PresetGuard(fps=24, warm=0)
        check(not any(g.note(100.0) for _ in range(200)), "AVATAR_X264_GUARD=0 disables it")
    with _Env(AVATAR_X264_GUARD_FRAC="0.8"):
        g = PresetGuard(fps=24, warm=0)
        check(abs(g.limit_ms - 33.333) < 0.01, "AVATAR_X264_GUARD_FRAC sets the fraction")


# ------------------------------------------------------------------ encoder
def _opts(data):
    k = data.find(b"options: ")
    return data[k:data.find(b"\x00", k)].decode("ascii", "replace") if k >= 0 else ""


def test_encoder():
    from avatar.encoder import H264Encoder
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (270, 480, 3), np.uint8)
    with _Env(AVATAR_X264_PRESET=None):
        for preset, want in ((None, "superfast"), ("ultrafast", "ultrafast")):
            e = H264Encoder(480, 270, 24, 250000, 2.0, prefer=("libx264",), threads=2, preset=preset)
            check(e.preset == want, f"opened preset {e.preset} (asked {preset})")
            first = e.encode(img, 0, True)
            o = _opts(first[0][1])
            check("deblock=1" in o if want == "superfast" else "deblock=0" in o, f"{want}: deblock in SEI")
            check("rc_lookahead=0" in o and "bframes=0" in o, f"{want}: zerolatency kept (no lookahead, no B-frames)")
            keys = []
            for j in range(1, 100):
                nk = e.next_is_key()
                out = e.encode(img, j)
                keys.append((nk, out[0][2]))
            check(all(a == b for a, b in keys), f"{want}: next_is_key() predicts every IDR (gop 48)")
            check(sum(b for _, b in keys) == 2, f"{want}: IDRs at 48 and 96")
            e.close()
    with _Env(AVATAR_X264_PRESET="ultrafast"):
        e = H264Encoder(480, 270, 24, 250000, 2.0, prefer=("libx264",), threads=2)
        check(e.preset == "ultrafast" and e.x264_preset_src == "env", "env pin opens ultrafast")
        e.close()


# ------------------------------------------------------------------ worker
def _child(conn, slow_ms, width, height):
    """Spawned child: render_worker.main with the superfast encode slowed by slow_ms (synthetic host CPU)."""
    sys.path.insert(0, ROOT)
    os.environ.pop("AVATAR_X264_PRESET", None)
    os.environ["AVATAR_ENC_SYNC"] = "1"
    from avatar import encoder, render_worker
    orig = encoder.H264Encoder.encode

    def encode(self, rgb, frame_no, force_key=False):
        if self.preset not in (None, encoder.X264_FAST) and slow_ms:
            time.sleep(slow_ms / 1000.0)
        return orig(self, rgb, frame_no, force_key)

    encoder.H264Encoder.encode = encode
    render_worker.main(conn, "mock", width, height, "live.testing:MouthBackend", {}, None, 1)


def _run_worker(slow_ms, n=170, w=480, h=270):
    from anim.schema import NPARAMS
    ctx = mp.get_context("spawn")
    conn, child = ctx.Pipe()
    p = ctx.Process(target=_child, args=(child, slow_ms, w, h), daemon=True)
    p.start()
    child.close()
    encs, frames = [], []
    try:
        while True:
            m = conn.recv()
            if m[0] == "ready":
                break
            if m[0] == "error":
                raise RuntimeError(m[1])
        cfg = {"codec": "h264", "fps": 24, "bitrate": 250000, "gop_s": 2.0, "nvenc": False, "quality": 80, "threads": 2}
        conn.send(("encoder", cfg))
        P = np.zeros(NPARAMS, np.float32)
        for j in range(n):
            conn.send(("job", j, j, P, j == 0, time.monotonic()))
            while True:
                m = conn.recv()
                if m[0] == "encoder":
                    encs.append((len(frames), m[1]))
                elif m[0] == "frame":
                    frames.append((m[1], bool(m[2] & 1), len(m[3])))
                    if m[1] == j:
                        break
                elif m[0] == "error":
                    raise RuntimeError(m[1])
        conn.send(("quit",))
    finally:
        p.join(10)
        if p.is_alive():
            p.kill()
    return encs, frames


def test_worker():
    encs, frames = _run_worker(slow_ms=25.0)
    check(len(frames) == 170 and all(f[2] > 0 for f in frames), f"slow host: every frame out ({len(frames)})")
    check(encs[0][1].get("preset") == "superfast" and not encs[0][1].get("preset_guard"), "starts on superfast")
    sw = [(i, e) for i, e in encs[1:]]
    check(len(sw) == 1, f"exactly one switch ({len(sw)})")
    if sw:
        i, e = sw[0]
        check(e.get("preset") == "ultrafast" and "p90" in str(e.get("preset_guard")), "switch -> ultrafast, reason given")
        check(e.get("codec_string") == encs[0][1].get("codec_string"), "same codec_string (no decoder change needed)")
        check(frames[i][1], f"first frame of the new encoder is an IDR (frame {frames[i][0]})")
        # trip after warm 12 + ~44 slow frames; the switch waits for the GOP's own IDR (frame 96 here) -> no extra IDR
        check(12 + 40 <= frames[i][0] <= 12 + 48 + 48, f"switch at the keyframe boundary after the trip ({frames[i][0]})")
        check(frames[i][0] % 48 == 0, "switch replaces the GOP's own IDR (no extra keyframe)")
    encs, frames = _run_worker(slow_ms=25.0, n=230)
    check(len(encs) == 2 and len([e for _, e in encs if e.get("preset") == "superfast"]) == 1,
          f"no flapping back to superfast ({[e.get('preset') for _, e in encs]})")
    encs, frames = _run_worker(slow_ms=0.0, n=130)
    check(len(encs) == 1 and encs[0][1].get("preset") == "superfast" and len(frames) == 130, "fast host: no switch")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    which = argv[0] if argv else "all"
    test_unit()
    if which != "unit":
        test_encoder()
        test_worker()
    print(f"{'FAILED ' + str(len(FAILS)) if FAILS else 'all passed'}", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
