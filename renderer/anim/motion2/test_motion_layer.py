"""live: offline test of anim/motion_layer.py driven exactly like render_server (20 ms PCM chunks on the 24 kHz
timeline -> push_audio on the anim thread, sample(t) per output frame, frame(...) per dispatched frame).

    AV_PY test_motion_layer.py [--wav voice24k.wav] [--fps 24] [--mode mm|replay:<npy>|neutral] [--secs 20] [--out poses.npy]
Prints per-call cost (p50 / p99 / max ms), weights over time, pose speed stats; --out writes the canon (T,189) at the
output fps (with the canon_w per frame in a .json beside it) for rendering through RIG's CLI.
"""
import argparse
import json
import os
import sys
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from anim import motion_layer as M  # noqa: E402


def read_wav(p):
    with wave.open(p) as w:
        assert w.getframerate() == 24000 and w.getnchannels() == 1, "24 kHz mono wav"
        return np.frombuffer(w.readframes(w.getnframes()), "<i2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="tests/data/line.wav")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--mode", default="mm")
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--gest", type=float, default=0.5, help="page gest_l/gest_r weight")
    ap.add_argument("--clip", default="", help="slug to play at 3 s")
    ap.add_argument("--out", default="")
    ap.add_argument("--realtime", type=int, default=0, help="pace the 20 ms chunks in real time (threaded sources)")
    a = ap.parse_args()
    voice = read_wav(a.wav)
    n = int(a.secs * 24000)
    pcm = np.zeros(n, np.int16)
    t = 1 * 24000                                     # voice at 1 s, silence gap, voice again
    while t < n:
        k = min(len(voice), n - t)
        pcm[t:t + k] = voice[:k]
        t += len(voice) + 3 * 24000
    t0 = time.perf_counter()
    ml = M.MotionLayer("junor", seed=1, mode=a.mode)
    print("construct s", round(time.perf_counter() - t0, 3))
    while ml.src is None and ml.dead is None:          # the package loads in a thread (live: off the session path)
        time.sleep(0.05)
    print("loaded s", round(time.perf_counter() - t0, 2), json.dumps(ml.info())[:300])
    spf = 24000 // a.fps
    lookahead = int(0.1 * 24000)
    nxt_frame = 0
    cs, cf, cp = [], [], []
    poses, ws = [], []
    t_rt = time.perf_counter()
    for s0 in range(0, n, 480):
        if a.realtime:
            dl = t_rt + s0 / 24000.0 - time.perf_counter()
            if dl > 0:
                time.sleep(dl)
        q = time.perf_counter()
        ml.push_audio(s0, pcm[s0:s0 + 480])
        cp.append(time.perf_counter() - q)
        have = s0 + 480
        while nxt_frame * spf + lookahead <= have:     # frames whose audio (+ lookahead) is in, like the lip core
            tt = nxt_frame / a.fps
            q = time.perf_counter()
            pose = ml.sample(tt)
            cs.append(time.perf_counter() - q)
            if a.clip and abs(tt - 3.0) < 0.5 / a.fps:
                ml.clip({"slug": a.clip})
            q = time.perf_counter()
            r = ml.frame(pose, tt, a.gest, a.gest)
            cf.append(time.perf_counter() - q)
            poses.append(r[0] if r is not None else M.neutral())
            ws.append(r[1] if r is not None else {})
            nxt_frame += 1
    ms = lambda x: (round(1e3 * np.percentile(x, 50), 3), round(1e3 * np.percentile(x, 99), 3), round(1e3 * max(x), 3))
    print("frames", len(poses), "push p50/p99/max ms", ms(cp), "sample", ms(cs), "frame", ms(cf))
    P = np.stack(poses)
    v = np.abs(np.diff(P[:, :135], axis=0)) * a.fps
    print("max joint speed rad/s", round(float(v.max()), 2), "mean", round(float(v.mean()), 4))
    for k in range(0, len(ws), a.fps * 2):
        print(f"t={k / a.fps:5.1f}", {g: round(x, 2) for g, x in ws[k].items()})
    print("stats", ml.stats())
    if a.out:
        np.save(a.out, P.astype(np.float32))
        json.dump({"fps": a.fps, "canon_w": ws}, open(a.out.replace(".npy", ".json"), "w"))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
