#!/usr/bin/env python3
"""Offline video from a rig-vector timeline (+ optional voice through lipstream for the mouth), through the SAME path
the live render server uses: LiveCore/StreamingLipAnimator (causal, lookahead 133 ms) for the lip columns and the
server's idle life, anim.rig.RigMixer for the page's rig vectors (interpolation, critically damped smoothing, legacy
mapping, jaw = max(lipsync, mouth_open), life off while rig vectors arrive), then the real-time engine.

  python tools/render_timeline.py --character fox --rig rig.jsonl [--audio voice.wav] --w 480 --h 270 \\
         --backend cpu|gpu --out out.mp4 [--fps 24] [--duration S] [--params-out rows.npy] [--threads 12]

rig.jsonl, one JSON object per line, any of:
  {"rig_params": ["smile", ...]}                  optional header: the order of "a" (default anim.rig.RIG_PARAMS)
  {"type": "rig", "t": 1.23, "a": [...]}          the contract's wire message (t = playback-timeline seconds)
  {"t": 1.23, "a": [...], "names": [...]}         an explicit order for this line
  {"t": 1.23, "p": {"hand_r_y": 1.0, ...}}        a sparse dict on top of the previous vector (handy by hand)
Without --rig the server's own life drives everything (as with no page).  On a shared GPU, serialize runs with a lock:
  flock gpu.lock python tools/render_timeline.py ... --backend gpu
"""
import argparse, json, os, subprocess, sys, time, wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine")]
from anim.schema import IDX, NPARAMS, SAMPLE_RATE as SR  # noqa: E402
from anim.rig import RigMixer, RIG_PARAMS  # noqa: E402

SPFS = {20: 1200, 24: 1000, 30: 800}


def read_wav(path):
    with wave.open(path, "rb") as w:
        n, ch, sw, rate = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32)/32768.0
    elif sw == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32)/2147483648.0
    elif sw == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0)/128.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    x = x.reshape(-1, ch).mean(axis=1)
    if rate != SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(rate, SR)
        x = resample_poly(x, SR//g, rate//g).astype(np.float32)
    return np.clip(x*32768.0, -32768, 32767).astype(np.int16)


def read_rig(path):
    """-> list of (t, kind, payload, names): kind 'a' (vector) or 'p' (sparse dict)."""
    out, names = [], None
    for ln, line in enumerate(open(path)):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        o = json.loads(line)
        if "rig_params" in o and "t" not in o:
            names = list(o["rig_params"])
            continue
        if o.get("type") not in (None, "rig"):
            continue
        t = float(o["t"])
        if "a" in o:
            out.append((t, "a", o["a"], o.get("names") or names))
        elif "p" in o:
            out.append((t, "p", o["p"], None))
    out.sort(key=lambda r: r[0])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--character", default="fox")
    ap.add_argument("--rig")
    ap.add_argument("--audio")
    ap.add_argument("--w", type=int, default=480)
    ap.add_argument("--h", type=int, default=270)
    ap.add_argument("--fps", type=int, default=24, choices=sorted(SPFS))
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--preset", default="auto")
    ap.add_argument("--lookahead-ms", type=float, default=133.0)
    ap.add_argument("--duration", type=float)
    ap.add_argument("--out", required=True)
    ap.add_argument("--params-out")
    ap.add_argument("--crf", type=int, default=18)
    a = ap.parse_args()

    from avatar.render_server import Core
    rig = read_rig(a.rig) if a.rig else []
    pcm = read_wav(a.audio) if a.audio else None
    dur = a.duration
    if dur is None:
        dur = max(pcm.size/SR if pcm is not None else 0.0, (rig[-1][0] + 0.5) if rig else 0.0, 1.0)
    n_s = int(round(dur*SR))
    if pcm is None:
        pcm = np.zeros(n_s, np.int16)
    elif pcm.size < n_s:
        pcm = np.concatenate([pcm, np.zeros(n_s - pcm.size, np.int16)])
    else:
        pcm = pcm[:n_s]
    # 1. lips + idle life, causal, the server's own animator; padded so the last frame is released
    core = Core(a.character, a.lookahead_ms, a.fps)
    jobs = []
    pad = np.zeros(int(SR*(a.lookahead_ms/1000.0 + 0.2)), np.int16)
    for k in range(0, pcm.size + pad.size, 2400):
        chunk = np.concatenate([pcm, pad])[k:k + 2400]
        jobs += core.push(chunk)
    n_frames = int(np.floor(dur*a.fps))
    jobs = [jb for jb in jobs if jb.j < n_frames]
    # 2. the page's rig vectors, streamed in timeline order, smoothed per frame exactly as the server does
    mix = RigMixer(character=a.character)
    cur = np.zeros(len(RIG_PARAMS), "f4")
    rows, k = [], 0
    for jb in jobs:
        t = jb.j/float(a.fps)
        while k < len(rig) and rig[k][0] <= t + 1e-9:
            tr, kind, payload, names = rig[k]
            if kind == "a":
                mix.push_rig(tr, payload, now=tr, names=names)
                if names is None:
                    cur = np.asarray(payload, "f4")[:len(RIG_PARAMS)]
            else:
                for n, v in payload.items():
                    if n in RIG_PARAMS:
                        cur[RIG_PARAMS.index(n)] = float(v)
                mix.push_rig(tr, cur, now=tr)
            k += 1
        rows.append((jb, mix.apply(jb.params, t, now=t)))
    if a.params_out:
        np.save(a.params_out, np.stack([r for _, r in rows]).astype("f4"))
    # 3. render
    import rt_engine
    preset = a.preset
    if preset == "auto":
        from live.backends.rt import default_preset
        preset = default_preset(a.character, a.w, a.h, 24)
    t0 = time.time()
    eng = rt_engine.RTEngine(a.character, a.w, a.h, preset, backend=a.backend, threads=a.threads)
    print(f"engine {a.character} {a.w}x{a.h} {a.backend} preset {preset}: {time.time()-t0:.1f}s; {len(rows)} frames",
          file=sys.stderr, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.w}x{a.h}",
           "-r", str(a.fps), "-i", "-"]
    tmp_wav = None
    if a.audio:
        tmp_wav = a.out + ".tmp.wav"
        with wave.open(tmp_wav, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
        cmd += ["-i", tmp_wav, "-c:a", "aac", "-b:a", "96k"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-shortest", a.out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    ts = []
    for i, (jb, row) in enumerate(rows):
        t1 = time.perf_counter()
        img = eng.render(int(jb.index), row)
        ts.append(time.perf_counter() - t1)
        ff.stdin.write(np.ascontiguousarray(img).tobytes())
        if i % 48 == 0:
            print(f"  frame {i}/{len(rows)}  {np.mean(ts[-48:])*1000:.0f} ms", file=sys.stderr, flush=True)
    ff.stdin.close()
    ff.wait()
    eng.close()
    if tmp_wav:
        os.unlink(tmp_wav)
    ts = np.array(ts)*1000
    print(json.dumps({"out": a.out, "frames": len(rows), "fps": a.fps, "render_mean_ms": round(float(ts.mean()), 1),
                      "render_p95_ms": round(float(np.percentile(ts, 95)), 1), "backend": a.backend}))


if __name__ == "__main__":
    main()
