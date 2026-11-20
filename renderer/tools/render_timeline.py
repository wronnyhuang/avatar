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
  {"t": 1.23, ..., "canon": [189 floats], "canon_w": {"torso":1,...}, "modes": {"head":"abs"}}
                                                  optional canon1 BODY pose on any line (held until the next
                                                  canon line): the rig drives Juno's bones from it (JunoRig.solve canon)
Without --rig the server's own life drives everything (as with no page).
 canon (Juno only): --canon clip.npy ((T,189) canon1 @30 fps; or a Seamless .npz, key motion (2,T,189), with
--canon-person 0|1 --canon-start FRAME) drives the body; --weights torso=1,head=1,arm_l=1,arm_r=1,fingers_l=1,fingers_r=1
(default all 1), --head-mode / --torso-mode abs|add (default add), --cam home|wide (wide = waist-up, both hands in frame).
 On a shared GPU, serialize runs with a lock:
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
# --life face keeps only the face columns of the server's idle life (the canon owns the body, root and camera)
_BODY_PFX = ("head_", "body_", "root_", "cam_", "hand_", "arm_", "wave_", "point_", "thumb", "shoulder_", "fingers_",
             "foot_", "gest_", "paw_", "prop", "tail_", "ear_")
FACE_ONLY = np.array([not (n.startswith(_BODY_PFX) or n in ("squash", "stand", "clap", "breath")) for n in
                      sorted(IDX, key=IDX.get)] + [False]*max(0, NPARAMS - len(IDX)))[:NPARAMS]


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
        if "canon" in o:                                       # a canon body pose rides this line
            out.append((t, "c", (o["canon"], o.get("canon_w"), o.get("modes")), None))
        if "a" in o:
            out.append((t, "a", o["a"], o.get("names") or names))
        elif "p" in o:
            out.append((t, "p", o["p"], None))
    out.sort(key=lambda r: r[0])
    return out


def load_canon(path, person=0, start=0, frames=None):
    """(T,189) float32 canon1 from a .npy or a Seamless canon .npz (motion (2,T,189), mask (2,T,9))."""
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        m = np.asarray(z["motion"][person], np.float32)
    else:
        m = np.asarray(np.load(path), np.float32)
        if m.ndim == 3:
            m = m[person]
    m = m[int(start):]
    if frames:
        m = m[:int(frames)]
    if m.ndim != 2 or m.shape[1] < 135:
        raise SystemExit(f"--canon {path}: expected (T,189) canon1, got {m.shape}")
    return m


def parse_weights(s):
    w = {g: 1.0 for g in ("torso", "head", "arm_l", "arm_r", "fingers_l", "fingers_r")}
    for kv in (s or "").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            if k.strip() not in w:
                raise SystemExit(f"--weights: unknown group {k!r} (one of {sorted(w)})")
            w[k.strip()] = float(v)
    return w


def wide_camera(char_cfg):
    """waist-up framing with both hands in frame while gesturing (scene units; Juno stands ~2.25 tall)."""
    c = dict(char_cfg.get("camera") or {})
    fov = float(c.get("fov", 22.0))
    tgt_y, half_h = 1.62, 0.78
    d = half_h/np.tan(np.radians(fov)*0.5)
    return {"pos": [0.0, tgt_y + 0.10, d], "target": [0.0, tgt_y, 0.0], "fov": fov, "focus": float(d),
            "dofk": float(c.get("dofk", 40.0)), "motion_layers": False,
            "quality": {"CHAR_T_NEAR": max(0.05, float(d) - 1.2), "CHAR_T_FAR": float(d) + 2.5}}


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
    ap.add_argument("--canon")                                               # canon body
    ap.add_argument("--canon-person", type=int, default=0)
    ap.add_argument("--canon-start", type=int, default=0)
    ap.add_argument("--weights")
    ap.add_argument("--head-mode", default="add", choices=["add", "abs"])
    ap.add_argument("--torso-mode", default="add", choices=["add", "abs"])
    ap.add_argument("--cam", default="home", choices=["home", "wide"])
    ap.add_argument("--lod")
    ap.add_argument("--life", default="on", choices=["on", "off", "face"],
                    help=": off = no server idle life (P = 0 + rig file); face = its face columns only")
    a = ap.parse_args()
    run(a)


def run(a):

    from avatar.render_server import Core
    rig = read_rig(a.rig) if a.rig else []
    crig = [r for r in rig if r[1] == "c"]                       # canon lines (held)
    rig = [r for r in rig if r[1] != "c"]
    canon = load_canon(a.canon, a.canon_person, a.canon_start,
                       int(np.ceil(a.duration*30)) + 1 if a.duration else None) if getattr(a, "canon", None) else None
    cw = parse_weights(getattr(a, "weights", None))
    cmodes = {"head": getattr(a, "head_mode", "add"), "torso": getattr(a, "torso_mode", "add")}
    pcm = read_wav(a.audio) if a.audio else None
    dur = a.duration
    if dur is None:
        dur = max(pcm.size/SR if pcm is not None else 0.0, (rig[-1][0] + 0.5) if rig else 0.0,
                  len(canon)/30.0 if canon is not None else 0.0, (crig[-1][0] + 1/30) if crig else 0.0, 1.0)
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
        lv = getattr(a, "life", "on")
        jp = jb.params if lv == "on" else np.zeros_like(jb.params) if lv == "off" else np.where(FACE_ONLY, jb.params, 0)
        rows.append((jb, mix.apply(jp, t, now=t)))
    if a.params_out:
        np.save(a.params_out, np.stack([r for _, r in rows]).astype("f4"))
    # 3. render
    import rt_engine
    preset = a.preset
    if preset == "auto":
        from live.backends.rt import default_preset
        preset = default_preset(a.character, a.w, a.h, 24)
    t0 = time.time()
    try:                                                       # verify: Juno (mesh) through MeshEngine like live
        from live.backends.rt import engine_class
        Eng = engine_class(a.character)
    except ImportError:
        Eng = rt_engine.RTEngine
    kw = {}
    if getattr(a, "cam", "home") == "wide" or getattr(a, "lod", None):      # (MeshEngine only)
        import render as _render
        if a.cam == "wide":
            kw["cam"] = wide_camera(_render.load_cfg(a.character))
        if a.lod:
            kw["lod"] = a.lod
    eng = Eng(a.character, a.w, a.h, preset, backend=a.backend, threads=a.threads, **kw)
    use_canon = (canon is not None or crig) and hasattr(eng, "canon_pose")
    if (canon is not None or crig) and not use_canon:
        print("canon given but this character's engine has no canon rig (Juno only): ignored", file=sys.stderr)
    ci = 0; ccur = None
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
        if use_canon and getattr(a, "cam", "home") == "wide":     # the wide camera is fixed (no page pans / zoom)
            row = np.array(row, "f4", copy=True)
            row[[IDX["cam_zoom"], IDX["cam_x"], IDX["cam_y"]]] = 0.0
            if not a.rig and getattr(a, "life", "on") != "on":     # nor any default root drift
                row[[IDX[n] for n in ("root_x", "root_y", "root_z", "root_yaw", "root_pitch", "root_roll", "squash")]] = 0.0
        if use_canon:
            t = jb.j/float(a.fps)
            if canon is not None:
                c = canon[min(len(canon) - 1, int(round(t*30.0)))]
                eng.canon_pose = (c, cw, cmodes)
            else:
                while ci < len(crig) and crig[ci][0] <= t + 1e-9:
                    ccur = crig[ci][2]; ci += 1
                if ccur is not None:
                    m_ = dict(cmodes); m_.update(ccur[2] or {})
                    eng.canon_pose = (np.asarray(ccur[0], np.float64), ccur[1] if ccur[1] is not None else cw, m_)
                else:
                    eng.canon_pose = None
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
