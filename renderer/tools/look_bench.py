#!/usr/bin/env python3
"""Smoothness bench of the live render path: one scripted sequence through the backend the render worker uses
(live.backends.rt.RTBackend) and the session's encoder, timed per frame.

  python tools/look_bench.py --animal muse --w 480 --h 270 [--preset auto] [--opts '{"tier":"auto"}'] [--json out.json]
  python tools/look_bench.py --animal muse --profile big_gpu          # size + preset from avatar/profiles.json
  python tools/look_bench.py --root /path/to/another/renderer ...     # the same sequence through another tree

The sequence (24 fps, paced like the live loop unless --no-pace): talk at rest, hands moving, hands raised by the face,
walk with the camera following, spin, backflip, the wide shot, and the desk scene (headphones worn, laptop open,
walking; a character without props or a root just keeps its hands moving).  Per frame: render ms (draw + readback,
synchronous: what the worker reports as render_ms), encode ms (H.264 of that frame), their sum.  A STUTTER is a frame
whose sum is over 41.7 ms (a 24 fps frame period): the live server would show the previous picture twice.
Prints one JSON line (and writes --json): overall and per segment p50 / p95 / p99 / max, stutters, the fps the sum
allows, and the quality-tier trace when the backend adapts.
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if "--root" in sys.argv:
    ROOT = os.path.abspath(sys.argv[sys.argv.index("--root") + 1])
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine"), HERE]
from anim.schema import IDX, NPARAMS  # noqa: E402
from rig_bench import pose_track  # noqa: E402  (this tree's tool: the motions are the same columns in every tree)

SEGMENTS = [("talk", "rest", 6.0), ("hands", "move", 6.0), ("raised", "raised", 5.0), ("walk", "walk", 6.0),
            ("spin", "spin", 4.0), ("flip", "flip", 4.0), ("wide", "wide", 6.0), ("desk", "worn", 8.0)]
PERIOD_MS = 1000.0/24.0


def sequence(fps, scale=1.0):
    rows, segs = [], []
    for name, motion, secs in SEGMENTS:
        n = max(int(round(secs*scale*fps)), 8)
        P = pose_track(n, motion, fps=fps)
        # every segment talks and looks about: the face is what the viewer watches
        t = np.arange(n)/fps
        for col, v in (("smile", 0.35 + 0.35*np.sin(t*0.7)), ("brow_l", 0.4*np.sin(t*1.1)), ("brow_r", 0.4*np.sin(t*1.1 + 0.3)),
                       ("gaze_x", 0.5*np.sin(t*0.9)), ("mouth_open", 0.5 + 0.5*np.sin(t*7.0))):
            if col in IDX and not P[:, IDX[col]].any():
                P[:, IDX[col]] = v
        segs.append((name, len(rows), len(rows) + n))
        rows.extend(P)
    return np.asarray(rows, "f4"), segs


def stats(render, enc, period=PERIOD_MS):
    tot = render + enc
    pc = lambda a, q: round(float(np.percentile(a, q)), 2)
    return dict(frames=int(len(tot)), render_p50=pc(render, 50), render_p95=pc(render, 95), render_p99=pc(render, 99),
                render_max=round(float(render.max()), 2), enc_p50=pc(enc, 50), enc_p95=pc(enc, 95), enc_p99=pc(enc, 99),
                total_p50=pc(tot, 50), total_p95=pc(tot, 95), total_p99=pc(tot, 99), total_max=round(float(tot.max()), 2),
                total_mean=round(float(tot.mean()), 2), over33=int((tot > 33.3).sum()), stutters=int((tot > period).sum()),
                fps_cap=round(1000.0/max(float(tot.mean()), 1e-3), 1), holds24=bool(np.percentile(tot, 99) <= period))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default="muse")
    ap.add_argument("--w", type=int, default=0)
    ap.add_argument("--h", type=int, default=0)
    ap.add_argument("--preset", default="auto")
    ap.add_argument("--profile", help="avatar/profiles.json section (local | big_gpu): size + preset when not given")
    ap.add_argument("--opts", default="{}", help="extra backend opts (JSON), e.g. '{\"tier\":\"auto\"}'")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--bitrate", type=int, default=450000)
    ap.add_argument("--nvenc", action="store_true")
    ap.add_argument("--enc-threads", type=int, default=4)
    ap.add_argument("--no-encode", action="store_true")
    ap.add_argument("--no-pace", action="store_true")
    ap.add_argument("--secs-scale", type=float, default=1.0)
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--label", default="")
    ap.add_argument("--json")
    ap.add_argument("--save-dir", help="write one PNG per segment (its middle frame)")
    ap.add_argument("--trace", help="write per-frame [render ms, encode ms, tier] + the controller's switch log (JSON)")
    ap.add_argument("--root")
    a = ap.parse_args()
    opts = json.loads(a.opts)
    w, h, preset = a.w, a.h, a.preset
    if a.profile:
        prof = json.load(open(os.path.join(ROOT, "avatar", "profiles.json")))["profiles"][a.profile]
        p = dict(prof.get("_default", {})); p.update(prof.get(a.animal, {}))
        if h and str(h) in p.get("alts", {}):
            alt = p["alts"][str(h)]; p.update(alt); p["opts"] = dict(p.get("opts", {}), **alt.get("opts", {}))
        w, h = w or p["width"], h or p["height"]
        if preset == "auto":
            preset = (p.get("opts") or {}).get("preset", "auto")
        for k, v in (p.get("opts") or {}).items():
            if k != "preset":
                opts.setdefault(k, v)
    w, h = w or 480, h or 270
    from live.backends.rt import RTBackend
    t0 = time.time()
    be = RTBackend(a.animal, w, h, preset=preset, target_fps=a.fps, **opts)
    init_s = time.time() - t0
    enc = None
    if not a.no_encode:
        from avatar.encoder import open_encoder
        enc = open_encoder("h264", w, h, fps=a.fps, bitrate=a.bitrate, gop_s=2.0, nvenc=a.nvenc, threads=a.enc_threads, quality=80)
    P, segs = sequence(a.fps, a.secs_scale)
    for i in range(12):                       # warm-up (not counted): shaders, history, encoder
        img = be.render(i, P[0])
        if enc is not None:
            enc.encode(img, i, i == 0)
    n = len(P)
    tr, te, tier = np.zeros(n*a.loops), np.zeros(n*a.loops), []
    period = 1.0/a.fps
    note = getattr(be, "note_frame", None)
    t_next = time.perf_counter()
    for L in range(a.loops):
        for i in range(n):
            if not a.no_pace:
                dt = t_next - time.perf_counter()
                if dt > 0:
                    time.sleep(dt)
                t_next = max(t_next + period, time.perf_counter() - period)
            j = L*n + i
            t = time.perf_counter()
            img = be.render(12 + j, P[i])
            t1 = time.perf_counter()
            if enc is not None:
                enc.encode(img, 12 + j, False)
            t2 = time.perf_counter()
            tr[j], te[j] = (t1 - t)*1000, (t2 - t1)*1000
            if note is not None:
                note(tr[j], te[j])
            tier.append(getattr(be, "tier_now", None))
            if a.save_dir and L == 0 and any(i == (s0 + s1)//2 for _, s0, s1 in segs):
                from PIL import Image
                os.makedirs(a.save_dir, exist_ok=True)
                name = next(nm for nm, s0, s1 in segs if i == (s0 + s1)//2)
                Image.fromarray(img).save(os.path.join(a.save_dir, f"{a.animal}_{name}.png"))
    out = dict(label=a.label, tree=ROOT, animal=a.animal, w=w, h=h, preset=preset if isinstance(preset, str) else "dict",
               backend_preset=str(getattr(be, "preset", "")), opts=opts, fps=a.fps, paced=not a.no_pace,
               encoder=getattr(enc, "name", None), bitrate=a.bitrate, init_s=round(init_s, 1), overall=stats(tr, te, 1000.0/a.fps*1.0))
    out["segments"] = {}
    for name, s0, s1 in segs:
        idx = np.concatenate([np.arange(L*n + s0, L*n + s1) for L in range(a.loops)])
        s = stats(tr[idx], te[idx], 1000.0/a.fps)
        out["segments"][name] = {k: s[k] for k in ("render_p50", "render_p95", "total_p50", "total_p95", "total_p99", "total_max", "stutters")}
        if tier and tier[0] is not None:
            tv = [tier[k] for k in idx]
            out["segments"][name]["tier_mean"] = round(float(np.mean(tv)), 2); out["segments"][name]["tier_min"] = int(min(tv))
    if tier and tier[0] is not None:
        tv = np.array(tier)
        out["tier"] = dict(mean=round(float(tv.mean()), 2), min=int(tv.min()), max=int(tv.max()),
                           switches=int((np.diff(tv) != 0).sum()), names=getattr(be, "tier_names", None))
    try:
        import subprocess
        out["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True,
                                    timeout=5).stdout.strip().splitlines()[0]
    except Exception:
        out["gpu"] = "unknown"
    if a.trace:
        ctl = getattr(be, "ctl", None)
        json.dump(dict(render=[round(float(x), 2) for x in tr], enc=[round(float(x), 2) for x in te], tier=tier,
                       switches=list(ctl.log) if ctl is not None else [], segs=segs), open(a.trace, "w"))
    s = json.dumps(out)
    print(s, flush=True)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        open(a.json, "w").write(s + "\n")
    be.close()


if __name__ == "__main__":
    main()
