#!/usr/bin/env python3
"""Encoder quality + timing through the LIVE H.264 encoder class (avatar/encoder.py H264Encoder), one preset at a time.

  python tools/enc_quality.py --frames DIR --bitrate 250000 --presets ultrafast,superfast [--roi x0,y0,x1,y1] [--json out.json]
  python tools/enc_quality.py --render muse --w 480 --h 270 --n 72 --bitrate 250000 --presets default,ultrafast
  python tools/enc_quality.py --frames DIR --size 1280x720 --bitrate 1100000 --presets ultrafast,superfast,veryfast \
      --min-frames 240 --timing-only

Frames: a directory of PNGs (sorted by name), or --render CHAR: a short synthetic talking clip rendered through a
live backend (--backend: auto | rt | v2 | sprite | raster | package.module:Class, as the render worker resolves it;
--render-opts JSON for its options; 8 settle frames, then lips / jaw / smile / small head motion / one blink).
The clip is played forward, backward, forward... (ping-pong) until --min-frames, so rate control settles; the first
pass is not scored.  Frames are handed to the encoder as RGB (the encoder's own colour conversion is timed too) at the
encoder's size (--size; a different source size is downscaled first with cv2 INTER_AREA).  Paced at --fps by default
(the live cadence); --no-pace encodes back to back.

Per preset (a name, or "default" = the encoder's size-aware choice): encoder actually opened, preset, the x264 option
string it wrote (deblock / lookahead / bframes check), kbps, bytes per frame (all / IDR / P), encode ms (mean, p50, p90,
p95, max), and against the source (luma): PSNR, SSIM, face-ROI PSNR + SSIM, and the block-edge ratio of the decoded
frames (mean |luma step| across 8-px block boundaries / mean step elsewhere; 1.0 = no block grid; the source's own
ratio is reported beside it).  JSON to stdout (and --json).  AVATAR_X264_PRESET is cleared for the run (every preset
here is explicit).  Threads: --threads (the server's --enc-threads, default 4).
"""
import argparse, glob, json, math, os, platform, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine")]
import numpy as np  # noqa: E402


# ------------------------------------------------------------------ metrics (luma, 8-bit)
def psnr(a, b):
    m = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if m == 0 else 10 * math.log10(255.0 ** 2 / m)


def ssim(a, b):
    import cv2
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu1, mu2 = g(a), g(b)
    s1, s2, s12 = g(a * a) - mu1 ** 2, g(b * b) - mu2 ** 2, g(a * b) - mu1 * mu2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))
    return float(m.mean())


def block_edge_ratio(y):
    """Mean |luma step| across 8-px block boundaries / mean |step| elsewhere (1.0 = no block grid)."""
    y = y.astype(np.float64)
    dx = np.abs(np.diff(y, axis=1))
    dy = np.abs(np.diff(y, axis=0))
    bx, ox = dx[:, 7::8].mean(), np.delete(dx, np.s_[7::8], axis=1).mean()
    by, oy = dy[7::8, :].mean(), np.delete(dy, np.s_[7::8], axis=0).mean()
    return float((bx + by) / (ox + oy + 1e-9))


def x264_options(data):
    """The 'options: ...' string libx264 writes into its SEI (first IDR), as a dict; {} if absent (NVENC)."""
    k = data.find(b"options: ")
    if k < 0:
        return {}
    end = data.find(b"\x00", k)
    txt = data[k + 9:end if end > 0 else k + 2000].decode("ascii", "replace")
    out = {}
    for tok in txt.split():
        if "=" in tok:
            a, b = tok.split("=", 1)
            out[a] = b
    return out


def ms_stats(ms):
    a = np.asarray(ms, np.float64)
    pc = lambda q: round(float(np.percentile(a, q)), 2)
    return dict(n=int(a.size), mean=round(float(a.mean()), 2), p50=pc(50), p90=pc(90), p95=pc(95), max=round(float(a.max()), 2))


# ------------------------------------------------------------------ frames
def load_frames(d):
    from PIL import Image
    fs = sorted(glob.glob(os.path.join(d, "*.png")))
    if not fs:
        raise SystemExit(f"no PNG frames in {d}")
    return [np.asarray(Image.open(f).convert("RGB")) for f in fs]


def render_frames(ch, w, h, n, backend, opts):
    """A short synthetic talking clip through a live backend (resolved like the render worker does)."""
    os.chdir(ROOT)
    from anim.rig_poses import KEY_POSES, pose_vector
    from avatar.render_worker import resolve_backend, default_opts
    cls, bname, _ = resolve_backend(backend)
    eng = cls(ch, w, h, **default_opts(bname, opts))
    base = dict(KEY_POSES["REST"])

    def vec(d):
        x = dict(base)
        x.update(d)
        return pose_vector(x)

    for f in range(8):
        eng.render(100 + f, vec({}))
    out = []
    for f in range(n):
        t = f / 24.0
        d = dict(lip_open=max(0.0, 0.35 + 0.35 * math.sin(2 * math.pi * 3.1 * t)),
                 jaw=max(0.0, 0.3 + 0.3 * math.sin(2 * math.pi * 3.1 * t)), lip_width=0.3 * math.sin(2 * math.pi * 1.3 * t),
                 smile=0.3, head_yaw=0.02 * math.sin(2 * math.pi * 0.4 * t), head_pitch=0.01 * math.sin(2 * math.pi * 0.7 * t),
                 gaze_x=0.1 * math.sin(2 * math.pi * 0.25 * t))
        if 20 <= f < 24:
            d["blink_l"] = d["blink_r"] = [0.5, 1.0, 1.0, 0.5][f - 20]
        out.append(np.ascontiguousarray(np.asarray(eng.render(108 + f, vec(d)))[:, :, :3]))
    try:
        eng.close()
    except Exception:
        pass
    return out


def ping_pong(frames, min_frames):
    seq, fwd = list(frames), False
    while len(seq) < max(min_frames, 3 * len(frames)):
        seq += frames if fwd else frames[::-1]
        fwd = not fwd
    return seq


# ------------------------------------------------------------------ one preset
def run_preset(name, src, a, w, h, threads):
    import av
    import cv2
    from avatar.encoder import H264Encoder
    preset = None if name == "default" else name
    enc = H264Encoder(w, h, fps=a.fps, bitrate=a.bitrate, gop_s=a.gop_s, prefer=(a.encoder,), threads=threads,
                      preset=preset)
    feed = src
    dec = None if a.timing_only else av.CodecContext.create("h264", "r")
    period = 1.0 / a.fps
    ms, sizes, keys, outs, opts = [], [], [], [], {}
    t_next = time.perf_counter()
    for i, fr in enumerate(feed):
        if a.pace:
            dt = t_next - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            t_next = max(t_next + period, time.perf_counter() - period)
        t = time.perf_counter()
        pk = enc.encode(fr, i, i == 0)
        ms.append(1000 * (time.perf_counter() - t))
        for _, data, key in pk:
            sizes.append(len(data))
            keys.append(bool(key))
            if key and not opts:
                opts = x264_options(data)
            if dec is not None:
                for p in dec.parse(data):
                    for f in dec.decode(p):
                        outs.append(f.to_ndarray(format="rgb24"))
    if dec is not None:
        for p in dec.parse(b""):
            for f in dec.decode(p):
                outs.append(f.to_ndarray(format="rgb24"))
    enc.close()
    skip = a.warm
    sz = np.asarray(sizes, np.float64)
    kk = np.asarray(keys, bool)
    r = dict(encoder=enc.name, preset=enc.preset, preset_source=enc.x264_preset_src if enc.name == "libx264" else None,
             codec_string=enc.codec_string, threads=threads, size=f"{w}x{h}", bitrate=a.bitrate, frames=len(feed),
             kbps=round(float(sz.sum()) * 8 / len(feed) * a.fps / 1000, 1),
             bytes_per_frame=round(float(sz.mean()), 1),
             bytes_idr=round(float(sz[kk].mean()), 1) if kk.any() else None,
             bytes_p=round(float(sz[~kk].mean()), 1) if (~kk).any() else None,
             encode_ms=ms_stats(ms[skip:]),
             x264=({k: opts.get(k) for k in ("deblock", "rc_lookahead", "bframes", "sliced_threads", "threads", "me",
                                               "subme", "trellis", "ref", "cabac")} if opts else None))
    if dec is not None:
        n = min(len(outs), len(src))
        sk = a.score_from if a.score_from is not None else min(len(src) // 3, n // 2)
        P, S, PF, SF, B, BS = [], [], [], [], [], []
        x0, y0, x1, y1 = a.roi_box
        for i in range(sk, n):
            ya = cv2.cvtColor(src[i], cv2.COLOR_RGB2YUV)[..., 0]
            yb = cv2.cvtColor(outs[i], cv2.COLOR_RGB2YUV)[..., 0]
            P.append(psnr(ya, yb))
            S.append(ssim(ya, yb))
            PF.append(psnr(ya[y0:y1, x0:x1], yb[y0:y1, x0:x1]))
            SF.append(ssim(ya[y0:y1, x0:x1], yb[y0:y1, x0:x1]))
            B.append(block_edge_ratio(yb))
            BS.append(block_edge_ratio(ya))
        r.update(scored=len(P), psnr_y=round(float(np.mean(P)), 2), ssim_y=round(float(np.mean(S)), 4),
                 face_psnr=round(float(np.mean(PF)), 2), face_ssim=round(float(np.mean(SF)), 4),
                 face_psnr_p10=round(float(np.percentile(PF, 10)), 2),
                 block_edge=round(float(np.mean(B)), 3), block_edge_src=round(float(np.mean(BS)), 3))
        if a.save_crop:
            i = min(n - 1, sk + 30)
            cv2.imwrite(a.save_crop.replace("{preset}", name), cv2.cvtColor(np.concatenate([src[i], outs[i]], 1),
                                                                             cv2.COLOR_RGB2BGR))
    return r


def host_info():
    cpu = platform.processor() or ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        avail = len(os.sched_getaffinity(0))
    except AttributeError:
        avail = os.cpu_count()
    info = dict(cpu=cpu, nproc=os.cpu_count(), affinity=avail)
    try:
        info["load1"] = round(os.getloadavg()[0], 2)
    except OSError:
        pass
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", help="directory of PNG frames")
    ap.add_argument("--render", metavar="CHAR", help="render a short talking clip of this character instead")
    ap.add_argument("--w", type=int, default=480)
    ap.add_argument("--h", type=int, default=270)
    ap.add_argument("--n", type=int, default=72, help="--render: frames to render")
    ap.add_argument("--backend", default="auto", help="--render: auto | rt | v2 | sprite | raster | package.module:Class")
    ap.add_argument("--render-opts", default="{}", help="--render: the backend's options as JSON")
    ap.add_argument("--save-frames", help="--render: also save the rendered PNGs here (reuse with --frames)")
    ap.add_argument("--size", help="encoded size WxH (default: the frames' size)")
    ap.add_argument("--bitrate", type=int, default=250000)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--gop-s", type=float, default=2.0)
    ap.add_argument("--threads", type=int, default=4, help="encoder threads (the server's --enc-threads)")
    ap.add_argument("--encoder", default="libx264")
    ap.add_argument("--presets", default="ultrafast,default", help="comma list: x264 preset names and/or 'default'")
    ap.add_argument("--profile", help="AVATAR_X264_PROFILE for the run (default: the encoder's baseline)")
    ap.add_argument("--roi", help="face region x0,y0,x1,y1 in encoded pixels (default: the centre 30%% x 40%%, upper half)")
    ap.add_argument("--min-frames", type=int, default=0, help="ping-pong the clip to at least this many frames")
    ap.add_argument("--warm", type=int, default=12, help="frames left out of the encode-ms statistics")
    ap.add_argument("--score-from", type=int, default=None)
    ap.add_argument("--no-pace", dest="pace", action="store_false")
    ap.add_argument("--timing-only", action="store_true", help="no decode / scoring (timing and bytes only)")
    ap.add_argument("--repeat", type=int, default=1, help="run the preset list this many times (interleaved)")
    ap.add_argument("--save-crop", help="PNG path with {preset}: source | decoded side by side of one scored frame")
    ap.add_argument("--label", default="")
    ap.add_argument("--json")
    a = ap.parse_args()
    os.environ.pop("AVATAR_X264_PRESET", None)
    if a.profile:
        os.environ["AVATAR_X264_PROFILE"] = a.profile

    if a.render:
        frames = render_frames(a.render, a.w, a.h, a.n, a.backend, json.loads(a.render_opts))
        if a.save_frames:
            from PIL import Image
            os.makedirs(a.save_frames, exist_ok=True)
            for i, f in enumerate(frames):
                Image.fromarray(f).save(os.path.join(a.save_frames, f"{i:03d}.png"))
    elif a.frames:
        frames = load_frames(a.frames)
    else:
        ap.error("--frames or --render")
    sh, sw = frames[0].shape[:2]
    w, h = (int(v) for v in a.size.lower().split("x")) if a.size else (sw, sh)
    if (w, h) != (sw, sh):
        import cv2
        frames = [np.ascontiguousarray(cv2.resize(np.ascontiguousarray(f), (w, h), interpolation=cv2.INTER_AREA))
                  for f in frames]
    if a.roi:
        a.roi_box = [int(v) for v in a.roi.split(",")]
    else:
        a.roi_box = [int(w * 0.35), int(h * 0.1), int(w * 0.65), int(h * 0.5)]
    threads = a.threads
    src = ping_pong(frames, a.min_frames)
    res = dict(label=a.label, host=host_info(), clip_frames=len(frames), frames=len(src), size=f"{w}x{h}", bitrate=a.bitrate,
               fps=a.fps, paced=a.pace, roi=a.roi_box, presets={})
    names = [p.strip() for p in a.presets.split(",") if p.strip()]
    for rep in range(a.repeat):
        for name in names:
            r = run_preset(name, src, a, w, h, threads)
            key = name if a.repeat == 1 else f"{name}#{rep}"
            res["presets"][key] = r
            print(json.dumps({"run": key, **r}), file=sys.stderr, flush=True)
    out = json.dumps(res, indent=1)
    if a.json:
        with open(a.json, "w") as f:
            f.write(out)
    print(out)


if __name__ == "__main__":
    main()
