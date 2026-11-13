#!/usr/bin/env python3
"""Render one animal across N processes x T threads, resumable, then mux.

  farm.py --animal fox --params params.npy --audio audio_in/clip.wav --procs 8 --threads 8 --w 1920 --h 1080 --spp 3

Frames land in out/<animal>/frames/%06d.png (written to a temp name, then renamed, so a killed run
leaves no half file; a re-run skips what exists).  Per-frame seconds go to out/<animal>/frames.log.
The cached background is baked once, before the workers start.  Output: out/<animal>/<animal>.mp4
(H.264 yuv420p, AAC, +faststart, audio and video both starting at 0).
"""
import argparse, multiprocessing as mp, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render  # noqa: E402


def worker(args, index, todo, frames_dir, log_path):
    from PIL import Image
    os.nice(args.nice)
    eng = render.Engine(args.animal, args.w, args.h, args.spp, args.backend, args.threads, args.bg_spp or None)
    arr = np.load(args.params, mmap_mode="r")
    with open(log_path, "a", buffering=1) as lg:
        for f in todo:
            path = os.path.join(frames_dir, f"{f:06d}.png")
            if os.path.exists(path):
                continue
            t0 = time.time()
            img = eng.render(f, np.array(arr[f]))
            tmp = path + f".tmp{os.getpid()}.png"
            Image.fromarray(img).save(tmp, compress_level=1)
            os.replace(tmp, path)
            lg.write(f"{f} {time.time()-t0:.3f} p{index}\n")


def mux(frames_dir, audio, out, fps, crf=17):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-start_number", "0",
           "-i", os.path.join(frames_dir, "%06d.png")]
    if audio:
        cmd += ["-i", audio]
    cmd += ["-map", "0:v:0"] + (["-map", "1:a:0"] if audio else [])
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]        # no -shortest: it drops the last video frame
    cmd += ["-movflags", "+faststart", out]     # mp4 edit lists start both streams at 0
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--audio", default="")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--spp", type=int, default=3)
    ap.add_argument("--bg-spp", type=int, default=0)
    ap.add_argument("--backend", default=os.environ.get("TA_BACKEND", "cpu"), choices=["cpu", "gpu"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--no-mux", action="store_true")
    ap.add_argument("--out-root", default="")
    a = ap.parse_args()
    if a.out_root:
        render.OUT_ROOT = a.out_root
    n = np.load(a.params, mmap_mode="r").shape[0]
    end = n if a.end < 0 else min(a.end, n)
    base = os.path.join(render.OUT_ROOT, a.animal)
    frames_dir = os.path.join(base, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for f in os.listdir(frames_dir):
        if ".tmp" in f:
            os.remove(os.path.join(frames_dir, f))
    log_path = os.path.join(base, "frames.log")
    missing = [f for f in range(a.start, end) if not os.path.exists(os.path.join(frames_dir, f"{f:06d}.png"))]
    print(f"{a.animal}: {end-a.start} frames, {len(missing)} to render", flush=True)
    t0 = time.time()
    if missing:
        # bake in a child so the parent never holds a GL context across fork
        ctx = mp.get_context("spawn")
        p = ctx.Process(target=bake, args=(a,)); p.start(); p.join()
        if p.exitcode != 0:
            raise SystemExit("background bake failed")
        procs = []
        for i in range(a.procs):
            p = ctx.Process(target=worker, args=(a, i, missing[i::a.procs], frames_dir, log_path))
            p.start(); procs.append(p)
        for p in procs:
            p.join()
        bad = [p.exitcode for p in procs if p.exitcode != 0]
        left = [f for f in range(a.start, end) if not os.path.exists(os.path.join(frames_dir, f"{f:06d}.png"))]
        if bad or left:
            raise SystemExit(f"{len(left)} frames missing, worker exits {bad}: re-run to resume")
    el = time.time() - t0
    print(f"rendered in {el:.1f}s ({el/max(len(missing),1):.2f} s/frame wall)", flush=True)
    if not a.no_mux and a.start == 0:
        out = os.path.join(base, f"{a.animal}.mp4")
        mux(frames_dir, a.audio, out, render.FPS, a.crf)
        print(out, flush=True)


def bake(a):
    render.Engine(a.animal, a.w, a.h, a.spp, a.backend, a.threads, a.bg_spp or None, log=True)


if __name__ == "__main__":
    main()
