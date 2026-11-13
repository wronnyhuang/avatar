#!/usr/bin/env python3
"""Render every system's output on the Muse (CPU) and package it for the judge: $T/eval/runs/<run>/<clip>/avatar.mp4.

  python -I evalkit/render.py --runs rules,gru-prior,unified,... [--ids ..] [--jobs 4] [--threads 8]

Rig systems (rules, gru-prior, a2f, rules+a2f-mouth): the avatar repo's renderer/tools/render_timeline.py (LiveCore lip sync from the avatar's
voice + RigMixer + the real-time engine) -- the rules-eval path; a run dir with lip_override.npz (a2f) goes through
evalkit/render_lipover.py = the same script with that lip source replacing lipstream's columns. Canon systems (unified, copy-delay, human): RETARGET's
retarget/render_canon.py (CANON v1 -> the Muse's rig controls -> the same renderer, the same lip sync from the same
voice). Same character, size (640x360), fps (24), quality preset (q50 = the Muse's live bench profile) and the same
voice.wav for every system, so the only difference is the motion. avatar.mp4 = render at 30 fps with the avatar's
voice (the judge listens to it); avatar_rows.npy = the per-frame rig parameter rows the renderer actually used.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import E, T, WT, AVATAR, RENDER_PY, VENV_PY, atomic_json  # noqa: E402

W, H, FPS_R, PRESET = 640, 360, 24, "q50"
CHAR = "muse"
RENDER_CANON = [WT / "retarget/render_canon.py"]
KIT_DIR = Path(__file__).resolve().parent


def manifest():
    return {c["id"]: c for c in json.loads((E / "clips/manifest.json").read_text())["clips"]}


def package(od: Path, clip: dict):
    """render.mp4 -> avatar.mp4 (30 fps, H.264, the avatar voice as audio, exact clip length)."""
    D = float(clip["duration_s"])
    voice = E / "clips" / clip["id"] / "voice.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(od / "render.mp4"), "-i", str(voice), "-map", "0:v:0",
                    "-map", "1:a:0", "-vf", "tpad=stop_mode=clone:stop_duration=2,fps=30,format=yuv420p", "-t", f"{D:.3f}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "96k", "-ac", "1",
                    "-ar", "48000", str(od / "avatar.mp4")], check=True)


def render_one(run: str, clip: dict, threads: int, force=False):
    od = E / "runs" / run / clip["id"]
    if (od / "avatar.mp4").exists() and not force:
        return run, clip["id"], "cached"
    voice = E / "clips" / clip["id"] / "voice.wav"
    D = float(clip["duration_s"])
    t0 = time.time()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": str(threads)}
    if (od / "rig.jsonl").exists():
        # a lip_override.npz (e.g. Audio2Face-3D's mouth) replaces lipstream's lip columns: same renderer, wrapped
        rt = ([KIT_DIR / "render_lipover.py", od / "lip_override.npz", "--"] if (od / "lip_override.npz").exists()
              else [AVATAR / "renderer/tools/render_timeline.py"])
        cmd = [RENDER_PY, "-B", *rt, "--character", CHAR, "--rig", od / "rig.jsonl",
               "--audio", voice, "--w", W, "--h", H, "--fps", FPS_R, "--backend", "cpu", "--threads", threads,
               "--preset", PRESET, "--duration", f"{D:.3f}", "--out", od / "render.mp4", "--params-out", od / "avatar_rows.npy"]
        cwd = AVATAR / "renderer"
    elif (od / "canon.npz").exists():
        rc = next((p for p in RENDER_CANON if p.exists()), None)
        if rc is None:
            return run, clip["id"], "render_canon.py not available yet"
        cmd = [VENV_PY, "-I", rc,
               "--canon", od / "canon.npz", "--character", CHAR, "--audio", voice, "--out", od / "render.mp4",
               "--w", W, "--h", H, "--fps", FPS_R, "--preset", PRESET, "--threads", threads,
               "--duration", f"{D:.3f}", "--rig-out", od / "rig.jsonl"]
        cwd = rc.parent
    else:
        return run, clip["id"], "no output to render"
    r = subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, capture_output=True, text=True)
    (od / "render.log").write_text(r.stdout[-4000:] + "\n" + r.stderr[-4000:])
    if r.returncode:
        return run, clip["id"], f"render failed: {r.stderr[-600:]}"
    package(od, clip)
    atomic_json(od / "render.json", {"run": run, "clip": clip["id"], "character": CHAR, "w": W, "h": H, "fps": FPS_R,
                                     "preset": PRESET, "seconds": round(time.time() - t0, 1), "cmd": list(map(str, cmd))})
    return run, clip["id"], f"ok {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True); ap.add_argument("--ids"); ap.add_argument("--group")
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    man = manifest()
    work = []
    for run in a.runs.split(","):
        rd = E / "runs" / run
        if not rd.exists():
            continue
        for cid in sorted(os.listdir(rd)):
            if cid in man and (not a.ids or cid in a.ids.split(",")) and (not a.group or man[cid]["group"] == a.group):
                work.append((run, man[cid]))
    with cf.ThreadPoolExecutor(a.jobs) as pool:
        for f in cf.as_completed([pool.submit(render_one, r, c, a.threads, a.force) for r, c in work]):
            print(*f.result(), flush=True)


if __name__ == "__main__":
    main()
