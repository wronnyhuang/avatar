#!/usr/bin/env python3
"""CANON v1 motion -> rig timeline -> mp4 through the avatar's own renderer on the CPU (docs/CONTRACT.md §5).

One clip:
  python -I retarget/render_canon.py --canon x.npz|x.npy [--person 0] [--start 0] [--dur 10] --character muse \\
         [--audio voice.wav] [--talk auto|vad|audio|none] --out clip.mp4 [--w 640 --h 360] [--threads 8]
Many clips in parallel (one JSON object per line with the same keys: canon, out, character, audio, person, start,
dur, talk, w, h, label):
  python -I retarget/render_canon.py --jobs jobs.jsonl --parallel 4 --threads 8      # <= 40 CPU threads in total

Input: a session npz (canon1/spec.py save_session: motion (P,T,189), mask (P,T,9), audio (P,T,42)) -> person
--person; or a bare .npy (T,189) / (P,T,189).  --audio is the AVATAR's voice for lip sync (the renderer's lipstream
makes the mouth); it is cut to the same --start/--dur.  Talking (lip sync owns the mouth, retarget/compositor.py):
--talk auto = the npz person's own vad column if present, else the --audio energy VAD, else never.
Output: mp4 (H.264 crf 24 + AAC when --audio, 30 fps, 640x360 default = phone-friendly) + <out>.rig.jsonl (the rig
timeline it rendered, render_timeline.py format) + <out>.json (what was done, timings).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from canon1 import spec  # noqa: E402
from retarget.retarget import Retargeter  # noqa: E402
from retarget.compositor import Compositor  # noqa: E402

# the avatar repo checkout whose renderer/ draws the clips (default: the repo this motion-model/ lives in) and the
# interpreter that runs its renderer (default: this one)
AVATAR = os.environ.get("RETARGET_AVATAR_ROOT", os.path.dirname(REPO))
AV_PY = os.environ.get("RETARGET_AVATAR_PY", sys.executable)


def load_canon(path, person=0):
    """-> motion (T,189) float32, mask (T,9) or None, vad (T,) or None, meta dict."""
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=False)
        mot = d["motion"].astype(np.float32)
        msk = d["mask"] if "mask" in d.files else None
        aud = d["audio"].astype(np.float32) if "audio" in d.files else None
        meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
        if mot.ndim == 2:
            mot = mot[None]
            msk = None if msk is None else msk[None]
            aud = None if aud is None else aud[None]
        m = mot[person]
        k = None if msk is None else msk[person]
        vad = None if aud is None else aud[person, :, 41]
        return m, k, vad, meta
    a = np.load(path, allow_pickle=False).astype(np.float32)
    if a.ndim == 3:
        a = a[person]
    return a, None, None, {}


def wav_cut(src, start, dur, dst):
    import soundfile as sf
    info = sf.info(src)
    sr = info.samplerate
    a0 = int(round(start * sr))
    n = -1 if dur is None else int(round(dur * sr))
    x, sr = sf.read(src, start=a0, frames=n, dtype="float32", always_2d=True)
    sf.write(dst, x.mean(1), sr, subtype="PCM_16")
    return x.mean(1), sr


def talk_from_audio(x, sr, T):
    x16 = spec.resample_to_16k(np.asarray(x, np.float32), sr)
    f = spec.audio_features(x16, n_frames=T)
    return f[:, 41]


def build_rig(motion, mask, character, talk, seed=0):
    rt = Retargeter(character, seed=seed)
    comp = Compositor(rt, planner=None)
    T = len(motion)
    rows = []
    names = None
    for i in range(T):
        r = comp.step(motion[i], talking=bool(talk[i]) if talk is not None else False,
                      mask=None if mask is None else mask[i])
        if names is None:
            names = list(r.keys())
        rows.append([r[n] for n in names])
    return names, np.asarray(rows, np.float32), comp.stats()


def render_one(job, threads=8, quiet=False):
    t0 = time.time()
    canon = job["canon"]; out = os.path.abspath(job["out"])
    character = job.get("character", "muse")
    person = int(job.get("person", 0))
    start = float(job.get("start", 0.0) or 0.0)
    dur = job.get("dur")
    dur = None if dur in (None, "", 0) else float(dur)
    w, h = int(job.get("w", 640)), int(job.get("h", 360))
    motion, mask, vad, meta = load_canon(canon, person)
    j0 = int(round(start * spec.FPS))
    j1 = len(motion) if dur is None else min(len(motion), j0 + int(round(dur * spec.FPS)))
    motion = motion[j0:j1]
    mask = None if mask is None else mask[j0:j1]
    vad = None if vad is None else vad[j0:j1]
    T = len(motion)
    if T == 0:
        raise ValueError(f"no frames in {canon} at start {start}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmpd = tempfile.mkdtemp(prefix="rc_", dir=os.path.dirname(out))
    audio = job.get("audio")
    wav = None
    talk_src = "none"
    talk = None
    if audio:
        wav = os.path.join(tmpd, "voice.wav")
        x, sr = wav_cut(audio, start, T / spec.FPS, wav)
    mode = job.get("talk", "auto")
    if mode in ("auto", "vad") and vad is not None:
        talk, talk_src = vad > 0.5, "npz vad"
    elif mode in ("auto", "audio") and wav is not None:
        talk, talk_src = talk_from_audio(x, sr, T) > 0.5, "audio energy vad"
    t1 = time.time()
    names, rows, cst = build_rig(motion, mask, character, talk, seed=int(job.get("seed", 0)))
    t_rig = time.time() - t1
    rig_path = out[:-4] + ".rig.jsonl" if out.endswith(".mp4") else out + ".rig.jsonl"
    with open(rig_path, "w") as f:
        f.write(json.dumps({"rig_params": names}) + "\n")
        for i in range(T):
            f.write(json.dumps({"t": round(i / spec.FPS, 4), "a": [round(float(v), 4) for v in rows[i]]}) + "\n")
    cmd = [AV_PY, "-I", os.path.join(AVATAR, "renderer", "tools", "render_timeline.py"), "--character", character,
           "--rig", rig_path, "--w", str(w), "--h", str(h), "--backend", "cpu", "--threads", str(threads),
           "--fps", "30", "--duration", f"{T / spec.FPS:.4f}", "--out", out, "--crf", str(int(job.get("crf", 24)))]
    if wav:
        cmd += ["--audio", wav]
    env = dict(os.environ, LP_NUM_THREADS=str(threads), OMP_NUM_THREADS="1")
    t2 = time.time()
    p = subprocess.run(["nice", "-n", "10"] + cmd, cwd=os.path.join(AVATAR, "renderer"), env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"render_timeline failed ({p.returncode}): {p.stderr[-800:]}")
    rinfo = {}
    for line in p.stdout.splitlines()[::-1]:
        try:
            rinfo = json.loads(line); break
        except ValueError:
            continue
    try:
        for fn in os.listdir(tmpd):
            os.unlink(os.path.join(tmpd, fn))
        os.rmdir(tmpd)
    except OSError:
        pass
    rep = {"out": out, "rig": rig_path, "canon": canon, "person": person, "start": start, "frames": T,
           "character": character, "size": [w, h], "audio": audio, "talk": talk_src,
           "talk_frac": None if talk is None else round(float(np.mean(talk)), 3), "label": job.get("label"),
           "retarget_s": round(t_rig, 2), "retarget_ms_per_frame": round(1000 * t_rig / T, 3),
           "compositor": {k: cst[k] for k in ("ms_p50", "ms_p95")}, "render": rinfo,
           "render_s": round(time.time() - t2, 1), "total_s": round(time.time() - t0, 1)}
    json.dump(rep, open(out[:-4] + ".json" if out.endswith(".mp4") else out + ".json", "w"), indent=1)
    if not quiet:
        print(json.dumps(rep), flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canon")
    ap.add_argument("--person", type=int, default=0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float)
    ap.add_argument("--character", default="muse")
    ap.add_argument("--audio")
    ap.add_argument("--talk", default="auto", choices=["auto", "vad", "audio", "none"])
    ap.add_argument("--out")
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=360)
    ap.add_argument("--crf", type=int, default=24)
    ap.add_argument("--threads", type=int, default=8, help="llvmpipe threads per clip")
    ap.add_argument("--jobs", help="jsonl of jobs (batch mode)")
    ap.add_argument("--parallel", type=int, default=4, help="clips rendered at once (parallel x threads <= 40)")
    a = ap.parse_args()
    if a.jobs:
        if a.parallel * a.threads > 40:
            sys.exit(f"--parallel {a.parallel} x --threads {a.threads} > 40 CPU threads (the task's cap)")
        jobs = [json.loads(l) for l in open(a.jobs) if l.strip() and not l.startswith("#")]
        ok = err = 0
        with cf.ThreadPoolExecutor(a.parallel) as ex:
            futs = {ex.submit(render_one, j, a.threads, True): j for j in jobs}
            for fu in cf.as_completed(futs):
                j = futs[fu]
                try:
                    rep = fu.result(); ok += 1
                    print(json.dumps({"ok": True, "out": rep["out"], "frames": rep["frames"], "s": rep["total_s"]}), flush=True)
                except Exception as e:  # noqa: BLE001
                    err += 1
                    print(json.dumps({"ok": False, "out": j.get("out"), "error": str(e)[:400]}), flush=True)
        print(json.dumps({"done": ok, "failed": err}))
        sys.exit(1 if err else 0)
    if not (a.canon and a.out):
        ap.error("--canon and --out (or --jobs)")
    render_one({"canon": a.canon, "person": a.person, "start": a.start, "dur": a.dur, "character": a.character,
                "audio": a.audio, "talk": a.talk, "out": a.out, "w": a.w, "h": a.h, "crf": a.crf}, a.threads)


if __name__ == "__main__":
    main()
