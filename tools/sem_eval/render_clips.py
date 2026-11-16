#!/usr/bin/env python3
"""Rig tracks recorded by run_live.cjs -> mp4 clips through the REAL renderer on CPU (offline, niced), + judge
contact sheets.

  python3 tools/sem_eval/render_clips.py --runs EVAL/runs/<tag> --out EVAL/clips/<tag> [--tree WT] [--only id,id]
        [--chars muse,fox] [--modes current,semantic] [--pass fresh] [--w 480 --h 270 --fps 24]
        [--par 3 --threads 4] [--juno-par 6 --juno-threads 2 --juno-lod lod1 --juno-fps 24] [--max-kb 400] [--force]

Input : <runs>/<char>/<mode>/<pass>/<id>.rig.jsonl  (+ <id>.json for the call/end times)
Output: <out>/<char>/<mode>/<id>.mp4      the page clip (H.264, yuv420p, faststart, <= --max-kb: crf ladder, then fps 20)
        <out>/<char>/<mode>/<id>.sheet.jpg judge sheet: 8 frames from the call to end + 0.5 s, cropped to the character
        <out>/render_log.jsonl            one row per clip: render_s, frames, kb, crf, errors
Renderer: <tree>/renderer/tools/render_timeline.py (the live server's RigMixer + engine; no audio -> lip columns from
the page's own rig values). Juno renders with JUNO_LOD (default lod1: ~2.2 s/frame on CPU at 480x270; lod0 ~4-5x slower).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import sys
import os
import shutil
import subprocess
import tempfile
import threading
import time

AV_PY = os.environ.get("AVATAR_PY") or sys.executable   # the renderer's python (cv2, moderngl)
DEF_TREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CROP = {"muse": (0.22, 0.78), "juno": (0.0, 1.0), "fox": (0.0, 1.0)}   # juno: the live page frames her close-up most of the time   # x-range of the frame the character occupies
LOCK = threading.Lock()


def probe_kb(p):
    return os.path.getsize(p) / 1024.0


def encode_small(src, dst, max_kb, fps):
    """re-encode to <= max_kb (crf ladder, then fps 20)"""
    for f in (fps, 20):
        for crf in (28, 31, 34, 37, 40):
            cmd = ["nice", "-n", "10", "ffmpeg", "-v", "error", "-y", "-i", src, "-an", "-c:v", "libx264", "-preset", "slow",
                   "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "2"]
            if f != fps:
                cmd += ["-r", str(f)]
            subprocess.run(cmd + [dst], check=True)
            if probe_kb(dst) <= max_kb:
                return {"crf": crf, "fps": f, "kb": round(probe_kb(dst), 1)}
    return {"crf": 40, "fps": 20, "kb": round(probe_kb(dst), 1), "over": True}


def sheet(mp4, out_jpg, t0, t1, char, n=8, cols=4, cell_h=300, tm=None):
    """8 cells: the first at the call (t0); the other 7 spread over the move itself (tm - 0.3 s .. t1) when its start
    tm is known (the thinking lead-in before it is then shown by the first cell only), else over t0..t1"""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(mp4)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if tm is not None and t1 - (tm - 0.3) > 0.5 and tm - 0.3 > t0:
        a0 = tm - 0.3
        ts = [t0] + [a0 + (t1 - a0) * i / (n - 2) for i in range(n - 1)]
    else:
        ts = [t0 + (t1 - t0) * i / (n - 1) for i in range(n)]
    x0, x1 = CROP.get(char, (0, 1))
    cells = []
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(0, int(round(t * fps))), max(0, nf - 1)))
        ok, fr = cap.read()
        if not ok:
            fr = np.zeros((270, 480, 3), np.uint8)
        h, w = fr.shape[:2]
        fr = fr[:, int(w * x0):int(w * x1)]
        s = cell_h / fr.shape[0]
        fr = cv2.resize(fr, (int(fr.shape[1] * s), cell_h), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(fr, (0, 0), (118, 26), (0, 0, 0), -1)
        cv2.putText(fr, f"t={t - t0:.1f}s", (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(fr)
    cap.release()
    rows = []
    for r in range(0, n, cols):
        row = cells[r:r + cols]
        while len(row) < cols:
            row.append(np.zeros_like(cells[0]))
        rows.append(np.hstack(row))
    cv2.imwrite(out_jpg, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return ts


def move_start(rj, t_call):
    try:
        fm = (json.load(open(rj)).get("timings") or {}).get("first_motion_ms")
        return t_call + fm / 1000.0 if fm is not None else None
    except (OSError, ValueError):
        return None


def render_one(job, a):
    rig, char, mode, cid, odir = job
    os.makedirs(odir, exist_ok=True)
    out = os.path.join(odir, cid + ".mp4")
    shp = os.path.join(odir, cid + ".sheet.jpg")
    row = {"char": char, "mode": mode, "id": cid, "rig": rig}
    if os.path.exists(out) and os.path.exists(shp) and not a.force:
        row["skipped"] = True
        return row
    meta = json.loads(open(rig).readline())
    if os.path.exists(out) and not a.force:                 # clip there, sheet missing: the sheet only
        try:
            dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out],
                                       capture_output=True, text=True).stdout.strip())
            t_call = float(meta.get("t_call_s") or 1.0)
            t_end = float(meta.get("t_end_s") or dur - 1.0)
            rj = os.path.join(os.path.dirname(rig), cid + ".json")
            tm = move_start(rj, t_call)
            tj = os.path.join(odir, cid + ".trim.json")
            if os.path.exists(tj):
                sh = json.load(open(tj))["trim_s"]
                t_call, t_end, tm = 1.0, t_end - sh, (tm - sh if tm is not None else None)
            row["sheet_t"] = [round(x, 2) for x in sheet(out, shp, t_call, min(dur - 0.05, t_end + 0.5), char, tm=tm)]
            row["sheet_only"] = True
        except Exception as e:  # noqa: BLE001
            row["error"] = f"sheet: {e}"
        return row
    rj = os.path.join(os.path.dirname(rig), cid + ".json")
    R = json.load(open(rj)) if os.path.exists(rj) else {}
    lines = sum(1 for _ in open(rig)) - 1
    last_t = 0.0
    with open(rig) as f:
        for ln in f:
            pass
        try:
            last_t = float(json.loads(ln)["t"])
        except Exception:  # noqa: BLE001
            pass
    dur = min(last_t, a.max_s)
    t_call = float(meta.get("t_call_s") or 1.0)
    t_end = meta.get("t_end_s")
    t_end = float(t_end) if t_end else dur - 1.0
    juno = char == "juno"
    fps = a.juno_fps if juno else a.fps
    th = a.juno_threads if juno else a.threads
    tmp = tempfile.mkdtemp(prefix="sem_rend_", dir=a.tmp)
    raw = os.path.join(tmp, "raw.mp4")
    # --trim-lead S: clips of commands NOT in --full-ids start S seconds before the planned move (the wait + thinking
    # lead-in before it is cut: judge-only clips render 30-60 % faster); the page's clips keep the whole window
    tm = move_start(rj, t_call)
    if a.trim_lead and cid not in a.full_ids and tm is not None and tm - a.trim_lead > 0.5:
        shift = tm - a.trim_lead
        trig = os.path.join(tmp, "rig.jsonl")
        with open(rig) as f, open(trig, "w") as g:
            hdr = json.loads(f.readline())
            hdr["t_call_s"], hdr["trim_s"] = 1.0, round(shift, 3)   # the first second = the offline camera settling: sheets skip it
            if hdr.get("t_end_s"):
                hdr["t_end_s"] = hdr["t_end_s"] - shift
            g.write(json.dumps(hdr) + "\n")
            for ln in f:
                o = json.loads(ln)
                if o["t"] >= shift:
                    o["t"] = round(o["t"] - shift, 4)
                    g.write(json.dumps(o) + "\n")
        rig, dur, t_call, t_end, tm = trig, dur - shift, 1.0, t_end - shift, tm - shift
        row["trim_s"] = round(shift, 2)
        json.dump({"trim_s": shift}, open(os.path.join(odir, cid + ".trim.json"), "w"))
    env = dict(os.environ, LP_NUM_THREADS=str(th), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if juno:
        env["JUNO_LOD"] = a.juno_lod
    cmd = ["nice", "-n", "10", AV_PY, "-I", "tools/render_timeline.py", "--character", char, "--rig", rig, "--w", str(a.w),
           "--h", str(a.h), "--backend", "cpu", "--threads", str(th), "--fps", str(fps), "--duration", f"{dur:.3f}",
           "--out", raw, "--crf", "18"]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=os.path.join(a.tree, "renderer"), env=env, capture_output=True, text=True, timeout=7200)
    row.update(render_s=round(time.time() - t0, 1), frames=int(dur * fps), dur_s=round(dur, 2), fps=fps, n_rig=lines)
    both = p.stdout + p.stderr
    if p.returncode != 0 or not os.path.exists(raw) or "TestRig" in both:
        row["error"] = f"render rc={p.returncode}: {both[-400:]}"
        shutil.rmtree(tmp, ignore_errors=True)
        return row
    try:
        row.update(encode_small(raw, out, a.max_kb, fps))
        row["sheet_t"] = [round(x, 2) for x in sheet(out, shp, t_call, min(dur - 0.05, t_end + 0.5), char, tm=tm)]
    except Exception as e:  # noqa: BLE001
        row["error"] = f"encode/sheet: {e}"
    shutil.rmtree(tmp, ignore_errors=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tree", default=DEF_TREE)
    ap.add_argument("--only", default="")
    ap.add_argument("--chars", default="muse,fox")
    ap.add_argument("--modes", default="current,semantic")
    ap.add_argument("--pass", dest="pass_", default="fresh")
    ap.add_argument("--w", type=int, default=480)
    ap.add_argument("--h", type=int, default=270)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--par", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--juno-par", type=int, default=6)
    ap.add_argument("--juno-threads", type=int, default=2)
    ap.add_argument("--juno-lod", default="lod1")
    ap.add_argument("--juno-fps", type=int, default=24)
    ap.add_argument("--max-kb", type=float, default=400)
    ap.add_argument("--max-s", type=float, default=16)
    ap.add_argument("--tmp", default="/tmp")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--trim-lead", type=float, default=0.0, help="judge-only clips start this many s before the move (0 = full)")
    ap.add_argument("--full-ids", default="", help="ids always rendered in full (the page's commands)")
    a = ap.parse_args()
    a.runs, a.out, a.tree, a.tmp = (os.path.abspath(x) for x in (a.runs, a.out, a.tree, a.tmp))
    only = set(filter(None, a.only.split(",")))
    a.full_ids = set(filter(None, a.full_ids.split(",")))
    jobs = {"juno": [], "other": []}
    for char in a.chars.split(","):
        for mode in a.modes.split(","):
            for rig in sorted(glob.glob(os.path.join(a.runs, char, mode, a.pass_, "*.rig.jsonl"))):
                cid = os.path.basename(rig)[:-len(".rig.jsonl")]
                if only and cid not in only:
                    continue
                jobs["juno" if char == "juno" else "other"].append((rig, char, mode, cid, os.path.join(a.out, char, mode)))
    order = [x for x in a.only.split(",") if x]                # --only order = priority; both modes of an id together
    for k in jobs:
        jobs[k].sort(key=lambda j: ((order.index(j[3]) if j[3] in order else len(order)), j[3], j[2], j[1]))
    os.makedirs(a.out, exist_ok=True)
    logp = os.path.join(a.out, "render_log.jsonl")
    t0 = time.time()

    def run(pool_jobs, par):
        with cf.ThreadPoolExecutor(max(1, par)) as ex:
            for row in ex.map(lambda j: render_one(j, a), pool_jobs):
                with LOCK:
                    with open(logp, "a") as f:
                        f.write(json.dumps(row) + "\n")
                print(json.dumps({k: row.get(k) for k in ("char", "mode", "id", "render_s", "frames", "kb", "crf", "error", "skipped")}),
                      flush=True)
    th = []
    for key, par in (("juno", a.juno_par), ("other", a.par)):
        if jobs[key]:
            t = threading.Thread(target=run, args=(jobs[key], par))
            t.start()
            th.append(t)
    for t in th:
        t.join()
    print(f"render_clips: {sum(len(v) for v in jobs.values())} clips in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
