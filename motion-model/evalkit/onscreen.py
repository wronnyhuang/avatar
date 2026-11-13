#!/usr/bin/env python3
"""What the renderer actually SHOWS for a system's rig stream: render_timeline.py's steps 1-2 without drawing.

  rows = onscreen_rows(rig_jsonl, lip)        # (N, NPARAMS) at 30 fps, anim.schema.PARAMS order
  lip  = lip_rows(clip_id)                    # the avatar voice through lipstream (cached per clip)

The live path (avatar repo renderer/tools/render_timeline.py == avatar/render_server.py):
  1. lips: LiveCore + StreamingLipAnimator(character, lookahead 133 ms) on the avatar's voice -> one row per 30 fps frame;
     its LIP_OWNED columns (jaw, round, spread, press, labdent, tongue*, cheek, voice, lip_*, teeth_gap) are the mouth.
  2. page rig: RigMixer (critically damped smoothing of the page columns) + anim.rig.combine: page columns replace the
     row's, smile = page smile + 0.25 * lipstream's own (life) smile, jaw / teeth_gap / lip_open = max(lipsync, mouth_open).
Page side (app/motion.js): LIPSYNC columns are left 0 by the controller; SPEECH_GROUP (mouth_open, tongue_out) of a keyframe
job / listener / program yields to speech while voiced (rules' own mouth_open is 0 while it speaks), so while the avatar
talks the jaw and lips on screen are lipstream's unless a renderer-side override replaces them (lip_override below).

lip_override (a2f): {"cols": [names], "values": (N, k) at 30 fps, "w": (N,) 0..1} -> the lip row's columns
are lerped to the override with weight w BEFORE the page rig is combined (= a lip source that replaces lipstream).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

# self-contained (same defaults as evalkit/common.py): importable from tools/audio, whose own `common` shadows evalkit's
_MM = Path(__file__).resolve().parents[1]
T = Path(os.environ.get("EVAL_T", os.environ.get("MM_WORK_DIR", str(_MM / "work"))))
E = Path(os.environ.get("EVAL_OUT", str(T / "eval")))
AVATAR = Path(os.environ.get("EVAL_AVATAR", str(_MM.parent)))
FPS = 30

REND = AVATAR / "renderer"
for _p in (REND, REND / "engine", REND / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
CHAR = "muse"
LOOKAHEAD_MS = 133.0
LIP_VERSION = "lip-v1"
CACHE = Path(os.environ.get("ONSCREEN_CACHE", str(Path(os.environ.get("SCORECARD_ROOT", str(T / "p2/scorecards"))) / "_cache/onscreen")))


def _rt():
    import render_timeline as RT  # noqa: N812  (avatar repo, read-only)
    return RT


def schema():
    from anim.schema import IDX, NPARAMS, PARAMS
    return IDX, NPARAMS, PARAMS


def lip_rows_pcm(pcm24, n_frames, character=CHAR, lookahead_ms=LOOKAHEAD_MS):
    """int16 24 kHz voice -> (n_frames, NPARAMS) lipstream rows at 30 fps (row j = frame j, the server's Core at 30 fps)."""
    from live.core import LiveCore
    from anim.lipstream import StreamingLipAnimator
    from anim.schema import NPARAMS
    core = LiveCore(character, lookahead_ms=lookahead_ms, fps=30, resample="hold",
                    animator=StreamingLipAnimator(character, lookahead_ms=lookahead_ms))
    n_s = int(round(n_frames / FPS * 24000))
    pcm = np.asarray(pcm24, np.int16)
    pcm = np.concatenate([pcm, np.zeros(max(0, n_s - pcm.size), np.int16)])[:n_s]
    x = np.concatenate([pcm, np.zeros(int(24000 * (lookahead_ms / 1000.0 + 0.2)), np.int16)])
    out = np.zeros((n_frames, NPARAMS), np.float32)
    got = np.zeros(n_frames, bool)
    for k in range(0, x.size, 2400):
        for jb in core.push(x[k:k + 2400]):
            if jb.j < n_frames:
                out[jb.j] = jb.params; got[jb.j] = True
    if not got.all():
        raise RuntimeError(f"lipstream released {got.sum()}/{n_frames} frames")
    return out


def lip_rows(cid, n_frames=None):
    """the clip's avatar voice (eval/clips/<cid>/voice.wav) through lipstream at 30 fps; cached."""
    p = CACHE / "_lip" / f"{cid}_{LIP_VERSION}.npy"
    if p.exists():
        return np.load(p)
    man = {c["id"]: c for c in json.loads((E / "clips/manifest.json").read_text())["clips"]}
    N = n_frames or int(round(float(man[cid]["duration_s"]) * FPS))
    rows = lip_rows_pcm(_rt().read_wav(str(E / "clips" / cid / "voice.wav")), N)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".tmp{os.getpid()}.npy")
    np.save(tmp, rows)
    os.replace(tmp, p)
    return rows


def apply_override(lip, ov):
    """lip rows (N, NPARAMS) with a lip_override dict/npz lerped in (returns a copy)."""
    if ov is None:
        return lip
    IDX = schema()[0]
    out = lip.copy()
    cols = [str(c) for c in ov["cols"]]
    vals = np.asarray(ov["values"], np.float32)
    w = np.asarray(ov["w"], np.float32)
    n = min(len(out), len(vals))
    ix = [IDX[c] for c in cols]
    out[:n, ix] = out[:n, ix] + (vals[:n] - out[:n, ix]) * w[:n, None]
    return out


def onscreen_rows(rig_path, lip, override=None, character=CHAR):
    """rig.jsonl (page rig stream, playback-timeline seconds) + lip rows -> the rows the renderer draws, 30 fps."""
    RT = _rt()
    from anim.rig import RigMixer, RIG_PARAMS
    rig = RT.read_rig(str(rig_path))
    lip = apply_override(lip, override)
    mix = RigMixer(character=character)
    cur = np.zeros(len(RIG_PARAMS), "f4")
    rows, k = [], 0
    for j in range(len(lip)):
        t = j / float(FPS)
        while k < len(rig) and rig[k][0] <= t + 1e-9:
            tr, kind, payload, names = rig[k]
            if kind == "a":
                mix.push_rig(tr, payload, now=tr, names=names)
                if names is None:
                    cur = np.asarray(payload, "f4")[:len(RIG_PARAMS)]
            else:
                for nm, v in payload.items():
                    if nm in RIG_PARAMS:
                        cur[RIG_PARAMS.index(nm)] = float(v)
                mix.push_rig(tr, cur, now=tr)
            k += 1
        rows.append(mix.apply(lip[j], t, now=t))
    return np.stack(rows).astype(np.float32)


def run_onscreen(run, cid, refresh=False):
    """cached on-screen rows of E/runs/<run>/<cid>: rig.jsonl (+ lip_override.npz) + the clip's lipstream rows."""
    od = E / "runs" / run / cid
    rp = od / "rig.jsonl"
    if not rp.exists():
        return None
    ovp = od / "lip_override.npz"
    st = rp.stat()
    key = f"{st.st_size}_{int(st.st_mtime)}" + (f"_{ovp.stat().st_size}_{int(ovp.stat().st_mtime)}" if ovp.exists() else "")
    p = CACHE / run / f"{cid}_{LIP_VERSION}_{key}.npy"
    if p.exists() and not refresh:
        return np.load(p)
    lip = lip_rows(cid)
    ov = dict(np.load(ovp, allow_pickle=False)) if ovp.exists() else None
    rows = onscreen_rows(rp, lip, ov)
    p.parent.mkdir(parents=True, exist_ok=True)
    for old in p.parent.glob(f"{cid}_*.npy"):
        old.unlink()
    tmp = p.with_suffix(f".tmp{os.getpid()}.npy")
    np.save(tmp, rows)
    os.replace(tmp, p)
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--ids")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    man = {c["id"]: c for c in json.loads((E / "clips/manifest.json").read_text())["clips"]}
    ids = a.ids.split(",") if a.ids else sorted(c for c, v in man.items() if v["group"] == "seamless")
    import concurrent.futures as cf
    import multiprocessing as mp
    with cf.ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as pool:
        futs = {pool.submit(lip_rows, c): c for c in ids}
        for f in cf.as_completed(futs):
            f.result()
        futs = {pool.submit(run_onscreen, r, c): (r, c) for r in a.runs.split(",") for c in ids}
        for f in cf.as_completed(futs):
            r = f.result()
            print(*futs[f], None if r is None else r.shape, flush=True)
