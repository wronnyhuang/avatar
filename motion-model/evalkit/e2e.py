#!/usr/bin/env python3
"""EVAL: end-to-end "user moves -> avatar visibly changes" through the REAL server path, offline.

  python -I evalkit/e2e.py --unified unified:<bundle> [--clips a,b,c] [--probes smile,turn] [--onsets 8,18,28]
                           [--rules] [--out f.json]

unified: the integrated build's app/unified_worker.py (read-only) as a subprocess, driven over its own
stdin/stdout framing exactly as app/unified_link.py does: per 30 fps tick one `step` with the page's raw MediaPipe 'U'
messages (wire.pack_mp, captured at the page's model rates from the clip's live-rate extraction), 20 ms mic chunks
(+ sync), the avatar's own voice tap (24 kHz, ~2 frames ahead), and the page's rules rig vector. Measured:
  * worker round trip per tick (ms: pack -> pipe -> fitter -> planner -> retarget/compositor -> pipe), p50/p95/p99
  * reaction delay: a CONTROL run vs a PROBE run where, from onset t0 for 2 s, the user (a) smiles (raw blendshapes
    mouthSmile 0.85, cheek/eye squint) or (b) turns the head 25 deg (face matrix rotated about the vertical axis).
    First tick after t0 whose final rig differs from the control by a VISIBLE amount (head yaw/pitch/roll >= 1 deg
    = 0.0175 rad, or smile/brows/gaze/squint/eye_wide >= 0.05), and by ANY amount (> 1e-3, blinks/mouth excluded).
rules (--rules): the same probes on the JS controller (evalkit/js/replay.cjs, the integrated build's motion.js) by
editing the perception features from t0 (smile: blend + expr + xp scores 0.85; turn: yaw + matrix) — the page
controller's own perception smoothing is SKIPPED (so its number is a lower bound), and its tick cost is the node step.
Not measured (common to both, or network): camera -> MediaPipe on the page, page <-> server link for unified, rendering.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import E, WT, KIT, AVATAR, FPS, VENV_PY, atomic_json, read_wav_mono  # noqa: E402

sys.path.insert(0, str(AVATAR / "app"))
import wire  # noqa: E402
import react_worker as RW  # noqa: E402

VIS = {"head_yaw": 0.0175, "head_pitch": 0.0175, "head_roll": 0.0175, "smile": 0.05, "brow_l": 0.05, "brow_r": 0.05,
       "gaze_x": 0.05, "gaze_y": 0.05, "squint": 0.05, "eye_wide": 0.05}
EXCLUDE_ANY = ("blink", "jaw", "mouth", "lip", "tongue", "teeth", "round", "spread", "press", "labdent", "voice", "breath")
PROBE_S = 2.0
BS_NAMES = None


def bs_index():
    global BS_NAMES
    if BS_NAMES is None:
        from common import canon_spec
        BS_NAMES = canon_spec().BS
    return BS_NAMES


def manifest():
    return {c["id"]: c for c in json.loads((E / "clips/manifest.json").read_text())["clips"]}


def load_rules_rig(cid):
    lines = (E / "runs/rules" / cid / "rig.jsonl").read_text().splitlines()
    params = json.loads(lines[0])["rig_params"]
    rows = [json.loads(l) for l in lines[1:] if l.strip()]
    A = np.array([r["a"] for r in rows if "a" in r], np.float32)
    return params, A


def rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def perturb_mp(mp, probe, i0, i1):
    """copy of the mp_live arrays with the probe applied on frames [i0, i1) (only where a face is present)."""
    q = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in mp.items()}
    B = bs_index()
    for i in range(i0, min(i1, len(q["face"]))):
        if not q["face"][i]:
            continue
        if probe == "smile":
            for n, v in (("mouthSmileLeft", .85), ("mouthSmileRight", .85), ("cheekSquintLeft", .4),
                         ("cheekSquintRight", .4), ("eyeSquintLeft", .3), ("eyeSquintRight", .3)):
                q["bs"][i, B[n]] = v
        elif probe == "turn":
            M4 = q["mat"][i].reshape(4, 4).astype(np.float64)
            M4[:3, :3] = rot_y(25.0) @ M4[:3, :3]
            q["mat"][i] = M4.reshape(16).astype(np.float32)
    return q


# ----------------------------------------------------------------------------- the worker path (unified)
class Worker:
    def __init__(self, code_root, bundle, threads=2, extra=None):
        args = [str(VENV_PY), "-I", str(AVATAR / "app/unified_worker.py"), "--roots", f"{code_root}{os.pathsep}{WT}",
                "--fit-root", str(WT), "--threads", str(threads)]
        if bundle:
            args += ["--bundle", str(bundle)]
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), PYTHONDONTWRITEBYTECODE="1")
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(E / "logs/e2e_worker.err", "a"),
                                  env=env)
        self.info, _ = RW.read_frame(self.p.stdout)
        self.id = 0

    def call(self, hdr, blobs=()):
        self.id += 1
        hdr = dict(hdr, id=self.id)
        RW.write_frame(self.p.stdin, hdr, blobs)
        return RW.read_frame(self.p.stdout)

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.p.kill()


def drive(w, cid, mp, params, page_rig, n_ticks, mic24, voice24):
    """one session over the worker: returns (final rig (n, P), per-tick round trip ms, stats)."""
    w.call({"op": "reset", "character": "muse", "rig_params": params, "blinks": "model"})
    wh = tuple(int(x) for x in mp["wh"])
    mic_seq = 0
    own_sent = 0
    out, rtt = [], []
    sync_sent = False
    lead = 2                                   # the voice tap leads the tick by ~2 frames
    for j in range(n_ticks):
        hdr = {"op": "step", "character": "muse", "mp": [], "mic": [], "own": [], "sync": [], "ticks": []}
        blobs = []
        t = j / FPS + 0.001
        if mp["face_fresh"][j]:
            mat_cm = mp["mat"][j].reshape(4, 4).T.reshape(16)      # row-major -> JS column-major
            blobs.append(wire.pack_mp(wire.U_FACE, t, wh[0], wh[1], bool(mp["face"][j]), face_bs=mp["bs"][j], face_mat=mat_cm))
            hdr["mp"].append([wire.U_FACE, t])
        if mp["pose_fresh"][j]:
            ok = bool(mp["pose_ok"][j])
            blobs.append(wire.pack_mp(wire.U_POSE, t, wh[0], wh[1], ok, pose_image=mp["pose"][j], pose_world=mp["pose_world"][j]))
            hdr["mp"].append([wire.U_POSE, t])
        if mp["hand_fresh"][j]:
            hs = [(int(mp["hand_label"][j][k]) if mp["hand_label"][j][k] >= 0 else 2, float(mp["hand_score"][j][k]),
                   mp["hands"][j][k], mp["hands_world"][j][k]) for k in range(int(mp["hand_n"][j]))]
            blobs.append(wire.pack_mp(wire.U_HANDS, t, wh[0], wh[1], bool(hs), hands=hs))
            hdr["mp"].append([wire.U_HANDS, t])
        if not sync_sent:
            hdr["sync"].append([0, 0.02]); sync_sent = True
        while (mic_seq + 1) * 0.02 <= (j + 1) / FPS + 1e-9:          # 20 ms mic chunks up to the tick's end
            x = mic24[mic_seq * 480:(mic_seq + 1) * 480]
            blobs.append((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
            hdr["mic"].append([mic_seq, 24000]); mic_seq += 1
        own_to = int(min(len(voice24), round((j + 1 + lead) / FPS * 24000)))
        if own_to > own_sent:
            blobs.append((np.clip(voice24[own_sent:own_to], -1, 1) * 32767).astype("<i2").tobytes())
            hdr["own"].append(own_sent); own_sent = own_to
        hdr["ticks"].append([j / FPS, 0, 0.0])
        blobs.append(np.asarray(page_rig[min(j, len(page_rig) - 1)], np.float32).tobytes())
        blobs.append(b"")
        if j == n_ticks - 1:
            hdr["want_stats"] = True
        t0 = time.perf_counter()
        rh, rb = w.call(hdr, blobs)
        rtt.append((time.perf_counter() - t0) * 1000)
        if rh.get("error"):
            raise RuntimeError(f"worker error at tick {j}: {rh['error']}")
        v = np.frombuffer(rb[0], np.float32).reshape(rh["n"], -1)
        out.append(v[-1] if len(v) else np.full(len(params), np.nan, np.float32))
        stats = rh.get("stats")
    return np.stack(out), np.array(rtt), stats


def first_change(ctrl, pert, params, j0, j1):
    """(first visible tick offset, first any-change tick offset, pre-onset max diff) in ticks from j0."""
    d = np.abs(pert[:j1] - ctrl[:j1])
    pre = float(np.nanmax(d[:j0])) if j0 > 0 else 0.0
    vis_cols = [(i, VIS[p]) for i, p in enumerate(params) if p in VIS]
    any_cols = [i for i, p in enumerate(params) if not any(x in p for x in EXCLUDE_ANY)]
    fv = fa = None
    for j in range(j0, j1):
        if fa is None and np.nanmax(d[j, any_cols]) > max(1e-3, 2 * pre):
            fa = j - j0
        if fv is None and any(d[j, i] >= th + pre for i, th in vis_cols):
            fv = j - j0
        if fv is not None and fa is not None:
            break
    which = None
    if fv is not None:
        j = j0 + fv
        which = [params[i] for i, th in vis_cols if d[j, i] >= th + pre]
    return fv, fa, pre, which


# ----------------------------------------------------------------------------- rules (node replay)
def perturb_perception(rows, probe, t0, t1):
    out = []
    for r in rows:
        r = json.loads(json.dumps(r)) if t0 <= r["t"] < t1 else r
        f = (r.get("features") or {}).get("face") or {}
        if t0 <= r["t"] < t1 and f.get("present"):
            if probe == "smile":
                for n, v in (("mouthSmileLeft", .85), ("mouthSmileRight", .85), ("cheekSquintLeft", .4),
                             ("cheekSquintRight", .4), ("eyeSquintLeft", .3), ("eyeSquintRight", .3)):
                    if "blend" in f:
                        f["blend"][n] = v
                if "expr" in f:
                    f["expr"]["smile"] = .85; f["expr"]["squint"] = max(f["expr"].get("squint", 0), .35)
                if "xp" in f and f["xp"]:
                    f["xp"]["state"] = "happy"; f["xp"]["intensity"] = .85
                    f["xp"].setdefault("scores", {})["happy"] = .85
            elif probe == "turn":
                f["yaw"] = float(f.get("yaw", 0)) + math.radians(25.0)
                if f.get("matrix"):
                    M4 = np.array(f["matrix"], float).reshape(4, 4, order="F")
                    M4[:3, :3] = rot_y(25.0) @ M4[:3, :3]
                    f["matrix"] = [round(float(x), 5) for x in M4.flatten(order="F")]
        out.append(r)
    return out


def rules_probe(clip, probe, onset, work):
    from systems import run_rig, clip_dir
    cd = clip_dir(clip["id"])
    rows = [json.loads(l) for l in (cd / "perception.jsonl").read_text().splitlines() if l.strip()]
    pr = perturb_perception(rows, probe, onset, onset + PROBE_S)
    idir = work / f"{clip['id']}_{probe}_{int(onset)}"
    idir.mkdir(parents=True, exist_ok=True)
    with open(idir / "perception.jsonl", "w") as f:
        for r in pr:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    os.link(cd / "mic.wav", idir / "mic.wav") if not (idir / "mic.wav").exists() else None
    od = idir / "out"
    if (od / "rig.jsonl").exists():
        (od / "rig.jsonl").unlink()
    run_rig(clip, "rules", od, idir)
    lines = (od / "rig.jsonl").read_text().splitlines()
    params = json.loads(lines[0])["rig_params"]
    return params, np.array([json.loads(l)["a"] for l in lines[1:] if l.strip() and '"a"' in l], np.float32)


# ----------------------------------------------------------------------------- main
def pctl(a, q):
    a = np.asarray(a, float)
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unified", default="")
    ap.add_argument("--clips", default="")
    ap.add_argument("--probes", default="smile,turn")
    ap.add_argument("--onsets", default="8,18,28")
    ap.add_argument("--rules", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out")
    a = ap.parse_args()
    man = manifest()
    if a.clips:
        clips = a.clips.split(",")
    else:   # default: 4 Seamless clips with the user's face visible most of the time
        clips = []
        for cid in sorted(c for c, v in man.items() if v["group"] == "seamless"):
            rc = json.loads((E / "clips" / cid / "receipt.json").read_text())
            if rc.get("face_frac", 0) > 0.9:
                clips.append(cid)
        clips = clips[::max(1, len(clips) // 4)][:4]
    probes = a.probes.split(",")
    onsets = [float(x) for x in a.onsets.split(",")]
    res = {"clips": clips, "probes": probes, "onsets_s": onsets, "probe_s": PROBE_S, "visible": VIS,
           "host_load": open("/proc/loadavg").read().split()[:3], "trials": []}
    (E / "logs").mkdir(parents=True, exist_ok=True)
    if a.unified:
        import bundles
        meta = bundles.resolve(a.unified)
        res["unified"] = meta
        w = Worker(meta["code"], meta["bundle"], a.threads)
        res["worker_info"] = w.info
        rtts = []
        try:
            for cid in clips:
                c = man[cid]
                z = np.load(E / "clips" / cid / "mp_live.npz")
                mp = {k: z[k] for k in z.files}
                params, page = load_rules_rig(cid)
                mic, _ = read_wav_mono(E / "clips" / cid / "mic.wav")
                voice, _ = read_wav_mono(E / "clips" / cid / "voice.wav")
                N = min(len(mp["face"]), len(page), int(round(float(c["duration_s"]) * FPS)))
                ctrl, rtt, stats = drive(w, cid, mp, params, page, N, mic, voice)
                rtts.append(rtt[30:])
                res.setdefault("worker_stats", {})[cid] = stats
                for probe in probes:
                    for t0 in onsets:
                        j0 = int(round(t0 * FPS)); j1 = min(N, j0 + int(3.0 * FPS))
                        if j0 >= N - 10:
                            continue
                        pm = perturb_mp(mp, probe, j0, j0 + int(PROBE_S * FPS))
                        pert, rtt2, _ = drive(w, cid, pm, params, page, j1, mic, voice)
                        rtts.append(rtt2[30:])
                        fv, fa, pre, which = first_change(ctrl, pert, params, j0, j1)
                        tr = {"system": "unified", "clip": cid, "probe": probe, "onset_s": t0,
                              "face_at_onset": bool(mp["face"][j0]),
                              "visible_ticks": fv, "any_ticks": fa, "pre_onset_maxdiff": pre, "visible_channels": which}
                        res["trials"].append(tr)
                        print(json.dumps(tr), flush=True)
        finally:
            w.close()
        r = np.concatenate(rtts) if rtts else np.array([])
        res["worker_rtt_ms"] = {"p50": pctl(r, 50), "p95": pctl(r, 95), "p99": pctl(r, 99), "max": pctl(r, 100), "n": int(len(r))}
    if a.rules:
        work = E / "e2e_rules"
        for cid in clips:
            c = man[cid]
            params, ctrl = load_rules_rig(cid)
            for probe in probes:
                for t0 in onsets:
                    j0 = int(round(t0 * FPS)); j1 = min(len(ctrl), j0 + int(3.0 * FPS))
                    p2, pert = rules_probe(c, probe, t0, work)
                    fv, fa, pre, which = first_change(ctrl, pert, p2, j0, j1)
                    tr = {"system": "rules", "clip": cid, "probe": probe, "onset_s": t0, "visible_ticks": fv,
                          "any_ticks": fa, "pre_onset_maxdiff": pre, "visible_channels": which}
                    res["trials"].append(tr)
                    print(json.dumps(tr), flush=True)
        st = [json.loads(p.read_text())["step_ms"] for p in (E / "runs/rules").glob("*/rig.jsonl.stats.json")]
        if st:
            res["rules_step_ms"] = {"p50_med": float(np.median([s["p50"] for s in st])),
                                    "p99_med": float(np.median([s["p99"] for s in st]))}
    # summary per system x probe
    summ = {}
    for sysn in ("unified", "rules"):
        for probe in probes:
            T = [t for t in res["trials"] if t["system"] == sysn and t["probe"] == probe]
            if not T:
                continue
            vis = [t["visible_ticks"] for t in T]
            got = [v for v in vis if v is not None]
            summ[f"{sysn}/{probe}"] = {"n": len(T), "visible_n": len(got),
                                       "visible_med_ms": float(np.median(got)) * 1000 / FPS if got else None,
                                       "visible_max_ms": float(np.max(got)) * 1000 / FPS if got else None,
                                       "any_med_ms": (float(np.median([t["any_ticks"] for t in T if t["any_ticks"] is not None])) * 1000 / FPS
                                                      if any(t["any_ticks"] is not None for t in T) else None)}
    res["summary"] = summ
    out = Path(a.out) if a.out else E / "measures" / "e2e.json"
    atomic_json(out, res)
    print(json.dumps({"summary": summ, "worker_rtt_ms": res.get("worker_rtt_ms"), "rules_step_ms": res.get("rules_step_ms")}, indent=1))


if __name__ == "__main__":
    main()
