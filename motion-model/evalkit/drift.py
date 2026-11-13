#!/usr/bin/env python3
"""EVAL: 5-minute closed-loop rollout of EVERY system on the same concatenated conversation -> drift / stability.

  python -I evalkit/drift.py build [--minutes 5]                       # E/clips/_long5 (once)
  python -I evalkit/drift.py run --system rules|gru-prior|copy-delay|human|unified:<bundle>[@style]
  python -I evalkit/drift.py measure --runs rules,gru-prior,copy-delay,human,unified-<tag> [--out f.json]

Input: the Seamless test clips (sorted) concatenated to `minutes` (user camera detections -> perception features and
the raw MediaPipe arrays; mic; the avatar's voice + words; the user's speech spans), fed causally to one instance of each
system (rules/gru-prior: node replay of the integrated build; copy-delay / unified: the planner with its own committed
output fed back; human = the real partners' canon concatenated, the reference for natural minute-to-minute variation).
Measures (rig space = what renders, tools/motion_stats.py conversion): per minute SD and mean of head yaw/pitch/roll,
gaze x/y, brows, smile; drift ratio = minute-k SD / minute-1 SD (min, max over k); mean-pose wander = max over minutes
of |minute mean - overall mean| (deg); longest frozen stretch (s); fraction of frames with |head yaw| or |pitch| > 30 deg;
finite. A system "drifts" if any SD ratio leaves [0.33, 3] or the pose wanders > 10 deg or freezes > 2 s.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import E, FPS, atomic_json, canon_spec, read_wav_mono, write_wav  # noqa: E402

LONG = "_long5"
CH = {"yaw": 0, "pitch": 1, "roll": 2, "gaze_x": 6, "gaze_y": 7, "brow": 10, "smile": 12}


def manifest():
    return {c["id"]: c for c in json.loads((E / "clips/manifest.json").read_text())["clips"]}


def long_clip():
    return json.loads((E / "clips" / LONG / "clip.json").read_text())


def build(minutes=5.0):
    man = manifest()
    od = E / "clips" / LONG
    if (od / "clip.json").exists():
        return long_clip()
    od.mkdir(parents=True, exist_ok=True)
    need = int(minutes * 60 * FPS)
    ids = sorted(c for c, v in man.items() if v["group"] == "seamless")
    used, n = [], 0
    for cid in ids:
        if n >= need:
            break
        used.append(cid); n += int(round(float(man[cid]["duration_s"]) * FPS))
    rows, mics, voices, ua, sa, words, usp, mps = [], [], [], [], [], [], [], []
    off = 0.0
    for cid in used:
        c = man[cid]; cd = E / "clips" / cid
        N = int(round(float(c["duration_s"]) * FPS))
        for l in (cd / "perception.jsonl").read_text().splitlines()[:N]:
            r = json.loads(l)
            t = round(r["t"] + off, 4)
            r["t"] = t
            if r.get("features"):
                r["features"]["t"] = t
                for ev in r["features"].get("events") or []:
                    ev["t"] = round(float(ev.get("t", 0)) + off, 4)
            r["events"] = (r.get("features") or {}).get("events") or []
            rows.append(r)
        x, sr = read_wav_mono(cd / "mic.wav"); mics.append(x[:int(N / FPS * sr)])
        v, sr2 = read_wav_mono(cd / "voice.wav"); voices.append(v[:int(N / FPS * sr2)])
        ua.append(np.load(cd / "audio42_user.npy")[:N]); sa.append(np.load(cd / "audio42_self.npy")[:N])
        for w0, w1, wd in c.get("avatar_words") or []:
            words.append([w0 + off, w1 + off, wd])
        ann = json.loads(Path(c["annotation"]).read_text())
        for s in ann.get("user_speech") or []:
            usp.append({"t0": s["t0"] + off, "t1": s["t1"] + off})
        z = np.load(cd / "mp_live.npz"); mps.append({k: z[k] for k in z.files})
        off += N / FPS
    D = need
    rows = rows[:D]
    with open(od / "perception.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    write_wav(od / "mic.wav", np.concatenate(mics)[:int(D / FPS * 24000)], 24000)
    write_wav(od / "voice.wav", np.concatenate(voices)[:int(D / FPS * 24000)], 24000)
    np.save(od / "audio42_user.npy", np.concatenate(ua)[:D]); np.save(od / "audio42_self.npy", np.concatenate(sa)[:D])
    mp = {}
    for k in mps[0]:
        if k in ("wh", "fps"):
            mp[k] = mps[0][k]
        else:
            mp[k] = np.concatenate([m[k] for m in mps])[:D]
    np.savez_compressed(od / "mp_live.npz", **mp)
    ann = {"id": LONG, "description": "concatenated Seamless test conversations", "user_speech": [s for s in usp if s["t0"] < D / FPS]}
    (od / "annotation.json").write_text(json.dumps(ann))
    clip = {"id": LONG, "group": "long", "duration_s": D / FPS, "avatar_words": [w for w in words if w[0] < D / FPS],
            "annotation": str(od / "annotation.json"), "sources": used}
    (od / "clip.json").write_text(json.dumps(clip))
    (od / "receipt.json").write_text(json.dumps({"id": LONG, "frames": D, "sources": used}))
    print("built", LONG, D, "frames from", len(used), "clips")
    return clip


def human_long(clip, od):
    """the real partners' canon over the same concatenation (each window's avatar-side person)."""
    import systems as SY
    man = manifest()
    mos, mas = [], []
    for cid in clip["sources"]:
        h = SY.human_canon(man[cid])
        if h is None:
            return None
        mos.append(h[0]); mas.append(h[1])
    D = int(round(clip["duration_s"] * FPS))
    mo, ma = np.concatenate(mos)[:D], np.concatenate(mas)[:D]
    spec = canon_spec()
    rest = SY.rest_pose()
    cm = spec.col_mask(ma).astype(bool)
    out = mo.copy(); held = rest.copy()
    for s in range(len(mo)):
        held = np.where(cm[s], mo[s], held); out[s] = held
    od.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(od / "canon.npz", motion=out, mask=ma, meta=np.array(json.dumps({"system": "human", "clip": LONG})))
    return od / "canon.npz"


def run(system):
    import systems as SY
    clip = long_clip()
    if system.startswith("unified"):
        import bundles
        meta = bundles.resolve(system)
        name = meta["run"]
    else:
        meta, name = None, system
    od = E / "runs" / name / LONG
    if (od / "rig.jsonl").exists():
        print("cached", name); return name
    if system in ("rules", "gru-prior"):
        SY.run_rig(clip, system, od)
    elif system == "copy-delay":
        SY.run_copy_delay(clip, od)
        SY.canon_to_rig(clip, od)
    elif system == "human":
        if human_long(clip, od) is None:
            print("no human canon"); return name
        SY.canon_to_rig(clip, od)
    else:
        kw = {"style": meta["style"]} if meta["style"] else {}
        SY.MODEL_DIR = Path(meta["code"])
        SY.run_unified(clip, od, meta["bundle"], **kw)
        SY.canon_to_rig(clip, od)
    print("ok", name)
    return name


def measure(runs, out=None):
    import measures as M
    res = {"clip": long_clip()["sources"], "minutes": None, "systems": {}}
    for run_ in runs:
        p = E / "runs" / run_ / LONG / "rig.jsonl"
        if not p.exists():
            continue
        Tb = M.table_from_rig(p)[0]
        n = len(Tb)
        mins = n // 1800
        res["minutes"] = mins
        o = {"finite": bool(np.isfinite(Tb[:, :16]).all() or np.isfinite(np.nan_to_num(Tb[:, :16])).all())}
        sd, mu = {}, {}
        for k, j in CH.items():
            sd[k] = [float(np.nanstd(Tb[a:a + 1800, j])) for a in range(0, mins * 1800, 1800)]
            mu[k] = [float(np.nanmean(Tb[a:a + 1800, j])) for a in range(0, mins * 1800, 1800)]
        o["per_minute_sd"] = sd
        o["per_minute_mean"] = mu
        o["sd_ratio_min_max"] = {k: [min(v) / max(v[0], 1e-6), max(v) / max(v[0], 1e-6)] for k, v in sd.items()}
        wander = {k: float(max(abs(np.array(v) - np.nanmean(Tb[:mins * 1800, CH[k]])))) for k, v in mu.items() if k in ("yaw", "pitch", "roll", "gaze_x", "gaze_y")}
        o["mean_wander_deg"] = wander
        mv = np.abs(np.diff(np.nan_to_num(Tb[:, [0, 1, 2, 6, 7, 10, 11, 12]]), axis=0)).max(1)
        import mstats
        o["longest_frozen_s"] = max((b - a for a, b in mstats._runs(mv < 1e-3)), default=0) / FPS
        o["frac_extreme_head"] = float(((np.abs(np.nan_to_num(Tb[:, 0])) > 30) | (np.abs(np.nan_to_num(Tb[:, 1])) > 30)).mean())
        ang = np.nan_to_num(Tb[:, :3])
        o["head_speed_med_per_min"] = [float(np.median(np.linalg.norm(np.gradient(ang[a:a + 1800], axis=0) * FPS, axis=1)))
                                       for a in range(0, mins * 1800, 1800)]
        rr = [v for k, v in o["sd_ratio_min_max"].items() if k in ("yaw", "pitch", "roll", "gaze_x", "smile")]
        o["drifts"] = bool(any(a < 0.33 or b > 3.0 for a, b in rr) or max(wander.values()) > 10.0 or o["longest_frozen_s"] > 2.0)
        o["worst_sd_ratio"] = float(max(max(b, 1 / max(a, 1e-6)) for a, b in rr))
        res["systems"][run_] = o
        print(f"{run_:30s} drifts={o['drifts']} worst_sd_ratio={o['worst_sd_ratio']:.2f} wander={max(wander.values()):.1f}deg "
              f"frozen={o['longest_frozen_s']:.1f}s extreme={o['frac_extreme_head']:.3f}")
    atomic_json(Path(out) if out else E / "measures/drift.json", res)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "run", "measure"])
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--system"); ap.add_argument("--runs"); ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.minutes)
    elif a.cmd == "run":
        build(a.minutes)
        run(a.system)
    else:
        measure(a.runs.split(","), a.out)


if __name__ == "__main__":
    main()
