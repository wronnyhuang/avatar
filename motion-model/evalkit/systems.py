#!/usr/bin/env python3
"""Run the compared systems on the clip set -> $T/eval/runs/<system>/<clip>/{rig.jsonl | canon.npz}.

  python -I evalkit/systems.py --systems rules,gru-prior,copy-delay,unified,human [--ids ..] [--jobs 8]
         [--bundle $T/runs/BEST] [--tag <run name suffix>] [--ablate zero|shuffle|shift]

Systems (what each one is, exactly):
  rules       today's dev controller (avatar 8e8f606 = avatar-dev): js/replay.cjs mode rules (motion.js + moves engine +
              Muse body/props + GRU prior via setPrior + cospeech on the reply words) -> rig.jsonl
  gru-prior   prior.js alone (js/replay.cjs mode gru-prior) -> rig.jsonl
  copy-delay  mirror the user's live-fitted canon (canon1.fit.Fitter on the same detections) with a 30-frame (1 s) delay
              (spec.mirror_motion; groups the user lacks -> REST pose) -> canon.npz
  unified     unified.runtime.UnifiedPlanner(bundle, threads=2, seed=0, defaults) stepped causally: user = Fitter canon,
              user audio42 (mic), self audio42 at frame s+L (L = p.self_audio_lookahead) -> canon.npz (+ plan_ms)
  human       the avatar-side person's REAL motion (the data pipeline's canon of that Seamless session) -> canon.npz
  a2f         the SPEAKING-FACE BASELINE: rules' rig stream, and while the avatar speaks (its reply spans, 0.15 s
              min-jerk ramps) the face dials NVIDIA Audio2Face-3D v3.0 covers (run on the avatar's own voice; evalkit/
              a2f_face.py): smile -> the rig smile, jaw / lips -> lip_override.npz = A2F replaces lipstream's lip columns
              at the renderer (a renderer-side change; evalkit/onscreen.py + render_lipover.py) -> rig.jsonl (+ override)
  rules+a2f-mouth  rules' rig unchanged (its smile too) + A2F's lips replacing lipstream's while speaking (lip_override.npz):
              isolates the MOUTH
  rules+a2f-page   A2F fed through TODAY's page with no renderer change: its jaw would arrive as mouth_open, which yields
              to speech while voiced (motion.js SPEECH_GROUP; rules' own mouth_open is 0 while speaking) and the renderer's
              jaw = max(lipsync, mouth_open) -> lipstream keeps the jaw/lips; only A2F's smile reaches the screen. Same
              rig.jsonl as a2f, no lip override.
Ablations (responsiveness): the USER stream (camera features / fit + mic) is replaced, everything else identical:
  zero = no user (no face, silent mic); shuffle = another clip's user (the next clip of the same group, cut/padded);
  shift = the same user circularly shifted by half the clip.
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import E, T, DATA, WT, KIT, AVATAR, FPS, VENV_PY, MODEL_CODE, atomic_json, canon_spec, read_wav_mono, write_wav  # noqa: E402

A2F_SYSTEMS = ("a2f", "rules+a2f-mouth", "rules+a2f-page")
RIG_SYSTEMS = ("rules", "gru-prior") + A2F_SYSTEMS

MUSE_BODY = AVATAR / "renderer/animals/muse.body.json"
MUSE_PROPS = AVATAR / "renderer/animals/muse.props.json"
AVATAR_REV = subprocess.run(["git", "-C", str(AVATAR), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip() or "?"
FIT_DIRS = [WT]                                      # canon1/fit.py (this checkout)
MODEL_DIR = MODEL_CODE                               # unified/runtime.py (overridable per run: set_model_code)


def manifest():
    return {c["id"]: c for c in json.loads((E / "clips/manifest.json").read_text())["clips"]}


def clip_dir(cid):
    return E / "clips" / cid


def run_dir(system, tag=""):
    return E / "runs" / (system + (("-" + tag) if tag else ""))


# ----------------------------------------------------------------------------- the avatar's replies (speech spans)
def utterances(clip):
    """[[t0, t1, text]]: the avatar's reply spans. Seamless: its words (gap >= 0.8 s splits); else energy spans."""
    dur = float(clip["duration_s"])
    if clip.get("avatar_words"):
        out = []
        for w0, w1, wd in clip["avatar_words"]:
            if out and w0 - out[-1][1] < 0.8:
                out[-1][1] = w1; out[-1][2] += " " + wd
            else:
                out.append([w0, w1, wd])
        return [[round(max(0, a - 0.05), 3), round(min(dur, b + 0.15), 3), t] for a, b, t in out]
    x, sr = read_wav_mono(clip_dir(clip["id"]) / "voice.wav")
    n = int(dur * 50)
    lv = np.array([np.sqrt(np.mean(x[int(k * sr / 50):int((k + 1) * sr / 50)] ** 2) + 1e-12) for k in range(n)])
    on = lv > 0.008
    spans = []
    k = 0
    while k < n:
        if on[k]:
            j = k
            while j < n and (on[j] or (j + 40 < n and on[j:j + 40].any())):
                j += 1
            spans.append([round(k / 50, 3), round(j / 50, 3), ""])
            k = j
        else:
            k += 1
    return spans


# ----------------------------------------------------------------------------- ablated user inputs
def ablated_user(clip, kind, man):
    """-> (perception.jsonl path, mic.wav path, mp_live dict) for the ablated USER stream."""
    cd = clip_dir(clip["id"])
    dur = float(clip["duration_s"])
    ad = E / "ablate_inputs" / kind / clip["id"]
    ad.mkdir(parents=True, exist_ok=True)
    N = int(round(dur * FPS))
    if kind == "zero":
        rows = [{"t": round(i / FPS + 0.001, 4), "features": {"t": round(i / FPS + 0.001, 4), "dt": 1 / FPS,
                 "face": {"present": False}, "hands": [], "pose": None, "motion": 0, "attending": False, "events": []},
                 "events": []} for i in range(N)]
        mic = np.zeros(int(dur * 24000), np.float32)
        mp = None
        src_note = "no user"
    else:
        if kind == "shuffle":
            same = sorted(c for c, v in man.items() if v["group"] == clip["group"] and c != clip["id"])
            if not same:
                raise RuntimeError("no other clip to shuffle with")
            ids = sorted([c for c, v in man.items() if v["group"] == clip["group"]])
            other = ids[(ids.index(clip["id"]) + 1) % len(ids)]
            sd, shift = clip_dir(other), 0
            src_note = f"user of {other}"
        else:
            sd, shift = cd, N // 2
            src_note = f"own user shifted {shift} frames (circular)"
        src = [json.loads(l) for l in (sd / "perception.jsonl").read_text().splitlines() if l.strip()]
        M = len(src)
        rows = []
        for i in range(N):
            r = json.loads(json.dumps(src[(i + shift) % M]))
            t = round(i / FPS + 0.001, 4)
            r["t"] = t
            if r.get("features"):
                r["features"]["t"] = t
                # perception events carry their own clip time; re-stamp so motion.js dedup keys stay unique
                for ev in (r["features"].get("events") or []):
                    ev["t"] = t
            r["events"] = (r.get("features") or {}).get("events") or []
            rows.append(r)
        x, sr = read_wav_mono(sd / "mic.wav")
        n = int(dur * sr)
        idx = (np.arange(n) + int(shift * sr / FPS)) % len(x)
        mic = x[idx]
        z = np.load(sd / "mp_live.npz")
        mp = {k: z[k] for k in z.files}
        Mm = len(mp["face"])
        sel = (np.arange(N) + shift) % Mm
        for k in list(mp):
            if isinstance(mp[k], np.ndarray) and mp[k].ndim >= 1 and len(mp[k]) == Mm:
                mp[k] = mp[k][sel]
    with (ad / "perception.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    write_wav(ad / "mic.wav", mic, 24000)
    spec = canon_spec()
    np.save(ad / "audio42_user.npy", spec.audio_features(spec.resample_to_16k(mic, 24000), n_frames=N))
    if mp is not None:
        np.savez_compressed(ad / "mp_live.npz", **mp)
    (ad / "note.txt").write_text(src_note + "\n")
    return ad


# ----------------------------------------------------------------------------- rig systems (node replay)
def run_rig(clip, system, od, inputs_dir=None, seed=9, ablation="none"):
    cd = clip_dir(clip["id"])
    idir = inputs_dir or cd
    od.mkdir(parents=True, exist_ok=True)
    job = {"avatar": str(AVATAR), "avatar_rev": AVATAR_REV, "mode": system, "seed": seed, "clip": clip["id"],
           "perception": str(idir / "perception.jsonl"), "mic": str(idir / "mic.wav"), "voice": str(cd / "voice.wav"),
           "duration": float(clip["duration_s"]), "voice_utts": utterances(clip), "rig": str(od / "rig.jsonl"),
           "body": str(MUSE_BODY), "props": str(MUSE_PROPS), "character": "muse", "ablation": ablation}
    jp = od / "replay.json"
    jp.write_text(json.dumps(job))
    r = subprocess.run(["node", str(KIT / "js/replay.cjs"), str(jp)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"replay {system} {clip['id']}: {r.stderr[-1500:]}")
    return od / "rig.jsonl"


# ----------------------------------------------------------------------------- canon systems
def _import_fitter():
    for d in FIT_DIRS:
        if (d / "canon1/fit.py").exists():
            sys.path.insert(0, str(d))
            from canon1.fit import Fitter  # noqa
            return Fitter, str(d)
    raise ImportError("canon1/fit.py (the canon1 Fitter) not available yet")


def mp_frames(mp):
    """mp_live arrays -> per-tick Fitter input dicts (sample-and-hold of the newest detections, as the page holds)."""
    n = len(mp["face"])
    wh = tuple(int(x) for x in mp["wh"])
    for i in range(n):
        hands = []
        for j in range(int(mp["hand_n"][i])):
            lab = int(mp["hand_label"][i][j])
            hands.append({"handedness": "Left" if lab == 0 else "Right" if lab == 1 else "?",
                          "world": mp["hands_world"][i][j], "image": mp["hands"][i][j], "score": float(mp["hand_score"][i][j])})
        yield {"face_present": bool(mp["face"][i]), "blendshapes": mp["bs"][i], "face_matrix": mp["mat"][i],
               "pose_world": mp["pose_world"][i] if mp["pose_ok"][i] else None,
               "pose_image": mp["pose"][i] if mp["pose_ok"][i] else None, "pose_present": bool(mp["pose_ok"][i]),
               "hands": hands, "image_wh": wh,
               "fresh": {"face": int(mp["face_fresh"][i]), "pose": int(mp["pose_fresh"][i]), "hand": int(mp["hand_fresh"][i])}}


def user_fit(clip, inputs_dir=None, force=False):
    """the user's live canon: the canon1 Fitter on the same detections (separate process: evalkit/fit_user.py; cached)."""
    idir = inputs_dir or clip_dir(clip["id"])
    cp = idir / "user_fit.npz"
    if not (idir / "mp_live.npz").exists():          # ablation 'zero': nobody in view -> all-zero mask every frame
        spec = canon_spec()
        N = int(round(float(clip["duration_s"]) * FPS))
        return np.zeros((N, spec.D_MOTION), np.float32), np.zeros((N, spec.D_MASK), np.uint8)
    if not cp.exists() or force:
        r = subprocess.run([str(VENV_PY), "-I", str(KIT / "fit_user.py"), str(idir)], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"fitter failed on {idir}: {r.stderr[-1200:]}")
    z = np.load(cp)
    return z["motion"], z["mask"]


def rest_pose():
    sys.path.insert(0, str(MODEL_DIR))
    try:
        from unified.runtime import REST_POSE  # noqa
        return np.asarray(REST_POSE, np.float32)
    except Exception:
        spec = canon_spec()
        r = np.zeros(spec.D_MOTION, np.float32)
        r[spec.COL["shoulder_l_z"]] = -1.2; r[spec.COL["shoulder_r_z"]] = 1.2
        return r


def run_copy_delay(clip, od, inputs_dir=None, delay=30):
    spec = canon_spec()
    um, uk = user_fit(clip, inputs_dir)
    mm, mk = spec.mirror_motion(um), spec.mirror_mask(uk)
    N = len(um)
    rest = rest_pose()
    out = np.tile(rest, (N, 1)).astype(np.float32)
    held = rest.copy()
    cm = spec.col_mask(mk).astype(bool)
    for s in range(N):
        src = s - delay
        if src >= 0:
            held = np.where(cm[src], mm[src], held)
        out[s] = held
    od.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(od / "canon.npz", motion=out, mask=np.ones((N, spec.D_MASK), np.uint8),
                        meta=np.array(json.dumps({"system": "copy-delay", "delay_frames": delay, "clip": clip["id"]})))
    return od / "canon.npz"


def run_unified(clip, od, bundle, inputs_dir=None, seed=0, threads=2, **kw):
    spec = canon_spec()
    sys.path.insert(0, str(MODEL_DIR))
    from unified.runtime import UnifiedPlanner  # noqa
    idir = inputs_dir or clip_dir(clip["id"])
    um, uk = user_fit(clip, inputs_dir)
    ua = np.load(idir / "audio42_user.npy")
    sa = np.load(clip_dir(clip["id"]) / "audio42_self.npy")
    N = len(um)
    style = kw.pop("style", None)
    p = UnifiedPlanner(str(bundle) if bundle else None, threads=threads, seed=seed, **kw)
    if style:
        apply_style(p, style)
    L = int(getattr(p, "self_audio_lookahead", 6))
    silent_self = sa[:, 41] < 0.5          # vad column
    # phase 2: a speech-encoder bundle computes its features from RAW 16 kHz audio inside the runtime
    # (UnifiedPlanner pcm_mode) -> feed the same audio the audio42 inputs came from: the user's mic.wav (the ablated one
    # for ~zero/~shuffle) and the avatar's voice.wav; frame k = samples [frame_end(k-1), frame_end(k)), clip start = 0.
    pcm = bool(getattr(p, "pcm_mode", False))
    if pcm:
        from unified.audio_feats import pcm_frame  # noqa
        xu, sru = read_wav_mono(idir / "mic.wav")
        xs, srs = read_wav_mono(clip_dir(clip["id"]) / "voice.wav")
        xu = np.asarray(spec.resample_to_16k(xu, sru) if sru != 16000 else xu, np.float32)
        xs = np.asarray(spec.resample_to_16k(xs, srs) if srs != 16000 else xs, np.float32)
        P = int(p.self_pcm_lookahead)
    out, plan_ms, step_ms = [], [], []
    for s in range(N):
        k = min(N - 1, s + L)
        self_a = None if (s + L >= N or not sa[k:k + 1, 41].any() and silent_self[max(0, s - 6):k + 1].all()) else sa[k]
        t0 = time.perf_counter()
        extra = {"user_pcm": pcm_frame(xu, s), "self_pcm_ahead": pcm_frame(xs, s + P)} if pcm else {}
        r = p.step(um[s] if uk[s].any() else None, uk[s] if uk[s].any() else None, ua[s], self_a, **extra)
        step_ms.append((time.perf_counter() - t0) * 1000)
        out.append(np.asarray(r["motion"], np.float32))
        plan_ms.append(np.nan if r.get("plan_ms") is None else float(r["plan_ms"]))
    od.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(od / "canon.npz", motion=np.stack(out), mask=np.ones((N, spec.D_MASK), np.uint8),
                        plan_ms=np.array(plan_ms, np.float32), step_ms=np.array(step_ms, np.float32),
                        meta=np.array(json.dumps({"system": "unified", "bundle": str(bundle), "seed": seed, "threads": threads,
                                                  "lookahead": L, "clip": clip["id"], "style": style, "pcm": pcm,
                                                  "code": str(MODEL_DIR), **{k: v for k, v in kw.items()}})))
    return od / "canon.npz"


def apply_style(p, style):
    """the CONTRACT's style vector {energy, mirroring, expressiveness} in [0,1] (None = infer) -> the planner.
    Accepts set_style(**kw) or set_style(dict); a runtime without set_style is an error (never a silent no-op)."""
    fn = getattr(p, "set_style", None) or getattr(getattr(p, "core", None), "set_style", None)
    if fn is None:
        raise RuntimeError("this unified runtime has no set_style (style vector not implemented in the bundle's code)")
    try:
        fn(**style)
    except TypeError:
        fn(dict(style))


def human_canon(clip):
    """the avatar-side person's real canon for the window (the data pipeline's npz of the session) -> (N,189), (N,9) or None."""
    spec = canon_spec()
    p = DATA / "canon1/seamless" / f"{clip['session']}.npz"
    if not p.exists():
        return None
    d = spec.load_session(str(p))
    persons = d["meta"]["persons"]
    pi = persons.index(clip["avatar_person"])
    s0 = int(clip["frame0"]); N = int(round(float(clip["duration_s"]) * FPS))
    mo = d["motion"][pi, s0:s0 + N]; ma = d["mask"][pi, s0:s0 + N]
    if len(mo) < N:
        return None
    return mo.astype(np.float32), ma.astype(np.uint8)


def run_human(clip, od):
    spec = canon_spec()
    h = human_canon(clip)
    if h is None:
        return None
    mo, ma = h
    rest = rest_pose()
    cm = spec.col_mask(ma).astype(bool)
    # invalid groups: hold the last valid value (else rest) — what a renderer would show
    out = mo.copy(); held = rest.copy()
    for s in range(len(mo)):
        held = np.where(cm[s], mo[s], held)
        out[s] = held
    od.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(od / "canon.npz", motion=out, mask=ma,
                        meta=np.array(json.dumps({"system": "human", "clip": clip["id"], "person": clip["avatar_person"]})))
    return od / "canon.npz"


RETARGET_DIR = WT


def canon_to_rig(clip, od):
    """canon.npz -> rig.jsonl through the retarget module's CANON->Muse map + compositor (retarget/render_canon.build_rig, imported
    live so its newest map is used); talking (lip sync owns the mouth) = the avatar's reply spans, the SAME spans the
    rules controller is told it is speaking."""
    if str(RETARGET_DIR) not in sys.path:
        sys.path.insert(0, str(RETARGET_DIR))
    from retarget.render_canon import build_rig  # noqa
    z = np.load(od / "canon.npz")
    motion = z["motion"].astype(np.float32)
    N = len(motion)
    talk = np.zeros(N, bool)
    for a, b, _ in utterances(clip):
        talk[max(0, int(a * FPS)):min(N, int(b * FPS) + 1)] = True
    names, rows, st = build_rig(motion, None, "muse", talk, seed=0)
    with open(od / "rig.jsonl", "w") as f:
        f.write(json.dumps({"rig_params": names, "source": "retarget.build_rig", "retarget_rev": _git_rev(RETARGET_DIR)}) + "\n")
        for i in range(N):
            f.write(json.dumps({"type": "rig", "t": round(i / FPS, 6), "a": [round(float(v), 4) for v in rows[i]],
                                "speaking": bool(talk[i])}) + "\n")
    return od / "rig.jsonl"


def _git_rev(d):
    try:
        return subprocess.run(["git", "-C", str(d), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


# ----------------------------------------------------------------------------- driver
def one(system, clip, man, tag, ablate, bundle, kw, idir=None):
    name = system + (("-" + tag) if tag and system == "unified" else "") + (("~" + ablate) if ablate else "")
    od = E / "runs" / name / clip["id"]
    done = od / ("rig.jsonl" if system in RIG_SYSTEMS or not ablate else "canon.npz")
    if done.exists():
        return name, clip["id"], "cached"
    kw = dict(kw)
    if system in ("rules", "gru-prior"):
        run_rig(clip, system, od, idir, ablation=ablate or "none")
    elif system in A2F_SYSTEMS:
        import a2f_face
        bd = E / "runs" / ("rules" + (("~" + ablate) if ablate else "")) / clip["id"]
        if not (bd / "rig.jsonl").exists():
            run_rig(clip, "rules", bd, idir, ablation=ablate or "none")
        a2f_face.build(clip, utterances(clip), bd / "rig.jsonl", od, system)
    elif system == "copy-delay":
        run_copy_delay(clip, od, idir)
    elif system == "unified":
        run_unified(clip, od, bundle, idir, **kw)
    elif system == "human":
        if clip["group"] != "seamless" or ablate:
            return name, clip["id"], "n/a"
        if run_human(clip, od) is None:
            return name, clip["id"], "no human canon"
    if system in ("copy-delay", "unified", "human") and not ablate:
        canon_to_rig(clip, od)
    return name, clip["id"], "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default="rules,gru-prior,copy-delay,human")
    ap.add_argument("--ids"); ap.add_argument("--group")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--bundle", default=str(T / "runs/BEST"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--ablate", default="")
    ap.add_argument("--kw", default="{}", help="extra UnifiedPlanner kwargs (json)")
    a = ap.parse_args()
    man = manifest()
    clips = [c for c in man.values() if (not a.ids or c["id"] in a.ids.split(",")) and (not a.group or c["group"] == a.group)]
    clips = [c for c in clips if (clip_dir(c["id"]) / "receipt.json").exists()]
    kw = json.loads(a.kw)
    bundle = None if a.bundle in ("", "none", "stub") else Path(a.bundle).resolve()
    jobs = [(s, c) for s in a.systems.split(",") for c in clips]
    # ablated user inputs are built ONCE per clip (shared by every system; cached), then the Fitter (shared cache)
    idirs = {}
    if a.ablate:
        for c in clips:
            ad = E / "ablate_inputs" / a.ablate / c["id"]
            idirs[c["id"]] = ad if (ad / "note.txt").exists() and (ad / "audio42_user.npy").exists() else ablated_user(c, a.ablate, man)
    if any(s in ("copy-delay", "unified") for s in a.systems.split(",")):
        for c in clips:
            user_fit(c, idirs.get(c["id"]))
    import multiprocessing as mp
    with cf.ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn")) as pool:   # GIL-free (canon->rig, planner)
        futs = [pool.submit(one, s, c, man, a.tag, a.ablate, bundle, kw, idirs.get(c["id"])) for s, c in jobs]
        for f in cf.as_completed(futs):
            try:
                print(*f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print("ERROR", repr(e)[:800], flush=True)


if __name__ == "__main__":
    main()
