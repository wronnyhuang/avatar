#!/usr/bin/env python3
"""Find, per character, the best (size, backend, preset) that holds the output fps through the WHOLE render-server
pipeline (animator -> render -> encode -> WebSocket -> client) on this GPU, and write the profile table.

    python -m avatar.bench_profiles [--lock out/gpu.lock] [--rt-tree DIR] \\
        --json avatar/profiles_local_measured.json --write-profile local

(run from the renderer/ directory). Starts its own render server (avatar/local_server.py), then for each character walks its
ladder (best look first) and stops at the first rung that holds: fps received >= --min-fps, <= --max-missing-pct frames
missing, and the server's render p95 inside one frame period.  Each character's ladder runs inside ONE hold of the GPU
lock (a few sessions of ~--seconds each, well under 5 minutes).  Every session streams --wav on a real-time timeline
(speech, then isolated syllables for the lip-sync check) through avatar/test_client.py and keeps its JSON;
uptime / nvidia-smi are recorded around each one (other GPU work may share the card).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

REPO = os.path.dirname(ROOT)
OUT = os.environ.get("AVATAR_BENCH_OUT") or os.path.join(REPO, "out", "bench-profiles")

# best look first; sizes are the ones whose backgrounds are baked in the shared cache; presets from rt-gpu's
# engine/rt_engine.py (qNN = TAA + character pass at NN% resolution, "r" = mouth box at full resolution); each ladder
# holds rt-gpu's own 640x360 pick (rt-gpu.json k620_24fps, render alone on a free card) as a rung
LADDERS = {
    "fox":     [(960, 540, "q33"), (640, 360, "q50"), (640, 360, "q40"), (640, 360, "q33")],
    "bear":    [(960, 540, "q33"), (640, 360, "q60"), (640, 360, "q50"), (640, 360, "q40")],
    "rabbit":  [(960, 540, "q40"), (960, 540, "q33"), (640, 360, "q60"), (640, 360, "q50")],
    "hippo":   [(960, 540, "q33"), (640, 360, "q50"), (640, 360, "q40"), (640, 360, "q33")],
    "monkey":  [(960, 540, "q33"), (640, 360, "q50"), (640, 360, "q40"), (640, 360, "q33")],
    "orb":     [(1280, 720, "q75"), (1280, 720, "q60"), (1280, 720, "q50"), (960, 540, "q60")],
    "cube":    [(1280, 720, "q60"), (1280, 720, "q50"), (1280, 720, "q40"), (960, 540, "q50")],
    "drop":    [(1280, 720, "q75"), (1280, 720, "q60"), (1280, 720, "q50"), (960, 540, "q60")],
    "star":    [(1280, 720, "q40"), (960, 540, "q50"), (960, 540, "q40"), (640, 360, "q50")],
    "pyramid": [(1280, 720, "q60"), (1280, 720, "q50"), (1280, 720, "q40"), (960, 540, "q50")],
    "blob":    [(1280, 720, "q60"), (1280, 720, "q50"), (1280, 720, "q40"), (960, 540, "q50")],
    "muse":    [(960, 540, "q33"), (640, 360, "q50"), (640, 360, "q40"), (640, 360, "q33")],   # provisional
}


def machine():
    m = {"t": time.strftime("%H:%M:%S"),
         "uptime": subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()}
    try:
        m["gpu_apps"] = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                        "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        m["gpu_util_mem"] = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                            "--format=csv,noheader"], capture_output=True, text=True,
                                           timeout=10).stdout.strip()
    except Exception:
        pass
    return m


def make_combo(path, speech_s=8.0):
    """speech (clip.wav) then 0.6 s of silence then the isolated-syllable onset WAV (live.latency make-wav)."""
    from avatar.test_client import read_wav
    onset = os.path.join(OUT, "onset.wav")
    if not os.path.exists(onset):
        subprocess.run([sys.executable, "-m", "live.latency", "make-wav", "--out", onset, "--n", "12", "--period", "1.0",
                        "--lead", "0.5"], cwd=ROOT, check=True)
    a = read_wav(os.path.join(ROOT, "audio_in", "clip.wav"))[:int(speech_s * 24000)]
    b = read_wav(onset)
    y = np.concatenate([a, np.zeros(int(0.6 * 24000), np.int16), b])
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(y.tobytes())
    return path, y.size / 24000


def holds(r, a):
    if "error" in r:
        return False
    rp95 = (r.get("server", {}).get("render_p95") or {}).get("p50")
    return ((r.get("fps_received") or 0) >= a.min_fps and r.get("missing_pct", 100) <= a.max_missing_pct
            and (rp95 is None or rp95 <= 1000.0 / a.fps * 1.1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--characters", default=",".join(LADDERS))
    ap.add_argument("--lock", default=os.path.join(REPO, "out", "gpu.lock"))
    ap.add_argument("--rt-tree", default=os.environ.get("AVATAR_RT_TREE", ""))
    ap.add_argument("--port", type=int, default=8772)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--min-fps", type=float, default=23.0)
    ap.add_argument("--max-missing-pct", type=float, default=3.0)
    ap.add_argument("--speech-s", type=float, default=8.0)
    ap.add_argument("--all-rungs", action="store_true", help="measure every rung, not just down to the first that holds")
    ap.add_argument("--ladder", default="", help="JSON {char: [[w,h,preset],...]} overriding the built-in ladders")
    ap.add_argument("--backend", default="rt")
    ap.add_argument("--json", default=os.path.join(ROOT, "avatar", "profiles_local_measured.json"))
    ap.add_argument("--write-profile", default="", help="profiles.json section to (re)write from the chosen rungs")
    a = ap.parse_args(argv)
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))      # so `finally` stops the render server it started
    os.makedirs(os.path.join(OUT, "bench"), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.lock)), exist_ok=True)
    os.environ.setdefault("TA_OUT", os.path.join(REPO, "out", "cache"))
    ladders = dict(LADDERS)
    if a.ladder:
        ladders.update({k: [tuple(x) for x in v] for k, v in json.loads(a.ladder).items()})
    combo, dur = make_combo(os.path.join(OUT, "bench", "combo.wav"), a.speech_s)

    from avatar.local_server import LocalServer
    from avatar import test_client
    P = LocalServer(state_dir=os.path.join(OUT, "state"), port=a.port, rt_tree=a.rt_tree,
                      log_path=os.path.join(OUT, "logs", f"bench_server_{a.port}.log"))
    P.stop()
    P.create_or_start()
    st = P.wait_ready(timeout=60, every=1)
    if st.get("state") != "running":
        sys.exit(f"render server did not come up: {st}")
    rt_rev = subprocess.run(["git", "-C", a.rt_tree, "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip() if a.rt_tree else None
    dirty = subprocess.run(["git", "-C", a.rt_tree, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip() if a.rt_tree else ""
    try:
        results = json.load(open(a.json))
    except (OSError, ValueError):
        results = {"runs": [], "chosen": {}}
    results.setdefault("runs", [])
    results.setdefault("chosen", {})
    results["meta"] = {"rt_tree": a.rt_tree, "rt_rev": rt_rev, "rt_dirty": bool(dirty), "fps": a.fps,
                       "criterion": f"fps_received >= {a.min_fps}, missing <= {a.max_missing_pct}%, "
                                    f"render p95 (median of the per-second p95s) <= 1.1 frame periods",
                       "wav": f"{a.speech_s:.0f} s of clip.wav + 0.6 s silence + 12 isolated pa/ma syllables ({dur:.1f} s)"}
    try:
        for ch in a.characters.split(","):
            lockf = open(a.lock, "a")
            t_wait = time.time()
            fcntl.flock(lockf, fcntl.LOCK_EX)
            waited = time.time() - t_wait
            t_hold = time.time()
            chosen = None
            try:
                for (w, h, preset) in ladders[ch]:
                    if time.time() - t_hold > 230:          # keep every hold under ~5 minutes
                        break
                    tag = f"{ch}_{w}x{h}_{a.backend}_{preset}"
                    before = machine()
                    t = time.time()
                    try:
                        r = test_client.main(["--url", P.render_url(), "--character", ch, "--width", str(w),
                                              "--height", str(h), "--backend", a.backend, "--opts",
                                              json.dumps({"preset": preset}), "--wav", combo, "--seconds", "0",
                                              "--lead-s", "1.0", "--tail-s", "0.5", "--fps", str(a.fps),
                                              "--json", os.path.join(OUT, "bench", tag + ".json"),
                                              "--out", os.path.join(OUT, "bench", tag + ".mp4")])
                    except Exception as e:
                        r = {"error": f"{type(e).__name__}: {e}"}
                    after = machine()
                    ok = holds(r, a)
                    srv = r.get("server", {})
                    row = {"character": ch, "width": w, "height": h, "backend": a.backend, "preset": preset,
                           "holds": ok, "fps_received": r.get("fps_received"), "missing_pct": r.get("missing_pct"),
                           "render_ms": (srv.get("render_ms") or {}).get("mean"),
                           "render_p95": (srv.get("render_p95") or {}).get("p50"),
                           "encode_ms": (srv.get("encode_ms") or {}).get("mean"),
                           "dropped": srv.get("dropped_total"), "late_sent": srv.get("late_sent_total"),
                           "delay_ms": r.get("delay_ms_frame_instant"),
                           "delay_after_lookahead_ms": r.get("delay_ms_after_lookahead_audio"),
                           "late_at_delay_ms": r.get("late_at_delay_ms"),
                           "sync_lag_frames": (r.get("sync") or {}).get("best_lag_frames"),
                           "sync_onset_offset_ms": (r.get("sync") or {}).get("onset_offset_ms"),
                           "ready_s": r.get("ready_s"), "encoder": (r.get("ready") or {}).get("encoder"),
                           "codec_string": (r.get("ready") or {}).get("codec_string"),
                           "backend_used": (r.get("ready") or {}).get("backend"), "kbps": r.get("kbps_payload"),
                           "session_s": round(time.time() - t, 1), "lock_waited_s": round(waited, 1),
                           "machine_before": before, "machine_after": after, "error": r.get("error"),
                           "json": os.path.join(OUT, "bench", tag + ".json")}
                    results["runs"].append(row)
                    print(json.dumps({k: row[k] for k in ("character", "width", "height", "preset", "holds",
                                                          "fps_received", "missing_pct", "render_ms", "render_p95",
                                                          "encode_ms", "delay_ms")}), flush=True)
                    json.dump(results, open(a.json, "w"), indent=1)
                    if ok and chosen is None:
                        chosen = row
                        if not a.all_rungs:
                            break
            finally:
                fcntl.flock(lockf, fcntl.LOCK_UN)
                lockf.close()
            if chosen is None:          # nothing held: the cheapest rung measured, flagged
                rows = [r for r in results["runs"] if r["character"] == ch and r.get("fps_received")]
                if rows:
                    chosen = dict(max(rows, key=lambda r: r["fps_received"]), holds=False)
            if chosen:
                results["chosen"][ch] = {k: chosen[k] for k in ("width", "height", "backend", "preset", "holds",
                                                                "fps_received", "render_ms", "render_p95",
                                                                "encode_ms", "delay_ms", "missing_pct",
                                                                "sync_onset_offset_ms", "sync_lag_frames")}
            json.dump(results, open(a.json, "w"), indent=1)
    finally:
        P.stop()
    if a.write_profile:
        write_profile(a.write_profile, results["chosen"], results["meta"])
    print(json.dumps(results["chosen"], indent=1))


def write_profile(section, chosen, meta):
    path = os.path.join(ROOT, "avatar", "profiles.json")
    try:
        prof = json.load(open(path))
    except (OSError, ValueError):
        prof = {}
    prof.setdefault("profiles", {})
    sec = prof["profiles"].setdefault(section, {})
    for ch, c in chosen.items():
        sec[ch] = {"width": c["width"], "height": c["height"], "backend": c["backend"],
                   "opts": {"preset": c["preset"]},
                   "measured": {k: c.get(k) for k in ("holds", "fps_received", "render_ms", "render_p95", "encode_ms",
                                                      "missing_pct")} | {"delay_ms_mean": (c.get("delay_ms") or {}).get("mean"),
                                                                         "delay_ms_p95": (c.get("delay_ms") or {}).get("p95")}}
    sec["_measured_with"] = meta
    json.dump(prof, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
