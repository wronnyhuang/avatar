#!/usr/bin/env python3
"""run_session.py — drive the REAL avatar page like a user and judge what happened (end-to-end).

  python3 tools/session_e2e/run_session.py --url 'http://127.0.0.1:18468/?k=KEY' --script tools/session_e2e/scripts/smoke.json \
          --out /path/to/outdir [--size 540p] [--voice realtime|mock] [--no-judge] [--asr] [--jobs 3]
  python3 tools/session_e2e/run_session.py --post-only --out OUTDIR [--script ...]   # re-run frames + judge on an earlier run

Script (JSON): {"name", "char": "muse", "cam": optional y4m, "voice": "ash", "start_wait_s": 4, "quiet_s": 1.8,
  "turns": [{"say": "...", "wait": "idle" | {"after_s": 3} | {"during": "writing", "delay_s": 1.5} | "immediate",
             "expect": "what a user should see/hear", "judge_frames": 6, "tone": "excited", "max_s": 90, "id": "t1"}]}
See README.md in this folder for the output layout.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_lib as L  # noqa: E402

HERE = L.HERE
CHROME = os.environ.get("E2E_CHROME") or os.environ.get("CHROME_PATH") or ""   # "" = playwright-core's own Chromium
DEFAULT_CAM = os.environ.get("E2E_CAM", "")   # fake camera .y4m; "" = Chromium's built-in test pattern
REC_ROOT = os.path.expanduser(os.environ.get("AVATAR_REC_ROOT", "~/recordings/avatar-sessions"))
SIZES = {"270p": (480, 1.0, "270p"), "360p": (640, 1.0, "360p"), "540p": (640, 1.5, "540p"), "720p": (640, 1.5, "540p")}   # stage css width, dsf, server quality


def find_recording(rid: str | None, sid_hint: str | None, t_after: float, wait_s: float = 180) -> tuple[str | None, dict | None]:
    """the server assembles the mp4 after the page's 'end'; match by rid (preferred) or newest meta started after t_after"""
    end = time.time() + wait_s
    while time.time() < end:
        for jp in sorted(glob.glob(os.path.join(REC_ROOT, "*", "*.json")), key=os.path.getmtime, reverse=True)[:60]:
            try:
                m = json.load(open(jp))
            except Exception:  # noqa: BLE001
                continue
            if rid and m.get("rid") == rid:
                mp = jp[:-5] + ".mp4"
                if os.path.exists(mp):
                    return mp, m
        time.sleep(3)
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, epilog=(
        "env: OPENAI_API_KEY (or E2E_TTS_KEY) = TTS of the user's lines; E2E_CHROME / CHROME_PATH = Chromium binary;\n"
        "     PLAYWRIGHT_CORE = playwright-core module; E2E_CAM = fake camera .y4m (default: test pattern);\n"
        "     AVATAR_REC_ROOT = where the app server writes session recordings; E2E_SCRATCH / E2E_TTS_CACHE = caches\n"
        "     (default <repo>/runs/session_e2e); JUDGE_CLAUDE / AVATAR_CLAUDE_BIN = the `claude` CLI for the judge;\n"
        "     JUDGE_ASR_SCRIPT / JUDGE_ASR_PY = optional external ASR for --asr"))
    ap.add_argument("--url")
    ap.add_argument("--script")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="540p", choices=list(SIZES))
    ap.add_argument("--voice", default="", help="'' = the page's default (realtime), 'mock' = scripted mock voice ($0)")
    ap.add_argument("--cam", default="")
    ap.add_argument("--char", default="")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--asr", action="store_true", help="transcript of the recording's audio (both sides) via $JUDGE_ASR_SCRIPT")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--judge-model", default="sonnet")
    ap.add_argument("--post-only", action="store_true")
    ap.add_argument("--no-rec", action="store_true", help="skip the app recorder (no frames, no judge)")
    ap.add_argument("--param", action="append", default=[], help="extra page query param k=v (repeatable)")
    ap.add_argument("--max-total-s", type=float, default=900)
    ap.add_argument("--only", default="", help="judge only these turn numbers, e.g. 3,4")
    a = ap.parse_args()
    out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)

    script_path = a.script or os.path.join(out, "script.json")
    script = json.load(open(script_path))
    for i, t in enumerate(script["turns"]):
        t.setdefault("id", f"t{i+1}")
    json.dump(script, open(os.path.join(out, "script.json"), "w"), indent=1)

    drv = {}
    rec_meta = None
    mp4 = None
    if not a.post_only:
        if not a.url:
            ap.error("--url required")
        # 1. user audio (cached TTS)
        tts_usd = 0.0
        cfgturns = []
        wdir = os.path.join(out, "user_audio"); os.makedirs(wdir, exist_ok=True)
        if True:
            for t in script["turns"]:
                r = L.tts_wav(t["say"], script.get("voice", "ash"), t.get("tone", ""))
                tts_usd += r["usd"]
                p = os.path.join(wdir, f"{t['id']}.wav")
                if not os.path.exists(p):
                    os.symlink(r["path"], p)
                cfgturns.append({**{k: v for k, v in t.items() if k not in ("expect",)}, "wav": r["path"]})
        u = urllib.parse.urlparse(a.url)
        q = dict(urllib.parse.parse_qsl(u.query))
        css_w, dsf, ql = SIZES[a.size]
        params = {"char": a.char or script.get("char", "muse"), "rec": "0" if a.no_rec else "1", "e2e": "1", "hud": "0"}
        if a.voice:
            params["voice"] = a.voice
        if a.voice == "mock":
            params["mock"] = "mock_01_hi.wav,mock_04_fact.wav,mock_02_food.wav"
        for kv in script.get("params", {}).items():
            params[kv[0]] = kv[1]
        for kv in a.param:
            k, _, v = kv.partition("="); params[k] = v
        cfg = {"url": f"{u.scheme}://{u.netloc}{u.path or '/'}?" + urllib.parse.urlencode(q), "out": out, "chrome": CHROME,
               "cam": a.cam or script.get("cam") or DEFAULT_CAM, "viewport": {"w": css_w + 40, "h": 1100}, "dsf": dsf, "quality": ql, "params": params,
               "turns": cfgturns, "start_wait_s": script.get("start_wait_s", 4), "quiet_s": script.get("quiet_s", 1.8), "no_reply_s": script.get("no_reply_s", 12),
               "max_total_s": a.max_total_s, "rec": not a.no_rec}
        json.dump(cfg, open(os.path.join(out, "driver_cfg.json"), "w"), indent=1)
        for f in ("raw.jsonl", "samples.jsonl", "driver.log"):
            try:
                os.remove(os.path.join(out, f))
            except FileNotFoundError:
                pass
        t_start = time.time()
        print(f"[e2e] driving {script.get('name')} ({len(script['turns'])} turns) -> {out}", flush=True)
        r = subprocess.run(["node", os.path.join(HERE, "driver.cjs"), os.path.join(out, "driver_cfg.json")], text=True)
        drvf = os.path.join(out, "driver.json")
        drv = json.load(open(drvf)) if os.path.exists(drvf) else {"ok": False}
        drv["tts_usd"] = tts_usd
        json.dump(drv, open(drvf, "w"), indent=1)
        if not drv.get("ok"):
            print("[e2e] DRIVER FAILED:", drv.get("error"), flush=True)
            return 2
        if not a.no_rec:
            print(f"[e2e] waiting for the server to assemble recording {drv.get('rid')} ...", flush=True)
            mp4, rec_meta = find_recording(drv.get("rid"), None, t_start)
    else:
        drv = json.load(open(os.path.join(out, "driver.json")))
        if os.path.exists(os.path.join(out, "recording.json")):
            rec_meta = json.load(open(os.path.join(out, "recording.json")))
            mp4 = os.path.join(out, "session.mp4")

    if mp4 and rec_meta and not a.post_only:
        link = os.path.join(out, "session.mp4")
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(mp4, link)
        json.dump(rec_meta, open(os.path.join(out, "recording.json"), "w"), indent=1)
        mp4 = link
    if not mp4:
        print("[e2e] NO RECORDING FOUND - timeline only", flush=True)

    # 2. timeline
    tl = L.build_timeline(out, script, drv, rec_meta)
    json.dump(tl, open(os.path.join(out, "timeline.json"), "w"), indent=1)

    # 3. video facts, frames, judge
    video = {}
    if mp4:
        video = L.probe(mp4); video.update(L.real_fps(mp4))
        try:   # distinct-picture rate of the avatar part of the frame (a 30 fps recording of a 12 fps render is 12 here)
            pj = json.loads(subprocess.run([sys.executable, os.path.join(HERE, "..", "rec_fps_probe.py"), mp4], capture_output=True, text=True).stdout)
            video["fps_distinct_avatar"] = (pj.get("distinct_avatar") or {}).get("fps")
            video["longest_unchanged_ms"] = (pj.get("distinct_avatar") or {}).get("longest_unchanged_ms")
        except Exception:  # noqa: BLE001
            pass
        video["renderer"] = (tl.get("page") or {}).get("render")
        video["audible"] = audio_levels(mp4, tl)
        L.strip_sheets(mp4, os.path.join(out, "strip"))
        only = {int(x) for x in a.only.split(",") if x} or None
        if not a.no_judge:
            print(f"[e2e] judging {sum(1 for t in tl['turns'] if t.get('expect'))} turns ...", flush=True)
            L.judge_all(tl["turns"], mp4, out, a.jobs, a.judge_model, only)
        else:
            for tn in tl["turns"]:
                tn["judge"] = {"verdict": "N/A", "reason": "--no-judge"}
        if a.asr:
            asr_audio(mp4, out)
    else:
        for tn in tl["turns"]:
            tn["judge"] = {"verdict": "UNSURE", "reason": "no recording"}

    pr = tl.get("page", {})
    session = {"script": script.get("name", os.path.basename(script_path)), "url_host": urllib.parse.urlparse(a.url).netloc if a.url else "(post-only)",
               "mp4": mp4, "video": video, "video_desc": f"{video.get('w')}x{video.get('h')} {video.get('fps')}fps rec, {video.get('fps_distinct_avatar')}fps distinct {video.get('duration_s')}s audio={video.get('audio')}" if video else "",
               "voice_usd": pr.get("voice"), "tts_usd": round(drv.get("tts_usd", 0), 4), "duration_s": tl["duration_s"], "build": rec_meta.get("app") if rec_meta else None}
    rep = {"session": session, "turns": tl["turns"], "errors": tl["errors"]}
    L.write_report(out, rep)
    print(open(os.path.join(out, "report.md")).read())
    print("[e2e] report:", os.path.join(out, "report.md"))
    return 0


def audio_levels(mp4: str, tl: dict) -> dict:
    """is there sound, on both sides?  mean volume of the whole track + per-turn: during the user's line / during the avatar's reply"""
    def vol(t0, t1):
        r = L.sh(["ffmpeg", "-v", "info", "-ss", f"{max(0,t0):.2f}", "-t", f"{max(0.2,t1-t0):.2f}", "-i", mp4, "-vn", "-af", "volumedetect", "-f", "null", "-"])
        import re
        m = re.search(r"mean_volume: (-?[0-9.]+) dB", r.stderr)
        return float(m.group(1)) if m else None
    rows = []
    for t in tl["turns"][:12]:
        if t.get("mp4_user_start") is None:
            continue
        rows.append({"turn": t["idx"], "user_line_db": vol(t["mp4_user_start"], t["mp4_user_end"]), "after_db": vol(t["mp4_user_end"] + 0.5, min(t["mp4_window_end"], t["mp4_user_end"] + 8))})
    return {"whole_db": vol(0, 9999), "turns": rows}


def asr_audio(mp4: str, out: str) -> None:
    wav = os.path.join(out, "audio16k.wav")
    L.sh(["ffmpeg", "-y", "-v", "error", "-i", mp4, "-vn", "-ac", "1", "-ar", "16000", wav])
    if not L.ASR_SCRIPT:
        open(os.path.join(out, "asr.txt"), "w").write("skipped: set JUDGE_ASR_SCRIPT to an ASR script (<wav> <out.json> <threads>)\n")
        return
    r = subprocess.run([L.ASR_PY, L.ASR_SCRIPT, wav, os.path.join(out, "asr.json"), "8"], capture_output=True, text=True)
    open(os.path.join(out, "asr.txt"), "w").write(r.stdout[-6000:])


if __name__ == "__main__":
    sys.exit(main())
