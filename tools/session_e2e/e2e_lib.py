"""e2e_lib.py — shared helpers of the session E2E harness (stdlib only): TTS (cached), raw timeline -> structured
timeline, frame extraction, Claude vision judge, report writers.  See run_session.py / cut_proof.py."""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import re
import struct
import subprocess
import time
import urllib.request
import wave

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
# scratch root for caches (TTS clips, browser downloads); default <repo>/runs/session_e2e
SCRATCH = os.environ.get("E2E_SCRATCH") or os.path.join(REPO, "runs", "session_e2e")
TTS_CACHE = os.environ.get("E2E_TTS_CACHE", os.path.join(SCRATCH, "cache", "tts"))
CLAUDE = os.environ.get("JUDGE_CLAUDE") or os.environ.get("AVATAR_CLAUDE_BIN") or "claude"
# optional --asr pass: an external ASR script `<py> <script> <wav> <out.json> <threads>` (not shipped; e.g. a
# faster-whisper wrapper writing {"segments": [...]}) and the interpreter that has its deps
ASR_PY = os.environ.get("JUDGE_ASR_PY") or sys.executable
ASR_SCRIPT = os.environ.get("JUDGE_ASR_SCRIPT", "")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SR = 24000


# ----------------------------------------------------------------------------------------------------------- TTS
TTS_BASE = ("An adult man in his thirties talking to a friend on a video call, natural conversational American English, "
            "relaxed pace, not theatrical, no announcer voice.")


def tts_wav(text: str, voice: str = "ash", tone: str = "", cache: str = TTS_CACHE) -> dict:
    """Synthesize `text` (OpenAI gpt-4o-mini-tts, OPENAI_API_KEY) -> cached 24 kHz mono wav with
    0.15 s lead and 0.35 s tail of near-silence. Returns {path, dur_s, speech_s, cached, usd}."""
    os.makedirs(cache, exist_ok=True)
    instr = TTS_BASE + (" Tone: " + tone + "." if tone else "")
    key = hashlib.sha256(f"{voice}|{instr}|{text}".encode()).hexdigest()[:20]
    wavp = os.path.join(cache, key + ".wav")
    if os.path.exists(wavp):
        with wave.open(wavp) as w:
            return {"path": wavp, "dur_s": w.getnframes() / w.getframerate(), "cached": True, "usd": 0.0}
    k = os.environ.get("OPENAI_API_KEY") or os.environ.get("E2E_TTS_KEY")
    if not k:
        raise RuntimeError("no OPENAI_API_KEY (or E2E_TTS_KEY) in env for TTS")
    body = json.dumps({"model": "gpt-4o-mini-tts", "voice": voice, "input": text, "response_format": "pcm", "instructions": instr}).encode()
    last = None
    for a in range(3):
        try:
            req = urllib.request.Request("https://api.openai.com/v1/audio/speech", data=body,
                                         headers={"Authorization": "Bearer " + k, "content-type": "application/json"})
            pcm = urllib.request.urlopen(req, timeout=90).read()
            break
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 + 3 * a)
    else:
        raise RuntimeError(f"tts failed: {last}")
    n = len(pcm) // 2
    s = struct.unpack("<%dh" % n, pcm[: 2 * n])
    # trim leading/trailing near-silence of the synthesis, then pad
    thr = max(250, int(0.03 * max(abs(v) for v in s))) if s else 250
    i0 = next((i for i, v in enumerate(s) if abs(v) > thr), 0)
    i1 = next((i for i in range(n - 1, -1, -1) if abs(s[i]) > thr), n - 1)
    core = s[max(0, i0 - int(0.05 * SR)): min(n, i1 + int(0.1 * SR))]
    out = [0] * int(0.15 * SR) + list(core) + [0] * int(0.35 * SR)
    with wave.open(wavp, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(struct.pack("<%dh" % len(out), *out))
    usd = len(text) * 15e-6   # list-price estimate
    return {"path": wavp, "dur_s": len(out) / SR, "cached": False, "usd": usd}


# ----------------------------------------------------------------------------------------------------------- timeline
def load_jsonl(p: str) -> list:
    out = []
    if not os.path.exists(p):
        return out
    for ln in open(p):
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except Exception:  # noqa: BLE001
                pass
    return out


def build_timeline(out_dir: str, script: dict, drv: dict, rec_meta: dict | None) -> dict:
    """raw.jsonl + samples.jsonl + driver.json -> timeline.json: per-turn structured events and latencies. All times in
    seconds from `live` (t=0). mp4_t = recording time (None when there is no recording)."""
    raw = load_jsonl(os.path.join(out_dir, "raw.jsonl"))
    samples = load_jsonl(os.path.join(out_dir, "samples.jsonl"))
    first_user = next((r["w"] for r in raw if r["kind"] == "user_turn_start"), 1 << 62)
    live = next((r["w"] for r in reversed(raw) if r["kind"] == "live" and r["w"] <= first_user), raw[0]["w"] if raw else 0)
    cts = (rec_meta or {}).get("client_ts")
    rel = lambda w: round((w - live) / 1000.0, 3)       # noqa: E731
    mp4t = (lambda w: round((w - cts) / 1000.0, 3)) if cts else (lambda w: None)    # noqa: E731
    starts = [r for r in raw if r["kind"] == "user_turn_start"]
    marks = {m.get("id"): m for m in (drv.get("turns") or [])}
    turns = []
    t_end_all = next((r["w"] for r in raw if r["kind"] == "stop_click"), (raw[-1]["w"] if raw else live))
    sturns = script.get("turns", [])
    for i, st in enumerate(starts):
        w0 = st["w"]; w1 = w0 + st["dur_ms"]
        nxt = starts[i + 1]["w"] if i + 1 < len(starts) else t_end_all
        sd = next((t for t in sturns if t.get("id") == st.get("id")), sturns[i] if i < len(sturns) else {})
        rows = [r for r in raw if w0 - 100 <= r["w"] < nxt]
        txu = [r for r in rows if r["kind"] == "tx" and r["who"] == "user"]
        txa = [r for r in rows if r["kind"] == "tx" and r["who"] == "avatar" and r["w"] >= w0]
        byid: dict = {}
        order = []
        for r in rows:
            if r["kind"] == "tool":
                key = r.get("id") or f"{r['name']}@{r['w']}"
                c = byid.get(key)
                if c is None:
                    c = byid[key] = {"t": rel(r["w"]), "name": r["name"], "args": "", "phases": []}; order.append(key)
                c["phases"].append(r["phase"])
                if r.get("args") and (r["phase"] in ("done", "early") or not c["args"]):
                    c["args"] = r["args"]
            elif r["kind"] == "tool_result":
                key = r.get("id") or next((k for k in reversed(order) if byid[k]["name"] == r["name"] and "result" not in byid[k]), None)
                c = byid.get(key)
                if c is None:
                    c = byid[key or r["name"]] = {"t": rel(r["w"]), "name": r["name"], "args": "", "phases": []}; order.append(key or r["name"])
                if r.get("phase") == "done" or "result" not in c:
                    c["result"] = r.get("result", ""); c["t_result"] = rel(r["w"])
        calls = [byid[k] for k in order if byid[k]["phases"] and any(p in ("done", "early", "start") for p in byid[k]["phases"])]
        motions = [{"t": rel(r["w"]), "mp4": mp4t(r["w"]), "name": r.get("name"), "src": r.get("src"), "phase": r.get("phase"), "reason": r.get("reason", "")}
                   for r in rows if r["kind"] == "motion" and r.get("phase") in ("start", "end", "interrupted", "fail", "refused")]
        states = [{"t": rel(r["w"]), "posture": r.get("posture"), "facing": r.get("facing"), "active": r.get("active"), "at": r.get("at")}
                  for r in rows if r["kind"] == "state"]
        extra = [{"t": rel(r["w"]), "kind": r["kind"], "v": r.get("v")} for r in rows if r["kind"].startswith("state:")]
        # latencies
        sm = [s for s in samples if w1 <= s[0] < nxt]
        first_audio = next((s[0] for s in sm if s[2]), None)
        first_motion = next((r["w"] for r in rows if r["kind"] == "motion" and r.get("phase") == "start" and r["w"] >= w1 - 200), None)
        if first_motion is None:
            first_motion = next((r["w"] for r in rows if r["kind"] == "state" and r.get("active") and r["w"] >= w1), None)
        first_tool = next((r["w"] for r in rows if r["kind"] == "tool" and r["w"] >= w0), None)
        mark_rows = [r for r in rows if (r["kind"] in ("motion",) and re.search(r"chalk|stroke|write|draw|mark", str(r.get("name")), re.I)) or r["kind"] in ("state:board",)]
        board_tool = next((r["w"] for r in rows if r["kind"] == "tool" and r["phase"] in ("done", "early") and re.search(r"board|teach|chalk|write|draw", r["name"], re.I)), None)
        first_mark = next((r["w"] for r in mark_rows if board_tool and r["w"] >= board_tool), None)
        mt = next((json.loads(r["m"]) if r["m"].startswith("{") else {} for r in rows if r["kind"] == "metric_turn"), {})
        lat = {
            "last_word_to_first_audio_ms": round(first_audio - w1) if first_audio else None,
            "last_word_to_first_motion_ms": round(first_motion - w1) if first_motion else None,
            "last_word_to_first_tool_ms": round(first_tool - w1) if first_tool else None,
            "tool_to_first_mark_ms": round(first_mark - board_tool) if first_mark and board_tool else None,
            "page_audible_lat_ms": mt.get("audible_lat_ms"),
        }
        busy_end = max([s[0] for s in sm if s[2] or s[5]] or [w1])
        mk = marks.get(st.get("id"), {})
        aborted = [m for m in motions if m["phase"] in ("interrupted",)] + [c for c in calls if re.search(r"abort|go_home|stop|cancel|erase_all", c["name"], re.I)]
        turns.append({
            "idx": i + 1, "id": st.get("id"), "say": st.get("say"), "wait": st.get("wait"), "started_how": st.get("how"),
            "expect": sd.get("expect"), "judge_frames": sd.get("judge_frames", 6),
            "t_user_start": rel(w0), "t_user_end": rel(w1), "t_window_end": rel(nxt), "t_active_end": rel(busy_end),
            "mp4_user_start": mp4t(w0), "mp4_user_end": mp4t(w1), "mp4_window_end": mp4t(nxt), "mp4_active_end": mp4t(busy_end),
            "heard_by_avatar": " ".join(r["text"] for r in txu), "avatar_said": " ".join(r["text"] for r in txa),
            "tool_calls": calls, "motions": motions, "state_changes": states, "scene_state": extra, "aborts": aborted,
            "latency": lat, "flags": {k: v for k, v in mk.items() if k in ("no_reply", "idle_timeout", "during_missed")},
        })
    page = {}
    pr = os.path.join(out_dir, "page_report.json")
    if os.path.exists(pr):
        page = json.load(open(pr))
    errors = [{"t": rel(r["w"]), "msg": r.get("msg") or r.get("text")} for r in raw if r["kind"] in ("page_error",) or (r["kind"] == "console" and r.get("type") == "error" and not str(r.get("text", "")).startswith("INFO:"))]
    return {"t0_wall_ms": live, "client_ts": cts, "turns": turns, "errors": errors[:40], "page": {"voice": (page.get("voice") or {}).get("usd_est"), "devices": page.get("devices"), "render": (page.get("last") or {}).get("render"), "fps": (page.get("last") or {}).get("fps")},
            "usage": [r for r in raw if r["kind"] == "usage"][-1:], "duration_s": rel(t_end_all)}


# ----------------------------------------------------------------------------------------------------------- video
def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def probe(mp4: str) -> dict:
    r = sh(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,nb_frames,sample_rate",
            "-of", "json", mp4])
    try:
        j = json.loads(r.stdout)
    except Exception:  # noqa: BLE001
        return {}
    v = next((s for s in j.get("streams", []) if s["codec_type"] == "video"), {})
    a = next((s for s in j.get("streams", []) if s["codec_type"] == "audio"), {})
    fr = v.get("avg_frame_rate", "0/1").split("/")
    dur = float(j.get("format", {}).get("duration") or 0)
    nb = int(v.get("nb_frames") or 0)
    fps_real = round(nb / dur, 2) if nb and dur else (round(float(fr[0]) / float(fr[1]), 2) if len(fr) == 2 and float(fr[1]) else 0)
    return {"duration_s": round(dur, 2), "w": v.get("width"), "h": v.get("height"), "avg_frame_rate": v.get("avg_frame_rate"), "nb_frames": nb,
            "fps_real": fps_real, "video": v.get("codec_name"), "audio": a.get("codec_name"), "audio_rate": a.get("sample_rate")}


def real_fps(mp4: str) -> dict:
    """distinct-picture frame rate over the whole file (ffprobe packet timestamps): avg and the longest gap"""
    r = sh(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time", "-of", "csv=p=0", mp4])
    ts = sorted(float(x) for x in r.stdout.split() if x.strip() not in ("", "N/A"))
    if len(ts) < 3:
        return {}
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    return {"packets": len(ts), "fps": round((len(ts) - 1) / (ts[-1] - ts[0]), 2), "max_gap_s": round(max(gaps), 3),
            "gaps_over_150ms": sum(1 for g in gaps if g > 0.15)}


def grab(mp4: str, t: float, out: str, width: int = 960) -> bool:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    r = sh(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", mp4, "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3", out])
    return os.path.exists(out) and os.path.getsize(out) > 500


def strip_sheets(mp4: str, out_dir: str, every_s: float = 2.0, cols: int = 5, rows: int = 4, w: int = 320) -> list[str]:
    """one thumbnail every `every_s`, tiled cols x rows per sheet with the recording time burned in"""
    os.makedirs(out_dir, exist_ok=True)
    vf = (f"fps=1/{every_s},scale={w}:-2,drawtext=fontfile={FONT}:text='%{{pts\\:hms}}':x=4:y=4:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6,"
          f"tile={cols}x{rows}")
    sh(["ffmpeg", "-y", "-v", "error", "-i", mp4, "-an", "-vf", vf, "-q:v", "4", os.path.join(out_dir, "strip_%03d.jpg")])
    return sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.startswith("strip_"))


def turn_frame_times(tn: dict, n: int, max_span: float = 150.0) -> list[float]:   # 150 s: a lesson runs 50-70 s; the judge must see its end
    """n frame times (recording seconds) spread over the turn's active part, the last one near its end"""
    a = (tn.get("mp4_user_end") or 0) + 0.6
    b = min((tn.get("mp4_window_end") or a + 5) - 0.2, a + max_span)
    act = tn.get("mp4_active_end") or b
    b = max(a + 1.0, min(b, act + 1.5)) if act > a else b
    if tn.get("mp4_judge_end"):   # an interrupted turn's own work runs on until the stop lands (judge_all)
        b = max(b, tn["mp4_judge_end"])
    if n <= 1:
        return [b]
    return [round(a + (b - a) * k / (n - 1), 2) for k in range(n)]


IGNORE_MOTION = {"props", "perk", "blink", "idle", "look", "gaze", "breathe"}


def burst_sheet(mp4: str, t0: float, dur: float, out: str, n: int = 8, cols: int = 4, w: int = 400) -> bool:
    """n frames evenly over [t0, t0+dur] tiled into one labelled sheet (a short move like a backflip falls between sparse frames)"""
    from PIL import Image, ImageDraw
    import tempfile
    tmp = tempfile.mkdtemp(prefix="burst_")
    ims = []
    for k in range(n):
        t = max(0.0, t0 + dur * k / max(1, n - 1))
        p = os.path.join(tmp, f"{k}.jpg")
        if grab(mp4, t, p, w):
            im = Image.open(p).convert("RGB"); ImageDraw.Draw(im).text((6, 4), f"t={t:.2f}s", fill=(255, 255, 0))
            ims.append(im)
    if not ims:
        return False
    rows = (len(ims) + cols - 1) // cols
    W, H = ims[0].size
    sheet = Image.new("RGB", (W * cols, H * rows))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * W, (i // cols) * H))
    sheet.save(out, quality=88)
    return True


def motion_bursts(tn: dict, mp4: str, fdir: str, max_n: int = 2) -> list[str]:
    out, seen = [], set()
    ms = tn["motions"]
    for m in ms:
        if m["phase"] != "start" or m.get("mp4") is None or (m["name"] or "") in IGNORE_MOTION or m["name"] in seen:
            continue
        seen.add(m["name"])
        end = next((x for x in ms if x["name"] == m["name"] and x["phase"] in ("end", "interrupted", "fail") and x["t"] >= m["t"]), None)
        full = (end["t"] - m["t"]) if end else 3.0
        dur = min(6.0, max(1.5, full))
        p = os.path.join(fdir, f"turn{tn['idx']:02d}_burst_{m['name']}_t{m['mp4']:.1f}.jpg")
        if burst_sheet(mp4, m["mp4"] - 0.2, dur + 0.4, p):
            out.append(p)
        # a long job (walk -> pick up -> hold up / show) does its visible point at the END; the 6 s start window
        # only saw the walk (the '3-5 s lag' of the bursts). Add a second sheet over the job's last 5 s + 1.5 s after.
        if end and full > 6.5 and end.get("mp4") is not None:
            p2 = os.path.join(fdir, f"turn{tn['idx']:02d}_burst_{m['name']}-end_t{end['mp4'] - 5.0:.1f}.jpg")
            if burst_sheet(mp4, end["mp4"] - 5.0, 6.5, p2):
                out.append(p2)
        if len(seen) >= max_n:
            break
    return out


# ----------------------------------------------------------------------------------------------------------- judge
JUDGE_SYSTEM = (
    "You are a strict QA judge for a live 3D avatar app. A scripted tester played the user's side of a video call with a cartoon "
    "teacher character called Muse. You get one turn of the session: what the user said, what the avatar said, the tool calls the "
    "avatar's voice model made, an expectation, and still frames from the screen recording (the avatar's room; a small self-view of "
    "the user sits in a corner - ignore it). Use the Read tool to LOOK at every frame file listed, then decide whether the "
    "EXPECTATION was met as the user would see it. Judge only what is visible/audible in the evidence. If the expectation is about "
    "writing or drawing, the marks must be actually legible in the frames (read them out in your reason). Do not give credit for "
    "intent, for what the avatar merely says it will do, or for what is hidden. PASS only with clear evidence; FAIL if the "
    "expected thing is absent, wrong or illegible; UNSURE only if the frames genuinely cannot show it. Also list any visual glitch "
    "you notice (clipping, freeze, avatar off-screen, black video, overlays, marks floating in front of the avatar)."
)
JUDGE_SCHEMA = {"type": "object", "properties": {
    "verdict": {"type": "string", "enum": ["PASS", "FAIL", "UNSURE"]},
    "reason": {"type": "string"},
    "seen": {"type": "string", "description": "what the frames show, in order, concise"},
    "glitches": {"type": "array", "items": {"type": "string"}}},
    "required": ["verdict", "reason", "seen", "glitches"]}


def judge_turn(tn: dict, frames: list[str], cwd: str, model: str = "sonnet") -> dict:
    ev = "\n".join((f"- {f}  (CONTACT SHEET of {os.path.basename(f).split('_')[2]}: 8 frames in time order, left-to-right then down, time labels on each tile; this is the dense view of the move)" if "_burst_" in f else
                    f"- {f}  (single frame, recording time {re.search(r'_t([0-9.]+)', os.path.basename(f)).group(1)} s)") for f in frames)
    calls = "; ".join(f"{c['name']}({c.get('args','')[:200]}) -> {str(c.get('result',''))[:160]}" for c in tn["tool_calls"]) or "(none)"
    mot = ", ".join(sorted({f"{m['name']}:{m['phase']}" for m in tn["motions"]})) or "(none)"
    prompt = (f"TURN {tn['idx']}\nUser said: {tn['say']!r}\nWhat the avatar heard (its ASR): {tn['heard_by_avatar']!r}\n"
              f"Avatar said (transcript): {tn['avatar_said']!r}\nTool calls: {calls}\nMotion events: {mot}\n"
              f"Timing: user finished at t={tn['t_user_end']}s; avatar active until t={tn['t_active_end']}s.\n"
              f"EXPECTATION: {tn['expect']}\n\nFrames (Read each, relative paths):\n{ev}\n\nReturn the JSON verdict.")
    cmd = [CLAUDE, "-p", "--model", model, "--tools", "Read", "--permission-mode", "dontAsk", "--no-session-persistence",
           "--setting-sources", "", "--add-dir", cwd, "--system-prompt", JUDGE_SYSTEM, "--output-format", "json", "--json-schema", json.dumps(JUDGE_SCHEMA)]
    last = None
    for _ in range(2):
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=cwd, timeout=600)
        try:
            d = json.loads(r.stdout)
            o = d.get("structured_output") or json.loads(re.search(r"\{.*\}", d.get("result") or "", re.S).group(0))
            o["cost_usd_list"] = d.get("total_cost_usd")
            if re.search(r"does not exist|could not (open|read)|couldn't (open|read)", str(o.get("reason", "")), re.I) and _ == 0:
                last = "frame files unreadable: " + str(o.get("reason"))[:100]; continue
            return o
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}; rc={r.returncode}; err={r.stderr[-200:]!r}; out={r.stdout[:200]!r}"
    return {"verdict": "UNSURE", "reason": "judge call failed: " + str(last), "seen": "", "glitches": []}


def judge_all(turns: list[dict], mp4: str, out_dir: str, jobs: int = 3, model: str = "sonnet", only: set | None = None) -> None:
    fdir = os.path.join(out_dir, "frames")
    todo = []
    # a turn cut by a 'during' interruption keeps doing its work (writing) until the stop
    # actually lands, ~2 s after the next line STARTS; its frames ended at that start and missed the marks it made. Its
    # window now runs to the next turn's first interrupted motion (+0.6 s); the next turn is judged on its own window as before.
    for i, tn in enumerate(turns[:-1]):
        nx = turns[i + 1]
        if isinstance(nx.get("wait"), dict) and nx["wait"].get("during"):
            cut = next((m["mp4"] for m in nx.get("motions", []) if m.get("phase") == "interrupted" and m.get("mp4") is not None), None)
            if cut is not None:
                tn["mp4_judge_end"] = round(cut + 0.6, 3)
    for tn in turns:
        if tn.get("mp4_user_end") is None:
            tn["judge"] = {"verdict": "UNSURE", "reason": "no recording to look at"}
            continue
        ts = turn_frame_times(tn, int(tn.get("judge_frames") or 6))
        fr = []
        for k, t in enumerate(ts):
            p = os.path.join(fdir, f"turn{tn['idx']:02d}_{k:02d}_t{t:.1f}.jpg")
            if grab(mp4, t, p):
                fr.append(p)
        fr = fr + motion_bursts(tn, mp4, fdir)
        tn["frames"] = [os.path.relpath(p, out_dir) for p in fr]
        if tn.get("expect") and fr and (only is None or tn["idx"] in only):
            todo.append((tn, fr))
        elif not tn.get("expect"):
            tn["judge"] = {"verdict": "N/A", "reason": "no expectation in the script"}

    def run(a):
        tn, fr = a
        rel = [os.path.relpath(p, out_dir) for p in fr]
        tn["judge"] = judge_turn(tn, rel, out_dir, model)
    with cf.ThreadPoolExecutor(jobs) as ex:
        list(ex.map(run, todo))


# ----------------------------------------------------------------------------------------------------------- report
def auto_checks(tn: dict) -> list[str]:
    fl = []
    lat = tn["latency"]
    if tn["flags"].get("no_reply"):
        fl.append("no reply/action within the wait")
    if tn["flags"].get("idle_timeout"):
        fl.append("turn never went idle (timeout)")
    if tn["flags"].get("during_missed"):
        fl.append("the 'during' condition never appeared")
    if not tn["heard_by_avatar"]:
        fl.append("avatar's ASR produced no transcript of the user's line")
    if lat["last_word_to_first_audio_ms"] is not None and lat["last_word_to_first_audio_ms"] > 4000:
        fl.append("slow first audio")
    return fl


def write_report(out_dir: str, rep: dict) -> None:
    json.dump(rep, open(os.path.join(out_dir, "report.json"), "w"), indent=1)
    L = []
    s = rep["session"]
    npass = sum(1 for t in rep["turns"] if (t.get("judge") or {}).get("verdict") == "PASS")
    nfail = sum(1 for t in rep["turns"] if (t.get("judge") or {}).get("verdict") == "FAIL")
    nun = sum(1 for t in rep["turns"] if (t.get("judge") or {}).get("verdict") == "UNSURE")
    L.append(f"# {s['script']}  -  {npass} PASS / {nfail} FAIL / {nun} UNSURE of {len(rep['turns'])} turns")
    L.append(f"url `{s['url_host']}`  build `{s.get('build','?')}`  mp4 `{s.get('mp4')}`  {s.get('video_desc','')}")
    L.append(f"voice cost est ${s.get('voice_usd')}  tts ${s.get('tts_usd')}  duration {s.get('duration_s')}s  errors {len(rep['errors'])}")
    L.append("")
    for t in rep["turns"]:
        j = t.get("judge") or {}
        lat = t["latency"]
        L.append(f"## {t['idx']}. {j.get('verdict','-')}  \"{t['say']}\"")
        L.append(f"- expect: {t.get('expect') or '-'}")
        if j.get("reason"):
            L.append(f"- judge: {j['reason']}")
        if j.get("glitches"):
            L.append(f"- glitches: {'; '.join(j['glitches'])}")
        L.append(f"- avatar said: {t['avatar_said'][:300] or '(nothing)'}")
        if t["tool_calls"]:
            L.append("- tools: " + "; ".join(f"{c['name']}({c.get('args','')[:90]})" for c in t["tool_calls"]))
        if t["aborts"]:
            L.append(f"- aborts: {len(t['aborts'])}")
        L.append(f"- latency ms: audio {lat['last_word_to_first_audio_ms']}  motion {lat['last_word_to_first_motion_ms']}  tool {lat['last_word_to_first_tool_ms']}  chalk {lat['tool_to_first_mark_ms']}  (started {t['started_how']} @ {t['t_user_start']}s)")
        ac = auto_checks(t)
        if ac:
            L.append("- AUTO: " + "; ".join(ac))
        if t.get("frames"):
            L.append("- frames: " + " ".join(t["frames"][:1] + t["frames"][-1:]) + f"  (+{max(0, len(t['frames'])-2)} between)")
        L.append("")
    if rep["errors"]:
        L.append("## page errors")
        for e in rep["errors"][:10]:
            L.append(f"- {e['t']}s {str(e['msg'])[:160]}")
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(L) + "\n")
