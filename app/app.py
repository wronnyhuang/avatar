"""avatar-live app server: the page + HTTP API, the voice session, the TIMELINE CLOCK, the render link.

    PORT=10250 python app/app.py        (listens on $HOST:$PORT, default 127.0.0.1:10250; state/ and data/ live in
                                         AVATAR_HOME, default this app/ directory; scripts/run_local.sh sets it)

Contract: docs/CONTRACT.md (wire format + browser transport over HTTP).
Every page/API request carries ?k=<secret> (AVATAR_HOME/state/secret, generated on first start, mode 600, never
logged); API calls also need the header x-requested-with: avatar-live. /health is open and carries no secrets.
Vendor keys come from the environment (see .env.example); the startup check prints which ones are set.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import hmac
import json
import os
import secrets
import signal
import sys
import time
import traceback
from collections import deque
from urllib.parse import quote

import aiohttp
import numpy as np
from aiohttp import web

HERE = os.path.dirname(os.path.abspath(__file__))


# a character's body / scene JSON (renderer/animals/<c>.body.json, <c>.props.json) for the page's
# move engine; absent for the 11 older characters (config carries nothing new for them)
_SCENE_CACHE: dict = {}


def char_scene(character: str) -> dict:
    if character in _SCENE_CACHE:
        return _SCENE_CACHE[character]
    out: dict = {}
    base = os.path.join(os.path.dirname(HERE), "renderer", "animals")
    for key in ("body", "props"):
        fp = os.path.join(base, f"{character}.{key}.json")
        if os.path.isfile(fp):
            try:
                with open(fp, encoding="utf-8") as f:
                    out[key] = json.load(f)
            except (OSError, ValueError):
                pass
    _SCENE_CACHE[character] = out
    return out
sys.path.insert(0, HERE)
import wire  # noqa: E402
from struct import error as struct_error  # noqa: E402
from providers import Box  # noqa: E402
from vendor import LiveVendor, scrub  # noqa: E402
from intent_hub import IntentHub  # noqa: E402  (fast local intent matcher on the mic)
import rtc  # noqa: E402
import gemini_live  # noqa: E402  (Gemini 3.8 Live tokens for the page's direct mode)
import react_link  # noqa: E402  (learned-reaction mode: planner worker process + batching)
import planner as kfplanner  # noqa: E402  (the live keyframe planner, docs/PLANNER.md 0-2)
import unified_link  # noqa: E402  (unified mode: fitter + unified planner + retarget worker, ?react=unified)
import behavior_coder  # noqa: E402  (program layer: standing behaviours -> checked per-frame programs)
from gesture_store import GestureStore  # noqa: E402  (the gestures the user taught, state/gestures/)
from attention_judge import AttentionJudge  # noqa: E402  (vision judge for attention candidates)
import recorder_hub  # noqa: E402  (session recordings -> <repo>/recordings/avatar-sessions or AVATAR_REC_ROOT, /api/rec/*)
import surfaces  # noqa: E402  (board + TV canvases streamed to the renderer as 'S' messages)
from teach_hub import TeachHub  # noqa: E402  (board plans -> the page, chalk marks -> the board surface)
import tv_agent  # noqa: E402  (the wall TV's illustration agent)
import affect  # noqa: E402  (the avatar's own feelings on its face and body)
import claude_door  # noqa: E402  (the `claude` CLI / Anthropic API door: board brain, TV, feelings)

HOME = os.environ.get("AVATAR_HOME") or HERE
STATE = os.path.join(HOME, "state")
DATA = os.path.join(HOME, "data")
SR = 24000
TICK = 480                     # 20 ms

CHARACTERS = {  # key: (name, what, voice)
    "fox": ("Felix", "fox", "verse"),
    "bear": ("Bruno", "bear", "ash"),
    "rabbit": ("Rosie", "rabbit", "shimmer"),
    "hippo": ("Hattie", "hippo", "ballad"),
    "monkey": ("Milo", "monkey", "echo"),
    "orb": ("Ori", "glowing orb", "sage"),
    "cube": ("Cubby", "cube", "alloy"),
    "drop": ("Dewdrop", "water drop", "marin"),
    "star": ("Stella", "star", "coral"),
    "pyramid": ("Pip", "pyramid", "cedar"),
    "blob": ("Bloop", "jelly blob", "coral"),
    "muse": ("Muse", "plush companion", "shimmer"),   # the new character (personality: rtc.PERSONALITY)
}

DEFAULTS = {
    "idle_stop_s": 900,              # stop the local render server after this long without activity (0 = never)
    "delay_ms": 300,
    "fps": 24,
    "bitrate": 1500000,
    "lookahead_ms": 100,            # 133 -> 100 (same viseme net, A-2L=2); -33 ms on every video round trip
    "backend": "auto",
    "quality_auto": "auto",
    "sizes": {"720p": [1280, 720], "540p": [960, 540], "360p": [640, 360]},
    "session_max_s": 0,          # 0 = no cap: a session lasts while the user uses it
    "no_speech_s": 0,            # 0 = never end for silence; the box idle rule (idle_stop_s) is the only stop
    "client_gone_s": 60,         # a phone tab that pauses (screen dim, app switch, network blip) gets a minute to come back
    "daily_cap_usd": 10.0,           # API spend cap per day (voice + planner + judges), metered locally in data/spend-*.json
    "vendor_jb_ms": 80,
    "ahead_ms": 600,
    "hang_ms": 600,
    "vad_rms": 350,
    "render_ready_timeout_s": 240,
    "box_ready_timeout_s": 600,
    "local": {"render_root": "", "rt_tree": "", "python": "", "args": None, "env": {"AVATAR_NVENC": "0"},
              "gpu_lock": None, "eta_s": None, "force_stub": False, "stub_args": []},
    "voice_default": "direct",      # 'direct' = the page's own WebRTC session to OpenAI; 'hub' = vendor.py bridge
    "playback_hold_ms": 150,        # reorder window for the page's playback-audio ('P') chunks
    # extra browser origins allowed to call this server's API across origins (CORS), e.g. a page served from another
    # host or a reverse proxy. Empty = same-origin only (the page this server serves).
    "allowed_origins": [],
    # direct voice defaults (measurements, REPORT-voice): server_vad 300 ms silence (250 measured -180 ms vs 400,
    # semantic_vad 'high' was slower), reasoning effort = model default, one-turn "say a few words + call" style
    "vad": "server:300", "effort": "", "move_style": "parallel",
    # learned-reaction mode (react_link.py header). DEFAULT rules: the page only goes learned with ?react=learned
    # or react.mode = "learned"; rules sessions never touch any of it.
    "react": dict(react_link.DEFAULTS),
    # commanded actions -> library / plan cache / streamed keyframes (app/planner.py). false = the server
    # refuses plans (the page's ?planner=0 path); a dict tunes it (planner.DEFAULTS). The model is a switch — default
    # "auto": the `claude` CLI if found, else ANTHROPIC_API_KEY, else OPENROUTER_API_KEY (Gemini 3.5 Flash Lite), else
    # OPENAI_API_KEY; env AVATAR_PLANNER_MODEL / AVATAR_PLANNER_FALLBACK override it.
    "planner": True,
}

P_HDR = __import__("struct").Struct("<BB2xQI")      # 'P' playback audio: type, flags(bit0 silent), pad, s0, n
P = 0x50


def load_rig_params(cfg: dict) -> list[str]:
    """config.rig_params = the renderer's anim/schema.py PARAMS (read without importing the renderer)."""
    import ast
    roots = [(cfg.get("local") or {}).get("render_root") or "", os.path.join(os.path.dirname(HERE), "renderer")]
    for root in roots:
        p = os.path.join(root, "anim", "schema.py") if root else ""
        if not p or not os.path.isfile(p):
            continue
        try:
            tree = ast.parse(open(p, encoding="utf-8").read())
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "PARAMS" for t in node.targets):
                    return [str(x) for x in ast.literal_eval(node.value)]
        except (OSError, ValueError, SyntaxError):
            continue
    return []


# config.body / config.props (docs/INTENT-RULES.md section 0) = renderer/animals/<c>.body.json and
# <c>.props.json next to the character's shader, in the first render root that has the shader; a key is absent when
# its file is (the 11 older characters have neither). Read per config message: two small files, edits apply live.
def load_char_files(cfg: dict, character: str) -> dict:
    roots = [(cfg.get("local") or {}).get("render_root") or "", os.path.join(os.path.dirname(HERE), "renderer")]
    for root in roots:
        d = os.path.join(root, "animals") if root else ""
        if not d or not os.path.isfile(os.path.join(d, character + ".glsl")):
            continue
        out = {}
        for kind in ("body", "props"):
            p = os.path.join(d, f"{character}.{kind}.json")
            try:
                with open(p, encoding="utf-8") as f:
                    out[kind] = json.load(f)
            except FileNotFoundError:
                pass
            except (OSError, ValueError) as e:
                log(f"config.{kind}: {p} unreadable: {type(e).__name__}: {e}")
        return out
    return {}
# end of body / scene files


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + scrub(msg), flush=True)


def rms16(pcm: bytes) -> float:
    if len(pcm) < 2:
        return 0.0
    a = np.frombuffer(pcm, dtype="<i2", count=len(pcm) // 2).astype(np.float32)
    return float(np.sqrt(np.mean(a * a)))


def pstats(xs) -> dict | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]  # noqa: E731
    return {"n": len(xs), "mean": round(sum(xs) / len(xs), 1), "p50": round(q(.5), 1), "p95": round(q(.95), 1),
            "min": round(xs[0], 1), "max": round(xs[-1], 1)}


def plan_summary(plans: list, performs: list) -> dict:
    """Planner: per-source counts and timings of the hub's plans + the page's perform marks (perf-clock ms):
    command -> first motion (first_motion_ms - call_ms) and command -> first speech (cue_audio_ms - call_ms)."""
    src: dict = {}
    for p in plans:
        k = str(p.get("source") or "?") + ("" if p.get("ok") else ":fail")
        src[k] = src.get(k, 0) + 1

    def d(m, a, b):
        x, y = m.get(a), m.get(b)
        return y - x if isinstance(x, (int, float)) and isinstance(y, (int, float)) else None
    marks = [p.get("marks") or {} for p in performs if isinstance(p.get("marks"), dict)]
    by = lambda s: [p.get("ms_first_pose") for p in plans if p.get("source") == s]   # noqa: E731
    return {"n": len(plans), "sources": src, "usd": round(sum(float(p.get("usd") or 0) for p in plans), 6),
            "planner_first_pose_ms": pstats(by("planner")),
            "planner_done_ms": pstats([p.get("ms_done") for p in plans if p.get("source") == "planner"]),
            "performs": len(performs),
            "command_to_first_motion_ms": pstats([d(m, "call_ms", "first_motion_ms") for m in marks]),
            "command_to_first_speech_ms": pstats([d(m, "call_ms", "cue_audio_ms") for m in marks]),
            "command_to_plan_first_pose_ms": pstats([d(m, "call_ms", "plan_first_pose_ms") for m in marks]),
            "command_to_ready_ms": pstats([d(m, "call_ms", "ready_ms") for m in marks])}


# ---------------------------------------------------------------------------------------------------------- spend
class Spend:
    """Per-day API spend file (data/spend-YYYY-MM-DD.json); `cap` is the daily limit sessions are refused at."""

    def __init__(self, cap: float):
        self.cap = cap
        self.day = ""
        self.rec: dict = {}
        self._load()

    def _path(self, day: str) -> str:
        return os.path.join(DATA, f"spend-{day}.json")

    def _load(self) -> None:
        day = dt.date.today().isoformat()
        if day == self.day:
            return
        self.day = day
        try:
            with open(self._path(day)) as f:
                self.rec = json.load(f)
        except (OSError, ValueError):
            self.rec = {"day": day, "usd": 0.0, "voice_usd": 0.0, "backend_usd": 0.0, "sessions": 0, "refused": 0}

    def add(self, cat: str, usd: float) -> None:
        if usd <= 0:
            return
        self._load()
        self.rec[cat + "_usd"] = round(self.rec.get(cat + "_usd", 0.0) + usd, 6)
        self.rec["usd"] = round(self.rec["usd"] + usd, 6)
        self.save()

    def bump(self, key: str) -> None:
        self._load()
        self.rec[key] = self.rec.get(key, 0) + 1
        self.save()

    def today(self) -> float:
        self._load()
        return float(self.rec["usd"])

    def save(self) -> None:
        tmp = self._path(self.day) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.rec, f, indent=1)
        os.replace(tmp, self._path(self.day))


# ---------------------------------------------------------------------------------------------------------- down queue
class DownQueue:
    """Long-poll batches. A poll carries after=<highest contiguous seq the page holds>; batches up to it are acked.
    A batch handed out but not acked within RESEND_S is presumed lost and handed out again (the page dedups by seq)."""
    RESEND_S = 1.5
    MAX_BATCH = 768 * 1024

    def __init__(self):
        self.pending: deque[bytes] = deque()
        self.pending_bytes = 0
        self.batches: dict[int, list] = {}
        self.next_seq = 1
        self.evt = asyncio.Event()
        self.dropped = 0

    def put(self, msg: bytes) -> None:
        self.pending.append(msg)
        self.pending_bytes += len(msg)
        if self.pending_bytes > 24 * 1024 * 1024:      # nobody polling: drop the oldest (the session dies soon)
            while self.pending and self.pending_bytes > 16 * 1024 * 1024:
                self.pending_bytes -= len(self.pending.popleft())
                self.dropped += 1
        self.evt.set()

    async def poll(self, after: int, hold: float) -> tuple[bytes, int]:
        for s in [s for s in self.batches if s <= after]:
            del self.batches[s]
        deadline = time.monotonic() + hold
        while True:
            now = time.monotonic()
            for s in sorted(self.batches):
                b = self.batches[s]
                if now - b[1] > self.RESEND_S:
                    b[1] = now
                    b[2] += 1
                    return wire.U32.pack(s) + b[0], s
            if self.pending:
                parts, n = [], 0
                while self.pending and (n == 0 or n + len(self.pending[0]) + 4 <= self.MAX_BATCH):
                    m = self.pending.popleft()
                    self.pending_bytes -= len(m)
                    parts.append(wire.U32.pack(len(m)) + m)
                    n += len(m) + 4
                s = self.next_seq
                self.next_seq += 1
                body = b"".join(parts)
                self.batches[s] = [body, now, 0]
                return wire.U32.pack(s) + body, s
            left = deadline - now
            if left <= 0:
                return wire.U32.pack(0), 0
            self.evt.clear()
            try:
                await asyncio.wait_for(self.evt.wait(), left)
            except asyncio.TimeoutError:
                pass


# ---------------------------------------------------------------------------------------------------------- timeline
class Timeline:
    """The hub's TIMELINE CLOCK. Every 20 ms the next 20 ms of avatar audio is guaranteed to exist: vendor audio if
    queued, else digital silence (inserted only on underrun, at its tick). Queued vendor audio gets its timeline
    position at once and is emitted ahead of the clock (bounded by ahead_ms), so the renderer can work ahead.
    Between turns the vendor's own digital silence is dropped (the clock fills silence); the first voiced packet of a
    turn lands jb_ms after the clock (a jitter cushion), later packets append contiguously."""

    def __init__(self, emit, cfg: dict):
        self.emit = emit
        self.jb = int(cfg["vendor_jb_ms"] * SR / 1000)
        self.ahead = int(cfg["ahead_ms"] * SR / 1000)
        self.hang = cfg["hang_ms"] / 1000
        self.q = bytearray()
        self.assigned = 0
        self.t0: float | None = None
        self.speaking = False
        self.last_voiced = -1e9
        self.st = {"silence_fill_ms": 0.0, "underrun_in_speech": 0, "underrun_in_speech_ms": 0.0,
                   "trimmed_vendor_silence_ms": 0.0, "voiced_ms": 0.0, "turns": 0, "max_ahead_ms": 0.0,
                   "late_ticks": 0, "max_tick_late_ms": 0.0}
        self.turn_starts: list[dict] = []

    def start(self, now: float) -> None:
        self.t0 = now

    def pos(self, now: float) -> int:
        return int((now - self.t0) * SR)

    def push(self, pcm: bytes, now: float) -> dict | None:
        """Vendor audio arrived. Returns turn info when this packet starts a new avatar turn."""
        if self.t0 is None:
            return None
        voiced = rms16(pcm) > 20
        turn = None
        if voiced:
            if not self.speaking:
                target = self.pos(now) + TICK + self.jb
                if not self.q and self.assigned < target:
                    self._silence(target - self.assigned)
                self.st["turns"] += 1
                turn = {"s": self.assigned + len(self.q) // 2, "at": now, "clock": self.pos(now)}
                self.turn_starts.append(turn)
            self.speaking = True
            self.last_voiced = now
            self.st["voiced_ms"] += len(pcm) / 2 / SR * 1000
        elif not self.speaking or now - self.last_voiced > self.hang:
            self.speaking = False
            self.st["trimmed_vendor_silence_ms"] += len(pcm) / 2 / SR * 1000
            return None
        self.q += pcm
        self._assign(now)
        return turn

    def _silence(self, n: int) -> None:
        if n <= 0:
            return
        self.emit(self.assigned, bytes(2 * n), False)
        self.assigned += n
        self.st["silence_fill_ms"] += n / SR * 1000

    def _assign(self, now: float) -> None:
        limit = self.pos(now) + TICK + self.ahead
        take = min(len(self.q) // 2, limit - self.assigned)
        if take > 0:
            chunk = bytes(self.q[:2 * take])
            del self.q[:2 * take]
            self.emit(self.assigned, chunk, True)
            self.assigned += take
            self.st["max_ahead_ms"] = max(self.st["max_ahead_ms"], (self.assigned - self.pos(now)) / SR * 1000)

    def tick(self, now: float, late_s: float) -> None:
        if late_s > 0.01:
            self.st["late_ticks"] += 1
        self.st["max_tick_late_ms"] = max(self.st["max_tick_late_ms"], late_s * 1000)
        self._assign(now)
        need = self.pos(now) + TICK
        if self.assigned < need:
            if self.speaking and now - self.last_voiced <= self.hang:
                self.st["underrun_in_speech"] += 1
                self.st["underrun_in_speech_ms"] += (need - self.assigned) / SR * 1000
            self._silence(need - self.assigned)
        if self.speaking and now - self.last_voiced > self.hang and not self.q:
            self.speaking = False


# ---------------------------------------------------------------------------------------------------------- render link
class RenderLink:
    """The server's WebSocket to the render server (the local child process)."""

    def __init__(self, hub: "Hub", endpoint: dict):
        self.hub, self.ep = hub, endpoint
        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self.q: asyncio.Queue = asyncio.Queue()
        self.tasks: list[asyncio.Task] = []
        self.ready_evt = asyncio.Event()
        self.ready_info: dict | None = None
        self.params: dict | None = None
        self.used = False              # audio was sent since the last start
        self.closed = False
        self.err = ""
        self.last_stats: dict = {}
        self.pings: deque = deque(maxlen=200)
        self.frames = 0

    async def connect(self) -> None:
        url = self.ep["ws_url"] + ("&" if "?" in self.ep["ws_url"] else "?") + "key=" + quote(self.ep.get("key", ""))
        self.ws = await self.hub.http.ws_connect(url, heartbeat=20, max_msg_size=0, compress=0,
                                                 timeout=aiohttp.ClientWSTimeout(ws_close=5))
        self.tasks = [asyncio.create_task(self._reader()), asyncio.create_task(self._writer()),
                      asyncio.create_task(self._pinger())]

    def alive(self) -> bool:
        return not self.closed and self.ws is not None and not self.ws.closed

    async def start(self, params: dict, timeout: float) -> dict:
        self.ready_evt.clear()
        self.ready_info = None
        self.err = ""
        self.params = dict(params)
        self.used = False
        self.send_json({"type": "start", **params})
        await asyncio.wait_for(self._wait_ready(), timeout)
        return self.ready_info

    async def switch(self, character: str, timeout: float) -> dict:
        self.ready_evt.clear()
        self.ready_info = None
        self.err = ""
        if self.params:
            self.params["character"] = character
        self.send_json({"type": "character", "character": character})
        await asyncio.wait_for(self._wait_ready(), timeout)
        return self.ready_info

    async def _wait_ready(self) -> None:
        while not self.ready_evt.is_set():
            if self.closed:
                raise RuntimeError(self.err or "render link closed")
            if self.err and not self.ready_info:
                raise RuntimeError(self.err)
            try:
                await asyncio.wait_for(self.ready_evt.wait(), 0.5)
            except asyncio.TimeoutError:
                pass

    def send_audio(self, msg: bytes) -> None:
        if self.alive():
            self.used = True
            self.q.put_nowait(msg)

    def send_json(self, obj: dict) -> None:
        if self.alive():
            self.q.put_nowait(json.dumps(obj))

    def send_bytes(self, msg: bytes) -> None:          # 'S' surface updates, same queue as audio
        if self.alive():
            self.q.put_nowait(msg)

    async def _writer(self) -> None:
        try:
            while True:
                m = await self.q.get()
                if isinstance(m, bytes):
                    await self.ws.send_bytes(m)
                else:
                    await self.ws.send_str(m)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self.err = f"render send failed: {type(e).__name__}"

    async def _pinger(self) -> None:
        try:
            while self.alive():
                self.send_json({"type": "ping", "t": time.monotonic()})
                await asyncio.sleep(2)
        except asyncio.CancelledError:
            raise

    async def _reader(self) -> None:
        try:
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    if msg.data and msg.data[0] == wire.V:
                        self.frames += 1
                        self.hub.on_video(self, msg.data)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        m = json.loads(msg.data)
                    except ValueError:
                        continue
                    t = m.get("type")
                    if t == "ready":
                        self.ready_info = m
                        self.ready_evt.set()
                    elif t == "pong":
                        try:
                            self.pings.append((time.monotonic() - float(m.get("t"))) * 1000)
                        except (TypeError, ValueError):
                            pass
                        continue
                    elif t == "stats":
                        self.last_stats = m
                    elif t == "error":
                        self.err = str(m.get("message", "render error"))[:300]
                    self.hub.on_render_json(self, m)
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self.err = f"render read failed: {type(e).__name__}"
        finally:
            self.closed = True
            self.hub.on_link_closed(self)

    async def close(self) -> None:
        self.closed = True
        for t in self.tasks:
            t.cancel()
        if self.ws and not self.ws.closed:
            try:
                await self.ws.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------------------------------------- session
def persona(character: str) -> str:
    name, what, _ = CHARACTERS[character]
    return (f"You are {name}, a friendly cartoon {what} the user is video-chatting with, an animated character. "
            "Speak English. Give short, warm, playful replies — usually one or two sentences — and let the other "
            "person do most of the talking. Children may talk to you: keep everything kind and age-appropriate. "
            f"If asked who you are, you are {name} the {what}. You hear continuously and may receive occasional "
            "still images from the phone camera. Be observant about shown objects and silent gestures. "
            "If asked to move any part of your body, generate <|SpawnThinking|> so your animation tool acts promptly, "
            "then describe what you did. You can move your ears, nose, eyes, brows, mouth, cheeks, paws, arms, "
            "tail and torso independently; never refuse a physical request your animated body can do. "
            "If you are a geometric shape and lack an anatomical ear or tail, say so playfully and do a visible "
            "wiggle or bounce instead. Keep expressions vivid and fit the person's emotion. "
            + rtc.PERSONALITY.get(what, "") +    # per-character personality line (muse)
            "Never mention being an AI model unless asked directly.\n\n"
            "Turn-taking: begin answering the moment the person finishes a sentence and goes quiet — within about "
            "half a second — rather than waiting to be sure they are done; if they keep talking, stop and listen.")


class Session:
    def __init__(self, hub: "Hub", character: str, quality: str, codec: str, voice: str = "hub"):
        self.hub = hub
        self.cfg = hub.cfg
        self.sid = secrets.token_urlsafe(12)
        # direct = the page holds its own WebRTC voice session (rtc.py); the hub renders from the page's playback tap
        self.direct = voice == "direct"
        # planner mode (perform_action for every commanded action); config planner:false / page ?planner=0 = off
        self.planner = rtc.planner_on(self.cfg.get("planner"))
        self.sb: rtc.Sideband | None = None
        self.rtc_calls: list[dict] = []
        self.voice_mode = ""
        self.pb_next: int | None = None
        self.pb_hold: dict[int, tuple[float, int, int, bytes]] = {}
        self.pb = {"chunks": 0, "silent_chunks": 0, "dup": 0, "skipped_samples": 0, "voiced_ms": 0.0, "head_s": 0.0,
                   "rig": 0, "page_usage": 0, "max_hold_ms": 0.0}
        self.page_calls: list[dict] = []
        self.page_usd = 0.0
        self.suppressed_render_errors = 0
        self.character = character
        self.quality = quality
        self.codec = codec
        w, h = hub.size_for(quality)
        self.w, self.h = w, h
        self.created = time.time()
        self.t_mono = time.monotonic()
        self.down = DownQueue()
        self.stage = "warming"
        self.ended = asyncio.Event()
        self.end_reason = ""
        self.vendor: LiveVendor | None = None
        self.vendor_ready = False
        self.link: RenderLink | None = None
        self.video_ok = False
        self.tl = Timeline(self._emit, self.cfg)
        self.last_poll = time.monotonic()
        self.last_user_speech = time.monotonic()
        self.live_at = 0.0
        self.mic_next = 0
        self.mic_hold: dict[int, tuple[float, bytes]] = {}
        self.mic_up = None                          # wire.Up16to24 for 16 kHz 'M' chunks
        self.vendor_in = 0                 # samples sent to the vendor
        self.vendor_t0 = 0.0
        self.mic_debt = 0
        self.last_mic_voiced_at = 0.0
        self.last_camera_at = 0.0
        self.m = {"mic_chunks": 0, "mic_dropped_dup": 0, "mic_skipped": 0, "mic_gap_fill_ms": 0.0, "mic_debt_drop": 0,
                  "polls": 0, "poll_resends": 0, "ups": 0, "frames_fwd": 0, "keyframes_fwd": 0, "audio_msgs": 0,
                  "keyframe_requests": 0, "switches": 0, "camera_stills": 0, "animation_tools": 0}
        self.turns: list[dict] = []        # hub-side per avatar turn
        self.page_turns: list[dict] = []   # page-side per turn (first-audio delay as heard)
        self.page_stats: list[dict] = []
        self.transcript: list[list] = []
        self.actions: list[dict] = []
        self.render_ready: list[dict] = []
        self.status_log: list[list] = []
        self.errors: list[str] = []
        self.tasks: list[asyncio.Task] = []
        self.usd = {"voice": 0.0, "backend": 0.0, "planner": 0.0, "behavior": 0.0}
        try:
            hub.attention.reset_session()          # a new session starts with no verdict memory (rules persist)
        except Exception:                          # noqa: BLE001
            pass
        self.plans: list[dict] = []        # one record per {type:'plan'} (source, timings, usd)
        self.performs: list[dict] = []     # the page's {type:'perform'} latency marks
        self.style: dict = {}              # set_style {energy, mirroring, expressiveness} (None = automatic)
        self.styles: list[dict] = []       # every change, for the session record
        self.plan_running: set = set()
        self.intent_log: dict = {}
        self.intent_on = hub.intent.attach(asyncio.get_running_loop(), self.send, self.intent_log,
                                           character=character)   # act grammar per character
        self.rx: react_link.SessionRx | None = None      # set only when the page asks for learned mode
        self.surfaces = surfaces.Surfaces()              # board + TV canvases (kept for the whole session)
        self._pt_rig: tuple[float, float] | None = None  # (last rig t, monotonic arrival) for playback_t
        self._pt_pb: tuple[float, float] | None = None   # (playback tap head s, monotonic) fallback
        self.teach = TeachHub(self, log=lambda s: print(s, file=sys.stderr, flush=True))
        if self.intent_on:                               # stop words from the hub's streaming recognizer
            hub.intent.on_text = self._on_fast_asr      # + the affect judge's early 'pre' verdict
        self.tv: tv_agent.TVAgent | None = None          # created for a character with the wall TV
        self.affect: affect.Affect | None = None         # feelings on the face / body (muse)

    # -- outbound to the page
    def send(self, obj: dict) -> None:
        self.down.put(wire.pack_json(obj))

    def status(self, stage: str, detail: str = "", eta_s=None, box: dict | None = None) -> None:
        self.stage = stage
        self.status_log.append([round(time.monotonic() - self.t_mono, 3), stage, detail[:160]])
        msg = {"type": "status", "stage": stage, "detail": detail, "eta_s": None if eta_s is None else round(eta_s)}
        if box and box.get("elapsed_s") is not None:     # the box's own start clock (eta_s None = estimate overrun)
            msg["elapsed_s"], msg["usual_s"] = round(box["elapsed_s"]), round(box.get("usual_s") or 0)
        self.send(msg)

    def config_msg(self, ready: dict | None) -> dict:
        r = ready or {}
        return {"type": "config", "fps": r.get("fps", self.cfg["fps"]), "width": r.get("width", self.w),
                "height": r.get("height", self.h), "codec": r.get("codec", self.codec),
                "codec_string": r.get("codec_string", "avc1.42E01F" if self.codec == "h264" else "jpeg"),
                "delay_ms": self.hub.delay_ms(), "sample_rate": SR, "character": self.character,
                "name": CHARACTERS[self.character][0], "video": self.video_ok, "backend": r.get("backend"),
                **load_char_files(self.cfg, self.character),   # config.body / config.props (muse)
                "gpu": r.get("gpu"), "session_max_s": self.cfg["session_max_s"], "no_speech_s": self.cfg["no_speech_s"],
                "voice": "direct" if self.direct else "hub", "audio_down": not self.direct,
                "intent_mic": self.intent_on, "rig_params": r.get("rig_params") or [], "lookahead_ms": r.get("lookahead_ms", self.cfg["lookahead_ms"]),
                "delay_max_ms": int(self.cfg.get("delay_max_ms") or 400),
                # design note: lips stay synced with the voice -- the page may show frames at most this much behind
                # the voice (beyond it the late-frame rule drops them); delay_max_ms is what absorbs a slow link.
                "video_lag_max_ms": int(self.cfg.get("video_lag_max_ms", 120)),
                "react_mode": (self.cfg.get("react") or {}).get("mode", "rules"),
                "planner": bool(self.planner) and bool(getattr(getattr(self.hub, "planner", None), "enabled", True)),  # page switch AND hub planner
                **char_scene(self.character)}   # config.body / config.props (the muse only)

    def _emit(self, s0: int, pcm: bytes, voiced: bool) -> None:
        msg = wire.pack_audio(s0, pcm)
        if self.link and self.video_ok:
            self.link.send_audio(msg)
        self.down.put(msg)
        self.m["audio_msgs"] += 1

    # -- lifecycle
    async def run(self) -> None:
        try:
            await self._bring_up()
            if self.ended.is_set():
                return
            self.live_at = time.monotonic()
            self.last_user_speech = self.live_at
            self.status("live", f"{CHARACTERS[self.character][0]} is listening" + ("" if self.video_ok else " (voice only — video unavailable)"))
            if self.direct:
                self.tasks.append(asyncio.create_task(self._direct_clock()))
            else:
                self.tasks.append(asyncio.create_task(self._clock()))
                self.tasks.append(asyncio.create_task(self._input_clock()))
            await self.ended.wait()
        except asyncio.CancelledError:
            self.end_reason = self.end_reason or "cancelled"
        except Exception as e:  # noqa: BLE001
            msg = scrub(str(e))[:300]
            self.errors.append(msg)
            log(f"session {self.sid[:6]} failed: {msg}")
            self.status("error", msg)
            self.end_reason = self.end_reason or "error"
        finally:
            await self._teardown()

    async def _bring_up(self) -> None:
        hub, cfg = self.hub, self.cfg
        bs = hub.box.status()
        self.status("warming", "starting the render box", bs.get("eta_s"), bs)
        await hub.box.ensure_up()
        box_t0 = time.monotonic()
        try:
            ep = await hub.box.wait_ready(cfg["box_ready_timeout_s"], on_progress=lambda s: self._box_progress(s))
            self.status("warming", f"loading {CHARACTERS[self.character][0]}", hub.char_eta())
            self.link = await hub.attach_render(self, ep)
            self.video_ok = True
            self.w = int((self.link.ready_info or {}).get("width", self.w))
            self.h = int((self.link.ready_info or {}).get("height", self.h))
            self.render_ready.append({**(self.link.ready_info or {}), "box_wait_ms": round((time.monotonic() - box_t0) * 1000)})
            self.surfaces_attach()                       # full canvases to the (new) render link
        except Exception as e:  # noqa: BLE001
            msg = scrub(str(e))[:200]
            self.errors.append("render: " + msg)
            log(f"session {self.sid[:6]}: video unavailable: {msg}")
            self.video_ok = False
            self.link = None
        if self.ended.is_set():
            return
        self.send(self.config_msg(self.link.ready_info if self.link else None))
        self.tv_attach()                                 # agent + warm model processes (muse only)
        if self.character == "muse" and (self.cfg.get("affect") or {}).get("enabled", True) is not False:
            try:
                self.affect = affect.Affect(send=self.send, log=lambda s: print(s, file=sys.stderr, flush=True))
                self.affect.prewarm()
            except Exception:  # noqa: BLE001
                self.affect = None
        if self.direct:
            return                       # the page dials the voice itself (/api/rtc); no hub vendor, no hub clock
        self.status("connecting", "calling " + CHARACTERS[self.character][0])
        await self._open_vendor()
        self.tl.start(time.monotonic())

    # the wall TV's illustration agent (docs/TV-AGENT.md). Frames go to self.surfaces "tv"; its state goes to
    # the page as {type:'tv_state'} (silent note for the voice model + window.__avatar.tv() for the e2e harness)
    def tv_attach(self) -> None:
        cfg = self.cfg.get("tv") or {}
        if self.character != "muse" or cfg.get("enabled") is False:
            return
        if self.tv is None:
            self.tv_log: list = []
            self.tv = tv_agent.TVAgent(surface=lambda: self.surfaces.get("tv"), on_state=self.send, cfg=cfg,
                                       log=lambda d: self.tv_log.append({"t": round(time.monotonic() - self.t_mono, 2), **d}),
                                       loop=asyncio.get_running_loop())
        if cfg.get("prewarm", True):
            self.tv.prewarm()

    def on_tv(self, m: dict) -> None:
        if self.tv is None:
            self.send({"type": "tv_state", "status": "failed", "phase": "idle", "error": "there is no TV here",
                       "id": m.get("id")})
            return
        if m.get("type") == "tv_clear":
            self.tv.clear()
        else:
            ctx = [(w, t) for w, t, _ in self.transcript[-8:]]
            self.tv.show(str(m.get("concept") or ""), context=ctx, mode="update" if m.get("mode") == "update" else "new")

    # surfaces sink + the playback clock their timestamps live on
    def surfaces_attach(self) -> None:
        """Point the surfaces at the current render link (muse only) and re-send the full canvases; else detach."""
        if self.link and self.video_ok and self.link.alive() and self.character == "muse" and not self.ended.is_set():
            self.surfaces.attach(self.link.send_bytes)
        else:
            self.surfaces.detach()

    def playback_t(self) -> float:
        """Seconds on the PLAYBACK timeline (the clock of the page's 'R' rig frames: AudioContext.currentTime - the
        playback start). Best estimate now: the newest rig frame's t + the time since it arrived (rig frames come every
        ~33 ms; this lags the page by its upload latency); else the newest playback-tap ('P') head + time since; else,
        hub voice, the hub Timeline clock (tl.pos). 0.0 before any of these exist."""
        now = time.monotonic()
        if self._pt_rig is not None and now - self._pt_rig[1] < 2.0:
            return self._pt_rig[0] + (now - self._pt_rig[1])
        if self._pt_pb is not None and now - self._pt_pb[1] < 2.0:
            return self._pt_pb[0] + (now - self._pt_pb[1])
        if self.tl.t0 is not None:
            return max(0.0, self.tl.pos(now) / SR)
        if self._pt_rig is not None:
            return self._pt_rig[0] + (now - self._pt_rig[1])
        return 0.0

    def _box_progress(self, s: dict) -> None:
        if self.stage == "warming":
            self.status("warming", s.get("detail") or "starting the render box", s.get("eta_s"), s)

    async def _open_vendor(self) -> None:
        name, what, voice = CHARACTERS[self.character]
        # the muse's move / object tools on this path too; mode: perform_action for every character
        act_tools = rtc.hub_act_tools(what, self.character, self.planner)
        v = LiveVendor(self.hub.http, on_audio=self._on_vendor_audio, on_transcript=self._on_transcript,
                       on_voice_usd=lambda u: self._usd("voice", u), on_backend_usd=lambda u, i, o: self._usd("backend", u),
                       on_end=lambda why: self.end(f"vendor: {why}"), on_action=self._on_action, log=log,
                       act_tools=act_tools, act_note=rtc.hub_act_note(what, self.character, self.planner),
                       on_act=self._on_act if act_tools else None, planner=self.planner)
        info = await v.open(rtc.hub_persona_planner(persona(self.character)) if self.planner else persona(self.character), voice)
        self.vendor = v
        self.vendor_ready = True
        self.vendor_t0 = time.monotonic()
        self.vendor_in = 0
        self.render_ready.append({"vendor": info})

    def _usd(self, cat: str, usd: float) -> None:
        self.usd[cat] += usd
        self.hub.spend.add(cat, usd)

    def end(self, reason: str) -> None:
        if not self.ended.is_set():
            self.end_reason = reason
            log(f"session {self.sid[:6]} ending: {reason}")
            self.ended.set()

    async def _teardown(self) -> None:
        for t in self.tasks:
            t.cancel()
        sb, self.sb = self.sb, None
        if sb:
            await sb.hangup()            # the hub ends the vendor call itself (caps hold even if the page is gone)
            self.vstats = sb.stats()
        v, self.vendor = self.vendor, None
        self.vendor_ready = False
        if v:
            await v.close()
            self.vstats = v.stats()
        self.surfaces.detach()
        if self.tv is not None:
            await self.tv.close()
        if self.link and self.link.alive():
            self.link.send_json({"type": "stop"})
        self.stage = "ended"
        self.send({"type": "status", "stage": "ended", "detail": self.end_reason})
        self.send({"type": "spend", "usd_today": round(self.hub.spend.today(), 4)})
        if self.intent_on:
            self.hub.intent.detach()
        self.hub.session_over(self)

    async def _clock(self) -> None:
        loop = asyncio.get_running_loop()
        tl = self.tl
        n = 0
        last_spend = time.monotonic()
        try:
            while not self.ended.is_set():
                n += 1
                target = tl.t0 + n * 0.02
                d = target - loop.time()
                if d > 0:
                    await asyncio.sleep(d)
                now = loop.time()
                tl.tick(now, now - target)
                if now - last_spend > 5:
                    last_spend = now
                    self.send({"type": "spend", "usd_today": round(self.hub.spend.today() + self.unbilled(), 4)})
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log(f"clock failed: {traceback.format_exc()[-400:]}")
            self.end(f"clock error {type(e).__name__}")

    def unbilled(self) -> float:
        if self.direct:
            return self.sb.unbilled() if self.sb else 0.0
        return self.vendor.unbilled_usd() if self.vendor else 0.0

    # -- direct voice: the page's WebRTC session; the hub meters it and renders from the page's playback tap
    async def open_rtc(self, mode: str, sdp: str, vad: str, effort: str = "", style: str = "callfirst",
                       history: list | None = None, planner: bool | None = None) -> dict:
        name, what, voice = CHARACTERS[self.character]
        old, self.sb = self.sb, None
        if old:                          # re-dial (renew / drop / switch): the old call ends a few s later, so a planned
            async def _later(sb=old):    # renew overlaps instead of cutting the user off; a dead one is gone anyway
                await asyncio.sleep(8)
                await sb.hangup()
            self.tasks.append(asyncio.create_task(_later()))
        info = await rtc.create_call(self.hub.http, mode, sdp, name, what, voice, vad, log, effort, style, history,
                                     character=self.character,   # action memory on the muse only
                                     planner=self.planner if planner is None else planner)
        self.voice_mode = mode
        self.rtc_calls.append({"t": round(time.monotonic() - self.t_mono, 2), "mode": mode, "model": info["model"],
                               "vad": vad if mode == "realtime" else None, "effort": effort or None, "style": style,
                               "create_ms": info["ms"],
                               "character": self.character, "voice": voice, "has_id": bool(info["id"])})
        if info["id"]:
            sb = rtc.Sideband(self.hub.http, mode, info["id"], on_usd=lambda u: self._usd("voice", u),
                              on_activity=self._voice_activity, on_transcript=self._on_transcript,
                              on_closed=lambda why: log(f"session {self.sid[:6]} sideband: {why}"), log=log)
            sb.start()
            self.sb = sb
        else:
            log(f"session {self.sid[:6]}: {mode} call has no id (no sideband: page usage reports bill it)")
        self.last_user_speech = time.monotonic()
        log(f"session {self.sid[:6]} rtc {mode} {info['model']} {CHARACTERS[self.character][0]}/{voice} "
            f"vad={vad if mode == 'realtime' else '-'} effort={effort or '-'} in {info['ms']} ms")
        return info

    def _voice_activity(self, kind: str) -> None:
        self.last_user_speech = time.monotonic()
        self.hub.activity("speech")

    def on_playback(self, msg: bytes) -> None:
        """'P' from the page's AudioWorklet tap: the voice as it plays, on the 24 kHz playback timeline."""
        if len(msg) < P_HDR.size:
            return
        _, flags, s0, n = P_HDR.unpack_from(msg, 0)
        if n <= 0 or n > 10 * SR:
            return
        if flags & 1:
            payload = b""
        elif flags & wire.P_ULAW:                                     # mu-law tap (half the bytes)
            raw = msg[P_HDR.size:P_HDR.size + n]
            if len(raw) != n:
                return
            payload = wire.ulaw_decode(raw)
        else:
            payload = msg[P_HDR.size:P_HDR.size + 2 * n]
            if len(payload) != 2 * n:
                return
        self.pb["chunks"] += 1
        now = time.monotonic()
        if self.pb_next is not None and s0 + n <= self.pb_next:
            self.pb["dup"] += 1
            return
        self.pb_hold[s0] = (now, flags, n, payload)
        self._pb_flush(now)

    def _pb_flush(self, now: float) -> None:
        hold = self.cfg["playback_hold_ms"] / 1000
        while self.pb_hold:
            if self.pb_next is None:
                self.pb_next = min(self.pb_hold)
            if self.pb_next in self.pb_hold:
                at, flags, n, payload = self.pb_hold.pop(self.pb_next)
                self.pb["max_hold_ms"] = max(self.pb["max_hold_ms"], (now - at) * 1000)
                self._pb_forward(self.pb_next, flags, n, payload)
                self.pb_next += n
                continue
            first = min(self.pb_hold)
            if first < self.pb_next:                         # overlaps what was sent: trim or drop
                at, flags, n, payload = self.pb_hold.pop(first)
                cut = self.pb_next - first
                if cut < n:
                    self.pb_hold[self.pb_next] = (at, flags, n - cut, payload[2 * cut:] if payload else b"")
                else:
                    self.pb["dup"] += 1
                continue
            if now - min(at for at, *_ in self.pb_hold.values()) > hold:   # a chunk is lost: move on
                self.pb["skipped_samples"] += first - self.pb_next
                self.pb_next = first
                continue
            break

    def _pb_forward(self, s0: int, flags: int, n: int, payload: bytes) -> None:
        if flags & 1:
            self.pb["silent_chunks"] += 1
            pcm = bytes(2 * n)
        else:
            pcm = payload
            self.pb["voiced_ms"] += n / SR * 1000
        self.pb["head_s"] = round((s0 + n) / SR, 3)
        self._pt_pb = ((s0 + n) / SR, time.monotonic())  # playback_t fallback
        if self.link and self.video_ok:
            self.link.send_audio(wire.pack_audio(s0, pcm))
        if self.rx is not None and not flags & 1:
            self.rx.add_own(s0, pcm)

    def on_rig(self, m: dict) -> None:
        a = m.get("a")
        if not isinstance(a, list) or len(a) > 512:
            return
        self.pb["rig"] += 1
        try:                                             # playback_t
            self._pt_rig = (float(m.get("t")), time.monotonic())
        except (TypeError, ValueError):
            pass
        if self.link and self.video_ok:
            self.link.send_json({"type": "rig", "t": m.get("t"), "a": a})

    def on_page_usage(self, m: dict) -> None:
        """Usage the page saw on its data channel. Billed only when the hub's own sideband is not metering."""
        self.pb["page_usage"] += 1
        if self.sb and self.sb.attached:
            return
        u = m.get("usage") if isinstance(m.get("usage"), dict) else {}
        if m.get("mode") == "realtime":
            usd = rtc.realtime_usd(u)
        elif m.get("mode") == "live":
            usd = rtc.live_backend_usd(int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0))
            s = float(m.get("seconds") or 0)
            if self.sb and s > self.sb.live_seconds:                   # live voice seconds snapshot
                usd += (s - self.sb.live_seconds) / 60 * rtc.LIVE_USD_PER_MIN
                self.sb.live_seconds = s
        else:
            usd = 0.0
        usd = min(usd, 0.5)                                            # one report can't be worth more than this
        self.page_usd += usd
        self._usd("voice", usd)

    async def _direct_clock(self) -> None:
        try:
            last_spend = time.monotonic()
            while not self.ended.is_set():
                await asyncio.sleep(0.05)
                now = time.monotonic()
                self._pb_flush(now)
                self._watch(now)
                if now - last_spend > 5:
                    last_spend = now
                    self.send({"type": "spend", "usd_today": round(self.hub.spend.today() + self.unbilled(), 4)})
        except asyncio.CancelledError:
            raise

    # -- vendor callbacks
    def _on_vendor_audio(self, pcm: bytes) -> None:
        now = time.monotonic()
        turn = self.tl.push(pcm, now)
        if turn:
            ue = self.last_mic_voiced_at
            self.turns.append({"t": round(now - self.t_mono, 2), "timeline_s": round(turn["s"] / SR, 3),
                               "user_end_to_vendor_ms": round((now - ue) * 1000) if ue and now - ue < 15 else None,
                               "placed_ahead_ms": round((turn["s"] - turn["clock"]) / SR * 1000)})
            self.send({"type": "turn", "s": turn["s"], "hub_ms": self.turns[-1]["user_end_to_vendor_ms"]})

    def _on_fast_asr(self, text: str, final: bool, asr: str = "full") -> None:
        self.teach.on_asr(text, final, asr)
        if self.affect is not None and final and asr == "full":   # the user's words ~0.3 s after they stop -> the 'pre' feeling
            try:
                self.affect.heard(text, final)
            except Exception:  # noqa: BLE001
                pass

    def _on_transcript(self, who: str, delta: str, final: bool = False) -> None:
        if self.tv is not None:                          # the conversation tap (live updates)
            self.tv.hear(who, delta, final=final)
        if self.affect is not None and who != "you" and not final:   # hub-side voice paths stream deltas
            try:
                self.affect.avatar_delta(delta)
            except Exception:  # noqa: BLE001
                pass
        elif self.affect is not None and who == "you" and final:
            self.affect.user_said(delta)
        if who == "you":
            self.last_user_speech = time.monotonic()
        if self.transcript and self.transcript[-1][0] == who and time.monotonic() - self.transcript[-1][2] < 2.5:
            self.transcript[-1][1] += delta
            self.transcript[-1][2] = time.monotonic()
        else:
            self.transcript.append([who, delta, time.monotonic()])
            if len(self.transcript) > 400:
                del self.transcript[:100]
        self.send({"type": "transcript", "who": who, "text": delta})

    async def _on_act(self, name: str, args: dict) -> dict:
        """A move / object call from the hub's GPT Live 1 session runs on the page's motion engine (page.html
        onJSON 'act' → runCommand) and its real result comes back ('act_result'); recorded like the page's own calls."""
        self.act_seq = getattr(self, "act_seq", 0) + 1
        k = self.act_seq
        fut = asyncio.get_running_loop().create_future()
        self.act_wait = getattr(self, "act_wait", {})
        self.act_wait[k] = fut
        self.m["animation_tools"] += 1
        self.actions.append({"t": round(time.monotonic() - self.t_mono, 2), "name": name, "args": args})
        self.send({"type": "act", "id": k, "name": name, "args": args})
        try:
            return await fut
        finally:
            self.act_wait.pop(k, None)

    def _on_action(self, name: str, args: dict) -> None:
        self.m["animation_tools"] += 1
        self.actions.append({"t": round(time.monotonic() - self.t_mono, 2), "name": name, "args": args})
        if len(self.actions) > 200:
            del self.actions[:50]
        self.send({"type": "action", "action": name, "args": args, "source": "gpt-live-1"})

    # -- page → hub
    def on_mic(self, seq: int, pcm: bytes) -> None:
        self.m["mic_chunks"] += 1
        if seq < self.mic_next:
            self.m["mic_dropped_dup"] += 1
            return
        now = time.monotonic()
        self.mic_hold[seq] = (now, pcm)
        self._mic_flush(now)

    def _mic_flush(self, now: float) -> None:
        while True:
            if self.mic_next in self.mic_hold:
                _, pcm = self.mic_hold.pop(self.mic_next)
                self.mic_next += 1
                self._mic_forward(pcm, now)
                continue
            if self.mic_hold and now - min(t for t, _ in self.mic_hold.values()) > 0.15:
                nxt = min(self.mic_hold)
                self.m["mic_skipped"] += nxt - self.mic_next
                self.mic_next = nxt
                continue
            break

    def _mic_forward(self, pcm: bytes, now: float) -> None:
        if self.intent_on:
            self.hub.intent.feed(pcm)
        voiced = rms16(pcm) > self.cfg["vad_rms"]
        if voiced:
            self.last_mic_voiced_at = now
            self.hub.activity("mic")
            # counts as speech for the idle timer only when the avatar has been quiet a moment (echo leak guard)
            if now - self.tl.last_voiced > 1.5:
                self.last_user_speech = now
        if not self.vendor_ready or self.stage not in ("live",):
            return
        if self.mic_debt > 0 and not voiced:
            n = len(pcm) // 2
            self.mic_debt -= n
            self.m["mic_debt_drop"] += 1
            return
        if voiced:
            self.mic_debt = 0
        self.vendor_in += len(pcm) // 2
        asyncio.ensure_future(self.vendor.send_audio(pcm))

    async def _input_clock(self) -> None:
        """gpt-live-1's output is clocked by its input: an uplink gap stalls its speech. Bridge gaps with silence."""
        try:
            while not self.ended.is_set():
                await asyncio.sleep(0.1)
                now = time.monotonic()
                self._mic_flush(now)
                if not self.vendor_ready or self.stage != "live":
                    continue
                expected = int((now - self.vendor_t0) * SR)
                behind = expected - self.vendor_in
                if behind > int(0.3 * SR):
                    n = behind - int(0.1 * SR)
                    self.vendor_in += n
                    self.mic_debt += n
                    self.m["mic_gap_fill_ms"] += n / SR * 1000
                    await self.vendor.send_audio(bytes(2 * n))
                self._watch(now)
        except asyncio.CancelledError:
            raise

    def _watch(self, now: float) -> None:
        c = self.cfg
        if now - self.last_poll > c["client_gone_s"]:
            self.end("page gone (no polls)")
        elif c["session_max_s"] and self.live_at and now - self.live_at > c["session_max_s"]:
            self.end(f"{c['session_max_s'] // 60} minute limit")
        elif c["no_speech_s"] and self.live_at and now - self.last_user_speech > c["no_speech_s"]:
            self.end(f"no speech for {c['no_speech_s']} s")
        elif self.hub.spend.today() + self.unbilled() >= self.hub.spend.cap:
            self.end("daily spend cap reached")

    def on_control(self, m: dict) -> None:
        t = m.get("type")
        if t in ("activity", "character", "call", "intent_hit", "ui", "plan"):
            self.hub.activity(str(t))
        if t == "rig":
            self.on_rig(m)
            return
        if t == "bye":
            self.end("page closed")
        elif t == "usage":
            self.on_page_usage(m)
        elif t == "turn":
            tr = {k: v for k, v in m.items() if k != "type"}
            tr["t"] = round(time.monotonic() - self.t_mono, 2)
            self.page_turns.append(tr)
            if len(self.page_turns) > 400:
                del self.page_turns[:100]
            self.last_user_speech = time.monotonic()
        elif t == "voice_prompt":                                # the page's silent note to the hub-relayed voice
            if self.vendor is not None and hasattr(self.vendor, "prompt") and isinstance(m.get("text"), str):
                self.tasks.append(asyncio.create_task(self.vendor.prompt(m["text"][:800], bool(m.get("respond")))))
        elif t == "act_result":                                  # the page's answer to a hub-session act call
            fut = getattr(self, "act_wait", {}).get(m.get("id"))
            if fut is not None and not fut.done():
                r = m.get("result")
                fut.set_result(r if isinstance(r, dict) else {"ok": bool(r)})
        elif t == "call":
            c = {k: v for k, v in m.items() if k != "type"}
            c["t"] = round(time.monotonic() - self.t_mono, 2)
            self.page_calls.append(c)
            self.actions.append({"t": c["t"], "name": str(m.get("name", ""))[:40], "args": m.get("args")})
            # a body action the voice model itself started (not a repeat of the fast matcher's) supersedes
            # the matcher's last pose: "more" / "higher" must not re-apply that one
            ran = m.get("ran")
            if (self.intent_on and isinstance(ran, dict) and ran.get("ok") and not ran.get("already") and not ran.get("dup")
                    and str(m.get("name", "")) in ("do_action", "perform_action", "pose", "gesture", "expression", "look",
                                                    "release", "move", "object")):
                self.hub.intent.forget_context()
            if len(self.page_calls) > 300:
                del self.page_calls[:100]
        elif t == "activity":
            self.last_user_speech = time.monotonic()
        elif t == "transcript":
            who = "you" if m.get("who") == "you" else "avatar"
            if not (self.sb and self.sb.attached):
                self._on_transcript(who, str(m.get("text") or "")[:2000], final=True)
        elif t == "affect_cue":                                    # the first words of a reply
            if self.affect is not None:
                self.affect.cue(str(m.get("user") or "")[:400], str(m.get("reply") or "")[:400], str(m.get("stage") or "reply"))   # 'pre' = the user's words alone
        elif t in ("tv_show", "tv_clear"):                         # the voice model's TV tools
            self.on_tv(m)
        elif t == "keyframe":
            self.m["keyframe_requests"] += 1
            if self.link and self.video_ok:
                self.link.send_json({"type": "keyframe"})
        elif t == "camera_frame":
            image = m.get("image")
            now = time.monotonic()
            if (self.stage == "live" and self.vendor_ready and isinstance(image, str)
                    and image.startswith("data:image/jpeg;base64,") and len(image) <= 120_000
                    and now - self.last_camera_at >= .9):
                self.last_camera_at = now
                self.last_user_speech = now  # camera presence keeps a silent interaction alive, bounded by session_max_s
                self.m["camera_stills"] += 1
                obs = str(m.get("observation") or "")[:120]
                self.tasks.append(asyncio.create_task(self.vendor.send_image(image, obs)))
        elif t == "pose":
            pose = m.get("pose")
            if self.link and self.video_ok and isinstance(pose, dict):
                fields = {k: v for k, v in (pose or {}).items() if k in (
                    "smile", "brow_knit", "head_yaw", "head_pitch", "head_roll", "gaze_x", "gaze_y",
                    "blink_l", "blink_r", "brow_l", "brow_r", "ear_l", "ear_r", "ear_flick_l", "ear_flick_r",
                    "cheek", "nostril", "lip_open", "teeth_gap", "lip_width", "lip_protrude", "tongue_dental",
                    "arm_l", "arm_r", "wave_l", "wave_r", "point_l", "point_r", "paw_open_l", "paw_open_r",
                    "clap", "thumbs_up", "body_lean", "body_bounce", "body_yaw", "tail_wag", "tail_raise")
                          and isinstance(v, (int, float)) and -2 <= v <= 2}
                if fields:
                    self.link.send_json({"type": "pose", "pose": fields})
        elif t == "stats":
            m = {k: v for k, v in m.items() if k != "type"}
            m["t"] = round(time.monotonic() - self.t_mono, 2)
            for tr in m.pop("turns", []) or []:
                if isinstance(tr, dict):
                    self.page_turns.append(tr)
            self.page_stats.append(m)
            if len(self.page_stats) > 400:
                del self.page_stats[:100]
        elif t == "character":
            ch = m.get("character")
            if ch in CHARACTERS and ch != self.character and self.stage == "live":
                self.tasks.append(asyncio.create_task(self._switch(ch)))
        elif t == "event":
            self.actions.append({"t": round(time.monotonic() - self.t_mono, 2), "event": str(m.get("name", ""))[:40],
                                 "spoke": bool(m.get("respond"))})
        elif t == "hello":
            pass
        elif t == "react":
            self._react_mode(m)
        elif t == "rx_sync":
            if self.rx is not None:
                self.rx.add_sync(m.get("seq"), m.get("t"))
        elif t == "plan":                                          # (contract section 2)
            self.on_plan(m)
        elif t == "behavior_code":                                 # program layer: the coder model
            self.on_behavior_code(m)
        elif t in ("gesture_save", "gesture_forget", "gesture_list"):
            r = self.hub.gestures.handle(m)
            if r:
                self.send(r)
        elif t == "attention_candidate":                                   # frames reach the judge ONLY here
            async def _judge(m=m):
                r = await self.hub.attention.handle(m)
                if r:
                    self.send(r)
                    self.actions.append({"t": round(time.monotonic() - self.t_mono, 2), "attention": r.get("what", "")[:80],
                                         "worth_reply": bool(r.get("worth_reply")), "skipped": r.get("skipped")})
            self.tasks.append(asyncio.create_task(_judge()))
        elif t == "attention_rule":
            self.send(self.hub.attention.handle_rule(m))
        elif t == "style":                                         # set_style -> the unified worker
            self.on_style(m)
        elif t in ("teach_hello", "teach_plan", "teach_erase", "teach_cancel", "board_ink", "board_item_done", "board_erased", "teach_utterance"):
            self.teach.on_msg(m)                                   # (app/teach_hub.py)
            if t == "teach_plan" and self.tv is not None:           # the TV tap gets strict while the board is in use
                try:
                    self.tv.board_started()
                except Exception:  # noqa: BLE001
                    pass
        elif t == "perform":
            c = {k: v for k, v in m.items() if k != "type"}
            c["t"] = round(time.monotonic() - self.t_mono, 2)
            self.performs.append(c)
            if len(self.performs) > 300:
                del self.performs[:100]

    # -- : {type:'style', energy?, mirroring?, expressiveness?} (each 0..1, null = automatic) from the voice
    # model's set_style (or a behaviour's api.set_style). Kept per session; passed to the unified worker when ?react=unified
    # runs (now or when it starts later); harmless in every other react mode.
    def on_style(self, m: dict) -> None:
        st = unified_link.clean_style(m)
        if not st:
            return
        self.style.update(st)
        self.styles.append({"t": round(time.monotonic() - self.t_mono, 2), **st})
        if len(self.styles) > 200:
            del self.styles[:100]
        if isinstance(self.rx, unified_link.UnifiedRx):
            self.rx.set_style(st)

    # -- : {type:'plan'} -> library / cache / streamed keyframes as {type:'kf'} messages
    def on_plan(self, m: dict) -> None:
        pid = str(m.get("id") or "")[:40] or f"h{len(self.plans) + 1}"
        ch = m.get("character") if m.get("character") in CHARACTERS else self.character
        desc = str(m.get("description") or "").strip()[:600]
        state = m.get("state") if isinstance(m.get("state"), dict) else None
        pl = self.hub.planner
        why = ("planner off" if not pl.enabled else "empty description" if not desc
               else "daily spend cap reached" if self.hub.spend.today() + self.unbilled() >= self.hub.spend.cap
               else "busy" if len(self.plan_running) >= int(pl.cfg.get("max_concurrent") or 3) else "")
        if why:
            ev = kfplanner.refusal(why)
            self.send(kfplanner.kf_message(pid, ev))
            self._plan_record({"id": pid, "character": ch, "description": desc[:200], "ok": False, "source": "planner",
                               "error": why, "ms_done": 0, "usd": 0.0, "n_poses": 0})
            return
        words = str(m.get("words") or "").strip()[:400] or None      # the user's own words (library second try)
        task = asyncio.create_task(kfplanner.run_plan(pl, pid, ch, desc, state, self.send, http=self.hub.http,
                                                      on_usd=lambda u: self._usd("planner", u), record=self._plan_record,
                                                      words=words))
        self.plan_running.add(task)
        self.tasks.append(task)

        def _done(t, self=self):
            self.plan_running.discard(t)
            if t in self.tasks:
                self.tasks.remove(t)
        task.add_done_callback(_done)

    # -- program layer: {type:'behavior_code'} -> coder model -> static check + dry run -> {type:'behavior_code'}
    def on_behavior_code(self, m: dict) -> None:
        bc = self.hub.behavior_coder
        rid = str(m.get("id") or "")[:60]
        running = getattr(self, "behavior_running", 0)
        why = ("behaviors off" if not bc.enabled else "daily spend cap reached"
               if self.hub.spend.today() + self.unbilled() >= self.hub.spend.cap else "busy" if running >= 3 else "")
        if why:
            self.send({"type": "behavior_code", "id": rid, "ok": False, "name": m.get("name"), "error": why, "ms": 0})
            return
        if m.get("character") not in CHARACTERS:
            m = dict(m, character=self.character)

        async def run():
            self.behavior_running = getattr(self, "behavior_running", 0) + 1
            try:
                res = await bc.code(m, self.hub.http, on_usd=lambda u: self._usd("behavior", u))
            except Exception as e:  # noqa: BLE001
                res = {"type": "behavior_code", "id": rid, "ok": False, "name": m.get("name"), "error": f"coder error {type(e).__name__}"}
            finally:
                self.behavior_running -= 1
            self.send(res)
            self.actions.append({"t": round(time.monotonic() - self.t_mono, 2), "behavior": str(m.get("name") or "")[:40],
                                 "ok": res.get("ok"), "ms": res.get("ms")})
        task = asyncio.create_task(run())
        self.tasks.append(task)
        task.add_done_callback(lambda t, self=self: self.tasks.remove(t) if t in self.tasks else None)

    def _plan_record(self, rec: dict) -> None:
        rec = dict(rec)
        rec["t"] = round(time.monotonic() - self.t_mono, 2)
        rec.pop("t0", None)
        self.plans.append(rec)
        if len(self.plans) > 300:
            del self.plans[:100]
        log(f"session {self.sid[:6]} plan {rec.get('id')} {rec.get('character')} {rec.get('source')} "
            f"ok={rec.get('ok')} poses={rec.get('n_poses')} first={rec.get('ms_first_pose')} done={rec.get('ms_done')} ms "
            f"${float(rec.get('usd') or 0):.4f}{' err=' + str(rec.get('error'))[:80] if rec.get('error') else ''}")

    # -- learned-reaction mode
    def _react_mode(self, m: dict) -> None:
        rc = self.cfg.get("react") or {}
        if m.get("mode") == "unified":                             # unified mode (unified_link.py)
            self._unified_mode(m, rc)
            return
        want = m.get("mode") == "learned"
        if want and self.rx is None:
            if not rc.get("allow", True):
                self.send({"type": "react", "mode": "rules", "ok": False, "why": "learned mode not allowed on this server"})
                return
            self.rx = react_link.SessionRx(self.hub.react_worker(), rc, log)
            if m.get("blinks") in ("rules", "learned"):          # the page's ?react_blinks= wins over the config
                self.rx.blinks = m["blinks"]
            self.tasks.append(asyncio.create_task(self._react_up()))
        elif not want and self.rx is not None:
            self.rx = None
            self.send({"type": "react", "mode": "rules", "ok": True})

    def _unified_mode(self, m: dict, rc: dict) -> None:
        uc = unified_link.config(rc)
        if self.rx is not None:
            return
        if not (rc.get("allow", True) and uc.get("allow", True)):
            self.send({"type": "react", "mode": "rules", "ok": False, "why": "unified mode not allowed on this server"})
            return
        self.rx = unified_link.UnifiedRx(self.hub.unified_worker(), uc, log)
        if self.style:                                             # a style set before unified mode started
            self.rx.set_style(self.style)
        if m.get("blinks") in ("model", "rules"):
            self.rx.blinks = m["blinks"]
        self.tasks.append(asyncio.create_task(self._react_up()))

    async def _react_up(self) -> None:
        rx = self.rx
        ok = rx is not None and await rx.w.ensure()
        if self.rx is not rx:
            return
        if not ok:
            self.rx = None
            self.errors.append("react: planner worker unavailable")
            self.send({"type": "react", "mode": "rules", "ok": False, "why": "planner worker unavailable"})
            return
        if isinstance(rx, unified_link.UnifiedRx):
            self.send({"type": "react", "mode": "unified", "ok": True, "planner": rx.w.info.get("planner"),
                       "fitter": rx.w.info.get("fitter"), "retarget": rx.w.info.get("retarget"), "blinks": rx.blinks})
            return
        self.send({"type": "react", "mode": "learned", "ok": True, "planner": rx.w.info.get("planner"),
                   "retarget": rx.w.info.get("retarget"), "blinks": rx.blinks})

    async def react_forward(self, ticks: list) -> None:
        """The page's rig vectors of one uplink batch -> the planner's merge -> the renderer (the page's own rules vector
        when the worker is down or late: the page always computes it)."""
        rx = self.rx
        names = (self.link.ready_info or {}).get("rig_params") if self.link else None
        finals = None
        if rx is not None and names and all(len(a) == len(names) for _, a, _, _ in ticks):
            now = time.monotonic()
            want = now - rx.last_stats_at > 2.0
            if want:
                rx.last_stats_at = now
            finals = await rx.step(ticks, self.character, list(names), want_stats=want)
        if ticks:                                        # playback_t
            self._pt_rig = (float(ticks[-1][0]), time.monotonic())
        for i, (t, a, _, _) in enumerate(ticks):
            v = finals[i] if finals is not None else a
            self.pb["rig"] += 1
            if self.link and self.video_ok:
                self.link.send_json({"type": "rig", "t": t, "a": [round(float(x), 4) for x in v]})
        if rx is not None and finals is not None:
            now = time.monotonic()
            if now - rx.last_shown_at >= float(rx.cfg.get("shown_every_s", 0.4)):
                rx.last_shown_at = now
                self.send({"type": "rx_shown", "t": ticks[-1][0], "a": [round(float(x), 4) for x in finals[-1]]})

    async def _switch(self, ch: str) -> None:
        self.m["switches"] += 1
        self.status("switching", f"switching to {CHARACTERS[ch][0]}")
        self.surfaces.detach()                           # nothing to the renderer mid-switch
        self.character = ch
        self.hub.intent.set_character(ch)
        if self.planner:                                           # warm the door for the new prompt
            try:
                self.hub.planner.prewarm(ch)
            except Exception:  # noqa: BLE001
                pass
        old, self.vendor = self.vendor, None
        self.vendor_ready = False
        try:
            if old:
                await old.close()
            if self.link and self.video_ok:
                info = await self.link.switch(ch, self.cfg["render_ready_timeout_s"])
                self.render_ready.append({**(info or {}), "switch": True})
                self.surfaces_attach()                   # back to the muse -> full canvases re-sent
            self.send(self.config_msg(self.link.ready_info if self.link else None))
            if not self.direct:            # direct: the page re-dials /api/rtc with the new character's voice
                await self._open_vendor()
            self.mic_debt = 0
            self.last_user_speech = time.monotonic()
            self.status("live", f"{CHARACTERS[ch][0]} is listening")
        except Exception as e:  # noqa: BLE001
            self.errors.append("switch: " + scrub(str(e))[:200])
            self.status("error", "switch failed: " + scrub(str(e))[:160])
            self.end("switch failed")

    # -- summaries
    def summary(self, full: bool = False) -> dict:
        ps = self.page_stats
        last = ps[-1] if ps else {}
        sync = [x.get("sync_ms", {}).get("p50") for x in ps if isinstance(x.get("sync_ms"), dict)]
        syncabs = [x.get("sync_ms", {}).get("abs_p95") for x in ps if isinstance(x.get("sync_ms"), dict)]
        fps = [x.get("fps") for x in ps if x.get("fps")]
        rtt = [x.get("poll_rtt_ms", {}).get("p50") for x in ps if isinstance(x.get("poll_rtt_ms"), dict)]
        rtt95 = [x.get("poll_rtt_ms", {}).get("p95") for x in ps if isinstance(x.get("poll_rtt_ms"), dict)]
        fa = [t.get("first_audio_ms") for t in self.page_turns]
        out = {
            "sid": self.sid[:8], "character": self.character, "quality": self.quality, "size": [self.w, self.h],
            "codec": self.codec, "started": dt.datetime.fromtimestamp(self.created).isoformat(timespec="seconds"),
            "duration_s": round(time.monotonic() - self.t_mono, 1),
            "live_s": round(time.monotonic() - self.live_at, 1) if self.live_at else 0, "stage": self.stage,
            "end_reason": self.end_reason, "video": self.video_ok, "delay_ms": self.hub.delay_ms(),
            "provider": self.hub.box.provider_name,
            "first_audio_ms": pstats(fa), "first_audio_each": fa[-20:],
            "hub_user_end_to_vendor_ms": pstats([t.get("user_end_to_vendor_ms") for t in self.turns]),
            "sync_offset_ms_p50": pstats(sync), "sync_offset_abs_p95_ms": pstats(syncabs),
            "fps_shown": pstats(fps), "poll_rtt_ms_p50": pstats(rtt), "poll_rtt_ms_p95": pstats(rtt95),
            "late_frames": last.get("late"), "dropped_frames": last.get("dropped"),
            "audio_underruns": last.get("underruns"), "page_last": last,
            "render": {"ready": self.render_ready, "stats": self.link.last_stats if self.link else None,
                       "ping_ms": pstats(list(self.link.pings)) if self.link else None,
                       "frames": self.link.frames if self.link else 0},
            "timeline": dict(self.tl.st), "hub": dict(self.m), "usd": {k: round(v, 5) for k, v in self.usd.items()},
            "vendor": getattr(self, "vstats", None) or (self.vendor.stats() if self.vendor else None)
                      or (self.sb.stats() if self.sb else None),
            "errors": self.errors[-10:],
            "voice": "direct" if self.direct else "hub", "voice_mode": self.voice_mode, "rtc": self.rtc_calls[-6:],
            "playback": dict(self.pb), "page_usd": round(self.page_usd, 5),
            "suppressed_render_errors": self.suppressed_render_errors,
        }
        if self.rx is not None:
            out["react"] = self.rx.summary()
        if self.direct:
            out["first_audio_ms"] = pstats([t.get("audible_lat_ms") for t in self.page_turns])
            out["first_audio_tap_ms"] = pstats([t.get("tap_lat_ms") for t in self.page_turns])
            out["first_audio_each"] = [t.get("audible_lat_ms") for t in self.page_turns][-20:]
            out["call_done_ms"] = pstats([c.get("done_lat_ms") for c in self.page_calls])
        if self.plans or self.performs:                            # report
            out["plans"] = plan_summary(self.plans, self.performs)
        if self.styles:                                            # set_style changes
            out["style"] = {"now": dict(self.style), "changes": self.styles[-20:]}
        if full:
            out["turns_hub"] = self.turns
            out["turns_page"] = self.page_turns
            out["page_stats"] = self.page_stats[-60:]
            out["status_log"] = self.status_log
            out["transcript"] = [[w, t[:2000]] for w, t, _ in self.transcript][-200:]
            out["actions"] = self.actions[-200:]
            out["calls_page"] = self.page_calls[-200:]
            out["intents"] = self.intent_log.get("intents", [])[-200:]
            out["intent_asr"] = self.intent_log.get("asr", [])[-200:]
            out["plans_each"] = self.plans[-200:]
            out["performs"] = self.performs[-200:]
            try:
                out["teach"] = self.teach.stats()
            except Exception:  # noqa: BLE001
                pass
        if self.tv is not None:                                    # (was nesting the full block under it)
            out["tv"] = dict(self.tv.stats(), log=getattr(self, "tv_log", [])[-100:],
                             program=[[round(m - self.t_mono, 2), ln] for m, ln in self.tv.timeline[-1500:]])   # replay: tools/tv_replay.py
        return out


# ---------------------------------------------------------------------------------------------------------- hub
class Hub:
    def __init__(self):
        os.makedirs(STATE, exist_ok=True)
        os.makedirs(DATA, exist_ok=True)
        os.chmod(STATE, 0o700)
        self.cfg = self._load_cfg()
        self.secret = self._secret()
        self.spend = Spend(float(self.cfg["daily_cap_usd"]))
        bcfg = dict(self.cfg)
        bcfg["log_dir"] = DATA
        self.box = Box(bcfg, STATE, log)
        self.box.on_idle = self._idle_end
        self.session: Session | None = None
        self.history: deque = deque(maxlen=20)
        self.link: RenderLink | None = None
        self.link_lock = asyncio.Lock()
        self.http: aiohttp.ClientSession | None = None
        self.started = time.time()
        self.warm_task: asyncio.Task | None = None
        self.char_init_ms: dict[str, float] = {}
        self.warm_info: dict | None = None
        self.page_html = open(os.path.join(HERE, "page.html"), encoding="utf-8").read()
        # browser modules inlined BEFORE the page script (no extra requests), in this order
        mods = []
        for name in ("voice.js", "react_feat.js", "perception.js", "prior.js", "moves.js", "affect.js", "motion.js", "intent.js",   # moves.js before motion.js (affect.js too)
                     "actionlog.js", "cospeech.js",    # action memory + co-speech gestures (muse only)
                     "thinking.js",                    # the thinking gesture while the planner writes the move
                     "perform.js",                     # perform_action -> planner -> keyframe job, cue / check-in notes
                     "behavior.js",                   # the program layer (window.Behaviors)
                     "gestures.js", "attention.js", "turnbid.js",   # learned gestures, attention, turn bids
                     "voicetools.js",                  # the eight voice tools + events -> silent notes
                     "quiet.js",                       # talkativeness + quiet mode (controller; gate in voice.js)
                     "teach.js",                       # the chalkboard / TV performance (window.AvatarTeach)
                     "recorder.js"):                   # session recording (stage + both voices) -> the hub
            fp = os.path.join(HERE, name)
            if os.path.isfile(fp):
                with open(fp, encoding="utf-8") as f:
                    mods.append(f"<script>/* {name} */\n" + f.read() + "\n</script>")
        with open(os.path.join(HERE, "intent_rules.json"), encoding="utf-8") as f:    # rules for the page-side matcher
            mods.append("<script>window.INTENT_RULES=" + json.dumps(json.load(f), separators=(",", ":")) + ";</script>")
        self.page_html = self.page_html.replace("<!--MODULES-->", "\n".join(mods))
        # the old interactive.js (browser overlay + its own camera pipeline) only when the new perception/motion are absent
        if not os.path.isfile(os.path.join(HERE, "motion.js")):
            with open(os.path.join(HERE, "interactive.js"), encoding="utf-8") as f:
                interactive = f.read()
            self.page_html = self.page_html.replace("</body>", "<script>\n" + interactive + "\n</script></body>")
        self.rig_params = load_rig_params(self.cfg)
        self.intent = IntentHub(self.cfg.get("intent"), log)
        self.planner = kfplanner.Planner(self.cfg.get("planner"), log, cache_dir=os.path.join(HOME, "cache", "plans"))
        self.behavior_coder = behavior_coder.BehaviorCoder(self.cfg.get("behavior"), log)
        # taught gestures (state/gestures/) + the attention judge (state/attention.json; config "attention")
        self.gestures = GestureStore(STATE, log)
        self.attention = AttentionJudge(HOME, log, http_session_factory=lambda: self.http,
                                        on_usd=lambda u: self.spend.add("attention", u), cfg=self.cfg.get("attention") or {})
        self._react_worker: react_link.Worker | None = None
        self._unified_worker: unified_link.UnifiedWorker | None = None    # unified mode
        self._load_history()

    def react_worker(self) -> react_link.Worker:
        if self._react_worker is None:
            self._react_worker = react_link.Worker(self.cfg.get("react") or dict(react_link.DEFAULTS), log, DATA)
        return self._react_worker

    def unified_worker(self) -> unified_link.UnifiedWorker:      # unified mode
        if self._unified_worker is None:
            self._unified_worker = unified_link.UnifiedWorker(unified_link.config(self.cfg.get("react")), log, DATA)
        return self._unified_worker

    def _load_cfg(self) -> dict:
        p = os.path.join(STATE, "config.json")
        cfg = dict(DEFAULTS)
        try:
            with open(p) as f:
                cfg = deep_merge(DEFAULTS, json.load(f))
        except FileNotFoundError:
            pass
        except ValueError as e:
            log(f"state/config.json unreadable ({e}); using defaults")
        return cfg

    def _secret(self) -> str:
        p = os.path.join(STATE, "secret")
        try:
            with open(p) as f:
                s = f.read().strip()
            if len(s) >= 24:
                return s
        except FileNotFoundError:
            pass
        s = secrets.token_urlsafe(24)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(s + "\n")
        log("generated a new page secret (state/secret)")
        return s

    def _load_history(self) -> None:
        try:
            files = sorted(f for f in os.listdir(DATA) if f.startswith("session-") and f.endswith(".json"))[-20:]
        except OSError:
            files = []
        for f in files:
            try:
                with open(os.path.join(DATA, f)) as fh:
                    d = json.load(fh)
                d.pop("turns_hub", None), d.pop("page_stats", None), d.pop("transcript", None), d.pop("status_log", None)
                self.history.append(d)
            except (OSError, ValueError):
                pass

    def delay_ms(self) -> int:
        d = self.cfg["delay_ms"]
        if isinstance(d, dict):
            d = d.get("local", 300)
        return int(d)

    def size_for(self, quality: str) -> tuple[int, int]:
        """(0, 0) = let the render server pick the measured size for the character (its profiles.json)."""
        q = quality if quality in self.cfg["sizes"] else self.cfg["quality_auto"]
        if q not in self.cfg["sizes"]:
            return 0, 0
        w, h = self.cfg["sizes"][q]
        return int(w), int(h)

    def char_eta(self) -> float:
        return max(self.char_init_ms.values(), default=20000) / 1000

    # -- render link
    def render_params(self, character: str, w: int, h: int, codec: str) -> dict:
        c = self.cfg
        # optional per-size bitrate (config "bitrate_sizes": {"540p": 700000}); a flat 250 kbps turned the
        # station camera moves at 540p into blocks. Absent = the flat "bitrate" as before.
        br = c["bitrate"]
        bs = c.get("bitrate_sizes") or {}
        if isinstance(bs, dict) and w and h:
            for q, wh in (c.get("sizes") or {}).items():
                if list(wh) == [w, h] and isinstance(bs.get(q), (int, float)) and bs[q] > 0:
                    br = int(bs[q])
        p = {"character": character, "fps": c["fps"], "codec": codec, "bitrate": br,
             "lookahead_ms": c["lookahead_ms"], "backend": c["backend"], "opts": {}}
        if "nvenc" in c:
            p["nvenc"] = bool(c["nvenc"])
        if w and h:
            p["width"], p["height"] = w, h
        return p

    async def _link_for(self, ep: dict) -> RenderLink:
        if self.link and self.link.alive() and self.link.ep.get("ws_url") == ep.get("ws_url"):
            return self.link
        if self.link:
            await self.link.close()
        self.link = RenderLink(self, ep)
        await self.link.connect()
        return self.link

    async def attach_render(self, s: Session, ep: dict) -> RenderLink:
        async with self.link_lock:
            link = await self._link_for(ep)
            want = self.render_params(s.character, s.w, s.h, s.codec)
            if link.params == want and link.ready_info and not link.used:
                return link
            t0 = time.monotonic()
            await link.start(want, self.cfg["render_ready_timeout_s"])
            self.char_init_ms[s.character] = (time.monotonic() - t0) * 1000
            return link

    async def prewarm(self, character: str, quality: str, codec: str) -> None:
        """Page opened / picker moved: bring the box up and load the character before Start is pressed."""
        try:
            await self.box.ensure_up()
            ep = await self.box.wait_ready(self.cfg["box_ready_timeout_s"])
            async with self.link_lock:
                if self.session and self.session.stage != "ended":
                    return
                link = await self._link_for(ep)
                w, h = self.size_for(quality)
                want = self.render_params(character, w, h, codec)
                if link.params == want and link.ready_info and not link.used:
                    return
                t0 = time.monotonic()
                await link.start(want, self.cfg["render_ready_timeout_s"])
                self.char_init_ms[character] = (time.monotonic() - t0) * 1000
                log(f"prewarmed {character} {w}x{h} {codec} in {self.char_init_ms[character]:.0f} ms")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log(f"prewarm failed: {scrub(str(e))[:200]}")

    def on_video(self, link: RenderLink, msg: bytes) -> None:
        s = self.session
        if s and link is s.link and s.stage in ("live", "switching") and not s.ended.is_set():
            s.down.put(msg)
            s.m["frames_fwd"] += 1
            if msg[1] & 1:
                s.m["keyframes_fwd"] += 1

    def on_render_json(self, link: RenderLink, m: dict) -> None:
        s = self.session
        if link is self.link and m.get("type") in ("status", "ready", "error") and not (s and not s.ended.is_set()):
            p = link.params or {}
            self.warm_info = {"character": p.get("character"), "ready": m.get("type") == "ready",
                              "stage": m.get("stage") or m.get("type"), "detail": str(m.get("detail") or m.get("message") or "")[:120],
                              "eta_s": m.get("eta_s"), "at": time.time()}
        if not s or link is not s.link or s.ended.is_set():
            return
        t = m.get("type")
        if t == "error" and "unknown message type 'rig'" in str(m.get("message", "")):
            s.suppressed_render_errors += 1      # renderer without the rig handler yet (rig agent's branch)
            return
        if t == "ready" and s.stage == "live" and s.video_ok:
            s.render_ready.append({**m, "midsession": True})       # encoder change: the page reconfigures its decoder
            s.send(s.config_msg(m))
            s.surfaces_attach()                                     # a re-initialised renderer -> resync
            return
        if t == "status" and s.stage in ("warming", "switching"):
            s.status(s.stage, f"{m.get('stage', '')}: {m.get('detail', '')}"[:160], m.get("eta_s") or self.char_eta())
        elif t == "stats":
            s.send({"type": "metrics", "render": {k: m.get(k) for k in ("fps", "render_ms", "render_p95", "encode_ms", "late", "queue")}})
        elif t == "error":
            s.errors.append("render: " + str(m.get("message", ""))[:200])
            s.send({"type": "metrics", "render_error": str(m.get("message", ""))[:200]})

    def on_link_closed(self, link: RenderLink) -> None:
        s = self.session
        if s and link is s.link and not s.ended.is_set() and s.stage in ("live", "switching"):
            s.video_ok = False
            s.surfaces.detach()
            s.errors.append("render link closed: " + link.err)
            s.send({"type": "status", "stage": "live", "detail": "video lost — voice continues"})
        if link is self.link:
            self.link = None

    # -- sessions
    def session_over(self, s: Session) -> None:
        summ = s.summary(full=True)
        fn = os.path.join(DATA, f"session-{dt.datetime.fromtimestamp(s.created).strftime('%Y%m%d-%H%M%S')}-{s.sid[:6]}.json")
        try:
            with open(fn, "w") as f:
                json.dump(summ, f, indent=1, default=str)
        except OSError as e:
            log(f"session log write failed: {e}")
        short = s.summary()
        short.pop("page_last", None)
        self.history.append(short)
        self.box.session_active = False      # idle is counted from the user's last real activity, not from the session end
        log(f"session {s.sid[:6]} over: {s.end_reason}; first-audio {short['first_audio_ms']}; sync p50 "
            f"{short['sync_offset_ms_p50']}; fps {short['fps_shown']}; ${sum(s.usd.values()):.4f}")

    def activity(self, why: str = "") -> None:
        """Real activity (the user's speech, Start, page open, switch, commands) resets the box's idle timer; throttled."""
        now = time.monotonic()
        if now - getattr(self, "_act_at", 0.0) >= 5:
            self._act_at = now
            self.box.touch()

    def _idle_end(self) -> None:
        s = self.session
        if s and not s.ended.is_set():
            s.end(f"idle {self.box.idle_s / 60:.0f} min")

    async def new_session(self, character: str, quality: str, codec: str, voice: str = "hub") -> Session:
        old = self.session
        if old and not old.ended.is_set():
            old.end("replaced by a new session")
            try:
                await asyncio.wait_for(old.task, 8)
            except Exception:  # noqa: BLE001
                pass
        s = Session(self, character, quality, codec, voice)
        self.session = s
        self.box.session_active = True
        self.box.touch()
        self.spend.bump("sessions")
        s.task = asyncio.create_task(s.run())
        return s


# ---------------------------------------------------------------------------------------------------------- http
def make_app(hub: Hub) -> web.Application:
    HDR = "avatar-live"

    def authed(req: web.Request, api: bool = True) -> bool:
        k = req.query.get("k", "")
        if not k or not hmac.compare_digest(k.encode(), hub.secret.encode()):
            return False
        if api and req.headers.get("x-requested-with") != HDR:
            return False
        return True

    def deny() -> web.Response:
        return web.json_response({"ok": False, "error": "forbidden"}, status=403)

    def nocache(r: web.StreamResponse) -> web.StreamResponse:
        r.headers["Cache-Control"] = "no-store"
        r.headers["Referrer-Policy"] = "no-referrer"
        r.headers["X-Robots-Tag"] = "noindex"
        return r

    async def page(req: web.Request):
        if not authed(req, api=False):
            return web.Response(status=403, text="This link is missing its key.", content_type="text/plain")
        return nocache(web.Response(text=hub.page_html, content_type="text/html", charset="utf-8"))

    async def vision_asset(req: web.Request):
        if not authed(req, api=False):
            return deny()
        name = req.match_info.get("name", "")
        allowed = {"vision_worker.js", "vision_bundle.js", "vision_wasm_internal.js", "vision_wasm_internal.wasm",
                   "face_landmarker.task", "gesture_recognizer.task", "percept_worker.js", "pose_landmarker_lite.task"}
        if name not in allowed:
            raise web.HTTPNotFound()
        path = os.path.join(HERE, "vision", name)
        if not os.path.isfile(path):
            raise web.HTTPNotFound()
        compressed = "gzip" in req.headers.get("Accept-Encoding", "") and os.path.isfile(path + ".gz")
        resp = web.FileResponse(path + ".gz" if compressed else path)
        if compressed:
            resp.headers["Content-Encoding"] = "gzip"
            resp.headers["Vary"] = "Accept-Encoding"
        resp.headers["Cache-Control"] = "private, max-age=86400"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["X-Robots-Tag"] = "noindex"
        if name.endswith(".wasm"):
            resp.content_type = "application/wasm"
        elif name.endswith(".js"):
            resp.content_type = "application/javascript"
        else:
            resp.content_type = "application/octet-stream"
        return resp

    async def health(req: web.Request):
        s = hub.session
        return web.json_response({"ok": True, "service": "avatar-live", "up_s": round(time.time() - hub.started),
                                  "session": bool(s and not s.ended.is_set()), "box": hub.box.status().get("state"),
                                  "provider": hub.box.provider_name})

    def overview() -> dict:
        s = hub.session
        live = s if s and not s.ended.is_set() else None
        return {"box": hub.box.status(), "usd_today": round(hub.spend.today(), 4), "cap_usd": hub.spend.cap,
                "session": {"stage": live.stage, "character": live.character} if live else None,
                "characters": [{"key": k, "name": v[0], "what": v[1], "voice": v[2]} for k, v in CHARACTERS.items()],
                "delay_ms": hub.delay_ms(), "quality_auto": hub.cfg["quality_auto"], "warm": hub.warm_info,
                "setup_pending": str(hub.cfg.get("setup_pending") or ""),
                "limits": {"session_max_s": hub.cfg["session_max_s"], "no_speech_s": hub.cfg["no_speech_s"],
                           "idle_stop_s": hub.cfg["idle_stop_s"]},
                "planner": {"enabled": hub.planner.enabled, **hub.planner.stats}}

    async def body_json(req: web.Request) -> dict:
        try:
            d = await req.json()
            return d if isinstance(d, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    async def api_state(req: web.Request):
        if not authed(req):
            return deny()
        return nocache(web.json_response(overview()))

    async def api_warm(req: web.Request):
        if not authed(req):
            return deny()
        if hub.cfg.get("setup_pending"):
            return nocache(web.json_response(overview()))
        d = await body_json(req)
        ch = d.get("character") if d.get("character") in CHARACTERS else "fox"
        if hub.spend.today() >= hub.spend.cap:
            return nocache(web.json_response({**overview(), "capped": True}))
        if not (hub.session and not hub.session.ended.is_set()):
            if hub.warm_task and not hub.warm_task.done():
                hub.warm_task.cancel()
            hub.warm_task = asyncio.create_task(hub.prewarm(ch, str(d.get("quality", "auto")),
                                                            "jpeg" if d.get("codec") == "jpeg" else "h264"))
        else:
            hub.box.touch()
        return nocache(web.json_response(overview()))

    async def api_session(req: web.Request):
        if not authed(req):
            return deny()
        if hub.cfg.get("setup_pending"):
            return nocache(web.json_response({"ok": False, "error": str(hub.cfg["setup_pending"])}, status=503))
        d = await body_json(req)
        ch = d.get("character")
        if ch not in CHARACTERS:
            return web.json_response({"ok": False, "error": "unknown character"}, status=400)
        if hub.spend.today() >= hub.spend.cap:
            hub.spend.bump("refused")
            return nocache(web.json_response({"ok": False, "error": f"Today's ${hub.spend.cap:.0f} limit for the talking "
                                              "avatar is used up. It resets at midnight."}, status=429))
        if hub.warm_task and not hub.warm_task.done():
            hub.warm_task.cancel()
            try:
                await hub.warm_task
            except BaseException:  # noqa: BLE001
                pass
        voice = d.get("voice") if d.get("voice") in ("direct", "hub") else hub.cfg.get("voice_default", "direct")
        s = await hub.new_session(ch, str(d.get("quality", "auto")), "jpeg" if d.get("codec") == "jpeg" else "h264", voice)
        if d.get("planner") is False:                            # page ?planner=0
            s.planner = False
        elif s.planner:                                          # a claude_door planner warms the door
            try:
                hub.planner.prewarm(ch)                          # for this character's prompt (no-op on other routes)
            except Exception as e:  # noqa: BLE001
                log(f"planner prewarm: {type(e).__name__}: {str(e)[:120]}")
        log(f"session {s.sid[:6]} opened: {ch} {s.w}x{s.h} {s.codec} voice={voice}")
        return nocache(web.json_response({"ok": True, "sid": s.sid, "config": s.config_msg(None)}))

    def find(req: web.Request) -> Session | None:
        s = hub.session
        sid = req.query.get("sid", "")
        if s and sid and hmac.compare_digest(sid.encode(), s.sid.encode()):
            return s
        return None

    async def api_up(req: web.Request):
        if not authed(req):
            return deny()
        s = find(req)
        if s is None:
            return web.json_response({"ok": False, "error": "no such session", "ended": True}, status=410)
        body = await req.read()
        s.m["ups"] += 1
        rx_ticks = [] if s.rx is not None else None
        try:
            for msg in wire.iter_msgs(body):
                if not msg:
                    continue
                if msg[0] == P:
                    s.on_playback(msg)
                elif msg[0] == wire.M and len(msg) >= 8:
                    _, seq = wire.M_HDR.unpack_from(msg, 0)
                    mf, pcm = msg[1], msg[8:]
                    if mf & wire.M_ULAW:
                        pcm = wire.ulaw_decode(pcm)
                    if s.rx is not None:
                        s.rx.add_mic(seq, pcm, 16000 if mf & wire.M_16K else SR)
                    if mf & wire.M_16K:                               # the mic path downstream is 24 kHz
                        if s.mic_up is None:
                            s.mic_up = wire.Up16to24()
                        pcm = s.mic_up(pcm)
                    s.on_mic(seq, pcm)
                elif msg[0] == wire.R and len(msg) >= wire.R_HDR.size and rx_ticks is not None:
                    try:                                              # learned mode: merged after the batch
                        t, a, ab, rb = wire.unpack_rig_rx(msg)
                    except (ValueError, struct_error):
                        continue
                    rx_ticks.append((t, a, ab, rb))
                elif msg[0] == wire.U and len(msg) >= wire.U_HDR.size:      # unified mode: raw MediaPipe
                    if isinstance(s.rx, unified_link.UnifiedRx):
                        s.rx.add_mp(msg)
                elif msg[0] == wire.H and len(msg) >= wire.R_HDR.size:
                    if s.rx is not None:
                        try:
                            t, v, mk = wire.unpack_human(msg)
                        except (ValueError, struct_error):
                            continue
                        s.rx.add_human(t, v, msg[1], len(msg))
                elif msg[0] == wire.R and len(msg) >= wire.R_HDR.size:
                    try:
                        t, a = wire.unpack_rig(msg)
                    except (ValueError, struct_error):
                        continue
                    s.on_rig({"t": t, "a": a})
                elif msg[0] == wire.J:
                    try:
                        s.on_control(json.loads(msg[1:].decode()))
                    except ValueError:
                        pass
        except ValueError:
            return web.json_response({"ok": False, "error": "bad body"}, status=400)
        if rx_ticks:
            await s.react_forward(rx_ticks)
        return nocache(web.json_response({"ok": True, "stage": s.stage}))

    async def api_down(req: web.Request):
        if not authed(req):
            return deny()
        s = find(req)
        if s is None:
            return web.Response(status=410, body=b"", content_type="application/octet-stream")
        try:
            after = int(req.query.get("after") or req.headers.get("x-after") or "0")
        except ValueError:
            after = 0
        s.last_poll = time.monotonic()
        s.m["polls"] += 1
        t0 = time.monotonic()
        hold = 0.3 if not s.ended.is_set() else 0.0
        body, seq = await s.down.poll(after, hold)
        s.last_poll = time.monotonic()
        r = web.Response(body=body, content_type="application/octet-stream")
        r.headers["x-hold-ms"] = str(round((time.monotonic() - t0) * 1000, 1))
        return nocache(r)

    async def api_exchange(req: web.Request):
        # One request carries both directions (a reverse proxy in front may rate-limit requests per client).
        # The browser sends at most 7 exchanges/s, batching PCM and timestamped video.
        uploaded = await api_up(req)
        if uploaded.status >= 400:
            return uploaded
        return await api_down(req)

    async def api_rtc(req: web.Request):
        """SDP broker for the page's direct voice session. The key stays here; vendor errors are never reflected."""
        if not authed(req):
            return deny()
        s = find(req)
        if s is None or not s.direct or s.ended.is_set():
            return web.json_response({"ok": False, "error": "no such session", "ended": True}, status=410)
        d = await body_json(req)
        sdp, mode = d.get("sdp"), d.get("mode", "realtime")
        if mode not in ("realtime", "live") or not isinstance(sdp, str) or not sdp.startswith("v=0") \
                or len(sdp) > rtc.MAX_OFFER:
            return web.json_response({"ok": False, "error": "bad offer"}, status=400)
        ch = d.get("character")
        if ch in CHARACTERS and ch != s.character and not s.link:
            s.character = ch
        vad = str(d.get("vad") or hub.cfg.get("vad") or "server")[:24]
        effort = d.get("effort") if d.get("effort") in rtc.EFFORTS else str(hub.cfg.get("effort") or "")
        style = d.get("style") if d.get("style") in rtc.MOVE_STYLES else str(hub.cfg.get("move_style") or "parallel")
        if hub.spend.today() + s.unbilled() >= hub.spend.cap:
            hub.spend.bump("refused")
            return nocache(web.json_response({"ok": False, "error": f"Today's ${hub.spend.cap:.0f} limit for the talking "
                                              "avatar is used up. It resets at midnight."}, status=429))
        # sessions last forever now (re-dials on drops / before the vendor's max length): a rate, not a total
        recent = [c for c in s.rtc_calls if time.monotonic() - s.t_mono - c["t"] < 120]
        if len(recent) >= 8:
            return web.json_response({"ok": False, "error": "too many voice connects right now"}, status=429)
        try:
            hist = [h for h in (d.get("history") or []) if isinstance(h, dict) and h.get("role") in ("user", "assistant")
                    and isinstance(h.get("text"), str)][-20:] if isinstance(d.get("history"), list) else []
            info = await s.open_rtc(mode, sdp, vad, effort, style, hist,
                                    planner=s.planner and d.get("planner") is not False)
        except rtc.MissingKey as e:
            s.errors.append(str(e))
            log(str(e))
            return nocache(web.json_response({"ok": False, "error": str(e), "missing_key": e.env}, status=503))
        except rtc.VendorRefused:
            s.errors.append(f"rtc {mode} refused")
            return nocache(web.json_response({"ok": False, "error": "voice session unavailable"}, status=502))
        except Exception as e:  # noqa: BLE001
            s.errors.append(f"rtc {mode}: {type(e).__name__}")
            log(f"rtc {mode} failed: {type(e).__name__}")
            return nocache(web.json_response({"ok": False, "error": "voice session unavailable"}, status=502))
        return nocache(web.json_response({"ok": True, "sdp": info["sdp"], "mode": mode, "model": info["model"],
                                          "create_ms": info["ms"], "sideband": bool(info["id"])}))

    async def mock_wav(req: web.Request):
        """Scripted reply WAVs for voice mode 'mock' (eval runs that cost nothing): <AVATAR_HOME>/mock/<name>."""
        if not authed(req, api=False):
            return deny()
        name = req.match_info.get("name", "")
        if not name or "/" in name or name.startswith(".") or not name.endswith(".wav"):
            raise web.HTTPNotFound()
        path = os.path.join(HOME, "mock", name)
        if not os.path.isfile(path):
            raise web.HTTPNotFound()
        return nocache(web.FileResponse(path))

    async def api_bye(req: web.Request):
        # sendBeacon cannot set headers: k + the live sid are required, and all this can do is end that session
        if not authed(req, api=False):
            return deny()
        s = find(req)
        if s:
            s.end("page hidden/closed")
        return web.Response(status=204)

    async def api_stats(req: web.Request):
        if not authed(req, api=False):
            return deny()
        s = hub.session
        return nocache(web.json_response({**overview(), "spend": hub.spend.rec,
                                          "live": s.summary() if s and not s.ended.is_set() else None,
                                          "sessions": list(hub.history)[::-1]}, dumps=lambda o: json.dumps(o, default=str)))

    # Cross-origin callers: only config "allowed_origins" get CORS (default none = same-origin only); auth is unchanged
    # (?k= + x-requested-with on every API call).
    cors_hdrs = {"Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                 "Access-Control-Allow-Headers": "content-type, x-requested-with, x-after",
                 "Access-Control-Expose-Headers": "x-hold-ms", "Access-Control-Max-Age": "7200", "Vary": "Origin"}

    @web.middleware
    async def cors(req: web.Request, handler):
        origin = req.headers.get("Origin", "")
        ok = bool(origin) and origin in (hub.cfg.get("allowed_origins") or [])
        if ok and req.method == "OPTIONS" and req.headers.get("Access-Control-Request-Method"):
            return web.Response(status=204, headers={"Access-Control-Allow-Origin": origin, **cors_hdrs})
        try:
            resp = await handler(req)
        except web.HTTPException as e:
            if ok:
                e.headers["Access-Control-Allow-Origin"] = origin
                e.headers.update(cors_hdrs)
            raise
        if ok and not resp.prepared:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers.update(cors_hdrs)
        return resp

    app = web.Application(client_max_size=4 * 1024 * 1024, middlewares=[cors])
    app.router.add_get("/", page)
    app.router.add_get("/index.html", page)
    app.router.add_get("/vision/{name}", vision_asset)
    app.router.add_get("/health", health)
    app.router.add_get("/api/state", api_state)
    app.router.add_post("/api/warm", api_warm)
    app.router.add_post("/api/session", api_session)
    app.router.add_post("/api/up", api_up)
    app.router.add_post("/api/exchange", api_exchange)
    app.router.add_get("/api/down", api_down)
    app.router.add_post("/api/bye", api_bye)
    async def api_gemini(req: web.Request):
        """A one-use Gemini Live token with the setup locked here (gemini_live.py); the key stays on the server."""
        if not authed(req):
            return deny()
        s = find(req)
        if s is None or not s.direct or s.ended.is_set():
            return web.json_response({"ok": False, "error": "no such session", "ended": True}, status=410)
        d = await body_json(req)
        mode = d.get("mode") if d.get("mode") in gemini_live.MODES else "gemini"
        ch = d.get("character")
        if ch in CHARACTERS and ch != s.character and not s.link:
            s.character = ch
        if hub.spend.today() + s.unbilled() >= hub.spend.cap:
            hub.spend.bump("refused")
            return nocache(web.json_response({"ok": False, "error": f"Today's ${hub.spend.cap:.0f} limit for the talking "
                                              "avatar is used up. It resets at midnight."}, status=429))
        recent = [c for c in s.rtc_calls if time.monotonic() - s.t_mono - c["t"] < 120]
        if len(recent) >= 8:
            return web.json_response({"ok": False, "error": "too many voice connects right now"}, status=429)
        name, what, voice = CHARACTERS[s.character]
        hist = [h for h in (d.get("history") or []) if isinstance(h, dict) and h.get("role") in ("user", "assistant")
                and isinstance(h.get("text"), str)][-20:] if isinstance(d.get("history"), list) else []
        t0 = time.monotonic()
        try:
            info = await gemini_live.mint(hub.http, mode, name, what, voice, s.character, hist,
                                          planner=s.planner and d.get("planner") is not False)
        except rtc.MissingKey as e:
            s.errors.append(str(e))
            log(str(e))
            return nocache(web.json_response({"ok": False, "error": str(e), "missing_key": e.env}, status=503))
        except Exception as e:  # noqa: BLE001
            s.errors.append(f"gemini: {type(e).__name__}")
            log(f"gemini token failed: {type(e).__name__} {scrub(e)[:120]}")
            return nocache(web.json_response({"ok": False, "error": "Gemini voice session unavailable"}, status=502))
        s.rtc_calls.append({"t": round(time.monotonic() - s.t_mono, 2), "mode": mode, "model": info["model"],
                            "ms": round((time.monotonic() - t0) * 1000)})
        log(f"session {s.sid[:6]} gemini {mode} {info['model']} {name}/{voice} token in {(time.monotonic() - t0) * 1000:.0f} ms")
        return nocache(web.json_response({"ok": True, **info, "mode": mode}))

    app.router.add_post("/api/rtc", api_rtc)
    app.router.add_post("/api/gemini", api_gemini)
    app.router.add_get("/mock/{name}", mock_wav)

    async def prior_weights(req: web.Request):
        """The learned motion prior's weights (app/prior_weights.json, gzipped when the client accepts it)."""
        if not authed(req, api=False):
            return deny()
        path = os.path.join(HERE, "prior_weights.json")
        if not os.path.isfile(path):
            raise web.HTTPNotFound()
        gz = "gzip" in req.headers.get("Accept-Encoding", "") and os.path.isfile(path + ".gz")
        resp = web.FileResponse(path + ".gz" if gz else path)
        if gz:
            resp.headers["Content-Encoding"] = "gzip"
            resp.headers["Vary"] = "Accept-Encoding"
        resp.content_type = "application/json"
        resp.headers["Cache-Control"] = "private, max-age=86400"
        return resp
    app.router.add_get("/prior_weights.json", prior_weights)
    app.router.add_get("/api/stats", api_stats)

    # debug door onto the live session's surfaces (docs/SCENE.md section 7 "Test hooks")
    async def debug_surface_post(req: web.Request):
        if not authed(req):
            return deny()
        s = hub.session
        if s is None or s.ended.is_set():
            return web.json_response({"ok": False, "error": "no session"}, status=404)
        d = await body_json(req)
        name, op = d.get("surface", "board"), d.get("op", "")
        if name not in ("board", "tv"):
            return web.json_response({"ok": False, "error": "bad surface"}, status=400)
        surf = s.surfaces.get(name)
        now_t = s.playback_t()
        t = None
        if d.get("t_rel") is not None:
            try:
                t = now_t + float(d["t_rel"])
            except (TypeError, ValueError):
                return web.json_response({"ok": False, "error": "bad t_rel"}, status=400)
        try:
            if op == "draw_polyline":
                ts = [now_t + float(x) for x in d["ts_rel"]] if isinstance(d.get("ts_rel"), list) else None
                msgs = surf.draw_polyline(d["pts"], float(d.get("width", 0.012)),
                                          tuple(d.get("rgba") or surfaces.CHALK), t=t, ts=ts)
            elif op == "erase_disc":
                msgs = surf.erase_disc(d["center"], float(d.get("r", 0.05)), t=t, residue=float(d.get("residue", 0.08)))
            elif op == "erase_rect":
                msgs = surf.erase_rect(d["uv0"], d["uv1"], t=t, residue=float(d.get("residue", 0.08)))
            elif op == "clear":
                msgs = surf.clear(t=t)
            elif op == "put_frame_test":
                if name != "tv":
                    return web.json_response({"ok": False, "error": "put_frame_test is a tv op"}, status=400)
                msgs = surf.put_frame(surfaces.test_pattern(float(d.get("phase", 0.0))), t=t)
            else:
                return web.json_response({"ok": False, "error": "bad op"}, status=400)
        except (KeyError, TypeError, ValueError, IndexError) as e:
            return web.json_response({"ok": False, "error": f"bad args: {type(e).__name__}"}, status=400)
        return nocache(web.json_response({"ok": True, "playback_t": round(now_t, 3), "t": t, "msgs": len(msgs),
                                          "bytes": sum(len(m) for m in msgs), "attached": s.surfaces.send is not None,
                                          "character": s.character}))

    async def debug_surface_get(req: web.Request):
        if not authed(req, api=False):
            return deny()
        s = hub.session
        if s is None or s.ended.is_set():
            return web.json_response({"ok": False, "error": "no session"}, status=404)
        name = req.query.get("surface", "board")
        if name not in ("board", "tv"):
            return web.json_response({"ok": False, "error": "bad surface"}, status=400)
        import io as _io
        from PIL import Image as _Image
        buf = _io.BytesIO()
        _Image.fromarray(s.surfaces.get(name).snapshot(), "RGBA").save(buf, "PNG")
        return nocache(web.Response(body=buf.getvalue(), content_type="image/png"))

    # drive the TV agent of the live session without a voice model:
    #   {"op":"show","concept":..,"mode":"new|update"} | {"op":"hear","who":"you","text":..} | {"op":"clear"} | {"op":"state"}
    async def debug_tv_post(req: web.Request):
        if not authed(req):
            return deny()
        s = hub.session
        if s is None or s.ended.is_set() or s.tv is None:
            return web.json_response({"ok": False, "error": "no session with a TV"}, status=404)
        d = await body_json(req)
        op = d.get("op", "state")
        if op == "show":
            r = s.tv.show(str(d.get("concept") or ""), mode=d.get("mode") or "new")
        elif op == "hear":
            s.tv.hear("avatar" if d.get("who") == "avatar" else "you", str(d.get("text") or ""), final=True)
            r = {"ok": True}
        elif op == "clear":
            r = s.tv.clear()
        else:
            r = {"ok": True}
        return nocache(web.json_response({**r, "state": s.tv.last_state, "stats": s.tv.stats()}))

    app.router.add_post("/debug/tv", debug_tv_post)
    app.router.add_post("/debug/surface", debug_surface_post)
    app.router.add_get("/debug/surface", debug_surface_get)
    recorder_hub.mount(app, authed=authed, home=HOME, cfg=hub.cfg, log=log)
    return app


# ---------------------------------------------------------------------------------------------------------- start-up
KEY_DOCS = {
    "OPENAI_API_KEY": "https://platform.openai.com/api-keys",
    "GEMINI_API_KEY": "https://aistudio.google.com/apikey",
    "ANTHROPIC_API_KEY": "https://console.anthropic.com/settings/keys",
    "OPENROUTER_API_KEY": "https://openrouter.ai/settings/keys",
    "META_API_KEY": "(Meta's model API; only the planner route \"meta\")",
}


def key_report(cfg: dict) -> tuple[list[str], list[str]]:
    """(lines to print, required keys that are missing) for the configured voice and the optional features."""
    def has(*names) -> bool:
        return any(os.environ.get(n) for n in names)

    voice = str(cfg.get("voice_default") or "direct")
    try:
        cli = claude_door.cli_available()
    except Exception:  # noqa: BLE001
        cli = False
    rows = [
        ("OPENAI_API_KEY", has("OPENAI_API_KEY"), True,
         "default voice (OpenAI Realtime, the page's direct WebRTC call) and the server-relayed voice"),
        ("GEMINI_API_KEY", has("GEMINI_API_KEY", "GOOGLE_API_KEY"), False,
         "optional voice ?voice=gemini (Gemini Live), last-choice planner route; GOOGLE_API_KEY also accepted"),
        ("ANTHROPIC_API_KEY", has("ANTHROPIC_API_KEY"), False,
         "planner route \"anthropic\"; board brain / TV / feelings when the `claude` CLI is not installed"),
        ("OPENROUTER_API_KEY", has("OPENROUTER_API_KEY"), False,
         "planner (Gemini Flash Lite) when neither the `claude` CLI nor ANTHROPIC_API_KEY is there; behavior coder and "
         "attention judge (else they use OPENAI_API_KEY)"),
        ("META_API_KEY", has("META_API_KEY"), False, "planner route \"meta\" only"),
    ]
    required = ["OPENAI_API_KEY"]                 # both voice paths ("direct" and "hub") run on OpenAI
    lines = ["API keys (from the environment / .env):"]
    for name, ok, _, what in rows:
        lines.append(f"  {'set    ' if ok else 'missing'}  {name:<19} {what}")
    lines.append(f"  {'found  ' if cli else 'missing'}  {'claude CLI':<19} board brain, TV agent, feelings, planner "
                 f"(default route when found){'' if cli else '; AVATAR_CLAUDE_BIN = its path'}")
    try:
        p = kfplanner.resolve(cfg.get("planner") if isinstance(cfg.get("planner"), dict) else True)
        f = kfplanner.resolve_fallback(cfg.get("planner") if isinstance(cfg.get("planner"), dict) else True)
        lines.append(f"  planner: {p['route']} / {p['model']}" + (f", fallback {f['route']} / {f['model']}" if f else ""))
    except Exception as e:  # noqa: BLE001
        lines.append(f"  planner: ({type(e).__name__})")
    missing = [n for n, ok, req, _ in rows if req and n in required and not ok]
    for n in missing:
        lines.append(f"WARNING: Missing {n}: the default voice (OpenAI Realtime, voice mode \"{voice}\") needs it. Get one "
                     f"at {KEY_DOCS[n]} and put it in .env (see .env.example). The page and the renderer still work; "
                     "starting a conversation will report the missing key.")
    if not cli and not has("ANTHROPIC_API_KEY"):
        lines.append("note: no `claude` CLI and no ANTHROPIC_API_KEY: the chalkboard brain, the wall TV and the "
                     "feelings classifier stay off.")
    return lines, missing


async def main() -> None:
    port = int(os.environ.get("PORT") or 10250)
    host = os.environ.get("HOST") or "127.0.0.1"
    hub = Hub()
    lines, missing = key_report(hub.cfg)
    for ln in lines:
        log(ln)
    if missing and os.environ.get("AVATAR_REQUIRE_KEYS", "").strip() not in ("", "0"):
        log(f"AVATAR_REQUIRE_KEYS is set and {', '.join(missing)} is missing: not starting.")
        sys.exit(2)
    hub.http = aiohttp.ClientSession()
    await hub.box.startup_check()
    app = make_app(hub)
    runner = web.AppRunner(app, access_log=None, handle_signals=False)
    await runner.setup()
    site = web.TCPSite(runner, host, port, reuse_address=True)
    await site.start()
    log(f"avatar-live listening on {host}:{port} (local renderer, delay {hub.delay_ms()} ms, "
        f"API cap ${hub.spend.cap:.2f}/day, spent today ${hub.spend.today():.4f})")
    log(f"open: http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/?k=<the secret in "
        f"{os.path.join(STATE, 'secret')}>")
    tasks = [asyncio.create_task(hub.box.watchdog())]
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    log("shutting down")
    if hub.session and not hub.session.ended.is_set():
        hub.session.end("service stopping")
        try:
            await asyncio.wait_for(hub.session.task, 6)
        except Exception:  # noqa: BLE001
            pass
    if hub.link:
        await hub.link.close()
    if hub._react_worker is not None:
        await hub._react_worker.close()
    if hub._unified_worker is not None:
        await hub._unified_worker.close()
    try:
        await hub.box.shutdown("service stopping")
    except Exception as e:  # noqa: BLE001
        log(f"render server stop at shutdown failed: {type(e).__name__}")
    for t in tasks:
        t.cancel()
    await hub.http.close()
    await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
