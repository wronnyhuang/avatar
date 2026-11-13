#!/usr/bin/env python3
"""The avatar render server (AVATAR.md): timeline audio in over a WebSocket, encoded video frames of the talking
character out, each tagged with the timeline instant it depicts.

    RENDER_KEY=... python -m avatar.render_server --host 127.0.0.1 --port 8770            # serve (key from the env)
    python -m avatar.render_server --prewarm 1280x720,960x540 --characters all [--lock gpu.lock]   # bake backgrounds

ENDPOINTS  ws://host:port/render?key=<RENDER_KEY> (one session at a time; a new connection replaces the old one) and
GET /health -> {"ok":true,"gpu":"...","busy":bool,...}.  Wire format: AVATAR.md ('A' audio in, 'V' video out, JSON
start/character/stop/ping/keyframe in, status/ready/stats/pong/error out); deviations in avatar/PROTOCOL-NOTES.txt.

PIPELINE  'A' (contiguous int16 PCM on the 24 kHz timeline, may run ahead of real time) -> animator thread:
live.core.LiveCore (anim.lipstream.StreamingLipAnimator, lookahead from start, causal 30 -> fps "hold" resampler) ->
pending jobs (global frame F = timeline sample / (24000/fps)) -> dispatcher thread -> render worker process
(avatar/render_worker.py: backend.render on the GPU, then the encoder on its own thread) -> 'V' out.

TIMELINE CLOCK  The app server emits timeline audio on its 20 ms tick and may send queued audio early, never late, so for every
'A' message arrival - end_sample / 24000 >= T0 + network delay, with equality for messages sent on the tick.  T0 (the
local wall time of timeline sample 0) is the max of that over the last --clock-window seconds.  A job's DUE time is
T0 + (the last sample it depends on) / 24000: the moment real-time audio would have released it.

SCHEDULING  Jobs are rendered in order as soon as they exist (audio ahead of the clock = frames rendered ahead).  When
the oldest due job is more than late_ms past due, the renderer has fallen behind: every due job but the newest is
dropped (counted) and the newest is rendered -- no backlog, ever.  Frames never wait for audio beyond the lookahead,
and silence on the timeline keeps the idle life (blinks, breath, head) rendering, since the app server never stops the audio.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hmac
import json
import multiprocessing as mp
import os
import queue
import signal
import struct
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit

# A frame is a small realtime job. Large BLAS/OpenMP pools contend with the shared
# machine and can turn a millisecond operation into a hundred-millisecond stall.
for _thread_env in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = os.environ.get("AVATAR_NUMERIC_THREADS", "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from anim.lipstream import StreamingLipAnimator      # noqa: E402
from anim.schema import FPS as ROW_FPS, SAMPLE_RATE as SR, IDX  # noqa: E402
from anim.rig import RigMixer, RIG_PARAMS            # noqa: E402
from live.core import LiveCore                       # noqa: E402

mono = time.monotonic
HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES = os.path.join(HERE, "profiles.json")
CHARACTERS = ["fox", "bear", "rabbit", "hippo", "monkey", "orb", "cube", "drop", "star", "pyramid", "blob", "muse"]   # + muse
A_HDR = struct.Struct("<B3xQ")          # 'A', pad, uint64 first sample
V_HDR = struct.Struct("<BBHId")         # 'V', flags, pad, uint32 frame number, float64 t
SPFS = {20: 1200, 24: 1000, 30: 800}    # timeline samples per output frame


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


class Core(LiveCore):
    """live.core.LiveCore at any output rate in SPFS (its resampler arithmetic is general; its constructor allows
    only 24 / 30)."""

    def __init__(self, character, lookahead_ms, fps):
        super().__init__(character, lookahead_ms=lookahead_ms, fps=30, resample="hold",
                         animator=StreamingLipAnimator(character, lookahead_ms=lookahead_ms))
        self.fps = int(fps)


# ------------------------------------------------------------------------------------------------ the timeline clock
class Clock:
    def __init__(self, window_s=10.0):
        self.window = window_s
        self.obs = collections.deque()       # (arrival, arrival - end/SR), values kept decreasing (sliding max)
        self.t0 = None
        self.newest = 0                       # newest sample received

    def observe(self, arrival, end_sample):
        v = arrival - end_sample / SR
        while self.obs and self.obs[-1][1] <= v:
            self.obs.pop()
        self.obs.append((arrival, v))
        while self.obs and self.obs[0][0] < arrival - self.window and len(self.obs) > 1:
            self.obs.popleft()
        self.t0 = self.obs[0][1]
        self.newest = max(self.newest, end_sample)

    def now_t(self, now=None):
        return None if self.t0 is None else (now if now is not None else mono()) - self.t0

    def ahead_ms(self, now=None):
        n = self.now_t(now)
        return None if n is None else 1000.0 * (self.newest / SR - n)


# ------------------------------------------------------------------------------------------------ engines (workers)
class Engine:
    """One render worker process and its reader thread; messages go to whichever session owns it."""

    def __init__(self, key, character, w, h, backend, opts, rt_tree):
        self.key = key
        self.character, self.w, self.h = character, w, h
        self.owner = None
        self.ready = False
        self.dead = False
        self.info = {}
        self.last_used = mono()
        self.t_spawn = mono()
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        from avatar import render_worker
        self.proc = ctx.Process(target=render_worker.main, args=(child, character, w, h, backend, opts, rt_tree),
                                daemon=True, name=f"render-{character}")
        self.proc.start()
        child.close()
        self.send_lock = threading.Lock()
        self.status_log = []
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def send(self, msg):
        with self.send_lock:
            try:
                self.conn.send(msg)
                return True
            except (OSError, BrokenPipeError, EOFError):
                self.dead = True
                return False

    def _read(self):
        while True:
            try:
                msg = self.conn.recv()
            except (EOFError, OSError):
                self.dead = True
                o = self.owner
                if o is not None:
                    o.on_engine_msg(self, ("dead", "render worker exited"))
                return
            if msg[0] == "ready":
                self.ready, self.info = True, msg[1]
            elif msg[0] == "status":
                self.status_log.append(msg)
            o = self.owner
            if o is not None:
                o.on_engine_msg(self, msg)

    def close(self):
        self.dead = True
        self.send(("quit",))
        self.proc.join(timeout=5)
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join(timeout=2)


class EngineCache:
    """Warm engines keyed by (character, size, backend, opts); at most max_engines alive (1 on the K620: the old
    engine is closed BEFORE the new one starts, so two never share the 2 GB)."""

    def __init__(self, max_engines=1, idle_release_s=600.0, rt_tree=None):
        self.max = max(1, int(max_engines))
        self.idle = float(idle_release_s)
        self.rt_tree = rt_tree
        self.engines: dict = {}
        self.lock = threading.Lock()

    @staticmethod
    def key_of(character, w, h, backend, opts):
        return json.dumps([character, w, h, backend, opts or {}], sort_keys=True)

    def get(self, character, w, h, backend, opts):
        key = self.key_of(character, w, h, backend, opts)
        with self.lock:
            e = self.engines.get(key)
            if e is not None and not e.dead:
                e.last_used = mono()
                return e
            if e is not None:
                del self.engines[key]
            while len(self.engines) >= self.max:
                free = [x for x in self.engines.values() if x.owner is None] or list(self.engines.values())
                old = min(free, key=lambda x: x.last_used)
                del self.engines[old.key]
                log(f"engine close {old.character} {old.w}x{old.h} (cache full)")
                old.owner = None
                old.close()
            e = Engine(key, character, w, h, backend, opts, self.rt_tree)
            self.engines[key] = e
            log(f"engine spawn {character} {w}x{h} {backend} {opts}")
            return e

    def sweep(self):
        with self.lock:
            for k, e in list(self.engines.items()):
                if e.dead or (e.owner is None and mono() - e.last_used > self.idle):
                    del self.engines[k]
                    log(f"engine release {e.character} {e.w}x{e.h} ({'dead' if e.dead else 'idle'})")
                    e.close()

    def close_all(self):
        with self.lock:
            for e in self.engines.values():
                e.close()
            self.engines.clear()


# ------------------------------------------------------------------------------------------------ profiles
def load_profiles():
    try:
        return json.load(open(PROFILES))
    except (OSError, ValueError):
        return {}


def profile_for(profile, character, height=None):
    """-> dict(width, height, backend, opts) from avatar/profiles.json, or {} when the profile does not say."""
    prof = load_profiles().get("profiles", {}).get(profile, {})
    ent = dict(prof.get("_default", {}))
    ent.update(prof.get(character, {}))
    if height and ent.get("alts"):
        alts = ent["alts"]
        if str(int(height)) in alts:
            ent = dict(ent, **alts[str(int(height))])
    return {k: ent[k] for k in ("width", "height", "backend", "opts") if k in ent}


# ------------------------------------------------------------------------------------------------ one session
class Job:
    __slots__ = ("F", "t", "index", "params", "horizon")

    def __init__(self, F, t, index, params, horizon):
        self.F, self.t, self.index, self.params, self.horizon = F, t, index, params, horizon


class Session:
    MAX_ENCODING = 2        # rendered frames waiting for / in the encoder before the dispatcher holds back

    def __init__(self, server, ws, loop):
        self.server, self.ws, self.loop = server, ws, loop
        self.out: asyncio.Queue = asyncio.Queue()
        self.clock = Clock(server.args.clock_window)
        self.cfg = None
        self.engine = None
        self.engine_ok = False          # engine ready AND encoder opened for this session
        self.enc_info = {}
        self.cond = threading.Condition()
        self.pending: collections.deque = collections.deque()
        self.inflight = 0               # jobs sent, not yet rendered (0 or 1)
        self.encoding = 0               # rendered, not yet encoded
        self.enc_opened = None          # (engine, gen) the encoder was opened for
        self.key_req = False
        self.running = True
        self.audio_q: queue.Queue = queue.Queue()
        self.gen = 0                    # bumps on every (re)start / character switch: stale results are ignored
        self.t_switch = None
        self.stats = self._new_stats()
        self.totals = collections.Counter()
        self.sent_frames = {}
        self.rig = RigMixer()           # page rig vectors + the old pose dicts, smoothed (anim/rig.py)
        from avatar.surfaces_rx import SurfaceMirror      # the app server's live surfaces ('S' messages)
        self.surf = SurfaceMirror()
        self.rig_lock = threading.Lock()
        self.anim_th = threading.Thread(target=self._anim_loop, daemon=True)
        self.disp_th = threading.Thread(target=self._dispatch_loop, daemon=True)
        self.anim_th.start()
        self.disp_th.start()

    # ---------------------------------------------------------------- messages out (thread-safe)
    def post(self, item):
        try:
            self.loop.call_soon_threadsafe(self.out.put_nowait, item)
        except RuntimeError:
            pass

    def post_json(self, obj):
        self.post(json.dumps(obj))

    # ---------------------------------------------------------------- control
    def start(self, msg):
        a = self.server.args
        ch = msg.get("character") or "orb"
        if ch not in CHARACTERS and ":" not in str(msg.get("backend", "")):
            raise ValueError(f"unknown character {ch!r}")
        fps = int(msg.get("fps") or a.fps)
        if fps not in SPFS:
            raise ValueError(f"fps must be one of {sorted(SPFS)}")
        self.cfg = {"character": ch, "fps": fps, "codec": msg.get("codec") or a.codec,
                    "bitrate": int(msg.get("bitrate") or a.bitrate), "lookahead_ms": float(msg.get("lookahead_ms") or 133),
                    "width": msg.get("width"), "height": msg.get("height"), "backend": msg.get("backend") or "auto",
                    "opts": dict(msg.get("opts") or {}), "late_ms": float(msg.get("late_ms") or a.late_ms),
                    "gop_s": float(msg.get("gop_s") or 2.0), "nvenc": bool(msg.get("nvenc", not a.no_nvenc)),
                    "quality": int(msg.get("jpeg_quality") or 80)}
        self._switch(ch, first=True)

    def _resolve(self, ch):
        c = self.cfg
        a = self.server.args
        backend, opts = c["backend"], dict(c["opts"])
        w, h = c["width"], c["height"]
        if backend in ("auto", "profile") or not w or not h:
            p = profile_for(a.profile, ch, h)
            if backend in ("auto", "profile"):
                backend = p.get("backend", a.backend)
                opts = dict(p.get("opts", {}), **opts)
            if not w or not h:
                w, h = p.get("width", a.width), p.get("height", a.height)
        if backend == "auto":
            backend = a.backend
        w, h = int(w) // 2 * 2, int(h) // 2 * 2
        return ch, int(w), int(h), backend, opts

    def _switch(self, ch, first=False):
        ch, w, h, backend, opts = self._resolve(ch)
        self.cfg.update(character=ch)
        self.rig.set_character(ch)       # per-character legacy-column rules (anim/rig.py point_moves_hand)
        self.t_switch = mono()
        with self.cond:
            self.gen += 1
            self.pending.clear()
            self.engine_ok = False
            old = self.engine
            if old is not None:
                old.owner = None
            self.inflight = 0
            self.encoding = 0
        self.audio_q.put(("reset", ch, self.gen))
        self.post_json({"type": "status", "stage": "init", "detail": f"{ch} {w}x{h} {backend}"})
        eng = self.server.engines.get(ch, w, h, backend, opts)
        self.res = (ch, w, h, backend, opts)
        self.surf.mark_all_dirty()       # a new / switched engine gets the whole canvases again
        with self.cond:
            self.engine = eng
            eng.owner = self
        if eng.ready:
            self._open_encoder(eng)
        elif eng.dead:
            self.post_json({"type": "error", "message": "render worker died before it was ready"})

    def _open_encoder(self, eng):
        with self.cond:                 # once per (engine, switch): the reader thread may race _switch here
            if eng is not self.engine or self.enc_opened == (eng, self.gen):
                return
            self.enc_opened = (eng, self.gen)
        c = self.cfg
        eng.send(("encoder", {"codec": c["codec"], "fps": c["fps"], "bitrate": c["bitrate"],
                                      "gop_s": c["gop_s"], "nvenc": c["nvenc"], "quality": c["quality"],
                                      "threads": self.server.args.enc_threads}))

    def on_engine_msg(self, eng, msg):
        if eng is not self.engine:
            return
        k = msg[0]
        if k == "status":
            _, stage, detail, eta = msg
            self.post_json({"type": "status", "stage": stage, "detail": detail, "eta_s": eta})
        elif k == "ready":
            self.post_json({"type": "status", "stage": "encoder", "detail": "opening the encoder"})
            self._open_encoder(eng)
        elif k == "encoder":
            info = msg[1]
            self.enc_info = info
            if info.get("fallback"):        # mid-stream encoder change: a new "ready" (new codec_string), IDR next
                self.post_json({"type": "status", "stage": "encoder", "detail": "encoder fallback: " + info["fallback"]})
            ch, w, h, backend, opts = self.res
            ei = eng.info
            self.post_json({"type": "ready", "character": ch, "width": w, "height": h, "fps": self.cfg["fps"],
                            "codec": info["codec"], "codec_string": info["codec_string"], "encoder": info["encoder"],
                            "backend": ei.get("backend", backend), "opts": ei.get("opts", opts),
                            "preset": ei.get("preset"), "gpu": ei.get("gpu"),
                            "init_ms": round(1000 * (mono() - self.t_switch), 1), "engine_init_ms": ei.get("init_ms"),
                            "lookahead_ms": self.cfg["lookahead_ms"], "encoder_errors": info.get("errors") or None,
                            "rig_params": RIG_PARAMS,
                            "encoder_open_ms": info.get("open_ms"), "encoder_fallback": info.get("fallback")})
            with self.cond:
                self.engine_ok = True
                self.key_req = True
                self.cond.notify_all()
        elif k == "rendered":
            with self.cond:
                self.inflight = max(0, self.inflight - 1)
                self.encoding += 1
                self.cond.notify_all()
        elif k == "frame":
            self._on_frame(msg)
        elif k in ("error", "dead"):
            self.post_json({"type": "error", "message": str(msg[1])[-800:]})
            if k == "dead":
                with self.cond:
                    self.engine_ok = False

    # ---------------------------------------------------------------- audio -> animator -> jobs
    def feed_audio(self, data, arrival):
        s0, = struct.unpack_from("<Q", data, 4)
        n = (len(data) - 12) // 2
        pcm = np.frombuffer(data, dtype="<i2", count=n, offset=12)
        self.clock.observe(arrival, s0 + pcm.size)
        self.audio_q.put((s0, pcm))

    def set_pose(self, msg):
        """The old reactive-pose dict ({"type":"pose","pose":{...}}): overrides the named columns for 1.5 s, the
        server's life stays on.  Audio-derived mouth columns only while the voice is quiet (anim/rig.py)."""
        raw = msg.get("pose") or {}
        if isinstance(raw, dict):
            with self.rig_lock:
                self.rig.push_pose(raw, mono())

    def set_rig(self, msg):
        """{"type":"rig","t":<playback-timeline s>,"a":[...]} in RIG_PARAMS order (the "ready" message's rig_params;
        an optional "names" list overrides the order).  While these arrive the server's life is off."""
        a = msg.get("a")
        if not isinstance(a, list):
            raise ValueError("rig: 'a' must be a list")
        names = msg.get("names")
        with self.rig_lock:
            self.rig.push_rig(msg.get("t"), a, mono(), names=names if isinstance(names, list) else None)
        self.totals["rig_msgs"] += 1

    def _anim_loop(self):
        core = None
        a0 = 0
        nxt = None
        gen = 0
        ch = None
        ring = collections.deque()        # recent (s0, pcm) for warming a new animator on a switch
        ring_n = 0
        RING = int(1.5 * SR)
        while self.running:
            item = self.audio_q.get()
            if item is None:
                return
            if item[0] == "reset":
                _, ch, gen = item
                core = None
                if nxt is not None and ring:        # a switch mid-stream: warm the new animator on the last 1.5 s
                    first = ring[0][0]
                    spf = SPFS[self.cfg["fps"]]
                    a0 = -(-first // spf) * spf
                    core = Core(ch, self.cfg["lookahead_ms"], self.cfg["fps"])
                    for s, p in list(ring):
                        if s + p.size <= a0:
                            continue
                        p = p[max(0, a0 - s):]
                        self._emit(core.push(p), a0, gen)
                continue
            s0, pcm = item
            if core is None:
                spf = SPFS[self.cfg["fps"]] if self.cfg else 1000
                if ch is None:
                    continue                          # audio before start: ignored
                a0 = (s0 // spf) * spf
                core = Core(ch, self.cfg["lookahead_ms"], self.cfg["fps"])
                nxt = a0
            if s0 > nxt:
                gap = s0 - nxt
                self.totals["gap_samples"] += gap
                if gap > 10 * SR:                     # a hole this long is a new timeline: restart cleanly
                    spf = SPFS[self.cfg["fps"]]
                    a0 = (s0 // spf) * spf
                    core = Core(ch, self.cfg["lookahead_ms"], self.cfg["fps"])
                    nxt = a0
                    gap = s0 - nxt
                if gap:
                    self._emit(core.push(np.zeros(gap, np.int16)), a0, gen)
                    nxt = s0
            elif s0 < nxt:
                cut = nxt - s0
                self.totals["overlap_samples"] += min(cut, pcm.size)
                if cut >= pcm.size:
                    continue
                pcm, s0 = pcm[cut:], nxt
            self._emit(core.push(pcm), a0, gen)
            nxt = s0 + pcm.size
            ring.append((s0, pcm))
            ring_n += pcm.size
            while ring and ring_n - ring[0][1].size >= RING:
                ring_n -= ring.popleft()[1].size

    def _emit(self, jobs, a0, gen):
        if not jobs:
            return
        fps = self.cfg["fps"]
        base = a0 // SPFS[fps]
        new = [Job(base + jb.j, (base + jb.j) / fps, jb.index, jb.params, a0 + jb.horizon) for jb in jobs]
        with self.cond:
            if gen != self.gen:
                return
            self.pending.extend(new)
            cap = int(self.server.args.max_pending_s * fps)
            while len(self.pending) > cap:
                self.pending.popleft()
                self.stats["dropped"] += 1
            self.cond.notify_all()

    # ---------------------------------------------------------------- dispatch
    def _dispatch_loop(self):
        while self.running:
            with self.cond:
                while self.running and not (self.engine_ok and self.pending and self.inflight == 0
                                            and self.encoding < self.MAX_ENCODING):
                    self.cond.wait(0.1)
                if not self.running:
                    return
                now = mono()
                t0 = self.clock.t0 if self.clock.t0 is not None else now
                late_s = self.cfg["late_ms"] / 1000.0
                first = self.pending[0]
                if now - (t0 + first.horizon / SR) > late_s:        # behind: skip to the newest due job
                    due = [jb for jb in self.pending if t0 + jb.horizon / SR <= now]
                    jb = due[-1]
                    nd = len(due) - 1
                    for _ in range(nd + 1):
                        self.pending.popleft()
                    self.stats["dropped"] += nd
                    self.totals["dropped"] += nd
                else:
                    jb = self.pending.popleft()
                self.inflight += 1
                force = self.key_req
                self.key_req = False
                eng, gen = self.engine, self.gen
                with self.rig_lock:
                    params = self.rig.apply(jb.params, jb.t, now)
                self.sent_frames[jb.F] = (now, t0 + jb.horizon / SR, gen)
            # surface updates due at this frame's timeline instant go up before the job (bounded bytes)
            self.surf.advance(jb.t)
            if eng.info.get("surfaces"):
                for up in self.surf.take():
                    eng.send(("surf",) + up)
            if not eng.send(("job", jb.F, jb.index, params, force, now)):
                with self.cond:
                    self.inflight = max(0, self.inflight - 1)

    def _on_frame(self, msg):
        _, F, flags, payload, rms, ems, t_r0, t_done = msg
        now = mono()
        with self.cond:
            self.encoding = max(0, self.encoding - 1)
            self.cond.notify_all()
            meta = self.sent_frames.pop(F, None)
            for k in [k for k in self.sent_frames if k < F - 240]:
                del self.sent_frames[k]
        if meta is None or meta[2] != self.gen or not payload:
            return
        t_disp, due, _ = meta
        fps = self.cfg["fps"]
        t = F / fps
        t0 = self.clock.t0 if self.clock.t0 is not None else now
        lag = 1000.0 * (now - (t0 + t))
        late = now - due > self.cfg["late_ms"] / 1000.0
        s = self.stats
        s["frames"] += 1
        s["bytes"] += len(payload)
        s["render"].append(rms)
        s["encode"].append(ems)
        s["lag"].append(lag)
        s["svc"].append(1000.0 * (now - t_disp))
        s["late"] += int(late)
        s["keys"] += flags & 1
        self.totals["frames"] += 1
        self.totals["late"] += int(late)
        self.post(V_HDR.pack(0x56, flags, 0, F & 0xFFFFFFFF, t) + payload)

    @staticmethod
    def _new_stats():
        return {"frames": 0, "bytes": 0, "render": [], "encode": [], "lag": [], "svc": [], "late": 0, "dropped": 0,
                "keys": 0, "t": mono()}

    def stats_json(self):
        s, self.stats = self.stats, self._new_stats()
        dt = max(1e-3, mono() - s["t"])
        p = lambda x, q: round(float(np.percentile(x, q)), 1) if x else None
        m = lambda x: round(float(np.mean(x)), 1) if x else None
        with self.cond:
            qn, inf = len(self.pending), self.inflight + self.encoding
        ahead = self.clock.ahead_ms()
        return {"type": "stats", "fps": round(s["frames"] / dt, 2), "render_ms": m(s["render"]),
                "render_p95": p(s["render"], 95), "encode_ms": m(s["encode"]), "encode_p95": p(s["encode"], 95),
                "late": s["late"] + s["dropped"], "late_sent": s["late"], "dropped": s["dropped"], "queue": qn,
                "inflight": inf, "lag_ms": m(s["lag"]), "lag_p95": p(s["lag"], 95), "service_ms": m(s["svc"]),
                "audio_ahead_ms": None if ahead is None else round(ahead, 1), "kbps": round(8 * s["bytes"] / dt / 1000, 1),
                "keyframes": s["keys"], "encoder": self.enc_info.get("encoder"), "out_queue": self.out.qsize(),
                "rig_active": self.rig.rig_active(mono()), "rig_msgs": int(self.totals["rig_msgs"]),
                "surfaces": dict(self.surf.stats)}

    def request_key(self):
        with self.cond:
            self.key_req = True

    def close(self):
        self.running = False
        self.audio_q.put(None)
        with self.cond:
            self.gen += 1
            self.cond.notify_all()
            eng = self.engine
            self.engine = None
        if eng is not None:
            eng.owner = None
            eng.last_used = mono()


# ------------------------------------------------------------------------------------------------ the server
class Server:
    def __init__(self, args):
        self.args = args
        self.key = os.environ.get(args.key_env, "")
        self.engines = EngineCache(args.max_engines, args.idle_release_s, args.rt_tree)
        self.session: Session | None = None
        self.gpu = None

    def health(self):
        if self.gpu is None:
            from avatar.render_worker import gpu_name
            self.gpu = gpu_name()
        s = self.session
        return {"ok": True, "gpu": self.gpu, "busy": s is not None,
                "engines": [{"character": e.character, "size": f"{e.w}x{e.h}", "ready": e.ready,
                             "backend": e.info.get("backend")} for e in list(self.engines.engines.values())],
                "profile": self.args.profile, "default_backend": self.args.backend, "pid": os.getpid()}

    def process_request(self, connection, request):
        u = urlsplit(request.path)
        if u.path == "/health":
            r = connection.respond(200, json.dumps(self.health()) + "\n")
            # websockets.Headers assignment appends; remove respond()'s text/plain
            # header so strict JSON clients see one unambiguous media type.
            del r.headers["Content-Type"]
            r.headers["Content-Type"] = "application/json"
            return r
        if u.path != "/render":
            return connection.respond(404, "not found\n")
        given = (parse_qs(u.query).get("key") or [""])[0]
        if self.key and not hmac.compare_digest(given.encode(), self.key.encode()):
            return connection.respond(403, "forbidden\n")
        return None

    async def handler(self, ws):
        loop = asyncio.get_running_loop()
        old = self.session
        if old is not None:
            log("new session replaces the old one")
            try:
                old.post_json({"type": "error", "message": "replaced by a new session"})
                old.close()
                await old.ws.close(4000, "replaced")
            except Exception:
                pass
        sess = Session(self, ws, loop)
        self.session = sess
        log(f"session open from {ws.remote_address}")

        async def sender():
            while True:
                item = await sess.out.get()
                if item is None:
                    return
                await ws.send(item)

        async def stats():
            while True:
                await asyncio.sleep(1.0)
                if sess.cfg is not None:
                    sess.post_json(sess.stats_json())
                await loop.run_in_executor(None, self.engines.sweep)

        tasks = [asyncio.create_task(sender()), asyncio.create_task(stats())]
        try:
            async for m in ws:
                now = mono()
                if isinstance(m, (bytes, bytearray)):
                    if len(m) >= 12 and m[0] == 0x41:
                        if sess.cfg is not None:
                            sess.feed_audio(m, now)
                    elif len(m) >= 24 and m[0] == 0x53:         # a live-surface update
                        sess.surf.feed(m)
                    continue
                try:
                    msg = json.loads(m)
                except ValueError:
                    sess.post_json({"type": "error", "message": "bad json"})
                    continue
                typ = msg.get("type")
                try:
                    if typ == "start" or (typ == "character" and sess.cfg is None):
                        await loop.run_in_executor(None, sess.start, msg)      # may close/spawn a worker: off-loop
                    elif typ == "character":
                        ch = msg.get("character")
                        if ch not in CHARACTERS:
                            raise ValueError(f"unknown character {ch!r}")
                        await loop.run_in_executor(None, sess._switch, ch)
                    elif typ == "ping":
                        sess.post_json({"type": "pong", "t": msg.get("t")})
                    elif typ == "keyframe":
                        sess.request_key()
                    elif typ == "pose":
                        sess.set_pose(msg)
                    elif typ == "rig":
                        sess.set_rig(msg)
                    elif typ == "stop":
                        break
                    else:
                        sess.post_json({"type": "error", "message": f"unknown message type {typ!r}"})
                except Exception as e:
                    sess.post_json({"type": "error", "message": f"{typ}: {e}"})
        except Exception as e:
            log(f"session error: {type(e).__name__}: {e}")
        finally:
            log(f"session end: {dict(sess.totals)}")
            sess.close()
            sess.out.put_nowait(None)
            for t in tasks:
                t.cancel()
            if self.session is sess:
                self.session = None
            try:
                await ws.close()
            except Exception:
                pass

    async def run(self):
        from websockets.asyncio.server import serve
        stop = asyncio.get_running_loop().create_future()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, lambda: stop.done() or stop.set_result(None))
        pw = self.args.prewarm_engines or ""
        for ch in (CHARACTERS if pw == "all" else [c for c in pw.split(",") if c]):
            p = profile_for(self.args.profile, ch)
            self.engines.get(ch, int(p.get("width", self.args.width)), int(p.get("height", self.args.height)),
                             p.get("backend", self.args.backend), p.get("opts", {}))
        async with serve(self.handler, self.args.host, self.args.port, process_request=self.process_request,
                         compression=None, max_size=2 ** 22, ping_interval=20, ping_timeout=60,
                         write_limit=2 ** 22):
            log(f"render server on ws://{self.args.host}:{self.args.port}/render (profile {self.args.profile}, "
                f"backend {self.args.backend}, key {'set' if self.key else 'NOT SET'})")
            await stop
        if self.session is not None:
            self.session.close()
        self.engines.close_all()


# ------------------------------------------------------------------------------------------------ prewarm (bakes)
def prewarm(args):
    """Bake every character's background at the given sizes into $TA_OUT (render.py's cache), one subprocess per
    bake so each GL context is released; with --lock each bake holds that flock (a shared GPU lock)."""
    import fcntl
    from avatar.render_worker import bg_cached
    chars = CHARACTERS if args.characters == "all" else args.characters.split(",")
    sizes = [tuple(int(v) for v in s.split("x")) for s in args.prewarm.split(",")]
    out = []
    for w, h in sizes:
        for ch in chars:
            if bg_cached(ch, w, h) and not getattr(args, "prewarm_compile", False):
                out.append({"character": ch, "size": f"{w}x{h}", "baked": False, "cached": True})
                continue
            # build the REAL-TIME engine at the profile's preset: it bakes the background AND compiles the rt
            # programs into the NVIDIA on-disk shader cache, so a session's first start of this character is warm
            preset = (profile_for(args.profile, ch, h).get("opts") or {}).get("preset", "auto")
            code = ("import sys,time; sys.path[:0]=[%r,%r]; import rt_engine; t=time.time(); "
                    "e=rt_engine.RTEngine(%r,%d,%d,%r if %r!='auto' else __import__('live.backends.rt',fromlist=['x']).default_preset(%r,%d,%d,24),'gpu'); "
                    "print(round(time.time()-t,1), e.init_times)" %
                    (ROOT, os.path.join(ROOT, "engine"), ch, w, h, preset, preset, ch, w, h))
            lockf = open(args.lock, "a") if args.lock else None
            try:
                if lockf:
                    fcntl.flock(lockf, fcntl.LOCK_EX)
                t = time.time()
                r = subprocess.run(["nice", "-n", "10", sys.executable, "-c", code], capture_output=True, text=True)
                ok = r.returncode == 0 and bg_cached(ch, w, h)
            finally:
                if lockf:
                    fcntl.flock(lockf, fcntl.LOCK_UN)
                    lockf.close()
            rec = {"character": ch, "size": f"{w}x{h}", "baked": ok, "s": round(time.time() - t, 1),
                   "engine": (r.stdout or "").strip()[-200:]}
            if not ok:
                rec["error"] = (r.stderr or r.stdout)[-400:]
            out.append(rec)
            log(json.dumps(rec))
    print(json.dumps(out, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8770")))
    ap.add_argument("--key-env", default="RENDER_KEY", help="environment variable holding the shared secret")
    ap.add_argument("--insecure-no-key", action="store_true", help="allow serving without a key (local tests only)")
    ap.add_argument("--profile", default=os.environ.get("AVATAR_PROFILE", "local"), help="profiles.json section")
    ap.add_argument("--backend", default=os.environ.get("AVATAR_BACKEND", "auto"),
                    help="auto (rt if it imports, else v2 spp=1) | rt | v2 | package.module:Class")
    ap.add_argument("--rt-tree", default=os.environ.get("AVATAR_RT_TREE", ""),
                    help="a checkout with live/backends/rt.py, used for 'rt' while this tree has none")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--codec", default="h264", choices=["h264", "jpeg"])
    ap.add_argument("--bitrate", type=int, default=900_000)
    ap.add_argument("--no-nvenc", action="store_true", default=os.environ.get("AVATAR_NVENC", "1") == "0",
                    help="libx264 only (AVATAR_NVENC=0; NVENC falls back by itself when slow)")
    ap.add_argument("--enc-threads", type=int, default=4)
    ap.add_argument("--late-ms", type=float, default=100.0, help="a job this far past due means the renderer is behind")
    ap.add_argument("--max-pending-s", type=float, default=20.0)
    ap.add_argument("--clock-window", type=float, default=10.0)
    ap.add_argument("--max-engines", type=int, default=int(os.environ.get("AVATAR_MAX_ENGINES", "1")),
                    help="warm engines kept (1 on a small card such as a 2 GB K620; 11 on a 24 GB card makes every switch instant)")
    ap.add_argument("--idle-release-s", type=float, default=600.0)
    ap.add_argument("--prewarm-engines", default=os.environ.get("AVATAR_PREWARM_ENGINES", ""),
                    help="'all' or fox,bear,...: engines started warm at boot at the profile's size (big GPU; set "
                         "--max-engines to at least that many)")
    ap.add_argument("--prewarm", default="", help="WxH[,WxH...]: bake backgrounds and exit")
    ap.add_argument("--characters", default="all")
    ap.add_argument("--prewarm-compile", action="store_true",
                    help="with --prewarm: build every engine even when its background is cached (fills the shader cache "
                         "after a shader change, so the first start of each character is warm)")
    ap.add_argument("--lock", default="", help="flock file held around each bake (a shared GPU lock)")
    a = ap.parse_args(argv)
    if a.rt_tree:
        os.environ["AVATAR_RT_TREE"] = a.rt_tree
    if a.prewarm:
        return prewarm(a)
    if not os.environ.get(a.key_env) and not a.insecure_no_key:
        sys.exit(f"refusing to serve: ${a.key_env} is empty (or pass --insecure-no-key for a local test)")
    asyncio.run(Server(a).run())


if __name__ == "__main__":
    main()
