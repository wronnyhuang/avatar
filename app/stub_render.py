#!/usr/bin/env python3
"""Stub render server for the live avatar (test fixture, CPU only).

Speaks the real render server's wire format (AVATAR.md "Wire format", render-server side) and draws a cheap
cartoon face whose mouth opening follows the audio RMS, so the hub, the browser page and lip-sync measurements
can be exercised end to end without a GPU.

  python stub_render.py --port N [--host 127.0.0.1] [--key KEY] [--render-ms 0] [--init-ms 300]
                        [--encoder auto|pyav|ffmpeg] [--x264-threads 2]

Key: --key, else env RENDER_KEY, else no key check. Prints `READY port=<n>` on stdout once listening
(--port 0 picks a free port).

Endpoints: GET /health, WebSocket GET /render?key=<KEY> (wrong key -> 403). One session at a time; a new
connection replaces (closes) the old one.

Schedule (causal, like the real server): frame j (instant t_j = j/fps) is rendered once the received audio covers
t_j + lookahead_ms, never earlier; frames go out in order, each number once. A frame more than 0.5 s behind (later
than both its audio's arrival and its due time on the estimated timeline clock) is skipped together with the rest
of the backlog up to the frame due now, and counted in stats.late.
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import math
import os
import queue
import shutil
import struct
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import numpy as np
import cv2
from aiohttp import web, WSMsgType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surfaces as surf_mod  # noqa: E402  (scene: 'S' surface messages -> mirror canvases)

try:
    import av  # PyAV (bundled FFmpeg with libx264)
except Exception:  # pragma: no cover - fallback path
    av = None

cv2.setNumThreads(1)

SR = 24000
LATE_S = 0.5
MSG_A = 0x41
MSG_V = 0x56
MSG_S = 0x53                     # scene: surface update (app/surfaces.py)
FLAG_KEY = 0x01
FLAG_JPEG = 0x02
V_HDR = struct.Struct("<BBHId")  # 16 bytes: 'V', flags, 0, uint32 frame, float64 t

CHARACTERS = {  # name -> (background BGR, face BGR)
    "fox": ((20, 80, 200), (170, 215, 245)),
    "bear": ((30, 60, 90), (150, 185, 210)),
    "rabbit": ((150, 120, 200), (235, 225, 245)),
    "hippo": ((140, 110, 120), (215, 200, 210)),
    "monkey": ((40, 110, 140), (165, 205, 225)),
    "orb": ((120, 40, 20), (250, 215, 170)),
    "cube": ((60, 140, 40), (190, 240, 190)),
    "drop": ((160, 100, 0), (250, 220, 160)),
    "star": ((0, 170, 200), (180, 245, 255)),
    "pyramid": ((40, 120, 170), (175, 220, 240)),
    "blob": ((130, 30, 120), (230, 190, 235)),
    "muse": ((150, 170, 200), (205, 225, 235)),   # body: cream plush on a warm studio
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[stub_render]", *a, file=sys.stderr, flush=True)


def h264_level(width, height, fps):
    mbs = ((width + 15) // 16) * ((height + 15) // 16)
    if mbs <= 3600 and mbs * fps <= 108000:
        return "3.1", 0x1F
    if mbs <= 8192 and mbs * fps <= 245760:
        return "4.0", 0x28
    return "5.1", 0x33


def nal_types(b: bytes):
    out, i = [], 0
    while True:
        k = b.find(b"\x00\x00\x01", i)
        if k < 0 or k + 3 >= len(b):
            return out
        out.append(b[k + 3] & 0x1F)
        i = k + 3


# ---------------------------------------------------------------- audio timeline
class AudioTimeline:
    """int16 mono samples on the 24 kHz timeline; indices below `base` read as silence."""

    def __init__(self):
        self.buf = np.zeros(SR * 4, np.int16)
        self.base = 0
        self.n = 0
        self.overlaps = 0

    @property
    def end(self):
        return self.base + self.n

    def append(self, idx: int, pcm: np.ndarray):
        end = self.end
        if idx > end:
            gap = idx - end
            if gap > SR * 2:  # big gap: drop everything, the past reads as silence
                self.base, self.n = idx, 0
            else:
                pcm = np.concatenate([np.zeros(gap, np.int16), pcm])
        elif idx < end:
            ov = end - idx
            self.overlaps += 1
            if ov >= len(pcm):
                return
            pcm = pcm[ov:]
        need = self.n + len(pcm)
        if need > len(self.buf):
            nb = np.zeros(max(need, len(self.buf) * 2), np.int16)
            nb[: self.n] = self.buf[: self.n]
            self.buf = nb
        self.buf[self.n: need] = pcm
        self.n = need

    def trim(self, keep_from: int):
        drop = min(keep_from - self.base, self.n)
        if drop >= SR:  # amortised compaction
            self.buf[: self.n - drop] = self.buf[drop: self.n]
            self.base += drop
            self.n -= drop

    def rms(self, a: int, b: int) -> float:
        """RMS (0..1 of full scale) of timeline samples [a, b), zero padded outside the known range."""
        lo, hi = max(a, self.base), min(b, self.end)
        if hi <= lo or b <= a:
            return 0.0
        x = self.buf[lo - self.base: hi - self.base].astype(np.float32)
        return float(math.sqrt(float(np.dot(x, x)) / (b - a))) / 32768.0


def level_from_rms(rms: float) -> float:
    if rms <= 1e-6:
        return 0.0
    db = 20.0 * math.log10(rms)
    return min(1.0, max(0.0, (db + 50.0) / 40.0))  # -50 dBFS -> closed, -10 dBFS -> fully open


# ---------------------------------------------------------------- drawing
class Painter:
    def __init__(self, character, width, height):
        self.w, self.h = width, height
        bg, face = CHARACTERS[character]
        img = np.empty((height, width, 3), np.uint8)
        img[:] = bg
        s = height / 720.0
        self.cx, self.cy, self.r = width // 2, int(height * 0.55), int(height * 0.32)
        cx, cy, r = self.cx, self.cy, self.r
        cv2.circle(img, (cx, cy), r, face, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r, (30, 30, 30), max(1, int(4 * s)), cv2.LINE_AA)
        for ex in (cx - int(0.38 * r), cx + int(0.38 * r)):
            cv2.circle(img, (ex, cy - int(0.25 * r)), max(2, int(0.11 * r)), (25, 25, 25), -1, cv2.LINE_AA)
            cv2.circle(img, (ex + int(0.03 * r), cy - int(0.29 * r)), max(1, int(0.035 * r)), (255, 255, 255), -1,
                       cv2.LINE_AA)
        cv2.putText(img, character, (int(24 * s), int(70 * s)), cv2.FONT_HERSHEY_SIMPLEX, 2.0 * s, (255, 255, 255),
                    max(1, int(4 * s)), cv2.LINE_AA)
        cv2.putText(img, "STUB", (int(24 * s), int(110 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (255, 255, 255),
                    max(1, int(2 * s)), cv2.LINE_AA)
        self.base = img
        self.s = s
        # mouth geometry (also used by the self-test)
        self.mouth_c = (cx, cy + int(0.42 * r))
        self.mouth_ax = int(0.42 * r)
        self.mouth_max = int(0.32 * r)

    def surface_rects(self):
        """scene: (x, y, w, h) of the board (left) and the TV (right) in the frame."""
        w, h = self.w, self.h
        bw = max(8, int(0.30 * w)) & ~1
        tw = max(8, int(0.24 * w)) & ~1
        return (int(0.03 * w), int(0.18 * h), bw, max(4, bw // 2)), \
               (w - tw - int(0.07 * w), int(0.18 * h), tw, max(4, tw * 288 // 512))

    def draw(self, frame_no, t, level, overlays=None):
        img = self.base.copy()
        s, w, h = self.s, self.w, self.h
        for ov, (x, y, ow, oh) in zip(overlays or (), self.surface_rects()):
            if ov is not None and ov.shape[:2] == (oh, ow) and x + ow <= w and y + oh <= h:
                img[y:y + oh, x:x + ow] = ov
        oy = int(round(level * self.mouth_max))
        if oy >= 2:
            cv2.ellipse(img, self.mouth_c, (self.mouth_ax, oy), 0, 0, 360, (40, 20, 30), -1, cv2.LINE_AA)
        else:
            mx, my = self.mouth_c
            cv2.line(img, (mx - self.mouth_ax, my), (mx + self.mouth_ax, my), (40, 20, 30), max(1, int(3 * s)),
                     cv2.LINE_AA)
        # level bar at the right edge
        bw, top, bot = max(4, int(20 * s)), int(20 * s), h - int(20 * s)
        x0 = w - bw - int(10 * s)
        cv2.rectangle(img, (x0, top), (x0 + bw, bot), (20, 20, 20), -1)
        lv = int(round(level * (bot - top)))
        if lv > 0:
            cv2.rectangle(img, (x0, bot - lv), (x0 + bw, bot), (60, 220, 60), -1)
        cv2.putText(img, f"f={frame_no} t={t:.2f}", (int(24 * s), h - int(24 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
        return img


# ---------------------------------------------------------------- encoders
class JpegEncoder:
    name = "jpeg"

    def encode(self, frame_no, img, force_kf):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return [(frame_no, buf.tobytes(), True)]

    def close(self):
        pass


def pyav_h264_ok():
    if av is None:
        return False
    try:
        av.codec.Codec("libx264", "w")
        return True
    except Exception:
        return False


class PyAVH264:
    name = "pyav-libx264"

    def __init__(self, width, height, fps, bitrate, threads=2):
        level, _ = h264_level(width, height, fps)
        cc = av.CodecContext.create("libx264", "w")
        cc.width, cc.height, cc.pix_fmt = width, height, "yuv420p"
        cc.time_base = Fraction(1, fps)
        cc.framerate = Fraction(fps, 1)
        cc.bit_rate = int(bitrate)
        cc.gop_size = 2 * fps
        cc.max_b_frames = 0
        cc.thread_count = max(1, int(threads))
        cc.options = {
            "preset": "ultrafast", "tune": "zerolatency", "profile": "baseline", "level": level, "forced-idr": "1",
            "x264-params": f"repeat-headers=1:scenecut=0:keyint={2 * fps}:min-keyint={2 * fps}:bframes=0",
        }
        self.cc = cc
        self.pts = 0
        self.fmap = {}

    def encode(self, frame_no, img, force_kf):
        f = av.VideoFrame.from_ndarray(img, format="bgr24")
        f.pts = self.pts
        self.fmap[self.pts] = frame_no
        self.pts += 1
        if force_kf:
            f.pict_type = av.video.frame.PictureType.I
        out = []
        for p in self.cc.encode(f):
            fn = self.fmap.pop(p.pts, frame_no)
            out.append((fn, bytes(p), bool(p.is_keyframe)))
        return out

    def close(self):
        self.cc = None


class FfmpegH264:
    """/usr/bin/ffmpeg libx264 pipe; access units split on AUD NALs (adds one frame of delay); a forced keyframe
    restarts the encoder."""
    name = "ffmpeg-pipe-libx264"

    def __init__(self, width, height, fps, bitrate, threads=2):
        self.w, self.h, self.fps, self.bitrate, self.threads = width, height, fps, int(bitrate), threads
        self.proc = None
        self.pending = []  # frame numbers written, AU not yet out
        self.q = queue.Queue()
        self.fresh = True

    def _spawn(self):
        level, _ = h264_level(self.w, self.h, self.fps)
        exe = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{self.w}x{self.h}", "-r", str(self.fps), "-i", "pipe:0",
               "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-profile:v", "baseline",
               "-bf", "0", "-g", str(2 * self.fps), "-b:v", str(self.bitrate),
               "-threads", str(self.threads),
               "-x264-params", f"aud=1:repeat-headers=1:scenecut=0:keyint={2 * self.fps}:level={level}",
               "-pix_fmt", "yuv420p", "-f", "h264", "-flush_packets", "1", "pipe:1"]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        q = self.q = queue.Queue()
        threading.Thread(target=self._reader, args=(self.proc, q), daemon=True).start()
        self.pending = []

    @staticmethod
    def _reader(proc, q):
        buf = b""
        aud = b"\x00\x00\x00\x01\x09"
        try:
            while True:
                chunk = proc.stdout.read1(1 << 16) if hasattr(proc.stdout, "read1") else os.read(
                    proc.stdout.fileno(), 1 << 16)
                if not chunk:
                    break
                buf += chunk
                while True:
                    k = buf.find(aud, 1)
                    if k < 0:
                        break
                    q.put(buf[:k])
                    buf = buf[k:]
        finally:
            if buf:
                q.put(buf)
            q.put(None)

    def _stop_proc(self, drain):
        out = []
        if self.proc is None:
            return out
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        if drain:
            while True:
                try:
                    au = self.q.get(timeout=2.0)
                except queue.Empty:
                    break
                if au is None:
                    break
                if self.pending:
                    fn = self.pending.pop(0)
                    out.append((fn, au, 5 in nal_types(au)))
        try:
            self.proc.wait(timeout=2.0)
        except Exception:
            self.proc.kill()
            self.proc.wait()
        self.proc = None
        return out

    def encode(self, frame_no, img, force_kf):
        out = []
        if force_kf and not self.fresh:
            out += self._stop_proc(drain=True)
        if self.proc is None:
            self._spawn()
        self.fresh = False
        self.proc.stdin.write(np.ascontiguousarray(img).tobytes())
        self.pending.append(frame_no)
        # AU n-1 is complete once AU n starts; wait briefly for it
        while len(self.pending) > 1:
            try:
                au = self.q.get(timeout=1.0)
            except queue.Empty:
                break
            if au is None:
                raise RuntimeError("ffmpeg encoder exited")
            fn = self.pending.pop(0)
            out.append((fn, au, 5 in nal_types(au)))
        return out

    def close(self):
        self._stop_proc(drain=False)


# ---------------------------------------------------------------- session
class Session:
    """One WebSocket connection (= one render session)."""

    def __init__(self, server, ws):
        self.srv, self.ws = server, ws
        self.loop = asyncio.get_running_loop()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stub-render")
        self.send_lock = asyncio.Lock()
        self.wake = asyncio.Event()
        self.gen = 0
        self.cfg = None
        self.character = None
        self.tl = None
        self.rendering = False
        self.ctl_task = None
        self.encoder = None
        self.painter = None
        self.closed = False
        self._reset_counters()
        self.render_task = asyncio.create_task(self._render_loop())
        self.stats_task = asyncio.create_task(self._stats_loop())

    def _reset_counters(self):
        self.next_j = 0
        self.force_kf = True
        self.since_kf = 0
        self.late = 0
        self.cover = []  # (audio end sample, arrival wall) per chunk, for frame deadlines
        self.cover_i = 0
        self.origin = None  # estimated wall time of timeline sample 0 on the hub clock
        self.win = []  # (wall, render_ms, encode_ms) per sent frame
        self.audio_ignored = 0
        # scene: surface mirrors + per-surface pending updates [(t or None, unpacked)]
        self.surf = {0: np.zeros((512, 1024, 4), np.uint8), 1: np.zeros((288, 512, 4), np.uint8)}
        self.surf_q = {0: [], 1: []}
        self.surf_ov = [None, None]
        self.surf_dirty = True
        self.surf_stats = {"msgs": 0, "applied": 0, "bytes": 0}

    # ---- sending
    async def send_json(self, obj):
        if self.closed or self.ws.closed:
            return
        async with self.send_lock:
            await self.ws.send_str(json.dumps(obj))

    async def send_bytes(self, b):
        if self.closed or self.ws.closed:
            return
        async with self.send_lock:
            await self.ws.send_bytes(b)

    async def error(self, msg):
        log("error:", msg)
        try:
            await self.send_json({"type": "error", "message": msg})
        except Exception:
            pass

    # ---- control messages
    async def on_text(self, data):
        try:
            m = json.loads(data)
        except Exception:
            return await self.error("bad JSON")
        typ = m.get("type") if isinstance(m, dict) else None
        if typ == "ping":
            await self.send_json({"type": "pong", "t": m.get("t")})
        elif typ == "start":
            await self._start(m)
        elif typ == "character":
            await self._character(m)
        elif typ == "stop":
            self._stop()
            log("stop")
        elif typ == "keyframe":
            self.force_kf = True
        else:
            await self.error(f"unknown message type {typ!r}")

    async def _start(self, m):
        try:
            ch = str(m.get("character", "fox"))
            w, h, fps = int(m.get("width") or 640), int(m.get("height") or 360), int(m.get("fps", 24))
            codec = str(m.get("codec", "h264"))
            cfg = {
                "character": ch, "width": w, "height": h, "fps": fps, "codec": codec,
                "bitrate": int(m.get("bitrate", 2000000)), "lookahead_ms": float(m.get("lookahead_ms", 133)),
                "backend": m.get("backend"), "opts": m.get("opts") or {},
            }
        except Exception as e:
            return await self.error(f"bad start: {e}")
        if ch not in CHARACTERS:
            return await self.error(f"unknown character {ch!r}")
        if codec not in ("h264", "jpeg"):
            return await self.error(f"unsupported codec {codec!r}")
        if not (16 <= w <= 3840 and 16 <= h <= 2160 and w % 2 == 0 and h % 2 == 0 and 1 <= fps <= 60):
            return await self.error(f"bad size/fps {w}x{h}@{fps}")
        if cfg["lookahead_ms"] < 0:
            return await self.error("lookahead_ms must be >= 0")
        self._stop()
        self.cfg = cfg
        self.tl = AudioTimeline()  # timeline restarts at sample 0; audio during init is buffered
        self.ctl_task = asyncio.create_task(self._init(ch, first=True, gen=self.gen))
        log(f"start {ch} {w}x{h}@{fps} {codec} lookahead={cfg['lookahead_ms']}ms")

    async def _character(self, m):
        ch = str(m.get("character", ""))
        if ch not in CHARACTERS:
            return await self.error(f"unknown character {ch!r}")
        if self.cfg is None:
            return await self.error("character before start")
        self.rendering = False  # pause after the frame in flight
        prev = self.ctl_task
        self.ctl_task = asyncio.create_task(self._init(ch, first=False, gen=self.gen, prev=prev))
        log(f"character -> {ch}")

    async def _init(self, ch, first, gen, prev=None):
        try:
            if prev is not None and not prev.done():
                try:
                    await prev
                except BaseException:
                    pass
            if gen != self.gen:
                return
            t0 = time.monotonic()
            init_s = self.srv.args.init_ms / 1000.0
            await self.send_json({"type": "status", "stage": "init", "detail": f"stub: loading {ch}"})
            await asyncio.sleep(init_s / 2)
            await self.send_json({"type": "status", "stage": "compile", "detail": "stub: compiling (simulated)"})
            await asyncio.sleep(init_s / 2)
            if gen != self.gen:
                return
            cfg = self.cfg
            await self.loop.run_in_executor(self.pool, self._make_render_state, ch, cfg, first)
            if gen != self.gen:
                return
            self.character = ch
            init_ms = int(round((time.monotonic() - t0) * 1000))
            h264 = cfg["codec"] == "h264"
            lvl = h264_level(cfg["width"], cfg["height"], cfg["fps"])[1]
            await self.send_json({
                "type": "ready", "character": ch, "width": cfg["width"], "height": cfg["height"], "fps": cfg["fps"],
                "codec": cfg["codec"], "codec_string": f"avc1.42E0{lvl:02X}" if h264 else "jpeg",
                "backend": "stub", "gpu": "stub-cpu", "init_ms": init_ms,
                "encoder": self.encoder.name,
            })
            self.force_kf = True
            self.rendering = True
            self.wake.set()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            traceback.print_exc()
            await self.error(f"init failed: {e}")

    def _make_render_state(self, ch, cfg, first):  # runs in the session thread
        self.painter = Painter(ch, cfg["width"], cfg["height"])
        if first or self.encoder is None:
            if self.encoder is not None:
                self.encoder.close()
            if cfg["codec"] == "jpeg":
                self.encoder = JpegEncoder()
            else:
                mode = self.srv.encoder_mode
                cls = PyAVH264 if mode == "pyav" else FfmpegH264
                self.encoder = cls(cfg["width"], cfg["height"], cfg["fps"], cfg["bitrate"], self.srv.args.x264_threads)

    def _stop(self):
        self.gen += 1
        self.rendering = False
        if self.ctl_task is not None and not self.ctl_task.done():
            self.ctl_task.cancel()
        self.ctl_task = None
        self.tl = None
        self.cfg = None
        enc, self.encoder = self.encoder, None
        if enc is not None:
            self.pool.submit(enc.close)
        self._reset_counters()

    # ---- surfaces (scene)
    def on_binary(self, data: bytes):
        if data and data[0] == MSG_S:
            return self.on_surface(data)
        return self.on_audio(data)

    def on_surface(self, data: bytes):
        d = surf_mod.unpack_msg(data)
        if d["sid"] not in self.surf:
            raise ValueError("unknown surface id")
        self.surf_stats["msgs"] += 1
        self.surf_stats["bytes"] += len(data)
        self.surf_q[d["sid"]].append((None if d["flags"] & surf_mod.F_NOW else d["t"], d))

    def _surf_apply(self, t):
        """Apply every pending update due by frame time t (NOW updates at the next frame); refresh the overlays."""
        for sid, q in self.surf_q.items():
            k = 0
            while k < len(q) and (q[k][0] is None or q[k][0] <= t + 1e-6):
                surf_mod.apply_msg(self.surf[sid], q[k][1])
                k += 1
            if k:
                del q[:k]
                self.surf_stats["applied"] += k
                self.surf_dirty = True
        if self.surf_dirty and self.painter is not None:
            self.surf_dirty = False
            (_, _, bw, bh), (_, _, tw, th) = self.painter.surface_rects()
            b = cv2.resize(self.surf[0], (bw, bh), interpolation=cv2.INTER_AREA).astype(np.float32)
            a = b[:, :, 3:4] / 255.0
            green = np.array([40, 70, 30], np.float32)            # BGR dark slate green
            self.surf_ov[0] = (green * (1 - a) + b[:, :, 2::-1] * a).astype(np.uint8)
            tv = cv2.resize(self.surf[1], (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
            a = tv[:, :, 3:4] / 255.0
            self.surf_ov[1] = (np.float32(12) * (1 - a) + tv[:, :, 2::-1] * a).astype(np.uint8)

    # ---- audio
    def on_audio(self, data: bytes):
        if len(data) < 12 or data[0] != MSG_A:
            raise ValueError("bad audio message")
        if (len(data) - 12) % 2:
            raise ValueError("odd PCM byte count")
        if self.tl is None:
            self.audio_ignored += 1
            if self.audio_ignored == 1:
                log("audio with no session started: ignored")
            return
        idx = struct.unpack_from("<Q", data, 4)[0]
        pcm = np.frombuffer(data, dtype="<i2", offset=12)
        now = time.monotonic()
        self.tl.append(idx, pcm)
        end = self.tl.end
        self.cover.append((end, now))
        o = now - end / SR
        if self.origin is None or o > self.origin:
            self.origin = o
        self.wake.set()

    # ---- schedule
    def _need(self, j):
        fps = self.cfg["fps"]
        return int(math.ceil((j / fps + self.cfg["lookahead_ms"] / 1000.0) * SR - 1e-6))

    def _last_ready(self):
        fps, la = self.cfg["fps"], self.cfg["lookahead_ms"] / 1000.0
        j = int(math.floor((self.tl.end / SR - la) * fps + 1e-9))
        while j >= 0 and self._need(j) > self.tl.end:
            j -= 1
        while self._need(j + 1) <= self.tl.end:
            j += 1
        return j

    def _cover_wall(self, j, i0):
        need = self._need(j)
        i = i0
        while i < len(self.cover) and self.cover[i][0] < need:
            i += 1
        return (self.cover[i][1] if i < len(self.cover) else time.monotonic()), i

    def _deadline(self, j, i0):
        cw, i = self._cover_wall(j, i0)
        due = self.origin + j / self.cfg["fps"] + self.cfg["lookahead_ms"] / 1000.0
        return max(cw, due) + LATE_S, i

    async def _render_loop(self):
        while not self.closed:
            await self.wake.wait()
            self.wake.clear()
            while self.rendering and not self.closed and self.tl is not None:
                gen = self.gen
                last = self._last_ready()
                j = self.next_j
                if j > last or self.origin is None:
                    break
                now = time.monotonic()
                dl, self.cover_i = self._deadline(j, self.cover_i)
                if now > dl:  # fallen behind by > 0.5 s: skip the backlog up to the frame due now
                    fps, la = self.cfg["fps"], self.cfg["lookahead_ms"] / 1000.0
                    j_now = int(math.floor((now - self.origin - la) * fps))
                    target = min(last, max(j + 1, j_now))
                    self.late += target - j
                    j = target
                # drop bookkeeping that is no longer needed
                if self.cover_i > 256:
                    del self.cover[: self.cover_i]
                    self.cover_i = 0
                fps = self.cfg["fps"]
                self._surf_apply(j / fps)
                c = int(round(j / fps * SR))
                level = level_from_rms(self.tl.rms(c - SR // 50, c + SR // 50))  # 40 ms window centred on t_j
                self.tl.trim(c - SR)
                # periodic keyframes: forced here for PyAV; the ffmpeg pipe's own -g does them (its output lags a
                # frame, so forcing would restart the encoder needlessly)
                periodic = self.since_kf >= 2 * fps and not isinstance(self.encoder, FfmpegH264)
                force = self.force_kf or periodic
                self.force_kf = False
                try:
                    outs, r_ms, e_ms = await self.loop.run_in_executor(self.pool, self._produce, j, level, force,
                                                                       tuple(self.surf_ov))
                except Exception as e:
                    traceback.print_exc()
                    if gen == self.gen:
                        await self.error(f"render/encode failed at frame {j}: {e}")
                        self.force_kf = True
                        self.next_j = j + 1
                    continue
                if gen != self.gen:
                    continue
                for fn, payload, kf in outs:
                    flags = (FLAG_KEY if kf else 0) | (FLAG_JPEG if self.cfg["codec"] == "jpeg" else 0)
                    self.since_kf = 0 if kf else self.since_kf + 1
                    await self.send_bytes(V_HDR.pack(MSG_V, flags, 0, fn, fn / fps) + payload)
                self.win.append((time.monotonic(), r_ms, e_ms))
                self.next_j = j + 1

    def _produce(self, j, level, force, overlays=None):  # session thread
        t0 = time.perf_counter()
        img = self.painter.draw(j, j / self.cfg["fps"], level, overlays)
        if self.srv.args.render_ms > 0:
            time.sleep(self.srv.args.render_ms / 1000.0)
        t1 = time.perf_counter()
        outs = self.encoder.encode(j, img, force)
        t2 = time.perf_counter()
        return outs, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0

    async def _stats_loop(self):
        while not self.closed:
            await asyncio.sleep(1.0)
            if self.cfg is None:
                continue
            now = time.monotonic()
            self.win = [x for x in self.win if now - x[0] <= 1.0]
            r = sorted(x[1] for x in self.win)
            e = [x[2] for x in self.win]
            queue_n = 0
            if self.tl is not None and self.rendering:
                queue_n = max(0, self._last_ready() - self.next_j + 1)
            try:
                await self.send_json({
                    "type": "stats", "fps": len(self.win),
                    "render_ms": round(sum(r) / len(r), 2) if r else 0.0,
                    "render_p95": round(r[min(len(r) - 1, int(0.95 * len(r)))], 2) if r else 0.0,
                    "encode_ms": round(sum(e) / len(e), 2) if e else 0.0,
                    "late": self.late, "queue": queue_n, "surfaces": dict(self.surf_stats),
                })
            except Exception:
                pass

    async def close(self, reason=""):
        if self.closed:
            return
        self._stop()
        self.closed = True
        self.wake.set()
        for t in (self.render_task, self.stats_task):
            t.cancel()
        self.pool.shutdown(wait=False)
        if not self.ws.closed:
            try:
                await self.ws.close(message=reason.encode()[:120])
            except Exception:
                pass
        log("session closed", reason)


# ---------------------------------------------------------------- server
class StubServer:
    def __init__(self, args):
        self.args = args
        self.key = args.key if args.key is not None else os.environ.get("RENDER_KEY")
        self.current = None
        if args.encoder == "auto":
            self.encoder_mode = "pyav" if pyav_h264_ok() else "ffmpeg"
        else:
            self.encoder_mode = args.encoder
        if self.encoder_mode == "pyav" and not pyav_h264_ok():
            raise SystemExit("PyAV with libx264 not available")

    async def health(self, request):
        busy = self.current is not None and not self.current.closed
        return web.json_response({"ok": True, "gpu": "stub-cpu", "busy": busy})

    async def render(self, request):
        if self.key:
            got = request.query.get("key", "")
            if not hmac.compare_digest(got.encode(), self.key.encode()):
                raise web.HTTPForbidden(text="bad key")
        ws = web.WebSocketResponse(heartbeat=20.0, max_msg_size=16 << 20)
        await ws.prepare(request)
        old = self.current
        if old is not None:
            await old.close("replaced by a new session")
        sess = Session(self, ws)
        self.current = sess
        log("session opened from", request.remote)
        try:
            async for msg in ws:
                try:
                    if msg.type == WSMsgType.TEXT:
                        await sess.on_text(msg.data)
                    elif msg.type == WSMsgType.BINARY:
                        sess.on_binary(msg.data)                   # scene: 'S' surfaces + 'A' audio
                    elif msg.type == WSMsgType.ERROR:
                        log("ws error", ws.exception())
                        break
                except Exception as e:
                    traceback.print_exc()
                    await sess.error(f"{type(e).__name__}: {e}")
        except Exception:
            traceback.print_exc()
        finally:
            await sess.close("connection ended")
            if self.current is sess:
                self.current = None
        return ws


async def amain(args):
    srv = StubServer(args)
    app = web.Application()
    app.router.add_get("/health", srv.health)
    app.router.add_get("/render", srv.render)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    log(f"listening on {args.host}:{port} encoder={srv.encoder_mode} key={'on' if srv.key else 'off'} "
        f"render_ms={args.render_ms} init_ms={args.init_ms}")
    print(f"READY port={port}", flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        if srv.current is not None:
            await srv.current.close("server shutdown")
        await runner.cleanup()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--key", default=None)
    ap.add_argument("--render-ms", type=float, default=0.0)
    ap.add_argument("--init-ms", type=float, default=300.0)
    ap.add_argument("--encoder", choices=["auto", "pyav", "ffmpeg"], default="auto")
    ap.add_argument("--x264-threads", type=int, default=2)
    args = ap.parse_args()
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
