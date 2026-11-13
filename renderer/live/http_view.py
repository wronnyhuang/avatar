"""Live view: ships the presenter's frames (JPEG) and audio (PCM) to a browser page that plays them timestamp-synced.

    from live.http_view import LiveView
    lv = LiveView(1280, 720, 30, sample_rate=24000, port=8765, info={"character": "fox", "delay_ms": 250})
    lv.push_audio(tau, pcm_s16le)       # at the instant sample tau is presented (T0 + tau + D)
    lv.push_frame(t_j, rgb, index=j)    # at the instant frame j is presented (T0 + t_j + D)
    print(lv.url) ... lv.close()

Endpoints (matched on the last path segment, so the page also works behind a reverse proxy's path prefix):
  /  the page    /stream  framed binary messages    /mjpeg  multipart JPEG, video only    /frame.jpg    /status  JSON
/stream is a chunked HTTP body of messages: 16-byte little-endian header (u8 type, 3 reserved bytes, u32 payload length,
f64 stream time t) + payload; type 1 = JPEG frame, 2 = PCM s16le mono, 3 = JSON hello (always the first message).
The page plays stream time t at AudioContext time base + t, base = clock at the first message + playout buffer - t.

Push calls never touch a socket or wake a thread: they append to each client's bounded queue under a private lock, and
one handler thread per client polls that queue every POLL_S and writes it out.  (Waking the handler from the push, as a
Condition.notify would, lets the kernel preempt the presenter thread on a busy host: measured here at load ~240, push
p99 0.1-1.8 ms with notify vs 0.04-0.08 ms polling.)  A slow client loses its oldest queued frames; one that falls more
than MAX_AUDIO_S of audio behind is dropped.
"""
from __future__ import annotations

import itertools
import json
import socket
import struct
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

HDR = struct.Struct("<B3xId")          # type, 3 reserved, payload length, stream time (s)
FRAME, AUDIO, HELLO = 1, 2, 3
MAX_FRAMES = 8                         # /stream client: a 9th queued frame drops the oldest queued frame
MJPEG_MAX_FRAMES = 2
MAX_AUDIO_S = 2.0                      # more queued audio than this and the client is dropped
SNDBUF = 256 * 1024                    # small kernel send buffer: a slow client's backlog stays in our queue, visible
GRACE_S = 0.5                          # close(): time clients get to end their streams cleanly
POLL_S = 0.004                         # handler poll period: adds 0-4 ms of delivery latency, keeps pushes syscall-free
MJPEG_PART = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"


class _Client:
    _ids = itertools.count(1)

    def __init__(self, kind: str, sock, peer: str, max_frames: int, max_audio: int):
        self.id, self.kind, self.sock, self.peer = next(_Client._ids), kind, sock, peer
        self.max_frames, self.max_audio = max_frames, max_audio
        self.lock = threading.Lock()
        self.q: deque = deque()                     # items: (type, parts tuple, payload bytes)
        self.qframes = self.qaudio = 0
        self.dead, self.reason = False, ""
        self.done = threading.Event()              # set once the handler has unregistered the client
        self.t0 = time.monotonic()
        self.sent_frames = self.sent_audio = self.sent_bytes = self.dropped_frames = 0

    def put(self, item) -> None:                    # presenter thread: a few list ops under a private lock, no I/O
        over = False
        with self.lock:
            if self.dead:
                return
            self.q.append(item)
            if item[0] == FRAME:
                self.qframes += 1
                if self.qframes > self.max_frames:
                    for i, it in enumerate(self.q):
                        if it[0] == FRAME:
                            del self.q[i]
                            break
                    self.qframes -= 1
                    self.dropped_frames += 1
            elif item[0] == AUDIO:
                self.qaudio += item[2]
                over = self.qaudio > self.max_audio
        if over:
            self.kill("behind", hard=True)

    def kill(self, reason: str, hard: bool = False) -> None:
        with self.lock:
            if not self.dead:
                self.dead, self.reason = True, reason
        if hard:                                    # wakes a handler blocked in sendall; shutdown itself never blocks
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def pump(self, write, chunked: bool) -> None:   # handler thread: drain the queue in batches, one write per batch
        while not self.dead:
            if not self.q:
                time.sleep(POLL_S)
                continue
            with self.lock:
                items, self.q = self.q, deque()
                self.qframes = self.qaudio = 0
            parts = [p for it in items for p in it[1]]
            n = sum(map(len, parts))
            write(b"".join([b"%X\r\n" % n, *parts, b"\r\n"] if chunked else parts))
            self.sent_bytes += n
            for it in items:
                self.sent_frames += it[0] == FRAME
                self.sent_audio += it[0] == AUDIO
        if chunked and self.reason == "closed":
            write(b"0\r\n\r\n")

    def stats(self) -> dict:
        return {"id": self.id, "kind": self.kind, "peer": self.peer, "alive": not self.dead, "reason": self.reason,
                "age_s": round(time.monotonic() - self.t0, 2), "sent_frames": self.sent_frames,
                "sent_audio": self.sent_audio, "sent_bytes": self.sent_bytes, "dropped_frames": self.dropped_frames,
                "queued": len(self.q)}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "LiveView/1"
    timeout = 10                                    # socket timeout: a client stuck for 10 s fails its send

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        lv: LiveView = self.server.lv
        path = urlsplit(self.path).path
        name = "" if path.endswith("/") else path.rsplit("/", 1)[-1]
        if name in ("", "index.html"):
            self._send(200, "text/html; charset=utf-8", lv._page_bytes)
        elif name in ("stream", "mjpeg"):
            self._stream(lv, name)
        elif name == "frame.jpg":
            jpg = lv._latest_jpeg()
            self._send(200, "image/jpeg", jpg) if jpg else self._send(404, "text/plain", b"no frame yet\n")
        elif name == "status":
            self._send(200, "application/json", json.dumps(lv.stats(), default=str).encode())
        else:
            self._send(404, "text/plain", b"not found\n")

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.close_connection = True
        self.send_response(code)
        for k, v in (("Content-Type", ctype), ("Content-Length", str(len(body))), ("Cache-Control", "no-store"),
                     ("Connection", "close")):
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _stream(self, lv: "LiveView", kind: str) -> None:
        if lv._closed:
            return self._send(503, "text/plain", b"closed\n")
        s = self.connection
        try:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SNDBUF)
        except OSError:
            pass
        self.close_connection = True
        self.send_response(200)
        ctype = "application/octet-stream" if kind == "stream" else "multipart/x-mixed-replace; boundary=frame"
        for k, v in (("Content-Type", ctype), ("Cache-Control", "no-store"), ("X-Accel-Buffering", "no"),
                     ("Connection", "close")):
            self.send_header(k, v)
        if kind == "stream":
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        c = lv._add(kind, s, "%s:%s" % self.client_address[:2])
        try:
            c.pump(self.wfile.write, kind == "stream")
        except OSError:
            c.kill("disconnected")                  # keeps an earlier reason ("behind", "closed")
        finally:
            lv._remove(c)


class LiveView:
    def __init__(self, width: int, height: int, fps: float, sample_rate: int = 24000, host: str = "127.0.0.1",
                 port: int = 8765, jpeg_quality: int = 80, info: dict | None = None):
        self.width, self.height, self.fps, self.sample_rate = int(width), int(height), float(fps), int(sample_rate)
        self.jpeg_quality = int(jpeg_quality)
        self._info = dict(info or {})
        self._lock = threading.Lock()
        self._clients: tuple = ()                   # copy-on-write: pushes read it without a lock
        self._gone: deque = deque(maxlen=32)
        self._latest = None                         # (t, index, rgb | None, jpeg | None); rgb is encoded on demand
        self._closed = False
        self._n_frames = self._n_audio = self._n_enc = 0
        self._enc_s = 0.0
        self._audio_cap = int(MAX_AUDIO_S * self.sample_rate * 2)
        self._page_bytes = PAGE.replace("__W__", str(self.width)).replace("__H__", str(self.height)).encode()
        self._srv = ThreadingHTTPServer((host, int(port)), _Handler)
        self._srv.daemon_threads = True
        self._srv.lv = self
        self.host, self.port = host, self._srv.server_address[1]
        self.url = f"http://{host}:{self.port}/"
        self._thread = threading.Thread(target=self._srv.serve_forever, kwargs={"poll_interval": 0.2},
                                        name=f"liveview:{self.port}", daemon=True)
        self._thread.start()

    # ---- presenter-side API -------------------------------------------------------------------------------------
    def set_info(self, **kv) -> None:
        self._info = {**self._info, **kv}

    def push_frame(self, t: float, rgb=None, jpeg: bytes | None = None, index: int = -1) -> None:
        self._n_frames += 1
        cl = self._clients
        if not cl:                                  # nobody watching: keep a reference only, no encode
            self._latest = (t, index, rgb, jpeg)
            return
        if jpeg is None:
            if rgb is None:
                return
            jpeg = self._encode(rgb)
        elif not isinstance(jpeg, bytes):
            jpeg = bytes(jpeg)
        self._latest = (t, index, None, jpeg)
        n, s_item, m_item = len(jpeg), None, None
        for c in cl:
            if c.kind == "stream":
                s_item = s_item or (FRAME, (HDR.pack(FRAME, n, t), jpeg), n)
                c.put(s_item)
            else:
                m_item = m_item or (FRAME, (MJPEG_PART % n, jpeg, b"\r\n"), n)
                c.put(m_item)

    def push_audio(self, t: float, pcm: bytes) -> None:
        self._n_audio += 1
        cl = self._clients
        if not cl:
            return
        if not isinstance(pcm, bytes):
            pcm = pcm.tobytes() if hasattr(pcm, "tobytes") else bytes(pcm)
        n = len(pcm)
        item = (AUDIO, (HDR.pack(AUDIO, n, t), pcm), n)
        for c in cl:
            if c.kind == "stream":
                c.put(item)

    def clients(self) -> int:
        return sum(not c.dead for c in self._clients)

    def stats(self) -> dict:
        cl, lt = self._clients, self._latest
        return {"url": self.url, "clients": sum(not c.dead for c in cl), "width": self.width, "height": self.height,
                "fps": self.fps, "sample_rate": self.sample_rate, "info": dict(self._info),
                "frames_pushed": self._n_frames, "audio_pushed": self._n_audio, "encodes": self._n_enc,
                "encode_ms_mean": round(1000 * self._enc_s / max(1, self._n_enc), 3),
                "latest": {"t": lt[0], "index": lt[1]} if lt else None,
                "per_client": [c.stats() for c in cl], "gone": list(self._gone)}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            cl = self._clients
        for c in cl:
            c.kill("closed")                        # graceful: the handler writes the final chunk and returns
        deadline = time.monotonic() + GRACE_S
        for c in cl:
            if not c.done.wait(max(0.0, deadline - time.monotonic())):
                c.kill("closed", hard=True)
        self._srv.shutdown()
        self._srv.server_close()
        self._thread.join(2.0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- internals ------------------------------------------------------------------------------------------------
    def _encode(self, rgb) -> bytes:
        import cv2
        import numpy as np
        t0 = time.perf_counter()
        a = np.ascontiguousarray(rgb, dtype=np.uint8)
        if a.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_RGBA2BGR if a.shape[2] == 4 else cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise ValueError("JPEG encode failed")
        self._n_enc += 1
        self._enc_s += time.perf_counter() - t0
        return buf.tobytes()

    def _latest_jpeg(self) -> bytes | None:
        lt = self._latest
        if lt is None:
            return None
        if lt[3] is None and lt[2] is not None:
            jpg = self._encode(lt[2])
            if self._latest is lt:
                self._latest = (lt[0], lt[1], None, jpg)
            return jpg
        return lt[3]

    def _add(self, kind: str, sock, peer: str) -> _Client:
        stream = kind == "stream"
        c = _Client(kind, sock, peer, MAX_FRAMES if stream else MJPEG_MAX_FRAMES, self._audio_cap)
        if stream:
            hello = json.dumps({**self._info, "width": self.width, "height": self.height, "fps": self.fps,
                                "sample_rate": self.sample_rate}, default=str).encode()
            c.q.append((HELLO, (HDR.pack(HELLO, len(hello), 0.0), hello), len(hello)))
        with self._lock:
            if self._closed:
                c.kill("closed")
            self._clients += (c,)
        return c

    def _remove(self, c: _Client) -> None:
        with self._lock:
            self._clients = tuple(x for x in self._clients if x is not c)
            self._gone.append(c.stats())
        c.done.set()


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>live</title>
<style>
html,body{margin:0;height:100%;background:#0b0c0e;color:#8e98a3;font:12px/1.4 system-ui,sans-serif;overflow:hidden}
#wrap{position:absolute;left:0;right:0;top:0;bottom:30px;display:flex;align-items:center;justify-content:center}
canvas{background:#000;display:block}
#go{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);font:600 22px system-ui,sans-serif;
  padding:18px 34px;border:0;border-radius:12px;background:#2f6fed;color:#fff;cursor:pointer;box-shadow:0 6px 30px #000a}
#msg{position:absolute;top:10px;left:0;right:0;text-align:center;color:#c8ced6;font-size:14px}
#bar{position:absolute;left:0;right:0;bottom:0;height:30px;padding:0 10px;display:flex;align-items:center;gap:16px;
  white-space:nowrap;overflow:hidden;font-variant-numeric:tabular-nums}
#info{color:#c8ced6}
</style></head><body>
<div id="wrap"><canvas id="cv" width="__W__" height="__H__"></canvas></div>
<button id="go">Start (sound on)</button>
<div id="msg">waiting for stream…</div>
<div id="bar"><span id="info"></span><span id="st"></span></div>
<script>
'use strict';
// <parser>  pure: feed byte chunks split anywhere; calls onMsg(type, t, payload) for every complete message
function makeParser(onMsg) {
  let buf = new Uint8Array(1 << 16), off = 0, end = 0;
  return function push(chunk) {
    if (end + chunk.length > buf.length) {
      const live = end - off;
      if (live + chunk.length <= buf.length) buf.copyWithin(0, off, end);
      else { const nb = new Uint8Array(Math.max(2 * buf.length, live + chunk.length)); nb.set(buf.subarray(off, end)); buf = nb; }
      off = 0; end = live;
    }
    buf.set(chunk, end); end += chunk.length;
    const dv = new DataView(buf.buffer);
    while (end - off >= 16) {
      const n = dv.getUint32(off + 4, true);
      if (n > (1 << 26)) throw new Error('bad message length ' + n);
      if (end - off < 16 + n) break;
      onMsg(dv.getUint8(off), dv.getFloat64(off + 8, true), buf.slice(off + 16, off + 16 + n));
      off += 16 + n;
    }
    if (off === end) off = end = 0;
  };
}
// </parser>
const $ = id => document.getElementById(id);
const cv = $('cv'), g = cv.getContext('2d'), go = $('go'), msgEl = $('msg'), infoEl = $('info'), stEl = $('st');
const Q = new URLSearchParams(location.search);
let B = (Q.has('buffer') ? +Q.get('buffer') : 80) / 1000;   // playout buffer (s): stream time t plays at base + t
const ADAPT = Q.get('adapt') !== '0';                         // each late audio chunk grows B by 20 ms (max 500 ms)
const VLAT = (Q.has('vlat') ? +Q.get('vlat') : 16) / 1000;    // a canvas drawn now is on screen about one refresh later
const TRIM = 0.05;          // audio up to 50 ms late starts at once, trimmed; later than that re-anchors
let ctx = null, sr = 24000, base = null, gen = 0, lastArr = 0, lastTransit = null;
let frames = [], drawnT = null, sources = [], audioEnd = 0;
let frTimes = [], aheadMin = Infinity, ahead = NaN, winStart = 0, lastBytes = 0, lastStat = performance.now();
const S = {lateF: 0, lateA: 0, resync: 0, jit: 0, bytes: 0, badF: 0};

const audioOn = () => !!ctx && ctx.state === 'running';
const now = () => audioOn() ? ctx.currentTime : performance.now() / 1000;      // the playback clock
const outLat = () => audioOn() ? (ctx.outputLatency || ctx.baseLatency || 0) : 0;
const showT = () => now() - outLat() + VLAT;     // clock time whose sound is at the speakers when this draw is seen

function stopAudio() { for (const s of sources) { try { s.stop(); } catch (e) {} } sources = []; audioEnd = 0; }
function anchor(t) { base = now() + B - t; stopAudio(); }
function resync() { S.resync++; base = null; }

function onMsg(type, t, p) {
  S.bytes += p.length + 16;
  if (type === 3) return hello(JSON.parse(new TextDecoder().decode(p)));
  const a = performance.now();
  if (base !== null && a - lastArr > 1000) resync();          // stream stalled > 1 s
  lastArr = a;
  if (base === null) anchor(t);
  const tr = a / 1000 - t;                                     // transit time; its variation = jitter (RFC 3550 style)
  if (lastTransit !== null) S.jit += (Math.abs(tr - lastTransit) - S.jit) / 16;
  lastTransit = tr;
  let slack = base + t - now();
  if (slack < -0.25 || slack > B + 2) { resync(); anchor(t); slack = B; }   // far from its slot
  aheadMin = Math.min(aheadMin, slack);
  if (a - winStart > 3000) {                                   // ran far ahead for a whole window: latency crept up
    if (winStart && aheadMin > B + 0.15) { resync(); anchor(t); }
    ahead = aheadMin; aheadMin = Infinity; winStart = a;
  }
  if (type === 2) audio(t, p); else if (type === 1) frame(t, p);
}

function audio(t, p) {
  if (!audioOn()) return;                                      // silent until Start
  const n = p.length >> 1, s16 = new Int16Array(p.buffer, 0, n);
  const ab = ctx.createBuffer(1, n, sr), f = ab.getChannelData(0);
  for (let i = 0; i < n; i++) f[i] = s16[i] / 32768;
  let when = base + t, off = 0;
  const cur = ctx.currentTime;
  if (when < cur) {                                            // late: trim a little, re-anchor a lot
    S.lateA++;
    if (ADAPT) B = Math.min(0.5, B + 0.02);
    off = cur - when;
    if (off > TRIM || off >= ab.duration) { resync(); anchor(t); when = base + t; off = 0; } else when = cur;
  }
  const s = ctx.createBufferSource();
  s.buffer = ab; s.connect(ctx.destination); s.start(when, off);
  s.onended = () => { const k = sources.indexOf(s); if (k >= 0) sources.splice(k, 1); };
  sources.push(s);
  audioEnd = Math.max(audioEnd, when + ab.duration - off);
}

function frame(t, p) {
  frTimes.push(performance.now());
  const my = gen;
  createImageBitmap(new Blob([p], {type: 'image/jpeg'})).then(bmp => {
    if (my !== gen) return bmp.close();
    if (base !== null && base + t < showT()) S.lateF++;        // its moment passed before it decoded
    let i = frames.length;
    while (i > 0 && frames[i - 1].t > t) i--;
    frames.splice(i, 0, {t, bmp});
    while (frames.length > 90) frames.shift().bmp.close();
  }, () => S.badF++);
}

function tick() {                                              // draw the newest frame whose time has come
  requestAnimationFrame(tick);
  if (base === null || !frames.length) return;
  const st = showT() - base;
  let k = -1;
  while (k + 1 < frames.length && frames[k + 1].t <= st) k++;
  if (k < 0) return;
  if (frames[k].t !== drawnT) { g.drawImage(frames[k].bmp, 0, 0, cv.width, cv.height); drawnT = frames[k].t; }
  for (const o of frames.splice(0, k)) o.bmp.close();
}

function hello(h) {
  gen++; sr = h.sample_rate || sr;
  if (h.width && h.height && (cv.width !== h.width || cv.height !== h.height)) { cv.width = h.width; cv.height = h.height; fit(); }
  for (const f of frames) f.bmp.close();
  frames = []; drawnT = null; base = null; lastTransit = null; stopAudio();
  const skip = new Set(['width', 'height', 'fps', 'sample_rate']);
  infoEl.textContent = Object.entries(h).filter(([k, v]) => !skip.has(k) && v !== null && typeof v !== 'object')
    .map(([k, v]) => k + ' ' + v).concat([h.width + '×' + h.height + ' @ ' + h.fps]).join(' · ');
  msgEl.textContent = '';
}

async function run() {
  for (;;) {
    try {
      const r = await fetch('stream', {cache: 'no-store'});
      if (!r.ok || !r.body) throw new Error('HTTP ' + r.status);
      const rd = r.body.getReader(), push = makeParser(onMsg);
      for (;;) { const {value, done} = await rd.read(); if (done) break; push(value); }
    } catch (e) { console.warn('stream:', e); }
    msgEl.textContent = 'waiting for stream…'; base = null; stopAudio();
    await new Promise(r => setTimeout(r, 1000));
  }
}

go.onclick = async () => {
  try {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!ctx) try { ctx = new AC({latencyHint: 'interactive', sampleRate: sr}); }   // chunks tile at their native rate
              catch (e) { ctx = new AC({latencyHint: 'interactive'}); }            // rate unsupported: browser resamples
    ctx.onstatechange = () => { base = null; go.style.display = audioOn() ? 'none' : ''; };
    await ctx.resume();
    go.style.display = 'none'; base = null;                    // switch to the audio clock: re-anchor
  } catch (e) { msgEl.textContent = 'audio failed: ' + e.message; }
};

function fit() {
  const s = Math.min(innerWidth / cv.width, (innerHeight - 30) / cv.height);
  cv.style.width = Math.floor(cv.width * s) + 'px'; cv.style.height = Math.floor(cv.height * s) + 'px';
}

setInterval(() => {
  const a = performance.now();
  frTimes = frTimes.filter(x => x > a - 1000);
  const mbit = (S.bytes - lastBytes) * 8e-3 / Math.max(1, a - lastStat); lastBytes = S.bytes; lastStat = a;
  const ab = audioOn() ? Math.round(Math.max(0, audioEnd - ctx.currentTime) * 1000) + ' ms buffered' : 'off';
  stEl.textContent = frTimes.length + ' fps · audio ' + ab + ' · late frames ' + S.lateF + ' · late audio ' + S.lateA +
    ' · jitter ' + (S.jit * 1000).toFixed(1) + ' ms · playout buffer ' + Math.round(B * 1000) + ' ms (min ahead ' +
    (isFinite(ahead) ? Math.round(ahead * 1000) : '–') + ' ms) · resyncs ' + S.resync + ' · ' + mbit.toFixed(1) + ' Mbit/s' +
    (S.badF ? ' · bad frames ' + S.badF : '');
}, 500);

addEventListener('resize', fit);
fit();
requestAnimationFrame(tick);
run();
</script>
</body></html>
"""
