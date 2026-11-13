"""scene: the render server's side of the live surfaces (docs/SCENE.md section 7).

The app server streams 'S' messages (a dirty rect of the board / TV canvas, timestamped on the playback timeline).  The
server keeps a MIRROR of each canvas: a message is applied to the mirror when the frame being dispatched reaches its
t (NOW messages at the next frame), the changed rect joins that surface's dirty box, and before each job the dirty
boxes go to the render worker as premultiplied RGBA sub-rect uploads, at most `budget` bytes per frame (a bigger
change spreads over the next frames; the newest pixels always win because uploads read the mirror at send time).
A new / switched engine gets the whole canvases again (mark_all_dirty).

The wire decoder is a copy of app/surfaces.py unpack_msg (so renderer/ runs without app/ on the path).
"""
from __future__ import annotations

import io
import os
import struct
import threading
import zlib

import numpy as np

S = 0x53
S_HDR = struct.Struct("<BBBBdIHHHH")          # 24 bytes
CODEC_RAW, CODEC_ZLIB, CODEC_JPEG = 0, 1, 2
F_NOW, F_RESYNC, F_CLEAR = 1, 2, 4
SURFACES = {0: ("board", 1024, 512), 1: ("tv", 512, 288)}


def unpack_msg(msg) -> dict:
    """'S' -> {sid, codec, flags, t, seq, x, y, w, h, pixels (h, w, 4) uint8}.  Raises ValueError on a bad message."""
    msg = bytes(msg)
    if len(msg) < S_HDR.size or msg[0] != S:
        raise ValueError("bad 'S' message")
    _, sid, codec, flags, t, seq, x, y, w, h = S_HDR.unpack_from(msg, 0)
    body = msg[S_HDR.size:]
    if w == 0 or h == 0:
        px = np.zeros((h, w, 4), np.uint8)
    elif codec in (CODEC_RAW, CODEC_ZLIB):
        raw = body if codec == CODEC_RAW else zlib.decompress(body)
        if len(raw) != w*h*4:
            raise ValueError("bad 'S' payload size")
        px = np.frombuffer(raw, np.uint8).reshape(h, w, 4)
    elif codec == CODEC_JPEG:
        from PIL import Image
        rgb = np.asarray(Image.open(io.BytesIO(body)).convert("RGB"))
        if rgb.shape[:2] != (h, w):
            raise ValueError("bad 'S' jpeg size")
        px = np.empty((h, w, 4), np.uint8)
        px[:, :, :3] = rgb
        px[:, :, 3] = 255
    else:
        raise ValueError("bad 'S' codec")
    return {"sid": sid, "codec": codec, "flags": flags, "t": t, "seq": seq, "x": x, "y": y, "w": w, "h": h, "pixels": px}


def premultiply(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3:4].astype(np.uint16)
    out = np.empty_like(rgba)
    out[..., :3] = ((rgba[..., :3].astype(np.uint16)*a + 127)//255).astype(np.uint8)
    out[..., 3] = rgba[..., 3]
    return out


class SurfaceMirror:
    """Thread-safe: feed() from the websocket thread, advance()/take() from the dispatcher."""

    def __init__(self, budget=None):
        self.lock = threading.Lock()
        self.canvas = {sid: np.zeros((h, w, 4), np.uint8) for sid, (_, w, h) in SURFACES.items()}
        self.pending = []            # (t, seq, msg dict); t = -inf for NOW
        self.dirty = {}              # sid -> [x0, y0, x1, y1]
        self.budget = int(budget or os.environ.get("AVATAR_SURF_BYTES", 1 << 20))
        self.stats = {"msgs": 0, "bad": 0, "applied": 0, "late": 0, "upload_bytes": 0, "uploads": 0}
        self.last_t = None

    def feed(self, msg) -> bool:
        try:
            d = unpack_msg(msg)
        except Exception:        # noqa: BLE001 — a bad message is dropped and counted, never fatal
            self.stats["bad"] += 1
            return False
        if d["sid"] not in self.canvas:
            self.stats["bad"] += 1
            return False
        t = float("-inf") if d["flags"] & F_NOW else float(d["t"])
        with self.lock:
            self.pending.append((t, d["seq"], d))
            self.stats["msgs"] += 1
            if self.last_t is not None and t != float("-inf") and t < self.last_t:
                self.stats["late"] += 1
        return True

    def _mark(self, sid, x0, y0, x1, y1):
        b = self.dirty.get(sid)
        self.dirty[sid] = [x0, y0, x1, y1] if b is None else [min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1)]

    def advance(self, t_frame: float) -> int:
        """Apply every pending message with t <= t_frame (in t, then seq order) to the mirrors."""
        n = 0
        with self.lock:
            self.last_t = t_frame
            if not self.pending:
                return 0
            due = [p for p in self.pending if p[0] <= t_frame]
            if not due:
                return 0
            self.pending = [p for p in self.pending if p[0] > t_frame]
            due.sort(key=lambda p: (p[0], p[1]))
            for _, _, d in due:
                c = self.canvas[d["sid"]]
                H, W = c.shape[:2]
                if d["flags"] & F_CLEAR:
                    c[:] = 0
                    self._mark(d["sid"], 0, 0, W, H)
                w, h, x, y = d["w"], d["h"], d["x"], d["y"]
                if w and h:
                    x1, y1 = min(W, x + w), min(H, y + h)
                    if x1 > x and y1 > y:
                        c[y:y1, x:x1] = d["pixels"][:y1 - y, :x1 - x]
                        self._mark(d["sid"], x, y, x1, y1)
                n += 1
            self.stats["applied"] += n
        return n

    def mark_all_dirty(self):
        with self.lock:
            for sid, c in self.canvas.items():
                self.dirty[sid] = [0, 0, c.shape[1], c.shape[0]]

    def take(self, budget=None):
        """-> [(name, x, y, w, h, premultiplied bytes)] for the dirty boxes, within the byte budget (row strips)."""
        budget = self.budget if budget is None else budget
        out = []
        with self.lock:
            for sid in sorted(self.dirty):
                x0, y0, x1, y1 = self.dirty[sid]
                row = (x1 - x0)*4
                if row <= 0 or y1 <= y0:
                    del self.dirty[sid]
                    continue
                rows = max(1, min(y1 - y0, budget//row))
                if budget < row and out:
                    break
                c = self.canvas[sid]
                data = premultiply(c[y0:y0 + rows, x0:x1]).tobytes()
                out.append((SURFACES[sid][0], x0, y0, x1 - x0, rows, data))
                budget -= len(data)
                self.stats["upload_bytes"] += len(data)
                self.stats["uploads"] += 1
                if y0 + rows >= y1:
                    del self.dirty[sid]
                else:
                    self.dirty[sid] = [x0, y0 + rows, x1, y1]
                if budget <= 0:
                    break
        return out

    def snapshot(self, name):
        sid = [k for k, v in SURFACES.items() if v[0] == name][0]
        with self.lock:
            return self.canvas[sid].copy()
