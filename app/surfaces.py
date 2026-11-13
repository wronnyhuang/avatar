"""scene: live surfaces -- the chalkboard ("board") and the wall TV ("tv") whose pixels the hub owns.

The hub keeps one RGBA8 canvas per surface for the whole session and streams every change to the renderer as an
'S' message (docs/SCENE.md section 7), timestamped on the PLAYBACK TIMELINE (seconds; the same clock
as the page's 'R' rig frames). The renderer keeps a mirror of each canvas and applies an update when the frame it
renders reaches the message's t.

    from surfaces import Surfaces
    S = Surfaces(send=link.send_bytes)        # or Surfaces() and attach(send) later
    B = S.get("board")                        # 1024x512, sid 0
    B.draw_polyline([[0.1, 0.2], [0.2, 0.3]], 0.012, t=12.5)      # width in v units (fraction of canvas height)
    B.draw_polyline(pts, 0.012, ts=[12.0, 12.1, 12.2])            # one op per segment, segment i at ts[i+1]
    B.erase_disc((0.5, 0.5), 0.05, t=None)    # t None = apply now
    T = S.get("tv")                           # 512x288, sid 1
    T.put_frame(rgb)                          # any size, resized; alpha 255 = picture on
    T.clear()                                 # alpha 0 = idle/off

Wire ('S', 24-byte little-endian header, struct "<BBBBdIHHHH"):
    0x53, uint8 sid (0 board, 1 tv), uint8 codec (0 raw, 1 zlib, 2 jpeg), uint8 flags,
    float64 t (0.0 when flags bit0), uint32 seq (per Surfaces, +1 per message), uint16 x, y, w, h, payload
    flags bit0 NOW (apply at the next frame), bit1 RESYNC (full canvas), bit2 CLEAR (zero the canvas first)
    payload codec 0: h rows x w x RGBA8, top row first; codec 1: the same, zlib (level 1); codec 2: a JPEG of the rect's
    RGB (alpha 255). zlib is used when it is smaller than raw; jpeg only for TV rects whose zlib output > 48 KB.

Semantics chosen here:
  * clear(): ONE message with flags bit2 and an EMPTY rect (x=y=w=h=0, no payload): the receiver zeroes the canvas.
  * Applying a message (apply_msg): if bit2, zero the canvas; then, if w*h > 0, overwrite canvas[y:y+h, x:x+w] with the
    pixels. Every message carries the rect's pixels AFTER the op, so applying in order rebuilds snapshot() exactly
    (jpeg rects are lossy).
  * attach(send) / resync(): for every surface, a non-blank one -> one full-canvas message (bit1|bit2, t now); a blank
    one -> an empty-rect resync clear (bit1|bit2, w=h=0), so a reused renderer drops a stale board or picture.
    Known limit: the resync carries the canvas with every op applied, including ops stamped in the future, so after a
    relink those few pending ops show at once instead of at their t.
  * t ordering per surface: an op stamped earlier than the surface's previous stamped op gets the previous t. An op
    with t=None after a stamped op also takes the previous t (a past t applies at once, a future one in order) so a
    "now" op never overtakes a pending one; t=None with no earlier stamped op goes out with the NOW flag.

Call from the hub's event loop (not thread-safe). Every op returns the list of message bytes it produced.
"""
from __future__ import annotations

import io
import struct
import zlib

import numpy as np

S = 0x53
S_HDR = struct.Struct("<BBBBdIHHHH")          # 24 bytes
CODEC_RAW, CODEC_ZLIB, CODEC_JPEG = 0, 1, 2
F_NOW, F_RESYNC, F_CLEAR = 1, 2, 4
JPEG_OVER = 48 * 1024                          # TV rects whose zlib output exceeds this go as JPEG
EDGE_PX = 1.5                                  # soft edge width (smoothstep band) of chalk and erasers
CHALK = (240, 240, 232, 235)


# ------------------------------------------------------------------------------------------------- pack / unpack
def pack_msg(sid: int, flags: int, t: float | None, seq: int, x: int, y: int, pixels: np.ndarray | None,
             allow_jpeg: bool = False) -> bytes:
    """pixels: h x w x 4 uint8 (None / empty = no payload). t None -> NOW flag, t field 0.0."""
    if t is None:
        flags |= F_NOW
        t = 0.0
    if pixels is None or pixels.size == 0:
        return S_HDR.pack(S, sid, CODEC_RAW, flags, float(t), seq & 0xFFFFFFFF, 0, 0, 0, 0)
    h, w = pixels.shape[:2]
    raw = np.ascontiguousarray(pixels, np.uint8).tobytes()
    z = zlib.compress(raw, 1)
    if allow_jpeg and len(z) > JPEG_OVER:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(pixels[:, :, :3])).save(buf, "JPEG", quality=85)
        codec, payload = CODEC_JPEG, buf.getvalue()
    elif len(z) < len(raw):
        codec, payload = CODEC_ZLIB, z
    else:
        codec, payload = CODEC_RAW, raw
    return S_HDR.pack(S, sid, codec, flags, float(t), seq & 0xFFFFFFFF, x, y, w, h) + payload


def unpack_msg(msg: bytes) -> dict:
    """'S' -> {sid, codec, flags, t, seq, x, y, w, h, pixels (h, w, 4) uint8}. Raises ValueError on a bad message."""
    if len(msg) < S_HDR.size or msg[0] != S:
        raise ValueError("bad 'S' message")
    _, sid, codec, flags, t, seq, x, y, w, h = S_HDR.unpack_from(msg, 0)
    body = msg[S_HDR.size:]
    if w == 0 or h == 0:
        px = np.zeros((h, w, 4), np.uint8)
    elif codec == CODEC_RAW or codec == CODEC_ZLIB:
        raw = body if codec == CODEC_RAW else zlib.decompress(body)
        if len(raw) != w * h * 4:
            raise ValueError("bad 'S' payload size")
        px = np.frombuffer(raw, np.uint8).reshape(h, w, 4).copy()
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
    return {"sid": sid, "codec": codec, "flags": flags, "t": t, "seq": seq, "x": x, "y": y, "w": w, "h": h,
            "pixels": px}


def apply_msg(canvas: np.ndarray, d: dict) -> None:
    """Apply one unpacked message to a mirror canvas (H x W x 4 uint8), in place."""
    if d["flags"] & F_CLEAR:
        canvas[:] = 0
    if d["w"] and d["h"]:
        x, y = d["x"], d["y"]
        canvas[y:y + d["h"], x:x + d["w"]] = d["pixels"]


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    u = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


# ------------------------------------------------------------------------------------------------- surfaces
class Surface:
    def __init__(self, owner: "Surfaces", name: str, sid: int, w: int, h: int):
        self.owner, self.name, self.sid, self.w, self.h = owner, name, sid, w, h
        self.canvas = np.zeros((h, w, 4), np.uint8)
        self.last_t: float | None = None
        self.jpeg = name == "tv"

    # -- common
    def _stamp(self, t: float | None) -> float | None:
        if t is None:
            return self.last_t                     # None (NOW) until the first stamped op
        t = float(t)
        if self.last_t is not None and t < self.last_t:
            t = self.last_t
        self.last_t = t
        return t

    def _emit_rect(self, x0: int, y0: int, x1: int, y1: int, t: float | None, flags: int = 0) -> list[bytes]:
        if x1 <= x0 or y1 <= y0:
            return []
        msg = pack_msg(self.sid, flags, self._stamp(t), self.owner._next_seq(), x0, y0, self.canvas[y0:y1, x0:x1],
                       allow_jpeg=self.jpeg)
        self.owner._out(msg)
        return [msg]

    def clear(self, t: float | None = None) -> list[bytes]:
        self.canvas[:] = 0
        msg = pack_msg(self.sid, F_CLEAR, self._stamp(t), self.owner._next_seq(), 0, 0, None)
        self.owner._out(msg)
        return [msg]

    def snapshot(self) -> np.ndarray:
        return self.canvas.copy()

    def blank(self) -> bool:
        return not self.canvas[:, :, 3].any() if self.name == "tv" else not self.canvas.any()

    def resync_msg(self) -> bytes:
        seq = self.owner._next_seq()
        if self.blank():
            return pack_msg(self.sid, F_RESYNC | F_CLEAR, None, seq, 0, 0, None)
        return pack_msg(self.sid, F_RESYNC | F_CLEAR, None, seq, 0, 0, self.canvas, allow_jpeg=self.jpeg)

    def _clip(self, x0, y0, x1, y1):
        return max(0, int(x0)), max(0, int(y0)), min(self.w, int(x1)), min(self.h, int(y1))

    # -- board ops
    def ink_grid(self, cols: int = 16, rows: int = 8) -> np.ndarray:
        """Mean alpha/255 per cell, (rows, cols) float."""
        a = self.canvas[:, :, 3].astype(np.float32) / 255.0
        ys = np.linspace(0, self.h, rows + 1).astype(int)
        xs = np.linspace(0, self.w, cols + 1).astype(int)
        out = np.zeros((rows, cols), np.float64)
        for r in range(rows):
            band = a[ys[r]:ys[r + 1]]
            for c in range(cols):
                cell = band[:, xs[c]:xs[c + 1]]
                out[r, c] = float(cell.mean()) if cell.size else 0.0
        return out

    def _seg_cov(self, p, q, half, box):
        """Coverage of the capsule p-q (pixel coords) of radius `half` over box (x0, y0, x1, y1)."""
        x0, y0, x1, y1 = box
        xs = np.arange(x0, x1, dtype=np.float32) + 0.5
        ys = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5
        dx, dy = q[0] - p[0], q[1] - p[1]
        L2 = dx * dx + dy * dy
        px, py = xs - p[0], ys - p[1]
        if L2 > 1e-9:
            s = np.clip((px * dx + py * dy) / L2, 0.0, 1.0)
            ex, ey = px - s * dx, py - s * dy
        else:
            ex, ey = px, py
        d = np.sqrt(ex * ex + ey * ey)
        return 1.0 - _smoothstep(half - EDGE_PX / 2, half + EDGE_PX / 2, d)

    def _seg_box(self, p, q, half):
        m = half + EDGE_PX
        return self._clip(np.floor(min(p[0], q[0]) - m), np.floor(min(p[1], q[1]) - m),
                          np.ceil(max(p[0], q[0]) + m) + 1, np.ceil(max(p[1], q[1]) + m) + 1)

    def _composite(self, box, cov, rgba):
        x0, y0, x1, y1 = box
        dst = self.canvas[y0:y1, x0:x1]
        a_s = cov * (rgba[3] / 255.0)
        a_d = dst[:, :, 3].astype(np.float32) / 255.0
        a_o = a_s + a_d * (1.0 - a_s)
        src = np.asarray(rgba[:3], np.float32)
        cd = dst[:, :, :3].astype(np.float32)
        w_d = (a_d * (1.0 - a_s))[:, :, None]
        num = src * a_s[:, :, None] + cd * w_d
        col = np.where(a_o[:, :, None] > 1e-6, num / np.maximum(a_o, 1e-6)[:, :, None], 0.0)
        out = np.empty_like(dst)
        out[:, :, :3] = np.clip(np.rint(col), 0, 255)
        out[:, :, 3] = np.clip(np.rint(a_o * 255.0), 0, 255)
        touched = cov > 0
        dst[touched] = out[touched]

    def draw_polyline(self, pts_uv, width: float, rgba=CHALK, t: float | None = None, ts=None) -> list[bytes]:
        pts = [(float(u) * self.w, float(v) * self.h) for u, v in pts_uv]
        if not pts:
            return []
        if len(pts) == 1:
            pts = pts * 2
        half = max(0.25, float(width) * self.h / 2.0)
        rgba = tuple(int(c) for c in rgba)
        if ts is not None:                          # one op per segment at the time the tip reaches its end
            ts = list(ts)
            if len(ts) == 1:
                ts = ts * 2
            if len(ts) != len(pts):
                raise ValueError("ts must have one time per point")
            out = []
            for i in range(len(pts) - 1):
                box = self._seg_box(pts[i], pts[i + 1], half)
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                self._composite(box, self._seg_cov(pts[i], pts[i + 1], half, box), rgba)
                out += self._emit_rect(*box, t=ts[i + 1])
            return out
        boxes = [self._seg_box(pts[i], pts[i + 1], half) for i in range(len(pts) - 1)]
        boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
        if not boxes:
            return []
        U = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
        cov = np.zeros((U[3] - U[1], U[2] - U[0]), np.float32)
        for i in range(len(pts) - 1):            # max over segments: joints are not composited twice
            b = self._seg_box(pts[i], pts[i + 1], half)
            if b[2] <= b[0] or b[3] <= b[1]:
                continue
            sub = cov[b[1] - U[1]:b[3] - U[1], b[0] - U[0]:b[2] - U[0]]
            np.maximum(sub, self._seg_cov(pts[i], pts[i + 1], half, b), out=sub)
        self._composite(U, cov, rgba)
        return self._emit_rect(*U, t=t)

    def _erase(self, box, inside, residue):
        x0, y0, x1, y1 = box
        f = 1.0 - (1.0 - float(residue)) * inside
        dst = self.canvas[y0:y1, x0:x1, 3]
        dst[:] = np.clip(np.rint(dst.astype(np.float32) * f), 0, 255).astype(np.uint8)

    def erase_disc(self, center_uv, r: float, t: float | None = None, residue: float = 0.08) -> list[bytes]:
        cx, cy = float(center_uv[0]) * self.w, float(center_uv[1]) * self.h
        rp = float(r) * self.h
        m = rp + EDGE_PX
        box = self._clip(np.floor(cx - m), np.floor(cy - m), np.ceil(cx + m) + 1, np.ceil(cy + m) + 1)
        if box[2] <= box[0] or box[3] <= box[1]:
            return []
        xs = np.arange(box[0], box[2], dtype=np.float32) + 0.5 - cx
        ys = np.arange(box[1], box[3], dtype=np.float32)[:, None] + 0.5 - cy
        d = np.sqrt(xs * xs + ys * ys)
        self._erase(box, 1.0 - _smoothstep(rp - EDGE_PX / 2, rp + EDGE_PX / 2, d), residue)
        return self._emit_rect(*box, t=t)

    def erase_rect(self, uv0, uv1, t: float | None = None, residue: float = 0.08) -> list[bytes]:
        ax, bx = sorted((float(uv0[0]) * self.w, float(uv1[0]) * self.w))
        ay, by = sorted((float(uv0[1]) * self.h, float(uv1[1]) * self.h))
        box = self._clip(np.floor(ax - EDGE_PX), np.floor(ay - EDGE_PX), np.ceil(bx + EDGE_PX) + 1,
                         np.ceil(by + EDGE_PX) + 1)
        if box[2] <= box[0] or box[3] <= box[1]:
            return []
        xs = np.arange(box[0], box[2], dtype=np.float32) + 0.5
        ys = np.arange(box[1], box[3], dtype=np.float32)[:, None] + 0.5
        ix = np.clip(np.minimum(xs - ax, bx - xs) / EDGE_PX + 0.5, 0.0, 1.0)
        iy = np.clip(np.minimum(ys - ay, by - ys) / EDGE_PX + 0.5, 0.0, 1.0)
        self._erase(box, ix * iy, residue)
        return self._emit_rect(*box, t=t)

    # -- tv ops
    def put_frame(self, rgb, t: float | None = None) -> list[bytes]:
        a = np.asarray(rgb)
        if a.ndim != 3 or a.shape[2] < 3:
            raise ValueError("put_frame wants H x W x 3")
        a = a[:, :, :3].astype(np.uint8, copy=False)
        if a.shape[:2] != (self.h, self.w):
            from PIL import Image
            a = np.asarray(Image.fromarray(np.ascontiguousarray(a)).resize((self.w, self.h), Image.BILINEAR))
        new = np.empty_like(self.canvas)
        new[:, :, :3] = a
        new[:, :, 3] = 255
        diff = np.any(new != self.canvas, axis=2)
        rows, cols = np.flatnonzero(diff.any(axis=1)), np.flatnonzero(diff.any(axis=0))
        if not len(rows):
            return []
        self.canvas[:] = new
        return self._emit_rect(int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1, t=t)

    # tv: put_frame split in two so the heavy half (diff + zlib/JPEG, ~5-15 ms) runs OFF the hub event loop.
    # encode_frame is pure (thread-safe: reads only its arguments); commit_frame (on the loop) stamps seq/t, updates the
    # canvas and sends. prev = the frame given to the previous encode_frame (None = send the whole frame).
    def encode_frame(self, rgb, prev=None):
        a = np.asarray(rgb)
        if a.ndim != 3 or a.shape[2] < 3:
            raise ValueError("encode_frame wants H x W x 3")
        a = a[:, :, :3].astype(np.uint8, copy=False)
        if a.shape[:2] != (self.h, self.w):
            from PIL import Image
            a = np.asarray(Image.fromarray(np.ascontiguousarray(a)).resize((self.w, self.h), Image.BILINEAR))
        if prev is not None and getattr(prev, "shape", None) == a.shape:
            diff = np.any(a != prev, axis=2)
            rows, cols = np.flatnonzero(diff.any(axis=1)), np.flatnonzero(diff.any(axis=0))
            if not len(rows):
                return None
            x0, y0, x1, y1 = int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1
        else:
            x0, y0, x1, y1 = 0, 0, self.w, self.h
        rgba = np.empty((y1 - y0, x1 - x0, 4), np.uint8)
        rgba[:, :, :3] = a[y0:y1, x0:x1]
        rgba[:, :, 3] = 255
        return {"frame": a, "rect": (x0, y0, x1, y1), "rgba": rgba,
                "msg": pack_msg(self.sid, 0, None, 0, x0, y0, rgba, allow_jpeg=self.jpeg)}

    def commit_frame(self, enc, t: float | None = None) -> list[bytes]:
        if not enc:
            return []
        x0, y0, x1, y1 = enc["rect"]
        self.canvas[y0:y1, x0:x1] = enc["rgba"]
        msg = enc["msg"]
        _, sid, codec, flags, _t, _seq, x, y, w, h = S_HDR.unpack_from(msg, 0)
        ts = self._stamp(t)
        flags = (flags & ~F_NOW) | (F_NOW if ts is None else 0)
        msg = S_HDR.pack(S, sid, codec, flags, float(ts or 0.0), self.owner._next_seq() & 0xFFFFFFFF, x, y, w, h) + msg[S_HDR.size:]
        self.owner._out(msg)
        return [msg]


def test_pattern(phase: float, w: int = 512, h: int = 288) -> np.ndarray:
    """A moving colour test frame (h x w x 3 uint8): colour bars, a circle that travels with `phase` (1.0 = one
    lap), and a frame counter (round(phase * 100))."""
    from PIL import Image, ImageDraw
    bars = np.array([(235, 235, 235), (235, 235, 16), (16, 235, 235), (16, 235, 16), (235, 16, 235), (235, 16, 16),
                     (16, 16, 235)], np.uint8)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = bars[(np.arange(w) * len(bars)) // w][None, :, :]
    img[int(h * 0.75):] = (24, 24, 32)
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    a = 2 * np.pi * (phase % 1.0)
    cx, cy, r = w / 2 + 0.35 * w * np.cos(a), h * 0.4 + 0.22 * h * np.sin(a), h * 0.12
    dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(20, 20, 20), outline=(255, 255, 255), width=3)
    dr.text((10, int(h * 0.78)), f"TV TEST  frame {int(round(phase * 100)):05d}", fill=(255, 255, 255))
    return np.asarray(im)


class Surfaces:
    SPECS = (("board", 0, 1024, 512), ("tv", 1, 512, 288))

    def __init__(self, send=None):
        self.send = send
        self.seq = 0
        self.by_name = {n: Surface(self, n, sid, w, h) for n, sid, w, h in self.SPECS}
        self.by_sid = {s.sid: s for s in self.by_name.values()}
        self.sent = 0                       # messages handed to the sink
        self.sent_bytes = 0

    def get(self, name: str) -> Surface:
        return self.by_name[name]

    def _next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def _out(self, msg: bytes) -> None:
        if self.send is not None:
            self.sent += 1
            self.sent_bytes += len(msg)
            self.send(msg)

    def attach(self, send) -> list[bytes]:
        """Set the sink and re-send every canvas (a new / re-created render link). send None = detach."""
        self.send = send
        return self.resync() if send is not None else []

    def detach(self) -> None:
        self.send = None

    def resync(self) -> list[bytes]:
        out = []
        for s in self.by_name.values():
            m = s.resync_msg()
            self._out(m)
            out.append(m)
        return out
