"""Binary wire format shared by the hub, the render server and the page (AVATAR.md "Wire format").

All integers little-endian.
  'A' audio  : 0x41, 3 x 0, uint64 first sample index (24 kHz timeline), int16 PCM mono
  'V' video  : 0x56, flags (bit0 keyframe, bit1 jpeg), 2 x 0, uint32 frame number, float64 t (s), payload
  'M' mic    : 0x4D, flags, 2 x 0, uint32 seq, PCM mono. flags 0 = int16 24 kHz (old pages); bit0 = G.711 mu-law
               bytes (1 per sample), bit1 = 16 kHz (latency: 16 kHz mu-law = 128 kbps instead of 384)
  'P' playback: 0x50, flags, 2 x 0, uint64 s0, uint32 n, PCM (app.py P_HDR). flags bit0 silent run (no payload),
               bit1 = mu-law payload (n bytes instead of 2n)
  'R' rig    : 0x52, flags, uint16 n, float64 t (playback-timeline s), n x int16 = value * 10000 (clamped +-3.2767)
               -- the binary form of {"type":"rig","t":..,"a":[...]} (latency: ~160 B instead of ~600 B)
  'J' json   : 0x4A, UTF-8 JSON  (HTTP transport only)
  'R' + rx   : (learned-reaction mode only) flags bit0 = after the n values: uint8 m, m x (uint8 index,
               int16 alpha*1e4, int16 beta*1e4), int16 reflex-blink*1e4 — per rig channel final = alpha*learned + beta
               (blinks: alpha*max(learned, reflex blink) + beta); channels not listed: alpha 1, beta 0
  'H' human  : (learned mode only) 0x48, flags (bit0 face, bit1 body, bit2 hand_l, bit3 hand_r valid),
               uint16 n (86), float64 t (playback-timeline s of the camera capture), n x int16 = value*1e4 in
               react/spec.py MOTION_COLS order, except cols 55-57 = RAW head translation in metres*1e4 (cm/100)
               and cols 60-77 (body, shoulder widths) = value*1e3
  'U' mp     : (unified mode only) one RAW MediaPipe result, as the page's percept workers produced it:
               0x55, kind (1 face, 2 pose, 3 hands), uint16 n, float64 t (playback-timeline s of the camera capture),
               uint16 w, uint16 h (the worker bitmap), uint8 present, then
                 face  n=52: 52 x int16 blendshape*1e4 (spec / MediaPipe order), 16 x float32 facial transformation
                       matrix COLUMN-major (as tasks-vision gives it in JS)
                 pose  n=33: 33 x 4 int16*1e4 image (x, y, z, visibility), 33 x 4 int16*1e4 world (x, y, z metres, vis)
                 hands n=number of hands: per hand uint8 label (0 'Left', 1 'Right', 2 unknown = MediaPipe's raw
                       handedness), int16 score*1e4, 21 x 3 int16*1e4 image, 21 x 3 int16*1e4 world (metres)
HTTP bodies: up = concatenated [uint32 len][message]; down = [uint32 seq][uint32 len][message]...
"""
from __future__ import annotations

import json
import struct

import numpy as np

SR = 24000
A_HDR = struct.Struct("<B3xQ")     # 12 bytes
V_HDR = struct.Struct("<BB2xId")   # 16 bytes
M_HDR = struct.Struct("<B3xI")     # 8 bytes
U32 = struct.Struct("<I")
A, V, M, J = 0x41, 0x56, 0x4D, 0x4A
R = 0x52
H = 0x48
U = 0x55                           # unified mode: raw MediaPipe result
U_HDR = struct.Struct("<BBHdHHB")  # 17 bytes: type, kind, n, t, w, h, present
U_FACE, U_POSE, U_HANDS = 1, 2, 3
R_RX = 1                           # 'R' flags bit0: learned-mode extension follows
R_HDR = struct.Struct("<BBHd")     # 12 bytes
RIG_SCALE = 10000.0
M_ULAW, M_16K = 1, 2               # 'M' flags
P_ULAW = 2                         # 'P' flags bit1
# scene: 'S' surface update (hub -> renderer): 0x53, uint8 surface (0 board, 1 tv), uint8 codec (0 raw, 1 zlib,
# 2 jpeg), uint8 flags (bit0 now, bit1 resync, bit2 clear first), float64 t, uint32 seq, uint16 x, y, w, h, payload
# (24 bytes; pack/unpack in app/surfaces.py)
S = 0x53
S_HDR = struct.Struct("<BBBBdIHHHH")


def _ulaw_table() -> np.ndarray:
    t = np.zeros(256, np.int16)
    for i in range(256):
        u = ~i & 0xFF
        sign, exp, mant = u & 0x80, (u >> 4) & 7, u & 0x0F
        v = (((mant << 3) + 0x84) << exp) - 0x84
        t[i] = -v if sign else v
    return t


ULAW = _ulaw_table()


def ulaw_decode(b: bytes) -> bytes:
    """G.711 mu-law bytes -> int16 LE PCM bytes."""
    return ULAW[np.frombuffer(b, np.uint8)].astype("<i2").tobytes()


def ulaw_encode(pcm: bytes) -> bytes:
    """int16 LE PCM -> mu-law (tests / tools; the page encodes in JS with the same algorithm)."""
    x = np.frombuffer(pcm, "<i2").astype(np.int32)
    sign = np.where(x < 0, 0x80, 0)
    x = np.minimum(np.abs(x), 32635) + 0x84
    exp = np.floor(np.log2(np.maximum(x, 1))).astype(np.int32) - 7
    exp = np.clip(exp, 0, 7)
    mant = (x >> (exp + 3)) & 0x0F
    return (~(sign | (exp << 4) | mant) & 0xFF).astype(np.uint8).tobytes()


class Up16to24:
    """Streaming 16 kHz -> 24 kHz (x1.5) linear resampler with one input sample of carry-over (62 us delay):
    320 samples in -> 480 out. Enough for the intent ASR (Vosk works at 16 kHz internally anyway)."""

    def __init__(self):
        self.prev = 0.0

    def __call__(self, pcm16k: bytes) -> bytes:
        x = np.frombuffer(pcm16k, "<i2").astype(np.float32)
        if not len(x):
            return b""
        ext = np.concatenate([[self.prev], x])
        n_out = (len(x) * 3) // 2
        q = np.arange(n_out, dtype=np.float32) * (2.0 / 3.0)
        y = np.interp(q, np.arange(len(ext), dtype=np.float32), ext)
        self.prev = float(x[-1])
        return np.clip(np.rint(y), -32768, 32767).astype("<i2").tobytes()


def unpack_rig(msg: bytes) -> tuple[float, list[float]]:
    _, _flags, n, t = R_HDR.unpack_from(msg, 0)
    v = np.frombuffer(msg, "<i2", count=n, offset=R_HDR.size).astype(np.float64) / RIG_SCALE
    return t, [round(float(x), 4) for x in v]


def unpack_rig_rx(msg: bytes) -> tuple[float, np.ndarray, list[tuple[int, float, float]] | None, float]:
    """'R' with the optional learned-mode extension -> (t, values float32, [(index, alpha, beta)] or None, reflex blink)."""
    _, flags, n, t = R_HDR.unpack_from(msg, 0)
    off = R_HDR.size
    a = np.frombuffer(msg, "<i2", count=n, offset=off).astype(np.float32) / RIG_SCALE
    off += 2 * n
    if not (flags & R_RX) or len(msg) < off + 1:
        return t, a, None, 0.0
    m = msg[off]
    off += 1
    if len(msg) < off + 5 * m + 2:
        return t, a, None, 0.0
    ab = []
    for i in range(m):
        idx = msg[off]
        al, be = struct.unpack_from("<hh", msg, off + 1)
        ab.append((idx, al / RIG_SCALE, be / RIG_SCALE))
        off += 5
    rb = struct.unpack_from("<h", msg, off)[0] / RIG_SCALE
    return t, a, ab, rb


def unpack_human(msg: bytes) -> tuple[float, np.ndarray, np.ndarray]:
    """'H' -> (t, motion (86,) float32 with cols 55-57 in CM (raw, not median-subtracted), mask (4,) uint8)."""
    _, flags, n, t = R_HDR.unpack_from(msg, 0)
    if n != 86 or len(msg) < R_HDR.size + 2 * n:
        raise ValueError("bad 'H' message")
    v = np.frombuffer(msg, "<i2", count=n, offset=R_HDR.size).astype(np.float32) / RIG_SCALE
    v[55:58] *= 100.0                     # metres -> cm
    v[60:78] *= 10.0                      # body cols travel x1e3 (wrist z reaches 4.4 shoulder widths)
    mask = np.array([(flags >> i) & 1 for i in range(4)], np.uint8)
    return t, v, mask


def pack_human(t: float, motion, mask) -> bytes:
    """tests / tools: the page's 'H' encoder (app/react_feat.js packHuman) in Python."""
    v = np.asarray(motion, np.float64).copy()
    v[55:58] /= 100.0
    v[60:78] /= 10.0
    q = np.clip(np.rint(v * RIG_SCALE), -32767, 32767).astype("<i2")
    flags = sum(int(bool(mask[i])) << i for i in range(4))
    return R_HDR.pack(H, flags, len(q), float(t)) + q.tobytes()


def pack_audio(s0: int, pcm: bytes) -> bytes:
    return A_HDR.pack(A, s0) + pcm


def unpack_audio(msg: bytes) -> tuple[int, bytes]:
    _, s0 = A_HDR.unpack_from(msg, 0)
    return s0, msg[A_HDR.size:]


def unpack_video_header(msg: bytes) -> tuple[int, int, float]:
    _, flags, n, t = V_HDR.unpack_from(msg, 0)
    return flags, n, t


def pack_json(obj) -> bytes:
    return bytes([J]) + json.dumps(obj, separators=(",", ":"), default=str).encode()


def frame_msgs(msgs) -> bytes:
    return b"".join(U32.pack(len(m)) + m for m in msgs)


def iter_msgs(body: bytes, limit: int = 1 << 22):
    off, n = 0, len(body)
    while off + 4 <= n:
        ln = U32.unpack_from(body, off)[0]
        off += 4
        if ln > limit or off + ln > n:
            raise ValueError("bad message length")
        yield body[off:off + ln]
        off += ln
    if off != n:
        raise ValueError("trailing bytes")


# ------------------------------------------------------------------------------------------------ unified 'U'
def _q16(v) -> bytes:
    return np.clip(np.rint(np.asarray(v, np.float64) * RIG_SCALE), -32767, 32767).astype("<i2").tobytes()


def pack_mp(kind: int, t: float, w: int, h: int, present: bool, face_bs=None, face_mat=None, pose_image=None,
            pose_world=None, hands=None) -> bytes:
    """tests / tools: the page's 'U' encoder (app/page.html mpMsg) in Python. hands = [(label 0|1|2, score, image
    (21,3), world (21,3))]; face_mat = 16 floats COLUMN-major (JS tasks-vision order)."""
    if kind == U_FACE:
        body = _q16(face_bs if present else np.zeros(52)) + np.asarray(
            face_mat if (present and face_mat is not None) else np.zeros(16), "<f4").tobytes()
        n = 52
    elif kind == U_POSE:
        body = (_q16(np.asarray(pose_image).reshape(33, 4)) + _q16(np.asarray(pose_world).reshape(33, 4))) if present else b""
        n = 33
    else:
        hands = list(hands or [])
        body = b"".join(struct.pack("<Bh", int(lab), int(round(sc * RIG_SCALE))) + _q16(np.asarray(im).reshape(21, 3))
                        + _q16(np.asarray(wo).reshape(21, 3)) for lab, sc, im, wo in hands)
        n = len(hands)
    return U_HDR.pack(U, kind, n, float(t), int(w), int(h), int(bool(present))) + body


def unpack_mp(msg: bytes) -> dict:
    """'U' -> {"kind", "t", "w", "h", "present", + face: bs (52,), mat (16,) row-major | pose: image (33,4), world
    (33,4) | hands: [{"handedness", "score", "image" (21,3), "world" (21,3)}]}."""
    _, kind, n, t, w, h, present = U_HDR.unpack_from(msg, 0)
    off = U_HDR.size
    out = {"kind": kind, "t": t, "w": w, "h": h, "present": bool(present)}
    q = lambda k, o: np.frombuffer(msg, "<i2", count=k, offset=o).astype(np.float32) / RIG_SCALE  # noqa: E731
    if kind == U_FACE:
        if len(msg) < off + 104 + 64:
            raise ValueError("bad 'U' face")
        out["bs"] = q(52, off)
        out["mat"] = np.frombuffer(msg, "<f4", count=16, offset=off + 104).reshape(4, 4).T.reshape(16).copy()
    elif kind == U_POSE:
        if present:
            if len(msg) < off + 528:
                raise ValueError("bad 'U' pose")
            out["image"] = q(132, off).reshape(33, 4)
            out["world"] = q(132, off + 264).reshape(33, 4)
    elif kind == U_HANDS:
        hs = []
        for i in range(n):
            if len(msg) < off + 255:
                raise ValueError("bad 'U' hands")
            lab, sc = struct.unpack_from("<Bh", msg, off)
            hs.append({"handedness": {0: "Left", 1: "Right"}.get(lab, ""), "score": sc / RIG_SCALE,
                       "image": q(63, off + 3).reshape(21, 3), "world": q(63, off + 129).reshape(21, 3)})
            off += 255
        out["hands"] = hs
    else:
        raise ValueError("bad 'U' kind")
    return out
