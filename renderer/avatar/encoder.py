"""Frame encoders for the render server: H.264 Annex-B for WebCodecs (NVENC when it opens, else libx264), or JPEG.

    enc = open_encoder("h264", 1280, 720, fps=24, bitrate=2_000_000)    # tries h264_nvenc, then libx264
    enc.name, enc.codec_string                                          # "libx264", "avc1.42C01F"
    for frame_no, payload, key in enc.encode(rgb, frame_no, force_key=False):   # (h, w, 3) uint8 RGB, top row first
        ...

H.264: constrained baseline (profile_idc 66 + constraint_set0/1), no B-frames, zero latency (one access unit out per
frame in), an IDR every gop_s seconds and on request (force_key), SPS+PPS in front of every IDR (the encoder repeats
them; encode() also re-inserts the cached pair if an IDR ever comes out without them).  codec_string is read from the
SPS the encoder actually writes (avc1.PPCCLL), so the browser's VideoDecoder.configure() gets the true profile/level.
JPEG: every frame stands alone; key is always True.
"""
from __future__ import annotations

import time
from fractions import Fraction
import os

import numpy as np

H264_ORDER = ("h264_nvenc", "libx264")


def nal_units(buf: bytes):
    """(type, start, end) of every NAL in an Annex-B buffer; start/end span the NAL including its start code."""
    out = []
    i, n = 0, len(buf)
    starts = []
    while True:
        k = buf.find(b"\x00\x00\x01", i)
        if k < 0:
            break
        s = k - 1 if k > 0 and buf[k - 1] == 0 else k
        starts.append((s, k + 3))
        i = k + 3
    for idx, (s, h) in enumerate(starts):
        e = starts[idx + 1][0] if idx + 1 < len(starts) else n
        if h < n:
            out.append((buf[h] & 0x1F, s, e))
    return out


def codec_string_from_sps(sps_nal: bytes) -> str:
    """sps_nal = the NAL with its start code; avc1.<profile_idc><constraint flags><level_idc> in hex."""
    i = sps_nal.find(b"\x00\x00\x01") + 3
    return "avc1.%02X%02X%02X" % (sps_nal[i + 1], sps_nal[i + 2], sps_nal[i + 3])


class H264Encoder:
    kind = "h264"

    def __init__(self, width, height, fps=24, bitrate=2_000_000, gop_s=2.0, prefer=H264_ORDER, threads=4):
        import cv2
        cv2.setNumThreads(max(1, int(os.environ.get("AVATAR_CV_THREADS", "1"))))
        if width % 2 or height % 2:
            raise ValueError("H.264 4:2:0 needs even width and height")
        self.w, self.h, self.fps, self.bitrate = int(width), int(height), int(fps), int(bitrate)
        self.gop = max(1, int(round(gop_s * fps)))
        self.threads = int(threads)
        self.errors = {}
        self.ctx = None
        for name in prefer:
            try:
                self.codec_string, self.headers = self._probe(name)
                self.ctx = self._open(name)
                self.name = name
                break
            except Exception as e:          # NVENC absent, out of memory on a shared card, too many sessions, ...
                self.errors[name] = f"{type(e).__name__}: {e}"[:300]
        if self.ctx is None:
            raise RuntimeError(f"no H.264 encoder opened: {self.errors}")
        self.since_key = None
        self.pending_key = True

    def _open(self, name):
        import av
        ctx = av.CodecContext.create(name, "w")
        ctx.width, ctx.height = self.w, self.h
        ctx.time_base = Fraction(1, self.fps)
        ctx.framerate = Fraction(self.fps, 1)
        ctx.pix_fmt = "yuv420p"
        ctx.bit_rate = self.bitrate
        ctx.gop_size = self.gop
        ctx.max_b_frames = 0
        common = {"maxrate": str(self.bitrate), "bufsize": str(self.bitrate // 2), "forced-idr": "1"}
        if name == "libx264":
            ctx.options = dict(common, preset="ultrafast", tune="zerolatency", profile="baseline",
                               **{"x264-params": f"keyint={self.gop}:min-keyint={self.gop}:scenecut=0:repeat-headers=1:"
                                                 f"threads={self.threads}:sliced-threads=1:aud=0"})
        elif name == "h264_nvenc":
            ctx.options = dict(common, preset="p2", tune="ull", profile="baseline", zerolatency="1", delay="0",
                               rc="cbr", bf="0", **{"rc-lookahead": "0", "strict_gop": "1", "no-scenecut": "1"})
        else:
            ctx.options = common
        ctx.open()
        return ctx

    def _probe(self, name):
        """Encode one grey frame on a throwaway instance: proves the encoder runs and yields the SPS/PPS pair."""
        ctx = self._open(name)
        f = self._frame(np.full((self.h, self.w, 3), 128, np.uint8), 0)
        data = b"".join(bytes(p) for p in ctx.encode(f))
        data += b"".join(bytes(p) for p in ctx.encode(None))
        nals = nal_units(data)
        sps = next((data[s:e] for t, s, e in nals if t == 7), None)
        pps = next((data[s:e] for t, s, e in nals if t == 8), None)
        if sps is None or pps is None:
            raise RuntimeError(f"{name}: no SPS/PPS in the first access unit")
        return codec_string_from_sps(sps), sps + pps

    def _frame(self, rgb, pts):
        import av
        import cv2
        yuv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2YUV_I420)
        f = av.VideoFrame.from_ndarray(yuv, format="yuv420p")
        f.pts = int(pts)
        f.time_base = Fraction(1, self.fps)
        return f

    def request_key(self):
        self.pending_key = True

    def encode(self, rgb, frame_no, force_key=False):
        from av.video.frame import PictureType
        f = self._frame(rgb, frame_no)
        if force_key or self.pending_key:
            f.pict_type = PictureType.I
            self.pending_key = False
        out = []
        for p in self.ctx.encode(f):
            data = bytes(p)
            key = bool(p.is_keyframe)
            if key:
                types = {t for t, _, _ in nal_units(data)}
                if 7 not in types or 8 not in types:
                    data = self.headers + data
            out.append((int(p.pts) if p.pts is not None else int(frame_no), data, key))
        return out

    def close(self):
        try:
            if self.ctx is not None:
                self.ctx.encode(None)
        except Exception:
            pass
        self.ctx = None


class JpegEncoder:
    kind = "jpeg"
    name = "jpeg"
    codec_string = "jpeg"
    errors = {}

    def __init__(self, width, height, fps=24, quality=80, **_):
        import cv2
        cv2.setNumThreads(max(1, int(os.environ.get("AVATAR_CV_THREADS", "1"))))
        self.w, self.h, self.fps, self.quality = width, height, fps, int(quality)

    def request_key(self):
        pass

    def encode(self, rgb, frame_no, force_key=False):
        import cv2
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return [(int(frame_no), buf.tobytes(), True)]

    def close(self):
        pass


def open_encoder(codec, width, height, fps=24, bitrate=2_000_000, gop_s=2.0, nvenc=True, threads=4, quality=80):
    if codec == "jpeg":
        return JpegEncoder(width, height, fps, quality=quality)
    if codec != "h264":
        raise ValueError(f"codec must be h264 or jpeg, not {codec!r}")
    prefer = H264_ORDER if nvenc else ("libx264",)
    return H264Encoder(width, height, fps, bitrate, gop_s, prefer=prefer, threads=threads)


if __name__ == "__main__":       # quick self-check: python -m avatar.encoder
    import json
    for codec in ("h264", "jpeg"):
        e = open_encoder(codec, 1280, 720)
        img = np.zeros((720, 1280, 3), np.uint8)
        ts = []
        for j in range(30):
            img[:, :, 0] = (j * 8) % 256
            t = time.perf_counter()
            out = e.encode(img, j, force_key=(j == 20))
            ts.append(time.perf_counter() - t)
        print(json.dumps({"codec": codec, "encoder": e.name, "codec_string": e.codec_string, "errors": e.errors,
                          "ms_mean": round(1000 * float(np.mean(ts[3:])), 2), "last_key": out[-1][2]}))
        e.close()
