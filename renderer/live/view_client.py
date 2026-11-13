#!/usr/bin/env python3
"""A headless viewer of the live page's /stream: logs every message's stream time and its arrival here, so the wire-level
A/V alignment can be checked without a browser.  For a frame and an audio chunk with the same stream time, the loop
hands both out at T0 + t + D, so (arrival - t) should be one constant for both kinds; its spread is the delivery jitter.

    python -m live.view_client --url http://127.0.0.1:8765/ --out client.json [--seconds 60]
"""
from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
from urllib.parse import urlsplit

import numpy as np

from live.http_view import AUDIO, FRAME, HDR, HELLO


def run(url, seconds=0.0, connect_timeout=30.0):
    u = urlsplit(url)
    t_end = time.monotonic() + connect_timeout
    while True:
        try:
            c = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
            c.request("GET", u.path.rstrip("/") + "/stream")
            r = c.getresponse()
            break
        except OSError:
            if time.monotonic() > t_end:
                raise
            time.sleep(0.2)
    buf = b""
    rows, hello = [], None
    stop = time.monotonic() + seconds if seconds else None
    while True:
        try:
            b = r.read1(65536)
        except (OSError, http.client.HTTPException):
            break
        if not b:
            break
        now = time.monotonic()
        buf += b
        while len(buf) >= HDR.size:
            typ, n, t = HDR.unpack_from(buf)
            if len(buf) < HDR.size + n:
                break
            payload = buf[HDR.size:HDR.size + n]
            buf = buf[HDR.size + n:]
            if typ == HELLO:
                hello = json.loads(payload)
            else:
                rows.append((typ, t, n, now))
        if stop and now > stop:
            break
    return hello, rows


def summarize(hello, rows):
    f = np.array([(t, a) for typ, t, n, a in rows if typ == FRAME])
    au = np.array([(t, a, n) for typ, t, n, a in rows if typ == AUDIO])
    out = {"hello": hello, "frames": int(len(f)), "audio_chunks": int(len(au))}
    if len(f) and len(au):
        sr = (hello or {}).get("sample_rate", 24000)
        off_f, off_a = f[:, 1] - f[:, 0], au[:, 1] - au[:, 0]
        base = float(np.median(off_a))
        out.update(audio_s=round(float(au[:, 2].sum() / 2 / sr), 3),
                   frame_offset_ms={"median": round(1000 * (np.median(off_f) - base), 2),
                                    "p95": round(1000 * (np.percentile(off_f, 95) - base), 2),
                                    "min": round(1000 * (off_f.min() - base), 2), "max": round(1000 * (off_f.max() - base), 2)},
                   audio_offset_ms={"p5": round(1000 * (np.percentile(off_a, 5) - base), 2),
                                    "p95": round(1000 * (np.percentile(off_a, 95) - base), 2),
                                    "max": round(1000 * (off_a.max() - base), 2)},
                   frame_stream_fps=round(float((len(f) - 1) / (f[-1, 0] - f[0, 0])), 2) if len(f) > 1 else None,
                   note="offsets are arrival - stream time, relative to the median audio offset: a frame and a sample "
                        "of the same stream time arrive this far apart (frames carry their JPEG encode)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765/")
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    hello, rows = run(a.url, a.seconds)
    s = summarize(hello, rows)
    if a.out:
        json.dump(s, open(a.out, "w"), indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    sys.exit(main())
