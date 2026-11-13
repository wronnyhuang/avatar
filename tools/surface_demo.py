#!/usr/bin/env python3
"""Scene: drive a running server's /debug/surface door (the scene contract in docs/, section "Test hooks").

    python tools/surface_demo.py --url 'http://127.0.0.1:PORT/?k=KEY' [--tv-seconds 6] [--erase] [--no-board]

Clears the board and the TV, writes "2+2=4" on the board as single-stroke chalk polylines over ~4 s (every segment
stamped with t_rel so it appears on the playback timeline as the 'hand' gets there), then plays a moving test
pattern on the TV at ~10 fps for --tv-seconds, then (--erase) wipes the writing with a few eraser discs and turns the
TV off. Stdlib only. The server must have a live session (404 otherwise); surfaces reach the renderer for the muse only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# single-stroke font: glyph -> list of strokes, points in a unit box (x 0..0.6 across, y 0..1 down)
FONT = {
    "0": [[(0.3, 0), (0.05, 0.15), (0, 0.5), (0.05, 0.85), (0.3, 1), (0.55, 0.85), (0.6, 0.5), (0.55, 0.15), (0.3, 0)]],
    "1": [[(0.15, 0.2), (0.35, 0), (0.35, 1)], [(0.1, 1), (0.55, 1)]],
    "2": [[(0.02, 0.2), (0.15, 0.03), (0.35, 0), (0.55, 0.12), (0.58, 0.3), (0.45, 0.5), (0, 1), (0.6, 1)]],
    "3": [[(0.03, 0.08), (0.3, 0), (0.55, 0.12), (0.55, 0.35), (0.25, 0.48), (0.58, 0.62), (0.58, 0.88), (0.3, 1),
           (0.02, 0.92)]],
    "4": [[(0.45, 1), (0.45, 0), (0, 0.7), (0.6, 0.7)]],
    "5": [[(0.55, 0), (0.08, 0), (0.03, 0.45), (0.3, 0.38), (0.55, 0.5), (0.58, 0.8), (0.35, 1), (0.02, 0.92)]],
    "6": [[(0.5, 0.03), (0.2, 0.15), (0.02, 0.55), (0.1, 0.92), (0.33, 1), (0.55, 0.85), (0.55, 0.6), (0.3, 0.48),
           (0.05, 0.6)]],
    "7": [[(0, 0), (0.6, 0), (0.2, 1)]],
    "8": [[(0.3, 0.48), (0.08, 0.32), (0.1, 0.08), (0.3, 0), (0.5, 0.08), (0.52, 0.32), (0.3, 0.48), (0.03, 0.68),
           (0.08, 0.92), (0.3, 1), (0.52, 0.92), (0.57, 0.68), (0.3, 0.48)]],
    "9": [[(0.55, 0.4), (0.3, 0.52), (0.05, 0.38), (0.08, 0.1), (0.3, 0), (0.55, 0.12), (0.55, 0.5), (0.45, 0.85),
           (0.15, 1)]],
    "+": [[(0.3, 0.25), (0.3, 0.75)], [(0.05, 0.5), (0.55, 0.5)]],
    "-": [[(0.05, 0.5), (0.55, 0.5)]],
    "=": [[(0.05, 0.38), (0.55, 0.38)], [(0.05, 0.62), (0.55, 0.62)]],
    "x": [[(0.08, 0.3), (0.52, 0.8)], [(0.52, 0.3), (0.08, 0.8)]],
}


def layout(text: str, u0: float, v0: float, height: float, aspect: float = 2.0):
    """-> list of strokes in board uv. aspect = canvas w/h (board 2:1): 1 v unit = 1/aspect u units."""
    strokes, x = [], u0
    for ch in text:
        if ch == " ":
            x += 0.4 * height / aspect
            continue
        for st in FONT[ch]:
            strokes.append([(x + px * height / aspect, v0 + py * height) for px, py in st])
        x += 0.85 * height / aspect
    return strokes


class Door:
    def __init__(self, url: str):
        p = urllib.parse.urlsplit(url)
        k = urllib.parse.parse_qs(p.query).get("k", [""])[0]
        if not k:
            raise SystemExit("--url must carry ?k=<key>")
        self.url = urllib.parse.urlunsplit((p.scheme, p.netloc, "/debug/surface", "k=" + urllib.parse.quote(k), ""))

    def __call__(self, **body):
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                     headers={"content-type": "application/json", "x-requested-with": "avatar-live"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise SystemExit(f"{body.get('op')}: HTTP {e.code} {e.read()[:200]!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--text", default="2+2=4")
    ap.add_argument("--write-seconds", type=float, default=4.0)
    ap.add_argument("--tv-seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--erase", action="store_true")
    ap.add_argument("--no-board", action="store_true")
    a = ap.parse_args()
    door = Door(a.url)
    print("clear board", door(surface="board", op="clear"))
    print("clear tv", door(surface="tv", op="clear"))
    if not a.no_board:
        strokes = layout(a.text, 0.08, 0.12, 0.09)
        # one segment = one post, spread over write-seconds by segment length; t_rel from the first post
        segs = [(st[i], st[i + 1]) for st in strokes for i in range(len(st) - 1)]
        lens = [((q[0] - p[0]) * 2) ** 2 + (q[1] - p[1]) ** 2 for p, q in segs]
        lens = [x ** 0.5 for x in lens]
        total = sum(lens) or 1.0
        t_rel, t_start, n_bytes = 0.3, time.monotonic(), 0
        for (p, q), L in zip(segs, lens):
            t_rel += a.write_seconds * L / total
            r = door(surface="board", op="draw_polyline", pts=[list(p), list(q)], width=0.012,
                     t_rel=round(t_rel - (time.monotonic() - t_start), 4))
            n_bytes += r.get("bytes", 0)
        print(f"board: {len(segs)} segments, {n_bytes} bytes, attached={r.get('attached')} "
              f"character={r.get('character')} playback_t={r.get('playback_t')}")
        time.sleep(max(0.0, a.write_seconds + 0.3 - (time.monotonic() - t_start)))
    n = int(a.tv_seconds * a.fps)
    t0, n_bytes = time.monotonic(), 0
    for i in range(n):
        r = door(surface="tv", op="put_frame_test", phase=i / max(1.0, a.fps * 3))
        n_bytes += r.get("bytes", 0)
        time.sleep(max(0.0, t0 + (i + 1) / a.fps - time.monotonic()))
    if n:
        print(f"tv: {n} frames, {n_bytes} bytes ({n_bytes / max(1, n) / 1024:.1f} KB/frame)")
    if a.erase:
        for k in range(8):
            door(surface="board", op="erase_disc", center=[0.1 + k * 0.04, 0.165], r=0.07, t_rel=0.15 * k)
        door(surface="tv", op="clear", t_rel=1.3)
        print("erased")
    return 0


if __name__ == "__main__":
    sys.exit(main())
