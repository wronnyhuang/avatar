#!/usr/bin/env python3
"""Board brain: raster preview of a board plan.

  board_preview.py --items plan.json --out board.png [--gif reveal.gif] [--mp4 reveal.mp4]
  board_preview.py --request "show how to add 1/2 and 1/3" [--model haiku] --out board.png [--gif ...]

plan.json = a list of S3 items (or {"items": [...]}). Renders the writable area at the board's pixel size
(1024x512, slate colour, chalk strokes ~0.012 v wide, erase items wipe with discs of erase_r), and optionally the
progressive reveal along the same time model TEACH uses (board.timing.time_parameterize).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))

from PIL import Image, ImageDraw  # noqa: E402

from board.timing import time_parameterize, drawn_until  # noqa: E402

W, H = 1024, 512
SLATE = (44, 58, 52)
CHALK = (236, 236, 226)
WIDTH_V = 0.011


def _px(p, w, h):
    return (p[0] * w, p[1] * h)


def draw_item(d, it, w=W, h=H, strokes=None, chalk=CHALK):
    lw = max(1, int(round(WIDTH_V * h)))
    strokes = it.get("strokes", []) if strokes is None else strokes
    if it.get("kind") == "erase":
        ew, eh = it.get("erase_wh") or (0.1, 0.084)
        rx, ry = ew * w / 2, eh * h / 2
        for s in strokes:
            for p in s:
                x, y = _px(p, w, h)
                d.rectangle((x - rx, y - ry, x + rx, y + ry), fill=SLATE)
        return
    for s in strokes:
        if len(s) < 2:
            continue
        d.line([_px(p, w, h) for p in s], fill=chalk, width=lw, joint="curve")
        for p in (s[0], s[-1]):
            x, y = _px(p, w, h)
            rr = lw / 2
            d.ellipse((x - rr, y - rr, x + rr, y + rr), fill=chalk)


REACH_V = 0.321          # SCENE station.reach_uv_y[0] — the chalk cannot reach above this
OUT_OF_REACH = (34, 45, 40)


def _slate(w, h):
    im = Image.new("RGB", (w, h), SLATE)
    d = ImageDraw.Draw(im)
    if REACH_V > 0:
        d.rectangle((0, 0, w, int(REACH_V * h)), fill=OUT_OF_REACH)
    return im, d


def render(items, w=W, h=H, boxes=False):
    im, d = _slate(w, h)
    for it in items:
        draw_item(d, it, w, h)
        if boxes and it.get("bbox") and it.get("kind") != "erase":
            b = it["bbox"]
            d.rectangle((b[0] * w, b[1] * h, b[2] * w, b[3] * h), outline=(90, 120, 160))
    return im


def render_pages(items, w=W, h=H):
    """Board snapshots just before each erase that wipes ink, plus the final board -> list of PIL images."""
    pages = []
    im, d = _slate(w, h)
    inked = False
    for it in items:
        if it.get("kind") == "erase" and inked:
            pages.append(im.copy())
        draw_item(d, it, w, h)
        if it.get("strokes") and it.get("kind") != "erase":
            inked = True
    pages.append(im)
    return pages


def storyboard(pages, w=W, h=H):
    """Stack pages vertically with a thin label bar ("1/3 before erase", ..., "final")."""
    bar = 22
    out = Image.new("RGB", (w, (h + bar) * len(pages)), (0, 0, 0))
    d = ImageDraw.Draw(out)
    for i, p in enumerate(pages):
        y = i * (h + bar)
        lab = "final board" if i == len(pages) - 1 else "board %d of %d (just before the teacher erased to make room)" % (i + 1, len(pages))
        d.text((6, y + 5), lab, fill=(255, 220, 0))
        out.paste(p, (0, y + bar))
    return out


def reveal_frames(items, fps=8, w=512, h=256, aspect=2.0, pause_s=0.6):
    """Frames of the progressive reveal (each item written along its timeline, a pause between items)."""
    frames = []
    base, _ = _slate(w, h)
    for it in items:
        st = it.get("strokes") or []
        if not st:
            for _ in range(int(pause_s * fps)):
                frames.append(base.copy())
            continue
        tl = time_parameterize(st, aspect)
        T = tl[-1][0] if tl else 0
        n = max(1, int(T * fps))
        for k in range(1, n + 1):
            t = T * k / n
            im = base.copy()
            dd = ImageDraw.Draw(im)
            if it.get("kind") == "erase":
                pts = [[(u, v) for (_, u, v, _) in tl if _ <= t]]
                draw_item(dd, it, w, h, strokes=pts)
            else:
                draw_item(dd, it, w, h, strokes=drawn_until(st, tl, t))
            frames.append(im)
        dd = ImageDraw.Draw(base)
        draw_item(dd, it, w, h)
        for _ in range(int(pause_s * fps)):
            frames.append(base.copy())
    return frames


def save_gif(frames, path, fps=8):
    if not frames:
        return
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)


def save_mp4(frames, path, fps=8):
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(frames):
            f.save(os.path.join(td, "f%05d.png" % i))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(td, "f%05d.png"),
                        "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", path], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items")
    ap.add_argument("--request")
    ap.add_argument("--context", default="[]", help="JSON list of turns")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gif")
    ap.add_argument("--mp4")
    ap.add_argument("--boxes", action="store_true")
    a = ap.parse_args()
    if a.items:
        data = json.load(open(a.items))
        items = data["items"] if isinstance(data, dict) else data
    else:
        from board_brain import plan
        stats = {}
        items = asyncio.run(plan(json.loads(a.context), a.request, None, model=a.model, stats=stats))
        print(json.dumps({k: v for k, v in stats.items() if k != "board_state"}, indent=1))
    render(items, boxes=a.boxes).save(a.out)
    if a.gif or a.mp4:
        fr = reveal_frames(items)
        if a.gif:
            save_gif(fr, a.gif)
        if a.mp4:
            save_mp4(fr, a.mp4)
    print(a.out)


if __name__ == "__main__":
    main()
