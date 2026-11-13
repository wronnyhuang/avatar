"""boardbrain: minimal reader for Hershey .jhf single-stroke fonts (glyph data public domain,
files vendored from the Hershey-Fonts 2.1.0 package, see fonts/README).

A glyph is (left, right, strokes) in Hershey units, y DOWN, origin at the glyph centre line; the cap line is
about -12 and the baseline +9 for the roman fonts."""
from __future__ import annotations

import os
from functools import lru_cache

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


def _parse_line(line: str):
    # columns: 0-4 id, 5-7 vertex count, then pairs; ' R' = pen up
    body = line[8:].rstrip("\n")
    if len(body) < 2:
        return None
    left = ord(body[0]) - ord("R")
    right = ord(body[1]) - ord("R")
    strokes, cur = [], []
    i = 2
    while i + 1 < len(body) + 1:
        pair = body[i:i + 2]
        if len(pair) < 2:
            break
        if pair == " R":
            if cur:
                strokes.append(cur)
            cur = []
        else:
            cur.append((ord(pair[0]) - ord("R"), ord(pair[1]) - ord("R")))
        i += 2
    if cur:
        strokes.append(cur)
    return left, right, strokes


@lru_cache(maxsize=None)
def load_font(name: str) -> dict:
    """Return {ascii_char: (left, right, strokes)} for a vendored font (first glyph = space, 32)."""
    path = os.path.join(FONT_DIR, name + ".jhf")
    glyphs = {}
    code = 32
    pending = ""
    with open(path, encoding="latin-1") as f:
        lines = f.read().split("\n")
    for raw in lines:
        if raw.startswith("#") or not raw.strip():
            continue
        # jhf lines can wrap; the vertex count tells the true length
        if pending:
            raw = pending + raw
            pending = ""
        try:
            n = int(raw[5:8])
        except ValueError:
            continue
        need = 8 + 2 * n
        if len(raw) < need:
            pending = raw
            continue
        g = _parse_line(raw[:need])
        if g is not None:
            glyphs[chr(code)] = g
        code += 1
    return glyphs
