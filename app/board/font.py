"""boardbrain: single-stroke handwriting font for the chalkboard.

Glyph source: Hershey Simplex (futural) for Latin/digits/punctuation, Hershey greek + mathlow for Greek and math
symbols (public domain), plus a few glyphs synthesized here (theta, approx, equilibrium arrows, prime, integral,
long arrows). Every glyph is returned in EM units: cap height = 1.0, baseline at y=0, y DOWN (cap top = -1),
advance starts at x=0. `HandFont.run()` lays out a string with seeded natural variation (slant, baseline wobble,
size jitter, bowed strokes) so it reads as handwriting rather than plotter output.
"""
from __future__ import annotations

import math
import random
import unicodedata
from functools import lru_cache

from .hershey import load_font
from . import geom

CAP = 21.0       # Hershey cap line (-12) to baseline (+9)
BASE = 9.0

# Unicode -> (font, ascii char)
_MAP: dict[str, tuple[str, str]] = {}
for _c in ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
           "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "):
    _MAP[_c] = ("futural", _c)
_GREEK = {
    "α": "a", "β": "b", "χ": "c", "δ": "d", "ε": "e", "φ": "f", "γ": "g", "η": "h", "ι": "i", "κ": "k",
    "λ": "l", "μ": "m", "ν": "n", "ο": "o", "π": "p", "ϑ": "q", "ρ": "r", "σ": "s", "τ": "t", "υ": "u",
    "ω": "w", "ξ": "x", "ψ": "y", "ζ": "z", "ϕ": "f", "ϵ": "e",
    "Α": "A", "Β": "B", "Χ": "C", "Δ": "D", "Ε": "E", "Φ": "F", "Γ": "G", "Η": "H", "Ι": "I", "Κ": "K",
    "Λ": "L", "Μ": "M", "Ν": "N", "Ο": "O", "Π": "P", "Θ": "Q", "Ρ": "R", "Σ": "S", "Τ": "T", "Υ": "U",
    "Ω": "W", "Ξ": "X", "Ψ": "Y", "Ζ": "Z", "µ": "m",
}
for _k, _v in _GREEK.items():
    _MAP[_k] = ("greek", _v)
_MATH = {
    "±": "!", "∓": "\"", "×": "#", "·": "$", "⋅": "$", "≦": "&", "≧": "'", "−": "-", "≠": "?", "≡": "@",
    "∝": "^", "∞": "_", "⊂": "d", "∪": "e", "⊃": "f", "∩": "g", "∈": "h", "↑": "j", "↓": "l",
    "∂": "m", "∇": "n", "∃": "v", "÷": "x", "∥": "y", "⊥": "z", "∠": "|", "∴": "~", "∼": "\x7f",
}
for _k, _v in _MATH.items():
    _MAP[_k] = ("mathlow", _v)

# characters drawn by code (em units, y down, baseline 0, cap -1); value = (advance, strokes)
_SYN: dict[str, tuple[float, list]] = {}


def _ell(cx, cy, rx, ry, a0=0.0, a1=360.0, n=24):
    out = []
    for i in range(n + 1):
        a = math.radians(a0 + (a1 - a0) * i / n)
        out.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
    return out


def _arrow(x0, x1, y, head=0.22, both=False):
    s = [[(x0, y), (x1, y)], [(x1 - head, y - head * 0.8), (x1, y), (x1 - head, y + head * 0.8)]]
    if both:
        s.append([(x0 + head, y - head * 0.8), (x0, y), (x0 + head, y + head * 0.8)])
    return s


def _build_syn():
    ax = -0.43  # math axis (minus sign height) in em
    _SYN["θ"] = (0.78, [_ell(0.38, -0.5, 0.26, 0.5, -90, 270), [(0.14, -0.5), (0.62, -0.5)]])
    _SYN["≈"] = (1.05, [[(0.15 + 0.75 * i / 12, ax - 0.13 - 0.09 * math.sin(2 * math.pi * i / 12)) for i in range(13)],
                        [(0.15 + 0.75 * i / 12, ax + 0.13 - 0.09 * math.sin(2 * math.pi * i / 12)) for i in range(13)]])
    _SYN["→"] = (1.25, _arrow(0.12, 1.1, ax))
    _SYN["←"] = (1.25, [[(1.1, ax), (0.12, ax)], [(0.34, ax - 0.18), (0.12, ax), (0.34, ax + 0.18)]])
    _SYN["↔"] = (1.35, _arrow(0.12, 1.2, ax, both=True))
    _SYN["⇒"] = (1.3, [[(0.12, ax - 0.1), (1.05, ax - 0.1)], [(0.12, ax + 0.1), (1.05, ax + 0.1)],
                       [(0.85, ax - 0.3), (1.15, ax), (0.85, ax + 0.3)]])
    _SYN["⇔"] = (1.4, [[(0.25, ax - 0.1), (1.1, ax - 0.1)], [(0.25, ax + 0.1), (1.1, ax + 0.1)],
                       [(0.95, ax - 0.3), (1.25, ax), (0.95, ax + 0.3)], [(0.4, ax - 0.3), (0.1, ax), (0.4, ax + 0.3)]])
    # equilibrium: half-headed arrows top (->) and bottom (<-)
    _SYN["⇌"] = (1.35, [[(0.12, ax - 0.12), (1.2, ax - 0.12), (0.98, ax - 0.32)],
                        [(1.2, ax + 0.12), (0.12, ax + 0.12), (0.34, ax + 0.32)]])
    _SYN["′"] = (0.28, [[(0.2, -1.05), (0.1, -0.72)]])
    _SYN["″"] = (0.42, [[(0.2, -1.05), (0.1, -0.72)], [(0.36, -1.05), (0.26, -0.72)]])
    _SYN["°"] = (0.45, [_ell(0.22, -0.88, 0.13, 0.13, 0, 360, 14)])
    _SYN["∫"] = (0.6, [[(0.62, -1.15), (0.5, -1.22), (0.4, -1.12), (0.33, -0.6), (0.27, 0.0), (0.21, 0.18),
                        (0.08, 0.2), (0.02, 0.12)]])
    _SYN["√"] = (0.95, [[(0.05, -0.45), (0.18, -0.52), (0.38, 0.05), (0.7, -1.15), (0.95, -1.15)]])
    _SYN["…"] = (1.0, [_ell(x, -0.03, 0.035, 0.035, 0, 360, 6) for x in (0.18, 0.5, 0.82)])
    _SYN["⋯"] = (1.0, [_ell(x, ax, 0.035, 0.035, 0, 360, 6) for x in (0.18, 0.5, 0.82)])
    _SYN["∑"] = (0.9, [[(0.8, -0.82), (0.8, -1.0), (0.08, -1.0), (0.45, -0.5), (0.08, 0.0), (0.8, 0.0), (0.8, -0.18)]])
    _SYN["∏"] = (0.95, [[(0.0, -1.0), (0.95, -1.0)], [(0.2, -1.0), (0.2, 0.0)], [(0.75, -1.0), (0.75, 0.0)]])
    _SYN["≅"] = (1.05, [[(0.15 + 0.75 * i / 12, ax - 0.25 - 0.08 * math.sin(2 * math.pi * i / 12)) for i in range(13)],
                        [(0.15, ax), (0.9, ax)], [(0.15, ax + 0.2), (0.9, ax + 0.2)]])
    _SYN["≤"] = (1.0, [[(0.85, ax - 0.42), (0.15, ax - 0.08), (0.85, ax + 0.26)], [(0.15, ax + 0.42), (0.85, ax + 0.42)]])
    _SYN["≥"] = (1.0, [[(0.15, ax - 0.42), (0.85, ax - 0.08), (0.15, ax + 0.26)], [(0.15, ax + 0.42), (0.85, ax + 0.42)]])
    _SYN["⩽"] = _SYN["≤"]
    _SYN["⩾"] = _SYN["≥"]
    _SYN["∘"] = (0.5, [_ell(0.25, ax, 0.12, 0.12, 0, 360, 12)])
    _SYN["✓"] = (0.9, [[(0.1, -0.45), (0.35, -0.1), (0.85, -1.0)]])
    _SYN["∅"] = (0.85, [_ell(0.42, -0.5, 0.32, 0.45, 0, 360, 20), [(0.12, 0.05), (0.75, -1.05)]])
    _SYN["⟨"] = (0.45, [[(0.38, -1.1), (0.1, -0.45), (0.38, 0.2)]])
    _SYN["⟩"] = (0.45, [[(0.07, -1.1), (0.35, -0.45), (0.07, 0.2)]])
    _SYN["∀"] = (0.9, [[(0.05, -1.0), (0.45, 0.0), (0.85, -1.0)], [(0.22, -0.55), (0.68, -0.55)]])
    _SYN["¬"] = (0.8, [[(0.1, ax), (0.7, ax), (0.7, ax + 0.25)]])
    _SYN["∧"] = (0.8, [[(0.1, 0.0), (0.4, -0.75), (0.7, 0.0)]])
    _SYN["∨"] = (0.8, [[(0.1, -0.75), (0.4, 0.0), (0.7, -0.75)]])
    _SYN["⊆"] = (0.95, [[(0.85, -0.95), (0.35, -0.95)] + _ell(0.35, -0.65, 0.25, 0.3, 270, 90, 10)[1:] + [(0.85, -0.35)],
                        [(0.1, -0.1), (0.85, -0.1)]])
    _SYN["ℓ"] = (0.55, [[(0.1, -0.15), (0.35, -0.6), (0.42, -0.95), (0.33, -1.05), (0.23, -0.9), (0.2, -0.3),
                         (0.28, -0.02), (0.45, 0.0)]])
    _SYN["ħ"] = (0.8, [[(0.2, -1.0), (0.2, 0.0)], [(0.2, -0.45), (0.35, -0.62), (0.55, -0.65), (0.65, -0.5), (0.65, 0.0)],
                       [(0.05, -0.8), (0.45, -0.9)]])
    _SYN["Å"] = (0.9, [[(0.05, 0.0), (0.45, -1.0), (0.85, 0.0)], [(0.2, -0.38), (0.7, -0.38)],
                       _ell(0.45, -1.2, 0.1, 0.1, 0, 360, 10)])


_build_syn()

# fallback: strip accents (é -> e) and draw an accent mark above
_ACCENTS = {"́": "acute", "̀": "grave", "̂": "circ", "̈": "uml", "̃": "tilde",
            "̧": "cedil", "̊": "ring", "̌": "caron"}


def _accent_strokes(kind, cx, top):
    y = top - 0.12
    if kind == "acute":
        return [[(cx - 0.08, y), (cx + 0.12, y - 0.22)]]
    if kind == "grave":
        return [[(cx + 0.08, y), (cx - 0.12, y - 0.22)]]
    if kind == "circ":
        return [[(cx - 0.15, y), (cx, y - 0.18), (cx + 0.15, y)]]
    if kind == "caron":
        return [[(cx - 0.15, y - 0.18), (cx, y), (cx + 0.15, y - 0.18)]]
    if kind == "uml":
        return [_ell(cx - 0.13, y - 0.06, 0.03, 0.03, 0, 360, 5), _ell(cx + 0.13, y - 0.06, 0.03, 0.03, 0, 360, 5)]
    if kind == "tilde":
        return [[(cx - 0.18 + 0.36 * i / 8, y - 0.08 - 0.06 * math.sin(2 * math.pi * i / 8)) for i in range(9)]]
    if kind == "macron":
        return [[(cx - 0.17, y - 0.06), (cx + 0.17, y - 0.06)]]
    if kind == "ring":
        return [_ell(cx, y - 0.08, 0.08, 0.08, 0, 360, 10)]
    if kind == "cedil":
        return [[(cx, 0.0), (cx + 0.08, 0.12), (cx - 0.06, 0.22)]]
    return []


@lru_cache(maxsize=4096)
def glyph(ch: str):
    """(advance_em, strokes_em, known) for one character."""
    if ch in _SYN:
        adv, st = _SYN[ch]
        return adv, [list(s) for s in st], True
    if ch in _MAP:
        font, c = _MAP[ch]
        g = load_font(font).get(c)
        if g:
            left, right, strokes = g
            adv = (right - left) / CAP
            st = [[((x - left) / CAP, (y - BASE) / CAP) for x, y in s] for s in strokes if len(s) >= 1]
            # a single-point stroke (a dot) becomes a tiny circle so it is traceable and visible
            st = [s if len(s) > 1 else _ell(s[0][0], s[0][1], 0.03, 0.03, 0, 360, 6) for s in st]
            return adv, st, True
    # accented Latin (é, ü, ñ ...): base letter + accent strokes
    d = unicodedata.normalize("NFD", ch)
    if len(d) >= 2 and d[0] in _MAP:
        adv, st, _ = glyph(d[0])
        top = -0.68 if d[0].islower() else -1.0
        if d[0] in "bdfhklt":
            top = -1.0
        if d[0] == "i":   # drop the dot
            st = [s for s in st if max(y for _, y in s) > -0.7]
        for m in d[1:]:
            st = st + _accent_strokes(_ACCENTS.get(m, ""), adv / 2, top)
        return adv, st, True
    if ch in "‐‑‒–—":
        return glyph("-")
    if ch in "‘’ʼ":
        return glyph("'")
    if ch in "“”":
        return glyph('"')
    if ch == " ":
        return glyph(" ")
    # unknown: a small open box (honest placeholder), flagged unknown
    return 0.8, [[(0.15, -0.8), (0.65, -0.8), (0.65, 0.0), (0.15, 0.0), (0.15, -0.8)]], False


def has_glyph(ch: str) -> bool:
    return glyph(ch)[2]


SPACE_EM = 0.45
TRACK_EM = 0.0   # extra letter spacing (chalk letters breathe a bit)
XSCALE = 0.86    # handwriting is narrower than the Hershey roman (more characters per line on the
                 # 2:1 board; cap height, i.e. legibility, unchanged)


class HandFont:
    """Seeded handwriting variation on top of the stroke glyphs."""
    serial = 0      # glyph instance counter (tags strokes so annotations can find a character)

    def __init__(self, seed: int = 0, slant: float = 0.10, wobble: float = 1.0):
        self.seed = seed
        self.slant = slant
        self.wobble = wobble

    def run(self, text: str, size: float, x: float = 0.0, y: float = 0.0, seed_offset: int = 0):
        """Lay out `text` at cap height `size` with baseline-left at (x, y) (same units as size, y down).

        Returns (strokes, advance, unknown_chars)."""
        rnd = random.Random((self.seed * 1000003 + seed_offset * 7919 + hash_text(text)) & 0xFFFFFFFF)
        strokes, unknown = [], []
        pen = x
        drift = 0.0
        for i, ch in enumerate(text):
            if ch == " ":
                pen += (SPACE_EM + rnd.uniform(-0.05, 0.05)) * size * XSCALE
                continue
            adv, st, ok = glyph(ch)
            if not ok:
                unknown.append(ch)
            w = self.wobble
            sc = size * (1.0 + w * rnd.uniform(-0.035, 0.035))
            drift = 0.7 * drift + w * rnd.uniform(-0.025, 0.025)
            dy = drift * size
            sl = self.slant + w * rnd.uniform(-0.035, 0.035)
            rot = w * rnd.uniform(-0.03, 0.03)
            cr, sr = math.cos(rot), math.sin(rot)
            for s in st:
                pts = []
                for gx, gy in s:
                    gx2 = gx * cr - gy * sr
                    gy2 = gx * sr + gy * cr
                    gx2 = gx2 - gy2 * sl
                    pts.append((pen + gx2 * sc * XSCALE, y + dy + gy2 * sc))
                pts = geom.resample(pts, max(size * 0.06, 1e-4))
                if w > 0 and len(pts) > 3:
                    pts = geom.bow(pts, amp=w * 0.018 * size, rnd=rnd)
                t = geom.TStroke(pts)
                t.ch = (ch, HandFont.serial)
                strokes.append(t)
            pen += (adv + TRACK_EM) * sc * XSCALE
            HandFont.serial += 1
        return strokes, pen - x, unknown

    @staticmethod
    def width(text: str, size: float) -> float:
        w = 0.0
        for ch in text:
            if ch == " ":
                w += SPACE_EM * size * XSCALE
            else:
                w += (glyph(ch)[0] + TRACK_EM) * size * XSCALE
        return w


def hash_text(s: str) -> int:
    h = 2166136261
    for ch in s:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h
