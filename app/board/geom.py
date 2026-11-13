"""boardbrain: small polyline geometry helpers (pure Python)."""
from __future__ import annotations

import math


def seglen(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def length(pts):
    return sum(seglen(pts[i - 1], pts[i]) for i in range(1, len(pts)))


def resample(pts, step):
    """Resample a polyline to roughly uniform spacing `step`, keeping every original vertex (corners stay sharp)."""
    if len(pts) < 2 or step <= 0:
        return list(pts)
    out = [pts[0]]
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        L = seglen(a, b)
        n = int(L / step)
        for k in range(1, n + 1):
            t = k / (n + 1)
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        out.append(b)
    return out


def bow(pts, amp, rnd):
    """Add a smooth low-frequency sideways bend (hand-drawn lines are never perfectly straight)."""
    L = length(pts)
    if L <= 0:
        return pts
    ph = rnd.uniform(0, math.pi * 2)
    a = rnd.uniform(-1, 1) * amp
    out = []
    acc = 0.0
    for i, p in enumerate(pts):
        if i:
            acc += seglen(pts[i - 1], p)
        s = acc / L
        # direction normal from neighbours
        q0 = pts[max(0, i - 1)]
        q1 = pts[min(len(pts) - 1, i + 1)]
        dx, dy = q1[0] - q0[0], q1[1] - q0[1]
        n = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / n, dx / n
        off = a * math.sin(math.pi * s) + 0.35 * a * math.sin(2 * math.pi * s + ph)
        out.append((p[0] + nx * off, p[1] + ny * off))
    return out


def bbox(strokes):
    xs = [p[0] for s in strokes for p in s]
    ys = [p[1] for s in strokes for p in s]
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


class TStroke(list):
    """A polyline that remembers which glyph it belongs to: .ch = (char, serial)."""
    __slots__ = ("ch",)


def retag(new, old):
    if isinstance(old, TStroke):
        t = TStroke(new)
        t.ch = old.ch
        return t
    return new


def transform(strokes, sx=1.0, sy=1.0, dx=0.0, dy=0.0):
    return [retag([(x * sx + dx, y * sy + dy) for x, y in s], s) for s in strokes]


def bezier(p0, p1, p2, p3, n=24):
    out = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        out.append((mt ** 3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t ** 3 * p3[0],
                    mt ** 3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t ** 3 * p3[1]))
    return out


def quad(p0, p1, p2, n=20):
    out = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        out.append((mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0], mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]))
    return out


def ellipse(cx, cy, rx, ry, a0=0.0, a1=360.0, n=None):
    """Points on an ellipse arc, angles in degrees (y axis as given; caller decides orientation)."""
    span = abs(a1 - a0)
    if n is None:
        n = max(8, int(span / 360.0 * 48 * max(1.0, min(3.0, (rx + ry) * 6))))
    return [(cx + rx * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cy + ry * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


def arrowhead(tip, frm, size, ang=28):
    """Two-stroke-free arrow head: polyline wing-tip-wing for a shaft arriving at `tip` from `frm`."""
    dx, dy = tip[0] - frm[0], tip[1] - frm[1]
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    a = math.radians(ang)
    ca, sa = math.cos(a), math.sin(a)
    w1 = (tip[0] - size * (ux * ca - uy * sa), tip[1] - size * (uy * ca + ux * sa))
    w2 = (tip[0] - size * (ux * ca + uy * sa), tip[1] - size * (uy * ca - ux * sa))
    return [w1, tip, w2]


def dashed(pts, dash, gap):
    """Split a polyline into dash strokes."""
    out, cur = [], []
    on = True
    left = dash
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        L = seglen(a, b)
        pos = 0.0
        if on and not cur:
            cur = [a]
        while L - pos > left:
            pos += left
            t = pos / L
            p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            if on:
                cur.append(p)
                out.append(cur)
                cur = []
                on, left = False, gap
            else:
                cur = [p]
                on, left = True, dash
        left -= (L - pos)
        if on:
            cur.append(b)
    if on and len(cur) > 1:
        out.append(cur)
    return out


def rects_overlap(a, b, gap=0.0):
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1])
