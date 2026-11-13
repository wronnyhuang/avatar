"""boardbrain: human-like chalk timing.

Distances are measured in BOARD HEIGHTS (u is scaled by the writable area's aspect, v as is). Each pen-down
stroke runs a trapezoidal speed profile (accelerate, cruise, decelerate) and also slows to a stop at sharp
corners (> CORNER_DEG); between strokes the hand travels in the air (faster, also eased) with a short lift/touch
pause. Small writing is slower per unit length than big writing (fine motor control).

    tl = time_parameterize(strokes_uv, aspect=2.0)   # [(t, u, v, pen_down), ...]
    est = tl[-1][0] + SETTLE_S

Semantics of the samples: they are waypoints in time order; the segment from sample i-1 to sample i is drawn
(chalk on the board) iff sample i has pen_down=True. The first sample of every stroke has pen_down=False (the
hand arrives there in the air), so `chalk_at(tl, t)` gives the tip position and whether it touches the board.
"""
from __future__ import annotations

import bisect
import math

# tunables (all in board heights / seconds); TEACH may scale everything with `speed`
CHALK_V = 0.9           # cruise speed while writing, at the reference letter size (~0.45 s per letter)
REF_SIZE = 0.09         # reference cap height (fraction of board height)
CHALK_A = 12.0           # acceleration while writing
AIR_V = 1.6             # pen-up travel cruise speed
AIR_A = 14.0
LIFT_S = 0.03          # lift + touch-down pause per stroke
CORNER_DEG = 65.0
SETTLE_S = 0.15


def _profile_times(cum, v, a):
    """Times at cumulative distances `cum` (monotonic, starting 0) for a trapezoid profile over cum[-1]."""
    L = cum[-1]
    if L <= 1e-12:
        return [0.0 for _ in cum]
    d_acc = v * v / (2 * a)
    if 2 * d_acc > L:          # triangular
        vp = math.sqrt(a * L)
        d_acc = L / 2
    else:
        vp = v
    t_acc = vp / a
    t_cruise = (L - 2 * d_acc) / vp
    out = []
    for s in cum:
        if s <= d_acc:
            t = math.sqrt(2 * s / a)
        elif s <= L - d_acc:
            t = t_acc + (s - d_acc) / vp
        else:
            r = max(0.0, L - s)
            t = 2 * t_acc + t_cruise - math.sqrt(2 * r / a)
        out.append(t)
    return out


def _dist(p, q, aspect):
    return math.hypot((q[0] - p[0]) * aspect, q[1] - p[1])


def _turn(a, b, c, aspect):
    v1 = ((b[0] - a[0]) * aspect, b[1] - a[1])
    v2 = ((c[0] - b[0]) * aspect, c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cs = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cs))


def stroke_size(strokes):
    """Rough letter size of a stroke set: median stroke extent (board heights)."""
    ext = []
    for s in strokes:
        if len(s) < 2:
            continue
        vs = [p[1] for p in s]
        ext.append(max(vs) - min(vs))
    if not ext:
        return REF_SIZE
    ext.sort()
    return max(0.02, ext[len(ext) // 2])


def time_parameterize(strokes, aspect: float = 2.0, t0: float = 0.0, speed: float = 1.0, start=None):
    """strokes: [[(u, v), ...], ...] in board uv -> [(t, u, v, pen_down)].

    `start` = where the chalk tip is before the first stroke (uv); default: the first point (no lead-in)."""
    out = []
    t = t0
    size = stroke_size(strokes)
    v_write = CHALK_V * speed * max(0.5, min(1.6, math.sqrt(size / REF_SIZE)))
    a_write = CHALK_A * speed * speed
    prev = tuple(start) if start is not None else None
    for s in strokes:
        if not s:
            continue
        p0 = (float(s[0][0]), float(s[0][1]))
        # air travel to the stroke start
        if prev is not None:
            D = _dist(prev, p0, aspect)
            if D > 1e-9:
                t += _profile_times([0.0, D], AIR_V * speed, AIR_A * speed * speed)[-1]
        t += LIFT_S
        out.append((t, p0[0], p0[1], False))
        if len(s) == 1:
            prev = p0
            continue
        # split at sharp corners; each piece gets its own accelerate/decelerate
        pts = [(float(p[0]), float(p[1])) for p in s]
        cuts = [0]
        for i in range(1, len(pts) - 1):
            if _turn(pts[i - 1], pts[i], pts[i + 1], aspect) > CORNER_DEG:
                cuts.append(i)
        cuts.append(len(pts) - 1)
        for a_i, b_i in zip(cuts[:-1], cuts[1:]):
            piece = pts[a_i:b_i + 1]
            cum = [0.0]
            for j in range(1, len(piece)):
                cum.append(cum[-1] + _dist(piece[j - 1], piece[j], aspect))
            ts = _profile_times(cum, v_write, a_write)
            base = t
            for j in range(1, len(piece)):
                tt = base + ts[j]
                if tt <= out[-1][0]:
                    tt = out[-1][0] + 1e-4
                out.append((tt, piece[j][0], piece[j][1], True))
            t = out[-1][0]
        t += LIFT_S
        prev = pts[-1]
    return out


def est_seconds(strokes, aspect: float = 2.0, speed: float = 1.0) -> float:
    tl = time_parameterize(strokes, aspect, speed=speed)
    if not tl:
        return 0.0
    return round(tl[-1][0] + LIFT_S + SETTLE_S, 3)


def chalk_at(tl, t):
    """(u, v, pen_down) of the chalk tip at time t along a timeline from time_parameterize."""
    if not tl:
        return None
    if t <= tl[0][0]:
        return (tl[0][1], tl[0][2], False)
    if t >= tl[-1][0]:
        return (tl[-1][1], tl[-1][2], False)
    ts = [x[0] for x in tl]
    i = bisect.bisect_right(ts, t)
    a, b = tl[i - 1], tl[i]
    f = (t - a[0]) / max(1e-9, b[0] - a[0])
    return (a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, b[3])


def drawn_until(strokes, tl, t):
    """The partial strokes visible at time t (for progressive previews): list of polylines."""
    out = []
    cur = None
    for k, (tt, u, v, down) in enumerate(tl):
        if tt > t:
            if down and cur is not None and k > 0:
                a = tl[k - 1]
                f = (t - a[0]) / max(1e-9, tt - a[0])
                cur.append((a[1] + (u - a[1]) * f, a[2] + (v - a[2]) * f))
            break
        if not down:
            if cur and len(cur) > 1:
                out.append(cur)
            cur = [(u, v)]
        else:
            if cur is None:
                cur = [(u, v)]
            else:
                cur.append((u, v))
    if cur and len(cur) > 1:
        out.append(cur)
    return out
