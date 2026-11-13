"""tvdsl.scene -- evaluate a Doc at a wall time into marks (world-space paths and texts), incl. generic optics tracing.

    fr = Frame(doc, now)            # evaluates params / lets for this instant
    marks = fr.marks()              # [Mark] in draw order
A Mark is ('path', ...) or ('text', ...); see Mark below.  Pure: the same doc + now give the same marks.
"""
import math
import re

from .expr import Env, NAN
from .lang import ColorSpec, Range, FADE_IN, FADE_OUT, MORPH, wavelength_rgb

DEF_COLOR = (0.93, 0.95, 0.98)
GLASS = (0.55, 0.8, 1.0)
CPK = {  # element -> (colour, radius in world units)
    "H": ((0.95, 0.95, 0.98), 0.32), "C": ((0.55, 0.58, 0.64), 0.48), "N": ((0.38, 0.52, 1.0), 0.48),
    "O": ((1.0, 0.33, 0.3), 0.48), "F": ((0.55, 0.95, 0.45), 0.42), "Cl": ((0.35, 0.95, 0.45), 0.6),
    "Br": ((0.75, 0.3, 0.2), 0.65), "I": ((0.6, 0.3, 0.9), 0.7), "S": ((1.0, 0.85, 0.25), 0.6),
    "P": ((1.0, 0.6, 0.2), 0.58), "Na": ((0.67, 0.4, 1.0), 0.66), "K": ((0.6, 0.3, 0.95), 0.75),
    "Li": ((0.8, 0.5, 1.0), 0.55), "Mg": ((0.4, 0.95, 0.3), 0.62), "Ca": ((0.35, 0.9, 0.3), 0.72),
    "Fe": ((0.95, 0.55, 0.25), 0.62), "Cu": ((0.95, 0.6, 0.35), 0.6), "Zn": ((0.6, 0.6, 0.75), 0.6),
    "He": ((0.85, 1.0, 1.0), 0.32), "Ne": ((0.7, 0.9, 1.0), 0.38), "Ar": ((0.5, 0.85, 0.95), 0.5),
    "Si": ((0.9, 0.75, 0.55), 0.6), "Al": ((0.75, 0.65, 0.7), 0.62), "Au": ((1.0, 0.82, 0.25), 0.62),
    "Ag": ((0.8, 0.8, 0.85), 0.62), "e": ((0.45, 0.85, 1.0), 0.12), "p": ((1.0, 0.4, 0.4), 0.2),
    "n": ((0.7, 0.72, 0.78), 0.2),
}


class Mark:
    __slots__ = ("kind", "pts", "closed", "stroke", "width", "fill", "alpha", "dash", "draw", "head", "text",
                 "size", "anchor", "layer", "order", "glow", "rot", "bold", "add", "soft", "side", "segs")

    def __init__(self, kind, **kw):
        self.kind = kind
        self.pts = kw.get("pts")
        self.closed = kw.get("closed", False)
        self.stroke = kw.get("stroke", DEF_COLOR)
        self.width = kw.get("width", 2.0)
        self.fill = kw.get("fill")
        self.alpha = kw.get("alpha", 1.0)
        self.dash = kw.get("dash", 0.0)
        self.draw = kw.get("draw", 1.0)
        self.head = kw.get("head", 0)          # 0 none, 1 end, 2 both
        self.text = kw.get("text")
        self.size = kw.get("size", 0.55)
        self.anchor = kw.get("anchor", "c")
        self.layer = kw.get("layer", 0.0)
        self.order = kw.get("order", 0)
        self.glow = kw.get("glow", 1.0)
        self.rot = kw.get("rot", 0.0)
        self.bold = kw.get("bold", False)
        self.add = kw.get("add", False)          # additive light (coloured rays overlap to white)
        self.soft = kw.get("soft", False)        # True: an automatic label the renderer may slide off lines; "tick": never moved, dropped if it would overlap a text
        #                                          "inline": exactly on its shape; "callout": placed by the renderer beside the picture + leader
        #                                          "axis": an axis title (placed after the axis numbers, avoids them)
        self.side = kw.get("side")               # callout side: l r t b (None = automatic)
        self.segs = kw.get("segs")               # extra polylines drawn with this mark in one pass (ripple rings)


def _ok(v):
    return isinstance(v, float) and math.isfinite(v)


def _okp(p):
    return isinstance(p, tuple) and len(p) == 2 and _ok(p[0]) and _ok(p[1])


def _blend(a, b, u):
    """morph between two evaluated property values of the same shape"""
    if isinstance(a, float) and isinstance(b, float):
        if not (_ok(a) and _ok(b)):
            return b
        return a + (b - a) * u
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return tuple(_blend(x, y, u) for x, y in zip(a, b))
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [_blend(x, y, u) for x, y in zip(a, b)]
    return b


def _smooth(u):
    u = 0.0 if u < 0 else 1.0 if u > 1 else u
    return u * u * (3 - 2 * u)


def _catmull(pts, closed, k=6):
    n = len(pts)
    if n < 3:
        return pts
    out = []
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        p0 = pts[(i - 1) % n] if (closed or i > 0) else pts[0]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n] if (closed or i + 2 < n) else pts[-1]
        for s in range(k):
            u = s / k
            u2, u3 = u * u, u * u * u
            out.append(tuple(0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * u + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * u2
                                    + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * u3) for j in (0, 1)))
    if not closed:
        out.append(pts[-1])
    return out


def _bezier(ctrl, n=48):
    out = []
    for s in range(n + 1):
        u = s / n
        pts = list(ctrl)
        while len(pts) > 1:
            pts = [(a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u) for a, b in zip(pts, pts[1:])]
        out.append(pts[0])
    return out


def _circle(c, rx, ry=None, a0=0.0, a1=360.0, n=None, rot=0.0):
    ry = rx if ry is None else ry
    if n is None:
        n = int(max(16, min(120, abs(a1 - a0) / 360.0 * 28 * max(1.0, max(abs(rx), abs(ry))) ** 0.5 * 2.2)))
    cr, sr = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    out = []
    for k in range(n + 1):
        a = math.radians(a0 + (a1 - a0) * k / n)
        x, y = rx * math.cos(a), ry * math.sin(a)
        out.append((c[0] + x * cr - y * sr, c[1] + x * sr + y * cr))
    return out


def path_length(pts):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def point_along(pts, u):
    if not pts:
        return (NAN, NAN)
    if len(pts) == 1:
        return pts[0]
    total = path_length(pts)
    if total <= 0:
        return pts[0]
    target = max(0.0, min(1.0, u)) * total
    acc = 0.0
    for a, b in zip(pts, pts[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if acc + L >= target and L > 0:
            k = (target - acc) / L
            return (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)
        acc += L
    return pts[-1]


def _inside(poly, p):
    x, y = p
    c = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / ((y2 - y1) or 1e-12)
            if x < xi:
                c = not c
    return c


def _seg_hit(o, d, a, b):
    """ray o + s d vs segment a-b -> (s, u) or None"""
    ex, ey = b[0] - a[0], b[1] - a[1]
    den = d[0] * ey - d[1] * ex
    if abs(den) < 1e-12:
        return None
    wx, wy = a[0] - o[0], a[1] - o[1]
    s = (wx * ey - wy * ex) / den
    u = (wx * d[1] - wy * d[0]) / den
    if s > 1e-7 and -1e-9 <= u <= 1 + 1e-9:
        return s, u
    return None


def _unit(v):
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n) if n else (1.0, 0.0)


# ---------------------------------------------------------------------------------------------------- the frame
class Frame:
    def __init__(self, doc, now, px_per_unit=32.0, width=512):
        self.doc = doc
        self.now = now
        self.ppu = px_per_unit
        # the renderer's legibility floor (12 px at 512 wide) in world units: builders size boxes / thin ticks with it
        self.fmin = 12.0 * (width / 512.0) / max(1e-6, px_per_unit)
        T = max(0.0, now - doc.t0)
        if hasattr(doc, "scene_t"):
            t = doc.scene_t(now)
        else:
            L = doc.loop_len() if hasattr(doc, "loop_len") else doc.loop
            t = T % L if L else T
        v = {"t": t, "T": T}
        self._t_wrapped, self._T_total = t, T          # one-shot traces/fades (see _once)
        sw = doc.sweep_param() if hasattr(doc, "sweep_param") else None
        for k, p in doc.params.items():
            v[k] = doc.sweep_value(k, now) if k == sw else p.value(now)
        self.base = v
        self.memo = {}
        self.busy = set()
        self.user_depth = 0
        self._optics = None
        x0, x1, y0, y1 = doc.view
        self.vb = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
        env = Env(v, self)
        lets = doc.ordered_lets() if hasattr(doc, "ordered_lets") else doc.lets
        for name, e in lets:                # (lets may call user functions / pos(): the hooks above exist first)
            val = e.eval(env)
            v[name] = val

    def _once(self, P, k, env, default):
        """a plain trace-in / fade-in (draw=lin(a,b), alpha=ramp(a,b)) happens ONCE: when the loop wraps,
        motion replays but finished traces stay drawn (every judge and the voice sessions flagged the 10 s 'rays vanish
        and re-trace' flicker). Only the literal one-shot forms; anything stage-dependent loops as written."""
        e = P.get(k)
        src = getattr(e, "src", None)
        T = getattr(self, "_T_total", 0.0)
        if src and T > getattr(self, "_t_wrapped", 0.0) + 1e-6 and _ONCE_RE.match(src):
            v = dict(env.vars)
            v["t"] = T
            return _num(P, k, Env(v, self), default)
        return _num(P, k, env, default)

    # ---- scene hooks for expressions
    def env_for(self, o):
        if not o.vars:
            return Env(self.base, self)
        v = dict(self.base)
        v.update(o.vars)
        return Env(v, self)

    def call_user(self, name, args, env):
        f = self.doc.funcs.get(name)
        if f is None or self.user_depth > 6:
            return NAN
        names, e = f
        v = dict(env.vars)
        for n, a in zip(names, args):
            v[n] = a
        self.user_depth += 1
        try:
            return e.eval(Env(v, self))
        finally:
            self.user_depth -= 1

    def _geom(self, ident):
        if ident is None:
            return None
        if ident in self.memo:
            return self.memo[ident]
        o = self.doc.objs.get(ident)
        if o is None or ident in self.busy:
            return None
        self.busy.add(ident)
        try:
            g = self._build(o)
        finally:
            self.busy.discard(ident)
        self.memo[ident] = g
        return g

    def ref_pos(self, ident):
        g = self._geom(ident)
        return g["anchor"] if g else (NAN, NAN)

    def ref_point(self, ident, u):
        g = self._geom(ident)
        if not g or not g["path"]:
            return (NAN, NAN)
        return point_along(g["path"], u)

    def ref_length(self, ident):
        g = self._geom(ident)
        return path_length(g["path"]) if g and g["path"] else NAN

    def ref_vertex(self, ident, k):
        g = self._geom(ident)
        if not g or not g["path"]:
            return (NAN, NAN)
        pts = g.get("verts") or g["path"]
        i = int(k) if _ok(k) else 0
        return pts[max(0, min(len(pts) - 1, i))]

    def ref_node(self, ident, k):
        """centre of step k of a flow (any object: its k-th vertex)"""
        g = self._geom(ident)
        if not g:
            return (NAN, NAN)
        pts = g.get("nodes") or g.get("verts") or g.get("path")
        if not pts:
            return (NAN, NAN)
        k = float(k) if isinstance(k, (int, float)) else NAN
        i = int(math.floor(k)) if _ok(k) else 0
        return pts[i % len(pts)] if g.get("nodes") else pts[max(0, min(len(pts) - 1, i))]

    def ref_radius(self, ident):
        g = self._geom(ident)
        return g.get("radius", 0.0) if g else 0.0

    # ---- property evaluation
    def _eval_props(self, o):
        env = self.env_for(o)
        P = {}
        for k, v in o.props.items():
            if isinstance(v, ColorSpec):
                P[k] = v.eval(env)
            elif isinstance(v, Range):
                P[k] = (v.a.eval(env), v.b.eval(env))
            elif isinstance(v, str):
                P[k] = v
            else:
                P[k] = v  # Expr: evaluated lazily by the builder (some are functions of x / s)
        return P, env

    def _build(self, o):
        P, env = self._eval_props(o)
        g = self._build_kind(o, P, env)
        if o.prev is not None and o.morph_t0 is not None and self.now - o.morph_t0 < MORPH:
            u = _smooth((self.now - o.morph_t0) / MORPH)
            Pp, envp = self._eval_props(o.prev)
            gp = self._build_kind(o.prev, Pp, envp)
            if gp is not None and g is not None:
                g = _morph_geom(gp, g, u)
        if g is None:
            return None
        # data coordinates of an axes box (on=AXES_ID)
        on = P.get("on")
        if isinstance(on, str) and on != o.id:
            mp = self._axes_map(on)
            if mp is not None:
                g = _map_geom(g, mp, rigid=o.kind in ("dot", "circle", "star", "atom", "text", "ellipse"))
        # group transform
        parent = P.get("in")
        if isinstance(parent, str) and parent != o.id:
            tr = self._group_tf(parent)
            if tr is not None:
                g = _transform_geom(g, tr)
        return g

    def _axes_map(self, aid):
        """(xr, yr, box) of an axes object with a box: data ranges and the world rectangle they fill"""
        o = self.doc.objs.get(aid)
        if o is None or o.kind not in ("axes", "grid"):
            return None
        P, env = self._eval_props(o)
        box = self._box(P, env)
        xr = P.get("x") if isinstance(P.get("x"), tuple) else None
        yr = P.get("y") if isinstance(P.get("y"), tuple) else None
        if box is None or xr is None or yr is None or not all(_ok(v) for v in (*xr, *yr)) or xr[0] == xr[1] or yr[0] == yr[1]:
            return None
        return (xr, yr, box)

    def _box(self, P, env):
        b = P.get("box")
        v = b.eval(env) if hasattr(b, "eval") else None
        if isinstance(v, list) and len(v) == 4 and all(_ok(x) for x in v):
            return tuple(v)
        x0, x1, y0, y1 = self.vb
        if not (isinstance(v, float) and v):
            # no box asked: still a data chart when its ranges cannot be world units (they would leave the screen)
            xr, yr = P.get("x"), P.get("y")
            if not (isinstance(xr, tuple) and isinstance(yr, tuple) and all(_ok(q) for q in (*xr, *yr))):
                return None
            at = _pval(P, "at", env, (0.0, 0.0)) if "at" in P else (0.0, 0.0)
            if not _okp(at):
                at = (0.0, 0.0)
            sx, sy = max(abs(x1 - x0), 1e-6), max(abs(y1 - y0), 1e-6)
            fits = (min(xr) >= x0 - 0.1 * sx and max(xr) <= x1 + 0.1 * sx and min(yr) >= y0 - 0.1 * sy and max(yr) <= y1 + 0.1 * sy)
            if fits:
                return None
        return (x0 + 1.4, x1 - 0.6, y0 + 0.9, y1 - 0.9)       # box=1: the view, inset for tick labels

    def _group_tf(self, gid, depth=0):
        o = self.doc.objs.get(gid)
        if o is None or o.kind != "group" or depth > 4:
            return None
        P, env = self._eval_props(o)
        at = _pval(P, "at", env, (0.0, 0.0))
        rot = _num(P, "rot", env, 0.0)
        sc = _num(P, "scale", env, 1.0)
        tf = (at, rot, sc)
        par = P.get("in")
        if isinstance(par, str) and par != gid:
            up = self._group_tf(par, depth + 1)
            if up is not None:
                return ("chain", tf, up)
        return tf

    # ---- the kinds
    def _build_kind(self, o, P, env):
        try:
            fn = getattr(self, "_k_" + o.kind, None)
            if fn is None:
                return None
            return fn(o, P, env)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError, IndexError):
            return None

    def _style(self, P, env, stroke=DEF_COLOR, width=2.0, fill=None):
        col = stroke
        for k in ("color", "c", "stroke"):
            if k in P:
                v = P[k]
                col = v if isinstance(v, tuple) or v is None else col
                break
        fl = fill
        if "fill" in P:
            v = P["fill"]
            fl = v if isinstance(v, tuple) else (col if hasattr(v, "eval") else None)
        return {
            "stroke": col,
            "width": max(0.5, min(14.0, _num(P, "width", env, _num(P, "w_px", env, width)))),
            "fill": fl,
            "alpha": max(0.0, min(1.0, self._once(P, "alpha", env, _num(P, "opacity", env, 1.0)))),
            "dash": max(0.0, min(40.0, _num(P, "dash", env, 0.0) * (6.0 if _num(P, "dash", env, 0.0) == 1 else 1.0)))
            if "dash" in P else (6.0 if "dashed" in P else 0.0),
            "draw": max(0.0, min(1.0, self._once(P, "draw", env, 1.0))),
            "layer": _num(P, "layer", env, 0.0),
            "glow": max(0.0, min(2.0, _num(P, "glow", env, 1.0))),
            "falpha": max(0.0, min(1.0, _num(P, "falpha", env, 0.35))),
        }

    def _path_marks(self, pts, st, closed=False, head=0, fill=None):
        if not pts:
            return []
        f = fill if fill is not None else st["fill"]
        return [Mark("path", pts=pts, closed=closed, stroke=st["stroke"], width=st["width"], fill=f,
                     alpha=st["alpha"], dash=st["dash"], draw=st["draw"], head=head, layer=st["layer"],
                     glow=st["glow"])]

    def _label(self, P, env, pos, st, default_off=(0.0, 0.42), ref=None, anchor="c"):
        lab = P.get("label")
        if not isinstance(lab, str) or not lab or not _okp(pos):
            return []
        off = _pval(P, "loff", env, default_off)
        size = _num(P, "lsize", env, 0.5)
        col = P.get("lcolor") if isinstance(P.get("lcolor"), tuple) else (st["stroke"] or DEF_COLOR)
        return [Mark("text", pts=[(pos[0] + off[0], pos[1] + off[1])], text=interp(lab, env), size=size, stroke=col,
                     alpha=st["alpha"] * (1.0 if st["draw"] >= 0.999 else _smooth(st["draw"] * 3 - 2)),
                     anchor=anchor if "loff" not in P else "c", layer=st["layer"] + 0.01, soft="loff" not in P)]

    def _k_line(self, o, P, env, head=0):
        a = _pval(P, "from", env, None)
        b = _pval(P, "to", env, None)
        if a is None:
            a = _pval(P, "at", env, (0.0, 0.0))
        if b is None:
            d = _pval(P, "d", env, None) or _pval(P, "vec", env, None)
            if d is None:
                ang = _num(P, "dir", env, 0.0)
                L = _num(P, "len", env, 1.0)
                d = (L * math.cos(math.radians(ang)), L * math.sin(math.radians(ang)))
            b = (a[0] + d[0], a[1] + d[1])
        if not (_okp(a) and _okp(b)):
            return None
        st = self._style(P, env)
        h = head
        if head and "both" in P:
            h = 2
        marks = self._path_marks([a, b], st, head=h)
        if head:
            L = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
            ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
            marks += self._label(P, env, b, st, (ux * 0.42, uy * 0.42 + (0.0 if abs(uy) > 0.5 else 0.3)))
        else:
            marks += self._label(P, env, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), st)
        return {"marks": marks, "path": [a, b], "anchor": ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)}

    def _k_arrow(self, o, P, env):
        return self._k_line(o, P, env, head=1)

    def _k_path(self, o, P, env, closed=False):
        pts = _plist(P, "pts", env)
        if len(pts) < 2:
            return None
        closed = closed or bool(_num(P, "closed", env, 0.0))
        raw = pts
        if _num(P, "smooth", env, 0.0):
            pts = _catmull(pts, closed)
        st = self._style(P, env)
        head = 1 if _num(P, "arrow", env, 0.0) else 0
        marks = self._path_marks(pts, st, closed=closed, head=head)
        c = (sum(p[0] for p in raw) / len(raw), sum(p[1] for p in raw) / len(raw))
        marks += self._label(P, env, c, st)
        return {"marks": marks, "path": pts + ([pts[0]] if closed else []), "anchor": c, "verts": raw}

    def _k_poly(self, o, P, env):
        g = self._k_path(o, P, env, closed=True)
        return g

    def _k_bezier(self, o, P, env):
        pts = _plist(P, "pts", env)
        if len(pts) < 2:
            return None
        cur = _bezier(pts)
        st = self._style(P, env)
        head = 1 if _num(P, "arrow", env, 0.0) else 0
        return {"marks": self._path_marks(cur, st, head=head) + self._label(P, env, point_along(cur, 0.5), st),
                "path": cur, "anchor": point_along(cur, 0.5), "verts": pts}

    def _k_circle(self, o, P, env, dot=False):
        c = _pval(P, "at", env, (0.0, 0.0))
        r = _num(P, "r", env, 0.12 if dot else 1.0)
        if not (_okp(c) and _ok(r)) or r <= 0:
            return None
        st = self._style(P, env, fill=None)
        fill = st["fill"]
        if dot and fill is None:
            fill = st["stroke"] or DEF_COLOR
        pts = _circle(c, r, n=max(14, int(min(120, 10 + r * self.ppu * 0.8))))
        marks = self._path_marks(pts, st, closed=True, fill=fill)
        if dot:
            marks[0].stroke = None if "color" not in P and "stroke" not in P and fill is not None else marks[0].stroke
            if marks[0].stroke is None:
                marks[0].fill = fill
        marks += self._label(P, env, c, st, default_off=(0.0, r + 0.38))
        return {"marks": marks, "path": pts, "anchor": c, "radius": r}

    def _k_dot(self, o, P, env):
        return self._k_circle(o, P, env, dot=True)

    def _k_ellipse(self, o, P, env):
        c = _pval(P, "at", env, (0.0, 0.0))
        rx = _num(P, "rx", env, 1.5)
        ry = _num(P, "ry", env, 1.0)
        rot = _num(P, "rot", env, 0.0)
        if not (_okp(c) and _ok(rx) and _ok(ry)):
            return None
        st = self._style(P, env)
        pts = _circle(c, rx, ry, rot=rot, n=72)
        return {"marks": self._path_marks(pts, st, closed=True) + self._label(P, env, c, st, (0.0, ry + 0.38)),
                "path": pts, "anchor": c, "radius": max(rx, ry)}

    def _k_arc(self, o, P, env):
        c = _pval(P, "at", env, (0.0, 0.0))
        r = _num(P, "r", env, 1.0)
        a0 = _num(P, "a0", env, 0.0)
        a1 = _num(P, "a1", env, 90.0)
        if not (_okp(c) and _ok(r) and _ok(a0) and _ok(a1)):
            return None
        st = self._style(P, env)
        pts = _circle(c, r, a0=a0, a1=a1)
        head = 1 if _num(P, "arrow", env, 0.0) else 0
        if st["fill"] is not None:
            pts = [c] + pts
        return {"marks": self._path_marks(pts, st, closed=st["fill"] is not None, head=head)
                + self._label(P, env, point_along(pts, 0.5), st, (0.0, 0.0)),
                "path": pts, "anchor": c, "radius": r}

    def _k_rect(self, o, P, env):
        w = _num(P, "w", env, 2.0)
        h = _num(P, "h", env, 1.0)
        corner = _pval(P, "corner", env, None)
        if corner is not None:
            c = (corner[0] + w / 2, corner[1] + h / 2)
        else:
            c = _pval(P, "at", env, (0.0, 0.0))
        rot = _num(P, "rot", env, 0.0)
        if not (_okp(c) and _ok(w) and _ok(h)):
            return None
        hw, hh = w / 2, h / 2
        rr = max(0.0, min(_num(P, "rr", env, 0.0), abs(hw), abs(hh)))
        if rr > 0:
            loc = []
            for cx, cy, a0 in ((hw - rr, hh - rr, 0), (-hw + rr, hh - rr, 90), (-hw + rr, -hh + rr, 180), (hw - rr, -hh + rr, 270)):
                loc += _circle((cx, cy), rr, a0=a0, a1=a0 + 90, n=6)
        else:
            loc = [(hw, hh), (-hw, hh), (-hw, -hh), (hw, -hh)]
        cr, sr = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        pts = [(c[0] + x * cr - y * sr, c[1] + x * sr + y * cr) for x, y in loc]
        st = self._style(P, env)
        marks = self._path_marks(pts, st, closed=True)
        if isinstance(P.get("label"), str):
            marks += self._label(P, env, c, st, (0.0, 0.0))
        return {"marks": marks, "path": pts + [pts[0]], "anchor": c, "radius": min(abs(hw), abs(hh)), "verts": pts}

    def _k_star(self, o, P, env):
        c = _pval(P, "at", env, (0.0, 0.0))
        r = _num(P, "r", env, 0.5)
        n = int(max(3, min(12, _num(P, "n", env, 5.0))))
        rot = _num(P, "rot", env, 90.0)
        pts = []
        for k in range(2 * n):
            rr = r if k % 2 == 0 else r * 0.45
            a = math.radians(rot + 180.0 * k / n)
            pts.append((c[0] + rr * math.cos(a), c[1] + rr * math.sin(a)))
        st = self._style(P, env)
        if st["fill"] is None:
            st["fill"] = st["stroke"]
        return {"marks": self._path_marks(pts, st, closed=True), "path": pts, "anchor": c, "radius": r}

    def _k_text(self, o, P, env):
        s = o.text if o.text is not None else (P.get("label") if isinstance(P.get("label"), str) else "")
        if not s:
            return None
        pos = _pval(P, "at", env, (0.0, 0.0))
        if not _okp(pos):
            return None
        st = self._style(P, env)
        size = max(0.45, min(3.0, _num(P, "size", env, 0.6)))   # TV captions were too small to read at 540p (floor 0.18 -> 0.45)
        anchor = P.get("anchor", "c") if isinstance(P.get("anchor"), str) else "c"
        txt = interp(s, env)
        # draw<1 reveals the text left-to-right (typing)
        if st["draw"] < 0.999:
            txt = txt[:int(round(len(txt) * st["draw"]))]
        return {"marks": [Mark("text", pts=[pos], text=txt, size=size, stroke=st["stroke"] or DEF_COLOR,
                               alpha=st["alpha"], anchor=anchor, layer=st["layer"] + 0.5,
                               rot=_num(P, "rot", env, 0.0), bold=bool(_num(P, "bold", env, 0.0)))],
                "path": [pos], "anchor": pos}

    def _sample_fx(self, ex, env, var, a, b, n):
        v = dict(env.vars)
        e2 = Env(v, self)
        out = []
        for k in range(n + 1):
            x = a + (b - a) * k / n
            v[var] = x
            y = ex.eval(e2)
            out.append((x, y))
        return out

    def _k_plot(self, o, P, env):
        ex = P.get("y") or P.get("f")
        if ex is None or isinstance(ex, (str, tuple)) or not hasattr(ex, "eval"):
            return None
        xr = P.get("x") if isinstance(P.get("x"), tuple) else P.get("domain") if isinstance(P.get("domain"), tuple) else None
        if xr is None:
            mp0 = self._axes_map(P["on"]) if isinstance(P.get("on"), str) else None
            xr = mp0[0] if mp0 else (self.vb[0], self.vb[1])
        a, b = xr
        if not (_ok(a) and _ok(b)) or a == b:
            return None
        n = int(max(24, min(400, _num(P, "n", env, 220.0))))
        smp = self._sample_fx(ex, env, "x", a, b, n)
        ylim = (self.vb[3] - self.vb[2]) * 3
        yc = (self.vb[2] + self.vb[3]) / 2
        mp = self._axes_map(P["on"]) if isinstance(P.get("on"), str) else None
        clip_box = mp is not None
        if mp is not None:
            # data coordinates: a graph's curve stops at its axes box (it ran on up to 1.5 box heights,
            # over the title and off the screen), cut exactly where it crosses the edge
            ylim = abs(mp[1][1] - mp[1][0]) * 0.5 * 1.02
            yc = (mp[1][0] + mp[1][1]) / 2
        segs, cur = [], []
        prev = None
        for x, y in smp:
            inside = _ok(y) and abs(y - yc) < ylim
            if clip_box and prev is not None and _ok(y) and _ok(prev[1]) and inside != (abs(prev[1] - yc) < ylim):
                edge = yc + math.copysign(ylim, (prev[1] if inside else y) - yc)
                u = (edge - prev[1]) / (y - prev[1]) if y != prev[1] else 0.0
                cut = (prev[0] + (x - prev[0]) * u, edge)
                if inside:
                    cur = [cut]
                else:
                    cur.append(cut)
            prev = (x, y)
            if inside:
                cur.append((x, y))
            else:
                if len(cur) > 1:
                    segs.append(cur)
                cur = []
        if len(cur) > 1:
            segs.append(cur)
        st = self._style(P, env, stroke=(0.31, 0.82, 1.0), width=2.5)
        marks = []
        allpts = [p for s in segs for p in s]
        fill_under = st["fill"] if isinstance(P.get("fill"), tuple) else None
        if fill_under is not None:
            for s in segs:
                k = st["draw"]
                m = max(2, int(len(s) * k))
                ss = s[:m]
                poly = ss + [(ss[-1][0], 0.0), (ss[0][0], 0.0)]
                marks.append(Mark("path", pts=poly, closed=True, stroke=None, fill=fill_under, alpha=st["alpha"] * st["falpha"],
                                  layer=st["layer"] - 0.01))
        for s in segs:
            marks.extend(self._path_marks(s, dict(st, fill=None)))
        if allpts:
            # the label starts just right of the curve's right end (the renderer slides it to free space nearby)
            ends = [s[-1] for s in segs]
            e = max(ends, key=lambda p: p[0]) if ends else allpts[-1]
            if st["draw"] < 0.999 and allpts:
                e = point_along(allpts, st["draw"])
            marks += self._label(P, env, e, st, (0.15, 0.0), anchor="l")
        g_ = {"marks": marks, "path": allpts, "anchor": allpts[len(allpts) // 2] if allpts else (NAN, NAN)}
        if clip_box:
            # how far the data goes past the box (in box heights) and its y extent, for the change lint
            ys_ = [y for _x, y in smp if _ok(y)]
            if ys_:
                span_ = abs(mp[1][1] - mp[1][0])
                g_["overshoot"] = max(0.0, (max(abs(max(ys_) - yc), abs(min(ys_) - yc)) - span_ / 2) / span_)
                g_["data_y"] = (min(ys_), max(ys_))
        return g_

    def _k_area(self, o, P, env):
        ex = P.get("y") or P.get("f")
        if ex is None or not hasattr(ex, "eval"):
            return None
        ex2 = P.get("y2")
        a, b = P.get("x") if isinstance(P.get("x"), tuple) else (self.vb[0], self.vb[1])
        if not (_ok(a) and _ok(b)):
            return None
        n = 120
        top = self._sample_fx(ex, env, "x", a, b, n)
        if ex2 is not None and hasattr(ex2, "eval"):
            bot = self._sample_fx(ex2, env, "x", a, b, n)
        else:
            bot = [(x, 0.0) for x, _ in top]
        top = [p for p in top if _ok(p[1])]
        bot = [p for p in bot if _ok(p[1])]
        if len(top) < 2:
            return None
        st = self._style(P, env, stroke=None, fill=(0.31, 0.82, 1.0))
        fill = P.get("fill") if isinstance(P.get("fill"), tuple) else P.get("color") if isinstance(P.get("color"), tuple) else (0.31, 0.82, 1.0)
        k = st["draw"]
        m = max(2, int(len(top) * k))
        poly = top[:m] + list(reversed(bot[:m]))
        alpha = st["alpha"] * _num(P, "falpha", env, 0.35)
        return {"marks": [Mark("path", pts=poly, closed=True, stroke=None, fill=fill, alpha=alpha, layer=st["layer"] - 0.05)],
                "path": top, "anchor": point_along(top, 0.5)}

    def _k_curve(self, o, P, env):
        ex, ey = P.get("x"), P.get("y")
        if not (hasattr(ex, "eval") and hasattr(ey, "eval")):
            return None
        sr = P.get("s") if isinstance(P.get("s"), tuple) else (0.0, 1.0)
        a, b = sr
        n = int(max(24, min(500, _num(P, "n", env, 240.0))))
        v = dict(env.vars)
        e2 = Env(v, self)
        pts = []
        for k in range(n + 1):
            v["s"] = a + (b - a) * k / n
            x, y = ex.eval(e2), ey.eval(e2)
            if _ok(x) and _ok(y):
                pts.append((x, y))
        if len(pts) < 2:
            return None
        st = self._style(P, env, stroke=(1.0, 0.42, 0.84), width=2.5)
        closed = bool(_num(P, "closed", env, 0.0))
        return {"marks": self._path_marks(pts, st, closed=closed, head=1 if _num(P, "arrow", env, 0.0) else 0)
                + self._label(P, env, pts[-1], st, (0.25, 0.3)),
                "path": pts, "anchor": pts[len(pts) // 2]}

    def _k_axes(self, o, P, env):
        if isinstance(P.get("x"), tuple) and isinstance(P.get("y"), tuple) and self._box(P, env) is not None:
            g = self._k_axes_box(o, P, env)
            if g is not None:
                return g
        at = _pval(P, "at", env, (0.0, 0.0))
        xr = P.get("x") if isinstance(P.get("x"), tuple) else (self.vb[0] + 0.6, self.vb[1] - 0.6)
        yr = P.get("y") if isinstance(P.get("y"), tuple) else (self.vb[2] + 0.5, self.vb[3] - 0.5)
        if not all(_ok(v) for v in (*xr, *yr)) or not _okp(at):
            return None
        st = self._style(P, env, stroke=(0.5, 0.54, 0.62), width=1.6)
        marks = []
        tick = _num(P, "ticks", env, 0.0)
        grid = _num(P, "grid", env, 0.0)
        if o.kind == "grid" and not grid:
            grid = tick or 1.0
        gstyle = dict(st, stroke=(0.18, 0.21, 0.28), width=1.0, dash=0.0, fill=None)
        if grid and grid > 0:
            gx = grid
            k0 = math.ceil((xr[0] - at[0]) / gx)
            k1 = math.floor((xr[1] - at[0]) / gx)
            for k in range(max(k0, -60), min(k1, 60) + 1):
                x = at[0] + k * gx
                marks += self._path_marks([(x, yr[0]), (x, yr[1])], gstyle)
            k0 = math.ceil((yr[0] - at[1]) / gx)
            k1 = math.floor((yr[1] - at[1]) / gx)
            for k in range(max(k0, -60), min(k1, 60) + 1):
                y = at[1] + k * gx
                marks += self._path_marks([(xr[0], y), (xr[1], y)], gstyle)
        if o.kind == "grid":
            return {"marks": marks, "path": [], "anchor": at}
        marks += self._path_marks([(xr[0], at[1]), (xr[1], at[1])], st, head=1)
        marks += self._path_marks([(at[0], yr[0]), (at[0], yr[1])], st, head=1)
        nums = _num(P, "nums", env, 1.0 if tick else 0.0)
        tsz = 0.12
        if tick and tick > 0:
            k0 = math.ceil((xr[0] - at[0]) / tick)
            k1 = math.floor((xr[1] - 0.3 - at[0]) / tick)
            # numbers never collide: every n-th tick is numbered when they would not fit side by side
            fs = max(0.36, self.fmin)
            wid = (max((len(_fmtnum(k * tick)) for k in (k0, k1)), default=2) * 0.62 + 0.9) * fs
            xs = self._stride(wid, tick)
            ys = self._stride(fs * 1.25, tick)
            for k in range(max(k0, -40), min(k1, 40) + 1):
                if k == 0:
                    continue
                x = at[0] + k * tick
                marks += self._path_marks([(x, at[1] - tsz), (x, at[1] + tsz)], dict(st, dash=0.0))
                if nums and k % xs == 0:
                    marks.append(Mark("text", pts=[(x, at[1] - 0.38)], text=_fmtnum(x - at[0]), size=0.36,
                                      stroke=(0.6, 0.64, 0.72), alpha=st["alpha"], anchor="c", layer=st["layer"], soft="tick"))
            k0 = math.ceil((yr[0] - at[1]) / tick)
            k1 = math.floor((yr[1] - 0.3 - at[1]) / tick)
            for k in range(max(k0, -40), min(k1, 40) + 1):
                if k == 0:
                    continue
                y = at[1] + k * tick
                marks += self._path_marks([(at[0] - tsz, y), (at[0] + tsz, y)], dict(st, dash=0.0))
                if nums and k % ys == 0:
                    marks.append(Mark("text", pts=[(at[0] - 0.3, y)], text=_fmtnum(y - at[1]), size=0.36,
                                      stroke=(0.6, 0.64, 0.72), alpha=st["alpha"], anchor="r", layer=st["layer"], soft="tick"))
        labs = P.get("labels")
        xl = P.get("xlabel") if isinstance(P.get("xlabel"), str) else None
        yl = P.get("ylabel") if isinstance(P.get("ylabel"), str) else None
        if isinstance(labs, str):
            parts = [x.strip() for x in labs.split(",")]
            xl = xl or (parts[0] if parts else None)
            yl = yl or (parts[1] if len(parts) > 1 else None)
        lc = (0.75, 0.78, 0.85)
        if xl:
            marks.append(Mark("text", pts=[(xr[1] - 0.05, at[1] + 0.4)], text=interp(xl, env), size=0.45, stroke=lc,
                              alpha=st["alpha"], anchor="r", layer=st["layer"], soft="axis"))
        if yl:
            marks.append(Mark("text", pts=[(at[0] + 0.25, yr[1] - 0.1)], text=interp(yl, env), size=0.45, stroke=lc,
                              alpha=st["alpha"], anchor="l", layer=st["layer"], soft="axis"))
        return {"marks": marks, "path": [(xr[0], at[1]), (xr[1], at[1])], "anchor": at}

    _k_grid = _k_axes

    @staticmethod
    def _stride(need, spacing):
        """number every n-th tick so labels of world size `need` fit at tick spacing `spacing`"""
        if not (spacing > 1e-9) or not math.isfinite(need):
            return 1
        n = max(1, math.ceil(need / spacing - 1e-9))
        for m in (1, 2, 5, 10, 20, 50):
            if m >= n:
                return m
        return n

    def _k_axes_box(self, o, P, env):
        mp = self._axes_map(o.id) if o.id in self.doc.objs else None
        if mp is None:
            return None
        (xa, xb), (ya, yb), (bx0, bx1, by0, by1) = mp
        X = lambda x: bx0 + (x - xa) / (xb - xa) * (bx1 - bx0)
        Y = lambda y: by0 + (y - ya) / (yb - ya) * (by1 - by0)
        st = self._style(P, env, stroke=(0.5, 0.54, 0.62), width=1.6)
        marks = []
        x0w = X(0.0) if xa <= 0 <= xb else bx0
        y0w = Y(0.0) if ya <= 0 <= yb else by0
        tk = P.get("ticks")
        tv = tk.eval(env) if hasattr(tk, "eval") else None
        sx = sy = None
        if isinstance(tv, list) and len(tv) == 2:
            sx, sy = tv
        elif isinstance(tv, tuple):
            sx, sy = tv
        ok_step = lambda st_, span: isinstance(st_, float) and _ok(st_) and st_ > 0 and 2 <= span / st_ <= 16
        sx = sx if ok_step(sx, xb - xa) else _nice((xb - xa) / 5)
        sy = sy if ok_step(sy, yb - ya) else _nice((yb - ya) / 5)
        grid = _num(P, "grid", env, 0.0)
        gst = dict(st, stroke=(0.18, 0.21, 0.28), width=1.0, dash=0.0, fill=None)
        lc = (0.6, 0.64, 0.72)
        fs = max(0.34, self.fmin)
        k0 = math.ceil(xa / sx - 1e-9)
        wid = (max(len(_fmtnum(k0 * sx)), len(_fmtnum(math.floor(xb / sx) * sx))) * 0.62 + 0.9) * fs
        xst = self._stride(wid, abs(X(sx) - X(0.0)))
        yst = self._stride(fs * 1.25, abs(Y(sy) - Y(0.0)))
        for k in range(k0, k0 + 32):
            x = k * sx
            if x > xb + 1e-9:
                break
            if grid:
                marks += self._path_marks([(X(x), by0), (X(x), by1)], gst)
            marks += self._path_marks([(X(x), y0w - 0.1), (X(x), y0w + 0.1)], dict(st, dash=0.0))
            if k % xst == 0:
                marks.append(Mark("text", pts=[(X(x), y0w - 0.36)], text=_fmtnum(x), size=0.34, stroke=lc,
                                  alpha=st["alpha"], soft="tick"))
        k0 = math.ceil(ya / sy - 1e-9)
        for k in range(k0, k0 + 32):
            y = k * sy
            if y > yb + 1e-9:
                break
            if grid:
                marks += self._path_marks([(bx0, Y(y)), (bx1, Y(y))], gst)
            marks += self._path_marks([(x0w - 0.1, Y(y)), (x0w + 0.1, Y(y))], dict(st, dash=0.0))
            if k % yst == 0:
                marks.append(Mark("text", pts=[(x0w - 0.2, Y(y))], text=_fmtnum(y), size=0.34, stroke=lc,
                                  alpha=st["alpha"], anchor="r", soft="tick"))
        marks += self._path_marks([(bx0, y0w), (bx1 + 0.3, y0w)], st, head=1)
        marks += self._path_marks([(x0w, by0), (x0w, by1 + 0.3)], st, head=1)
        labs = P.get("labels")
        xl = P.get("xlabel") if isinstance(P.get("xlabel"), str) else None
        yl = P.get("ylabel") if isinstance(P.get("ylabel"), str) else None
        if isinstance(labs, str):
            parts = [x.strip() for x in labs.split(",")]
            xl = xl or (parts[0] if parts else None)
            yl = yl or (parts[1] if len(parts) > 1 else None)
        if xl:
            marks.append(Mark("text", pts=[(bx1 + 0.2, y0w - 0.75)], text=interp(xl, env), size=0.42, stroke=(0.75, 0.78, 0.85),
                              alpha=st["alpha"], anchor="r", soft="axis"))
        if yl:
            marks.append(Mark("text", pts=[(x0w + 0.25, by1 + 0.3)], text=interp(yl, env), size=0.42, stroke=(0.75, 0.78, 0.85),
                              alpha=st["alpha"], anchor="l", soft="axis"))
        return {"marks": marks, "path": [(bx0, y0w), (bx1, y0w)], "anchor": (x0w, y0w)}

    def _k_angle(self, o, P, env):
        c = _pval(P, "at", env, (0.0, 0.0))
        r = _num(P, "r", env, 0.6)
        p1 = _pval(P, "from", env, None)
        p2 = _pval(P, "to", env, None)
        if p1 is not None and p2 is not None:
            a0 = math.degrees(math.atan2(p1[1] - c[1], p1[0] - c[0]))
            a1 = math.degrees(math.atan2(p2[1] - c[1], p2[0] - c[0]))
            d = (a1 - a0 + 540) % 360 - 180
            a1 = a0 + d
        else:
            a0 = _num(P, "a0", env, 0.0)
            a1 = _num(P, "a1", env, 45.0)
        if not all(_ok(v) for v in (a0, a1, r)) or not _okp(c):
            return None
        st = self._style(P, env, stroke=(1.0, 0.82, 0.4), width=1.8)
        pts = _circle(c, r, a0=a0, a1=a1, n=24)
        marks = []
        if abs(abs(a1 - a0) - 90) < 0.5 and _num(P, "square", env, 1.0):
            u = (math.cos(math.radians(a0)), math.sin(math.radians(a0)))
            v = (math.cos(math.radians(a1)), math.sin(math.radians(a1)))
            s = r * 0.7
            pts = [(c[0] + u[0] * s, c[1] + u[1] * s), (c[0] + (u[0] + v[0]) * s, c[1] + (u[1] + v[1]) * s),
                   (c[0] + v[0] * s, c[1] + v[1] * s)]
        if st["fill"] is not None:
            marks.append(Mark("path", pts=[c] + pts, closed=True, stroke=None, fill=st["fill"], alpha=st["alpha"] * 0.35,
                              layer=st["layer"] - 0.01))
        marks += self._path_marks(pts, dict(st, fill=None))
        lab = P.get("label")
        if isinstance(lab, str) and lab:
            am = math.radians((a0 + a1) / 2)
            txt = interp(lab, env)
            lsz = _num(P, "lsize", env, 0.48)
            ca = math.cos(am)
            anc = "r" if ca < -0.15 else "l" if ca > 0.15 else "c"      # the label grows away from the angle
            rr = r + 0.25 + (0.3 * lsz if anc == "c" else 0.1)
            marks.append(Mark("text", pts=[(c[0] + rr * math.cos(am), c[1] + rr * math.sin(am))], text=txt,
                              size=lsz, stroke=st["stroke"] or DEF_COLOR, alpha=st["alpha"], anchor=anc, soft=True,
                              layer=st["layer"] + 0.01))
        return {"marks": marks, "path": pts, "anchor": c}

    def _k_wave(self, o, P, env):
        a = _pval(P, "from", env, (-4.0, 0.0))
        b = _pval(P, "to", env, (4.0, 0.0))
        amp = _num(P, "amp", env, 0.5)
        wl = max(0.05, _num(P, "wl", env, _num(P, "lambda", env, 1.5)))
        sp = _num(P, "speed", env, 1.0)
        ph = _num(P, "phase", env, 0.0)
        if not (_okp(a) and _okp(b) and _ok(amp)):
            return None
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L <= 0:
            return None
        ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        nx, ny = -uy, ux
        n = int(max(40, min(500, L / wl * 24)))
        t = env.vars.get("t", 0.0)
        damp = _num(P, "decay", env, 0.0)
        mode = P.get("mode") if isinstance(P.get("mode"), str) else ("standing" if "standing" in P else "")
        standing = mode.startswith("stand")
        k2 = 2 * math.pi / wl
        w2 = k2 * sp                                   # angular frequency: crests move at `speed` (units/s)
        pts = []
        for k in range(n + 1):
            s = L * k / n
            if standing:
                # the sum of two waves of amplitude amp travelling opposite ways: 2 amp sin(ks + phase) cos(wt)
                y = 2 * amp * math.exp(-damp * s) * math.sin(k2 * s + ph) * math.cos(w2 * t)
            else:
                y = amp * math.exp(-damp * s) * math.sin(k2 * (s - sp * t) + ph)
            pts.append((a[0] + ux * s + nx * y, a[1] + uy * s + ny * y))
        st = self._style(P, env, stroke=(0.31, 0.82, 1.0), width=2.2)
        marks = self._path_marks(pts, st)
        if standing and _num(P, "nodes", env, 0.0):
            # nodes (never move) where sin(ks + phase) = 0
            j0 = math.ceil((ph / math.pi) - 1e-9)
            for j in range(j0, j0 + 200):
                s_ = (j * math.pi - ph) / k2
                if s_ < -1e-9:
                    continue
                if s_ > L + 1e-9:
                    break
                q = (a[0] + ux * s_, a[1] + uy * s_)
                marks.append(Mark("path", pts=_circle(q, 0.09, n=12), closed=True, stroke=None, fill=(1.0, 0.85, 0.35),
                                  alpha=st["alpha"], layer=st["layer"] + 0.01))
        return {"marks": marks + self._label(P, env, b, st, (0.3, 0.4)), "path": pts,
                "anchor": ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)}

    def _k_spring(self, o, P, env):
        a = _pval(P, "from", env, (-2.0, 0.0))
        b = _pval(P, "to", env, (2.0, 0.0))
        coils = int(max(2, min(40, _num(P, "coils", env, 10.0))))
        w = _num(P, "w", env, 0.3)
        if not (_okp(a) and _okp(b)):
            return None
        L = math.hypot(b[0] - a[0], b[1] - a[1]) or 1e-6
        ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        nx, ny = -uy, ux
        pts = [a]
        lead = 0.1 * L
        for k in range(coils * 2 + 1):
            s = lead + (L - 2 * lead) * k / (coils * 2)
            y = (w if k % 2 else -w) if 0 < k < coils * 2 else 0.0
            pts.append((a[0] + ux * s + nx * y, a[1] + uy * s + ny * y))
        pts.append(b)
        st = self._style(P, env, stroke=(0.75, 0.78, 0.85), width=2.0)
        return {"marks": self._path_marks(pts, st), "path": pts, "anchor": ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)}

    def _k_field(self, o, P, env):
        fx, fy = P.get("fx"), P.get("fy")
        if not (hasattr(fx, "eval") and hasattr(fy, "eval")):
            return None
        step = max(0.3, _num(P, "step", env, 1.0))
        reg = P.get("region")
        x0, x1, y0, y1 = self.vb
        if hasattr(reg, "eval"):
            rv = reg.eval(env)
            if isinstance(rv, list) and len(rv) == 4 and all(_ok(v) for v in rv):
                x0, x1, y0, y1 = rv
        scale = _num(P, "scale", env, 0.0)
        st = self._style(P, env, stroke=(0.45, 0.65, 1.0), width=1.4)
        v = dict(env.vars)
        e2 = Env(v, self)
        marks = []
        vecs = []
        y = y0 + step / 2
        while y < y1 and len(vecs) < 500:
            x = x0 + step / 2
            while x < x1 and len(vecs) < 500:
                v["x"], v["y"] = x, y
                a, b = fx.eval(e2), fy.eval(e2)
                if _ok(a) and _ok(b):
                    vecs.append((x, y, a, b))
                x += step
            y += step
        mx = max((math.hypot(a, b) for _, _, a, b in vecs), default=0.0) or 1.0
        for x, y, a, b in vecs:
            m = math.hypot(a, b)
            if m < 1e-9:
                continue
            L = (scale * m) if scale else step * 0.7
            L = min(L, step * 1.2)
            ux, uy = a / m, b / m
            p0 = (x - ux * L / 2, y - uy * L / 2)
            p1 = (x + ux * L / 2, y + uy * L / 2)
            al = st["alpha"] * (0.35 + 0.65 * (m / mx) ** 0.5)
            marks.append(Mark("path", pts=[p0, p1], stroke=st["stroke"], width=st["width"], alpha=al, head=1,
                              layer=st["layer"], draw=st["draw"], glow=st["glow"]))
        return {"marks": marks, "path": [], "anchor": ((x0 + x1) / 2, (y0 + y1) / 2)}

    def _k_atom(self, o, P, env):
        c = _pval(P, "at", env, (0.0, 0.0))
        el = P.get("el") if isinstance(P.get("el"), str) else "C"
        el = el.strip()
        base = CPK.get(el) or CPK.get(el.capitalize()) or ((0.75, 0.6, 0.95), 0.5)
        r = _num(P, "r", env, base[1])
        col = P.get("color") if isinstance(P.get("color"), tuple) else base[0]
        if not (_okp(c) and _ok(r)):
            return None
        st = self._style(P, env)
        a = st["alpha"]
        n = max(20, int(r * self.ppu * 0.9))
        marks = [Mark("path", pts=_circle(c, r, n=n), closed=True, stroke=tuple(min(1, x * 1.1) for x in col), width=1.5,
                      fill=tuple(x * 0.55 for x in col), alpha=a, layer=st["layer"], glow=0.6),
                 Mark("path", pts=_circle((c[0] - r * 0.5, c[1] + r * 0.5), r * 0.2, n=16), closed=True, stroke=None,
                      fill=tuple(min(1, 0.4 + x * 0.7) for x in col), alpha=a * 0.35, layer=st["layer"] + 0.001, glow=0.0)]
        lab = P.get("label") if isinstance(P.get("label"), str) else el
        q = P.get("q")
        if hasattr(q, "eval"):                  # an expression choosing the charge text
            qv = q.eval(env)
            q = qv if isinstance(qv, str) else (("+" if qv > 0 else "−" if qv < 0 else "") if isinstance(qv, float) and qv == qv else "")
        q = q if isinstance(q, str) else ""
        if lab:
            marks.append(Mark("text", pts=[c], text=lab + (("^[" + q + "]") if q else ""), size=max(0.3, min(1.2, r * 1.15)),
                              stroke=(1, 1, 1) if el not in ("H", "He", "S", "Au") else (0.1, 0.12, 0.18), alpha=a,
                              layer=st["layer"] + 0.002, bold=True, soft="inline"))
        shells = P.get("shells")
        if hasattr(shells, "eval"):
            sv = shells.eval(env)
            if isinstance(sv, float):
                sv = [sv]
            if isinstance(sv, list):
                t = env.vars.get("t", 0.0)
                spin = _num(P, "spin", env, 1.0)
                for k, ne in enumerate(sv[:5]):
                    if not _ok(ne):
                        continue
                    rr = r + 0.45 * (k + 1)
                    marks.append(Mark("path", pts=_circle(c, rr, n=64), closed=True, stroke=(0.35, 0.45, 0.6), width=1.0,
                                      alpha=a * 0.8, layer=st["layer"] - 0.01, glow=0.3))
                    ne = int(max(0, min(18, ne)))
                    for j in range(ne):
                        ang = 2 * math.pi * j / max(1, ne) + t * spin * 1.4 / (k + 1)
                        p = (c[0] + rr * math.cos(ang), c[1] + rr * math.sin(ang))
                        marks.append(Mark("path", pts=_circle(p, 0.085, n=10), closed=True, stroke=None,
                                          fill=(0.45, 0.85, 1.0), alpha=a, layer=st["layer"] + 0.003))
        return {"marks": marks, "path": _circle(c, r, n=24), "anchor": c, "radius": r}

    def _k_bond(self, o, P, env):
        ia, ib = P.get("a"), P.get("b")
        if isinstance(ia, str) and isinstance(ib, str):
            pa, pb = self.ref_pos(ia), self.ref_pos(ib)
            ra, rb = self.ref_radius(ia), self.ref_radius(ib)
        else:
            pa, pb = _pval(P, "from", env, None), _pval(P, "to", env, None)
            ra = rb = 0.0
        if not (pa is not None and pb is not None and _okp(pa) and _okp(pb)):
            return None
        L = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
        if L < 1e-6:
            return None
        ux, uy = (pb[0] - pa[0]) / L, (pb[1] - pa[1]) / L
        nx, ny = -uy, ux
        s0, s1 = ra * 0.85, L - rb * 0.85
        if s1 <= s0:
            return None
        order = int(max(1, min(3, _num(P, "order", env, 1.0))))
        st = self._style(P, env, stroke=(0.8, 0.83, 0.9), width=3.0)
        gap = 0.13
        marks = []
        for k in range(order):
            off = (k - (order - 1) / 2) * gap
            p0 = (pa[0] + ux * s0 + nx * off, pa[1] + uy * s0 + ny * off)
            p1 = (pa[0] + ux * s1 + nx * off, pa[1] + uy * s1 + ny * off)
            marks += self._path_marks([p0, p1], dict(st, layer=st["layer"] - 0.1))
        mid = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
        marks += self._label(P, env, mid, st, (nx * 0.4, ny * 0.4))
        return {"marks": marks, "path": [pa, pb], "anchor": mid}

    def _k_brace(self, o, P, env):
        a = _pval(P, "from", env, (-1.0, 0.0))
        b = _pval(P, "to", env, (1.0, 0.0))
        if not (_okp(a) and _okp(b)):
            return None
        L = math.hypot(b[0] - a[0], b[1] - a[1]) or 1e-6
        ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        nx, ny = uy, -ux                                   # bulges to the right of a->b
        d = _num(P, "depth", env, 0.25)
        q = lambda s, h: (a[0] + ux * s + nx * h, a[1] + uy * s + ny * h)
        pts = _bezier([q(0, 0), q(0, d), q(L * 0.25, d * 0.5), q(L * 0.5, d)], 16) + \
            _bezier([q(L * 0.5, d), q(L * 0.75, d * 0.5), q(L, d), q(L, 0)], 16)
        st = self._style(P, env, stroke=(0.75, 0.78, 0.85), width=1.8)
        return {"marks": self._path_marks(pts, st) + self._label(P, env, q(L * 0.5, d), st, (nx * 0.45, ny * 0.45)),
                "path": pts, "anchor": q(L * 0.5, d)}

    # ---------------------------------------------------------------- r2-tv-prim: callouts, flows, ripples
    def _k_callout(self, o, P, env):
        """a label placed by the renderer in free space beside the picture, joined to its point by a leader"""
        p = _pval(P, "at", env, None) or _pval(P, "to", env, None)
        s = o.text if o.text is not None else (P.get("label") if isinstance(P.get("label"), str) else "")
        if not s or p is None or not _okp(p):
            return None
        st = self._style(P, env, stroke=(0.9, 0.92, 0.96))
        side = P.get("side") if isinstance(P.get("side"), str) else None
        side = {"left": "l", "right": "r", "top": "t", "bottom": "b", "above": "t", "below": "b"}.get(side, side)
        side = side[:1] if side and side[:1] in "lrtb" else None
        size = max(0.4, min(1.2, _num(P, "size", env, 0.45)))
        txt = interp(s, env)
        return {"marks": [Mark("text", pts=[p], text=txt, size=size, stroke=st["stroke"] or DEF_COLOR,
                               alpha=st["alpha"], layer=st["layer"] + 0.5, soft="callout", side=side)],
                "path": [p], "anchor": p}

    def _k_flow(self, o, P, env):
        """process / cycle: boxed steps on a ring (closed loop) or in a row / column, arrows between consecutive steps,
        the active step highlighted with a dot travelling along the arrow out of it"""
        c = _pval(P, "at", env, (0.0, 0.0))
        names = P.get("nodes") if "nodes" in P else P.get("steps")
        nv = names.eval(env) if hasattr(names, "eval") else None
        if not isinstance(nv, list) or not nv or not _okp(c):
            return None
        labels = [v if isinstance(v, str) else _fmtv(v, None) if isinstance(v, float) else "?" for v in nv[:12]]
        n = len(labels)
        mode = P.get("mode") if isinstance(P.get("mode"), str) else None
        if mode not in ("ring", "row", "col", "column", "cycle"):
            mode = "col" if ("column" in P or ("col" in P and not isinstance(P.get("col"), ColorSpec))) \
                else "row" if "row" in P else "ring"
        mode = {"column": "col", "cycle": "ring"}.get(mode, mode)
        s = max(0.42, self.fmin * 1.02, _num(P, "size", env, 0.45))
        cw = 0.62 * s                                     # glyph advance estimate (DejaVu Sans)
        bw_req = _num(P, "w", env, NAN)
        gap = max(0.3, _num(P, "gap", env, 0.85))
        x0v, x1v, y0v, y1v = self.vb
        minword = max((len(w) for l in labels for w in l.replace("\n", " ").split(" ")), default=1)
        for per in (16, 12, 9, 7, 5):
            if per < minword and per != 16:
                break                         # narrower wrapping cannot make the boxes any narrower
            if _ok(bw_req) and bw_req > 0:
                wrapped = [_wrap_chars(l, max(3, int((bw_req - 0.3) / cw))) for l in labels]
            else:
                wrapped = [_wrap_chars(l, per) if len(l) > per else l for l in labels]
            nl = max(w.count("\n") + 1 for w in wrapped)
            mx = max(len(x) for w in wrapped for x in w.split("\n"))
            em = max(_em(x) for w in wrapped for x in w.split("\n"))
            bw = bw_req if _ok(bw_req) and bw_req > 0 else em * s * 1.08 + 0.5        # (bold when active)
            bh = max(_num(P, "h", env, 0.0), nl * 1.22 * s + 0.4)
            # a row / column must fit on screen around its centre: wrap the names narrower, then tighten the gaps
            if mode == "row":
                room = 2 * min(c[0] - x0v, x1v - c[0]) - 0.6
                need = n * bw + (n - 1) * gap
                if need > room and n > 1:
                    gap = max(0.45, (room - n * bw) / (n - 1))
                    need = n * bw + (n - 1) * gap
                if need <= room or (_ok(bw_req) and bw_req > 0):
                    break
            elif mode == "col":
                room = 2 * min(c[1] - y0v, y1v - c[1]) - 0.4
                need = n * bh + (n - 1) * gap
                if need > room and n > 1:
                    gap = max(0.4, (room - n * bh) / (n - 1))
                break
            else:
                # a ring of long names ('metamorphic rock'): two-line names keep the boxes compact
                if per > 10 and minword <= 10 and max(len(l) for l in labels) > 10 and not (_ok(bw_req) and bw_req > 0):
                    continue
                break
        if mode == "row" and need > room and not (_ok(bw_req) and bw_req > 0):
            # still too wide with the narrowest names: smaller type, down to the legibility floor only
            s2 = max(self.fmin * 1.02, s * (room - (n - 1) * gap - n * 0.55) / max(1e-6, n * (bw - 0.55)))
            if s2 < s:
                s, cw = s2, 0.62 * s2
                bw = em * s * 1.08 + 0.5
                bh = max(_num(P, "h", env, 0.0), nl * 1.22 * s + 0.4)
        st = self._style(P, env, stroke=(0.31, 0.82, 1.0), width=2.0)
        col = st["stroke"] or (0.31, 0.82, 1.0)
        fillc = st["fill"] if isinstance(st["fill"], tuple) else col
        # node centres
        if mode == "ring":
            angs = [90 - 360 * i / n for i in range(n)]
            cs = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in angs]
            ry_u, rx_u = _num(P, "r", env, NAN), _num(P, "rx", env, NAN)
            ylim = max(1.0, min(c[1] - self.vb[2], self.vb[3] - c[1]) - 0.25)
            for _shrink in range(8):
                ry = ry_u if _ok(ry_u) and ry_u > 0 else max(1.5, bh * 1.6)
                rx = rx_u if _ok(rx_u) and rx_u > 0 else max(ry * 1.4, bw * 0.6)
                # the smallest ellipse on which no two steps touch (room for an arrow between neighbours): grow the
                # width first (the screen is wide), the height once the width would leave the screen
                g = 0.5
                xlim = max(1.0, min(c[0] - self.vb[0], self.vb[1] - c[0]) - bw / 2 - 0.3)
                for _ in range(60):
                    bad = any(abs(rx * (cs[i][0] - cs[j][0])) < bw + gg and abs(ry * (cs[i][1] - cs[j][1])) < bh + gg
                              for i in range(n) for j in range(i + 1, n)
                              for gg in ((1.0 if (j == i + 1 or (i == 0 and j == n - 1)) else g),))
                    if not bad or n < 2:
                        break
                    if not (_ok(rx_u) and rx_u > 0) and rx * 1.06 <= xlim:
                        rx *= 1.06
                    elif not (_ok(ry_u) and ry_u > 0):
                        ry *= 1.06
                    elif not (_ok(rx_u) and rx_u > 0):
                        rx *= 1.06
                    else:
                        break
                # the ring must stay on the screen (5 long names grew it 3x the screen height, the
                # auto-fit then showed two giant lines): smaller type down to the legibility floor, then squash
                if ry + bh / 2 <= ylim or (_ok(ry_u) and ry_u > 0) or s <= self.fmin * 1.03:
                    break
                s = max(self.fmin * 1.02, s * 0.88)
                bw = bw_req if _ok(bw_req) and bw_req > 0 else em * s * 1.08 + 0.5
                bh = max(_num(P, "h", env, 0.0), nl * 1.22 * s + 0.4)
            if not (_ok(ry_u) and ry_u > 0):
                ry = min(ry, max(0.8, ylim - bh / 2))
            if not (_ok(rx_u) and rx_u > 0):
                rx = min(rx, max(1.0, min(c[0] - self.vb[0], self.vb[1] - c[0]) - bw / 2 - 0.2))
            ctr = [(c[0] + rx * math.cos(math.radians(a)), c[1] + ry * math.sin(math.radians(a))) for a in angs]
        else:
            step = (bw if mode == "row" else bh) + gap
            ctr = [(c[0] + (i - (n - 1) / 2) * step, c[1]) if mode == "row" else (c[0], c[1] - (i - (n - 1) / 2) * step)
                   for i in range(n)]
        av = _num(P, "active", env, NAN)
        act, frac = -1, 0.0
        if _ok(av):
            act = int(math.floor(av)) % n
            frac = av - math.floor(av)
        marks = []
        L = st["layer"]
        hw, hh = bw / 2, bh / 2
        rr = min(0.22, hh * 0.6)
        for i, (p, lab) in enumerate(zip(ctr, wrapped)):
            on = i == act
            box = _rrect(p, hw, hh, rr)
            # opaque boxes, colours pre-mixed with the background (no per-mark blending: a flow is ~20 marks)
            marks.append(Mark("path", pts=box, closed=True, stroke=None, fill=_premix(fillc, 0.5 if on else 0.16),
                              alpha=st["alpha"], layer=L, glow=0.0))
            marks.append(Mark("path", pts=box, closed=True, stroke=(1.0, 1.0, 1.0) if on else _premix(col, 1.0 if act < 0 else 0.75),
                              width=3.0 if on else st["width"], alpha=st["alpha"], layer=L + 0.01, glow=1.0 if on else 0.6))
            marks.append(Mark("text", pts=[p], text=lab, size=s, stroke=(1.0, 1.0, 1.0) if on else (0.9, 0.93, 0.97),
                              alpha=st["alpha"], layer=L + 0.02, soft="inline", bold=on))
        # arrows between consecutive steps (a ring closes the loop)
        pairs = [(i, (i + 1) % n) for i in range(n if (mode == "ring" and n > 2) else n - 1)]
        arrows = {}
        pad = 0.1
        for i, j in pairs:
            a, b = ctr[i], ctr[j]
            if mode == "ring":
                a0, a1 = math.radians(angs[i]), math.radians(angs[i] - 360.0 / n)
                arc = [(c[0] + rx * math.cos(a0 + (a1 - a0) * k / 48), c[1] + ry * math.sin(a0 + (a1 - a0) * k / 48))
                       for k in range(49)]
                arc = [q for q in arc if not (_in_box(q, a, hw + pad, hh + pad) or _in_box(q, b, hw + pad, hh + pad))]
                pts = arc if len(arc) >= 2 else None
            else:
                pts = None
            if pts is None:
                pa, pb = _box_exit(a, b, hw + pad, hh + pad), _box_exit(b, a, hw + pad, hh + pad)
                pts = [pa, pb]
            arrows[i] = pts
            marks.append(Mark("path", pts=pts, stroke=col if i == act else _premix(col, 0.7), width=st["width"],
                              alpha=st["alpha"], head=1, layer=L - 0.01, glow=0.6, draw=st["draw"]))
        if act >= 0 and act in arrows:
            u = frac if frac > 1e-3 else (env.vars.get("t", 0.0) % 1.2) / 1.2
            q = point_along(arrows[act], u)
            dc = P.get("dot") if isinstance(P.get("dot"), tuple) else (1.0, 0.85, 0.35)
            marks.append(Mark("path", pts=_circle(q, 0.13, n=14), closed=True, stroke=None, fill=dc, alpha=st["alpha"],
                              layer=L + 0.03, glow=1.2))
        return {"marks": marks, "path": ctr + ([ctr[0]] if mode == "ring" else []), "anchor": c, "verts": ctr,
                "nodes": ctr, "radius": max(hw, hh)}

    def _k_ripple(self, o, P, env):
        """circular wavefronts (crests) expanding from a point source at `speed`, `wl` apart, fading with distance
        (amplitude ~ 1/sqrt(r)); two ripples overlap additively, so crests crossing light up (interference)"""
        c = _pval(P, "at", env, (0.0, 0.0))
        wl = max(0.12, _num(P, "wl", env, _num(P, "lambda", env, 1.0)))
        sp = _num(P, "speed", env, 1.0)
        rmax = max(0.2, _num(P, "r", env, _num(P, "rmax", env, 7.0)))
        ph = _num(P, "phase", env, 0.0)
        t0 = _num(P, "start", env, 0.0)
        if not (_okp(c) and _ok(sp) and _ok(rmax)):
            return None
        t = env.vars.get("t", 0.0)
        front = sp * (t - t0) if sp > 0 else rmax          # how far the first crest has travelled
        front = min(front, rmax)
        nmax = int(max(1, min(60, _num(P, "n", env, 60.0))))
        st = self._style(P, env, stroke=(0.31, 0.82, 1.0), width=2.0)
        amp = max(0.0, min(2.0, _num(P, "amp", env, 1.0)))
        segs = []
        tr = bool(_num(P, "troughs", env, 0.0))
        off = (sp * (t - t0) - ph / 360.0 * wl) % wl if sp > 0 else (-ph / 360.0 * wl) % wl
        for k in range(nmax * (2 if tr else 1)):
            r = off + k * (wl / 2 if tr else wl)
            if r > front + 1e-6 or r > rmax:
                break
            if r < 0.04:
                continue
            fade = min(1.0, math.sqrt(wl / max(r, wl))) * (1.0 - _smooth((r - 0.8 * rmax) / (0.2 * rmax)))
            trough = tr and (k % 2 == 1)
            a = amp * fade * (0.35 if trough else 1.0)
            if a < 0.02:
                continue
            npts = int(max(24, min(160, r * self.ppu * 0.35)))
            segs.append((_circle(c, r, n=npts), a))
        if not segs:
            return {"marks": [], "path": [c], "anchor": c, "radius": 0.0}
        m = Mark("path", pts=segs[0][0], closed=False, stroke=st["stroke"] or (0.31, 0.82, 1.0), width=st["width"],
                 alpha=st["alpha"], layer=st["layer"], glow=st["glow"] * 0.6, segs=segs, add=True)
        return {"marks": [m] + self._label(P, env, c, st, (0.0, -0.5)), "path": segs[-1][0], "anchor": c, "radius": front}

    def _k_group(self, o, P, env):
        at = _pval(P, "at", env, (0.0, 0.0))
        return {"marks": [], "path": [at], "anchor": at}

    # ---------------------------------------------------------------- optics
    def _lens_shape(self, c, ax, h, R1, R2, dmin=0.08):
        """outline of a lens with surface radii R1 (first surface, facing -axis) and R2 (0 = flat), half-height h"""
        def u_of(R, v, base):
            if R == 0 or not _ok(R):
                return base
            rr = max(abs(R), h * 1.02)
            R = math.copysign(rr, R)
            return base + R - math.copysign(1.0, R) * math.sqrt(max(0.0, R * R - v * v))
        s1 = u_of(R1, h, 0.0)
        s2 = u_of(R2, h, 0.0)
        d = max(dmin, s1 - s2 + dmin)
        pv = (-ax[1], ax[0])
        n = 28
        front, back = [], []
        for k in range(n + 1):
            v = -h + 2 * h * k / n
            front.append((u_of(R1, v, -d / 2), v))
            back.append((u_of(R2, v, d / 2), v))
        loc = front + list(reversed(back))
        w = lambda u, v: (c[0] + ax[0] * u + pv[0] * v, c[1] + ax[1] * u + pv[1] * v)
        pts = [w(u, v) for u, v in loc]
        info = {"d": d, "front": [w(u, v) for u, v in front], "back": [w(u, v) for u, v in back]}
        # arc centres for exact normals
        info["c1"] = w(-d / 2 + R1, 0.0) if R1 and _ok(R1) else None
        info["c2"] = w(d / 2 + R2, 0.0) if R2 and _ok(R2) else None
        return pts, info

    def _k_lens(self, o, P, env):
        c = _pval(P, "at", env, (0.0, 0.0))
        h = abs(_num(P, "h", env, 1.6))
        rot = _num(P, "rot", env, 0.0)
        n_idx = _num(P, "n", env, 1.5)
        R1 = _num(P, "r1", env, NAN)
        R2 = _num(P, "r2", env, NAN)
        f = _num(P, "f", env, NAN)
        if _ok(R1) or _ok(R2):
            R1 = R1 if _ok(R1) else 0.0
            R2 = R2 if _ok(R2) else 0.0
            from .expr import f_lensmaker
            if not _ok(f):
                f = f_lensmaker(n_idx, R1, R2)
            dR1, dR2 = R1, R2
            if not _num(P, "real", env, 0.0):
                # an IDEAL thin lens (rays bend at its mid-plane) drawn with its true radii looked like a
                # ball when R ~ h, its focus inside the glass. Drawn sag saturates at 0.45 h (tanh), so a more curved
                # surface still looks more curved but the lens stays thin enough for the thin-lens rays
                def vis(R):
                    if not R or not _ok(R):
                        return R
                    rr = max(abs(R), h)
                    sag = rr - math.sqrt(max(0.0, rr * rr - h * h)) if abs(R) >= h else abs(R)
                    sv = 0.45 * h * math.tanh(sag / (0.45 * h))
                    return math.copysign((h * h + sv * sv) / (2 * max(1e-6, sv)), R)
                dR1, dR2 = vis(R1), vis(R2)
        else:
            if not _ok(f):
                f = 2.0
            # drawn shape only (the trace is the ideal thin lens): fatter for a shorter focal length
            sag = max(0.06 * h, min(0.3 * h, 0.25 * h * h / max(1e-6, abs(f))))
            R = math.copysign((h * h + sag * sag) / (2 * sag), f) if f else 1e6
            dR1, dR2 = R, -R
        if not (_okp(c) and _ok(h) and h > 0):
            return None
        real_ = bool(_num(P, "real", env, 0.0)) and (_ok(R1) or _ok(R2))
        if real_:
            # a REAL lens is the glass its n and radii say: a half-height above |R| is impossible, so the
            # aperture shrinks (it used to inflate R to fit h, so the traced glass was a weaker lens than the F marks)
            rs = [abs(r) for r in (R1, R2) if r and _ok(r)]
            if rs:
                h = min(h, 0.92 * min(rs))
        ax = (math.cos(math.radians(rot)), math.sin(math.radians(rot)))
        pts, info = self._lens_shape(c, ax, h, dR1, dR2)
        fo = {}
        if real_ and not _ok(_num(P, "f", env, NAN)):
            # thick lens: true focal length and the focal points measured from the vertices (where the traced
            # paraxial rays really cross), so F, F' and 2F sit where the light goes
            d = info["d"]
            i1 = 0.0 if not R1 else 1.0 / R1
            i2 = 0.0 if not R2 else 1.0 / R2
            pw = (n_idx - 1) * (i1 - i2 + (n_idx - 1) * d * i1 * i2 / n_idx)
            if pw and _ok(pw):
                f = 1.0 / pw
                bfd = f * (1 - (n_idx - 1) * d * i1 / n_idx)
                ffd = f * (1 + (n_idx - 1) * d * i2 / n_idx)
                fo = {1: d / 2 + bfd, -1: -(d / 2 + ffd)}
                fo[2], fo[-2] = fo[1] + f, fo[-1] - f
        st = self._style(P, env, stroke=(0.6, 0.85, 1.0), width=1.8)
        fill = st["fill"] if st["fill"] is not None else GLASS
        marks = [Mark("path", pts=pts, closed=True, stroke=st["stroke"], width=st["width"], fill=fill, alpha=st["alpha"],
                      layer=st["layer"] - 0.2, glow=0.7)]
        marks[0].dash = 0
        # translucent fill drawn as a separate mark so the outline stays crisp
        marks = [Mark("path", pts=pts, closed=True, stroke=None, fill=fill, alpha=st["alpha"] * 0.22, layer=st["layer"] - 0.21,
                      glow=0.0),
                 Mark("path", pts=pts, closed=True, stroke=st["stroke"], width=st["width"], alpha=st["alpha"],
                      layer=st["layer"] - 0.2, glow=0.7)]
        pv = (-ax[1], ax[0])
        if _num(P, "axis", env, 0.0):
            x0, x1, y0, y1 = self.vb
            L = (x1 - x0) + (y1 - y0)
            marks.append(Mark("path", pts=[(c[0] - ax[0] * L, c[1] - ax[1] * L), (c[0] + ax[0] * L, c[1] + ax[1] * L)],
                              stroke=(0.4, 0.44, 0.52), width=1.0, dash=7.0, alpha=st["alpha"] * 0.9, layer=st["layer"] - 0.3,
                              glow=0.0))
        if _num(P, "marks", env, _num(P, "foci", env, 0.0)) and _ok(f) and abs(f) < 1e3:
            for sgn, name in ((-1, "F"), (1, "F'")):
                uf = fo.get(sgn, f * sgn)
                p = (c[0] + ax[0] * uf, c[1] + ax[1] * uf)
                marks.append(Mark("path", pts=_circle(p, 0.08, n=10), closed=True, stroke=None, fill=(1.0, 0.82, 0.4),
                                  alpha=st["alpha"], layer=st["layer"] + 0.1))
                marks.append(Mark("text", pts=[(p[0] + pv[0] * -0.38, p[1] + pv[1] * -0.38)], text=name, size=0.42,
                                  stroke=(1.0, 0.82, 0.4), alpha=st["alpha"], layer=st["layer"] + 0.1))
                if _num(P, "marks", env, 0.0) >= 2:
                    u2 = fo.get(2 * sgn, 2 * f * sgn)
                    p2 = (c[0] + ax[0] * u2, c[1] + ax[1] * u2)
                    marks.append(Mark("path", pts=_circle(p2, 0.06, n=10), closed=True, stroke=None, fill=(0.6, 0.64, 0.72),
                                      alpha=st["alpha"], layer=st["layer"] + 0.1))
                    marks.append(Mark("text", pts=[(p2[0] + pv[0] * -0.38, p2[1] + pv[1] * -0.38)], text="2" + name,
                                      size=0.38, stroke=(0.6, 0.64, 0.72), alpha=st["alpha"], layer=st["layer"] + 0.1))
        marks += self._label(P, env, (c[0] + pv[0] * (h + 0.35), c[1] + pv[1] * (h + 0.35)), st, (0.0, 0.0))
        real = bool(_num(P, "real", env, 0.0))
        el = {"type": "glass" if real else "thinlens", "c": c, "ax": ax, "h": h, "f": f, "n": n_idx,
              "disp": _num(P, "disp", env, 0.0)}
        if real:
            el["poly"] = pts
            el["centres"] = (info["c1"], info["c2"], len(info["front"]))
        return {"marks": marks, "path": pts, "anchor": c, "optic": el, "radius": h, "f": f}

    def _k_mirror(self, o, P, env):
        a = _pval(P, "from", env, None)
        b = _pval(P, "to", env, None)
        st = self._style(P, env, stroke=(0.85, 0.9, 1.0), width=3.0)
        if a is not None and b is not None:
            if not (_okp(a) and _okp(b)):
                return None
            L = math.hypot(b[0] - a[0], b[1] - a[1]) or 1e-6
            nrm = (-(b[1] - a[1]) / L, (b[0] - a[0]) / L)
            hatch = []
            for k in range(1, int(L / 0.35) + 1):
                p = (a[0] + (b[0] - a[0]) * k * 0.35 / L, a[1] + (b[1] - a[1]) * k * 0.35 / L)
                hatch.append([p, (p[0] - nrm[0] * 0.18 - (b[0] - a[0]) / L * 0.15, p[1] - nrm[1] * 0.18 - (b[1] - a[1]) / L * 0.15)])
            marks = self._path_marks([a, b], st)
            if _num(P, "hatch", env, 1.0):
                for hp in hatch:
                    marks += self._path_marks(hp, dict(st, width=1.0, alpha=st["alpha"] * 0.6, glow=0.0))
            marks += self._label(P, env, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), st, (nrm[0] * 0.4, nrm[1] * 0.4))
            return {"marks": marks, "path": [a, b], "anchor": ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2),
                    "optic": {"type": "flatmirror", "a": a, "b": b}}
        c = _pval(P, "at", env, (0.0, 0.0))
        h = abs(_num(P, "h", env, 1.6))
        rot = _num(P, "rot", env, 180.0)           # the side it faces (toward the light): default -x
        f = _num(P, "f", env, NAN)
        R = _num(P, "r", env, NAN)
        if not _ok(f):
            f = R / 2 if _ok(R) else math.inf
        if not _okp(c):
            return None
        ax = (math.cos(math.radians(rot)), math.sin(math.radians(rot)))
        pv = (-ax[1], ax[0])
        pts = []
        for k in range(33):
            v = -h + 2 * h * k / 32
            sag = (v * v) / (4 * f) if _ok(f) and abs(f) > 1e-6 else 0.0   # parabola: concave toward the light when f > 0
            pts.append((c[0] + ax[0] * sag + pv[0] * v, c[1] + ax[1] * sag + pv[1] * v))
        marks = self._path_marks(pts, st)
        if _num(P, "hatch", env, 1.0):
            for k in range(0, 33, 3):
                p = pts[k]
                marks += self._path_marks([p, (p[0] - ax[0] * 0.2 - pv[0] * 0.12, p[1] - ax[1] * 0.2 - pv[1] * 0.12)],
                                          dict(st, width=1.0, alpha=st["alpha"] * 0.6, glow=0.0))
        if _num(P, "axis", env, 0.0):
            x0, x1, y0, y1 = self.vb
            L = (x1 - x0) + (y1 - y0)
            marks.append(Mark("path", pts=[c, (c[0] + ax[0] * L, c[1] + ax[1] * L)], stroke=(0.4, 0.44, 0.52), width=1.0,
                              dash=7.0, alpha=st["alpha"] * 0.9, layer=st["layer"] - 0.3, glow=0.0))
        if _num(P, "marks", env, _num(P, "foci", env, 0.0)) and _ok(f) and abs(f) < 1e3:
            p = (c[0] + ax[0] * f, c[1] + ax[1] * f)
            marks.append(Mark("path", pts=_circle(p, 0.08, n=10), closed=True, stroke=None, fill=(1.0, 0.82, 0.4),
                              alpha=st["alpha"], layer=st["layer"] + 0.1))
            marks.append(Mark("text", pts=[(p[0] - pv[0] * 0.38, p[1] - pv[1] * 0.38)], text="F", size=0.42,
                              stroke=(1.0, 0.82, 0.4), alpha=st["alpha"], layer=st["layer"] + 0.1))
            if _num(P, "marks", env, 0.0) >= 2:
                p = (c[0] + ax[0] * 2 * f, c[1] + ax[1] * 2 * f)
                marks.append(Mark("text", pts=[(p[0] - pv[0] * 0.38, p[1] - pv[1] * 0.38)], text="C", size=0.42,
                                  stroke=(0.6, 0.64, 0.72), alpha=st["alpha"], layer=st["layer"] + 0.1))
        marks += self._label(P, env, (c[0] + pv[0] * (h + 0.35), c[1] + pv[1] * (h + 0.35)), st, (0.0, 0.0))
        return {"marks": marks, "path": pts, "anchor": c, "f": f,
                "optic": {"type": "curvedmirror", "c": c, "ax": ax, "h": h, "f": f}}

    def _k_glass(self, o, P, env):
        pts = _plist(P, "pts", env)
        if len(pts) < 3:
            # a block given by at/w/h, or a prism by at/r (equilateral)
            c = _pval(P, "at", env, (0.0, 0.0))
            if "w" in P or "h" in P:
                w, h = _num(P, "w", env, 3.0), _num(P, "h", env, 2.0)
                rot = math.radians(_num(P, "rot", env, 0.0))
                cr, sr = math.cos(rot), math.sin(rot)
                pts = [(c[0] + x * cr - y * sr, c[1] + x * sr + y * cr)
                       for x, y in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]
            else:
                # at + r: a round drop / ball (sides=N for a regular polygon, e.g. sides=3 = an equilateral prism)
                r = _num(P, "r", env, 1.5)
                rot = _num(P, "rot", env, 90.0)
                ns = int(max(3, min(160, _num(P, "sides", env, 120.0))))
                pts = [(c[0] + r * math.cos(math.radians(rot + 360.0 * k / ns)), c[1] + r * math.sin(math.radians(rot + 360.0 * k / ns)))
                       for k in range(ns)]
        if not all(_okp(p) for p in pts):
            return None
        st = self._style(P, env, stroke=(0.6, 0.85, 1.0), width=1.6)
        fill = st["fill"] if st["fill"] is not None else (P.get("color") if isinstance(P.get("color"), tuple) else GLASS)
        n_idx = _num(P, "n", env, 1.5)
        marks = [Mark("path", pts=pts, closed=True, stroke=None, fill=fill, alpha=st["alpha"] * _num(P, "falpha", env, 0.2),
                      layer=st["layer"] - 0.21, glow=0.0),
                 Mark("path", pts=pts, closed=True, stroke=st["stroke"], width=st["width"], alpha=st["alpha"],
                      layer=st["layer"] - 0.2, glow=0.6)]
        c = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        marks += self._label(P, env, c, st, (0.0, 0.0))
        return {"marks": marks, "path": pts + [pts[0]], "anchor": c, "verts": pts,
                "optic": {"type": "glass", "poly": pts, "n": n_idx, "disp": _num(P, "disp", env, 0.0042),
                          "inner": int(max(0, min(4, _num(P, "inner", env, 0.0))))}}

    def _k_screen(self, o, P, env):
        a = _pval(P, "from", env, (5.0, -2.0))
        b = _pval(P, "to", env, (5.0, 2.0))
        if not (_okp(a) and _okp(b)):
            return None
        st = self._style(P, env, stroke=(0.5, 0.54, 0.62), width=4.0)
        return {"marks": self._path_marks([a, b], st) + self._label(P, env, b, st, (0.0, 0.35)), "path": [a, b],
                "anchor": ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), "optic": {"type": "screen", "a": a, "b": b}}

    def optics(self):
        if self._optics is None:
            els = []
            for ident, o in self.doc.objs.items():
                if o.kind in ("lens", "mirror", "glass", "screen"):
                    g = self._geom(ident)
                    if g and g.get("optic"):
                        els.append(g["optic"])
            self._optics = els
        return self._optics

    def trace(self, o0, d, wl=None, max_len=60.0, bounces=24, big=None):
        """trace one ray through every optical element; returns the polyline and the index of the last bend.
        big = an element treated as unbounded (principal-ray construction: textbook rays may pass beyond its rim)"""
        els = self.optics()
        if big is not None:
            els = [dict(e, h=1e3) if e is big else e for e in els]
        x0, x1, y0, y1 = self.vb
        mx = 0.15 * max(x1 - x0, y1 - y0)
        pts = [o0]
        p = o0
        d = _unit(d)
        total = 0.0
        last_bend = 0
        inner_done = {}                  # glass id -> internal reflections taken (glass inner=N)
        for _ in range(bounces):
            best = None
            for el in els:
                ty = el["type"]
                if ty == "thinlens" or ty == "curvedmirror":
                    c, ax, h = el["c"], el["ax"], el["h"]
                    pv = (-ax[1], ax[0])
                    hit = _seg_hit(p, d, (c[0] - pv[0] * h, c[1] - pv[1] * h), (c[0] + pv[0] * h, c[1] + pv[1] * h))
                    if hit and (best is None or hit[0] < best[0]):
                        best = (hit[0], el, None)
                elif ty == "flatmirror" or ty == "screen":
                    hit = _seg_hit(p, d, el["a"], el["b"])
                    if hit and (best is None or hit[0] < best[0]):
                        best = (hit[0], el, None)
                elif ty == "glass":
                    poly = el["poly"]
                    n = len(poly)
                    for i in range(n):
                        a, b = poly[i], poly[(i + 1) % n]
                        hit = _seg_hit(p, d, a, b)
                        if hit and hit[0] > 1e-6 and (best is None or hit[0] < best[0]):
                            best = (hit[0], el, i)
            if best is None:
                break
            s, el, ei = best
            if total + s > max_len:
                break
            q = (p[0] + d[0] * s, p[1] + d[1] * s)
            total += s
            pts.append(q)
            ty = el["type"]
            if ty == "screen":
                return pts, len(pts) - 1, True
            if ty == "flatmirror":
                a, b = el["a"], el["b"]
                t_ = _unit((b[0] - a[0], b[1] - a[1]))
                nrm = (-t_[1], t_[0])
                k = d[0] * nrm[0] + d[1] * nrm[1]
                d = (d[0] - 2 * k * nrm[0], d[1] - 2 * k * nrm[1])
            elif ty == "thinlens":
                c, ax, f = el["c"], el["ax"], el["f"]
                pv = (-ax[1], ax[0])
                v = (q[0] - c[0]) * pv[0] + (q[1] - c[1]) * pv[1]
                du = d[0] * ax[0] + d[1] * ax[1]
                dv = d[0] * pv[0] + d[1] * pv[1]
                sgn = 1.0 if du >= 0 else -1.0
                du = abs(du) or 1e-9
                fe = f
                if wl is not None and el.get("disp"):
                    from .expr import f_cauchy
                    n0 = el.get("n", 1.5)
                    nl = f_cauchy(wl, n0, el["disp"])
                    fe = f * (n0 - 1) / ((nl - 1) or 1e-9)
                m = dv / du - (v / fe if _ok(fe) and fe != 0 else 0.0)
                d = _unit((ax[0] * sgn + pv[0] * m, ax[1] * sgn + pv[1] * m))
            elif ty == "curvedmirror":
                c, ax, f = el["c"], el["ax"], el["f"]
                pv = (-ax[1], ax[0])
                v = (q[0] - c[0]) * pv[0] + (q[1] - c[1]) * pv[1]
                du = d[0] * ax[0] + d[1] * ax[1]
                dv = d[0] * pv[0] + d[1] * pv[1]
                if du < 0:     # coming toward its face: reflect into +ax, then focus
                    m = dv / (-du) - (v / f if _ok(f) and f != 0 else 0.0)
                    d = _unit((ax[0] + pv[0] * m, ax[1] + pv[1] * m))
                else:          # its back: plain reflection
                    d = (d[0] - 2 * du * ax[0], d[1] - 2 * du * ax[1])
            elif ty == "glass":
                poly = el["poly"]
                a, b = poly[ei], poly[(ei + 1) % len(poly)]
                t_ = _unit((b[0] - a[0], b[1] - a[1]))
                nrm = (-t_[1], t_[0])
                cen = el.get("centres")
                if cen:   # a real lens: exact normals on its spherical faces
                    c1, c2, nf = cen
                    cc = c1 if ei < nf else c2 if (nf < ei < 2 * nf + 1) else None
                    if cc is not None:
                        nrm = _unit((q[0] - cc[0], q[1] - cc[1]))
                n_g = el["n"]
                if wl is not None and el.get("disp"):
                    from .expr import f_cauchy
                    n_g = f_cauchy(wl, n_g, el["disp"])
                probe_in = (q[0] - d[0] * 1e-5, q[1] - d[1] * 1e-5)
                inside = _inside(poly, probe_in)
                n1, n2 = (n_g, self._medium(q, d, el, wl)) if inside else (self._medium(q, (-d[0], -d[1]), el, wl), n_g)
                if d[0] * nrm[0] + d[1] * nrm[1] > 0:
                    nrm = (-nrm[0], -nrm[1])            # normal against the ray
                ci = -(d[0] * nrm[0] + d[1] * nrm[1])
                r = n1 / n2
                k = 1 - r * r * (1 - ci * ci)
                if k < 0:
                    d = (d[0] + 2 * ci * nrm[0], d[1] + 2 * ci * nrm[1])     # total internal reflection
                elif inside and inner_done.get(id(el), 0) < el.get("inner", 0):
                    # glass inner=N: follow the (partial) internal reflection N times before the ray leaves
                    # (a raindrop's rainbow path: refract in, reflect at the back, refract out)
                    inner_done[id(el)] = inner_done.get(id(el), 0) + 1
                    d = (d[0] + 2 * ci * nrm[0], d[1] + 2 * ci * nrm[1])
                else:
                    sk = math.sqrt(k)
                    d = (r * d[0] + (r * ci - sk) * nrm[0], r * d[1] + (r * ci - sk) * nrm[1])
                d = _unit(d)
            last_bend = len(pts) - 1
            p = (q[0] + d[0] * 1e-6, q[1] + d[1] * 1e-6)
        # run out to the edge of the view (or max_len)
        rem = max(0.0, max_len - total)
        far = 0.0
        for s in (rem,):
            far = s
        # clip to the view box (+ margin)
        bx0, bx1, by0, by1 = x0 - mx, x1 + mx, y0 - mx, y1 + mx
        ts = []
        if d[0] > 1e-12: ts.append((bx1 - p[0]) / d[0])
        if d[0] < -1e-12: ts.append((bx0 - p[0]) / d[0])
        if d[1] > 1e-12: ts.append((by1 - p[1]) / d[1])
        if d[1] < -1e-12: ts.append((by0 - p[1]) / d[1])
        s_edge = min([t for t in ts if t > 0], default=0.0)
        far = min(far, s_edge)
        pts.append((p[0] + d[0] * far, p[1] + d[1] * far))
        return pts, last_bend, False

    def _medium(self, q, d, skip, wl):
        probe = (q[0] + d[0] * 1e-4, q[1] + d[1] * 1e-4)
        for el in self.optics():
            if el is skip or el["type"] != "glass":
                continue
            if _inside(el["poly"], probe):
                n_g = el["n"]
                if wl is not None and el.get("disp"):
                    from .expr import f_cauchy
                    n_g = f_cauchy(wl, n_g, el["disp"])
                return n_g
        return 1.0

    def _ray_marks(self, P, env, p0, d, wl, st, ext, big=None):
        L = _num(P, "len", env, 80.0)
        pts, lb, _ = self.trace(p0, d, wl, L, big=big)
        marks = self._path_marks(pts, st, head=1 if _num(P, "arrow", env, 0.0) else 0)
        if wl is not None:
            for m in marks:
                m.add = True
        if ext and len(pts) >= 2:
            q = pts[-2] if lb == len(pts) - 2 else pts[lb]
            q2 = pts[-1]
            dd = _unit((q2[0] - q[0], q2[1] - q[1]))
            back = [q, (q[0] - dd[0] * ext, q[1] - dd[1] * ext)]
            if lb > 0:
                marks.append(Mark("path", pts=back, stroke=st["stroke"], width=max(1.0, st["width"] * 0.7), dash=5.0,
                                  alpha=st["alpha"] * 0.6 * (1.0 if st["draw"] >= 0.999 else 0.0), layer=st["layer"] - 0.05,
                                  glow=0.3))
        if _num(P, "arrows", env, 0.0):
            # small direction chevrons at the middle of each straight piece
            for a, b in zip(pts, pts[1:]):
                Ls = math.hypot(b[0] - a[0], b[1] - a[1])
                if Ls > 0.8:
                    m = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                    dd = ((b[0] - a[0]) / Ls * 0.05, (b[1] - a[1]) / Ls * 0.05)
                    marks.append(Mark("path", pts=[(m[0] - dd[0], m[1] - dd[1]), m], stroke=st["stroke"], width=st["width"],
                                      alpha=st["alpha"] * (1.0 if st["draw"] >= 0.6 else 0.0), head=1, layer=st["layer"]))
        return marks, pts

    def _k_ray(self, o, P, env):
        p0 = _pval(P, "from", env, _pval(P, "at", env, (-6.0, 0.0)))
        tgt = _pval(P, "to", env, None) or _pval(P, "through", env, None)
        if tgt is not None and _okp(tgt):
            d = (tgt[0] - p0[0], tgt[1] - p0[1])
        else:
            a = _num(P, "dir", env, 0.0)
            d = (math.cos(math.radians(a)), math.sin(math.radians(a)))
        if not (_okp(p0) and _okp(d)) or (d[0] == 0 and d[1] == 0):
            return None
        wl = _num(P, "wl", env, NAN)
        wl = wl if _ok(wl) else None
        col = P.get("color") if isinstance(P.get("color"), tuple) else (wavelength_rgb(wl) if wl else (1.0, 0.82, 0.4))
        st = self._style(P, env, stroke=col, width=2.0)
        st["stroke"] = col
        marks, pts = self._ray_marks(P, env, p0, d, wl, st, _num(P, "ext", env, 0.0) * (6.0 if _num(P, "ext", env, 0.0) == 1 else 1.0))
        marks += self._label(P, env, p0, st, (0.0, 0.35))
        return {"marks": marks, "path": pts, "anchor": p0, "verts": pts}

    def _k_beam(self, o, P, env):
        p0 = _pval(P, "from", env, _pval(P, "at", env, (-6.0, 0.0)))
        a = _num(P, "dir", env, 0.0)
        tgt = _pval(P, "to", env, None)
        if tgt is not None and _okp(tgt):
            a = math.degrees(math.atan2(tgt[1] - p0[1], tgt[0] - p0[0]))
        n = int(max(1, min(25, _num(P, "n", env, 5.0))))
        h = _num(P, "h", env, 0.0)
        spread = _num(P, "spread", env, 0.0)
        if not _okp(p0):
            return None
        wls = [None]
        wv = P.get("wl")
        if hasattr(wv, "eval"):
            w = wv.eval(env)
            wls = [x for x in (w if isinstance(w, list) else [w]) if _ok(x)][:9] or [None]
        if _num(P, "white", env, 0.0):
            wls = [410.0, 450.0, 490.0, 530.0, 570.0, 610.0, 660.0]
        ang = math.radians(a)
        dvec = (math.cos(ang), math.sin(ang))
        pv = (-dvec[1], dvec[0])
        st0 = self._style(P, env, stroke=(1.0, 0.82, 0.4), width=1.8)
        user_col = P.get("color") if isinstance(P.get("color"), tuple) else None
        ext = _num(P, "ext", env, 0.0) * (6.0 if _num(P, "ext", env, 0.0) == 1 else 1.0)
        marks, paths = [], []
        for k in range(n):
            u = 0.0 if n == 1 else (k / (n - 1) - 0.5)
            if spread:
                aa = ang + math.radians(spread) * u
                d = (math.cos(aa), math.sin(aa))
                p = p0
            else:
                d = dvec
                p = (p0[0] + pv[0] * h * u, p0[1] + pv[1] * h * u)
            for wl in wls:
                col = user_col or (wavelength_rgb(wl) if wl else st0["stroke"])
                st = dict(st0, stroke=col, alpha=st0["alpha"] * (0.85 if len(wls) > 1 else 1.0))
                mk, pts = self._ray_marks(P, env, p, d, wl, st, ext)
                marks += mk
                paths.append(pts)
        marks += self._label(P, env, p0, st0, (0.0, 0.4 + abs(h) / 2))
        return {"marks": marks, "path": paths[n // 2] if paths else [p0], "anchor": p0}

    def _optic_of(self, P):
        ref = P.get("lens") or P.get("mirror") or P.get("optic")
        if isinstance(ref, str):
            g = self._geom(ref)
            if g and g.get("optic") and g["optic"]["type"] in ("thinlens", "curvedmirror", "glass"):
                el = g["optic"]
                if el["type"] == "glass" and "f" in el and "ax" in el:
                    el = dict(el, type="thinlens")
                return el if el["type"] != "glass" else None
        for el in self.optics():
            if el["type"] in ("thinlens", "curvedmirror"):
                return el
        return None

    @staticmethod
    def _map_point(el, P):
        """paraxial image of point P through a thin lens / mirror -> (image point, virtual?)"""
        c, ax, f = el["c"], el["ax"], el["f"]
        pv = (-ax[1], ax[0])
        u = (P[0] - c[0]) * ax[0] + (P[1] - c[1]) * ax[1]
        v = (P[0] - c[0]) * pv[0] + (P[1] - c[1]) * pv[1]
        if el["type"] == "thinlens":
            side = -1.0 if u <= 0 else 1.0          # the object's side; light goes toward the other side
            do = abs(u)
            q = 1.0 / f - 1.0 / do if do and f and _ok(f) else NAN
            di = 1.0 / q if q else math.inf
            m = -di / do if do else NAN
            u2 = -side * di
        else:
            do = u                                  # in front of the mirror = +ax side
            q = 1.0 / f - 1.0 / do if do and f and _ok(f) else (-1.0 / do if do else NAN)
            di = 1.0 / q if q else math.inf
            m = -di / do if do else NAN
            u2 = di
        if not (_ok(di) and _ok(m)):
            return None, False
        return (c[0] + ax[0] * u2 + pv[0] * v * m, c[1] + ax[1] * u2 + pv[1] * v * m), di < 0

    def _k_image(self, o, P, env):
        el = self._optic_of(P)
        src = P.get("of")
        if el is None or not isinstance(src, str):
            return None
        g = self._geom(src)
        if not g or not g.get("path") or len(g["path"]) < 2:
            return None
        a, b = g["path"][0], g["path"][-1]
        ia, va = self._map_point(el, a)
        ib, vb = self._map_point(el, b)
        if ia is None or ib is None:
            return None
        x0, x1, y0, y1 = self.vb
        span = max(x1 - x0, y1 - y0) * 3
        if max(abs(ia[0]), abs(ia[1]), abs(ib[0]), abs(ib[1])) > span:
            return None                              # image at infinity (object at F)
        virtual = va or vb
        st = self._style(P, env, stroke=(1.0, 0.42, 0.84), width=3.0)
        if virtual and "dash" not in P:
            st["dash"] = 6.0
        marks = self._path_marks([ia, ib], st, head=1)
        if "label" not in P:
            P = dict(P, label="virtual image" if virtual else "real image")
        L = math.hypot(ib[0] - ia[0], ib[1] - ia[1]) or 1.0
        ux, uy = (ib[0] - ia[0]) / L, (ib[1] - ia[1]) / L
        marks += self._label(P, env, ib, st, (ux * 0.42, uy * 0.42 + (0.0 if abs(uy) > 0.5 else 0.3)))
        return {"marks": marks, "path": [ia, ib], "anchor": ib, "virtual": virtual}

    def _k_rays3(self, o, P, env):
        el = self._optic_of(P)
        tip = _pval(P, "from", env, None) or _pval(P, "at", env, None)
        if tip is None:
            src = P.get("of")
            if isinstance(src, str):
                g = self._geom(src)
                if g and g.get("path"):
                    tip = g["path"][-1]
        if el is None or tip is None or not _okp(tip):
            return None
        c, ax, f = el["c"], el["ax"], el["f"]
        pv = (-ax[1], ax[0])
        u = (tip[0] - c[0]) * ax[0] + (tip[1] - c[1]) * ax[1]
        v = (tip[0] - c[0]) * pv[0] + (tip[1] - c[1]) * pv[1]
        img, virtual = self._map_point(el, tip)
        cols = [(1.0, 0.82, 0.4), (1.0, 0.62, 0.26), (1.0, 0.55, 0.75)]
        user_col = P.get("color") if isinstance(P.get("color"), tuple) else None
        st0 = self._style(P, env, width=2.0)
        n = int(max(1, min(3, _num(P, "n", env, 3.0))))
        toward = (-ax[0], -ax[1]) if (el["type"] == "curvedmirror" or u > 0) else ax   # from the object toward the element
        if el["type"] == "curvedmirror":
            toward = (-ax[0], -ax[1]) if u > 0 else ax
        dirs = [toward]                                            # 1: parallel to the axis
        dirs.append((c[0] - tip[0], c[1] - tip[1]) if el["type"] == "thinlens"
                    else (c[0] + ax[0] * 2 * f - tip[0], c[1] + ax[1] * 2 * f - tip[1]))   # 2: centre / centre of curvature
        if el["type"] == "thinlens":
            F = (c[0] - toward[0] * f, c[1] - toward[1] * f)       # the focal point on the object's side (far side if f<0)
        else:
            F = (c[0] + ax[0] * f, c[1] + ax[1] * f)
        dF = (F[0] - tip[0], F[1] - tip[1])
        if dF[0] * toward[0] + dF[1] * toward[1] < 0:
            dF = (-dF[0], -dF[1])                                 # F behind the object: the ray comes FROM F through it
        dirs.append(dF)
        # a ray aimed away from the element (e.g. the centre-of-curvature ray from beyond C) is flipped
        dirs = [d if d[0] * toward[0] + d[1] * toward[1] >= 0 else (-d[0], -d[1]) for d in dirs]
        ext = 0.0
        if virtual and img is not None:
            ext = math.hypot(img[0] - tip[0], img[1] - tip[1]) + abs(u) + 1.0
        marks, path = [], []
        draw = st0["draw"]
        stagger = _num(P, "stagger", env, 0.0)
        for k, d in enumerate(dirs[:n]):
            if math.hypot(*d) < 1e-9:
                continue
            col = user_col or cols[k]
            st = dict(st0, stroke=col)
            if stagger and draw < 0.999:
                st["draw"] = max(0.0, min(1.0, draw * (1 + stagger * (n - 1)) - stagger * k))
            mk, pts = self._ray_marks(P, env, tip, d, None, st, ext, big=el)
            marks += mk
            if not path:
                path = pts
        return {"marks": marks, "path": path, "anchor": tip}

    # ---------------------------------------------------------------- all marks
    def marks(self):
        out = []
        now = self.now
        doc = self.doc
        live = list(doc.objs.values())
        for o in live + doc.dead:
            if o.kind == "group":
                continue
            fade = 1.0
            if o.dying is not None:
                fade = 1.0 - _smooth((now - o.dying) / FADE_OUT)
                if fade <= 0:
                    continue
                g = self._build(o)
            else:
                if now - o.born < FADE_IN:
                    fade = _smooth((now - o.born) / FADE_IN)
                g = self._geom(o.id)
            if not g:
                continue
            for m in g["marks"]:
                if fade < 1.0:
                    m.alpha *= fade
                m.order = o.order
                out.append(m)
        out.sort(key=lambda m: (m.layer, m.order))
        return out


def _morph_geom(ga, gb, u):
    ma, mb = ga["marks"], gb["marks"]
    if len(ma) != len(mb):
        return gb
    out = []
    for a, b in zip(ma, mb):
        if a.kind != b.kind or (a.pts is None) or (b.pts is None):
            out.append(b)
            continue
        pa, pb = a.pts, b.pts
        if len(pa) != len(pb):
            pa = _resample(pa, len(pb))
        if pa is None:
            out.append(b)
            continue
        b.pts = [(x0 + (x1 - x0) * u, y0 + (y1 - y0) * u) for (x0, y0), (x1, y1) in zip(pa, pb)]
        if isinstance(a.stroke, tuple) and isinstance(b.stroke, tuple):
            b.stroke = _blend(a.stroke, b.stroke, u)
        if isinstance(a.fill, tuple) and isinstance(b.fill, tuple):
            b.fill = _blend(a.fill, b.fill, u)
        b.width = a.width + (b.width - a.width) * u
        b.alpha = a.alpha + (b.alpha - a.alpha) * u
        b.size = a.size + (b.size - a.size) * u
        out.append(b)
    g = dict(gb)
    g["marks"] = out
    if _okp(ga.get("anchor")) and _okp(gb.get("anchor")):
        g["anchor"] = _blend(ga["anchor"], gb["anchor"], u)
    if ga.get("path") and gb.get("path") and len(ga["path"]) == len(gb["path"]):
        g["path"] = [(x0 + (x1 - x0) * u, y0 + (y1 - y0) * u) for (x0, y0), (x1, y1) in zip(ga["path"], gb["path"])]
    return g


def _nice(x):
    if not (x > 0) or not math.isfinite(x):
        return 1.0
    e = 10 ** math.floor(math.log10(x))
    for m in (1, 2, 2.5, 5, 10):
        if m * e >= x:
            return m * e
    return 10 * e


def _map_geom(g, mp, rigid=False):
    """data -> world for an object drawn on an axes box; rigid = only its anchor moves (dots keep their size)"""
    (xa, xb), (ya, yb), (bx0, bx1, by0, by1) = mp
    fx = (bx1 - bx0) / (xb - xa)
    fy = (by1 - by0) / (yb - ya)
    ap = lambda p: (bx0 + (p[0] - xa) * fx, by0 + (p[1] - ya) * fy)
    if rigid and _okp(g.get("anchor")):
        a = g["anchor"]
        b = ap(a)
        dx, dy = b[0] - a[0], b[1] - a[1]
        ap = lambda p, dx=dx, dy=dy: (p[0] + dx, p[1] + dy)
    for m in g["marks"]:
        if m.pts:
            m.pts = [ap(p) for p in m.pts]
    g = dict(g)
    g["path"] = [ap(p) for p in g.get("path") or []]
    if _okp(g.get("anchor")):
        g["anchor"] = ap(g["anchor"])
    return g


def _resample(pts, n):
    if not pts or n < 1:
        return None
    if len(pts) == 1:
        return [pts[0]] * n
    return [point_along(pts, k / max(1, n - 1)) for k in range(n)]


def _transform_geom(g, tf):
    def ap(p, tf=tf):
        if tf[0] == "chain":
            return ap(ap(p, tf[1]), tf[2])
        at, rot, sc = tf
        cr, sr = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        x, y = p[0] * sc, p[1] * sc
        return (at[0] + x * cr - y * sr, at[1] + x * sr + y * cr)

    def scale_of(tf):
        if tf[0] == "chain":
            return scale_of(tf[1]) * scale_of(tf[2])
        return tf[2]
    k = scale_of(tf)
    for m in g["marks"]:
        if m.pts:
            m.pts = [ap(p) for p in m.pts]
        if m.kind == "text":
            m.size *= k
    g = dict(g)
    g["path"] = [ap(p) for p in g.get("path") or []]
    g["anchor"] = ap(g["anchor"]) if _okp(g.get("anchor")) else g.get("anchor")
    if "radius" in g:
        g["radius"] = g["radius"] * k
    return g


_ONCE_RE = re.compile(r"^\s*(lin|ramp)\(\s*[-+]?[\d.]+\s*,\s*[-+]?[\d.]+\s*\)\s*$")


def _num(P, k, env, default):
    v = P.get(k)
    if v is None:
        return default
    if hasattr(v, "eval"):
        v = v.eval(env)
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        v = float(v)
        return v if math.isfinite(v) or v == math.inf or v == -math.inf else default
    return default


def _pval(P, k, env, default):
    v = P.get(k)
    if v is None:
        return default
    if hasattr(v, "eval"):
        v = v.eval(env)
    if isinstance(v, tuple) and len(v) == 2 and _ok(float(v[0])) and _ok(float(v[1])):
        return (float(v[0]), float(v[1]))
    return default


def _plist(P, k, env):
    v = P.get(k)
    if v is None:
        return []
    if hasattr(v, "eval"):
        v = v.eval(env)
    if not isinstance(v, list):
        return []
    return [p for p in v if _okp(p)]


_EMS = {}


def _em(line):
    """width of a line of text at size 1 (world units per unit of text size), measured with the renderer's font"""
    v = _EMS.get(line)
    if v is None:
        try:
            from .render import _font, _plain
            v = _font(100).getlength(_plain(line)) / 100.0 * (100.0 / 100.0)
        except Exception:  # noqa: BLE001
            v = 0.62 * len(line)
        if len(_EMS) > 2000:
            _EMS.clear()
        _EMS[line] = v
    return v


def _wrap_chars(s, per):
    """word-wrap to lines of at most ~per characters (explicit newlines kept)"""
    out = []
    for ln in str(s).split("\n"):
        cur = ""
        for w in ln.split(" "):
            cand = (cur + " " + w) if cur else w
            if cur and len(cand) > per:
                out.append(cur)
                cur = w
            else:
                cur = cand
        out.append(cur)
    return "\n".join(out)


_BGC = (0.027, 0.035, 0.06)


def _premix(col, a):
    """a colour at opacity a over the TV's background, as an opaque colour"""
    return tuple(b + (c - b) * a for c, b in zip(col, _BGC))


def _rrect(c, hw, hh, rr):
    loc = []
    for cx, cy, a0 in ((hw - rr, hh - rr, 0), (-hw + rr, hh - rr, 90), (-hw + rr, -hh + rr, 180), (hw - rr, -hh + rr, 270)):
        loc += _circle((cx, cy), rr, a0=a0, a1=a0 + 90, n=5)
    return [(c[0] + x, c[1] + y) for x, y in loc]


def _in_box(q, c, hw, hh):
    return abs(q[0] - c[0]) <= hw and abs(q[1] - c[1]) <= hh


def _box_exit(a, b, hw, hh):
    """where the segment from box centre a toward b leaves the box (half sizes hw, hh)"""
    dx, dy = b[0] - a[0], b[1] - a[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return a
    s = min(hw / abs(dx) if abs(dx) > 1e-9 else math.inf, hh / abs(dy) if abs(dy) > 1e-9 else math.inf)
    return (a[0] + dx * s, a[1] + dy * s)


def _fmtnum(x):
    if abs(x) >= 1e4:                 # compact: 250k, 1.5M (a 7-digit tick number ran over the axis)
        for d, suf in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
            if abs(x) >= d:
                return f"{x / d:.3g}{suf}".replace("-", "−")
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x))).replace("-", "−")
    return (f"{x:.2g}" if abs(x) < 1 else f"{x:.1f}").replace("-", "−")


_INTERP = re.compile(r"\{([^{}]+?)(?::(\.?\d{0,2}[fge%]?))?\}")


_ICACHE = {}


def interp(s, env):
    """'{expr}' / '{expr:.2f}' inside text -> its value now"""
    if "{" not in s:
        return s
    from .expr import compile_expr, ExprError

    def rep(m):
        src, spec = m.group(1), m.group(2)
        e = _ICACHE.get(src)
        if e is None:
            try:
                e = compile_expr(src)
            except (ExprError, RecursionError):
                e = False
            if len(_ICACHE) > 2000:
                _ICACHE.clear()
            _ICACHE[src] = e
        if e is False:
            return "?"
        v = e.eval(env)
        if isinstance(v, tuple):
            return "(" + ", ".join(_fmtv(x, spec) for x in v) + ")"
        if isinstance(v, (int, float)):
            return _fmtv(float(v), spec)
        if isinstance(v, str):
            return v
        return "?"
    return _INTERP.sub(rep, s)


def _fmtv(v, spec):
    if not math.isfinite(v):
        return "∞" if v == math.inf else "−∞" if v == -math.inf else "?"
    if spec:
        try:
            if spec[-1] not in "fge%":
                spec = spec + "f"
            return format(v, spec).replace("-", "−")
        except (ValueError, TypeError):
            pass
    if abs(v - round(v)) < 1e-9 and abs(v) < 1e9:
        return str(int(round(v))).replace("-", "−")
    return f"{v:.3g}".replace("-", "−")
