"""boardbrain: a tiny vector-figure DSL -> hand-traceable polylines.

The model writes math coordinates (y UP) inside a `view`; the figure is fitted (equal aspect unless `stretch`)
into the region the placer gives it. Commands, one per ';' or newline, drawn in the order written:

  view x0 x1 y0 y1 [stretch]          math window (default: fitted to everything drawn)
  axes [xlabel] [ylabel] [ticks=STEP] [labels=STEP] [grid=STEP]   axes through the origin (clamped into view)
  numberline A B STEP                 a number line with ticks + numbers (sets the view if none)
  line x1 y1 x2 y2 [dashed]           arrow x1 y1 x2 y2 [dashed] ["label"]   (also: ray, darrow = double-headed)
  polyline x y x y ... [closed] [dashed]        polygon x y x y ...        rect x y w h
  circle cx cy r       ellipse cx cy rx ry       arc cx cy r a0 a1 [arrow]  (degrees, counter-clockwise)
  curve x y x y ... [closed]          smooth curve through the points (sketches)
  bezier x0 y0 x1 y1 x2 y2 x3 y3      quad x0 y0 x1 y1 x2 y2
  plot "expr in x" [x0 x1] [dashed]   function graph (sin cos tan exp log ln sqrt abs pi e, ^ for power)
  param "x(t)" "y(t)" t0 t1           parametric curve
  dot x y            point x y "A"    (dot + label)
  label x y "text" [above|below|left|right|center] [size=0.8]   text, $..$ math allowed
  angle x y a0 a1 [r] ["θ"]           angle arc at a vertex          rightangle x y a0 [s]
  vector x1 y1 x2 y2 ["label"]        same as arrow
Bad commands are skipped with a warning, never raised.
"""
from __future__ import annotations

import ast
import math
import operator
import random
import re
import shlex

from . import geom
from .font import HandFont
from .mathlayout import layout_math, split_text_math, looks_like_tex

# ----------------------------------------------------------------- safe expression evaluation

_FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "arcsin": math.asin, "arccos": math.acos, "arctan": math.atan, "sinh": math.sinh, "cosh": math.cosh,
    "tanh": math.tanh, "exp": math.exp, "log": math.log, "ln": math.log, "log10": math.log10, "log2": math.log2,
    "sqrt": math.sqrt, "abs": abs, "floor": math.floor, "ceil": math.ceil, "sec": lambda v: 1 / math.cos(v),
    "csc": lambda v: 1 / math.sin(v), "cot": lambda v: 1 / math.tan(v), "sign": lambda v: (v > 0) - (v < 0),
    "min": min, "max": max, "pow": pow,
}
_CONST = {"pi": math.pi, "e": math.e, "π": math.pi}
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _prep_expr(s: str) -> str:
    s = s.strip().strip('"').strip("'")
    s = s.replace("^", "**").replace("π", "pi").replace("·", "*").replace("×", "*").replace("−", "-")
    s = re.sub(r"^\s*[a-zA-Z]\s*\(\s*[a-z]\s*\)\s*=", "", s)   # "f(x) = ..."
    s = re.sub(r"^\s*y\s*=", "", s)
    s = re.sub(r"(\d)\s*([a-zA-Z(])", r"\1*\2", s)               # 2x -> 2*x, 3(x+1)
    s = re.sub(r"\)\s*([a-zA-Z0-9(])", r")*\1", s)               # (x+1)(x-1)
    s = re.sub(r"\b(sin|cos|tan|exp|ln|log|sqrt|abs|sinh|cosh|tanh)\s+([a-z0-9.]+)", r"\1(\2)", s)
    s = s.replace("**-*", "**-")
    return s


def compile_expr(src: str, var="x"):
    tree = ast.parse(_prep_expr(src), mode="eval")

    def ev(node, env):
        if isinstance(node, ast.Expression):
            return ev(node.body, env)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in _CONST:
                return _CONST[node.id]
            raise ValueError("unknown name " + node.id)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            return _BIN[type(node.op)](ev(node.left, env), ev(node.right, env))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
            return _UN[type(node.op)](ev(node.operand, env))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            return _FUNCS[node.func.id](*[ev(a, env) for a in node.args])
        raise ValueError("unsupported expression")

    # validate structure once
    def check(node):
        for ch in ast.walk(node):
            if isinstance(ch, (ast.Attribute, ast.Subscript, ast.Lambda, ast.Dict, ast.List, ast.Starred)):
                raise ValueError("unsupported expression")
    check(tree)

    def f(v):
        try:
            r = ev(tree, {var: v})
            if isinstance(r, complex) or r != r or abs(r) == float("inf"):
                return None
            return float(r)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            return None
    return f


# ----------------------------------------------------------------- parsing

def split_commands(src: str):
    """Split on ';' and newlines that are outside quotes."""
    cmds, cur, q = [], [], None
    for ch in src:
        if q:
            cur.append(ch)
            if ch == q:
                q = None
            continue
        if ch in "\"'":
            # apostrophes inside words (don't) are not quotes
            if ch == "'" and cur and cur[-1].isalnum():
                cur.append(ch)
                continue
            q = ch
            cur.append(ch)
            continue
        if ch in ";\n":
            if "".join(cur).strip():
                cmds.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if "".join(cur).strip():
        cmds.append("".join(cur).strip())
    return cmds


_TOKRE = re.compile(r'"([^"]*)"?|\'([^\']*)\'|(\S+)')


def _tokens(cmd):
    """-> [(text, quoted)]"""
    out = []
    for m in _TOKRE.finditer(cmd):
        if m.group(3) is not None:
            out.append((m.group(3), False))
        else:
            out.append((m.group(1) if m.group(1) is not None else m.group(2), True))
    return out


def _num(s):
    s = s.strip().rstrip(",").lstrip("(").rstrip(")")
    try:
        return float(s)
    except ValueError:
        if not re.fullmatch(r"[0-9.+\-*/()^ a-z√π]*", s) or not s:
            raise ValueError(s)
        f = compile_expr(s, var="_novar_")
        v = f(0.0)
        if v is None:
            raise ValueError(s)
        return v


def parse(src: str):
    """-> (commands [(name, nums, strs, opts, flags)], warnings)"""
    out, warns = [], []
    for raw in split_commands(src):
        toks = _tokens(raw)
        if not toks:
            continue
        name = toks[0][0].lower()
        nums, strs, opts, flags = [], [], {}, set()
        for t, quoted in toks[1:]:
            if quoted:
                strs.append(t)
                continue
            if "=" in t and re.match(r"^[a-z]+=", t):
                k, v = t.split("=", 1)
                opts[k] = v.strip('"')
                continue
            if name in ("plot", "param") and re.search(r"[a-z]", t) and t.lower() not in ("dashed", "dotted") \
                    and not re.fullmatch(r"[\[\]()0-9.,+\-*/ ]*(pi|π)?[\[\]()0-9.,+\-*/ ]*", t):
                strs.append(t)
                continue
            for piece in re.split(r"[,()\[\]]+", t) if not re.search(r"[a-z]\(", t) else [t]:
                if not piece:
                    continue
                try:
                    nums.append(_num(piece))
                except (ValueError, SyntaxError):
                    if piece.lower() in ("dashed", "closed", "arrow", "stretch", "above", "below", "left", "right",
                                         "center", "filled", "grid", "both", "dotted"):
                        flags.add(piece.lower())
                    else:
                        strs.append(piece)
        out.append((name, nums, strs, opts, flags))
    return out, warns


# ----------------------------------------------------------------- rendering


class _Canvas:
    """Collects strokes in math coordinates; mapped to the region at the end."""

    def __init__(self):
        self.items = []    # ("poly", pts_math, dashed) | ("label", x, y, text, pos, scale) | ("head", tip, frm)

    def poly(self, pts, dashed=False):
        pts = [p for p in pts if p is not None]
        if len(pts) >= 2:
            self.items.append(("poly", pts, dashed))

    def head(self, tip, frm):
        self.items.append(("head", tip, frm))

    def label(self, x, y, text, pos="center", scale=1.0):
        self.items.append(("label", x, y, text, pos, scale))


def _catmull(pts, closed=False, n=10):
    if len(pts) < 3:
        return pts
    P = list(pts)
    if closed:
        P = [P[-1]] + P + [P[0], P[1]]
    else:
        P = [P[0]] + P + [P[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(n):
            t = k / n
            t2, t3 = t * t, t * t * t
            out.append(tuple(0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2 +
                                    (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3) for j in range(2)))
    out.append(P[-2])
    return out


def render_figure(src: str, font: HandFont, region_w: float, region_h: float, text_size: float, seed: int = 0,
                  label_min: float = 0.065):
    """Render a figure program into a region_w x region_h box (origin top-left, y down, board units).

    Returns {"strokes", "w", "h", "warnings"} (w/h = the used extent, <= region)."""
    cmds, warns = parse(src)
    cv = _Canvas()
    view = None
    stretch = False
    pending_axes = []
    for name, nums, strs, opts, flags in cmds:
        try:
            if name == "view" and len(nums) >= 4:
                view = (nums[0], nums[1], nums[2], nums[3])
                stretch = "stretch" in flags
            elif name in ("axes", "axis"):
                pending_axes.append((nums, strs, opts, flags, len(cv.items)))
                cv.items.append(("axes_slot",))
            elif name == "numberline" and len(nums) >= 2:
                a, b = nums[0], nums[1]
                step = nums[2] if len(nums) > 2 else _nice_step(b - a)
                cv.items.append(("numberline", a, b, step, flags))
            elif name in ("line", "ray", "segment") and len(nums) >= 4:
                cv.poly([(nums[0], nums[1]), (nums[2], nums[3])], "dashed" in flags or "dotted" in flags)
                if strs:
                    cv.label((nums[0] + nums[2]) / 2, (nums[1] + nums[3]) / 2, strs[0], _pos_from(flags, "above"))
            elif name in ("arrow", "vector", "darrow") and len(nums) >= 4:
                a, b = (nums[0], nums[1]), (nums[2], nums[3])
                cv.poly([a, b], "dashed" in flags)
                cv.head(b, a)
                if name == "darrow" or "both" in flags:
                    cv.head(a, b)
                if strs:
                    cv.label((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, strs[0], _pos_from(flags, "above"))
            elif name in ("polyline", "polygon", "path") and len(nums) >= 4:
                pts = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
                if name == "polygon" or "closed" in flags:
                    pts.append(pts[0])
                cv.poly(pts, "dashed" in flags)
            elif name == "rect" and len(nums) >= 4:
                x, y, w, h = nums[:4]
                cv.poly([(x, y + h), (x + w, y + h), (x + w, y), (x, y), (x, y + h)], "dashed" in flags)
            elif name == "circle" and len(nums) >= 3:
                cv.items.append(("ellipse", nums[0], nums[1], nums[2], nums[2], 90, 450, "dashed" in flags, False))
            elif name == "ellipse" and len(nums) >= 4:
                cv.items.append(("ellipse", nums[0], nums[1], nums[2], nums[3], 90, 450, "dashed" in flags, False))
            elif name == "arc" and len(nums) >= 5:
                cv.items.append(("ellipse", nums[0], nums[1], nums[2], nums[2], nums[3], nums[4], "dashed" in flags,
                                 "arrow" in flags))
            elif name in ("curve", "spline", "sketch") and len(nums) >= 4:
                pts = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
                cv.poly(_catmull(pts, "closed" in flags), "dashed" in flags)
            elif name == "bezier" and len(nums) >= 8:
                cv.poly(geom.bezier(*[(nums[i], nums[i + 1]) for i in range(0, 8, 2)], n=32), "dashed" in flags)
            elif name == "quad" and len(nums) >= 6:
                cv.poly(geom.quad(*[(nums[i], nums[i + 1]) for i in range(0, 6, 2)], n=24), "dashed" in flags)
            elif name == "plot" and strs:
                f = compile_expr(strs[0] if len(strs) == 1 else " ".join(strs))
                rng = nums[:2] if len(nums) >= 2 else None
                cv.items.append(("plot", f, rng, "dashed" in flags))
            elif name == "param" and len(strs) >= 2 and len(nums) >= 2:
                fx, fy = compile_expr(strs[0], "t"), compile_expr(strs[1], "t")
                n = 160
                pts = []
                for i in range(n + 1):
                    t = nums[0] + (nums[1] - nums[0]) * i / n
                    x, y = fx(t), fy(t)
                    if x is not None and y is not None:
                        pts.append((x, y))
                cv.poly(pts, "dashed" in flags)
            elif name == "dot" and len(nums) >= 2:
                cv.items.append(("dot", nums[0], nums[1]))
            elif name == "point" and len(nums) >= 2:
                cv.items.append(("dot", nums[0], nums[1]))
                if strs:
                    cv.label(nums[0], nums[1], strs[0], _pos_from(flags, "above right"))
            elif name in ("label", "text") and len(nums) >= 2 and strs:
                sc = float(opts.get("size", 1.0)) if _isnum(opts.get("size", "1")) else 1.0
                cv.label(nums[0], nums[1], " ".join(strs), _pos_from(flags, "center"), sc)
            elif name == "angle" and len(nums) >= 4:
                r = nums[4] if len(nums) > 4 else None
                cv.items.append(("angle", nums[0], nums[1], nums[2], nums[3], r, strs[0] if strs else None))
            elif name == "rightangle" and len(nums) >= 3:
                cv.items.append(("rightangle", nums[0], nums[1], nums[2], nums[3] if len(nums) > 3 else None))
            else:
                warns.append("skipped: " + name)
        except Exception as e:  # noqa: BLE001 — a bad command must never sink the figure
            warns.append("bad %s: %s" % (name, e))

    # ---- view: the model's window, tightened to what is actually drawn (a triangle in a roomy `view` used to come
    # out a third of its region); axes / number lines span their view, so graphs keep theirs
    names = {c[0] for c in cmds}
    ext = _content_ext(cv, bool(pending_axes))
    if view is None:
        view = ext
    elif ext is not None and not (names & {"axes", "axis", "numberline"}):
        vx0, vx1, vy0, vy1 = view
        if vx1 > vx0 and vy1 > vy0:
            cx0, cx1 = max(vx0, ext[0]), min(vx1, ext[1])
            cy0, cy1 = max(vy0, ext[2]), min(vy1, ext[3])
            if cx1 - cx0 > (vx1 - vx0) * 0.05 and cy1 - cy0 > (vy1 - vy0) * 0.05:
                px, py = (cx1 - cx0) * 0.04, (cy1 - cy0) * 0.04
                view = (max(vx0, cx0 - px), min(vx1, cx1 + px), max(vy0, cy0 - py), min(vy1, cy1 + py))
    if view is None:
        view = (0.0, 10.0, 0.0, 10.0)
    vx0, vx1, vy0, vy1 = view
    if vx1 <= vx0:
        vx0, vx1 = vx1 - 1, vx0 + 1
    if vy1 <= vy0:
        vy0, vy1 = vy1 - 1, vy0 + 1
    view = (vx0, vx1, vy0, vy1)

    graph_like = bool(names & {"axes", "axis", "plot", "param", "numberline"}) and not (
        names & {"circle", "ellipse", "arc", "polygon", "rect", "angle", "rightangle", "curve", "bezier"})

    def scales(aw, ah):
        sx, sy = aw / (vx1 - vx0), ah / (vy1 - vy0)
        if not stretch:
            s = min(sx, sy)
            if graph_like:
                # a function graph may use the region's shape (units on x and y are independent), up to 2.5:1
                sx, sy = min(sx, s * 2.5), min(sy, s * 2.5)
            else:
                sx, sy = s, s
        return sx, sy

    # first guess: the region less a label rim; then refit to the measured ink (labels included) so the drawing
    # fills the region — labels keep their size, the geometry grows or shrinks around them
    m = text_size * 0.9
    sx, sy = scales(max(region_w - 2 * m, region_w * 0.4), max(region_h - 2 * m, region_h * 0.4))
    best = None
    gcache = {}               # label glyphs per (text, size): the passes re-place the same labels
    for _ in range(5):
        strokes, spans = _draw_pass(cv, pending_axes, view, sx, sy, font, text_size, label_min, seed, src, gcache)
        if not strokes:
            best = (strokes, spans, (0.0, 0.0, 0.0, 0.0), 0.0)
            break
        bb = geom.bbox(strokes)
        W, H = bb[2] - bb[0], bb[3] - bb[1]
        fit = min(1.0, region_w / max(W, 1e-9), region_h / max(H, 1e-9))
        # what the drawing's scale will be once the placer has shrunk an overflowing figure into its region
        eff = math.sqrt(sx * sy) * fit
        if best is None or eff > best[3] * 1.0001:
            best = (strokes, spans, bb, eff)
        gw, gh = (vx1 - vx0) * sx, (vy1 - vy0) * sy      # geometry extent; the rest is label overhang
        kx = (region_w - (W - gw)) / max(gw, 1e-9)
        ky = (region_h - (H - gh)) / max(gh, 1e-9)
        if stretch or graph_like:
            kx, ky = max(0.5, min(4.0, kx)), max(0.5, min(4.0, ky))
            nsx, nsy = sx * kx, sy * ky
            if not stretch:
                nsx, nsy = min(nsx, nsy * 2.5), min(nsy, nsx * 2.5)
        else:
            k = max(0.5, min(4.0, min(kx, ky)))
            nsx, nsy = sx * k, sy * k
        if abs(nsx / sx - 1) < 0.03 and abs(nsy / sy - 1) < 0.03:
            break
        if fit < 1.0:         # overflow: aim a little inside so the next pass fits
            nsx, nsy = nsx * 0.98, nsy * 0.98
        sx, sy = nsx, nsy
    strokes, spans, bb = best[0], best[1], best[2]
    dx, dy = -bb[0], -bb[1]
    strokes = geom.transform(strokes, 1, 1, dx, dy)
    keep, remap = [], {}
    for i, s_ in enumerate(strokes):
        if len(s_) >= 2:
            remap[i] = len(keep)
            keep.append(s_)
    labels = []
    for sp in spans:
        idx = [remap[i] for i in range(sp["strokes"][0], sp["strokes"][1]) if i in remap]
        if not idx:
            continue
        L = dict(sp, strokes=[idx[0], idx[-1] + 1])
        if sp.get("leader") is not None:
            L["leader"] = remap.get(sp["leader"])
        if sp.get("anchor") is not None:
            L["anchor"] = (sp["anchor"][0] + dx, sp["anchor"][1] + dy)
        labels.append(L)
    return {"strokes": keep, "w": bb[2] - bb[0], "h": bb[3] - bb[1], "warnings": warns, "labels": labels}


def _content_ext(cv, has_axes):
    """Math-coordinate extent (x0, x1, y0, y1) of what the figure draws, padded 8%; None if nothing."""
    xs, ys = [], []
    for it in cv.items:
        if it[0] == "poly":
            xs += [p[0] for p in it[1]]
            ys += [p[1] for p in it[1]]
        elif it[0] in ("label", "dot", "angle", "rightangle"):
            xs.append(it[1])
            ys.append(it[2])
        elif it[0] == "numberline":
            xs += [it[1], it[2]]
            ys += [0.0]
        elif it[0] == "head":
            xs.append(it[1][0])
            ys.append(it[1][1])
        elif it[0] == "ellipse":
            xs += [it[1] - it[3], it[1] + it[3]]
            ys += [it[2] - it[4], it[2] + it[4]]
    for it in cv.items:
        if it[0] == "plot":
            rng = it[2] or ((min(xs), max(xs)) if xs else (-5, 5))
            xs += list(rng)
            for i in range(41):
                v = it[1](rng[0] + (rng[1] - rng[0]) * i / 40)
                if v is not None:
                    ys.append(v)
    if has_axes:
        xs.append(0.0)
        ys.append(0.0)
    xs = [x for x in xs if math.isfinite(x)]
    ys = [y for y in ys if math.isfinite(y)]
    if not xs and not ys:
        return None
    if not xs:
        xs = [0, 10]
    if not ys:
        ys = [0, 10]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if x1 - x0 < 1e-9:
        x0, x1 = x0 - 1, x1 + 1
    if y1 - y0 < 1e-9:
        y0, y1 = y0 - 1, y1 + 1
    px, py = (x1 - x0) * 0.08, (y1 - y0) * 0.08
    return (x0 - px, x1 + px, y0 - py, y1 + py)


def _draw_pass(cv, pending_axes, view, sx, sy, font, text_size, label_min, seed, src, gcache=None):
    """Draw every command at scale (sx, sy) (math -> board units, origin at the view's top-left), then place the
    labels. Returns (strokes, label spans)."""
    vx0, vx1, vy0, vy1 = view
    pending_axes = list(pending_axes)

    def M(p):
        return ((p[0] - vx0) * sx, (vy1 - p[1]) * sy)

    rnd = random.Random(seed * 7777 + len(src))
    step = text_size * 0.12
    strokes = []

    def add(pts_board, dashed=False, bow=True):
        if len(pts_board) < 2:
            return
        if dashed:
            for d in geom.dashed(pts_board, text_size * 0.35, text_size * 0.25):
                strokes.append(geom.resample(d, step))
            return
        p = geom.resample(pts_board, step)
        if bow and len(p) > 4 and geom.length(p) > text_size * 2:
            p = geom.bow(p, amp=min(text_size * 0.06, geom.length(p) * 0.008), rnd=rnd)
        strokes.append(p)

    def add_label(x, y, text, pos, scale=1.0, board=False, role="label", spread=None):
        bx, by = (x, y) if board else M((x, y))
        size = text_size * max(0.6, min(1.5, scale))
        fixed = board and role == "tick"
        strokes.append({"label": True, "text": text, "anchor": (bx, by), "pos": pos, "size": size,
                        "fixed": fixed, "role": role, "spread": spread,
                        "minsize": size if fixed else max(min(size, label_min), size * 0.8)})

    for it in cv.items:
        kind = it[0]
        if kind == "poly":
            add([M(p) for p in it[1]], it[2])
        elif kind == "head":
            tip, frm = M(it[1]), M(it[2])
            add(geom.arrowhead(tip, frm, text_size * 0.45), bow=False)
        elif kind == "ellipse":
            _, cx, cy, rx, ry, a0, a1, dashed, arrow = it
            n = max(16, int(abs(a1 - a0) / 6))
            pts = [(cx + rx * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
                    cy + ry * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]
            bp = [M(p) for p in pts]
            add(bp, dashed, bow=False)
            if arrow and len(bp) > 2:
                add(geom.arrowhead(bp[-1], bp[-3], text_size * 0.4), bow=False)
        elif kind == "plot":
            f, rng, dashed = it[1], it[2], it[3]
            a, b = rng if rng else (vx0, vx1)
            n = 240
            seg = []
            prev = None
            span = vy1 - vy0
            for i in range(n + 1):
                x = a + (b - a) * i / n
                y = f(x)
                ok = y is not None and vy0 - span * 0.02 <= y <= vy1 + span * 0.02
                if ok and prev is not None and abs(y - prev) > span * 0.5:
                    ok_jump = False
                else:
                    ok_jump = True
                if ok and ok_jump:
                    seg.append(M((x, max(vy0, min(vy1, y)))))
                else:
                    if len(seg) > 1:
                        add(seg, dashed, bow=False)
                    seg = [M((x, max(vy0, min(vy1, y))))] if ok else []
                prev = y
            if len(seg) > 1:
                add(seg, dashed, bow=False)
        elif kind == "dot":
            c = M((it[1], it[2]))
            r = text_size * 0.09
            strokes.append(geom.ellipse(c[0], c[1], r, r, 0, 360, 10))
            strokes.append(geom.ellipse(c[0], c[1], r * 0.45, r * 0.45, 0, 360, 8))
        elif kind == "label":
            add_label(it[1], it[2], it[3], it[4], it[5], role=it[6] if len(it) > 6 else "label")
        elif kind == "angle":
            _, x, y, a0, a1, r, lab = it
            c = M((x, y))
            rr = text_size * 0.9 if r is None else r * sx
            n = max(8, int(abs(a1 - a0) / 6))
            pts = [(c[0] + rr * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
                    c[1] - rr * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]
            add(pts, bow=False)
            if lab:
                am = math.radians((a0 + a1) / 2)
                # an angle's label sits on the bisector just outside its arc (further out / turned a little if busy)
                add_label(c[0], c[1], lab, "center", 0.85, board=True, role="angle",
                          spread=(am, rr, abs(math.radians(a1 - a0))))
        elif kind == "rightangle":
            _, x, y, a0, s = it
            c = M((x, y))
            ss = text_size * 0.5 if s is None else s * sx
            u = (math.cos(math.radians(a0)), -math.sin(math.radians(a0)))
            v = (math.cos(math.radians(a0 + 90)), -math.sin(math.radians(a0 + 90)))
            add([(c[0] + u[0] * ss, c[1] + u[1] * ss), (c[0] + (u[0] + v[0]) * ss, c[1] + (u[1] + v[1]) * ss),
                 (c[0] + v[0] * ss, c[1] + v[1] * ss)], bow=False)
        elif kind == "axes_slot":
            nums, strs, opts, flags, _ = pending_axes.pop(0)
            _draw_axes(add, add_label, M, (vx0, vx1, vy0, vy1), strs, opts, flags, text_size, sx, sy)
        elif kind == "numberline":
            _, a, b, stp, flags = it
            p0, p1 = M((a, 0)), M((b, 0))
            pad = text_size * 0.6
            add([(p0[0] - pad, p0[1]), (p1[0] + pad, p1[1])], bow=False)
            add(geom.arrowhead((p1[0] + pad, p1[1]), p0, text_size * 0.4), bow=False)
            add(geom.arrowhead((p0[0] - pad, p0[1]), p1, text_size * 0.4), bow=False)
            k = 0
            v = a
            lw = max(len(_fmt(a)), len(_fmt(b))) * 0.75 * 0.85 * text_size + 0.5 * text_size
            stride = max(1, math.ceil(lw / max(1e-9, stp * sx)))
            while v <= b + 1e-9 and k < 60:
                p = M((v, 0))
                add([(p[0], p[1] - text_size * 0.3), (p[0], p[1] + text_size * 0.3)], bow=False)
                if k % stride == 0:
                    add_label(p[0], p[1] + text_size * 0.3, _fmt(v), "below", 0.85, board=True, role="tick")
                v = a + (k + 1) * stp
                k += 1
    return _resolve_labels(strokes, text_size, font, gcache)


# ----------------------------------------------------------------- label placement

def _glyphs(text, size, font):
    """-> (strokes at origin with baseline y=0, width, ascent, descent)"""
    if "$" in text or looks_like_tex(text):
        parts = split_text_math(text) if "$" in text else [(text, True)]
        from .mathlayout import hbox, Box
        boxes = []
        for s_, is_m in parts:
            if is_m:
                boxes.append(layout_math(s_, size, font))
            else:
                st, adv, _ = font.run(s_, size, 0, 0)
                boxes.append(Box(adv, size, 0.2 * size, st))
        b = hbox(boxes)
        return b.strokes, b.w, b.a, b.d
    st, w_, _ = font.run(text, size, 0, 0)
    return st, w_, size, 0.25 * size


_DIRS = {"above": (0, -1), "below": (0, 1), "right": (1, 0), "left": (-1, 0), "above right": (1, -1),
         "above left": (-1, -1), "below right": (1, 1), "below left": (-1, 1)}


def _candidates(lab, w, a, d, ts):
    """Candidate top-left text origins (tx, ty_baseline) with a rank cost and whether a leader line is needed.
    Rings at increasing distance; the requested side first."""
    bx, by = lab["anchor"]
    size = lab["size"]
    pos = lab["pos"]
    g = 0.3 * size
    out = []

    def box_at(cx, cy):             # label centred at (cx, cy)
        return cx - w / 2, cy + (a - d) / 2

    if lab["role"] == "tick":
        # tick numbers: only straight below/above (x axis) or left/right (y axis), dropped when busy
        alt = {"below": "above", "above": "below", "left": "right", "right": "left"}.get(pos, pos)
        for k, p in enumerate((pos, alt)):
            dx, dy = _DIRS.get(p, (0, 1))
            cx = bx + dx * (g + w / 2)
            cy = by + dy * (g + (a + d) / 2)
            out.append((box_at(cx, cy), 3.0 * k, False))
        return out
    if lab["role"] == "angle":
        am, rr, span = lab["spread"]
        # further out along the bisector a narrow angle's wedge is wide enough for its label
        for ring, extra in enumerate((0.45, 0.9, 1.4, 2.1, 2.9)):
            for k, turn in enumerate((0.0, 0.18, -0.18, 0.36, -0.36)):
                t = am + turn * min(1.0, span / 0.9 + 0.3)
                rad = rr + extra * size + 0.5 * max(w, a + d) * (1.0 if ring else 0.6)
                cx, cy = bx + rad * math.cos(t), by - rad * math.sin(t)
                out.append((box_at(cx, cy), 1.2 * ring + 0.35 * k, False))
        return out
    order = [pos] if pos in _DIRS else []
    order += [p for p in ("above", "below", "right", "left", "above right", "above left", "below right",
                          "below left") if p not in order]
    if pos == "center":
        out.append((box_at(bx, by), 0.0, False))
    for ring, extra in enumerate((0.0, 0.45, 1.0, 1.7)):
        for k, p in enumerate(order):
            dx, dy = _DIRS[p]
            gg = g + extra * size
            cx = bx + dx * (gg + w / 2)
            cy = by + dy * (gg + (a + d) / 2)
            # diagonal corners: the box corner sits gg away from the anchor along both axes
            out.append((box_at(cx, cy), (0.6 if pos == "center" else 0.0) + 1.5 * ring + 0.3 * k,
                        ring >= 2 and pos != "center" and lab["role"] == "label"))
    return out


def _resolve_labels(chunks, ts, font, gcache=None):
    """Place every label at the candidate that keeps a clear halo from all ink and from labels placed before it;
    the most constrained labels are placed first. Tries: candidate rings at increasing distance (with a leader
    line from the anchor when far), then a smaller size (never below the legible minimum), then (last resort) an
    abbreviation. Tick numbers that find no clear spot are left out. Returns (strokes, spans)."""
    from .collide import SegIndex, seg_rect_dist
    gcache = {} if gcache is None else gcache

    def _label_glyphs(text, size, font):
        k = (text, round(size, 6))
        g = gcache.get(k)
        if g is None:
            g = gcache[k] = _glyphs(text, size, font)
        return g
    ink = [c for c in chunks if not isinstance(c, dict)]
    idx = SegIndex(cell=max(ts * 0.6, 1e-3))
    for s in ink:
        idx.add_stroke(s, 0)
    halo = ts * 0.2
    gb = geom.bbox(ink) if ink else None      # the drawing's own extent: labels that stick far out make it smaller
    placed = []          # label rects (with their leader segments)
    leaders = []

    def ink_cost(r):
        hits = idx.count_near(r, halo)
        return 0 if not hits else 10 + hits

    def label_cost(r):
        c = 0
        for q in placed:
            if geom.rects_overlap(r, q, halo * 1.5):
                c += 40
        for (p, q) in leaders:
            if seg_rect_dist(p, q, r) < halo:
                c += 12
        return c

    def leader_seg(anchor, r, size):
        cx = min(max(anchor[0], r[0]), r[2])
        cy = min(max(anchor[1], r[1]), r[3])
        L = math.hypot(cx - anchor[0], cy - anchor[1])
        if L < size * 0.6:
            return None
        ux, uy = (cx - anchor[0]) / L, (cy - anchor[1]) / L
        return ((anchor[0] + ux * size * 0.3, anchor[1] + uy * size * 0.3),
                (cx - ux * size * 0.15, cy - uy * size * 0.15))

    def evaluate(lab, size, text):
        st, w, a, d = _label_glyphs(text, size, font)
        # the real ink extent (descenders of β, ρ, g; tall math) — not just the nominal ascent/descent
        gb_ = geom.bbox(st) if st else (0.0, -a, w, d)
        a, d = max(a, -gb_[1]), max(d, gb_[3])
        lab2 = dict(lab, size=size)
        best = None
        for (tx, ty), rank, lead in _candidates(lab2, w, a, d, ts):
            r = (tx, ty - a, tx + w, ty + d)
            c = ink_cost(r) + label_cost(r) + rank
            if gb is not None:
                out_ = max(0.0, gb[0] - r[0]) + max(0.0, r[2] - gb[2]) + max(0.0, gb[1] - r[1]) + max(0.0, r[3] - gb[3])
                c += 0.8 * out_ / max(size, 1e-9)
            # a label belongs next to its anchor: the farther, the more ambiguous (a label nearer another mark)
            ax, ay = lab["anchor"]
            dd = math.hypot(max(r[0] - ax, 0.0, ax - r[2]), max(r[1] - ay, 0.0, ay - r[3]))
            c += 2.0 * dd / max(size, 1e-9)
            seg = None
            if lead:
                seg = leader_seg(lab["anchor"], r, size)
                if seg is not None:
                    mid = (seg[0][0] * 0.7 + seg[1][0] * 0.3, seg[0][1] * 0.7 + seg[1][1] * 0.3)
                    lr = (min(mid[0], seg[1][0]), min(mid[1], seg[1][1]), max(mid[0], seg[1][0]),
                          max(mid[1], seg[1][1]))
                    if idx.count_near(lr, halo * 0.5, cap=1):
                        c += 8          # a leader across other lines reads as one more line
                    c += 0.5
            if best is None or c < best[0]:
                best = (c, tx, ty, r, seg, st)
        return best

    labs = [c for c in chunks if isinstance(c, dict)]
    # the most constrained first: fewest clear candidates at full size (ties: drawing order)
    order = []
    for k, lab in enumerate(labs):
        st, w, a, d = _label_glyphs(lab["text"], lab["size"], font)
        gb_ = geom.bbox(st) if st else (0.0, -a, w, d)
        a, d = max(a, -gb_[1]), max(d, gb_[3])
        nfree = sum(1 for (tx, ty), _, _ in _candidates(lab, w, a, d, ts)
                    if ink_cost((tx, ty - a, tx + w, ty + d)) == 0)
        order.append((0 if lab["role"] != "tick" else 1, nfree, k))
    order.sort()
    result = {}
    for _, _, k in order:
        lab = labs[k]
        size, text = lab["size"], lab["text"]
        best = evaluate(lab, size, text)
        if best[0] >= 10 and not lab["fixed"]:
            # no clear spot: a smaller size (down to the legible minimum), then an abbreviation
            s2 = size
            while best[0] >= 10 and s2 * 0.9 >= lab["minsize"] - 1e-9:
                s2 *= 0.9
                b2 = evaluate(lab, s2, text)
                if b2[0] < best[0] - 1:
                    best = b2
            if best[0] >= 10:
                ab = _abbreviate(text)
                if ab and ab != text:
                    b2 = evaluate(lab, lab["minsize"], ab)
                    if b2[0] < 10:
                        best = b2
        if lab["fixed"] and best[0] >= 4:
            continue            # a tick number that would sit on the curve / a label is simply left out
        c, tx, ty, r, seg, st = best
        placed.append(r)
        if seg is not None:
            leaders.append(seg)
        result[k] = (tx, ty, r, seg, st)
    out, spans = [], []
    k = 0
    for c in chunks:
        if not isinstance(c, dict):
            out.append(c)
            continue
        got = result.get(k)
        lab = c
        k += 1
        if got is None:
            continue
        tx, ty, r, seg, st = got
        i0 = len(out)
        lead = None
        if seg is not None:
            lead = len(out)
            out.append(geom.resample([seg[0], seg[1]], ts * 0.12))     # a short plain leader stroke
        out.extend(geom.transform(st, 1, 1, tx, ty))
        spans.append({"text": lab["text"], "role": lab["role"], "anchor": lab["anchor"], "strokes": [i0, len(out)],
                      "leader": lead})
    return out, spans


def _abbreviate(text):
    """Last resort for a label with no clear spot: the first word, or a long word cut to 4 letters + '.'"""
    if "$" in text or looks_like_tex(text):
        return None
    words = text.split()
    if len(words) > 1:
        return words[0]
    if len(text) > 5 and text.isalpha():
        return text[:4] + "."
    return None


def _draw_axes(add, add_label, M, view, strs, opts, flags, ts, sx, sy):
    vx0, vx1, vy0, vy1 = view
    ox = min(max(0.0, vx0), vx1)
    oy = min(max(0.0, vy0), vy1)
    xa, xb = M((vx0, oy)), M((vx1, oy))
    ya, yb = M((ox, vy0)), M((ox, vy1))
    grid = opts.get("grid")
    if grid and _isnum(grid) and float(grid) > 0:
        g = float(grid)
        k = math.ceil(vx0 / g)
        while k * g <= vx1 and k < 1e4:
            add([M((k * g, vy0)), M((k * g, vy1))], True, bow=False)
            k += 1
        k = math.ceil(vy0 / g)
        while k * g <= vy1 and k < 1e4:
            add([M((vx0, k * g)), M((vx1, k * g))], True, bow=False)
            k += 1
    add([xa, xb], bow=False)
    add(geom.arrowhead(xb, xa, ts * 0.4), bow=False)
    add([ya, yb], bow=False)
    add(geom.arrowhead(yb, ya, ts * 0.4), bow=False)
    xl = strs[0] if len(strs) > 0 else "x"
    yl = strs[1] if len(strs) > 1 else "y"
    if xl and xl != "-":
        add_label(xb[0] + ts * 0.15, xb[1], xl, "right", 0.9, board=True, role="axis")
    if yl and yl != "-":
        add_label(yb[0], yb[1] - ts * 0.1, yl, "above", 0.9, board=True, role="axis")
    tick = opts.get("ticks") or opts.get("labels")
    if tick and _isnum(tick) and float(tick) > 0:
        st = float(tick)
        lab = "nolabels" not in flags
        lw = max(len(_fmt(vx0)), len(_fmt(vx1))) * 0.75 * 0.7 * ts + 0.5 * ts
        xstride = max(1, math.ceil(lw / max(1e-9, st * sx)))
        ystride = max(1, math.ceil(0.7 * ts * 1.4 / max(1e-9, st * sy)))
        n = 0
        k = math.ceil(vx0 / st)
        while k * st <= vx1 + 1e-9 and n < 40:
            v = k * st
            if abs(v) > 1e-9:
                p = M((v, oy))
                add([(p[0], p[1] - ts * 0.18), (p[0], p[1] + ts * 0.18)], bow=False)
                if lab and k % xstride == 0 and abs(p[0] - xb[0]) > ts * 0.8 and abs(p[0] - ya[0]) > ts * 0.9:
                    add_label(p[0], p[1] + ts * 0.25, _fmt(v), "below", 0.7, board=True, role="tick")
            k += 1
            n += 1
        n = 0
        k = math.ceil(vy0 / st)
        while k * st <= vy1 + 1e-9 and n < 40:
            v = k * st
            if abs(v) > 1e-9:
                p = M((ox, v))
                add([(p[0] - ts * 0.18, p[1]), (p[0] + ts * 0.18, p[1])], bow=False)
                if lab and k % ystride == 0 and abs(p[1] - yb[1]) > ts * 0.8 and abs(p[1] - xa[1]) > ts * 0.9:
                    add_label(p[0] - ts * 0.3, p[1], _fmt(v), "left", 0.7, board=True, role="tick")
            k += 1
            n += 1


def _fmt(v):
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v))).replace("-", "−")
    return ("%g" % v).replace("-", "−")


def _nice_step(span):
    if span <= 0:
        return 1.0
    raw = span / 10
    p = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * p:
            return m * p
    return 10 * p


def _isnum(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _pos_from(flags, default):
    ps = [f for f in ("above", "below", "left", "right", "center") if f in flags]
    return " ".join(ps) if ps else default
