"""tvdsl.lang -- the TV illustration language: line parser + the live document.

One statement per line; a malformed line is skipped (reported), never fatal.

    scene "<one-line description>" [view=x0,x1,y0,y1] [loop=SECONDS] [bg=#hex]   start a new picture (replaces)
    note "<one-line description>"                  update the description the voice model gets
    param NAME = EXPR [min..max]                   a tunable number; changing it later glides smoothly
    set NAME = EXPR                                (patch) change a param (same as param)
    let NAME = EXPR                                a derived value; may use t and params (evaluated every frame)
    func NAME(a, b) = EXPR                         a user function usable in any expression
    KIND ID key=value ... ["text"]                 an object; the same ID again replaces it (morphs smoothly)
    for i=A..B[:STEP] <statement with {i} in ids>  instancing (i bound in its expressions); nest up to 2 deep
    del ID [ID ...]                                (patch) remove (fades out); ID* = every id with that prefix
    clear                                          blank the TV (idle look)
    # comment

Values are expressions (tvdsl.expr) except: colours (name, #hex, rgb()/hsv()/wl() expressions), quoted strings,
ranges a..b, and object references (ids).
"""
import math
import re

from .expr import compile_expr, ExprError, Env, NAN

MAX_LINE = 1200
MAX_OBJECTS = 600
MAX_FOR = 120
COLOR_KEYS = {"color", "fill", "stroke", "c", "col", "bg", "color2", "ink"}
STRING_KEYS = {"anchor", "font", "el", "a", "b", "in", "label", "labels", "mode", "style", "text", "xlabel", "ylabel",
               "align", "name", "src", "shape", "kind", "q", "of", "path", "follow", "title", "lens", "mirror", "optic", "on", "side"}
RANGE_KEYS = {"x", "y", "s", "range", "xr", "yr", "domain", "u"}
FLAG_WORDS = {"closed", "smooth", "fill", "dashed", "dash", "grid", "ticks", "both", "real", "white", "bold", "under",
              "ext", "glow", "hidden", "noarrow", "arrows", "ring", "row", "col", "column", "standing", "troughs"}
PATCH_KINDS = {"set", "del", "delete", "remove", "clear", "note"}
KINDS = {
    "line", "arrow", "vector", "path", "polyline", "poly", "polygon", "bezier", "circle", "dot", "point", "ellipse",
    "arc", "rect", "box", "text", "label", "tex", "plot", "curve", "area", "axes", "grid", "angle", "wave", "spring",
    "field", "atom", "bond", "group", "lens", "mirror", "glass", "prism", "block", "screen", "ray", "beam", "brace",
    "electron", "particle", "spot", "star", "graph", "function", "parametric", "rays3", "principal", "image",
    "callout", "annotate", "pointer", "flow", "cycle", "process", "ripple", "wavefront", "waves",
}
ALIASES = {"vector": "arrow", "polyline": "path", "polygon": "poly", "point": "dot", "box": "rect", "label": "text",
           "tex": "text", "graph": "plot", "function": "plot", "parametric": "curve", "prism": "glass",
           "block": "glass", "principal": "rays3", "annotate": "callout", "pointer": "callout",
           "cycle": "flow", "process": "flow", "wavefront": "ripple", "waves": "ripple", "electron": "dot", "particle": "dot", "spot": "dot", "func_plot": "plot"}

_ID_RE = re.compile(r"^[A-Za-z_][\w.\-{}]*$")
_KEY_RE = re.compile(r"(?<![<>!=])([A-Za-z_][\w]*)=(?!=)")


class Stmt:
    __slots__ = ("op", "kind", "id", "props", "text", "name", "expr", "rng", "ids", "vars", "line", "src")

    def __init__(self, op, **kw):
        self.op = op
        self.kind = kw.get("kind")
        self.id = kw.get("id")
        self.props = kw.get("props") or {}
        self.text = kw.get("text")
        self.name = kw.get("name")
        self.expr = kw.get("expr")
        self.rng = kw.get("rng")
        self.ids = kw.get("ids")
        self.vars = kw.get("vars") or {}
        self.line = kw.get("line", 0)
        self.src = kw.get("src", "")

    def __repr__(self):
        return f"Stmt({self.op} {self.kind or ''} {self.id or self.name or ''})"


class ParseError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------------- lexical helpers
def _split_kv(s):
    """'at=(0, 1) r=1 + t "hello" closed' -> ([(key, raw)], [quoted strings], [bare words])"""
    kvs, strs, bare = [], [], []
    i, n = 0, len(s)
    while i < n:
        while i < n and s[i] in " \t,;":
            i += 1
        if i >= n:
            break
        if s[i] in "\"'“”":
            q = s[i]
            close = "\"" if q in "\"“”" else "'"
            j = i + 1
            buf = []
            while j < n and s[j] not in (close, "”") :
                if s[j] == "\\" and j + 1 < n:
                    # "\n" is a line break (it was kept as a bare 'n': "P = 5.5nQ = 5.6" on screen)
                    buf.append("\n" if s[j + 1] == "n" else s[j + 1])
                    j += 2
                    continue
                buf.append(s[j])
                j += 1
            strs.append("".join(buf))
            i = j + 1
            continue
        m = _KEY_RE.match(s, i)
        if m:
            key = m.group(1)
            j = m.end()
            depth = 0
            start = j
            inq = None
            while j < n:
                c = s[j]
                if inq:
                    if c == inq:
                        inq = None
                elif c in "\"'":
                    inq = c
                elif c in "([{":
                    depth += 1
                elif c in ")]}":
                    depth -= 1
                elif c in " \t" and depth <= 0:
                    # does a new key=, a quoted string or a flag start here?
                    k = j
                    while k < n and s[k] in " \t":
                        k += 1
                    if k >= n or _KEY_RE.match(s, k) or s[k] in "\"'“":
                        break
                    w = re.match(r"[A-Za-z_]\w*", s[k:])
                    if w and w.group(0) in FLAG_WORDS and (k + w.end() >= n or s[k + w.end()] in " \t"):
                        # a flag word only when what precedes is a complete value (balanced, not an operator)
                        prev = s[start:j].rstrip()
                        if prev and prev[-1] not in "+-*/^(,<>=?:":
                            break
                j += 1
            kvs.append((key, s[start:j].strip()))
            i = j
            continue
        m = re.match(r"[^\s]+", s[i:])
        bare.append(m.group(0))
        i += m.end()
    return kvs, strs, bare


def _split_range(raw):
    depth = 0
    for k in range(len(raw) - 1):
        c = raw[k]
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "." and raw[k + 1] == "." and depth == 0:
            return raw[:k], raw[k + 2:]
    return None


_NAMED = {
    "cyan": (0.31, 0.82, 1.0), "blue": (0.36, 0.55, 1.0), "sky": (0.55, 0.8, 1.0), "teal": (0.2, 0.85, 0.75),
    "green": (0.49, 0.95, 0.61), "lime": (0.75, 1.0, 0.35), "yellow": (1.0, 0.82, 0.4), "gold": (1.0, 0.72, 0.2),
    "orange": (1.0, 0.62, 0.26), "red": (1.0, 0.36, 0.36), "pink": (1.0, 0.55, 0.75), "magenta": (1.0, 0.42, 0.84),
    "purple": (0.69, 0.55, 1.0), "violet": (0.62, 0.45, 1.0), "white": (0.93, 0.95, 0.98), "grey": (0.5, 0.54, 0.62),
    "gray": (0.5, 0.54, 0.62), "dim": (0.25, 0.28, 0.36), "dark": (0.12, 0.14, 0.2), "black": (0.0, 0.0, 0.0),
    "brown": (0.72, 0.5, 0.3), "glass": (0.55, 0.8, 1.0), "water": (0.3, 0.6, 1.0), "fg": (0.93, 0.95, 0.98),
    "accent": (1.0, 0.82, 0.4), "none": None, "transparent": None, "off": None,
}


def wavelength_rgb(nm):
    """visible wavelength (380..750 nm) -> an emissive rgb in 0..1"""
    w = float(nm)
    if w != w:
        return (1.0, 1.0, 1.0)
    if w < 380: w = 380
    if w > 750: w = 750
    if w < 440: r, g, b = -(w - 440) / 60, 0.0, 1.0
    elif w < 490: r, g, b = 0.0, (w - 440) / 50, 1.0
    elif w < 510: r, g, b = 0.0, 1.0, -(w - 510) / 20
    elif w < 580: r, g, b = (w - 510) / 70, 1.0, 0.0
    elif w < 645: r, g, b = 1.0, -(w - 645) / 65, 0.0
    else: r, g, b = 1.0, 0.0, 0.0
    f = 0.45 + 0.55 * (w - 380) / 40 if w < 420 else 0.45 + 0.55 * (750 - w) / 50 if w > 700 else 1.0
    lift = 0.12
    return tuple(min(1.0, lift + (1 - lift) * (c * f) ** 0.8) for c in (r, g, b))


def _hex(s):
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6 or not re.fullmatch(r"[0-9a-fA-F]{6}", s):
        return None
    return tuple(int(s[k:k + 2], 16) / 255.0 for k in (0, 2, 4))


class ColorSpec:
    """a colour: constant rgb tuple / None (no fill) or a dynamic rgb()/hsv()/wl()/mix() expression"""
    __slots__ = ("const", "fn", "src")

    def __init__(self, src):
        self.src = src
        self.const = None
        self.fn = None
        s = src.strip().strip("\"'")
        low = s.lower()
        if low in _NAMED:
            self.const = _NAMED[low] if _NAMED[low] is not None else "none"
            return
        if low.startswith("#"):
            h = _hex(low)
            if h is None:
                raise ParseError(f"bad colour {s!r}")
            self.const = h
            return
        m = re.fullmatch(r"(rgb|hsv|wl|mix)\((.*)\)", low)
        if not m:
            # a named colour with a brightness factor: "cyan*0.5"
            m2 = re.fullmatch(r"([a-z]+)\s*\*\s*([0-9.]+)", low)
            if m2 and _NAMED.get(m2.group(1)):
                k = float(m2.group(2))
                self.const = tuple(min(1.0, c * k) for c in _NAMED[m2.group(1)])
                return
            raise ParseError(f"bad colour {s!r}")
        fname, inner = m.group(1), m.group(2)
        if fname == "mix":
            parts = _top_split(inner)
            if len(parts) != 3:
                raise ParseError("mix(colour, colour, u)")
            a, b, u = ColorSpec(parts[0]), ColorSpec(parts[1]), compile_expr(parts[2])

            def fmix(env, a=a, b=b, u=u):
                ca, cb = a.eval(env), b.eval(env)
                if ca is None or cb is None:
                    return ca or cb
                k = u.eval(env)
                k = 0.0 if k != k else max(0.0, min(1.0, float(k)))
                return tuple(x + (y - x) * k for x, y in zip(ca, cb))
            self.fn = fmix
            return
        parts = [compile_expr(p) for p in _top_split(inner)]
        if fname == "wl":
            if len(parts) != 1:
                raise ParseError("wl(nm)")
            self.fn = lambda env, p=parts[0]: wavelength_rgb(p.eval(env))
        elif fname == "rgb":
            if len(parts) != 3:
                raise ParseError("rgb(r,g,b)")

            def frgb(env, ps=parts):
                v = [float(p.eval(env)) for p in ps]
                sc = 255.0 if any(x > 1.0 for x in v if x == x) else 1.0
                return tuple(max(0.0, min(1.0, (x if x == x else 0) / sc)) for x in v)
            self.fn = frgb
        else:
            if len(parts) != 3:
                raise ParseError("hsv(h,s,v)")

            def fhsv(env, ps=parts):
                import colorsys
                h, s_, v = [float(p.eval(env)) for p in ps]
                h = (h / 360.0) % 1.0 if h > 1 else h % 1.0
                return colorsys.hsv_to_rgb(h if h == h else 0, max(0, min(1, s_)), max(0, min(1, v)))
            self.fn = fhsv
        if self.fn is not None and all(p.const for p in parts):
            self.const = self.fn(Env())
            self.fn = None

    def eval(self, env):
        if self.fn is None:
            return None if self.const == "none" else self.const
        try:
            return self.fn(env)
        except Exception:  # noqa: BLE001
            return (1.0, 1.0, 1.0)


def _top_split(s, sep=","):
    out, depth, cur = [], 0, []
    for c in s:
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        if c == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    out.append("".join(cur))
    return [x.strip() for x in out if x.strip()]


class Range:
    __slots__ = ("a", "b")

    def __init__(self, a, b):
        self.a, self.b = a, b


def _value(key, raw):
    key_l = key.lower()
    if key_l in COLOR_KEYS and not (key_l == "col" and raw.strip() in ("1", "0", "true")):   # col=1: a column flag
        return ColorSpec(raw)
    if "?" in raw and ":" in raw and not re.fullmatch(r"([\"'“]).*[\"'”]", raw.strip()):
        try:
            return compile_expr(raw)          # a choice of words: q=tr>0.99 ? "+" : ""
        except ExprError:
            pass
    if raw[:1] in "\"'":
        return raw.strip("\"'“”")
    if key_l in STRING_KEYS:
        return raw.strip("\"'")
    if key_l in RANGE_KEYS or ".." in raw:
        r = _split_range(raw)
        if r is not None:
            return Range(compile_expr(r[0]), compile_expr(r[1]))
    return compile_expr(raw)


# ---------------------------------------------------------------------------------------------------- statements
def _expand_for(rest, lineno, src, depth=0, outer=None):
    m = re.match(r"for\s+([A-Za-z_]\w*)\s*(?:=|in)\s*(.+?)\s*\.\.\s*([^\s:]+?)(?::([^\s:]+))?\s*:?\s+(.*)$", rest)
    if not m:
        raise ParseError("for i=A..B <statement>")
    var, a, b, st, body = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    env = Env(dict(outer or {}))
    lo, hi = compile_expr(a).eval(env), compile_expr(b).eval(env)
    step = compile_expr(st).eval(env) if st else 1.0
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (lo, hi, step)) or step == 0:
        raise ParseError("for range must be constant numbers")
    n = int(math.floor((hi - lo) / step + 1e-9)) + 1
    if n <= 0:
        return []
    if n > MAX_FOR:
        raise ParseError(f"for loop over {n} items (max {MAX_FOR})")
    out = []
    for k in range(n):
        v = lo + k * step
        if abs(v - round(v)) < 1e-9:
            v = float(round(v))
        vars_ = dict(outer or {})
        vars_[var] = v
        b2 = _subst_ids(body.strip(), {var: v})
        if b2.startswith("for ") and depth < 1:
            out.extend(_expand_for(b2, lineno, src, depth + 1, vars_))
        else:
            out.extend(parse_line(b2, lineno, vars_=vars_, src=src))
    return out


def _subst_ids(s, vars_):
    if not vars_:
        return s

    def rep(m):
        name = m.group(1)
        if name in vars_:
            v = vars_[name]
            return str(int(v)) if float(v).is_integer() else str(v).replace(".", "p")
        return m.group(0)
    return re.sub(r"\{([A-Za-z_]\w*)\}", rep, s)


def parse_line(line, lineno=0, vars_=None, src=None):
    """one source line -> [Stmt] (a for loop expands to several). Raises ParseError/ExprError on a bad line."""
    s = line.strip()
    src = src if src is not None else s
    if not s or s.startswith("#") or s.startswith("//") or s.startswith("```"):
        return []
    if len(s) > MAX_LINE:
        raise ParseError("line too long")
    s = s.rstrip(";")
    head = s.split(None, 1)
    kw = head[0].lower()
    rest = head[1] if len(head) > 1 else ""
    if kw == "for":
        return _expand_for(s, lineno, src, 0, vars_)
    if kw in ("scene", "morph"):
        # `morph "title" [view=] [loop=]` = the agent's soft replace (a careful version / a rewrite of the
        # current picture): new title/view/loop, but time, params and objects carry on -- objects re-declared with the
        # same ID morph, params glide, and the agent sends `del` for the IDs the new program no longer has
        kvs, strs, bare = _split_kv(rest)
        props = {}
        for k, raw in kvs:
            k = k.lower()
            if k == "view":
                parts = [compile_expr(p).eval(Env()) for p in _top_split(raw.strip("()[]"))]
                if len(parts) == 4 and all(isinstance(p, float) and math.isfinite(p) for p in parts):
                    props["view"] = tuple(parts)
            elif k == "loop":
                v = compile_expr(raw).eval(Env())
                if isinstance(v, float) and math.isfinite(v) and v > 0:
                    props["loop"] = min(120.0, max(1.0, v))
            elif k == "bg":
                props["bg"] = ColorSpec(raw)
            elif k == "restart" and kw == "morph":
                props["restart"] = raw.strip() not in ("0", "false")
        title = strs[0] if strs else " ".join(bare)
        return [Stmt(kw, text=title.strip()[:200], props=props, line=lineno, src=src)]
    if kw == "note":
        kvs, strs, bare = _split_kv(rest)
        return [Stmt("note", text=(strs[0] if strs else rest.strip("\"' ")).strip()[:200], line=lineno, src=src)]
    if kw == "clear":
        return [Stmt("clear", line=lineno, src=src)]
    if kw == "hold":                        # stop the automatic sweep (the agent sends it before every update)
        return [Stmt("hold", line=lineno, src=src)]
    if kw in ("del", "delete", "remove"):
        ids = [_subst_ids(x.strip(","), vars_) for x in rest.split()]
        if not ids:
            raise ParseError("del needs ids")
        return [Stmt("del", ids=ids, line=lineno, src=src)]
    if kw in ("set", "let") and re.match(r"note\b\s*=?\s*[\"'“]", rest):        # "set note = ..." -> note
        return parse_line("note " + re.sub(r"^note\b\s*=?\s*", "", rest), lineno, vars_, src)
    if kw in ("let", "set", "var") and re.match(r"[A-Za-z_]\w*\s*\([^)]*\)\s*=", rest):   # let f(x) = ... -> func
        return parse_line("func " + rest, lineno, vars_, src)
    if kw in ("param", "set", "let", "var", "const"):
        m = re.match(r"([A-Za-z_]\w*)\s*=\s*(.+)$", rest)
        if not m:
            raise ParseError(f"{kw} NAME = EXPR")
        name, val = m.group(1), m.group(2).strip()
        rng = None
        sweep = False
        ms = re.match(r"^(.*?)\s+sweep\s*$", val)
        if ms and kw == "param":
            val, sweep = ms.group(1), True
        m2 = re.match(r"^(.*?)\s+(?:range\s*=?\s*)?\[?\s*(-?[\d.]+)\s*\.\.\s*(-?[\d.]+)\s*\]?\s*$", val)
        if m2 and kw != "let":
            val = m2.group(1)
            rng = (float(m2.group(2)), float(m2.group(3)))
        e = compile_expr(val)
        op = "let" if kw == "let" else "param"
        if op == "param" and not e.const and (e.names - {"@scene"}) & {"t"}:
            op = "let"                     # a time-varying 'param' is a let
        return [Stmt(op, name=name, expr=e, rng=rng, vars=vars_, line=lineno, src=src, kind="sweep" if sweep else kw)]
    if kw == "func" or kw == "def":
        m = re.match(r"([A-Za-z_]\w*)\s*\(([^)]*)\)\s*=\s*(.+)$", rest)
        if not m:
            raise ParseError("func NAME(args) = EXPR")
        args = [a.strip() for a in m.group(2).split(",") if a.strip()]
        return [Stmt("func", name=m.group(1), ids=args, expr=compile_expr(m.group(3)), line=lineno, src=src)]
    # an object
    kind = ALIASES.get(kw, kw)
    if kind not in KINDS and kw not in KINDS:
        raise ParseError(f"unknown statement {kw!r}")
    kind = ALIASES.get(kind, kind)
    ident = None
    parts = rest.split(None, 1)
    if parts and "=" not in parts[0] and parts[0][:1] not in "\"'“" and _ID_RE.match(parts[0]):
        ident = _subst_ids(parts[0], vars_)
        rest = parts[1] if len(parts) > 1 else ""
    kvs, strs, bare = _split_kv(rest)
    props = {}
    for k, raw in kvs:
        try:
            props[k.lower()] = _value(k, raw)
        except (ExprError, ParseError):
            continue                      # one bad property does not lose the object
    for w in bare:
        wl = w.lower().strip(",")
        if wl in FLAG_WORDS:
            props.setdefault(wl, compile_expr("1"))
    text = strs[0] if strs else None
    if text is None and kind == "text" and "text" in props:
        text = props.pop("text")
    return [Stmt("obj", kind=kind, id=ident, props=props, text=text, vars=vars_, line=lineno, src=src)]


def parse_program(text):
    """whole program -> (stmts, errors[(lineno, line, why)])"""
    stmts, errs = [], []
    for k, line in enumerate(str(text).splitlines(), 1):
        try:
            stmts.extend(parse_line(line, k))
        except (ParseError, ExprError, RecursionError) as e:
            errs.append((k, line.strip()[:200], str(e)[:120]))
    return stmts, errs


class LineFeeder:
    """streaming: feed text deltas, get back complete statements as soon as each line ends"""

    def __init__(self):
        self.buf = ""
        self.n = 0
        self.errors = []
        self.stmts = 0

    def feed(self, delta):
        self.buf += delta
        out = []
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            out.extend(self._one(line))
        return out

    def flush(self):
        line, self.buf = self.buf, ""
        return self._one(line) if line.strip() else []

    def _one(self, line):
        self.n += 1
        try:
            r = parse_line(line, self.n)
            self.stmts += len(r)
            return r
        except (ParseError, ExprError, RecursionError) as e:
            self.errors.append((self.n, line.strip()[:200], str(e)[:120]))
            return []


# ---------------------------------------------------------------------------------------------------- the live document
FADE_IN = 0.35
FADE_OUT = 0.35
MORPH = 0.7
PARAM_GLIDE = 0.9


class Obj:
    __slots__ = ("kind", "id", "props", "text", "vars", "born", "dying", "prev", "morph_t0", "order", "src")

    def __init__(self, st, born, order):
        self.kind = st.kind
        self.id = st.id
        self.props = st.props
        self.text = st.text
        self.vars = dict(st.vars or {})
        self.born = born
        self.dying = None
        self.prev = None
        self.morph_t0 = None
        self.order = order
        self.src = st.src


class Param:
    __slots__ = ("old", "new", "t0", "rng", "sweep")

    def __init__(self, v, t0, rng=None, sweep=False):
        self.old = v
        self.new = v
        self.t0 = t0
        self.rng = rng
        self.sweep = sweep

    def value(self, now):
        u = (now - self.t0) / PARAM_GLIDE
        if u >= 1 or self.old == self.new:
            return self.new
        u = 0.0 if u < 0 else u
        u = u * u * (3 - 2 * u)
        return self.old + (self.new - self.old) * u

    def set(self, v, now):
        cur = self.value(now)
        if cur * v < 0:
            cur = v                       # a sign flip (converging -> diverging) snaps: gliding through 0 is nonsense
        self.old = cur
        self.new = v
        self.t0 = now


class Doc:
    """the live picture: params, lets, funcs, objects; every change is timestamped (wall seconds) so rendering at a
    given wall time is deterministic"""

    def __init__(self):
        self.reset(0.0)
        self.empty = True
        self.version = 0

    def reset(self, now, title=""):
        self.sweep_off = False        # a param was set after the picture was drawn: no automatic sweep any more
        self._sweep = (None, None)
        self.title = title
        self.note = ""
        self.view = (-8.0, 8.0, -4.5, 4.5)
        self.loop = None
        self.bg = None
        self.t0 = now
        self.params = {}
        self.lets = []            # [(name, Expr)] in order
        self.funcs = {}
        self.objs = {}            # id -> Obj (insertion = draw order)
        self.dead = []            # fading-out objects
        self.n = 0
        self.scene_at = now
        self.empty = False
        self._phase = None
        self._animated = {}       # params the program animates (let NAME = <time-varying>): name -> Param

    # a change (hold: every update / careful version) must not be followed by the loop wrapping and the
    # picture blanking to re-trace itself while the user looks at what they asked for: the scene clock runs on WITHOUT
    # wrapping for LOOP_HOLD s (traces stay drawn, steps() stay at their stage, periodic motions go on), then the loop
    # starts over from 0
    LOOP_HOLD = 8.0

    def scene_t(self, now):
        T = max(0.0, now - self.t0)
        L = self.loop_len()
        if not L:
            return T
        ph = getattr(self, "_phase", None)
        if ph:
            t_h, t_at, until = ph
            if now < until:
                return t_at + max(0.0, now - t_h)
            return (now - until) % L
        return T % L

    def _stop_animated(self, now):
        """a param the program animates (set n = cycle(...)) is a demo; once the user takes control (any change)
        it becomes the user's param again at its declared value, gliding there from where the animation was -- otherwise the user's
        "lower the index" met a let and the model changed something else"""
        anim = getattr(self, "_animated", None)
        if not anim:
            return
        lets = dict(self.lets)
        cur = {}
        try:
            from .scene import Frame
            cur = Frame(self, now).base
        except Exception:  # noqa: BLE001
            pass
        for name, p in list(anim.items()):
            e = lets.get(name)
            if e is None or "t" not in getattr(e, "names", ()):
                continue
            self.lets = [(n_, e_) for n_, e_ in self.lets if n_ != name]
            v = cur.get(name)
            p.old = v if isinstance(v, float) and math.isfinite(v) else p.new
            p.t0 = now
            self.params[name] = p
        self._animated = {}

    def _hold_loop(self, now):
        if self.empty or not self.loop:
            return
        self._phase = (now, self.scene_t(now), now + self.LOOP_HOLD)

    def ordered_lets(self):
        """lets in DEPENDENCY order (stable). A morph/patch that redefines `th` to use a new `T` kept th's
        old slot ahead of T, so th, bob, the whole pendulum went NaN and froze (fu3 f_pendulum: 'the string is the same
        length'). Cached per lets list."""
        key = tuple((n, id(e)) for n, e in self.lets)
        c = getattr(self, "_lets_order", None)
        if c is not None and c[0] == key:
            return c[1]
        names = {n for n, _ in self.lets}
        deps = {n: (set(getattr(e, "names", ())) & names) - {n} for n, e in self.lets}
        out, done, busy = [], set(), set()

        def visit(n, e):
            if n in done or n in busy:
                return
            busy.add(n)
            for d in sorted(deps[n], key=lambda m: idx[m]):
                visit(d, ex[d])
            busy.discard(n)
            done.add(n)
            out.append((n, e))
        ex = dict(self.lets)
        idx = {n: k for k, (n, _) in enumerate(self.lets)}
        for n, e in self.lets:
            visit(n, e)
        self._lets_order = (key, out)
        return out

    def describe(self):
        d = self.note or self.title
        d = d or ("blank" if self.empty else "an illustration")
        sw = None if self.empty else self.sweep_param()
        if sw:
            p = self.params[sw]
            d += f" (it slowly varies {sw} across {min(p.rng):g}..{max(p.rng):g} and back to {p.new:g})"
        return d

    def apply(self, st, now):
        self.version += 1
        op = st.op
        if op == "scene":
            self.reset(now, st.text)
            self.view = st.props.get("view", self.view)
            self.loop = st.props.get("loop")
            self.bg = st.props.get("bg")
            return
        if op == "morph":
            if self.empty:
                self.reset(now, st.text)
            self.title = st.text or self.title
            self.note = ""
            if "view" in st.props:
                self.view = st.props["view"]
            self._hold_loop(now)                  # (with the old loop: t stays continuous)
            if "loop" in st.props:
                self.loop = st.props["loop"]
            self.stop_sweep(now)
            if st.props.get("restart"):
                # the careful version is a new program whose stages are timed from t = 0: shapes still
                # morph from the sketch, but its clock starts over so its process plays from the beginning (with the
                # clock carried on, the careful pictures sat at their last stage: 'frames 3 and 4 identical', acc dev
                # 'animated' fails 11 -> 22)
                self.t0 = now
                self._phase = None
            return
        if op == "clear":
            self.reset(now)
            self.empty = True
            return
        if op == "hold":
            if not self.empty:
                self.stop_sweep(now)
                self._hold_loop(now)
                self._stop_animated(now)
            return
        if self.empty:
            self.empty = False
            self.scene_at = now
            self.t0 = now
        if op == "note":
            self.note = st.text
        elif op == "param":
            v = st.expr.eval(self._const_env(now, st.vars))
            if not isinstance(v, float) or not math.isfinite(v):
                return
            # a value the user asks for is never clamped to the range the program guessed (fu4: "make the
            # string twice as long" stopped at L = 4 of [1..4], "move the point to x = -1" stopped at 0.3, "a 5, 12,
            # 13 triangle" stopped at 6): the range widens to hold it (ranges only bound the automatic sweep)
            def widen(r):
                if not r:
                    return r
                lo, hi = min(r), max(r)
                return (min(lo, v), max(hi, v)) if not (lo <= v <= hi) else r
            p = self.params.get(st.name)
            if p is None:
                # a param that shadows an existing let: the let goes
                self.lets = [(n, e) for n, e in self.lets if n != st.name]
                self.params[st.name] = Param(v, now, widen(st.rng), sweep=st.kind == "sweep")
            else:
                if st.rng is not None and st.kind != "set":
                    p.rng = st.rng            # a re-declared param takes its new range
                p.rng = widen(p.rng)
                self.stop_sweep(now)
                p.set(v, now)
        elif op == "let":
            p_old = self.params.pop(st.name, None)
            if p_old is not None and "t" in getattr(st.expr, "names", ()):
                self._animated[st.name] = p_old
            elif st.name in getattr(self, "_animated", {}) and "t" not in getattr(st.expr, "names", ()):
                self._animated.pop(st.name, None)
            for k, (n, _) in enumerate(self.lets):
                if n == st.name:
                    self.lets[k] = (st.name, st.expr)
                    break
            else:
                self.lets.append((st.name, st.expr))
        elif op == "func":
            self.funcs[st.name] = (st.ids, st.expr)
        elif op == "del":
            for ident in st.ids:
                if ident.endswith("*"):
                    pre = ident[:-1]
                    hits = [k for k in self.objs if k.startswith(pre)]
                else:
                    hits = [ident] if ident in self.objs else []
                for k in hits:
                    o = self.objs.pop(k)
                    o.dying = now
                    self.dead.append(o)
        elif op == "obj":
            if st.id is None:
                st.id = f"_{st.kind}{self.n}"
            old = self.objs.get(st.id)
            if old is None and len(self.objs) >= MAX_OBJECTS:
                return
            self.n += 1
            o = Obj(st, now, self.n)
            if old is not None:
                o.order = old.order
                o.born = old.born
                if old.kind == o.kind:
                    o.prev = old
                    o.morph_t0 = now
                    old.prev = None
                else:
                    old.dying = now
                    self.dead.append(old)
                    o.born = now
            self.objs[st.id] = o
        self.dead = [o for o in self.dead if now - o.dying < FADE_OUT]

    # ---- sweep: `param N = v [lo..hi] sweep` -- a picture that would otherwise stand still explores that param.
    # Opt-in (11-11 r6: an automatic sweep of the first ranged param crossed regimes -- an object swept from beyond 2F
    # to inside F under a fixed caption "real, smaller image" -- and the judge rightly called the picture wrong)
    _MOTION = re.compile(r"\b(osc|saw|tri|cycle|steps)\s*\(|(?<![\w.])t(?![\w(])")

    def sweep_param(self):
        """name of the param the TV sweeps by itself (lo..hi range, declared value first), or None: only when nothing in
        the program moves continuously (osc/saw/tri/cycle/steps or plain t) and nobody has set a param yet"""
        if self.sweep_off or not self.params:
            return None
        if self._sweep[0] == self.version:
            return self._sweep[1]
        name = None
        srcs = [getattr(e, "src", "") for _, e in self.lets] + [o.src or "" for o in self.objs.values()]
        if not any(self._MOTION.search(x) for x in srcs):
            for k, p in self.params.items():
                if p.sweep and p.rng and len(p.rng) == 2 and max(p.rng) > min(p.rng) and any(re.search(r"\b%s\b" % re.escape(k), x) for x in srcs):
                    name = k
                    break
        self._sweep = (self.version, name)
        return name

    _SEQ = re.compile(r"\b(steps|cycle)\s*\(\s*([\d.]+)\s*,")

    def loop_len(self):
        """the loop actually used: never shorter than the longest steps()/cycle() sequence (+1.5 s to see its last
        stage) -- a 6-stage cycle(2.5, ...) in a loop=10 scene would otherwise never reach its last stages"""
        if not self.loop:
            return None
        key = getattr(self, "_loopk", None)
        if key and key[0] == self.version:
            return key[1]
        need = 0.0
        srcs = [getattr(e, "src", "") for _, e in self.lets] + [o.src or "" for o in self.objs.values()]
        for x in srcs:
            for m in self._SEQ.finditer(x):
                depth, n, i = 0, 1, m.end()
                while i < len(x):
                    c = x[i]
                    if c in "([":
                        depth += 1
                    elif c in ")]":
                        if depth == 0:
                            break
                        depth -= 1
                    elif c == "," and depth == 0:
                        n += 1
                    i += 1
                try:
                    need = max(need, float(m.group(2)) * n + 1.5)
                except ValueError:
                    pass
        L = min(120.0, max(self.loop, need))
        self._loopk = (self.version, L)
        return L

    def stop_sweep(self, now):
        """no more automatic sweep; a swept param glides back from where it is to its own value"""
        if self.sweep_off:
            return
        sw = self.sweep_param()
        if sw:
            p = self.params[sw]
            cur = self.sweep_value(sw, now)
            p.old, p.t0 = cur, now
        self.sweep_off = True

    def sweep_value(self, name, now):
        """declared value for 3.5 s, then out to the far end of its range and back (12 s), again and again"""
        p = self.params[name]
        v = p.value(now)
        T = now - self.t0 - 3.5
        if T <= 0:
            return v
        lo, hi = min(p.rng), max(p.rng)
        far = hi if hi - v >= v - lo else lo
        u = 0.5 - 0.5 * math.cos(2 * math.pi * (T % 12.0) / 12.0)
        return v + (far - v) * u

    def _const_env(self, now, vars_=None):
        v = {k: p.value(now) for k, p in self.params.items()}
        if vars_:
            v.update(vars_)
        return Env(v, None)

    def apply_text(self, text, now):
        stmts, errs = parse_program(text)
        for st in stmts:
            self.apply(st, now)
        return errs

    def program_text(self, max_chars=6000):
        """the current program as source (for the patcher's context)"""
        out = []
        if not self.empty:
            v = self.view
            extra = f" view={v[0]:g},{v[1]:g},{v[2]:g},{v[3]:g}" + (f" loop={self.loop:g}" if self.loop else "")
            out.append(f'scene "{self.title}"{extra}')
        if self.note:
            out.append(f'note "{self.note}"')
        for n, p in self.params.items():
            out.append(f"param {n} = {p.new:g}" + (f" {p.rng[0]:g}..{p.rng[1]:g}" if p.rng else ""))
        for n, e in self.ordered_lets():
            out.append(f"let {n} = {e.src}")
        for n, (args, e) in self.funcs.items():
            out.append(f"func {n}({', '.join(args)}) = {e.src}")
        seen = set()
        for o in self.objs.values():
            s = o.src
            if s in seen:
                continue                  # one for-loop line covers several instances
            seen.add(s)
            out.append(s)
        txt = "\n".join(out)
        return txt[-max_chars:]
