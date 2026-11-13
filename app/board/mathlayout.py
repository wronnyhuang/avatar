"""boardbrain: a small LaTeX-subset math layouter producing single-stroke polylines.

`layout_math(src, size, font, text_mode=False)` -> Box with strokes relative to the baseline-left origin
(y DOWN, same units as `size` = cap height). Supported: fractions (\\frac, \\dfrac, \\tfrac, \\binom), ^ and _
(also Unicode super/subscript digits), \\sqrt[n]{}, \\int \\sum \\prod \\lim with limits, \\left( \\right) sized
delimiters, \\text{}, \\overline \\vec \\hat \\bar \\dot \\underline, \\boxed, \\cancel, \\xrightarrow[below]{above},
\\overset/\\underset/\\stackrel, matrices/cases environments, aligned multi-line (\\\\ and &), \\ce{} chemistry,
Greek and the usual symbol commands, ASCII shorthands (->, <=, >=, !=, <=>). Unknown commands render their name
(never crash). Malformed input (unbalanced braces) is repaired best-effort.
"""
from __future__ import annotations

import math
import re

from . import geom
from .font import HandFont, glyph

AXIS = 0.43      # math axis above baseline, in cap heights
SCRIPT = 0.68
SCRIPT2 = 0.58

SYMS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ", "eta": "η",
    "theta": "θ", "vartheta": "θ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "rho": "ρ", "varrho": "ρ", "sigma": "σ", "varsigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ",
    "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω", "omicron": "ο",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ",
    "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    "times": "×", "cdot": "·", "div": "÷", "pm": "±", "mp": "∓", "le": "≤", "leq": "≤", "leqslant": "≤",
    "ge": "≥", "geq": "≥", "geqslant": "≥", "ne": "≠", "neq": "≠", "approx": "≈", "equiv": "≡", "sim": "∼",
    "simeq": "≅", "cong": "≅", "propto": "∝", "infty": "∞", "partial": "∂", "nabla": "∇", "to": "→",
    "rightarrow": "→", "longrightarrow": "→", "leftarrow": "←", "longleftarrow": "←", "gets": "←",
    "leftrightarrow": "↔", "Rightarrow": "⇒", "Longrightarrow": "⇒", "implies": "⇒", "iff": "⇔",
    "Leftrightarrow": "⇔", "rightleftharpoons": "⇌", "leftrightharpoons": "⇌", "rightleftarrows": "⇌",
    "uparrow": "↑", "downarrow": "↓", "in": "∈", "subset": "⊂", "subseteq": "⊆", "supset": "⊃", "cup": "∪",
    "cap": "∩", "forall": "∀", "exists": "∃", "neg": "¬", "lnot": "¬", "land": "∧", "wedge": "∧", "lor": "∨",
    "vee": "∨", "angle": "∠", "perp": "⊥", "parallel": "∥", "therefore": "∴", "degree": "°", "circ": "∘",
    "prime": "′", "ldots": "…", "dots": "…", "dotsc": "…", "cdots": "⋯", "emptyset": "∅", "varnothing": "∅",
    "ell": "ℓ", "hbar": "ħ", "checkmark": "✓", "langle": "⟨", "rangle": "⟩", "mid": "|", "vert": "|",
    "lvert": "|", "rvert": "|", "AA": "Å", "%": "%", "$": "$", "{": "{", "}": "}", "_": "_", "#": "#",
    "&": "&", "lbrace": "{", "rbrace": "}", "backslash": "\\", "star": "*", "ast": "*", "bullet": "·",
    "triangle": "Δ", "square": "□", "setminus": "\\", "colon": ":", "lt": "<", "gt": ">",
}
FUNCS = {"sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "exp", "lim", "max", "min", "det", "arcsin",
         "arccos", "arctan", "sinh", "cosh", "tanh", "gcd", "lcm", "mod", "deg", "dim", "ker", "Pr", "arg", "sup", "inf",
         "lg", "bmod"}
BIGOPS = {"int": "∫", "iint": "∫∫", "iiint": "∫∫∫", "oint": "∮", "sum": "∑", "prod": "∏", "bigcup": "∪",
          "bigcap": "∩", "coprod": "∏"}
REL = set("=<>≤≥≠≈≡∼≅∝→←↔⇒⇔⇌∈⊂⊆⊃:")
BIN = set("+−×÷·±∓∪∩∧∨")
SUPMAP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
          "⁺": "+", "⁻": "-", "ⁿ": "n", "ˣ": "x", "⁼": "="}
SUBMAP = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
          "₊": "+", "₋": "-", "ₓ": "x", "ₙ": "n", "ᵢ": "i", "ⱼ": "j"}


class Box:
    __slots__ = ("w", "a", "d", "strokes", "anchors")

    def __init__(self, w=0.0, a=0.0, d=0.0, strokes=None, anchors=None):
        self.w, self.a, self.d = w, a, d
        self.strokes = strokes or []
        self.anchors = anchors or {}


def _shift(strokes, dx, dy):
    return [geom.retag([(x + dx, y + dy) for x, y in s], s) for s in strokes]


def hbox(boxes):
    out = Box()
    x = 0.0
    for b in boxes:
        out.strokes += _shift(b.strokes, x, 0.0)
        for k, v in b.anchors.items():
            out.anchors.setdefault(k, v + x)
        x += b.w
        out.a = max(out.a, b.a)
        out.d = max(out.d, b.d)
    out.w = x
    return out


# ---------------------------------------------------------------- tokenizer / parser

_TOK = re.compile(r"\\([A-Za-z]+\*?|.)|(\s+)|(.)", re.S)


def _preprocess(src: str) -> str:
    s = src
    s = s.replace("\u2212", "-")
    # unicode super/sub scripts -> ^{..} / _{..}
    def rep(m, table, op):
        return op + "{" + "".join(table[c] for c in m.group(0)) + "}"
    s = re.sub("[" + "".join(SUPMAP) + "]+", lambda m: rep(m, SUPMAP, "^"), s)
    s = re.sub("[" + "".join(SUBMAP) + "]+", lambda m: rep(m, SUBMAP, "_"), s)
    s = s.replace("√", "\\sqrt ")
    return s


def tokenize(src):
    toks = []
    for m in _TOK.finditer(src):
        if m.group(1) is not None:
            toks.append(("cmd", m.group(1)))
        elif m.group(2) is not None:
            toks.append(("sp", m.group(2)))
        else:
            toks.append(("ch", m.group(3)))
    return toks


class Parser:
    def __init__(self, src, text_mode=False):
        self.toks = tokenize(_preprocess(src))
        self.i = 0
        self.text_mode = text_mode

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self):
        t = self.peek()
        self.i += 1
        return t

    def skip_sp(self):
        while self.peek() and self.peek()[0] == "sp":
            self.i += 1

    def group(self):
        """Parse one argument: {...} or a single token."""
        self.skip_sp()
        t = self.peek()
        if t is None:
            return []
        if t == ("ch", "{"):
            self.i += 1
            return self.seq(close="}")
        return self.atom_list_one()

    def opt(self):
        self.skip_sp()
        if self.peek() == ("ch", "["):
            self.i += 1
            return self.seq(close="]")
        return None

    def raw_group(self):
        """Raw text inside {...} (for \\text, \\ce, env names)."""
        self.skip_sp()
        if self.peek() != ("ch", "{"):
            t = self.next()
            if t is None:
                return ""
            return t[1] if t[0] != "cmd" else "\\" + t[1]
        self.i += 1
        depth, out = 1, []
        while self.peek() is not None:
            k, v = self.next()
            if k == "ch" and v == "{":
                depth += 1
            elif k == "ch" and v == "}":
                depth -= 1
                if depth == 0:
                    break
            out.append("\\" + v if k == "cmd" else v)
        return "".join(out)

    def atom_list_one(self):
        n = self.atom()
        return [n] if n is not None else []

    def seq(self, close=None, stop_env=None):
        nodes = []
        while True:
            t = self.peek()
            if t is None:
                return nodes
            if close and t == ("ch", close):
                self.i += 1
                return nodes
            if t == ("cmd", "right") and close == "\\right":
                return nodes
            if stop_env and t == ("cmd", "end"):
                return nodes
            if t[0] == "ch" and t[1] in "^_":
                self.i += 1
                arg = self.group()
                base = nodes.pop() if nodes and nodes[-1][0] != "space" else ("chars", "", "m")
                if base[0] == "scripts":
                    _, b, sub, sup = base
                else:
                    b, sub, sup = base, None, None
                if t[1] == "^":
                    sup = arg if sup is None else sup + arg
                else:
                    sub = arg if sub is None else sub + arg
                if b[0] == "bigop":
                    b = ("bigop", b[1], sub if sub is not None else b[2], sup if sup is not None else b[3])
                    nodes.append(b)
                else:
                    nodes.append(("scripts", b, sub, sup))
                continue
            if t[0] == "ch" and t[1] == "'" and not self.text_mode and nodes:
                self.i += 1
                nodes.append(("chars", "′", "m"))
                continue
            n = self.atom()
            if n is not None:
                nodes.append(n)

    def atom(self):
        t = self.next()
        if t is None:
            return None
        k, v = t
        if k == "sp":
            return ("space", 0.45 if self.text_mode else 0.22)
        if k == "ch":
            if v == "{":
                return ("group", self.seq(close="}"))
            if v == "}":
                return None          # stray close brace: drop
            if v == "&":
                return ("align",)
            if v == "~":
                return ("space", 0.45)
            if not self.text_mode:
                # ASCII shorthands
                rest = "".join(tv for tk, tv in self.toks[self.i:self.i + 2] if tk == "ch")
                for pat, sym in (("<=>", "⇌"), ("<->", "↔"), ("->", "→"), ("=>", "⇒"), ("<=", "≤"), (">=", "≥"),
                                 ("!=", "≠"), ("<-", "←"), ("+-", "±")):
                    if (v + rest).startswith(pat):
                        self.i += len(pat) - 1
                        return ("chars", sym, "m")
                if v == "*":
                    return ("chars", "·", "m")
                if v == "-":
                    return ("chars", "−", "m")
            return ("chars", v, "t" if self.text_mode else "m")
        # commands
        if v == "\\":
            return ("newline",)
        if v in (",", ";", ":", " ", ">"):
            return ("space", {",": 0.18, ";": 0.3, ":": 0.25, " ": 0.4, ">": 0.25}[v])
        if v == "!":
            return ("space", -0.1)
        if v in ("quad", "qquad", "enspace", "hspace"):
            if v == "hspace":
                self.raw_group()
            return ("space", {"quad": 1.0, "qquad": 2.0, "enspace": 0.5, "hspace": 1.0}[v])
        if v in ("frac", "dfrac", "tfrac", "cfrac"):
            return ("frac", self.group(), self.group())
        if v == "binom":
            return ("binom", self.group(), self.group())
        if v == "sqrt":
            idx = self.opt()
            return ("sqrt", self.group(), idx)
        if v in ("text", "textrm", "mathrm", "textbf", "mathbf", "textit", "mathit", "operatorname", "mbox",
                 "textsf", "mathsf", "texttt", "mathtt", "mathbb", "mathcal", "mathfrak", "boldsymbol", "bm",
                 "emph", "textnormal"):
            raw = self.raw_group()
            if v in ("text", "textrm", "textbf", "textit", "mbox", "textsf", "texttt", "emph", "textnormal"):
                return ("group", Parser(raw, text_mode=True).seq())
            return ("group", Parser(raw, text_mode=self.text_mode).seq())
        if v in ("ce", "pu"):
            return ("group", Parser(chem_to_tex(self.raw_group())).seq())
        if v in BIGOPS:
            return ("bigop", BIGOPS[v], None, None)
        if v == "lim" or v in ("max", "min", "sup", "inf") :
            return ("bigop", v, None, None)
        if v in FUNCS:
            return ("func", v)
        if v in ("left", "right", "bigl", "bigr", "Bigl", "Bigr", "big", "Big", "bigg", "Bigg", "biggl", "biggr"):
            if v == "left":
                d = self._delim()
                body = self.seq(close="\\right")
                rd = "."
                if self.peek() == ("cmd", "right"):
                    self.i += 1
                    rd = self._delim()
                return ("delim", d, body, rd)
            if v == "right":
                self._delim()
                return None
            return None   # size hints: ignore, the next token is the delimiter itself
        if v in ("overline", "bar", "vec", "hat", "widehat", "dot", "ddot", "tilde", "widetilde", "underline",
                 "overrightarrow", "overleftrightarrow"):
            return ("accent", v, self.group())
        if v in ("boxed", "fbox", "framebox"):
            return ("boxed", self.group())
        if v in ("cancel", "bcancel", "xcancel", "sout", "cancelto"):
            if v == "cancelto":
                self.group()
            return ("cancel", self.group())
        if v in ("xrightarrow", "xleftarrow", "xrightleftharpoons", "xleftrightarrow", "xLeftrightarrow",
                 "xRightarrow"):
            below = self.opt()
            above = self.group()
            sym = {"xrightarrow": "→", "xleftarrow": "←", "xrightleftharpoons": "⇌", "xleftrightarrow": "↔",
                   "xLeftrightarrow": "⇔", "xRightarrow": "⇒"}[v]
            return ("xarrow", sym, above, below)
        if v in ("overset", "stackrel"):
            top = self.group()
            return ("stack", self.group(), top, None)
        if v == "underset":
            bot = self.group()
            return ("stack", self.group(), None, bot)
        if v in ("color", "textcolor", "colorbox"):
            self.raw_group()
            return None if v == "color" else ("group", self.group())
        if v in ("displaystyle", "textstyle", "scriptstyle", "limits", "nolimits", "nonumber", "notag",
                 "label", "tag", "phantom", "hfill", "centering", "middle"):
            if v in ("label", "tag", "phantom"):
                self.raw_group()
            return None
        if v == "begin":
            env = self.raw_group().strip()
            if env in ("array", "tabular"):
                self.raw_group()
            rows = self._env_rows()
            return ("env", env.rstrip("*"), rows)
        if v == "end":
            self.raw_group()
            return None
        if v in ("deg",):
            return ("chars", "°", "m")
        if v in SYMS:
            return ("chars", SYMS[v], "m")
        if v == "not":
            n = self.atom()
            return ("cancel", [n]) if n else None
        if len(v) == 1:
            return ("chars", v, "m")
        return ("chars", v, "t")   # unknown command: write its name rather than failing

    def _delim(self):
        self.skip_sp()
        t = self.next()
        if t is None:
            return "."
        k, v = t
        if k == "cmd":
            return {"{": "{", "}": "}", "langle": "⟨", "rangle": "⟩", "lvert": "|", "rvert": "|", "vert": "|",
                    "|": "‖", "lbrack": "[", "rbrack": "]", "lbrace": "{", "rbrace": "}", "lfloor": "⌊",
                    "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉"}.get(v, ".")
        return v

    def _env_rows(self):
        rows, row, cell = [], [], []
        while True:
            t = self.peek()
            if t is None:
                break
            if t == ("cmd", "end"):
                self.i += 1
                self.raw_group()
                break
            if t == ("cmd", "\\"):
                self.i += 1
                row.append(cell)
                rows.append(row)
                row, cell = [], []
                continue
            if t == ("ch", "&"):
                self.i += 1
                row.append(cell)
                cell = []
                continue
            if t[0] == "ch" and t[1] in "^_":
                # reuse seq machinery for scripts inside cells
                sub = self.seq_until_cell()
                cell += sub
                continue
            n = self.atom()
            if n is not None:
                cell.append(n)
        if cell or row:
            row.append(cell)
            rows.append(row)
        return [r for r in rows if any(c for c in r)]

    def seq_until_cell(self):
        # handle a script attached to the previous atom of the current cell: parse "^x" as a bare scripts node
        t = self.next()
        arg = self.group()
        return [("scripts", ("chars", "", "m"), arg if t[1] == "_" else None, arg if t[1] == "^" else None)]


# ---------------------------------------------------------------- chemistry

_CHEM_ARROWS = [("<=>", "\\rightleftharpoons "), ("<->", "\\leftrightarrow "), ("->", "\\rightarrow "),
                ("<-", "\\leftarrow "), ("=", " = ")]


def chem_to_tex(s: str) -> str:
    """mhchem-ish: 2H2 + O2 -> 2H2O, Fe^3+, SO4^{2-}, CO3^2-, (aq)/(s)/(l)/(g) kept as text."""
    out = []
    i = 0
    n = len(s)
    prev_alpha = False
    while i < n:
        c = s[i]
        hit = False
        for pat, rep in _CHEM_ARROWS:
            if s.startswith(pat, i):
                # arrow with a condition label: ->[heat]
                j = i + len(pat)
                if pat != "=" and j < n and s[j] == "[":
                    k = s.find("]", j)
                    if k > 0:
                        sym = {"->": "xrightarrow", "<=>": "xrightleftharpoons", "<-": "xleftarrow",
                               "<->": "xleftrightarrow"}[pat]
                        out.append(" \\%s{\\text{%s}} " % (sym, s[j + 1:k]))
                        i = k + 1
                        hit = True
                        break
                out.append(" " + rep + " ")
                i = j
                hit = True
                break
        if hit:
            prev_alpha = False
            continue
        if c == "(" and re.match(r"\((aq|s|l|g)\)", s[i:]):
            m = re.match(r"\((aq|s|l|g)\)", s[i:])
            out.append("_{\\text{(%s)}}" % m.group(1))
            i += len(m.group(0))
            prev_alpha = False
            continue
        if c == "^":
            m = re.match(r"\^(\{[^}]*\}|[0-9]*[+-]|[0-9]+)", s[i:])
            if m:
                body = m.group(1).strip("{}")
                out.append("^{%s}" % body)
                i += len(m.group(0))
                continue
        if c.isdigit() and prev_alpha:
            j = i
            while j < n and s[j].isdigit():
                j += 1
            out.append("_{%s}" % s[i:j])
            i = j
            continue
        if c in "+-" and prev_alpha and (i + 1 == n or s[i + 1] in " )^,") and not (i + 1 < n and s[i + 1] == ">"):
            out.append("^{%s}" % c)
            i += 1
            continue
        if c == " ":
            out.append("\\, " if out and out[-1].strip() and i + 1 < n and s[i + 1] not in "+-<=" else " ")
            i += 1
            prev_alpha = False
            continue
        out.append(c if c not in "\\" else "")
        prev_alpha = c.isalpha() or c in ")]"
        i += 1
    return "".join(out)


# ---------------------------------------------------------------- layout


class Layouter:
    def __init__(self, font: HandFont):
        self.font = font
        self.k = 0

    def chars(self, s, size, mode):
        if not s:
            return Box(0, 0, 0)
        self.k += 1
        pad = 0.0
        if mode == "m" and len(s) == 1 and s in REL:
            pad = 0.3
        elif mode == "m" and len(s) == 1 and s in BIN:
            pad = 0.22
        elif mode == "m" and s in (",", ";"):
            pad = 0.0
        strokes, adv, _ = self.font.run(s, size, pad * size, 0.0, seed_offset=self.k)
        a, d = self._metrics(s, size)
        b = Box(adv + 2 * pad * size + (0.12 * size if s in (",", ";") and mode == "m" else 0), a, d, strokes)
        if s in "=≈≡≤≥<>⇌→" and len(s) == 1:
            b.anchors["eq"] = 0.0
        return b

    def _metrics(self, s, size):
        a = 0.0
        d = 0.0
        for ch in s:
            if ch == " ":
                continue
            _, st, _ = glyph(ch)
            ys = [y for stroke in st for _, y in stroke]
            if ys:
                a = max(a, -min(ys))
                d = max(d, max(ys))
        return max(a, 0.72) * size, max(d, 0.0) * size

    def seq(self, nodes, size, level=0):
        # split at newline nodes -> multiline (aligned)
        if any(n[0] == "newline" for n in nodes):
            lines, cur = [], []
            for n in nodes:
                if n[0] == "newline":
                    lines.append(cur)
                    cur = []
                else:
                    cur.append(n)
            lines.append(cur)
            return self.lines([l for l in lines if l], size, level)
        # drop source spaces next to operators/relations (they get their own spacing)
        def _op(n):
            return n is not None and n[0] == "chars" and n[2] == "m" and len(n[1]) == 1 and (n[1] in REL or n[1] in BIN)
        kept = []
        for i, n in enumerate(nodes):
            if n[0] == "space" and n[1] > 0 and n[1] < 0.3:
                prev = kept[-1] if kept else None
                nxt = nodes[i + 1] if i + 1 < len(nodes) else None
                if prev is None or nxt is None or _op(prev) or _op(nxt) or prev[0] == "space":
                    continue
            kept.append(n)
        nodes = kept
        boxes = []
        for n in nodes:
            b = self.node(n, size, level)
            if b is not None:
                boxes.append(b)
        return hbox(boxes)

    def lines(self, lines, size, level=0, gap=0.55):
        boxes = [self.seq(l, size, level) for l in lines]
        if not boxes:
            return Box()
        # align on '&' anchor (else on first relation), else left
        keys = [b.anchors.get("align", b.anchors.get("eq")) for b in boxes]
        use = all(k is not None for k in keys) and len(boxes) > 1
        ax = max(keys) if use else 0.0
        out = Box()
        y = 0.0
        for i, b in enumerate(boxes):
            dx = (ax - keys[i]) if use else 0.0
            if i:
                y += boxes[i - 1].d + gap * size + b.a
            out.strokes += _shift(b.strokes, dx, y)
            out.w = max(out.w, dx + b.w)
            for k, v in b.anchors.items():
                out.anchors.setdefault(k, v + dx)
        out.a = boxes[0].a
        out.d = y + boxes[-1].d
        return out

    def node(self, n, size, level):
        kind = n[0]
        if kind == "chars":
            return self.chars(n[1], size, n[2])
        if kind == "space":
            return Box(n[1] * size, 0, 0)
        if kind == "align":
            return Box(0, 0, 0, anchors={"align": 0.0})
        if kind == "group":
            return self.seq(n[1], size, level)
        if kind == "func":
            b = self.chars(n[1], size, "t")
            b.w += 0.15 * size
            return b
        if kind == "frac":
            return self.frac(n[1], n[2], size, level)
        if kind == "binom":
            inner = self.stackv(n[1], n[2], size * (0.8 if level == 0 else SCRIPT), level + 1, bar=False)
            return self.delimit("(", inner, ")", size)
        if kind == "scripts":
            return self.scripts(n[1], n[2], n[3], size, level)
        if kind == "sqrt":
            return self.sqrt(n[1], n[2], size, level)
        if kind == "bigop":
            return self.bigop(n[1], n[2], n[3], size, level)
        if kind == "delim":
            inner = self.seq(n[2], size, level)
            return self.delimit(n[1], inner, n[3], size)
        if kind == "accent":
            return self.accent(n[1], self.seq(n[2], size, level), size)
        if kind == "boxed":
            b = self.seq(n[1], size, level)
            p = 0.25 * size
            r = [(0, -b.a - p), (b.w + 2 * p, -b.a - p), (b.w + 2 * p, b.d + p), (0, b.d + p), (0, -b.a - p)]
            return Box(b.w + 2 * p, b.a + p, b.d + p, [geom.resample(r, size * 0.1)] + _shift(b.strokes, p, 0))
        if kind == "cancel":
            b = self.seq(n[1], size, level)
            line = geom.resample([(-0.05 * size, b.d * 0.6 + 0.1 * size), (b.w + 0.05 * size, -b.a * 0.95)], size * 0.1)
            return Box(b.w, b.a, b.d, b.strokes + [line])
        if kind == "xarrow":
            return self.xarrow(n[1], n[2], n[3], size, level)
        if kind == "stack":
            base = self.seq(n[1], size, level)
            top = self.seq(n[2], size * SCRIPT, level + 1) if n[2] else None
            bot = self.seq(n[3], size * SCRIPT, level + 1) if n[3] else None
            return self.over_under(base, top, bot, size)
        if kind == "env":
            return self.env(n[1], n[2], size, level)
        if kind == "newline":
            return None
        return None

    def frac(self, num, den, size, level):
        s2 = size * (0.85 if level == 0 else SCRIPT)
        return self.stackv(num, den, s2, level + 1, bar=True, outer=size)

    def stackv(self, num, den, s2, level, bar=True, outer=None):
        outer = outer or s2
        nb = self.seq(num, s2, level)
        db = self.seq(den, s2, level)
        pad = 0.15 * outer
        w = max(nb.w, db.w) + 2 * pad
        axis = -AXIS * outer
        gap = 0.22 * outer
        strokes = []
        nx = (w - nb.w) / 2
        ny = axis - gap - nb.d
        strokes += _shift(nb.strokes, nx, ny)
        if bar:
            strokes.append(geom.resample([(0.02 * outer, axis), (w - 0.02 * outer, axis)], outer * 0.12))
        dx = (w - db.w) / 2
        dy = axis + gap + db.a
        strokes += _shift(db.strokes, dx, dy)
        sp = 0.12 * outer
        return Box(w + 2 * sp, -(ny - nb.a), dy + db.d, _shift(strokes, sp, 0))

    def scripts(self, base, sub, sup, size, level):
        bb = self.node(base, size, level) if base else Box()
        if bb is None:
            bb = Box()
        ss = size * (SCRIPT if level == 0 else SCRIPT2)
        strokes = list(bb.strokes)
        w = bb.w
        a, d = bb.a, bb.d
        x0 = bb.w + 0.04 * size
        wsup = wsub = 0.0
        if sup:
            sb = self.seq(sup, ss, level + 1)
            y = -max(0.55 * size, bb.a - 0.45 * size) if base and base[0] != "chars" else -0.55 * size
            if sub:
                y -= 0.08 * size
            if base and base[0] == "chars" and bb.a > 1.05 * size:
                y = -(bb.a - 0.4 * size)
            strokes += _shift(sb.strokes, x0, y)
            wsup = sb.w
            a = max(a, -y + sb.a)
        if sub:
            sb = self.seq(sub, ss, level + 1)
            y = max(0.3 * size, bb.d * 0.7) if base and base[0] not in ("chars", "func") else 0.3 * size
            if sup:
                y += 0.08 * size
            strokes += _shift(sb.strokes, x0, y)
            wsub = sb.w
            d = max(d, y + sb.d)
        w = x0 + max(wsup, wsub) + 0.06 * size
        return Box(w, a, d, strokes, dict(bb.anchors))

    def sqrt(self, body, idx, size, level):
        b = self.seq(body, size, level)
        top = b.a + 0.22 * size
        bot = max(b.d, 0.0) + 0.05 * size
        hook = 0.55 * size
        x1 = 0.12 * size
        rad = [(0, -0.45 * size), (x1, -0.55 * size), (x1 + 0.22 * size, bot), (hook, -top), (hook + b.w + 0.25 * size, -top),
               (hook + b.w + 0.25 * size, -top + 0.12 * size)]
        strokes = [geom.resample(rad, size * 0.08)] + _shift(b.strokes, hook + 0.12 * size, 0)
        w = hook + b.w + 0.4 * size
        if idx:
            ib = self.seq(idx, size * SCRIPT2, level + 2)
            strokes += _shift(ib.strokes, max(0.0, 0.05 * size), -0.65 * size)
        return Box(w, top + 0.05 * size, bot, strokes)

    def bigop(self, sym, sub, sup, size, level):
        ss = size * (SCRIPT if level == 0 else SCRIPT2)
        if sym in ("lim", "max", "min", "sup", "inf"):
            base = self.chars(sym, size, "t")
            sb = self.seq(sub, ss, level + 1) if sub else None
            out = self.over_under(base, None, sb, size)
            out.w += 0.18 * size
            return out
        big = 1.6 if level == 0 else 1.1
        if sym in ("∫", "∫∫", "∫∫∫", "∮"):
            boxes = []
            for _ in sym if sym != "∮" else "∫":
                gb = self.chars("∫", size * big, "m")
                # centre on axis
                boxes.append(gb)
            g = hbox(boxes)
            dy = (-AXIS * size) - (-AXIS * size * big)
            g = Box(g.w, g.a - dy, g.d + dy, _shift(g.strokes, 0, dy))
            if sym == "∮":
                g.strokes.append(geom.ellipse(g.w * 0.45, -AXIS * size, 0.18 * size, 0.18 * size))
            strokes = list(g.strokes)
            a, d = g.a, g.d
            wl = 0.0
            if sup:
                sb = self.seq(sup, ss, level + 1)
                y = -g.a + sb.a * 0.6
                strokes += _shift(sb.strokes, g.w + 0.12 * size, y)
                wl = max(wl, sb.w + 0.1 * size)
                a = max(a, -y + sb.a)
            if sub:
                sb = self.seq(sub, ss, level + 1)
                y = g.d
                strokes += _shift(sb.strokes, g.w - 0.18 * size, y)
                wl = max(wl, sb.w - 0.2 * size)
                d = max(d, y + sb.d)
            return Box(g.w + wl + 0.25 * size, a, d, strokes)
        bs = 1.6 if level == 0 else 1.1
        gb = self.chars(sym, size * bs, "m")
        dy = (-AXIS * size) - (-AXIS * size * bs)
        gb = Box(gb.w, gb.a - dy, gb.d + dy, _shift(gb.strokes, 0, dy))
        tb = self.seq(sup, ss, level + 1) if sup else None
        bb = self.seq(sub, ss, level + 1) if sub else None
        out = self.over_under(gb, tb, bb, size)
        out.w += 0.15 * size
        return out

    def over_under(self, base, top, bot, size):
        w = max(base.w, top.w if top else 0, bot.w if bot else 0)
        strokes = _shift(base.strokes, (w - base.w) / 2, 0)
        a, d = base.a, base.d
        if top:
            y = -base.a - 0.15 * size - top.d
            strokes += _shift(top.strokes, (w - top.w) / 2, y)
            a = -y + top.a
        if bot:
            y = base.d + 0.18 * size + bot.a
            strokes += _shift(bot.strokes, (w - bot.w) / 2, y)
            d = y + bot.d
        return Box(w, a, d, strokes, {k: v + (w - base.w) / 2 for k, v in base.anchors.items()})

    def xarrow(self, sym, above, below, size, level):
        ab = self.seq(above, size * SCRIPT, level + 1) if above else None
        bb = self.seq(below, size * SCRIPT, level + 1) if below else None
        L = max(1.3 * size, (ab.w if ab else 0) + 0.5 * size, (bb.w if bb else 0) + 0.5 * size)
        ax = -AXIS * size
        h = 0.2 * size
        if sym == "⇌":
            st = [[(0.1 * size, ax - 0.1 * size), (L, ax - 0.1 * size), (L - 1.2 * h, ax - 0.1 * size - h)],
                  [(L, ax + 0.1 * size), (0.1 * size, ax + 0.1 * size), (0.1 * size + 1.2 * h, ax + 0.1 * size + h)]]
        elif sym == "←":
            st = [[(L, ax), (0.1 * size, ax)], [(0.1 * size + h, ax - h), (0.1 * size, ax), (0.1 * size + h, ax + h)]]
        elif sym in ("↔", "⇔"):
            st = [[(0.1 * size, ax), (L, ax)], [(L - h, ax - h), (L, ax), (L - h, ax + h)],
                  [(0.1 * size + h, ax - h), (0.1 * size, ax), (0.1 * size + h, ax + h)]]
        else:
            st = [[(0.1 * size, ax), (L, ax)], [(L - h, ax - h), (L, ax), (L - h, ax + h)]]
        st = [geom.resample(s, size * 0.1) for s in st]
        base = Box(L + 0.1 * size, -ax + 0.3 * size, 0.0, st)
        out = self.over_under(base, ab, bb, size)
        pad = 0.25 * size
        out.strokes = _shift(out.strokes, pad, 0)
        out.w += 2 * pad
        out.anchors["eq"] = 0.0
        return out

    def delimit(self, ld, inner, rd, size):
        h_top = max(inner.a, 0.8 * size) + 0.1 * size
        h_bot = max(inner.d, 0.15 * size) + 0.1 * size
        lb = self.delim_box(ld, h_top, h_bot, size, left=True)
        rb = self.delim_box(rd, h_top, h_bot, size, left=False)
        pad = Box(0.1 * size if h_top + h_bot > 1.35 * size else 0.03 * size, 0, 0)
        return hbox([lb, pad, inner, pad, rb])

    def delim_box(self, d, top, bot, size, left):
        if d in (".", "", None):
            return Box(0.05 * size, 0, 0)
        H = top + bot
        if H <= 1.35 * size and d in "()[]|{}":
            return self.chars(d, size, "m") if d not in "|" else self._vbar(top, bot, size)
        w = min(0.45 * size, 0.18 * size + 0.08 * H)
        if d in "()":
            sgn = 1 if left else -1
            x0 = w * 0.85 if left else w * 0.15
            pts = geom.quad((x0, -top), (x0 - sgn * w * 1.4, (bot - top) / 2), (x0, bot), n=24)
        elif d in "[]⌊⌋⌈⌉":
            xi, xo = (w * 0.85, w * 0.25) if left else (w * 0.15, w * 0.75)
            pts = [(xi, -top), (xo, -top), (xo, bot), (xi, bot)]
            if d in "⌊⌋":
                pts = pts[1:]
            if d in "⌈⌉":
                pts = pts[:3]
        elif d in "{}":
            sgn = 1 if left else -1
            xa = w * 0.8 if left else w * 0.2
            xm = w * 0.2 if left else w * 0.8
            mid = (bot - top) / 2
            pts = (geom.quad((xa, -top), (xa - sgn * w * 0.45, -top), (xa - sgn * w * 0.4, (mid - top) / 2), 8) +
                   geom.quad((xa - sgn * w * 0.4, (mid - top) / 2), (xa - sgn * w * 0.4, mid), (xm, mid), 8)[1:] +
                   geom.quad((xm, mid), (xa - sgn * w * 0.4, mid), (xa - sgn * w * 0.4, (mid + bot) / 2), 8)[1:] +
                   geom.quad((xa - sgn * w * 0.4, (mid + bot) / 2), (xa - sgn * w * 0.45, bot), (xa, bot), 8)[1:])
        elif d in "⟨⟩":
            pts = [(w, -top), (0.1 * w, (bot - top) / 2), (w, bot)] if left else [(0.1 * w, -top), (w, (bot - top) / 2), (0.1 * w, bot)]
        elif d in "|‖":
            b = self._vbar(top, bot, size)
            if d == "‖":
                b2 = self._vbar(top, bot, size)
                return hbox([b, b2])
            return b
        else:
            return self.chars(d, size, "m")
        return Box(w + 0.08 * size, top, bot, [geom.resample(pts, size * 0.1)])

    def _vbar(self, top, bot, size):
        x = 0.15 * size
        return Box(0.3 * size, top, bot, [geom.resample([(x, -top), (x, bot)], size * 0.1)])

    def accent(self, kind, b, size):
        y = -b.a - 0.22 * size
        if kind == "underline":
            y = b.d + 0.15 * size
            line = [(0, y), (b.w, y)]
            return Box(b.w, b.a, -y + 0.05 * size if False else y + 0.05 * size, b.strokes + [geom.resample(line, size * 0.1)])
        if kind in ("overline", "bar"):
            st = [[(0.05 * size, y), (b.w - 0.05 * size, y)]]
        elif kind in ("vec", "overrightarrow"):
            L = max(b.w, 0.6 * size)
            h = 0.14 * size
            st = [[(0.0, y), (L, y)], [(L - h, y - h), (L, y), (L - h, y + h)]]
        elif kind == "overleftrightarrow":
            h = 0.14 * size
            st = [[(0, y), (b.w, y)], [(b.w - h, y - h), (b.w, y), (b.w - h, y + h)], [(h, y - h), (0, y), (h, y + h)]]
        elif kind in ("hat", "widehat"):
            st = [[(b.w * 0.15, y + 0.05 * size), (b.w / 2, y - 0.15 * size), (b.w * 0.85, y + 0.05 * size)]]
        elif kind in ("tilde", "widetilde"):
            st = [[(b.w * (0.15 + 0.7 * i / 8), y - 0.06 * size * math.sin(2 * math.pi * i / 8)) for i in range(9)]]
        elif kind == "dot":
            st = [geom.ellipse(b.w / 2, y, 0.04 * size, 0.04 * size, n=6)]
        elif kind == "ddot":
            st = [geom.ellipse(b.w / 2 - 0.12 * size, y, 0.04 * size, 0.04 * size, n=6),
                  geom.ellipse(b.w / 2 + 0.12 * size, y, 0.04 * size, 0.04 * size, n=6)]
        else:
            st = []
        st = [geom.resample(s, size * 0.1) for s in st]
        return Box(max(b.w, 0.6 * size if kind in ("vec", "overrightarrow") else b.w), -y + 0.2 * size, b.d, b.strokes + st, b.anchors)

    def env(self, name, rows, size, level):
        if name in ("aligned", "align", "gathered", "split", "eqnarray", "gather", "alignat", "multline"):
            lines = []
            for r in rows:
                line = []
                for ci, c in enumerate(r):
                    if ci:
                        line.append(("align",))
                    line += c
                lines.append(line)
            return self.lines(lines, size, level)
        # grid: matrix / cases / array
        cells = [[self.seq(c, size * (0.9 if level == 0 else SCRIPT), level + 1) for c in r] for r in rows]
        ncol = max(len(r) for r in cells) if cells else 0
        colw = [max((r[j].w for r in cells if j < len(r)), default=0) for j in range(ncol)]
        cg = 0.7 * size
        rg = 0.35 * size
        strokes = []
        y = 0.0
        heights = []
        for i, r in enumerate(cells):
            a = max((b.a for b in r), default=0.7 * size)
            d = max((b.d for b in r), default=0)
            if i:
                y += heights[-1][1] + rg + a
            heights.append((a, d))
            x = 0.0
            for j, b in enumerate(r):
                dx = x if name == "cases" else x + (colw[j] - b.w) / 2
                strokes += _shift(b.strokes, dx, y)
                x += colw[j] + cg
        total_w = sum(colw) + cg * max(0, ncol - 1)
        top = heights[0][0] if heights else 0
        bot = y + (heights[-1][1] if heights else 0)
        # centre on axis
        mid = (bot - top) / 2
        dy = -AXIS * size - mid
        inner = Box(total_w, top - dy, bot + dy, _shift(strokes, 0, dy))
        delims = {"pmatrix": ("(", ")"), "bmatrix": ("[", "]"), "vmatrix": ("|", "|"), "Bmatrix": ("{", "}"),
                  "Vmatrix": ("‖", "‖"), "cases": ("{", "."), "matrix": (".", "."), "array": (".", "."),
                  "smallmatrix": (".", "."), "tabular": (".", ".")}
        l, r = delims.get(name, (".", "."))
        return self.delimit(l, inner, r, size)


def layout_math(src: str, size: float, font: HandFont | None = None, text_mode: bool = False) -> Box:
    font = font or HandFont()
    p = Parser(src, text_mode=text_mode)
    nodes = p.seq()
    L = Layouter(font)
    return L.seq(nodes, size)


def split_text_math(src: str):
    """Split 'Area = $\\pi r^2$ cm' into [(text, False), (math, True), ...]."""
    parts = []
    pat = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)", re.S)
    pos = 0
    for m in pat.finditer(src):
        if m.start() > pos:
            parts.append((src[pos:m.start()], False))
        parts.append((m.group(1) or m.group(2) or m.group(3), True))
        pos = m.end()
    if pos < len(src):
        parts.append((src[pos:], False))
    return parts


def looks_like_tex(s: str) -> bool:
    return bool(re.search(r"\\[A-Za-z]+|[\^_]\{|[\^_][A-Za-z0-9]", s))
