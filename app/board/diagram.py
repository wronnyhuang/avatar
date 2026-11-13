"""boardbrain: Mermaid diagram -> single-stroke polylines a hand can trace on a chalkboard.

    render_mermaid(src, font, text_size, max_w=None, max_h=None) -> {
        "strokes": [[(x, y), ...], ...],   # ordered as a person draws them, resampled at ~0.06*text_size
        "w": float, "h": float,            # extent; origin top-left, y DOWN, units of text_size (= label cap height)
        "nodes": {id: (x0, y0, x1, y1)},   # node boxes in the same frame
        "edge_labels": [(x0, y0, x1, y1)], # flowchart edge-label boxes (never within 0.3*text_size of a node box,
                                           # never on another label; a 0.35*text_size margin rims w/h)
        "warnings": [str],
    }

Supported input (model-written, one line with ';' separators or multi-line, optional ``` fences, %% comments):
  * graph / flowchart  TD TB BT LR RL; shapes [rect] (rounded) {diamond} ((circle)) (((double))) ([stadium])
    [[subroutine]] [(cylinder)] >asym] {{hexagon}} [/para/] [\\para\\] [/trap\\] [\\trap/]; bare ids; "quoted"
    labels; edges --> --- -.-> -.- ==> === ~~~ <--> --o --x, `--text-->`, `-- text -->`, `-->|text|`, chains
    a-->b-->c, `a & b --> c & d`; <br> line breaks; inline $math$ in any label; subgraph ... end (flattened, a
    dashed labelled box is drawn around the members when it does not collide with anything);
    classDef/class/style/linkStyle/click/direction/accTitle are ignored; A@{ label: "x" } partially.
  * stateDiagram(-v2) (transitions, [*] start/end dots) and sequenceDiagram (messages) are converted to a
    flowchart best-effort, with a warning.
  * timeline (title, section, `period : event : event`, `: event` continuation) -> horizontal axis with ticks
    (vertical if the target aspect is tall).
  * mindmap (indentation tree, shapes) -> two-sided left/right tree around the root.

Layout: layered (Sugiyama-lite): DFS cycle breaking, longest-path layering (sources pulled down next to their
children), dummy vertices for long edges, barycenter sweeps + adjacent transposition for crossing reduction,
isotonic (PAVA) coordinate assignment that centres parents over children with guaranteed separation. A graph
that is one simple cycle of 3-10 nodes (water cycle, ...) is drawn on a ring with curved arrows instead.
Edges are clipped exactly against each shape's border (point-in-shape bisection), long edges are smoothed
(Catmull-Rom) through their dummy vertices, parallel edges between one pair bow apart, edge labels sit beside
their edge at the free spot with the least overlap with nodes, labels and other lines.

Stroke order (how a teacher draws it): nodes are visited in layer order (top to bottom for TD, left to right
for LR; within a layer along the cross axis). Before a node is drawn, every edge INTO it from an already drawn
node is drawn (shaft, arrowhead, edge label) - "box, arrow, box". Edges whose source is drawn after their
target (back edges) are drawn right after their source node. A node = its outline (one continuous stroke
starting top-left and going clockwise where the shape allows) followed by its label strokes. Subgraph boxes come
last. Nothing ever raises: unparsable pieces are skipped and reported in "warnings".
"""
from __future__ import annotations

import math
import os
import random
import re

from . import geom
from .font import HandFont, hash_text
from .mathlayout import layout_math, split_text_math

SPACING = 0.06            # resample step, x text_size
LINE_A, LINE_D, LINE_GAP = 1.0, 0.22, 0.42   # label line metrics (x size)
PAD_X, PAD_Y = 0.6, 0.5
NODE_SEP = 1.3            # min gap between nodes in a layer (x size)
DUMMY_W = 0.5
GAP_TD, GAP_LR = 1.9, 2.6  # min gap between layers along the flow
ARROW = 0.5
MARGIN = 0.35             # empty border inside w/h (x size)

DIRS = ("TD", "TB", "BT", "LR", "RL")


# ======================================================================= labels

def _clean_label(s: str) -> str:
    s = (s or "").strip()
    if len(s) >= 2 and s[0] == s[-1] == '"':
        s = s[1:-1]
    s = s.replace("`", "")
    out = []
    for part, is_m in split_text_math(s):
        if is_m:
            out.append("$" + part + "$")
            continue
        p = re.sub(r"<\s*br\s*/?\s*>", "\n", part, flags=re.I)
        p = re.sub(r"\\n(?![a-z])", "\n", p)
        p = re.sub(r"</?[A-Za-z][^<>]*>", "", p)
        p = re.sub(r"fa[bsr]?:fa-[\w-]+\s*", "", p)
        p = re.sub(r"[^\s$]*[\u2070-\u209f\u00b2\u00b3\u00b9][^\s$]*", lambda m: "$" + m.group(0) + "$", p)
        for a, b in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("#quot;", '"'),
                     ("&nbsp;", " "), ("&#39;", "'"), ("**", ""), ("__", "")):
            p = p.replace(a, b)
        out.append(p)
    s = "".join(out)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in s.split("\n")]
    return "\n".join(ln for ln in lines if ln) or ""


class LabStroke(list):
    """A label glyph stroke: .lab = (serial, text, role) of the label draw it came from (labels are tagged
    so the item can say which strokes are labels — collision checks / TEACH never need to guess)."""
    __slots__ = ("lab",)


_LAB_SERIAL = [0]


class _Label:
    """A measured multi-line label: lines of text with $math$, centred."""

    def __init__(self, R, text, wrap):
        self.text = text
        self.lines = []          # (string, w, a, d)
        paras = text.split("\n") if text else []
        for para in paras:
            for ln in R.wrap(para, wrap):
                w, a, d = R.measure(ln)
                self.lines.append((ln, w, a, d))
        self.w = max((l[1] for l in self.lines), default=0.0)
        s = R.size
        self.h = sum(l[2] + l[3] for l in self.lines) + LINE_GAP * s * max(0, len(self.lines) - 1)

    def draw(self, R, x0, y0, seed, role="node"):
        """Strokes with the label box's top-left at (x0, y0)."""
        out = []
        y = y0
        for i, (ln, w, a, d) in enumerate(self.lines):
            base = y + a
            pen = x0 + (self.w - w) / 2
            for k, (part, is_m) in enumerate(split_text_math(ln)):
                if is_m:
                    b = R.math(part)
                    out += [[(px + pen, py + base) for px, py in st] for st in b.strokes]
                    pen += b.w
                else:
                    st, adv, unk = R.font.run(part, R.size, pen, base, seed_offset=seed * 31 + i * 7 + k)
                    for ch in unk:
                        R.unknown.setdefault(ch, None)
                    out += st
                    pen += adv
            y = base + d + LINE_GAP * R.size
        _LAB_SERIAL[0] += 1
        tag = (_LAB_SERIAL[0], self.text, role)
        tagged = []
        for st in out:
            t = LabStroke(st)
            t.lab = tag
            tagged.append(t)
        return tagged


class _Ctx:
    def __init__(self, font, size, warnings, seed):
        self.font, self.size, self.warnings = font, size, warnings
        self._math, self._labels = {}, {}
        self.unknown = {}
        self.rnd = random.Random(seed)

    def warn(self, msg):
        if msg not in self.warnings and len(self.warnings) < 40:
            self.warnings.append(msg)

    def math(self, src):
        b = self._math.get(src)
        if b is None:
            try:
                b = layout_math(src, self.size, self.font)
            except Exception as ex:   # never let a bad formula sink the diagram
                self.warn(f"math label {src!r} failed ({type(ex).__name__}); drawn as text")
                from .mathlayout import Box
                st, adv, _ = self.font.run(src, self.size)
                b = Box(adv, self.size, 0.2 * self.size, st)
            self._math[src] = b
        return b

    def measure(self, ln):
        s = self.size
        w, a, d = 0.0, LINE_A * s, LINE_D * s
        for part, is_m in split_text_math(ln):
            if is_m:
                b = self.math(part)
                w += b.w
                a, d = max(a, b.a), max(d, b.d)
            else:
                w += HandFont.width(part, s)
        return w, a, d

    def wrap(self, para, wrap):
        if wrap is None or wrap <= 0:
            return [para]
        toks, cur, in_m = [], "", False
        for ch in para:
            if ch == "$":
                in_m = not in_m
            if ch == " " and not in_m:
                if cur:
                    toks.append(cur)
                cur = ""
            else:
                cur += ch
        if cur:
            toks.append(cur)
        if not toks:
            return []
        broken = []
        for t in toks:      # a single unbreakable word longer than the wrap is split by characters
            lim = max(wrap, 12.0 * self.size)
            if "$" not in t and HandFont.width(t, self.size) > lim:
                cur = ""
                for ch in t:
                    if cur and HandFont.width(cur + ch, self.size) > lim:
                        broken.append(cur)
                        cur = ""
                    cur += ch
                if cur:
                    broken.append(cur)
            else:
                broken.append(t)
        toks = broken
        lines, line = [], toks[0]
        for t in toks[1:]:
            cand = line + " " + t
            if self.measure(cand)[0] <= wrap:
                line = cand
            else:
                lines.append(line)
                line = t
        lines.append(line)
        return lines

    def label(self, text, wrap):
        key = (text, None if wrap is None else round(wrap, 3))
        L = self._labels.get(key)
        if L is None:
            L = self._labels[key] = _Label(self, text, wrap)
        return L


# ======================================================================= model

class Node:
    __slots__ = ("id", "label", "shape", "sub", "w", "h", "x", "y", "lab", "explicit")

    def __init__(self, nid, label=None, shape="rect"):
        self.id, self.label, self.shape = nid, label if label is not None else nid, shape
        self.sub = None
        self.explicit = label is not None
        self.w = self.h = self.x = self.y = 0.0
        self.lab = None

    def rect(self):
        return (self.x - self.w / 2, self.y - self.h / 2, self.x + self.w / 2, self.y + self.h / 2)


class Edge:
    __slots__ = ("s", "t", "label", "style", "head", "tail", "pts", "lab", "lab_xy", "gap")

    def __init__(self, s, t, label="", style="solid", head="arrow", tail=None):
        self.s, self.t, self.label, self.style, self.head, self.tail = s, t, label, style, head, tail
        self.pts, self.lab, self.lab_xy, self.gap = None, None, None, None


class Graph:
    def __init__(self):
        self.direction = "TD"
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.subs: dict[str, dict] = {}     # id -> {"title", "members": [ids]}

    def node(self, nid, label=None, shape=None, sub=None):
        n = self.nodes.get(nid)
        if n is None:
            n = self.nodes[nid] = Node(nid)
            n.sub = sub
            if sub and sub in self.subs:
                self.subs[sub]["members"].append(nid)
        if label is not None:
            n.label, n.explicit = label, True
        if shape is not None:
            n.shape = shape
        return n


# ======================================================================= preprocessing / splitting

def _prep(src: str) -> str:
    s = src.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    s = "\n".join(ln for ln in s.split("\n") if not ln.strip().startswith("```"))
    s = re.sub(r"%%\{.*?\}%%", "", s, flags=re.S)
    s = re.sub(r"%%[^\n]*", "", s)
    return s


def _split_statements(s: str, brackets=True):
    out, cur, depth, q, m = [], [], 0, False, False
    for ch in s:
        if ch == '"' and not m:
            q = not q
        elif ch == "$" and not q:
            m = not m
        elif not q and not m and brackets:
            if ch in "[({":
                depth += 1
            elif ch in "])}":
                depth = max(0, depth - 1)
        if ch == "\n" or (ch == ";" and not q and not m and depth == 0):
            if ch == "\n" and depth > 0:
                pass
            out.append("".join(cur))
            cur, depth, q, m = [], 0, False, False
            continue
        cur.append(ch)
    out.append("".join(cur))
    return [x for x in (t.strip() for t in out) if x]


# ======================================================================= flowchart parser

_SHAPES = [
    ("(((", (")))",), "dcircle"), ("((", ("))",), "circle"), ("([", ("])",), "stadium"),
    ("[[", ("]]",), "subroutine"), ("[(", (")]",), "cylinder"), ("[/", ("/]", "\\]"), "slash"),
    ("[\\", ("\\]", "/]"), "bslash"), ("{{", ("}}",), "hexagon"), ("[", ("]",), "rect"),
    ("(", (")",), "rounded"), ("{", ("}",), "diamond"), (">", ("]",), "asym"),
]
_ID = re.compile(r"[\w\u0080-￿](?:[\w\u0080-￿]|-(?=[\w\u0080-￿])|\.(?=[\w\u0080-￿]))*")
_LINK = re.compile(r"(<|o(?=-)|x(?=-))?(-{2,}|={2,}|-\.+-|~{3,})(>|o(?=\s|$)|x(?=\s|$))?")
_TERM = {"--": re.compile(r"-{2,}>|-{3,}"), "==": re.compile(r"={2,}>|={3,}"), "-.": re.compile(r"\.-+>|\.-+")}


def _scan_label(s, i, closers):
    """Scan a node label starting at i; returns (raw_label, end_index, closed?)."""
    n = len(s)
    j = i
    while j < n and s[j] == " ":
        j += 1
    if j < n and s[j] == '"':
        k = s.find('"', j + 1)
        if k > 0:
            e = k + 1
            while e < n and s[e] == " ":
                e += 1
            for cl in closers:
                if s.startswith(cl, e):
                    return s[j:k + 1], e + len(cl), True
    depth = 0
    k = i
    while k < n:
        c = s[k]
        if c == "$":
            e = s.find("$", k + 1)
            if e > 0:
                k = e + 1
                continue
        if depth == 0:
            for cl in closers:
                if s.startswith(cl, k):
                    return s[i:k], k + len(cl), True
        if c in "([{":
            depth += 1
        elif c in ")]}" and depth > 0:
            depth -= 1
        k += 1
    return s[i:], n, False


class _FlowParser:
    def __init__(self, ctx, g):
        self.ctx, self.g = ctx, g
        self.sub_stack = []
        self.anon = 0

    def cur_sub(self):
        return self.sub_stack[-1] if self.sub_stack else None

    def statement(self, st):
        ctx, g = self.ctx, self.g
        low = st.lower()
        first = low.split(None, 1)[0] if low.split() else ""
        if first in ("classdef", "class", "style", "linkstyle", "click", "acctitle", "accdescr", "accdescr:",
                     "acctitle:", "title") or first.startswith("acc"):
            return
        if first == "direction":
            return
        if first == "end" and low.strip() == "end":
            if self.sub_stack:
                self.sub_stack.pop()
            return
        if first == "subgraph":
            rest = st[len("subgraph"):].strip()
            m = re.match(r'([\w\-]+)\s*\[\s*"?(.*?)"?\s*\]\s*$', rest)
            if m:
                sid, title = m.group(1), m.group(2)
            elif rest.startswith('"'):
                title = rest.strip('"')
                sid = title
            else:
                sid = title = rest or f"_sub{len(g.subs)}"
            sid = sid.strip()
            if sid not in g.subs:
                g.subs[sid] = {"title": _clean_label(title), "members": [], "parent": self.cur_sub()}
            self.sub_stack.append(sid)
            return
        self.chain(st)

    def node_group(self, s, i):
        ids = []
        while True:
            i = self._sp(s, i)
            nid, i2 = self.node(s, i)
            if nid is None:
                break
            ids.append(nid)
            i = self._sp(s, i2)
            if i < len(s) and s[i] == "&":
                i += 1
                continue
            break
        return ids, i

    def _sp(self, s, i):
        while i < len(s) and s[i] in " \t":
            i += 1
        return i

    def node(self, s, i):
        g, ctx = self.g, self.ctx
        m = _ID.match(s, i)
        nid = None
        if m:
            nid = m.group(0)
            # a lone 'o'/'x' id glued to a link is a link end marker, not a node (handled by _LINK)
            i = m.end()
        label, shape = None, None
        for op, cls, shp in _SHAPES:
            if s.startswith(op, i) and not (op == ">" and nid is None):
                raw, e, closed = _scan_label(s, i + len(op), cls)
                if not closed:
                    ctx.warn(f"unclosed {op!r} in node {nid or raw.strip()[:20]!r}; took the rest as its label")
                    ma = re.search(r"\s*(?:-->|---|==>|-\.->|-\.-|===)", raw)
                    if ma:
                        e = i + len(op) + ma.start()
                        raw = raw[:ma.start()]
                if shp in ("slash", "bslash"):
                    closer = s[e - 2:e] if closed else cls[0]
                    if shp == "slash":
                        shp = "para" if closer == "/]" else "trap"
                    else:
                        shp = "para_alt" if closer == "\\]" else "trap_alt"
                label, shape, i = _clean_label(raw), shp, e
                break
        if nid is None and shape is None:
            return None, i
        if nid is None:
            self.anon += 1
            nid = f"_n{self.anon}"
        if s.startswith("@{", i):
            e = s.find("}", i)
            body = s[i + 2:e if e > 0 else len(s)]
            ml = re.search(r'label\s*:\s*"([^"]*)"', body)
            if ml:
                label = _clean_label(ml.group(1))
            ms = re.search(r"shape\s*:\s*([\w-]+)", body)
            if ms:
                shape = {"rect": "rect", "rounded": "rounded", "circle": "circle", "circ": "circle",
                         "diamond": "diamond", "diam": "diamond", "decision": "diamond", "stadium": "stadium",
                         "hex": "hexagon", "cyl": "cylinder", "db": "cylinder"}.get(ms.group(1), "rect")
            i = e + 1 if e > 0 else len(s)
        mm = re.match(r":::[\w-]+", s[i:])
        if mm:
            i += mm.end()
        if label is not None and label == "" and shape:
            label = " "
        g.node(nid, label, shape, self.cur_sub())
        return nid, i

    def link(self, s, i):
        """Returns (edge_props or None, new_i)."""
        i = self._sp(s, i)
        m = _LINK.match(s, i)
        if m and not (m.group(2) in ("--", "==") and not m.group(3)) and not (m.group(2) == "-.-" and False):
            body = m.group(2)
            props = self._props(body, m.group(1), m.group(3))
            i = m.end()
            i2 = self._sp(s, i)
            if i2 < len(s) and s[i2] == "|":
                e = s.find("|", i2 + 1)
                props["label"] = _clean_label(s[i2 + 1:e if e > 0 else len(s)])
                if e < 0:
                    self.ctx.warn("unclosed |edge label|")
                i = e + 1 if e > 0 else len(s)
            return props, i
        for op in ("--", "==", "-."):
            if s.startswith(op, i) or (s.startswith("<" + op, i)):
                st = i + (1 if s[i] == "<" else 0) + 2
                t = _TERM[op].search(s, st)
                if not t:
                    self.ctx.warn(f"edge text without an arrow end near {s[i:i + 20]!r}")
                    return None, len(s)
                props = self._props(op + t.group(0) if op != "-." else "-.-", "<" if s[i] == "<" else None,
                                    ">" if t.group(0).endswith(">") else None)
                props["label"] = _clean_label(s[st:t.start()])
                return props, t.end()
        return None, i

    @staticmethod
    def _props(body, left, right):
        style = "dotted" if "." in body else "thick" if body.startswith("=") else "invis" if body.startswith("~") \
            else "solid"
        hm = {">": "arrow", "o": "circle", "x": "cross"}
        return {"style": style, "head": hm.get(right), "tail": hm.get(left) if left != "<" else "arrow",
                "label": ""}

    def chain(self, st):
        ctx, g = self.ctx, self.g
        srcs, i = self.node_group(st, 0)
        if not srcs:
            ctx.warn(f"could not parse statement {st[:40]!r}")
            return
        while i < len(st):
            props, i2 = self.link(st, i)
            if props is None:
                rest = st[i:].strip()
                if rest:
                    ctx.warn(f"ignored unparsed text {rest[:30]!r}")
                return
            dsts, i3 = self.node_group(st, i2)
            if not dsts:
                ctx.warn(f"edge from {srcs[-1]!r} has no target (truncated?)")
                return
            for a in srcs:
                for b in dsts:
                    g.edges.append(Edge(a, b, props["label"], props["style"], props["head"], props["tail"]))
            srcs, i = dsts, i3


def _parse_flowchart(ctx, body, direction):
    g = Graph()
    g.direction = direction
    stmts = _split_statements(body)
    joined = re.sub(r'"[^"]*"|\$[^$]*\$', "", " ".join(stmts))
    joined = re.sub(r"(?<=[\w])>[^\[\]]*\]", "", joined)
    if joined.count("[") != joined.count("]") or joined.count("(") != joined.count(")") or \
            joined.count("{") != joined.count("}"):
        ctx.warn("unbalanced brackets in the input; parsed best-effort")
        stmts2 = _split_statements(body, brackets=False)
        if len(stmts2) > len(stmts):
            stmts = stmts2
    p = _FlowParser(ctx, g)
    for st in stmts:
        try:
            p.statement(st)
        except Exception as ex:   # pragma: no cover - parser bug safety net
            ctx.warn(f"statement {st[:30]!r} skipped ({type(ex).__name__})")
    if p.sub_stack:
        ctx.warn("subgraph without 'end'")
    # edges that point at a subgraph id -> its middle member
    for e in g.edges:
        for attr in ("s", "t"):
            v = getattr(e, attr)
            sg = g.subs.get(v)
            if sg and sg["members"] and v in g.nodes and not g.nodes[v].explicit and v not in sg["members"]:
                mem = sg["members"][len(sg["members"]) // 2]
                setattr(e, attr, mem)
    used = {e.s for e in g.edges} | {e.t for e in g.edges}
    for sid, sg in g.subs.items():
        n = g.nodes.get(sid)
        if n is not None and not n.explicit and sid not in used and sg["members"]:
            del g.nodes[sid]
    return g


def _state_to_flow(ctx, lines):
    g = Graph()
    g.direction = "TD"
    nstart = nend = 0
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith(("note", "direction", "classDef", "class ", "[*] :")):
            m = re.match(r"direction\s+(\w+)", s)
            if m and m.group(1).upper() in DIRS:
                g.direction = m.group(1).upper()
            continue
        m = re.match(r'state\s+"([^"]*)"\s+as\s+(\w+)', s)
        if m:
            g.node(m.group(2), _clean_label(m.group(1)), "rounded")
            continue
        m = re.match(r"(\[\*\]|[\w.]+)\s*-+>\s*(\[\*\]|[\w.]+)\s*(?::\s*(.*))?$", s)
        if m:
            a, b = m.group(1), m.group(2)
            if a == "[*]":
                nstart += 1
                a = f"_start{nstart}"
                g.node(a, "", "dot")
            if b == "[*]":
                nend += 1
                b = f"_end{nend}"
                g.node(b, "", "dot2")
            for v in (a, b):
                if not v.startswith("_"):
                    n = g.node(v)
                    if n.shape == "rect":
                        n.shape = "rounded"
            g.edges.append(Edge(a, b, _clean_label(m.group(3) or "")))
            continue
        m = re.match(r"([\w.]+)\s*:\s*(.+)$", s)
        if m:
            g.node(m.group(1), _clean_label(m.group(2)), "rounded")
            continue
        if not s.startswith(("state", "}", "{")):
            ctx.warn(f"stateDiagram: ignored {s[:30]!r}")
    return g


def _seq_to_flow(ctx, lines):
    g = Graph()
    g.direction = "LR"
    for ln in lines:
        s = ln.strip()
        m = re.match(r"(?:participant|actor)\s+(\w+)(?:\s+as\s+(.+))?$", s)
        if m:
            g.node(m.group(1), _clean_label(m.group(2) or m.group(1)), "rect")
            continue
        m = re.match(r"(\w+)\s*(--?>>|--?>|--?x|--?\))\s*[+-]?\s*(\w+)\s*:\s*(.*)$", s)
        if m:
            g.node(m.group(1))
            g.node(m.group(3))
            g.edges.append(Edge(m.group(1), m.group(3), _clean_label(m.group(4)),
                                "dotted" if m.group(2).startswith("--") else "solid"))
    return g


# ======================================================================= shapes

def _size_node(R, n, wrap):
    s = R.size
    shape = n.shape
    w_eff = wrap * (0.62 if shape == "diamond" else 0.8 if shape in ("circle", "dcircle") else 1.0) \
        if wrap else None
    lab = R.label(n.label, w_eff)
    n.lab = lab
    tw, th = max(lab.w, 0.6 * s), max(lab.h, LINE_A * s)
    px, py = PAD_X * s, PAD_Y * s
    if shape in ("dot", "dot2"):
        n.w = n.h = 0.8 * s
    elif shape in ("circle", "dcircle"):
        d = math.hypot(tw, th) + 0.5 * s + (0.5 * s if shape == "dcircle" else 0.0)
        n.w = n.h = max(d, 2.2 * s)
    elif shape == "diamond":
        hw, hh = tw / 2 + 0.4 * s, th / 2 + 0.3 * s      # clear of the outline at the corners
        B = hh + 0.35 * tw / 2 + 0.45 * s
        A = hw / max(0.25, 1 - hh / B)
        n.w, n.h = 2 * A, 2 * B
    else:
        h = th + 2 * py
        w = tw + 2 * px
        if shape == "stadium":
            w = tw + h * 0.9
        elif shape == "subroutine":
            w += 0.7 * s
        elif shape == "hexagon":
            w += h * 0.55
        elif shape in ("para", "para_alt", "trap", "trap_alt"):
            w += h * 0.5 if shape.startswith("para") else h * 0.7
        elif shape == "asym":
            w += h * 0.4
        elif shape == "cylinder":
            h += 0.7 * s
        n.w, n.h = max(w, min(h * 1.1, 3.0 * s), 2.0 * s), h


def _poly(n):
    x0, y0, x1, y1 = n.rect()
    cx, cy, h = n.x, n.y, n.h
    sh = n.shape
    if sh == "diamond":
        return [(cx, y0), (x1, cy), (cx, y1), (x0, cy)]
    if sh == "hexagon":
        k = h * 0.275
        return [(x0 + k, y0), (x1 - k, y0), (x1, cy), (x1 - k, y1), (x0 + k, y1), (x0, cy)]
    k = h * (0.5 if sh.startswith("para") else 0.35)
    if sh == "para":
        return [(x0 + k, y0), (x1, y0), (x1 - k, y1), (x0, y1)]
    if sh == "para_alt":
        return [(x0, y0), (x1 - k, y0), (x1, y1), (x0 + k, y1)]
    if sh == "trap":
        return [(x0 + k, y0), (x1 - k, y0), (x1, y1), (x0, y1)]
    if sh == "trap_alt":
        return [(x0, y0), (x1, y0), (x1 - k, y1), (x0 + k, y1)]
    if sh == "asym":
        k = h * 0.4
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0 + k, cy)]
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _inside(n, p):
    x, y = p
    sh = n.shape
    if sh in ("circle", "dcircle", "dot", "dot2"):
        return math.hypot(x - n.x, y - n.y) <= n.w / 2
    x0, y0, x1, y1 = n.rect()
    if sh == "stadium":
        r = n.h / 2
        qx = min(max(x, x0 + r), x1 - r)
        return math.hypot(x - qx, y - n.y) <= r
    if sh in ("diamond", "hexagon", "para", "para_alt", "trap", "trap_alt", "asym"):
        poly = _poly(n)
        c = False
        for i in range(len(poly)):
            (ax, ay), (bx, by) = poly[i - 1], poly[i]
            if (ay > y) != (by > y) and x < ax + (bx - ax) * (y - ay) / (by - ay):
                c = not c
        return c
    return x0 <= x <= x1 and y0 <= y <= y1


def _sk_line(a, b, R, amp=0.03):
    """Straight segment with a gentle hand bow (zero at both ends)."""
    s = R.size
    L = geom.seglen(a, b)
    n = max(1, int(L / (0.25 * s)))
    if L < 1e-9:
        return [a, b]
    k = R.rnd.uniform(-1, 1) * min(amp * L, 0.09 * s)
    nx, ny = -(b[1] - a[1]) / L, (b[0] - a[0]) / L
    out = []
    for i in range(n + 1):
        t = i / n
        o = k * math.sin(math.pi * t)
        out.append((a[0] + (b[0] - a[0]) * t + nx * o, a[1] + (b[1] - a[1]) * t + ny * o))
    return out


def _closed_poly(pts, R):
    out = [pts[0]]
    for i in range(len(pts)):
        out += _sk_line(pts[i], pts[(i + 1) % len(pts)], R)[1:]
    return out


def _outline(n, R):
    s = R.size
    x0, y0, x1, y1 = n.rect()
    cx, cy = n.x, n.y
    sh = n.shape
    if sh in ("circle", "dcircle", "dot", "dot2"):
        r = n.w / 2
        rr = r * (1 + R.rnd.uniform(-0.02, 0.02))
        out = [geom.ellipse(cx, cy, rr, r, 225, 225 + 368)]
        if sh == "dcircle":
            out.append(geom.ellipse(cx, cy, r - 0.25 * s, r - 0.25 * s, 225, 225 + 362))
        if sh == "dot":
            out.append(geom.ellipse(cx, cy, r * 0.45, r * 0.45, 0, 720, 24))
            out.append(geom.ellipse(cx, cy, r * 0.2, r * 0.2, 0, 360, 10))
        if sh == "dot2":
            out.append(geom.ellipse(cx, cy, r * 0.5, r * 0.5, 0, 720, 24))
        return out
    if sh in ("rounded", "stadium"):
        r = n.h / 2 if sh == "stadium" else min(0.45 * s, n.h / 3)
        r = min(r, n.w / 2)
        pts = _sk_line((x0 + r, y0), (x1 - r, y0), R)
        pts += geom.ellipse(x1 - r, y0 + r, r, r, 270, 360, 8)[1:] if sh == "rounded" else \
            geom.ellipse(x1 - r, cy, r, r, 270, 450, 18)[1:]
        if sh == "rounded":
            pts += _sk_line((x1, y0 + r), (x1, y1 - r), R)[1:]
            pts += geom.ellipse(x1 - r, y1 - r, r, r, 0, 90, 8)[1:]
        pts += _sk_line((x1 - r, y1), (x0 + r, y1), R)[1:]
        if sh == "rounded":
            pts += geom.ellipse(x0 + r, y1 - r, r, r, 90, 180, 8)[1:]
            pts += _sk_line((x0, y1 - r), (x0, y0 + r), R)[1:]
            pts += geom.ellipse(x0 + r, y0 + r, r, r, 180, 270, 8)[1:]
        else:
            pts += geom.ellipse(x0 + r, cy, r, r, 90, 270, 18)[1:]
        return [pts]
    if sh == "cylinder":
        ry = 0.3 * s
        rx = n.w / 2
        top = geom.ellipse(cx, y0 + ry, rx, ry, 180, 540, 40)
        body = _sk_line((x1, y0 + ry), (x1, y1 - ry), R)
        body += geom.ellipse(cx, y1 - ry, rx, ry, 0, 180, 24)[1:]
        body += _sk_line((x0, y1 - ry), (x0, y0 + ry), R)[1:]
        return [top, body]
    out = [_closed_poly(_poly(n), R)]
    if sh == "subroutine":
        e = 0.3 * s
        out.append(_sk_line((x0 + e, y0), (x0 + e, y1), R))
        out.append(_sk_line((x1 - e, y0), (x1 - e, y1), R))
    return out


def _label_origin(n):
    lab = n.lab
    dx = 0.0
    dy = 0.0
    if n.shape == "cylinder":
        dy = 0.3 * 0.5 * (n.h - lab.h) / max(n.h, 1e-9) + 0.0
    if n.shape == "asym":
        dx = n.h * 0.2
    x = n.x - lab.w / 2 + dx
    y = n.y - lab.h / 2 + dy
    if n.shape == "cylinder":
        y = n.y - lab.h / 2 + 0.3 * (n.h - lab.h) / 4
    return x, y


# ======================================================================= path helpers

def _dense(pts, step):
    return geom.resample(pts, step)


def _cut_start(pts, d):
    acc = 0.0
    for i in range(1, len(pts)):
        L = geom.seglen(pts[i - 1], pts[i])
        if acc + L >= d:
            t = (d - acc) / L if L > 0 else 0
            a, b = pts[i - 1], pts[i]
            return [(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)] + pts[i:]
        acc += L
    return pts[-1:]


def _trim(pts, n, gap):
    """Remove the head of `pts` lying inside node n's shape, then back off `gap` along the path."""
    if n is None or len(pts) < 2:
        return pts
    i = 0
    while i < len(pts) and _inside(n, pts[i]):
        i += 1
    if i == len(pts):
        return pts[-1:]
    if i > 0:
        a, b = pts[i - 1], pts[i]
        for _ in range(18):
            m = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            if _inside(n, m):
                a = m
            else:
                b = m
        pts = [b] + pts[i:]
    return _cut_start(pts, gap) if gap > 0 else pts


def _catmull(pts, per=10):
    if len(pts) < 3:
        return list(pts)
    P = [pts[0]] + list(pts) + [pts[-1]]
    out = [pts[0]]
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(1, per + 1):
            t = k / per
            t2, t3 = t * t, t * t * t
            out.append(tuple(0.5 * (2 * p1[j] + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2
                                    + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3) for j in (0, 1)))
    return out


def _point_at(pts, frac):
    L = geom.length(pts)
    d = L * frac
    acc = 0.0
    for i in range(1, len(pts)):
        l = geom.seglen(pts[i - 1], pts[i])
        if acc + l >= d and l > 0:
            t = (d - acc) / l
            a, b = pts[i - 1], pts[i]
            return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t), ((b[0] - a[0]) / l, (b[1] - a[1]) / l)
        acc += l
    if len(pts) >= 2:
        a, b = pts[-2], pts[-1]
        l = geom.seglen(a, b) or 1.0
        return b, ((b[0] - a[0]) / l, (b[1] - a[1]) / l)
    return pts[0], (1.0, 0.0)


def _back_point(pts, d):
    """Point at arc distance d before the end."""
    rev = pts[::-1]
    return _cut_start(rev, d)[0]


def _end_marks(kind, tip, frm, R):
    s = R.size
    if kind == "arrow":
        return [geom.arrowhead(tip, frm, ARROW * s, 27)]
    dx, dy = tip[0] - frm[0], tip[1] - frm[1]
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    if kind == "circle":
        r = 0.2 * s
        return [geom.ellipse(tip[0] - ux * r, tip[1] - uy * r, r, r, 0, 370, 14)]
    if kind == "cross":
        r = 0.22 * s
        c = (tip[0] - ux * r, tip[1] - uy * r)
        return [[(c[0] - r, c[1] - r), (c[0] + r, c[1] + r)], [(c[0] + r, c[1] - r), (c[0] - r, c[1] + r)]]
    return []


# ======================================================================= layered layout

def _pava(y, w):
    blocks = []
    for yi, wi in zip(y, w):
        blocks.append([yi * wi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            b = blocks.pop()
            blocks[-1][0] += b[0]
            blocks[-1][1] += b[1]
            blocks[-1][2] += b[2]
    out = []
    for sw, ww, c in blocks:
        out += [sw / ww] * c
    return out


def _crossings(la, lb, pos, succ):
    segs = []
    for u in la:
        for v in succ[u]:
            if v in lb:
                segs.append((pos[u], pos[v]))
    c = 0
    for i in range(len(segs)):
        a1, b1 = segs[i]
        for j in range(i + 1, len(segs)):
            a2, b2 = segs[j]
            if (a1 - a2) * (b1 - b2) < 0:
                c += 1
    return c


def _layered(g, R, gap_extra=None):
    s = R.size
    ids = list(g.nodes)
    N = len(ids)
    ix = {v: i for i, v in enumerate(ids)}
    td = g.direction in ("TD", "TB", "BT")
    real = [e for e in g.edges if e.s != e.t]
    loops = [e for e in g.edges if e.s == e.t]
    out_adj = [[] for _ in range(N)]
    indeg = [0] * N
    for k, e in enumerate(real):
        out_adj[ix[e.s]].append((ix[e.t], k))
        indeg[ix[e.t]] += 1
    # --- cycle breaking (iterative DFS, roots first)
    state = [0] * N
    rev = [False] * len(real)
    order = [i for i in range(N) if indeg[i] == 0] + [i for i in range(N) if indeg[i] > 0]
    for r in order:
        if state[r]:
            continue
        stack = [(r, 0)]
        state[r] = 1
        while stack:
            u, j = stack[-1]
            if j < len(out_adj[u]):
                stack[-1] = (u, j + 1)
                v, k = out_adj[u][j]
                if state[v] == 1:
                    rev[k] = True
                elif state[v] == 0:
                    state[v] = 1
                    stack.append((v, 0))
            else:
                state[u] = 2
                stack.pop()
    dag = []
    for k, e in enumerate(real):
        a, b = ix[e.s], ix[e.t]
        dag.append((b, a) if rev[k] else (a, b))
    succ = [[] for _ in range(N)]
    pred = [[] for _ in range(N)]
    for a, b in dag:
        succ[a].append(b)
        pred[b].append(a)
    # --- longest-path layering
    indg = [len(pred[i]) for i in range(N)]
    topo, q = [], [i for i in range(N) if indg[i] == 0]
    while q:
        u = q.pop(0)
        topo.append(u)
        for v in succ[u]:
            indg[v] -= 1
            if indg[v] == 0:
                q.append(v)
    layer = [0] * N
    for u in topo:
        for v in succ[u]:
            layer[v] = max(layer[v], layer[u] + 1)
    for u in reversed(topo):
        if not pred[u] and succ[u]:
            layer[u] = max(layer[u], min(layer[v] for v in succ[u]) - 1)
    # --- vertices (nodes + dummies)
    V_layer = list(layer)
    V_cw, V_mw, V_node = [], [], []
    loop_nodes = {ix[e.s] for e in loops}
    for i, nid in enumerate(ids):
        n = g.nodes[nid]
        extra = 1.4 * s if i in loop_nodes else 0.0
        V_cw.append((n.w if td else n.h) + extra)
        V_mw.append(n.h if td else n.w)
        V_node.append(nid)
    chains = []
    lab_dummy = {}
    vsucc = [[] for _ in range(N)]
    for k, (a, b) in enumerate(dag):
        ch = [a]
        prev = a
        for L in range(layer[a] + 1, layer[b]):
            d = len(V_layer)
            V_layer.append(L)
            V_cw.append(DUMMY_W * s)
            V_mw.append(0.0)
            V_node.append(None)
            vsucc.append([])
            vsucc[prev].append(d)
            ch.append(d)
            prev = d
        vsucc[prev].append(b)
        ch.append(b)
        chains.append(ch)
        e = real[k]
        if e.label and len(ch) > 2:
            lab = R.label(e.label, R.wrap_w * 0.8 if R.wrap_w else None)
            dm = ch[len(ch) // 2]
            ext = lab.w if td else lab.h
            V_cw[dm] = ext + 1.0 * s
            V_mw[dm] = lab.h if td else lab.w
            lab_dummy[k] = (dm, lab)
    NV = len(V_layer)
    vpred = [[] for _ in range(NV)]
    for u in range(NV):
        for v in vsucc[u]:
            vpred[v].append(u)
    nL = max(V_layer) + 1 if NV else 0
    layers = [[] for _ in range(nL)]
    # initial order: layer 0 by declaration, later layers by barycenter of preds
    for v in range(NV):
        if V_layer[v] == 0:
            layers[0].append(v)
    pos = {v: float(i) for i, v in enumerate(layers[0])}
    for L in range(1, nL):
        mem = [v for v in range(NV) if V_layer[v] == L]
        key = {}
        for j, v in enumerate(mem):
            ps = [pos[u] for u in vpred[v] if u in pos]
            key[v] = (sum(ps) / len(ps) if ps else 1e6 + j, j)
        mem.sort(key=lambda v: key[v])
        layers[L] = mem
        for i, v in enumerate(mem):
            pos[v] = float(i)

    def total_cross():
        return sum(_crossings(layers[L], set(layers[L + 1]), pos, vsucc) for L in range(nL - 1))

    def setpos():
        for Lr in layers:
            for i, v in enumerate(Lr):
                pos[v] = float(i)

    best = total_cross()
    best_layers = [list(x) for x in layers]
    for it in range(8 if NV < 120 else 2):
        rng = range(1, nL) if it % 2 == 0 else range(nL - 2, -1, -1)
        for L in rng:
            nb = vpred if it % 2 == 0 else vsucc
            key = {}
            for j, v in enumerate(layers[L]):
                ps = [pos[u] for u in nb[v]]
                key[v] = (sum(ps) / len(ps) if ps else pos[v], j)
            layers[L].sort(key=lambda v: key[v])
            for i, v in enumerate(layers[L]):
                pos[v] = float(i)
        # transpose pass
        if NV < 80:
            for L in range(nL):
                Lr = layers[L]
                improved = True
                guard = 0
                while improved and guard < 4:
                    improved = False
                    guard += 1
                    for i in range(len(Lr) - 1):
                        def local():
                            c = 0
                            if L > 0:
                                c += _crossings(layers[L - 1], set(Lr), pos, vsucc)
                            if L < nL - 1:
                                c += _crossings(Lr, set(layers[L + 1]), pos, vsucc)
                            return c
                        c0 = local()
                        Lr[i], Lr[i + 1] = Lr[i + 1], Lr[i]
                        pos[Lr[i]], pos[Lr[i + 1]] = float(i), float(i + 1)
                        if local() < c0:
                            improved = True
                        else:
                            Lr[i], Lr[i + 1] = Lr[i + 1], Lr[i]
                            pos[Lr[i]], pos[Lr[i + 1]] = float(i), float(i + 1)
        c = total_cross()
        if c < best:
            best, best_layers = c, [list(x) for x in layers]
        if best == 0:
            break
    layers = best_layers
    setpos()
    # --- coordinate assignment (cross axis)
    # a node whose short edges carry labels keeps long edges (dummies) further away, so the labels fit
    clear = [0.0] * NV
    clear_in = [0.0] * NV
    pairs = {}
    for k, ch in enumerate(chains):
        if len(ch) == 2:
            pairs[(min(ch), max(ch))] = pairs.get((min(ch), max(ch)), 0) + 1
    for k, ch in enumerate(chains):
        if len(ch) == 2:
            e = real[k]
            c = 0.8 * s if pairs[(min(ch), max(ch))] > 1 else 0.0
            if e.label:
                lab = R.label(e.label, R.wrap_w * 0.8 if R.wrap_w else None)
                c += min(lab.w if td else lab.h, 8 * s) * 0.6 + 0.5 * s
            for v in ch:
                clear[v] = max(clear[v], c)
            if e.label:
                clear_in[ch[-1]] = max(clear_in[ch[-1]], c)
    cx = getattr(R, "crossx", {})
    for v in range(N):
        clear_in[v] += cx.get(V_node[v], 0.0)

    def sep(u, v):
        nu, nv = V_node[u] is not None, V_node[v] is not None
        if nu and nv and g.nodes[V_node[u]].sub != g.nodes[V_node[v]].sub:
            return (V_cw[u] + V_cw[v]) / 2 + NODE_SEP * s + 1.4 * s
        if nu and nv:
            gap = NODE_SEP * s + max(clear_in[u], clear_in[v]) * 0.5 + min(clear_in[u], clear_in[v]) * 0.5
        elif nu or nv:
            gap = 0.9 * s + clear[u] + clear[v] + (clear_in[u] if nu else clear_in[v]) * 0.5
        else:
            gap = 0.45 * s
        return (V_cw[u] + V_cw[v]) / 2 + gap

    cross = {}
    for Lr in layers:
        x = 0.0
        for i, v in enumerate(Lr):
            if i:
                x += sep(Lr[i - 1], v)
            cross[v] = x
        off = x / 2
        for v in Lr:
            cross[v] -= off

    def project(Lr, desired):
        S = [0.0]
        for i in range(1, len(Lr)):
            S.append(S[-1] + sep(Lr[i - 1], Lr[i]))
        w = [1.6 if V_node[v] is None else 1.0 for v in Lr]
        fit = _pava([d - S[i] for i, d in enumerate(desired)], w)
        for i, v in enumerate(Lr):
            cross[v] = fit[i] + S[i]

    for it in range(10):
        if it % 2 == 0:
            rng, nb = range(1, nL), vpred
        else:
            rng, nb = range(nL - 2, -1, -1), vsucc
        for L in rng:
            Lr = layers[L]
            des = []
            for v in Lr:
                ns = nb[v]
                des.append(sum(cross[u] for u in ns) / len(ns) if ns else cross[v])
            project(Lr, des)
    for _ in range(3):
        for L in range(nL):
            Lr = layers[L]
            des = []
            for v in Lr:
                ns = vpred[v] + vsucc[v]
                des.append(sum(cross[u] for u in ns) / len(ns) if ns else cross[v])
            project(Lr, des)
    # --- main axis
    thick = [max([V_mw[v] for v in layers[L]] + [0.5 * s]) for L in range(nL)]
    base_gap = (GAP_TD if td else GAP_LR) * s
    gaps = [base_gap] * max(0, nL - 1)
    if g.subs:     # room for a subgraph frame + title where membership changes between layers
        def subs_in(L):
            return sorted({g.nodes[V_node[v]].sub or "" for v in layers[L] if V_node[v] is not None})
        for L in range(nL - 1):
            if subs_in(L) != subs_in(L + 1) or len(subs_in(L)) > 1:
                gaps[L] += (2.0 if td else 1.4) * s
    for k, e in enumerate(real):
        if not e.label:
            continue
        ch = chains[k]
        if k in lab_dummy:
            e.gap = min(V_layer[lab_dummy[k][0]], len(gaps) - 1)
            continue
        lab = R.label(e.label, R.wrap_w * 0.8 if R.wrap_w else None)
        Lm = V_layer[ch[(len(ch) - 2) // 2]]
        if 0 <= Lm < len(gaps):
            e.gap = Lm
            ext = (lab.h if td else lab.w) + 1.5 * s
            gaps[Lm] = max(gaps[Lm], ext)
    for L, add in (gap_extra or {}).items():     # widened by the label placer when a label found no free spot
        if 0 <= L < len(gaps):
            gaps[L] += add
    main_c = []
    m = 0.0
    for L in range(nL):
        if L:
            m += thick[L - 1] / 2 + gaps[L - 1] + thick[L] / 2
        main_c.append(m)
    sign = -1.0 if g.direction in ("BT", "RL") else 1.0
    XY = []
    for v in range(NV):
        mc, cc = sign * main_c[V_layer[v]], cross[v]
        XY.append((cc, mc) if td else (mc, cc))
    for k, (dm, lab) in lab_dummy.items():
        cc, mc = cross[dm], sign * main_c[V_layer[dm]]
        line_c = cc - V_cw[dm] / 2 + 0.2 * s
        lab_c = cc + V_cw[dm] / 2 - (lab.w if td else lab.h) / 2
        XY[dm] = (line_c, mc) if td else (mc, line_c)
        lx, ly = (lab_c, mc) if td else (mc, lab_c)
        real[k].lab = lab
        real[k].lab_xy = (lx - lab.w / 2, ly - lab.h / 2)
    for i, nid in enumerate(ids):
        n = g.nodes[nid]
        n.x, n.y = XY[i]
    # --- edge routing
    pair_count, pair_seen = {}, {}
    for k, ch in enumerate(chains):
        if len(ch) == 2:
            key = (min(ch), max(ch))
            pair_count[key] = pair_count.get(key, 0) + 1
    for k, e in enumerate(real):
        ch = chains[k]
        pts = [XY[v] for v in ch]
        a_node, b_node = g.nodes[V_node[ch[0]]], g.nodes[V_node[ch[-1]]]
        if len(ch) == 2:
            key = (min(ch), max(ch))
            cnt = pair_count[key]
            if cnt > 1:
                j = pair_seen.get(key, 0)
                pair_seen[key] = j + 1
                off = (j - (cnt - 1) / 2) * 1.6 * s
                (ax, ay), (bx, by) = pts
                L = math.hypot(bx - ax, by - ay) or 1.0
                nx, ny = -(by - ay) / L, (bx - ax) / L
                if ch[0] > ch[-1]:
                    nx, ny = -nx, -ny
                c = ((ax + bx) / 2 + nx * off * 2, (ay + by) / 2 + ny * off * 2)
                pts = geom.quad(pts[0], c, pts[1], 24)
        else:
            pts = _catmull(pts, 12)
        pts = _dense(pts, 0.2 * s)
        pts = _trim(pts, a_node, 0.1 * s)
        pts = _trim(pts[::-1], b_node, 0.12 * s)[::-1]
        if rev[k]:
            pts = pts[::-1]
        if len(pts) == 2 or (len(ch) == 2 and pair_count[(min(ch), max(ch))] == 1):
            pts = _sk_line(pts[0], pts[-1], R, 0.02)
        e.pts = pts
    for e in loops:
        n = g.nodes[e.s]
        x0, y0, x1, y1 = n.rect()
        if td:
            a, b = (x1 - 0.05 * s, n.y - n.h * 0.25), (x1 - 0.05 * s, n.y + n.h * 0.25)
            pts = geom.bezier(a, (x1 + 1.5 * s, n.y - n.h * 0.6), (x1 + 1.5 * s, n.y + n.h * 0.6), b, 24)
        else:
            a, b = (n.x - n.w * 0.2, y0), (n.x + n.w * 0.2, y0)
            pts = geom.bezier(a, (n.x - n.w * 0.3, y0 - 1.5 * s), (n.x + n.w * 0.3, y0 - 1.5 * s), b, 24)
        e.pts = pts
    # drawing order of nodes: layer, then cross position
    return sorted(ids, key=lambda nid: (layer[ix[nid]], cross[ix[nid]]))


def _cycle_layout(g, R):
    """If g is one simple directed cycle of 3..10 nodes, lay it out on a ring. Returns node order or None."""
    ids = list(g.nodes)
    n = len(ids)
    if not (3 <= n <= 10) or len(g.edges) != n:
        return None
    nxt = {}
    for e in g.edges:
        if e.s == e.t or e.s in nxt:
            return None
        nxt[e.s] = e.t
    if len(set(nxt.values())) != n or len(nxt) != n:
        return None
    order = [ids[0]]
    while len(order) < n:
        v = nxt[order[-1]]
        if v == order[0]:
            return None
        order.append(v)
    if nxt[order[-1]] != order[0]:
        return None
    s = R.size
    nodes = [g.nodes[v] for v in order]
    sgn = -1.0 if g.direction in ("BT", "RL") else 1.0   # clockwise by default
    ang = [-math.pi / 2 + sgn * 2 * math.pi * i / n for i in range(n)]
    lab_ext = max([R.label(e.label, R.wrap_w * 0.8 if R.wrap_w else None).w for e in g.edges if e.label] + [0.0])
    ax = 1.25 if g.direction in ("TD", "TB", "BT") else 1.6
    Rr = 2.0 * s
    for _ in range(60):
        rx, ry = Rr * ax, Rr
        for nd, a in zip(nodes, ang):
            nd.x, nd.y = rx * math.cos(a), ry * math.sin(a)
        ok = True
        for i in range(n):
            a, b = nodes[i], nodes[(i + 1) % n]
            ra, rb = a.rect(), b.rect()
            if geom.rects_overlap(ra, rb, 2.2 * s + 0.5 * lab_ext):
                ok = False
                break
        if ok and n > 3:
            for i in range(n):
                for j in range(i + 2, n):
                    if (i, j) != (0, n - 1) and geom.rects_overlap(nodes[i].rect(), nodes[j].rect(), 1.0 * s):
                        ok = False
        if ok:
            break
        Rr *= 1.08
    Rr *= getattr(R, "ring_scale", 1.0)
    rx, ry = Rr * ax, Rr
    for nd, a in zip(nodes, ang):
        nd.x, nd.y = rx * math.cos(a), ry * math.sin(a)
    for i, e in enumerate(g.edges):
        e.gap = 0
        a, b = g.nodes[e.s], g.nodes[e.t]
        ia, ib = order.index(e.s), order.index(e.t)
        a0, a1 = ang[ia], ang[ib]
        if sgn > 0 and a1 < a0:
            a1 += 2 * math.pi
        if sgn < 0 and a1 > a0:
            a1 -= 2 * math.pi
        k = 48
        pts = [(rx * math.cos(a0 + (a1 - a0) * t / k), ry * math.sin(a0 + (a1 - a0) * t / k)) for t in range(k + 1)]
        pts = _trim(pts, a, 0.15 * s)
        pts = _trim(pts[::-1], b, 0.15 * s)[::-1]
        e.pts = pts
    return order


def _chain_order(g):
    """('path'|'cycle', node order) when g is one simple directed chain or one simple loop (no
    branching, no subgraphs), else None. Such a graph can be wrapped into rows when one line is too long."""
    ids = list(g.nodes)
    n = len(ids)
    if n < 4 or g.subs or any(e.s == e.t for e in g.edges):
        return None
    nxt, prv = {}, {}
    for e in g.edges:
        if e.s in nxt or e.t in prv:
            return None
        nxt[e.s], prv[e.t] = e.t, e.s
    if len(g.edges) == n - 1:
        heads = [v for v in ids if v not in prv]
        if len(heads) != 1:
            return None
        order = heads
        while order[-1] in nxt and len(order) <= n:
            order.append(nxt[order[-1]])
        return ("path", order) if len(order) == n else None
    if len(g.edges) == n:
        order = [ids[0]]
        while len(order) < n and order[-1] in nxt and nxt[order[-1]] != order[0]:
            order.append(nxt[order[-1]])
        if len(order) == n and nxt.get(order[-1]) == order[0]:
            return ("cycle", order)
    return None


def _rows_layout(g, R, ncols):
    """a long chain wrapped into rows of `ncols` nodes, reading like text but snaking (row 1 left to
    right, row 2 right to left, ...) so every arrow is a short straight hop; a loop becomes two rows going round
    (top row rightwards, bottom row leftwards, closing up the left side). Straight edges only: nothing crosses.
    Returns the node order or None."""
    co = _chain_order(g)
    if co is None:
        return None
    kind, order = co
    n = len(order)
    s = R.size
    if kind == "cycle":
        ncols = (n + 1) // 2
    if not (2 <= ncols < n):
        return None
    rows = [order[i:i + ncols] for i in range(0, n, ncols)]
    if kind == "cycle" and len(rows) != 2:
        return None
    # grid slots: (row, col); odd rows run right to left
    slot = {}
    for r, row in enumerate(rows):
        for k, v in enumerate(row):
            if kind == "cycle" and r == 1:
                c = ncols - 1 - k - (ncols - len(row)) * 0  # bottom row from the right end
            else:
                c = k if r % 2 == 0 else ncols - 1 - k
            slot[v] = (r, c)
    colw = [0.0] * ncols
    rowh = [0.0] * len(rows)
    for v, (r, c) in slot.items():
        nd = g.nodes[v]
        colw[c] = max(colw[c], nd.w)
        rowh[r] = max(rowh[r], nd.h)
    lab_w = max([R.label(e.label, R.wrap_w * 0.8 if R.wrap_w else None).w for e in g.edges if e.label] + [0.0])
    lab_h = max([R.label(e.label, R.wrap_w * 0.8 if R.wrap_w else None).h for e in g.edges if e.label] + [0.0])
    gx = max(1.8 * s, lab_w + 0.8 * s) * getattr(R, "ring_scale", 1.0)
    gy = max(1.6 * s, lab_h + 0.8 * s) * getattr(R, "ring_scale", 1.0)
    xs, x = [], 0.0
    for c in range(ncols):
        xs.append(x + colw[c] / 2)
        x += colw[c] + gx
    ys, y = [], 0.0
    for r in range(len(rows)):
        ys.append(y + rowh[r] / 2)
        y += rowh[r] + gy
    for v, (r, c) in slot.items():
        if kind == "cycle" and r == 1 and len(rows[1]) < ncols:
            # an odd loop: the shorter bottom row spreads under the whole top row
            k = rows[1].index(v)
            m = len(rows[1])
            xa, xb = xs[0], xs[-1]
            nd = g.nodes[v]
            nd.x = xb - (xb - xa) * (k + 0.5) / m if m > 1 else (xa + xb) / 2
        else:
            g.nodes[v].x = xs[c]
        g.nodes[v].y = ys[r]
    for e in g.edges:
        e.gap = None
        a, b = g.nodes[e.s], g.nodes[e.t]
        pts = [(a.x + (b.x - a.x) * t / 24, a.y + (b.y - a.y) * t / 24) for t in range(25)]
        pts = _trim(pts, a, 0.15 * s)
        pts = _trim(pts[::-1], b, 0.15 * s)[::-1]
        e.pts = pts
    return order


# ======================================================================= edge labels + drawing

LABEL_CLEAR = 0.3     # min clearance label <-> node box (x size)


def _place_edge_labels(g, R):
    """Put each edge label beside its edge. Hard rules: never within LABEL_CLEAR of a node box, never on another
    label. Soft rule: avoid covering any edge line. Candidates: 7 points along the edge x (both sides of the line
    at two distances + straight left/right/up/down), outer side first. Returns [(edge, "hard"|"soft")] for labels
    that could not meet the rules; the caller widens the layout and retries."""
    s = R.size
    cl = LABEL_CLEAR * s
    obst_nodes = [n.rect() for n in g.nodes.values()]
    pts_all = []
    for e in g.edges:
        if e.pts and e.style != "invis":
            pts_all += _dense(e.pts, 0.25 * s)
    placed = []
    bad = []

    def hard_ok(r):
        for o in obst_nodes:
            if r[0] < o[2] + cl and o[0] - cl < r[2] and r[1] < o[3] + cl and o[1] - cl < r[3]:
                return False
        m = 0.5 * s
        for o in placed:
            if r[0] < o[2] + m and o[0] - m < r[2] and r[1] < o[3] + m and o[1] - m < r[3]:
                return False
        return True

    def line_hits(r):
        m = 0.25 * s          # a chalk-free halo round each edge label (was 0.12)
        x0, y0, x1, y1 = r[0] - m, r[1] - m, r[2] + m, r[3] + m
        return sum(1 for q in pts_all if x0 <= q[0] <= x1 and y0 <= q[1] <= y1)

    def overlap_cost(r):
        c = 0.0
        for o in obst_nodes:
            ox = min(r[2], o[2] + cl) - max(r[0], o[0] - cl)
            oy = min(r[3], o[3] + cl) - max(r[1], o[1] - cl)
            if ox > 0 and oy > 0:
                c += ox * oy
        for o in placed:
            ox, oy = min(r[2], o[2]) - max(r[0], o[0]), min(r[3], o[3]) - max(r[1], o[1])
            if ox > 0 and oy > 0:
                c += ox * oy
        return c

    preset = [e for e in g.edges if e.lab is not None and e.lab_xy is not None]
    order = preset + [e for e in g.edges if e.label and e not in preset]
    for e in order:
        if e.lab is not None and e.lab_xy is not None:      # preset by the layout (label dummy)
            x, y = e.lab_xy
            r = (x, y, x + e.lab.w, y + e.lab.h)
            if hard_ok(r) and line_hits(r) == 0:
                placed.append(r)
                continue
            e.lab_xy = None
        if not e.label or not e.pts or len(e.pts) < 2:
            continue
        lab = e.lab or R.label(e.label, R.wrap_w * 0.8 if R.wrap_w else None)
        e.lab = lab
        hw, hh = lab.w / 2, lab.h / 2
        src = g.nodes[e.s]
        cands = []
        for fr in (0.5, 0.4, 0.6, 0.3, 0.7, 0.22, 0.78):
            p, t = _point_at(e.pts, fr)
            nx, ny = -t[1], t[0]
            dirs = []
            for extra in (0.0, 0.5 * s):
                off = abs(nx) * hw + abs(ny) * hh + 0.3 * s + extra
                dirs += [(nx * off, ny * off), (-nx * off, -ny * off)]
            # axis-aligned escapes: a label beside a shallow line can sit straight above/below or left/right
            dirs += [(hw + 0.35 * s, 0.0), (-hw - 0.35 * s, 0.0), (0.0, hh + 0.35 * s), (0.0, -hh - 0.35 * s)]
            # outer side first (away from the source node): reads as belonging to this branch
            dirs.sort(key=lambda d: -math.hypot(p[0] + d[0] - src.x, p[1] + d[1] - src.y))
            for d in dirs:
                cands.append((p[0] + d[0], p[1] + d[1]))
        best_soft = None
        best_any = None
        for i, c in enumerate(cands):
            r = (c[0] - hw, c[1] - hh, c[0] + hw, c[1] + hh)
            if hard_ok(r):
                h = line_hits(r)
                if h == 0:
                    best_soft = (0, i, c, r)
                    break
                if best_soft is None or h < best_soft[0]:
                    best_soft = (h, i, c, r)
            elif best_soft is None:
                oc = overlap_cost(r)
                if best_any is None or oc < best_any[0]:
                    best_any = (oc, c, r)
        if best_soft is not None:
            c, r = best_soft[2], best_soft[3]
            if best_soft[0] > 0:
                bad.append((e, "soft"))
        else:
            c, r = best_any[1], best_any[2]
            bad.append((e, "hard"))
        placed.append(r)
        e.lab_xy = (c[0] - hw, c[1] - hh)
    return bad


def _edge_strokes(e, R, seed):
    s = R.size
    out = []
    if not e.pts or len(e.pts) < 2 or e.style == "invis":
        return out
    pts = e.pts
    if e.style == "dotted":
        out += geom.dashed(_dense(pts, 0.05 * s), 0.32 * s, 0.24 * s)
    else:
        out.append(list(pts))
        if e.style == "thick":
            # second pass slightly offset: chalk pressed twice
            sh = []
            for i, p in enumerate(pts):
                a, b = pts[max(0, i - 1)], pts[min(len(pts) - 1, i + 1)]
                L = geom.seglen(a, b) or 1.0
                sh.append((p[0] - (b[1] - a[1]) / L * 0.07 * s, p[1] + (b[0] - a[0]) / L * 0.07 * s))
            out.append(sh[::-1])
    if geom.length(pts) > 0.3 * s:
        if e.head:
            out += _end_marks(e.head, pts[-1], _back_point(pts, 0.7 * s), R)
        if e.tail:
            out += _end_marks(e.tail, pts[0], _back_point(pts[::-1], 0.7 * s), R)
    if e.lab is not None and e.lab_xy is not None:
        out += e.lab.draw(R, e.lab_xy[0], e.lab_xy[1], seed, role="edge")
    return out


def _draw_graph(g, R, order):
    strokes = []
    drawn = {}
    pending = list(range(len(g.edges)))
    seed = 0
    for nid in order:
        n = g.nodes[nid]
        # edges into n from drawn nodes
        rest = []
        for k in pending:
            e = g.edges[k]
            if e.t == nid and e.s in drawn and e.s != nid:
                seed += 1
                strokes += _edge_strokes(e, R, 500 + seed)
            else:
                rest.append(k)
        pending = rest
        strokes += _outline(n, R)
        if n.label.strip():
            lx, ly = _label_origin(n)
            strokes += n.lab.draw(R, lx, ly, list(g.nodes).index(nid))
        drawn[nid] = True
        rest = []
        for k in pending:
            e = g.edges[k]
            if e.s == nid and (e.t in drawn):
                seed += 1
                strokes += _edge_strokes(e, R, 500 + seed)
            else:
                rest.append(k)
        pending = rest
    for k in pending:
        strokes += _edge_strokes(g.edges[k], R, 900 + k)
    strokes += _subgraph_boxes(g, R)
    return strokes


def _subgraph_boxes(g, R):
    s = R.size
    out = []
    labs = [((e.lab_xy[0], e.lab_xy[1], e.lab_xy[0] + e.lab.w, e.lab_xy[1] + e.lab.h)) for e in g.edges
            if e.lab is not None and e.lab_xy is not None]
    for sid, sg in g.subs.items():
        mem = [g.nodes[m] for m in sg["members"] if m in g.nodes]
        if not mem:
            continue
        x0 = min(n.rect()[0] for n in mem) - 0.6 * s
        y0 = min(n.rect()[1] for n in mem) - 0.6 * s
        x1 = max(n.rect()[2] for n in mem) + 0.6 * s
        y1 = max(n.rect()[3] for n in mem) + 0.6 * s
        title = sg["title"]
        tl = R.label(title, None) if title else None
        if tl:
            y0 -= tl.h + 0.3 * s
            x1 = max(x1, x0 + tl.w + 0.8 * s)
        box = (x0, y0, x1, y1)
        memset = {n.id for n in mem}
        bad = any(geom.rects_overlap(box, n.rect(), 0.0) for n in g.nodes.values() if n.id not in memset) or \
            any(geom.rects_overlap(box, r, 0.0) and not all(geom.rects_overlap(r, n.rect(), 0) is False for n in mem)
                for r in labs if not (box[0] <= r[0] and r[2] <= box[2] and box[1] <= r[1] and r[3] <= box[3]))
        if bad:
            R.warn(f"subgraph {sid!r}: box omitted (members not contiguous in the layout)")
            continue
        rect = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        out += geom.dashed(_dense(rect, 0.05 * s), 0.45 * s, 0.3 * s)
        if tl:
            out += tl.draw(R, x0 + 0.4 * s, y0 + 0.25 * s, 77 + len(out), role="title")
    return out


def _graph_render(g, R, labels=True):
    """Layout + label placement; widens layer gaps (or the ring) until every label has a free spot."""
    if not g.nodes:
        return []
    R.ring_scale = 1.0
    R.crossx = {}
    extra = {}
    order = None
    td = g.direction in ("TD", "TB", "BT")
    for rnd in range(9):
        for e in g.edges:
            e.pts = e.lab = e.lab_xy = e.gap = None
        order = _graph_render1(g, R, extra)
        if not labels:
            return order
        bad = _place_edge_labels(g, R)
        hard = [e for e, k in bad if k == "hard"]
        if not hard and (not bad or rnd >= 5):
            break
        if rnd == 8:
            R.warn("an edge label could not be given a free spot")
            break
        grown = False
        for e, k in bad:
            if e.gap is None:
                continue
            lab = e.lab
            add = ((lab.h if td else lab.w) if lab else 1.0) * 0.5 + 0.6 * R.size
            extra[e.gap] = extra.get(e.gap, 0.0) + add
            ext = ((lab.w if td else lab.h) if lab else 1.0) * 0.5 + 0.3 * R.size
            if k == "hard" or rnd >= 2:     # widen sideways too once the layer gap alone did not help
                for v in (e.s, e.t):
                    R.crossx[v] = R.crossx.get(v, 0.0) + (ext if k == "hard" else ext * 0.5)
            grown = True
        R.ring_scale *= 1.2
        if not grown:
            break
    return order


def _graph_render1(g, R, extra):
    if not g.nodes:
        return []
    for n in g.nodes.values():
        _size_node(R, n, R.wrap_w)
    if getattr(R, "rows", None):
        order = _rows_layout(g, R, R.rows)
        if order is not None:
            return order
    order = _cycle_layout(g, R)
    if order is None:
        order = _layered(g, R, extra)
    return order


# ======================================================================= timeline

def _parse_timeline(ctx, lines):
    title, items, section = "", [], None   # items: [period, [events], section]
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("title"):
            title = _clean_label(s[5:].strip())
            continue
        if low.startswith("section"):
            section = _clean_label(s[7:].strip())
            continue
        if s.startswith(":"):
            evs = [_clean_label(x) for x in s[1:].split(" : ") if x.strip()]
            if items:
                items[-1][1].extend(evs)
            continue
        parts = [x for x in re.split(r"\s+:\s+|\s*:\s+|\s+:\s*$", s)]
        if ":" in s and len(parts) == 1:
            parts = s.split(":")
        per = _clean_label(parts[0])
        evs = [_clean_label(x) for x in parts[1:] if x.strip()]
        items.append([per, evs, section])
    if not items:
        ctx.warn("timeline has no periods")
    return title, items


def _timeline_render(ctx, title, items, vertical):
    s = ctx.size
    strokes, nodes = [], {}
    if not items:
        if title:
            lab = ctx.label(title, None)
            strokes += lab.draw(ctx, 0, 0, 1)
        return strokes, nodes
    ew = ctx.wrap_w * 0.75 if ctx.wrap_w else 8 * s
    tlab = ctx.label(title, ctx.wrap_w * 3 if ctx.wrap_w else None) if title else None
    P = [ctx.label(p, ew) for p, _, _ in items]
    EV = [[ctx.label(e, ew) for e in evs] for _, evs, _ in items]
    secs = []
    for i, it in enumerate(items):
        if it[2] is not None and (not secs or secs[-1][0] != it[2]):
            secs.append([it[2], i, i])
        elif secs and it[2] == secs[-1][0]:
            secs[-1][2] = i
    if not vertical:
        colw = [max([P[i].w] + [e.w for e in EV[i]] + [3.0 * s]) + 1.2 * s for i in range(len(items))]
        xs, x = [], 0.0
        for w in colw:
            xs.append(x + w / 2)
            x += w
        total_w = x
        y = 0.0
        if tlab:
            strokes += tlab.draw(ctx, (total_w - tlab.w) / 2, y, 3)
            y += tlab.h + 1.0 * s
        sec_y = y
        if secs:
            y += LINE_A * s + 1.0 * s
        ph = max(p.h for p in P)
        line_y = y + ph + 0.7 * s
        axis = _sk_line((0.0, line_y), (total_w + 0.6 * s, line_y), ctx, 0.01)
        strokes.append(axis)
        strokes.append(geom.arrowhead(axis[-1], (total_w - 0.4 * s, line_y), ARROW * s, 27))
        si = 0
        for i in range(len(items)):
            while si < len(secs) and secs[si][1] == i:
                nm, a, b = secs[si]
                xa, xb = xs[a] - colw[a] / 2 + 0.3 * s, xs[b] + colw[b] / 2 - 0.3 * s
                yb = sec_y + LINE_A * s + 0.45 * s
                strokes.append(_sk_line((xa, yb + 0.3 * s), (xa, yb), ctx)[:-1] + _sk_line((xa, yb), (xb, yb), ctx)
                               + _sk_line((xb, yb), (xb, yb + 0.3 * s), ctx)[1:])
                sl = ctx.label(nm, None)
                strokes += sl.draw(ctx, (xa + xb) / 2 - sl.w / 2, sec_y - 0.1 * s, 40 + si)
                si += 1
            cx = xs[i]
            strokes.append(_sk_line((cx, line_y - 0.35 * s), (cx, line_y + 0.35 * s), ctx))
            px0, py0 = cx - P[i].w / 2, line_y - 0.6 * s - P[i].h
            strokes += P[i].draw(ctx, px0, py0, 100 + i)
            nodes[items[i][0] or f"p{i}"] = (px0, py0, px0 + P[i].w, py0 + P[i].h)
            ey = line_y + 0.9 * s
            for j, e in enumerate(EV[i]):
                strokes += e.draw(ctx, cx - e.w / 2, ey, 200 + i * 10 + j)
                ey += e.h + 0.85 * s
    else:
        pw = max(p.w for p in P)
        lx = pw + 1.0 * s
        y = 0.0
        if tlab:
            strokes += tlab.draw(ctx, 0.0, y, 3)
            y += tlab.h + 1.0 * s
        rows = []
        yy = y + 0.6 * s
        for i in range(len(items)):
            hgt = max([P[i].h] + [sum(e.h for e in EV[i]) + 0.45 * s * max(0, len(EV[i]) - 1)])
            rows.append(yy)
            yy += hgt + 1.0 * s
        axis = _sk_line((lx, y), (lx, yy), ctx, 0.01)
        strokes.append(axis)
        strokes.append(geom.arrowhead(axis[-1], (lx, yy - 0.8 * s), ARROW * s, 27))
        for i in range(len(items)):
            ty = rows[i] + 0.5 * s
            strokes.append(_sk_line((lx - 0.35 * s, ty), (lx + 0.35 * s, ty), ctx))
            px0 = lx - 0.7 * s - P[i].w
            strokes += P[i].draw(ctx, px0, rows[i], 100 + i)
            nodes[items[i][0] or f"p{i}"] = (px0, rows[i], px0 + P[i].w, rows[i] + P[i].h)
            ey = rows[i]
            for j, e in enumerate(EV[i]):
                strokes += e.draw(ctx, lx + 0.8 * s, ey, 200 + i * 10 + j)
                ey += e.h + 0.45 * s
        if secs:
            ctx.warn("timeline sections are not drawn in the vertical layout")
    return strokes, nodes


# ======================================================================= mindmap

_MM_SHAPES = [("((", "))", "circle"), ("))", "((", "circle"), ("{{", "}}", "hexagon"), ("(", ")", "rounded"),
              ("[", "]", "rect"), (")", "(", "rounded")]


def _parse_mindmap(ctx, lines):
    items = []   # (indent, text, shape)
    for ln in lines:
        if not ln.strip():
            continue
        st = ln.strip()
        if st.startswith("::icon") or st.startswith(":::"):
            continue
        ind = len(ln) - len(ln.lstrip(" "))
        st = re.sub(r":::[\w\- ]+$", "", st).strip()
        text, shape = st, None
        for op, cl, shp in _MM_SHAPES:
            k = st.find(op)
            if k >= 0 and st.endswith(cl) and k + len(op) <= len(st) - len(cl) and re.match(r"^[\w\-]*$", st[:k]):
                text, shape = st[k + len(op):len(st) - len(cl)], shp
                break
        items.append((ind, _clean_label(text), shape))
    if not items:
        ctx.warn("mindmap has no nodes")
        return None
    root = {"text": items[0][1], "shape": items[0][2] or "circle", "kids": [], "ind": items[0][0]}
    stack = [root]
    for ind, text, shape in items[1:]:
        while len(stack) > 1 and stack[-1]["ind"] >= ind:
            stack.pop()
        if ind <= root["ind"] and len(stack) == 1:
            ctx.warn(f"mindmap: extra top-level node {text[:20]!r} attached to the root")
        nd = {"text": text, "shape": shape, "kids": [], "ind": ind}
        stack[-1]["kids"].append(nd)
        stack.append(nd)
    return root


def _mindmap_render(ctx, root):
    s = ctx.size
    wrap = ctx.wrap_w * 0.8 if ctx.wrap_w else 9 * s
    strokes, nodes = [], {}

    def prep(nd, depth):
        nd["depth"] = depth
        lab = ctx.label(nd["text"], wrap if depth else wrap * 0.8)
        nd["lab"] = lab
        if depth == 0:
            n = Node("root", nd["text"], nd["shape"] or "circle")
        elif depth == 1 or nd["shape"]:
            n = Node("n", nd["text"], nd["shape"] or "rounded")
        else:
            n = Node("n", nd["text"], "plain")
        if n.shape == "plain":
            n.lab = lab
            n.w, n.h = lab.w + 0.2 * s, lab.h + 0.5 * s
        else:
            _size_node(ctx, n, wrap if depth else wrap * 0.8)
        nd["node"] = n
        for k in nd["kids"]:
            prep(k, depth + 1)

    prep(root, 0)
    vgap = 0.6 * s
    hgap = 2.2 * s

    def span(nd):
        if not nd["kids"]:
            nd["span"] = nd["node"].h
        else:
            nd["span"] = max(nd["node"].h, sum(span(k) for k in nd["kids"]) + vgap * (len(nd["kids"]) - 1))
        return nd["span"]

    kids = root["kids"]
    for k in kids:
        span(k)
    total = sum(k["span"] for k in kids) + vgap * max(0, len(kids) - 1)
    right, left, acc = [], [], 0.0
    for k in kids:
        if len(kids) > 1 and acc + k["span"] / 2 > total / 2 and right:
            left.append(k)
        else:
            right.append(k)
            acc += k["span"] + vgap
    rn = root["node"]
    rn.x = rn.y = 0.0

    def place(nd_list, parent_edge_x, side, y_center):
        tot = sum(k["span"] for k in nd_list) + vgap * max(0, len(nd_list) - 1)
        y = y_center - tot / 2
        for k in nd_list:
            n = k["node"]
            n.y = y + k["span"] / 2
            n.x = parent_edge_x + side * (hgap + n.w / 2)
            y += k["span"] + vgap
            if k["kids"]:
                place(k["kids"], n.x + side * n.w / 2, side, n.y)

    place(right, rn.w / 2, 1, 0.0)
    place(left, -rn.w / 2, -1, 0.0)

    def draw_node(nd, idx):
        n = nd["node"]
        if n.shape == "plain":
            lab = n.lab
            x0 = n.x - lab.w / 2
            y0 = n.y - n.h / 2
            out = lab.draw(ctx, x0, y0, idx)
            out.append(_sk_line((n.x - n.w / 2, n.y + n.h / 2), (n.x + n.w / 2, n.y + n.h / 2), ctx))
            return out
        out = _outline(n, ctx)
        lx, ly = _label_origin(n)
        return out + n.lab.draw(ctx, lx, ly, idx)

    cnt = [0]
    strokes += draw_node(root, 0)
    nodes["root"] = rn.rect()

    def walk(nd, parent, side):
        for k in nd["kids"]:
            n, p = k["node"], parent["node"]
            if p.shape == "plain":
                a = (p.x + side * p.w / 2, p.y + p.h / 2)
            else:
                a = (p.x, p.y)
            b = (n.x - side * n.w / 2, n.y + (n.h / 2 if n.shape == "plain" else 0.0))
            dx = (b[0] - a[0]) * 0.55
            pts = geom.bezier(a, (a[0] + dx, a[1]), (b[0] - dx, b[1]), b, 28)
            if p.shape != "plain":
                pts = _trim(_dense(pts, 0.2 * s), p, 0.08 * s)
            if n.shape != "plain":
                pts = _trim(_dense(pts, 0.2 * s)[::-1], n, 0.08 * s)[::-1]
            strokes.append(pts)
            cnt[0] += 1
            strokes.extend(draw_node(k, cnt[0]))
            key = k["text"] or f"n{cnt[0]}"
            while key in nodes:
                key += "'"
            nodes[key] = n.rect()
            walk(k, k, side)

    for k in right:
        walk({"node": rn, "kids": [k]}, root, 1)
    for k in left:
        walk({"node": rn, "kids": [k]}, root, -1)
    return strokes, nodes


# ======================================================================= top level

def _fit(w, h, max_w, max_h):
    w, h = max(w, 1e-6), max(h, 1e-6)
    if max_w and max_h:
        return min(max_w / w, max_h / h)
    if max_w:
        return max_w / w
    if max_h:
        return max_h / h
    return 1.0


def _render_flow(ctx, g, max_w, max_h):
    s = ctx.size
    constrained = bool(max_w or max_h)
    wraps = [11.0 * s, 8.0 * s, 6.0 * s] if constrained else [11.0 * s]
    declared = g.direction

    def attempt(direction, wrap, labels=False, rows=None):
        g.direction = direction
        ctx.wrap_w = wrap
        ctx.rows = rows
        for e in g.edges:
            e.pts = e.lab = e.lab_xy = None
        order = _graph_render(g, ctx, labels)
        rects = [n.rect() for n in g.nodes.values()]
        for e in g.edges:
            if e.pts:
                rects.append(geom.bbox([e.pts]))
            if e.lab is not None and e.lab_xy is not None:
                rects.append((e.lab_xy[0], e.lab_xy[1], e.lab_xy[0] + e.lab.w, e.lab_xy[1] + e.lab.h))
        if not rects:
            return order, 1.0
        W = max(r[2] for r in rects) - min(r[0] for r in rects)
        H = max(r[3] for r in rects) - min(r[1] for r in rects)
        return order, _fit(W, H, max_w, max_h)

    flip = {"TD": "LR", "TB": "LR", "BT": "RL", "LR": "TD", "RL": "BT"}[declared]

    def pick(direction):
        bst = None
        for wr in wraps:
            _, sc = attempt(direction, wr)
            if bst is None or sc > bst[0] * 1.03:   # narrower wrap only when it clearly fits better
                bst = (sc, wr)
        return bst

    sc_d, wr = pick(declared)
    d = declared
    if constrained:
        sc_f, wr_f = pick(flip)
        if sc_f > 1.15 * sc_d:                       # clearly out of aspect -> switch (was 1.5x)
            d, wr = flip, wr_f
            ctx.warn(f"direction {declared} switched to {flip} to fit the {max_w}x{max_h} region")
    # decisions drawn as diamonds are about twice as wide as their words; when the diagram only
    # fits its region with its words below the given size, decision nodes become hexagons (the compact decision
    # shape) if that makes the words clearly bigger (an umbrella flowchart had its words at ~1/3 of the label size)
    dia = [n for n in g.nodes.values() if n.shape == "diamond"]
    if constrained and dia:
        _, sc_now = attempt(d, wr)
        if sc_now < 0.9:
            for n in dia:
                n.shape = "hexagon"
            best_h = None
            for dd in dict.fromkeys((declared, flip)):
                sc_h, wr_h = pick(dd)
                if best_h is None or sc_h > best_h[0]:
                    best_h = (sc_h, wr_h, dd)
            if best_h[0] > 1.12 * sc_now:
                d, wr = best_h[2], best_h[1]
                ctx.warn("decision diamonds drawn as hexagons to fit the region")
            else:
                for n in dia:
                    n.shape = "diamond"
    rows = None
    co = _chain_order(g) if constrained else None
    if co is not None:
        # a long chain / loop that only fits its region small is wrapped into rows (snaking), when
        # that makes the writing clearly bigger (a 5-step food chain on one line had its words at half the label size)
        _, sc_now = attempt(d, wr)
        n = len(co[1])
        cands = [(n + 1) // 2] if co[0] == "cycle" else list(range(2, n))
        for nc in cands:
            for wr2 in wraps:
                _, sc_r = attempt("LR", wr2, rows=nc)
                if sc_r > 1.15 * sc_now and (rows is None or sc_r > rows[0] * 1.03):
                    rows = (sc_r, nc, wr2)
        if rows is not None:
            d, wr = "LR", rows[2]
            ctx.warn(f"chain wrapped into rows of {rows[1]} to fit the {max_w}x{max_h} region")
    order, _ = attempt(d, wr, True, rows=rows[1] if rows else None)
    strokes = _draw_graph(g, ctx, order)
    ctx.out["edge_labels"] = [(e.lab_xy[0], e.lab_xy[1], e.lab_xy[0] + e.lab.w, e.lab_xy[1] + e.lab.h)
                              for e in g.edges if e.lab is not None and e.lab_xy is not None]
    nodes = {nid: n.rect() for nid, n in g.nodes.items() if not nid.startswith(("_start", "_end"))}
    for nid, n in g.nodes.items():
        if nid.startswith(("_start", "_end")):
            nodes[nid] = n.rect()
    return strokes, nodes


def _render(src, font, size, max_w, max_h, warnings, out=None):
    ctx = _Ctx(font, size, warnings, hash_text(src))
    ctx.out = out if out is not None else {}
    try:
        return _render_kind(ctx, src, font, size, max_w, max_h)
    finally:
        if ctx.unknown:
            ctx.warn("no glyph for: " + "".join(sorted(ctx.unknown)))


def _render_kind(ctx, src, font, size, max_w, max_h):
    ctx.wrap_w = 11.0 * size
    s = _prep(src)
    if not s.strip():
        ctx.warn("empty diagram")
        return [], {}
    body = s.lstrip(" \n;")
    m = re.match(r"(\w[\w-]*)", body)
    kind = m.group(1) if m else ""
    kl = kind.lower()
    if kl in ("graph", "flowchart", "flowchart-elk"):
        rest = body[m.end():]
        md = re.match(r"[ \t]*(TD|TB|BT|LR|RL|DT)\b", rest, re.I)
        direction = "TD"
        if md:
            direction = md.group(1).upper()
            direction = "BT" if direction == "DT" else direction
            rest = rest[md.end():]
        else:
            mb = re.match(r"[ \t]*([A-Za-z]{2})[ \t]*(?=;|\n|$)", rest)
            if mb:
                ctx.warn(f"unknown direction {mb.group(1)!r}; using TD")
                rest = rest[mb.end():]
        g = _parse_flowchart(ctx, rest, direction)
        return _render_flow(ctx, g, max_w, max_h)
    if kl == "timeline":
        lines = re.split(r"[\n;]", body[m.end():])
        title, items = _parse_timeline(ctx, lines)
        hz = _timeline_render(ctx, title, items, False)
        if max_w and max_h and len(items) > 2:
            vt = _timeline_render(ctx, title, items, True)
            bh, bv = geom.bbox(hz[0]), geom.bbox(vt[0])
            sh = _fit(bh[2] - bh[0], bh[3] - bh[1], max_w, max_h)
            sv = _fit(bv[2] - bv[0], bv[3] - bv[1], max_w, max_h)
            if sv > sh * 1.3:
                ctx.warn("timeline drawn vertically to fit the region")
                return vt
            if "timeline sections are not drawn in the vertical layout" in ctx.warnings:
                ctx.warnings.remove("timeline sections are not drawn in the vertical layout")
        return hz
    if kl == "mindmap":
        text = body[m.end():]
        lines = text.split("\n")
        if len([x for x in lines if x.strip()]) <= 1 and ";" in text:
            ctx.warn("mindmap on one line: indentation lost, all nodes attached to the root")
            parts = [p.strip() for p in text.split(";") if p.strip()]
            lines = parts[:1] + ["  " + p for p in parts[1:]]
        root = _parse_mindmap(ctx, lines)
        if root is None:
            return [], {}
        return _mindmap_render(ctx, root)
    if kl.startswith("statediagram"):
        ctx.warn("stateDiagram rendered as a flowchart")
        g = _state_to_flow(ctx, re.split(r"[\n;]", body[m.end():]))
        return _render_flow(ctx, g, max_w, max_h)
    if kl == "sequencediagram":
        ctx.warn("sequenceDiagram rendered as a flowchart of its messages")
        g = _seq_to_flow(ctx, re.split(r"[\n;]", body[m.end():]))
        return _render_flow(ctx, g, max_w, max_h)
    if re.search(r"-->|---|==>|-\.->", body):
        ctx.warn(f"missing or unknown diagram header {kind!r}; parsed as flowchart TD")
        g = _parse_flowchart(ctx, body, "TD")
        return _render_flow(ctx, g, max_w, max_h)
    ctx.warn(f"unsupported diagram type {kind!r}; rendered as flowchart best-effort")
    g = _parse_flowchart(ctx, body[m.end():] if m else body, "TD")
    return _render_flow(ctx, g, max_w, max_h)


def render_mermaid(src: str, font: HandFont, text_size: float, max_w: float | None = None,
                   max_h: float | None = None) -> dict:
    warnings: list[str] = []
    size = float(text_size) if text_size and text_size > 0 else 1.0
    font = font or HandFont()
    out: dict = {}
    try:
        strokes, nodes = _render(src if isinstance(src, str) else str(src or ""), font, size, max_w, max_h,
                                 warnings, out)
    except Exception as ex:   # last-resort safety net: never raise to the board
        if os.environ.get("BOARD_DIAGRAM_DEBUG"):
            raise
        warnings.append(f"internal error: {type(ex).__name__}: {ex}")
        strokes, nodes = [], {}
    step = SPACING * size
    clean, tags = [], []
    for st in strokes:
        tag = getattr(st, "lab", None)
        st = [(float(x), float(y)) for x, y in st if math.isfinite(x) and math.isfinite(y)]
        if len(st) >= 2:
            clean.append(geom.resample(st, step))
            tags.append(tag)
    if not clean:
        return {"strokes": [], "w": 0.0, "h": 0.0, "nodes": {}, "edge_labels": [], "warnings": warnings,
                "labels": []}
    labels = _label_spans(tags)
    x0, y0, x1, y1 = geom.bbox(clean)
    for r in list(nodes.values()) + list(out.get("edge_labels", [])):
        x0, y0, x1, y1 = min(x0, r[0]), min(y0, r[1]), max(x1, r[2]), max(y1, r[3])
    mg = MARGIN * size           # nothing touches the outer bounds
    x0, y0, x1, y1 = x0 - mg, y0 - mg, x1 + mg, y1 + mg
    clean = geom.transform(clean, dx=-x0, dy=-y0)
    nodes = {k: (r[0] - x0, r[1] - y0, r[2] - x0, r[3] - y0) for k, r in nodes.items()}
    labs = [(r[0] - x0, r[1] - y0, r[2] - x0, r[3] - y0) for r in out.get("edge_labels", [])]
    return {"strokes": clean, "w": x1 - x0, "h": y1 - y0, "nodes": nodes, "edge_labels": labs, "warnings": warnings,
            "labels": labels}


def _label_spans(tags):
    """Parallel stroke tags -> [{"text", "role", "strokes": [i0, i1]}] (contiguous runs of one label draw)."""
    out = []
    i = 0
    while i < len(tags):
        if tags[i] is None:
            i += 1
            continue
        j = i
        while j < len(tags) and tags[j] == tags[i]:
            j += 1
        out.append({"text": tags[i][1], "role": tags[i][2], "strokes": [i, j]})
        i = j
    return out
