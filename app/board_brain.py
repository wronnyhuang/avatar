"""The chalkboard "board brain".

    from board_brain import plan, BoardSession
    items = await plan(context, request, board_state, on_item=cb)

The voice model asks for something to be written ("show how to add 1/2 and 1/3"); a fast text model (Claude Haiku
by default, Sonnet by config) sees the recent conversation, the request and what is already on the board, and
streams a compact LINE FORMAT — one board item per line — that code turns into strokes:

    KIND ID key=value ... | say: SAY_BEFORE | write: CONTENT

Kinds: text, math (LaTeX subset), column (column arithmetic), diagram (Mermaid), figure (vector-figure DSL),
underline/box/circle/strike/check/arrow (annotations of an earlier id), erase, say. Code does all geometry:
handwriting font, math layout, diagram layout, placement in free space (no overlap, min legible size, clamped to the
writable area, auto-erase of the least relevant item when the board is full) and timing.

Each emitted ITEM (board uv: 0..1 x 0..1 of the writable area, y down):
    {"id", "kind", "say_before", "say_after", "strokes": [[(u, v), ...], ...], "bbox": [u0, v0, u1, v1],
     "est_s", "text", "size", ...}            erase items also carry "target": [ids], "erase_r" (v units)
Config (env): BOARD_BRAIN_MODEL=sonnet|haiku|opus (default sonnet), BOARD_BRAIN_MAX_S (default 25), BOARD_BRAIN_SPEED (chalk speed
multiplier, default 1.0).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time

from board.layout import Board, ANNOT
from board.timing import time_parameterize, chalk_at, est_seconds  # noqa: F401  (re-exported for TEACH)

DIAGRAM_WORDS = {"timeline", "mindmap", "flowchart", "graph", "sequence", "sequencediagram", "statediagram"}
KINDS = {"text", "math", "column", "diagram", "figure", "erase", "say"} | set(ANNOT)
KIND_ALIASES = {"eq": "math", "equation": "math", "formula": "math", "latex": "math", "mermaid": "diagram",
                "flowchart": "diagram", "graph": "diagram", "chart": "diagram", "timeline": "diagram", "mindmap": "diagram",
                "sequence": "diagram", "sequencediagram": "diagram", "statediagram": "diagram", "fig": "figure", "plot": "figure",
                "sketch": "figure", "draw": "figure", "drawing": "figure", "shape": "figure", "label": "text",
                "title": "text", "heading": "text", "note": "text", "arith": "column", "columns": "column",
                "longmath": "column", "clear": "erase", "wipe": "erase", "speak": "say", "speech": "say",
                "line": "underline", "ring": "circle", "cross": "strike", "tick": "check", "frame": "box"}

SYSTEM = r"""You plan what a friendly teacher (Muse, an avatar) writes on a chalkboard, step by step. Code draws everything; you only decide WHAT to write, in what ORDER, roughly WHERE, and the short line Muse SAYS before writing each item.

OUTPUT: only item lines, nothing else (no prose, no code fences, no numbering). One item per line:
KIND ID key=value ... | say: SAY | write: CONTENT
- KIND: text | math | column | diagram | figure | underline | box | circle | strike | check | arrow | erase | say
- ID: short unique id you invent (t1, m1, m2, d1...). Later lines may refer to it.
- SAY (after "say:"): the exact words Muse speaks right BEFORE writing this item: one short natural sentence (<= 12 words), teacher voice, varied. Don't read long content aloud verbatim.
- CONTENT (after "write:"): what goes on the board, ONE line.

KINDS
text: a short phrase (<= 6 words), may include inline math in $...$.  text t1 at=top-left | say: Here's our problem. | write: Add $\frac{1}{2}+\frac{1}{3}$
math: raw LaTeX (no $): \frac{a}{b}, x^{2}, a_{n}, \sqrt{x}, \sqrt[3]{x}, \int_{a}^{b}, \sum_{i=1}^{n}, \lim_{x\to 0}, \left( \right), \text{...}, \vec{F}, \overline{AB}, \cancel{x}, \boxed{x}, \pm \times \div \cdot \le \ge \ne \approx \to \infty \pi \theta \Delta ..., \begin{pmatrix}a&b\\c&d\end{pmatrix}, cases. Chemistry: \ce{2H2 + O2 -> 2H2O}, \ce{N2 + 3H2 <=> 2NH3}, \ce{Fe^3+}, \ce{CaCO3 ->[heat] CaO + CO2}.
   math m2 at=below:m1 align=eq | say: Now add the tops. | write: = \frac{5}{6}
column: column arithmetic; rows separated by ';', '---' is a rule line, a leading + - x ÷ is the operator, 'c: 1 1' is a row of small carry digits.  column c1 | say: Let's stack them. | write: 347 ; x 28 ; --- ; 2776 ; 6940 ; --- ; 9716
diagram: Mermaid on ONE line, ';' between statements, labels 1-3 words, at most 6 nodes (merge small steps; more nodes make every word too small to read on a 2:1 board). A diagram fills most of the board: it is the only item of its lesson (no heading, no notes: SAY the title and the explanation instead).  diagram d1 | say: Here's the cycle. | write: graph LR; A[Evaporation] --> B[Condensation]; B --> C[Precipitation]; C --> A
   Also: write: timeline; 1775 : Lexington; 1776 : Declaration   and   write: mindmap ...
figure: vector figure in math coordinates (y up), commands separated by ';':
  view x0 x1 y0 y1 | axes [xlabel] [ylabel] ticks=STEP | plot "expr in x" [x0 x1] [dashed] | param "x(t)" "y(t)" t0 t1 | line x1 y1 x2 y2 [dashed] | arrow x1 y1 x2 y2 ["label"] | polygon x y x y ... | polyline x y ... | rect x y w h | circle cx cy r | ellipse cx cy rx ry | arc cx cy r deg0 deg1 [arrow] | curve x y x y ... [closed] (smooth, for sketches) | bezier x0 y0 x1 y1 x2 y2 x3 y3 | dot x y | point x y "A" | label x y "text" [above|below|left|right] | angle x y deg0 deg1 ["θ"] | rightangle x y deg | numberline a b step
  figure f1 at=top-right wh=0.45,0.75 | say: Let me sketch the parabola. | write: view -3 3 -1 9; axes x y ticks=1; plot "x^2"; point 2 4 "(2,4)"
  For graphs, geometry, number lines, physics/optics (rays; a lens = thin ellipse; mirrors), vectors, simple sketches of objects/animals from circles, ellipses, curves, lines. Keep coordinates simple and correct (check every point lies where the math says). Few labels (<= 6), each 1-3 characters or a short word, placed off the lines. Make figures big: wh at least 0.4,0.6.
underline / box / circle / strike / check: mark an earlier item: target=ID, or just some characters of it: target=ID:TEXT (e.g. target=m1:c, target=m2:5x, target=m1:x#2 for the second x).  box b1 target=m3 | say: And that's our answer! |
arrow: arrow from one item (or ID:TEXT part) to another: target=ID to=ID:TEXT; write: optional 1-3 word label. To explain a symbol: write a short note item at=right, then arrow from the note to ID:symbol.
erase: target=all | target=ID,ID | region=u0,v0,u1,v1. Targets (erase and marks) may also be #n (the n-th thing written, as numbered in "Board now"), last, or answer (the boxed answer).  erase e1 target=all | say: Let me clear some space. |
say: no writing, just a spoken line.  say s1 | say: That's all there is to it!

PLACEMENT: code places everything (columns down the board, no overlaps, legible size); you give only ORDER and, when useful, a hint: at=next (default: next line) | at=right (same line, right of the previous item: for a short continuation like "= 5/6") | at=top-right (start a new column, e.g. a figure or a check beside the work) | at=below:ID. align=eq lines up '=' with the line above (steps that start with '='). h=small|large (relative). wh=W,H figure/diagram size (fractions of the board).
SPACE: Muse can only reach a WIDE strip of the board: the full width (u 0..1) but only v 0.32..0.95 of its height — about 3.2 times as wide as it is tall. Coordinates (the bboxes and free space below, at=U,V) are fractions of the WHOLE board; wh=W,H: W of the board width, H of the board height (H 0.63 = the whole strip). A figure fills the strip best when it is wider than tall (a view about 2-3 times as wide as tall). About 4 short lines per column (a line with a fraction counts as 2), 2-3 columns of ~14 characters. Budget: at most 6 content items (a line with a fraction or a stacked column counts as 2) — skip trivial steps, merge obvious ones. Each step a SHORT line (one '=' step per item). If you write more, code erases the earlier work to make room, which loses context.
An unrelated new topic: if ALL of it (every item you plan) fits in the free space listed, write there without erasing; otherwise start with 'erase e1 target=all' (never write first and then erase it to make room). A follow-up about the current board: keep it and refer to items by id.

TEACHING STYLE
- Key points only, like a real teacher: numbers, symbols, very short phrases. 3-6 items is typical. Abbreviate long requests to what matters (the answer and the key steps).
- Worked problems: one step per item, in order, ending with the FINAL ANSWER on its own item, then 'box' it. The math must be CORRECT: work it out before writing, check every number; use \approx when you round. Code re-checks plain arithmetic.
- Mark items by their id: underline/box/circle target=ID (the whole item) or target=ID:TEXT (just those characters). An arrow label must not repeat the note it points at.
- Board letters: Latin (with accents), Greek, digits and math symbols only. Write other scripts (Chinese, Japanese, Arabic, ...) as romanization, e.g. pinyin with tone marks (nǐ hǎo); SAY may name the characters.
- Use the user's language for board words. The first SAY says what you're about to do; you may end with a 'say' item that wraps up.
"""


def _shim_flags(model, system):
    return ["-p", "--output-format", "stream-json", "--include-partial-messages", "--verbose", "--model", model,
            "--system-prompt", system, "--tools", "", "--setting-sources", "", "--strict-mcp-config",
            "--disable-slash-commands", "--no-session-persistence"]


class _ShimDoor:
    """Minimal stand-in for app/claude_door.py used only when it is not importable: one cold
    `claude -p` per request."""

    async def stream(self, model="haiku", system="", user="", images=None, max_s=20.0, tag="board"):
        t0 = time.monotonic()
        exe = os.environ.get("AVATAR_CLAUDE_BIN") or shutil.which("claude") or "claude"
        proc = await asyncio.create_subprocess_exec(exe, *_shim_flags(model, system), user,
                                                    stdin=asyncio.subprocess.DEVNULL,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        text, first, err = [], None, None
        try:
            while True:
                left = max_s - (time.monotonic() - t0)
                if left <= 0:
                    err = "timeout"
                    break
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=left)
                if not line:
                    break
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("type") == "stream_event":
                    ev = d.get("event") or {}
                    dl = ev.get("delta") or {}
                    if ev.get("type") == "content_block_delta" and dl.get("type") == "text_delta" and dl.get("text"):
                        if first is None:
                            first = time.monotonic()
                        text.append(dl["text"])
                        yield {"delta": dl["text"]}
                elif d.get("type") == "result":
                    if d.get("is_error"):
                        err = str(d.get("result") or d.get("subtype") or "error")
                    elif not text and isinstance(d.get("result"), str):
                        text.append(d["result"])
                        yield {"delta": d["result"]}
                    break
        except asyncio.TimeoutError:
            err = "timeout"
        finally:
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            try:
                await asyncio.wait_for(proc.wait(), 2)
            except Exception:  # noqa: BLE001
                pass
        now = time.monotonic()
        yield {"done": True, "ok": err is None, "text": "".join(text), "error": err,
               "ms_first": int(((first or now) - t0) * 1000), "ms_done": int((now - t0) * 1000), "model": model}

    def prewarm(self, model, system, n=2):
        return None


def get_door():
    try:
        from claude_door import door
        return door
    except Exception:  # noqa: BLE001
        return _ShimDoor()


def prewarm(model=None):
    """Call when a session that may teach starts: keeps warm board-brain processes in the door."""
    try:
        get_door().prewarm(model or os.environ.get("BOARD_BRAIN_MODEL", "sonnet"), SYSTEM)
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------ line parser (streaming, forgiving)

_HEAD = re.compile(r"^\s*(?:[-*•]\s*|\d+[.)]\s+)?([A-Za-z_][A-Za-z0-9_]*)\b\s*(.*)$")
_KV = re.compile(r"([A-Za-z_]+)\s*=\s*(\"[^\"]*\"|'[^']*'|\S+)")


_KEYS = r"(?:at|pos|h|size|wh|w|align|target|to|of|from|region|box)"
_KV_SPACED = re.compile(r"\b(target|to)\s*=\s*(\S[^|]*?)(?=\s+%s\s*=|\s*$)" % _KEYS, re.I)
_SAYM = re.compile(r"\|\s*(say|says|speak|spoken)\s*:", re.I)
_WRM = re.compile(r"\|\s*(write|draw|content|board)\s*:", re.I)
CONTENT_KINDS = ("text", "math", "column", "diagram", "figure")


def _looks_math(s):
    return bool(re.search(r"\\[A-Za-z]+|[\^_]|=|\d\s*[+\-*/]\s*\d", s))


def _split_fields(rest, kind):
    """-> (head, say, content). Labeled fields ('| say: ... | write: ...', any order) win; otherwise positional
    'head | say | content' with repairs for a missing or swapped SAY."""
    ms, mw = _SAYM.search(rest), _WRM.search(rest)
    if ms or mw:
        cut = min(x.start() for x in (ms, mw) if x)
        head = rest[:cut]
        say = content = ""
        if ms and mw:
            if ms.start() < mw.start():
                say, content = rest[ms.end():mw.start()], rest[mw.end():]
            else:
                content, say = rest[mw.end():ms.start()], rest[ms.end():]
        elif ms:
            say = rest[ms.end():]
            if "|" in say and kind != "say":       # 'say: words | content' (write label forgotten)
                say, content = say.split("|", 1)
        else:
            content = rest[mw.end():]
        return head, say.strip().rstrip("|").strip(), content.strip()
    parts = rest.split("|", 2)
    head = parts[0]
    say = parts[1].strip() if len(parts) > 1 else ""
    content = parts[2].strip() if len(parts) > 2 else ""
    if kind in CONTENT_KINDS:
        if len(parts) == 2:
            say, content = "", say                   # SAY omitted: "math m1 | \frac12"
        elif not content and say:
            say, content = "", say                   # content put in the SAY slot
        elif kind == "math" and _looks_math(say) and not _looks_math(content):
            say, content = content, say              # swapped
    return head, say, content


def parse_line(line: str):
    """One model line -> op dict or None. Never raises."""
    s = line.strip()
    if not s or s.startswith("```") or s.startswith("#"):
        return None
    m = _HEAD.match(s)
    if not m:
        return None
    kind = m.group(1).lower()
    kind = KIND_ALIASES.get(kind, kind)
    rest = m.group(2)
    if kind not in KINDS:
        # "m1 math at=... | ..." (id before kind)
        m2 = re.match(r"\s*([A-Za-z_]+)\b(.*)$", rest, re.S)
        k2 = KIND_ALIASES.get(m2.group(1).lower(), m2.group(1).lower()) if m2 else None
        if not m2 or k2 not in KINDS:
            return None
        rest = m.group(1) + " " + m2.group(2)
        kind = k2
        m = re.match(r"^\s*(" + re.escape(m2.group(1)) + r")", m2.group(1))
    head, say, content = _split_fields(rest, kind)
    w0 = m.group(1).lower()
    if kind == "diagram" and w0 in DIAGRAM_WORDS and not re.match(
            r"\s*(graph|flowchart|timeline|mindmap|sequenceDiagram|stateDiagram)", content, re.I):
        content = w0 + ("; " if w0 in ("timeline", "mindmap") else " TD; ") + content
    if kind in ("text", "math", "column", "diagram", "figure") and not content.strip():
        return None
    # head: ID then key=value pairs
    opts = {}
    for k, v in _KV.findall(head):
        opts[k.lower()] = v.strip("\"'")
    # target=m4:x = 4 (a character target may contain spaces): up to the next known key or the end of the head
    for m in _KV_SPACED.finditer(head):
        opts[m.group(1).lower()] = m.group(2).strip().strip("\"'")
    head = _KV_SPACED.sub(" ", head)
    head_wo = _KV.sub(" ", head).split()
    rid = head_wo[0] if head_wo else None
    if rid and rid.lower() in ("at", "h", "target"):
        rid = None
    say = say.strip().strip('"').strip()
    if kind == "say":
        # 'say: words | figure f1 ... | say: ... | write: ...' -> the item, with the leading words as its SAY
        mi = re.search(r"\|\s*(?=(?:%s)\s+[A-Za-z_][A-Za-z0-9_]*\b)" % "|".join(sorted(KINDS - {"say"}, key=len, reverse=True)),
                       rest)
        if mi:
            inner = parse_line(rest[mi.end():])
            if inner is not None:
                lead = re.sub(r"^\s*:?\s*(say\s*:)?", "", rest[:mi.start()].split("|")[-1]).strip() or say
                if lead and not inner.get("say"):
                    inner["say"] = lead
                return inner
    if kind == "say" and not say and content:
        say, content = content, ""
    # forgive "say:" prefixes
    say = re.sub(r"^((muse|teacher)\s+)?(say|says|said)\s*:\s*", "", say, flags=re.I).strip().strip('"')
    if kind == "math":
        c = content.strip()
        if c.startswith("$$") and c.endswith("$$") and len(c) > 4:
            c = c[2:-2]
        elif c.startswith("$") and c.endswith("$") and len(c) > 2:
            c = c[1:-1]
        elif c.startswith("\\[") and c.endswith("\\]"):
            c = c[2:-2]
        content = c.strip()
    return {"kind": kind, "id": rid, "opts": opts, "say": say or None, "content": content}


class _Stream:
    """Accumulates text deltas, yields ops as soon as their line is complete. Diagram/figure ops wait for the next
    item header (or the end) so a model that spreads them over several lines still works."""

    def __init__(self):
        self.buf = ""
        self.held = None

    def feed(self, delta):
        self.buf += delta
        out = []
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            out += self._line(line)
        return out

    def _line(self, line):
        op = parse_line(line)
        out = []
        if op is None:
            if self.held is not None and line.strip() and not line.strip().startswith("```"):
                sep = "; " if self.held["kind"] in ("diagram", "figure") else " "
                self.held["content"] = (self.held["content"] + sep + line.strip()).strip("; ")
            return out
        if self.held is not None:
            out.append(self.held)
            self.held = None
        if op["kind"] in ("diagram", "figure"):
            self.held = op
        else:
            out.append(op)
        return out

    def end(self):
        out = []
        if self.buf.strip():
            out += self._line(self.buf)
            self.buf = ""
        if self.held is not None:
            out.append(self.held)
            self.held = None
        return out


# ------------------------------------------------------------------ prompt

def _fmt_context(context, limit=8):
    lines = []
    for t in (context or [])[-limit:]:
        if isinstance(t, dict):
            who = t.get("role") or t.get("who") or t.get("speaker") or "?"
            txt = t.get("text") or t.get("content") or ""
        elif isinstance(t, (list, tuple)) and len(t) >= 2:
            who, txt = t[0], t[1]
        else:
            who, txt = "?", str(t)
        who = {"assistant": "Muse", "model": "Muse", "user": "User"}.get(str(who), str(who))
        txt = re.sub(r"\s+", " ", str(txt)).strip()
        if len(txt) > 400:
            txt = txt[:400] + "…"
        lines.append("%s: %s" % (who, txt))
    return "\n".join(lines) if lines else "(none)"


def build_user(context, request, board: Board):
    st = board.state()
    items = st["items"]
    if items:
        rows, n = [], 0
        for i in items:
            if not i.get("attached"):
                n += 1
            rows.append("%s%s %s at %s: %s%s" % ("#%d " % n if not i.get("attached") else "   ", i["id"], i["kind"],
                                                 ",".join("%.2f" % x for x in i["bbox"]),
                                                 (i["text"] or "")[:60].replace("\n", " "),
                                                 (" (marks %s)" % i["attached"]) if i.get("attached") else ""))
        bl = "\n".join(rows)
        free = board.free_regions()
        fr = "; ".join(",".join("%.2f" % x for x in r) for r in free) if free else "none (board is full)"
        bs = "Board now, in writing order (#n = n-th thing written; id kind at u0,v0,u1,v1: content):\n%s\nFree space (u0,v0,u1,v1): %s" % (bl, fr)
    else:
        bs = "Board now: empty."
    return "Conversation (latest last):\n%s\n\nRequest: %s\n\n%s\n\nWrite the item lines now." % (
        _fmt_context(context), (request or "").strip(), bs)


# ------------------------------------------------------------------ the brain

class BoardSession:
    """Keeps the board model across requests in one session (TEACH holds one per session).

    The board model assumes every planned item gets written. If TEACH stops early (abort, new topic), it reports
    what really happened: `mark_done(item)` for every item it wrote (fully or partly) or erased, then `abort()`
    rebuilds the model from exactly those, so the next plan sees the real board."""

    def __init__(self, aspect=2.0, seed=0, speed=None):
        sp = float(os.environ.get("BOARD_BRAIN_SPEED", "1.0")) if speed is None else speed
        self.aspect, self.seed, self.speed = aspect, seed, sp
        self.board = Board(aspect=aspect, seed=seed, speed=sp)
        self.done = []
        self.tracking = False

    def state(self):
        return self.board.state()

    async def plan(self, context, request, on_item=None, model=None, max_s=None, door=None, stats=None):
        return await _run(self.board, context, request, on_item, model, max_s, door, stats)

    def mark_done(self, item):
        self.tracking = True
        self.done.append(item)

    def abort(self):
        """Forget planned-but-unwritten items: rebuild the board model from the items marked done."""
        if not self.tracking:
            return self.state()
        old = self.board
        b = Board(aspect=self.aspect, seed=self.seed, speed=self.speed)
        b.plan_no, b.order = old.plan_no, old.order
        for it in self.done:
            if it.get("kind") == "erase":
                b.erase_ids(list(it.get("target") or []))
            elif it.get("bbox"):
                b.load({"items": [it]})
        self.board = b
        return self.state()

    def clear(self):
        """The board was wiped outside the brain (e.g. TEACH cleared it): reset the model."""
        self.board = Board(aspect=self.aspect, seed=self.seed, speed=self.speed)
        self.done, self.tracking = [], False


async def plan(context, request, board_state=None, on_item=None, model=None, max_s=None, door=None, stats=None,
               aspect=2.0, seed=0):
    """Stateless entry: board_state = {"aspect", "items": [...]} as returned by a previous plan's
    `stats["board_state"]` or BoardSession.state(). Returns the list of items (also streamed via on_item, which
    may be sync or async)."""
    board = Board(aspect=(board_state or {}).get("aspect", aspect) if isinstance(board_state, dict) else aspect,
                  seed=seed, speed=float(os.environ.get("BOARD_BRAIN_SPEED", "1.0")))
    board.load(board_state)
    return await _run(board, context, request, on_item, model, max_s, door, stats)


class Feeder:
    """Turns parsed ops into items with the board's two-phase layout: everything up to and including the FIRST
    content item is placed at once (writing starts immediately); the rest of the plan is held and laid out as one
    batch (`Board.apply_batch`: best of a small knob grid — size, wrap width, figure scale — with reserved room for
    later marks) when the stream ends or the hold deadline passes. The performer is busy saying + writing the
    first item meanwhile, so holding costs nothing visible."""

    def __init__(self, board):
        self.board = board
        self.started = False
        self.pending = []

    def push(self, op):
        if not self.started and op.get("kind") in ("figure", "diagram"):
            # a picture first: hold it with the rest so the batch can size/centre it with the whole plan in view
            self.started = True
            self.pending.append(op)
            return []
        if not self.started:
            items = self.board.apply(op)
            if op.get("kind") in CONTENT_KINDS and any(i.get("strokes") for i in items):
                self.started = True
            return items
        self.pending.append(op)
        return []

    def flush(self):
        ops, self.pending = self.pending, []
        return self.board.apply_batch(ops) if ops else []


def _say_s(it):
    return 0.4 * len((it.get("say_before") or "").split())


async def _run(board, context, request, on_item, model, max_s, door, stats):
    model = model or os.environ.get("BOARD_BRAIN_MODEL", "sonnet")
    max_s = max_s or float(os.environ.get("BOARD_BRAIN_MAX_S", "25"))
    door = door or get_door()
    board.new_plan()
    user = build_user(context, request, board)
    t0 = time.monotonic()
    items = []
    st = _Stream()
    first_ms = None
    raw = []
    errors = []

    feeder = Feeder(board)
    hold_s = os.environ.get("BOARD_BRAIN_HOLD_S")
    deadline = [None]

    async def out(got):
        nonlocal first_ms
        for it in got:
            if first_ms is None and it.get("strokes"):
                first_ms = int((time.monotonic() - t0) * 1000)
            items.append(it)
            if on_item is not None:
                r = on_item(it)
                if asyncio.iscoroutine(r):
                    await r
        busy = sum(float(it.get("est_s") or 0) + _say_s(it) for it in got)
        if got and busy:
            deadline[0] = time.monotonic() + (float(hold_s) if hold_s else max(1.5, 0.6 * busy))

    async def emit(op):
        try:
            got = feeder.push(op)
        except Exception as e:  # noqa: BLE001 — one bad item never ends the lesson
            errors.append("%s %s: %s" % (op.get("kind"), op.get("id"), e))
            return
        await out(got)

    async def flush():
        try:
            got = feeder.flush()
        except Exception as e:  # noqa: BLE001
            errors.append("batch: %s" % e)
            return
        await out(got)

    done = None
    async for ev in door.stream(model=model, system=SYSTEM, user=user, max_s=max_s, tag="board"):
        if ev.get("delta"):
            raw.append(ev["delta"])
            for op in st.feed(ev["delta"]):
                await emit(op)
            if feeder.pending and deadline[0] is not None and time.monotonic() > deadline[0]:
                await flush()           # the performer would run dry: hand over what is planned so far
        if ev.get("done"):
            done = ev
    for op in st.end():
        await emit(op)
    await flush()
    # repair: the model answered but nothing drawable came out (wrong format) -> one retry with a reminder
    retried = False
    wanted_ink = not re.search(r"\b(erase|wipe|clear|underline|circle|box)\b", request or "", re.I)
    if wanted_ink and not any(it.get("strokes") for it in items) and (time.monotonic() - t0) < max_s * 0.6:
        retried = True
        st = _Stream()
        bad = "".join(raw)[:600]
        user2 = (user + "\n\nYour previous answer was not in the item-line format and nothing could be drawn:\n" + bad +
                 "\n\nAnswer again using ONLY lines like: text t1 at=top-left | say: ... | write: ...")
        async for ev in door.stream(model=model, system=SYSTEM, user=user2, max_s=max(4.0, max_s - (time.monotonic() - t0)),
                                    tag="board"):
            if ev.get("delta"):
                raw.append(ev["delta"])
                for op in st.feed(ev["delta"]):
                    await emit(op)
            if ev.get("done"):
                done = ev
        for op in st.end():
            await emit(op)
        await flush()
    if stats is not None:
        stats.update({"model": model, "ms_first_item": first_ms, "ms_total": int((time.monotonic() - t0) * 1000),
                      "ms_first_token": (done or {}).get("ms_first"), "ok": bool(done and done.get("ok")), "retried": retried,
                      "error": (done or {}).get("error"), "raw": "".join(raw), "errors": errors,
                      "board_state": board.state()})
    return items
