"""boardbrain unit tests (no model calls): font coverage, layout invariants, parser robustness, erase ops,
timing, streaming through a fake door.  Run: python -m unittest tests/test_board_brain.py  (or python this file)."""
import asyncio
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))

from board import geom  # noqa: E402
from board.font import HandFont, has_glyph  # noqa: E402
from board.figure import render_figure  # noqa: E402
from board.layout import Board  # noqa: E402
from board.mathlayout import layout_math  # noqa: E402
from board.timing import chalk_at, est_seconds, time_parameterize  # noqa: E402
import board_brain  # noqa: E402
from board_brain import _Stream, parse_line  # noqa: E402

REQUIRED = ("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,;:!?'\"()[]{}|/\\-+=<>%&*#@$_~"
            "−×÷≠≈≤≥±√∫Σ∑πθλΔαβμ∞→⇌°′" "ε σ ω φ γ δ ρ τ Ω Φ ∂ ∇ ← ↔ ⇒ · … éèàçñüöäôîûâ".replace(" ", ""))


def _ops(text):
    st = _Stream()
    return st.feed(text) + st.end()


def _apply_all(board, text):
    board.new_plan()
    items = []
    for op in _ops(text):
        items += board.apply(op)
    return items


def _content_rects(board):
    return [(r["id"], r["ext"] or r["bbox"]) for r in board.items.values() if not r["attached"]]


class FontTest(unittest.TestCase):
    def test_coverage(self):
        missing = [c for c in REQUIRED if not has_glyph(c)]
        self.assertEqual(missing, [])

    def test_strokes_dense_and_deterministic(self):
        f = HandFont(seed=4)
        a, adv, unk = f.run("Hello 123", 0.1)
        b, _, _ = HandFont(seed=4).run("Hello 123", 0.1)
        self.assertEqual(a, b)
        self.assertEqual(unk, [])
        for s in a:
            for i in range(1, len(s)):
                self.assertLessEqual(geom.seglen(s[i - 1], s[i]), 0.1 * 0.25)

    def test_variation(self):
        a, _, _ = HandFont(seed=1).run("888", 0.1)
        b, _, _ = HandFont(seed=2).run("888", 0.1)
        self.assertNotEqual(a, b)


class MathTest(unittest.TestCase):
    def test_constructs(self):
        for src in [r"\frac{1}{2}+\frac{1}{3}", r"x^{2}+y_1^2", r"\sqrt[3]{x+1}", r"\int_0^1 x^2\,dx",
                    r"\sum_{n=1}^{\infty}\frac{1}{n^2}", r"\lim_{x\to 0}\frac{\sin x}{x}", r"\ce{2H2 + O2 -> 2H2O}",
                    r"\ce{N2 + 3H2 <=> 2NH3}", r"\begin{pmatrix}1&2\\3&4\end{pmatrix}", r"\vec{F}=m\vec{a}",
                    r"3x+5 &= 20 \\ 3x &= 15", r"\left(\frac{a}{b}\right)^2", r"\boxed{x=5}", r"\cancel{6}",
                    r"\xrightarrow[\Delta]{heat}", r"f(x)=\begin{cases}x&x\ge0\\-x&x<0\end{cases}", "x² + 2x₁"]:
            b = layout_math(src, 0.1)
            self.assertGreater(len(b.strokes), 0, src)
            self.assertTrue(all(math.isfinite(c) for s in b.strokes for p in s for c in p), src)

    def test_malformed_never_raises(self):
        rnd = random.Random(7)
        pieces = ["\\frac", "{", "}", "^", "_", "\\sqrt", "[", "]", "\\left(", "\\right)", "x", "2", "\\begin{pmatrix}",
                  "&", "\\\\", "\\end{pmatrix}", "\\ce{", "->", "\\unknowncmd", "$", "\\text{", "∫", "\\"]
        for _ in range(400):
            src = "".join(rnd.choice(pieces) for _ in range(rnd.randint(1, 12)))
            layout_math(src, 0.1)


class FigureTest(unittest.TestCase):
    def test_commands(self):
        r = render_figure('view -3 3 -1 9; axes x y ticks=1; plot "x^2"; point 2 4 "(2,4)"; circle 0 0 1; '
                          'arc 0 0 2 0 90 arrow; curve 0 0 1 1 2 0; numberline -2 2 1; angle 0 0 0 45 "θ"',
                          HandFont(), 0.8, 0.6, 0.07)
        self.assertGreater(len(r["strokes"]), 10)
        x0, y0, x1, y1 = geom.bbox(r["strokes"])
        self.assertTrue(x1 - x0 < 0.95 and y1 - y0 < 0.75)

    def test_garbage(self):
        for src in ["", ";;;", "plot", "plot \"x^\"", "circle a b c", "label 1", "view 1 1 1 1", "polygon 1 2 3",
                    "plot \"__import__('os')\"", "plot \"x.real\"", "line 0 0 1e308 1e308", "zzz 1 2 3"]:
            r = render_figure(src, HandFont(), 0.8, 0.6, 0.07)
            self.assertTrue(all(math.isfinite(c) for s in r["strokes"] for p in s for c in p))


class ParserTest(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(parse_line("math m1 at=below:t1 | say: Now add. | write: = \\frac{5}{6}")["content"], "= \\frac{5}{6}")
        p = parse_line("m2 math | \\frac{1}{2}")
        self.assertEqual((p["kind"], p["id"], p["content"]), ("math", "m2", "\\frac{1}{2}"))
        self.assertEqual(parse_line("- text t1 | Hello there | Hi")["say"], "Hello there")
        self.assertEqual(parse_line("timeline d1 | say: x | write: 1775 : A")["content"], "timeline; 1775 : A")
        self.assertIsNone(parse_line("Sure! Here is the plan:"))
        self.assertIsNone(parse_line("```"))
        self.assertIsNone(parse_line("text t1 | say: nothing to write |"))
        self.assertEqual(parse_line("math m1 | x^2 | The square.")["content"], "x^2")   # swapped fields repaired

    def test_random_garbage_never_raises(self):
        rnd = random.Random(3)
        alphabet = "text math figure diagram erase box | say: write: = \\frac{1}{2} at=1,2 h=big ; \n -->[]"
        b = Board()
        for _ in range(300):
            s = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 80)))
            for op in _ops(s):
                b.apply(op)

    def test_partial_stream(self):
        st = _Stream()
        got = []
        text = "text t1 | say: a | write: First\nmath m1 | say: b | write: x=1\ndiagram d1 | say: c | write: graph TD; A-->B\n"
        for ch in text[:-15]:
            got += st.feed(ch)
        self.assertEqual([o["id"] for o in got], ["t1", "m1"])    # the diagram waits for its line to end
        got += st.end()
        self.assertEqual(got[-1]["kind"], "diagram")


class BoardTest(unittest.TestCase):
    PLAN = "\n".join([
        "text t1 at=top-left | say: Here's the problem. | write: Solve 3x + 7 = 22",
        "math m1 | say: Subtract seven. | write: 3x + 7 - 7 = 22 - 7",
        "math m2 align=eq | say: Simplify. | write: 3x = 15",
        "math m3 align=eq | say: Divide by three. | write: x = \\frac{15}{3}",
        "math m4 align=eq | say: So | write: x = 5",
        "box b1 target=m4 | say: Done! |",
        "figure f1 at=top-right wh=0.4,0.6 | say: A picture. | write: numberline 0 6 1; dot 5 0",
        "text t2 | say: More. | write: Check: 3(5)+7 = 22",
        "column c1 | say: Columns. | write: 347 ; x 28 ; --- ; 2776 ; 6940 ; --- ; 9716",
        "diagram d1 | say: Flow. | write: graph LR; A[Start] --> B[Mid]; B --> C[End]",
    ])

    def test_no_overlap_in_bounds_min_size(self):
        for seed in range(4):
            b = Board(seed=seed)
            items = _apply_all(b, self.PLAN)
            rects = _content_rects(b)
            for i in range(len(rects)):
                r = rects[i][1]
                self.assertTrue(r[0] >= -1e-6 and r[1] >= b.top - 1e-6 and r[2] <= b.aspect + 1e-6 and r[3] <= b.bot + 1e-6, rects[i])
                for j in range(i + 1, len(rects)):
                    self.assertFalse(geom.rects_overlap(r, rects[j][1], 0.0), (rects[i], rects[j]))
            for it in items:
                if it["kind"] in ("text", "math", "column"):
                    self.assertGreaterEqual(it["size"], b.min_size - 1e-9)
                for s in it["strokes"]:
                    for u, v in s:
                        self.assertTrue(-0.01 <= u <= 1.01 and 0.321 - 0.006 <= v <= 0.95 + 0.006, (it["id"], u, v))
            self.assertTrue(all(isinstance(it["est_s"], float) for it in items))

    def test_full_board_auto_erases(self):
        b = Board()
        lines = "\n".join("text l%d | say: x | write: Line number %d here" % (i, i) for i in range(30))
        items = _apply_all(b, lines)
        er = [it for it in items if it["kind"] == "erase"]
        self.assertTrue(er and all(it.get("auto") for it in er))
        self.assertTrue(all(it["strokes"] for it in er))
        # every content item that was written is either still on the board or was erased by an erase item
        erased = {t for it in er for t in it["target"]}
        for it in items:
            if it["kind"] == "text":
                self.assertTrue(it["id"] in b.items or it["id"] in erased)

    def test_erase_ops(self):
        b = Board()
        _apply_all(b, "text a | say: | write: one\ntext c | say: | write: two\ntext d | say: | write: three\nunderline u target=c | say: |")
        items = _apply_all(b, "erase e1 target=c | say: Bye. |")
        self.assertEqual(items[0]["kind"], "erase")
        self.assertEqual(set(items[0]["target"]), {"c", "u"})         # its annotation goes with it
        self.assertNotIn("c", b.items)
        reg = b.state()["items"][0]["bbox"]
        items = _apply_all(b, "erase e2 region=%f,%f,%f,%f | say: |" % tuple(reg))
        self.assertIn("a", items[0]["target"])
        items = _apply_all(b, "erase e3 target=all | say: Clean slate. |")
        self.assertEqual(b.items, {})
        self.assertEqual(items[0]["say_before"], "Clean slate.")

    def test_eraser_covers_ink(self):
        b = Board()
        items = _apply_all(b, "text a | say: | write: Hello there\nbox bx target=a | say: |")
        er = _apply_all(b, "erase e target=a | say: |")[0]
        ew, eh = er["erase_wh"]
        pts = [p for s in er["strokes"] for p in s]
        for it in items:
            for s in it["strokes"]:
                for u, v in s:
                    self.assertTrue(any(abs(u - a) <= ew / 2 + 1e-6 and abs(v - c) <= eh / 2 + 1e-6 for a, c in pts))

    def test_state_roundtrip(self):
        b = Board()
        _apply_all(b, self.PLAN)
        st = b.state()
        b2 = Board().load(st)
        self.assertEqual(set(b2.items), set(b.items))
        items = _apply_all(b2, "text z | say: | write: one more line")
        r = b2.items["z"]["bbox"]
        for k, rr in _content_rects(b2):
            if k != "z":
                self.assertFalse(geom.rects_overlap(r, rr, 0.0))


class TimingTest(unittest.TestCase):
    def test_monotonic_and_pen(self):
        b = Board()
        items = _apply_all(b, BoardTest.PLAN)
        for it in items:
            if not it["strokes"]:
                continue
            tl = time_parameterize(it["strokes"], 2.0)
            ts = [x[0] for x in tl]
            self.assertTrue(all(ts[i] > ts[i - 1] for i in range(1, len(ts))), it["id"])
            n_up = sum(1 for x in tl if not x[3])
            self.assertEqual(n_up, len([s for s in it["strokes"] if s]))
            self.assertAlmostEqual(it["est_s"], est_seconds(it["strokes"], 2.0), places=2)
            mid = chalk_at(tl, ts[len(ts) // 2])
            self.assertTrue(0 <= mid[0] <= 1 and 0 <= mid[1] <= 1)

    def test_speed_plausible(self):
        b = Board()
        it = _apply_all(b, "text t | say: | write: hello world 12345")[0]
        per = it["est_s"] / 15
        self.assertTrue(0.3 < per < 1.0, per)


class LayoutR7Test(unittest.TestCase):
    """r7: reach band, batch layout, page turns, arithmetic self-check, item addressing."""

    def _batch(self, b, text):
        from board_brain import Feeder
        b.new_plan()
        fd = Feeder(b)
        items = []
        for op in _ops(text):
            items += fd.push(op)
        return items + fd.flush()

    def _ink_in_band(self, items):
        for it in items:
            if it["kind"] in ("erase", "say"):
                continue
            for s_ in it["strokes"]:
                for u, v in s_:
                    self.assertTrue(0.321 - 0.006 <= v <= 0.95 + 0.006, (it["id"], v))
                    self.assertTrue(-0.005 <= u <= 1.005, (it["id"], u))

    def test_batch_in_band_no_overlap_legible(self):
        for seed in range(3):
            b = Board(seed=seed)
            items = self._batch(b, BoardTest.PLAN)
            self._ink_in_band(items)
            rects = _content_rects(b)
            for i in range(len(rects)):
                for j in range(i + 1, len(rects)):
                    self.assertFalse(geom.rects_overlap(rects[i][1], rects[j][1], 0.0), (rects[i], rects[j]))
            for it in items:
                if it["kind"] in ("text", "math", "column"):
                    self.assertGreaterEqual(it["size"], 0.08 * 0.85 - 1e-9)

    def test_page_turn_keeps_heading_and_reading_order(self):
        b = Board()
        lines = ["text h | say: x | write: Times table"] + [
            "math m%d | say: x | write: 7 \\times %d = %d" % (i, i, 7 * i) for i in range(1, 25)]
        items = self._batch(b, "\n".join(lines))
        self._ink_in_band(items)
        er = [it for it in items if it["kind"] == "erase"]
        self.assertTrue(er and all(it.get("auto") for it in er))
        self.assertIn("m24", b.items)
        written = [it["id"] for it in items if it["kind"] == "math"]
        self.assertEqual(written, ["m%d" % i for i in range(1, 25)])

    def test_reach_override(self):
        b = Board(reach=(0.0, 1.0))
        items = self._batch(b, "text a | say: | write: Hello")
        self.assertLess(items[0]["bbox"][1], 0.05)

    def test_arith_fix(self):
        b = Board()
        items = self._batch(b, "math m1 | say: | write: 60 + 80 = 150\ncolumn c1 | say: | write: 347 ; x 28 ; --- ; 9715")
        by = {it["id"]: it for it in items}
        self.assertEqual(by["m1"]["text"], "60 + 80 = 140")
        self.assertTrue(any("arith fixed" in w for w in by["m1"].get("warnings", [])))
        self.assertTrue(by["c1"]["text"].endswith("9716"))
        from board.arith import check_chain
        self.assertEqual(check_chain("2x + 3 = 11")[1], [])                      # letters: not checked
        self.assertEqual(check_chain("\\frac{730}{140} \\approx 5.21")[1], [])  # rounding with approx is fine

    def test_target_with_spaces_and_aliases(self):
        op = parse_line("underline u1 target=m4:x = 4 | say: There. |")
        self.assertEqual(op["opts"]["target"], "m4:x = 4")
        b = Board()
        self._batch(b, "text t1 | say: | write: Solve 2x = 8\nmath m4 | say: | write: x = 4\nbox b1 target=m4 | say: |")
        u = self._batch(b, "underline u1 target=m4:x = 4 | say: |")[0]
        m4 = b.items["m4"]["bbox"]
        self.assertGreater((u["bbox"][2] - u["bbox"][0]) * b.aspect, 0.6 * (m4[2] - m4[0]))   # the whole 'x = 4'
        e = self._batch(b, "erase e1 target=#1 | say: |")[0]
        self.assertEqual(e["target"], ["t1"])
        c = self._batch(b, "circle c1 target=answer | say: |")[0]
        self.assertEqual(c["target"], ["m4"])

    def test_column_box_marks_answer_row(self):
        b = Board()
        items = self._batch(b, "column c1 | say: | write: 4786 ; + 3459 ; --- ; 8245\nbox b1 target=c1 | say: |")
        col, box = items[0]["bbox"], items[1]["bbox"]
        self.assertGreater(box[1], col[1] + 0.5 * (col[3] - col[1]))        # only the last row is boxed

    def test_long_equation_splits(self):
        b = Board()
        items = self._batch(b, "math m1 | say: | write: 373 + 80 \\times 5.21 + 12345 - 678 = 373 + 416.8 + 12345 - 678 = 12456.8")
        self._ink_in_band(items)
        self.assertGreaterEqual(items[0]["size"], 0.08 - 1e-9)


class _FakeDoor:
    def __init__(self, text, chunk=7):
        self.text, self.chunk = text, chunk

    async def stream(self, **kw):
        for i in range(0, len(self.text), self.chunk):
            await asyncio.sleep(0.001)
            yield {"delta": self.text[i:i + self.chunk]}
        yield {"done": True, "ok": True, "text": self.text, "ms_first": 1, "ms_done": 2, "error": None}


class PlanTest(unittest.TestCase):
    def test_streams_items(self):
        seen = []
        stats = {}
        items = asyncio.run(board_brain.plan([{"role": "user", "text": "solve it"}], "solve 3x+7=22", None,
                                             on_item=lambda it: seen.append(it["id"]), door=_FakeDoor(BoardTest.PLAN),
                                             stats=stats))
        self.assertEqual(seen, [it["id"] for it in items])
        self.assertIn("m4", seen)
        self.assertTrue(stats["ok"])
        self.assertIn("items", stats["board_state"])
        # follow-up on the returned board state
        items2 = asyncio.run(board_brain.plan([], "underline the answer", stats["board_state"],
                                              door=_FakeDoor("underline u1 target=%s | say: That one. |" % stats["board_state"]["items"][0]["id"])))
        self.assertEqual(items2[0]["kind"], "underline")


if __name__ == "__main__":
    unittest.main()
