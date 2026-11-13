"""boardvis: collision checker + label placement tests.  python -m unittest tests/test_board_labels.py"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))

from board import geom  # noqa: E402
from board.collide import check_items, seg_rect_dist, min_ink_dist  # noqa: E402
from board.figure import render_figure  # noqa: E402
from board.font import HandFont  # noqa: E402
from board.layout import Board  # noqa: E402
from board_brain import parse_line  # noqa: E402


def _item(iid, kind, strokes, labels=None, target=None):
    it = {"id": iid, "kind": kind, "strokes": strokes, "bbox": None}
    if labels is not None:
        it["labels"] = labels
    if target is not None:
        it["target"] = target
    return it


def _plan(lines, board=None):
    b = board or Board(aspect=2.0, seed=3)
    b.new_plan()
    ops = [parse_line(l) for l in lines]
    return b, b.apply_batch([o for o in ops if o])


class CheckerTest(unittest.TestCase):
    def test_primitives(self):
        r = (0.0, 0.0, 1.0, 1.0)
        self.assertEqual(seg_rect_dist((-1, 0.5), (2, 0.5), r), 0.0)          # crosses
        self.assertAlmostEqual(seg_rect_dist((1.5, -1), (1.5, 2), r), 0.5)     # beside
        self.assertAlmostEqual(seg_rect_dist((2, 2), (3, 3), r), 2 ** 0.5)   # off the corner
        self.assertAlmostEqual(min_ink_dist([[(0, 0), (1, 0)]], [[(0, 0.3), (1, 0.3)]], 1.0), 0.3)

    def test_categories(self):
        # a figure: one horizontal line at v=0.6, a label run (strokes 1..2) sitting on it, and one clear of it
        line = [(0.2, 0.6), (0.4, 0.6)]
        lab_on = [(0.25, 0.58), (0.27, 0.62)]          # glyph crossing the line
        lab_off = [(0.3, 0.45), (0.32, 0.48)]
        lab_off2 = [(0.305, 0.46), (0.315, 0.47)]      # overlaps lab_off
        fig = _item("f1", "figure", [line, lab_on, lab_off, lab_off2],
                    labels=[{"text": "A", "strokes": [1, 2]}, {"text": "B", "strokes": [2, 3]},
                            {"text": "C", "strokes": [3, 4]}])
        t1 = _item("t1", "text", [[(0.6, 0.5), (0.7, 0.5)]])
        t2 = _item("t2", "text", [[(0.6, 0.505), (0.7, 0.505)]])   # touches t1
        t3 = _item("t3", "text", [[(0.5, 0.96), (0.6, 0.96)]])     # below the reach band
        rep = check_items([fig, t1, t2, t3])
        c = rep["counts"]
        self.assertEqual(c["label_ink"], 1)            # A on the line
        self.assertEqual(c["label_label"], 1)          # B / C
        self.assertEqual(c["item_touch"], 1)
        self.assertEqual(c["out_band"], 1)

    def test_erase_and_marks(self):
        t1 = _item("t1", "text", [[(0.2, 0.5), (0.4, 0.5)]])
        box = _item("b1", "box", [[(0.19, 0.48), (0.41, 0.48), (0.41, 0.52), (0.19, 0.52), (0.19, 0.48)]],
                    target=["t1"])
        er = {"id": "e1", "kind": "erase", "target": ["t1"], "strokes": [[(0.3, 0.5)]]}
        t2 = _item("t2", "text", [[(0.2, 0.5), (0.4, 0.5)]])          # where t1 was: fine after the erase
        rep = check_items([t1, box, er, t2])
        self.assertEqual(rep["total"], 0, rep["events"])
        # a mark touching an item it does not mark is reported
        t3 = _item("t3", "text", [[(0.6, 0.5), (0.8, 0.5)]])
        box2 = _item("b2", "box", [[(0.59, 0.5), (0.85, 0.5)]], target=["t1"])
        self.assertEqual(check_items([t1, t3, box2])["counts"]["annot_touch"], 1)


class LabelPlacementTest(unittest.TestCase):
    CROWDED = [
        'view -1 7 -1 4; polygon 0 0 6 0 2 3; line -1 3 7 3 dashed; angle 0 0 0 56 "α"; angle 6 0 143 180 "β"; '
        'label 0 -0.4 "A" left; label 6 -0.4 "B" right; label 2 3.4 "C" above',
        'view -0.5 7 -1.6 1.6; axes x y ticks=1; plot "sin(x)" [0 6.283]; line 0 1 6.283 1 dashed; '
        'point 1.571 1 "max"; point 4.712 -1 "min"; label 6.283 0 "2π" below',
        'view -1 6 -1 5; arrow 0 0 4 0 "4 N"; arrow 0 0 0 3 "3 N"; line 4 0 4 3 dashed; line 0 3 4 3 dashed; '
        'arrow 0 0 4 3 "R"; rightangle 0 0 0',
        'view -5 5 -3 3; circle -2.5 0 1.2; circle 2.5 0 1.5; label -2.5 0 "Na"; label 2.5 0 "Cl"; '
        'arrow -1.2 1.8 1.2 1.8 "e-"; label 0 -2.5 "ionic bond" above',
    ]

    def test_figures_have_clear_labels(self):
        for src in self.CROWDED:
            b, items = _plan(["figure f1 wh=0.6,0.7 | say: Look. | write: " + src])
            figs = [i for i in items if i["kind"] == "figure"]
            self.assertEqual(len(figs), 1)
            f = figs[0]
            self.assertTrue(f.get("labels"), src)
            for L in f["labels"]:
                i0, i1 = L["strokes"]
                self.assertTrue(0 <= i0 < i1 <= len(f["strokes"]))
                self.assertEqual(len(L["bbox"]), 4)
            rep = check_items(items)
            self.assertEqual(rep["counts"]["label_ink"] + rep["counts"]["label_label"], 0, (src, rep["events"]))
            self.assertEqual(rep["counts"]["out_band"] + rep["counts"]["edge_band"], 0, (src, rep["events"]))

    def test_figure_fills_region(self):
        # a small triangle in a roomy view comes out as big as the region allows (view tightened, refit)
        r = render_figure("view -10 10 -10 10; polygon 0 0 4 0 0 3", HandFont(), 1.0, 0.6, 0.08)
        self.assertGreater(r["h"], 0.55)
        self.assertLessEqual(r["h"], 0.6 + 1e-6)
        self.assertLessEqual(r["w"], 1.0 + 1e-6)

    def test_label_minimum_size(self):
        r = render_figure(self.CROWDED[0], HandFont(), 0.9, 0.5, 0.08, label_min=0.065)
        for L in r["labels"]:
            if L["role"] in ("label", "axis"):
                st = r["strokes"][L["strokes"][0]:L["strokes"][1]]
                if L.get("leader") is not None:
                    st = [s for k, s in enumerate(st) if k + L["strokes"][0] != L["leader"]]
                x0, y0, x1, y1 = geom.bbox(st)
                if L["text"].isupper():
                    self.assertGreater(y1 - y0, 0.065 * 0.9, L)   # cap height kept at the legible minimum

    def test_diagram_labels_tagged_and_clear(self):
        b, items = _plan(["diagram d1 | say: Here. | write: graph LR; A[Start] -->|yes| B[Middle]; "
                          "B -->|no| C[End]; A -->|maybe| C"])
        d = [i for i in items if i["kind"] == "diagram"][0]
        roles = {L["role"] for L in d.get("labels", [])}
        self.assertIn("edge", roles)
        self.assertIn("node", roles)
        rep = check_items(items)
        self.assertEqual(rep["counts"]["label_ink"] + rep["counts"]["label_label"], 0, rep["events"])

    def test_item_shape_unchanged(self):
        # TEACH's fields stay as they were; labels is additive
        b, items = _plan(["text t1 | say: Hi. | write: Hello", "figure f1 | say: Look. | write: " + self.CROWDED[2]])
        for it in items:
            for k in ("id", "kind", "say_before", "say_after", "strokes", "bbox", "est_s", "text"):
                self.assertIn(k, it)
            for s in it["strokes"]:
                self.assertTrue(all(len(p) == 2 for p in s))


if __name__ == "__main__":
    unittest.main()


class LessonKeepsItsStepsTest(unittest.TestCase):
    """labels: a lesson that outgrows the band at the normal size is squeezed / flowed into
    free room — it never wipes its own steps while it is being taught."""

    def test_long_derivation_not_self_erased(self):
        lines = ["text t1 at=top-left | say: Fractions. | write: Adding fractions",
                 "math m1 | say: a | write: \\frac{1}{2}+\\frac{1}{4}",
                 "math m2 | say: b | write: =\\frac{2}{4}+\\frac{1}{4}",
                 "math m3 | say: c | write: =\\frac{3}{4}",
                 "text t2 at=top-right | say: d | write: Check with eighths",
                 "math m4 | say: e | write: \\frac{4}{8}+\\frac{2}{8}=\\frac{6}{8}",
                 "math m5 | say: f | write: \\frac{6}{8}=\\frac{3}{4}",
                 "box b1 target=m5 | say: g |"]
        b, items = _plan(lines)
        mine = set()
        for it in items:
            if it["kind"] == "erase":
                self.assertFalse(set(it.get("target") or []) & mine, "the lesson erased its own steps")
            elif it.get("strokes"):
                mine.add(it["id"])
        for rid in ("m1", "m2", "m3", "m4", "m5"):
            self.assertIn(rid, mine)


class ChainRowsTest(unittest.TestCase):
    def test_long_chain_wraps_into_rows(self):
        from board.diagram import render_mermaid
        src = "graph LR; A[Grass] --> B[Grasshopper]; B --> C[Frog]; C --> D[Snake]; D --> E[Hawk]"
        one = render_mermaid(src, HandFont(), 1.0)
        fit = render_mermaid(src, HandFont(), 1.0, max_w=25.5, max_h=6.9)
        s_one = min(25.5 / one["w"], 6.9 / one["h"])
        s_fit = min(25.5 / fit["w"], 6.9 / fit["h"])
        self.assertGreater(s_fit, 1.3 * s_one)          # the text comes out clearly bigger
        self.assertTrue(any("rows" in w for w in fit["warnings"]))

    def test_branching_graph_not_wrapped(self):
        from board.diagram import render_mermaid
        r = render_mermaid("graph LR; A-->B; A-->C; C-->D; D-->E", HandFont(), 1.0, max_w=10, max_h=6)
        self.assertFalse(any("rows" in w for w in r["warnings"]))
