"""teach: the hub side (app/teach_hub.py) — plans stream to the page, ink lands in the board surface with the page's
playback stamps, cancel stops a plan, erase updates the board state."""
import asyncio, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from teach_hub import TeachHub, summarize  # noqa: E402


class FakeSession:
    def __init__(self):
        self.out = []

    def send(self, m):
        self.out.append(m)


class FakeBrainSession:
    def __init__(self, n=3, gap=0.05):
        self.n, self.gap = n, gap

    async def plan(self, ctx, request, on_item=None):
        for i in range(self.n):
            await asyncio.sleep(self.gap)
            r = on_item({"id": f"m{i}", "kind": "math", "say_before": f"line {i}", "strokes": [[(0.1, 0.1 + i * 0.2), (0.2, 0.1 + i * 0.2)]],
                         "bbox": (0.1, 0.1 + i * 0.2, 0.2, 0.15 + i * 0.2), "text": f"x={i}"})
            if asyncio.iscoroutine(r):
                await r


class Rec:
    def __init__(self):
        self.ops = []

    def get(self, name):
        return self

    def draw_polyline(self, pts, width, rgba=None, t=None, ts=None):
        self.ops.append(("line", pts, width, t))

    def erase_disc(self, c, r, t=None, residue=0.08):
        self.ops.append(("erase", c, r, t))


class T(unittest.TestCase):
    def run_(self, coro):
        return asyncio.new_event_loop().run_until_complete(coro)

    def test_plan_stream_ink_state(self):
        async def go():
            s, surf = FakeSession(), Rec()
            h = TeachHub(s, surfaces=surf, brain=False)
            h.bsess = FakeBrainSession()
            h.on_msg({"type": "teach_plan", "id": "q1", "request": "solve", "context": [{"role": "user", "text": "hi"}]})
            await asyncio.sleep(0.4)
            items = [m for m in s.out if m["type"] == "board_item"]
            self.assertEqual(len(items), 3)
            self.assertTrue(any(m["type"] == "board_plan_done" and m["n"] == 3 for m in s.out))
            self.assertIsInstance(items[0]["item"]["bbox"], list)          # tuples -> JSON lists
            h.on_msg({"type": "board_ink", "t": 12.5, "segs": [{"item": "m0", "stroke": 0, "pts": [[0.1, 0.1], [0.15, 0.1]]},
                                                               {"erase": True, "pts": [[0.3, 0.3]], "rv": 0.05}]})
            self.assertEqual(surf.ops[0][0], "line"); self.assertEqual(surf.ops[0][3], 12.5)
            self.assertEqual(surf.ops[1][0], "erase"); self.assertAlmostEqual(surf.ops[1][2], 0.05)
            h.on_msg({"type": "board_item_done", "id": "q1", "item": "m0", "ok": True})
            st = [m for m in s.out if m["type"] == "board_state"][-1]
            self.assertIn('"x=0"', st["summary"])
            h.on_msg({"type": "board_erased", "id": "q1", "item": "e", "regions": [[0, 0, 1, 1]]})
            st = [m for m in s.out if m["type"] == "board_state"][-1]
            self.assertEqual(st["summary"], "the board is empty")
        self.run_(go())

    def test_cancel(self):
        async def go():
            s = FakeSession()
            h = TeachHub(s, surfaces=Rec(), brain=False)
            h.bsess = FakeBrainSession(n=10, gap=0.1)
            h.on_msg({"type": "teach_plan", "id": "q1", "request": "a long one"})
            await asyncio.sleep(0.25)
            h.on_msg({"type": "teach_cancel", "id": "q1"})
            await asyncio.sleep(0.4)
            n = len([m for m in s.out if m["type"] == "board_item"])
            self.assertLess(n, 4)
            self.assertFalse(any(m["type"] == "board_plan_done" for m in s.out))
        self.run_(go())

    def test_erase_all_and_stub(self):
        async def go():
            s = FakeSession()
            h = TeachHub(s, surfaces=Rec(), brain=False)
            h.on_msg({"type": "teach_erase", "id": "e1", "what": "everything"})
            await asyncio.sleep(0.05)
            it = [m for m in s.out if m["type"] == "board_item"][0]["item"]
            self.assertEqual(it["kind"], "erase"); self.assertEqual(it["regions"], [[0, 0, 1, 1]])
            h.on_msg({"type": "teach_plan", "id": "q2", "request": "2 + 2 = 4"})
            await asyncio.sleep(1.2)
            items = [m["item"] for m in s.out if m["type"] == "board_item" and m["id"] == "q2"]
            self.assertEqual(len(items), 2)
            self.assertTrue(all(it["strokes"] for it in items))
        self.run_(go())

    def test_reach_band(self):
        # teach: a brain that lays out inside the reach band itself -> full-board coords, the band goes to its env
        # switch; an older brain (no reach) -> its 0..1 is mapped onto the band the page names
        h = TeachHub(type("S", (), {"send": lambda self, m: None})())
        h._reach_ok = True
        h._area({"v0": 0.321, "v1": 0.95, "aspect": 2.0})
        self.assertEqual(h.band, (0.0, 1.0))
        it = {"id": "a", "kind": "text", "strokes": [[[0.1, 0.5], [0.2, 0.6]]], "bbox": [0.1, 0.5, 0.2, 0.6]}
        self.assertEqual(h._to_board(it), it)
        h2 = TeachHub(type("S", (), {"send": lambda self, m: None})())
        h2._reach_ok = False
        h2._area({"v0": 0.3, "v1": 0.9, "aspect": 2.0})
        self.assertEqual(h2.band, (0.3, 0.9)); self.assertAlmostEqual(h2.aspect, 2.0 / 0.6, 3)
        b = h2._to_board(dict(it, regions=[[0, 0, 1, 1], [0, 0.5, 1, 1]]))
        self.assertAlmostEqual(b["bbox"][1], 0.6, 4); self.assertAlmostEqual(b["strokes"][0][1][1], 0.66, 4)
        self.assertEqual(b["regions"][0], [0, 0, 1, 1]); self.assertAlmostEqual(b["regions"][1][3], 0.9, 4)

    def test_fast_asr_stop_words(self):   # integ
        out = []
        h = TeachHub(type("S", (), {"send": lambda self, m: out.append(m)})())
        h.on_asr("wait wait", False)
        h.on_asr("wait wait stop", False)
        h.on_asr("wait wait stop", False)                  # same partial again: once
        h.on_asr("so the bus stop is near the school and then we walk to the park", True)   # long: not a stop
        self.assertEqual([m["text"] for m in out], ["wait wait stop"])
        self.assertEqual(out[0]["type"], "teach_heard")

    def test_sweep_and_partial_erase(self):   # erase
        class RR(Rec):
            def erase_rect(self, a, b, t=None, residue=0.08):
                self.ops.append(("rect", a, b, t, residue))
        s, surf = FakeSession(), RR()
        h = TeachHub(s, surfaces=surf, brain=False)
        done = []
        h.bsess = type("B", (), {"mark_done": lambda self, it: done.append(it["id"])})()
        h.raw["e1"] = {"id": "e1", "kind": "erase"}
        # a swept stamp: the felt moved 0.3 uv in one frame -> rects all along it, none further apart than the pad
        h.on_msg({"type": "board_ink", "t": 3.0, "segs": [{"erase": True, "sweep": True, "pts": [[0.1, 0.5], [0.4, 0.5]], "ru": 0.05, "rv": 0.04}]})
        cs = [((o[1][0] + o[2][0]) / 2) for o in surf.ops if o[0] == "rect"]
        self.assertGreaterEqual(len(cs), 8)
        self.assertAlmostEqual(min(cs), 0.1, 3); self.assertAlmostEqual(max(cs), 0.4, 3)
        self.assertTrue(all(b - a <= 0.04 + 1e-9 for a, b in zip(sorted(cs), sorted(cs)[1:])))
        # a lazy page turn: partial wipes update the board state; the brain hears only the final one
        h.items = {"a": {"id": "a", "kind": "text", "bbox": [0.1, 0.4, 0.2, 0.5]}, "b": {"id": "b", "kind": "text", "bbox": [0.7, 0.4, 0.8, 0.5]}}
        h.on_msg({"type": "board_erased", "id": "q", "item": "e1", "regions": [[0, 0.3, 0.35, 1]], "partial": True, "t": 4.0})
        self.assertEqual(list(h.items), ["b"]); self.assertEqual(done, [])
        h.on_msg({"type": "board_erased", "id": "q", "item": "e1", "regions": [[0.65, 0.3, 1, 1]], "t": 6.0})
        self.assertEqual(list(h.items), []); self.assertEqual(done, ["e1"])

    def test_erase_that_no_brain(self):   # erase: "erase that" after a lesson = its items, no brain call
        async def go():
            s = FakeSession()
            h = TeachHub(s, surfaces=Rec(), brain=False)
            h.bsess = FakeBrainSession(n=2, gap=0.01)
            h.on_msg({"type": "teach_plan", "id": "q1", "request": "a"}); await asyncio.sleep(0.1)
            for i in ("m0", "m1"):
                h.on_msg({"type": "board_item_done", "id": "q1", "item": i, "ok": True})
            h.bsess = FakeBrainSession(n=1, gap=0.01)
            h.on_msg({"type": "teach_plan", "id": "q2", "request": "b"}); await asyncio.sleep(0.1)
            h.on_msg({"type": "board_item_done", "id": "q2", "item": "m0", "ok": True})   # (same id: the newer lesson's m0 replaces it)
            n0 = len(s.out)
            h.on_msg({"type": "teach_erase", "id": "e1", "what": "that"}); await asyncio.sleep(0.05)
            it = [m for m in s.out[n0:] if m["type"] == "board_item"][0]["item"]
            self.assertEqual(it["kind"], "erase"); self.assertEqual(it["target"], ["m0"]); self.assertEqual(len(it["regions"]), 1)
        self.run_(go())

    def test_summary(self):
        self.assertEqual(summarize([]), "the board is empty")
        self.assertIn("top left", summarize([{"id": "a", "kind": "text", "text": "hi", "bbox": [0, 0, 0.2, 0.1]}]))


if __name__ == "__main__":
    unittest.main()
