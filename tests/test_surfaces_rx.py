"""scene: the render server's surface mirror (renderer/avatar/surfaces_rx.py) fed by the hub's app/surfaces.py."""
import os, sys, unittest
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "app"), os.path.join(HERE, "..", "renderer")]
import surfaces as hub                     # noqa: E402
from avatar import surfaces_rx as rx       # noqa: E402


class T(unittest.TestCase):
    def test_timed_apply_and_mirror(self):
        S = hub.Surfaces(); B = S.get("board"); m = rx.SurfaceMirror()
        msgs = B.draw_polyline([(0.1, 0.1), (0.3, 0.2), (0.5, 0.1)], 0.02, ts=[1.0, 1.5, 2.0])
        for x in msgs: m.feed(x)
        m.advance(1.2); self.assertEqual(m.stats["applied"], 0)
        m.advance(1.5); self.assertEqual(m.stats["applied"], 1)
        m.advance(5.0); self.assertEqual(m.stats["applied"], 2)
        self.assertTrue(np.array_equal(m.snapshot("board"), B.snapshot()))

    def test_now_and_clear(self):
        S = hub.Surfaces(); B = S.get("board"); m = rx.SurfaceMirror()
        for x in B.draw_polyline([(0.1, 0.5), (0.9, 0.5)], 0.05): m.feed(x)
        m.advance(-1e9)
        self.assertGreater(m.snapshot("board")[..., 3].max(), 200)
        for x in B.clear(): m.feed(x)
        m.advance(0.0)
        self.assertEqual(m.snapshot("board")[..., 3].max(), 0)

    def test_budget_strips_and_premul(self):
        m = rx.SurfaceMirror(budget=100_000)
        T = hub.Surfaces().get("tv")
        rgb = np.zeros((288, 512, 3), np.uint8); rgb[..., 0] = 200
        for x in T.put_frame(rgb): m.feed(x)
        m.advance(0.0)
        ups, rows = [], 0
        for _ in range(20):
            u = m.take()
            if not u: break
            ups += u
        self.assertGreater(len(ups), 1)                           # spread over several frames
        self.assertTrue(all(len(u[5]) <= 100_000 for u in ups))
        self.assertEqual(sum(u[4] for u in ups), 288)            # every row exactly once
        self.assertEqual(ups[0][5][0], 200)                      # alpha 255: premultiplied = straight

    def test_resend_all(self):
        m = rx.SurfaceMirror()
        m.mark_all_dirty()
        names = sorted(u[0] for u in m.take(budget=1 << 30))
        self.assertEqual(names, ["board", "tv"])

    def test_bad_message(self):
        m = rx.SurfaceMirror()
        self.assertFalse(m.feed(b"S" + b"\0" * 5)); self.assertEqual(m.stats["bad"], 1)


if __name__ == "__main__":
    unittest.main()
