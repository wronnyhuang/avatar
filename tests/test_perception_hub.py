"""perception — the server side: gesture store + app.py's on_control routing for gestures / attention.

    python3 tests/test_perception_hub.py      (needs aiohttp for app.py)
Checks: save / list / forget / caps / slug safety of app/gesture_store.py; Session.on_control (unbound, on a stub
session) answers gesture_list with the stored models, refuses a bad save, routes attention_candidate to the judge as a
task and answers attention_rule at once — with a mocked vendor (no network, no spend).
"""
import asyncio
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
from gesture_store import GestureStore, slug  # noqa: E402
from attention_judge import AttentionJudge  # noqa: E402


def model(name, kind="static"):
    return {"name": name, "kind": kind, "examples": [{"mean": [0.1, None]}], "counters": [], "thr": 3.0}


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="avatar-gs_")
        self.s = GestureStore(self.d, log=lambda *_: None)

    def test_roundtrip(self):
        self.assertIsNone(self.s.handle({"type": "gesture_save", "name": "Peace sign", "model": model("Peace sign")}))
        self.assertIsNone(self.s.handle({"type": "gesture_save", "name": "peace-sign", "model": model("peace-sign", "moving")}))
        r = self.s.handle({"type": "gesture_list"})
        self.assertEqual(r["type"], "gestures")
        self.assertEqual(sorted(m["name"] for m in r["items"]), ["Peace sign", "peace-sign"])
        self.s.handle({"type": "gesture_forget", "name": "Peace sign"})
        self.assertEqual([m["name"] for m in self.s.handle({"type": "gesture_list"})["items"]], ["peace-sign"])
        self.s.handle({"type": "gesture_forget", "name": "all"})
        self.assertEqual(self.s.handle({"type": "gesture_list"})["items"], [])

    def test_refusals_and_paths(self):
        r = self.s.handle({"type": "gesture_save", "name": "x", "model": model("y")})
        self.assertEqual(r["type"], "gesture_saved"); self.assertFalse(r["ok"])
        r = self.s.handle({"type": "gesture_save", "name": "big", "model": dict(model("big"), pad="x" * 2_000_000)})
        self.assertFalse(r["ok"])
        self.assertNotIn("/", slug("../../etc/passwd")); self.assertTrue(slug("../../etc/passwd").startswith("etc-passwd-"))
        self.s.save("../../evil", model("../../evil"))
        self.assertTrue(all(os.path.dirname(os.path.join(self.s.dir, f)) == self.s.dir for f in os.listdir(self.s.dir)))


class StubHub:
    def __init__(self, d):
        self.gestures = GestureStore(d, log=lambda *_: None)
        async def vendor(payload, timeout_s):
            return {"choices": [{"message": {"content": json.dumps({"what": "a cat walks in", "worth_reply": True, "salience": 0.9, "reason": "asked"})}}],
                    "usage": {"cost": 0.0003}, "model": "mock"}
        self.attention = AttentionJudge(d, log=lambda *_: None, vendor=vendor)

    def activity(self, *_):
        pass


class RouteTest(unittest.TestCase):
    def test_on_control(self):
        try:
            import app as A  # noqa: E402  (needs aiohttp)
        except ImportError as e:
            self.skipTest(f"app.py needs aiohttp ({e})")
        d = tempfile.mkdtemp(prefix="avatar-route_")

        async def run():
            sess = type("S", (), {})()
            sess.hub = StubHub(d); sess.sent = []; sess.send = sess.sent.append; sess.tasks = []; sess.actions = []
            import time
            sess.t_mono = time.monotonic()
            on = A.Session.on_control
            on(sess, {"type": "gesture_save", "name": "wave", "model": model("wave", "moving")})
            on(sess, {"type": "gesture_list"})
            self.assertEqual(sess.sent[-1]["type"], "gestures"); self.assertEqual(sess.sent[-1]["items"][0]["name"], "wave")
            on(sess, {"type": "attention_rule", "op": "add", "text": "tell me if my cat walks in"})
            self.assertEqual(sess.sent[-1]["type"], "attention_rules"); self.assertEqual(len(sess.sent[-1]["rules"]), 1)
            img = "data:image/jpeg;base64," + "A" * 400
            on(sess, {"type": "attention_candidate", "id": "c1", "t": 1.0, "kinds": ["scene_change"], "images": [img, img], "summary": {}})
            self.assertEqual(len(sess.tasks), 1)
            await asyncio.gather(*sess.tasks)
            v = sess.sent[-1]
            self.assertEqual(v["type"], "attention"); self.assertTrue(v["worth_reply"]); self.assertEqual(v["id"], "c1")
            self.assertTrue(any("attention" in a for a in sess.actions))
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main(verbosity=1)
