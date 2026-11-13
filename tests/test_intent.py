"""python -m unittest tests/test_intent.py  (or python tests/test_intent.py) — the matcher on the shared vectors."""
import json, os, sys, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
sys.path.insert(0, HERE)
import intent  # noqa: E402
from check_cases import summ  # noqa: E402


class Vectors(unittest.TestCase):
    def test_cases(self):
        vec = json.load(open(os.path.join(HERE, "intent_vectors.json")))["cases"]
        self.assertGreaterEqual(len(vec), 150)
        self.assertGreaterEqual(sum(1 for c in vec if not c["expect"]), 50)
        for c in vec:
            cmds = [h["cmd"] for h in intent.Matcher().parse(c["text"])]
            with self.subTest(text=c["text"]):
                self.assertEqual([summ(x) for x in cmds], c["expect"])
                self.assertEqual(json.loads(json.dumps(cmds)), c["cmds"])

    def test_streams(self):
        for s in json.load(open(os.path.join(HERE, "stream_vectors.json")))["streams"]:
            st = intent.Matcher().stream()
            outs = [st.update(t, fin, ms) for t, fin, ms in s["steps"]]
            got = [[summ(h["cmd"]) for h in o] for o in outs]
            with self.subTest(name=s["name"]):
                self.assertEqual(got, s["expect"])
                if "replaces" in s:
                    self.assertEqual([summ(o[0]["replaces"]) if o and o[0].get("replaces") else None for o in outs], s["replaces"])

    def test_legacy_exact(self):
        """Matcher(act=False) is the matcher exactly (frozen vectors)"""
        vec = json.load(open(os.path.join(HERE, "intent_vectors_v1.json")))["cases"]
        for c in vec:
            cmds = [h["cmd"] for h in intent.Matcher(act=False).parse(c["text"])]
            with self.subTest(text=c["text"]):
                self.assertEqual(json.loads(json.dumps(cmds)), c["cmds"])

    def test_side_convention(self):
        c = intent.Matcher().parse("raise your left paw")[0]["cmd"]
        self.assertEqual(c["side"], "r")
        self.assertIn("hand_r_y", c["targets"])
        self.assertEqual(c["own"], "left")
        c = intent.Matcher().parse("look to your left")[0]["cmd"]
        self.assertEqual((c["target"], c["screen_side"]), ("left", "r"))

    def test_keys(self):
        k = intent.cmd_keys
        self.assertEqual(k({"action": "pose", "targets": {"hand_r_y": 1, "point_r": 1}}), ["hand_r"])
        self.assertEqual(k({"action": "gesture", "name": "wave", "side": "l"}), ["hand_l"])
        self.assertEqual(k({"action": "gesture", "name": "nod"}), ["head"])
        self.assertEqual(k({"action": "release", "parts": "all"}), ["release:all"])
        hit = intent.Matcher().parse("raise your left paw")[0]["cmd"]
        self.assertEqual(k(hit), ["hand_r"])


class Context(unittest.TestCase):
    def test_forget_context(self):
        """fixes: "more" / "higher" re-applies the last pose only while that pose is still the context"""
        m = intent.Matcher()
        hits = m.parse("raise your left paw", now_ms=1000)
        self.assertTrue(hits)
        for h in hits:
            m.commit(h, 1000)
        self.assertTrue(m.parse("higher", now_ms=3000))
        m.forget_context()
        self.assertFalse(m.parse("higher", now_ms=3500))
        for h in m.parse("raise your left paw", now_ms=5000):
            m.commit(h, 5000)
        for h in m.parse("relax", now_ms=6000):
            m.commit(h, 6000)
        self.assertFalse(m.parse("higher", now_ms=7000))


if __name__ == "__main__":
    unittest.main()
