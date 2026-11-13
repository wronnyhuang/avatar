"""intent: per-character grammar in the hub, the move/object voice tools, eval gates on the frozen sets.

    python3 -m unittest tests/test_intent_rules.py      (rtc part needs aiohttp, else skipped)
"""
import json, os, sys, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
sys.path.insert(0, HERE)
import intent  # noqa: E402
import intent_hub  # noqa: E402
from check_cases import summ  # noqa: E402

try:
    import rtc  # noqa: E402
except ImportError:          # aiohttp absent (system python): the tool tests skip
    rtc = None


class HubGrammar(unittest.TestCase):
    def test_muse_act_others_legacy(self):
        m = intent_hub.matcher_for("muse")
        self.assertEqual([summ(h["cmd"]) for h in m.parse("spin around three times")], ["move/spin count=3"])
        self.assertTrue(m.parse("put on your headphones"))
        fox = intent_hub.matcher_for("fox")
        self.assertIsNone(fox.A)
        with open(os.path.join(HERE, "intent_vectors_v1.json")) as f:
            vec = json.load(f)["cases"]
        for c in vec:
            with self.subTest(text=c["text"]):
                self.assertEqual(json.loads(json.dumps([h["cmd"] for h in fox.parse(c["text"])])), c["cmds"])

    def test_scene_nouns(self):
        sc = intent_hub.scene_for("muse")
        self.assertTrue(sc and sc.get("props"), "muse scene (renderer/animals/muse.props.json or the mock)")
        hit = intent_hub.matcher_for("muse").parse("pull the chair up to the desk")
        self.assertEqual(hit[0]["cmd"]["action"], "object")
        self.assertEqual(hit[0]["cmd"]["verb"], "pull")

    def test_set_character_swaps(self):
        h = intent_hub.IntentHub({"enabled": False}, lambda m: None)
        h.sink, h.character = ("loop", None, {}, 1), "fox"
        h.matcher = intent_hub.matcher_for("fox")
        h.set_character("muse")
        self.assertIsNotNone(h.matcher.A)
        h.set_character("fox")
        self.assertIsNone(h.matcher.A)


class LlmFallbackHub(unittest.TestCase):
    """intent c2: the hub asks the LLM only on a final the grammar missed (muse), sends source 'matcher-llm'"""

    def _hub(self, character="muse"):
        sent = []

        class Loop:
            def call_soon_threadsafe(self, fn):
                fn()
        h = intent_hub.IntentHub({"enabled": False}, lambda m: None)
        h.sink, h.character, h.gen = (Loop(), sent.append, {}, 1), character, 1
        h.matcher = intent_hub.matcher_for(character)
        asked = []

        class Stub:
            def on_for(self, c):
                return c == "muse"

            def wants(self, text, scene=None):
                import intent_llm
                return intent_llm.LlmFallback.wants(intent_llm.LlmFallback.__new__(intent_llm.LlmFallback), text, scene)

            def ask(self, text, scene=None):
                asked.append(text)
                return [{"action": "move", "name": "backflip"}], {"ms": 5.0, "usd": 0.0, "model": "stub"}
        h.llm = Stub()
        return h, sent, asked

    def _final(self, h, text):
        import threading
        before = set(threading.enumerate())
        h._on_line({"t": "f", "text": text, "a": 2.0, "r": "full"})
        for t in set(threading.enumerate()) - before:
            t.join(2)

    def test_grammar_miss_asks(self):
        h, sent, asked = self._hub()
        self._final(h, "yo muse lemme see a sick backwards flip thingy")
        self.assertEqual(asked, ["yo muse lemme see a sick backwards flip thingy"])
        self.assertEqual([(m["cmd"]["name"], m["source"]) for m in sent], [("backflip", "matcher-llm")])

    def test_grammar_hit_does_not_ask(self):
        h, sent, asked = self._hub()
        self._final(h, "do a backflip")
        self.assertEqual(asked, [])
        self.assertEqual([m.get("source") for m in sent], [None])

    def test_negatives_and_others_do_not_ask(self):
        h, sent, asked = self._hub()
        for t in ("did you do a backflip", "dont do a backflip", "my sister said do a backflip",
                  "if you could do a backflip that would be cool", "the laptop is open", "what a nice day"):
            self._final(h, t)
        self.assertEqual(asked, [])
        h, sent, asked = self._hub("fox")
        self._final(h, "yo lemme see a sick backwards flip thingy")
        self.assertEqual(asked, [])


def _base(stem):
    """the ORIGINAL (planner-off) builder: rtc.py keeps each earlier layer of <stem> under a private _<stem>_<tag>
    alias; the base is the one defined first in the file"""
    fs = {getattr(rtc, n) for n in dir(rtc) if n.startswith("_" + stem + "_") and callable(getattr(rtc, n))}
    fs = [f for f in fs if hasattr(f, "__code__")]
    return min(fs, key=lambda f: f.__code__.co_firstlineno)


@unittest.skipIf(rtc is None, "aiohttp missing (run with a venv that has aiohttp)")
class Tools(unittest.TestCase):
    def test_others_unchanged(self):
        # planner off (?planner=0 / config planner:false) = these sessions exactly
        for name, what in (("Felix", "fox"), ("Ori", "orb"), ("Bloop", "jelly blob")):
            a = rtc.realtime_session(name, what, "ash", "server", planner=False)
            b = _base("realtime_session")(name, what, "ash", "server")
            self.assertEqual(a, b)
            self.assertEqual(rtc.live_session(name, what, "ash", planner=False), _base("live_session")(name, what, "ash"))
            self.assertNotIn("move", [t["name"] for t in a["tools"]])
            # planner mode (default): perform_action for every character, still no move
            p = [t["name"] for t in rtc.realtime_session(name, what, "ash", "server")["tools"]]
            self.assertIn(rtc.ACTION_TOOL, p)                  # do_action (perform_action = alias)
            self.assertNotIn("move", p)

    def test_muse_tools(self):
        for kw in ({"what": "plush companion"}, {"what": "plush toy", "character": "muse"}):
            ck = {"character": kw["character"]} if "character" in kw else {}
            s = rtc.realtime_session("Muse", kw["what"], "shimmer", "server", planner=False, **ck)
            names = [t["name"] for t in s["tools"]]
            self.assertEqual(names[-2:], ["move", "object"])
            l = rtc.live_session("Muse", kw["what"], "shimmer", planner=False, **ck)
            self.assertEqual([t["name"] for t in l["delegation"]["responses"]["tools"]][-2:], ["move", "object"])
            # planner mode: the library is a cache — no move tool; object stays (scene state)
            sp = [t["name"] for t in rtc.realtime_session("Muse", kw["what"], "shimmer", "server", **ck)["tools"]]
            self.assertEqual(sp[sp.index(rtc.ACTION_TOOL):sp.index(rtc.ACTION_TOOL) + 2], [rtc.ACTION_TOOL, "object"])   # + program tools after
            self.assertNotIn("move", sp)
            lp = [t["name"] for t in rtc.live_session("Muse", kw["what"], "shimmer", **ck)["delegation"]["responses"]["tools"]]
            self.assertEqual(lp[lp.index(rtc.ACTION_TOOL):lp.index(rtc.ACTION_TOOL) + 2], [rtc.ACTION_TOOL, "object"])
        mv, ob = rtc.act_tools()
        self.assertEqual(mv["parameters"]["required"], ["name"])
        self.assertEqual(ob["parameters"]["required"], ["verb"])
        # every name / verb the matcher emits on the muse dev set is in the tool enums (one command language)
        heads = set()
        with open(os.path.join(HERE, "intent_dev.txt")) as f:
            for ln in f:
                for tok in ln.partition("|")[2].split():
                    if tok.startswith(("move/", "object/")):
                        heads.add(tok)
        missing = {h for h in heads if h.split("/")[1] not in (rtc.MOVES if h.startswith("move/") else rtc.OBJECT_VERBS)}
        # aliases the page/motion fold: flip/get_up/hold_up/catch/use (motion VERB_ALIAS / MOVE_ALIAS)
        self.assertLessEqual(missing, {"move/flip", "move/get_up", "object/hold_up", "object/catch", "object/use"})
        # short enough not to slow first audio: the two tools stay under ~2.7 KB of JSON
        self.assertLess(len(json.dumps(rtc.act_tools())), 2700)


if __name__ == "__main__":
    unittest.main()
