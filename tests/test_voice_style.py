"""voice style, server side of set_style: page {type:'style', energy?, mirroring?, expressiveness?} -> app.Session.on_style
-> unified_link.UnifiedRx.set_style -> the next worker step header {"style": {...}} -> unified_worker.Engine.set_style ->
the planner's set_style when it has one. Harmless in every other react mode. No network, no worker process.
    python3 tests/test_voice_style.py
"""
import asyncio
import os
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
import unified_link  # noqa: E402


class FakeWorker:
    def __init__(self):
        self.sent = []

    def alive(self):
        return True

    async def ensure(self):
        return True

    def send(self, hdr, blobs=None):
        self.sent.append(hdr)
        fut = asyncio.get_running_loop().create_future()
        if hdr.get("op") == "reset":
            fut.set_result(({"id": 1, "ok": True}, b""))
        else:
            n = len(hdr.get("ticks", []))
            import numpy as np
            fut.set_result(({"id": 2, "n": n}, np.zeros((n, 3), np.float32).tobytes()))
        return fut


def rx():
    return unified_link.UnifiedRx(FakeWorker(), dict(unified_link.DEFAULTS, timeout_ms=500), lambda *a: None)


async def step(r):
    return await r.step([(0.0, [0.0, 0.0, 0.0], [], 0.0)], "muse", ["a", "b", "c"])


class Style(unittest.TestCase):
    def test_clean_style(self):
        c = unified_link.clean_style
        self.assertEqual(c({"energy": 0.3, "mirroring": 2, "expressiveness": None}), {"energy": 0.3, "mirroring": 1.0, "expressiveness": None})
        self.assertEqual(c({"energy": -1}), {"energy": None})                       # -1 = automatic
        self.assertEqual(c({"energy": "auto", "x": 1, "mirroring": True, "expressiveness": "high"}), {"energy": None})
        self.assertEqual(c({}), {})
        self.assertEqual(c(None), {})

    def test_rx_header(self):
        async def run():
            r = rx()
            await step(r)                                                             # reset + a step without style
            r.set_style({"energy": 0.8})
            await step(r)
            r.set_style({"mirroring": None})
            await step(r)
            await step(r)                                                             # unchanged: no style key
            r.need_reset = True                                                       # the worker restarted
            await step(r)
            return r
        r = asyncio.run(run())
        steps = [h for h in r.w.sent if h.get("op") == "step"]
        self.assertNotIn("style", steps[0])
        self.assertEqual(steps[1]["style"], {"energy": 0.8})
        self.assertEqual(steps[2]["style"], {"energy": 0.8, "mirroring": None})
        self.assertNotIn("style", steps[3])
        self.assertEqual(steps[4]["style"], {"energy": 0.8, "mirroring": None})      # re-sent after a reset
        self.assertEqual(r.m["style_changes"], 2)

    def test_session_on_style(self):
        import app
        s = types.SimpleNamespace(style={}, styles=[], t_mono=0.0, rx=None)
        app.Session.on_style(s, {"type": "style", "energy": 0.2})                    # learned / rules mode: kept, harmless
        self.assertEqual(s.style, {"energy": 0.2})
        r = rx()
        s.rx = r
        app.Session.on_style(s, {"type": "style", "expressiveness": 0.9, "junk": 3})
        self.assertEqual(r.style, {"expressiveness": 0.9})
        self.assertEqual(s.style, {"energy": 0.2, "expressiveness": 0.9})
        self.assertEqual(len(s.styles), 2)
        app.Session.on_style(s, {"type": "style"})                                   # nothing to change
        self.assertEqual(len(s.styles), 2)
        with open(os.path.join(HERE, "..", "app", "app.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn('elif t == "style":', src)
        self.assertIn("self.rx.set_style(self.style)", src)                           # applied when unified starts later

    def test_worker_engine_set_style(self):
        import unified_worker as UW
        import collections

        class P1:
            def __init__(self):
                self.got = []

            def set_style(self, st):
                self.got.append(st)
        e = types.SimpleNamespace(style={}, planner=P1(), stats=collections.Counter())
        UW.Engine.set_style(e, {"energy": 0.7, "mirroring": None})
        UW.Engine.set_style(e, {"expressiveness": 5})
        self.assertEqual(e.planner.got[-1], {"energy": 0.7, "mirroring": None, "expressiveness": 1.0})
        self.assertEqual(e.stats["style_applied"], 2)
        e2 = types.SimpleNamespace(style={}, planner=object(), stats=collections.Counter())
        UW.Engine.set_style(e2, {"energy": 0.1})                                      # a planner without set_style yet
        self.assertEqual(e2.style, {"energy": 0.1})
        self.assertEqual(e2.stats["style_unsupported"], 1)
        with open(os.path.join(HERE, "..", "app", "unified_worker.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn('if "style" in hdr:', src)


class WordsLibrary(unittest.TestCase):
    """voice: the server library's second try on his own words (the page sends them with the plan request)"""

    def test_words_second_try(self):
        import planner

        async def run(desc, words):
            P = planner.Planner({"model": "sol", "fallback": None}, lambda *a: None, cache_dir=None)

            async def no_model(*a, **k):                       # never reach a model in this test
                raise planner.PlanError("no model in tests")
                yield ""
            P.stream_fn = no_model
            evs = []
            async for ev in P.plan("muse", desc, None, words=words):
                evs.append(ev)
            return evs[-1]
        long = "Do a backward somersault now, one full rotation backward, landing softly on both feet"
        ev = asyncio.run(run(long, "Do a backward somersault."))
        self.assertEqual(ev["source"], "library")
        self.assertEqual(ev["engine"][0]["name"], "backflip")
        self.assertTrue(ev["library"].startswith("user's words"))
        ev2 = asyncio.run(run(long, None))                     # without the user's words: not the library
        self.assertNotEqual(ev2["source"], "library")
        ev3 = asyncio.run(run("jump once", "uh I said something unrelated about the weather"))
        self.assertEqual(ev3["source"], "library")             # the description first, as before
        with open(os.path.join(HERE, "..", "app", "app.py"), encoding="utf-8") as f:
            self.assertIn('words = str(m.get("words") or "")', f.read())


if __name__ == "__main__":
    unittest.main(verbosity=1)
