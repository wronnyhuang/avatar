"""the no-think hint on the claude_door user turn, the door's thinking event, and the fallback started at
the primary's thinking block (cfg fallback_on_thinking). Fake door, no claude process, no paid calls.
    python tests/planner_planlat_test.py"""
import asyncio
import os
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import planner_door_test as pdt  # noqa: E402  (sets AVATAR_HOME, imports planner)
planner = pdt.planner
SONNET = "claude-sonnet-5-5"
os.environ.pop("AVATAR_PLANNER_NO_THINK", None)


class ThinkingDoor(pdt.FakeDoor):
    """a script item "THINK" = the real door's {"thinking": True, "ms"} event (a hidden thinking block opened)"""

    async def stream(self, model="haiku", system="", user="", images=None, max_s=20.0, tag="door"):
        self.calls.append({"model": model, "system": system, "user": user})
        t0 = time.monotonic()
        text = []
        try:
            for delay, x in self.script.get(model, []):
                await asyncio.sleep(delay)
                if x == "THINK":
                    yield {"thinking": True, "ms": int((time.monotonic() - t0) * 1000)}
                    continue
                text.append(x)
                yield {"delta": x}
        finally:
            self.killed += 1
        yield {"done": True, "ok": True, "text": "".join(text), "ms_first": 1, "ms_done": 2, "error": None,
               "model": model, "tag": tag, "warm": True, "usage": {"input_tokens": 2, "output_tokens": 300},
               "cost_usd": 0.0, "ms_thinking": 5}


def install(door):
    """pdt.install, but the per-effort private doors (effort "none") are ThinkingDoors too"""
    made = pdt.install(door)
    sys.modules["claude_door"].ClaudeDoor = lambda: made.append(ThinkingDoor(door.script)) or made[-1]
    sys.modules["claude_door"].make_door = sys.modules["claude_door"].ClaudeDoor
    return made


def mixed_fn(fb_first_at):
    async def mixed(http, system, user, deadline, info):
        if info["spec"]["route"] == "claude_door":
            async for x in planner.door_stream(info["spec"], system, user, deadline, info):
                yield x
            return
        for delay, x in pdt.steps(0.9, fb_first_at):
            await asyncio.sleep(delay)
            yield x
    return mixed


def cfg(**kw):
    return {**pdt.NOLIB, "model": "sonnet55door_nt", "fallback": {"model": "gem35lite", "after_s": 2.0}, **kw}


class Hint(unittest.TestCase):
    def test_user_turn(self):
        u = planner.USER_TMPL.format(text="hop") + "\nnote"
        h = planner.no_think_user(u)
        self.assertIn("Answer with the JSON object. " + planner.NO_THINK_HINT, h)
        self.assertTrue(h.endswith("\nnote"))
        self.assertEqual(planner.no_think_user(h), h)            # idempotent

    def test_door_gets_hint_default_on(self):
        d = ThinkingDoor({SONNET: pdt.steps(0.7, 0.05)})
        install(d)
        pl = planner.Planner(cfg(fallback=None), lambda m: None)
        done = asyncio.run(pdt.collect(pl))[-1]
        self.assertTrue(done["ok"], done)
        self.assertTrue(any(planner.NO_THINK_HINT in c["user"] for c in d.calls + sum((x.calls for x in planner._DOORS.values()), [])))

    def test_switch_off(self):
        d = ThinkingDoor({SONNET: pdt.steps(0.7, 0.05)})
        install(d)
        pl = planner.Planner(cfg(fallback=None, no_think=False), lambda m: None)
        asyncio.run(pdt.collect(pl))
        calls = d.calls + sum((x.calls for x in planner._DOORS.values()), [])
        self.assertTrue(calls)
        self.assertFalse(any(planner.NO_THINK_HINT in c["user"] for c in calls))

    def test_env_off(self):
        os.environ["AVATAR_PLANNER_NO_THINK"] = "0"
        try:
            self.assertFalse(planner.no_think_on({"no_think": True}))
        finally:
            os.environ.pop("AVATAR_PLANNER_NO_THINK")
        self.assertTrue(planner.no_think_on({}))

    def test_other_routes_untouched(self):
        seen = []

        async def fn(http, system, user, deadline, info):
            seen.append(user)
            for delay, x in pdt.steps(0.5, 0.01):
                await asyncio.sleep(delay)
                yield x
        pl = planner.Planner({**pdt.NOLIB, "model": "gem35lite", "fallback": None}, lambda m: None)
        pl.stream_fn = fn
        asyncio.run(pdt.collect(pl))
        self.assertTrue(seen and all(planner.NO_THINK_HINT not in u for u in seen))


class ThinkingFallback(unittest.TestCase):
    def run_plan(self, on, think_at=0.05, claude_first=1.0, fb_first=0.1):
        script = [(think_at, "THINK")] + pdt.steps(0.7, claude_first)
        d = ThinkingDoor({SONNET: script})
        install(d)
        logs = []
        pl = planner.Planner(cfg(fallback_on_thinking=on), logs.append)
        pl.stream_fn = mixed_fn(fb_first)
        t0 = time.monotonic()
        evs = asyncio.run(pdt.collect(pl))
        return evs, logs, time.monotonic() - t0

    def test_on_starts_fallback_at_thinking(self):
        evs, logs, _ = self.run_plan(True)
        done = evs[-1]
        self.assertTrue(done["fallback_used"], done)
        self.assertEqual(pdt.smiles(evs), {0.9})
        fb = [a for a in done["attempts"] if a["role"] == "fallback"][0]
        self.assertLess(fb["ms_start"], 500)                     # at the thinking event, not after_s (2.0 s)
        self.assertTrue(any("is thinking" in m for m in logs))

    def test_off_waits(self):
        evs, logs, _ = self.run_plan(False)
        done = evs[-1]
        self.assertFalse(done["fallback_used"], done)             # Claude's pose at ~1.05 s < after_s 2.0 s
        self.assertEqual(pdt.smiles(evs), {0.7})
        self.assertFalse(any("is thinking" in m for m in logs))

    def test_claude_still_wins_if_first(self):
        evs, _, _ = self.run_plan(True, claude_first=0.1, fb_first=1.5)
        done = evs[-1]
        self.assertFalse(done["fallback_used"], done)
        self.assertEqual(pdt.smiles(evs), {0.7})


class Hedge(unittest.TestCase):
    """hedge_after_s: a stalled primary (no text) gets a second request to the same model; first pose wins"""

    class StallDoor(ThinkingDoor):
        n = 0

        async def stream(self, model="haiku", system="", user="", images=None, max_s=20.0, tag="door"):
            type(self).n += 1
            first = 1.5 if type(self).n == 1 else 0.05          # the first request stalls, the hedge does not
            self.script = {SONNET: pdt.steps(0.7, first)}
            async for ev in ThinkingDoor.stream(self, model, system, user, images, max_s, tag):
                yield ev

    def plan(self, **kw):
        Hedge.StallDoor.n = 0
        d = Hedge.StallDoor({})
        made = pdt.install(d)
        sys.modules["claude_door"].ClaudeDoor = lambda: made.append(Hedge.StallDoor({})) or made[-1]
        sys.modules["claude_door"].make_door = sys.modules["claude_door"].ClaudeDoor
        logs = []
        pl = planner.Planner(cfg(fallback=None, **kw), logs.append)
        t0 = time.monotonic()
        evs = asyncio.run(pdt.collect(pl))
        return evs, logs, time.monotonic() - t0

    def test_hedge_wins_stall(self):
        evs, logs, dt = self.plan(hedge_after_s=0.3)
        done = evs[-1]
        self.assertTrue(done["ok"], done)
        self.assertEqual([a["role"] for a in done["attempts"]], ["primary", "hedge"])
        self.assertLess(done["ms_first_pose"], 1000)
        self.assertTrue(done["attempts"][0]["cancelled"])
        self.assertFalse(done["fallback_used"])
        self.assertTrue(any("planner: hedge" in m for m in logs))

    def test_off_by_default(self):
        evs, logs, _ = self.plan()
        done = evs[-1]
        self.assertEqual([a["role"] for a in done["attempts"]], ["primary"])
        self.assertGreater(done["ms_first_pose"], 1400)


class Retry(unittest.TestCase):
    """no_think "retry": the plain request goes first; when it thinks, a hinted second request races it"""

    class HintDoor(ThinkingDoor):
        async def stream(self, model="haiku", system="", user="", images=None, max_s=20.0, tag="door"):
            hinted = planner.NO_THINK_HINT in user
            self.script = {SONNET: pdt.steps(0.8, 0.05) if hinted else [(0.05, "THINK")] + pdt.steps(0.7, 1.5)}
            async for ev in ThinkingDoor.stream(self, model, system, user, images, max_s, tag):
                yield ev

    def plan(self, mode, think=True):
        d = Retry.HintDoor({})
        made = pdt.install(d)
        sys.modules["claude_door"].ClaudeDoor = lambda: made.append(Retry.HintDoor({})) or made[-1]
        sys.modules["claude_door"].make_door = sys.modules["claude_door"].ClaudeDoor
        logs = []
        pl = planner.Planner(cfg(fallback=None, no_think=mode), logs.append)
        return asyncio.run(pdt.collect(pl)), logs

    def test_retry_races_hinted(self):
        evs, logs = self.plan("retry")
        done = evs[-1]
        self.assertEqual([a["role"] for a in done["attempts"]], ["primary", "nothink"])
        self.assertEqual(pdt.smiles(evs), {0.8})
        self.assertLess(done["ms_first_pose"], 600)
        self.assertTrue(done["attempts"][0]["cancelled"])
        self.assertTrue(any("no-think hint" in m for m in logs))

    def test_always_hints_first_request(self):
        evs, _ = self.plan(True)
        done = evs[-1]
        self.assertEqual([a["role"] for a in done["attempts"]], ["primary"])
        self.assertEqual(pdt.smiles(evs), {0.8})

    def test_off_waits_for_thinking(self):
        evs, _ = self.plan(False)
        done = evs[-1]
        self.assertEqual([a["role"] for a in done["attempts"]], ["primary"])
        self.assertEqual(pdt.smiles(evs), {0.7})
        self.assertGreater(done["ms_first_pose"], 1400)

    def test_modes(self):
        self.assertEqual(planner.no_think_mode({"no_think": "retry"}), "retry")
        self.assertEqual(planner.no_think_mode({"no_think": True}), "always")
        self.assertEqual(planner.no_think_mode({"no_think": False}), "off")
        self.assertFalse(planner.no_think_on({"no_think": "retry"}))


if __name__ == "__main__":
    unittest.main()
