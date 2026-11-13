"""planner route "claude_door" (app/planner.py): presets / route resolution, streaming poses through a fake
claude_door module (the real door's event shapes, no claude process), the same parser / hedge / cache / timeout
semantics as the HTTP routes, cancel and timeout closing the door stream, usd 0 with the API-equivalent cost recorded,
prewarm. No paid calls, no subscription calls.
    python3 tests/planner_door_test.py"""
import asyncio
import os
import sys
import tempfile
import time
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
os.environ["AVATAR_HOME"] = tempfile.mkdtemp(prefix="avatar-door-home-")
for _k in ("AVATAR_PLANNER_MODEL", "AVATAR_PLANNER_FALLBACK", "AVATAR_PLANNER_FALLBACK_AFTER_S"):
    os.environ.pop(_k, None)
import planner  # noqa: E402

NOLIB = {"library": False, "cache": False}


def answer(tag, n=4):
    return (['{"summary":"d%s","poses":[' % tag]
            + ['{"t":%s,"smile":%s,"head_yaw":%s},' % (0.3 + 0.5 * i, tag, i / 10) for i in range(n - 1)]
            + ['{"t":%s,"smile":%s}]}' % (0.3 + 0.5 * (n - 1), tag)])


class FakeDoor:
    """the S1 interface: stream(model, system, user, images, max_s, tag) -> {"delta"} ... {"done", ...}"""

    def __init__(self, script=None, effort=None):
        self.script = script or {}            # model -> list of (delay_s, text | "FAIL:<err>")
        self.effort = effort
        self.calls, self.prewarms, self.closed, self.killed = [], [], 0, 0

    def prewarm(self, model, system, n=2):
        self.prewarms.append((model, len(system), n, self.effort))

    async def stream(self, model="haiku", system="", user="", images=None, max_s=20.0, tag="door"):
        self.calls.append({"model": model, "system": system, "user": user, "max_s": max_s, "tag": tag,
                           "effort": self.effort})
        t0 = time.monotonic()
        text, err = [], None
        try:
            for delay, x in self.script.get(model, []):
                if time.monotonic() + delay - t0 > max_s:
                    await asyncio.sleep(max(0.0, max_s - (time.monotonic() - t0)))
                    err = "timeout after %.1fs" % max_s
                    break
                await asyncio.sleep(delay)
                if x.startswith("FAIL:"):
                    err = x[5:]
                    break
                text.append(x)
                yield {"delta": x}
        finally:
            self.killed += 1                   # the real door kills its process here, always
        yield {"done": True, "ok": err is None, "text": "".join(text), "ms_first": 1, "ms_done": 2, "error": err,
               "model": model, "tag": tag, "warm": True, "usage": {"input_tokens": 2, "output_tokens": 300},
               "cost_usd": 0.0042}


def install(door, cls_doors=None):
    """a fake claude_door module: door = the singleton; ClaudeDoor() = new FakeDoors sharing the script"""
    mod = types.ModuleType("claude_door")
    mod.door = door
    made = cls_doors if cls_doors is not None else []

    def ClaudeDoor():
        d = FakeDoor(door.script)
        made.append(d)
        return d
    mod.ClaudeDoor = ClaudeDoor
    sys.modules["claude_door"] = mod
    planner._DOORS.clear()
    return made


def steps(tag, first_at, every=0.02, n=4):
    t = answer(tag, n)
    return [(first_at, t[0] + t[1])] + [(every, x) for x in t[2:]]


async def collect(pl, desc="do a curtsy", character="muse"):
    evs = []
    async for ev in pl.plan(character, desc, None, http=object()):
        evs.append(ev)
    return evs


def smiles(evs):
    return {p["smile"] for e in evs if not e["done"] for p in (planner.wire_pose(x) for x in e["poses"]) if "smile" in p}


HAIKU = "claude-haiku-5-5"


class Resolve(unittest.TestCase):
    def test_presets(self):
        for k, m in (("haiku55door", HAIKU), ("sonnet55door", "claude-sonnet-5-5"), ("opus55door", "claude-opus-5-5")):
            self.assertEqual(planner.resolve({"model": k}), {"model": m, "route": "claude_door", "effort": "low"})

    def test_raw_ids_and_aliases(self):
        self.assertEqual(planner.resolve({"model": "claude-opus-5-5"})["route"], "claude_door")
        self.assertEqual(planner.resolve({"model": "sonnet"}),
                         {"model": "claude-sonnet-5-5", "route": "claude_door", "effort": None})
        # an OpenRouter Anthropic id stays on OpenRouter unless the route says door
        self.assertEqual(planner.resolve({"model": "anthropic/claude-haiku-5.5"})["route"], "openrouter")
        self.assertEqual(planner.resolve({"model": "anthropic/claude-haiku-5.5", "route": "claude_door"})["model"], HAIKU)

    def test_fallback_block(self):
        f = planner.resolve_fallback({"model": "haiku55door", "fallback": {"model": "gem35lite", "after_s": 3.0}})
        self.assertEqual(f, {"model": "google/gemini-3.5-flash-lite", "route": "openrouter", "effort": None,
                             "after_s": 3.0})

    def test_recommended_block(self):
        # planner: the recommended config.json "planner" block (the sealed-eval winner), spelled out in full
        blk = {"model": "claude-sonnet-5-5", "route": "claude_door", "effort": "none",
               "fallback": {"model": "google/gemini-3.5-flash-lite", "route": "openrouter", "after_s": 4.5}}
        self.assertEqual(planner.resolve(blk), {"model": "claude-sonnet-5-5", "route": "claude_door", "effort": "none"})
        self.assertEqual(planner.resolve_fallback(blk), {"model": "google/gemini-3.5-flash-lite", "route": "openrouter",
                                                         "effort": None, "after_s": 4.5})
        self.assertEqual(planner.resolve({"model": "sonnet55door_nt"}), planner.resolve(blk))   # == the preset

    def test_usd_zero(self):
        self.assertEqual(planner.usd_of(HAIKU, {"input_tokens": 9, "output_tokens": 9}, "s", "u", "t", "claude_door"),
                         (0.0, False))


class Stream(unittest.TestCase):
    def test_poses_stream_through_parser(self):
        d = FakeDoor({HAIKU: steps(0.7, 0.05)}, effort="low")      # the singleton already runs at low
        install(d)
        pl = planner.Planner({**NOLIB, "model": "haiku55door", "fallback": None}, lambda m: None)
        evs = asyncio.run(collect(pl))
        done = evs[-1]
        self.assertTrue(done["ok"], done)
        self.assertEqual((done["model"], done["route"]), (HAIKU, "claude_door"))
        self.assertEqual(len(done["all_poses"]), 4)
        self.assertGreaterEqual(sum(1 for e in evs if not e["done"]), 2)        # poses streamed in batches, not at end
        self.assertEqual(done["usd"], 0.0)
        a = done["attempts"][0]
        self.assertEqual(a["door_usd"], 0.0042)
        self.assertTrue(a["door"]["warm"])
        c = d.calls[0]
        self.assertEqual(c["tag"], "planner")
        self.assertEqual(c["system"], pl.prompts["muse"]["text"])               # the character prompt = the warm key
        self.assertIn("do a curtsy", c["user"])
        self.assertTrue(20 <= c["max_s"] <= 25.0)                                 # the plan's own deadline
        self.assertEqual(d.killed, 1)

    def test_effort_gets_its_own_door(self):
        d = FakeDoor({HAIKU: steps(0.5, 0.01)}, effort=None)
        made = install(d)
        pl = planner.Planner({**NOLIB, "model": "haiku55door", "fallback": None}, lambda m: None)
        asyncio.run(collect(pl))
        self.assertEqual(d.calls, [])                                             # the singleton has no effort
        self.assertEqual(len(made), 1)
        self.assertEqual(made[0].effort, "low")
        self.assertEqual(len(made[0].calls), 1)
        # effort None -> the singleton
        install(d)
        pl2 = planner.Planner({**NOLIB, "model": "haiku55door", "effort": None, "fallback": None}, lambda m: None)
        asyncio.run(collect(pl2))
        self.assertEqual(len(d.calls), 1)

    def test_door_error_falls_back_to_hedge(self):
        d = FakeDoor({HAIKU: [(0.05, "FAIL:claude exited rc=1: rate limit")]})
        install(d)
        pl = planner.Planner({**NOLIB, "model": "haiku55door", "effort": None,
                              "fallback": {"model": "dsv41flash", "after_s": 3.0}}, lambda m: None)

        async def fb_stream(http, system, user, deadline, info):
            if info["spec"]["route"] == "claude_door":
                async for x in planner.door_stream(info["spec"], system, user, deadline, info):
                    yield x
                return
            for delay, x in steps(0.9, 0.05):
                await asyncio.sleep(delay)
                yield x
        pl.stream_fn = fb_stream
        t0 = time.monotonic()
        evs = asyncio.run(collect(pl))
        done = evs[-1]
        self.assertTrue(done["ok"] and done["fallback_used"], done)
        self.assertEqual(smiles(evs), {0.9})
        self.assertLess(time.monotonic() - t0, 1.5)                              # failed first -> hedge at once
        self.assertIn("rate limit", done["attempts"][0]["error"])

    def test_slow_door_hedged_after_s_and_cancelled(self):
        d = FakeDoor({HAIKU: steps(0.7, 0.6)})
        install(d)
        pl = planner.Planner({**NOLIB, "model": "haiku55door", "effort": None,
                              "fallback": {"model": "dsv41flash", "after_s": 0.15}}, lambda m: None)

        async def mixed(http, system, user, deadline, info):
            if info["spec"]["route"] == "claude_door":
                async for x in planner.door_stream(info["spec"], system, user, deadline, info):
                    yield x
                return
            for delay, x in steps(0.9, 0.05):
                await asyncio.sleep(delay)
                yield x
        pl.stream_fn = mixed
        evs = asyncio.run(collect(pl))
        done = evs[-1]
        self.assertTrue(done["fallback_used"])
        self.assertEqual(smiles(evs), {0.9})
        self.assertTrue(done["attempts"][0]["cancelled"])
        self.assertEqual(d.killed, 1)                                              # the cancel closed the door stream

    def test_timeout(self):
        d = FakeDoor({HAIKU: steps(0.7, 5.0)})
        install(d)
        pl = planner.Planner({**NOLIB, "model": "haiku55door", "effort": None, "fallback": None, "timeout_s": 0.4,
                              "retry_min_s": 99}, lambda m: None)
        t0 = time.monotonic()
        evs = asyncio.run(collect(pl))
        self.assertFalse(evs[-1]["ok"])
        self.assertLess(time.monotonic() - t0, 1.5)
        self.assertEqual(d.killed, 1)

    def test_no_door_module(self):
        sys.modules["claude_door"] = None          # import fails -> PlanError, not a crash
        planner._DOORS.clear()
        pl = planner.Planner({**NOLIB, "model": "haiku55door", "fallback": None, "retry_min_s": 99}, lambda m: None)
        evs = asyncio.run(collect(pl))
        self.assertFalse(evs[-1]["ok"])
        self.assertIn("claude_door unavailable", evs[-1]["error"])
        del sys.modules["claude_door"]

    def test_cache_written_and_served(self):
        d = FakeDoor({HAIKU: steps(0.7, 0.01)})
        install(d)
        cdir = tempfile.mkdtemp(prefix="avatar-door-cache-")
        cfg = {"library": False, "model": "haiku55door", "fallback": None}
        pl = planner.Planner(cfg, lambda m: None, cache_dir=cdir)
        asyncio.run(collect(pl))
        evs = asyncio.run(collect(pl))
        self.assertEqual(evs[-1]["source"], "cache")
        self.assertEqual(len(d.calls) + len(planner._DOORS.get("low").calls if planner._DOORS.get("low") else []), 1)


class NoThink(unittest.TestCase):
    def test_effort_none_is_low_without_thinking(self):
        d = FakeDoor({HAIKU: steps(0.5, 0.01)}, effort="low")
        made = install(d)
        pl = planner.Planner({**NOLIB, "model": "haiku55door_nt", "fallback": None}, lambda m: None)
        self.assertEqual(planner.resolve({"model": "haiku55door_nt"})["effort"], "none")
        evs = asyncio.run(collect(pl))
        self.assertTrue(evs[-1]["ok"])
        self.assertEqual(d.calls, [])                      # never the shared door (it thinks)
        self.assertEqual((made[0].effort, getattr(made[0], "thinking", None)), ("low", False))

    def test_real_door_env(self):
        import importlib
        sys.modules.pop("claude_door", None)
        cd = importlib.import_module("claude_door")
        planner._DOORS.clear()
        dd = planner.door_for("none")
        self.assertIsInstance(dd, cd.ClaudeDoor)
        self.assertFalse(dd.thinking)
        self.assertTrue(cd.door.thinking)
        self.assertIs(planner.door_for("low"), cd.door)
        sys.modules.pop("claude_door", None)
        planner._DOORS.clear()


class Prewarm(unittest.TestCase):
    def test_prewarm_door_route_only(self):
        d = FakeDoor({})
        made = install(d)
        pl = planner.Planner({"model": "haiku55door", "fallback": None}, lambda m: None)
        self.assertEqual(len(pl.prewarm("muse")), 1)
        self.assertEqual(made[0].prewarms, [(HAIKU, len(pl.prompts["muse"]["text"]), 2, "low")])
        self.assertEqual(pl.prewarm("nobody"), [])
        pl2 = planner.Planner({"model": "gem35lite"}, lambda m: None)              # HTTP routes: nothing to warm
        self.assertEqual(pl2.prewarm("muse"), [])
        pl3 = planner.Planner(False, lambda m: None)
        self.assertEqual(pl3.prewarm("muse"), [])

    def test_prewarm_fallback_on_door(self):
        d = FakeDoor({})
        install(d)
        pl = planner.Planner({"model": "gem35lite", "fallback": {"model": "sonnet", "after_s": 2}}, lambda m: None)
        self.assertEqual([s["model"] for s in pl.prewarm("fox")], ["claude-sonnet-5-5"])
        self.assertEqual(d.prewarms[0][0], "claude-sonnet-5-5")


if __name__ == "__main__":
    unittest.main(verbosity=1)
