"""the semantic-API planner path (app/planner.py plan(..., planner='semantic'), run_plan mode='semantic',
app.py on_plan / on_semwarm) against the in-tree FAKE semantic package (tests/sem_hub_fake) and a fake claude_door
module (the real door's event shapes, no claude process). No paid calls, no subscription calls.
    .venv/bin/python tests/test_sem_hub.py
(the real core end to end: tests/sem_hub_smoke.py against a running server; latency: tests/sem_hub_latency.py)"""
import asyncio
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
os.environ["AVATAR_HOME"] = tempfile.mkdtemp(prefix="semhub-home-")
for _k in ("AVATAR_PLANNER_MODEL", "AVATAR_PLANNER_FALLBACK", "AVATAR_PLANNER_FALLBACK_AFTER_S", "AVATAR_SEM_MODEL",
           "AVATAR_SEM_EFFORT"):
    os.environ.pop(_k, None)
os.environ["AVATAR_SEM_ROUTE"] = "claude_door"   # these tests drive the fake door; AnthropicRoute below the API route
import planner  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "sem_hub_fake"))
import semantic as fakesem  # noqa: E402
sys.path.pop(0)
planner.set_sem_module(fakesem)


def program(*stmts):
    """a streamed answer: fence, one statement per chunk (each chunk = a line), closing fence"""
    return ["```python\n"] + [s + "\n" for s in stmts] + ["```\n"]


class FakeDoor:
    """the claude_door interface; scripts[model] = list of answers, one per call (an answer = list of chunks or
    'FAIL:<err>' items); delay = seconds between chunks"""

    def __init__(self, scripts=None, delay=0.01, effort=None):
        self.scripts = scripts if scripts is not None else {}
        self.delay = delay
        self.effort = effort
        self.calls, self.prewarms, self.killed = [], [], 0
        self.thinking = True

    def prewarm(self, model, system, n=2, reserve=False):
        self.prewarms.append((model, system, n))
        self.warm_keys = getattr(self, "warm_keys", set()) | {(model, system)}

    async def stream(self, model="haiku", system="", user="", images=None, max_s=20.0, tag="door"):
        self.calls.append({"model": model, "system": system, "user": user, "max_s": max_s, "tag": tag})
        lst = self.scripts.get(model) or [[]]
        answer = lst.pop(0) if len(lst) > 1 else lst[0]
        t0 = time.monotonic()
        text, err = [], None
        try:
            for x in answer:
                if time.monotonic() + self.delay - t0 > max_s:
                    await asyncio.sleep(max(0.0, max_s - (time.monotonic() - t0)))
                    err = "timeout after %.1fs" % max_s
                    break
                await asyncio.sleep(self.delay)
                if x.startswith("FAIL:"):
                    err = x[5:]
                    break
                text.append(x)
                yield {"delta": x}
        finally:
            self.killed += 1
        yield {"done": True, "ok": err is None, "text": "".join(text), "ms_first": 1, "ms_done": 2, "error": err,
               "model": model, "tag": tag, "warm": (model, system) in getattr(self, "warm_keys", set()), "usage": {"input_tokens": 2, "output_tokens": 30},
               "cost_usd": 0.001}


def install(door):
    mod = types.ModuleType("claude_door")
    mod.door = door
    made = []

    def ClaudeDoor():
        d = FakeDoor(door.scripts, door.delay)
        made.append(d)
        return d
    mod.ClaudeDoor = ClaudeDoor
    mod.make_door = ClaudeDoor                       # this tree's door_for builds private doors with make_door()
    sys.modules["claude_door"] = mod
    planner._DOORS.clear()
    return made


HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"


def mkplanner(sem=None, cache=True, library=False):
    d = tempfile.mkdtemp(prefix="semhub-cache-")
    pl = planner.Planner({"library": library,
                          "semantic": dict({"model": HAIKU, "effort": "none", "cache": cache}, **(sem or {}))},
                         lambda m: None, cache_dir=os.path.join(d, "plans"))
    return pl, d


def run_sem(pl, desc, character="muse", state=None, words=None):
    sent, recs = [], []

    async def go():
        return await planner.run_plan(pl, "p1", character, desc, state, sent.append, http=object(),
                                      record=recs.append, words=words, mode="semantic", t_recv=time.monotonic())
    rec = asyncio.run(go())
    return sent, rec


ANS = program("look_at('camera', dur=0.5)", "nod(dur=0.4)", "smile(dur=0.3)")


class Shapes(unittest.TestCase):
    def setUp(self):
        planner.set_sem_module(fakesem)
        self.door = FakeDoor({HAIKU: [ANS]})
        self.made = install(self.door)

    def test_messages(self):
        pl, d = mkplanner()
        sent, rec = run_sem(pl, "look at me and nod happily")
        kf = [m for m in sent if m["type"] == "kf"]
        sm = [m for m in sent if m["type"] == "sem"]
        self.assertGreaterEqual(len(kf), 3)
        for m in kf[:-1]:
            self.assertEqual(list(m), ["type", "id", "space", "hz", "poses", "done"])
            self.assertEqual((m["space"], m["hz"], m["done"]), ("rig", 20, False))
            self.assertTrue(m["poses"] and all(list(p)[0] == "t" for p in m["poses"]))
        last = kf[-1]
        self.assertEqual(list(last)[:6], ["type", "id", "space", "hz", "poses", "done"])
        for k in ("ok", "source", "planner", "engine", "summary", "ms_first_pose", "ms_done", "usd", "channels"):
            self.assertIn(k, last)
        self.assertEqual((last["ok"], last["source"], last["planner"], last["usd"], last["poses"]),
                         (True, "planner", "semantic", 0.0, []))
        ts = [p["t"] for m in kf for p in m["poses"]]
        self.assertEqual(ts, sorted(ts))
        self.assertEqual(len(ts), len(set(ts)))
        self.assertEqual([m["phase"] for m in sm][0], "start")
        self.assertEqual([m["phase"] for m in sm][-1], "done")
        self.assertIn("stmt", [m["phase"] for m in sm])
        done = sm[-1]
        for k in ("program", "source", "model", "cached", "ms", "exec_ms", "warnings", "errors", "targets"):
            self.assertIn(k, done)
        self.assertEqual(set(done["ms"]), {"recv", "first_token", "first_stmt", "first_pose", "done"})
        self.assertTrue(done["ms"]["first_token"] <= done["ms"]["first_stmt"] <= done["ms"]["first_pose"]
                        <= done["ms"]["done"])
        self.assertIn("nod(dur=0.4)", done["program"])
        self.assertEqual(done["model"], HAIKU)
        self.assertIn("user", done["targets"])
        # the order: every sem 'stmt' follows its kf batch; the kf done precedes the sem done
        self.assertEqual(sent[-1]["type"], "sem")
        self.assertEqual(sent[-2]["type"], "kf")
        self.assertTrue(sent[-2]["done"])
        json.dumps(sent)
        # the record
        for k in ("planner", "program", "ms", "model", "n_statements", "warnings", "errors", "ok", "source"):
            self.assertIn(k, rec)
        self.assertEqual(rec["planner"], "semantic")
        self.assertEqual(rec["n_statements"], 3)
        # the door call: the semantic tag, the fake's system prompt, the user prompt, the effort-none door
        self.assertEqual(len(self.made), 1)
        call = self.made[0].calls[0]
        self.assertEqual(call["tag"], "semantic")
        self.assertTrue(call["system"].startswith("FAKE SEMANTIC SYSTEM for muse"))
        self.assertIn("look at me and nod happily", call["user"])
        shutil.rmtree(d, ignore_errors=True)

    def test_library_hit(self):
        pl, d = mkplanner(library=True)
        sent, rec = run_sem(pl, "wave", character="fox")
        kf = [m for m in sent if m["type"] == "kf"]
        self.assertEqual(len(kf), 1)
        today = planner.kf_message("p1", {"done": True, "ok": True, "source": "library", "poses": [],
                                          "engine": kf[0]["engine"], "summary": "", "ms_first_pose": None,
                                          "ms_done": kf[0]["ms_done"], "usd": 0.0})
        self.assertEqual(kf[0], dict(today, planner="semantic"))
        self.assertEqual([m["phase"] for m in sent if m["type"] == "sem"], ["done"])
        self.assertEqual(self.made, [])                     # no door call
        shutil.rmtree(d, ignore_errors=True)

    def test_unavailable_character(self):
        pl, d = mkplanner(library=True)
        sent, rec = run_sem(pl, "wave", character="orb")
        self.assertEqual(len(sent), 1)
        self.assertTrue(sent[0]["sem_unavailable"])
        self.assertEqual(sent[0]["source"], "library")
        self.assertNotIn("space", sent[0])
        self.assertNotIn("planner", sent[0])
        self.assertTrue(rec["sem_unavailable"])
        # a planner-path request: today's planner (its own stream fn), never the door
        pl.cfg.update({"library": False, "cache": False})
        calls = []

        async def fn(http, system, user, deadline, info):
            calls.append(user)
            for x in ['{"summary":"x","poses":[', '{"t":0.3,"smile":0.5},', '{"t":0.9,"smile":0}]}']:
                yield x
        pl.stream_fn = fn
        sent, rec = run_sem(pl, "do something odd", character="orb")
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.made, [])
        self.assertTrue(sent[-1]["done"] and sent[-1]["sem_unavailable"] and sent[-1]["ok"])
        self.assertTrue(all(list(m)[:4] == ["type", "id", "poses", "done"] for m in sent))
        shutil.rmtree(d, ignore_errors=True)

    def test_not_installed(self):
        planner.set_sem_module(None)
        try:
            pl, d = mkplanner()
            sent, rec = run_sem(pl, "do a little dance")
            self.assertEqual(len(sent), 1)
            self.assertEqual((sent[0]["ok"], sent[0]["error"], sent[0]["planner"]),
                             (False, "semantic planner not installed", "semantic"))
            self.assertEqual(pl.prewarm_semantic("muse"), [])
        finally:
            planner.set_sem_module(fakesem)

    def test_non_door_model_refused(self):
        pl, d = mkplanner({"model": "gem35lite"})
        sent, rec = run_sem(pl, "do a little dance")
        self.assertFalse(sent[-1]["ok"])
        self.assertIn("Claude model", sent[-1]["error"])
        self.assertEqual(self.made, [])


class Cache(unittest.TestCase):
    def setUp(self):
        planner.set_sem_module(fakesem)
        self.door = FakeDoor({HAIKU: [program("look_at('user', dur=0.5)", "nod(dur=0.4)")]})
        self.made = install(self.door)

    def test_namespace_and_hit(self):
        pl, d = mkplanner()
        st = {"sem": {"user": {"present": True, "ax": 0.2, "ay": 0.0}}}
        sent, rec = run_sem(pl, "look at me and nod", state=st)
        self.assertTrue(rec["ok"])
        self.assertEqual(os.listdir(os.path.join(d)), ["semplans"])    # never the keyframe planner's cache/plans
        self.assertEqual(len(os.listdir(os.path.join(d, "semplans"))), 1)
        first = [p for m in sent if m["type"] == "kf" for p in m["poses"]]
        # a repeat: the cache, every pose at once, no door call, fast
        t = time.monotonic()
        sent2, rec2 = run_sem(pl, "Look at me and nod please", state=st)
        dt = (time.monotonic() - t) * 1000
        kf2 = [m for m in sent2 if m["type"] == "kf"]
        self.assertEqual(len(kf2), 1)
        self.assertEqual((kf2[0]["source"], kf2[0]["done"], kf2[0]["space"]), ("cache", True, "rig"))
        self.assertEqual(kf2[0]["poses"], first)
        self.assertEqual(len(self.made[0].calls), 1)
        self.assertLess(dt, 50, f"cache hit {dt:.1f} ms")
        sm = [m for m in sent2 if m["type"] == "sem"]
        self.assertTrue(all(m["cached"] for m in sm))
        print(f"  cache hit {dt:.1f} ms (asyncio.run included)")
        # the person moved: the cached program is re-run with the fresh target (not stale poses)
        st2 = {"sem": {"user": {"present": True, "ax": -0.3, "ay": 0.0}}}
        sent3, rec3 = run_sem(pl, "look at me and nod", state=st2)
        kf3 = [m for m in sent3 if m["type"] == "kf"]
        self.assertEqual(kf3[0]["source"], "cache")
        self.assertEqual(rec3.get("replayed"), ["user"])
        self.assertNotEqual(kf3[0]["poses"], first)
        self.assertEqual(len(self.made[0].calls), 1)
        # the keyframe planner never sees the semantic cache (and its own cache dir stays empty)
        pl.cfg.update({"library": False})
        calls = []

        async def fn(http, system, user, deadline, info):
            calls.append(1)
            for x in ['{"summary":"x","poses":[{"t":0.3,"smile":0.5}]}']:
                yield x
        pl.stream_fn = fn

        async def today():
            return [ev async for ev in pl.plan("muse", "look at me and nod", None, http=object())]
        evs = asyncio.run(today())
        self.assertEqual(evs[-1]["source"], "planner")
        self.assertEqual(len(calls), 1)
        self.assertEqual(sorted(os.listdir(d)), ["plans", "semplans"])
        # ...and the semantic path never serves the keyframe plan just written for another description
        evs = asyncio.run(today())
        self.assertEqual(evs[-1]["source"], "cache")
        shutil.rmtree(d, ignore_errors=True)

    def test_model_change_misses(self):
        pl, d = mkplanner()
        run_sem(pl, "look at me and nod")
        pl.cfg["semantic"]["model"] = SONNET
        self.door.scripts[SONNET] = [program("nod(dur=0.4)")]
        sent, rec = run_sem(pl, "look at me and nod")
        self.assertEqual(rec["source"], "planner")
        self.assertEqual(rec["model"], SONNET)
        shutil.rmtree(d, ignore_errors=True)

    def test_cache_off(self):
        pl, d = mkplanner(cache=False)
        run_sem(pl, "look at me and nod")
        sent, rec = run_sem(pl, "look at me and nod")
        self.assertEqual(rec["source"], "planner")
        self.assertFalse(os.path.exists(os.path.join(d, "semplans")))
        shutil.rmtree(d, ignore_errors=True)


class OffLoop(unittest.TestCase):
    def setUp(self):
        planner.set_sem_module(fakesem)

    def test_feed_off_loop(self):
        install(FakeDoor({HAIKU: [program("slow(dur=0.3)", "slow(dur=0.3)", "slow(dur=0.3)", "nod()")]}, delay=0.005))
        pl, d = mkplanner(cache=False)
        fakesem.FEED_THREADS.clear()
        gaps = []

        async def go():
            stop = False

            async def ticker():
                last = time.perf_counter()
                while not stop:
                    await asyncio.sleep(0.002)
                    now = time.perf_counter()
                    gaps.append(now - last)
                    last = now
            tk = asyncio.create_task(ticker())
            sent = []
            await planner.run_plan(pl, "p", "muse", "x", None, sent.append, mode="semantic")
            stop = True
            await tk
            return sent
        sent = asyncio.run(go())
        self.assertTrue(sent[-2]["ok"])
        main = threading.current_thread().name
        self.assertTrue(fakesem.FEED_THREADS and all(n.startswith("semantic") for n in fakesem.FEED_THREADS),
                        fakesem.FEED_THREADS)
        self.assertNotIn(main, fakesem.FEED_THREADS)
        worst = max(gaps) * 1000
        self.assertLess(worst, 60, f"event loop blocked {worst:.0f} ms (feed sleeps 200 ms per statement)")
        print(f"  feed off loop: worst loop gap {worst:.1f} ms with 3 x 200 ms statements")


class RetryCancel(unittest.TestCase):
    def setUp(self):
        planner.set_sem_module(fakesem)

    def test_retry_after_door_error(self):
        made = install(FakeDoor({HAIKU: [["FAIL:claude exited rc=1: boom"], ANS]}))
        pl, d = mkplanner(cache=False)
        sent, rec = run_sem(pl, "x")
        self.assertTrue(rec["ok"])
        self.assertTrue(rec["retried"])
        self.assertEqual([a["role"] for a in rec["attempts"]], ["primary", "retry"])
        self.assertEqual(len(made[0].calls), 2)
        self.assertEqual(made[0].calls[1]["model"], HAIKU)       # the same model, never another one

    def test_retry_after_zero_valid_statements(self):
        made = install(FakeDoor({HAIKU: [program("boom()", "boom()"), ANS]}))
        pl, d = mkplanner(cache=False)
        sent, rec = run_sem(pl, "x")
        self.assertTrue(rec["ok"] and rec["retried"])
        self.assertEqual(len(made[0].calls), 2)

    def test_one_retry_only_then_plain_error(self):
        made = install(FakeDoor({HAIKU: [["FAIL:overloaded"], ["FAIL:overloaded"], ANS]}))
        pl, d = mkplanner(cache=False)
        sent, rec = run_sem(pl, "x")
        self.assertFalse(rec["ok"])
        self.assertEqual(len(made[0].calls), 2)
        last = [m for m in sent if m["type"] == "kf"][-1]
        self.assertEqual((last["ok"], last["done"], last["planner"]), (False, True, "semantic"))
        self.assertIn("overloaded", last["error"])
        self.assertEqual([m for m in sent if m["type"] == "sem"][-1]["phase"], "done")

    def test_no_retry_when_little_time_left(self):
        made = install(FakeDoor({HAIKU: [["FAIL:overloaded"], ANS]}))
        pl, d = mkplanner({"timeout_s": 5.0}, cache=False)          # < retry_min_s 8 s remain
        sent, rec = run_sem(pl, "x")
        self.assertFalse(rec["ok"])
        self.assertEqual(len(made[0].calls), 1)

    def test_error_after_first_pose_keeps_motion(self):
        made = install(FakeDoor({HAIKU: [["```python\n", "nod(dur=0.4)\n", "smile()\n", "FAIL:claude exited rc=1"]]}))
        pl, d = mkplanner(cache=False)
        sent, rec = run_sem(pl, "x")
        self.assertTrue(rec["ok"])                    # poses are out: no retry, the error is reported
        self.assertIn("exited", rec["error"])
        self.assertEqual(len(made[0].calls), 1)
        self.assertFalse(rec["retried"])
        self.assertFalse(os.path.exists(os.path.join(d, "semplans")))   # an errored plan is not cached

    def test_timeout_closes_door_stream(self):
        made = install(FakeDoor({HAIKU: [["```python\n"] + ["nod()\n"] * 200]}, delay=0.05))
        pl, d = mkplanner({"timeout_s": 1.0}, cache=False)
        t = time.monotonic()
        sent, rec = run_sem(pl, "x")
        dt = time.monotonic() - t
        self.assertLess(dt, 2.0)
        self.assertEqual(made[0].killed, 1)
        self.assertTrue(rec["ok"])                     # statements done before the timeout played
        self.assertIn("timeout", rec["error"])

    def test_cancel_closes_door_stream(self):
        made = install(FakeDoor({HAIKU: [["```python\n"] + ["nod()\n"] * 200]}, delay=0.05))
        pl, d = mkplanner(cache=False)
        recs = []

        async def go():
            sent = []
            t = asyncio.create_task(planner.run_plan(pl, "p", "muse", "x", None, sent.append, record=recs.append,
                                                     mode="semantic"))
            await asyncio.sleep(0.4)
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
            await asyncio.sleep(0.1)
            return sent
        sent = asyncio.run(go())
        self.assertEqual(made[0].killed, 1)
        self.assertEqual((recs[0]["ok"], recs[0]["error"], recs[0]["planner"]), (False, "cancelled", "semantic"))

    def test_hedge(self):
        made = install(FakeDoor({HAIKU: [["```python\n"] + ["FAIL:stuck"]], SONNET: [ANS]}, delay=0.05))
        pl, d = mkplanner({"hedge": {"model": SONNET, "effort": "none", "after_s": 0.02}}, cache=False)
        # haiku fails first -> the hedge (sonnet) runs
        sent, rec = run_sem(pl, "x")
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["model"], SONNET)
        self.assertTrue(rec["hedge_started"])


class WarmKey(unittest.TestCase):
    """plan #1 after semwarm must find a warm door process for EXACTLY its prompt"""

    def setUp(self):
        planner.set_sem_module(fakesem)
        planner._SEM_LAST_USE.clear()

    def test_plan1_after_semwarm_hits_warm_key(self):
        import hashlib
        door = FakeDoor({SONNET: [ANS]})
        made = install(door)
        pl, d = mkplanner({"model": SONNET, "effort": "none"}, cache=False)
        sha = lambda x: hashlib.sha256(x.encode()).hexdigest()   # noqa: E731
        for ch, st in (("muse", {"sem": {"user": {"present": True, "ax": 0.3, "ay": -0.1}},
                                 "things": [{"id": "ball", "pos": [0.4, 0.1, 0.2]}], "root": {"x": 0.5, "z": 0}}),
                       ("fox", None)):
            async def go():
                specs = await pl.prewarm_semantic_async(ch)
                await asyncio.gather(*list(getattr(pl, "_sem_prime_tasks", ())))
                sent = []
                rec = await planner.run_plan(pl, "p", ch, "do an odd thing " + ch, st, sent.append, mode="semantic")
                again = await pl.prewarm_semantic_async(ch)          # used just now: no second prime
                return specs, rec, again
            n0 = len(made[0].calls) if made else 0
            specs, rec, again = asyncio.run(go())
            dd = made[0]
            warmed = {(m, sha(s)) for m, s, n in dd.prewarms}
            calls = dd.calls[n0:]
            self.assertEqual([c["tag"] for c in calls], ["semantic-prime", "semantic"], ch)
            self.assertEqual(sha(calls[0]["system"]), sha(calls[1]["system"]), ch)      # the prime = the plan's prompt
            self.assertIn((calls[1]["model"], sha(calls[1]["system"])), warmed, ch)     # plan #1 = a warmed key
            self.assertTrue(rec["attempts"][0]["warm"], ch)
            self.assertEqual(len(made), 1)                         # one door instance (model + effort) for both
            self.assertEqual(len(dd.calls), n0 + 2)                # the re-warm did not prime again
        shutil.rmtree(d, ignore_errors=True)

    def test_own_door_pool(self):
        """the keyframe planner (same model + effort) never shares the semantic door's warm budget"""
        made = install(FakeDoor({SONNET: [ANS]}))
        self.assertIsNot(planner.sem_door("none"), planner.door_for("none"))
        self.assertIs(planner.sem_door("none"), planner.sem_door("none"))
        self.assertEqual(len(made), 2)

    def test_real_core_prompt_stable_across_states(self):
        """the real app/semantic: the warm prompt (state None) == the plan prompt for any plan state, per character"""
        import importlib.util
        root = os.path.join(HERE, "..", "app", "semantic")
        if not os.path.isfile(os.path.join(root, "__init__.py")):
            self.skipTest("app/semantic not in this tree")
        spec = importlib.util.spec_from_file_location("semantic_real", os.path.join(root, "__init__.py"),
                                                      submodule_search_locations=[root])
        real = importlib.util.module_from_spec(spec)
        sys.modules["semantic_real"] = real
        spec.loader.exec_module(real)
        pl, d = mkplanner()
        for ch in real.CHARACTERS:
            warm = pl._sem_system(real, ch, real.resolve_targets(ch, None))
            cfg = planner.load_scene(ch) or {}
            things = [{"id": p["id"], "pos": [0.3, 0.2, 0.1]} for p in cfg.get("props", [])][:6]
            for st in ({}, {"sem": {"user": {"present": True, "ax": 0.4, "ay": 0.1}}},
                       {"things": things + [{"id": "not_in_props", "pos": [1, 0, 1]}], "root": {"x": 0.7, "z": -0.2},
                        "posture": "seated"}):
                self.assertEqual(pl._sem_system(real, ch, real.resolve_targets(ch, st)), warm, (ch, st))
        shutil.rmtree(d, ignore_errors=True)


class HubWiring(unittest.TestCase):
    def setUp(self):
        planner.set_sem_module(fakesem)

    def test_session(self):
        import app
        door = FakeDoor({HAIKU: [ANS]})
        made = install(door)
        hub = app.Hub()
        hub.intent.cfg["enabled"] = False
        cache = tempfile.mkdtemp(prefix="semhub-cache2-")
        hub.planner.cache_dir = os.path.join(cache, "plans")
        hub.planner.cfg["semantic"] = {"model": HAIKU, "effort": "none"}
        sent = []

        async def run():
            s = app.Session(hub, "muse", "720p", "h264", "direct")
            s.send = sent.append
            s.on_control({"type": "semwarm", "character": "muse"})
            await asyncio.sleep(0.05)
            s.on_control({"type": "plan", "id": "s1", "character": "muse", "description": "nod at me twice slowly",
                          "planner": "semantic", "state": {"sem": {"user": {"present": False}}}})
            while s.plan_running:
                await asyncio.sleep(0.02)
            s.on_control({"type": "plan", "id": "s2", "character": "muse", "description": "wave",
                          "planner": "semantic"})
            while s.plan_running:
                await asyncio.sleep(0.02)
            hub.planner.cfg["enabled"] = False
            s.on_control({"type": "plan", "id": "s3", "character": "muse", "description": "x", "planner": "semantic"})
            hub.planner.cfg["enabled"] = True
            return s
        s = asyncio.run(run())
        warm = [m for m in sent if m["type"] == "semwarm"]
        self.assertEqual(len(warm), 1)
        self.assertEqual((warm[0]["ok"], warm[0]["available"], warm[0]["model"]), (True, True, HAIKU))
        pw = [p for d in made for p in d.prewarms]
        self.assertTrue(pw and pw[0][0] == HAIKU and pw[0][1].startswith("FAKE SEMANTIC SYSTEM for muse"))
        self.assertEqual(door.prewarms, [])            # the shared door (the hub's other users) is not touched
        s1 = [m for m in sent if m.get("id") == "s1"]
        self.assertTrue(any(m["type"] == "kf" and m.get("space") == "rig" for m in s1))
        self.assertTrue(any(m["type"] == "sem" for m in s1))
        s2 = [m for m in sent if m.get("id") == "s2" and m["type"] == "kf"]
        self.assertEqual((s2[0]["source"], s2[0]["planner"]), ("library", "semantic"))
        s3 = [m for m in sent if m.get("id") == "s3"]
        self.assertEqual((s3[0]["ok"], s3[0]["error"], s3[0]["planner"]), (False, "planner off", "semantic"))
        recs = {r["id"]: r for r in s.plans}
        self.assertEqual(recs["s1"]["planner"], "semantic")
        self.assertIn("nod", recs["s1"]["program"])
        self.assertIn("first_pose", recs["s1"]["ms"])
        self.assertEqual(recs["s3"]["planner"], "semantic")
        summ = s.summary(full=True)
        self.assertIn("plans", summ)
        shutil.rmtree(cache, ignore_errors=True)

    def test_off_mode_never_warms(self):
        import app
        door = FakeDoor({HAIKU: [ANS]})
        made = install(door)
        hub = app.Hub()
        hub.intent.cfg["enabled"] = False
        sent = []

        async def run():
            s = app.Session(hub, "muse", "720p", "h264", "direct")
            s.send = sent.append
            s.on_control({"type": "plan", "id": "o1", "character": "muse", "description": "wave"})
            while s.plan_running:
                await asyncio.sleep(0.02)
            return s
        asyncio.run(run())
        self.assertEqual([p for d in made for p in d.prewarms], [])
        self.assertEqual(made, [])
        kf = [m for m in sent if m["type"] == "kf"]
        self.assertEqual(list(kf[0]), ["type", "id", "poses", "done", "ok", "source", "engine", "summary",
                                       "ms_first_pose", "ms_done", "usd"])
        self.assertFalse([m for m in sent if m["type"] in ("sem", "semwarm")])


class AnthropicRoute(unittest.TestCase):
    """route 'anthropic' (ANTHROPIC_API_KEY, no CLI): the same plan through Planner._route_stream (faked: no network)"""

    def setUp(self):
        planner.set_sem_module(fakesem)
        self.door = FakeDoor({HAIKU: [ANS]})
        self.made = install(self.door)
        self.calls = []

    def fake_stream(self, chunks):
        async def route_stream(http, system, user, deadline, info):
            self.calls.append({"route": info["spec"]["route"], "model": info["spec"]["model"], "system": system})
            for c in chunks:
                await asyncio.sleep(0.005)
                yield c
            info["usage"] = {"input_tokens": 1200, "output_tokens": 80, "anthropic": True}
        return route_stream

    def test_api_route(self):
        os.environ["AVATAR_SEM_ROUTE"] = "anthropic"
        try:
            pl, d = mkplanner()
            pl._route_stream = self.fake_stream(ANS)
            usd = []
            sent, recs = [], []

            async def go():
                return await planner.run_plan(pl, "p1", "muse", "look at me and nod", None, sent.append, http=object(),
                                              on_usd=usd.append, record=recs.append, mode="semantic",
                                              t_recv=time.monotonic())
            rec = asyncio.run(go())
        finally:
            os.environ["AVATAR_SEM_ROUTE"] = "claude_door"
        kf = [m for m in sent if m["type"] == "kf"]
        self.assertTrue(kf[-1]["ok"], kf[-1])
        self.assertEqual(rec.get("route"), "anthropic")
        self.assertEqual([c["route"] for c in self.calls], ["anthropic"])
        self.assertEqual([p for dd in self.made for p in dd.prewarms], [])        # nothing warmed, no door used
        self.assertEqual([c for dd in self.made for c in dd.calls], [])
        self.assertTrue(usd and usd[0] > 0 and kf[-1]["usd"] > 0, (usd, kf[-1]["usd"]))
        self.assertGreater(sum(len(m["poses"]) for m in kf), 0)

    def test_auto_route(self):
        os.environ.pop("AVATAR_SEM_ROUTE", None)
        old = os.environ.get("AVATAR_CLAUDE_BIN")
        try:
            os.environ["AVATAR_CLAUDE_BIN"] = os.path.join(tempfile.gettempdir(), "no-such-claude-cli")
            self.assertEqual(planner.Planner.sem_route("auto"), "anthropic")
            self.assertEqual(planner.Planner.sem_route("claude_door"), "claude_door")
        finally:
            os.environ["AVATAR_SEM_ROUTE"] = "claude_door"
            if old is None:
                os.environ.pop("AVATAR_CLAUDE_BIN", None)
            else:
                os.environ["AVATAR_CLAUDE_BIN"] = old


if __name__ == "__main__":
    unittest.main(verbosity=2)
