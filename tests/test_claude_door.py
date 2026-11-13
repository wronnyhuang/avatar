"""door tests. Fake `claude` script for parsing/timeout/replacement; live test opt-in: AVATAR_DOOR_LIVE=1.
   python3 -m unittest tests/test_claude_door.py"""
import asyncio, os, stat, sys, tempfile, unittest, textwrap
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))
import claude_door as cd

FAKE = textwrap.dedent('''\
    #!/usr/bin/env python3
    import sys, json, time, os
    mode = os.environ.get("FAKE_MODE", "ok")
    open(os.environ["FAKE_LOG"], "a").write("spawn %d\\n" % os.getpid())
    line = sys.stdin.readline()
    if mode == "hang":
        time.sleep(60)
    if mode == "die":
        sys.stderr.write("boom: not logged in\\n"); sys.exit(3)
    def ev(e): print(json.dumps(e), flush=True)
    ev({"type": "system", "subtype": "init"})
    for w in ["Hel", "lo ", "there"]:
        ev({"type": "stream_event", "event": {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": w}}})
        time.sleep(0.05)
    ev({"type": "result", "subtype": "success", "is_error": False, "result": "Hello there", "total_cost_usd": 0.001,
        "usage": {"input_tokens": 2, "output_tokens": 3, "cache_read_input_tokens": 10}})
    time.sleep(30)
''')


class T(unittest.TestCase):
    def setUp(self):
        d = tempfile.mkdtemp()
        self.bin = os.path.join(d, "fakeclaude")
        open(self.bin, "w").write(FAKE)
        os.chmod(self.bin, 0o755)
        self.log = os.path.join(d, "log")
        os.environ.update(AVATAR_CLAUDE_BIN=self.bin, FAKE_LOG=self.log, FAKE_MODE="ok")
        self.door = cd.ClaudeDoor()

    def spawns(self):
        return open(self.log).read().count("spawn") if os.path.exists(self.log) else 0

    def run_(self, coro):
        async def w():
            try:
                return await coro(self.door)
            finally:
                await self.door.close()
        return asyncio.run(w())

    def test_stream_parse(self):
        async def go(door):
            evs = [e async for e in door.stream(model="haiku", system="s", user="u")]
            return evs
        evs = self.run_(go)
        self.assertEqual([e["delta"] for e in evs[:-1]], ["Hel", "lo ", "there"])
        d = evs[-1]
        self.assertTrue(d["done"] and d["ok"])
        self.assertEqual(d["text"], "Hello there")
        self.assertGreaterEqual(d["ms_first"], 0)
        self.assertEqual(d["usage"]["cache_read_input_tokens"], 10)

    def test_warm_and_replace(self):
        async def go(door):
            door.prewarm("haiku", "s", n=2)
            await asyncio.sleep(1.0)
            a = self.spawns()
            e = await door.complete(model="haiku", system="s", user="u")
            await asyncio.sleep(1.0)
            return a, e, self.spawns()
        a, e, b = self.run_(go)
        self.assertEqual(a, 2)
        self.assertTrue(e["warm"] and e["ok"])
        self.assertEqual(b, 3)      # one replacement spawned

    def test_pool_bound(self):
        async def go(door):
            door.prewarm("haiku", "a", n=5)
            door.prewarm("sonnet", "b", n=5)
            await asyncio.sleep(1.5)
            return self.spawns()
        self.assertLessEqual(self.run_(go), cd.MAX_TOTAL)

    def test_timeout_kills(self):
        os.environ["FAKE_MODE"] = "hang"
        async def go(door):
            e = await door.complete(model="haiku", system="s", user="u", max_s=0.8)
            slot_gone = True
            return e
        e = self.run_(go)
        self.assertFalse(e["ok"])
        self.assertIn("timeout", e["error"])

    def test_dead_process_stderr(self):
        os.environ["FAKE_MODE"] = "die"
        async def go(door):
            return await door.complete(model="haiku", system="s", user="u")
        e = self.run_(go)
        self.assertFalse(e["ok"])
        self.assertIn("not logged in", e["error"])

    def test_no_zombies(self):
        async def go(door):
            door.prewarm("haiku", "s", n=2)
            await asyncio.sleep(0.8)
            await door.complete(model="haiku", system="s", user="u")
        self.run_(go)
        import subprocess, time
        time.sleep(0.5)
        out = subprocess.run(["pgrep", "-f", self.bin], capture_output=True, text=True).stdout.split()
        # pgrep may match itself's parent shell; fakes must be gone
        alive = [p for p in out if os.path.exists("/proc/%s" % p) and open("/proc/%s/stat" % p).read().split(")")[1].split()[0] != "Z"]
        self.assertEqual(alive, [])

    def test_bad_model(self):
        async def go(door):
            return await door.complete(model="gpt", system="s", user="u")
        self.assertFalse(self.run_(go)["ok"])


@unittest.skipUnless(os.environ.get("AVATAR_DOOR_LIVE"), "set AVATAR_DOOR_LIVE=1")
class Live(unittest.TestCase):
    def test_live_haiku(self):
        os.environ.pop("AVATAR_CLAUDE_BIN", None)
        d = cd.ClaudeDoor()
        async def go():
            try:
                return await d.complete(model="haiku", system="Reply with one word.", user="Capital of France?")
            finally:
                await d.close()
        e = asyncio.run(go())
        self.assertTrue(e["ok"], e)
        self.assertIn("Paris", e["text"])


if __name__ == "__main__":
    unittest.main()
