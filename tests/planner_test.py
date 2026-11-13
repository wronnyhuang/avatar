"""server-side keyframe planner (app/planner.py + app/planner_prompts + the app.py wiring). No paid calls.
    python3 tests/planner_test.py
Parser parity: the Python StreamParser against exec_stream.cjs parseStream (the planner streaming bench, $PLANNER_BENCH, run with node) — on the stored
fixtures (tests/fixtures/planner_streams: recorded streams + node's parse at fixture time), on every recorded stream of
$PLANNER_BENCH when it is there, and on synthetic edge-case streams; normalization; library; plan cache; prompt manifest;
contract section 2 message shapes through the real Session.on_control path with a replayed recorded stream."""
import asyncio
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "..", "app")
sys.path.insert(0, APP)
os.environ["AVATAR_HOME"] = tempfile.mkdtemp(prefix="avatar-plan-home-")   # never the live server's state/data
for _k in ("AVATAR_PLANNER_MODEL", "AVATAR_PLANNER_FALLBACK", "AVATAR_PLANNER_FALLBACK_AFTER_S"):
    os.environ.pop(_k, None)                                  # the model switch comes from the config below
SOL = {"model": "sol", "fallback": None}   # the default became gem35lite + dsv41flash; these fixtures are Sol's
import planner  # noqa: E402

BENCH = os.environ.get("PLANNER_BENCH", "")   # the planner streaming bench (stream/exec_stream.cjs + out/); optional
EXEC = os.path.join(BENCH, "stream", "exec_stream.cjs")
FIX = os.path.join(HERE, "fixtures", "planner_streams")
REF = os.path.join(HERE, "planner_parse_ref.cjs")
HAVE_NODE = bool(shutil.which("node")) and os.path.isfile(EXEC)


def jload(path):
    with open(path) as f:
        return json.load(f)


def rbytes(path):
    with open(path, "rb") as f:
        return f.read()


def nz(x):
    """numbers compared as numbers (JSON 3 == 3.0, -0 == 0)"""
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return float(x) + 0.0
    if isinstance(x, list):
        return [nz(v) for v in x]
    if isinstance(x, dict):
        return {k: nz(v) for k, v in x.items()}
    return x


def mine_parse(S):
    t_done = S.get("t_done_ms")
    if t_done is None:
        t_done = S["chunks"][-1][0] if S.get("chunks") else 0
    p = planner.parse_stream(S.get("chunks") or [], t_done)
    return {**p, "dropped": len(p["dropped"]), "salvaged": len(p["salvaged"])}


def node_ref(paths):
    out = subprocess.run(["node", REF, EXEC, *paths], capture_output=True, text=True, timeout=300, cwd="/tmp")
    if out.returncode:
        raise RuntimeError(out.stderr[-500:])
    return [json.loads(ln) for ln in out.stdout.splitlines() if ln.strip()]


def diff(ref, mine):
    return [k for k in ref if json.dumps(nz(ref[k]), sort_keys=True) != json.dumps(nz(mine.get(k)), sort_keys=True)]


class Parity(unittest.TestCase):
    def test_fixtures(self):
        files = sorted(glob.glob(os.path.join(FIX, "*.json")))
        self.assertGreaterEqual(len(files), 5)
        n = 0
        for f in files:
            S = jload(f)
            ref, mine = S["ref_parse"], mine_parse(S)
            self.assertEqual(diff(ref, mine), [], f)
            self.assertEqual([(p["ms"], p["t"]) for p in ref["poses"]], [(p["ms"], p["t"]) for p in mine["poses"]], f)
            n += ref["n_poses"]
        print(f"  parity fixtures: {len(files)} streams, {n} poses identical (ms, t, controls)")

    def test_incremental_equals_whole(self):
        for f in sorted(glob.glob(os.path.join(FIX, "*.json"))):
            S = jload(f)
            sp, got = planner.StreamParser(), []
            for ms, s in S["chunks"]:
                got += [(p["ms"], round(p["t"], 3)) for p in sp.feed(ms, s)]
                if sp.complete:
                    break
            self.assertEqual(got, [(p["ms"], p["t"]) for p in S["ref_parse"]["poses"]], f)

    @unittest.skipUnless(HAVE_NODE and os.path.isdir(os.path.join(BENCH, "out")), "node or PLANNER_BENCH missing")
    def test_all_recorded_streams_vs_node(self):
        paths = sorted(glob.glob(os.path.join(BENCH, "out", "stream_kf-*", "*", "stream.json"))
                       + glob.glob(os.path.join(BENCH, "stream", "dev", "*", "*", "*", "stream.json")))
        refs = node_ref(paths)
        bad, poses = [], 0
        for r in refs:
            S = jload(r["path"])
            d = diff(r["parse"], mine_parse(S))
            poses += r["parse"]["n_poses"]
            if d:
                bad.append((r["path"], d))
        print(f"  parity vs node: {len(refs)} recorded streams ({len({p.split('/')[-3] for p in paths})} models/sets), "
              f"{poses} poses, {len(bad)} mismatches")
        self.assertEqual(bad, [])
        sol = [r for r in refs if "/stream_kf-sol/" in r["path"]]
        self.assertGreaterEqual(len(sol), 5)

    @unittest.skipUnless(HAVE_NODE, "node or PLANNER_BENCH missing")
    def test_synthetic_edge_cases_vs_node(self):
        texts = {
            "trailing_comma": '{"summary":"x","poses":[{"t":0.3,"smile":0.5,},{"t":0.8,"head_yaw":0.2}]}',
            "salvage": '{"summary":"x","poses":[{"t":0.3,smile:0.5,"head_yaw":0.1},{"t":0.9,"squash":0.1}]}',
            "array_form": '{"poses":[[0.4,{"smile":1}],[1.0,{"root_y":0.2}]]}',
            "time_set": '{"poses":[{"time":0.5,"set":{"smile":1,"root_y":"0.3"}},{"time":1.2,"set":{"squash":true}}]}',
            "shift": '{"poses":[{"t":0,"smile":1},{"t":0.1,"smile":0.2},{"t":0.5,"head_yaw":0.3}]}',
            "unknown_and_bad": '{"poses":[{"t":0.3,"knee":1,"smile":null,"eye_wide":"",'
                               '"squint":[0.4],"head_roll":{"a":1}},{"t":0.2,"smile":1},{"t":19.5,"smile":0.3},{"t":0.6}]}',
            "fenced_prose": 'Sure!\n```json\n{"summary":"a \\"quoted\\" } brace","poses":[{"t":0.3,"smile":0.4}, {"t":0.6, "smile":0.1}]}\n```',
            "stray_close": '{"poses":[}{"t":0.3,"smile":0.4}],"x":[{"t":0.5}]}',
            "nested": '{"poses":[{"t":0.3,"set":{"smile":0.4,"x":{"y":[1,2]}}},{"t":0.7,"smile":0.2}]}',
            "nan": '{"poses":[{"t":0.3,"smile":NaN},{"t":0.6,"smile":0.2}]}',
            "no_poses": '{"summary":"nothing"}',
            "truncated": '{"summary":"x","poses":[{"t":0.3,"smile":0.4},{"t":0.6,"sm',
        }
        d = tempfile.mkdtemp(prefix="avatar-synth-")
        paths = {}
        for name, txt in texts.items():
            for split in (1, 3, 7):                       # the same text arriving in different chunkings
                chunks = [[100 + 10 * k, txt[i:i + split]] for k, i in enumerate(range(0, len(txt), split))]
                p = os.path.join(d, f"{name}_{split}.json")
                with open(p, "w") as fh:
                    json.dump({"chunks": chunks, "t_done_ms": chunks[-1][0] + 5}, fh)
                paths[p] = name
        refs = node_ref(sorted(paths))
        for r in refs:
            S = jload(r["path"])
            self.assertEqual(diff(r["parse"], mine_parse(S)), [], r["path"])
        shutil.rmtree(d, ignore_errors=True)
        print(f"  parity synthetic: {len(refs)} edge-case streams identical to node")


class Normalize(unittest.TestCase):
    def test_normalize(self):
        N = planner.normalize
        self.assertEqual(N("Can you please do THREE jumping-jacks for me now?"), "do 3 jumping jacks")
        self.assertEqual(N("do three jumping jacks"), N("Could you do 3 jumping jacks, please!"))
        self.assertEqual(N("Spin around twice"), "spin around 2 times")
        self.assertEqual(N("jump twenty five times"), "jump 25 times")
        self.assertEqual(N("Would you wave for 1.5 seconds?"), "wave for 1.5 seconds")
        self.assertEqual(N("give me a high five"), "give me a high five")
        self.assertEqual(N("I want you to   do a backflip right now"), "do a backflip")
        self.assertEqual(N("don't stop"), "dont stop")
        self.assertNotEqual(N("wave your left paw"), N("wave your right paw"))
        k1 = planner.cache_key("fox", N("Please wave three times"))
        self.assertEqual(k1, hashlib.sha256(("fox\n" + "wave 3 times").encode()).hexdigest())
        self.assertNotEqual(k1, planner.cache_key("orb", N("Please wave three times")))


def replay(path, scale=0.0, usage=True):
    """stream_fn replaying a recorded stream (scale = real-time factor; 0 = as fast as possible)"""
    S = jload(path)

    async def fn(http, system, user, deadline, info):
        last = 0
        for ms, s in S["chunks"]:
            if scale:
                await asyncio.sleep(max(0, ms - last) / 1000 * scale)
            last = ms
            if s:
                yield s
        if usage:
            info["usage"] = {"input_tokens": 3101, "input_tokens_details": {"cached_tokens": 3079}, "output_tokens": 1090}
    fn.calls = 0
    return S, fn


async def collect(pl, ch, desc, state=None):
    evs = []
    async for ev in pl.plan(ch, desc, state, http=object()):
        evs.append(ev)
    return evs


class PlannerFlow(unittest.TestCase):
    def setUp(self):
        self.cache = tempfile.mkdtemp(prefix="avatar-plan-cache-")
        self.pl = planner.Planner(SOL, lambda m: None, cache_dir=self.cache)

    def tearDown(self):
        shutil.rmtree(self.cache, ignore_errors=True)

    def test_library_hits(self):
        cmds, why = self.pl.library("muse", "do a backflip")
        self.assertEqual(cmds, [{"action": "move", "name": "backflip"}], why)
        cmds, why = self.pl.library("fox", "wave")
        self.assertEqual((cmds[0]["action"], cmds[0]["name"]), ("gesture", "wave"), why)
        evs = asyncio.run(collect(self.pl, "muse", "do a backflip"))
        self.assertEqual(len(evs), 1)
        m = planner.kf_message("p1", evs[0])
        self.assertEqual((m["source"], m["done"], m["ok"], m["poses"], m["engine"]),
                         ("library", True, True, [], [{"action": "move", "name": "backflip"}]))
        evs = asyncio.run(collect(self.pl, "fox", "wave"))
        self.assertEqual((evs[0]["source"], evs[0]["engine"][0]["name"]), ("library", "wave"))

    def test_library_misses(self):
        for ch, d in [("fox", "do a backflip"), ("fox", "spin around three times"), ("fox", "wave quickly"),
                      ("fox", "jump"), ("fox", "do a little dance"), ("orb", "wiggle your ears"), ("bear", "wag your tail"),
                      ("muse", "do three jumping jacks"), ("fox", "clap ten times"), ("fox", "wave like a robot"),
                      ("muse", "wave excitedly"), ("fox", "wave your right paw three times quickly")]:
            cmds, why = self.pl.library(ch, d)
            self.assertIsNone(cmds, f"{ch} {d!r} -> {cmds}")
        for ch, d in [("fox", "nod twice"), ("orb", "wave three times"), ("muse", "spin around 3 times"),
                      ("muse", "jump 5 times quickly"), ("fox", "smile"), ("muse", "wink with your left eye")]:
            cmds, why = self.pl.library(ch, d)
            self.assertTrue(cmds, f"{ch} {d!r} -> {why}")

    def test_planner_stream_then_cache(self):
        f = os.path.join(FIX, "stream_kf-sol__m06_jacks.json")
        S, fn = replay(f)
        self.pl.stream_fn = fn
        usd = []
        evs = []

        async def run():
            async for ev in self.pl.plan("muse", "Please do three jumping jacks!", None, http=object(), on_usd=usd.append):
                evs.append(ev)
        asyncio.run(run())
        batches, done = evs[:-1], evs[-1]
        self.assertTrue(all(not e["done"] and e["poses"] for e in batches))
        self.assertTrue(done["done"] and done["ok"] and done["source"] == "planner" and done["poses"] == [])
        got = [p for e in batches for p in e["poses"]]
        ref = S["ref_parse"]["poses"]
        self.assertEqual([round(p["t"], 3) for p in got], [p["t"] for p in ref])
        self.assertEqual(nz([p["set"] for p in got]), nz([p["set"] for p in ref]))
        self.assertEqual(len(done["all_poses"]), len(ref))
        self.assertEqual(done["summary"], "Three cheerful jumping jacks")
        self.assertAlmostEqual(sum(usd), done["usd"], 6)
        self.assertAlmostEqual(done["usd"], ((3101 - 3079) * 2.0 + 3079 * 0.10 + 1090 * 10.0) / 1e6, 6)
        key = planner.cache_key("muse", "do 3 jumping jacks")
        self.assertTrue(os.path.isfile(os.path.join(self.cache, key + ".json")))
        # a repeat (other politeness, digits) is a cache hit: every pose at once, done in the first message
        evs2 = asyncio.run(collect(self.pl, "muse", "can you do 3 jumping jacks"))
        self.assertEqual(len(evs2), 1)
        m = planner.kf_message("p2", evs2[0])
        self.assertEqual((m["source"], m["done"], m["ok"], m["usd"]), ("cache", True, True, 0.0))
        self.assertEqual(m["poses"], done["all_poses"])
        self.assertEqual(m["poses"][0], {"t": 0.25, "root_y": -0.05, "squash": 0.12, "smile": 0.7})
        # another character / a changed prompt is a miss
        self.assertIsNone(self.pl.cache_get(planner.cache_key("fox", "do 3 jumping jacks"), self.pl.prompts["fox"]["sha256"], "gpt-6.1-sol"))
        self.assertIsNone(self.pl.cache_get(key, "0" * 64, "gpt-6.1-sol"))
        self.assertIsNotNone(self.pl.cache_get(key, self.pl.prompts["muse"]["sha256"], "gpt-6.1-sol"))
        # a non-default state is part of the key (the planner hears it)
        note = planner.state_note(self.pl.prompts["muse"]["traits"], {"posture": "seated", "seated": "chair", "holding": {"left": "ball"}})
        self.assertEqual(note, "Right now it is sitting on the chair; it holds the ball in its left paw.")
        self.assertEqual(planner.state_note(self.pl.prompts["fox"]["traits"], {"posture": "seated"}), "")
        self.assertEqual(planner.state_note(self.pl.prompts["muse"]["traits"], {"posture": "standing", "holding": {"left": None, "right": None}}), "")

    def test_retry_and_errors(self):
        calls = []
        f = os.path.join(FIX, "stream_kf-sol__n02_jump7.json")
        S, ok_fn = replay(f)

        async def flaky(http, system, user, deadline, info):
            calls.append(1)
            if len(calls) == 1:
                raise planner.PlanError("HTTP 503: busy", transient=True)
            async for s in ok_fn(http, system, user, deadline, info):
                yield s
        self.pl.stream_fn = flaky
        evs = asyncio.run(collect(self.pl, "fox", "jump seven times"))
        self.assertEqual(len(calls), 2)
        self.assertTrue(evs[-1]["ok"] and len(evs[-1]["attempts"]) == 2 and evs[-1]["error"] is None)

        async def broken(http, system, user, deadline, info):
            yield '{"summary":"x","poses":[{"t":0.3,"smile":0.5},'
            raise planner.PlanError("timeout", transient=True)
        self.pl.stream_fn = broken
        evs = asyncio.run(collect(self.pl, "fox", "do something odd"))
        d = evs[-1]
        self.assertEqual((d["ok"], d["error"], len(d["all_poses"]), len(d["attempts"])), (True, "timeout", 1, 1))  # no retry after a pose
        self.assertGreater(d["usd"], 0)                                               # aborted stream: estimated cost
        self.assertFalse(os.path.exists(os.path.join(self.cache, d["cache_key"] + ".json")))   # a broken answer is not cached

        async def hard(http, system, user, deadline, info):
            raise planner.PlanError("HTTP 400: bad", transient=False)
            yield ""
        self.pl.stream_fn = hard
        d = asyncio.run(collect(self.pl, "orb", "float"))[-1]
        self.assertEqual((d["ok"], d["error"][:8]), (False, "HTTP 400"))
        off = planner.Planner(False, lambda m: None, cache_dir=self.cache)
        d = asyncio.run(collect(off, "fox", "wave"))[-1]
        self.assertEqual(planner.kf_message("p9", d), {"type": "kf", "id": "p9", "poses": [], "done": True, "ok": False,
                                                       "source": "planner", "engine": None, "summary": "", "ms_first_pose": None,
                                                       "ms_done": 0, "usd": 0.0, "error": "planner off"})

    def test_cancel_bills_and_records(self):
        S, fn = replay(os.path.join(FIX, "stream_kf-sol__m06_jacks.json"), scale=1.0, usage=False)
        self.pl.stream_fn = fn
        sent, usd, recs = [], [], []

        async def run():
            t = asyncio.create_task(planner.run_plan(self.pl, "p7", "muse", "do three jumping jacks", None, sent.append,
                                                     http=object(), on_usd=usd.append, record=recs.append))
            while not sent:
                await asyncio.sleep(0.01)
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        asyncio.run(run())
        self.assertTrue(sent and not sent[-1]["done"])
        self.assertEqual((recs[0]["id"], recs[0]["ok"], recs[0]["error"]), ("p7", False, "cancelled"))
        self.assertTrue(usd and usd[0] > 0)                       # the tokens streamed so far are counted (estimate)

    def test_seated_controls(self):
        # a seated character's parser keeps body_bounce and drops root / feet (its prompt never offers them)
        sp = planner.StreamParser(self.pl.prompts["fox"]["controls"])
        sp.feed(10, '{"poses":[{"t":0.3,"body_bounce":0.5,"root_y":0.3,"right_foot_y":0.1,"tail_wag":0.4}]}')
        self.assertEqual(sp.poses[0]["set"], {"body_bounce": 0.5, "tail_wag": 0.4})
        self.assertEqual(sp.errors[0]["unknown"], ["root_y", "right_foot_y"])
        sp = planner.StreamParser(self.pl.prompts["orb"]["controls"])
        sp.feed(10, '{"poses":[{"t":0.3,"right_ear":1,"tail_wag":0.4,"nose_wiggle":1,"right_paw_y":1}]}')
        self.assertEqual(sp.poses[0]["set"], {"right_paw_y": 1.0})


class Prompts(unittest.TestCase):
    def test_manifest(self):
        man = jload(os.path.join(APP, "planner_prompts", "MANIFEST.json"))
        chars = man["characters"]
        self.assertEqual(set(chars), {"fox", "bear", "rabbit", "hippo", "monkey", "orb", "cube", "drop", "star", "pyramid",
                                      "blob", "muse"})
        for k, e in chars.items():
            with open(os.path.join(APP, "planner_prompts", e["file"]), "rb") as fh:
                b = fh.read()
            self.assertEqual(hashlib.sha256(b).hexdigest(), e["sha256"], k)
        # the frozen prompt is kept as frozen_v1.txt; muse.txt = it + MANIFEST "fixes" (the head_pitch sign)
        fz = rbytes(os.path.join(APP, "planner_prompts", "frozen_v1.txt"))
        self.assertEqual(hashlib.sha256(fz).hexdigest(), "dc070126dd6ae359eb989b7de81ad51e08051eb4e685a6b7a219d8ad273e79e0")
        frozen = os.path.join(BENCH, "stream", "prompt_kf_stream.txt")
        if os.path.isfile(frozen):
            self.assertEqual(fz, rbytes(frozen))
        muse = rbytes(os.path.join(APP, "planner_prompts", "muse.txt")).decode("utf-8")
        fixed = fz.decode("utf-8")
        for fx in man["fixes"]:
            fixed = fixed.replace(fx["from"], fx["to"])
        self.assertEqual(muse, fixed)
        self.assertIn("+ = tips the nose down", muse)                      # renderer: head_pitch > 0 = nose down
        r = subprocess.run([sys.executable, os.path.join(APP, "planner_prompts", "make_prompts.py"), "--check"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for k, e in chars.items():
            with open(os.path.join(APP, "planner_prompts", e["file"])) as fh:
                txt = fh.read()
            if k == "muse":
                continue
            for bad in ("root_x", "root_y", "root_yaw", "root_pitch", "root_roll", "squash", "_foot_", "fingers", "THE ROOM"):
                self.assertNotIn(bad, txt, f"{k} prompt offers {bad}")
            self.assertIn("body_bounce", txt)
            tr = e["traits"]
            for ctl, has in (("right_ear", tr["ears"]), ("tail_wag", tr["tail"]), ("nose_wiggle", tr["nose"])):
                self.assertEqual(ctl in txt, has, f"{k} {ctl}")
            self.assertIn("seated version" if not tr["shape"] else "in-place version", txt)
        self.assertTrue(chars["fox"]["traits"]["tail"] and not chars["bear"]["traits"]["tail"])
        self.assertFalse(any(chars[s]["traits"]["ears"] for s in ("orb", "cube", "drop", "star", "pyramid", "blob")))
        pl = planner.Planner(None, lambda m: None)
        self.assertTrue(all(p["ok"] for p in pl.prompts.values()))
        print("  prompts: " + ", ".join(f"{k} {e['sha256'][:12]}" for k, e in chars.items()))


class HubWiring(unittest.TestCase):
    """Session.on_control {type:'plan'} / {type:'perform'} -> {type:'kf'} on the page's down queue (contract 2)."""

    def test_session_plan_messages(self):
        import app
        hub = app.Hub()
        hub.intent.cfg["enabled"] = False                         # no ASR worker in this test
        cache = tempfile.mkdtemp(prefix="avatar-hub-cache-")
        hub.planner.cache_dir = cache
        S, fn = replay(os.path.join(FIX, "stream_kf-sol__m06_jacks.json"), scale=0.02)
        hub.planner.stream_fn = fn
        sent = []

        async def run():
            s = app.Session(hub, "fox", "720p", "h264", "direct")
            s.send = sent.append
            before = hub.spend.today()
            s.on_control({"type": "plan", "id": "p1", "character": "fox", "description": "do three jumping jacks",
                          "state": {"posture": "seated"}})
            s.on_control({"type": "plan", "id": "p2", "character": "fox", "description": "wave"})
            while s.plan_running:
                await asyncio.sleep(0.02)
            s.on_control({"type": "plan", "id": "p3", "character": "fox", "description": "Do 3 jumping jacks please"})
            while s.plan_running:
                await asyncio.sleep(0.02)
            s.on_control({"type": "perform", "id": "p1", "description": "do three jumping jacks", "source": "planner",
                          "marks": {"call_ms": 1000, "plan_first_pose_ms": 3400, "ready_ms": 3900, "cue_audio_ms": 4100,
                                    "first_motion_ms": 4200, "end_ms": 9000}})
            hub.planner.cfg["enabled"] = False
            s.on_control({"type": "plan", "id": "p4", "character": "fox", "description": "wave"})
            hub.planner.cfg["enabled"] = True
            return s, hub.spend.today() - before
        s, spent = asyncio.run(run())
        kf = [m for m in sent if m.get("type") == "kf"]
        for m in kf:
            base = ["type", "id", "poses", "done"]
            keys = base if not m["done"] else base + ["ok", "source", "engine", "summary", "ms_first_pose", "ms_done", "usd"] + \
                (["error"] if "error" in m else [])
            self.assertEqual(list(m), keys, m)
            json.dumps(m)
            for p in m["poses"]:
                self.assertEqual(list(p)[0], "t")
        p1 = [m for m in kf if m["id"] == "p1"]
        self.assertTrue(all(not m["done"] and len(m["poses"]) >= 1 for m in p1[:-1]) and p1[-1]["done"])
        self.assertEqual((p1[-1]["source"], p1[-1]["ok"], p1[-1]["poses"]), ("planner", True, []))
        self.assertEqual(sum(len(m["poses"]) for m in p1), 20 - 0)       # (fox keeps only its own controls in each pose)
        self.assertTrue(all("root_y" not in p and "right_foot_x" not in p for m in p1 for p in m["poses"]))
        p2 = [m for m in kf if m["id"] == "p2"]
        self.assertEqual((len(p2), p2[0]["source"], p2[0]["engine"][0]["name"]), (1, "library", "wave"))
        p3 = [m for m in kf if m["id"] == "p3"]
        self.assertEqual((len(p3), p3[0]["source"], p3[0]["done"], len(p3[0]["poses"])), (1, "cache", True, 20))
        p4 = [m for m in kf if m["id"] == "p4"]
        self.assertEqual((p4[0]["ok"], p4[0]["error"]), (False, "planner off"))
        self.assertAlmostEqual(spent, p1[-1]["usd"], 6)
        self.assertAlmostEqual(s.usd["planner"], p1[-1]["usd"], 6)
        self.assertAlmostEqual(hub.spend.rec.get("planner_usd", 0), p1[-1]["usd"], 6)
        summ = s.summary(full=True)
        self.assertEqual(summ["plans"]["sources"], {"planner": 1, "library": 1, "cache": 1, "planner:fail": 1})
        self.assertEqual(summ["plans"]["command_to_first_motion_ms"]["p50"], 3200)
        self.assertEqual(summ["plans"]["command_to_first_speech_ms"]["p50"], 3100)
        self.assertEqual(len(summ["performs"]), 1)
        self.assertTrue(s.config_msg(None)["planner"])
        shutil.rmtree(cache, ignore_errors=True)
        print(f"  hub: {len(kf)} kf messages, planner ${p1[-1]['usd']:.5f} counted in spend")


if __name__ == "__main__":
    unittest.main(verbosity=2)
