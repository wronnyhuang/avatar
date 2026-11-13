"""OBJECTS planner layer, hub side (app/planner.py + app/planner_prompts/objects_appendix.txt). No paid calls.
    python3 tests/planner_objects_test.py
The keyframe language WITH things (held controls, paw events, "do" steps) in the stream parser; plans with things only
for a scene character whose page sent its things (appendix + live things note; muse.txt untouched, make_prompts --check
passes); the cache key / prompt sha follow the things; the library does not swallow a request naming a thing."""
import asyncio
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
os.environ["AVATAR_HOME"] = tempfile.mkdtemp(prefix="avatar-plan-home-")
for _k in ("AVATAR_PLANNER_MODEL", "AVATAR_PLANNER_FALLBACK", "AVATAR_PLANNER_FALLBACK_AFTER_S"):
    os.environ.pop(_k, None)
import planner  # noqa: E402

CFG = {"model": "sol", "fallback": None}
THINGS = [{"id": "ball", "name": "yellow ball", "where": "home", "held": None, "on": "floor", "pos": [0.313, 0.043, 0.27]},
          {"id": "book", "name": "blue book", "where": "home", "held": None, "on": "side_table", "pos": [-0.813, 0.203, -0.243]},
          {"id": "chair", "where": "home", "afford": ["furniture"], "pos": [0.65, 0.25, 0.18]}]
ANSWER = ('{"summary":"spin the book","poses":[{"t":0.3,"do":{"verb":"pick_up","object":"book","side":"right"}},'
          '{"t":0.8,"right_paw_y":0.8,"right_held_y":0.05},{"t":1.1,"right_held_spin":1},{"t":1.4,"right_held_spin":2,'
          '"right_toss_up":0.5},{"t":1.98,"right_catch":1},{"t":2.4,"do":"put_back book"}]}')


def chunks(text, n=7):
    return [[i * 10, text[i:i + n]] for i in range(0, len(text), n)]


def parse(text, allow_do=True, known=None):
    known = known if known is not None else planner.KF_CONTROLS + planner.KF_OBJ_CONTROLS
    return planner.parse_stream(chunks(text), known=known, allow_do=allow_do)


class Parser(unittest.TestCase):
    def test_do_held_events_accepted(self):
        r = parse(ANSWER)
        P = r["poses"]
        self.assertEqual(len(P), 6)
        self.assertEqual(P[0]["do"], {"verb": "pick_up", "object": "book", "side": "right"})
        self.assertEqual(P[0]["set"], {})
        self.assertEqual(P[1]["set"], {"right_paw_y": 0.8, "right_held_y": 0.05})
        self.assertEqual(P[3]["set"]["right_toss_up"], 0.5)
        self.assertEqual(P[4]["set"], {"right_catch": 1.0})
        self.assertEqual(P[5]["do"], {"verb": "put_back", "object": "book"})
        self.assertEqual(r["unknown"], [])
        self.assertEqual(r["shift_s"], 0)

    def test_unknown_keys_still_rejected(self):
        r = parse('{"poses":[{"t":0.3,"right_paw_y":0.5,"right_held_wobble":1,"juggle":3},{"t":0.9,"do":{"verb":"!!"}}]}')
        un = sorted(u for e in r["unknown"] for u in e["unknown"])
        self.assertIn("right_held_wobble", un)
        self.assertIn("juggle", un)
        self.assertTrue(any(u.startswith("do=") for u in un), un)
        self.assertEqual(r["poses"][0]["set"], {"right_paw_y": 0.5})
        self.assertNotIn("do", r["poses"][1])

    def test_do_only_for_scene_plans(self):
        r = parse(ANSWER, allow_do=False, known=planner.KF_CONTROLS)
        un = {u for e in r["unknown"] for u in e["unknown"]}
        self.assertTrue({"do", "right_held_y", "right_held_spin", "right_toss_up", "right_catch"} <= un, un)
        self.assertFalse(any("do" in p for p in r["poses"]))

    def test_shift_counts_a_do_step(self):
        r = parse('{"poses":[{"t":0,"do":"pick up ball"},{"t":0.6,"right_paw_y":0.4}]}')
        self.assertEqual(r["shift_s"], 0.3)
        self.assertAlmostEqual(r["poses"][0]["t"], 0.3)
        self.assertEqual(r["poses"][0]["do"], {"verb": "pick_up", "object": "ball"})

    def test_norm_do_strings(self):
        N = planner.norm_do
        self.assertEqual(N("pick_up ball"), {"verb": "pick_up", "object": "ball"})
        self.assertEqual(N("put back the ball"), {"verb": "put_back", "object": "the ball"})
        self.assertEqual(N("put ball on block_a"), {"verb": "place", "object": "ball", "relation": "on", "target": "block_a"})
        self.assertEqual(N("place the hat next to gift_box"), {"verb": "place", "object": "the hat", "relation": "next_to", "target": "gift_box"})
        self.assertIsNone(N(""))
        self.assertIsNone(N({"object": "ball"}))
        self.assertEqual(N({"verb": "attach", "to": "head", "side": "right", "junk": [1]}), {"verb": "attach", "to": "head", "side": "right"})

    def test_touch_go_to_steps(self):   # objvoice
        N = planner.norm_do
        self.assertIn("touch", planner.DO_VERBS)
        self.assertIn("go_to", planner.DO_VERBS)
        self.assertEqual(N("go to board"), {"verb": "go_to", "object": "board"})
        self.assertEqual(N("walk over to tv"), {"verb": "walk_over_to", "object": "tv"})
        self.assertEqual(N("knock on desk"), {"verb": "knock_on", "object": "desk"})
        self.assertEqual(N({"verb": "touch", "object": "desk", "style": "knock", "count": 3, "side": "left"}),
                         {"verb": "touch", "object": "desk", "style": "knock", "count": 3, "side": "left"})
        r = parse('{"poses":[{"t":0.3,"do":{"verb":"touch","object":"desk","style":"pat","count":2}},'
                  '{"t":0.6,"do":{"verb":"go_to","object":"board"}},{"t":1.2,"right_paw_y":0.5}]}')
        self.assertEqual(r["unknown"], [])
        self.assertEqual(r["poses"][0]["do"], {"verb": "touch", "object": "desk", "style": "pat", "count": 2})
        self.assertEqual(r["poses"][1]["do"], {"verb": "go_to", "object": "board"})

    def test_wire_pose_carries_do(self):
        r = planner.StreamParser(planner.KF_CONTROLS + planner.KF_OBJ_CONTROLS, True)
        r.feed(0, ANSWER)
        w = [planner.wire_pose(p) for p in r.poses]
        self.assertEqual(w[0], {"t": 0.3, "do": {"verb": "pick_up", "object": "book", "side": "right"}})
        self.assertNotIn("do", w[1])

    def test_js_parser_same_semantics(self):
        """the page's kfNorm (app/motion.js) reads the same poses: same do / events / held keys, same shift"""
        js = r"""
const load = require(process.argv[1] + '/tests/load.cjs'); globalThis.AvatarMovesLib = load('app/moves.js');
const M = load('app/motion.js'); const m = M.createMotion({seed: 1, character: 'fox'}); m.setProps(null, null, 'fox');
const poses = JSON.parse(process.argv[2]).poses;
m.command({action: 'keyframes', id: 'x', description: 'x', poses, done: true, hold: true}, {src: 'tool'});
const st = m.kfState('x'); console.log(JSON.stringify({n: st.poses, unknown: st.unknown}));"""
        out = subprocess.run(["node", "-e", js, os.path.join(HERE, ".."), json.loads(json.dumps(ANSWER))],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        o = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(o["n"], 6)
        self.assertEqual(o["unknown"], [])


def replayer(text, seen):
    async def fn(http, system, user, deadline, info):
        seen.append({"system": system, "user": user})
        for _, s in chunks(text):
            yield s
        info["usage"] = {"input_tokens": 4000, "output_tokens": 300}
    return fn


async def collect(pl, ch, desc, state=None):
    out = []
    async for ev in pl.plan(ch, desc, state, http=object()):
        out.append(ev)
    return out


class Flow(unittest.TestCase):
    def setUp(self):
        self.cache = tempfile.mkdtemp(prefix="avatar-plan-cache-")
        self.pl = planner.Planner(dict(CFG, library=False), lambda m: None, cache_dir=self.cache)
        self.seen = []
        self.pl.stream_fn = replayer(ANSWER, self.seen)

    def tearDown(self):
        shutil.rmtree(self.cache, ignore_errors=True)

    def run_plan(self, ch, desc, state):
        return asyncio.run(collect(self.pl, ch, desc, state))

    def test_scene_plan_uses_appendix_and_things(self):
        evs = self.run_plan("muse", "spin the book on your finger", {"posture": "standing", "things": THINGS,
                                                                      "root": {"x": 0, "z": 0}})
        done = evs[-1]
        self.assertTrue(done["ok"], done.get("error"))
        sysm, user = self.seen[0]["system"], self.seen[0]["user"]
        muse = self.pl.prompts["muse"]["text"]
        self.assertTrue(sysm.startswith(muse.rstrip("\n")))
        self.assertIn("THINGS MUSE CAN REALLY HANDLE", sysm)
        self.assertIn("- book = blue book:", sysm)
        self.assertIn("Things now:", user)
        self.assertIn("book: on the side_table", user)
        self.assertNotIn("chair:", user)                      # furniture: the prompt's room paragraph places it
        self.assertEqual(done["prompt_sha"], hashlib.sha256(sysm.encode()).hexdigest())
        wire = done["all_poses"]
        self.assertEqual(wire[0]["do"]["verb"], "pick_up")
        self.assertEqual(wire[-1]["do"], {"verb": "put_back", "object": "book"})
        self.assertEqual(done["unknown"], [])
        # the same request + the same things: the cache; a moved thing: planned afresh
        evs2 = self.run_plan("muse", "spin the book on your finger", {"posture": "standing", "things": THINGS})
        self.assertEqual(evs2[-1]["source"], "cache")
        self.assertEqual(evs2[-1]["all_poses"], wire)
        moved = json.loads(json.dumps(THINGS))
        moved[1].update({"held": "right", "where": "right paw"})
        evs3 = self.run_plan("muse", "spin the book on your finger", {"posture": "standing", "things": moved})
        self.assertEqual(evs3[-1]["source"], "planner")
        self.assertNotEqual(evs3[-1]["cache_key"], done["cache_key"])
        self.assertIn("book: held in its right paw", self.seen[-1]["user"])

    def test_no_things_no_appendix(self):
        self.pl.stream_fn = replayer('{"summary":"x","poses":[{"t":0.3,"right_paw_y":0.5},{"t":1.2,"right_paw_y":0}]}', self.seen)
        evs = self.run_plan("muse", "wave", {"posture": "standing"})
        P = self.pl.prompts["muse"]
        self.assertEqual(self.seen[0]["system"], P["text"])           # byte-identical to the base prompt
        self.assertEqual(evs[-1]["prompt_sha"], P["sha256"])
        self.assertEqual(evs[-1]["cache_key"], planner.cache_key("muse", planner.normalize("wave"), ""))
        evs = self.run_plan("fox", "wave", {"posture": "seated", "things": THINGS})   # no props file: no scene
        self.assertEqual(self.seen[-1]["system"], self.pl.prompts["fox"]["text"])
        self.assertNotIn("Things now", self.seen[-1]["user"])

    def test_do_rejected_without_scene(self):
        evs = self.run_plan("muse", "spin the book", {"posture": "standing"})   # an old page: no things sent
        done = evs[-1]
        self.assertFalse(any("do" in p for p in done["all_poses"]))
        self.assertIn("do", done["unknown"])

    def test_library_does_not_swallow_a_thing(self):
        pl = planner.Planner(CFG, lambda m: None, cache_dir=self.cache)
        pl.stream_fn = replayer(ANSWER, self.seen)
        pl.library = lambda ch, d: ([{"action": "move", "name": "spin"}], "ok")       # a matcher reading "spin"
        evs = asyncio.run(collect(pl, "muse", "spin the book on your finger", {"things": THINGS}))
        self.assertEqual(evs[-1]["source"], "planner")
        self.assertIn("names a thing", evs[-1]["library"])
        pl.library = lambda ch, d: ([{"action": "object", "verb": "pick_up", "object": {"name": "book"}}], "ok")
        evs = asyncio.run(collect(pl, "muse", "pick up the book", {"things": THINGS}))
        self.assertEqual(evs[-1]["source"], "library")                 # a plain object step stays instant
        pl.library = lambda ch, d: ([{"action": "move", "name": "spin"}], "ok")
        evs = asyncio.run(collect(pl, "muse", "spin around", {"things": THINGS}))
        self.assertEqual(evs[-1]["source"], "library")                 # no thing named: the library as before


class Prompts(unittest.TestCase):
    def test_muse_txt_untouched_and_check_passes(self):
        with open(os.path.join(APP, "planner_prompts", "MANIFEST.json")) as f:
            man = json.load(f)
        with open(os.path.join(APP, "planner_prompts", "muse.txt"), "rb") as f:
            txt = f.read()
        self.assertEqual(hashlib.sha256(txt).hexdigest(), man["characters"]["muse"]["sha256"])
        r = subprocess.run([sys.executable, os.path.join(APP, "planner_prompts", "make_prompts.py"), "--check"],
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_appendix_compact_and_complete(self):
        a = planner.scene_appendix(planner.load_scene("muse"))
        self.assertLess(len(a), 6700)   # objvoice: +touch / go_to (~360 chars)
        for k in ('touch', 'go_to', '"style"'):
            self.assertIn(k, a)
        for k in ("right_held_spin", "right_toss_up", "right_catch", "right_let_go", "right_grab", '"do"', "put_back",
                  "FRESH START", "0.82"):
            self.assertIn(k, a)
        self.assertNotIn("{CATALOGUE}", a)
        self.assertNotIn("{VERBS}", a)


if __name__ == "__main__":
    unittest.main(verbosity=1)
