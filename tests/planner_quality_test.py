"""quality, hub side (app/planner.py + app/planner_prompts/make_prompts.py VARIANTS). No paid calls.
    python tests/planner_quality_test.py
The q1 prompt variant: generated (make_prompts --check), served to the Muse and the characters on its hand map only (Juno
keeps the frozen prompt; the seated characters are untouched), switched off by config / env; the plan check: what it calls
static / short / held_tiny, that it gates the plan cache both ways (write and read) and is switched off by config / env;
tools/kf_reach.py agrees with the numbers the prompt quotes."""
import asyncio
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
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
os.environ["AVATAR_HOME"] = tempfile.mkdtemp(prefix="planq-home-")
for _k in ("AVATAR_PLANNER_MODEL", "AVATAR_PLANNER_FALLBACK", "AVATAR_PLANNER_FALLBACK_AFTER_S", "AVATAR_PLANNER_PROMPT",
           "AVATAR_PLANNER_CHECK"):
    os.environ.pop(_k, None)
import planner  # noqa: E402
import kf_reach  # noqa: E402

CFG = {"model": "sol", "fallback": None, "library": False}
GOOD = '{"summary":"wave","poses":[{"t":0.3,"right_paw_x":0,"right_paw_y":1.2},{"t":0.8,"right_paw_x":1.0,"right_paw_y":0.8},{"t":1.3,"right_paw_x":0,"right_paw_y":1.2}]}'
STATIC = '{"summary":"hm","poses":[{"t":0.3,"right_paw_x":-0.1,"smile":0.3},{"t":1.5,"right_paw_x":0.1,"head_roll":0.05}]}'


def chunks(text, n=9):
    return [text[i:i + n] for i in range(0, len(text), n)]


def replayer(text, seen):
    async def fn(http, system, user, deadline, info):
        seen.append({"system": system, "user": user})
        for s in chunks(text):
            yield s
        info["usage"] = {"input_tokens": 4000, "output_tokens": 300}
    return fn


def run(pl, ch, desc):
    async def go():
        out = []
        async for ev in pl.plan(ch, desc, None, http=object()):
            out.append(ev)
        return out
    return asyncio.run(go())


class Variant(unittest.TestCase):
    def test_generated_and_checked(self):
        r = subprocess.run([sys.executable, os.path.join(APP, "planner_prompts", "make_prompts.py"), "--check"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        with open(os.path.join(APP, "planner_prompts", "MANIFEST.json")) as f:
            man = json.load(f)
        v = man["variants"]["muse"]["q1"]
        self.assertEqual(v["controls"], man["characters"]["muse"]["controls"])
        self.assertIn("muse", v["for"])
        self.assertNotIn("juno", v["for"])
        self.assertNotIn("junor", v["for"])
        self.assertNotIn("variants", man["characters"])            # the 12 characters' entries are as they were

    def test_served_to_the_muse_hand_map_only(self):
        pl = planner.Planner(CFG, lambda m: None)
        fr = planner.Planner(dict(CFG, prompt_variant="frozen"), lambda m: None)
        self.assertEqual(pl.variant, "q1")
        self.assertEqual(fr.variant, "")
        self.assertNotEqual(pl.prompts["muse"]["sha256"], fr.prompts["muse"]["sha256"])
        for c in ("juno", "junor", "fox", "bear", "orb", "blob"):
            if c in fr.prompts:
                self.assertEqual(pl.prompts[c]["sha256"], fr.prompts[c]["sha256"], c)
        import charcaps
        for c, a in charcaps.ALIAS.items():
            if a == "muse" and c not in ("juno", "junor"):
                self.assertEqual(pl.prompts[c]["sha256"], pl.prompts["muse"]["sha256"], c)     # the mascots
                self.assertEqual(fr.prompts[c]["sha256"], fr.prompts["muse"]["sha256"], c)
        self.assertTrue(pl.prompts["muse"]["ok"])
        t = pl.prompts["muse"]["text"]
        for name in ("right_point", "right_thumb", "PROFILE", "x -0.9"):
            self.assertIn(name, t)
        # nothing the parser would drop is named, and the format lines are the frozen ones
        self.assertTrue(t.startswith(fr.prompts["muse"]["text"][:400]))

    def test_switches(self):
        for off in ("", "frozen", False, None, "off"):
            self.assertEqual(planner.Planner(dict(CFG, prompt_variant=off), lambda m: None).variant, "", repr(off))
        os.environ["AVATAR_PLANNER_PROMPT"] = "frozen"
        try:
            self.assertEqual(planner.Planner(CFG, lambda m: None).variant, "")
        finally:
            del os.environ["AVATAR_PLANNER_PROMPT"]
        logs = []
        pl = planner.Planner(dict(CFG, prompt_variant="nope"), logs.append)       # an unknown variant: the frozen prompt
        self.assertEqual(pl.prompts["muse"]["sha256"],
                         planner.Planner(dict(CFG, prompt_variant="frozen"), lambda m: None).prompts["muse"]["sha256"])

    def test_the_system_prompt_sent(self):
        seen = []
        pl = planner.Planner(dict(CFG, cache=False), lambda m: None)
        pl.stream_fn = replayer(GOOD, seen)
        ev = run(pl, "muse", "wave at me")[-1]
        self.assertTrue(ev["ok"])
        self.assertEqual(seen[0]["system"], pl.prompts["muse"]["text"])
        self.assertEqual(ev["prompt_sha"], pl.prompts["muse"]["sha256"])


class Check(unittest.TestCase):
    def test_issues(self):
        P = planner.plan_issues
        self.assertEqual(P(json.loads(GOOD)["poses"]), [])
        self.assertEqual(P(json.loads(STATIC)["poses"]), ["static"])
        self.assertEqual(P([{"t": 0.2, "root_yaw": 1}]), ["short"])
        self.assertEqual(P([]), ["static", "short"])
        self.assertEqual(P([{"t": 0.3, "root_y": 0.5}, {"t": 1.2, "root_y": 0}]), [])
        self.assertEqual(P([{"t": 0.3, "smile": 1, "eye_wide": 0.8}, {"t": 2.0, "smile": 0.9}]), [])       # a face is a move
        self.assertEqual(P([{"t": 0.3, "body_lean": 0.2}, {"t": 2.0, "body_lean": 0.25}]), ["static"])     # 3 degrees
        tiny = [{"t": 0.3, "right_paw_x": 0.4, "right_paw_y": 0.4}] + \
               [{"t": 0.5 + 0.2 * i, "right_paw_x": 0.4 + (0.1 if i % 2 else 0)} for i in range(8)]
        self.assertEqual(P(tiny), ["held_tiny"])
        self.assertEqual(P([{"t": 0.3, "do": {"verb": "pick_up", "object": "ball"}}]), [])                 # scene steps: not judged
        self.assertEqual(P([{"t": 0.3, "right_grab": 1}, {"t": 1, "right_held_y": 0.1}]), [])

    def test_limb_hidden(self):                                
        P = planner.plan_issues
        kick = {"right_foot_y": 0.15, "right_foot_z": 0.15}
        hold = lambda **c: [{"t": 0.4, **c}, {"t": 2.4, **c}]
        self.assertEqual(P(hold(**kick)), ["limb_hidden"])                               # front-on: behind the belly
        self.assertEqual(P(hold(root_yaw=0.25, **kick)), ["limb_hidden"])                # 0.25 shows the back
        self.assertEqual(P(hold(root_yaw=-0.15, **kick)), [])                            # three-quarter: seen
        self.assertEqual(P(hold(right_foot_x=-0.18, left_foot_x=0.18, root_y=-0.2)), [])  # sideways: seen front-on
        self.assertEqual(P(hold(right_paw_x=-0.5, right_paw_y=0.15, right_paw_z=-1)), ["limb_hidden"])
        self.assertEqual(P(hold(root_pitch=0.25, **kick)), [])                           # lying: not judged
        self.assertNotIn("limb_hidden", planner.CHECK_HARD)                              # reported, never a cache gate

    def test_q2_variant(self):
        q2 = planner.Planner(dict(CFG, prompt_variant="q2"), lambda m: None)
        q1 = planner.Planner(CFG, lambda m: None)
        self.assertEqual(q2.variant, "q2")
        self.assertTrue(q2.prompts["muse"]["ok"])
        t2, t1 = q2.prompts["muse"]["text"], q1.prompts["muse"]["text"]
        self.assertIn("THREE-QUARTER", t2)
        self.assertNotIn("THREE-QUARTER", t1)                                            # q1 is unchanged
        self.assertTrue(t2.startswith(t1[:400]))

    def test_cache_gate(self):
        d = tempfile.mkdtemp(prefix="planq-cache-")
        try:
            seen = []
            pl = planner.Planner(CFG, lambda m: None, cache_dir=d)
            pl.stream_fn = replayer(STATIC, seen)
            ev = run(pl, "muse", "do the thing")[-1]
            self.assertTrue(ev["ok"])                       # the plan still plays: the check never changes or stops a plan
            self.assertEqual(ev["issues"], ["static"])
            self.assertEqual(os.listdir(d), [])             # ... but it is not kept
            ev = run(pl, "muse", "do the thing")[-1]
            self.assertEqual(ev["source"], "planner")
            pl.stream_fn = replayer(GOOD, seen)
            ev = run(pl, "muse", "wave at me")[-1]
            self.assertNotIn("issues", ev)
            self.assertEqual(len(os.listdir(d)), 1)
            self.assertEqual(run(pl, "muse", "wave at me")[-1]["source"], "cache")
            # an OLD entry today's check rejects is not served (re-scored on read), and is replaced by the new plan
            off = planner.Planner(dict(CFG, check=False), lambda m: None, cache_dir=d)
            off.stream_fn = replayer(STATIC, seen)
            self.assertNotIn("issues", run(off, "muse", "another thing")[-1])
            self.assertEqual(len(os.listdir(d)), 2)
            self.assertEqual(run(off, "muse", "another thing")[-1]["source"], "cache")       # check off: as before
            pl.stream_fn = replayer(GOOD, seen)
            ev = run(pl, "muse", "another thing")[-1]
            self.assertEqual(ev["source"], "planner")
            self.assertEqual(pl.stats.get("cache_rejected"), 1)
            self.assertEqual(run(pl, "muse", "another thing")[-1]["source"], "cache")
            # the prompt variant is part of what a cache entry was written for: the frozen planner does not serve q1 plans
            fr = planner.Planner(dict(CFG, prompt_variant="frozen"), lambda m: None, cache_dir=d)
            fr.stream_fn = replayer(GOOD, seen)
            self.assertEqual(run(fr, "muse", "wave at me")[-1]["source"], "planner")
            os.environ["AVATAR_PLANNER_CHECK"] = "0"
            try:
                self.assertFalse(planner.check_on({"check": True}))
            finally:
                del os.environ["AVATAR_PLANNER_CHECK"]
        finally:
            shutil.rmtree(d, ignore_errors=True)


class Reach(unittest.TestCase):
    """the prompt's numbers against the port of the shader's arm block"""

    def test_midline_and_reach(self):
        old = kf_reach.fk(-0.44, 0.45, 0.45)["wrist"]          # the frozen prompt's "paws meeting for a clap"
        new = kf_reach.fk(-0.9, 0.5, 0.75)["wrist"]
        self.assertGreater(old[0], 0.3)                         # 0.6+ apart: never met
        self.assertLess(new[0], 0.13)
        hip = kf_reach.fk(-0.1, 0.15, 0.05)
        self.assertFalse(hip["on_head"])
        self.assertLess(hip["short"], 0.01)
        up = kf_reach.fk(0.6, 1.2, 0.1)["wrist"]
        self.assertLess(up[1], 1.64)                            # "as high as it goes" is below the head's centre
        a, b = kf_reach.fk(0.0, 1.2, 0.0)["wrist"], kf_reach.fk(1.0, 0.8, 0.0)["wrist"]
        self.assertGreater(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5, 0.3)      # the wave's two ends are apart
        with open(os.path.join(HERE, "..", "renderer", "animals", "muse.body.json")) as f:
            hm = json.load(f)["hand_map"]
        self.assertEqual(tuple(hm["rest_wrist_scene"]), kf_reach.REST)
        self.assertEqual(hm["L_scene"], kf_reach.L)


if __name__ == "__main__":
    unittest.main(verbosity=1)
