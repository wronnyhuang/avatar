"""The headphones and their stand are out of the classroom scene. The props files keep the slot-0
entry flagged "removed": true (the other props' prop{k}_* columns never move). For the REAL scene:
  hub   — intent_hub.scene_for, planner.load_scene, semantic.targets, app.char_scene / load_char_files (animals + the mesh
          path for junor, and a mascot) carry no headphones and no stand; every other prop is still there;
  render — props_defs.defines for muse, junor and a mascot: PROPS_N 16, slot 0 kind 0, no PROPKIND_HEADPHONES, the
          side table is a plain table (PLACE kind 3, no post), every other slot keeps its kind and column vs the
          pre-removal scene (tests/fixtures/muse_props_with_headphones.json).
The page side (moves.js / the matcher / the voice tools) is tests/scene_removed_prop_test.cjs.
    python tests/scene_removed_prop_test.py   (no GPU, no network)"""
import json
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
APP = os.path.join(ROOT, "app")
REN = os.path.join(ROOT, "renderer")
FIXTURE = os.path.join(HERE, "fixtures", "muse_props_with_headphones.json")
sys.path[:0] = [APP, REN, os.path.join(REN, "engine")]
for k in ("AVATAR_PROPS_JSON", "AVATAR_PROPS_FOR", "AVATAR_NO_PROPS"):
    os.environ.pop(k, None)

GONE = {"headphones"}
STAND_WORDS = re.compile(r"headphone|headset|\bstand\b", re.I)
MASCOT = "gekkie"


def fixture():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def kept_slots():
    """id -> slot of every prop of the pre-removal scene except the headphones (chair 1 ... eraser 15)"""
    return {p["id"]: p["slot"] for p in fixture()["props"] if p["id"] not in GONE}


def strip_doc(p):
    """a prop entry without its free-text note keys (named differently in the two files; not scene data)"""
    return {k: v for k, v in p.items() if k and not k.startswith("_") and not re.match(r"t\d+_", k)}


def ids(cfg):
    return [p["id"] for p in (cfg or {}).get("props", [])]


class Fixture(unittest.TestCase):
    def test_fixture_is_the_pre_removal_scene(self):
        fx = fixture()
        hp = [p for p in fx["props"] if p["id"] == "headphones"]
        self.assertEqual(len(hp), 1)
        self.assertEqual(hp[0]["slot"], 0)
        self.assertNotIn("removed", hp[0])
        st = [p for p in fx["places"] if p["id"] == "side_table"][0]
        self.assertEqual(st["kind"], "table_stand")
        self.assertIn("stand_top_y", st)

    def test_slots(self):
        ks = kept_slots()
        self.assertEqual(ks["chair"], 1)
        self.assertEqual(ks["eraser"], 15)
        self.assertEqual(sorted(ks.values()), list(range(1, 16)))


class RealFiles(unittest.TestCase):
    """the files themselves: slot 0 kept and flagged, every other entry unchanged"""
    def test_props_files(self):
        fx = {p["id"]: p for p in fixture()["props"]}
        for name in ("muse", "junor"):
            with self.subTest(file=name):
                with open(os.path.join(REN, "animals", f"{name}.props.json"), encoding="utf-8") as f:
                    cfg = json.load(f)
                hp = [p for p in cfg["props"] if p["id"] == "headphones"]
                self.assertEqual(len(hp), 1)
                self.assertTrue(hp[0].get("removed"))
                self.assertEqual(hp[0]["slot"], 0)
                for p in cfg["props"]:
                    if p["id"] not in GONE:
                        self.assertEqual(strip_doc(p), strip_doc(fx[p["id"]]), p["id"])
                st = [p for p in cfg["places"] if p["id"] == "side_table"][0]
                self.assertEqual(st["kind"], "table")
                self.assertNotIn("stand_top_y", st)
                self.assertFalse(any(STAND_WORDS.search(a) for a in st.get("aliases", [])), st.get("aliases"))


class Hub(unittest.TestCase):
    def check_scene(self, cfg, label):
        self.assertTrue(cfg and isinstance(cfg.get("props"), list), label)
        got = ids(cfg)
        self.assertFalse(GONE & set(got), f"{label}: {got}")
        self.assertEqual(set(got), set(kept_slots()), label)
        self.assertFalse(any(p.get("removed") for p in cfg["props"]), label)
        for p in cfg["props"]:
            self.assertEqual(p["slot"], kept_slots()[p["id"]], f"{label}: {p['id']}")
        st = [p for p in cfg.get("places", []) if p.get("id") == "side_table"]
        self.assertTrue(st, label)
        self.assertNotEqual(st[0].get("kind"), "table_stand", label)
        self.assertNotIn("stand_top_y", st[0], label)

    def test_intent_hub(self):
        import intent_hub
        for c in ("muse", "junor", MASCOT):
            with self.subTest(character=c):
                sc = intent_hub.scene_for(c)
                if c == "muse":
                    self.check_scene(sc, f"intent_hub.scene_for({c})")
                elif sc is not None:
                    self.assertFalse(GONE & set(ids(sc)), c)
        m = intent_hub.matcher_for("muse")
        self.assertTrue(m.parse("pull the chair up to the desk"))

    def test_planner(self):
        import planner
        self.check_scene(planner.load_scene("muse"), "planner.load_scene(muse)")
        # an explicit scene dir with the pre-removal file still loads it whole (the loader drops only flagged props)
        import shutil
        import tempfile
        d = tempfile.mkdtemp()
        try:
            shutil.copy(FIXTURE, os.path.join(d, "muse.props.json"))
            self.assertIn("headphones", ids(planner.load_scene("muse", d)))
        finally:
            shutil.rmtree(d)

    def test_semantic_targets(self):
        from semantic import targets
        for c in ("muse",):
            with self.subTest(character=c):
                sc = targets._scene(c)
                self.assertIsNotNone(sc, c)
                self.assertFalse(any("headphone" in str(k).lower() for k in json.dumps(sc, default=str).split('"')), c)
                names = targets.scene_names(c)
                self.assertFalse(any("headphone" in str(n).lower() for n in names), names)

    def test_app_char_scene(self):
        import app
        for c in ("muse", "junor", MASCOT):
            with self.subTest(character=c):
                app._SCENE_CACHE.pop(c, None)
                s = app.char_scene(c)
                self.check_scene(s.get("props"), f"app.char_scene({c})")
                f = app.load_char_files({}, c)
                self.check_scene(f.get("props"), f"app.load_char_files({c})")
        self.check_scene(app._mesh_files(REN, "junor").get("props"), "app._mesh_files(junor)")


def const_array(src, name):
    m = re.search(r"const (\w+) %s\[(\d+)\] = \w+\[\d+\]\((.*?)\);\n" % name, src)
    assert m, name
    return m.group(1), int(m.group(2)), [v.strip() for v in re.split(r",(?![^()]*\))", m.group(3))]


class Renderer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import props_defs
        cls.pd = props_defs
        cls.out = {c: props_defs.defines(c) for c in ("muse", "junor", MASCOT)}
        os.environ["AVATAR_PROPS_JSON"] = FIXTURE           # the pre-removal scene through the same code (muse only)
        os.environ["AVATAR_PROPS_FOR"] = "muse"
        try:
            cls.before = props_defs.defines("muse")
        finally:
            os.environ.pop("AVATAR_PROPS_JSON", None)
            os.environ.pop("AVATAR_PROPS_FOR", None)

    def test_defines(self):
        kinds0 = const_array(self.before[1], "PR_KIND")[2]
        px0 = const_array(self.before[1], "PR_PX")[2]
        self.assertNotEqual(kinds0[0], "0")                 # the fixture really had the headphones model
        self.assertIn("#define PROPKIND_HEADPHONES 1", self.before[1])
        for c, (ph, out) in self.out.items():
            with self.subTest(character=c):
                self.assertIn("#define PROPS_N 16\n", out)
                _, n, kinds = const_array(out, "PR_KIND")
                self.assertEqual(n, 16)
                self.assertEqual(kinds[0], "0")
                self.assertNotIn("PROPKIND_HEADPHONES", out)
                self.assertNotIn("PROPS_HP_W", out)
                _, _, px = const_array(out, "PR_PX")
                self.assertEqual(px, px0, "prop{k}_x columns moved")
                if c in ("muse", MASCOT, "junor"):        # same props file as the fixture apart from slot 0
                    self.assertEqual(kinds[1:], kinds0[1:])
                _, _, free = const_array(out, "PR_FREE")
                self.assertEqual(free[0], "1")
                # the side table: a plain table (kind 3), no post
                i = self.side_table_index(ph)
                self.assertEqual(re.search(r"#define PLACE%d_KIND (\d+)" % i, ph).group(1), "3", ph)
                self.assertEqual(float(re.search(r"#define PLACE%d_POST (\S+)" % i, ph).group(1)), 0.0, ph)
                self.assertNotIn(" %d\n" % self.pd.PLACE_KINDS["table_stand"], "".join(re.findall(r"#define PLACE\d+_KIND \d+\n", ph)))
        i0 = self.side_table_index(self.before[0])
        self.assertEqual(re.search(r"#define PLACE%d_KIND (\d+)" % i0, self.before[0]).group(1), str(self.pd.PLACE_KINDS["table_stand"]))
        self.assertGreater(float(re.search(r"#define PLACE%d_POST (\S+)" % i0, self.before[0]).group(1)), 0.1)

    def side_table_index(self, ph):
        """the side table's PLACE index: the places the renderer draws, in file order (props_defs.defines)"""
        places = [p for p in fixture()["places"] if p.get("kind") in self.pd.PLACE_KINDS and not p.get("movable")][:4]
        i = [p["id"] for p in places].index("side_table")
        self.assertIn("#define PLACE%d_KIND" % i, ph)
        return i


if __name__ == "__main__":
    unittest.main()
