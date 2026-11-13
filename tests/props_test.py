"""OBJECTS renderer: the props tables for the muse's new props (slots 8-13) and the 16-slot limit.
    python3 tests/props_test.py      (no GPU, no render: the generated GLSL)"""
import json
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REN = os.path.join(HERE, "..", "renderer")
sys.path[:0] = [REN, os.path.join(REN, "engine")]
os.environ.pop("AVATAR_PROPS_JSON", None)
os.environ.pop("AVATAR_NO_PROPS", None)
import props_defs  # noqa: E402

NEW = {8: ("hat", "hat", 30), 9: ("glasses", "glasses", 31), 10: ("gift_box", "box", 32),
       11: ("watering_can", "watering_can", 33), 12: ("block_a", "block", 34), 13: ("block_b", "block", 34)}
OTHERS = ["bear", "blob", "cube", "drop", "fox", "hippo", "monkey", "orb", "pyramid", "rabbit", "star"]


def const_array(src, name):
    m = re.search(r"const (\w+) %s\[(\d+)\] = \w+\[\d+\]\((.*?)\);\n" % name, src)
    assert m, name
    return m.group(1), int(m.group(2)), m.group(3)


class MuseProps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.load(open(os.path.join(REN, "animals", "muse.props.json")))
        cls.place, cls.head = props_defs.defines("muse")
        cls.n = len(cls.cfg["props"])

    def test_count(self):
        self.assertGreaterEqual(self.n, 14)
        self.assertLessEqual(self.n, 16)
        self.assertIn(f"#define PROPS_N {self.n}\n", self.head)

    def test_slots_and_kinds(self):
        by_slot = {p["slot"]: p for p in self.cfg["props"]}
        _, n, kinds = const_array(self.head, "PR_KIND")
        kinds = [int(v) for v in kinds.split(",")]
        self.assertEqual(n, self.n)
        for s, (pid, kind, pk) in NEW.items():
            self.assertEqual(by_slot[s]["id"], pid)
            self.assertEqual(by_slot[s]["kind"], kind)
            self.assertEqual(props_defs.CATALOG[kind][0], pk)
            self.assertEqual(kinds[s], pk)
            self.assertIn(f"#define PROPKIND_{kind.upper()} 1", self.head)
        for k in ("hat", "glasses", "box", "watering_can", "block"):
            self.assertTrue(30 <= props_defs.CATALOG[k][0] <= 39)

    def test_tables_rows(self):
        tab = props_defs.tables("muse")
        for name in ("PR_PX", "PR_KIND", "PR_HOME", "PR_SIZE", "PR_BND", "PR_BND2", "PR_BBC2", "PR_BBH2", "PR_VAR", "PR_GA"):
            self.assertIn(name, tab)
            self.assertEqual(len(tab[name][1]), self.n, name)
            self.assertIn(f"uniform {tab[name][0]} {name}[{self.n}];", self.head)
        # the rig columns of slots 8-13 (anim/schema.py appended prop8..prop15)
        from anim.schema import IDX
        px = tab["PR_PX"][1]
        for s in NEW:
            self.assertEqual(px[s], IDX[f"prop{s}_x"])
        # per-slot letters on the blocks; the articulated bounds reach further than shut where the model does
        var = tab["PR_VAR"][1]
        self.assertEqual((var[12], var[13]), (ord("A"), ord("B")))
        bnd, bnd2 = tab["PR_BND"][1], tab["PR_BND2"][1]
        for s in (9, 10, 11):
            self.assertGreater(bnd2[s], bnd[s])
        for s in range(8):
            self.assertEqual(bnd2[s], bnd[s])

    def test_json_numbers(self):
        by_id = {p["id"]: p for p in self.cfg["props"]}
        self.assertEqual(by_id["glasses"]["home"]["on"], "gift_box")
        self.assertEqual(by_id["block_b"]["home"]["on"], "block_a")
        for pid in ("hat", "glasses"):
            self.assertEqual(len(by_id[pid]["afford"]["wear"]["offset"]), 3)
        box = by_id["gift_box"]["afford"]
        self.assertLess(box["container"]["floor_y"], 0.0)
        self.assertGreater(box["surface"]["top_y"], 0.0)
        # stacked homes agree with the surfaces: block_b's centre = block_a's top + its half height
        a, b = by_id["block_a"], by_id["block_b"]
        self.assertAlmostEqual(b["home"]["pos"][1], a["home"]["pos"][1] + a["afford"]["surface"]["top_y"] + b["afford"]["surface"]["top_y"], places=3)
        g = by_id["glasses"]
        self.assertAlmostEqual(g["home"]["pos"][1], by_id["gift_box"]["home"]["pos"][1] + box["surface"]["top_y"] + 0.30*g["scale"], places=3)

    def test_homes_clear_of_lanes(self):
        """the new props' floor homes keep out of the muse's straight walks (home -> every station / its line / aside,
        home -> every pick_up_from.stand_at, a step forward) and out of each other: body half width + footprint."""
        import math
        body = json.load(open(os.path.join(REN, "animals", "muse.body.json")))["extents"]["x"]
        P = {p["id"]: p for p in self.cfg["props"]}

        def foot(p):
            c, h = props_defs.BBOX[p["kind"]]
            return p["scale"]*(max(h[0], h[2]) + math.hypot(c[0], c[2]))

        def segd(p, a, b):
            dx, dz = b[0] - a[0], b[1] - a[1]; L = dx*dx + dz*dz
            t = max(0.0, min(1.0, ((p[0] - a[0])*dx + (p[1] - a[1])*dz)/L)) if L else 0.0
            return math.hypot(p[0] - a[0] - t*dx, p[1] - a[1] - t*dz)
        O, lanes = (0.0, 0.0), [("walk forward", (0.0, 0.0), (0.0, 0.6))]
        for pl in self.cfg["places"]:
            st = pl.get("station")
            if not st:
                continue
            pts = [st["pos"]] + ([st["line"]["from"], st["line"]["to"]] if st.get("line") else []) + ([st["aside"]] if st.get("aside") else [])
            lanes += [(pl["id"] + " station", O, (q[0], q[2])) for q in pts]
            if st.get("line"):
                lanes.append((pl["id"] + " line", (st["line"]["from"][0], st["line"]["from"][2]), (st["line"]["to"][0], st["line"]["to"][2])))
        for p in self.cfg["props"]:
            sa = p.get("pick_up_from", {}).get("stand_at")
            if sa:
                lanes.append((p["id"] + " stand_at", O, (sa[0], sa[2])))
        mine = [P[NEW[s][0]] for s in NEW if "on" not in P[NEW[s][0]]["home"]]
        for p in mine:
            x, _, z = p["home"]["pos"]
            for n, a, b in lanes:
                self.assertGreaterEqual(segd((x, z), a, b) - body - foot(p), 0.0, (p["id"], n))
        for i, p in enumerate(mine):
            for q in mine[i + 1:]:
                d = math.hypot(p["home"]["pos"][0] - q["home"]["pos"][0], p["home"]["pos"][2] - q["home"]["pos"][2])
                self.assertGreater(d, foot(p) + foot(q), (p["id"], q["id"]))

    def test_sixteen_slot_limit(self):
        src = open(os.path.join(REN, "engine", "props.glsl")).read()
        self.assertIn("int gPrMask = 0xFFFF;", src)
        for arr in ("gPrC[16]", "gPrM[16]", "gPrArt[16]", "gPrRest[16]", "uPrC[16]", "uPrM[16]", "uPrAR[16]"):
            self.assertIn(arr, src)
        self.assertNotIn("0xFF;", open(os.path.join(REN, "engine", "scene_main.glsl")).read())
        import rt_engine
        self.assertEqual(rt_engine.PROPS_SLOTS, 16)


class OtherCharacters(unittest.TestCase):
    def test_no_props(self):
        for c in OTHERS:
            place, head = props_defs.defines(c)
            self.assertEqual((place, head), ("", "#define PROPS_N 0\n"), c)


if __name__ == "__main__":
    unittest.main(verbosity=2)
