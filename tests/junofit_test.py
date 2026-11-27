#!/usr/bin/env python3
"""the mesh character's (junor) props override keeps a rescaled prop's GRIP ANCHOR with its
size.  props_defs turns grip_point_local (H) into the unit-frame anchor by / scale: an override that changes a prop's
scale without its grip_point_local leaves the anchor at the Muse's size (the held hat floated 0.5 scene units off
Juno's hand).  Rule checked for EVERY prop the override rescales: unit anchor = the shared file's unit anchor; and the
worn hat is sized to her hair, not her skull (brim wider than the old skull fit).   python tests/junofit_test.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); TREE = os.path.dirname(HERE)
R = os.path.join(TREE, "renderer")
sys.path[:0] = [R, os.path.join(R, "engine")]
os.environ.pop("AVATAR_PROPS_JSON", None)
from humans import charfiles
import props_defs
n = [0, 0]


def ok(c, name, info=""):
    n[0 if c else 1] += 1
    print(("PASS " if c else "FAIL ") + name + (f"  {info}" if info and not c else ""))


shared = {p["id"]: p for p in json.load(open(os.path.join(R, "animals", "muse.props.json")))["props"]}
for c in ("junor",):          # (juno has no scene in this repository)
    cfg = charfiles.props_for(c)
    ok(props_defs.load(c) == cfg, f"{c}: the renderer loads the merged override")
    by = {p["id"]: p for p in cfg["props"]}
    for pid, p in by.items():
        m = shared[pid]
        if "grip_point_local" not in m or m.get("kind") == "headphones" or p.get("removed"):
            continue          # (headphones: the anchor comes from the fit)
        ua = [v/p["scale"] for v in p["grip_point_local"]]; um = [v/m["scale"] for v in m["grip_point_local"]]
        ok(max(abs(a - b) for a, b in zip(ua, um)) < 0.02, f"{c}: {pid} grip anchor rides its scale (unit frame = shared)", (ua, um))
    hat, gl = by["hat"], by["glasses"]
    ok(1.9 <= hat["scale"]/gl["scale"] <= 2.6, f"{c}: hat sized over the hair (brim / glasses width)", hat["scale"]/gl["scale"])
    ok(abs(hat["size"]/hat["scale"] - shared["hat"]["size"]/shared["hat"]["scale"]) < 0.02, f"{c}: hat page size follows its scale")
    ho, go = hat["afford"]["wear"]["offset"], gl["afford"]["wear"]["offset"]
    ok(0.4*hat["scale"] <= ho[1] <= 0.7*hat["scale"], f"{c}: hat sits on the hair (offset y / scale)", ho[1]/hat["scale"])
    ok(go[2] >= 1.9*gl["scale"], f"{c}: glasses in front of the mesh face (offset z / scale)", go[2]/gl["scale"])
print(f"{n[0]} passed, {n[1]} failed")
sys.exit(1 if n[1] else 0)
