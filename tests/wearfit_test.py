#!/usr/bin/env python3
"""a character with its own head carries its own worn-prop fit (animals/<c>.props.override.json, written
by renderer/tools/wear_fit.py), merged over the shared scene file by the hub loader and the renderer; the Muse and the
mesh characters are untouched; AVATAR_WEARFIT=0 turns it off.   python tests/wearfit_test.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); TREE = os.path.dirname(HERE)
R = os.path.join(TREE, "renderer")
sys.path[:0] = [R, os.path.join(R, "engine"), os.path.join(TREE, "app")]
os.environ.setdefault("AVATAR_HOME", "/tmp/wearfit_test_avhome")
os.environ.pop("AVATAR_WEARFIT", None); os.environ.pop("AVATAR_PROPS_JSON", None)
import props_defs
n = [0, 0]


def ok(c, name, info=""):
    n[0 if c else 1] += 1
    print(("PASS " if c else "FAIL ") + name + (f"  {info}" if info and not c else ""))


def wear(cfg, pid):
    return next(p for p in cfg["props"] if p.get("id") == pid)["afford"]["wear"]


shared = json.load(open(os.path.join(R, "animals", "muse.props.json")))
FIT = ("gekkie", "hooty", "doughlad", "bonbon", "pneu", "goober", "jade", "ruby", "flurry")
ok(props_defs.load("muse") == shared, "muse: the shared scene file exactly")
ok("PROPS_WEARFIT" not in props_defs.defines("muse")[1], "muse: no worn-fit path compiled")
for c in FIT:
    cfg = props_defs.load(c)
    h, g = wear(cfg, "hat"), wear(cfg, "glasses")
    ok(len(cfg["props"]) == len(shared["props"]) and [p.get("slot") for p in cfg["props"]] == [p.get("slot") for p in shared["props"]],
       f"{c}: same props, same slots")
    ok(0.2 <= h.get("scale", 0) <= 1.0 and 0.2 <= g.get("scale", 0) <= 1.5 and len(h["offset"]) == 3 and "carry" in h,
       f"{c}: hat + glasses carry offset / scale / carry", (h, g))
    ok(h["rot"] == wear(shared, "hat")["rot"], f"{c}: the shared worn tilt is kept")
    d = props_defs.defines(c)[1]
    ok(("#define PROPS_WEARFIT 1" in d) == (abs(h["scale"] - 1) > 1e-6 or abs(g["scale"] - 1) > 1e-6), f"{c}: PROPS_WEARFIT iff a scale differs")
    rest = [(p if p.get("id") not in ("hat", "glasses") else None) for p in cfg["props"]]
    ok(rest == [(p if p.get("id") not in ("hat", "glasses") else None) for p in shared["props"]], f"{c}: every other prop untouched")
# the hub loader gives the page the same numbers
import app
cfgd = {"local": {"render_root": R}}
for c in ("gekkie", "jade"):
    f = app.load_char_files(cfgd, c)
    ok(wear(f["props"], "hat") == wear(props_defs.load(c), "hat"), f"{c}: hub loader = renderer")
    ok(all(not p.get("removed") for p in f["props"]["props"]), f"{c}: removed props still hidden from the page")
fm = app.load_char_files(cfgd, "muse")
ok(wear(fm["props"], "hat") == wear(shared, "hat") and "carry" not in wear(fm["props"], "hat"), "muse: hub loader unchanged")
fj = app.load_char_files(cfgd, "junor")
ok("scale" not in wear(fj["props"], "hat"), "junor (mesh): her own override only")
# the switch
os.environ["AVATAR_WEARFIT"] = "0"
ok(props_defs.load("gekkie") == shared and "PROPS_WEARFIT" not in props_defs.defines("gekkie")[1], "AVATAR_WEARFIT=0: renderer back on the shared fit")
ok(wear(app.load_char_files(cfgd, "gekkie")["props"], "hat") == wear(shared, "hat"), "AVATAR_WEARFIT=0: hub loader back on the shared fit")
print(f"{n[0]} passed, {n[1]} failed")
sys.exit(1 if n[1] else 0)
