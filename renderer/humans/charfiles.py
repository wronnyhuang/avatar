"""rig: a character's body / props JSON with the per-character OVERRIDE of the shared scene file applied at load
time (never a copied props file: muse.props.json stays the one source edits).

  props_for("juno")  -> deep_merge(muse.props.json, renderer/humans/juno/juno.props.override.json)
  props_for("muse")  -> renderer/animals/muse.props.json exactly as before
  body_for("juno")   -> renderer/humans/juno/juno.body.json

Merge rules (= app.deep_merge for dicts) plus: a list of objects that all carry "id" (places, props) merges BY id
(an override entry with an unknown id is appended); any other list is replaced.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RENDER_ROOT = os.path.dirname(HERE)
ANIMALS = os.path.join(RENDER_ROOT, "animals")


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def merge(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = merge(out[k], v) if k in out else v
        return out
    if (isinstance(a, list) and isinstance(b, list) and a and all(isinstance(x, dict) and "id" in x for x in a)
            and all(isinstance(x, dict) and "id" in x for x in b)):
        out = [dict(x) for x in a]
        pos = {x["id"]: i for i, x in enumerate(out)}
        for x in b:
            if x["id"] in pos:
                out[pos[x["id"]]] = merge(out[pos[x["id"]]], x)
            else:
                out.append(x)
        return out
    return b


def override_path(character):
    """humans/<c>/<c>.props.override.json (a mesh character), else animals/<c>.props.override.json (a
    character on the Muse skeleton with its own head: the worn props' fit, tools/wear_fit.py)."""
    p = os.path.join(HERE, character, f"{character}.props.override.json")
    return p if os.path.isfile(p) else os.path.join(ANIMALS, f"{character}.props.override.json")


def props_for(character):
    """The props/scene dict for a character, or None."""
    op = override_path(character)
    if os.path.isfile(op):
        ov = _load(op)
        base = _load(os.path.join(ANIMALS, f"{ov.get('base', 'muse')}.props.json"))
        ov = {k: v for k, v in ov.items() if k not in ("base", "_doc")}
        return merge(base, ov)
    p = os.path.join(ANIMALS, f"{character}.props.json")
    return _load(p) if os.path.isfile(p) else None


def body_for(character):
    for p in (os.path.join(HERE, character, f"{character}.body.json"), os.path.join(ANIMALS, f"{character}.body.json")):
        if os.path.isfile(p):
            return _load(p)
    return None
