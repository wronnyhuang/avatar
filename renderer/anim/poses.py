"""Pose tables: the base table in anim/viseme_poses.json, merged with an optional animals/<name>.poses.json.

Character file, every key optional:
    {"poses": {"AA": {"lip_open": 0.9}},   a named parameter replaces the base value; the rest of that pose stays
     "gains": {"teeth_gap": 1.1},          multiplies the final column
     "speed": 0.85,                        scales every mouth follower: heavy jaws < 1, small quick mouths > 1
     "box":   [x0, y0, x1, y1]}            mouth crop in 1080p pixels, for strips and sheets
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LIP_PARAMS = ["lip_open", "lip_upper", "lip_lower", "lip_width", "lip_protrude", "lip_press", "lip_tuck",
              "teeth_gap", "tongue_tip", "tongue_dental", "tongue_back"]
FINE = ["sil", "PP", "FF", "TH", "DD", "NN", "LL", "kk", "CH", "SS", "RR", "WW",
        "AA", "AE", "AH", "AX", "EH", "IH", "IY", "AO", "OW", "UH", "UW"]


def load_table(character: str | None = None) -> dict:
    t = json.loads((ROOT / "anim" / "viseme_poses.json").read_text())
    t = {k: dict(v) for k, v in t.items() if not k.startswith("_")}
    meta = {"gains": {}, "speed": 1.0, "box": None}
    p = ROOT / "animals" / f"{character}.poses.json" if character else None
    if p is not None and p.exists():
        c = json.loads(p.read_text())
        for v, over in c.get("poses", {}).items():
            t[v].update(over)
        meta.update({k: c[k] for k in ("gains", "speed", "box") if k in c})
    return {"poses": t, **meta}


def pose_matrix(character: str | None = None, names: list[str] = FINE) -> np.ndarray:
    t = load_table(character)["poses"]
    m = np.zeros((len(names), len(LIP_PARAMS)))
    for i, v in enumerate(names):
        for k, x in t[v].items():
            m[i, LIP_PARAMS.index(k)] = x
    return m


def gains(character: str | None) -> np.ndarray:
    g = load_table(character)["gains"]
    return np.array([g.get(n, 1.0) for n in LIP_PARAMS])


# the first net's 15 visemes, spread over the fine set, until the new net replaces it
OLD15 = ["sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "ih", "oh", "ou"]
SPREAD = {"nn": {"NN": .6, "LL": .4}, "aa": {"AA": .35, "AE": .3, "AH": .35}, "E": {"EH": 1}, "ih": {"IH": .5, "IY": .5},
          "oh": {"OW": .6, "AO": .4}, "ou": {"UW": .7, "WW": .3}}


def old15_matrix(character: str | None = None) -> np.ndarray:
    P = pose_matrix(character)
    out = np.zeros((len(OLD15), P.shape[1]))
    for i, v in enumerate(OLD15):
        for name, w in SPREAD.get(v, {v: 1}).items():
            out[i] += w * P[FINE.index(name)]
    return out
