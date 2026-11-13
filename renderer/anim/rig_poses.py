"""The contract's key poses (docs/CONTRACT.md) as param dicts, and helpers.  Used by the QA tools and tests; the page's
motion controller has its own copy in JS.  Hand tuples are (x, y, z ; open ; turn) per side, x = outward."""
import numpy as np
from .schema import IDX, NPARAMS

REST = {"hand_l_open": 0.2, "hand_r_open": 0.2}


def hands(l=None, r=None, **extra):
    d = dict(REST)
    for side, v in (("l", l), ("r", r)):
        if v is None:
            continue
        x, y, z = v[0], v[1], v[2]
        d[f"hand_{side}_x"], d[f"hand_{side}_y"], d[f"hand_{side}_z"] = x, y, z
        if len(v) > 3 and v[3] is not None:
            d[f"hand_{side}_open"] = v[3]
        if len(v) > 4 and v[4] is not None:
            d[f"hand_{side}_turn"] = v[4]
    d.update(extra)
    return d


KEY_POSES = {
    "REST": dict(REST),
    "WAVE_UP": hands(r=(0.20, 1.00, 0.20, 1.0, 1.0)),
    "WAVE_UP_L": hands(l=(0.20, 1.00, 0.20, 1.0, 1.0)),
    "POINT_CAMERA": hands(r=(-0.10, 0.55, 0.85, 0.0, 0.0), point_r=1.0),
    "POINT_SIDE": hands(r=(0.60, 0.60, 0.30, 0.0, 0.0), point_r=1.0),
    "CLAP": hands(l=(-0.45, 0.45, 0.45, 1.0, 0.0), r=(-0.45, 0.45, 0.45, 1.0, 0.0)),
    "THUMBS_UP": hands(r=(0.10, 0.45, 0.45, 0.0, 0.0), thumb_r=1.0),
    "CHIN": hands(r=(-0.35, 0.75, 0.35, 0.3, 0.0)),
    "SHRUG": hands(l=(0.35, 0.30, 0.20, 1.0, 1.0), r=(0.35, 0.30, 0.20, 1.0, 1.0), shoulder_l=1.0, shoulder_r=1.0),
    "COVER_EYES": hands(l=(-0.35, 1.00, 0.45, 1.0, 0.0), r=(-0.35, 1.00, 0.45, 1.0, 0.0)),
    "HAND_ON_HEART": hands(r=(-0.45, 0.35, 0.25, 1.0, 0.0)),
}


def pose_vector(d):
    v = np.zeros(NPARAMS, "f4")
    for k, x in d.items():
        v[IDX[k]] = x
    return v
