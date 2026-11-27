"""Live targets: the person (perception), the camera, the character's scene things -> the API's pelvis frame (cm).

resolve_targets(character, state) ALWAYS returns the same NAME set for a character (hub ask: the warm system prompt is
keyed by the names), whatever the state says: 'camera', 'user' (+ 'user_left', 'user_right'), and for Muse / Juno every
scene place / furniture / thing at its home position (moved things at the position the page reports, when it does).

state (the page's plan state, all optional):
  state['sem']['user'] = {present: bool, ax: rad (+ = screen right), ay: rad (+ = up), dist: m}   perception's face
  state['things'] = [{id, pos: [x, y, z] (H, world), ...}]   live things (planner.things_note format)
  state['root'] = {x, z}  where the character stands now (H) — things are made relative to it
Frames: scene world (H) = x screen-right (= the avatar's own LEFT = API +x), y up, z toward the camera (= API +z).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from . import kin

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
SCENE_DIR = os.path.join(ROOT, "renderer", "animals")
SCENE_CHARS = {"muse": "muse", "juno": "juno"}
M_PER_SU = 0.83          # canon metres per scene unit, from the muse's shoulder height above the pelvis (0.41 m / 0.495 su)
_CACHE = {}


def _scene(character):
    name = SCENE_CHARS.get(character)
    if not name:
        return None
    path = os.path.join(SCENE_DIR, f"{name}.props.json")
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return None
    c = _CACHE.get(path)
    if c and c[0] == mt:
        return c[1]
    try:
        cfg = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(cfg, dict) and isinstance(cfg.get("props"), list):      # a prop flagged "removed" is not in the scene
        cfg = dict(cfg, props=[p for p in cfg["props"] if not (isinstance(p, dict) and p.get("removed"))])
    body = {}
    bpath = os.path.join(SCENE_DIR, f"{name}.body.json")
    if character == "juno":
        bpath = os.path.join(ROOT, "renderer", "humans", "juno", "juno.body.json")
    try:
        body = json.load(open(bpath, encoding="utf-8"))
    except (OSError, ValueError):
        body = {}
    H = float(body.get("H", 2.30) or 2.30)
    pel = ((body.get("joints_scene") or {}).get("pelvis") or [0.0, 0.54, 0.0])
    pts = {}
    for pl in cfg.get("places") or []:
        if not isinstance(pl, dict) or not pl.get("id") or pl.get("id") in ("floor",):
            continue
        c_ = pl.get("center")
        if isinstance(c_, list) and len(c_) == 3:
            pts[_name(pl["id"])] = [float(c_[0]), float(pl.get("top_y", c_[1]) if pl.get("kind") != "board" else c_[1]),
                                    float(c_[2])]
    for p in cfg.get("props") or []:
        if not isinstance(p, dict) or not p.get("id"):
            continue
        h = p.get("home") or {}
        pos = h.get("pos")
        if isinstance(pos, list) and len(pos) == 3:
            pts.setdefault(_name(p["id"]), [float(v) for v in pos])
    out = (pts, H, [float(v) / H for v in pel])     # pelvis in H (the props file is in H already)
    _CACHE[path] = (mt, out)
    return out


def _name(s):
    n = str(s).strip().lower().replace(" ", "_").replace("-", "_")
    return {"desk_top": "desk", "chair_seat": "chair", "board_tray": "board_tray"}.get(n, n)


def _h_to_cm(p_h, pel_h, H, root=(0.0, 0.0)):
    """scene point in H (world) -> API pelvis frame cm"""
    su = (np.asarray(p_h, float) - np.array([root[0], 0.0, root[1]]) - np.asarray(pel_h, float)) * H
    return tuple(float(round(v * M_PER_SU * 100.0, 1)) for v in su)


def _user_point(u):
    cam = kin.camera_point()
    eyes = kin.point("eyes", kin.neutral())
    d = cam - eyes
    dist = 1.5
    ax = ay = 0.0
    if isinstance(u, dict) and u.get("present"):
        try:
            ax = float(u.get("ax") or 0.0)
            ay = float(u.get("ay") or 0.0)
            if u.get("dist") is not None:
                dist = float(np.clip(float(u["dist"]), 0.4, 4.0))
        except (TypeError, ValueError):
            ax = ay = 0.0
        if not (math.isfinite(ax) and math.isfinite(ay)):
            ax = ay = 0.0
    # the camera's view: the person at angle (ax right, ay up) off the camera axis, `dist` from the camera.  Screen
    # right as the avatar sees the frame = perception's mirrored x: ax > 0 = the person is on the avatar's LEFT (+x).
    p = cam + dist * np.array([math.sin(ax), math.sin(ay), 0.0]) + np.array([0.0, 0.0, 0.0])
    return p, cam, eyes


def resolve_targets(character, state=None):
    out = {}
    cam = kin.camera_point()
    out["camera"] = tuple(float(round(v * 100.0, 1)) for v in cam)
    st = state if isinstance(state, dict) else {}
    sem = st.get("sem") if isinstance(st.get("sem"), dict) else {}
    u = sem.get("user") if isinstance(sem.get("user"), dict) else None
    p, cam, eyes = _user_point(u)
    out["user"] = tuple(float(round(v * 100.0, 1)) for v in p)
    side = np.array([0.35, 0.0, 0.0])
    out["user_left"] = tuple(float(round(v * 100.0, 1)) for v in (p - side))    # the person's own left = avatar's right
    out["user_right"] = tuple(float(round(v * 100.0, 1)) for v in (p + side))
    sc = _scene(character)
    if sc:
        pts, H, pel = sc
        root = (0.0, 0.0)
        r = st.get("root")
        if isinstance(r, dict):
            try:
                root = (float(r.get("x") or 0.0), float(r.get("z") or 0.0))
            except (TypeError, ValueError):
                root = (0.0, 0.0)
        live = {}
        for t in st.get("things") or []:
            if isinstance(t, dict) and t.get("id") and isinstance(t.get("pos"), list) and len(t["pos"]) == 3:
                try:
                    live[_name(t["id"])] = [float(v) for v in t["pos"]]
                except (TypeError, ValueError):
                    pass
        for n, ph in pts.items():
            out[n] = _h_to_cm(live.get(n, ph), pel, H, root)
    return out


def scene_names(character):
    sc = _scene(character)
    return sorted(sc[0]) if sc else []
