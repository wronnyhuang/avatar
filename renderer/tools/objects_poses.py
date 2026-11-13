#!/usr/bin/env python3
"""OBJECTS: named test poses for the six new muse props (slots 8-13: hat, glasses, gift box, watering can,
block A, block B), computed from renderer/animals/muse.props.json (prints rig_still pose specs).
  python tools/objects_poses.py NAME [NAME ...]      (names: see POSES)
  python tools/rig_still.py --character muse --poses "$(python tools/objects_poses.py hat_worn),..." ..."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import props_poses as pp  # noqa: E402

P = pp.P


def worn(d, pid):
    k = pp.slot(pid); w = P[pid]["afford"]["wear"]
    d[f"prop{k}_hh"] = 1.0
    for a, v in zip("xyz", w["offset"]):
        d[f"prop{k}_{a}"] = v
    for a, v in zip(("rx", "ry", "rz"), w.get("rot", [0, 0, 0])):
        d[f"prop{k}_{a}"] = v


def rest_on(d, pid, sup, dx=0.0, dz=0.0):
    """a free offset that puts pid's centre on (or in) prop sup at home (sup's surface top / container floor)."""
    s, p = P[sup], P[pid]
    half = p["afford"].get("surface", {}).get("top_y") or 0.5*p["size"]
    top = np.array(s["home"]["pos"]) + np.array([dx, s["afford"]["surface"]["top_y"] + half, dz])
    k = pp.slot(pid)
    d[f"prop{k}_x"], d[f"prop{k}_y"], d[f"prop{k}_z"] = top - np.array(p["home"]["pos"])


def floor_glasses(d):
    """the glasses set down on the rug in front of the box (they start on its lid)."""
    g = pp.slot("glasses"); p = P["glasses"]
    half_y = 0.30*p["scale"]                     # the model's lower extent (lens rim), unit 0.30
    d[f"prop{g}_z"] = 0.20
    d[f"prop{g}_y"] = (half_y + 0.007) - p["home"]["pos"][1]


def pose(name):
    d = {}
    hold_r = {"hand_r_x": 0.1, "hand_r_y": 0.55, "hand_r_z": 0.7}
    if name == "hat_worn":
        worn(d, "hat")
    elif name == "glasses_worn":
        worn(d, "glasses")
    elif name == "both_worn":
        worn(d, "hat"); worn(d, "glasses")
    elif name == "both_worn_turn":
        worn(d, "hat"); worn(d, "glasses"); d.update({"head_yaw": 0.12, "head_pitch": -0.05})
    elif name.startswith("hold_") and name[5:] in ("hat", "glasses", "watering_can", "block_a", "block_b"):
        d.update(hold_r); d[f"prop{pp.slot(name[5:])}_hr"] = 1.0
    elif name == "hold_box":
        d.update({"hand_r_x": -0.25, "hand_r_y": 0.35, "hand_r_z": 0.8, "hand_l_x": -0.25, "hand_l_y": 0.35, "hand_l_z": 0.8})
        k = pp.slot("gift_box"); d[f"prop{k}_hl"] = 0.5; d[f"prop{k}_hr"] = 0.5
        floor_glasses(d)
    elif name == "box_open":
        # lid open, block A inside it (on the inside floor), glasses + block B set down on the floor beside it
        k = pp.slot("gift_box"); d[f"prop{k}_s"] = 1.0
        b, a = P["gift_box"], P["block_a"]
        c = np.array(b["home"]["pos"]) + np.array([0.0, b["afford"]["container"]["floor_y"] + a["afford"]["surface"]["top_y"], 0.0])
        ka = pp.slot("block_a"); d[f"prop{ka}_x"], d[f"prop{ka}_y"], d[f"prop{ka}_z"] = c - np.array(a["home"]["pos"])
        kb = pp.slot("block_b"); d[f"prop{kb}_y"] = -0.10
        floor_glasses(d)
        d[f"prop{pp.slot('hat')}_x"] = 0.34           # the hat set aside (it stands in front of the box from the camera)
    elif name == "can_pour":
        k = pp.slot("watering_can")
        d.update({"hand_r_x": 0.25, "hand_r_y": 0.75, "hand_r_z": 0.6, f"prop{k}_hr": 1.0, f"prop{k}_rz": -P["watering_can"]["afford"]["pour"]["tilt_turns"], f"prop{k}_s": 1.0})
    elif name == "blocks_apart":
        kb = pp.slot("block_b"); d[f"prop{kb}_x"] = 0.14; d[f"prop{kb}_y"] = -0.10
    elif name == "wide":
        d["cam_zoom"] = 1.0
    elif name.startswith("c_"):
        # close-ups: the camera pushed in (cam_zoom < 0) and aimed at the thing (cam_x/y, H, from its home)
        base = {"c_box": "box_open", "c_worn": "both_worn", "c_pour": "can_pour", "c_stack": None}[name]
        if base:
            d.update({k: float(v) for k, v in (t.split("=") for t in pose(base).split())} if pose(base) != "REST" else {})
        tgt = {"c_box": P["gift_box"]["home"]["pos"], "c_stack": P["block_a"]["home"]["pos"],
               "c_worn": [0.0, 0.72, 0.0], "c_pour": [0.25, 0.45, 0.2]}[name]
        d.update({"cam_zoom": -0.75, "cam_x": tgt[0], "cam_y": tgt[1] - 0.45})
    return pp.spec(d) if d else "REST"


POSES = ["hat_worn", "glasses_worn", "both_worn", "both_worn_turn", "hold_hat", "hold_glasses", "hold_watering_can",
         "hold_block_a", "hold_box", "box_open", "can_pour", "blocks_apart", "wide", "c_box", "c_worn", "c_pour", "c_stack"]

if __name__ == "__main__":
    print(",".join(pose(n) for n in (sys.argv[1:] or POSES)))
