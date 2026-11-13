#!/usr/bin/env python3
"""props: named test poses for the muse scene, computed from renderer/animals/muse.props.json (prints
rig_still pose specs).  python tools/props_poses.py NAME [NAME ...]"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(os.path.dirname(HERE), "animals", "muse.props.json")))
P = {p["id"]: p for p in cfg["props"]}
PL = {p["id"]: p for p in cfg["places"]}


def ry(t, v):
    a = 2*math.pi*t; c, s = math.cos(a), math.sin(a)
    return np.array([c*v[0] + s*v[2], v[1], -s*v[0] + c*v[2]])


def slot(pid):
    return P[pid]["slot"]


def off_to(pid, target):
    """free offset (H) that moves prop pid's centre to world point target (H)."""
    return np.array(target) - np.array(P[pid]["home"]["pos"])


def on_desk(pid, local_xz, desk_off=(0, 0, 0)):
    d = P["desk"]
    top = np.array(d["home"]["pos"]) + np.array(desk_off) + ry(d["home"]["rot"][1], [local_xz[0], d["dims"][2]*0.5, local_xz[1]])
    lift = {"laptop": 0.08*P["laptop"]["scale"], "mug": 0.84*P["mug"]["scale"], "book": 0.89*P["book"]["scale"],
            "plant": 0.97*P["plant"]["scale"], "ball": 0.98*P["ball"]["scale"]}[pid]
    return off_to(pid, top + np.array([0, lift, 0]))


# ---- the reach map (muse.props.json "reach"): which hand target puts the palm contact at a point
def _reach_eval(side, h):
    r = cfg["reach"][side]
    ax = [np.array(a) for a in r["axes"]]
    pos = np.array(r["pos"]).reshape(r["shape"])
    idx, fr = [], []
    for a, v in zip(ax, h):
        i = int(np.clip(np.searchsorted(a, v) - 1, 0, len(a) - 2))
        idx.append(i); fr.append((v - a[i])/(a[i+1] - a[i]))
    out = np.zeros(3)
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                w = (fr[0] if di else 1 - fr[0])*(fr[1] if dj else 1 - fr[1])*(fr[2] if dk else 1 - fr[2])
                out += w*pos[idx[0]+di, idx[1]+dj, idx[2]+dk]
    return out


def reach_inv(side, target, h0=None):
    """hand target (x, y, z) whose palm contact (rest body, root 0) lands nearest target (world H); + the miss (H).
    Damped Gauss-Newton from the nearest grid node (the map saturates at full reach: plain GN overshoots there)."""
    r = cfg["reach"][side]
    pos = np.array(r["pos"]).reshape(-1, 3)
    ax = r["axes"]
    if h0 is None:
        i = int(np.argmin(np.linalg.norm(pos - target, axis=1)))
        n1, n2 = len(ax[1]), len(ax[2])
        h0 = (ax[0][i//(n1*n2)], ax[1][(i//n2) % n1], ax[2][i % n2])
    h = np.array(h0, float)
    lam = 1e-3
    f = _reach_eval(side, h) - target
    for _ in range(60):
        J = np.zeros((3, 3))
        for c in range(3):
            e = np.zeros(3); e[c] = 1e-3
            J[:, c] = (_reach_eval(side, h + e) - _reach_eval(side, h - e))/2e-3
        step = np.linalg.solve(J.T @ J + lam*np.eye(3), -J.T @ f)
        hn = np.clip(h + np.clip(step, -0.15, 0.15), -1.0, 1.2)
        fn = _reach_eval(side, hn) - target
        if fn @ fn < f @ f:
            h, f, lam = hn, fn, lam*0.5
        else:
            lam *= 4.0
    return h, float(np.linalg.norm(f))


# ---- the root (muse.body.json): world point <-> the character's rest frame (H), root = (x, y, z, yaw)
_BJ = os.path.join(os.path.dirname(HERE), "animals", "muse.body.json")
# the body agent's file (branch body); the numbers it had at 01:40 when this tree has none of its own
BODY = json.load(open(_BJ)) if os.path.exists(_BJ) else {
    "H": 2.30, "com": [0.0, 0.58, 0.0],
    "sitting": {"seat_contact_local_scene": [0.0, 0.37, 0.0],
                "feet_for_thighs_horizontal_H": {"foot_l": [0.0087, 0.0696, 0.078], "foot_r": [-0.0087, 0.0696, 0.078]}}}
COM = np.array(BODY["com"])


def to_local(world, root):
    x, y, z, yaw = root
    return ry(-yaw, np.array(world) - COM - np.array([x, y, z])) + COM


def hands_at(d, root, l=None, r=None):
    """set hand_l/r targets so the palms land on world points l / r with the root (x, y, z, yaw); returns the misses."""
    miss = {}
    if l is not None and r is not None and to_local(l, root)[0] > to_local(r, root)[0]:
        l, r = r, l                            # turned round: the screen-left point is on the character's _r side
    for side, P in (("l", l), ("r", r)):
        if P is None:
            continue
        h, m = reach_inv(side, to_local(P, root))
        d[f"hand_{side}_x"], d[f"hand_{side}_y"], d[f"hand_{side}_z"] = h
        miss[side] = m
    return miss


def seated_on_chair(d, chair_off, chair_ry_total):
    """root + feet that sit the muse on the chair (chair centre = home + chair_off, its yaw chair_ry_total turns)."""
    ch = P["chair"]
    sp = np.array(ch["home"]["pos"]) + np.array(chair_off) + ry(chair_ry_total, ch["seat"]["sit_point_local"])
    yaw = chair_ry_total                       # the chair's front (+z local) is the way the sitter faces
    seat_scene = ch["seat"]["height"]*BODY["H"]
    root = (float(sp[0]), float((seat_scene - BODY["sitting"]["seat_contact_local_scene"][1])/BODY["H"]), float(sp[2]), float(yaw))
    d["root_x"], d["root_y"], d["root_z"], d["root_yaw"] = root
    ft = BODY["sitting"]["feet_for_thighs_horizontal_H"]
    for side, key in (("l", "foot_l"), ("r", "foot_r")):
        v = ry(yaw, ft[key])
        d[f"foot_{side}_x"], d[f"foot_{side}_y"], d[f"foot_{side}_z"] = v
    return root


def laptop_world(lap_off, lap_ry_total, local):
    return np.array(P["laptop"]["home"]["pos"]) + np.array(lap_off) + ry(lap_ry_total, np.array(local))


def spec(d):
    return " ".join(f"{k}={v:.4f}" for k, v in d.items())


def pose(name):
    d = {}
    def put(pid, o, extra=None):
        k = slot(pid)
        d[f"prop{k}_x"], d[f"prop{k}_y"], d[f"prop{k}_z"] = o
        for kk, vv in (extra or {}).items():
            d[f"prop{k}_{kk}"] = vv
    if name == "worn":
        d["prop0_hh"] = 1.0
    elif name == "laptop_desk_shut":
        put("laptop", on_desk("laptop", (0.0, 0.12)), {"ry": -0.06 - P["laptop"]["home"]["rot"][1]})
    elif name == "laptop_desk_open":
        put("laptop", on_desk("laptop", (0.0, 0.12)), {"ry": -0.06 - P["laptop"]["home"]["rot"][1], "s": 1.0})
        put("book", off_to("book", np.array(P["book"]["home"]["pos"])), {"s": 1.0})
    elif name == "hold_mug_r":
        d.update({"hand_r_x": 0.1, "hand_r_y": 0.55, "hand_r_z": 0.7, "prop4_hr": 1.0})
    elif name == "hold_ball_l":
        d.update({"hand_l_x": 0.1, "hand_l_y": 0.55, "hand_l_z": 0.7, "prop6_hl": 1.0})
    elif name == "hold_plant_r_book_l":
        d.update({"hand_r_x": 0.15, "hand_r_y": 0.45, "hand_r_z": 0.7, "prop7_hr": 1.0,
                  "hand_l_x": 0.15, "hand_l_y": 0.45, "hand_l_z": 0.7, "prop5_hl": 1.0})
    elif name == "hold_book_open":
        d.update({"hand_r_x": -0.2, "hand_r_y": 0.5, "hand_r_z": 0.8, "hand_l_x": -0.2, "hand_l_y": 0.5, "hand_l_z": 0.8,
                  "prop5_hl": 1.0, "prop5_hr": 1.0, "prop5_s": 1.0})
    elif name == "carry_laptop":
        d.update({"hand_r_x": -0.25, "hand_r_y": 0.35, "hand_r_z": 0.8, "hand_l_x": -0.25, "hand_l_y": 0.35, "hand_l_z": 0.8,
                  "prop3_hl": 1.0, "prop3_hr": 1.0})
    elif name == "hold_headphones_r":
        d.update({"hand_r_x": 0.2, "hand_r_y": 0.8, "hand_r_z": 0.6, "prop0_hr": 1.0})
    elif name == "chair_at_desk":
        de = P["desk"]; ch = P["chair"]
        sp = np.array(de["home"]["pos"]) + ry(de["home"]["rot"][1], de["sit_point_local"])          # where the sit point goes
        cur = np.array(ch["home"]["pos"]) + ry(ch["home"]["rot"][1], ch["seat"]["sit_point_local"])
        dry = (de["home"]["rot"][1] + 0.5) - ch["home"]["rot"][1]          # the chair faces the desk (its front to the desk)
        new_c = sp - ry(ch["home"]["rot"][1] + dry, ch["seat"]["sit_point_local"])
        o = new_c - np.array(ch["home"]["pos"])
        put("chair", (o[0], 0.0, o[2]), {"ry": dry})
        put("laptop", on_desk("laptop", (0.0, 0.12)), {"ry": -0.06 - P["laptop"]["home"]["rot"][1], "s": 1.0})
    elif name.startswith("seated_desk"):
        # the headline end state: the chair pulled up to the desk, the muse in it, the laptop on the desk in front of it
        # (open, or _shut), paws on the near rows of keys.  seated_desk = the desk where it stands at home;
        # seated_desk34 = the desk turned 0.10 turn so the camera sees the muse three-quarter from behind and the screen;
        # _wide = the wide shot.  Things on the desk (mug, plant) turn with it.
        de = P["desk"]; ch = P["chair"]
        ddy = -0.10 if "34" in name else 0.0
        dhome = np.array(de["home"]["pos"]); dyaw = de["home"]["rot"][1] + ddy
        if ddy:
            put("desk", (0.0, 0.0, 0.0), {"ry": ddy})
            for pid in ("mug", "plant"):
                h = np.array(P[pid]["home"]["pos"])
                put(pid, dhome + ry(ddy, h - dhome) - h, {"ry": ddy})
        sp = dhome + ry(dyaw, de["sit_point_local"])
        cyaw = dyaw + 0.5                                   # the chair faces the desk
        new_c = sp - ry(cyaw, ch["seat"]["sit_point_local"])
        co = new_c - np.array(ch["home"]["pos"])
        put("chair", (co[0], 0.0, co[2]), {"ry": cyaw - ch["home"]["rot"][1]})
        top = dhome + ry(dyaw, [0.0, de["dims"][2]*0.5, 0.12]) + np.array([0, 0.08*P["laptop"]["scale"], 0])
        lo = off_to("laptop", top)
        put("laptop", lo, {"ry": dyaw - P["laptop"]["home"]["rot"][1], "s": 0.0 if "shut" in name else 1.0})
        root = seated_on_chair(d, co, cyaw)
        kb = P["laptop"]["keyboard"]
        up = np.array([0.0, 0.012, 0.0])
        kl = laptop_world(lo, dyaw, kb["keys_left_local"]) + up
        kr = laptop_world(lo, dyaw, kb["keys_right_local"]) + up
        miss = hands_at(d, root, l=kl, r=kr)
        d["_miss"] = max(miss.values())
        d["gaze_free"] = 1.0
        if "wide" in name:
            d["cam_zoom"] = 1.0
    elif name == "pull_chair":
        # the muse (turned to face screen-right) holds the chair's backrest in both paws, the chair's back toward it
        ch = P["chair"]
        root = (0.0, 0.0, 0.0, 0.25)
        d["root_yaw"] = 0.25
        cry = 0.25
        hl = [h for h in ch["handles"] if h["id"] == "back_l"][0]["local"]
        cpos = np.array([0.22 - ry(cry, hl)[0], ch["home"]["pos"][1], 0.0])
        co = cpos - np.array(ch["home"]["pos"])
        put("chair", (co[0], 0.0, co[2]), {"ry": cry - ch["home"]["rot"][1]})
        bw = {h["id"]: np.array(ch["home"]["pos"]) + co + ry(cry, h["local"]) for h in ch["handles"]}
        miss = hands_at(d, root, l=bw["back_l"], r=bw["back_r"])
        d["_miss"] = max(miss.values())
    elif name == "hp_pick":
        # standing where muse.props.json says (pick_up_from), the right paw on the band handle of the hanging headphones
        hp = P["headphones"]
        sa = hp["pick_up_from"]["stand_at"]; yaw = hp["pick_up_from"]["facing_yaw"]
        root = (sa[0], 0.0, sa[2], yaw)
        d["root_x"], d["root_z"], d["root_yaw"] = sa[0], sa[2], yaw
        band = [x for x in hp["handles"] if x["id"] == "band"][0]["world_home"]
        miss = hands_at(d, root, r=np.array(band))
        d["_miss"] = max(miss.values())
    elif name == "neck_seated":
        # seated at the desk (34), the headphones taken off round the neck (worn.neck), the mug at a seat spot
        out = pose("seated_desk34")
        for t in out.split():
            k, v = t.split("="); d[k] = float(v)
        nk = P["headphones"]["worn"]["neck"]
        d["prop0_hh"] = 1.0; d["prop0_y"] = nk["y"]; d["prop0_rx"] = nk["rx"]
        de = P["desk"]; dhome = np.array(de["home"]["pos"]); dyaw = de["home"]["rot"][1] - 0.10
        loc = PL["desk_top"]["seat_spots_local"][1]
        w = dhome + ry(dyaw, loc) + np.array([0, 0.84*P["mug"]["scale"], 0])
        o = w - np.array(P["mug"]["home"]["pos"])
        d["prop4_x"], d["prop4_y"], d["prop4_z"] = o
    elif name == "hp_puton":
        # half way through putting them on: both paws on the cups by the head, hh easing in
        d.update({"prop0_hl": 0.5, "prop0_hr": 0.5, "prop0_hh": 0.5,
                  "hand_l_x": 0.55, "hand_l_y": 1.0, "hand_l_z": 0.2, "hand_r_x": 0.55, "hand_r_y": 1.0, "hand_r_z": 0.2})
    elif name == "worn_turn":
        d.update({"prop0_hh": 1.0, "head_yaw": 0.12, "head_pitch": -0.04})
    miss = d.pop("_miss", None)
    if miss is not None and os.environ.get("PROPS_POSES_MISS"):
        print(f"{name}: reach miss {miss:.4f} H", file=sys.stderr)
    return spec(d) if d else "REST"


if __name__ == "__main__":
    for n in sys.argv[1:]:
        print(pose(n))
