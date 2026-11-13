#!/usr/bin/env python3
"""props: write renderer/animals/muse.props.json (docs/INTENT-RULES.md sections 0 + 2) from the scene layout
below (authored in SCENE units, converted to body heights H), and optionally the calibrated reach map measured from the
renderer's own arm solve (tools/props_probe.py: the GLSL grasp readback of museGrasp).

  python tools/props_build.py                 # layout only (keeps an existing reach map)
  python tools/props_build.py --reach         # + measure the reach map (CPU GLSL readback; needs the muse character)

World frame (contract): H units, origin on the floor under the character's centre at rest, x screen-right, y up,
z toward the camera.  A prop's local frame: its own x/y/z (y up, z = its front), origin at its centre (pos), turned by
its ry; "local" points below are offsets from that centre in H, in that frame (world = pos + Ry(ry_turns)*local).
"""
import argparse, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine")]

CHAR = "muse"
OUT = os.path.join(ROOT, "animals", f"{CHAR}.props.json")


def body_units():
    bp = os.path.join(ROOT, "animals", f"{CHAR}.body.json")
    if os.path.exists(bp):
        b = json.load(open(bp))
        return float(b["H"]), float(b.get("base_y", 0.0)), "body"
    return 2.30, 0.0, "props (muse.body.json absent: H = 2.30 from animals/muse.glsl M_H)"


H, BASE, USRC = body_units()
GRASP_OFS = 0.022                           # museGrasp sits 0.022 off the mitten surface (palm pad + 0.07 from its centre, pad half-thickness 0.048)
RUG = (-2.70, -1.60, 2.20, 1.90, 0.016)      # the studio set's rug: x0 z0 x1 z1 top (scene) — props on it sit 0.016 up


def floor_at(x, z):
    return RUG[4] if (RUG[0] <= x <= RUG[2] and RUG[1] <= z <= RUG[3]) else 0.0


def h(v):
    """scene -> H (points: y measured from the floor BASE)."""
    return [round(v[0]/H, 5), round((v[1] - BASE)/H, 5), round(v[2]/H, 5)]


def hl(v):
    """scene offsets -> H (no origin shift)."""
    return [round(x/H, 5) for x in v]


def ry_mat(turns):
    a = 2*math.pi*turns
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])      # GLSL prRY(a): x' = c x + s z, z' = -s x + c z


def world(pos, ry, local):
    return (np.array(pos) + ry_mat(ry) @ np.array(local)).tolist()


# ------------------------------------------------------------------ the layout (scene units)
# the headline scenario (contract section 0): headphones hang on a stand on a small side table at the left; the chair
# and the desk stand off to the right, a few steps away (to be pulled over); the laptop lies shut on the floor by the
# desk; a red mug and a green plant on the desk, a blue book on the side table, a yellow ball on the rug.
# a LOW side table (its top under the muse's belly) with the stand post near the front: the band hangs at 0.51 H,
# ~0.23 H in front of a muse standing clear of the cups -> in reach (tools/props_poses.py hp_pick, reach map)
SIDE_TABLE = dict(c=(-2.02, -0.70), top=0.36, w=0.56, d=0.34, stand_h=0.675)   # top face centre x, z; stand post above it
HP_W, HP_ARC, HP_BZ = 0.75, 0.72, 0.19       # headphone fit to the muse head (scene units, head frame), see fit_headphones
HP_WR = 0.42                                 # the relaxed band (off the head): cups pulled in, a smaller thing in a paw
HP_NECK = dict(y=-0.62, rx=-0.25)            # "round the neck" (worn frame): cups below the head, band behind it
CHAIR = dict(pos_xz=(1.50, 0.42), ry=0.20, dims=(0.98, 0.92, 0.44, 1.12))       # seat w, d, seat height, back top
DESK = dict(pos_xz=(2.55, -1.50), ry=-0.06, dims=(1.62, 0.84, 0.92))            # w, d, top height
LAPTOP = dict(pos_xz=(1.02, -0.80), ry=0.10, scale=0.28)                        # base 0.56 x 0.38, shut 0.045 thick
MUG = dict(local=(-0.42, 0.0, 0.10), scale=0.085)        # on the desk, in the desk's frame (x along it, z to its front)
PLANT = dict(local=(0.58, 0.0, -0.18), scale=0.13)
BOOK = dict(local=(0.0, 0.0, 0.10), scale=0.12)          # on the side table, lying? standing: leaning on the stand post
BALL = dict(pos_xz=(0.72, 0.62), scale=0.085)


def build(reach=None):
    st = SIDE_TABLE
    chair_c = [CHAIR["pos_xz"][0], floor_at(*CHAIR["pos_xz"]) + 0.5*CHAIR["dims"][3], CHAIR["pos_xz"][1]]
    desk_c = [DESK["pos_xz"][0], floor_at(*DESK["pos_xz"]) + 0.5*DESK["dims"][2], DESK["pos_xz"][1]]
    desk_top_y = desk_c[1] + 0.5*DESK["dims"][2]
    lap_c = [LAPTOP["pos_xz"][0], floor_at(*LAPTOP["pos_xz"]) + 0.08*LAPTOP["scale"] + 0.0, LAPTOP["pos_xz"][1]]
    # headphones hang on the stand: the band's pad rests on the stand's saddle
    hp_c = [st["c"][0], st["top"] + st["stand_h"] - (HP_ARC - 0.085), st["c"][1]]
    mug_c = world(desk_c, DESK["ry"], [MUG["local"][0], 0.5*DESK["dims"][2] + 0.84*MUG["scale"], MUG["local"][2]])
    plant_c = world(desk_c, DESK["ry"], [PLANT["local"][0], 0.5*DESK["dims"][2] + 0.97*PLANT["scale"], PLANT["local"][2]])
    book_c = [st["c"][0] + 0.15, st["top"] + 0.89*BOOK["scale"], st["c"][1] + 0.14]
    ball_c = [BALL["pos_xz"][0], floor_at(*BALL["pos_xz"]) + 0.98*BALL["scale"], BALL["pos_xz"][1]]
    cw, cd, cs, cb = CHAIR["dims"]
    dw, dd, dh = DESK["dims"]
    ls = LAPTOP["scale"]
    places = [
        {"id": "floor", "name": "the floor", "aliases": ["floor", "ground", "rug", "carpet"], "kind": "floor",
         "center": [0.0, 0.0, 0.0], "size": hl([9.4, 5.0]), "top_y": 0.0,
         "spots": [h([0.70, 0.016, 0.55]), h([-0.70, 0.016, 0.55]), h([1.20, 0.016, -0.30]), h([-1.00, 0.016, 0.90])],
         "notes": "the studio floor; the rug (x -2.70..2.20, z -1.60..1.90 scene) is 0.016 scene above it"},
        {"id": "side_table", "name": "the side table", "aliases": ["side table", "little table", "small table", "stand",
         "headphone stand", "end table"], "kind": "table_stand", "movable": False,
         "center": h([st["c"][0], st["top"], st["c"][1]]), "size": hl([st["w"], st["d"]]), "top_y": round(st["top"]/H, 5),
         "floor_y": 0.0, "stand_top_y": round((st["top"] + st["stand_h"])/H, 5), "rgb": [176, 128, 86],
         "spots": [h([st["c"][0] + 0.17, st["top"], st["c"][1] + 0.06]), h([st["c"][0] - 0.17, st["top"], st["c"][1] + 0.06])],
         "notes": "a low oak side table with a headphone stand (post + padded saddle) in the middle; the headphones hang there, their cups beside the table"},
        {"id": "desk_top", "name": "the desk", "aliases": ["desk", "table", "desktop", "on the desk"], "kind": "desk_top",
         "movable": True, "on_prop": "desk", "local_center": hl([0.0, 0.5*dh, 0.0]), "center": h([desk_c[0], desk_top_y, desk_c[2]]),
         "size": hl([dw, dd]), "top_y": round(desk_top_y/H, 5),
         "spots_local": [hl([-0.42, 0.5*dh, 0.10]), hl([0.0, 0.5*dh, 0.12]), hl([0.42, 0.5*dh, 0.10]), hl([0.0, 0.5*dh, 0.26])],
         "spots": [h(world(desk_c, DESK["ry"], [x, 0.5*dh, z])) for x, z in ((-0.42, 0.10), (0.0, 0.12), (0.42, 0.10), (0.0, 0.26))],
         "seat_spots_local": [hl([-0.45, 0.5*dh, 0.36]), hl([0.45, 0.5*dh, 0.36])],
         "seat_spots": [h(world(desk_c, DESK["ry"], [x, 0.5*dh, 0.36])) for x in (-0.45, 0.45)],
         "seat_spots_notes": "reachable from the chair pulled up at the desk (the sit point): the near strip (local z >= 0.34 scene of 0.42), left/right of the laptop at the work spot; small things only (mug, book, ball, plant) - the headphones are head-sized: seated, take them off round the neck (headphones worn.neck)",
         "work_spot_local": hl([0.0, 0.5*dh, 0.18]),
         "notes": "moves with the desk: world = desk pos + Ry(desk ry)*local (+ the desk's prop offset). spots_local[3] = the front, closest to a sitter"},
        {"id": "chair_seat", "name": "the chair", "aliases": ["chair", "seat"], "kind": "chair_seat", "movable": True,
         "on_prop": "chair", "local_center": hl([0.0, cs - 0.5*cb, 0.04]), "center": h(world(chair_c, CHAIR["ry"], [0.0, cs - 0.5*cb, 0.04])),
         "size": hl([cw - 0.10, cd - 0.14]), "top_y": round(cs/H, 5), "spots_local": [hl([0.0, cs - 0.5*cb, 0.12])],
         "spots": [h(world(chair_c, CHAIR["ry"], [0.0, cs - 0.5*cb, 0.12]))]},
    ]

    def handle(id_, side, local, pos, ry, note):
        return {"id": id_, "side": side, "local": hl(local), "world_home": h(world(pos, ry, local)), "note": note}

    props = [
        {"slot": 0, "id": "headphones", "name": "pink headphones", "color": "pink", "colors": ["pink", "silver"],
         "aliases": ["headphones", "headset", "earphones", "cans", "phones"], "category": "wearable", "kind": "headphones",
         "size": round(1.05/H, 5), "mass": "light", "movable": True, "furniture": False, "wearable": True,
         "scale": round(1.0/H, 6), "rgb": [236, 128, 150], "rgb2": [214, 216, 226],
         "fit": {"w": HP_W, "arc": HP_ARC, "bz": HP_BZ, "w_rest": HP_WR},
         "articulation": {"param": "prop0_s", "meaning": "band spread: 0 relaxed (cups pulled in, scene half-gap 0.42) .. 1 opened to the head (0.75); worn (hh) opens it by itself"},
         "home": {"place": "side_table", "pos": h(hp_c), "rot": [0, 0, 0]}, "grip": "power", "throwable": False,
         "facing": [0, 0, 1],
         "handles": [handle("band", "either", [0.0, HP_ARC + 0.05, HP_BZ], hp_c, 0.0, "the top of the band: one paw lifts it"),
                     handle("cup_l", "l", [-(HP_WR + 0.17), 0.0, 0.0], hp_c, 0.0, "screen-left cup, relaxed band (two paws: put it on; opened to the head it is at -(0.75 + 0.17) scene)"),
                     handle("cup_r", "r", [HP_WR + 0.17, 0.0, 0.0], hp_c, 0.0, "screen-right cup, relaxed band")],
         "pick_up_from": {"stand_at": h([hp_c[0], 0.0, hp_c[2] + 0.71]), "facing_yaw": 0.5,
                          "notes": "a standing muse there (root_x/z, facing the table) clears the hanging cups; the band handle is then ~0.23 H in front at 0.51 H: in reach (reach map miss < 0.01 H)"},
         "grip_point_local": hl([0.0, HP_ARC + 0.05, 0.0]),
         "worn": {"param": "prop0_hh", "frame": "museHeadFrame (head axes: x right, y up, z out of the face)",
                  "offset_local": [0, 0, 0],
                  "neck": {"y": round(HP_NECK["y"]/H, 5), "rx": HP_NECK["rx"],
                           "notes": "round the neck: hh = 1 with prop0_y = this y (H, head frame) and prop0_rx = this rx: the cups below the head, the band behind it (take them off while seated: there is no room for them on the desk)"},
                  "notes": "hh = 1: the band over the crown (forward of the ears), cups on the head's sides at the ear line; put on = hold it by both cups (hl = hr = 1) at the head, then ease hh 0 -> 1 while hl/hr ease 1 -> 0"}},
        {"slot": 1, "id": "chair", "name": "chair", "color": "teal", "colors": ["teal", "oak"], "aliases": ["chair", "seat", "stool"],
         "category": "furniture", "kind": "chair", "size": round(0.5*math.hypot(cw, cd)/H, 5), "mass": "heavy", "movable": True,
         "furniture": True, "dims": hl(CHAIR["dims"]), "rgb": [178, 126, 82], "rgb2": [62, 150, 158],
         "home": {"place": "floor", "pos": h(chair_c), "rot": [0, CHAIR["ry"], 0]}, "grip": "two_hand", "throwable": False,
         "facing": [0, 0, 1],
         "handles": [handle("back_l", "l", [-(0.5*cw - 0.08), 0.5*cb - 0.05, -(0.5*cd - 0.07) - 0.06], chair_c, CHAIR["ry"], "top rail of the backrest, its left end"),
                     handle("back_r", "r", [0.5*cw - 0.08, 0.5*cb - 0.05, -(0.5*cd - 0.07) - 0.06], chair_c, CHAIR["ry"], "top rail, right end")],
         "seat": {"height": round(cs/H, 5), "sit_point_local": hl([0.0, cs - 0.5*cb, 0.10]),
                  "sit_point_home": h(world(chair_c, CHAIR["ry"], [0.0, cs - 0.5*cb, 0.10])), "sit_facing_local": [0, 0, 1],
                  "notes": "sit: root at the sit point (root_y = (seat height scene - 0.37)/2.30, muse.body.json), facing the chair's front"},
         "notes": "FREE furniture (never paw-attached): prop1_x/z move it on the floor, prop1_ry turns it about its vertical centre line; keep prop1_y/rx/rz 0"},
        {"slot": 2, "id": "desk", "name": "desk", "color": "oak", "colors": ["oak", "brown", "wood"], "aliases": ["desk", "table", "writing desk"],
         "category": "furniture", "kind": "desk", "size": round(0.5*math.hypot(dw, dd)/H, 5), "mass": "heavy", "movable": True,
         "furniture": True, "dims": hl(DESK["dims"]), "rgb": [196, 150, 100],
         "home": {"place": "floor", "pos": h(desk_c), "rot": [0, DESK["ry"], 0]}, "grip": "two_hand", "throwable": False,
         "facing": [0, 0, 1],
         "handles": [handle("edge_l", "l", [-(0.5*dw - 0.12), 0.5*dh - 0.02, 0.5*dd], desk_c, DESK["ry"], "front edge of the top, left end"),
                     handle("edge_r", "r", [0.5*dw - 0.12, 0.5*dh - 0.02, 0.5*dd], desk_c, DESK["ry"], "front edge, right end"),
                     handle("side_l", "l", [-0.5*dw, 0.5*dh - 0.02, 0.0], desk_c, DESK["ry"], "the left end of the top (pull it sideways)"),
                     handle("side_r", "r", [0.5*dw, 0.5*dh - 0.02, 0.0], desk_c, DESK["ry"], "the right end of the top")],
         "top": {"height": round(dh/H, 5), "place": "desk_top"},
         "sit_point_local": hl([0.0, cs - 0.5*dh, 0.5*dd + 0.56]),
         "notes": "FREE furniture; things on its top (place desk_top) move with it (the page keeps them consistent). sit_point_local = where a chair's sit point goes for working at the desk (front, centre)"},
        {"slot": 3, "id": "laptop", "name": "silver laptop", "color": "silver", "colors": ["silver", "grey", "gray"],
         "aliases": ["laptop", "computer", "notebook", "macbook", "pc"], "category": "device", "kind": "laptop",
         "size": round(1.1*ls/H, 5), "mass": "medium", "movable": True, "furniture": False, "scale": round(ls/H, 6),
         "rgb": [196, 198, 206], "home": {"place": "floor", "pos": h(lap_c), "rot": [0, LAPTOP["ry"], 0]}, "grip": "two_hand",
         "throwable": False, "facing": [0, 0, 1],
         "articulation": {"param": "prop3_s", "meaning": "lid: 0 shut .. 1 open (112 degrees about the back hinge)"},
         "handles": [handle("side_l", "l", [-ls, 0.0, 0.0], lap_c, LAPTOP["ry"], "left side of the base (carry in two paws)"),
                     handle("side_r", "r", [ls, 0.0, 0.0], lap_c, LAPTOP["ry"], "right side of the base"),
                     handle("lid", "either", [0.0, 0.08*ls, 0.66*ls], lap_c, LAPTOP["ry"], "front edge of the shut lid (open: lift it here)")],
         "keyboard": {"center_local": hl([0.0, 0.0, 0.15*ls]), "size_local": hl([1.76*ls, 0.0, 0.90*ls]),
                      "keys_left_local": hl([-0.45*ls, 0.0, 0.40*ls]), "keys_right_local": hl([0.45*ls, 0.0, 0.40*ls]),
                      "notes": "the key well on the base top, running to the front edge; typing = paws at keys_left/right (the near rows) with small up/down taps.  Reach: the muse's palm reaches <= 0.66 scene forward of its centre at desk height and its seated belly front is ~0.53, so the laptop's FRONT edge goes to the desk's near edge and the desk's near edge sits ~0.55 from the muse's centre (desk sit_point_local)"},
         "screen": {"center_open_local": hl([0.0, 0.62*ls, -0.91*ls]), "normal_open_local": [0.0, 0.37, 0.93],
                    "notes": "open (s = 1): the panel centre is ~0.62 scale above the hinge at the back edge, facing the sitter (+z, tipped back 22 deg)"},
         "notes": "carry: hl = hr = 1 at side_l / side_r; slide on the desk: free x/z with the paws resting on it"},
        {"slot": 4, "id": "mug", "name": "red mug", "color": "red", "colors": ["red", "cream"], "aliases": ["mug", "cup", "coffee", "tea", "red one"],
         "category": "drinkware", "kind": "mug", "size": round(0.95*MUG["scale"]/H, 5), "mass": "light", "movable": True,
         "furniture": False, "scale": round(MUG["scale"]/H, 6), "rgb": [206, 44, 40], "rgb2": [246, 236, 220],
         "home": {"place": "desk_top", "pos": h(mug_c), "rot": [0, 0.0, 0]}, "grip": "power", "throwable": False, "facing": [0, 0, 1]},
        {"slot": 5, "id": "book", "name": "blue book", "color": "blue", "colors": ["blue", "gold"], "aliases": ["book", "notebook", "storybook", "blue one"],
         "category": "book", "kind": "book", "size": round(0.9*BOOK["scale"]/H, 5), "mass": "light", "movable": True,
         "furniture": False, "scale": round(BOOK["scale"]/H, 6), "rgb": [44, 82, 176], "rgb2": [222, 182, 72],
         "home": {"place": "side_table", "pos": h(book_c), "rot": [0, -0.05, 0]}, "grip": "power", "throwable": False,
         "facing": [0, 0, 1], "articulation": {"param": "prop5_s", "meaning": "0 shut .. 1 open flat, pages to the front"}},
        {"slot": 6, "id": "ball", "name": "yellow ball", "color": "yellow", "colors": ["yellow", "red"], "aliases": ["ball", "toy", "yellow one"],
         "category": "toy", "kind": "ball", "size": round(BALL["scale"]/H, 5), "mass": "light", "movable": True, "furniture": False,
         "scale": round(BALL["scale"]/H, 6), "rgb": [246, 202, 40], "rgb2": [222, 58, 48],
         "home": {"place": "floor", "pos": h(ball_c), "rot": [0, 0, 0]}, "grip": "power", "throwable": True, "facing": [0, 0, 1]},
        {"slot": 7, "id": "plant", "name": "little green plant", "color": "green", "colors": ["green", "terracotta"],
         "aliases": ["plant", "flower pot", "pot", "succulent", "green one"], "category": "plant", "kind": "plant",
         "size": round(1.0*PLANT["scale"]/H, 5), "mass": "light", "movable": True, "furniture": False,
         "scale": round(PLANT["scale"]/H, 6), "rgb": [192, 100, 64], "rgb2": [72, 152, 62],
         "home": {"place": "desk_top", "pos": h(plant_c), "rot": [0, 0, 0]}, "grip": "power", "throwable": False, "facing": [0, 0, 1]},
    ]
    cfg = {
        "character": CHAR, "version": 1,
        "units": {"H": H, "base_y": BASE, "source": USRC,
                  "frame": "H units; origin on the floor under the character's centre at rest; x screen-right, y up, z toward the camera"},
        "renderer": {"grasp_fn": "museGrasp", "head_fn": "museHeadFrame", "grasp_offset": 0.022, "char_bound": "muse",
                     "notes": "engine/props.glsl: held = the grip anchor meets the palm (museGrasp pos - 0.07 palm normal), upright + camera-facing then the column turns; worn = museHeadFrame; furniture = free only"},
        "columns": {"x,y,z": "H: free = offset from home (world axes); held/worn = small offset in the grasp/head frame",
                    "rx,ry,rz": "turns (periodic): free = about its own centre on top of the home rotation (world axes); held = ry turn it round (about up), rx tip toward/away from the camera, rz roll",
                    "hl,hr": "attach to the screen-left/right paw (both = two-paw hold / hand-over)",
                    "hh": "attach to the head (worn: headphones)", "s": "articulation (laptop lid, book cover)"},
        "places": places, "props": props,
        "floor_steps": [{"rect_scene": RUG[:4], "top_scene": RUG[4], "note": "the rug: a prop resting on the floor inside it sits this much higher"}],
    }
    if reach:
        cfg["reach"] = reach
    return cfg


def measure_reach():
    """The paw's grasp point (museGrasp, minus its 0.07 palm offset = the palm contact) and palm normal for a grid of hand
    targets at rest body pose, read back from the renderer's own GLSL arm solve."""
    from props_probe import Probe
    from anim.schema import IDX, NPARAMS
    pr = Probe(CHAR)
    gx = np.linspace(-1.0, 1.2, 12); gy = np.linspace(-1.0, 1.2, 12); gz = np.linspace(-1.0, 1.2, 12)
    out = {}
    for side, pre in ((0, "hand_l_"), (1, "hand_r_")):
        vecs = []
        for x in gx:
            for y in gy:
                for z in gz:
                    v = np.zeros(NPARAMS, "f4"); v[IDX[pre + "x"]] = x; v[IDX[pre + "y"]] = y; v[IDX[pre + "z"]] = z
                    vecs.append(v)
        g = pr.grasp(vecs)[:, side]                      # (N, 4, 3)
        pos = g[:, 0] - g[:, 3]*GRASP_OFS                # palm contact point (scene)
        P = np.stack([pos[:, 0]/H, (pos[:, 1] - BASE)/H, pos[:, 2]/H], -1).reshape(len(gx), len(gy), len(gz), 3)
        N = g[:, 3].reshape(len(gx), len(gy), len(gz), 3)
        out["lr"[side]] = {"param": [pre + "x", pre + "y", pre + "z"], "axes": [gx.round(4).tolist(), gy.round(4).tolist(), gz.round(4).tolist()],
                           "shape": [len(gx), len(gy), len(gz), 3], "layout": "flat, C order: index ((i*ny + j)*nz + k)*3 + c",
                           "pos": [round(float(v), 4) for v in P.reshape(-1)], "palm_normal": [round(float(v), 2) for v in N.reshape(-1)]}
    return out


def reach_stats(reach):
    """trilinear prediction error at the cell centres (held-out points) and inverse-solve residuals."""
    from scipy.interpolate import RegularGridInterpolator
    from props_probe import Probe
    from anim.schema import IDX, NPARAMS
    pr = Probe(CHAR)
    res = {}
    rng = np.random.default_rng(7)
    for side, pre in ((0, "hand_l_"), (1, "hand_r_")):
        r = reach["lr"[side]]
        ax = [np.array(a) for a in r["axes"]]
        P = np.array(r["pos"]).reshape(r["shape"])
        f = RegularGridInterpolator(ax, P)
        pts = np.stack([rng.uniform(a[0], a[-1], 200) for a in ax], -1)
        vecs = []
        for p in pts:
            v = np.zeros(NPARAMS, "f4"); v[IDX[pre + "x"]], v[IDX[pre + "y"]], v[IDX[pre + "z"]] = p
            vecs.append(v)
        g = pr.grasp(vecs)[:, side]
        truth = g[:, 0] - g[:, 3]*GRASP_OFS
        truth = np.stack([truth[:, 0]/H, (truth[:, 1] - BASE)/H, truth[:, 2]/H], -1)
        err = np.linalg.norm(f(pts) - truth, axis=-1)
        res["lr"[side]] = {"mean_H": float(err.mean()), "p95_H": float(np.percentile(err, 95)), "max_H": float(err.max()),
                           "mean_scene": float(err.mean()*H)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reach", action="store_true")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    reach = None
    if os.path.exists(OUT):
        reach = json.load(open(OUT)).get("reach")
    if a.reach:
        reach = measure_reach()
        reach["notes"] = ("pos (flat, see layout) = the palm contact point (world H, rest body pose, root 0, hand_open/turn 0) for hand "
                          "target (axes[0][i], axes[1][j], axes[2][k]); trilinear interpolation predicts it; invert by solving "
                          "pos(h) = P (e.g. a few Gauss-Newton steps from the nearest grid node). palm_normal = out of the palm. "
                          "A held object's grip anchor sits at pos + palm_normal * its extent toward the palm.")
    if a.stats and reach:
        reach["accuracy"] = reach_stats(reach)
    cfg = build(reach)
    reach_c = cfg.pop("reach", None)
    txt = json.dumps(cfg, indent=1)
    if reach_c:      # the reach grids compact (one line each), the rest readable
        txt = txt[:-2] + ',\n "reach": ' + json.dumps(reach_c, separators=(",", ":")) + "\n}"
    open(OUT, "w").write(txt + "\n")
    print("wrote", OUT, "H", H, "source", USRC, "reach", bool(reach))
    if reach and "accuracy" in reach:
        print(json.dumps(reach["accuracy"]))


if __name__ == "__main__":
    main()
