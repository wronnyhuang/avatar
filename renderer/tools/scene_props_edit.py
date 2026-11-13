import json, numpy as np, sys
p = sys.argv[1]
d = json.load(open(p))
H = 2.3
r4 = lambda a: [round(float(x), 4) for x in a]
yaw = np.radians(15.0)
n = np.array([np.sin(yaw), 0, np.cos(yaw)]); u = np.array([np.cos(yaw), 0, -np.sin(yaw)]); up = np.array([0, 1.0, 0])
Cw = np.array([-1.25, 1.27, -2.5]); W, Hb = 2.80, 1.40     # design: a real classroom board
O = Cw - u*W/2 + up*Hb/2
def st(a):
    P = O + u*(a*W + 0.54) + n*0.70; P[1] = 0; return r4(P/H)
TRAY = -(Hb/2 + 0.04 + 0.05)          # local tray top (below the frame)
tray_y = Cw[1] + TRAY
def tray(xl, yoff=0.0):
    P = Cw + u*xl + n*0.05 + up*(TRAY + yoff); return r4(P/H)
board = {"id": "board", "name": "the chalkboard", "aliases": ["board", "chalkboard", "blackboard", "chalk board", "slate"],
  "kind": "chalkboard", "movable": False, "center": r4(Cw/H), "top_y": round(Cw[1]/H, 4), "floor_y": 0.0,
  "stand_top_y": round((Cw[1] + Hb/2 + 0.17)/H, 4), "size": r4([W/H, Hb/H]), "yaw": round(15/360, 6), "rgb": [150, 104, 64],
  "plane": {"origin": r4(O/H), "u": r4(u), "v": [0.0, -1.0, 0.0], "n": r4(n), "size": r4([W/H, Hb/H]),
            "uv": "(0,0) top-left of the writable area, x along u, y along v (down); P = origin + u*a*size[0] + v*b*size[1]",
            "surface": {"name": "board", "px": [1024, 512], "aspect": 2.0, "width_units": "v (fraction of the writable height)"}},
  "station": {"pos": st(0.5), "yaw": -0.0833, "side": "l", "line": {"from": st(0.0), "to": st(1.0)},
              "s_of_a": "station(a) = origin_floor + u*(a*size[0] + 0.2348) + n*0.3043 (H), a = board uv-x of the writing column",
              "aside": st(1.2), "window_uv": [-0.10, 0.10], "y_range_H": [0.278, 0.661], "reach_uv_y": [round((Cw[1]+Hb/2-1.52)/Hb, 3), round((Cw[1]+Hb/2-0.64)/Hb, 3)],
              "note": "stand right of the writing point, turned 30 deg toward screen-left (root_yaw -0.0833), write with the screen-left paw (hand_l); shift along the line between columns"},
  "camera": {"cam_zoom": -0.50, "cam_x": -0.18, "cam_y": 0.10},
  "spots": [], "notes": "scene: free-standing oak-framed slate on two posts, faces the room, turned 15 deg toward screen-right, behind and left of home; chalk + eraser on its tray"}
traysp = [tray(0.80, 0.0), tray(-0.60, 0.0), tray(0.30, 0.0), tray(-0.10, 0.0)]
board["spots"] = traysp           # "put it on the board" = on its tray
board_tray = {"id": "board_tray", "name": "the chalk tray", "aliases": ["tray", "chalk tray", "board tray", "ledge"],
  "kind": "board_tray", "movable": False, "on_place": "board", "center": tray(0.0), "top_y": round(tray_y/H, 4),
  "size": r4([(W+0.18)/H, 0.10/H]), "yaw": round(15/360, 6), "spots": traysp,
  "notes": "the ledge along the bottom of the board frame (0.10 su deep); spots: chalk, eraser, two spare"}
tvC = np.array([2.75, 1.95, -3.34]); tW, tH = 1.92, 1.08
tO = tvC + np.array([-tW/2, tH/2, 0])
tv = {"id": "tv", "name": "the TV", "aliases": ["tv", "television", "screen", "monitor", "telly", "wall screen"],
  "kind": "wall_tv", "movable": False, "center": r4(tvC/H), "top_y": round(tvC[1]/H, 4), "floor_y": round((tvC[1]-0.62)/H, 4),
  "stand_top_y": round((tvC[1]+0.62)/H, 4), "size": r4([tW/H, tH/H]), "yaw": 0.0, "rgb": [18, 18, 20],
  "screen": {"origin": r4(tO/H), "u": [1.0, 0.0, 0.0], "v": [0.0, -1.0, 0.0], "n": [0.0, 0.0, 1.0], "size": r4([tW/H, tH/H]),
             "surface": {"name": "tv", "px": [512, 288]}},
  "station": {"pos": r4(np.array([0.80, 0.0, -1.95])/H), "yaw": 0.07, "side": "r",
              "note": "present: stand left of the screen, slightly turned toward it, point / gesture with the screen-right paw"},
  "camera": {"cam_zoom": -0.60, "cam_x": 0.565, "cam_y": 0.21},
  "placeable": False, "spots": [], "notes": "scene: wall-mounted on the back wall (z -1.478 H), above/behind the desk; emissive picture"}
ST_NEW = np.array([2.30, 1.00]) / H         # side table centre x, z (H): front-right of home
def shift_side_table(d):
    st = [p for p in d["places"] if p["id"] == "side_table"][0]
    if abs(st["center"][0] - ST_NEW[0]) < 1e-3:
        return
    dx, dz = ST_NEW[0] - st["center"][0], ST_NEW[1] - st["center"][2]
    mv = lambda v: [round(v[0] + dx, 5), v[1], round(v[2] + dz, 5)]
    st["center"] = mv(st["center"]); st["spots"] = [mv(v) for v in st["spots"]]
    st["notes"] = st.get("notes", "") + " (scene: moved front-right of home so it no longer hides the chalkboard)"
    for pr in d["props"]:
        if pr["home"].get("place") != "side_table":
            continue
        pr["home"]["pos"] = mv(pr["home"]["pos"])
        for h in pr.get("handles") or []:
            if "world_home" in h: h["world_home"] = mv(h["world_home"])
        pu = pr.get("pick_up_from")
        if pu and "stand_at" in pu: pu["stand_at"] = mv(pu["stand_at"])
shift_side_table(d)
# replace / append places
ids = [pl["id"] for pl in d["places"]]
for pl in (board, board_tray, tv):
    if pl["id"] in ids: d["places"][ids.index(pl["id"])] = pl
    else: d["places"].append(pl)
d["home"] = {"pos": [0.0, 0.0, 0.0], "yaw": 0.0, "camera": {"cam_zoom": 0.0, "cam_x": 0.0, "cam_y": 0.0}}
chalk = {"slot": 14, "id": "chalk", "name": "white chalk", "color": "white", "colors": ["white"],
  "aliases": ["chalk", "piece of chalk", "chalk stick", "stick of chalk"], "category": "stationery", "kind": "chalk",
  "size": round(0.075/H, 5), "mass": "light", "movable": True, "furniture": False, "scale": round(0.075/H, 6),
  "rgb": [236, 234, 224], "home": {"place": "board_tray", "pos": tray(0.80, 0.0165), "rot": [0.25, round(105/360, 5), 0.0]},
  "grip": "pinch", "throwable": False, "facing": [0, 0, 1],
  "tip": {"local_unit": [0.0, 1.0, 0.0], "grasp_to_tip_scene": [0.0, 0.1013, 0.0], "grasp_to_tip_H": [0.0, 0.044, 0.0],
          "frame": "held: tip = gp + G*(offs + Rcol*grasp_to_tip), G = upright camera-facing frame, Rcol = prEuler(2pi(rx,ry,rz))*prEuler(2pi*home.rot) (docs/SCENE.md 3)"}}
eraser = {"slot": 15, "id": "eraser", "name": "board eraser", "color": "grey", "colors": ["grey", "oak", "brown"],
  "aliases": ["eraser", "board eraser", "duster", "board duster", "board rubber", "rubber"], "category": "stationery",
  "kind": "eraser", "size": round(0.09/H, 5), "mass": "light", "movable": True, "furniture": False, "scale": round(0.09/H, 6),
  "rgb": [168, 120, 76], "rgb2": [128, 128, 130], "home": {"place": "board_tray", "pos": tray(-0.60, 0.0324), "rot": [0.0, round(15/360, 5), 0.0]},
  "grip": "power", "throwable": False, "facing": [0, 0, 1],
  "face": {"local_unit": [0.0, -1.0, 0.0], "grasp_to_face_scene": [0.0, -0.0648, 0.0], "footprint_scene": [0.18, 0.076],
           "erase": "erase_rect 0.10 x 0.084 uv (or erase_disc r 0.05 v) per contact"}}
pids = [pr["id"] for pr in d["props"]]
for pr in (chalk, eraser):
    if pr["id"] in pids: d["props"][pids.index(pr["id"])] = pr
    else: d["props"].append(pr)
json.dump(d, open(p, "w"), indent=1)
print(json.dumps(board["plane"]), board["station"]["pos"], st(0), st(1))
# re-serialise in the file's own style: indent 1, the reach map compact on one line
reach = d.pop("reach")
txt = json.dumps(d, indent=1)
assert txt.endswith("\n}")
txt = txt[:-2] + ',\n "reach": ' + json.dumps(reach, separators=(",", ":")) + "\n}"
open(p, "w").write(txt + ("\n" if open(p).read().endswith("\n") and False else ""))
