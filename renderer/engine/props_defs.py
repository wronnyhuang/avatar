"""props: GLSL defines for a character's props and places, generated from renderer/props/<character>.json.

    place_head, props_head = defines("fox")

place_head goes into the shader header (it shapes the baked set, so it is part of the background cache key);
props_head is appended to every frame pass only (props are drawn live, never baked).

Units: the JSON is in body heights H (docs/INTENT-RULES.md): world point = ORG + H*pos, ORG = (0, base_y, 0) in scene
units.  H and base_y come from renderer/body/<character>.json when the body agent has published it, else from the props
file's own "units" block (the props agent measured them with the same definition).
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# kind -> (PK id, half extents of its box (unit frame), grip anchor, grip radius, default grip, bounding radius, furniture)
#   grip anchor = where the paw closes on it (unit frame); grip radius = what the digits wrap there
CATALOG = {
    "ball":       (1,  (0.98, 0.98, 0.98), (0.0, 0.0, 0.0),   0.98, "power",    1.00, 0),
    "mug":        (2,  (0.80, 0.84, 0.68), (0.0, -0.05, 0.0), 0.68, "power",    1.12, 0),
    "book":       (3,  (0.68, 0.88, 0.21), (0.0, -0.30, 0.0), 0.21, "power",    1.60, 0),
    "duck":       (4,  (0.66, 0.82, 1.00), (0.0, -0.25, 0.0), 0.66, "power",    1.08, 0),
    "apple":      (10, (0.86, 0.95, 0.86), (0.0, -0.06, 0.0), 0.86, "power",    1.06, 0),
    "plant":      (21, (0.55, 0.97, 0.55), (0.0, -0.55, 0.0), 0.50, "power",    1.10, 0),
    "headphones": (22, (1.30, 1.15, 0.34), (0.0, 0.95, 0.0),  0.06, "power",    1.65, 0),
    "chair":      (23, (0.55, 1.00, 0.55), (0.0, 0.0, 0.0),   0.05, "two_hand", 1.30, 1),
    "desk":       (24, (1.00, 0.68, 0.55), (0.0, 0.0, 0.0),   0.05, "two_hand", 1.35, 1),
    "laptop":     (25, (1.00, 0.08, 0.70), (0.0, 0.0, 0.0),   0.08, "two_hand", 1.75, 0),
    # OBJECTS (PK 30-39; SCENE uses 26-29 / 40+): unit-frame models in engine/props.glsl
    "hat":          (30, (1.00, 0.36, 1.00), (0.0, 0.30, 0.0),  0.30, "power",    1.08, 0),
    "glasses":      (31, (0.80, 0.27, 0.10), (0.48, 0.0, 0.0),  0.06, "pinch",    1.00, 0),
    "box":          (32, (1.00, 0.80, 0.90), (0.0, 0.0, 0.0),   0.05, "two_hand", 1.58, 0),
    "watering_can": (33, (1.00, 0.80, 0.55), (-0.62, 0.55, 0.0), 0.08, "power",   1.45, 0),
    "block":        (34, (0.90, 0.90, 0.90), (0.0, 0.0, 0.0),   0.90, "power",    1.30, 0),
    # scene (docs/SCENE.md section 3): held by the stick behind its middle / by the eraser's back
    "chalk":      (26, (0.22, 1.00, 0.22), (0.0, -0.35, 0.0), 0.22, "pinch",    1.03, 0),
    "eraser":     (27, (1.00, 0.36, 0.42), (0.0, 0.36, 0.0),  0.36, "power",    1.12, 0),
}
# a CONSERVATIVE box round each model in its unit frame at any articulation (centre, half extents): the ray masks
# (engine/props.glsl propsRayMask) cull with it.  None = from the fitted extents (+ a margin).
BBOX = {
    "ball": ((0.0, 0.0, 0.0), (1.02, 1.02, 1.02)), "mug": ((0.0, 0.0, 0.0), (1.20, 0.92, 0.92)),
    "book": ((0.0, 0.0, 0.0), (1.48, 0.98, 0.80)), "duck": ((0.0, 0.0, 0.0), (0.90, 1.05, 1.10)),
    "apple": ((0.0, 0.0, 0.0), (0.95, 1.10, 0.95)), "plant": ((0.0, 0.0, 0.0), (1.00, 1.10, 1.00)),
    "laptop": ((0.0, 0.62, -0.25), (1.05, 0.74, 1.00)), "headphones": None, "chair": None, "desk": None,
    "hat": ((0.0, 0.0, 0.0), (1.03, 0.40, 1.03)), "block": ((0.0, 0.0, 0.0), (0.94, 0.94, 0.94)),
    "chalk": ((0.0, 0.0, 0.0), (0.26, 1.06, 0.26)), "eraser": ((0.0, 0.0, 0.0), (1.04, 0.40, 0.46)),
}
# the box and the bounding radius while ARTICULATED (prop{k}_s > 0), when the model reaches further than shut
# (a lid swung open, arms unfolded, a stream of water below a spout).  (centre, half extents), radius.
BBOX_ART = {
    "box": (((0.0, 0.62, -0.30), (1.04, 1.44, 1.24)), 1.95),
    "glasses": (((0.0, 0.0, -0.62), (1.30, 0.32, 0.74)), 1.50),
    "watering_can": (((0.20, -1.10, 0.0), (2.60, 2.70, 2.60)), 3.10),
}
BBOX.update({"glasses": ((0.0, 0.0, -0.10), (0.82, 0.30, 0.24)), "box": ((0.0, 0.0, 0.0), (1.04, 0.86, 0.94)),
             "watering_can": ((0.10, 0.10, 0.0), (1.12, 0.92, 0.62))})
# kinds held by a handle the paw closes round (contact = the grip radius at the anchor, not the box extent)
ANCHOR_GRIP = {"headphones", "watering_can", "chalk"}
# cost: kinds whose shadow ON THE CHARACTER (and on other props) is a soft ellipsoid inscribed in their BBOX,
# not their model: the character pass then carries no shadow copy of those models (every model call site inlines all
# of them; K620: ~2 ms at 270p).  Only solid, compact objects (never a band round the head, never thin frames on the
# face: those keep their exact shadow).  A prop never shadows itself through its proxy.
SHADOW_PROXY = {"hat", "box", "watering_can", "block"}
GRIPS = {"power": 0, "pinch": 1, "two_hand": 2}
PROPS_MAX = 16          # slots the renderer draws (anim/schema.py prop0..prop15)
# place kinds (engine/places.glsl)
PLACE_KINDS = {"stump": 1, "rock": 2, "table": 3, "plinth": 4, "crate": 5, "leaf": 6, "log": 7, "table_stand": 8,
               "chalkboard": 9, "wall_tv": 10}      # scene: the chalkboard (+ tray, posts) and the wall TV


def _lit(v):
    if isinstance(v, (list, tuple)):
        return f"vec{len(v)}(" + ",".join(f"{float(x):.6f}" for x in v) + ")"
    return f"{float(v):.6f}"


def srgb_to_lin(c):
    return [((x/255.0 if x > 1.0 else x)/12.92) if (x/255.0 if x > 1.0 else x) <= 0.04045
            else (((x/255.0 if x > 1.0 else x) + 0.055)/1.055)**2.4 for x in c]


def props_path(character):
    t = os.environ.get("AVATAR_PROPS_JSON")          # test hook: any character with a given props file
    if t and (os.environ.get("AVATAR_PROPS_FOR", character) == character):
        return t
    return os.path.join(ROOT, "animals", f"{character}.props.json")


def units(character):
    """(H scene units per body height, base_y scene y of the H frame's origin, source)."""
    bp = os.path.join(ROOT, "animals", f"{character}.body.json")
    if os.path.exists(bp):
        try:
            b = json.load(open(bp))
            return float(b["H"]), float(b["base_y"]), "body"
        except Exception:
            pass
    pp = props_path(character)
    if os.path.exists(pp):
        u = json.load(open(pp)).get("units", {})
        if "H" in u:
            return float(u["H"]), float(u["base_y"]), "props"
    return 1.0, 0.0, "none"


def load(character):
    pp = props_path(character)
    return json.load(open(pp)) if os.path.exists(pp) else None


def to_scene(pos, H, base_y):
    return [H*pos[0], base_y + H*pos[1], H*pos[2]]


TABLES = {}       # character -> {name: (glsl type, values)}: rt_engine sets them as uniforms in the PROPS_UNI passes


def _arr(name, typ, vals):
    """A per-slot table: a uniform array in the PROPS_UNI passes (GPU constant memory: a dynamically indexed const
    array is materialised per fragment in slow local memory on the K620), a const array elsewhere (bake, probes)."""
    n = len(vals)
    lit = (lambda v: str(int(v))) if typ == "int" else (lambda v: _lit(v))
    return (f"#ifdef PROPS_UNI\nuniform {typ} {name}[{n}];\n#else\n"
            f"const {typ} {name}[{n}] = {typ}[{n}](" + ", ".join(lit(v) for v in vals) + ");\n#endif")


def tables(character):
    return TABLES.get(character, {})


def defines(character):
    cfg = load(character)
    if not cfg or os.environ.get("AVATAR_NO_PROPS"):
        return "", "#define PROPS_N 0\n"      # no props file: nothing in the (keyed) header
    import sys
    sys.path.insert(0, ROOT)
    from anim.schema import IDX
    H, base_y, _ = units(character)
    ph = []
    places = [p for p in cfg.get("places", []) if p.get("kind") in PLACE_KINDS and not p.get("movable")]
    places = places[:4]
    ph.append(f"#define PLACES_N {len(places)}")
    for i, pl in enumerate(places):
        c = to_scene(pl["center"], H, base_y)
        top = base_y + H*float(pl["top_y"])
        sz = [H*float(pl["size"][0]), H*float(pl["size"][1])]
        ph += [f"#define PLACE{i}_KIND {PLACE_KINDS[pl['kind']]}",
               f"#define PLACE{i}_C {_lit([c[0], top, c[2]])}",        # centre of its TOP face
               f"#define PLACE{i}_SIZE {_lit(sz)}",                     # width (x), depth (z)
               f"#define PLACE{i}_YAW {_lit(6.283185307*float(pl.get('yaw', 0.0)))}",
               f"#define PLACE{i}_FLOOR {_lit(base_y + H*float(pl.get('floor_y', 0.0)))}",
               f"#define PLACE{i}_COL {_lit(srgb_to_lin(pl.get('rgb', [120, 90, 60])))}",
               f"#define PLACE{i}_SEED {_lit(float(i)*3.7 + 1.3)}",
               f"#define PLACE{i}_POST {_lit(H*(float(pl.get('stand_top_y', pl['top_y'])) - float(pl['top_y'])))}"]
        if pl["kind"] in ("chalkboard", "wall_tv"):      # scene: the live surface pass's planes (scene units)
            import math
            a = 6.283185307*float(pl.get("yaw", 0.0))
            tag = "BOARD" if pl["kind"] == "chalkboard" else "TV"
            ph += [f"#define PLACE_{tag} {i}", f"#define {tag}_C {_lit([c[0], top, c[2]])}",
                   f"#define {tag}_U {_lit([math.cos(a), 0.0, -math.sin(a)])}", f"#define {tag}_N {_lit([math.sin(a), 0.0, math.cos(a)])}",
                   f"#define {tag}_SZ {_lit(sz)}"]
    # (scene: an entry may name its "slot"; slots beyond the renderer's capacity are left out — the page still
    # sees them in config.props.  Slot = list index, as before, for every entry that names none.)
    props = [p for i, p in enumerate(cfg.get("props", [])) if int(p.get("slot", i)) == i][:PROPS_MAX]
    rnd = cfg.get("renderer", {})
    out = [f"#define PROPS_N {len(props)}", f"#define PROPS_H {_lit(H)}"]
    if rnd.get("grasp_fn"):
        out.append(f"#define PROPS_GRASP_FN {rnd['grasp_fn']}")
    if rnd.get("head_fn"):
        out.append(f"#define PROPS_HEAD_FN {rnd['head_fn']}")
    if rnd.get("palm_sign"):
        out.append(f"#define PROPS_PALM_SIGN {_lit(rnd['palm_sign'])}")
    if rnd.get("char_bound") == "muse":
        out.append("#define PROPS_BOUND_MUSE 1")
    if rnd.get("grasp_offset"):
        out.append(f"#define PROPS_GRASP_OFS {_lit(rnd['grasp_offset'])}")
    cols = {k: [] for k in ("PR_PX", "PR_PHH", "PR_PS", "PR_KIND", "PR_GRIP", "PR_FREE", "PR_HOME", "PR_ROT", "PR_SIZE",
                            "PR_BND", "PR_COL", "PR_COL2", "PR_GA", "PR_GR", "PR_EXT", "PR_SUPR", "PR_ONFLOOR",
                            "PR_BBC", "PR_BBH", "PR_BBC2", "PR_BBH2", "PR_BND2", "PR_VAR")}
    kinds = set()
    for k, pr in enumerate(props):
        kind = pr["kind"]
        pk, ext, ga, gr, grip, bnd, free = CATALOG[kind]
        kinds.add(kind)
        fitted = kind == "headphones"     # its grip anchor comes from the fit, not from grip_point_local
        if kind == "headphones":          # fitted to the head: cup spacing W and the crown arc (unit frame = head frame)
            hp = pr.get("fit", {})
            W, A = float(hp.get("w", 0.80)), float(hp.get("arc", 1.10))
            out += [f"#define PROPS_HP_W {_lit(W)}", f"#define PROPS_HP_ARC {_lit(A)}",
                    f"#define PROPS_HP_WR {_lit(float(hp.get('w_rest', W)))}",
                    f"#define PROPS_HP_BZ {_lit(hp.get('bz', 0.0))}"]
            ext = (W + 0.30, max(A + 0.06, 0.34), 0.34)
            ga = (0.0, A + 0.05, 0.0)       # held by the band's crown
            bnd = ((W + 0.30)**2 + (A + 0.06)**2 + 0.34**2)**0.5
        size = H*float(pr.get("scale", 1.0))
        if kind in ("chair", "desk"):     # furniture is built in scene units from its dimensions (H in the file)
            dims = [H*float(v) for v in pr["dims"]]
            size = 1.0
            if kind == "chair":
                out.append(f"#define PROPS_CHAIR {_lit(dims)}")
                ext = (0.5*dims[0] + 0.02, 0.5*dims[3], 0.5*dims[1] + 0.12)
            else:
                out.append(f"#define PROPS_DESK {_lit(dims)}")
                ext = (0.5*dims[0], 0.5*dims[2], 0.5*dims[1])
            bnd = (ext[0]**2 + ext[1]**2 + ext[2]**2)**0.5 + 0.05
        cols["PR_PX"].append(IDX[f"prop{k}_x"])
        cols["PR_PHH"].append(IDX.get(f"prop{k}_hh", -1))
        cols["PR_PS"].append(IDX.get(f"prop{k}_s", -1))
        cols["PR_KIND"].append(pk)
        cols["PR_GRIP"].append(GRIPS[pr.get("grip", grip)])
        cols["PR_FREE"].append(1 if pr.get("furniture", free) else 0)
        cols["PR_HOME"].append(to_scene(pr["home"]["pos"], H, base_y))
        cols["PR_ROT"].append([6.283185307*float(a) for a in pr["home"].get("rot", [0, 0, 0])])   # turns in the JSON
        cols["PR_SIZE"].append(size)
        cols["PR_BND"].append(bnd)
        cols["PR_COL"].append(srgb_to_lin(pr.get("rgb", [200, 60, 50])))
        cols["PR_COL2"].append(srgb_to_lin(pr.get("rgb2", pr.get("rgb", [200, 60, 50]))))
        if "grip_point_local" in pr and not fitted:      # the page's grip point (prop-local H) -> unit frame
            ga = tuple(float(v)/float(pr.get("scale", 1.0)) for v in pr["grip_point_local"])
        cols["PR_GA"].append(pr.get("grip_anchor", ga))
        cols["PR_GR"].append(pr.get("grip_r", gr))
        cols["PR_EXT"].append(ext)
        bb = BBOX.get(kind)
        cols["PR_BBC"].append(bb[0] if bb else (0.0, 0.0, 0.0))
        cols["PR_BBH"].append(bb[1] if bb else tuple(float(v) + 0.04 for v in ext))
        ba = BBOX_ART.get(kind)          # articulated: its own box + radius, else the shut ones
        cols["PR_BBC2"].append(ba[0][0] if ba else cols["PR_BBC"][-1])
        cols["PR_BBH2"].append(ba[0][1] if ba else cols["PR_BBH"][-1])
        cols["PR_BND2"].append(ba[1] if ba else bnd)
        # contact rule on the palm: 0 = the object's box extent toward the palm (books, mugs, boxes: whatever side
        # faces the palm), 1 = the grip radius at its anchor (a thin handle / band the paw closes round)
        # (props JSON "grip_contact": "anchor" | "extent" overrides the kind's default)
        cols["PR_SUPR"].append(1 if pr.get("grip_contact", "anchor" if kind in ANCHOR_GRIP else "extent") == "anchor" else 0)
        lt = str(pr.get("letter", ""))[:1]   # a per-slot variant (the toy block's letter) for the painter
        cols["PR_VAR"].append(ord(lt) if lt else 0)
        cols["PR_ONFLOOR"].append(1 if pr["home"].get("place") == "floor" else 0)
    for st in cfg.get("floor_steps", [])[:1]:
        out += [f"#define PROPS_FLOOR_STEP {_lit(st['rect_scene'])}", f"#define PROPS_FLOOR_STEP_H {_lit(st['top_scene'])}"]
    tab = TABLES.setdefault(character, {})
    for name, vals in cols.items():
        typ = "int" if name in ("PR_PX", "PR_PHH", "PR_PS", "PR_KIND", "PR_GRIP", "PR_FREE", "PR_SUPR", "PR_ONFLOOR", "PR_VAR") else \
              ("float" if name in ("PR_SIZE", "PR_BND", "PR_BND2", "PR_GR") else "vec3")
        out.append(_arr(name, typ, vals))
        tab[name] = (typ, vals)
    for kind in sorted(kinds):
        out.append(f"#define PROPKIND_{kind.upper()} 1")
        if kind in SHADOW_PROXY:
            out.append(f"#define SPROXY_{kind.upper()} 1")
    return "\n".join(ph) + "\n", "\n".join(out) + "\n"


if __name__ == "__main__":
    import sys
    a, b = defines(sys.argv[1] if len(sys.argv) > 1 else "muse")
    print(a); print(b)
