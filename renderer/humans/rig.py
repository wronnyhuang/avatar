"""RIG (seam G1, docs/CONTRACT--rig.md): the shared control vector (anim/schema.py PARAMS) -> Juno's bone
world matrices, blendshape weights, grasp / head frames and shadow proxy capsules.  Pure numpy, deterministic, no GL.

The rig follows animals/muse.glsl's semantics param for param (same signs, same hand-target mapping, same finger-curl
formula, same prop-hold curls, same gaze rule as engine/face.glsl + scene_main.glsl rootGaze), with a human skeleton:
  * every posed bone = a WORLD-space rotation delta D on its rest orientation:  Q = D @ Q_rest,  so the rig never depends
    on the asset's bone-local axes (only on rest joint positions and rest orientations);
  * FK runs in skeleton order: p_child = p_parent + D_parent @ (r_child - r_parent) * k  (k = asset metres -> scene units);
  * solver hooks (spine, neck/head, clavicle, two-bone IK arms/legs, hands, fingers, feet, jaw, eyes) set D for their bone.

Frames: CHARACTER-LOCAL = Muse's (origin on the floor under the character, x screen-right, y up, z toward the camera,
scene units); world = engine root transform (A, o), exactly rt_engine._body_frame's.  Sides: param _l = screen-left =
anatomical RIGHT (bones *_r, ARKit ...Right); side index 0 = screen-left everywhere (museGrasp's side).

    rig = JunoRig(asset_dir)                         # skeleton.json + juno.asset.json (+ lodN/mesh.json for correctives)
    f = rig.solve(P, t=0.0, cam_pos=(0, 1.3, 7.7))   # RigFrame
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RENDER_ROOT = os.path.dirname(HERE)
if RENDER_ROOT not in sys.path:
    sys.path.insert(0, RENDER_ROOT)
from anim.schema import IDX, NPARAMS  # noqa: E402

JUNO_DIR = os.path.join(HERE, "juno")
SQUASH_K = 0.55          # = rt_engine.SQUASH_K
MOTION_H = 2.30         # the shared motion unit (muse.glsl M_H: foot offsets; props H)
GRASP_OFS = 0.022        # = muse.props.json renderer.grasp_offset (props.glsl PROPS_GRASP_OFS)
MUSE_ARM_L = 0.57        # Muse's arm length (scene): Muse-tuned height thresholds scale by L / MUSE_ARM_L
SIDES = ("l", "r")       # param side names, index 0 = screen-left
ANAT = ("r", "l")        # the anatomical side for each param side
ARK = ("Right", "Left")  # ARKit suffix for each param side
DIGITS = ("index", "middle", "ring", "pinky", "thumb")
FLEX = (math.radians(70.0), math.radians(95.0), math.radians(65.0))      # finger joints at curl 1
TFLEX = (0.0, math.radians(35.0), math.radians(45.0))                    # thumb ip joints at curl 1


# ------------------------------------------------------------------ small math (all 3x3 / 3-vectors, float64)
def _v(x):
    return np.asarray(x, dtype=np.float64)


def _n(v):
    L = math.sqrt(float(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]))
    return v/L if L > 1e-12 else v*0.0


def cross(a, b):
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2]); b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    return np.array([a1*b2 - a2*b1, a2*b0 - a0*b2, a0*b1 - a1*b0])


def axis_angle(ax, a):
    """Rodrigues rotation matrix (right-handed, column vectors)."""
    ax = _n(_v(ax)); x, y, z = ax
    c, s = math.cos(a), math.sin(a); t = 1.0 - c
    return np.array([[t*x*x + c, t*x*y - s*z, t*x*z + s*y],
                     [t*x*y + s*z, t*y*y + c, t*y*z - s*x],
                     [t*x*z - s*y, t*y*z + s*x, t*z*z + c]])


def rotX(a):
    c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rotY(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rotZ(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def align(a, b):
    """The smallest rotation taking direction a onto direction b."""
    a = _n(_v(a)); b = _n(_v(b))
    c = float(np.dot(a, b)); ax = cross(a, b); s = float(np.linalg.norm(ax))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        p = cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else cross(a, [0, 1.0, 0])
        return axis_angle(p, math.pi)
    return axis_angle(ax/s, math.atan2(s, c))


def frame_from(u, n):
    """Orthonormal frame (columns) from a primary direction u and a secondary n (made perpendicular)."""
    u = _n(u); n = n - u*np.dot(n, u); n = _n(n)
    return np.array([u, n, cross(u, n)]).T


def rot_power(R, f):
    """R^f (fraction of the rotation about its own axis)."""
    c = (np.trace(R) - 1.0)*0.5
    a = math.acos(max(-1.0, min(1.0, c)))
    if a < 1e-9:
        return np.eye(3)
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])/(2.0*math.sin(a))
    return axis_angle(ax, a*f)


def rest_hinge(u1, u2, bend_dir):
    """The rest-pose hinge axis of a two-bone chain: cross(u1, u2) when the rest chain is visibly bent, else the
    anatomical bend direction (arms: the forearm bends forward; legs: the shin bends back)."""
    n = cross(u1, u2)
    if np.linalg.norm(n) > math.sin(math.radians(4.0)):
        return _n(n)
    return _n(cross(u1, bend_dir))


def rodrigues(ax, a):
    """Batched axis_angle: ax (n,3) unit, a (n,) -> (n,3,3)."""
    x, y, z = ax[:, 0], ax[:, 1], ax[:, 2]
    c, s = np.cos(a), np.sin(a); t = 1.0 - c
    R = np.empty((len(a), 3, 3))
    R[:, 0, 0] = t*x*x + c; R[:, 0, 1] = t*x*y - s*z; R[:, 0, 2] = t*x*z + s*y
    R[:, 1, 0] = t*x*y + s*z; R[:, 1, 1] = t*y*y + c; R[:, 1, 2] = t*y*z - s*x
    R[:, 2, 0] = t*x*z - s*y; R[:, 2, 1] = t*y*z + s*x; R[:, 2, 2] = t*z*z + c
    return R


def ss(e0, e1, x):
    if e0 == e1:
        return 1.0 if x >= e1 else 0.0
    t = min(1.0, max(0.0, (x - e0)/(e1 - e0))); return t*t*(3.0 - 2.0*t)


def clamp(x, a, b):
    return a if x < a else b if x > b else x


def mix(a, b, t):
    return a + (b - a)*t


def ik2(a, t, l1, l2, pole, reach=0.995):
    """animals/muse.glsl mIK: two-bone IK, returns (middle joint, clamped target)."""
    d = t - a; L = float(np.linalg.norm(d))
    lmax = (l1 + l2)*reach; lmin = abs(l1 - l2) + 0.02*(l1 + l2)/0.57
    dr = d/max(L, 1e-6)
    L = clamp(L, lmin, lmax)
    t = a + dr*L
    ca = clamp((l1*l1 + L*L - l2*l2)/(2.0*l1*L), -1.0, 1.0)
    pp = pole - dr*np.dot(pole, dr)
    pp = _n(pp) if np.dot(pp, pp) > 1e-10 else _n(cross(dr, [0, 0, 1.0]))
    return a + l1*(ca*dr + math.sqrt(max(0.0, 1.0 - ca*ca))*pp), t


# ------------------------------------------------------------------ the engine's root transform (rt_engine._body_frame)
def engine_rot(yaw, pitch, roll):
    a, b, c = 2*np.pi*yaw, 2*np.pi*pitch, -2*np.pi*roll
    return rotY(a) @ rotX(b) @ rotZ(c)


def engine_body(char_cfg, body_json=None, body_dir=None):
    """rt_engine.load_body for this character: renderer/body/<c>.json if present, else the camera-framing fallback."""
    if body_dir and os.path.isfile(os.path.join(body_dir, "juno.json")):
        b = json.load(open(os.path.join(body_dir, "juno.json")))
        H = float(b["H"]); org = _v(b["origin_scene"]); com = org + _v(b["com"])*H
        return dict(H=H, origin=org, C=com, B=np.array([com[0], org[1], com[2]]))
    cam = char_cfg["camera"]
    vis = 2.0*np.linalg.norm(np.subtract(cam["target"], cam["pos"]))*np.tan(np.radians(cam["fov"])*0.5)
    H = 0.8*vis; gy = float(char_cfg.get("set_params", {}).get("ground_y", cam["target"][1] - 0.45*vis))
    org = np.array([0.0, gy, 0.0]); com = org + np.array([0.0, 0.45*H, 0.0])
    return dict(H=H, origin=org, C=com, B=org.copy())


def root_transform(P, eb):
    x, y, z = P[IDX["root_x"]], P[IDX["root_y"]], P[IDX["root_z"]]
    yaw, pitch, roll, sq = P[IDX["root_yaw"]], P[IDX["root_pitch"]], P[IDX["root_roll"]], P[IDX["squash"]]
    if not any(abs(float(v)) > 1e-7 for v in (x, y, z, yaw, pitch, roll, sq)):
        return np.eye(3), np.zeros(3), np.eye(3), 1.0
    R = engine_rot(yaw, pitch, roll)
    sy = float(np.exp(-SQUASH_K*np.clip(sq, -1.0, 1.0))); sx = 1.0/np.sqrt(sy)
    A = R @ np.diag([sx, sy, sx])
    o = eb["H"]*np.array([x, y, z], float) + eb["C"] + R @ (eb["B"] - eb["C"]) - A @ eb["B"]
    return A, o, R, min(sx, sy)


# ------------------------------------------------------------------ output
@dataclass
class RigFrame:
    bones: np.ndarray                 # f32[nb,4,4] world (root applied), applies to asset-space rest via inverse_bind
    skin: np.ndarray                  # f32[nb,4,4] bones @ inverse_bind
    bones_local: np.ndarray           # f32[nb,4,4] char-local (before the root)
    weights: np.ndarray               # f32[nshapes] asset shape order
    eyes: np.ndarray                  # f32[2,3] eyeball centres world (anatomical: 0 = eye_l bone, 1 = eye_r)
    gaze: np.ndarray                  # f32[3] world point the eyes converge on
    grasp_local: np.ndarray           # f32[2,4,4] side 0 = screen-left; cols x across, y fingertips, z palm normal, w pos
    grasp: np.ndarray                 # f32[2,4,4] world
    head_local: np.ndarray            # f32[4,4] worn-prop head frame (museHeadFrame semantics)
    head: np.ndarray                  # f32[4,4] world
    head_center_local: np.ndarray     # f32[3]  (gHO)
    head_rot_local: np.ndarray        # f32[3,3] head axes in char-local (= transpose(gHR))
    proxies_local: np.ndarray         # f32[K,8] capsules (a, b, r, kind) char-local
    proxies: np.ndarray               # f32[K,8] world (r scaled by root_lip)
    bound: np.ndarray                 # f32[4] world sphere
    hold_w: np.ndarray                # f32[2]
    hold_r: np.ndarray                # f32[2]
    hold_g: np.ndarray                # i32[2]  0 power, 1 pinch, 2 two-hand
    flags: int
    root_A: np.ndarray
    root_o: np.ndarray
    root_R: np.ndarray
    root_lip: float
    mat: dict = field(default_factory=dict)
    marks: dict = field(default_factory=dict)
    joints: dict = field(default_factory=dict)   # named char-local joint positions (tests / sheets)


MAT_KEYS = ("blush", "pale", "green", "tear_well", "tear_drop", "sweat", "vein", "brow_bold", "pupil_scale",
            "catchlight", "eye_shake", "eye_scale", "flush_ears")
MARK_KEYS = ("fx_question", "fx_exclaim", "fx_heart", "fx_sparkle", "fx_steam", "fx_scribble", "fx_dots", "fx_lines")
PROXY_KINDS = {"head": 0, "neck": 1, "torso": 2, "pelvis": 3, "arm": 4, "hand": 5, "leg": 6, "foot": 7, "bun": 8}


_HF_CACHE = {}


def _human_face(face, shape_names, asset_dir, lod):
    """rig4: HumanFace for this asset (lid coefficients cached per asset + lod); None when the face json says
    "human_face": false."""
    cfg = face.get("human_face", {}) if isinstance(face, dict) else {}
    if cfg is False:
        return None
    try:
        from humans.human_face import HumanFace, lid_coefficients
    except ImportError:
        from human_face import HumanFace, lid_coefficients  # type: ignore
    key = (os.path.abspath(asset_dir), lod, tuple(shape_names))
    hf = HumanFace(cfg, shape_names, None, lod)
    if key not in _HF_CACHE:
        _HF_CACHE[key] = lid_coefficients(asset_dir, shape_names, lod) if os.environ.get("JUNO_LIDGOV", "1") != "0" else None
    hf.lid = _HF_CACHE[key]
    return hf


def _face_path(asset_dir, meta):
    """the face table of the character this asset belongs to: asset meta "character", else the asset folder's
    stem (renderer/humans/junor/junor.asset -> junor), when renderer/humans/<c>/<c>.face.json exists; else Juno's."""
    for c in ((meta or {}).get("character"), os.path.basename(os.path.normpath(asset_dir)).split(".")[0]):
        if c and c != "juno":
            p = os.path.join(HERE, c, f"{c}.face.json")
            if os.path.isfile(p):
                return p
    return os.path.join(JUNO_DIR, "juno.face.json")


def _human_life(face):
    """motion: the living-person layer (humans/human_life.py) when the face json has a "life" section."""
    cfg = face.get("life") if isinstance(face, dict) else None
    if not isinstance(cfg, dict) or os.environ.get("JUNO_LIFE") == "0":
        return None
    try:
        from humans.human_life import HumanLife
    except ImportError:
        from human_life import HumanLife  # type: ignore
    return HumanLife(cfg, IDX)


def _load_json(p, default=None):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


# ------------------------------------------------------------------ the rig
class JunoRig:
    def __init__(self, asset_dir, lod="lod0", stature_scene=None, face_json=None, props_json=None, char_cfg=None,
                 engine_body_dict=None, tuning=None):
        self.asset_dir = asset_dir
        self.meta = _load_json(os.path.join(asset_dir, "juno.asset.json"), {})
        sk = _load_json(os.path.join(asset_dir, self.meta.get("skeleton", "skeleton.json")))
        if sk is None:
            raise FileNotFoundError(os.path.join(asset_dir, "skeleton.json"))
        self.sk = sk
        self.hints = sk.get("skeleton_hints", {})
        bones = sk["bones"]
        self.names = [b["name"] for b in bones]
        self.nb = len(bones)
        self.bi = {n: i for i, n in enumerate(self.names)}
        self.parent = [int(b["parent"]) for b in bones]
        self.helper = [bool(b.get("helper", False)) for b in bones]
        rest = np.stack([np.asarray(b["rest"], float).reshape(4, 4) for b in bones])
        self.inv_bind = np.stack([np.asarray(b["inverse_bind"], float).reshape(4, 4) for b in bones]).astype(np.float32)
        Q = rest[:, :3, :3].copy()
        for i in range(self.nb):                      # orthonormalise (an asset may carry scale)
            U, _, Vt = np.linalg.svd(Q[i]); Q[i] = U @ Vt
        self.Qr = Q
        self.Mrest_lin = rest[:, :3, :3]              # the rest linear part as stored (incl. any scale)
        h_m = float(self.meta.get("height_m") or 1.68)
        tun = dict(DEFAULT_TUNING)
        tun.update(_load_json(os.path.join(JUNO_DIR, "juno.rig.json"), {}).get("tuning", {}))
        if tuning:
            tun.update(tuning)
        if os.environ.get("JUNO_RIG_TUNE"):          # rig3: look-dev knob (a json dict over the tuning)
            tun.update(json.loads(os.environ["JUNO_RIG_TUNE"]))
        self.tun = tun
        self.stature = float(stature_scene or tun["stature_scene"])
        self.k = self.stature/h_m                      # metres -> scene units
        self.r = rest[:, :3, 3]*self.k                 # rest joint positions, scene units (char-local: floor at 0)
        self.shape_names = list(self.meta.get("shapes") or [])
        mj = None
        for l in self.meta.get("lods", []):
            if l.get("name") == lod:
                mj = _load_json(os.path.join(asset_dir, l["mesh"]))
        if mj and not self.shape_names:
            self.shape_names = [s["name"] for s in mj.get("shapes", [])]
        self.si = {n: i for i, n in enumerate(self.shape_names)}
        self.correctives = []
        cmap = {}
        for nm, c in (self.meta.get("correctives") or {}).items():
            cmap[nm] = c
        for s in (mj or {}).get("shapes", []):
            if s.get("corrective"):
                cmap[s["name"]] = s["corrective"]
        for nm, c in cmap.items():
            of = c.get("of", []) if isinstance(c, dict) else list(c)
            if nm in self.si and of and all(o in self.si for o in of):
                self.correctives.append((self.si[nm], [self.si[o] for o in of]))
        self.face = _load_json(face_json or os.environ.get("JUNO_FACE_JSON") or _face_path(asset_dir, self.meta), DEFAULT_FACE)   # ( rig3: env = look-dev knob)
        if isinstance(self.face, dict) and isinstance(self.face.get("rig_tuning"), dict):   # per-character tuning
            tun.update(self.face["rig_tuning"])
            self.stature = float(stature_scene or tun["stature_scene"]); self.k = self.stature/h_m
            self.r = rest[:, :3, 3]*self.k
        self.life = _human_life(self.face)                                   # motion: None unless "life"
        self.hf = _human_face(self.face, self.shape_names, asset_dir, lod)   # rig4: the human expression governor
        self._props(props_json)
        cfg = char_cfg or _load_json(os.path.join(JUNO_DIR, "juno.json")) or _load_json(
            os.path.join(RENDER_ROOT, "animals", "muse.json"))
        self.char_cfg = cfg
        self.eb = engine_body_dict or engine_body(cfg, body_dir=os.path.join(RENDER_ROOT, "body"))
        self._derive()

    # ---- props: what each slot's grip is (engine/props_defs.py semantics, H = 2.30)
    def _props(self, props_json):
        self.prop_cols = []
        cfg = None
        if isinstance(props_json, dict):
            cfg = props_json
        else:
            cfg = _load_json(props_json or os.path.join(RENDER_ROOT, "animals", "muse.props.json"))
        if not cfg:
            return
        try:
            from engine import props_defs as PD
        except Exception:                               # noqa: BLE001
            PD = None
        H = float((cfg.get("units") or {}).get("H", 2.30))
        grips = {"power": 0, "pinch": 1, "two": 2, "two_hand": 2}
        for pr in cfg.get("props", []):
            k = int(pr.get("slot", -1))
            if k < 0:
                continue
            base = f"prop{k}_"
            if base + "hl" not in IDX:
                continue
            g = grips.get(pr.get("grip", "power"), 0)
            size = float(pr.get("size", 0.1))*H
            gr = pr.get("grip_r")
            if gr is None:
                gr = 0.12 if PD is None else getattr(PD, "DEFAULT_GRIP_R", 0.12)
            free = bool(pr.get("furniture", False))
            self.prop_cols.append((k, IDX[base + "hl"], IDX[base + "hr"], IDX.get(base + "hh"), g, float(gr)*size, free))

    # ---- rest-derived geometry
    def _derive(self):
        r, bi = self.r, self.bi
        R = lambda n: r[bi[n]]
        self.side_bones = []
        for ps in range(2):                       # param side -> anatomical bones
            a = ANAT[ps]
            self.side_bones.append({n: bi.get(f"{n}_{a}") for n in (
                "clavicle", "upperarm", "lowerarm", "hand", "thigh", "calf", "foot", "ball",
                "upperarm_twist_01", "lowerarm_twist_01", "thigh_twist_01")})
        sb = self.side_bones
        # arms
        self.sh_rest = [R(f"upperarm_{ANAT[i]}") for i in range(2)]
        self.el_rest = [R(f"lowerarm_{ANAT[i]}") for i in range(2)]
        self.wr_rest = [R(f"hand_{ANAT[i]}") for i in range(2)]
        self.l_up = float(np.linalg.norm(self.el_rest[0] - self.sh_rest[0]))
        self.l_fo = float(np.linalg.norm(self.wr_rest[0] - self.el_rest[0]))
        self.L = self.l_up + self.l_fo
        self.ka = self.L/MUSE_ARM_L
        # legs
        self.hip_rest = [R(f"thigh_{ANAT[i]}") for i in range(2)]
        self.kn_rest = [R(f"calf_{ANAT[i]}") for i in range(2)]
        self.an_rest = [R(f"foot_{ANAT[i]}") for i in range(2)]
        self.l_th = float(np.linalg.norm(self.kn_rest[0] - self.hip_rest[0]))
        self.l_sh = float(np.linalg.norm(self.an_rest[0] - self.kn_rest[0]))
        self.pelvis_rest = R("pelvis")
        self.waist = R("spine_01")
        self.neck = R("neck_01")
        self.headb = R("head")
        self.arm_hinge_r = [rest_hinge(_n(self.el_rest[i] - self.sh_rest[i]), _n(self.wr_rest[i] - self.el_rest[i]),
                                       np.array([0.0, 0.0, 1.0])) for i in range(2)]
        self.leg_hinge_r = [rest_hinge(_n(self.kn_rest[i] - self.hip_rest[i]), _n(self.an_rest[i] - self.kn_rest[i]),
                                       np.array([0.0, 0.0, -1.0])) for i in range(2)]
        hh = dict(self.meta.get("landmarks") or {}); hh.update(self.hints)
        k = self.k
        self.head_top = _v(hh["head_top"])*k if "head_top" in hh else np.array([0.0, self.stature, self.headb[2]])
        chin = _v(hh["chin"])*k if "chin" in hh else np.array([0.0, self.headb[1] - 0.01*self.stature, self.headb[2] + 0.04*self.stature])
        eyez = R("eye_l")[2]
        hr = 0.5*(self.head_top[1] - chin[1])
        self.head_c = _v(hh["head_center"])*k if "head_center" in hh else np.array(
            [0.0, 0.5*(self.head_top[1] + chin[1]) + 0.04*hr, eyez - 0.80*hr])
        self.chin = chin
        self.head_radii = np.array([0.78*hr, 1.0*hr, 0.95*hr])*float(self.tun["head_radii_k"])
        self.eye_rest = [R("eye_r"), R("eye_l")]                 # param side 0 = screen-left = anatomical right
        self.eye_mid = 0.5*(self.eye_rest[0] + self.eye_rest[1])
        # rest hand frames (x across the palm toward the thumb side.. made right-handed: y fingertips, z out of palm)
        self.hand_rest_frame = []
        self.finger_bones = []
        for i in range(2):
            a = ANAT[i]; s = -1.0 if i == 0 else 1.0
            w = self.wr_rest[i]
            mid = R(f"middle_01_{a}")
            y = _n(mid - w)
            acr = _n(R(f"index_01_{a}") - R(f"pinky_01_{a}"))
            z = _n(cross(acr, y))
            # palm normal "out of the palm" = the side the fingers curl to: down / toward the midline in the A-pose
            if np.dot(z, np.array([-np.sign(w[0]), -1.0, 0.0])) < 0:
                z = -z
            x = cross(y, z)
            self.hand_rest_frame.append(np.array([x, y, z]).T)
            ch = {}
            for d in DIGITS:
                ch[d] = [bi.get(f"{d}_0{j}_{a}") for j in (1, 2, 3)]
            self.finger_bones.append(ch)
        # the palm contact point in the rest hand frame (palm centre on the palm surface)
        pc = []
        for i in range(2):
            a = ANAT[i]
            F = self.hand_rest_frame[i]
            w = self.wr_rest[i]
            mcp = np.mean([R(f"{d}_01_{a}") for d in ("index", "middle", "ring", "pinky")], 0)
            plen = float(np.dot(mcp - w, F[:, 1]))
            pc.append(np.array([0.0, 0.52*plen, float(self.tun["palm_half_thick"])*self.stature]))
        self.palm_local = pc
        self.tip_len = float(np.linalg.norm(R("middle_03_r") - R("hand_r")))*1.25    # body: wrist -> fingertip
        # torso keep-out ellipsoid (upper-body space): chest round spine_03, sized by the shoulders
        sw = abs(self.sh_rest[1][0] - self.sh_rest[0][0])
        s3 = R("spine_03"); s2 = R("spine_02")
        self.torso_c = np.array([0.0, 0.5*(s2[1] + s3[1]), s3[2] + 0.01*self.stature])
        self.torso_r = np.array([0.40*sw, 0.30*(self.neck[1] - self.waist[1]) + 0.05*self.stature, 0.24*sw])
        # stances / sitting
        self.com = np.array([0.0, 0.55*self.stature, 0.0])
        self.base = np.array([0.0, 0.0, 0.0])

    # ------------------------------------------------------------------ helpers
    def _curls(self, P, i, holdw, holdr, holdg):
        """animals/muse.glsl's per-digit curl c (index, middle, ring, pinky, thumb) + spread, for param side i."""
        sd = SIDES[i]
        op = clamp(P[IDX[f"hand_{sd}_open"]], 0.0, 1.0)
        pt = clamp(P[IDX[f"point_{sd}"]], 0.0, 1.0)
        th = clamp(P[IDX[f"thumb_{sd}"]], 0.0, 1.0)
        cnt = clamp(P[IDX[f"fingers_{sd}"]], 0.0, 5.0)
        relax = mix(0.55, 0.0, op)
        c = [0.0]*5
        for k in range(5):
            v = mix(0.35, 0.0, op) if k == 4 else relax
            if 0 < k < 4:
                v = mix(v, 1.0, pt)
            if k == 0:
                v = mix(v, 0.0, pt)
            if k == 4:
                v = mix(v, 0.0, th)
            else:
                v = mix(v, 1.0, th*(1.0 - pt))
            upk = clamp(cnt - k, 0.0, 1.0)
            v = mix(v, mix(1.0, 0.0, upk), clamp(cnt, 0.0, 1.0))
            c[k] = v
        spread = 0.25*op + 0.20*clamp(cnt - 0.5, 0.0, 1.0)
        hw = clamp(holdw, 0.0, 1.0)
        if hw > 0.0:
            g = holdg
            wrapc = clamp(0.70*0.10*self.ka/((holdr + 0.028*self.ka)*1.45), 0.25, 0.95)
            for k in range(5):
                tc = (0.10 if k == 4 else 0.18) if g == 2 else ((0.55 if k in (0, 4) else 0.70) if g == 1 else
                                                                 (0.55 if k == 4 else wrapc))
                c[k] = mix(c[k], tc, hw)
            spread = mix(spread, 0.08, hw)
        return c, spread

    def _holds(self, P):
        hw = [0.0, 0.0]; hr = [0.1, 0.1]; hg = [0, 0]; flags = 0
        for (k, cl, cr, ch, g, r, free) in self.prop_cols:
            if free:
                continue
            wl = clamp(P[cl], 0.0, 1.0); wr = clamp(P[cr], 0.0, 1.0)
            gg = 2 if (wl > 0.05 and wr > 0.05) else g
            if wl > hw[0]:
                hw[0], hr[0], hg[0] = wl, r, gg
            if wr > hw[1]:
                hw[1], hr[1], hg[1] = wr, r, gg
            if wl > 0.05 or wr > 0.05:
                flags |= 1 << k
            if ch is not None and P[ch] > 0.05:
                flags |= 1 << (16 + k)
        return hw, hr, hg, flags

    # ------------------------------------------------------------------ the solve
    def solve(self, P, t=0.0, cam_pos=None, root=None):
        P = np.asarray(P, dtype=np.float64)
        if P.shape[0] < NPARAMS:
            P = np.concatenate([P, np.zeros(NPARAMS - P.shape[0])])
        LF = self.life
        if LF is not None:                              # motion: speech curves, expression slew (a copy)
            P = LF.pre(P, t)
        T = self.tun
        nb = self.nb; bi = self.bi; r = self.r
        D = [None]*nb                       # world rotation deltas
        pos = np.zeros((nb, 3))
        J = {}
        g = lambda n: P[IDX[n]]
        # ---- root (engine) + human squash compensation
        if root is None:
            A, o, Rr, lip = root_transform(P, self.eb)
        else:
            A, o = np.asarray(root[0], float), np.asarray(root[1], float)
            Rr = root[2] if len(root) > 2 else engine_rot(g("root_yaw"), g("root_pitch"), g("root_roll"))
            lip = root[3] if len(root) > 3 else 1.0
        Ai = np.linalg.inv(A)
        sq = clamp(g("squash"), -1.0, 1.0)
        if abs(sq) > 1e-7:
            sy = math.exp(-SQUASH_K*sq*(1.0 - T["squash_human"])); sx = 1.0/math.sqrt(sy)
            Csq = np.diag([1.0/sx, 1.0/sy, 1.0/sx])     # undo (1 - squash_human) of the engine's squash
        else:
            Csq = None
        if cam_pos is None:
            cam_pos = np.asarray(self.char_cfg["camera"]["pos"], float)
        cam_l = Ai @ (np.asarray(cam_pos, float) - o)                          # the camera in char-local
        # ---- pelvis
        bounce = clamp(g("body_bounce"), -1.0, 1.0)
        knee_sq = T["squash_knee"]*max(sq, 0.0)
        pel = self.pelvis_rest + np.array([0.0, T["bounce_scene"]*bounce - knee_sq*self.stature, 0.0])
        # ---- low reach (a human's hands hang at mid-thigh; Muse's short arms reach the floor): a hand target
        # below the hanging range bends her down by itself — knees (pelvis drops, hips back, feet planted) + a forward
        # bend at the waist — so hand_*_y = -1 reaches the floor and the page's reach map (built from this rig) covers
        # the floor, the chalk tray and the board's lowest row with the page's own bend logic unchanged.
        low = self._low_reach(P)
        self._low = low
        if low["d"] > 0.0:
            pel = pel + np.array([0.0, -low["d"], -low["back"]])
        # ---- upper body (Muse's waist rotation, distributed over the spine)
        W = rotY(0.40*g("body_yaw")) @ rotX(0.28*g("body_pitch") + low["th"]) @ rotZ(-0.20*g("body_lean"))
        br = g("breath")
        spineD = {"spine_01": rot_power(W, 0.30), "spine_02": rot_power(W, 0.65), "spine_03": W @ rotX(-0.012*br)}
        # rig2: the upper-body frame WITHOUT the rig's own low-reach bend (pelvis drop + waist pitch): hand targets
        # are defined in it (they ride the page's body_pitch/yaw/lean exactly as the page's reach model assumes), and the
        # low-reach bend is then only how the body gets the hand there (_clavicle)
        self._s3_0 = self._spine3_frame(self.pelvis_rest + np.array([0.0, T["bounce_scene"]*bounce - knee_sq*self.stature, 0.0]),
                                        rotY(0.40*g("body_yaw")) @ rotX(0.28*g("body_pitch")) @ rotZ(-0.20*g("body_lean")), br) \
            if low["d"] > 0.0 or low["th"] > 0.0 else None
        F = rotY(g("head_yaw")) @ rotX(g("head_pitch")) @ rotZ(-g("head_roll"))
        self._Fc = None; neck_share = T["neck_share"]; self._eye_off = (0.0, 0.0)
        if LF is not None:                              # motion: head-jaw / voice coupling, the spine shares the turn
            hy, hp, hrl = g("head_yaw"), g("head_pitch"), g("head_roll")
            ey, ep, er = LF.head_extra(P, t, self._jaw_open(P))
            hy += ey; hp += ep; hrl += er
            F = rotY(hy) @ rotX(hp) @ rotZ(-hrl)
            if LF.on["body"]:
                B = LF.cfg["body"]
                a_ = LF.chest_angles(hy, hp, hrl, t)
                self._Fc = rot_power(rotY(a_[0]) @ rotX(a_[1]) @ rotZ(-a_[2]), float(B["chest_share"]))
                neck_share = float(B["chest_share"]) + float(B["neck_share"])
                spineD["spine_03"] = spineD["spine_03"] @ self._Fc
            self._eye_off = LF.eye_offset(t, g("gaze_x"), g("gaze_y"), g("gaze_free"))
        hoff = T["head_xyz_scene"]*np.array([g("head_x"), g("head_y"), g("head_z")])
        hw, hr, hg, flags = self._holds(P)
        side_out = [dict(), dict()]
        # per-bone FK in skeleton order
        I3 = np.eye(3); off = self.off; handler = self.handler
        for i in range(nb):
            n = self.names[i]; p = self.parent[i]
            hk = handler[i]
            if hk is not None and hk[0] in ("finger", "thumb"):
                D[i] = I3; continue                      # batched after the loop (_fingers_batch)
            if p < 0 or n == "root":
                D[i] = I3; pos[i] = r[i]; continue
            if n == "pelvis":
                D[i] = I3; pos[i] = pel; continue
            Dp = D[p]
            pos[i] = pos[p] + Dp @ off[i]
            if n in spineD:
                D[i] = spineD[n]; continue
            if n == "neck_01":
                D[i] = W @ rot_power(F, neck_share); continue
            if n == "head":
                pos[i] = pos[i] + W @ hoff + np.array([0.0, 0.006*br*self.ka, 0.0])
                D[i] = W @ F
                continue
            if n == "jaw":
                Rj, tj = self._jaw_xform(P)
                D[i] = D[p] @ Rj
                pos[i] = pos[i] + D[p] @ tj; continue
            if n in ("eye_l", "eye_r"):
                D[i] = self._eye(P, n, D[p], pos[i], cam_l, t)
                if LF is not None:                      # motion: the globe retracts under a closing lid
                    rm = LF.eye_retract_m(clamp(float(P[IDX["blink_r" if n == "eye_l" else "blink_l"]]), 0.0, 1.0))
                    if rm > 0.0:
                        pos[i] = pos[i] - D[p] @ np.array([0.0, 0.0, rm*self.k])
                continue
            if hk is None:
                D[i] = Dp; continue
            hit = False
            for ps in (hk[1],):
                sb = self.side_bones[ps]
                if i == sb["clavicle"]:
                    D[i] = self._clavicle(P, ps, Dp, pos[i], side_out[ps], F if self._Fc is None else self._Fc.T @ F, hw[ps]); hit = True; break
                if i == sb["upperarm"]:
                    D[i] = self._upperarm(ps, Dp, pos[i], side_out[ps]); hit = True; break
                if i == sb["lowerarm"]:
                    D[i] = self._lowerarm(ps, pos[i], side_out[ps]); hit = True; break
                if i == sb["lowerarm_twist_01"]:
                    D[i] = self._twist(ps, D[p], side_out[ps], 0.5); hit = True; break
                if i == sb["upperarm_twist_01"]:
                    D[i] = D[p]; hit = True; break
                if i == sb["hand"]:
                    D[i] = self._hand(P, ps, pos[i], side_out[ps], hw[ps], hr[ps], hg[ps]); hit = True; break
                if i == sb["thigh"]:
                    D[i] = self._thigh(P, ps, pos[i], side_out[ps], Ai); hit = True; break
                if i == sb["calf"]:
                    D[i] = self._calf(ps, pos[i], side_out[ps]); hit = True; break
                if i == sb["thigh_twist_01"]:
                    D[i] = D[p]; hit = True; break
                if i == sb["foot"]:
                    D[i] = self._foot(ps, side_out[ps]); hit = True; break
                if i == sb["ball"]:
                    D[i] = self._ball(ps, D[p], side_out[ps]); hit = True; break
            if hit:
                continue
            fd = self._finger(i, D[p], side_out)
            D[i] = fd if fd is not None else Dp
        self._fingers_batch(D, pos, side_out)
        if T.get("forearm_roll_share") is not None:    # body (gated): no twist bones in the asset, so the hand's
            self._forearm_roll(D, float(T["forearm_roll_share"]))     # roll about the forearm all sat in the wrist
        # ---- assemble matrices (char-local, then the human squash compensation, then the engine root)
        Ms = np.zeros((nb, 4, 4)); Ms[:, 3, 3] = 1.0
        Dst = np.stack(D)
        Ms[:, :3, :3] = np.einsum("bij,bjk->bik", Dst, self.Mrest_lin)*self.k
        Ms[:, :3, 3] = pos
        if LF is not None and LF.on["body"] and abs(br) > 1e-9:     # motion: the chest / belly volume breathes
            for bn, key in (("spine_03", "breath_chest"), ("spine_01", "breath_belly")):
                j = bi.get(bn)
                if j is not None:
                    sc = 1.0 + br*np.asarray(LF.cfg["body"][key], float)
                    Ms[j, :3, :3] = (Dst[j]*sc[None, :]) @ self.Mrest_lin[j]*self.k
        if Csq is not None:
            C4 = np.eye(4); C4[:3, :3] = Csq; C4[:3, 3] = self.base - Csq @ self.base
            Ms = np.einsum("ij,bjk->bik", C4, Ms)
        R4 = np.eye(4); R4[:3, :3] = A; R4[:3, 3] = o
        Mw = np.einsum("ij,bjk->bik", R4, Ms)
        skin = np.einsum("bij,bjk->bik", Mw.astype(np.float32), self.inv_bind)
        # ---- frames (char-local, after the squash compensation)
        def loc(p):
            return p if Csq is None else self.base + Csq @ (p - self.base)

        def locR(Rm):
            if Csq is None:
                return Rm
            M = Csq @ Rm
            U, _, Vt = np.linalg.svd(M); return U @ Vt
        gl = np.zeros((2, 4, 4), np.float32)
        for ps in range(2):
            so = side_out[ps]
            Fh = locR(so["hand_frame"])
            pc = loc(so["palm"])
            G = np.eye(4); G[:3, :3] = Fh; G[:3, 3] = pc + Fh[:, 2]*GRASP_OFS
            gl[ps] = G
            J[f"wrist_{SIDES[ps]}"] = loc(so["wrist"]); J[f"elbow_{SIDES[ps]}"] = loc(so["elbow"])
            J[f"shoulder_{SIDES[ps]}"] = loc(so["shoulder"]); J[f"palm_{SIDES[ps]}"] = pc
            J[f"hip_{SIDES[ps]}"] = loc(so["hip"]); J[f"knee_{SIDES[ps]}"] = loc(so["knee"]); J[f"ankle_{SIDES[ps]}"] = loc(so["ankle"])
            J[f"palm_normal_{SIDES[ps]}"] = Fh[:, 2]
        hi = bi["head"]
        HR = locR(D[hi])                                  # head axes in local (rest head faces +z)
        hc = loc(pos[hi] + D[hi] @ (self.head_c - r[hi]))
        hf = np.eye(4); hf[:3, :3] = HR; hf[:3, 3] = hc + HR @ (np.asarray(T["head_frame_ofs"], float)*self.stature)
        J["head_center"] = hc
        # ---- face weights
        w = self._face(P, t, D, bi)
        # ---- proxies
        prox = self._proxies(pos, D, side_out, hc, HR, loc)
        prw = prox.copy()
        prw[:, 0:3] = (A @ prox[:, 0:3].T).T + o
        prw[:, 3:6] = (A @ prox[:, 3:6].T).T + o
        # radius: A is volume-preserving and the local pose already carries the human squash: keep r
        allp = np.concatenate([prw[:, 0:3], prw[:, 3:6]])
        bc = 0.5*(allp.min(0) + allp.max(0))
        brad = float(np.max(np.linalg.norm(allp - bc, axis=1) + np.concatenate([prw[:, 6], prw[:, 6]])))
        R3 = np.eye(4); R3[:3, :3] = A; R3[:3, 3] = o
        grasp_w = np.einsum("ij,bjk->bik", R3, gl.astype(np.float64))
        for ps in range(2):                           # world frames: orthonormal columns (rotation part of the root)
            grasp_w[ps, :3, :3] = Rr @ gl[ps, :3, :3]
        head_w = R3 @ hf; head_w[:3, :3] = Rr @ hf[:3, :3]
        el = loc(pos[bi["eye_l"]]); er = loc(pos[bi["eye_r"]])
        eyes = np.stack([A @ el + o, A @ er + o])
        gz = A @ self._gaze_pt + o
        mat = self._mat(P, t)
        marks = {k: float(clamp(P[IDX[k]], 0.0, 1.0)) for k in MARK_KEYS}
        return RigFrame(bones=Mw.astype(np.float32), skin=skin.astype(np.float32), bones_local=Ms.astype(np.float32),
                        weights=w, eyes=eyes.astype(np.float32), gaze=gz.astype(np.float32),
                        grasp_local=gl, grasp=grasp_w.astype(np.float32), head_local=hf.astype(np.float32),
                        head=head_w.astype(np.float32), head_center_local=hc.astype(np.float32),
                        head_rot_local=HR.astype(np.float32), proxies_local=prox.astype(np.float32),
                        proxies=prw.astype(np.float32), bound=np.array([*bc, brad], np.float32),
                        hold_w=np.array(hw, np.float32), hold_r=np.array(hr, np.float32), hold_g=np.array(hg, np.int32),
                        flags=int(flags), root_A=A.astype(np.float32), root_o=o.astype(np.float32),
                        root_R=np.asarray(Rr, np.float32), root_lip=float(lip), mat=mat, marks=marks, joints=J)

    # ---- arms
    def _low_reach(self, P):
        """rig2: the low reach (below) + a FORWARD lean for targets in front beyond the arm (a desk's far side, an
        object held out): the least waist bend that brings the shoulder within 0.95 L of the standard target, capped at
        far_pitch_max, faded out for targets behind the shoulder or above it.  Continuous in the params; the reach map
        (built from this rig) carries it, so the page's object reach needs no change."""
        lo = self._low_reach0(P)
        T = self.tun; L = self.L; hm = self.hand_map; wst = self.waist
        thmax = math.radians(T["far_pitch_max"])
        if thmax <= 0.0:
            return lo
        thf = 0.0; far = [0.0, 0.0]
        for ps in range(2):
            sd = SIDES[ps]; s = -1.0 if ps == 0 else 1.0
            hx, hy, hz = P[IDX[f"hand_{sd}_x"]], P[IDX[f"hand_{sd}_y"]], P[IDX[f"hand_{sd}_z"]]
            if hz <= 0.0:
                continue
            rest = self.wr_hang[ps]
            gx = hm["gain_x_out"] if hx > 0 else hm["gain_x_in"]
            Tt = rest + np.array([s*gx*L*hx, hm["gain_y"]*L*max(hy, T["low_from"]), hm["gain_z"]*L*hz])
            sh0 = self.sh_rest[ps]
            fade = ss(T["far_from"], 1.2, hz)*ss(sh0[2] + 0.2*L, sh0[2] + 0.6*L, Tt[2])*(1.0 - ss(sh0[1] - 0.1*L, sh0[1] + 0.4*L, Tt[1]))
            if fade <= 0.0:
                continue
            v = sh0 - wst

            def ex(th):
                c, sn = math.cos(th), math.sin(th)
                sh = wst + np.array([v[0], c*v[1] - sn*v[2], sn*v[1] + c*v[2]])
                return float(np.linalg.norm(Tt - sh)) - T["far_reach"]*L
            if ex(0.0) <= 0.0:
                continue
            if ex(thmax) > 0.0:
                th = thmax
            else:
                a_, b_ = 0.0, thmax
                for _ in range(10):
                    m = 0.5*(a_ + b_)
                    a_, b_ = (m, b_) if ex(m) > 0.0 else (a_, m)
                th = b_
            thf = max(thf, th*fade)
            far[ps] = th*fade
        lo = dict(lo)
        if thf > lo["th"]:
            lo["th"] = thf
        if lo["th"] > 0.0:      # which hand the bend is FOR: its target stays put while the body bends; the other rides along
            lo["own"] = [max(lo["own"][ps], clamp(far[ps]/lo["th"], 0.0, 1.0)) for ps in range(2)]
        return lo

    def _low_reach0(self, P):
        """Targets below the hanging hand: a local-frame target (y down to the floor at hand_y = -1) and the knee bend
        + waist bend that brings the shoulders within reach of the lowest one.  Deterministic in the params."""
        T = self.tun; S = self.stature; L = self.L; hm = self.hand_map
        b = [0.0, 0.0]; TL = [None, None]; need = 0.0; ramp = 0.0; own = [0.0, 0.0]
        y0, y1 = T["low_from"], -1.0
        for ps in range(2):
            sd = SIDES[ps]; s = -1.0 if ps == 0 else 1.0
            hx, hy, hz = P[IDX[f"hand_{sd}_x"]], P[IDX[f"hand_{sd}_y"]], P[IDX[f"hand_{sd}_z"]]
            if hy >= y0:
                continue
            w = ss(y0, y1, hy)
            b[ps] = w
            rest = self.wr_hang[ps]
            gx = hm["gain_x_out"] if hx > 0 else hm["gain_x_in"]
            ystd = rest[1] + hm["gain_y"]*L*y0
            ylow = ystd + (T["low_floor"]*S - ystd)*(y0 - hy)/(y0 - y1)       # linear to the floor at -1 (and beyond)
            tl = np.array([rest[0] + s*gx*L*hx, ylow, rest[2] + hm["gain_z"]*L*hz + T["low_fwd"]*S*w])
            tl[0] = s*max(s*tl[0], hm["min_x_scene"] + 0.05*S*w)
            TL[ps] = tl
            sh = self.sh_rest[ps]
            rp = ss(y0, y0 - T["low_ramp"], hy)
            ramp = max(ramp, rp); own[ps] = rp
            need = max(need, rp*max(0.0, sh[1] - 0.93*L - tl[1]))   # continuous at low_from
        if need <= 0.0:
            return {"d": 0.0, "th": 0.0, "back": 0.0, "b": b, "T": TL, "own": [0.0, 0.0]}
        dmax = T["low_squat"]*S
        d = min(need*T["low_squat_share"], dmax)
        # the least waist bend that brings the lowest target within 0.95 L of its shoulder (scalar search, no IK)
        wst = self.waist
        lo_ps = [ps for ps in range(2) if TL[ps] is not None]
        th_max = math.radians(T["low_pitch_max"])

        def short(th):
            # rig2: UNWEIGHTED distance excess (the b-weighted one made every bend look equally good at the onset
            # of the low regime, so the waist snapped to its maximum at hand_y just below low_from: a folded reach map
            # the page's inverse could not invert); the onset is continuous through `ramp` below
            back = T["low_back"]*float(self.sh_rest[0][1] - wst[1])*math.sin(th) + 0.25*d
            c, sn = math.cos(th), math.sin(th)
            worst = -1e9
            for ps in lo_ps:
                v = self.sh_rest[ps] - wst
                sh = wst + np.array([0.0, -d, -back]) + np.array([v[0], c*v[1] - sn*v[2], sn*v[1] + c*v[2]])
                worst = max(worst, float(np.linalg.norm(TL[ps] - sh)) - 0.95*L)
            return worst
        th = 0.0                                     # the first bend (coarse scan, then a refine) that reaches; else the closest
        if short(0.0) > 0.0:
            best, bth = short(0.0), 0.0
            for k in range(1, 17):
                t_ = th_max*k/16.0; v = short(t_)
                if v <= 0.0:
                    lo_, hi_ = th_max*(k - 1)/16.0, t_
                    for _ in range(6):
                        mid = 0.5*(lo_ + hi_)
                        lo_, hi_ = (mid, hi_) if short(mid) > 0.0 else (lo_, mid)
                    bth = hi_; break
                if v < best - 0.01*L:                # a bend must buy real reach to be taken
                    best, bth = v, t_
            th = bth
        th *= ramp
        back = T["low_back"]*float(self.sh_rest[0][1] - wst[1])*math.sin(th) + 0.25*d
        return {"d": d, "th": th, "back": back, "b": b, "T": TL, "own": own}

    def _spine3_frame(self, pel, W, br):
        """spine_03's (position, rotation delta) for a pelvis position and a waist rotation (the FK of the spine chain)"""
        bi = self.bi; off = self.off
        D1 = rot_power(W, 0.30); D2 = rot_power(W, 0.65); D3 = W @ rotX(-0.012*br)
        p1 = pel + off[bi["spine_01"]]
        p2 = p1 + D1 @ off[bi["spine_02"]]
        p3 = p2 + D2 @ off[bi["spine_03"]]
        return p3, D3

    def _upper_to_local(self, spine3_D, spine3_pos):
        s3 = self.r[self.bi["spine_03"]]
        return lambda u: spine3_pos + spine3_D @ (u - s3)

    def _clavicle(self, P, ps, Dp, pos_c, so, F, holdw):
        """Clavicle: shrug + a share of far reaches.  Also computes the wrist target (upper-body space -> local)."""
        T = self.tun
        sd = SIDES[ps]; s = -1.0 if ps == 0 else 1.0
        i3 = self.bi["spine_03"]
        # upper-body space = spine_03's rest space; map with spine_03's posed delta (Dp of the clavicle = spine_03 D)
        s3r = self.r[i3]
        s3D = Dp
        # spine_03 posed position = the clavicle's parent position: pos_c - Dp @ (r_clav - r_s3)
        ic = self.side_bones[ps]["clavicle"]
        s3p = pos_c - Dp @ (self.r[ic] - s3r)
        toL = lambda u: s3p + s3D @ (u - s3r)
        so["toL"] = toL; so["s3D"] = s3D
        L = self.L
        hm = self.hand_map
        hx, hy, hz = P[IDX[f"hand_{sd}_x"]], P[IDX[f"hand_{sd}_y"]], P[IDX[f"hand_{sd}_z"]]
        rest = self.wr_hang[ps]
        gx = hm["gain_x_out"] if hx > 0 else hm["gain_x_in"]
        Tt = rest + np.array([s*gx*L*hx, hm["gain_y"]*L*hy, hm["gain_z"]*L*hz])
        Tt[0] = s*max(s*Tt[0], hm["min_x_scene"])
        lw = self._low["b"][ps]
        if self._s3_0 is not None:                     # rig2: low reach. Targets live in the frame without the
            p0, D0 = self._s3_0                        # rig's own bend (rest-local + the page's bend/yaw/lean): the
            Tloc = p0 + D0 @ (Tt - s3r)                # standard target and the low target blend THERE, then go into
            if lw > 0.0:                               # the bent upper body -> a reach map the page can invert
                Tloc = Tloc + (p0 + D0 @ (self._low["T"][ps] - s3r) - Tloc)*lw
            own = self._low["own"][ps]                 # (the hand the bend is not for rides the bent body)
            Tt = Tt + (s3D.T @ (Tloc - s3p) + s3r - Tt)*own
        e = (Tt - self.torso_c)/self.torso_r; el = float(np.linalg.norm(e))
        if el < 1.0:
            Tt = self.torso_c + e/max(el, 1e-6)*self.torso_r
        # an EMPTY hand that comes to the face rests on it (head space, inflated by the hand's thickness)
        fr = 1.0 - clamp(holdw, 0.0, 1.0)
        hcu = self.neck + F @ (self.head_c - self.neck)
        up = self.palm_above_wrist
        qh = F.T @ (Tt + np.array([0.0, up, 0.0]) - hcu)
        Rh = self.head_radii + T["face_pad"]*self.stature
        eh = qh/Rh; elh = float(np.linalg.norm(eh))
        if elh < 1.0 and fr > 0.0:
            radial = eh/max(elh, 1e-6)*Rh
            front = np.array([qh[0], qh[1], Rh[2]*math.sqrt(max(1.0 - eh[0]**2 - eh[1]**2, 0.0))])
            qn = radial + (front - radial)*ss(0.0, 0.35, qh[2]/Rh[2])
            Tn = hcu + F @ qn - np.array([0.0, up, 0.0])
            Tt = Tt + (Tn - Tt)*fr
        # a hand at the face (empty, inside the inflated head or just outside it) stands its fingers up, so the palm
        # sits palm_above_wrist over the wrist — the page's face-target formula (motion.js makeFace) assumes exactly that
        so["face_w"] = fr*(1.0 - ss(1.0, 1.5, elh))
        ft = T.get("face_tip")
        if ft is not None and so["face_w"] > 0.0:   # body (gated): at the lips the FINGERTIPS meet the face (blow a
            # kiss, cover the mouth), not the palm (an adult hand's fingers then covered her nose and eyes); ramps in under
            # the nose, out again toward the chin (a chin rest keeps the palm under it). ft = [nose, mouth, chin] -y/radius
            yq = -float(qh[1]/Rh[1])
            wt = ss(ft[0], ft[1], yq)*(1.0 - ss(ft[1], ft[2], yq))*so["face_w"]
            if wt > 0.0:
                Tt = Tt - F @ np.array([0.0, (self.tip_len - up)*wt, 0.0])
        so["face_hc"] = hcu
        if self._Fc is not None:                       # motion: the chest's share of a head turn must not move the
            own = self._low["own"][ps] if self._s3_0 is not None else 0.0     # hands (a hand at the face rides along)
            so["fc_w"] = (1.0 - so["face_w"])*(1.0 - own)
            Tt = Tt + (s3r + self._Fc.T @ (Tt - s3r) - Tt)*so["fc_w"]
        so["target_u"] = Tt
        # shrug + reach share
        sh = clamp(P[IDX[f"shoulder_{sd}"]], 0.0, 1.0)
        S0 = self.sh_rest[ps]; C0 = self.r[ic]
        cl_dir = _n(S0 - C0)
        shd = math.radians(T["shrug_deg"])*sh + math.radians(T["breath_shrug_deg"])*float(P[IDX["breath"]])   # rig2: inhale lifts the shoulders
        if self.life is not None and self.life.on["body"]:      # motion: scapulohumeral rhythm (arm overhead)
            B = self.life.cfg["body"]
            dA = _n(Tt - S0); el = math.degrees(math.acos(clamp(-float(dA[1]), -1.0, 1.0)))
            shd += math.radians(float(B["clav_raise_deg"]))*ss(float(B["clav_from_deg"]), float(B["clav_full_deg"]), el)
        Dc = axis_angle(np.array([0.0, 0.0, 1.0]), -s*shd) if abs(shd) > 1e-9 else np.eye(3)
        dS = Tt - S0; dl = float(np.linalg.norm(dS))
        reach = ss(0.80*L, 1.05*L, dl)
        if reach > 0.0:
            ax = cross(cl_dir, dS/max(dl, 1e-6))
            if np.linalg.norm(ax) > 1e-6:
                Dc = axis_angle(ax, math.radians(T["clav_reach_deg"])*reach) @ Dc
        # D for the clavicle is world: spine delta then its own (rest-space axes)
        return s3D @ Dc

    def _upperarm(self, ps, Dp, pos_s, so):
        s = -1.0 if ps == 0 else 1.0
        toL = so["toL"]; s3D = so["s3D"]
        # IK in upper-body space: the shoulder in upper-body space
        s3r = self.r[self.bi["spine_03"]]
        S_u = s3D.T @ (pos_s - toL(s3r)) + s3r
        Tt = so["target_u"].copy()
        T_ = self.tun
        up = ss(0.55*self.ka + self._dy, 1.35*self.ka + self._dy, Tt[1])
        pole = _n(np.array([s*mix(T_["elbow_out_lo"], 0.9, up), mix(-0.35, 0.25, up), mix(-0.9, -0.35, up)]))
        if T_.get("elbow_pole_hi") is not None:        # body (gated, junor.face.json rig_tuning): the adult arm's
            ph = T_["elbow_pole_hi"]; pf = T_.get("elbow_pole_face")     # high pole + a hand at the face hangs its elbow
            pole = _n(np.array([s*mix(T_["elbow_out_lo"], ph[0], up), mix(-0.35, ph[1], up), mix(-0.9, ph[2], up)]))
            fw = so.get("face_w", 0.0)
            if pf is not None and fw > 0.0:            # below the hand (forearm up, not the elbow thrown over the shoulder)
                pole = _n(pole + (_n(np.array([s*pf[0], pf[1], pf[2]])) - pole)*fw)
        E, Tt = ik2(S_u, Tt, self.l_up, self.l_fo, pole)
        so["E_u"], so["T_u"], so["S_u"] = E, Tt, S_u
        so["shoulder"] = toL(S_u); so["elbow"] = toL(E); so["wrist"] = toL(Tt)
        so["pole_u"] = pole
        # the upper arm's world delta: rest frame (bone dir, hinge) -> posed frame
        iu = self.side_bones[ps]["upperarm"]
        u_r = _n(self.el_rest[ps] - self.sh_rest[ps])
        n_r = self.arm_hinge_r[ps]
        u = _n(E - S_u); v = _n(Tt - E)
        n = cross(u, v)
        if np.linalg.norm(n) < 1e-4:
            n = cross(u, -pole)
        n = _n(n)
        so["hinge_u"] = n
        Fr = frame_from(u_r, n_r); Fp = frame_from(u, n)
        Du = Fp @ Fr.T
        so["Du_u"] = Du
        return s3D @ Du

    def _lowerarm(self, ps, pos_e, so):
        s3D = so["s3D"]
        u_r = _n(self.wr_rest[ps] - self.el_rest[ps])
        n_r = self.arm_hinge_r[ps]
        v = _n(so["T_u"] - so["E_u"])
        Fr = frame_from(u_r, n_r); Fp = frame_from(v, so["hinge_u"])
        Dl = Fp @ Fr.T
        so["Dl_u"] = Dl
        # forearm roll share (decided in _hand; here the IK-only delta, hand adds the twist split)
        so["Dl_noroll"] = s3D @ Dl
        return so["Dl_noroll"]

    def _twist(self, ps, Dparent, so, f):
        return Dparent

    def _forearm_roll(self, D, f):
        """body: a share f of the hand's roll about the forearm (swing-twist split, rest-space axis) moves into the
        lowerarm bone (pronation/supination turns the radius along the forearm; the wrist keeps the rest).  The roll axis
        is the bone's own, so the wrist position (on the axis) is unchanged."""
        for ps in range(2):
            il, ih = self.side_bones[ps]["lowerarm"], self.side_bones[ps]["hand"]
            if il is None or ih is None:
                continue
            Dr = D[il].T @ D[ih]
            ax = _n(self.wr_rest[ps] - self.el_rest[ps])
            w = math.sqrt(max(0.0, 1.0 + float(np.trace(Dr))))*0.5
            v = np.array([Dr[2, 1] - Dr[1, 2], Dr[0, 2] - Dr[2, 0], Dr[1, 0] - Dr[0, 1]])/(4.0*max(w, 1e-6))
            th = 2.0*math.atan2(float(np.dot(v, ax)), w)
            if th > math.pi:
                th -= 2*math.pi
            if abs(th) > 1e-6:
                D[il] = D[il] @ axis_angle(ax, f*th)

    def _hand(self, P, ps, pos_w, so, holdw, holdr, holdg):
        """Muse's hand frame: y along the forearm (fingers up when raised high), z = palm normal toward the midline at
        turn 0, rolled by hand_turn about the forearm (1.45 rad per unit); x = y cross z.  Upper-body space -> local."""
        sd = SIDES[ps]; s = -1.0 if ps == 0 else 1.0
        s3D = so["s3D"]
        E, Tt = so["E_u"], so["T_u"]
        f = _n(Tt - E)
        hh = ss(0.85*self.ka + self._dy, 1.35*self.ka + self._dy, Tt[1])*ss(-0.2, 0.4, f[1] + 0.3)
        f = _n(f + (np.array([0.0, 1.0, 0.0]) - f)*max(0.75*hh, so.get("face_w", 0.0)))
        n0 = np.array([-s, 0.0, 0.0]); n0 = n0 - f*np.dot(n0, f)
        n0 = _n(n0) if np.dot(n0, n0) > 1e-4 else _n(cross(f, [0, 0, 1.0]))
        turn = clamp(P[IDX[f"hand_{sd}_turn"]], -1.0, 1.0)
        nz = axis_angle(f, s*turn*1.45) @ n0
        fp = self.tun.get("face_palm")
        if fp is not None and so.get("face_w", 0.0) > 0.0:   # body (gated): a hand at the face turns its palm TO the
            q = so["face_hc"] - Tt; q = q - f*np.dot(q, f)  # face (cover eyes / chin / cheek / mouth), not out at the camera
            if np.dot(q, q) > 1e-8:
                nz = _n(nz + (_n(q) - nz)*float(fp)*so["face_w"])
                nz = _n(nz - f*np.dot(nz, f))
        nx = _n(cross(f, nz))
        Fh_u = np.array([nx, f, nz]).T
        Fh = s3D @ Fh_u
        if self._Fc is not None and so.get("fc_w", 0.0) > 0.0:    # motion: nor turn the hand with the chest
            Fh = s3D @ rot_power(self._Fc.T, so["fc_w"]) @ Fh_u
        Dh = Fh @ self.hand_rest_frame[ps].T
        so["hand_frame"] = Fh
        so["palm"] = so["wrist"] + Fh @ self.palm_local[ps]
        curls, spread = self._curls(P, ps, holdw, holdr, holdg)
        so["curls"], so["spread"], so["holdg"], so["holdw"] = curls, spread, holdg, holdw
        so["Dh"] = Dh
        return Dh

    def _build_handlers(self):
        self.handler = [None]*self.nb
        for ps in range(2):
            for nme, bidx in self.side_bones[ps].items():
                if bidx is not None:
                    self.handler[bidx] = ("side", ps)
            Fr = self.hand_rest_frame[ps]
            xr, yr, zr = Fr[:, 0], Fr[:, 1], Fr[:, 2]
            s = -1.0 if ps == 0 else 1.0
            fb = self.finger_bones[ps]
            for k, d in enumerate(DIGITS):
                ch = fb[d]
                for j, bidx in enumerate(ch):
                    if bidx is None:
                        continue
                    if d != "thumb":
                        sgn = 1.0 if np.dot(xr, [1, 0, 0])*s >= 0 else -1.0
                        self.handler[bidx] = ("finger", ps, k, j, xr, zr*sgn, None, 0.0)
                    else:
                        tdir = _n(self.r[ch[1]] - self.r[ch[0]]) if ch[1] is not None else yr
                        fax = _n(cross(tdir, zr))
                        side = 1.0 if np.dot(cross(yr, tdir), zr) >= 0 else -1.0
                        self.handler[bidx] = ("thumb", ps, k, j, yr, fax, None, side)

    def _finger_levels(self):
        """Finger bones grouped by joint level (0, 1, 2), all digits of both hands per level: batched rotations."""
        lv = [[], [], []]
        for i, hk in enumerate(self.handler):
            if hk is not None and hk[0] in ("finger", "thumb"):
                lv[hk[3]].append(i)
        self.flev = []
        for j in range(3):
            idx = np.array(lv[j], int)
            if not len(idx):
                continue
            par = np.array([self.parent[i] for i in idx], int)
            hk = [self.handler[i] for i in idx]
            ax1 = np.array([h[4] for h in hk], float); ax2 = np.array([h[5] for h in hk], float)
            th = np.array([h[0] == "thumb" for h in hk]); ps = np.array([h[1] for h in hk], int)
            dg = np.array([h[2] for h in hk], int); sd = np.array([h[7] for h in hk], float)
            self.flev.append((j, idx, par, ax1, ax2, th, ps, dg, sd, self.r[idx] - self.r[par]))

    def _fingers_batch(self, D, pos, side_out):
        cur = np.array([side_out[0]["curls"], side_out[1]["curls"]], float)          # (2, 5)
        spr = np.array([side_out[0]["spread"], side_out[1]["spread"]], float)
        for (j, idx, par, ax1, ax2, th, ps, dg, sd, off) in self.flev:
            c = cur[ps, dg]
            Dp = np.stack([D[q] for q in par])
            pos[idx] = pos[par] + np.einsum("bij,bj->bi", Dp, off)
            if j == 0:
                a1 = np.where(th, -sd*self._opp*c, FLEX[0]*c)
                a2 = np.where(th, -0.349*c, spr[ps]*self._spread*(1.5 - dg)/1.5)
                L = np.einsum("bij,bjk->bik", rodrigues(ax1, a1), rodrigues(ax2, a2))
            else:
                a2 = np.where(th, -TFLEX[j]*c, FLEX[j]*c)
                L = rodrigues(np.where(th[:, None], ax2, ax1), a2)
            Dn = np.einsum("bij,bjk->bik", Dp, L)
            for m, i in enumerate(idx):
                D[i] = Dn[m]

    def human_params(self, P):
        """rig3: the HUMAN reading of the shared control vector (MeshEngine applies it to the whole vector once per
        frame, before the rig and the shared marks pass).  The page's affect anchors / life were built cartoon-exaggerated
        for the plush (eye size, popping eyes, pouts, floating marks); a Pixar human acts with brows, lids, cheeks and mouth
        shape.  juno.face.json "human": {"gain": {param: [g_pos, g_neg]}, "marks_keep": [...]}: each listed column is scaled
        on its sign; every mark column not in marks_keep is zeroed (no floating cartoon marks on a human).  Body / root /
        hand / camera / lip columns are untouched.  Deterministic, idempotent only through this one call site."""
        hm = self._human
        if hm is None or os.environ.get("JUNO_HUMAN") == "0":       # (A/B knob for look checks)
            return P
        Q = np.array(P, dtype=np.float32, copy=True)
        if Q.shape[0] < NPARAMS:
            Q = np.concatenate([Q, np.zeros(NPARAMS - Q.shape[0], np.float32)])
        ix, gp, gn = hm
        v = Q[ix]
        Q[ix] = np.where(v >= 0.0, v*gp, v*gn)
        if self.hf is not None:                     # rig4: chin-up / roll soft caps (with the body that carries the head)
            Q = self.hf.head(Q, IDX)
        return Q

    def _human_init(self):
        self._brow_ix = [self.si[n] for n in ("browInnerUp", "browOuterUpLeft", "browOuterUpRight", "browDownLeft",
                                              "browDownRight", "browTiltLeft", "browTiltRight") if n in self.si]
        self._brow_dn_ix = set(self.si[n] for n in ("browDownLeft", "browDownRight") if n in self.si)
        h = self.face.get("human") or DEFAULT_FACE.get("human")
        if not h:
            self._human = None
            return
        ix, gp, gn = [], [], []
        for n, g in (h.get("gain") or {}).items():
            if n in IDX:
                g = g if isinstance(g, (list, tuple)) else [g, g]
                ix.append(IDX[n]); gp.append(float(g[0])); gn.append(float(g[1]))
        keep = set(h.get("marks_keep") or [])
        for n in MARK_KEYS:
            if n in IDX and n not in keep and IDX[n] not in ix:
                ix.append(IDX[n]); gp.append(0.0); gn.append(0.0)
        self._human = (np.array(ix, int), np.array(gp, np.float32), np.array(gn, np.float32))

    def _face_matrix(self):
        ps, ss_, ff, gg, mx = [], [], [], [], []
        for rule in self.face.get("rules", []):
            if rule["s"] not in self.si or rule["p"] not in IDX:
                continue
            ps.append(IDX[rule["p"]]); ss_.append(self.si[rule["s"]])
            ff.append({"pos": 0, "neg": 1, "abs": 2, "lin": 3}[rule.get("f", "pos")])
            gg.append(float(rule.get("g", 1.0))); mx.append(float(rule["max"]) if rule.get("max") is not None else 1e9)
        self._human_init()
        self._fr_n = len(ps)
        self._fr_p = np.array(ps, int); self._fr_s = np.array(ss_, int); self._fr_f = np.array(ff, int)
        self._fr_g = np.array(gg, float); self._fr_max = np.array(mx, float)

    def _finger(self, i, Dp, side_out):
        for ps in range(2):
            fb = self.finger_bones[ps]
            for k, d in enumerate(DIGITS):
                ch = fb[d]
                if i in ch:
                    so = side_out[ps]
                    if "curls" not in so:
                        return None
                    j = ch.index(i)
                    c = so["curls"][k]
                    Fr = self.hand_rest_frame[ps]
                    xr, yr, zr = Fr[:, 0], Fr[:, 1], Fr[:, 2]
                    T = self.tun
                    if d != "thumb":
                        ang = math.radians((70.0, 95.0, 65.0)[j])*c
                        L = axis_angle(xr, ang)
                        if j == 0:
                            spr = so["spread"]*math.radians(T["spread_deg"])*(1.5 - k)/1.5
                            s = -1.0 if ps == 0 else 1.0
                            L = L @ axis_angle(zr, spr*(1.0 if np.dot(xr, [1, 0, 0]) * s >= 0 else -1.0))
                        return Dp @ L
                    # thumb: opposition (about the hand's y) on the first joint, flexion of all three toward the palm
                    a = self.finger_bones[ps]["thumb"]
                    tdir = _n(self.r[a[1]] - self.r[a[0]]) if a[1] is not None else yr
                    fax = _n(cross(tdir, zr))
                    if j == 0:
                        opp = math.radians(T["thumb_opp_deg"])*c
                        side = 1.0 if np.dot(cross(yr, tdir), zr) >= 0 else -1.0
                        return Dp @ axis_angle(yr, -side*opp) @ axis_angle(fax, -math.radians(20.0)*c)
                    return Dp @ axis_angle(fax, -math.radians((0, 35.0, 45.0)[j])*c)
        return None

    # ---- legs
    def _thigh(self, P, ps, pos_h, so, Ai):
        T = self.tun
        sd = SIDES[ps]; s = -1.0 if ps == 0 else 1.0
        off = MOTION_H*np.array([P[IDX[f"foot_{sd}_x"]], P[IDX[f"foot_{sd}_y"]], P[IDX[f"foot_{sd}_z"]]])
        ry = float(P[IDX["root_y"]])
        if ry < 0.0:                                   # rig2: the page squats by LOWERING the root (bendFor y,
            off = off + np.array([0.0, -ry*MOTION_H, 0.0])   # Muse's plush sinks into her legs); a human keeps her feet
                                                       # on the floor and bends her knees instead
        if np.any(np.abs(off) > 0):
            off = Ai @ off
        Tt = self.an_rest[ps] + off
        Tt[1] = max(Tt[1], self.an_rest[ps][1] - 0.005*self.stature)          # never through the floor
        pole = _n(np.array([s*T["knee_out"], 0.0, 1.0]))
        K, Tt = ik2(pos_h, Tt, self.l_th, self.l_sh, pole, 0.9995)
        so["hip"], so["knee"], so["ankle"] = pos_h, K, Tt
        u_r = _n(self.kn_rest[ps] - self.hip_rest[ps])
        n_r = self.leg_hinge_r[ps]
        u = _n(K - pos_h); v = _n(Tt - K)
        n = cross(u, v)
        if np.linalg.norm(n) < 1e-4:
            n = cross(u, -pole)
        n = _n(n); so["hinge_l"] = n
        return frame_from(u, n) @ frame_from(u_r, n_r).T

    def _calf(self, ps, pos_k, so):
        u_r = _n(self.an_rest[ps] - self.kn_rest[ps])
        n_r = self.leg_hinge_r[ps]
        v = _n(so["ankle"] - so["knee"])
        return frame_from(v, so["hinge_l"]) @ frame_from(u_r, n_r).T

    def _foot(self, ps, so):
        s = -1.0 if ps == 0 else 1.0
        lift = ss(0.0, 0.11*self.stature, so["ankle"][1] - self.an_rest[ps][1])
        so["lift"] = lift
        return rotY(0.0) @ rotX(math.radians(self.tun["toe_down_deg"])*lift)

    def _ball(self, ps, Dp, so):
        return Dp @ rotX(-math.radians(self.tun["toe_down_deg"])*0.6*so.get("lift", 0.0))

    # ---- head parts
    def _jaw_weights(self, P):
        """(jawOpen, jawLeft, jawRight, jawForward): the SAME numbers drive the jaw bone and the jaw shapes."""
        jo = self._jaw_open(P)
        sk = clamp(float(P[IDX["mouth_skew"]]), -1.0, 1.0)*self.tun["jaw_skew"]
        if self.hf is not None and os.environ.get("JUNO_HUMAN") != "0":    # rig4: no crooked jaw on a human
            m = float(self.hf.cfg["sym"]["skew_max"]); sk = clamp(sk, -m, m)
        fw = self.tun["jaw_fwd_tongue"]*clamp(float(P[IDX["tongue_dental"]]), 0.0, 1.0)
        return jo, max(sk, 0.0), max(-sk, 0.0), fw

    def _jaw_xform(self, P):
        """The jaw bone in rest-world axes: jawOpen * jaw_open_deg about +x and jawLeft/Right * jaw_side_deg about +y
        (jawLeft = chin to +x), both through the hinge (skeleton_hints.jaw_hinge, else the jaw bone origin), plus
        jawForward * jaw_forward_m along +z.  Returns (R, t): R the rotation, t the bone-origin offset (rest space)."""
        jo, jl, jr, jf = self._jaw_weights(P)
        h = self.hints
        R = rotY(math.radians(float(h.get("jaw_side_deg", 4.0)))*(jl - jr)) @ rotX(math.radians(float(h.get("jaw_open_deg", 22.0)))*jo)
        o = self.r[self.bi["jaw"]]
        hinge = _v(h["jaw_hinge"])*self.k if "jaw_hinge" in h else o
        t = (hinge + R @ (o - hinge)) - o + np.array([0.0, 0.0, float(h.get("jaw_forward_m", 0.003))*self.k*jf])
        return R, t

    def _smile_p(self, P):
        """rig4: the smile drive of the vector (the table's mouthSmile: smile + 0.8 mouth_curve)."""
        return max(float(P[IDX["smile"]]), 0.0) + 0.8*max(float(P[IDX["mouth_curve"]]), 0.0)

    def _smile_jaw(self, P):
        if self.hf is None or os.environ.get("JUNO_HUMAN") == "0":
            return 0.0
        return self.hf.smile_jaw(self._smile_p(P))

    def _jaw_open(self, P):
        g = lambda n: clamp(P[IDX[n]], 0.0, 1.0)
        mo, to = g("mouth_open"), g("tongue_out")
        press = g("lip_press")*(1.0 - max(mo, to))
        lips_any = any(P[IDX[n]] != 0 for n in ("lip_open", "lip_upper", "lip_lower", "teeth_gap", "lip_press", "lip_width"))
        open_ = max(g("lip_open")*(1.0 - press), max(mo, 0.42*to))
        gap = max(max(g("teeth_gap"), open_ - 0.30), max(mo, 0.36*to))
        if not lips_any:
            gap = max(gap, 0.8*g("jaw")*(1.0 - g("press")))
        teeth = g("mouth_teeth")*(1.0 - ss(0.15, 0.40, open_))
        gap *= 1.0 - 0.9*teeth
        gap = max(gap, self._smile_jaw(P))       # rig4: a broad smile opens (upper teeth row), bone == shape
        gap *= 1.0 - press                       # bilabial priority: m / b / p close the jaw too (a human mouth)
        if lips_any:                             # rig2: the lip columns are Muse-calibrated; a human talks with less jaw
            gap = max(gap*float(self.tun["speech_jaw_gain"]), mo, 0.36*to)
        if g("tongue_out") > 0:
            gap = max(gap, 0.25*g("tongue_out"))
        return clamp(gap, 0.0, 1.0)

    def _eye(self, P, n, Dhead, pos_e, cam_l, t):
        """engine/face.glsl: eyeF = normalize(normalize(toCam) + (gaze_x*0.28, gaze_y*0.22, 0)) in head space;
        toCam through scene_main.glsl rootGaze (gaze_free slerps toward head-forward, kept inside ~66 deg)."""
        HR = Dhead                       # head axes (local): rest head faces +z
        mid = pos_e
        to_cam = HR.T @ (cam_l - mid)
        L = float(np.linalg.norm(to_cam))
        a = to_cam/max(L, 1e-6)
        b = _n(self._cam_h0)
        gf = clamp(P[IDX["gaze_free"]], 0.0, 1.0)
        c = float(np.dot(a, b)); CMAX = 0.40
        if c < CMAX:
            oo = a - b*c
            oo = _n(oo) if np.dot(oo, oo) > 1e-8 else _n(cross(b, [1.0, 0, 0]))
            a = b*CMAX + oo*math.sqrt(1.0 - CMAX*CMAX); c = CMAX
        th = math.acos(clamp(c, -1.0, 1.0))
        if gf > 0.0 and th > 1e-4:
            a = (math.sin((1.0 - gf)*th)*a + math.sin(gf*th)*b)/math.sin(th)
        gx, gy = P[IDX["gaze_x"]], P[IDX["gaze_y"]]
        sh = clamp(P[IDX["eye_shake"]], 0.0, 1.0)
        if sh > 0:
            gx += 0.02*sh*math.sin(t*37.0 + (1.3 if n == "eye_l" else 0.0))
            gy += 0.015*sh*math.sin(t*41.0 + 0.7)
        eF = _n(a + np.array([gx*0.28, gy*0.22, 0.0]))
        if self.life is not None and self.life.on["eyes"]:      # motion: fixation scanning + drift (conjugate)
            oy, op = self._eye_off
            eF = rotY(oy) @ rotX(-op) @ eF
        # vergence: both eyes converge on one point at the camera's distance (clamped)
        dist = clamp(L, 0.6*self.stature, 4.0*self.stature)
        tgt = mid + HR @ (eF*dist)
        if n == "eye_l":
            self._gaze_pt = tgt
        d = _n(tgt - pos_e)
        # head-relative limits
        dh = HR.T @ d
        yaw = math.atan2(dh[0], dh[2]); pit = math.atan2(-dh[1], math.hypot(dh[0], dh[2]))
        T = self.tun
        yaw = clamp(yaw, -math.radians(T["eye_yaw_max"]), math.radians(T["eye_yaw_max"]))
        pit = clamp(pit, -math.radians(T["eye_pitch_max"]), math.radians(T["eye_pitch_max"]))
        self._eye_ang[n] = (yaw, pit)
        Rl = rotY(yaw) @ rotX(pit)
        return HR @ Rl

    # ---- face shapes
    def _face(self, P, t, D, bi):
        w = np.zeros(len(self.shape_names), np.float32)
        si = self.si
        T = self.tun

        def add(name, v):
            j = si.get(name)
            if j is not None and v:
                w[j] += v

        def p(n):
            return float(P[IDX[n]])
        hf = self.hf if os.environ.get("JUNO_HUMAN") != "0" else None
        btr = None
        if hf is not None:                       # rig4: the held part of the lid columns vs a fast blink
            P = np.array(P, dtype=np.float64, copy=True)
            btr = {}
            for prm, suf in (("blink_l", "Right"), ("blink_r", "Left")):
                held, r = hf.blink_split(clamp(float(P[IDX[prm]]), 0.0, 1.0), prm, t)
                P[IDX[prm]] = held; btr[suf] = r
        # table rules (juno.face.json): {"p": param, "s": shape, "g": gain, "f": "pos"|"neg"|"lin"|"abs"}
        if self._fr_n:
            v = P[self._fr_p]
            v = np.where(self._fr_f == 0, np.maximum(v, 0.0), np.where(self._fr_f == 1, np.maximum(-v, 0.0),
                         np.where(self._fr_f == 2, np.abs(v), v)))
            v = np.minimum(v, self._fr_max)*self._fr_g
            np.add.at(w, self._fr_s, v.astype(np.float32))
        # lid follow from the eye angles
        lf = T["lid_follow"]
        for n, suf in (("eye_l", "Left"), ("eye_r", "Right")):
            yaw, pit = self._eye_ang.get(n, (0.0, 0.0))
            ym, pm = math.radians(T["eye_yaw_max"]), math.radians(T["eye_pitch_max"])
            add("eyeLookDown" + suf, lf*max(pit, 0.0)/pm)
            add("eyeLookUp" + suf, lf*max(-pit, 0.0)/pm)
            out = yaw if suf == "Left" else -yaw            # the left eye (screen-right) looks OUT toward +x
            add("eyeLookOut" + suf, lf*max(out, 0.0)/ym)
            add("eyeLookIn" + suf, lf*max(-out, 0.0)/ym)
        LF = self.life
        if LF is not None:                       # motion: lids + brows follow the gaze beyond the eyeLook shapes
            LF.lid_weights(w, si, {"Left": self._eye_ang.get("eye_l", (0.0, 0.0)), "Right": self._eye_ang.get("eye_r", (0.0, 0.0))},
                           math.radians(T["eye_yaw_max"]), math.radians(T["eye_pitch_max"]))
            LF.expr_weights(w, si, self._smile_p(P), t)
            LF.tone_weights(w, si)
        # mouth (lips.glsl semantics, bilabial priority)
        c01 = lambda n: clamp(p(n), 0.0, 1.0)
        mo, to = c01("mouth_open"), c01("tongue_out")
        lips_any = any(p(n) != 0 for n in ("lip_open", "lip_upper", "lip_lower", "teeth_gap", "lip_press", "lip_width",
                                            "lip_protrude", "lip_tuck"))
        press = c01("lip_press") if lips_any else c01("press")
        press *= 1.0 - max(mo, to)
        keep = 1.0 - press
        open_ = max(c01("lip_open")*keep, max(mo, 0.42*to)) if lips_any else max(0.8*c01("jaw")*keep, mo)
        up = c01("lip_upper")*keep; lo = c01("lip_lower")*keep
        wid = clamp(p("lip_width"), -1.0, 1.0) if lips_any else clamp(p("spread") - p("round"), -1.0, 1.0)
        pro = c01("lip_protrude") if lips_any else c01("round")*0.6
        tuck = (c01("lip_tuck") if lips_any else c01("labdent"))*keep
        jo = self._jaw_open(P)
        open_ = max(open_, self._smile_jaw(P)*keep)   # rig4: the smile's own opening parts the lips (no seal)
        teeth = c01("mouth_teeth")*(1.0 - ss(0.15, 0.40, open_))
        up = max(up, 0.85*teeth); lo = max(lo, 0.85*teeth)
        _jo, jl, jr, jf = self._jaw_weights(P)
        add("jawOpen", jo); add("jawLeft", jl); add("jawRight", jr); add("jawForward", jf)
        # lips sealed over a jaw that is lower than the lip opening asks
        add("mouthClose", clamp(jo - open_, 0.0, 1.0)*0.9*(1.0 - 0.6*pro))
        add("mouthUpperUpLeft", up*0.8); add("mouthUpperUpRight", up*0.8)
        add("mouthLowerDownLeft", lo*0.8 + 0.4*clamp(open_ - jo, 0.0, 1.0))
        add("mouthLowerDownRight", lo*0.8 + 0.4*clamp(open_ - jo, 0.0, 1.0))
        sp, pk = max(wid, 0.0), max(-wid, 0.0)
        add("mouthStretchLeft", 0.75*sp); add("mouthStretchRight", 0.75*sp)
        add("mouthSmileLeft", 0.15*sp); add("mouthSmileRight", 0.15*sp)
        ow = ss(0.2, 0.8, open_)*(1.0 - pk)*(1.0 - pro)   # rig2: an open vowel widens the corners a little (ah, not oh)
        add("mouthStretchLeft", T["open_stretch"]*ow); add("mouthStretchRight", T["open_stretch"]*ow)
        add("mouthPucker", pk*(1.0 - ss(0.15, 0.45, open_)) + 0.5*pro*(1.0 - ss(0.1, 0.35, open_)))
        add("mouthFunnel", (pk*0.8 + pro)*ss(0.10, 0.40, open_))
        add("mouthPressLeft", press); add("mouthPressRight", press)
        add("mouthRollLower", 0.4*press); add("mouthRollUpper", 0.3*press)
        add("lipsTogether", press)
        add("labioDental", tuck); add("lipTuckLower", 0.5*tuck)
        if self.life is not None:                # motion: f/v shows the upper incisors on the lower lip
            self.life.speech_weights(w, si, tuck)
        add("tongueTipUp", max(c01("tongue_tip"), 0.0 if lips_any else c01("tongue")))
        add("tongueDental", c01("tongue_dental")); add("tongueBack", c01("tongue_back"))
        add("tongueOut", to)
        # mouth_teeth: lips drawn back off a closed bite
        add("mouthStretchLeft", 0.5*teeth); add("mouthStretchRight", 0.5*teeth)
        # mouth_wobble: a trembling line (animated on t)
        wb = c01("mouth_wobble")
        if wb > 0:
            ph = math.sin(t*13.0); ph2 = math.sin(t*17.0 + 1.1)
            add("mouthPressLeft", 0.25*wb*max(ph, 0)); add("mouthPressRight", 0.25*wb*max(-ph, 0))
            add("mouthFrownLeft", 0.20*wb*(0.5 + 0.5*ph2)); add("mouthFrownRight", 0.20*wb*(0.5 - 0.5*ph2))
            add("mouthShrugLower", 0.3*wb)
        # knit + raised brows = worried (inner ends up), Muse's BROW_SAD_TILT
        kn = c01("brow_knit"); bu = max(0.5*(p("brow_l") + p("brow_r")), 0.0)
        add("browInnerUp", 0.8*kn*min(bu*1.5, 1.0))
        # rig3: knit under an inner-up tilt (sad / worried) draws the brows together, it does not lower them (the
        # table's knit -> browDown cancelled the inner lift: the sad brows came out flat)
        tu = ss(0.0, 0.4, max(p("brow_tilt"), 0.0))
        if tu > 0.0 and kn > 0.0:
            for nm in ("browDownLeft", "browDownRight"):
                j = si.get(nm)
                if j is not None:
                    w[j] = max(0.0, w[j] - 0.55*kn*tu)
        # surprise reads in the brows as much as the lids (face.glsl: brow += 0.55 eye_wide)
        ew = c01("eye_wide")
        add("browOuterUpLeft", 0.33*ew); add("browOuterUpRight", 0.33*ew); add("browInnerUp", 0.22*ew)
        # the frown: knit over a smile-less mouth turns the corners down (muse.glsl MUSE_FROWN)
        fr = kn*(1.0 - ss(0.0, 0.5, c01("smile")))
        add("mouthFrownLeft", T["knit_frown"]*fr); add("mouthFrownRight", T["knit_frown"]*fr)
        # rig3: a broad smile shows the upper teeth (ASSET2: mouthUpperUp .2-.3 over the smile shapes; the
        # corr_jawOpen_mouthSmile* product correctives part the lips when the jaw is open too)
        sm = 0.5*(w[si["mouthSmileLeft"]] + w[si["mouthSmileRight"]]) if "mouthSmileLeft" in si else 0.0
        ts = T["teeth_smile"]*ss(T["teeth_smile_from"], T["teeth_smile_full"], float(sm))*keep
        if ts > 0.0:
            # a broad smile lifts the corners: the open-vowel / wide stretch and the dropped lower lip fade under it
            # (a smile shows the UPPER teeth; stretch + lower-lip drop read as a grimace)
            k0 = ss(T["teeth_smile_from"], T["teeth_smile_full"], float(sm))
            for nm, kk in (("mouthStretchLeft", "smile_unstretch"), ("mouthStretchRight", "smile_unstretch"),
                           ("mouthLowerDownLeft", "smile_unlower"), ("mouthLowerDownRight", "smile_unlower")):
                j = si.get(nm)
                if j is not None:
                    w[j] *= 1.0 - T[kk]*k0
            add("mouthUpperUpLeft", ts); add("mouthUpperUpRight", ts)
        # bilabial priority: m/b/p close whatever else asked the lips open
        if press > 0.0:
            for nm in ("mouthUpperUpLeft", "mouthUpperUpRight", "mouthLowerDownLeft", "mouthLowerDownRight",
                       "mouthFunnel"):          # (jawOpen already carries the press: it must equal the jaw bone)
                j = si.get(nm)
                if j is not None:
                    w[j] *= 1.0 - 0.85*press
            j = si.get("mouthClose")
            if j is not None:
                w[j] = max(w[j], jo*press)
        np.clip(w, 0.0, 1.2, out=w)
        if LF is not None:
            btr = LF.blink_transient(btr)
        if hf is not None:                       # rig4: smile open/cap, symmetry, brows under a smile, lid aperture
            hf.weights(w, self._smile_p(P), float(jo), btr,
                       {k: si.get("eyeBlink" + k) for k in ("Left", "Right")}, speaking=bool(lips_any))
        if LF is not None:                       # motion: blink micro-compression, brow asymmetry
            LF.blink_weights(w, si, btr)
            LF.brow_asym(w, si)
        # rig3: the asset's brow shapes move only 3-6 mm at weight 1 (a Pixar brow travels ~10-15 mm: SPEC s10 "brows
        # are the major expression carrier"); they stay clean when extrapolated, so the brows are driven past 1
        ba = float(T["brow_amp"])
        if ba != 1.0:
            for j in self._brow_ix:
                if j in self._brow_dn_ix:        # rig4: lowered brows (anger, focus) need more travel to read
                    w[j] = min(w[j]*float(T["brow_amp_down"]), float(T["brow_max_down"]))
                else:
                    w[j] = min(w[j]*ba, float(T["brow_max"]))
        for j, of in self.correctives:
            w[j] = float(np.prod([w[o] for o in of]))
        return w

    def _mat(self, P, t):
        c01 = lambda n: clamp(float(P[IDX[n]]), 0.0, 1.0)
        tear = c01("fx_tear")
        return {"blush": c01("fx_blush"), "pale": c01("fx_pale"), "green": c01("fx_green"),
                "tear_well": ss(0.15, 0.55, tear), "tear_drop": ss(0.55, 0.80, tear), "sweat": c01("fx_sweat"),
                "vein": c01("fx_vein"), "brow_bold": c01("brow_thick"),
                "pupil_scale": -0.6*c01("eye_ring"), "catchlight": clamp(float(P[IDX["eye_spark"]]), -1.0, 1.0),
                "eye_shake": c01("eye_shake"), "eye_scale": clamp(float(P[IDX["eye_scale"]]), -1.0, 1.0),
                "flush_ears": 0.6*c01("fx_blush")}

    def _proxies(self, pos, D, side_out, hc, HR, loc):
        bi = self.bi; S = self.stature
        out = []

        def cap(a, b, r, kind):
            out.append([*loc(a), *loc(b), r, PROXY_KINDS[kind]])

        hcl = hc                                             # already local
        hr = self.head_radii
        out.append([*(hcl - HR[:, 1]*hr[1]*0.25), *(hcl + HR[:, 1]*hr[1]*0.25), float(hr[0]*0.98), PROXY_KINDS["head"]])
        cap(pos[bi["neck_01"]], pos[bi["head"]] + D[bi["head"]] @ np.array([0, 0.02*S, 0]), 0.030*S, "neck")
        s1, s2, s3 = pos[bi["spine_01"]], pos[bi["spine_02"]], pos[bi["spine_03"]]
        cap(s2, s3 + D[bi["spine_03"]] @ np.array([0, 0.05*S, 0]), 0.085*S, "torso")
        cap(s1, s2, 0.078*S, "torso")
        hl = side_out[0]["hip"]; hrr = side_out[1]["hip"]
        cap(hl + np.array([0, 0.02*S, 0]), hrr + np.array([0, 0.02*S, 0]), 0.075*S, "pelvis")
        for ps in range(2):
            so = side_out[ps]
            cap(so["shoulder"], so["elbow"], 0.030*S, "arm")
            cap(so["elbow"], so["wrist"], 0.024*S, "arm")
        for ps in range(2):
            so = side_out[ps]
            Fh = so["hand_frame"]
            cap(so["wrist"] + Fh[:, 1]*0.02*S, so["wrist"] + Fh[:, 1]*0.075*S, 0.026*S, "hand")
        for ps in range(2):
            so = side_out[ps]
            cap(so["hip"], so["knee"], 0.050*S, "leg")
            cap(so["knee"], so["ankle"], 0.036*S, "leg")
        for ps in range(2):
            so = side_out[ps]
            fi = self.side_bones[ps]["foot"]
            toe = so["ankle"] + D[fi] @ np.array([0.0, -0.035*S, 0.11*S])
            cap(so["ankle"] + np.array([0, -0.03*S, -0.02*S]), toe, 0.030*S, "foot")
        top = hcl + HR @ (np.array([0.0, hr[1]*1.0, -hr[2]*0.35]))
        out.append([*top, *(top + HR[:, 1]*0.02*S), 0.055*S, PROXY_KINDS["bun"]])
        return np.array(out, float)

    # ---- derived constants that need the full rest set up (called from _derive)
    @property
    def hand_map(self):
        return self._hand_map


DEFAULT_TUNING = {
    "stature_scene": 2.25,        # floor to the top of the head (bun excluded), scene units
    "squash_human": 0.35,         # share of the engine's squash a human keeps
    "squash_knee": 0.04,          # pelvis drop (of stature) per unit positive squash
    "bounce_scene": 0.05,         # body_bounce: pelvis travel (scene units per unit)
    "neck_share": 0.40,           # of the head rotation in neck_01
    "head_xyz_scene": 0.12,       # head_x/y/z translation, scene units per unit
    "head_radii_k": 1.0,
    "head_frame_ofs": [0.0, 0.005, -0.01],   # worn-prop head frame offset from the head centre (of stature)
    "palm_half_thick": 0.012,     # palm contact off the hand bone axis (of stature)
    "face_pad": 0.013,            # hand half-thickness for the face rest (of stature)
    "shrug_deg": 14.0,
    "clav_reach_deg": 12.0,
    "spread_deg": 9.0,
    "thumb_opp_deg": 55.0,
    "knee_out": 0.18,
    "toe_down_deg": 17.0,
    "eye_yaw_max": 32.0, "eye_pitch_max": 24.0,
    "lid_follow": 0.6,
    "hand_gain_x_out": 0.95, "hand_gain_x_in": 1.05, "hand_gain_y": 1.10, "hand_gain_z": 0.95,
    "hand_min_x": 0.016,          # of stature (palms meet at the midline: clap)
    "hang_out": 0.14, "hang_fwd": 0.10, "hang_reach": 0.96,
    "knee_soft": 0.0,
    "elbow_out_lo": 0.35,
    "low_from": -0.35,            # hand_*_y below this reaches below the hanging hand (the body bends down)
    "low_floor": 0.04,            # target height at hand_*_y = -1 (of stature): the floor
    "low_fwd": 0.10,              # the low target moves forward (of stature) so the arm clears the knees
    "low_squat": 0.40,            # max pelvis drop (of stature)
    "low_squat_share": 0.70,      # share of the lowering the knees take
    "low_pitch_max": 70.0,        # max forward bend at the waist (deg)
    "jaw_skew": 0.35,             # mouth_skew -> jawLeft/Right (a crooked smirk shifts the chin a little)
    "jaw_fwd_tongue": 0.5,        # tongue_dental (th) pushes the jaw forward a touch
    "low_ramp": 0.15,             # hand_y span over which the bend fades in below low_from (no snap)
    "low_back": 0.0,              # hips go back for balance (of the shoulder's forward travel)
    "breath_shrug_deg": 1.6,      # rig2: clavicle lift per unit breath (the shoulders rise on the inhale; SPEC s10 3-5 px)
    "speech_jaw_gain": 0.75,      # rig2: jaw opening from the lip columns (Muse-calibrated) for a human mouth
    "knit_frown": 0.3,            # rig3: knit over a smile-less mouth -> corners down (Muse 0.45; the human frown shape pouts)
    "brow_amp": 2.2,              # rig3: brow shape weights x this (the asset's brows move 3-6 mm at 1) ...
    "brow_max": 2.6,              # ... capped here
    "brow_amp_down": 3.4,         # rig4: browDown* x this (anger read flat at 2.2) ...
    "brow_max_down": 4.0,         # ... capped here
    "smile_unstretch": 0.65,      # rig3: share of the corner stretch removed under a full smile ...
    "smile_unlower": 0.65,        # ... and of the lower-lip drop (a smile shows the upper teeth)
    "teeth_smile": 0.28,          # rig3: mouthUpperUp added by a broad smile (teeth show) ...
    "teeth_smile_from": 0.55,     # ... from this mean mouthSmile weight ( rig4: .30 -> .55, a closed smile keeps its lips) ...
    "teeth_smile_full": 0.95,     # ... to full here (rig4: .75 -> .95)
    "open_stretch": 0.3,          # rig2: corner stretch on open vowels (an O-hole otherwise)
    "far_from": 0.85,             # rig2: hand_*_z where the forward lean fades in (the map's far edge: objects, not gestures)
    "far_reach": 1.0,             # rig2: the forward lean starts when the standard target is this far (of L) from the shoulder
    "far_pitch_max": 45.0,        # rig2: max forward lean for targets in front beyond the arm (deg; 0 = off)
}

# the face table: param -> ARKit/extras shape (side params: _l = screen-left = anatomical Right)
DEFAULT_FACE = {
 # rig3: the human reading of the plush-built columns (rig.human_params; MeshEngine applies it per frame)
 "human": {"_doc": "per column [gain if >= 0, gain if < 0]; every fx mark column not in marks_keep -> 0",
           "gain": {"eye_scale": [0.0, 0.0], "eye_ring": [0.25, 0.25], "eye_wide": [0.75, 1.0], "eye_shake": [0.3, 0.3],
                    "brow_l": [1.0, 0.55], "brow_r": [1.0, 0.55], "brow_thick": [0.3, 0.3], "brow_tilt": [1.0, 0.9],
                    "mouth_pout": [0.2, 1.0], "mouth_wide": [1.0, 0.2], "mouth_curve": [1.0, 0.6],
                    "mouth_teeth": [0.55, 1.0], "lid_happy": [0.9, 1.0], "squint": [0.85, 1.0],
                    "gaze_x": [0.6, 0.6], "gaze_y": [0.6, 0.6], "head_pitch": [1.0, 0.7], "mouth_skew": [0.35, 0.35], "mouth_wobble": [0.6, 0.6],
                    "fx_sweat": [0.0, 0.0], "fx_vein": [0.0, 0.0], "fx_green": [0.3, 0.3], "fx_tear": [0.8, 0.8], "fx_blush": [0.5, 0.5], "lid_asym": [0.4, 0.4]},
           "marks_keep": []},
 "rules": [
    {"p": "blink_l", "s": "eyeBlinkRight"}, {"p": "blink_r", "s": "eyeBlinkLeft"},
    {"p": "eye_wide", "s": "eyeWideLeft"}, {"p": "eye_wide", "s": "eyeWideRight"},
    {"p": "squint", "s": "eyeSquintLeft"}, {"p": "squint", "s": "eyeSquintRight"},
    {"p": "squint", "s": "cheekSquintLeft", "g": 0.4}, {"p": "squint", "s": "cheekSquintRight", "g": 0.4},
    {"p": "lid_droop", "s": "eyeBlinkLeft", "g": 0.40}, {"p": "lid_droop", "s": "eyeBlinkRight", "g": 0.40},
    {"p": "lid_slant", "s": "eyeBlinkLeft", "g": 0.25}, {"p": "lid_slant", "s": "eyeBlinkRight", "g": 0.25},
    {"p": "lid_slant", "s": "eyeSquintLeft", "g": 0.5}, {"p": "lid_slant", "s": "eyeSquintRight", "g": 0.5},
    {"p": "lid_slant", "s": "browDownLeft", "g": 0.4}, {"p": "lid_slant", "s": "browDownRight", "g": 0.4},
    {"p": "lid_happy", "s": "eyeHappyLeft"}, {"p": "lid_happy", "s": "eyeHappyRight"},
    {"p": "lid_happy", "s": "cheekSquintLeft", "g": 0.6}, {"p": "lid_happy", "s": "cheekSquintRight", "g": 0.6},
    {"p": "lid_asym", "s": "eyeSquintLeft", "f": "pos", "g": 0.8}, {"p": "lid_asym", "s": "eyeSquintRight", "f": "neg", "g": 0.8},
    {"p": "eye_scale", "s": "eyeWideLeft", "f": "pos", "g": 0.5}, {"p": "eye_scale", "s": "eyeWideRight", "f": "pos", "g": 0.5},
    {"p": "eye_scale", "s": "eyeSquintLeft", "f": "neg", "g": 0.5}, {"p": "eye_scale", "s": "eyeSquintRight", "f": "neg", "g": 0.5},
    {"p": "eye_ring", "s": "eyeWideLeft"}, {"p": "eye_ring", "s": "eyeWideRight"},
    {"p": "brow_l", "s": "browOuterUpRight", "f": "pos", "g": 0.6}, {"p": "brow_r", "s": "browOuterUpLeft", "f": "pos", "g": 0.6},
    {"p": "brow_l", "s": "browInnerUp", "f": "pos", "g": 0.3}, {"p": "brow_r", "s": "browInnerUp", "f": "pos", "g": 0.3},
    {"p": "brow_l", "s": "browDownRight", "f": "neg"}, {"p": "brow_r", "s": "browDownLeft", "f": "neg"},
    {"p": "brow_knit", "s": "browDownLeft", "g": 0.55}, {"p": "brow_knit", "s": "browDownRight", "g": 0.55},
    {"p": "brow_tilt", "s": "browInnerUp", "f": "pos"},
    {"p": "brow_tilt", "s": "browTiltLeft", "f": "neg"}, {"p": "brow_tilt", "s": "browTiltRight", "f": "neg"},
    # rig2: a smile is Duchenne (cheeks up, the lower lids arc up: SPEC s10) and a little lopsided (her right corner
    # rides ~10 % lower: perfectly symmetric faces are a tell)
    {"p": "smile", "s": "mouthSmileLeft", "f": "pos"}, {"p": "smile", "s": "mouthSmileRight", "f": "pos", "g": 0.9},
    {"p": "smile", "s": "cheekSquintLeft", "f": "pos", "g": 0.4}, {"p": "smile", "s": "cheekSquintRight", "f": "pos", "g": 0.36},
    {"p": "smile", "s": "eyeHappyLeft", "f": "pos", "g": 0.3}, {"p": "smile", "s": "eyeHappyRight", "f": "pos", "g": 0.27},
    {"p": "mouth_curve", "s": "mouthSmileLeft", "f": "pos", "g": 0.8}, {"p": "mouth_curve", "s": "mouthSmileRight", "f": "pos", "g": 0.8},
    {"p": "mouth_curve", "s": "mouthFrownLeft", "f": "neg"}, {"p": "mouth_curve", "s": "mouthFrownRight", "f": "neg"},
    {"p": "mouth_wide", "s": "mouthStretchLeft", "f": "pos", "g": 0.6}, {"p": "mouth_wide", "s": "mouthStretchRight", "f": "pos", "g": 0.6},
    {"p": "mouth_wide", "s": "mouthPucker", "f": "neg", "g": 0.6},
    {"p": "mouth_skew", "s": "mouthSmileLeft", "f": "pos", "g": 0.6}, {"p": "mouth_skew", "s": "mouthFrownRight", "f": "pos", "g": 0.4},
    {"p": "mouth_skew", "s": "mouthSmileRight", "f": "neg", "g": 0.6}, {"p": "mouth_skew", "s": "mouthFrownLeft", "f": "neg", "g": 0.4},
    {"p": "mouth_skew", "s": "mouthLeft", "f": "pos", "g": 0.2}, {"p": "mouth_skew", "s": "mouthRight", "f": "neg", "g": 0.2},
    {"p": "mouth_pout", "s": "mouthShrugLower", "g": 0.8}, {"p": "mouth_pout", "s": "mouthPucker", "g": 0.3},
    {"p": "mouth_teeth", "s": "mouthUpperUpLeft", "g": 0.3}, {"p": "mouth_teeth", "s": "mouthUpperUpRight", "g": 0.3},
    {"p": "cheek", "s": "cheekSquintLeft", "g": 0.3}, {"p": "cheek", "s": "cheekSquintRight", "g": 0.3},
    {"p": "cheek", "s": "cheekPuff", "g": 0.15},
    {"p": "nostril", "s": "noseSneerLeft", "g": 0.3}, {"p": "nostril", "s": "noseSneerRight", "g": 0.3},
    {"p": "nose_wiggle", "s": "noseSneerLeft", "f": "pos", "g": 0.25}, {"p": "nose_wiggle", "s": "noseSneerRight", "f": "pos", "g": 0.25},
    {"p": "nose_wiggle", "s": "mouthShrugUpper", "f": "neg", "g": 0.25},
]}


# hand map + eye constants need the tuning + rest: attach after _derive
_orig_derive = JunoRig._derive


def _derive_full(self):
    _orig_derive(self)
    T = self.tun; S = self.stature
    self._hand_map = {"gain_x_out": T["hand_gain_x_out"], "gain_x_in": T["hand_gain_x_in"], "gain_y": T["hand_gain_y"],
                      "gain_z": T["hand_gain_z"], "min_x_scene": T["hand_min_x"]*S}
    self.palm_above_wrist = float(self.palm_local[0][1])
    self._dy = float(self.sh_rest[0][1] - 1.035*self.ka)     # Muse-tuned heights (shoulder 1.035) moved to her shoulder
    self.eb_H = float(self.eb["H"])
    # the RELAXED rest (all hand params 0): arms hanging at her sides, elbows soft, wrists beside the thighs a little
    # forward (Muse's rest is its paws beside the tummy); this is hand_map.rest_wrist_scene
    self.wr_hang = []
    for i in range(2):
        s = -1.0 if i == 0 else 1.0
        d = _n(np.array([s*T["hang_out"], -1.0, T["hang_fwd"]]))
        self.wr_hang.append(self.sh_rest[i] + d*self.L*T["hang_reach"])
    # standing: soft knees (the pelvis a little below the straight-leg height, so the ankles reach the floor)
    self.pelvis_rest = self.pelvis_rest - np.array([0.0, T["knee_soft"]*S, 0.0])
    self._eye_ang = {}
    self._gaze_pt = self.eye_mid + np.array([0.0, 0.0, 3.0])
    cam = np.asarray(self.char_cfg["camera"]["pos"], float)
    self._cam_h0 = _n(cam - self.eye_mid)                     # rest head = identity axes
    self._spread = math.radians(T["spread_deg"]); self._opp = math.radians(T["thumb_opp_deg"])
    self._build_handlers()
    self._finger_levels()
    self.off = np.array([self.r[i] - (self.r[p] if p >= 0 else 0.0) for i, p in enumerate(self.parent)])
    self._face_matrix()


JunoRig._derive = _derive_full
