"""render: RigFrame + a stand-in rig (until RIG's renderer/humans/rig.py lands) — FK poses and named
expressions straight on the asset's skeleton/shapes, placed in the scene like Muse (origin on the floor at (0,0,0),
facing the camera).  RigFrame fields = docs/CONTRACT--render.md section 2."""
import numpy as np


class RigFrame:
    __slots__ = ("bones", "weights", "grasp", "head", "proxies", "bound", "gaze", "flags")

    def __init__(self, bones, weights, grasp=None, head=None, proxies=None, bound=None, gaze=None, flags=0):
        self.bones, self.weights = bones, weights
        self.grasp = grasp if grasp is not None else np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        self.head = head if head is not None else np.eye(4, dtype=np.float32)
        self.proxies = proxies if proxies is not None else np.zeros((0, 8), np.float32)
        self.bound = bound if bound is not None else np.array([0, 1.15, 0, 1.6], np.float32)
        self.gaze = gaze if gaze is not None else np.zeros(3, np.float32)
        self.flags = flags


def _rx(a):
    c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler(x=0.0, y=0.0, z=0.0):
    """degrees, applied X then Y then Z about the ASSET's world axes"""
    return _rz(np.radians(z)) @ _ry(np.radians(y)) @ _rx(np.radians(x))


# named poses: bone -> world-axis rotation (degrees) applied about the bone's head, in hierarchy order
POSES = {
    "rest": {},
    "relaxed": {"upperarm_l": (0, 0, -22), "upperarm_r": (0, 0, 22), "lowerarm_l": (0, -10, 0), "lowerarm_r": (0, 10, 0)},
    "wave": {"upperarm_l": (0, 0, -22), "upperarm_r": (0, 0, 115), "lowerarm_r": (0, 0, 35)},
    "point": {"upperarm_l": (0, 0, -22), "upperarm_r": (-70, 0, 30), "lowerarm_r": (-10, 0, 0)},
    "think": {"upperarm_l": (0, 0, -22), "upperarm_r": (-40, 0, 30), "lowerarm_r": (-120, 0, 0), "head": (6, -8, 4)},
    "turn": {"upperarm_l": (0, 0, -22), "upperarm_r": (0, 0, 22), "spine_02": (0, 12, 0), "head": (0, 18, 0)},
}
EXPRS = {
    "neutral": {},
    "talk_a": {"jawOpen": 0.55, "mouthLowerDownLeft": 0.2, "mouthLowerDownRight": 0.2},
    "talk_o": {"jawOpen": 0.35, "mouthFunnel": 0.6, "mouthPucker": 0.3},
    "talk_e": {"jawOpen": 0.25, "mouthStretchLeft": 0.4, "mouthStretchRight": 0.4, "mouthUpperUpLeft": 0.2, "mouthUpperUpRight": 0.2},
    "smile": {"mouthSmileLeft": 0.85, "mouthSmileRight": 0.85, "cheekSquintLeft": 0.45, "cheekSquintRight": 0.45,
              "eyeSquintLeft": 0.25, "eyeSquintRight": 0.25, "jawOpen": 0.12, "eyeHappyLeft": 0.4, "eyeHappyRight": 0.4},
    "surprise": {"browInnerUp": 0.9, "browOuterUpLeft": 0.8, "browOuterUpRight": 0.8, "eyeWideLeft": 0.8,
                 "eyeWideRight": 0.8, "jawOpen": 0.35, "mouthFunnel": 0.2},
    "sad": {"browInnerUp": 0.7, "mouthFrownLeft": 0.6, "mouthFrownRight": 0.6, "eyeSquintLeft": 0.1, "eyeSquintRight": 0.1},
    "angry": {"browDownLeft": 0.85, "browDownRight": 0.85, "noseSneerLeft": 0.4, "noseSneerRight": 0.4,
              "mouthPressLeft": 0.4, "mouthPressRight": 0.4},
    "blink": {"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0},
}
JAW_DEG = 18.0


class TestRig:
    """FK on the asset's rest skeleton.  scene placement: origin_scene + s*p_asset, s = H_scene/height_m."""

    def __init__(self, asset, H_scene=2.30, origin=(0.0, 0.0, 0.0)):
        A = asset
        self.A = A
        self.names = [b["name"] for b in A.bones]
        self.bi = {n: i for i, n in enumerate(self.names)}
        self.parents = A.parents
        self.rest = A.rest.astype(np.float64)
        hm = float(A.meta.get("height_m", 1.68))
        self.s = H_scene/hm
        S = np.eye(4); S[:3, :3] *= self.s; S[:3, 3] = origin
        self.S = S
        self.shape_idx = {n: i for i, n in enumerate(A.shape_names)}
        self.jaw_deg = float(A.skeleton.get("skeleton_hints", {}).get("jaw_open_deg", JAW_DEG))
        # render4: product correctives (corr_* = product of its drivers), as RIG's JunoRig drives them -- the judge
        # shots (juno_still --params) went through here WITHOUT them (open smile / jaw-open lip correctives missing)
        self.correctives = []
        for nm, c in (A.meta.get("correctives") or {}).items():
            of = c.get("of", []) if isinstance(c, dict) else list(c)
            if nm in self.shape_idx and of and all(o in self.shape_idx for o in of):
                self.correctives.append((self.shape_idx[nm], [self.shape_idx[o] for o in of]))

    def world(self, rots):
        """rots: bone -> 3x3 world-axis rotation about the bone head.  Returns posed asset-space matrices."""
        nb = len(self.names)
        G = np.zeros((nb, 4, 4))
        for i in range(nb):
            p = self.parents[i]
            if p < 0:
                G[i] = self.rest[i]
            else:
                G[i] = G[p] @ np.linalg.inv(self.rest[p]) @ self.rest[i]
            R = rots.get(self.names[i])
            if R is not None:
                h = G[i][:3, 3].copy()
                M = np.eye(4); M[:3, :3] = R
                T1 = np.eye(4); T1[:3, 3] = h
                T0 = np.eye(4); T0[:3, 3] = -h
                G[i] = T1 @ M @ T0 @ G[i]
        return G

    def frame(self, pose="relaxed", expr="neutral", weights=None, extra_rots=None, gaze=(0.0, 0.0)):
        rots = {k: euler(*v) for k, v in POSES.get(pose, {}).items()}
        w = np.zeros(len(self.A.shape_names), np.float32)
        for k, v in EXPRS.get(expr, {}).items():
            if k in self.shape_idx:
                w[self.shape_idx[k]] = v
        for k, v in (weights or {}).items():
            if k in self.shape_idx:
                w[self.shape_idx[k]] = v
        # render4: mouthClose never exceeds jawOpen (as RIG's live clamp, rig.py): without an open jaw it pushes
        # the lower lip up/forward = the 'sad pout' the judge saw in neutral / speak_closed_lips (ASSET2 04:19)
        if "mouthClose" in self.shape_idx and "jawOpen" in self.shape_idx:
            jc, jj = self.shape_idx["mouthClose"], self.shape_idx["jawOpen"]
            w[jc] = min(w[jc], w[jj])
        for j, of in self.correctives:          # render4
            if not (weights and self.A.shape_names[j] in weights):
                w[j] = float(np.prod([w[o] for o in of]))
        jo = w[self.shape_idx["jawOpen"]] if "jawOpen" in self.shape_idx else 0.0
        if jo and "jaw" in self.bi:
            rots["jaw"] = euler(self.jaw_deg*jo, 0, 0)
        for e in ("eye_l", "eye_r"):
            if e in self.bi and (gaze[0] or gaze[1]):
                rots[e] = euler(-gaze[1], gaze[0], 0)
        rots.update(extra_rots or {})
        G = self.world(rots)
        # rotations about parents' heads must carry the children: world() applies each bone's own rotation after its
        # parent's chain, so a child rotation composes on top (FK).  Eyes / jaw are leaves.
        B = np.einsum("ij,bjk->bik", self.S, G)
        return self._frame(B, w)

    def _frame(self, B, w):
        bi = self.bi
        P = lambda n: B[bi[n]][:3, 3] if n in bi else None
        s = self.s
        # head frame: the head bone's rotation relative to rest, origin at the ear line
        Rh = B[bi["head"]][:3, :3]/s @ np.linalg.inv(self.rest[bi["head"]][:3, :3])
        head = np.eye(4); head[:3, :3] = Rh
        head[:3, 3] = P("head") + Rh @ np.array([0.0, 0.09, 0.0])*s
        # grasp frames: rig side 0 = screen-left = anatomical RIGHT hand
        grasp = np.zeros((2, 4, 4))
        lm = self.A.meta.get("landmarks", {})
        for gi, (side, key) in enumerate((("r", "grasp_r"), ("l", "grasp_l"))):
            hb = B[bi[f"hand_{side}"]]
            if key in lm and "offset" in lm[key]:
                g = hb @ np.asarray(lm[key]["offset"], float).reshape(4, 4)
            else:
                g = hb.copy(); g[:3, 3] = hb[:3, 3] + hb[:3, 1]*0.08
            Rg = g[:3, :3]/np.linalg.norm(g[:3, 0])
            grasp[gi] = np.eye(4); grasp[gi][:3, :3] = Rg; grasp[gi][:3, 3] = g[:3, 3]
        # proxies (capsules, scene units)
        caps = []
        def cap(a, b, r, kind=0):
            pa, pb = P(a), P(b) if isinstance(b, str) else b
            if pa is not None and pb is not None:
                caps.append(list(pa) + list(pb) + [r*s, kind])
        cap("pelvis", "spine_03", 0.15)
        cap("spine_03", "neck_01", 0.13)
        hc = head[:3, 3]
        caps.append(list(hc - Rh @ np.array([0, 0.05, 0])*s) + list(hc + Rh @ np.array([0, 0.03, 0])*s) + [0.105*s, 1])
        for sd in ("l", "r"):
            cap(f"upperarm_{sd}", f"lowerarm_{sd}", 0.05)
            cap(f"lowerarm_{sd}", f"hand_{sd}", 0.04)
            cap(f"hand_{sd}", f"middle_02_{sd}", 0.04, 2)
            cap(f"thigh_{sd}", f"calf_{sd}", 0.075)
            cap(f"calf_{sd}", f"foot_{sd}", 0.055)
            cap(f"foot_{sd}", f"ball_{sd}", 0.045)
        prox = np.array(caps, np.float32)
        pts = np.concatenate([prox[:, :3], prox[:, 3:6]], 0)
        c = 0.5*(pts.min(0) + pts.max(0))
        r = float(np.max(np.linalg.norm(pts - c, axis=1))) + 0.25*s
        return RigFrame(B.astype(np.float32), w, grasp.astype(np.float32), head.astype(np.float32), prox,
                        np.array([*c, r], np.float32), np.zeros(3, np.float32), 0)
