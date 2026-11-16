"""Arms in RIG space, per character: an API arm goal (a named target, a coordinate, a direction from the shoulder, ...)
-> the rig's hand target (hand_<s>_x/y/z) + forearm roll (hand_<s>_turn), through each character's OWN geometry:

  muse / juno  the renderer's linear hand map (body.json hand_map: wrist = rest_wrist + (s*gx*L*hx, gy*L*hy, gz*L*hz)),
               face points from body.json face (head space, rotated by the head pose about the neck), torso points
               (muse: the shader's torso ellipsoid; juno: the canon human scaled to her stature)
  fox          animals/fox.glsl handTarget() inverted numerically; face points in the fox's head space (NECK, YAW0,
               EYEC, NOSEP ...) rotated by the head pose; torso points from TORSO_C/R

Frames: CHARACTER space = x screen-right (= the avatar's own LEFT = canon +x), y up, z toward the camera.  Rig side
'l' = screen-left (s = -1) = the avatar's own RIGHT hand = canon side 'r'.
Everything is vectorised over frames (head pose arrays) so a whole statement costs a few hundred microseconds.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from . import kin

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CAL_DIR = os.path.join(HERE, "charmaps")

L_CANON = 0.51                       # canon upper arm + forearm (m)
_N = None
_CANON_PTS = {}


def canon_point(name):
    """canon neutral position (m, pelvis frame) of a named point."""
    global _N
    if _N is None:
        _N = kin.neutral().astype(np.float64)
    p = _CANON_PTS.get(name)
    if p is None:
        p = _CANON_PTS[name] = np.asarray(kin.point_fast(name, _N), np.float64)
    return p


def rot_head(yaw, pitch, roll, v):
    """the shaders' head rotation F = rotY(yaw) rotX(pitch) rotZ(-roll) applied to v (3,) or (T,3); angles (T,) rad."""
    yaw, pitch, roll = (np.asarray(a, np.float64) for a in (yaw, pitch, roll))
    v = np.broadcast_to(np.asarray(v, np.float64), np.broadcast_shapes(np.shape(v), yaw.shape + (3,)))
    r = -roll
    x = np.cos(r) * v[..., 0] - np.sin(r) * v[..., 1]
    y = np.sin(r) * v[..., 0] + np.cos(r) * v[..., 1]
    z = v[..., 2]
    y2 = np.cos(pitch) * y - np.sin(pitch) * z
    z2 = np.sin(pitch) * y + np.cos(pitch) * z
    return np.stack([np.cos(yaw) * x + np.sin(yaw) * z2, y2, -np.sin(yaw) * x + np.cos(yaw) * z2], -1)


# API target name -> (kind, key): face points are given for the paw's OWN side (x outward); 'L'/'R' = the API side
FACE_NAMES = {"nose": "nose", "mouth": "mouth_mid", "lips": "mouth_mid", "chin": "chin", "jaw": "jaw", "eyes": "eyes",
              "forehead": "forehead", "top_of_head": "top", "back_of_head": "back", "face": "nose",
              "in_front_of_face": "front", "above_head": "above"}
SIDED_FACE = {"eye": "eye", "ear": "ear", "cheek": "cheek", "temple": "temple"}


class CharArms:
    """per-character arm geometry; build with for_character(name)."""
    character = "?"
    y_hi = 1.2
    off_gain = 1.0          # move_hand / offset_cm centimetres are a human's: short-armed characters need more

    def __init__(self):
        self.cal = {}
        p = os.path.join(CAL_DIR, f"{self.character}.json")
        if os.path.exists(p):
            try:
                self.cal = json.load(open(p))
            except ValueError:
                self.cal = {}

    # ---- to implement per character
    def face_local(self, key):          # -> (3,) head-space point for the paw's own side (x outward), or None
        raise NotImplementedError

    def head_to_body(self, q, head):    # head-space (T,3) -> character space (T,3)
        raise NotImplementedError

    def body_local(self, key):          # -> (3,) character-space point for an API body target name (x = API side +)
        raise NotImplementedError

    def wrist_to_h(self, s, W):         # (T,3) character-space wrist -> (T,3) hand params, side s = 'l'|'r' (rig)
        raise NotImplementedError

    def shoulder(self, s):
        raise NotImplementedError

    # ---- shared
    def palm_offset(self, part, n):
        """wrist = palm target - this (character units), part = 'palm' | 'index_tip' | ..."""
        up = self.palm_up
        k = {"wrist": 0.0, "palm": 1.0, "back_of_hand": 1.0, "knuckles": 1.3, "fingertips": 1.6, "index_tip": 1.6,
             "middle_tip": 1.6, "ring_tip": 1.5, "pinky_tip": 1.4, "thumb_tip": 1.2}.get(part, 1.0)
        return np.array([0.0, up * k, 0.0]) if n is None else up * k * n

    def canon_to_char(self, p, s):
        """canon point (m, pelvis frame) -> character space.  Near points: scaled about the canon/character shoulder
        of rig side s.  Far points (the camera, the person, the room): about the EYES, so 'the camera' stays at the
        character's own eye height (a big-headed character points up at the lens, not at its chest)."""
        p = np.asarray(p, np.float64)
        cs = "r" if s == "l" else "l"           # rig screen-left = canon right
        sh_c = canon_point(f"{'left' if cs == 'l' else 'right'}_shoulder")
        if np.linalg.norm(p - sh_c) > 0.9:
            return self.eyes() + (p - canon_point("eyes")) * (self.L / L_CANON)
        return self.shoulder(s) + (p - sh_c) * (self.L / L_CANON)

    def eyes(self):
        z = np.zeros(1)
        return self.head_to_body(np.array(self.face_local("eyes"))[None], (z, z, z))[0]

    def h_override(self, name, s):
        """a character-specific hand pose (hx, hy, hz) for a named target, or None (geometry decides)"""
        return None

    def fix_cross(self, q, s):
        """a body target across the midline that this arm cannot reach: the nearest spot on its own side"""
        sg = 1.0 if s == "r" else -1.0
        if sg * q[0] < self.min_x:
            # across the midline the arm cannot go: keep the hand at the target's HEIGHT (a hand-to-the-shoulder
            # gesture reads from its height), just in front of the body on the near side of the midline
            q = np.array(q, np.float64)
            q[0] = sg * self.min_x
            q[1] = min(q[1], self.cross_y)
            q[2] = max(q[2], self.cross_z)
        return q

    def push_head(self, W):
        return W

    def overhead(self, s, W):
        return W

    def goal_h(self, spec, s, head, T, heading=0.0):
        """-> (T,3) rig hand params for an interp arm spec on rig side s.  head = rig (yaw, pitch, roll) arrays (face
        targets follow the head every frame); heading = the body's turn (deg, root yaw + torso twist) at the goal:
        world-fixed targets (camera, the person, room things) are re-aimed in the turned body's frame."""
        tg = spec["target"]
        if isinstance(tg, str) and not tg.endswith("_wrist_rest"):
            ho = self.h_override(tg, s)
            if ho is not None:
                sg = 1.0 if s == "r" else -1.0          # canon +x (the avatar's left) is outward for rig side r
                o = np.asarray(spec.get("offset", np.zeros(3)), np.float64)
                hpm = self.h_per_m()
                dh = np.array([sg * o[0] * hpm[0], o[1] * hpm[1], o[2] * hpm[2]])
                return np.repeat((np.asarray(ho, np.float64) + dh)[None], T, 0)
        h = self.wrist_to_h(s, self.target_points(spec, s, head, T, heading))
        return self.post_h(h, spec)

    def post_h(self, h, spec):
        return h

    def world_to_body(self, p, heading):
        """canon world point -> the body frame after the body turned by `heading` deg (about the pelvis, y axis)"""
        if not heading:
            return p
        a = -math.radians(float(heading))
        c, s_ = math.cos(a), math.sin(a)
        return np.array([c * p[0] + s_ * p[2], p[1], -s_ * p[0] + c * p[2]])

    def target_points(self, spec, s, head, T, heading=0.0):
        """-> (T,3) character-space WRIST goal for an interp arm spec on rig side s, head = (yaw, pitch, roll) arrays."""
        tg = spec["target"]
        part = spec.get("part", "palm")
        off = np.asarray(spec.get("offset", np.zeros(3)), np.float64) * (self.L / L_CANON) * self.off_gain
        sg = 1.0 if s == "r" else -1.0
        aim = bool(spec.get("aim"))
        if isinstance(tg, tuple) and tg[0] == "xyz":
            pc = np.array(tg[1], np.float64)
            if len(tg) > 2 and tg[2] == "world":
                pc = self.world_to_body(pc, heading)
            p = self.canon_to_char(pc, s)
            W = self._aim(p, s, 0.97) if aim else self._reach(p, s)
            return self.overhead(s, np.repeat(self.push_head(W)[None], T, 0) + off)
        if isinstance(tg, tuple) and tg[0] == "shoulder_dir":
            _, cs, dv, ext = tg
            d = np.asarray(dv, np.float64)
            W = self.shoulder(s) + d / max(np.linalg.norm(d), 1e-9) * self.L * float(ext) * self.ext_gain
            return self.overhead(s, np.repeat(self.push_head(W)[None], T, 0) + off)
        if isinstance(tg, tuple) and tg[0] == "shoulder_off":
            _, cs, o = tg
            W = self.shoulder(s) + np.asarray(o, np.float64) * (self.L / L_CANON)
            return np.repeat(W[None], T, 0) + off
        if isinstance(tg, str) and tg.endswith("_wrist_rest"):
            return np.repeat(self.rest_wrist(s)[None], T, 0) + off
        name = str(tg)
        ho = self.h_override(name, s)
        if ho is not None:
            return np.repeat(self.wrist_to_h_inv(s, np.asarray(ho, np.float64))[None], T, 0) + off
        q, kind = self.named(name, s)
        self._crossed = False
        if kind == "body" and not kin.normalize_point_name(name).startswith("in_front_of"):
            q0 = q
            q = self.fix_cross(q, s)
            self._crossed = q is not q0
        if kind == "face":
            dy = getattr(self, "face_dy", 0.0)
            nn_ = kin.normalize_point_name(name)
            if self.character == "juno" and nn_ in ("chin", "jaw"):
                dy += 0.08            # ... while her chin touch needs to come up (the hand arrives from below)
            elif self.character == "juno" and nn_ in ("mouth", "lips"):
                dy += 0.04            # ... and her mouth point was right as measured
            q = q + np.array([0.0, dy, 0.0])
            P = self.head_to_body(np.repeat(q[None], T, 0), head)
            # the palm rests on the face from in front of it: the wrist below the palm
            return P - self.palm_offset(part, None)[None] + off
        if kind == "body":
            if getattr(self, "_crossed", False):     # across the midline: the wrist at the target's height
                return np.repeat(q[None], T, 0) + off
            return np.repeat((q - self.palm_offset(part, None))[None], T, 0) + off
        # space / scene: point a straight arm at it (aim) or reach toward it
        pc = canon_point(name)
        if kin.normalize_point_name(name) in ("camera", "viewer"):
            pc = self.world_to_body(pc, heading)
        p = self.canon_to_char(pc, s)
        W = self._aim(p, s, 0.97) if (aim or kind == "far") else self._reach(p, s)
        return self.overhead(s, np.repeat(self.push_head(W)[None], T, 0) + off)

    def named(self, name, s):
        """-> (point, kind) kind: 'face' (head space, follows the head), 'body' (character space), 'far' (space)."""
        n = kin.normalize_point_name(name)
        sg_api = None
        base = n
        for pre, sgn in (("left_", 1.0), ("right_", -1.0)):
            if n.startswith(pre):
                sg_api, base = sgn, n[len(pre):]
        cal = self.cal.get("targets", {}).get(n)
        if base in SIDED_FACE or n in FACE_NAMES:
            key = SIDED_FACE.get(base) if base in SIDED_FACE else FACE_NAMES[n]
            q = np.array(self.face_local(key), np.float64)
            if base in SIDED_FACE:
                # x outward on the paw's own side: the API side's point seen from rig side s
                own = (1.0 if s == "r" else -1.0)
                tgt_sg = sg_api if sg_api is not None else own
                q[0] = abs(q[0]) * tgt_sg
            else:
                q[0] = q[0] * (1.0 if s == "r" else -1.0)     # head-space x is outward on the paw's own side
            if cal:
                q = q + np.asarray(cal, np.float64)
            return q, "face"
        q = self.body_local(n)
        if q is not None:
            q = np.array(q, np.float64)
            if cal:
                q = q + np.asarray(cal, np.float64)
            return q, "body"
        return None, ("far" if n in kin.SPACE_DIRS or n in ("camera", "viewer", "above_head") else "near")

    def _aim(self, p, s, ext):
        sh = self.shoulder(s)
        d = p - sh
        L = np.linalg.norm(d)
        # a point / reach BEHIND the body hides the arm behind it: aim out to the side instead (a little back at most)
        if L > 1e-9 and d[2] < -0.3 * L:
            d = d.copy()
            horiz = math.hypot(d[0], d[2])
            sgx = 1.0 if d[0] > 0 else (-1.0 if d[0] < 0 else (1.0 if s == "r" else -1.0))
            d[2] = -0.3 * L
            d[0] = sgx * math.sqrt(max(horiz * horiz - d[2] * d[2], 0.0))
            L = np.linalg.norm(d)
        return sh + d / max(L, 1e-9) * min(L, self.L * ext * self.ext_gain)

    def _reach(self, p, s):
        sh = self.shoulder(s)
        d = p - sh
        L = np.linalg.norm(d)
        if L > self.L * 0.98:
            return sh + d / L * self.L * 0.98
        return p - self.palm_offset("palm", None)

    def rest_wrist(self, s):
        return self.wrist_to_h_inv(s, np.zeros(3))


# ============================================================================================ muse / juno (linear map)
class LinearArms(CharArms):
    def __init__(self, character, body, human_scale=None):
        self.character = character
        self.body = body
        hm = body["hand_map"]
        self.rw = np.asarray(hm["rest_wrist_scene"], np.float64)
        self.L = float(hm.get("L_scene", 0.57))
        self.gxo, self.gxi = float(hm.get("gain_x_out", 0.95)), float(hm.get("gain_x_in", 1.05))
        self.gy, self.gz = float(hm.get("gain_y", 1.4)), float(hm.get("gain_z", 0.8))
        self.palm_up = float(hm.get("palm_above_wrist_scene", 0.09))
        f = body.get("face") or {}
        self.face = f
        self.neck = np.asarray(f.get("neck_pivot_scene") or body["joints_scene"]["neck"], np.float64)
        self.hc0 = np.asarray(body["head_scene"], np.float64)
        self.js = {k: np.asarray(v, np.float64) for k, v in body["joints_scene"].items()}
        self.human_scale = human_scale
        rr = (body.get("rig_ranges") or {}).get("hand_l_y")
        self.y_hi = float(rr[1]) if rr else 1.2
        if character == "juno":
            # her face sits above hand_*_y 1.2 (eyes need ~1.31); the renderer accepts up to 1.4 (anim/rig.py RANGES),
            # the page clamps to body.rig_ranges (1.2 today) — requested: juno.body.json rig_ranges hand_*_y 1.4
            self.y_hi = 1.4
        # measured on renders ( round 2): with hand_*_y up to 1.4 Juno's face points land (nose, eyes, mouth,
        # forehead, ears); only her chin / jaw touch sat ~0.04 su low (the hand comes from below and stops early)
        self.face_dy = -0.04 if character == "juno" else 0.0   # her nose / eye / mouth touches landed ~0.04 su high
        self.ext_gain = 1.0
        self.off_gain = 1.6 if character == "muse" else 1.0
        self.min_x = max(0.1, float(hm.get("min_x_scene", 0.07)) + 0.05) if character == "muse" else 0.06
        self.cross_y = 1.08 if character == "muse" else float(self.js["shoulder_r"][1]) - 0.02
        self.cross_z = 0.55 if character == "muse" else float(self.js["shoulder_r"][2]) + 0.14
        self.head_r = np.asarray(body.get("head_radii_scene") or [0.0, 0.0, 0.0], np.float64)
        super().__init__()

    def shoulder(self, s):
        return self.js["shoulder_r" if s == "r" else "shoulder_l"].copy()

    def push_head(self, W):
        """a reach / point that would end inside the (big) head goes out sideways past it (the muse's head is wider
        than its arm is long: 'point up' = up beside the head)"""
        if self.character not in ("muse", "juno") or not self.head_r.any():
            return W
        r = self.head_r + (0.12 if self.character == "muse" else 0.10)
        q = (W - self.hc0) / r
        l = float(np.linalg.norm(q))
        if l >= 1.0:
            return W
        sg = 1.0 if W[0] >= 0 else -1.0
        x2 = 1.0 - q[1] ** 2 - q[2] ** 2
        W = W.copy()
        W[0] = self.hc0[0] + sg * r[0] * math.sqrt(max(x2, 0.0)) if x2 > 0 else W[0]
        return W

    def wrist_to_h(self, s, W):
        sg = 1.0 if s == "r" else -1.0
        W = np.asarray(W, np.float64)
        ox = sg * W[..., 0] - self.rw[0]
        hx = ox / (np.where(ox > 0, self.gxo, self.gxi) * self.L)
        hy = (W[..., 1] - self.rw[1]) / (self.gy * self.L)
        hz = (W[..., 2] - self.rw[2]) / (self.gz * self.L)
        return np.stack([hx, hy, hz], -1)

    def overhead(self, s, W):
        """a reach / point above what the rig can lift (hand_y cap) must not end in front of the face: that spot goes
        out sideways past the head ('point up' = up beside the head)"""
        W = np.array(W, np.float64)
        sg = 1.0 if s == "r" else -1.0
        hy = (W[..., 1] - self.rw[1]) / (self.gy * self.L)
        side = self.head_r[0] + (0.12 if self.character == "muse" else 0.13)
        m = (hy > self.y_hi) & (sg * (W[..., 0] - self.hc0[0]) < side) & (W[..., 2] > self.hc0[2] - 0.1)
        if np.any(m):
            W[..., 0] = np.where(m, self.hc0[0] + sg * side, W[..., 0])
        return W

    def wrist_to_h_inv(self, s, h):
        sg = 1.0 if s == "r" else -1.0
        h = np.asarray(h, np.float64)
        ox = h[..., 0] * np.where(h[..., 0] > 0, self.gxo, self.gxi) * self.L
        return np.stack([sg * (ox + self.rw[0]), self.rw[1] + h[..., 1] * self.gy * self.L,
                         self.rw[2] + h[..., 2] * self.gz * self.L], -1)

    def h_per_m(self):
        """hand units per canon metre (x, y, z) for oscillations"""
        k = self.L / L_CANON
        return np.array([k / (self.gxo * self.L), k / (self.gy * self.L), k / (self.gz * self.L)])

    def head_to_body(self, q, head):
        yaw, pitch, roll = head
        c = rot_head(yaw, pitch, roll, self.hc0 - self.neck)
        o = rot_head(yaw, pitch, roll, q)
        return self.neck + c + o

    def face_local(self, key):
        f = self.face
        if self.character == "muse":
            tbl = {
                "nose": [0.0, -0.09, 0.62], "mouth_mid": f.get("mouth_mid", [0, -0.215, 0.58]),
                "chin": [0.03, -0.40, 0.48], "jaw": [0.25, -0.35, 0.40], "eyes": [0.0, 0.035, 0.58],
                "eye": f.get("eye", [0.228, 0.035, 0.56]), "cheek": f.get("cheek", [0.42, -0.12, 0.45]),
                "temple": f.get("temple", [0.56, 0.2, 0.3]), "ear": [0.60, 0.25, 0.05],
                "forehead": [0.0, 0.30, 0.52], "top": [0.52, 0.40, 0.18], "back": [0.55, 0.15, -0.25],
                "front": [0.0, -0.05, 0.95], "above": [0.25, 0.85, 0.2],
            }
        else:   # juno: her face table + human proportions (head-space metres x k)
            k = self.human_scale or 1.24
            def cp(n):
                return ((canon_point(n) - canon_point("head")) * k).tolist()
            tbl = {
                "nose": [0.0, cp("nose")[1], cp("nose")[2] + 0.01], "mouth_mid": f.get("mouth_mid", cp("mouth")),
                "chin": f.get("chin", cp("chin")), "jaw": [0.06, cp("chin")[1] + 0.02, cp("chin")[2] - 0.06],
                "eyes": [0.0, f.get("eye", cp("left_eye"))[1], f.get("eye", cp("left_eye"))[2]],
                "eye": f.get("eye", cp("left_eye")), "cheek": f.get("cheek", cp("left_cheek")),
                "temple": f.get("temple", cp("left_temple")), "ear": [abs(cp("left_ear")[0]) + 0.01,
                                                                      cp("left_ear")[1], cp("left_ear")[2]],
                "forehead": [0.0, cp("forehead")[1], cp("forehead")[2]], "top": [0.0, cp("top_of_head")[1] + 0.02,
                                                                                 cp("top_of_head")[2] - 0.03],
                "back": [0.05, cp("back_of_head")[1], cp("back_of_head")[2] - 0.02],
                "front": [0.0, cp("nose")[1], cp("nose")[2] + 0.25], "above": [0.0, cp("top_of_head")[1] + 0.2, 0.0],
            }
        return tbl.get(key, tbl["nose"])

    def body_local(self, n):
        sg = 1.0
        base = n
        for pre, sgn in (("left_", 1.0), ("right_", -1.0)):
            if n.startswith(pre):
                sg, base = sgn, n[len(pre):]
        if self.character == "muse":
            tbl = {
                "chest": [0.0, 0.86, 0.50], "sternum": [0.0, 0.86, 0.50], "heart": [0.16, 0.86, 0.48],
                "left_chest": [0.17, 0.86, 0.48], "right_chest": [-0.17, 0.86, 0.48],
                "chest_side": [0.42, 0.85, 0.25], "belly": [0.0, 0.66, 0.52], "stomach": [0.0, 0.70, 0.52],
                "navel": [0.0, 0.62, 0.52], "waist": [0.0, 0.60, 0.50], "hip": [0.42, 0.52, 0.28], "lap": [0.0, 0.42, 0.45],
                "neck_front": [0.0, 1.0, 0.45], "throat": [0.0, 1.0, 0.45], "shoulder": [0.42, 1.06, 0.10],
                "shoulder_top": [0.42, 1.10, 0.06], "upper_arm": [0.50, 0.92, 0.12], "elbow": [0.52, 0.80, 0.0],
                "forearm": [0.54, 0.68, 0.10], "wrist": [0.54, 0.58, 0.15], "palm": [0.54, 0.55, 0.18],
                "back": [0.0, 0.9, -0.5],
                # measured ( r3): in front of the chest = armpit height and well forward; the face front higher
                "in_front_of_chest": [0.0, 1.0, 0.80], "in_front_of_face": [0.0, 1.38, 1.0],
                "knee": [0.24, 0.36, 0.40], "knees": [0.0, 0.36, 0.42], "thigh": [0.24, 0.50, 0.42],
            }
            key = n if n in tbl else base
            if key not in tbl:
                return None
            p = list(tbl[key])
            if key == base and n != base:
                p[0] = abs(p[0]) * sg
            return p
        # juno: the canon human, scaled about the pelvis to her stature
        sh = self.js["shoulder_r"]
        extra = {"in_front_of_chest": [0.0, sh[1] - 0.10, sh[2] + 0.38],
                 "in_front_of_face": [0.0, float(self.hc0[1]) - 0.05, float(self.hc0[2]) + 0.42]}
        if n in extra:
            return extra[n]
        if n in ("camera", "viewer") or n in kin.SPACE_DIRS or n in ("above_head",):
            return None
        if n not in kin.POINT_NAMES:
            return None
        k = self.human_scale or 1.24
        p = self.js["pelvis"] + canon_point(n) * k
        return p.tolist()


# ============================================================================================ fox (fox.glsl)
def _softp(u):
    return 0.5 * (np.sqrt(u * u + 0.020) + u) - 0.0707


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class FoxArms(CharArms):
    character = "fox"
    ARM_U, ARM_F = 0.60, 0.58
    NECK = np.array([0.0, -0.62, -0.18])
    YAW0 = 0.36
    SHO_P = np.array([0.56, -1.02, -0.20])
    REST = {"l": np.array([-0.372, -1.37, 0.42]), "r": np.array([0.347, -1.41, 0.30])}

    COVER = (-0.35, 1.0, 0.45)            # the contract's COVER_EYES: fox.glsl puts the paw centre on the eye
    CHIN = (-0.33, 0.80, 0.40)            # fox.glsl's chin mode, high in its window: knuckles under the chin
    FRONT = (-0.25, 0.52, 0.60)           # both paws forward at chest height (out of the chin window: x > -0.28 there)
    HEART = (-0.42, 0.50, 0.30)           # the contract's HAND_ON_HEART, raised into the torso-up frame

    def __init__(self):
        self.L = self.ARM_U + self.ARM_F
        self.palm_up = 0.17
        self.ext_gain = 1.0
        self.min_x = -0.10
        self.cross_y = -0.80
        self.cross_z = 0.6
        super().__init__()

    VIS_Y = 0.38                          # hand_y below this: the paw is under the bottom of the torso-up frame

    def post_h(self, h, spec):
        tg = spec.get("target")
        if isinstance(tg, str) and tg.endswith("_wrist_rest"):
            return h
        h = np.array(h, np.float64)
        h[..., 1] = np.maximum(h[..., 1], self.VIS_Y)      # keep every commanded paw in the visible frame
        return h

    def _aim(self, p, s, ext):
        W = CharArms._aim(self, p, s, ext)
        d = p - self.shoulder(s)
        fz = max(0.0, d[2] / max(np.linalg.norm(d), 1e-9))
        return W + np.array([0.0, 0.5 * fz, 0.0])      # its forward reach sags in fox.glsl: aim higher to read level

    def h_override(self, name, s):
        n = kin.normalize_point_name(name)
        base = n.split("_", 1)[1] if n.startswith(("left_", "right_")) else n
        if n in ("eyes", "forehead") or base == "eye":
            return self.COVER
        if n in ("nose", "mouth", "lips", "chin", "jaw", "face") or base in ("cheek",):
            return self.CHIN
        if n in ("chest", "heart", "sternum", "neck_front", "throat"):
            return self.HEART
        if n == "in_front_of_chest":
            return self.FRONT
        if base in ("chest",):          # left_chest / right_chest: own side flat, other side across
            own = n.startswith("left_") == (s == "r")
            return (-0.15, 0.38, 0.30) if own else self.HEART
        if base in ("shoulder", "shoulder_top", "upper_arm") and n.startswith("left_") != (s == "r"):
            return (-0.10, 0.74, 0.35)  # the other shoulder: at its height, as far across as the paw goes
        return None

    def wrist_to_h(self, s, W):
        h = self._wrist_to_h(s, W)
        # stay out of fox.glsl's special zones (cover-eyes / chin) unless the target asked for them
        hx, hy = h[..., 0], h[..., 1]
        keep = np.zeros(hx.shape, bool)
        for k in (self.COVER, self.CHIN, self.HEART, self.FRONT, (-0.10, 0.74, 0.35), (-0.15, 0.38, 0.30)):
            keep |= np.abs(h - np.array(k)).max(-1) < 0.03       # an asked-for special pose (h_override)
        zone = (hy > 0.5) & (hx < -0.06) & ~keep
        h[..., 0] = np.where(zone, -0.06, hx)
        return h

    def shoulder(self, s):
        sg = 1.0 if s == "r" else -1.0
        return np.array([sg * self.SHO_P[0], self.SHO_P[1], self.SHO_P[2]])

    def hand_target(self, s, h):
        """fox.glsl handTarget, vectorised: h (T,3) -> T (T,3) body frame"""
        sg = 1.0 if s == "r" else -1.0
        h = np.clip(np.asarray(h, np.float64), -1.0, 1.2)
        AL = self.L
        wide = _smoothstep(0.25, 0.60, h[..., 0])
        rest = self.REST[s]
        Tx = rest[0] + AL * sg * h[..., 0] * (0.62 + 0.75 * wide)
        Ty = rest[1] + AL * h[..., 1] * (1.12 - 0.30 * wide)
        Tz = rest[2] + AL * h[..., 2] * (0.74 + (0.40 - 0.74) * _smoothstep(0.3, 1.0, h[..., 1]))
        up = np.clip((Ty + 1.45) / 1.0, 0.0, 1.2)
        Tx = Tx + sg * 0.30 * up * up
        Tz = Tz + 0.10 * up - 0.62 * _softp(sg * Tx - 0.52)
        Ty = Ty - 0.05 * np.maximum(sg * Tx - 0.8, 0.0) ** 2
        dx = sg * Tx + 0.16
        Tx = sg * (-0.16 + 0.5 * (np.sqrt(dx * dx + 0.02) + dx))
        return np.stack([Tx, Ty, Tz], -1)

    def _wrist_to_h(self, s, W):
        W = np.asarray(W, np.float64)
        sh = W.shape
        W2 = W.reshape(-1, 3)
        if len(W2) > 1:                    # goals are mostly constant over a segment: solve each distinct one once
            U, inv = np.unique(np.round(W2, 6), axis=0, return_inverse=True)
            if len(U) < len(W2):
                return self._wrist_to_h(s, U)[np.asarray(inv).reshape(-1)].reshape(sh)
        AL = self.L
        sg = 1.0 if s == "r" else -1.0
        h = np.zeros_like(W2)
        h[:, 0] = (sg * (W2[:, 0] - self.REST[s][0])) / (0.62 * AL)
        h[:, 1] = (W2[:, 1] - self.REST[s][1]) / (1.12 * AL)
        h[:, 2] = (W2[:, 2] - self.REST[s][2]) / (0.74 * AL)
        h = np.clip(h, -1.0, 1.2)
        eps = 1e-3
        for _ in range(12):           # Newton with a finite-difference Jacobian (3 extra evaluations, vectorised)
            f0 = self.hand_target(s, h)
            r = W2 - f0
            if np.abs(r).max() < 2e-4:
                break
            J = np.zeros((len(h), 3, 3))
            for k in range(3):
                hk = h.copy(); hk[:, k] += eps
                J[:, :, k] = (self.hand_target(s, hk) - f0) / eps
            try:
                step = np.linalg.solve(J + np.eye(3)[None] * 1e-4, r[..., None])[..., 0]
            except np.linalg.LinAlgError:
                break
            step = np.clip(step, -0.3, 0.3)
            h = np.clip(h + step, -1.0, 1.2)
        return h.reshape(sh)

    def wrist_to_h_inv(self, s, h):
        return self.hand_target(s, np.asarray(h, np.float64)[None])[0]

    def h_per_m(self):
        k = self.L / L_CANON
        return np.array([k / (0.62 * self.L), k / (1.12 * self.L), k / (0.74 * self.L)])

    def head_to_body(self, q, head):
        yaw, pitch, roll = head
        return self.NECK + rot_head(np.asarray(yaw) + self.YAW0, pitch, roll, np.asarray(q) - self.NECK)

    def face_local(self, key):
        tbl = {"nose": [0.0, -0.085, 1.10], "mouth_mid": [0.0, -0.30, 0.95], "chin": [0.0, -0.47, 0.75],
               "jaw": [0.30, -0.40, 0.55], "eyes": [0.0, 0.10, 0.60], "eye": [0.262, 0.10, 0.58],
               "cheek": [0.45, -0.18, 0.55], "temple": [0.50, 0.25, 0.30], "ear": [0.55, 0.55, 0.0],
               "forehead": [0.0, 0.38, 0.55], "top": [0.15, 0.62, 0.15], "back": [0.5, 0.2, -0.4],
               "front": [0.0, -0.05, 1.5], "above": [0.2, 1.1, 0.1]}
        return tbl.get(key, tbl["nose"])

    def body_local(self, n):
        sg = 1.0
        base = n
        for pre, sgn in (("left_", 1.0), ("right_", -1.0)):
            if n.startswith(pre):
                sg, base = sgn, n[len(pre):]
        tbl = {"chest": [0.0, -0.85, 0.75], "sternum": [0.0, -0.85, 0.75], "heart": [0.22, -0.88, 0.72],
               "left_chest": [0.25, -0.9, 0.70], "right_chest": [-0.25, -0.9, 0.70], "chest_side": [0.75, -1.3, 0.1],
               "belly": [0.0, -1.70, 0.52], "stomach": [0.0, -1.62, 0.52], "navel": [0.0, -1.75, 0.50],
               "waist": [0.0, -1.80, 0.48], "hip": [0.62, -1.95, 0.20], "lap": [0.0, -1.95, 0.70],
               "neck_front": [0.0, -0.95, 0.55], "throat": [0.0, -0.95, 0.55], "shoulder": [0.62, -0.90, 0.0],
               "shoulder_top": [0.62, -0.85, -0.05], "upper_arm": [0.78, -1.15, -0.05], "elbow": [0.85, -1.45, -0.1],
               "forearm": [0.75, -1.5, 0.15], "wrist": [0.40, -1.40, 0.40], "palm": [0.40, -1.35, 0.42],
               "back": [0.0, -1.2, -0.7], "in_front_of_chest": [0.0, -0.85, 0.95],
               "in_front_of_face": [0.0, -0.3, 1.4], "knee": [0.45, -1.9, 0.6], "knees": [0.0, -1.9, 0.65],
               "thigh": [0.45, -1.8, 0.5]}
        key = n if n in tbl else base
        if key not in tbl:
            return None
        p = list(tbl[key])
        if key == base and n != base:
            p[0] = abs(p[0]) * sg
        return p


_ARMS = {}


def for_character(character):
    a = _ARMS.get(character)
    if a is not None:
        return a
    if character == "fox":
        a = FoxArms()
    else:
        if character == "juno":
            path = os.path.join(REPO, "renderer", "humans", "juno", "juno.body.json")
        else:
            path = os.path.join(REPO, "renderer", "animals", "muse.body.json")
        body = json.load(open(path))
        hs = None
        if character == "juno":
            js = body["joints_scene"]
            hs = (js["shoulder_r"][1] - js["pelvis"][1]) / float(canon_point("left_shoulder")[1])
        a = LinearArms(character, body, human_scale=hs)
    _ARMS[character] = a
    return a
