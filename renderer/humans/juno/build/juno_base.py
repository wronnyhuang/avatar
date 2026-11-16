"""asset — the Juno BASE: MakeHuman hm08 topology morphed to Juno's design, skeleton, skin weights, landmarks.

Every other build module (face_shapes, eyes_mouth, skin_tex, hair, wardrobe) receives a `Base` and must work for ANY
base shape (procedural, landmark-driven, never hard-coded coordinates) — the ASSET lead keeps changing the design.

    base = Base(data_dir)            # data_dir = T/asset-src/mpfb2/data
    base.VA        (nV,3) float64    rest positions, metres, asset space (+y up, +z forward, +x = her LEFT)
    base.obj       MHObj             faces (lists of (v, vt)), fgroup, vt (MH UVs), groups
    base.verts(g)  vertex ids of an MH group ("body", "helper-tongue", "helper-l-eye", ...)
    base.faces_of(g) face ids of an MH group
    base.skin_mask (nV,) bool        vertices of the "body" group (the skin surface)
    base.N         (nV,3)            smooth normals of the welded MH mesh at rest
    base.tris      (nT,3)            triangulated welded MH mesh (all groups) — for normals
    base.bidx      {bone: index}     base.bone_names, base.rest (nb,4,4), base.skeleton (json dict)
    base.J, base.W (nV,4)            skin weights (u16 joints, f32 weights)
    base.lm        landmarks dict: eye_c_l/eye_c_r (centre), eye_r_l/eye_r_r (radius), mouth (lip centre), jaw_hinge,
                   chin, head_top, nose_tip, ...
    base.target(path) -> (nV,3) metres   any MH target (path relative to data/targets), NOT applied
    base.cube(name) -> (3,)          MH joint cube centre (after morph)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mh_base import MHObj, bone_matrix_asset, target_dense  # noqa: E402

DM = 0.1


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def tri_of(f):
    return [[f[0], f[1], f[2]]] + ([[f[0], f[2], f[3]]] if len(f) == 4 else [])


def vertex_normals(V, tris):
    fn = np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]])
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


class Base:
    def __init__(self, data_dir, design=None):
        self.data = data_dir
        self.obj = MHObj(os.path.join(data_dir, "3dobjs", "base.obj"))
        self.nV = len(self.obj.v)
        self._tcache = {}
        self.skin_mask = np.zeros(self.nV, bool)
        self.skin_mask[self.verts("body")] = True
        self.tris = np.array([t for f in self.obj.faces for t in tri_of([vi for vi, _ in f])], np.int64)
        V = self.obj.v * DM
        from juno_design import stylize  # the ASSET lead's design (targets + custom sculpt deformers)
        V = stylize(self, V, design)
        self.floor_shift = float(V[self.verts("body"), 1].min())
        V[:, 1] -= self.floor_shift
        self.VA = V
        self.N = vertex_normals(self.VA, self.tris)
        self._skeleton()
        self._weights()
        self._landmarks()

    # ---- helpers
    def verts(self, g):
        return self.obj.group_verts(g)

    def faces_of(self, g):
        return np.array(self.obj.groups.get(g, []), np.int64)

    def target(self, rel):
        if rel not in self._tcache:
            self._tcache[rel] = target_dense(os.path.join(self.data, "targets", rel), self.nV) * DM
        return self._tcache[rel]

    def cube(self, name, V=None):
        V = self.VA if V is None else V
        return V[self.verts(name)].mean(0)

    # ---- skeleton
    def _skeleton(self):
        rig = json.load(open(os.path.join(self.data, "rigs", "standard", "rig.game_engine.json")))
        VA = self.VA

        def endpoint(spec):
            if spec["strategy"] == "CUBE":
                return self.cube(spec["cube_name"])
            return VA[np.array(spec["vertex_indices"])].mean(0)

        bones = {}
        for name, b in rig.items():
            nm = "root" if name == "Root" else name
            par = b["parent"]
            par = "root" if par == "Root" else (par or None)
            bones[nm] = dict(parent=par, head=endpoint(b["head"]), tail=endpoint(b["tail"]), roll=b.get("roll", 0.0))
        bones["root"]["head"] = np.zeros(3)
        bones["root"]["tail"] = np.array([0.0, 0.0, -0.1])
        hinge, chin = self.cube("joint-mouth"), self.cube("joint-jaw")  # MH: joint-mouth = jaw hinge, joint-jaw = chin
        bones["jaw"] = dict(parent="head", head=hinge, tail=chin, roll=0.0)
        self.eye = {}
        for side in ("l", "r"):
            ev = self.verts(f"helper-{side}-eye")
            c = VA[ev].mean(0)
            r = float(np.linalg.norm(VA[ev] - c, axis=1).mean())
            df = getattr(self, "design_fit", None)
            if df:   # SCULPT's fitted eyeball wins (centre in stylize space -> floor-shifted; radius exact)
                c = np.asarray(df[f"eye_center_{side}"], float) - np.array([0.0, self.floor_shift, 0.0])
                r = float(df["eye_radius"])
                self.iris_radius = float(df.get("iris_radius", 0.0)) or None
            self.eye[side] = (c, r)
            bones[f"eye_{side}"] = dict(parent="head", head=c, tail=c + np.array([0, 0, 0.02]), roll=0.0)
        order = []

        def visit(n):
            if n in order:
                return
            if bones[n]["parent"]:
                visit(bones[n]["parent"])
            order.append(n)

        for n in ["root"] + sorted(k for k in bones if k != "root"):
            visit(n)
        self.bone_names = order
        self.bidx = {n: i for i, n in enumerate(order)}
        self.bones = bones
        self.rest = np.stack([bone_matrix_asset(bones[n]["head"], bones[n]["tail"], bones[n]["roll"]) for n in order])
        self.skeleton = {"bones": [{"name": n, "parent": self.bidx[bones[n]["parent"]] if bones[n]["parent"] else -1,
                                    "rest": self.rest[i].reshape(-1).tolist(),
                                    "inverse_bind": np.linalg.inv(self.rest[i]).reshape(-1).tolist(), "helper": False,
                                    "length": float(np.linalg.norm(bones[n]["tail"] - bones[n]["head"]))}
                                   for i, n in enumerate(order)]}

    def _weights(self):
        wj = json.load(open(os.path.join(self.data, "rigs", "standard", "weights.game_engine.json")))["weights"]
        per = [[] for _ in range(self.nV)]
        for bn, lst in wj.items():
            b = self.bidx["root" if bn == "Root" else bn]
            for vi, w in lst:
                per[vi].append((w, b))
        for vi in self.verts("helper-lower-teeth").tolist() + self.verts("helper-tongue").tolist():
            per[vi] = [(1.0, self.bidx["jaw"])]
        for vi in self.verts("helper-upper-teeth").tolist():
            per[vi] = [(1.0, self.bidx["head"])]
        for side in ("l", "r"):
            for vi in self.verts(f"helper-{side}-eye").tolist():
                per[vi] = [(1.0, self.bidx[f"eye_{side}"])]
        J = np.zeros((self.nV, 4), np.uint16)
        W = np.zeros((self.nV, 4), np.float32)
        for vi in range(self.nV):
            l = sorted(per[vi], reverse=True)[:4] or [(1.0, self.bidx["root"])]
            s = sum(w for w, _ in l)
            for k, (w, b) in enumerate(l):
                J[vi, k] = b
                W[vi, k] = w / s
        self.J, self.W = J, W

    def weights_dense(self):
        D = np.zeros((self.nV, len(self.bone_names)), np.float32)
        for k in range(4):
            np.add.at(D, (np.arange(self.nV), self.J[:, k].astype(np.int64)), self.W[:, k])
        return D

    def _landmarks(self):
        VA = self.VA
        c = np.linalg.norm(self.target("expression/units/caucasian/mouth-compression.target.gz"), axis=1) ** 2
        mouth = (VA * c[:, None]).sum(0) / c.sum()
        mouth[0] = 0.0
        body = self.verts("body")
        front_mid = body[np.abs(VA[body, 0]) < 0.004]
        head_y0 = self.cube("joint-head")[1]
        hv = front_mid[VA[front_mid, 1] > head_y0]
        nose_tip = VA[hv[np.argmax(VA[hv, 2])]]
        self.lm = {"eye_c_l": self.eye["l"][0], "eye_c_r": self.eye["r"][0], "eye_r_l": self.eye["l"][1],
                   "eye_r_r": self.eye["r"][1], "mouth": mouth, "jaw_hinge": self.cube("joint-mouth"),
                   "chin": self.cube("joint-jaw"), "head_top": VA[body][np.argmax(VA[body, 1])],
                   "nose_tip": nose_tip, "neck": self.cube("joint-neck"), "head_joint": self.cube("joint-head")}

    def height(self):
        return float(self.VA[self.verts("body"), 1].max())
